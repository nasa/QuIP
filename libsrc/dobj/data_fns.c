#include "quip_config.h"

#include <stdio.h>
#include <ctype.h>

#include "quip_prot.h"
#include "data_obj.h"
#include "debug.h"
#include "platform.h"
#include "dobj_private.h"

//#include "img_file.h"

/*
 * set up the increments for a contiguous object
 *
 * The increment tells us how many basic elements we have to skip to get to
 * the next one...  Originally, this would be set to one, even when we only had
 * a dimension of 1...  But for processing, we need increments for dimension=1
 * to be equal to zero, so we can stay in place...  what trouble will that cause?
 */

#define SETUP_INC(stem)						\
								\
		if( dp->dt_##stem##_dim[i] == 1 )		\
			dp->dt_##stem##_inc[i] = 0;		\
		else	dp->dt_##stem##_inc[i] = n_##stem;	\
		n_##stem *= dp->dt_##stem##_dim[i];


#define SETUP_TYPE_INC						\
		if( OBJ_TYPE_DIM(dp,i) == 1 )			\
			SET_OBJ_TYPE_INC(dp,i,0);		\
		else	SET_OBJ_TYPE_INC(dp,i,n_type);		\
		n_type *= OBJ_TYPE_DIM(dp,i);

#define SETUP_MACH_INC						\
		if( OBJ_MACH_DIM(dp,i) == 1 )			\
			SET_OBJ_MACH_INC(dp,i,0);		\
		else	SET_OBJ_MACH_INC(dp,i,n_mach);		\
		n_mach *= OBJ_MACH_DIM(dp,i);


void make_contiguous(Data_Obj *dp)
{
	int i;
	int n_type;
	int n_mach;

	n_type = 1;
	n_mach = 1;
	for(i=0;i<N_DIMENSIONS;i++){
		SETUP_MACH_INC
		SETUP_TYPE_INC
	}

	SET_OBJ_FLAG_BITS(dp, DT_CONTIG | DT_CHECKED );
}

/* Rename an existing object.
 * Returns 0 on success, -1 if the new name contains an illegal character,
 * or if an object with the requested name already exists.
 */

int obj_rename(QSP_ARG_DECL  Data_Obj *dp,const char *newname)
{
	Data_Obj *dp2;

	if( !is_valid_dname(QSP_ARG  newname) ) return(-1);

	// We expect that the passed object is in the namespace.
	assert( dobj_of(OBJ_NAME(dp)) != NULL );

	dp2=dobj_of(newname);
	if( dp2 != NULL ){
		sprintf(ERROR_STRING,
			"name \"%s\" is already in use in area \"%s\"",
			newname,AREA_NAME( OBJ_AREA(dp) ) );
		WARN(ERROR_STRING);
		return(-1);
	}
	/* BUG?  where is the object's node? */

	del_dobj(dp);	// remove from database, add to free list
	SET_OBJ_NAME(dp,savestr(newname));

	/* now add this to the database */
	/* We might have a memory leak, with the item node? */
	assert( remove_from_item_free_list(dobj_itp, dp) == 0 );
	ADD_OBJ_ITEM(dp);

	return(0);
}

/* this routine is made obsolete by make_dobj */

Data_Obj *
_make_obj(QSP_ARG_DECL  const char *name,
	dimension_t frames,
	dimension_t rows,
	dimension_t cols,
	dimension_t type_dim,
	Precision * prec_p )
{
	Data_Obj *dp;
	Dimension_Set *dsp;

	INIT_DIMSET_PTR(dsp)

	SET_DIMENSION(dsp,0,type_dim);
	SET_DIMENSION(dsp,1,cols);
	SET_DIMENSION(dsp,2,rows);
	SET_DIMENSION(dsp,3,frames);
	SET_DIMENSION(dsp,4,1);

	dp = make_dobj(name,dsp,prec_p);

	RELEASE_DIMSET(dsp);

	return(dp);
}

/* What is a list object?
 */

Data_Obj *_make_obj_list(QSP_ARG_DECL  const char *name, List *lp)
{
	Data_Obj *dp;
    	Data_Obj **dp_tbl;
	Dimension_Set *dsp;
	Precision * prec_p;
	Node *np;
	int uk_leaf=0;

	INIT_DIMSET_PTR(dsp)

	dp = dobj_of(name);
	if( dp != NULL ){
		sprintf(ERROR_STRING,"make_obj_list:  object %s already exists!?",name);
		WARN(ERROR_STRING);
		return(NULL);
	}

	SET_DIMENSION(dsp,0,1);
	SET_DIMENSION(dsp,1,eltcount(lp));
	if( DIMENSION(dsp,1) < 1 ){
		sprintf(ERROR_STRING,"make_obj_list %s:  object list has no elements!?",name);
		WARN(ERROR_STRING);
		return(NULL);
	}

	SET_DIMENSION(dsp,2,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,4,1);

	if( sizeof(Data_Obj *)==4 )
		prec_p=prec_for_code(PREC_DI);
	else if( sizeof(Data_Obj *)==8 )
		prec_p=prec_for_code(PREC_LI);
	else {
		prec_p=NULL;	// error1 doesn't return, but silence compiler.
		error1("Unexpected pointer size!?");
	}

	dp = make_dobj(name,dsp,prec_p);	/* specify prec_long because sizeof(long) == sizeof(dp) */

	SET_OBJ_PREC_PTR(dp,NULL);

	dp_tbl=(Data_Obj **)OBJ_DATA_PTR(dp);

	np=QLIST_HEAD(lp);
	while(np!=NULL){
		*dp_tbl = (Data_Obj *) NODE_DATA(np);
		if( UNKNOWN_SHAPE( OBJ_SHAPE(*dp_tbl) ) )
			uk_leaf++;
		dp_tbl++;
		np=NODE_NEXT(np);
	}

	if( dp_tbl != OBJ_DATA_PTR(dp) ){	/* one or more objects? */
		dp_tbl=(Data_Obj **)OBJ_DATA_PTR(dp);
		SET_OBJ_PREC_PTR(dp, OBJ_PREC_PTR((*dp_tbl)) );
	}

	auto_shape_flags( OBJ_SHAPE(dp) );

	if( uk_leaf ){
		CLEAR_OBJ_FLAG_BITS(dp,SHAPE_DIM_MASK);
		SET_OBJ_FLAG_BITS(dp,DT_UNKNOWN_SHAPE);
	}
	SET_OBJ_FLAG_BITS(dp,DT_OBJ_LIST);

	return(dp);
}

Data_Obj *mk_scalar(QSP_ARG_DECL  const char *name,Precision * prec_p)
{
	Data_Obj *dp;

	dp=make_obj(name,1,1,1,1,prec_p);
	return(dp);
}

// Doesn't support CUDA???

void assign_scalar_obj(QSP_ARG_DECL  Data_Obj *dp,Scalar_Value *svp)
{
	Precision *prec_p;

	assert( svp != NULL );

#ifdef HAVE_ANY_GPU
	if( ! OBJ_IS_RAM(dp) ){
		index_t offset;

		offset = OBJ_OFFSET(dp) * PREC_SIZE( OBJ_PREC_PTR(dp) );
		// BUG may not be cuda, use platform-specific function!
		(*(PF_MEM_UPLOAD_FN(PFDEV_PLATFORM(OBJ_PFDEV(dp)))))
			(QSP_ARG  OBJ_DATA_PTR(dp), &svp->u_d, PREC_SIZE(OBJ_PREC_PTR(dp)),
				offset,
				OBJ_PFDEV(dp) );
		return;
	}
#endif // HAVE_ANY_GPU

	prec_p = OBJ_PREC_PTR(dp);
	if( (*(prec_p->assign_scalar_obj_func))(dp,svp) < 0 ){
		sprintf(ERROR_STRING,
			"Unable to set scalar value for object %s!?",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}
	SET_OBJ_FLAG_BITS(dp,DT_ASSIGNED);
}

double cast_from_scalar_value(QSP_ARG_DECL  Scalar_Value *svp, Precision *prec_p)
{
	return (*(prec_p->cast_to_double_func))(svp);
}

void cast_dbl_to_scalar_value(QSP_ARG_DECL  Scalar_Value *svp, Precision *prec_p,double val)
{
	(*(prec_p->cast_from_double_func))(svp,val);
}

// This function casts to a single component

void cast_dbl_to_cpx_scalar(QSP_ARG_DECL  int index, Scalar_Value *svp, Precision *prec_p,double val)
{
	assert( index >= 0 && index <= 1 );
	(*(prec_p->cast_indexed_type_from_double_func))(svp,index,val);
}


void cast_dbl_to_quat_scalar(QSP_ARG_DECL  int index, Scalar_Value *svp, Precision *prec_p,double val)
{
	assert( index >= 0 && index <= 3 );
	(*(prec_p->cast_indexed_type_from_double_func))(svp,index,val);
}

void cast_dbl_to_color_scalar(QSP_ARG_DECL  int index, Scalar_Value *svp, Precision *prec_p,double val)
{
	assert( index >= 0 && index <= 2 );
	(*(prec_p->cast_indexed_type_from_double_func))(svp,index,val);
}

void extract_scalar_value(QSP_ARG_DECL  Scalar_Value *svp, Data_Obj *dp)
{
	Precision *prec_p;

	if( ! OBJ_IS_RAM(dp) ){
		index_t offset;

		offset = OBJ_OFFSET(dp) * PREC_SIZE( OBJ_PREC_PTR(dp) );
		// BUG may not be cuda, use platform-specific function!
		( * PF_MEM_DNLOAD_FN(OBJ_PLATFORM(dp)) )
			(QSP_ARG  &svp->u_d, OBJ_DATA_PTR(dp),
				PREC_SIZE(OBJ_PREC_PTR(dp)), offset, OBJ_PFDEV(dp) );
		return;
	}

	prec_p = OBJ_PREC_PTR(dp);
	(*(prec_p->extract_scalar_func))(svp,dp);
}


Data_Obj *
mk_cscalar(QSP_ARG_DECL  const char *name,double rval,double ival)
{
	Data_Obj *dp;

	dp=make_obj(name,1,1,1,2,prec_for_code(PREC_SP));
	if( dp != NULL ){
		*((float *)OBJ_DATA_PTR(dp)) = (float)rval;
		*( ((float *)OBJ_DATA_PTR(dp)) + 1 ) = (float)ival;
		SET_OBJ_FLAG_BITS(dp,DT_COMPLEX);
	}
	return(dp);
}

Data_Obj *
_mk_img(QSP_ARG_DECL  const char *name,dimension_t rows,dimension_t cols,dimension_t type_dim,Precision *prec_p)		/**/
{
	return( make_obj(name,1,rows,cols,type_dim,prec_p) );
}


Data_Obj *
mk_vec(QSP_ARG_DECL  const char *name,dimension_t dim,dimension_t type_dim,Precision *prec_p)		/**/
{
	return( make_obj(name,1,1,dim,type_dim,prec_p) );
}

/* what is this for? half size?? */

Data_Obj *
dup_half(QSP_ARG_DECL  Data_Obj *dp,const char *name)
{
	Data_Obj *dp2;

	if( !IS_IMAGE(dp) ){
		sprintf(ERROR_STRING,"dup_half:  \"%s\" is not an image",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return(NULL);
	}
	dp2=make_obj(name,1,(OBJ_ROWS(dp))>>1,(OBJ_COLS(dp))>>1,
			OBJ_COMPS(dp),OBJ_PREC_PTR(dp));
	return(dp2);
}

Data_Obj *
dup_dbl(QSP_ARG_DECL  Data_Obj *dp,const char *name)
{
	Data_Obj *dp2;

	if( !IS_IMAGE(dp) ){
		sprintf(ERROR_STRING,"dup_half:  \"%s\" is not an image",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return(NULL);
	}
	dp2=make_obj(name,1,(OBJ_ROWS(dp))<<1,(OBJ_COLS(dp))<<1,
			OBJ_COMPS(dp),OBJ_PREC_PTR(dp));
	return(dp2);
}

/* Create a new object with the given name, having the same size
 * and type as the supplied object.
 * Returns a pointer to the new object.
 */

Data_Obj *
dup_obj(QSP_ARG_DECL  Data_Obj *dp,const char *name)
{
	Data_Obj *dp2;

	dp2=make_obj(name,OBJ_FRAMES(dp),OBJ_ROWS(dp),OBJ_COLS(dp),
			OBJ_COMPS(dp),OBJ_PREC_PTR(dp));
	return(dp2);
}

const char *localname(void)
{
	static short localn=0;
	char buf[32];
	const char *s;

	localn++;
	sprintf(buf,"L.%d",localn);
	s = savestr(buf);
	return s;
}

/* Make an object of the same size and type, with an arbitrary, unique name 
 * Returns a pointer to the new object.
 */

Data_Obj *
dupdp(QSP_ARG_DECL  Data_Obj *dp)
{
	Data_Obj *new_dp;
	new_dp = dup_obj(QSP_ARG  dp,localname());
	assert(new_dp!=NULL);
	return new_dp;
}

int is_valid_dname(QSP_ARG_DECL  const char *name)
{
	const char *s=name;
	while( *s ){
		if( !isalnum(*s) && !is_in_string(*s,DNAME_VALID) ){
			sprintf(ERROR_STRING,
			"illegal character '%c' (0x%x) in data name (\"%s\")",
				*s,*s,name);
			WARN(ERROR_STRING);
			return(0);
		}
		s++;
	}
	return(1);
}

/* Return a pointer to the element in the object specified by a separate
 * index for each dimension.
 * 
 * The increments are inherited from the parent image by subimages, so
 * that this will work for non-contiguous objects.
 */

void *multiply_indexed_data(Data_Obj *dp, dimension_t *index_array)
{
	char *cp;
	dimension_t offset;
	int i;

	cp = (char *)OBJ_DATA_PTR(dp);

	offset = 0;
	for(i=0;i<N_DIMENSIONS;i++)
		offset += index_array[i] * OBJ_MACH_INC(dp,i);

	return( cp + offset*OBJ_PREC_MACH_SIZE( dp ) );
}

/* Return a pointer to the Nth element in the object.  This isn't just
 * N added to the base pointer, because the object may not be contiguous.
 */

void *indexed_data(Data_Obj *dp, dimension_t offset )
{
	dimension_t index_array[N_DIMENSIONS];
	dimension_t count_array[N_DIMENSIONS];
	dimension_t remainder;
	int i;

	count_array[0] = OBJ_MACH_DIM(dp,0);
	for(i=1;i<N_DIMENSIONS;i++)
		count_array[i] = count_array[i-1] * OBJ_MACH_DIM(dp,i-1);

	index_array[N_DIMENSIONS-1] =  offset / count_array[N_DIMENSIONS-1];
	remainder = offset - index_array[N_DIMENSIONS-1] * count_array[N_DIMENSIONS-1];
	for(i=N_DIMENSIONS-2;i>=0;i--){
		index_array[i] =  remainder / count_array[i];
		remainder -= index_array[i] * count_array[i];
	}

	return( multiply_indexed_data(dp,index_array) );
}

