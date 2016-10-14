#include "quip_config.h"

#include <stdio.h>
#include <ctype.h>

#include "quip_prot.h"
#include "data_obj.h"
#include "debug.h"

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

//dp2=dobj_of(QSP_ARG  OBJ_NAME(dp));
//if( dp2 == NO_OBJ ){
//sprintf(ERROR_STRING,"CAUTIOUS:  obj_rename:  object %s has already been removed from the database",
//OBJ_NAME(dp));
//WARN(ERROR_STRING);
//}

	// We expect that the passed object is in the namespace.
	assert( dobj_of(QSP_ARG  OBJ_NAME(dp)) != NO_OBJ );

	dp2=dobj_of(QSP_ARG  newname);
	if( dp2 != NO_OBJ ){
		sprintf(ERROR_STRING,
			"name \"%s\" is already in use in area \"%s\"",
			newname,AREA_NAME( OBJ_AREA(dp) ) );
		WARN(ERROR_STRING);
		return(-1);
	}
	/* BUG?  where is the object's node? */
	//del_item(QSP_ARG  dobj_itp,dp);
	DELETE_OBJ_ITEM(dp);

	rls_str((char *)OBJ_NAME(dp));	/* release old name (obj_rename) */
	SET_OBJ_NAME(dp,savestr(newname));

	/* now add this to the database */
	/* We might have a memory leak, with the item node? */
	//add_item(QSP_ARG  dobj_itp,dp,NO_NODE);
	ADD_OBJ_ITEM(dp);

	return(0);
}

/* this routine is made obsolete by make_dobj */

Data_Obj *
make_obj(QSP_ARG_DECL  const char *name,
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

	dp = make_dobj(QSP_ARG  name,dsp,prec_p);

	RELEASE_DIMSET(dsp);

	return(dp);
}

/* What is a list object?
 */

Data_Obj * make_obj_list(QSP_ARG_DECL  const char *name, List *lp)
{
	Data_Obj *dp;
    	Data_Obj **dp_tbl;
	Dimension_Set *dsp;
	Precision * prec_p;
	Node *np;
	int uk_leaf=0;

	INIT_DIMSET_PTR(dsp)

	dp = dobj_of(QSP_ARG  name);
	if( dp != NO_OBJ ){
		sprintf(ERROR_STRING,"make_obj_list:  object %s already exists!?",name);
		WARN(ERROR_STRING);
		return(NO_OBJ);
	}

	SET_DIMENSION(dsp,0,1);
	SET_DIMENSION(dsp,1,eltcount(lp));
	if( DIMENSION(dsp,1) < 1 ){
		sprintf(ERROR_STRING,"make_obj_list %s:  object list has no elements!?",name);
		WARN(ERROR_STRING);
		return(NO_OBJ);
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
		ERROR1("Unexpected pointer size!?");
	}

	dp = make_dobj(QSP_ARG  name,dsp,prec_p);	/* specify prec_long because sizeof(long) == sizeof(dp) */

	SET_OBJ_PREC_PTR(dp,NULL);

	dp_tbl=(Data_Obj **)OBJ_DATA_PTR(dp);

	np=QLIST_HEAD(lp);
	while(np!=NO_NODE){
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

	set_shape_flags( OBJ_SHAPE(dp),dp,AUTO_SHAPE);

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

	dp=make_obj(QSP_ARG  name,1,1,1,1,prec_p);
	return(dp);
}

// Doesn't support CUDA???

void assign_scalar(QSP_ARG_DECL  Data_Obj *dp,Scalar_Value *svp)
{
//#ifdef CAUTIOUS
//	if( svp == NULL ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  assign_scalar:  passed null scalar ptr");
//		WARN(ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( svp != NULL );

#ifdef HAVE_ANY_GPU
	if( ! OBJ_IS_RAM(dp) ){
		// BUG may not be cuda, use platform-specific function!
		(*(PF_MEM_UPLOAD_FN(PFDEV_PLATFORM(OBJ_PFDEV(dp)))))
			(QSP_ARG  OBJ_DATA_PTR(dp), &svp->u_d, PREC_SIZE(OBJ_PREC_PTR(dp)),
				OBJ_PFDEV(dp) );
		return;
	}
#endif // HAVE_ANY_GPU

	switch( OBJ_PREC(dp) ){
		case PREC_BY:  *((char     *)OBJ_DATA_PTR(dp)) = svp->u_b ; break;
		case PREC_IN:  *((short    *)OBJ_DATA_PTR(dp)) = svp->u_s ; break;
		case PREC_DI:  *((int32_t  *)OBJ_DATA_PTR(dp)) = svp->u_l ; break;
		case PREC_LI:  *((int64_t  *)OBJ_DATA_PTR(dp)) = svp->u_ll; break;
		case PREC_CHAR:
		case PREC_UBY: *((u_char   *)OBJ_DATA_PTR(dp)) = svp->u_ub; break;
		case PREC_UIN: *((u_short  *)OBJ_DATA_PTR(dp)) = svp->u_us; break;
		case PREC_UDI: *((uint32_t *)OBJ_DATA_PTR(dp)) = svp->u_ul; break;
		case PREC_ULI: *((uint64_t *)OBJ_DATA_PTR(dp)) = svp->u_ull; break;

		case PREC_SP: *((float  *)OBJ_DATA_PTR(dp)) = svp->u_f ; break;
		case PREC_DP: *((double *)OBJ_DATA_PTR(dp)) = svp->u_d; break;

		case PREC_CPX:
			*( (float  *)OBJ_DATA_PTR(dp)  ) = svp->u_fc[0];
			*(((float *)OBJ_DATA_PTR(dp))+1) = svp->u_fc[1];
			break;

		case PREC_QUAT:
			*( (float  *)OBJ_DATA_PTR(dp)  ) = svp->u_fq[0];
			*(((float *)OBJ_DATA_PTR(dp))+1) = svp->u_fq[1];
			*(((float *)OBJ_DATA_PTR(dp))+2) = svp->u_fq[2];
			*(((float *)OBJ_DATA_PTR(dp))+3) = svp->u_fq[3];
			break;

		case PREC_DBLCPX:
			*( (double *)OBJ_DATA_PTR(dp)   ) = svp->u_dc[0];
			*(((double *)OBJ_DATA_PTR(dp))+1) = svp->u_dc[1];
			break;
			break;
		case PREC_BIT:
			if( svp->u_l )
				*( (u_long *)OBJ_DATA_PTR(dp) ) |= 1 << OBJ_BIT0(dp) ;
			else
				*( (u_long *)OBJ_DATA_PTR(dp) ) &= ~( 1 << OBJ_BIT0(dp) );
			break;

		default:
			sprintf(ERROR_STRING,
		"assign_scalar:  unsupported scalar precision %s",OBJ_PREC_NAME(dp));
			NERROR1(ERROR_STRING);
			break;
	}
	SET_OBJ_FLAG_BITS(dp,DT_ASSIGNED);
}

double cast_from_scalar_value(QSP_ARG_DECL  Scalar_Value *svp, Precision *prec_p)
{
	double retval;

	switch( PREC_CODE(prec_p) ){
		case PREC_BY:  retval = svp->u_b; break;
		case PREC_IN:  retval = svp->u_s; break;
		case PREC_DI:  retval = svp->u_l; break;
		case PREC_STR:
		case PREC_CHAR:
		case PREC_UBY: retval = svp->u_ub; break;
		case PREC_UIN: retval = svp->u_us; break;
		case PREC_UDI: retval = svp->u_ul; break;
		case PREC_SP: retval = svp->u_f; break;
		case PREC_DP: retval = svp->u_d; break;
		case PREC_BIT:
			if( svp->u_l )  retval =1;
			else		retval =0;
			break;

		case PREC_CPX:
		case PREC_QUAT:
		case PREC_DBLCPX:
			WARN("cast_from_scalar_value:  can't cast multi-component types to double");
			retval =0;
			break;
//#ifdef CAUTIOUS
		default:
//			WARN("CAUTIOUS:  cast_from_scalar_value:  unrecognized precision");
//			retval =0;
			assert( AERROR("cast_from_scalar_value:  unrecognized precision") );

			break;
//#endif /* CAUTIOUS */
	}
	return(retval);
}

void cast_to_scalar_value(QSP_ARG_DECL  Scalar_Value *svp, Precision *prec_p,double val)
{
	switch( PREC_CODE(prec_p) ){
		case PREC_BY:  svp->u_b = (char) val; break;
		case PREC_IN:  svp->u_s = (short) val; break;
		case PREC_DI:  svp->u_l = (int32_t)val; break;
		case PREC_LI:  svp->u_ll = (int64_t)val; break;
		case PREC_CHAR:
		case PREC_UBY: svp->u_ub = (u_char) val; break;
		case PREC_UIN: svp->u_us = (u_short) val; break;
		case PREC_UDI: svp->u_ul = (uint32_t) val; break;
		case PREC_ULI: svp->u_ull = (uint64_t) val; break;
		case PREC_SP: svp->u_f = (float) val; break;
		case PREC_DP: svp->u_d = val; break;
#ifdef USE_LONG_DOUBLE
		case PREC_LP: svp->u_ld = val; break;
#endif // USE_LONG_DOUBLE
		case PREC_BIT:
			if( val != 0 )
				svp->u_l =  1;
			else
				svp->u_l =  0;
			break;

		case PREC_CPX:
		case PREC_QUAT:
		case PREC_DBLCPX:
			WARN("cast_to_scalar_value:  can't cast to multi-component types from double");
			break;
//#ifdef CAUTIOUS
		default:
//			WARN("CAUTIOUS:  cast_to_scalar_value:  unrecognized precision");
			assert( AERROR("cast_to_scalar_value:  unrecognized precision") );
			break;
//#endif /* CAUTIOUS */
	}
}

void cast_to_cpx_scalar(QSP_ARG_DECL  int index, Scalar_Value *svp, Precision *prec_p,double val)
{
//#ifdef CAUTIOUS
//	if( index < 0 || index > 1 ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  cast_to_cpx_scalar:  index (%d) out of range.",index);
//		WARN(ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( index >= 0 && index <= 1 );

	switch( PREC_CODE(prec_p) & MACH_PREC_MASK ){
		case PREC_SP: svp->u_fc[index] = (float) val; break;
		case PREC_DP: svp->u_dc[index] = val; break;
//#ifdef CAUTIOUS
		default:
//			WARN("CAUTIOUS:  cast_to_cpx_scalar:  unexpected machine precision");
			assert( AERROR("cast_to_cpx_scalar:  unexpected machine precision") );
			break;
//#endif /* CAUTIOUS */
	}
}

void cast_to_quat_scalar(QSP_ARG_DECL  int index, Scalar_Value *svp, Precision *prec_p,double val)
{
//#ifdef CAUTIOUS
//	if( index < 0 || index > 3 ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  cast_to_cpx_scalar:  index (%d) out of range.",index);
//		WARN(ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( index >= 0 && index <= 3 );

	switch( PREC_CODE(prec_p) & MACH_PREC_MASK ){
		case PREC_SP: svp->u_fq[index] = (float) val; break;
		case PREC_DP: svp->u_dq[index] = val; break;
//#ifdef CAUTIOUS
		default:
//			WARN("CAUTIOUS:  cast_to_quat_scalar:  unexpected machine precision");
			assert( AERROR("cast_to_quat_scalar:  unexpected machine precision") );
			break;
//#endif /* CAUTIOUS */
	}
}

void extract_scalar_value(QSP_ARG_DECL  Scalar_Value *svp, Data_Obj *dp)
{
	if( ! OBJ_IS_RAM(dp) ){
		// BUG may not be cuda, use platform-specific function!
		( * PF_MEM_DNLOAD_FN(OBJ_PLATFORM(dp)) )
			(QSP_ARG  &svp->u_d, OBJ_DATA_PTR(dp),
				PREC_SIZE(OBJ_PREC_PTR(dp)), OBJ_PFDEV(dp) );
		return;
	}

	switch( OBJ_PREC(dp) ){
		case PREC_BY:  svp->u_b  = *((char     *)OBJ_DATA_PTR(dp)) ; break;
		case PREC_IN:  svp->u_s  = *((short    *)OBJ_DATA_PTR(dp)) ; break;
		case PREC_DI:  svp->u_l  = *((int32_t     *)OBJ_DATA_PTR(dp)) ; break;
		case PREC_STR:
		case PREC_UBY: svp->u_ub = *((u_char   *)OBJ_DATA_PTR(dp)) ; break;
		case PREC_UIN: svp->u_us = *((u_short  *)OBJ_DATA_PTR(dp)) ; break;
		case PREC_UDI: svp->u_ul = *((uint32_t   *)OBJ_DATA_PTR(dp)) ; break;

		case PREC_SP: svp->u_f = *((float  *)OBJ_DATA_PTR(dp)) ; break;
		case PREC_DP: svp->u_d = *((double *)OBJ_DATA_PTR(dp)) ; break;

		case PREC_CPX:
			svp->u_fc[0] = *( (float  *)OBJ_DATA_PTR(dp)  ) ;
			svp->u_fc[1] = *(((float *)OBJ_DATA_PTR(dp))+1) ;
			break;

		case PREC_QUAT:
			svp->u_fq[0] = *( (float  *)OBJ_DATA_PTR(dp)  ) ;
			svp->u_fq[1] = *(((float *)OBJ_DATA_PTR(dp))+1) ;
			svp->u_fq[2] = *(((float *)OBJ_DATA_PTR(dp))+2) ;
			svp->u_fq[3] = *(((float *)OBJ_DATA_PTR(dp))+3) ;
			break;

		case PREC_DBLCPX:
			svp->u_dc[0] = *( (double *)OBJ_DATA_PTR(dp)   ) ;
			svp->u_dc[1] = *(((double *)OBJ_DATA_PTR(dp))+1) ;
			break;
			break;
		default:
			sprintf(DEFAULT_ERROR_STRING,
		"extract_scalar_value:  unsupported scalar precision %s",OBJ_PREC_NAME(dp));
			NERROR1(DEFAULT_ERROR_STRING);
			break;
	}
}

Data_Obj *
mk_cscalar(QSP_ARG_DECL  const char *name,double rval,double ival)
{
	Data_Obj *dp;

	dp=make_obj(QSP_ARG  name,1,1,1,2,prec_for_code(PREC_SP));
	if( dp != NO_OBJ ){
		*((float *)OBJ_DATA_PTR(dp)) = (float)rval;
		*( ((float *)OBJ_DATA_PTR(dp)) + 1 ) = (float)ival;
		SET_OBJ_FLAG_BITS(dp,DT_COMPLEX);
	}
	return(dp);
}

Data_Obj *
mk_img(QSP_ARG_DECL  const char *name,dimension_t rows,dimension_t cols,dimension_t type_dim,Precision *prec_p)		/**/
{
	return( make_obj(QSP_ARG  name,1,rows,cols,type_dim,prec_p) );
}


Data_Obj *
mk_vec(QSP_ARG_DECL  const char *name,dimension_t dim,dimension_t type_dim,Precision *prec_p)		/**/
{
	return( make_obj(QSP_ARG  name,1,1,dim,type_dim,prec_p) );
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
		return(NO_OBJ);
	}
	dp2=make_obj(QSP_ARG  name,1,(OBJ_ROWS(dp))>>1,(OBJ_COLS(dp))>>1,
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
		return(NO_OBJ);
	}
	dp2=make_obj(QSP_ARG  name,1,(OBJ_ROWS(dp))<<1,(OBJ_COLS(dp))<<1,
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

	dp2=make_obj(QSP_ARG  name,OBJ_FRAMES(dp),OBJ_ROWS(dp),OBJ_COLS(dp),
			OBJ_COMPS(dp),OBJ_PREC_PTR(dp));
	return(dp2);
}

const char *localname(void)
{
	static short localn=0;
	char buf[32];

	localn++;
	sprintf(buf,"L.%d",localn);
	return( savestr(buf) );
}

/* Make an object of the same size and type, with an arbitrary, unique name 
 * Returns a pointer to the new object.
 */

Data_Obj *
dupdp(QSP_ARG_DECL  Data_Obj *dp)
{
	return( dup_obj(QSP_ARG  dp,localname()) );
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

