#include "quip_config.h"

char VersionId_dataf_data_fns[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include <ctype.h>

#include "data_obj.h"
#include "items.h"
#include "debug.h"
#include "img_file.h"
#include "savestr.h"

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


void make_contiguous(Data_Obj *dp)
{
	int i;
	int n_type;
	int n_mach;

	n_type = 1;
	n_mach = 1;
	for(i=0;i<N_DIMENSIONS;i++){
		SETUP_INC(type)
		SETUP_INC(mach)
	}

	dp->dt_flags |= DT_CONTIG | DT_CHECKED;
}

/* Rename an existing object.
 * Returns 0 on success, -1 if the new name contains an illegal character,
 * or if an object with the requested name already exists.
 */

int obj_rename(QSP_ARG_DECL  Data_Obj *dp,const char *newname)
{
	Data_Obj *dp2;

	if( !is_valid_dname(QSP_ARG  newname) ) return(-1);

dp2=dobj_of(QSP_ARG  dp->dt_name);
if( dp2 == NO_OBJ ){
sprintf(error_string,"CAUTIOUS:  obj_rename:  object %s has already been removed from the database",
dp->dt_name);
WARN(error_string);
}
	/* BUG?  where is the object's node? */
	del_item(QSP_ARG  dobj_itp,dp);

	dp2=dobj_of(QSP_ARG  newname);
	if( dp2 != NO_OBJ ){
		sprintf(error_string,
			"name \"%s\" is already in use in area \"%s\"",
			newname,dp->dt_ap->da_name);
		WARN(error_string);
		return(-1);
	}
	rls_str((char *)dp->dt_name);	/* release old name */
	dp->dt_name = savestr(newname);

	/* now add this to the database */
	/* We might have a memory leak, with the item node? */
	add_item(QSP_ARG  dobj_itp,dp,NO_NODE);

	return(0);
}

/* this routine is made obsolete by make_dobj */

Data_Obj *
make_obj(QSP_ARG_DECL  const char *name,dimension_t frames,dimension_t rows,dimension_t cols,dimension_t type_dim,prec_t prec)
{
	Data_Obj *dp;
	Dimension_Set dimset;

	dimset.ds_dimension[0]=type_dim;
	dimset.ds_dimension[1]=cols;
	dimset.ds_dimension[2]=rows;
	dimset.ds_dimension[3]=frames;
	dimset.ds_dimension[4]=1;

	dp = make_dobj(QSP_ARG  name,&dimset,prec);

	return(dp);
}

Data_Obj *
make_obj_list(QSP_ARG_DECL  const char *name, List *lp)
{
	Data_Obj *dp, **dp_tbl;
	Dimension_Set dimset;
	Node *np;
	int uk_leaf=0;

	dp = dobj_of(QSP_ARG  name);
	if( dp != NO_OBJ ){
		sprintf(error_string,"make_obj_list:  object %s already exists!?",name);
		WARN(error_string);
		return(NO_OBJ);
	}

	dimset.ds_dimension[0]=1;
	dimset.ds_dimension[1]=eltcount(lp);
	if( dimset.ds_dimension[1] < 1 ){
		sprintf(error_string,"make_obj_list %s:  object list has no elements!?",name);
		WARN(error_string);
		return(NO_OBJ);
	}

	dimset.ds_dimension[2]=1;
	dimset.ds_dimension[3]=1;
	dimset.ds_dimension[4]=1;

	dp = make_dobj(QSP_ARG  name,&dimset,PREC_DI);	/* specify prec_long because sizeof(long) == sizeof(dp) */

	dp->dt_prec = PREC_NONE;

	dp_tbl=(Data_Obj **)dp->dt_data;

	np=lp->l_head;
	while(np!=NO_NODE){
		*dp_tbl = (Data_Obj *) np->n_data;
		if( UNKNOWN_SHAPE( &(*dp_tbl)->dt_shape ) )
			uk_leaf++;
		dp_tbl++;
		np=np->n_next;
	}

	if( dp_tbl != dp->dt_data ){	/* one or more objects? */
		dp_tbl=(Data_Obj **)dp->dt_data;
		dp->dt_prec = (*dp_tbl)->dt_prec;
	}

	set_shape_flags(&dp->dt_shape,dp,AUTO_SHAPE);

	if( uk_leaf ){
		dp->dt_flags &= SHAPE_DIM_MASK;
		dp->dt_flags |= DT_UNKNOWN_SHAPE;
	}
	dp->dt_flags |= DT_OBJ_LIST;

	return(dp);
}

Data_Obj *mk_scalar(QSP_ARG_DECL  const char *name,prec_t prec)
{
	Data_Obj *dp;

	dp=make_obj(QSP_ARG  name,1,1,1,1,prec);
	return(dp);
}

void assign_scalar(QSP_ARG_DECL  Data_Obj *dp,Scalar_Value *svp)
{
#ifdef CAUTIOUS
	if( svp == NULL ){
		sprintf(error_string,"CAUTIOUS:  assign_scalar:  passed null scalar ptr");
		WARN(error_string);
		return;
	}
#endif /* CAUTIOUS */

	switch( dp->dt_prec ){
		case PREC_BY:  *((char     *)dp->dt_data) = svp->u_b ; break;
		case PREC_IN:  *((short    *)dp->dt_data) = svp->u_s ; break;
		case PREC_DI:  *((long     *)dp->dt_data) = svp->u_l ; break;
		case PREC_CHAR:
		case PREC_UBY: *((u_char   *)dp->dt_data) = svp->u_ub; break;
		case PREC_UIN: *((u_short  *)dp->dt_data) = svp->u_us; break;
		case PREC_UDI: *((u_long   *)dp->dt_data) = svp->u_ul; break;

		case PREC_SP: *((float  *)dp->dt_data) = svp->u_f ; break;
		case PREC_DP: *((double *)dp->dt_data) = svp->u_d; break;

		case PREC_CPX:
			*( (float  *)dp->dt_data  ) = svp->u_fc[0];
			*(((float *)dp->dt_data)+1) = svp->u_fc[1];
			break;

		case PREC_QUAT:
			*( (float  *)dp->dt_data  ) = svp->u_fq[0];
			*(((float *)dp->dt_data)+1) = svp->u_fq[1];
			*(((float *)dp->dt_data)+2) = svp->u_fq[2];
			*(((float *)dp->dt_data)+3) = svp->u_fq[3];
			break;

		case PREC_DBLCPX:
			*( (double *)dp->dt_data   ) = svp->u_dc[0];
			*(((double *)dp->dt_data)+1) = svp->u_dc[1];
			break;
			break;
		case PREC_BIT:
			if( svp->u_l )
				*( (u_long *)dp->dt_data ) |= 1 << dp->dt_bit0 ;
			else
				*( (u_long *)dp->dt_data ) &= ~( 1 << dp->dt_bit0 );
			break;

		default:
			sprintf(error_string,
		"assign_scalar:  unsupported scalar precision %s",name_for_prec(dp->dt_prec));
			NERROR1(error_string);
			break;
	}
	dp->dt_flags |= DT_ASSIGNED;
}

double cast_from_scalar_value(QSP_ARG_DECL  Scalar_Value *svp, prec_t prec)
{
	double retval;

	switch( prec ){
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
#ifdef CAUTIOUS
		default:
			WARN("CAUTIOUS:  cast_from_scalar_value:  unrecognized precision");
			retval =0;
			break;
#endif /* CAUTIOUS */
	}
	return(retval);
}

void cast_to_scalar_value(QSP_ARG_DECL  Scalar_Value *svp, prec_t prec,double val)
{
	switch( prec ){
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
		case PREC_BIT:
			if( val )
				svp->u_l =  1;
			else
				svp->u_l =  0;
			break;

		case PREC_CPX:
		case PREC_QUAT:
		case PREC_DBLCPX:
			WARN("cast_to_scalar_value:  can't cast to multi-component types from double");
			break;
#ifdef CAUTIOUS
		default:
			WARN("CAUTIOUS:  cast_to_scalar_value:  unrecognized precision");
			break;
#endif /* CAUTIOUS */
	}
}

void cast_to_cpx_scalar(QSP_ARG_DECL  int index, Scalar_Value *svp, prec_t prec,double val)
{
#ifdef CAUTIOUS
	if( index < 0 || index > 1 ){
		sprintf(error_string,"CAUTIOUS:  cast_to_cpx_scalar:  index (%d) out of range.",index);
		WARN(error_string);
		return;
	}
#endif /* CAUTIOUS */

	switch( prec & MACH_PREC_MASK ){
		case PREC_SP: svp->u_fc[index] = (float) val; break;
		case PREC_DP: svp->u_dc[index] = val; break;
#ifdef CAUTIOUS
		default:
			WARN("CAUTIOUS:  cast_to_cpx_scalar:  unexpected machine precision");
			break;
#endif /* CAUTIOUS */
	}
}

void cast_to_quat_scalar(QSP_ARG_DECL  int index, Scalar_Value *svp, prec_t prec,double val)
{
#ifdef CAUTIOUS
	if( index < 0 || index > 3 ){
		sprintf(error_string,"CAUTIOUS:  cast_to_cpx_scalar:  index (%d) out of range.",index);
		WARN(error_string);
		return;
	}
#endif /* CAUTIOUS */

	switch( prec & MACH_PREC_MASK ){
		case PREC_SP: svp->u_fq[index] = (float) val; break;
		case PREC_DP: svp->u_dq[index] = val; break;
#ifdef CAUTIOUS
		default:
			WARN("CAUTIOUS:  cast_to_quat_scalar:  unexpected machine precision");
			break;
#endif /* CAUTIOUS */
	}
}

void extract_scalar_value(Scalar_Value *svp, Data_Obj *dp)
{
	switch( dp->dt_prec ){
		case PREC_BY:  svp->u_b  = *((char     *)dp->dt_data) ; break;
		case PREC_IN:  svp->u_s  = *((short    *)dp->dt_data) ; break;
		case PREC_DI:  svp->u_l  = *((long     *)dp->dt_data) ; break;
		case PREC_STR:
		case PREC_UBY: svp->u_ub = *((u_char   *)dp->dt_data) ; break;
		case PREC_UIN: svp->u_us = *((u_short  *)dp->dt_data) ; break;
		case PREC_UDI: svp->u_ul = *((u_long   *)dp->dt_data) ; break;

		case PREC_SP: svp->u_f = *((float  *)dp->dt_data) ; break;
		case PREC_DP: svp->u_d = *((double *)dp->dt_data) ; break;

		case PREC_CPX:
			svp->u_fc[0] = *( (float  *)dp->dt_data  ) ;
			svp->u_fc[1] = *(((float *)dp->dt_data)+1) ;
			break;

		case PREC_QUAT:
			svp->u_fq[0] = *( (float  *)dp->dt_data  ) ;
			svp->u_fq[1] = *(((float *)dp->dt_data)+1) ;
			svp->u_fq[2] = *(((float *)dp->dt_data)+2) ;
			svp->u_fq[3] = *(((float *)dp->dt_data)+3) ;
			break;

		case PREC_DBLCPX:
			svp->u_dc[0] = *( (double *)dp->dt_data   ) ;
			svp->u_dc[1] = *(((double *)dp->dt_data)+1) ;
			break;
			break;
		default:
			sprintf(DEFAULT_ERROR_STRING,
		"extract_scalar_value:  unsupported scalar precision %s",name_for_prec(dp->dt_prec));
			NERROR1(DEFAULT_ERROR_STRING);
			break;
	}
}

Data_Obj *
mk_cscalar(QSP_ARG_DECL  const char *name,double rval,double ival)
{
	Data_Obj *dp;

	dp=make_obj(QSP_ARG  name,1,1,1,2,PREC_SP);
	if( dp != NO_OBJ ){
		*((float *)dp->dt_data) = (float)rval;
		*( ((float *)dp->dt_data) + 1 ) = (float)ival;
		dp->dt_flags |= DT_COMPLEX;
	}
	return(dp);
}

Data_Obj *
mk_img(QSP_ARG_DECL  const char *name,dimension_t rows,dimension_t cols,dimension_t type_dim,prec_t prec)		/**/
{
	return( make_obj(QSP_ARG  name,1,rows,cols,type_dim,prec) );
}


Data_Obj *
mk_vec(QSP_ARG_DECL  const char *name,dimension_t dim,dimension_t type_dim,prec_t prec)		/**/
{
	return( make_obj(QSP_ARG  name,1,1,dim,type_dim,prec) );
}

/* what is this for? half size?? */

Data_Obj *
dup_half(QSP_ARG_DECL  Data_Obj *dp,const char *name)
{
	Data_Obj *dp2;

	if( !IS_IMAGE(dp) ){
		sprintf(error_string,"dup_half:  \"%s\" is not an image",
			dp->dt_name);
		WARN(error_string);
		return(NO_OBJ);
	}
	dp2=make_obj(QSP_ARG  name,1,(dp->dt_rows)>>1,(dp->dt_cols)>>1,
			dp->dt_comps,dp->dt_prec);
	return(dp2);
}

Data_Obj *
dup_dbl(QSP_ARG_DECL  Data_Obj *dp,const char *name)
{
	Data_Obj *dp2;

	if( !IS_IMAGE(dp) ){
		sprintf(error_string,"dup_half:  \"%s\" is not an image",
			dp->dt_name);
		WARN(error_string);
		return(NO_OBJ);
	}
	dp2=make_obj(QSP_ARG  name,1,(dp->dt_rows)<<1,(dp->dt_cols)<<1,
			dp->dt_comps,dp->dt_prec);
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

	dp2=make_obj(QSP_ARG  name,dp->dt_frames,dp->dt_rows,dp->dt_cols,
			dp->dt_comps,dp->dt_prec);
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

int is_valid_dname(QSP_ARG_DECL  const char *s)
{
	while( *s ){
		if( !isalnum(*s) && !is_in_string(*s,DNAME_VALID) ){
			sprintf(error_string,
			"illegal character '%c' (0x%x) in data name",*s,*s);
			WARN(error_string);
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

	cp = (char *)dp->dt_data;

	offset = 0;
	for(i=0;i<N_DIMENSIONS;i++)
		offset += index_array[i] * dp->dt_mach_inc[i];

	return( cp + offset*siztbl[ MACHINE_PREC(dp) ] );
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

	count_array[0] = dp->dt_mach_dim[0];
	for(i=1;i<N_DIMENSIONS;i++)
		count_array[i] = count_array[i-1] * dp->dt_mach_dim[i-1];

	index_array[N_DIMENSIONS-1] =  offset / count_array[N_DIMENSIONS-1];
	remainder = offset - index_array[N_DIMENSIONS-1] * count_array[N_DIMENSIONS-1];
	for(i=N_DIMENSIONS-2;i>=0;i--){
		index_array[i] =  remainder / count_array[i];
		remainder -= index_array[i] * count_array[i];
	}

	return( multiply_indexed_data(dp,index_array) );
}

