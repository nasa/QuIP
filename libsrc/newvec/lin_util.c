
#include "quip_config.h"

char VersionId_newvec_lin_util[] = QUIP_VERSION_STRING;

#include <nvf.h>
#include "sp_prot.h"
#include "dp_prot.h"


int same_pixel_type(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2)		/* BUG? needed or redundant? */
{
	if( !dp_same_prec(QSP_ARG  dp1,dp2,"same_pixel_type") ) return(0);

	if( dp1->dt_mach_dim[0] != dp2->dt_mach_dim[0] ){
		sprintf(DEFAULT_ERROR_STRING,"component count mismatch:  %s (%d),  %s (%d)",
			dp1->dt_name,dp2->dt_mach_dim[0],
			dp2->dt_name,dp2->dt_mach_dim[0]);
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	return(1);
}

/* Now handled by outer op */

/* BUG use call_wfunc to allow chaining */

int prodimg(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *rowobj,Data_Obj *colobj)	/** make the product image */
{
	Vec_Obj_Args oargs;

	if( rowobj->dt_cols != dpto->dt_cols ){
		sprintf(DEFAULT_ERROR_STRING,
	"prodimg:  row size mismatch, target %s (%d) and row %s (%d)",
			dpto->dt_name,dpto->dt_cols,rowobj->dt_name,
			rowobj->dt_cols);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	} else if( colobj->dt_rows != dpto->dt_rows ){
		sprintf(DEFAULT_ERROR_STRING,
	"prodimg:  column size mismatch, target %s (%d) and column %s (%d)",
			dpto->dt_name,dpto->dt_rows,colobj->dt_name,
			colobj->dt_rows);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	} else if( !same_pixel_type(QSP_ARG  dpto,rowobj) ){
		NWARN("type/precision mismatch");
		return(-1);
	} else if( !same_pixel_type(QSP_ARG  dpto,colobj) ){
		NWARN("type precision mismatch");
		return(-1);
	}
#ifdef FOOBAR
	else if( ! FLOATING_OBJ(dpto) ){
		NWARN("sorry, only float and double supported for prodimg");
		return(-1);
	} else if( IS_COMPLEX(dpto) || IS_COMPLEX(colobj)
			|| IS_COMPLEX(rowobj) ){
		NWARN("Sorry, complex not supported");
		return(-1);
	}
#endif /* FOOBAR */

	setvarg3(&oargs,dpto,rowobj,colobj);

	vmul(&oargs);
	return(0);
}

/* is_good_for_inner
 *
 * make sure the object is either float or double, and real or complex
 */

static int is_good_for_inner(Data_Obj *dp,const char *func_name)
{
	int retval=1;

#ifdef CAUTIOUS
	if( dp == NO_OBJ ){
		NWARN("CAUTIOUS:  is_good_for_inner passed null object pointer!?");
		return(0);
	}
#endif /* CAUTIOUS */
	if( dp->dt_comps > 1 ){
		sprintf(DEFAULT_ERROR_STRING,"%s:  object %s has %d components (should be 1)",
			func_name,dp->dt_name,dp->dt_comps);
		NWARN(DEFAULT_ERROR_STRING);
		retval=0;
	}
	if( MACHINE_PREC(dp) != PREC_SP && MACHINE_PREC(dp) != PREC_DP ){
		sprintf(DEFAULT_ERROR_STRING,"%s:  object %s has machine prec %s (should be float or double)",
			func_name,dp->dt_name,prec_name[MACHINE_PREC(dp)]);
		NWARN(DEFAULT_ERROR_STRING);
		retval=0;
	}
	return(retval);
}

static int prec_and_type_match(Data_Obj *dp1,Data_Obj *dp2,const char *func_name)
{
	if( dp1->dt_prec != dp2->dt_prec ){
		sprintf(DEFAULT_ERROR_STRING,"Function %s:  precisions of objects %s (%s) and %s (%s) do not match!?",
			func_name,dp1->dt_name,name_for_prec(dp1->dt_prec),dp2->dt_name,name_for_prec(dp2->dt_prec));
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	return(1);
}

/* inner (matrix) product */

void inner(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr1,Data_Obj *dpfr2)
{
	dimension_t _n;		/* dot prod len */
	dimension_t i,j;
	Vec_Obj_Args oargs;
	Dimension_Set sizes={{1,1,1,1,1}};
	index_t dst_indices[N_DIMENSIONS]={0,0,0,0,0};
	index_t src1_indices[N_DIMENSIONS]={0,0,0,0,0};
	index_t src2_indices[N_DIMENSIONS]={0,0,0,0,0};
	Data_Obj *col_dp;

#ifdef CAUTIOUS
	clear_obj_args(&oargs);
#endif /* CAUTIOUS */

	/* The types and precisions should be whatever is allowed by vdot,
	 * which is float, double, real and complex...
	 */

	if( ! is_good_for_inner(dpto,"inner") ) return;
	if( ! is_good_for_inner(dpfr1,"inner") ) return;
	if( ! is_good_for_inner(dpfr2,"inner") ) return;

	/* we need to make sure that the types and precisions MATCH! */
	if( ! prec_and_type_match(dpto,dpfr1,"inner") ) return;
	if( ! prec_and_type_match(dpto,dpfr2,"inner") ) return;

	if( dpto->dt_rows != dpfr1->dt_rows ){
		sprintf(DEFAULT_ERROR_STRING,
	"inner:  dpto %s (%d) and first operand %s (%d) must have same # rows",
			dpto->dt_name,dpto->dt_rows,dpfr1->dt_name,dpfr1->dt_rows);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( dpto->dt_cols != dpfr2->dt_cols ){
		sprintf(DEFAULT_ERROR_STRING,
	"inner:  target %s (%d) and second operand %s (%d) must have same # columns",
			dpto->dt_name,dpto->dt_cols,dpfr2->dt_name,dpfr2->dt_cols);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( dpfr1->dt_cols != dpfr2->dt_rows ){
		sprintf(DEFAULT_ERROR_STRING,
	"inner:  # cols of operand %s (%d) must match # rows of operand %s (%d)",
			dpfr1->dt_name,dpfr1->dt_cols,dpfr2->dt_name,dpfr2->dt_rows);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	_n=dpfr1->dt_cols;		/* the length of each dot product we will compute */

	if( IS_COMPLEX(dpto) )	oargs.oa_argstype = COMPLEX_ARGS;
	else				oargs.oa_argstype = REAL_ARGS;

	/* vdot things it's inputs have the same shape, so if we are taking the inner
	 * product of a column vector with a row vector, we have to transpose one of
	 * the inputs...
	 */

	if( dpfr1->dt_rows > 1 )
		oargs.oa_dp[0]=d_subscript(QSP_ARG  dpfr1,0);	/* subscript first row */
	else
		oargs.oa_dp[0]=dpfr1;			/* object is a row */

	if( dpto->dt_cols > 1 )
		col_dp=c_subscript(QSP_ARG  dpfr2,0);
	else 
		col_dp=dpfr2;

	oargs.oa_dest=mk_subimg(QSP_ARG  dpto,0,0,"target pixel",1,1);

	sizes.ds_dimension[1]=col_dp->dt_rows;
	oargs.oa_dp[1]=make_equivalence(QSP_ARG  "_transposed_column",col_dp,&sizes,col_dp->dt_prec);

	for(i=0;i<dpto->dt_rows;i++){
		src1_indices[2]=i;
		oargs.oa_dp[0]->dt_data = multiply_indexed_data(dpfr1,src1_indices);
		for(j=0;j<dpto->dt_cols;j++){
			dst_indices[2]=i;		/* k_th component */
			dst_indices[1]=j;		/* k_th component */
			oargs.oa_dest->dt_data = multiply_indexed_data(dpto,dst_indices);
			src2_indices[1]=j;
			oargs.oa_dp[1]->dt_data = multiply_indexed_data(dpfr2,src2_indices);
			vdot(&oargs);
		}
	}

	delvec(QSP_ARG  oargs.oa_dp[1]);		/* "_transposed_column" */

	if( oargs.oa_dp[0] != dpfr1 )
		delvec(QSP_ARG  oargs.oa_dp[0]);
	if( col_dp != dpfr2 )
		delvec(QSP_ARG  col_dp);

	delvec(QSP_ARG  oargs.oa_dest);
}

/* Here we assume the matrix acts on vectors in the tdim direction... */

int xform_chk(Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform)
{
	if( dpto==NO_OBJ || dpfr==NO_OBJ || xform==NO_OBJ )
		return(-1);

	if( !IS_IMAGE(xform) ){
		sprintf(DEFAULT_ERROR_STRING,
	"transformation %s must be a matrix (image)",
			xform->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( xform->dt_comps != 1 ){
		sprintf(DEFAULT_ERROR_STRING,
	"transform matrix %s must have single-component elements",xform->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( dpto->dt_comps != xform->dt_rows ){
		sprintf(DEFAULT_ERROR_STRING,
	"target %s component dimension (%d) must match # rows of xform %s (%d)",
			dpto->dt_name,dpto->dt_comps,xform->dt_name,xform->dt_rows);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( dpto->dt_n_type_elts != dpfr->dt_n_type_elts ){
		sprintf(DEFAULT_ERROR_STRING,
	"target %s (%d) and source %s (%d) must have same # elements",
			dpto->dt_name,dpto->dt_n_type_elts,
			dpfr->dt_name,dpfr->dt_n_type_elts);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( xform->dt_cols != dpfr->dt_comps ){
		sprintf(DEFAULT_ERROR_STRING,
	"source %s component dimension (%d) must match # cols of xform %s (%d)",
			dpfr->dt_name,dpfr->dt_comps,xform->dt_name,xform->dt_cols);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	/* BUG these contiguity requirements may no longer be necessary... */

	if( !is_contiguous(dpto) ){
		sprintf(DEFAULT_ERROR_STRING,
			"xform target %s must be contiguous",dpto->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( !is_contiguous(dpfr) ){
		sprintf(DEFAULT_ERROR_STRING,
			"xform source %s must be contiguous",dpfr->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( !is_contiguous(xform) ){
		sprintf(DEFAULT_ERROR_STRING,
			"xform %s must be contiguous",xform->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	return(0);
}

/* apply a matrix to a list of elements */
/* this routine vectorizes the dot products;
	good for big matrices or short lists */

/* there should be a better routine for long lists of short elts. */

void xform_list(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform)
{
	if( xform_chk(dpto,dpfr,xform) == -1 )
		return;

	switch( MACHINE_PREC(dpto) ){
		case PREC_SP:	sp_obj_xform_list(QSP_ARG  dpto,dpfr,xform); break;
		case PREC_DP:	dp_obj_xform_list(QSP_ARG  dpto,dpfr,xform); break;
		default:
			sprintf(DEFAULT_ERROR_STRING,"xform_list:  destination object %s (%s) should have float or double precision",
				dpto->dt_name,name_for_prec(dpto->dt_prec));
			NWARN(DEFAULT_ERROR_STRING);
	}
}

/* like xform_list(), but vectorizes over list instead of matrix row */
/* good for long lists or big images */

void vec_xform(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform)
{
	if( xform_chk(dpto,dpfr,xform) == -1 )
		return;

	switch( MACHINE_PREC(dpto) ){
		case PREC_SP:	sp_obj_vec_xform(QSP_ARG  dpto,dpfr,xform); break;
		case PREC_DP:	dp_obj_vec_xform(QSP_ARG  dpto,dpfr,xform); break;
		default:
			sprintf(ERROR_STRING,"vec_xform:  destination object %s (%s) should have float or double precision",
				dpto->dt_name,name_for_prec(dpto->dt_prec));
			WARN(ERROR_STRING);
	}
}

/*
 * like vec_xform, but does the division for homgenous coords
 */

void homog_xform(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform_dp)
{
	switch(MACHINE_PREC(dpto)){
		case PREC_SP:	sp_obj_homog_xform(QSP_ARG  dpto,dpfr,xform_dp); break;
		case PREC_DP:	dp_obj_homog_xform(QSP_ARG  dpto,dpfr,xform_dp); break;
		default: sprintf(ERROR_STRING,"homog_xform:  unsupported precision");
			WARN(ERROR_STRING);
			break;
	}

} /* end homog_xform */

void unity(Data_Obj *mp)				/**/
{
	dimension_t i,j;
	float *f;

	f=(float *)(mp->dt_data);
	for(i=0;i<mp->dt_cols;i++){
		for(j=0;j<mp->dt_cols;j++){
			if(i==j) *f++ = 1.0;
			else *f++ = 0.0;
		}
	}
}

void newmtrx(QSP_ARG_DECL  const char *s,int dim)
{
	Data_Obj *mp;

	if( dim <= 1 ){
		WARN("bad dimension");
		return;
	}
	mp=dobj_of(QSP_ARG  s);
	if( mp!=(NO_OBJ) ){
		WARN("name in use already");
		return;
	}
	mp=make_obj(QSP_ARG  s,1,dim,dim,1,PREC_SP);
	if( mp == NO_OBJ ){
		WARN("couldn't create new matrix");
		return;
	}
	unity(mp);
}

void transpose(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr)
{
	/* new version using gen_xpose */

	Data_Obj tmpobj;
	Vec_Obj_Args oargs;

	tmpobj = *dpfr;
	gen_xpose(&tmpobj,1,2);		/* switch rows, cols */

	setvarg2(&oargs,dpto,&tmpobj);
	perf_vfunc(QSP_ARG  FVMOV,&oargs);
}

#ifdef FOOBAR

/* Compute the correlation matrix of a bunch of vectors.
 *
 * Assume that the vectors are the row of an image.
 */

void corr_matrix(Data_Obj *dpto,Data_Obj *dpfr)
{
	int had_err=0;
	float *op1, *op2;
	float *dest, *dest2;
	dimension_t i,j;
	Vec_Args args;

	if( ! is_real(dpto,"corr_matrix") ) return;
	if( ! is_real(dpfr,"corr_matrix") ) return;

	if( dpto->dt_cols != dpto->dt_rows ){
		sprintf(ERROR_STRING,"target matrix %s (%dx%d) must be square",dpto->dt_name,
			dpto->dt_rows,dpto->dt_cols);
		WARN(ERROR_STRING);
		had_err++;
	}

	if( dpto->dt_cols != dpfr->dt_rows ){
		sprintf(ERROR_STRING,
	"target matrix %s size %d not equal to source matrix %s rows (%d)",
			dpto->dt_name,dpto->dt_cols,dpfr->dt_name,dpfr->dt_rows);
		WARN(ERROR_STRING);
		had_err++;
	}

	if( had_err ) return;

	if( IS_COMPLEX(dpto) )
		args.arg_argstype = COMPLEX_ARGS;
	else
		args.arg_argstype = REAL_ARGS;

	args.arg_inc1 = dpfr->dt_pinc;
	args.arg_inc2 = dpfr->dt_pinc;
	args.arg_n1 = dpfr->dt_cols;
	args.arg_n2 = dpfr->dt_cols;
	args.arg_prec1 = dpfr->dt_prec;
	args.arg_prec2 = dpfr->dt_prec;

	op1 = dpfr->dt_data;
	for(i=0;i<dpfr->dt_rows;i++){
		dest = dest2 = dpto->dt_data;
		dest += i*dpto->dt_rinc;
		dest += i*dpto->dt_pinc;
		dest2 += i*dpto->dt_pinc;
		dest2 += i*dpto->dt_rinc;
		op2 = dpfr->dt_data;
		op2 += i*dpfr->dt_rinc;
		for(j=i;j<dpfr->dt_rows;j++){

			args.arg_v1 = op1;
			args.arg_v2 = op2;
			vdot(&args);

			*dest2 = *dest;		/* symmetric matrix */

			op2 += dpfr->dt_rinc;
			dest += dpto->dt_pinc;
			dest2 += dpto->dt_rinc;
		}
		op1 += dpfr->dt_rinc;
	}
} /* end corr_matrix() */
#endif /* FOOBAR */

/* Compute the determinant of a square matrix */

double determinant(Data_Obj *dp)
{
	switch(MACHINE_PREC(dp)){
		case PREC_SP:  return sp_obj_determinant(dp);
		case PREC_DP:  return dp_obj_determinant(dp);
		default:
			sprintf(DEFAULT_ERROR_STRING,"determinant:  object %s has unsupported precision %s",
				dp->dt_name,name_for_prec(MACHINE_PREC(dp)));
			NWARN(DEFAULT_ERROR_STRING);
			return(0.0);
	}
	/* NOTREACHED */
	return(0.0);
}
			

