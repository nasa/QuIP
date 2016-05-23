
#include "quip_config.h"
#include "nvf.h"
#include "veclib_prot.h"
#include "veclib/vl2_veclib_prot.h"


static void vdot(/*HOST_CALL_ARG_DECLS*/ QSP_ARG_DECL  Vec_Obj_Args *oap )
{
	// BUG determine proper platform
	//h_vl2_vdot(HOST_CALL_ARGS);
	platform_dispatch_by_code(QSP_ARG  FVDOT, oap);
}

static void vmul(/*HOST_CALL_ARG_DECLS*/ QSP_ARG_DECL  Vec_Obj_Args *oap )
{
	//h_vl2_vmul(HOST_CALL_ARGS);
	platform_dispatch_by_code(QSP_ARG  FVMUL, oap);
}

static int same_pixel_type(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2);

static int same_pixel_type(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2)		/* BUG? needed or redundant? */
{
	if( !dp_same_prec(QSP_ARG  dp1,dp2,"same_pixel_type") ) return(0);

	if( OBJ_MACH_DIM(dp1,0) != OBJ_MACH_DIM(dp2,0) ){
		sprintf(DEFAULT_ERROR_STRING,"component count mismatch:  %s (%d),  %s (%d)",
			OBJ_NAME(dp1),OBJ_MACH_DIM(dp2,0),
			OBJ_NAME(dp2),OBJ_MACH_DIM(dp2,0));
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	return(1);
}

/* Now handled by outer op */

/* BUG use call_wfunc to allow chaining */

int prodimg(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *rowobj,Data_Obj *colobj)	/** make the product image */
{
	Vec_Obj_Args oa1, *oap=&oa1;

	if( OBJ_COLS(rowobj) != OBJ_COLS(dpto) ){
		sprintf(DEFAULT_ERROR_STRING,
	"prodimg:  row size mismatch, target %s (%d) and row %s (%d)",
			OBJ_NAME(dpto),OBJ_COLS(dpto),OBJ_NAME(rowobj),
			OBJ_COLS(rowobj));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	} else if( OBJ_ROWS(colobj) != OBJ_ROWS(dpto) ){
		sprintf(DEFAULT_ERROR_STRING,
	"prodimg:  column size mismatch, target %s (%d) and column %s (%d)",
			OBJ_NAME(dpto),OBJ_ROWS(dpto),OBJ_NAME(colobj),
			OBJ_ROWS(colobj));
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

	setvarg3(oap,dpto,rowobj,colobj);

	vmul(QSP_ARG  oap);
	return(0);
}

/* is_good_for_inner
 *
 * make sure the object is either float or double, and real or complex
 */

static int is_good_for_inner(Data_Obj *dp,const char *func_name)
{
	int retval=1;

//#ifdef CAUTIOUS
//	if( dp == NO_OBJ ){
//		NWARN("CAUTIOUS:  is_good_for_inner passed null object pointer!?");
//		return(0);
//	}
//#endif /* CAUTIOUS */
	assert( dp != NO_OBJ );

	if( OBJ_COMPS(dp) > 1 ){
		sprintf(DEFAULT_ERROR_STRING,"%s:  object %s has %d components (should be 1)",
			func_name,OBJ_NAME(dp),OBJ_COMPS(dp));
		NWARN(DEFAULT_ERROR_STRING);
		retval=0;
	}
	if( OBJ_MACH_PREC(dp) != PREC_SP && OBJ_MACH_PREC(dp) != PREC_DP ){
		sprintf(DEFAULT_ERROR_STRING,"%s:  object %s has machine prec %s (should be float or double)",
			func_name,OBJ_NAME(dp),OBJ_MACH_PREC_NAME(dp) );
		NWARN(DEFAULT_ERROR_STRING);
		retval=0;
	}
	return(retval);
}

static int prec_and_type_match(Data_Obj *dp1,Data_Obj *dp2,const char *func_name)
{
	if( OBJ_PREC(dp1) != OBJ_PREC(dp2) ){
		sprintf(DEFAULT_ERROR_STRING,"Function %s:  precisions of objects %s (%s) and %s (%s) do not match!?",
			func_name,OBJ_NAME(dp1),OBJ_PREC_NAME(dp1),OBJ_NAME(dp2),OBJ_PREC_NAME(dp2));
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	return(1);
}

/* inner (matrix) product */

void inner(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr1,Data_Obj *dpfr2)
{
	//dimension_t _n;		/* dot prod len */
	dimension_t i,j;
	Vec_Obj_Args oa1, *oap=&oa1;
	//Dimension_Set sizes={{1,1,1,1,1}};
	Dimension_Set *sizes;
	index_t dst_indices[N_DIMENSIONS]={0,0,0,0,0};
	index_t src1_indices[N_DIMENSIONS]={0,0,0,0,0};
	index_t src2_indices[N_DIMENSIONS]={0,0,0,0,0};
	Data_Obj *col_dp;

	sizes=NEW_DIMSET;
	for(i=0;i<N_DIMENSIONS;i++)
		SET_DIMENSION(sizes,i,1);

#ifdef CAUTIOUS
	clear_obj_args(oap);
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

	if( OBJ_ROWS(dpto) != OBJ_ROWS(dpfr1) ){
		sprintf(DEFAULT_ERROR_STRING,
	"inner:  dpto %s (%d) and first operand %s (%d) must have same # rows",
			OBJ_NAME(dpto),OBJ_ROWS(dpto),OBJ_NAME(dpfr1),OBJ_ROWS(dpfr1));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( OBJ_COLS(dpto) != OBJ_COLS(dpfr2) ){
		sprintf(DEFAULT_ERROR_STRING,
	"inner:  target %s (%d) and second operand %s (%d) must have same # columns",
			OBJ_NAME(dpto),OBJ_COLS(dpto),OBJ_NAME(dpfr2),OBJ_COLS(dpfr2));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( OBJ_COLS(dpfr1) != OBJ_ROWS(dpfr2) ){
		sprintf(DEFAULT_ERROR_STRING,
	"inner:  # cols of operand %s (%d) must match # rows of operand %s (%d)",
			OBJ_NAME(dpfr1),OBJ_COLS(dpfr1),OBJ_NAME(dpfr2),OBJ_ROWS(dpfr2));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	//_n=OBJ_COLS(dpfr1);		/* the length of each dot product we will compute */

#ifdef FOOBAR
	// this stuff is encapsulated into set_obj_arg_flags
	if( IS_COMPLEX(dpto) )	SET_OA_ARGSTYPE(oap,COMPLEX_ARGS);
	else			SET_OA_ARGSTYPE(oap,REAL_ARGS);

	SET_OA_ARGSPREC(oap,ARGSET_PREC( OBJ_MACH_PREC(dpto) ) );
#endif // FOOBAR
	set_obj_arg_flags(oap);

	/* vdot things it's inputs have the same shape, so if we are taking the inner
	 * product of a column vector with a row vector, we have to transpose one of
	 * the inputs...
	 */

	if( OBJ_ROWS(dpfr1) > 1 )
		SET_OA_SRC1(oap,d_subscript(QSP_ARG  dpfr1,0) );	/* subscript first row */
	else
		SET_OA_SRC1(oap,dpfr1);			/* object is a row */

	if( OBJ_COLS(dpto) > 1 )
		col_dp=c_subscript(QSP_ARG  dpfr2,0);
	else 
		col_dp=dpfr2;

	SET_OA_DEST(oap,mk_subimg(QSP_ARG  dpto,0,0,"target pixel",1,1) );

	//[sizes setDimensionAtIndex : 1 withValue : OBJ_ROWS(col_dp) ];
	SET_DIMENSION(sizes,1,OBJ_ROWS(col_dp));
	SET_DIMENSION(sizes,0,OBJ_COMPS(col_dp));

	SET_OA_SRC2(oap,make_equivalence(QSP_ARG  "_transposed_column",
						col_dp,sizes,OBJ_PREC_PTR(col_dp)) );

	for(i=0;i<OBJ_ROWS(dpto);i++){
		src1_indices[2]=i;
		SET_OBJ_DATA_PTR( OA_SRC1(oap), multiply_indexed_data(dpfr1,src1_indices) );
		for(j=0;j<OBJ_COLS(dpto);j++){
			dst_indices[2]=i;		/* k_th component */
			dst_indices[1]=j;		/* k_th component */
			SET_OBJ_DATA_PTR( OA_DEST(oap), multiply_indexed_data(dpto,dst_indices) );
			src2_indices[1]=j;
			SET_OBJ_DATA_PTR( OA_SRC2(oap), multiply_indexed_data(dpfr2,src2_indices) );
			vdot(QSP_ARG  oap);
		}
	}

	delvec(QSP_ARG  OA_SRC2(oap) );		/* "_transposed_column" */

	if( OA_SRC1(oap) != dpfr1 )
		delvec(QSP_ARG  OA_SRC1(oap) );
	if( col_dp != dpfr2 )
		delvec(QSP_ARG  col_dp);

	delvec(QSP_ARG  OA_DEST(oap) );
}

/* Here we assume the matrix acts on vectors in the tdim direction...
 */

int xform_chk(Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform)
{
	if( dpto==NO_OBJ || dpfr==NO_OBJ || xform==NO_OBJ )
		return(-1);

	if( !IS_IMAGE(xform) ){
		sprintf(DEFAULT_ERROR_STRING,
	"xform_chk:  transformation %s must be a matrix (image)",
			OBJ_NAME(xform));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( OBJ_COMPS(xform) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,
	"xform_chk:  transform matrix %s must have single-component elements",OBJ_NAME(xform));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( OBJ_COMPS(dpto) != OBJ_ROWS(xform) ){
		sprintf(DEFAULT_ERROR_STRING,
	"xform_chk:  target %s component dimension (%d) must match # rows of xform %s (%d)",
			OBJ_NAME(dpto),OBJ_COMPS(dpto),OBJ_NAME(xform),OBJ_ROWS(xform));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( OBJ_COMPS(dpfr) != OBJ_COLS(xform) ){
		sprintf(DEFAULT_ERROR_STRING,
	"xform_chk:  source %s component dimension (%d) must match # columns of xform %s (%d)",
			OBJ_NAME(dpto),OBJ_COMPS(dpto),OBJ_NAME(xform),OBJ_ROWS(xform));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( OBJ_N_TYPE_ELTS(dpto)/OBJ_COMPS(dpto) != OBJ_N_TYPE_ELTS(dpfr)/OBJ_COMPS(dpfr) ){
		sprintf(DEFAULT_ERROR_STRING,
	"xform_chk:  target %s (%d/%d) and source %s (%d/%d) must have same # of elements",
			OBJ_NAME(dpto),OBJ_N_TYPE_ELTS(dpto),OBJ_COMPS(dpto),
			OBJ_NAME(dpfr),OBJ_N_TYPE_ELTS(dpfr),OBJ_COMPS(dpfr));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	/* BUG these contiguity requirements may no longer be necessary... */

	if( !is_contiguous(DEFAULT_QSP_ARG  dpto) ){
		sprintf(DEFAULT_ERROR_STRING,
			"xform_chk:  xform target %s must be contiguous",OBJ_NAME(dpto));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( !is_contiguous(DEFAULT_QSP_ARG  dpfr) ){
		sprintf(DEFAULT_ERROR_STRING,
			"xform_chk:  xform source %s must be contiguous",OBJ_NAME(dpfr));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( !is_contiguous(DEFAULT_QSP_ARG  xform) ){
		sprintf(DEFAULT_ERROR_STRING,
			"xform_chk:  xform %s must be contiguous",OBJ_NAME(xform));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	return(0);
} // end xform_chk

/* apply a matrix to a list of elements */
/* this routine vectorizes the dot products;
	good for big matrices or short lists */

/* there should be a better routine for long lists of short elts. */

void xform_list(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform)
{
	Vec_Obj_Args oa1,*oap=&oa1;

	if( xform_chk(dpto,dpfr,xform) == -1 )
		return;

	/*
	switch( OBJ_MACH_PREC(dpto) ){
		case PREC_SP:	sp_obj_xform_list(QSP_ARG  dpto,dpfr,xform); break;
		case PREC_DP:	dp_obj_xform_list(QSP_ARG  dpto,dpfr,xform); break;
		default:
			sprintf(DEFAULT_ERROR_STRING,"xform_list:  destination object %s (%s) should have float or double precision",
				OBJ_NAME(dpto),OBJ_PREC_NAME(dpto));
			NWARN(DEFAULT_ERROR_STRING);
	}
	*/
	//WARN("xform_list:  need to implement sp_obj_xform_list and dp_obj_xform_list !?");
	// BUG should use platform-specific routine!?
#ifdef CAUTIOUS
	clear_obj_args(oap);
#endif /* CAUTIOUS */
	SET_OA_DEST(oap,dpto);
	SET_OA_SRC1(oap,dpfr);
	SET_OA_SRC2(oap,xform);
	set_obj_arg_flags(oap);

	h_vl2_xform_list(-1,oap);
}

/* like xform_list(), but vectorizes over list instead of matrix row */
/* good for long lists or big images */

void vec_xform(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform)
{
	Vec_Obj_Args oa1,*oap=&oa1;

	if( xform_chk(dpto,dpfr,xform) == -1 )
		return;

#ifdef CAUTIOUS
	clear_obj_args(oap);
#endif /* CAUTIOUS */
	SET_OA_DEST(oap,dpto);
	SET_OA_SRC1(oap,dpfr);
	SET_OA_SRC2(oap,xform);
	set_obj_arg_flags(oap);

	h_vl2_vec_xform(-1,oap);

	/*
	switch( OBJ_MACH_PREC(dpto) ){
		case PREC_SP:	sp_obj_vec_xform(QSP_ARG  dpto,dpfr,xform); break;
		case PREC_DP:	dp_obj_vec_xform(QSP_ARG  dpto,dpfr,xform); break;
		default:
			sprintf(ERROR_STRING,"vec_xform:  destination object %s (%s) should have float or double precision",
				OBJ_NAME(dpto),OBJ_PREC_NAME(dpto));
			WARN(ERROR_STRING);
	}
	*/
}

/*
 * like vec_xform, but does the division for homgenous coords
 */

void homog_xform(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform_dp)
{
	/*
	switch(OBJ_MACH_PREC(dpto)){
		case PREC_SP:	sp_obj_homog_xform(QSP_ARG  dpto,dpfr,xform_dp); break;
		case PREC_DP:	dp_obj_homog_xform(QSP_ARG  dpto,dpfr,xform_dp); break;
		default: sprintf(ERROR_STRING,"homog_xform:  unsupported precision");
			WARN(ERROR_STRING);
			break;
	}
	*/

} /* end homog_xform */

// BUG this assumes contiguity
void unity(Data_Obj *mp)				/**/
{
	dimension_t i,j;
	float *f;

	f=(float *)OBJ_DATA_PTR(mp);
	for(i=0;i<OBJ_COLS(mp);i++){
		for(j=0;j<OBJ_COLS(mp);j++){
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
	mp=make_obj(QSP_ARG  s,1,dim,dim,1,prec_for_code(PREC_SP));
	if( mp == NO_OBJ ){
		WARN("couldn't create new matrix");
		return;
	}
	unity(mp);
}

void transpose(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr)
{
	/* new version using gen_xpose */

	Data_Obj *tmp_dp;
	Vec_Obj_Args oa1, *oap=&oa1;

	// BUG use a macro for this...
	tmp_dp=NEW_DOBJ;

	OBJ_COPY_FROM(tmp_dp,dpfr);
	// after the copy, the shape pointer is the same as the parent's...
	DUP_OBJ_SHAPE(tmp_dp,dpfr);

	gen_xpose(tmp_dp,1,2);		/* switch rows, cols */

	setvarg2(oap,dpto,tmp_dp);
	perf_vfunc(QSP_ARG  FVMOV,oap);

	// tmp_dp has no resources except is shape...
	rls_shape( OBJ_SHAPE(tmp_dp) );
	givbuf(tmp_dp);
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

	if( OBJ_COLS(dpto) != OBJ_ROWS(dpto) ){
		sprintf(ERROR_STRING,"target matrix %s (%dx%d) must be square",OBJ_NAME(dpto),
			OBJ_ROWS(dpto),OBJ_COLS(dpto));
		WARN(ERROR_STRING);
		had_err++;
	}

	if( OBJ_COLS(dpto) != OBJ_ROWS(dpfr) ){
		sprintf(ERROR_STRING,
	"target matrix %s size %d not equal to source matrix %s rows (%d)",
			OBJ_NAME(dpto),OBJ_COLS(dpto),OBJ_NAME(dpfr),OBJ_ROWS(dpfr));
		WARN(ERROR_STRING);
		had_err++;
	}

	if( had_err ) return;

	if( IS_COMPLEX(dpto) )
		args.arg_argstype = COMPLEX_ARGS;
	else
		args.arg_argstype = REAL_ARGS;

	args.arg_inc1 = OBJ_PXL_INC(dpfr);
	args.arg_inc2 = OBJ_PXL_INC(dpfr);
	args.arg_n1 = OBJ_COLS(dpfr);
	args.arg_n2 = OBJ_COLS(dpfr);
	args.arg_prec1 = OBJ_PREC(dpfr);
	args.arg_prec2 = OBJ_PREC(dpfr);

	op1 = OBJ_DATA_PTR(dpfr);
	for(i=0;i<OBJ_ROWS(dpfr);i++){
		dest = dest2 = OBJ_DATA_PTR(dpto);
		dest += i*OBJ_ROW_INC(dpto);
		dest += i*OBJ_PXL_INC(dpto);
		dest2 += i*OBJ_PXL_INC(dpto);
		dest2 += i*OBJ_ROW_INC(dpto);
		op2 = OBJ_DATA_PTR(dpfr);
		op2 += i*OBJ_ROW_INC(dpfr);
		for(j=i;j<OBJ_ROWS(dpfr);j++){

			args.arg_v1 = op1;
			args.arg_v2 = op2;
			vdot(&args);

			*dest2 = *dest;		/* symmetric matrix */

			op2 += OBJ_ROW_INC(dpfr);
			dest += OBJ_PXL_INC(dpto);
			dest2 += OBJ_ROW_INC(dpto);
		}
		op1 += OBJ_ROW_INC(dpfr);
	}
} /* end corr_matrix() */
#endif /* FOOBAR */

/* Compute the determinant of a square matrix */

double determinant(Data_Obj *dp)
{
	/*
	switch(OBJ_MACH_PREC(dp)){
		case PREC_SP:  return sp_obj_determinant(dp);
		case PREC_DP:  return dp_obj_determinant(dp);
		default:
			sprintf(DEFAULT_ERROR_STRING,"determinant:  object %s has unsupported precision %s",
				OBJ_NAME(dp),OBJ_MACH_PREC_NAME(dp));
			NWARN(DEFAULT_ERROR_STRING);
			return(0.0);
	}
	*/
	NWARN("Need to implement sp_obj_determinant...");
	/* NOTREACHED */
	return(0.0);
}
			

