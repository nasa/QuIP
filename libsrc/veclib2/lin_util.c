
#include "quip_config.h"
#include "quip_prot.h"
#include "veclib_api.h"
#include "veclib/vl2_veclib_prot.h"

#ifdef NOT_USED
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
#endif // NOT_USED

#ifdef FOOBAR
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

	vmul(oap);
	return(0);
}
#endif // FOOBAR

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

/* inner (matrix) product
 *
 */

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

	if( IS_COMPLEX(dpto) )	SET_OA_ARGSTYPE(oap,COMPLEX_ARGS);
	else			SET_OA_ARGSTYPE(oap,REAL_ARGS);

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
			h_vl2_vdot(FVDOT,oap);
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
	Vec_Obj_Args oa1, *oap=(&oa1);
	
	if( xform_chk(dpto,dpfr,xform) == -1 )
		return;

	// BUG shouldn't use vl2 platform!?
	setvarg3(oap,dpto,dpfr,xform);
	h_vl2_xform_list(-1,oap);

#ifdef FOOBAR
	switch( OBJ_MACH_PREC(dpto) ){
		case PREC_SP:	sp_obj_xform_list(QSP_ARG  dpto,dpfr,xform); break;
		case PREC_DP:	dp_obj_xform_list(QSP_ARG  dpto,dpfr,xform); break;
		default:
			sprintf(DEFAULT_ERROR_STRING,"xform_list:  destination object %s (%s) should have float or double precision",
				OBJ_NAME(dpto),OBJ_PREC_NAME(dpto));
			NWARN(DEFAULT_ERROR_STRING);
	}
#endif // FOOBAR
}

