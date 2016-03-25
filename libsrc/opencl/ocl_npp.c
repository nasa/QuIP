
/* interface to nVidia NPP image processing library */

#include "quip_config.h"

#ifdef HAVE_CUDA		// can we have cuda but not libnpp???

#include <stdio.h>

#include <cuda_runtime.h>

#ifdef HAVE_NPP_H
#include <npp.h>
#endif /* HAVE_NPP_H */
#include <nppi.h>
#include <npps.h>
#include <nppversion.h>

#define NPP_VERSION_CODE(major,minor,build)			\
	((major<<16)|(minor<<8)|build)

#define NPP_VERSION	NPP_VERSION_CODE(NPP_VERSION_MAJOR,NPP_VERSION_MINOR,NPP_VERSION_BUILD)

#include "quip_prot.h"
#include "my_cuda.h"
#include "data_obj.h"
#include "cuda_supp.h"

#ifndef HAVE_LIBNPP
#define NO_NPP_MSG(whence)			\
	sprintf(DEFAULT_ERROR_STRING,			\
"%s:  Program was not configured with libnpp support!?",whence);	\
	NWARN(DEFAULT_ERROR_STRING);
#endif /* ! HAVE_LIBNPP */

#define REPORT_NPP_STATUS(whence,funcname)			\
								\
	if( s != NPP_SUCCESS ){					\
		report_npp_error(whence,funcname,s);		\
		return;						\
	}


#ifdef HAVE_LIBNPP

#define NPP_ERR_CASE(code,msg)					\
								\
	case code: m = msg; break;

static void report_npp_error(const char *whence, const char *funcname, NppStatus s)
{
	const char *m=NULL;

	switch(s){
#if CUDA_VERSION >= 6050
		NPP_ERR_CASE(NPP_OVERFLOW_ERROR,"overflow")
#endif
		NPP_ERR_CASE(NPP_NOT_SUPPORTED_MODE_ERROR, "unsupported mode" )
		NPP_ERR_CASE(NPP_ROUND_MODE_NOT_SUPPORTED_ERROR,"unsupported round mode" )
		NPP_ERR_CASE( NPP_RESIZE_NO_OPERATION_ERROR, "No resize operation" )
		NPP_ERR_CASE( NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY, "insufficient compute capability" )
// Mac MBP 4020
// Wheatstone 4000
#if CUDA_VERSION >= 4000	// wheatstone
		NPP_ERR_CASE( NPP_WRONG_INTERSECTION_ROI_ERROR, "wrong intersection ROI" )
#endif // CUDA_VERSION >= 4000

#if CUDA_VERSION >= 4020	// mac MBP
		NPP_ERR_CASE( NPP_WRONG_INTERSECTION_ROI_WARNING, "wrong intersection ROI warning" )
#endif // CUDA_VERSION >= 4020

#if CUDA_VERSION >= 5050
		NPP_ERR_CASE( NPP_BAD_ARGUMENT_ERROR, "bad argument" )
		NPP_ERR_CASE( NPP_COEFFICIENT_ERROR, "coefficient error" )
		NPP_ERR_CASE( NPP_RECTANGLE_ERROR, "rect error" )
		NPP_ERR_CASE( NPP_QUADRANGLE_ERROR, "quad error" )
		NPP_ERR_CASE( NPP_MEMFREE_ERROR, "memfree error" )
		NPP_ERR_CASE( NPP_MEMSET_ERROR, "memset error" )
		NPP_ERR_CASE(NPP_MEMORY_ALLOCATION_ERR, "memory allocation error" )
		NPP_ERR_CASE(NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR, "bad number of histogram levels" )
		NPP_ERR_CASE(NPP_MIRROR_FLIP_ERROR, "mirror flip error" )
#else
		NPP_ERR_CASE( NPP_BAD_ARG_ERROR, "bad argument" )
		NPP_ERR_CASE( NPP_COEFF_ERROR, "coefficient error" )
		NPP_ERR_CASE( NPP_RECT_ERROR, "rect error" )
		NPP_ERR_CASE( NPP_QUAD_ERROR, "quad error" )
		NPP_ERR_CASE( NPP_MEMFREE_ERR, "memfree error" )
		NPP_ERR_CASE( NPP_MEMSET_ERR, "memset error" )
		NPP_ERR_CASE(NPP_MEM_ALLOC_ERR, "memory allocation error" )
		NPP_ERR_CASE(NPP_HISTO_NUMBER_OF_LEVELS_ERROR, "bad number of histogram levels" )
		NPP_ERR_CASE(NPP_MIRROR_FLIP_ERR, "mirror flip error" )
		// No updated version for these ones???
		NPP_ERR_CASE(NPP_INVALID_INPUT, "invalid input" )
		NPP_ERR_CASE(NPP_POINTER_ERROR, "pointer error" )
		NPP_ERR_CASE(NPP_WARNING, "general NPP warning" )
		// roi warning error needed on mac MBP...
		NPP_ERR_CASE(NPP_ODD_ROI_WARNING, "ROI size forced to an even value" )
#endif

		NPP_ERR_CASE( NPP_LUT_NUMBER_OF_LEVELS_ERROR, "bad number of LUT levels" )
		NPP_ERR_CASE( NPP_TEXTURE_BIND_ERROR, "texture binding error" )
		NPP_ERR_CASE( NPP_NOT_EVEN_STEP_ERROR, "not an even step" )
		NPP_ERR_CASE( NPP_INTERPOLATION_ERROR, "interpolation error" )
		NPP_ERR_CASE( NPP_RESIZE_FACTOR_ERROR, "bad resize factor" )
		NPP_ERR_CASE( NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR, "Haar classifier pixel match error" )
		NPP_ERR_CASE( NPP_MEMCPY_ERROR, "memcpy error" )
		NPP_ERR_CASE(NPP_ALIGNMENT_ERROR, "alignment error" )
		NPP_ERR_CASE(NPP_STEP_ERROR, "step error" )
		NPP_ERR_CASE(NPP_SIZE_ERROR, "size error" )
		NPP_ERR_CASE(NPP_NULL_POINTER_ERROR, "null pointer" )
		NPP_ERR_CASE(NPP_CUDA_KERNEL_EXECUTION_ERROR, "kernel execution error" )
		NPP_ERR_CASE(NPP_NOT_IMPLEMENTED_ERROR, "not implemented" )
		NPP_ERR_CASE(NPP_ERROR, "general NPP error" )
		NPP_ERR_CASE(NPP_WRONG_INTERSECTION_QUAD_WARNING, "wrong intersection quad" )
		NPP_ERR_CASE(NPP_MISALIGNED_DST_ROI_WARNING, "mis-aligned dst ROI" )
		NPP_ERR_CASE(NPP_AFFINE_QUAD_INCORRECT_WARNING, "affine quad incorrect" )
		NPP_ERR_CASE(NPP_DOUBLE_SIZE_WARNING, "ROI size was modified" )
		NPP_ERR_CASE(NPP_SUCCESS, "Success" )

		// These added to elim unhandled case warnings in 5.5, may
		// not work with earlier versions...

		// euler is 5.0
#if CUDA_VERSION >= 5000
		NPP_ERR_CASE(NPP_COI_ERROR, "COI error" )
		NPP_ERR_CASE(NPP_ZC_MODE_NOT_SUPPORTED_ERROR, "ZC mode not supported" )
#endif

#if CUDA_VERSION >= 5050
		NPP_ERR_CASE(NPP_INVALID_HOST_POINTER_ERROR,"Invalid host pointer" )
		NPP_ERR_CASE(NPP_INVALID_DEVICE_POINTER_ERROR, "Invalid device pointer" )
		NPP_ERR_CASE(NPP_LUT_PALETTE_BITSIZE_ERROR, "LUT palette bitsize error" )
		NPP_ERR_CASE(NPP_QUALITY_INDEX_ERROR, "Quality index error" )
		NPP_ERR_CASE(NPP_CHANNEL_ORDER_ERROR, "Channel order error" )
		NPP_ERR_CASE(NPP_ZERO_MASK_VALUE_ERROR, "Zero mask value error" )
		NPP_ERR_CASE(NPP_NUMBER_OF_CHANNELS_ERROR, "Number of channels error" )
		NPP_ERR_CASE(NPP_DIVISOR_ERROR, "Divisor error" )
		NPP_ERR_CASE(NPP_CHANNEL_ERROR, "Channel error" )
		NPP_ERR_CASE(NPP_STRIDE_ERROR, "Stride error" )
		NPP_ERR_CASE(NPP_ANCHOR_ERROR, "Anchor error" )
		NPP_ERR_CASE(NPP_MASK_SIZE_ERROR, "Mask size error" )
		NPP_ERR_CASE(NPP_MOMENT_00_ZERO_ERROR, "Moment 00 zero error" )
		NPP_ERR_CASE(NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR, "Threshold negative level error" )
		NPP_ERR_CASE(NPP_THRESHOLD_ERROR, "Threshold error" )
		NPP_ERR_CASE(NPP_CONTEXT_MATCH_ERROR, "Context match error" )
		NPP_ERR_CASE(NPP_FFT_FLAG_ERROR, "FFT flag error" )
		NPP_ERR_CASE(NPP_FFT_ORDER_ERROR, "FFT order error" )
		NPP_ERR_CASE(NPP_SCALE_RANGE_ERROR, "Scale range error" )
		NPP_ERR_CASE(NPP_DATA_TYPE_ERROR, "Data type error" )
		NPP_ERR_CASE(NPP_OUT_OFF_RANGE_ERROR, "Out of range error" )
		NPP_ERR_CASE(NPP_DIVIDE_BY_ZERO_ERROR, "Divide by zero error" )
		NPP_ERR_CASE(NPP_RANGE_ERROR, "Range error" )
		NPP_ERR_CASE(NPP_NO_MEMORY_ERROR, "No memory error" )
		NPP_ERR_CASE(NPP_ERROR_RESERVED, "Error reserved" )
		NPP_ERR_CASE(NPP_NO_OPERATION_WARNING, "No operation warning" )
		NPP_ERR_CASE(NPP_DIVIDE_BY_ZERO_WARNING, "Divide by zero warning" )
		// duplicate case warning???
		//NPP_ERR_CASE(NPP_WRONG_INTERSECTION_ROI_WARNING, "Wrong intersection ROI warning" )
#endif	// cuda_version > 5050

		// no default case so compiler will report missing cases.
	}
	if( m == NULL ){
		sprintf(DEFAULT_ERROR_STRING,
	"%s:  %s:  Unrecognized error (%d) in report_npp_error!?",
			whence,funcname,s);
		NWARN(DEFAULT_ERROR_STRING);
		m = "Unrecognized error";
	}
		
	if( s < 0 ){
		sprintf(DEFAULT_ERROR_STRING,"%s:  %s:  NPP error:  %s",
			whence,funcname,m);
		NWARN(DEFAULT_ERROR_STRING);
	} else {
		sprintf(DEFAULT_ERROR_STRING,"%s:  %s:  NPP warning:  %s",
			whence,funcname,m);
		NWARN(DEFAULT_ERROR_STRING);
	}
}
#endif /* HAVE_LIBNPP */

/* For dilation and erosion, the mask needs to fall completely
 * within the input image.  So the output that we can set is
 * necessarily smaller, by the size of the mask.  For example,
 * for a 3x3 mask with the anchor in the center, we need to operate
 * on an image with an extra pixel all around the margin.
 *
 * There are several options:  in the first implementation, before
 * I understood any of this, we just passed along the images from the user
 * as-is, which caused an error until we added the padding (by
 * creating a larger image, and passing a subimage).  That didn't
 * reveal all the error conditions, because overflows off the back
 * wouldn't create a problem if there was another (different) image
 * there, because the gpu doesn't know where our image boundaries are.
 *
 * It seems more logical for the code here to be passed the entire image
 * and adjust the pointers and lengths according to the mask.
 * That relieves the user of having to understand things...  The 
 * user can always add a pad to their image if they need to.
 *
 * Now the code seems to work as expected with 3x3 masks, but not with 5x5???
 */

#define SET_SIZES						\
								\
	siz.width = OBJ_COLS(dst_dp) + 1 - OBJ_COLS(mask_dp);	\
	siz.height = OBJ_ROWS(dst_dp) + 1 - OBJ_ROWS(mask_dp);	\
	msiz.width = OBJ_COLS(mask_dp);				\
	msiz.height = OBJ_ROWS(mask_dp);

#define MORPH_DECLS					\
							\
	Data_Obj *dst_dp, *src_dp, *mask_dp;		\
	NppStatus s;					\
	NppiSize siz,msiz;				\
	NppiPoint anchor;				\
	Npp8u *dst_p, *src_p;

#define GET_MORPH_ARGS					\
							\
	dst_dp = PICK_OBJ("target image");		\
	src_dp = PICK_OBJ("source image");		\
	mask_dp = PICK_OBJ("mask image");		\
	anchor.x = HOW_MANY("anchor x");		\
	anchor.y = HOW_MANY("anchor y");


#define GET_FILTER_ARGS					\
							\
	dst_dp = PICK_OBJ("target image");		\
	src_dp = PICK_OBJ("source image");		\
	mask_dp = PICK_OBJ("kernel image");		\
	anchor.x = HOW_MANY("anchor x");		\
	anchor.y = HOW_MANY("anchor y");		\
	divisor = HOW_MANY("divisor");


static int good_img_for_morph(QSP_ARG_DECL  Data_Obj *dp, const char *whence )
{
	if( OBJ_PREC(dp) != PREC_UBY ){
		sprintf(ERROR_STRING,
			"%s:  object %s (%s) must have %s precision",
			whence,OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)),
			PREC_NAME(PREC_FOR_CODE(PREC_UBY)));
		WARN(ERROR_STRING);
		return(0);
	}
	if( OBJ_COMPS(dp) != 1 ){
		sprintf(ERROR_STRING,
			"%s:  object %s (%d) must have 1 component",
			whence,OBJ_NAME(dp),OBJ_COMPS(dp) );
		WARN(ERROR_STRING);
		return(0);
	}
	if( OBJ_FRAMES(dp) != 1 || OBJ_SEQS(dp) != 1 ){
		sprintf(ERROR_STRING,
	"%s:  object %s (%d sequence%s of %d frame%s) must be a single image",
			whence,OBJ_NAME(dp),OBJ_SEQS(dp),
			OBJ_SEQS(dp)==1?"":"s",OBJ_FRAMES(dp),
			OBJ_FRAMES(dp)==1?"":"s");
		WARN(ERROR_STRING);
		return(0);
	}
	return(1);
}

static int good_kernel_for_filter(Data_Obj *dp, const char *whence )
{
	if( OBJ_PREC(dp) != PREC_DI ){
		sprintf(DEFAULT_ERROR_STRING,
			"%s:  object %s (%s) must have %s precision",
			whence,OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)),
			PREC_NAME(PREC_FOR_CODE(PREC_DI)));
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	if( OBJ_COMPS(dp) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,
			"%s:  object %s (%d) must have 1 component",
			whence,OBJ_NAME(dp),OBJ_COMPS(dp) );
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	if( OBJ_FRAMES(dp) != 1 || OBJ_SEQS(dp) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,
	"%s:  object %s (%d sequence%s of %d frame%s) must be a single image",
			whence,OBJ_NAME(dp),OBJ_SEQS(dp),
			OBJ_SEQS(dp)==1?"":"s",OBJ_FRAMES(dp),
			OBJ_FRAMES(dp)==1?"":"s");
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	return(1);
}

#ifdef HAVE_LIBNPP

static int good_for_morph( QSP_ARG_DECL   Data_Obj *dst_dp, Data_Obj *src_dp,
				Data_Obj *mask_dp, const char * whence )
{
	if( ! dp_same_size(QSP_ARG  dst_dp,src_dp,whence) )
		return(0);
	if( ! good_img_for_morph(QSP_ARG  dst_dp,whence) )
		return(0);
	if( ! good_img_for_morph(QSP_ARG  src_dp,whence) )
		return(0);
	if( ! good_img_for_morph(QSP_ARG  mask_dp,whence) )
		return(0);
	if( OBJ_ROWS(dst_dp) < OBJ_ROWS(mask_dp) ||
			OBJ_COLS(dst_dp) < OBJ_COLS(mask_dp) ){
		sprintf(DEFAULT_ERROR_STRING,
	"%s:  destination %s (%d x %d) must be larger than mask %s (%d x %d)",
			whence,OBJ_NAME(dst_dp),OBJ_COLS(dst_dp),OBJ_ROWS(dst_dp),
			OBJ_NAME(mask_dp),OBJ_COLS(mask_dp),OBJ_ROWS(mask_dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	return(1);
}

static int good_for_filter( QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp,
				Data_Obj *mask_dp, const char * whence )
{
	if( ! dp_same_size(QSP_ARG  dst_dp,src_dp,whence) )
		return(0);
	if( ! dp_same_prec(QSP_ARG  dst_dp,src_dp,whence) )
		return(0);
	if( OBJ_PREC(mask_dp) != PREC_DI ){
		sprintf(ERROR_STRING,
	"%s:  kernel image %s (%s) should have %s precision",
			whence,OBJ_NAME(mask_dp),PREC_NAME(OBJ_PREC_PTR(mask_dp)),
			PREC_NAME(PREC_FOR_CODE(PREC_DI)));
		WARN(ERROR_STRING);
		return(0);
	}
	if( ! good_img_for_morph(QSP_ARG  dst_dp,whence) )
		return(0);
	if( ! good_img_for_morph(QSP_ARG  src_dp,whence) )
		return(0);
	if( ! good_kernel_for_filter(mask_dp,whence) )
		return(0);
	if( OBJ_ROWS(dst_dp) < OBJ_ROWS(mask_dp) ||
			OBJ_COLS(dst_dp) < OBJ_COLS(mask_dp) ){
		sprintf(ERROR_STRING,
	"%s:  destination %s (%d x %d) must be larger than kernel %s (%d x %d)",
			whence,OBJ_NAME(dst_dp),OBJ_COLS(dst_dp),OBJ_ROWS(dst_dp),
			OBJ_NAME(mask_dp),OBJ_COLS(mask_dp),OBJ_ROWS(mask_dp));
		WARN(ERROR_STRING);
		return(0);
	}
	return(1);
}

#endif // HAVE_LIBNPP

#define CHECK_MORPH_ARGS(whence)					\
									\
	if( dst_dp == NO_OBJ || src_dp == NO_OBJ || mask_dp == NO_OBJ )	\
		return;							\
									\
	if( !good_for_morph(QSP_ARG  dst_dp,src_dp,mask_dp,whence) )		\
		return;


#define CHECK_FILTER_ARGS(whence)					\
									\
	if( dst_dp == NO_OBJ || src_dp == NO_OBJ || mask_dp == NO_OBJ )	\
		return;							\
									\
	if( !good_for_filter(QSP_ARG  dst_dp,src_dp,mask_dp,whence) )		\
		return;


#define SET_OFFSETS						\
								\
	dst_p = (Npp8u *)OBJ_DATA_PTR(dst_dp);			\
	src_p = (Npp8u *)OBJ_DATA_PTR(src_dp);			\
	dst_p += anchor.x + anchor.y * OBJ_ROW_INC(dst_dp);	\
	src_p += anchor.x + anchor.y * OBJ_ROW_INC(src_dp);


COMMAND_FUNC( do_npp_erosion )
{
#ifdef HAVE_LIBNPP
	MORPH_DECLS
	GET_MORPH_ARGS
	CHECK_MORPH_ARGS("do_npp_erosion")
	SET_SIZES
	SET_OFFSETS

	s = nppiErode_8u_C1R(src_p,OBJ_ROW_INC(src_dp),
				dst_p, OBJ_ROW_INC(dst_dp),
				siz,(Npp8u *)OBJ_DATA_PTR(mask_dp),msiz,anchor);

	REPORT_NPP_STATUS("do_npp_erosion","nppiErode_8u_C1R")

#else /* ! HAVE_LIBNPP */
	NO_NPP_MSG("do_npp_erosion")
#endif /* ! HAVE_LIBNPP */
}

COMMAND_FUNC( do_npp_dilation )
{
#ifdef HAVE_LIBNPP
	MORPH_DECLS
	GET_MORPH_ARGS
	CHECK_MORPH_ARGS("do_npp_dilation")
	SET_SIZES
	SET_OFFSETS

	s = nppiDilate_8u_C1R(src_p, OBJ_ROW_INC(src_dp),
				dst_p, OBJ_ROW_INC(dst_dp),
				siz,(Npp8u *)OBJ_DATA_PTR(mask_dp),msiz,anchor);

	REPORT_NPP_STATUS("do_npp_dilation","nppiDilate_8u_C1R")

#else /* ! HAVE_LIBNPP */
	NO_NPP_MSG("do_npp_dilation")
#endif /* ! HAVE_LIBNPP */
}

COMMAND_FUNC( do_npp_filter )
{
#ifdef HAVE_LIBNPP
	MORPH_DECLS
	int32_t divisor;

	// The NPP documentation appears to be in error, it says
	// that the peak value is preserved by using a divisor
	// equal to the sum of the coefficients, but for a gaussian
	// low-pass filter with a peak value of 255, using a divisor
	// of 256 gives us an image with at peak value of 254
	// regardless of the filter width...

	GET_FILTER_ARGS
	CHECK_FILTER_ARGS("do_npp_filter")
	SET_SIZES
	SET_OFFSETS

	s = nppiFilter_8u_C1R(src_p, OBJ_ROW_INC(src_dp),
				dst_p, OBJ_ROW_INC(dst_dp), siz,
				(Npp32s *)OBJ_DATA_PTR(mask_dp),msiz,anchor,divisor);

	REPORT_NPP_STATUS("do_npp_filter","nppiFilter_8u_C1R")

#else /* ! HAVE_LIBNPP */
	NO_NPP_MSG("do_npp_dilation")
#endif /* ! HAVE_LIBNPP */
}

static Precision * validate_npp_prec( Precision * prec_p )
{
	switch(PREC_CODE(prec_p) ){
		case PREC_IN:
		case PREC_UIN:
		case PREC_SP:
		case PREC_DI:
		case PREC_UDI:
		case PREC_DP:
		case PREC_LI:
		case PREC_ULI:
		case PREC_BY:
		case PREC_UBY:
			return(prec_p);
			break;
		default:
			return(PREC_FOR_CODE(PREC_NONE));
			break;
	}
}

#define M_CASE(prec,suffix)					\
	case prec: SET_OBJ_DATA_PTR(dp, 				\
	nppiMalloc_##suffix(OBJ_COLS(dp),OBJ_ROWS(dp),&stride));	\
	break;

#define M_DEFAULT						\
	default:						\
	sprintf(DEFAULT_ERROR_STRING,				\
"Sorry, no NPP allocator for %s precision with %d channels",	\
		PREC_NAME(prec_p),ds.ds_dimension[0]);		\
	NWARN(DEFAULT_ERROR_STRING);				\
	delvec(QSP_ARG  dp);					\
	return;

COMMAND_FUNC( do_npp_malloc )
{
#ifdef HAVE_LIBNPP
	Precision *prec_p , *p;
	Dimension_Set ds;
	Data_Obj *dp;
	const char *s;
	int stride;

	s=NAMEOF("name for image");
	ds.ds_dimension[2] = HOW_MANY("number of rows");
	ds.ds_dimension[1] = HOW_MANY("number of columns");
	ds.ds_dimension[0] = HOW_MANY("number of channels");
	prec_p = get_precision(SINGLE_QSP_ARG);
	p = validate_npp_prec(prec_p);

	if( PREC_CODE(p) == PREC_NONE ){
		sprintf(ERROR_STRING,
	"Sorry, precision (%s) requested for image %s is not supported by NPP",
			PREC_NAME(prec_p),s);
		WARN(ERROR_STRING);
		return;
	}

	if( ds.ds_dimension[0] > 4 ){
		sprintf(ERROR_STRING,
"Sorry, n_channels (%d) requested for image %s cannot be greater than 4.",
			ds.ds_dimension[0],s);
		WARN(ERROR_STRING);
		return;
	}

	ds.ds_dimension[3] = 1;
	ds.ds_dimension[4] = 1;

	dp = _make_dp(QSP_ARG  s,&ds,prec_p);
	if( dp == NO_OBJ ) return;

	/* Now call the allocator */
	if( ds.ds_dimension[0] == 1 ){
		switch(PREC_CODE(prec_p)){
			M_CASE(PREC_UBY,8u_C1)
			M_CASE(PREC_UIN,16u_C1)
			M_CASE(PREC_IN,16s_C1)
			M_CASE(PREC_DI,32s_C1)
			M_CASE(PREC_SP,32f_C1)
			M_DEFAULT
		}
	} else if( ds.ds_dimension[0] == 2 ){
		switch(PREC_CODE(prec_p)){
			M_CASE(PREC_UBY,8u_C2)
			M_CASE(PREC_UIN,16u_C2)
			M_CASE(PREC_IN,16s_C2)
			M_CASE(PREC_SP,32f_C2)
			M_DEFAULT
		}
	} else if( ds.ds_dimension[0] == 3 ){
		switch(PREC_CODE(prec_p)){
			M_CASE(PREC_UBY,8u_C3)
			M_CASE(PREC_UIN,16u_C3)
			M_CASE(PREC_DI,32s_C3)
			M_CASE(PREC_SP,32f_C3)
			M_DEFAULT
		}
	} else if( ds.ds_dimension[0] == 4 ){
		switch(PREC_CODE(prec_p)){
			M_CASE(PREC_UBY,8u_C4)
			M_CASE(PREC_UIN,16u_C4)
			M_CASE(PREC_IN,16s_C4)
			M_CASE(PREC_DI,32s_C4)
			M_CASE(PREC_SP,32f_C4)
			M_DEFAULT
		}
	} else {
		sprintf(ERROR_STRING,
	"do_npp_malloc:  number of channels (%d) must be between 1 and 4.",
			ds.ds_dimension[0]);
		WARN(ERROR_STRING);
		return;
	}
	/* stride is in bytes */
	SET_OBJ_ROW_INC(dp, stride / PREC_SIZE(prec_p) );

#else /* ! HAVE_LIBNPP */
	NO_NPP_MSG("do_npp_malloc");
#endif /* ! HAVE_LIBNPP */
} /* end do_npp_malloc */

#define A_CASE(prec,suffix,type)					\
	case prec:						\
	s = nppiAdd_##suffix( (const type *)OBJ_DATA_PTR(src1_dp), src1_step,\
		(const type *)OBJ_DATA_PTR(src2_dp), src2_step,		\
		(type *)OBJ_DATA_PTR(dst_dp), dst_step, size);		\
	break;

#define A_DEFAULT						\
	default:	sprintf(DEFAULT_ERROR_STRING,			\
"No NPP addition function for %d channels of type %s",	\
OBJ_COMPS(dst_dp),PREC_NAME(OBJ_PREC_PTR(dst_dp)));		\
		NWARN(DEFAULT_ERROR_STRING);				\
		return;						\
		break;


COMMAND_FUNC( do_npp_vadd )
{
#ifdef HAVE_LIBNPP
	Data_Obj *dst_dp, *src1_dp, *src2_dp;
	int dst_step, src1_step, src2_step;
	NppStatus s;
	NppiSize size;
	int pxl_size;

	dst_dp = PICK_OBJ("destination image");
	src1_dp = PICK_OBJ("first source image");
	src2_dp = PICK_OBJ("second source image");

	/* BUG - make sure that the sizes all match */

	pxl_size = PREC_SIZE( OBJ_MACH_PREC_PTR(dst_dp) );
	if( IS_COMPLEX(dst_dp) ) pxl_size *= 2;
	dst_step = OBJ_ROW_INC(dst_dp) * pxl_size;
	src1_step = OBJ_ROW_INC(src1_dp) * pxl_size;
	src2_step = OBJ_ROW_INC(src2_dp) * pxl_size;

	size.width = OBJ_COLS(dst_dp);
	size.height = OBJ_ROWS(dst_dp);

	if( OBJ_COMPS(dst_dp) == 1 ){
		switch(OBJ_PREC(dst_dp)){
			//A_CASE(PREC_UBY,8u_C1R)
			A_CASE(PREC_DI,32s_C1R,Npp32s)
			A_CASE(PREC_SP,32f_C1R,Npp32f)
			case PREC_UBY:
	s = nppiAdd_8u_C1RSfs( (const Npp8u *)OBJ_DATA_PTR(src1_dp), src1_step,
				(const Npp8u *)OBJ_DATA_PTR(src2_dp), src2_step,
				(Npp8u *)OBJ_DATA_PTR(dst_dp), dst_step, size, 1);
				break;

			A_DEFAULT
		}
	} else {
		sprintf(DEFAULT_ERROR_STRING,
			"npp_vadd:  component dimension %d not supported.",
			OBJ_COMPS(dst_dp));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	REPORT_NPP_STATUS("do_npp_vadd", "nppiAdd_8u_C1RSfs")

	/* No complex addition? */
	/*
	else if( OBJ_COMPS(dst_dp) == 2 ){
		switch(OBJ_PREC(dst_dp)){
			A_CASE(PREC_CPX,32fc_C1R,Npp32fc)
		}
	}
	*/

#else /* ! HAVE_LIBNPP */
	NO_NPP_MSG("do_npp_vadd");
#endif /* ! HAVE_LIBNPP */
}

COMMAND_FUNC( do_report_npp_version )
{
	sprintf(ERROR_STRING,"Cuda NPP version:  %d.%d.%d",
		NPP_VERSION_MAJOR,NPP_VERSION_MINOR,NPP_VERSION_BUILD);
	prt_msg(ERROR_STRING);
}

#ifdef HAVE_LIBNPP

static int sum_scratch_size = 0;
static int sum_scratch_n = 0;
static Npp8u *scratch_buf=NULL;

static void get_scratch_for(Data_Obj *dp)
{
	int len, buf_size;
	NppStatus s;

	if( ! N_IS_CONTIGUOUS(dp) ){
		sprintf(DEFAULT_ERROR_STRING,
	"sum_scratch:  Object %s must be contiguous",OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	len = OBJ_N_TYPE_ELTS(dp);

/* Mac has 4.2.9 and needs SumGetBufferSize */
/* Linus has 4.0.17 and needs ReductionGetBufferSize */

#if NPP_VERSION <= NPP_VERSION_CODE(4,0,17)
	s = nppsReductionGetBufferSize_32f( len, &buf_size );
#else	/* newer version of NPP */
	/* What versions use this??? */
	s = nppsSumGetBufferSize_32f( len, &buf_size );
#endif	/* newer version of NPP */


	REPORT_NPP_STATUS("do_npp_vadd","nppsReductionGetBufferSize_32f")

	// now allocate the memory...
sprintf(DEFAULT_ERROR_STRING,"Requesting %d scratch bytes",buf_size);
NADVISE(DEFAULT_ERROR_STRING);
	scratch_buf = nppsMalloc_8u(buf_size);
	// BUG error check?
	sum_scratch_size = buf_size;
	sum_scratch_n = len;
}

#endif // HAVE_LIBNPP

COMMAND_FUNC( do_npp_sum_scratch )
{
	Data_Obj *src_dp;

	src_dp = PICK_OBJ("source object");
	if( src_dp == NO_OBJ ) return;

	// BUG make sure correct type...
	// BUG make sure contiguous...

#ifdef HAVE_LIBNPP
	get_scratch_for(src_dp);
#endif // HAVE_LIBNPP
}

COMMAND_FUNC( do_npp_sum )
{
	Data_Obj *src_dp,*dst_dp;
#ifdef HAVE_LIBNPP
	NppStatus s;
#endif // HAVE_LIBNPP

	dst_dp = PICK_OBJ("destination object");
	src_dp = PICK_OBJ("source object");

	if( dst_dp == NO_OBJ || src_dp == NO_OBJ )
		return;

#ifdef HAVE_LIBNPP
	if( sum_scratch_size == 0 ){
		get_scratch_for(src_dp);
	} else if( (int) OBJ_N_TYPE_ELTS(src_dp) > sum_scratch_n ){
		// BUG
		nppsFree(scratch_buf);
		get_scratch_for(src_dp);
	}
	// BUG make sure the scratch space is the correct size

#if NPP_VERSION >= NPP_VERSION_CODE(5,0,0)
	// cuda 5
	s = nppsSum_32f((Npp32f*)OBJ_DATA_PTR(src_dp),OBJ_N_TYPE_ELTS(src_dp),
		(Npp32f*)OBJ_DATA_PTR(dst_dp),scratch_buf);
#else
	s = nppsSum_32f((Npp32f*)OBJ_DATA_PTR(src_dp),OBJ_N_TYPE_ELTS(src_dp),
		(Npp32f*)OBJ_DATA_PTR(dst_dp),nppAlgHintNone,scratch_buf);
#endif

	REPORT_NPP_STATUS("do_npp_sum","nppsSum_32f")
#endif // HAVE_LIBNPP
}

COMMAND_FUNC( do_nppi_vmul )
{
	Data_Obj *src1_dp, *src2_dp, *dst_dp;
#ifdef HAVE_LIBNPP
	NppStatus s;
	NppiSize roi_size;
#endif // HAVE_LIBNPP

	dst_dp = PICK_OBJ("destination object");
	src1_dp = PICK_OBJ("first source object");
	src2_dp = PICK_OBJ("second source object");

	if( dst_dp == NO_OBJ || src1_dp == NO_OBJ || src2_dp == NO_OBJ )
		return;

	// BUG make sure sizes match
	// BUG make sure float precision, or switch on pixel type

#ifdef HAVE_LIBNPP
	roi_size.width = OBJ_COLS(dst_dp);
	roi_size.height = OBJ_ROWS(dst_dp);

	s = nppiMul_32f_C1R(	(Npp32f*) OBJ_DATA_PTR(src1_dp), OBJ_ROW_INC(src1_dp)*sizeof(Npp32f),
				(Npp32f*) OBJ_DATA_PTR(src2_dp), OBJ_ROW_INC(src2_dp)*sizeof(Npp32f),
				(Npp32f*) OBJ_DATA_PTR(dst_dp),  OBJ_ROW_INC(dst_dp)*sizeof(Npp32f),
				roi_size );

	REPORT_NPP_STATUS("do_nppi_vmul","nppiMul_32f_C1R")
#endif // HAVE_LIBNPP
}

COMMAND_FUNC( do_npps_vmul )
{
	Data_Obj *src_dp, *dst_dp;
#ifdef HAVE_LIBNPP
	NppStatus s;
#endif // HAVE_LIBNPP

	dst_dp = PICK_OBJ("destination/source object");
	src_dp = PICK_OBJ("source object");

	if( dst_dp == NO_OBJ || src_dp == NO_OBJ )
		return;

	// BUG make sure sizes match
	// BUG make sure float precision, or switch on pixel type

#ifdef HAVE_LIBNPP
	s = nppsMul_32f_I(	(Npp32f*) OBJ_DATA_PTR(src_dp),
				(Npp32f*) OBJ_DATA_PTR(dst_dp),
				OBJ_N_TYPE_ELTS(dst_dp) );

	REPORT_NPP_STATUS("do_npps_vmul","nppsMul_32f_I")
#endif // HAVE_LIBNPP
}

#endif /* HAVE_CUDA */
