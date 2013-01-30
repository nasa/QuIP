
/* interface to nVidia NPP image processing library */

#include "quip_config.h"

char VersionId_cuda_cuda_npp[] = QUIP_VERSION_STRING;


#ifdef HAVE_CUDA

#include <stdio.h>

#include <cuda_runtime.h>

#ifdef HAVE_NPPI_H
#include <nppi.h>
#include <npps.h>
#endif /* HAVE_NPPI_H */

#define NPP_VERSION_CODE(major,minor,build)			\
	((major<<16)|(minor<<8)|build)

#define NPP_VERSION	NPP_VERSION_CODE(NPP_VERSION_MAJOR,NPP_VERSION_MINOR,NPP_VERSION_BUILD)

#include "query.h"
#include "data_obj.h"
#include "cuda_supp.h"

#ifndef HAVE_LIBNPP
#define NO_MPP_MSG(whence)			\
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

void report_npp_error(const char *whence, const char *funcname, NppStatus s)
{
	const char *m;

	switch(s){
		NPP_ERR_CASE(NPP_NOT_SUPPORTED_MODE_ERROR, "unsupported mode" )
		NPP_ERR_CASE(NPP_ROUND_MODE_NOT_SUPPORTED_ERROR,"unsupported round mode" )
		NPP_ERR_CASE( NPP_RESIZE_NO_OPERATION_ERROR, "No resize operation" )
		NPP_ERR_CASE( NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY, "insufficient compute capability" )
		NPP_ERR_CASE( NPP_BAD_ARG_ERROR, "bad argument" )
		NPP_ERR_CASE( NPP_LUT_NUMBER_OF_LEVELS_ERROR, "bad number of LUT levels" )
		NPP_ERR_CASE( NPP_TEXTURE_BIND_ERROR, "texture binding error" )
		NPP_ERR_CASE( NPP_COEFF_ERROR, "coefficient error" )
		NPP_ERR_CASE( NPP_RECT_ERROR, "rect error" )
		NPP_ERR_CASE( NPP_QUAD_ERROR, "quad error" )
		NPP_ERR_CASE( NPP_WRONG_INTERSECTION_ROI_ERROR, "wrong intersection ROI" )
		NPP_ERR_CASE( NPP_NOT_EVEN_STEP_ERROR, "not an even step" )
		NPP_ERR_CASE( NPP_INTERPOLATION_ERROR, "interpolation error" )
		NPP_ERR_CASE( NPP_RESIZE_FACTOR_ERROR, "bad resize factor" )
		NPP_ERR_CASE( NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR, "Haar classifier pixel match error" )
		NPP_ERR_CASE( NPP_MEMFREE_ERR, "memfree error" )
		NPP_ERR_CASE( NPP_MEMSET_ERR, "memset error" )
		NPP_ERR_CASE( NPP_MEMCPY_ERROR, "memcpy error" )
		NPP_ERR_CASE(NPP_MEM_ALLOC_ERR, "memory allocation error" )
		NPP_ERR_CASE(NPP_HISTO_NUMBER_OF_LEVELS_ERROR, "bad number of histogram levels" )
		NPP_ERR_CASE(NPP_MIRROR_FLIP_ERR, "mirror flip error" )
		NPP_ERR_CASE(NPP_INVALID_INPUT, "invalid input" )
		NPP_ERR_CASE(NPP_ALIGNMENT_ERROR, "alignment error" )
		NPP_ERR_CASE(NPP_STEP_ERROR, "step error" )
		NPP_ERR_CASE(NPP_SIZE_ERROR, "size error" )
		NPP_ERR_CASE(NPP_POINTER_ERROR, "pointer error" )
		NPP_ERR_CASE(NPP_NULL_POINTER_ERROR, "null pointer" )
		NPP_ERR_CASE(NPP_CUDA_KERNEL_EXECUTION_ERROR, "kernel execution error" )
		NPP_ERR_CASE(NPP_NOT_IMPLEMENTED_ERROR, "not implemented" )
		NPP_ERR_CASE(NPP_ERROR, "general NPP error" )
		NPP_ERR_CASE(NPP_WARNING, "general NPP warning" )
		NPP_ERR_CASE(NPP_WRONG_INTERSECTION_QUAD_WARNING, "wrong intersection quad" )
		NPP_ERR_CASE(NPP_MISALIGNED_DST_ROI_WARNING, "mis-aligned dst ROI" )
		NPP_ERR_CASE(NPP_AFFINE_QUAD_INCORRECT_WARNING, "affine quad incorrect" )
		NPP_ERR_CASE(NPP_DOUBLE_SIZE_WARNING, "ROI size was modified" )
		NPP_ERR_CASE(NPP_ODD_ROI_WARNING, "ROI size forced to an even value" )
		default:
			sprintf(DEFAULT_ERROR_STRING,
		"%s:  %s:  Unrecognized error (%d) in report_npp_error!?",
				whence,funcname,s);
			advise(DEFAULT_ERROR_STRING);
			//describe_cuda_error(whence,"xxx",e);
			return;
			break;
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
	siz.width = dst_dp->dt_cols + 1 - mask_dp->dt_cols;	\
	siz.height = dst_dp->dt_rows + 1 - mask_dp->dt_rows;	\
	msiz.width = mask_dp->dt_cols;				\
	msiz.height = mask_dp->dt_rows;

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
	if( dp->dt_prec != PREC_UBY ){
		sprintf(ERROR_STRING,
			"%s:  object %s (%s) must have %s precision",
			whence,dp->dt_name,name_for_prec(dp->dt_prec),
			name_for_prec(PREC_UBY));
		WARN(ERROR_STRING);
		return(0);
	}
	if( dp->dt_comps != 1 ){
		sprintf(ERROR_STRING,
			"%s:  object %s (%d) must have 1 component",
			whence,dp->dt_name,dp->dt_comps );
		WARN(ERROR_STRING);
		return(0);
	}
	if( dp->dt_frames != 1 || dp->dt_seqs != 1 ){
		sprintf(ERROR_STRING,
	"%s:  object %s (%d sequence%s of %d frame%s) must be a single image",
			whence,dp->dt_name,dp->dt_seqs,
			dp->dt_seqs==1?"":"s",dp->dt_frames,
			dp->dt_frames==1?"":"s");
		WARN(ERROR_STRING);
		return(0);
	}
	return(1);
}

static int good_kernel_for_filter(Data_Obj *dp, const char *whence )
{
	if( dp->dt_prec != PREC_DI ){
		sprintf(DEFAULT_ERROR_STRING,
			"%s:  object %s (%s) must have %s precision",
			whence,dp->dt_name,name_for_prec(dp->dt_prec),
			name_for_prec(PREC_DI));
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	if( dp->dt_comps != 1 ){
		sprintf(DEFAULT_ERROR_STRING,
			"%s:  object %s (%d) must have 1 component",
			whence,dp->dt_name,dp->dt_comps );
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	if( dp->dt_frames != 1 || dp->dt_seqs != 1 ){
		sprintf(DEFAULT_ERROR_STRING,
	"%s:  object %s (%d sequence%s of %d frame%s) must be a single image",
			whence,dp->dt_name,dp->dt_seqs,
			dp->dt_seqs==1?"":"s",dp->dt_frames,
			dp->dt_frames==1?"":"s");
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	return(1);
}

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
	if( dst_dp->dt_rows < mask_dp->dt_rows ||
			dst_dp->dt_cols < mask_dp->dt_cols ){
		sprintf(DEFAULT_ERROR_STRING,
	"%s:  destination %s (%d x %d) must be larger than mask %s (%d x %d)",
			whence,dst_dp->dt_name,dst_dp->dt_cols,dst_dp->dt_rows,
			mask_dp->dt_name,mask_dp->dt_cols,mask_dp->dt_rows);
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
	if( mask_dp->dt_prec != PREC_DI ){
		sprintf(ERROR_STRING,
	"%s:  kernel image %s (%s) should have %s precision",
			whence,mask_dp->dt_name,name_for_prec(mask_dp->dt_prec),
			name_for_prec(PREC_DI));
		WARN(ERROR_STRING);
		return(0);
	}
	if( ! good_img_for_morph(QSP_ARG  dst_dp,whence) )
		return(0);
	if( ! good_img_for_morph(QSP_ARG  src_dp,whence) )
		return(0);
	if( ! good_kernel_for_filter(mask_dp,whence) )
		return(0);
	if( dst_dp->dt_rows < mask_dp->dt_rows ||
			dst_dp->dt_cols < mask_dp->dt_cols ){
		sprintf(ERROR_STRING,
	"%s:  destination %s (%d x %d) must be larger than kernel %s (%d x %d)",
			whence,dst_dp->dt_name,dst_dp->dt_cols,dst_dp->dt_rows,
			mask_dp->dt_name,mask_dp->dt_cols,mask_dp->dt_rows);
		WARN(ERROR_STRING);
		return(0);
	}
	return(1);
}

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
	dst_p = (Npp8u *)dst_dp->dt_data;			\
	src_p = (Npp8u *)src_dp->dt_data;			\
	dst_p += anchor.x + anchor.y * dst_dp->dt_rowinc;	\
	src_p += anchor.x + anchor.y * src_dp->dt_rowinc;


COMMAND_FUNC( do_npp_erosion )
{
#ifdef HAVE_LIBNPP
	MORPH_DECLS
	GET_MORPH_ARGS
	CHECK_MORPH_ARGS("do_npp_erosion")
	SET_SIZES
	SET_OFFSETS

	s = nppiErode_8u_C1R(src_p,src_dp->dt_rowinc,
				dst_p, dst_dp->dt_rowinc,
				siz,(Npp8u *)mask_dp->dt_data,msiz,anchor);

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

	s = nppiDilate_8u_C1R(src_p, src_dp->dt_rowinc,
				dst_p, dst_dp->dt_rowinc,
				siz,(Npp8u *)mask_dp->dt_data,msiz,anchor);

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

	s = nppiFilter_8u_C1R(src_p, src_dp->dt_rowinc,
				dst_p, dst_dp->dt_rowinc, siz,
				(Npp32s *)mask_dp->dt_data,msiz,anchor,divisor);

	REPORT_NPP_STATUS("do_npp_filter","nppiFilter_8u_C1R")

#else /* ! HAVE_LIBNPP */
	NO_NPP_MSG("do_npp_dilation")
#endif /* ! HAVE_LIBNPP */
}

static prec_t validate_npp_prec( prec_t p )
{
	switch(p){
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
			return(p);
			break;
		default:
			return(PREC_NONE);
			break;
	}
}

#define M_CASE(prec,suffix)					\
	case prec: dp->dt_data = 				\
	nppiMalloc_##suffix(dp->dt_cols,dp->dt_rows,&stride);	\
	break;

#define M_DEFAULT						\
	default:						\
	sprintf(DEFAULT_ERROR_STRING,					\
"Sorry, no NPP allocator for %s precision with %d channels",	\
		name_for_prec(prec),ds.ds_dimension[0]);	\
	NWARN(DEFAULT_ERROR_STRING);

COMMAND_FUNC( do_npp_malloc )
{
#ifdef HAVE_LIBNPP
	Dimension_Set ds;
	prec_t prec,p;
	Data_Obj *dp;
	const char *s;
	int stride;

	s=NAMEOF("name for image");
	ds.ds_dimension[2] = HOW_MANY("number of rows");
	ds.ds_dimension[1] = HOW_MANY("number of columns");
	ds.ds_dimension[0] = HOW_MANY("number of channels");
	prec = get_precision(SINGLE_QSP_ARG);
	p = validate_npp_prec(prec);

	if( p == PREC_NONE ){
		sprintf(ERROR_STRING,
	"Sorry, precision (%s) requested for image %s is not supported by NPP",
			name_for_prec(prec),s);
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

	dp = _make_dp(QSP_ARG  s,&ds,prec);
	if( dp == NO_OBJ ) return;

	/* Now call the allocator */
	if( ds.ds_dimension[0] == 1 ){
		switch(prec){
			M_CASE(PREC_UBY,8u_C1)
			M_CASE(PREC_UIN,16u_C1)
			M_CASE(PREC_IN,16s_C1)
			M_CASE(PREC_DI,32s_C1)
			M_CASE(PREC_SP,32f_C1)
			M_DEFAULT
		}
	} else if( ds.ds_dimension[0] == 2 ){
		switch(prec){
			M_CASE(PREC_UBY,8u_C2)
			M_CASE(PREC_UIN,16u_C2)
			M_CASE(PREC_IN,16s_C2)
			M_CASE(PREC_SP,32f_C2)
			M_DEFAULT
		}
	} else if( ds.ds_dimension[0] == 3 ){
		switch(prec){
			M_CASE(PREC_UBY,8u_C3)
			M_CASE(PREC_UIN,16u_C3)
			M_CASE(PREC_DI,32s_C3)
			M_CASE(PREC_SP,32f_C3)
			M_DEFAULT
		}
	} else if( ds.ds_dimension[0] == 4 ){
		switch(prec){
			M_CASE(PREC_UBY,8u_C4)
			M_CASE(PREC_UIN,16u_C4)
			M_CASE(PREC_IN,16s_C4)
			M_CASE(PREC_DI,32s_C4)
			M_CASE(PREC_SP,32f_C4)
			M_DEFAULT
		}
	}
	/* stride is in bytes */
	dp->dt_rowinc = stride / siztbl[prec];

#else /* ! HAVE_LIBNPP */
	NO_NPP_MSG("do_npp_malloc");
#endif /* ! HAVE_LIBNPP */
} /* end do_npp_malloc */

#define A_CASE(prec,suffix,type)					\
	case prec:						\
	s = nppiAdd_##suffix( (const type *)src1_dp->dt_data, src1_step,\
		(const type *)src2_dp->dt_data, src2_step,		\
		(type *)dst_dp->dt_data, dst_step, size);		\
	break;

#define A_DEFAULT						\
	default:	sprintf(DEFAULT_ERROR_STRING,			\
"No NPP addition function for %d channels of type %s",	\
dst_dp->dt_comps,name_for_prec(dst_dp->dt_prec));		\
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

	pxl_size = siztbl[ MACHINE_PREC(dst_dp) ];
	if( IS_COMPLEX(dst_dp) ) pxl_size *= 2;
	dst_step = dst_dp->dt_rowinc * pxl_size;
	src1_step = src1_dp->dt_rowinc * pxl_size;
	src2_step = src2_dp->dt_rowinc * pxl_size;

	size.width = dst_dp->dt_cols;
	size.height = dst_dp->dt_rows;

	if( dst_dp->dt_comps == 1 ){
		switch(dst_dp->dt_prec){
			//A_CASE(PREC_UBY,8u_C1R)
			A_CASE(PREC_DI,32s_C1R,Npp32s)
			A_CASE(PREC_SP,32f_C1R,Npp32f)
			case PREC_UBY:
	s = nppiAdd_8u_C1RSfs( (const Npp8u *)src1_dp->dt_data, src1_step,
				(const Npp8u *)src2_dp->dt_data, src2_step,
				(Npp8u *)dst_dp->dt_data, dst_step, size, 1);
				break;

			A_DEFAULT
		}
	} else {
		sprintf(DEFAULT_ERROR_STRING,
			"npp_vadd:  component dimension %d not supported.",
			dst_dp->dt_comps);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	REPORT_NPP_STATUS("do_npp_vadd", "nppiAdd_8u_C1RSfs")

	/* No complex addition? */
	/*
	else if( dst_dp->dt_comps == 2 ){
		switch(dst_dp->dt_prec){
			A_CASE(PREC_CPX,32fc_C1R,Npp32fc)
		}
	}
	*/

#else /* ! HAVE_NPPLIB */
	NO_NPP_MSG("do_npp_vadd");
#endif /* ! HAVE_NPPLIB */
}

static int sum_scratch_size = 0;
static int sum_scratch_n = 0;
static Npp8u *scratch_buf=NULL;

static void get_scratch_for(Data_Obj *dp)
{
	int len, buf_size;
	NppStatus s;

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(DEFAULT_ERROR_STRING,
	"sum_scratch:  Object %s must be contiguous",dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	len = dp->dt_n_type_elts;
	/* BUG this function is gone in cuda 4.2,
	 * Not sure when it went away...
	 */
#ifdef FOOBAR
#if NPP_VERSION <= NPP_VERSION_CODE(4,0,17)
	s = nppsReductionGetBufferSize_32f( len, &buf_size );
#else	/* newer version of NPP */
	s = nppsSumGetBufferSize_32f( len, &buf_size );
#endif	/* newer version of NPP */
#endif /* FOOBAR */
/* not working on mac cuda 4.2, fix later... */
	s = nppsSumGetBufferSize_32f( len, &buf_size );

	REPORT_NPP_STATUS("do_npp_vadd","nppsReductionGetBufferSize_32f")

	// now allocate the memory...
sprintf(DEFAULT_ERROR_STRING,"Requesting %d scratch bytes",buf_size);
advise(DEFAULT_ERROR_STRING);
	scratch_buf = nppsMalloc_8u(buf_size);
	// BUG error check?
	sum_scratch_size = buf_size;
	sum_scratch_n = len;
}

COMMAND_FUNC( do_npp_sum_scratch )
{
	Data_Obj *src_dp;

	src_dp = PICK_OBJ("source object");
	if( src_dp == NO_OBJ ) return;

	// BUG make sure correct type...
	// BUG make sure contiguous...
	get_scratch_for(src_dp);
}

COMMAND_FUNC( do_npp_sum )
{
	Data_Obj *src_dp,*dst_dp;
	NppStatus s;

	dst_dp = PICK_OBJ("destination object");
	src_dp = PICK_OBJ("source object");

	if( dst_dp == NO_OBJ || src_dp == NO_OBJ )
		return;

	if( sum_scratch_size == 0 ){
		get_scratch_for(src_dp);
	} else if( (int) src_dp->dt_n_type_elts > sum_scratch_n ){
		// BUG
		nppsFree(scratch_buf);
		get_scratch_for(src_dp);
	}
	// BUG make sure the scratch space is the correct size

	s = nppsSum_32f((Npp32f*)src_dp->dt_data,src_dp->dt_n_type_elts,
		(Npp32f*)dst_dp->dt_data,nppAlgHintNone,scratch_buf);

	REPORT_NPP_STATUS("do_npp_sum","nppsSum_32f")
}

COMMAND_FUNC( do_nppi_vmul )
{
	Data_Obj *src1_dp, *src2_dp, *dst_dp;
	NppStatus s;
	NppiSize roi_size;

	dst_dp = PICK_OBJ("destination object");
	src1_dp = PICK_OBJ("first source object");
	src2_dp = PICK_OBJ("second source object");

	if( dst_dp == NO_OBJ || src1_dp == NO_OBJ || src2_dp == NO_OBJ )
		return;

	// BUG make sure sizes match
	// BUG make sure float precision, or switch on pixel type

	roi_size.width = dst_dp->dt_cols;
	roi_size.height = dst_dp->dt_rows;

	s = nppiMul_32f_C1R(	(Npp32f*) src1_dp->dt_data, src1_dp->dt_rowinc*sizeof(Npp32f),
				(Npp32f*) src2_dp->dt_data, src2_dp->dt_rowinc*sizeof(Npp32f),
				(Npp32f*) dst_dp->dt_data,  dst_dp->dt_rowinc*sizeof(Npp32f),
				roi_size );

	REPORT_NPP_STATUS("do_nppi_vmul","nppiMul_32f_C1R")
}

COMMAND_FUNC( do_npps_vmul )
{
	Data_Obj *src_dp, *dst_dp;
	NppStatus s;

	dst_dp = PICK_OBJ("destination/source object");
	src_dp = PICK_OBJ("source object");

	if( dst_dp == NO_OBJ || src_dp == NO_OBJ )
		return;

	// BUG make sure sizes match
	// BUG make sure float precision, or switch on pixel type

	s = nppsMul_32f_I(	(Npp32f*) src_dp->dt_data,
				(Npp32f*) dst_dp->dt_data,
				dst_dp->dt_n_type_elts );

	REPORT_NPP_STATUS("do_npps_vmul","nppsMul_32f_I")
}

#endif /* HAVE_CUDA */
