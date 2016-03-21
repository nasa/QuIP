#include "quip_config.h"

#ifdef HAVE_OPENCL

#include <stdio.h>
#include <stdlib.h>
//#include "data_obj.h"
//#include "veclib/vecgen.h"

#define BUILD_FOR_OPENCL
#include "my_ocl.h"

/* No need to explicitely include the OpenCL headers */
#ifdef HAVE_CLFFT_H
#include <clFFT.h>
#endif // HAVE_CLFFT_H

#include "quip_prot.h"
#include "veclib_api.h"
#include "veclib/ocl_veclib_prot.h"
#include "veclib/fftsupp.h"

// we need this for fftrows function...
// But it won't work because of how we handle offsets in OpenCL...

#define ROW_LOOP(real_dp,func,src_typ,dst_typ)				\
									\
	{								\
		for (i = 0; i < OBJ_ROWS( real_dp ); ++i) {		\
			SET_FFT_SRC_OFFSET( fap, i*OBJ_ROW_INC( OA_SRC1(oap) ) );	\
			SET_FFT_DST_OFFSET( fap, i*OBJ_ROW_INC( OA_DEST(oap) ) );	\
			func(fap);					\
		}							\
	}

#ifdef FOOBAR


/* In a 2D fft, the column transforms are always complex,
 * regardless of whether the input is real or comples.
 * That is because we do the rows first, so only the row
 * transforms are real->complex.
 */


#define COLUMN_LOOP(dp,func)						\
									\
	{								\
		dimension_t i;						\
									\
		SET_FFT_SRC( fap, NULL );					\
		SET_FFT_DST( fap, OBJ_DATA_PTR( dp ) );				\
									\
		for(i=0;i<OBJ_COLS( dp );i++){				\
			SET_FFT_DST_OFFSET( fap, i*OBJ_PXL_INC( dp ) );\
			func(fap);					\
		}							\
	}

#endif // FOOBAR

#define CLFFT_RESULT_CHECK(whence)					\
									\
	if( err != CL_SUCCESS ){					\
		fprintf(stderr,"Error %d in %s!?\n",err,#whence);	\
		exit(1);						\
	}

#ifdef HAVE_CLFFT
static void ocl_fft_shutdown(SINGLE_QSP_ARG_DECL)
{
	cl_int err;
	err=clfftTeardown();
	CLFFT_RESULT_CHECK(clfftTeardown)
}

static int ocl_fft_inited=FALSE;

#define OCL_FFT_INIT							\
	if( ! ocl_fft_inited ){						\
		ocl_fft_inited = TRUE;					\
fprintf(stderr,"OCL_FFT_INIT setting exit function\n");\
		do_on_exit(ocl_fft_shutdown);				\
	}
#else // !HAVE_CLFFT

#define OCL_FFT_INIT		// nop

#endif // !HAVE_CLFFT

#include "veclib/sp_defs.h"
#include "ocl_typed_fft.c"
#include "veclib/type_undefs.h"

#include "veclib/dp_defs.h"
#include "ocl_typed_fft.c"
#include "veclib/type_undefs.h"


#define FP_PREC_SWITCH(sw_dp,func)							\
										\
	switch( OBJ_MACH_PREC(sw_dp) ){						\
		case PREC_SP:							\
			HOST_TYPED_CALL_NAME(func,sp)(HOST_CALL_ARGS);		\
			break;							\
		case PREC_DP:							\
			HOST_TYPED_CALL_NAME(func,dp)(HOST_CALL_ARGS);		\
			break;							\
		default:							\
			fprintf(stderr,"Unexpected destination precision!?\n");	\
			break;							\
	}

#define FP_PREC_SWITCH_ISINV(sw_dp,func,isinv)					\
										\
	switch( OBJ_MACH_PREC(sw_dp) ){						\
		case PREC_SP:							\
			HOST_TYPED_CALL_NAME(func,sp)(HOST_CALL_ARGS,isinv);	\
			break;							\
		case PREC_DP:							\
			HOST_TYPED_CALL_NAME(func,dp)(HOST_CALL_ARGS,isinv);	\
			break;							\
		default:							\
			fprintf(stderr,"Unexpected destination precision!?\n");	\
			break;							\
	}

#define RC_SWITCH(sw_dp,rfunc,cfunc,isinv)					\
	Vec_Obj_Args oa1, *oap=(&oa1);						\
fprintf(stderr,"%s BEGIN, oap = 0x%lx\n",STRINGIFY(HOST_CALL_NAME(rfunc)),(u_long)oap);\
	setvarg2(oap,dst_dp,src_dp);						\
	SET_OA_PFDEV(oap,OBJ_PFDEV(OA_DEST(oap)));				\
	if( IS_REAL(sw_dp) ){							\
		FP_PREC_SWITCH(dst_dp,r##rfunc)					\
	} else {								\
		FP_PREC_SWITCH_ISINV(dst_dp,c##cfunc,0)				\
	}

void HOST_CALL_NAME(fft2d)( VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp )
{
	OCL_FFT_INIT
	RC_SWITCH(src_dp,fft2d,fft2d,0)
}

void HOST_CALL_NAME(ift2d)( VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp )
{
	OCL_FFT_INIT
	RC_SWITCH(dst_dp,ift2d,fft2d,1)
}

void HOST_CALL_NAME(fftrows)( VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp )
{
	OCL_FFT_INIT
	RC_SWITCH(src_dp,fftrows,fftrows,0)
}

void HOST_CALL_NAME(iftrows)( VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp )
{
	OCL_FFT_INIT
	RC_SWITCH(dst_dp,iftrows,fftrows,0)
}

#endif // HAVE_OPENCL

