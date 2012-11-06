#include "quip_config.h"

#ifdef HAVE_CUDA

#include "quip_version.h"

char VersionId_cuda_cuda_centroid[] = QUIP_VERSION_STRING;

#include <cutil.h>
#include <cutil_inline.h>

#include "my_cuda.h"
#include "cuda_supp.h"
#include "vecgen.h"

#include "host_calls.h"

// The host call


#define HOST_CENT_FUNC( typ )					\
								\
void typ##_cent(Vector_Args *vap)				\
{								\
	BLOCK_VARS_DECLS					\
	dim3 len;						\
	int max_threads_per_block;				\
	DECL_SLOW_INCRS_3					\
								\
	max_threads_per_block =	curr_cdp->cudev_prop.maxThreadsPerBlock;\
	CLEAR_CUDA_ERROR(type_code##_slow_cent_helper)		\
	XFER_SLOW_LEN_3						\
	SETUP_BLOCKS_XY_					\
	SETUP_SLOW_INCRS_3					\
	REPORT_THREAD_INFO					\
REPORT_ARGS_3							\
	typ##_slow_cent_helper<<< NN_GPU >>>			\
		(GPU_SLEN_ARGS_3 );				\
    	cutilCheckMsg("kernel launch failure");			\
}								\
								\
/* Now the entry point */					\
								\
void typ##_cuda_centroid(Vec_Obj_Args *oap)			\
{								\
	Vector_Args va1;					\
	Spacing_Info spi1;					\
	Size_Info szi1;						\
								\
	va1.va_spi_p = &spi1;					\
	va1.va_szi_p = &szi1;					\
	insure_cuda_device( oap->oa_dest );			\
	XFER_SLOW_ARGS_3					\
	CHAIN_CHECK( typ##_cent )				\
	if( is_chaining ){					\
		if( insure_static(oap) < 0 ) return;		\
		add_link( & typ##_cent , &va1 );			\
		return;						\
	} else {						\
		typ##_cent(&va1);				\
		oap->oa_dest->dt_flags |= DT_ASSIGNED;		\
		oap->oa_dp[0]->dt_flags |= DT_ASSIGNED;		\
	}							\
}


#define HCF( t )	HOST_CENT_FUNC( t )


// The kernels

// SP stuff

#define std_type float
#define dest_type float
#define type_code sp
#include "gpu_cent.cu"
HCF( type_code )
#undef std_type
#undef dest_type
#undef type_code

#define std_type double
#define dest_type double
#define type_code dp
#include "gpu_cent.cu"
HCF( type_code )
#undef std_type
#undef dest_type
#undef type_code

#endif /* HAVE_CUDA */

