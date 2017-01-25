#include "quip_config.h"

#ifdef HAVE_CUDA

//#include "quip_version.h"
#define BUILD_FOR_CUDA

#include <cuda.h>
#include <curand.h>
#if CUDA_VERSION >= 5000
// but gone on 7.5...
//#include <helper_cuda.h>
#else
#include <cutil.h>
#include <cutil_inline.h>
#endif

#include "quip_prot.h"
#include "my_cuda.h"
#include "cuda_supp.h"
#include "veclib/vecgen.h"
#include "veclib/slow_len.h"
#include "veclib/gpu_args.h"
#include "veclib/gpu_call_utils.h"

// The kernels

#include "veclib/slow_defs.h"	// has to be slow to access x and y coords

// SP stuff

//#define std_type float
//#define dest_type float
//#define type_code sp
#include "veclib/sp_defs.h"
#include "gpu_cent.cu"
//HCF( type_code )
//#undef std_type
//#undef dest_type
//#undef type_code
#include "veclib/type_undefs.h"

//#define std_type double
//#define dest_type double
//#define type_code dp
#include "veclib/dp_defs.h"
#include "gpu_cent.cu"		// the kernel
//HCF( type_code )		// the host function
//#undef std_type
//#undef dest_type
//#undef type_code
#include "veclib/type_undefs.h"

#include "veclib/speed_undefs.h"

#include "veclib/host_typed_call_defs.h"
#include "../cu2/cu2_host_call_defs.h"
//#include "../cu2/cu2_kern_call_defs.h"

//#include "host_calls.h"

// The host call


#define HCF( t )	HOST_CENT_FUNC( t )


#define HOST_CENT_FUNC( typ )					\
								\
void typ##_cent(LINK_FUNC_ARG_DECLS)				\
{								\
	DECLARE_PLATFORM_VARS					\
	/*BLOCK_VARS_DECLS*/					\
	dim3 len;						\
	/*int max_threads_per_block;*/				\
	DECL_SLOW_INCRS_3					\
								\
	/*max_threads_per_block =	curr_cdp->cudev_prop.maxThreadsPerBlock;*/\
	CLEAR_CUDA_ERROR(type_code##_slow_cent_helper)		\
	/*XFER_SLOW_LEN_3*/					\
	/*SETUP_BLOCKS_XYZ_(VA_PFDEV(vap))*/			\
	SETUP_SLOW_INCS_3					\
	REPORT_THREAD_INFO					\
/*REPORT_ARGS_3*/							\
	typ##_slow_cent_helper<<< NN_GPU >>>			\
		(KERN_ARGS_SLEN_3 );				\
    	CUDA_ERROR_CHECK("kernel launch failure");		\
}								\
								\
/* Now the entry point */					\
								\
void typ##_cuda_centroid(HOST_CALL_ARG_DECLS)			\
{								\
	Vector_Args va1;					\
	Vector_Args *vap=(&va1);				\
	/*Spacing_Info spi1;*/					\
	/*Size_Info szi1;*/						\
								\
	/*SET_VA_SPACING(vap,&spi1);*/				\
	/*SET_VA_SIZE_INFO(vap,&szi1);*/				\
	insure_cuda_device( oap->oa_dest );			\
	XFER_SLOW_ARGS_3					\
	SETUP_SLOW_LEN_3					\
	CHAIN_CHECK( typ##_cent )				\
	if( is_chaining ){					\
		if( insure_static(oap) < 0 ) return;		\
		add_link( & typ##_cent , LINK_FUNC_ARGS );		\
		return;						\
	} else {						\
		typ##_cent(LINK_FUNC_ARGS);				\
		SET_OBJ_FLAG_BITS(OA_DEST(oap), DT_ASSIGNED);	\
		/* WHY set assigned flag on a source obj??? */	\
		/* Maybe because it's really a second destination? */	\
		SET_OBJ_FLAG_BITS(OA_SRC1(oap), DT_ASSIGNED);	\
	}							\
}

#include "veclib/sp_defs.h"
HCF( type_code )
#include "veclib/type_undefs.h"

#include "veclib/dp_defs.h"
HCF( type_code )
#include "veclib/type_undefs.h"

#endif /* HAVE_CUDA */

