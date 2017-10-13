#ifndef _MY_CU2_H_
#define _MY_CU2_H_

#ifdef _WIN32
//#define USE_DLL_LINKING
#endif

//#if defined(USE_DLL_LINKING)
//// using CUDA runtime dynamic linking instead of static linking
//#include <dynlink/cuda_runtime_dynlink.h>
//using namespace dyn;
//#else

#include "platform.h"	// MAX_DEVICES_PER_PLATFORM

#ifdef HAVE_CUDA
#define BUILD_FOR_CUDA

#include <cuda.h>

#if CUDA_VERSION >= 5000
// CUDA 5
#define CUDA_ERROR_CHECK(string)	getLastCudaError(string);
#else // CUDA_VERSION < 5000
// CUDA 4
#define CUDA_ERROR_CHECK(string)	cutilCheckMsg(string);
#endif // CUDA_VERSION < 5000

#ifdef FOOBAR
#if CUDA_VERSION >= 5000
#include <helper_cuda.h>
#else
#include <cutil.h>
#include <cutil_inline.h>
#endif
#endif // FOOBAR


#ifdef HAVE_CURAND_H
#include <curand.h>
#endif /* HAVE_CURAND_H */

#ifdef HAVE_CUFFT_H
#include <cufft.h>
#endif /* HAVE_CUFFT_H */

#include <cuda_runtime_api.h>

#endif /* HAVE_CUDA */

// These need to come after the cuda includes...



#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include "item_type.h"
#include "veclib/vec_func.h"
#include "veclib/obj_args.h"

#include "veclib/cu2_veclib_prot.h"	// BUILD_FOR_GPU, BUILD_FOR_CUDA
//#include "platform.h"

#include "veclib_api.h"

/* prototypes here */

extern void query_cuda_device(QSP_ARG_DECL  int dev);

// from old lib cuda - BUG consolidate!
extern void init_cuda_devices(SINGLE_QSP_ARG_DECL);

// cu2.c
extern void insure_cu2_device( QSP_ARG_DECL   Data_Obj *dp );

// whence?
#ifdef HAVE_CUDA
extern const char* getCUFFTError(cufftResult status);
#endif // HAVE_CUDA

extern void insure_cuda_device( Data_Obj *dp );

#ifdef __cplusplus
}
#endif // __cplusplus

struct cuda_kernel_info {
	char *	cki_ptx[MAX_DEVICES_PER_PLATFORM];
};

#define CUDA_KI_KERNEL(kip,idx)		((kip).cuda_kernel_info_p)->cki_ptx[idx]
#define SET_CUDA_KI_KERNEL(kip,idx,v)	((kip).cuda_kernel_info_p)->cki_ptx[idx] = v

#define MAX_CUDA_GLOBAL_OBJECTS	2048
#define MAX_CUDA_MAPPED_OBJECTS	128


#ifdef CUDA_CONSTANT_AREA_INDEX
#define N_CUDA_DEVICE_AREAS	4	/* global, host, host_mapped, const */
#else
#define N_CUDA_DEVICE_AREAS	3	/* global, host, host_mapped */
#endif

#ifdef FOOBAR
extern bitmap_word *gpu_bit_val_array;	/* BUG should have one per device */
#endif /* FOOBAR */

// These define's aren't too critical, they set the size of the
// initial namespace hashtable (? confirm)
#define MAX_CUDA_GLOBAL_OBJECTS	2048
#define MAX_CUDA_MAPPED_OBJECTS	128

#define cudev_name	cudev_item.item_name

#define CONST_MEM_SIZE	0x8000	/* 32k bytes */


//-------------------------old my_cuda.h above, my_ocl.h below-----------

#undef BUILD_FOR_OPENCL

#define NO_CUDA_MSG(whence)				\
							\
	sprintf(ERROR_STRING,"%s:  Sorry, no CUDA support in this build.",#whence); \
	advise(ERROR_STRING);

#define index_type	int32_t	// for vmaxi etc
#define INDEX_PREC	PREC_DI	// for vmaxi etc


// cu2.c
#ifdef FOOBAR
extern void cu2_mem_upload(QSP_ARG_DECL  void *dst, void *src, size_t siz );
extern void cu2_mem_dnload(QSP_ARG_DECL  void *dst, void *src, size_t siz );
extern int cu2_mem_alloc(QSP_ARG_DECL  Data_Obj *dp, dimension_t size, int align );
extern void cu2_mem_free(QSP_ARG_DECL  Data_Obj *dp );
#endif // FOOBAR


#endif // ! _MY_CU2_H_
