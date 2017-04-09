#ifdef _WIN32
//#define USE_DLL_LINKING
#endif

//#if defined(USE_DLL_LINKING)
//// using CUDA runtime dynamic linking instead of static linking
//#include <dynlink/cuda_runtime_dynlink.h>
//using namespace dyn;
//#else

#ifdef HAVE_CUDA

#include <cuda.h>

#ifndef CUDA_ERROR_CHECK
#if CUDA_VERSION >= 6000
#define CUDA_ERROR_CHECK(string)	/* what to do? */
#elif CUDA_VERSION >= 5000
// CUDA 5
#define CUDA_ERROR_CHECK(string)	getLastCudaError(string);
#else // CUDA 4
#define CUDA_ERROR_CHECK(string)	cutilCheckMsg(string);
#endif
#endif // ! CUDA_ERROR_CHECK

//#include "cuda_port.h"	// BUILD_FOR_GPU, BUILD_FOR_CUDA
#define BUILD_FOR_GPU
#define BUILD_FOR_CUDA

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

#else // ! HAVE_CUDA

#define NO_CUDA_MSG(whence)					\
							\
	sprintf(ERROR_STRING,"%s:  Sorry, no CUDA support in this build.",#whence); \
	advise(ERROR_STRING);


#endif /* ! HAVE_CUDA */


#include "query.h"
#include "veclib_api.h"

typedef struct cuda_device {
	Item			cudev_item;
	int			cudev_index;
#ifdef HAVE_CUDA
	struct cudaDeviceProp	cudev_prop;
#endif
} Cuda_Device;

extern Cuda_Device *curr_cdp;

#define MAX_CUDA_DEVICES	2		// for now, in the vision lab.

#ifdef CUDA_CONSTANT_AREA_INDEX
#define N_CUDA_DEVICE_AREAS	4	/* global, host, host_mapped, const */
#else
#define N_CUDA_DEVICE_AREAS	3	/* global, host, host_mapped */
#endif

enum {
	CUDA_GLOBAL_AREA_INDEX,
	CUDA_HOST_AREA_INDEX,
	CUDA_HOST_MAPPED_AREA_INDEX,
};

//extern Data_Area *cuda_data_area[MAX_CUDA_DEVICES][N_CUDA_DEVICE_AREAS];

#ifdef FOOBAR
extern bitmap_word *gpu_bit_val_array;	/* BUG should have one per device */
#endif /* FOOBAR */

// These define's aren't too critical, they set the size of the
// initial namespace hashtable (? confirm)
#define MAX_CUDA_GLOBAL_OBJECTS	2048
#define MAX_CUDA_MAPPED_OBJECTS	128

#define NO_CUDA_DEVICE ((Cuda_Device *)NULL)
#define cudev_name	cudev_item.item_name

#define CONST_MEM_SIZE	0x8000	/* 32k bytes */

#define index_type	int32_t	// for vmaxi etc
#define INDEX_PREC	PREC_DI

ITEM_INTERFACE_PROTOTYPES( Cuda_Device, cudev )



#ifdef HAVE_CUDA
extern void print_cudev_properties(QSP_ARG_DECL  int, struct cudaDeviceProp *);
#endif // HAVE_CUDA

/* gpu_func_tbl.cpp */
extern int gpu_dispatch( Vector_Function *vfp, Vec_Obj_Args *oap );

extern COMMAND_FUNC( do_list_cudevs );
extern COMMAND_FUNC( do_cudev_info );
extern COMMAND_FUNC( do_report_npp_version );

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
extern void query_cuda_device(QSP_ARG_DECL  int dev);
#ifdef __cplusplus
}
#endif // __cplusplus

/* cuda.cpp */
extern void _init_cuda_devices(SINGLE_QSP_ARG_DECL);
extern void set_cuda_device(Cuda_Device *);
extern void *tmpvec(int,int,const char *whence);
extern void freetmp(void *,const char *whence);


extern COMMAND_FUNC( do_gpu_obj_upload );
extern COMMAND_FUNC( do_gpu_obj_dnload );


// More prototypes that aren't standard...

//FFT Functions
extern COMMAND_FUNC( do_gpu_fwdfft );

// NPP stuff - cuda_npp.cpp

extern COMMAND_FUNC( do_npp_malloc );
extern COMMAND_FUNC( do_npp_dilation );
extern COMMAND_FUNC( do_npp_erosion );
extern COMMAND_FUNC( do_npp_vadd );
extern COMMAND_FUNC( do_npp_filter );
extern COMMAND_FUNC( do_npp_sum );
extern COMMAND_FUNC( do_npp_sum_scratch );
extern COMMAND_FUNC( do_nppi_vmul );
extern COMMAND_FUNC( do_npps_vmul );

extern COMMAND_FUNC( do_init_checkpoints );
extern COMMAND_FUNC( do_set_checkpoint );
extern COMMAND_FUNC( do_clear_checkpoints );
extern COMMAND_FUNC( do_show_checkpoints );

extern void insure_cuda_device( Data_Obj *dp );

extern void h_sp_ifl(Data_Obj *dp, int x, int y, float tol, float v );
extern void h_sp_ifl2(Data_Obj *dp, int x, int y, float tol, float v );

extern void sp_cuda_centroid(Vec_Obj_Args *oap);
extern void dp_cuda_centroid(Vec_Obj_Args *oap);

// yuv2rgb.cu
extern void cuda_yuv422_to_rgb24(Data_Obj *,Data_Obj *);

// deviceQuery.cpp
extern COMMAND_FUNC( query_cuda_devices );

// simpleCUBLAS.cpp
extern int test_cublas(void);



