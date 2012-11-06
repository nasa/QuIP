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

#ifdef HAVE_CURAND_H
#include <curand.h>
#endif /* HAVE_CURAND_H */

#include <cuda_runtime_api.h>

#endif /* HAVE_CUDA */

#include "query.h"
#include "nvf_api.h"

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

extern Data_Area *cuda_data_area[MAX_CUDA_DEVICES][N_CUDA_DEVICE_AREAS];

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


extern void print_cudev_properties(int, struct cudaDeviceProp *);

/* gpu_func_tbl.cpp */
extern int gpu_dispatch( Vec_Func *vfp, Vec_Obj_Args *oap );

extern COMMAND_FUNC( do_list_cudevs );
extern COMMAND_FUNC( do_cudev_info );

extern void query_cuda_device(int dev);

/* cuda.cpp */
extern void init_cuda_devices(SINGLE_QSP_ARG_DECL);
extern void set_cuda_device(QSP_ARG_DECL  Cuda_Device *);
extern void *tmpvec(int,int,const char *whence);
extern void freetmp(void *,const char *whence);
#ifdef HAVE_CUDA
extern int setup_slow_len(dim3 *,Size_Info *,dimension_t start_dim, int *dim_indices,int i_first,int n_vec);
#endif /* HAVE_CUDA */


extern COMMAND_FUNC( do_gpu_upload );
extern COMMAND_FUNC( do_gpu_dnload );

//Three vec methods
extern COMMAND_FUNC( do_gpu_vcmp );
extern COMMAND_FUNC( do_gpu_vibnd );
extern COMMAND_FUNC( do_gpu_vibnd );
extern COMMAND_FUNC( do_gpu_vbnd );
extern COMMAND_FUNC( do_gpu_vmaxm );
extern COMMAND_FUNC( do_gpu_vminm );
extern COMMAND_FUNC( do_gpu_vmax );
extern COMMAND_FUNC( do_gpu_vmin );
extern COMMAND_FUNC( do_gpu_vadd );
extern COMMAND_FUNC( do_gpu_rvsub );
extern COMMAND_FUNC( do_gpu_rvmul );
extern COMMAND_FUNC( do_gpu_rvdiv );
extern COMMAND_FUNC( do_gpu_vibnd );
extern COMMAND_FUNC( do_gpu_vibnd );
extern COMMAND_FUNC( do_gpu_vbnd );
extern COMMAND_FUNC( do_gpu_vmaxm );
extern COMMAND_FUNC( do_gpu_vminm );
extern COMMAND_FUNC( do_gpu_vmax );
extern COMMAND_FUNC( do_gpu_vmin );
extern COMMAND_FUNC( do_gpu_vand );
extern COMMAND_FUNC( do_gpu_vnand );
extern COMMAND_FUNC( do_gpu_vor );
extern COMMAND_FUNC( do_gpu_vxor );
extern COMMAND_FUNC( do_gpu_vmod );
extern COMMAND_FUNC( do_gpu_vshr );
extern COMMAND_FUNC( do_gpu_vshl );
extern COMMAND_FUNC( do_gpu_rvpow );
extern COMMAND_FUNC( do_gpu_vatan2 );
extern COMMAND_FUNC( do_gpu_vexp );

//Two vec methods
extern COMMAND_FUNC( do_gpu_convert );
extern COMMAND_FUNC( do_gpu_vsign );
extern COMMAND_FUNC( do_gpu_vabs );
extern COMMAND_FUNC( do_gpu_vnot );
extern COMMAND_FUNC( do_gpu_rvmov );
extern COMMAND_FUNC( do_gpu_vset );
extern COMMAND_FUNC( do_gpu_rvneg );
extern COMMAND_FUNC( do_gpu_rvsqr );
extern COMMAND_FUNC( do_gpu_rvrand );
extern COMMAND_FUNC( do_gpu_vnot );
extern COMMAND_FUNC( do_gpu_rvsqr );
extern COMMAND_FUNC( do_gpu_rvrand );
extern COMMAND_FUNC( do_gpu_vnot );
extern COMMAND_FUNC( do_gpu_vcomp );
//extern COMMAND_FUNC( do_gpu_vj0 );
//extern COMMAND_FUNC( do_gpu_vj1 );
extern COMMAND_FUNC( do_gpu_vrint );
extern COMMAND_FUNC( do_gpu_vfloor );
extern COMMAND_FUNC( do_gpu_vround );
extern COMMAND_FUNC( do_gpu_vceil );
extern COMMAND_FUNC( do_gpu_vlog );
extern COMMAND_FUNC( do_gpu_vlog10 );
extern COMMAND_FUNC( do_gpu_vatan );
extern COMMAND_FUNC( do_gpu_vtan );
extern COMMAND_FUNC( do_gpu_vcos ); 
extern COMMAND_FUNC( do_gpu_verf ); 
extern COMMAND_FUNC( do_gpu_vacos ); 
extern COMMAND_FUNC( do_gpu_vsin ); 
extern COMMAND_FUNC( do_gpu_vasin ); 
//extern COMMAND_FUNC( do_gpu_vpow );
extern COMMAND_FUNC( do_gpu_vsqrt ); 

//Two vec scalar methods
extern COMMAND_FUNC( do_gpu_vscmp );
extern COMMAND_FUNC( do_gpu_vscmp2 );
extern COMMAND_FUNC( do_gpu_vsmnm );
extern COMMAND_FUNC( do_gpu_vsmxm );
extern COMMAND_FUNC( do_gpu_viclp );
extern COMMAND_FUNC( do_gpu_vclip );
extern COMMAND_FUNC( do_gpu_vsmax );
extern COMMAND_FUNC( do_gpu_vsmin );
extern COMMAND_FUNC( do_gpu_rvsadd );
extern COMMAND_FUNC( do_gpu_rvssub );
extern COMMAND_FUNC( do_gpu_rvsmul );
extern COMMAND_FUNC( do_gpu_rvsdiv );
extern COMMAND_FUNC( do_gpu_rvsdiv2 );
extern COMMAND_FUNC( do_gpu_vscmp );
extern COMMAND_FUNC( do_gpu_viclp );
extern COMMAND_FUNC( do_gpu_vclip );
extern COMMAND_FUNC( do_gpu_vsmax );
extern COMMAND_FUNC( do_gpu_vsmin );
extern COMMAND_FUNC( do_gpu_rvsadd );
extern COMMAND_FUNC( do_gpu_rvssub );
extern COMMAND_FUNC( do_gpu_rvsmul );
extern COMMAND_FUNC( do_gpu_rvsdiv );
extern COMMAND_FUNC( do_gpu_rvsdiv2 );
extern COMMAND_FUNC( do_gpu_vsand );
extern COMMAND_FUNC( do_gpu_vsnand );
extern COMMAND_FUNC( do_gpu_vsor );
extern COMMAND_FUNC( do_gpu_vsxor );
extern COMMAND_FUNC( do_gpu_vsmod );
extern COMMAND_FUNC( do_gpu_vsmod2 );
extern COMMAND_FUNC( do_gpu_vsshr );
extern COMMAND_FUNC( do_gpu_vsshr2 );
extern COMMAND_FUNC( do_gpu_vsshl );
extern COMMAND_FUNC( do_gpu_vsshl2 );
extern COMMAND_FUNC( do_gpu_vsatan2 );
extern COMMAND_FUNC( do_gpu_vsatan22 );
extern COMMAND_FUNC( do_gpu_vspow );
extern COMMAND_FUNC( do_gpu_vspow2 );

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

extern void insure_cuda_device(Data_Obj *dp );

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



