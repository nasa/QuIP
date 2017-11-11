#ifndef _PLATFORM_H_
#define _PLATFORM_H_

#include "data_obj.h"	// but this file is included in data_obj...
#include "veclib/vecgen.h"
#include "veclib/obj_args.h"
struct vector_function;

#define MAX_DEVICES_PER_PLATFORM	4	// somewhat arbitrary...

#ifdef HAVE_OPENCL
#ifdef BUILD_FOR_OPENCL
#include "ocl_platform.h"
#endif // BUILD_FOR_OPENCL
#endif // HAVE_OPENCL

struct opencl_kernel_info;
typedef struct opencl_kernel_info OpenCL_Kernel_Info;

typedef union {
#ifdef HAVE_OPENCL
	OpenCL_Kernel_Info *ocl_kernel_info_p;
#endif // HAVE_OPENCL
#ifdef HAVE_CUDA
	CUDA_Kernel_Info *cuda_kernel_info_p;
#endif // HAVE_CUDA
	void *any_kernel_info_p;
} Kernel_Info_Ptr;




/* a platform uses a common set of subroutines, which
 * can operate on multiple devices.  For example, with
 * CUDA, nVidia provides a programming interface that
 * works over multiple devices.  With platforms, we
 * try to extend that further to non-nVidia devices supported
 * by the OpenCL platform...
 */

typedef enum {
	PLATFORM_CPU,
	PLATFORM_CUDA,
	PLATFORM_OPENCL,
	N_PLATFORM_TYPES
} platform_type;


enum {
	PFDEV_GLOBAL_AREA_INDEX,
	PFDEV_HOST_AREA_INDEX,
	PFDEV_HOST_MAPPED_AREA_INDEX,
	N_PFDEV_AREA_TYPES
};

typedef enum {
	PKS_KERNEL_QUALIFIER,
	PKS_ARG_QUALIFIER,
	N_PLATFORM_KERNEL_STRINGS
} Platform_Kernel_String_ID;

// platform or API?

typedef struct ocl_platform_data OCL_Platform_Data;

typedef struct dispatch_function {
	const int	df_index;
	void		(*df_func)(const int,const Vec_Obj_Args *);
} Dispatch_Function;

typedef enum {
	KERNEL_ARG_VECTOR,
	KERNEL_ARG_DBL,
	KERNEL_ARG_INT,
	N_KERNEL_ARG_TYPES
} Kernel_Arg_Type;

typedef struct compute_platform {
	Item		cp_item;
	const char *	cp_prefix_str;
	platform_type	cp_type;
	Item_Context *	cp_icp;	// context for devices

	// These are only relevant for GPUs...

	// upload:  host-to-device
	void (*cp_mem_upload_func)(QSP_ARG_DECL  void *dst, void *src, size_t siz, index_t offset, struct platform_device *pdp );

	// dnload:  device-to-host
	void (*cp_mem_dnload_func)(QSP_ARG_DECL  void *dst, void *src, size_t siz, index_t offset, struct platform_device *pdp );

	void * (*cp_mem_alloc_func)(QSP_ARG_DECL  Platform_Device *pdp, dimension_t size, int align);
	int (*cp_obj_alloc_func)(QSP_ARG_DECL  Data_Obj *dp, dimension_t size, int align);
	void (*cp_mem_free_func)(QSP_ARG_DECL  void *ptr);
	void (*cp_obj_free_func)(QSP_ARG_DECL  Data_Obj *dp);
	void (*cp_offset_data_func)(QSP_ARG_DECL  Data_Obj *dp, index_t o );
	void (*cp_update_offset_func)(QSP_ARG_DECL  Data_Obj *dp);
	int (*cp_regbuf_func)(QSP_ARG_DECL  Data_Obj *dp);
	int (*cp_mapbuf_func)(QSP_ARG_DECL  Data_Obj *dp);
	int (*cp_unmapbuf_func)(QSP_ARG_DECL  Data_Obj *dp);
	void (*cp_devinfo_func)(QSP_ARG_DECL  struct platform_device *pdp);
	void (*cp_info_func)(QSP_ARG_DECL  struct compute_platform *pdp);

	struct vec_func_array *	cp_vfa_tbl;

	// most useful for GPUs, but could compile kernels for CPU also???
	void * (*cp_make_kernel_func)(QSP_ARG_DECL  const char *src, const char *name, struct platform_device *pdp);
	const char * (*cp_kernel_string_func)(QSP_ARG_DECL  Platform_Kernel_String_ID which_str );
	void (*cp_store_kernel_func)(QSP_ARG_DECL  Kernel_Info_Ptr *kip_p, void *kp, struct platform_device *pdp);
	// The fetch function returns the platform-specific kernel pointer for the device,
	// while the Kernel_Info_Ptr points to a struct with kernels for all platform devices for the platform
	void * (*cp_fetch_kernel_func)(QSP_ARG_DECL  Kernel_Info_Ptr kip, struct platform_device *pdp);
	void (*cp_run_kernel_func)(QSP_ARG_DECL  void * kp, Vec_Expr_Node *arg_enp, struct platform_device *pdp);
	void (*cp_set_kernel_arg_func)(QSP_ARG_DECL  void *kp, int *idx_p, void *vp, Kernel_Arg_Type arg_type );

#ifdef HAVE_ANY_GPU

	// This doesn't really need to be a union, as there is no
	// cuda platform-specific data?

	union {

#ifdef HAVE_OPENCL
		OCL_Platform_Data *	u_opd_p;
#define PF_ODP(pfp)		(pfp)->cp_u.u_opd_p
#endif // HAVE_OPENCL

#ifdef HAVE_CUDA
		/*Cuda_Platform_Data*/int	u_cpd;
#endif // HAVE_CUDA

	} cp_u;	// platform-specific
#endif // HAVE_ANY_GPU

} Compute_Platform;

ITEM_INTERFACE_PROTOTYPES( Compute_Platform, platform )

#define pick_platform(pmpt)		_pick_platform(QSP_ARG  pmpt)
#define new_platform(name)		_new_platform(QSP_ARG  name)
#define del_platform(name)		_del_platform(QSP_ARG  name)
#define list_platforms(fp)		_list_platforms(QSP_ARG  fp)
#define platform_list()			_platform_list(SINGLE_QSP_ARG)

#define PLATFORM_NAME(cpp)		(cpp)->cp_item.item_name

#define PF_PREFIX_STR(cpp)		(cpp)->cp_prefix_str
#define SET_PF_PREFIX_STR(cpp,s)	(cpp)->cp_prefix_str = s

#define PF_CONTEXT(cpp)			(cpp)->cp_icp

#define PF_MEM_UPLOAD_FN(cpp)		(cpp)->cp_mem_upload_func
#define PF_MEM_DNLOAD_FN(cpp)		(cpp)->cp_mem_dnload_func
#define PF_MEM_ALLOC_FN(cpp)		(cpp)->cp_mem_alloc_func
#define PF_OBJ_ALLOC_FN(cpp)		(cpp)->cp_obj_alloc_func
#define PF_MEM_FREE_FN(cpp)		(cpp)->cp_mem_free_func
#define PF_OBJ_FREE_FN(cpp)		(cpp)->cp_obj_free_func
#define PF_OFFSET_DATA_FN(cpp)		(cpp)->cp_offset_data_func
#define PF_UPDATE_OFFSET_FN(cpp)	(cpp)->cp_update_offset_func
#define PF_MAPBUF_FN(cpp)		(cpp)->cp_mapbuf_func
#define PF_UNMAPBUF_FN(cpp)		(cpp)->cp_unmapbuf_func
#define PF_REGBUF_FN(cpp)		(cpp)->cp_regbuf_func
#define PF_DEVINFO_FN(cpp)		(cpp)->cp_devinfo_func
#define PF_INFO_FN(cpp)			(cpp)->cp_info_func
#define PF_MAKE_KERNEL_FN(cpp)		(cpp)->cp_make_kernel_func
#define PF_KERNEL_STRING_FN(cpp)	(cpp)->cp_kernel_string_func
#define PF_STORE_KERNEL_FN(cpp)		(cpp)->cp_store_kernel_func
#define PF_FETCH_KERNEL_FN(cpp)		(cpp)->cp_fetch_kernel_func
#define PF_RUN_KERNEL_FN(cpp)		(cpp)->cp_run_kernel_func
#define PF_SET_KERNEL_ARG_FN(cpp)	(cpp)->cp_set_kernel_arg_func

#define PF_FFT2D_FN(cpp)		(cpp)->cp_fft2d_func
#define PF_IFT2D_FN(cpp)		(cpp)->cp_ift2d_func
#define PF_FFTROWS_FN(cpp)		(cpp)->cp_fftrows_func
#define PF_IFTROWS_FN(cpp)		(cpp)->cp_iftrows_func

#define IS_CPU_DEVICE(pdp)	IS_CPU_PLATFORM( PFDEV_PLATFORM(pdp) )

// What is RNGEN ???  random number generator???

#define SET_PF_CONTEXT(cpp,v)	(cpp)->cp_icp = v
#define SET_PF_MEM_UPLOAD_FN(cpp,v)	(cpp)->cp_mem_upload_func = v
#define SET_PF_MEM_DNLOAD_FN(cpp,v)	(cpp)->cp_mem_dnload_func = v
#define SET_PF_MEM_ALLOC_FN(cpp,v)	(cpp)->cp_mem_alloc_func = v
#define SET_PF_OBJ_ALLOC_FN(cpp,v)	(cpp)->cp_obj_alloc_func = v
#define SET_PF_MEM_FREE_FN(cpp,v)	(cpp)->cp_mem_free_func = v
#define SET_PF_OBJ_FREE_FN(cpp,v)	(cpp)->cp_obj_free_func = v
#define SET_PF_OFFSET_DATA_FN(cpp,v)	(cpp)->cp_offset_data_func = v
#define SET_PF_UPDATE_OFFSET_FN(cpp,v)	(cpp)->cp_update_offset_func = v
#define SET_PF_MAPBUF_FN(cpp,v)		(cpp)->cp_mapbuf_func = v
#define SET_PF_UNMAPBUF_FN(cpp,v)	(cpp)->cp_unmapbuf_func = v
#define SET_PF_REGBUF_FN(cpp,v)		(cpp)->cp_regbuf_func = v
#define SET_PF_DEVINFO_FN(cpp,v)	(cpp)->cp_devinfo_func = v
#define SET_PF_INFO_FN(cpp,v)		(cpp)->cp_info_func = v
#define SET_PF_MAKE_KERNEL_FN(cpp,v)		(cpp)->cp_make_kernel_func = v
#define SET_PF_KERNEL_STRING_FN(cpp,v)		(cpp)->cp_kernel_string_func = v
#define SET_PF_STORE_KERNEL_FN(cpp,v)		(cpp)->cp_store_kernel_func = v
#define SET_PF_FETCH_KERNEL_FN(cpp,v)		(cpp)->cp_fetch_kernel_func = v
#define SET_PF_RUN_KERNEL_FN(cpp,v)		(cpp)->cp_run_kernel_func = v
#define SET_PF_SET_KERNEL_ARG_FN(cpp,v)	(cpp)->cp_set_kernel_arg_func = v

#define SET_PLATFORM_FUNCTIONS(cpp,stem)					\
										\
	SET_PF_MEM_UPLOAD_FN(		cpp,	stem##_mem_upload	);	\
	SET_PF_MEM_DNLOAD_FN(		cpp,	stem##_mem_dnload	);	\
	SET_PF_MEM_ALLOC_FN(		cpp,	stem##_mem_alloc	);	\
	SET_PF_OBJ_ALLOC_FN(		cpp,	stem##_obj_alloc	);	\
	SET_PF_MEM_FREE_FN(		cpp,	stem##_mem_free		);	\
	SET_PF_OBJ_FREE_FN(		cpp,	stem##_obj_free		);	\
	SET_PF_OFFSET_DATA_FN(		cpp,	stem##_offset_data	);	\
	SET_PF_UPDATE_OFFSET_FN(	cpp,	stem##_update_offset	);	\
	SET_PF_REGBUF_FN(		cpp,	stem##_register_buf	);	\
	SET_PF_MAPBUF_FN(		cpp,	stem##_map_buf		);	\
	SET_PF_UNMAPBUF_FN(		cpp,	stem##_unmap_buf	);	\
	SET_PF_DEVINFO_FN(		cpp,	stem##_dev_info		);	\
	SET_PF_INFO_FN(			cpp,	stem##_info		);	\
	SET_PF_KERNEL_STRING_FN(	cpp,	stem##_kernel_string	);	\
	SET_PF_MAKE_KERNEL_FN(		cpp,	stem##_make_kernel	);	\
	SET_PF_STORE_KERNEL_FN(		cpp,	stem##_store_kernel	);	\
	SET_PF_FETCH_KERNEL_FN(		cpp,	stem##_fetch_kernel	);	\
	SET_PF_RUN_KERNEL_FN(		cpp,	stem##_run_kernel	);	\
	SET_PF_SET_KERNEL_ARG_FN(	cpp,	stem##_set_kernel_arg	);	\
	/* end of function initializations */


#define PF_FUNC_TBL(cpp)		(cpp)->cp_vfa_tbl
#define SET_PF_FUNC_TBL(cpp,v)	(cpp)->cp_vfa_tbl = v

#define PF_TYPE(cpp)	(cpp)->cp_type
#define PF_DATA(cpp)	(cpp)->cp_u
#define SET_PF_TYPE(cpp,v)	(cpp)->cp_type = v
// Not a pointer!
//#define SET_PF_DATA(cpp,v)	(cpp)->cp_u = v
#define IS_CPU_PLATFORM(cpp)	(PF_TYPE(cpp)==PLATFORM_CPU)

#define PF_OPD(cpp)	PF_DATA(cpp).u_opd_p

#define OCLPF_ID(cpp)			OPD_ID(PF_OPD(cpp))
#define OCLPF_PROFILE(cpp)		OPD_PROFILE(PF_ODP(cpp))
#define OCLPF_VERSION(cpp)		OPD_VERSION(PF_ODP(cpp))
#define OCLPF_VENDOR(cpp)		OPD_VENDOR(PF_ODP(cpp))
#define OCLPF_EXTENSIONS(cpp)		OPD_EXTENSIONS(PF_ODP(cpp))

#define SET_OCLPF_ID(cpp,v)		SET_OPD_ID(PF_OPD(cpp),v)
#define SET_OCLPF_PROFILE(cpp,v)	SET_OPD_PROFILE(PF_ODP(cpp),v)
#define SET_OCLPF_VERSION(cpp,v)	SET_OPD_VERSION(PF_ODP(cpp),v)
#define SET_OCLPF_VENDOR(cpp,v)		SET_OPD_VENDOR(PF_ODP(cpp),v)
#define SET_OCLPF_EXTENSIONS(cpp,v)	SET_OPD_EXTENSIONS(PF_ODP(cpp),v)

struct cuda_dev_info;
typedef struct cuda_dev_info Cuda_Dev_Info;

#ifdef HAVE_CUDA
#ifdef BUILD_FOR_CUDA

#ifndef __cplusplus
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#endif // ! __cplusplus

struct cuda_dev_info {
	int			cudev_index;
	struct cudaDeviceProp	cudev_prop;
	curandGenerator_t	cudev_rngen;
} ;

#define CUDA_DEV_INDEX(cdi_p)		(cdi_p)->cudev_index
#define SET_CUDA_DEV_INDEX(cdi_p,v)	(cdi_p)->cudev_index = v
#define CUDA_DEV_PROP(cdi_p)		(cdi_p)->cudev_prop
#define SET_CUDA_DEV_PROP(cdi_p,v)	(cdi_p)->cudev_prop = v
#define CUDA_MAX_THREADS_PER_BLOCK(cdi_p)	(cdi_p)->cudev_prop.maxThreadsPerBlock

#define CUDA_DEV_RNGEN(cdi_p)		(cdi_p)->cudev_rngen
#define SET_CUDA_DEV_RNGEN(cdi_p,v)	(cdi_p)->cudev_rngen = v

#define PFDEV_CUDA_DEV_INDEX(pdp)		CUDA_DEV_INDEX( PFDEV_CUDA_INFO(pdp) )
#define SET_PFDEV_CUDA_DEV_INDEX(pdp,v)		CUDA_DEV_INDEX( PFDEV_CUDA_INFO(pdp) ) = v
#define PFDEV_CUDA_DEV_PROP(pdp)		CUDA_DEV_PROP( PFDEV_CUDA_INFO(pdp) )
#define SET_PFDEV_CUDA_DEV_PROP(pdp,v)		CUDA_DEV_PROP( PFDEV_CUDA_INFO(pdp) ) = v

#define PFDEV_CUDA_RNGEN(pdp)		CUDA_DEV_RNGEN( PFDEV_CUDA_INFO(pdp) )
#define SET_PFDEV_CUDA_RNGEN(pdp,v)	SET_CUDA_DEV_RNGEN( PFDEV_CUDA_INFO(pdp), v )

#define PFDEV_CUDA_MAX_THREADS_PER_BLOCK(pdp)		CUDA_MAX_THREADS_PER_BLOCK( PFDEV_CUDA_INFO(pdp) )


#endif // BUILD_FOR_CUDA
#endif // HAVE_CUDA

#ifdef HAVE_OPENCL
typedef struct ocl_dev_info  OCL_Dev_Info;

#define PFDEV_ODI(pdp)		(pdp)->pd_dev_info.u_odi_p
#define SET_PFDEV_ODI(pdp,v)	(pdp)->pd_dev_info.u_odi_p = v

#define PFDEV_OCL_DEV_ID(pdp)			ODI_DEV_ID( PFDEV_ODI(pdp) )
#define SET_PFDEV_OCL_DEV_ID(pdp,v)		ODI_DEV_ID( PFDEV_ODI(pdp) ) = v

#define OCLDEV_DEV_ID(pdp)		ODI_DEV_ID( PFDEV_ODI(pdp) )
#define OCLDEV_CTX(pdp)			ODI_CTX( PFDEV_ODI(pdp) )
#define OCLDEV_QUEUE(pdp)		ODI_QUEUE( PFDEV_ODI(pdp) )
#define OCLDEV_EXTENSIONS(pdp)		ODI_EXTENSIONS(PFDEV_ODI(pdp))

#define SET_OCLDEV_DEV_ID(pdp,v)	SET_ODI_DEV_ID( PFDEV_ODI(pdp),v)
#define SET_OCLDEV_CTX(pdp,v)		SET_ODI_CTX( PFDEV_ODI(pdp),v)
#define SET_OCLDEV_QUEUE(pdp,v)		SET_ODI_QUEUE( PFDEV_ODI(pdp),v)
#define SET_OCLDEV_EXTENSIONS(pdp,v)	SET_ODI_EXTENSIONS(PFDEV_ODI(pdp),v)


#endif // HAVE_OPENCL

struct platform_device {
	Item			pd_item;
	int			pd_idx;		// serial number of this device (0,1,2 ...)
	Compute_Platform *	pd_cpp;
	Data_Area *		pd_ap[N_PFDEV_AREA_TYPES];
	int			pd_max_dims;	// a proxy for compute capability
#define DEFAULT_PFDEV_MAX_DIMS	3	// newer cuda devices, openCL
#ifdef HAVE_ANY_GPU
	union {
		// We could save a few bytes by using pointers and dynamically
		// allocating...
#ifdef HAVE_OPENCL
		OCL_Dev_Info *	u_odi_p;
#endif // HAVE_OPENCL
#ifdef HAVE_CUDA
		Cuda_Dev_Info *	u_cdi_p;
#endif // HAVE_CUDA
	} pd_dev_info;
#endif // HAVE_ANY_GPU
} ;

#define PFDEV_SERIAL(pdp)		(pdp)->pd_idx
#define SET_PFDEV_SERIAL(pdp,v)		(pdp)->pd_idx = v

#define PFDEV_CUDA_INFO(pdp)		((pdp)->pd_dev_info.u_cdi_p)
#define SET_PFDEV_CUDA_INFO(pdp,v)	((pdp)->pd_dev_info.u_cdi_p) = v

#define PFDEV_OCL_DEV_INFO(pdp)		((pdp)->pd_dev_info.u_odi_p)
#define SET_PFDEV_OCL_DEV_INFO(pdp,v)	((pdp)->pd_dev_info.u_odi_p) = v

#define PFDEV_NAME(pdp)			(pdp)->pd_item.item_name

#define PFDEV_MAX_DIMS(pdp)		(pdp)->pd_max_dims
#define SET_PFDEV_MAX_DIMS(pdp,v)	(pdp)->pd_max_dims = v
#define PFDEV_PLATFORM(pdp)		(pdp)->pd_cpp
#define SET_PFDEV_PLATFORM(pdp,v)	(pdp)->pd_cpp = v
#define PFDEV_AREA(pdp,index)		(pdp)->pd_ap[index]
#define SET_PFDEV_AREA(pdp,index,ap)	(pdp)->pd_ap[index] = ap

#define PFDEV_DISPATCH_FUNC(pdp)	PF_DISPATCH_FN( PFDEV_PLATFORM(pdp) )
#define PFDEV_PLATFORM_TYPE(pdp)	PF_TYPE(PFDEV_PLATFORM(pdp))


ITEM_INTERFACE_PROTOTYPES( Platform_Device, pfdev )

#define pick_pfdev(pmpt)	_pick_pfdev(QSP_ARG  pmpt)
#define new_pfdev(name)		_new_pfdev(QSP_ARG  name)
#define pfdev_of(name)		_pfdev_of(QSP_ARG  name)
#define init_pfdevs()		_init_pfdevs(SINGLE_QSP_ARG)
#define pfdev_list()		_pfdev_list(SINGLE_QSP_ARG)
#define list_pfdevs(fp)		_list_pfdevs(QSP_ARG  fp)

// BUG these should be per-thread
extern Platform_Device *curr_pdp;
extern List *pdp_stack;

extern void list_pf_devs(QSP_ARG_DECL  platform_type t);

#ifdef HAVE_OPENCL
typedef struct ocl_stream_info  OCL_Stream_Info;
#endif // HAVE_OPENCL

struct cuda_stream_info;
typedef struct cuda_stream_info Cuda_Stream_Info;

#ifdef HAVE_CUDA
#ifdef BUILD_FOR_CUDA

struct cuda_stream_info {
	cudaStream_t	csi_stream;
};

#endif // BUILD_FOR_CUDA
#endif // HAVE_CUDA


/*typedef*/ struct platform_stream {
	Item	ps_item;
#ifdef HAVE_ANY_GPU
	union {
#ifdef HAVE_OPENCL
		OCL_Stream_Info *	u_osi_p;
#endif // HAVE_OPENCL
#ifdef HAVE_CUDA
		Cuda_Stream_Info *	u_csi_p;
#endif // HAVE_CUDA
	} ps_si;
#endif // HAVE_ANY_GPU
} /*Platform_Stream*/;

#define STREAM_NAME(psp)	(psp)->ps_item.item_name

#define PICK_STREAM(s)	pick_stream(QSP_ARG s)
#define GET_STREAM(p)	get_stream(QSP_ARG  p)

ITEM_INTERFACE_PROTOTYPES( Platform_Stream , stream )

extern void _select_pfdev(QSP_ARG_DECL  Platform_Device *pdp);
extern void insure_obj_pfdev(QSP_ARG_DECL  Data_Obj *dp, Platform_Device *pdp);
extern void gen_obj_upload(QSP_ARG_DECL  Data_Obj *dpto, Data_Obj *dpfr);
extern void gen_obj_dnload(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr);

#define select_pfdev(pdp)	_select_pfdev(QSP_ARG  pdp)

extern Compute_Platform *creat_platform(QSP_ARG_DECL  const char *name, platform_type t);
extern void delete_platform(QSP_ARG_DECL  Compute_Platform *cpp);

extern void init_all_platforms(SINGLE_QSP_ARG_DECL);

extern void vl2_init_platform(SINGLE_QSP_ARG_DECL);
#ifdef HAVE_OPENCL
extern void ocl_init_platform(SINGLE_QSP_ARG_DECL);
#endif // HAVE_OPENCL
#ifdef HAVE_CUDA
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
extern void cu2_init_platform(SINGLE_QSP_ARG_DECL);
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HAVE_CUDA

extern void _push_pfdev(QSP_ARG_DECL  Platform_Device *pdp);
extern Platform_Device * _pop_pfdev(SINGLE_QSP_ARG_DECL);

#define push_pfdev(pdp)	_push_pfdev(QSP_ARG  pdp)
#define pop_pfdev()	_pop_pfdev(SINGLE_QSP_ARG)

extern Item_Context *create_pfdev_context(QSP_ARG_DECL  const char *name);
extern void push_pfdev_context(QSP_ARG_DECL  Item_Context *icp);
extern Item_Context *pop_pfdev_context(SINGLE_QSP_ARG_DECL);

extern int platform_dispatch(QSP_ARG_DECL  const Compute_Platform *cpp,
				const struct vector_function *vfp,
				Vec_Obj_Args *oap );

extern int platform_dispatch_by_code(QSP_ARG_DECL  int code, Vec_Obj_Args *oap );
extern void dp_convert(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);

// currently in ocl.c but should be moved - BUG
extern void show_gpu_vector(QSP_ARG_DECL  Platform_Device *pdp, void *ptr, int len);

extern long set_fused_kernel_args(QSP_ARG_DECL  void *kernel, int *idx_p, Vec_Expr_Node *enp, Compute_Platform *cpp);

extern Platform_Device *default_pfdev(void);
extern void set_default_pfdev(Platform_Device *pdp);
#endif // _PLATFORM_H_
