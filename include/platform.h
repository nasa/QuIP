#ifndef _PLATFORM_H_
#define _PLATFORM_H_

#include "data_obj.h"	// but this file is included in data_obj...
#include "veclib/obj_args.h"
struct vector_function;
//#include "veclib/vec_func.h"
//struct vec_func_array;
//struct vec_obj_args;

#ifdef HAVE_OPENCL
#define MAX_OPENCL_DEVICES	4
#endif // HAVE_OPENCL

#ifdef HAVE_OPENCL
#ifdef BUILD_FOR_OPENCL
#include "ocl_platform.h"
#endif // BUILD_FOR_OPENCL
#endif // HAVE_OPENCL




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

// platform or API?

typedef struct ocl_platform_data OCL_Platform_Data;

typedef struct dispatch_function {
	const int	df_index;
	void		(*df_func)(const int,const Vec_Obj_Args *);
} Dispatch_Function;

typedef struct compute_platform {
	Item		cp_item;
	platform_type	cp_type;
	Item_Context *	cp_icp;	// context for devices
//	Dispatch_Function *	cp_dispatch_tbl;
//#define PLATFORM_DISPATCH_TBL(cpp)	(cpp)->cp_dispatch_tbl
//#define PLATFORM_DISPATCH_FUNC(cpp,i)	(cpp)->cp_dispatch_tbl[i].df_func
//#define SET_PLATFORM_DISPATCH_TBL(cpp,v)	(cpp)->cp_dispatch_tbl = v

	// These are only relevant for GPUs...
	// These are probably device functions, not platform functions!
	void (*cp_mem_upload_func)(QSP_ARG_DECL  void *dst, void *src, size_t siz, struct platform_device *pdp );
	void (*cp_mem_dnload_func)(QSP_ARG_DECL  void *dst, void *src, size_t siz, struct platform_device *pdp );
	/*
	void (*cp_obj_upload_func)(QSP_ARG_DECL  Data_Obj *dpto, Data_Obj *dpfr);
	void (*cp_obj_dnload_func)(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr);
	*/

	/*int (*cp_dispatch_func)(QSP_ARG_DECL  struct dispatch_function *dfp, struct vec_obj_args *oap);*/
	int (*cp_mem_alloc_func)(QSP_ARG_DECL  Data_Obj *dp, dimension_t size, int align);
	void (*cp_mem_free_func)(QSP_ARG_DECL  Data_Obj *dp);
	void (*cp_offset_data_func)(QSP_ARG_DECL  Data_Obj *dp, index_t o );
	void (*cp_update_offset_func)(QSP_ARG_DECL  Data_Obj *dp);
	int (*cp_regbuf_func)(QSP_ARG_DECL  Data_Obj *dp);
	int (*cp_mapbuf_func)(QSP_ARG_DECL  Data_Obj *dp);
	int (*cp_unmapbuf_func)(QSP_ARG_DECL  Data_Obj *dp);
	void (*cp_devinfo_func)(QSP_ARG_DECL  struct platform_device *pdp);
	void (*cp_info_func)(QSP_ARG_DECL  struct compute_platform *pdp);

	void (*cp_fft2d_func)();
	void (*cp_ift2d_func)();
	/*
	void (*cp_fft2d_2_func)();
	void (*cp_ift2d_2_func)();
	*/
	void (*cp_fftrows_func)();
	void (*cp_iftrows_func)();

	struct vec_func_array *	cp_vfa_tbl;

	// This doesn't really need to be a union, as there is no
	// cuda platform-specific data?
#ifdef HAVE_ANY_GPU
	union {

#ifdef HAVE_OPENCL
		OCL_Platform_Data *	u_opd_p;
#define PF_ODP(pfp)		(pfp)->cp_data.u_opd_p
#endif // HAVE_OPENCL

#ifdef HAVE_CUDA
		/*Cuda_Platform_Data*/int	u_cpd;
#endif // HAVE_CUDA

	} cp_data;	// platform-specific
#endif // HAVE_ANY_GPU

} Compute_Platform;

/*
extern int cpu_dispatch(QSP_ARG_DECL  struct vector_function *vfp, struct vec_obj_args *oap);

#ifdef HAVE_OPENCL
extern int ocl_dispatch(QSP_ARG_DECL  struct vector_function *vfp, struct vec_obj_args *oap);
#endif // HAVE_OPENCL

#ifdef HAVE_CUDA
extern int cu2_dispatch(QSP_ARG_DECL  struct vector_function *vfp, struct vec_obj_args *oap);
#endif // HAVE_CUDA
*/

#define NO_PLATFORM	((Compute_Platform *)NULL)

ITEM_INTERFACE_PROTOTYPES( Compute_Platform, platform )

#define PICK_PLATFORM(pmpt)		pick_platform(QSP_ARG  pmpt)

#define PLATFORM_NAME(cpp)		(cpp)->cp_item.item_name

#define PF_CONTEXT(cpp)			(cpp)->cp_icp

#define PF_MEM_UPLOAD_FN(cpp)		(cpp)->cp_mem_upload_func
#define PF_MEM_DNLOAD_FN(cpp)		(cpp)->cp_mem_dnload_func
#define PF_ALLOC_FN(cpp)		(cpp)->cp_mem_alloc_func
#define PF_FREE_FN(cpp)			(cpp)->cp_mem_free_func
#define PF_OFFSET_DATA_FN(cpp)		(cpp)->cp_offset_data_func
#define PF_UPDATE_OFFSET_FN(cpp)	(cpp)->cp_update_offset_func
//#define PF_DISPATCH_FN(cpp)		(cpp)->cp_dispatch_func
//#define PF_DISPATCH_TBL(cpp)		(cpp)->cp_dispatch_tbl
#define PF_MAPBUF_FN(cpp)		(cpp)->cp_mapbuf_func
#define PF_UNMAPBUF_FN(cpp)		(cpp)->cp_unmapbuf_func
#define PF_REGBUF_FN(cpp)		(cpp)->cp_regbuf_func
#define PF_DEVINFO_FN(cpp)		(cpp)->cp_devinfo_func
#define PF_INFO_FN(cpp)			(cpp)->cp_info_func

#define PF_FFT2D_FN(cpp)		(cpp)->cp_fft2d_func
#define PF_IFT2D_FN(cpp)		(cpp)->cp_ift2d_func
/*
#define PF_FFT2D_2_FN(cpp)		(cpp)->cp_fft2d_2_func
#define PF_IFT2D_2_FN(cpp)		(cpp)->cp_ift2d_2_func
*/
#define PF_FFTROWS_FN(cpp)		(cpp)->cp_fftrows_func
#define PF_IFTROWS_FN(cpp)		(cpp)->cp_iftrows_func

#define IS_CPU_DEVICE(pdp)	IS_CPU_PLATFORM( PFDEV_PLATFORM(pdp) )

// What is RNGEN ???  random number generator???

#define SET_PF_CONTEXT(cpp,v)	(cpp)->cp_icp = v
#define SET_PF_MEM_UPLOAD_FN(cpp,v)	(cpp)->cp_mem_upload_func = v
#define SET_PF_MEM_DNLOAD_FN(cpp,v)	(cpp)->cp_mem_dnload_func = v
#define SET_PF_ALLOC_FN(cpp,v)		(cpp)->cp_mem_alloc_func = v
#define SET_PF_FREE_FN(cpp,v)		(cpp)->cp_mem_free_func = v
#define SET_PF_OFFSET_DATA_FN(cpp,v)	(cpp)->cp_offset_data_func = v
#define SET_PF_UPDATE_OFFSET_FN(cpp,v)	(cpp)->cp_update_offset_func = v
//#define SET_PF_DISPATCH_FN(cpp,v)	(cpp)->cp_dispatch_func = v
//#define SET_PF_DISPATCH_TBL(cpp,v)	(cpp)->cp_dispatch_tbl = v
#define SET_PF_MAPBUF_FN(cpp,v)		(cpp)->cp_mapbuf_func = v
#define SET_PF_UNMAPBUF_FN(cpp,v)	(cpp)->cp_unmapbuf_func = v
#define SET_PF_REGBUF_FN(cpp,v)		(cpp)->cp_regbuf_func = v
#define SET_PF_DEVINFO_FN(cpp,v)	(cpp)->cp_devinfo_func = v
#define SET_PF_INFO_FN(cpp,v)		(cpp)->cp_info_func = v

#define SET_PF_FFT2D_FN(cpp,v)		(cpp)->cp_fft2d_func = v
#define SET_PF_IFT2D_FN(cpp,v)		(cpp)->cp_ift2d_func = v
/*
#define SET_PF_FFT2D_2_FN(cpp,v)	(cpp)->cp_fft2d_2_func = v
#define SET_PF_IFT2D_2_FN(cpp,v)	(cpp)->cp_ift2d_2_func = v
*/
#define SET_PF_FFTROWS_FN(cpp,v)	(cpp)->cp_fftrows_func = v
#define SET_PF_IFTROWS_FN(cpp,v)	(cpp)->cp_iftrows_func = v

#define SET_PLATFORM_FUNCTIONS(cpp,stem)				\
									\
	SET_PF_MEM_UPLOAD_FN(	cpp,	stem##_mem_upload	);	\
	SET_PF_MEM_DNLOAD_FN(	cpp,	stem##_mem_dnload	);	\
	SET_PF_ALLOC_FN(	cpp,	stem##_mem_alloc	);	\
	SET_PF_FREE_FN(		cpp,	stem##_mem_free		);	\
	SET_PF_OFFSET_DATA_FN(	cpp,	stem##_offset_data	);	\
	SET_PF_UPDATE_OFFSET_FN(cpp,	stem##_update_offset	);	\
	SET_PF_REGBUF_FN(	cpp,	stem##_register_buf	);	\
	SET_PF_MAPBUF_FN(	cpp,	stem##_map_buf		);	\
	SET_PF_UNMAPBUF_FN(	cpp,	stem##_unmap_buf	);	\
	SET_PF_DEVINFO_FN(	cpp,	stem##_dev_info		);	\
	SET_PF_INFO_FN(		cpp,	stem##_info		);	\
	/*SET_PF_DISPATCH_FN(	cpp,	stem##_dispatch		);*/	\
									\
	SET_PF_FFT2D_FN(	cpp,	h_##stem##_fft2d	);	\
	SET_PF_IFT2D_FN(	cpp,	h_##stem##_ift2d	);	\
	/*SET_PF_FFT2D_2_FN(	cpp,	h_##stem##_fft2d_2	);	\
	SET_PF_IFT2D_2_FN(	cpp,	h_##stem##_ift2d_2	);*/	\
	SET_PF_FFTROWS_FN(	cpp,	h_##stem##_fftrows	);	\
	SET_PF_IFTROWS_FN(	cpp,	h_##stem##_iftrows	);	\


#define PF_FUNC_TBL(cpp)		(cpp)->cp_vfa_tbl
#define SET_PF_FUNC_TBL(cpp,v)	(cpp)->cp_vfa_tbl = v

#define PF_TYPE(cpp)	(cpp)->cp_type
#define PF_DATA(cpp)	(cpp)->cp_data
#define SET_PF_TYPE(cpp,v)	(cpp)->cp_type = v
// Not a pointer!
//#define SET_PF_DATA(cpp,v)	(cpp)->cp_data = v
#define IS_CPU_PLATFORM(cpp)	(PF_TYPE(cpp)==PLATFORM_CPU)

#define PF_OPD(cpp)	PF_DATA(cpp).u_opd_p
#define PF_OPD_ID(cpp)	PF_OPD(cpp)->opd_id
#define SET_PF_OPD_ID(cpp,v)	PF_OPD(cpp)->opd_id = v

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

#define MAX_CUDA_DEVICES	2		// for now, in the vision lab.

#endif // BUILD_FOR_CUDA
#endif // HAVE_CUDA

#ifdef HAVE_OPENCL
typedef struct ocl_dev_info  OCL_Dev_Info;

#endif // HAVE_OPENCL

struct platform_device {
	Item			pd_item;
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

#define NO_PFDEV	((Platform_Device *)NULL)

#define PFDEV_CUDA_INFO(pdp)		((pdp)->pd_dev_info.u_cdi_p)
#define SET_PFDEV_CUDA_INFO(pdp,v)	((pdp)->pd_dev_info.u_cdi_p) = v

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
#define PICK_PFDEV(pmpt)	pick_pfdev(QSP_ARG  pmpt)

extern Platform_Device *curr_pdp;

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


typedef struct platform_stream {
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
} Platform_Stream;

#define STREAM_NAME(psp)	(psp)->ps_item.item_name

#define NO_STREAM ((Platform_Stream *)NULL)
#define PICK_STREAM(s)	pick_stream(QSP_ARG s)
#define GET_STREAM(p)	get_stream(QSP_ARG  p)

ITEM_INTERFACE_PROTOTYPES( Platform_Stream , stream )

extern void select_pfdev(QSP_ARG_DECL  Platform_Device *pdp);
extern void insure_obj_pfdev(QSP_ARG_DECL  Data_Obj *dp, Platform_Device *pdp);
extern void gen_obj_upload(QSP_ARG_DECL  Data_Obj *dpto, Data_Obj *dpfr);
extern void gen_obj_dnload(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr);

extern Compute_Platform *creat_platform(QSP_ARG_DECL  const char *name, platform_type t);
extern void delete_platform(QSP_ARG_DECL  Compute_Platform *cpp);

extern void init_all_platforms(SINGLE_QSP_ARG_DECL);

extern void vl2_init_platform(SINGLE_QSP_ARG_DECL);
#ifdef HAVE_OPENCL
extern void ocl_init_platform(SINGLE_QSP_ARG_DECL);
#endif // HAVE_OPENCL
#ifdef HAVE_CUDA
extern void cu2_init_platform(SINGLE_QSP_ARG_DECL);
#endif // HAVE_CUDA

extern Item_Context *create_pfdev_context(QSP_ARG_DECL  const char *name);
extern void push_pfdev_context(QSP_ARG_DECL  Item_Context *icp);
extern Item_Context *pop_pfdev_context(SINGLE_QSP_ARG_DECL);

extern int platform_dispatch(QSP_ARG_DECL  const Compute_Platform *cpp,
				const struct vector_function *vfp,
				Vec_Obj_Args *oap );

extern int platform_dispatch_by_code(QSP_ARG_DECL  int code, Vec_Obj_Args *oap );
extern void dp_convert(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);

#endif // _PLATFORM_H_
