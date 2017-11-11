
#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//#include "item_type.h"
#include "quip_fwd.h"
#include "item_obj.h"

#include "veclib/vecgen.h"
//#include "veclib/vec_func.h"
//#include "veclib/obj_args.h"
//#include "platform.h"

#define BUILD_FOR_OPENCL

#define MAX_OPENCL_DEVICES	4

#define index_type	int32_t	// for vmaxi etc
#define INDEX_PREC	PREC_DI	// for vmaxi etc

#ifdef HAVE_OPENCL

typedef struct opencl_kernel_info {
	cl_kernel	oki_kernel[MAX_OPENCL_DEVICES];
} OpenCL_Kernel_Info;

#define OCL_KI_KERNEL( kip, idx )		((kip).ocl_kernel_info_p)->oki_kernel[idx]
#define SET_OCL_KI_KERNEL( kip, idx, kp )	((kip).ocl_kernel_info_p)->oki_kernel[idx] = kp

#endif // HAVE_OPENCL

/* structs */
typedef struct kernel {
	Item		k_item;
#ifdef HAVE_OPENCL
	cl_kernel	k_kernel;
#endif // HAVE_OPENCL
} Kernel;

#define KERN_NAME(kp)		(kp)->k_item.item_name
#define KERN_KERNEL(kp)		(kp)->k_kernel
#define SET_KERN_KERNEL(kp,v)	(kp)->k_kernel = v

// OpenCL platform is "Apple" on mac...

#define MAX_OCL_GLOBAL_OBJECTS	512
#define MAX_OCL_MAPPED_OBJECTS	128

extern Vec_Func_Array ocl_vfa_tbl[N_VEC_FUNCS];

/* prototypes here */

//extern void init_opencl_platform(void);
extern void shutdown_opencl_platform(void);
extern cl_kernel create_kernel(QSP_ARG_DECL  const char * name, const char *pathname);
//extern const char *load_file(QSP_ARG_DECL  const char *pathname, size_t *len);
extern void delete_kernel(QSP_ARG_DECL  Kernel *kp);

// ocl_utils.c

extern void report_ocl_error(QSP_ARG_DECL  cl_int status, const char *whence);
extern cl_program ocl_create_program(const char *buf, Platform_Device *pdp );
//extern cl_kernel ocl_create_kernel(cl_program program,
extern cl_kernel ocl_create_kernel(/*QSP_ARG_DECL*/  cl_program program,
			const char *name, Platform_Device *pdp );
extern void * ocl_make_kernel( QSP_ARG_DECL  const char *src, const char *name, Platform_Device *pdp );

extern void _insure_ocl_device(QSP_ARG_DECL  Data_Obj *dp);

#define insure_ocl_device(dp) _insure_ocl_device(QSP_ARG  dp)

extern void h_ocl_set_seed(int seed);

//extern void init_ocl(SINGLE_QSP_ARG_DECL);
//extern void ocl_alloc_data(QSP_ARG_DECL  Data_Obj *dp, dimension_t size);

#define OCL_STATUS_CHECK(stat,whence)				\
								\
	if( stat != CL_SUCCESS ){				\
		report_ocl_error(QSP_ARG  stat, #whence );	\
		return;						\
	}

#define OCL_STATUS_CHECK_WITH_RETURN(stat,whence,retval)	\
								\
	if( stat != CL_SUCCESS ){				\
		report_ocl_error(QSP_ARG  stat, #whence );	\
		return retval;					\
	}

