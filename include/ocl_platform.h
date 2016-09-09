#ifndef _OCL_PLATFORM_H_
#define _OCL_PLATFORM_H_

#ifdef HAVE_OPENCL

#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

struct ocl_platform_data {
	cl_platform_id		opd_id;
	const char *		opd_profile;
	const char *		opd_version;
	const char *		opd_vendor;
	const char *		opd_extensions;
};

#define OCLPF_PROFILE(pfp)		(PF_OPD(pfp))->opd_profile
#define OCLPF_VERSION(pfp)		(PF_OPD(pfp))->opd_version
#define OCLPF_VENDOR(pfp)		(PF_OPD(pfp))->opd_vendor
#define OCLPF_EXTENSIONS(pfp)		(PF_OPD(pfp))->opd_extensions

#define SET_OCLPF_PROFILE(pfp,v)	(PF_OPD(pfp))->opd_profile = v
#define SET_OCLPF_VERSION(pfp,v)	(PF_OPD(pfp))->opd_version = v
#define SET_OCLPF_VENDOR(pfp,v)		(PF_OPD(pfp))->opd_vendor = v
#define SET_OCLPF_EXTENSIONS(pfp,v)	(PF_OPD(pfp))->opd_extensions = v

struct ocl_dev_info {
	// implementation-specific stuff here...
	//cl_platform_id	odi_oclpf_id;
	//const char *		odi_pf_name;	// redundant storage but who cares.
	cl_device_id		odi_dev_id;
	cl_context		odi_ctx;
	cl_command_queue	odi_queue; //"stream" in CUDA
	int			odi_idx;
} ;

#define PFDEV_ODI(pdp)		(pdp)->pd_dev_info.u_odi_p

#define OCLDEV_DEV_ID(pdp)		(PFDEV_ODI(pdp))->odi_dev_id
#define OCLDEV_CTX(pdp)			(PFDEV_ODI(pdp))->odi_ctx
#define OCLDEV_QUEUE(pdp)		(PFDEV_ODI(pdp))->odi_queue
#define OCLDEV_IDX(pdp)			(PFDEV_ODI(pdp))->odi_idx

#define SET_OCLDEV_DEV_ID(pdp,v)	(PFDEV_ODI(pdp))->odi_dev_id = v
#define SET_OCLDEV_CTX(pdp,v)		(PFDEV_ODI(pdp))->odi_ctx = v
#define SET_OCLDEV_QUEUE(pdp,v)		(PFDEV_ODI(pdp))->odi_queue = v
#define SET_OCLDEV_IDX(pdp,v)		(PFDEV_ODI(pdp))->odi_idx = v

struct ocl_stream_info {
	int foobar;
} ;

#endif // HAVE_OPENCL

#endif // _OCL_PLATFORM_H_

