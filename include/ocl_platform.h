#ifndef _OCL_PLATFORM_H_
#define _OCL_PLATFORM_H_

#ifdef HAVE_OPENCL

#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

struct ocl_platform_data {
	cl_platform_id		opd_pf_id;
	const char *		opd_profile;
	const char *		opd_version;
	const char *		opd_vendor;
	const char *		opd_extensions;
};

#define OPD_ID(opd_p)			(opd_p)->opd_pf_id
#define OPD_PROFILE(opd_p)		(opd_p)->opd_profile
#define OPD_VERSION(opd_p)		(opd_p)->opd_version
#define OPD_VENDOR(opd_p)		(opd_p)->opd_vendor
#define OPD_EXTENSIONS(opd_p)		(opd_p)->opd_extensions

#define SET_OPD_ID(opd_p,v)		(opd_p)->opd_pf_id = v
#define SET_OPD_PROFILE(opd_p,v)	(opd_p)->opd_profile = v
#define SET_OPD_VERSION(opd_p,v)	(opd_p)->opd_version = v
#define SET_OPD_VENDOR(opd_p,v)		(opd_p)->opd_vendor = v
#define SET_OPD_EXTENSIONS(opd_p,v)	(opd_p)->opd_extensions = v

struct ocl_dev_info {
	// implementation-specific stuff here...
	//cl_platform_id	odi_oclpf_id;
	//const char *		odi_pf_name;	// redundant storage but who cares.
	cl_device_id		odi_dev_id;
	cl_context		odi_ctx;
	cl_command_queue	odi_queue; //"stream" in CUDA
	const char *		odi_extensions;
} ;

#define ODI_DEV_ID(odi_p)	(odi_p)->odi_dev_id
#define ODI_CTX(odi_p)		(odi_p)->odi_ctx
#define ODI_QUEUE(odi_p)	(odi_p)->odi_queue
#define ODI_EXTENSIONS(odi_p)	(odi_p)->odi_extensions

#define SET_ODI_DEV_ID(odi_p,v)		(odi_p)->odi_dev_id = v
#define SET_ODI_CTX(odi_p,v)		(odi_p)->odi_ctx = v
#define SET_ODI_QUEUE(odi_p,v)		(odi_p)->odi_queue = v
#define SET_ODI_EXTENSIONS(odi_p,v)	(odi_p)->odi_extensions = v

struct ocl_stream_info {
	int foobar;
} ;

#endif // HAVE_OPENCL

#endif // _OCL_PLATFORM_H_

