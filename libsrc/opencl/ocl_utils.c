#include "quip_config.h"

#ifdef HAVE_OPENCL

#include <string.h>
#include "quip_prot.h"
#include "my_ocl.h"
//#include "veclib/ocl_port.h"
#include "veclib_api.h"
#include "veclib/vec_func.h"
#include "ocl_platform.h"
#include "fileck.h"

//#define MEM_SIZE (16)//suppose we have a vector with 128 elements
#define MAX_SOURCE_SIZE (0x100000)

// BUG these need to go into a struct...
//static cl_device_id	device_id = NULL;
//static cl_mem		memobj = NULL;		//device memory
//static cl_program	program = NULL;		//cl_prgram is a program executable created from the source or binary
//static cl_kernel	kernel = NULL;		//kernel function
//	cl_int i;

#ifdef FOOBAR

//#define DEFAULT_OCL_DEV_VAR	"DEFAULT_OCL_DEVICE"
#define OCL_STATUS_CHECK(stat,whence)				\
								\
	if( stat != CL_SUCCESS ){				\
		report_ocl_error(QSP_ARG  stat, #whence );	\
		return;						\
	}
#endif // FOOBAR

#define ERROR_CASE(code,string)	case code: msg = string; break;

void report_ocl_error(QSP_ARG_DECL  cl_int status, const char *whence)
{
	char *msg;

	switch(status){
ERROR_CASE( CL_DEVICE_NOT_FOUND,		"device not found"				)
ERROR_CASE( CL_DEVICE_NOT_AVAILABLE,		"device not available"				)
ERROR_CASE( CL_COMPILER_NOT_AVAILABLE,		"compiler not available"			)
ERROR_CASE( CL_MEM_OBJECT_ALLOCATION_FAILURE,	"mem object allocation failure"			)
ERROR_CASE( CL_OUT_OF_RESOURCES,		"out of resources"				)
ERROR_CASE( CL_OUT_OF_HOST_MEMORY,		"out of host memory"				)
ERROR_CASE( CL_PROFILING_INFO_NOT_AVAILABLE,	"profiling info not available"			)
ERROR_CASE( CL_MEM_COPY_OVERLAP,		"mem copy overlap"				)
ERROR_CASE( CL_IMAGE_FORMAT_MISMATCH,		"image format mismatch"				)
ERROR_CASE( CL_IMAGE_FORMAT_NOT_SUPPORTED,	"image format not supported"			)
ERROR_CASE( CL_BUILD_PROGRAM_FAILURE,		"build program failure"				)
ERROR_CASE( CL_MAP_FAILURE,			"map failure"					)
ERROR_CASE( CL_MISALIGNED_SUB_BUFFER_OFFSET,	"misaligned sub buffer offset"			)
ERROR_CASE( CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,	"exec status error for events in wait list"	)
ERROR_CASE( CL_COMPILE_PROGRAM_FAILURE,		"compile program failure"			)
ERROR_CASE( CL_LINKER_NOT_AVAILABLE,		"linker not available"				)
ERROR_CASE( CL_LINK_PROGRAM_FAILURE,		"link program failure"				)
ERROR_CASE( CL_DEVICE_PARTITION_FAILED,		"device partition failed"			)
ERROR_CASE( CL_KERNEL_ARG_INFO_NOT_AVAILABLE,	"kernel arg info not available"			)

ERROR_CASE( CL_INVALID_VALUE,			"invalid value"					)
ERROR_CASE( CL_INVALID_DEVICE_TYPE,		"invalid device type"				)
ERROR_CASE( CL_INVALID_PLATFORM,		"invalid platform"				)
ERROR_CASE( CL_INVALID_DEVICE,			"invalid device"				)
ERROR_CASE( CL_INVALID_CONTEXT,			"invalid context"				)
ERROR_CASE( CL_INVALID_QUEUE_PROPERTIES,	"invalid queue properties"			)
ERROR_CASE( CL_INVALID_COMMAND_QUEUE,		"invalid command queue"				)
ERROR_CASE( CL_INVALID_HOST_PTR,		"invalid host ptr"				)
ERROR_CASE( CL_INVALID_MEM_OBJECT,		"invalid mem object"				)
ERROR_CASE( CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,	"invalid image format descriptor"		)
ERROR_CASE( CL_INVALID_IMAGE_SIZE,		"invalid image size"				)
ERROR_CASE( CL_INVALID_SAMPLER,			"invalid sampler"				)
ERROR_CASE( CL_INVALID_BINARY,			"invalid binary"				)
ERROR_CASE( CL_INVALID_BUILD_OPTIONS,		"invalid build options"				)
ERROR_CASE( CL_INVALID_PROGRAM,			"invalid program"				)
ERROR_CASE( CL_INVALID_PROGRAM_EXECUTABLE,	"invalid program execuatble"			)
ERROR_CASE( CL_INVALID_KERNEL_NAME,		"invalid kernel name"				)
ERROR_CASE( CL_INVALID_KERNEL_DEFINITION,	"invalid kernel definition"			)
ERROR_CASE( CL_INVALID_KERNEL,			"invalid kernel"				)
ERROR_CASE( CL_INVALID_ARG_INDEX,		"invalid arg index"				)
ERROR_CASE( CL_INVALID_ARG_VALUE,		"invalid arg value"				)
ERROR_CASE( CL_INVALID_ARG_SIZE,		"invalid arg size"				)
ERROR_CASE( CL_INVALID_KERNEL_ARGS,		"invalid kernel args"				)
ERROR_CASE( CL_INVALID_WORK_DIMENSION,		"invalid work dimension"			)
ERROR_CASE( CL_INVALID_WORK_GROUP_SIZE,		"invalid work group size"			)
ERROR_CASE( CL_INVALID_WORK_ITEM_SIZE,		"invalid work item size"			)
ERROR_CASE( CL_INVALID_GLOBAL_OFFSET,		"invalid global offset"				)
ERROR_CASE( CL_INVALID_EVENT_WAIT_LIST,		"invalid event wait list"			)
ERROR_CASE( CL_INVALID_EVENT,			"invalid event"					)
ERROR_CASE( CL_INVALID_OPERATION,		"invalid operation"				)
ERROR_CASE( CL_INVALID_GL_OBJECT,		"invalid gl object"				)
ERROR_CASE( CL_INVALID_BUFFER_SIZE,		"invalid buffer size"				)
ERROR_CASE( CL_INVALID_MIP_LEVEL,		"invalid mip level"				)
ERROR_CASE( CL_INVALID_GLOBAL_WORK_SIZE,	"invalid global work size"			)
ERROR_CASE( CL_INVALID_PROPERTY,		"invalid property"				)
ERROR_CASE( CL_INVALID_IMAGE_DESCRIPTOR,	"invalid image descriptor"			)
ERROR_CASE( CL_INVALID_COMPILER_OPTIONS,	"invalid compiler options"			)
ERROR_CASE( CL_INVALID_LINKER_OPTIONS,		"invalid linker options"			)
ERROR_CASE( CL_INVALID_DEVICE_PARTITION_COUNT,	"invalid device partition count"		)

		default:
			msg="unhandled";
			sprintf(ERROR_STRING,
	"report_ocl_error:  Unhandled error code %d (0x%x)!?",status,status);
			WARN(ERROR_STRING);
			break;
	}
	sprintf(ERROR_STRING,"%s:  %s",whence,msg);
	WARN(ERROR_STRING);
}

/* cl_device_type - bitfield
 *
 * CL_DEVICE_TYPE_DEFAULT
 * CL_DEVICE_TYPE_CPU
 * CL_DEVICE_TYPE_GPU
 * CL_DEVICE_TYPE_ACCELERATOR
 * CL_DEVICE_TYPE_CUSTOM
 * CL_DEVICE_TYPE_ALL
 */

/* Possible values for device XXX:
 *
 * CL_DEVICE_EXECUTION_CAPABILITIES
 * CL_DEVICE_NAME
 * CL_DEVICE_VENDOR
 * CL_DEVICE_PLATFORM
 * CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF
 * CL_DEVICE_HOST_UNIFIED_MEMORY
 * CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR
 * CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT
 * CL_DEVICE_NATIVE_VECTOR_WIDTH_INT
 * CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG
 * CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT
 * CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE
 * CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF
 * CL_DEVICE_OPENCL_C_VERSION
 * CL_DEVICE_BUILT_IN_KERNELS
 * CL_DEVICE_IMAGE_MAX_BUFFER_SIZE
 * CL_DEVICE_IMAGE_MAX_ARRAY_SIZE
 * CL_DEVICE_PARENT_DEVICE
 * CL_DEVICE_PARTITION_MAX_SUB_DEVICES
 * CL_DEVICE_PARTITION_PROPERTIES
 * CL_DEVICE_PARTITION_AFFINITY_DOMAIN
 * CL_DEVICE_PARTITION_TYPE
 * CL_DEVICE_REFERENCE_COUNT
 * CL_DEVICE_PREFERRED_INTEROP_USER_SYNC
 * CL_DEVICE_PRINTF_BUFFER_SIZE
 * CL_DEVICE_IMAGE_PITCH_ALIGNMENT
 * CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT
 */

typedef enum {
	PS_UINT,	// cl_uint
	PS_BOOL,	// cl_bool
	PS_FP_CONF,	// cl_device_fp_config (enum)
	PS_EXCAP,	// cl_device_exec_capabilities (enum)
	PS_STR,		// char[]
	PS_UL,		// cl_ulong
	PS_CTYP,	// cl_device_mem_cache_type
	PS_LTYP,	// cl_device_local_mem_type
	PS_SIZ,		// size_t
	PS_SIZA,	// size_t[]
	PS_DTYP,	// cl_device_type
	PS_QP,		// cl_command_queue_properties
	N_PS_TYPES
} OCL_Dev_Param_Type;

typedef struct ocl_dev_param_spec {
	const char *		devprm_name;
	cl_int			devprm_code;
	OCL_Dev_Param_Type	devprm_type;
} OCL_Dev_Param_Spec;

#define PS_NAME(psp)	(psp)->devprm_name
#define PS_CODE(psp)	(psp)->devprm_code
#define PS_TYPE(psp)	(psp)->devprm_type

#ifdef NOT_USED
static OCL_Dev_Param_Spec dev_param_tbl_short[]={
{ "type",			CL_DEVICE_TYPE,				PS_DTYP	},
{ "vendor",			CL_DEVICE_VENDOR,			PS_STR	},
{ "version",			CL_DEVICE_VERSION,			PS_STR	},
{ "driver_version",		CL_DRIVER_VERSION,			PS_STR	},
{ "available",			CL_DEVICE_AVAILABLE,			PS_BOOL	},
{ "global_mem_size",		CL_DEVICE_GLOBAL_MEM_SIZE,		PS_UL },
{ "global_mem_cache_size",	CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,	PS_UL },
{ "local_mem_size",		CL_DEVICE_LOCAL_MEM_SIZE,		PS_UL },
{ "max_compute_units",		CL_DEVICE_MAX_COMPUTE_UNITS,		PS_UINT	},
{ "max_work_group_size",	CL_DEVICE_MAX_WORK_GROUP_SIZE,		PS_SIZ	},
};

#define N_DEV_PARAMS_SHORT	(sizeof(dev_param_tbl_short)/sizeof(OCL_Dev_Param_Spec))
#endif // NOT_USED

#ifdef NOT_YET
static OCL_Dev_Param_Spec dev_param_tbl_long[]={
{ "type",				CL_DEVICE_TYPE,					PS_DTYP	},
{ "available",				CL_DEVICE_AVAILABLE,				PS_BOOL	},
{ "compiler_available",			CL_DEVICE_COMPILER_AVAILABLE,			PS_BOOL	},
{ "linker_available",			CL_DEVICE_LINKER_AVAILABLE,			PS_BOOL	},
{ "double_fp_config",			CL_DEVICE_DOUBLE_FP_CONFIG,			PS_FP_CONF	},
{ "endian_little",			CL_DEVICE_ENDIAN_LITTLE,			PS_BOOL	},
{ "error_correction_support",		CL_DEVICE_ERROR_CORRECTION_SUPPORT,		PS_BOOL	},
{ "execution_capabilities",		CL_DEVICE_ERROR_CORRECTION_SUPPORT,		PS_EXCAP },
{ "extensions",				CL_DEVICE_EXTENSIONS,				PS_STR },
{ "global_mem_cache_size",		CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,		PS_UL },
{ "global_mem_cache_type",		CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,		PS_CTYP },
{ "global_mem_cacheline_size",		CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,		PS_UINT },
{ "global_mem_size",			CL_DEVICE_GLOBAL_MEM_SIZE,			PS_UL },
// half fp is reserved but not currently supported...
//{ "half_fp_config",			CL_DEVICE_HALF_FP_CONFIG,			PS_FP_CONF	},
{ "image_support",			CL_DEVICE_IMAGE_SUPPORT,			PS_BOOL	},
{ "image2d_max_height",			CL_DEVICE_IMAGE2D_MAX_HEIGHT,			PS_SIZ	},
{ "image2d_max_width",			CL_DEVICE_IMAGE2D_MAX_WIDTH,			PS_SIZ	},
{ "image3d_max_depth",			CL_DEVICE_IMAGE3D_MAX_DEPTH,			PS_SIZ	},
{ "image3d_max_height",			CL_DEVICE_IMAGE3D_MAX_HEIGHT,			PS_SIZ	},
{ "image3d_max_width",			CL_DEVICE_IMAGE3D_MAX_WIDTH,			PS_SIZ	},
{ "local_mem_size",			CL_DEVICE_LOCAL_MEM_SIZE,			PS_UL },
{ "local_mem_type",			CL_DEVICE_LOCAL_MEM_TYPE,			PS_LTYP },
{ "max_clock_frequency",		CL_DEVICE_MAX_CLOCK_FREQUENCY,			PS_UINT	},
{ "max_compute_units",			CL_DEVICE_MAX_COMPUTE_UNITS,			PS_UINT	},
{ "max_constant_args",			CL_DEVICE_MAX_CONSTANT_ARGS,			PS_UINT	},
{ "max_constant_buffer_size",		CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,		PS_UL },
{ "max_mem_alloc_size",			CL_DEVICE_MAX_MEM_ALLOC_SIZE,			PS_UL	},
{ "max_parameter_size",			CL_DEVICE_MAX_PARAMETER_SIZE,			PS_SIZ	},
{ "max_read_image_args",		CL_DEVICE_MAX_READ_IMAGE_ARGS,			PS_UINT	},
{ "max_samplers",			CL_DEVICE_MAX_SAMPLERS,				PS_UINT	},
{ "max_work_group_size",		CL_DEVICE_MAX_WORK_GROUP_SIZE,			PS_SIZ	},
{ "max_work_item_dimensions",		CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,		PS_UINT	},
{ "max_work_item_sizes",		CL_DEVICE_MAX_WORK_ITEM_SIZES,			PS_SIZA	},
{ "max_write_image_args",		CL_DEVICE_MAX_WRITE_IMAGE_ARGS,			PS_UINT	},
{ "mem_base_addr_align",		CL_DEVICE_MEM_BASE_ADDR_ALIGN,			PS_UINT	},
{ "min_data_type_align_size",		CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,		PS_UINT	},
{ "preferred_vector_width_char",	CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,		PS_UINT	},
{ "preferred_vector_width_short",	CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,		PS_UINT	},
{ "preferred_vector_width_int",		CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,		PS_UINT	},
{ "preferred_vector_width_long",	CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,		PS_UINT	},
{ "preferred_vector_width float",	CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,		PS_UINT	},
{ "preferred_vector_width_double",	CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,	PS_UINT	},
{ "profile",				CL_DEVICE_PROFILE,				PS_STR },
{ "profiling_timer_resolution",		CL_DEVICE_PROFILING_TIMER_RESOLUTION,		PS_SIZ	},
{ "queue_properties",			CL_DEVICE_QUEUE_PROPERTIES,			PS_QP	},
{ "single_fp_config",			CL_DEVICE_SINGLE_FP_CONFIG,			PS_FP_CONF	},
{ "vendor",				CL_DEVICE_VENDOR,				PS_STR	},
{ "vendor_id",				CL_DEVICE_VENDOR_ID,				PS_UINT	},
{ "version",				CL_DEVICE_VERSION,				PS_STR	},
{ "driver_version",			CL_DRIVER_VERSION,				PS_STR	},

/* not documented on web page??? */
//{ "address_bits",			CL_DEVICE_ADDRESS_BITS,				PS_INT	},
};

#define N_DEV_PARAMS_LONG	(sizeof(dev_param_tbl_long)/sizeof(OCL_Dev_Param_Spec))

#endif // NOT_YET

#define MAX_PARAM_SIZE	128

#ifdef NOT_USED

static void display_dev_param(QSP_ARG_DECL  OCL_Dev_Param_Spec *psp,
								Platform_Device *pdp)
{
	long param_data[MAX_PARAM_SIZE/sizeof(long)];	// force alignment
	cl_int status;
	size_t psize;
	cl_uint *uip;
	cl_ulong *ulp;
	size_t *szp;
	cl_bool *bp;
	cl_device_type *tp;
	char str[LLEN];

	status = clGetDeviceInfo(OCLDEV_DEV_ID(pdp),PS_CODE(psp),MAX_PARAM_SIZE,
		param_data,&psize);
	if( status != CL_SUCCESS ){
		switch(status){
			case CL_INVALID_DEVICE:
				sprintf(MSG_STR,"\t%s:  %s",PS_NAME(psp), "invalid device");
				prt_msg(MSG_STR);
				return;
			case CL_INVALID_VALUE:
				sprintf(MSG_STR,"\t%s:  %s",PS_NAME(psp), "invalid value");
				prt_msg(MSG_STR);
				return;
			case CL_OUT_OF_RESOURCES:
				sprintf(MSG_STR,"\t%s:  %s",PS_NAME(psp), "out of resources");
				prt_msg(MSG_STR);
				return;
			case CL_OUT_OF_HOST_MEMORY:
				sprintf(MSG_STR,"\t%s:  %s",PS_NAME(psp), "out of host memory");
				prt_msg(MSG_STR);
				return;
			default:
				sprintf(MSG_STR,"\t%s:  %s",PS_NAME(psp), "unhandled error");
				prt_msg(MSG_STR);
				report_ocl_error(QSP_ARG  status,"clGetDeviceInfo");
				return;
			}
	}
	// Now figure out how to display...
	switch(PS_TYPE(psp)){
		case PS_UINT:
			// cl_uint
			uip = (cl_uint *) param_data;
			sprintf(MSG_STR,"\t%s:  %d",PS_NAME(psp),*uip);
			prt_msg(MSG_STR);
			break;
		case PS_BOOL:
			// cl_bool
			bp = (cl_bool *) param_data;
			sprintf(MSG_STR,"\t%s:  %s",PS_NAME(psp),
				(*bp)?"yes":"no");
			prt_msg(MSG_STR);
			break;
		case PS_STR:
			// char[]
			sprintf(MSG_STR,"\t%s:  %s",PS_NAME(psp), (char *) param_data);
			prt_msg(MSG_STR);
			break;
		case PS_UL:
			// cl_ulong
			ulp = (cl_ulong *) param_data;
			sprintf(MSG_STR,"\t%s:  %llu",PS_NAME(psp),*ulp);
			prt_msg(MSG_STR);
			break;
		case PS_SIZ:
			// size_t
			szp = (size_t *) param_data;
			sprintf(MSG_STR,"\t%s:  %ld",PS_NAME(psp),*szp);
			prt_msg(MSG_STR);
			break;

		case PS_DTYP:
			// cl_device_type
			tp = (cl_device_type *) param_data;
			// this is a bit field
			str[0]=0;
#define CHECK_BIT(code,tag) if( (*tp) & code ) { strcat(str," "); strcat(str,#tag); }
			CHECK_BIT(CL_DEVICE_TYPE_CPU,cpu)
			CHECK_BIT(CL_DEVICE_TYPE_GPU,gpu)
			CHECK_BIT(CL_DEVICE_TYPE_ACCELERATOR,accelerator)
			CHECK_BIT(CL_DEVICE_TYPE_DEFAULT,default)
			CHECK_BIT(CL_DEVICE_TYPE_CUSTOM,custom)
			sprintf(MSG_STR,"\t%s:  %s",PS_NAME(psp),str);
			prt_msg(MSG_STR);

			break;

		case PS_QP:
			// cl_command_queue_properties
		case PS_CTYP:
			// cl_device_mem_cache_type
		case PS_LTYP:
			// cl_device_local_mem_type
		case PS_SIZA:
			// size_t[]
		case PS_FP_CONF:
			// cl_device_fp_config (enum)
		case PS_EXCAP:
			// cl_device_exec_capabilities (enum)
			sprintf(MSG_STR,"\t%s:  %s",PS_NAME(psp),
				"unhandled type case");
			prt_msg(MSG_STR);
			break;
		default:
			sprintf(ERROR_STRING,
	"CAUTIOUS:  display_dev_param:  unexpected parameter type (%d)!?",
				PS_TYPE(psp));
			WARN(ERROR_STRING);
			assert(0);
			break;
	}
}

static void display_dev_params(QSP_ARG_DECL  Platform_Device *pdp, OCL_Dev_Param_Spec *tbl, int n )
{
	int i;

	for(i=0;i<n;i++)
		display_dev_param(QSP_ARG  &tbl[i], pdp );
}

static void print_pfdev_info_short(QSP_ARG_DECL  Platform_Device *pdp)
{
	sprintf(MSG_STR,"Device %s (%s platform):",PFDEV_NAME(pdp),PLATFORM_NAME(PFDEV_PLATFORM(pdp)));
	prt_msg(MSG_STR);
	display_dev_params(QSP_ARG  pdp,dev_param_tbl_short,N_DEV_PARAMS_SHORT);
}
#endif // NOT_USED


void shutdown_opencl_platform(void)
{
	//cl_int status;

	/*
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	*/
	NWARN("shutdown_opencl_platform NOT implemented!?");

	// Need to iterate over all devices...
}

#ifdef NOT_USED
/* This utility routine could useful beyond opencl... */

static const char *load_file(QSP_ARG_DECL  const char *pathname, size_t *len)
{
	FILE *fp;
	size_t n_read, siz=0;
	char *buf=NULL;

 	if( ! path_exists(QSP_ARG  pathname) ){
		sprintf(ERROR_STRING,"load_file:  file %s does not exist!?",pathname);
		WARN(ERROR_STRING);
		goto done;
	}

 	siz = file_content_size(QSP_ARG  pathname);
	if( siz == (off_t) -1 ){
		sprintf(ERROR_STRING,"load_file:  couldn't determine size of file %s!?",pathname);
		WARN(ERROR_STRING);
		goto done;
	}

	fp = try_open(QSP_ARG  pathname, "r");
	if (!fp) goto done;

	buf = getbuf(siz);
	n_read = fread( buf, 1, siz, fp );
	if( n_read != siz ){
		sprintf(ERROR_STRING,"load_file %s:  read error, expected %ld bytes, got %ld!?",
			pathname,siz,n_read);
		WARN(ERROR_STRING);
		givbuf(buf);
		siz=n_read;
		buf = NULL;
	}
	fclose( fp );
done:
	*len=siz;
	return buf;
}
#endif // NOT_USED

/* This utility routine could useful beyond opencl... */

// Apparently we have to create kernels on a per-context basis...

cl_program ocl_create_program( const char *buf, Platform_Device *pdp )
{
	cl_program program;	//cl_program is a program executable
	//size_t len;		// NULL len array indicates null-terminated strings
	cl_int status;

	//len = strlen(buf);		// don't count trailing null

	// BUG?  should we check that device is OCL device?
	program = clCreateProgramWithSource(OCLDEV_CTX(pdp), 1,
		(const char **)&buf, /*(const size_t *)&len*/ NULL, &status);

	if( status != CL_SUCCESS ){
		report_ocl_error(DEFAULT_QSP_ARG  status,
					"clCreateProgramWithSource");
		return NULL;
	}
	return program;
}

#define BUF_SIZE	256

static void report_build_info(QSP_ARG_DECL  cl_program prog, Platform_Device *pdp)
{
	cl_int ret;
	char buf[BUF_SIZE];
	char *bufp=buf;
	size_t bytes_returned;
	cl_build_status bs;

	ret = clGetProgramBuildInfo( prog,
				OCLDEV_DEV_ID(pdp),
				CL_PROGRAM_BUILD_STATUS,
				sizeof(bs),
				&bs,
				&bytes_returned
				);

	if( ret != CL_SUCCESS ){
		report_ocl_error(QSP_ARG  ret,"clGetProgramBuildInfo");
	}
	switch(bs){
		case CL_BUILD_NONE:  NADVISE("No build!?"); break;
		case CL_BUILD_ERROR:  NADVISE("Build errors:"); break;
		case CL_BUILD_SUCCESS:  NADVISE("Build success!"); break;
		case CL_BUILD_IN_PROGRESS:  NADVISE("Build in progress..."); break;
		default: NWARN("Unexpected build status!?"); break;
	}

	if( bs == CL_BUILD_ERROR ){
		ret = clGetProgramBuildInfo( prog,
				OCLDEV_DEV_ID(pdp),
				CL_PROGRAM_BUILD_LOG,
				BUF_SIZE,
				bufp,
				&bytes_returned
				);
		if( ret == CL_INVALID_VALUE ){
			// probably insufficient buffer size?
			if( bytes_returned > BUF_SIZE ){
				int n;
				bufp = getbuf(n=(int)bytes_returned);	// BUG?  memory leak?
				ret = clGetProgramBuildInfo( prog,
						OCLDEV_DEV_ID(pdp),
						CL_PROGRAM_BUILD_LOG,
						n,
						bufp,
						&bytes_returned
						);
			}
		}

		if( ret != CL_SUCCESS ){
			report_ocl_error(QSP_ARG  ret,"clGetProgramBuildInfo");
		}

//fprintf(stderr,"%zu log bytes returned, strlen(bufp) = %ld...\n",bytes_returned,
//strlen(bufp));
		//prt_msg(bufp);
		fputs(bufp,stderr);

		if( bufp != buf ) givbuf(bufp);

#ifdef FOOBAR
		fputs(buf,stderr);
		fflush(stderr);
		{
			FILE *fp;
			fp=try_open(QSP_ARG  "msgs.txt","w");
			if( fp != NULL ){
				if( fwrite(buf,1,bytes_returned,fp) != bytes_returned )
					fprintf(stderr,"error writing buffer to file...\n");
				fclose(fp);
			}
		}
#endif // FOOBAR
	}
}

// this routine seems to only be used for the random number generator???

cl_kernel ocl_make_kernel(const char *ksrc,const char *kernel_name,Platform_Device *pdp)
{
	cl_program program;
	cl_kernel kernel;

	program = ocl_create_program(ksrc,pdp);
	if( program == NULL )
		NERROR1("program creation failure!?");

	kernel = ocl_create_kernel(program, kernel_name, pdp);
	if( kernel == NULL ){
		NADVISE("Source code of failed program:");
		NADVISE(ksrc);
		NERROR1("kernel creation failure!?");
	}

	return kernel;
}

cl_kernel ocl_create_kernel(cl_program program,
			const char *name, Platform_Device *pdp )
{
	cl_kernel kernel;
	cl_int status;

	// build (compiles and links) a program executable
	// from the program source or binary
	status = clBuildProgram(program,	// compiled program
				1,		// num_devices
				&OCLDEV_DEV_ID(pdp),	// device list
				NULL,		// options
				NULL,		// notify_func
				NULL		// user_data
				);
	if( status != CL_SUCCESS ){
		report_ocl_error(DEFAULT_QSP_ARG  status,"clBuildProgram");
		report_build_info(DEFAULT_QSP_ARG  program,pdp);
		return NULL;
	}

	//create a kernel object with specified name
	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,"ocl_create_kernel:  creating kernel with name '%s'\n",name);
		NADVISE(DEFAULT_ERROR_STRING);
	}
	kernel = clCreateKernel(program, name, &status);
	if( status != CL_SUCCESS ){
		report_ocl_error(DEFAULT_QSP_ARG  status,"clCreateKernel");
		if( status == CL_INVALID_KERNEL_NAME ){
			sprintf(DEFAULT_ERROR_STRING,"Name:  \"%s\"",name);
			NADVISE(DEFAULT_ERROR_STRING);
		}
		return NULL;
	}

	return kernel;
}

#ifdef NOT_USED
cl_kernel create_kernel_from_file(QSP_ARG_DECL  const char * name, const char *pathname)
{
	const char *buf;
	size_t len;
	cl_program prog;
	cl_kernel kern;

	//INSURE_ODP_RET

	buf = load_file(QSP_ARG  pathname, &len);
	if( buf == NULL ) return NULL;

	//create a program object for a context
	//load the source code specified by the text strings into the program object
	// BUG this uses curr_pdp, but we need different kernels for different devices...
	prog = ocl_create_program(buf,curr_pdp);
	if( prog == NULL ) return NULL;

	// name needs to match a kernel routine name?
	kern = ocl_create_kernel( prog, "foo", curr_pdp );
	if( kern == NULL ) {
		advise("Source code of failed program:");
		advise(buf);
		return NULL;
	}

	return kern;
}
#endif // NOT_USED

#ifdef NOT_USED
static void PF_FUNC_NAME(sync)(SINGLE_QSP_ARG_DECL)
{
	cl_int status;

	assert( curr_pdp != NULL );

	if( OCLDEV_QUEUE(curr_pdp) == NULL ){
		WARN("ocl_sync:  no command queue!?");
		return;
	}

	// clFlush only guarantees that all queued commands to
	// command_queue get issued to the appropriate device
	// There is no guarantee that they will be complete
	// after clFlush returns

	status = clFlush( OCLDEV_QUEUE(curr_pdp) );
	OCL_STATUS_CHECK(status,clFlush)

	// clFinish blocks until all previously queued OpenCL commands
	// in command_queue are issued to the associated device
	// and have completed.

	status = clFinish( OCLDEV_QUEUE(curr_pdp) );
	OCL_STATUS_CHECK(status,clFinish)
}
#endif // NOT_USED

void delete_kernel(QSP_ARG_DECL  Kernel *kp)
{
	//cl_int		ret;

	/*
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	*/
	WARN("delete_kernel:  not implemented!?");
}


#ifdef FOOBAR
int get_max_threads_per_block(Data_Obj *dp)
{
	NWARN("get_max_threads_per_block:  unimplemented, returning 8!?");
	return 8;
}
#endif // FOOBAR

#endif // HAVE_OPENCL

