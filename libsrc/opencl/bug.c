#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <OpenGL/OpenGL.h>		// apple only?
#include <OpenGL/gl.h>			// apple only?

#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

static int verbose=0;
static char error_string[256];	// BUG fixed len string buffer overflow risk!
typedef struct { int x; int y; int z; } dim3 ;
#define N_TEST_COLUMNS	8
#define N_TEST_ROWS	8

static cl_command_queue the_queue;
static cl_context the_context;
static cl_device_id the_dev_id;
static char *the_extensions;

void warn(const char *msg)
{
	fprintf(stderr,"WARNING:  %s\n",msg);
	fflush(stderr);
}

void error1(const char *msg)
{
	fprintf(stderr,"ERROR:  %s\n",msg);
	fflush(stderr);
	exit(1);
}

void advise(const char *msg)
{
	fprintf(stderr,"%s\n",msg);
	fflush(stderr);
}

void shutdown_opencl_platform(void)
{
	//cl_int status;

	/*
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	*/
	warn("shutdown_opencl_platform NOT implemented!?");
}


#define ERROR_CASE(code,string)	case code: msg = string; break;

void report_ocl_error(cl_int status, const char *whence)
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
			sprintf(error_string,
	"report_ocl_error:  Unhandled error code %d (0x%x)!?",status,status);
			warn(error_string);
			break;
	}
	sprintf(error_string,"%s:  %s",whence,msg);
	warn(error_string);
}
char *get_platform_string(cl_platform_id pf_id, int code)
{
	cl_int status;
	size_t ret_size;
	char *str;

	/* First figure out the required size */
	status = clGetPlatformInfo(pf_id,code,
		0,NULL,&ret_size);
	if( status != CL_SUCCESS ){
		report_ocl_error(status, "clGetPlatformInfo");
		return NULL;
		/* BUG make sure to return cleanly... */
	}
	str = malloc(ret_size+1);
	status = clGetPlatformInfo(pf_id,code,
		ret_size+1,str,&ret_size);
	if( status != CL_SUCCESS ){
		report_ocl_error(status, "clGetPlatformInfo");
		return NULL;
		/* BUG make sure to return cleanly... */
	}
	return str;
}

static int extension_supported( const char *ext_str )
{
	char *s;
	s = strstr( the_extensions, ext_str );
	return s==NULL ? 0 : 1;
}


#define OCL_STATUS_CHECK(stat,whence)				\
								\
	if( stat != CL_SUCCESS ){				\
		report_ocl_error(stat, #whence );		\
		return;						\
	}


#define MAX_PARAM_SIZE	128

static void init_ocl_device(cl_device_id dev_id, cl_platform_id pf_id)
{
	cl_int status;
	size_t psize;
	char *this_name;
	char *extensions;
	CGLContextObj cgl_ctx=NULL;
	cl_context_properties props[3]={
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
		0,	// need to put cgl_ctx here
		0
		};

	// Find out space required
	status = clGetDeviceInfo(dev_id,CL_DEVICE_NAME,0,
		NULL,&psize);
	OCL_STATUS_CHECK(status,clGetDeviceInfo)
	this_name=malloc(psize+1);
	status = clGetDeviceInfo(dev_id,CL_DEVICE_NAME,psize+1, this_name,&psize);
	OCL_STATUS_CHECK(status,clGetDeviceInfo)

	// Check for other properties...
	// find out how much space required...
	status = clGetDeviceInfo(dev_id,CL_DEVICE_EXTENSIONS,0,NULL,&psize);
	OCL_STATUS_CHECK(status,clGetDeviceInfo)
	extensions = malloc(psize+1);
	status = clGetDeviceInfo(dev_id,CL_DEVICE_EXTENSIONS,psize+1,extensions,&psize);
	OCL_STATUS_CHECK(status,clGetDeviceInfo)

	// On the new MacBook Pro, with two devices, the Iris_Pro
	// throws an error at clCreateCommandQueue *iff* we set
	// the share group property here...  Presumably because
	// that device doesn't handle the display?
	// We insert a hack below by excluding that device name,
	// but maybe there is another model where that would be
	// inappropriate?

	if( extension_supported("cl_APPLE_gl_sharing") &&
			strcmp(this_name,"Iris_Pro")){

		CGLShareGroupObj share_group;
	
		the_extensions=extensions;
		the_dev_id = dev_id;
		cgl_ctx = CGLGetCurrentContext();
		if( cgl_ctx != NULL){
			// This means that we have an OpenGL window available...
			share_group = CGLGetShareGroup(cgl_ctx);
			if( share_group != NULL )
				props[1] = (cl_context_properties) share_group;
			else
				error1("CAUTIOUS:  init_ocl_device:  CGL context found, but null share group!?");
		} else {
			advise("OpenCL initialized without an OpenGL context.");
		}
	}


	// Check for OpenGL capabilities
	//opengl_check(pdp);
#ifdef TAKEN_FROM_DEMO_PROG
#if (USE_GL_ATTACHMENTS)

    printf(SEPARATOR);
    printf("Using active OpenGL context...\n");

    CGLContextObj kCGLContext = CGLGetCurrentContext();              
    CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
    
    cl_context_properties properties[] = { 
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, 
        (cl_context_properties)kCGLShareGroup, 0 
    };
        
    // Create a context from a CGL share group
    //
    ComputeContext = clCreateContext(properties, 0, 0, clLogMessagesToStdoutAPPLE, 0, 0);
	if(!ComputeContext)
		return -2;

#else	// ! USE_GL_ATTACHMENTS	

    // Connect to a compute device
    //
    err = clGetDeviceIDs(NULL, ComputeDeviceType, 1, &ComputeDeviceId, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to locate compute device!\n");
        return EXIT_FAILURE;
    }
  
    // Create a compute context 
    //
    ComputeContext = clCreateContext(0, 1, &ComputeDeviceId, clLogMessagesToStdoutAPPLE, NULL, &err);
    if (!ComputeContext)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
#endif	// ! USE_GL_ATTACHMENTS	
#endif // TAKEN_FROM_DEMO_PROG

	//create context on the specified device
//if( cgl_ctx != NULL )
//fprintf(stderr,"creating clContext with share properties for %s...\n",PFDEV_NAME(pdp));
	if( cgl_ctx == NULL ){
		the_context = clCreateContext(
			NULL,		// cl_context_properties *properties
			1,		// num_devices
			&dev_id,	// devices
			NULL,		// void *pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data )
			NULL,		// void *user_data
			&status		// cl_int *errcode_ret
		);
	} else {
		the_context = clCreateContext(
			props,		// cl_context_properties *properties
			0,		// num_devices
			NULL,	// devices
			clLogMessagesToStdoutAPPLE,	// void *pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data )
			NULL,		// void *user_data
			&status		// cl_int *errcode_ret
		);
	}
	if( status != CL_SUCCESS ){
		report_ocl_error(status, "clCreateContext");
		//return;
	}
	// BUG check return value for error

	//create the command_queue (stream)
//fprintf(stderr,"clContext = 0x%lx...\n",(long)context);
//fprintf(stderr,"init_ocl_device:  dev_id = 0x%lx\n",(long)dev_id);

	// At least once we have gotten an invalid value error here,
	// after receiving the advisory "OpenCL initialized without an OpenGL context
	// (which may or may not be relevant).  This behavior was not repeatable,
	// perhaps because of different stack contents???

	// The third arg is a properties bit field, with valid values being:
	// CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
	// CL_QUEUE_PROFILING_ENABLE
	the_queue = clCreateCommandQueue(the_context, dev_id, 0, &status);
	if( status != CL_SUCCESS ){
		report_ocl_error(status, "clCreateCommandQueue");
		error1("failed to create command queue");
	}
	// set a ready flag?

	//init_ocl_dev_memory();
}

#define MAX_OPENCL_DEVICES	4

static void init_ocl_devices(cl_platform_id pf_id)
{
	cl_int	status;
	cl_uint		n_devs;
	int i;
	cl_device_id dev_tbl[MAX_OPENCL_DEVICES];

	//get the device info
	status = clGetDeviceIDs( pf_id, CL_DEVICE_TYPE_DEFAULT,
		MAX_OPENCL_DEVICES, dev_tbl, &n_devs);

	if( status != CL_SUCCESS ){
		report_ocl_error(status, "clGetDeviceIDs");
		return;
		// BUG make sure to return cleanly...
	}
//fprintf(stderr,"init_ocl_devices:  %d device%s found\n",n_devs,n_devs==1?"":"s");

	if( verbose ){
		sprintf(error_string,"%d OpenCL device%s found...",n_devs,
			n_devs==1?"":"s");
		advise(error_string);
	}

	for(i=0;i<n_devs;i++){
		init_ocl_device(dev_tbl[i],pf_id);
	}
} // end init_ocl_devices

static void init_ocl_platform(cl_platform_id pf_id)
{
	//cl_int status;
	//char param_data[MAX_PARAM_SIZE];
	//char *str;
	//size_t ret_size;

	/*
	str=get_platform_string(platform_id,CL_PLATFORM_NAME)

	cpp = creat_platform(platform_str, PLATFORM_OPENCL);
	givbuf(platform_str);

	str=get_platform_string(platform_id,CL_PLATFORM_PROFILE)
	if( str == NULL ) return;
	SET_OCLPF_PROFILE(cpp,platform_str);

	str=get_platform_string(platform_id,CL_PLATFORM_VERSION)
	if( str == NULL ) return;
	SET_OCLPF_VERSION(cpp,platform_str);

	str=get_platform_string(platform_id,CL_PLATFORM_VENDOR)
	if( str == NULL ) return;
	SET_OCLPF_VENDOR(cpp,platform_str);
	*/

	the_extensions=get_platform_string(pf_id,CL_PLATFORM_EXTENSIONS);
	if( the_extensions == NULL ) return;

	/*
	SET_OCLPF_EXTENSIONS(cpp,platform_str);
	SET_PF_OPD_ID(cpp,platform_id);
	SET_PLATFORM_FUNCTIONS(cpp,ocl)
	SET_PF_FUNC_TBL(cpp,ocl_vfa_tbl);
	*/

	// BUG need to set vfa_tbl here too!

	//icp = create_item_context(pfdev_itp, PLATFORM_NAME(cpp) );
	//push_item_context(pfdev_itp, icp );
	//push_pfdev_context(PF_CONTEXT(cpp) );
	init_ocl_devices(pf_id);
	//if( pop_pfdev_context(SINGLE_) == NO_ITEM_CONTEXT )
	//	ERROR1("init_ocl_platform:  Failed to pop platform device context!?");
}

//In general Intel CPU and NV/AMD's GPU are in different platforms
//But in Mac OSX, all the OpenCL devices are in the platform "Apple"

#define MAX_CL_PLATFORMS	3

static int init_ocl_platforms(void)
{
	cl_platform_id	platform_ids[MAX_CL_PLATFORMS];
	cl_uint		num_platforms;
	cl_int		ret;
	int		i;

	// BUG need to add error checking on the return values...

	ret = clGetPlatformIDs(MAX_CL_PLATFORMS, platform_ids, &num_platforms);
//fprintf(stderr,"init_ocl_platform:  %d platform%s found\n",num_platforms,num_platforms==1?"":"s");

	for(i=0;i<num_platforms;i++)
		init_ocl_platform(platform_ids[i]);

	return num_platforms;
}

// Apparently we have to create kernels on a per-context basis...

cl_program ocl_create_program( const char *buf, cl_context context )
{
	cl_program program;	//cl_program is a program executable
	size_t len;
	cl_int status;

	len = strlen(buf);		// count trailing null?
	program = clCreateProgramWithSource(context, 1,
		(const char **)&buf, (const size_t *)&len, &status);

	if( status != CL_SUCCESS ){
		report_ocl_error(status, "clCreateProgramWithSource");
		return NULL;
	}
	return program;
}


cl_kernel ocl_create_kernel(cl_program program, const char *name, cl_device_id dev_id )
{
	cl_kernel kernel;
	cl_int status;

	// build (compiles and links) a program executable
	// from the program source or binary
	status = clBuildProgram(program,	// compiled program
				1,		// num_devices
				&dev_id,	// device list
				NULL,		// options
				NULL,		// notify_func
				NULL		// user_data
				);
	if( status != CL_SUCCESS ){
		report_ocl_error(status,"clBuildProgram");
		//report_build_info(program,pdp);
		return NULL;
	}

	//create a kernel object with specified name
	if( verbose ){
		sprintf(error_string,"ocl_create_kernel:  creating kernel with name '%s'\n",name);
		advise(error_string);
	}
	kernel = clCreateKernel(program, name, &status);
	if( status != CL_SUCCESS ){
		report_ocl_error(status,"clCreateKernel");
		return NULL;
	}

	return kernel;
}



char		kernel_source_ocl_slow_sp_vramp2d[] =

"typedef struct { int x; int y; int z; } dim3 ;			"

"__kernel void g_ocl_slow_sp_vramp2d( __global float* a ,	"
"					int a_offset ,		"
"					dim3 inc1 ,		"
"					float scalar1_val,	"
"					float scalar2_val,	"
"					float scalar3_val)	"
"{								"
"	dim3 index1;						"
"	index1.x = get_global_id(0);				"
"	index1.y = get_global_id(1);				"
"	index1.z = get_global_id(2);				"
"	index1.x *= inc1.x;					"
"	index1.y *= inc1.y;					"
"	index1.z *= inc1.z;					"
"	a[index1.x+index1.y + a_offset ] =			"
"		scalar1_val +					"
"		scalar2_val * (index1.x / inc1.x ) +		"
"		scalar3_val * (index1.y / inc1.y );		"
"}								"
;

static void	h_ocl_slow_sp_vramp2d(cl_mem dest_ptr,cl_device_id pfdev, cl_context ctx, cl_command_queue queue)
{
	static cl_program program = ((void *)0);
	static cl_kernel kernel = ((void *)0);
	cl_int		status;
	cl_event	event;
	int		ki_idx = 0;
	int 		offset=0;
	const char     *ksrc;
	size_t		global_work_size[3] = {1, 1, 1};
	dim3		dst_xyz_incr;
	float		base=1.0;
	float		x_delta=1.0;
	float		y_delta=10.0;
	dst_xyz_incr.x = 1;
	dst_xyz_incr.y = N_TEST_COLUMNS;
	dst_xyz_incr.z = N_TEST_COLUMNS * N_TEST_ROWS;
	if (kernel == ((void *)0)) {
		ksrc = kernel_source_ocl_slow_sp_vramp2d;
		program = ocl_create_program(ksrc, ctx);
		if (program == ((void *)0))
			error1("program creation failure!?");
		kernel = ocl_create_kernel(program, "g_ocl_slow_sp_vramp2d", pfdev);
		if (kernel == ((void *)0)) {
			advise("Source code of failed program:");
			advise(ksrc);
			error1("kernel creation failure!?");
		}
	}
	status = clSetKernelArg(kernel, ki_idx++, sizeof(void *), & dest_ptr );
	if (status != 0)
		report_ocl_error(status, "clSetKernelArg");
	status = clSetKernelArg(kernel, ki_idx++, sizeof(int), & offset );
	if (status != 0)
		report_ocl_error(status, "clSetKernelArg");
	status = clSetKernelArg(kernel, ki_idx++, sizeof(dim3), &dst_xyz_incr);
	if (status != 0)
		report_ocl_error(status, "clSetKernelArg");
	status = clSetKernelArg(kernel, ki_idx++, sizeof(float), &base );
	if (status != 0)
		report_ocl_error(status, "clSetKernelArg");
	status = clSetKernelArg(kernel, ki_idx++, sizeof(float), &x_delta);
	if (status != 0)
		report_ocl_error(status, "clSetKernelArg");
	status = clSetKernelArg(kernel, ki_idx++, sizeof(float), &y_delta);
	if (status != 0)
		report_ocl_error(status, "clSetKernelArg");
	global_work_size[0] = N_TEST_COLUMNS;
	global_work_size[1] = N_TEST_ROWS;
	global_work_size[2] = 1;
	status = clEnqueueNDRangeKernel(queue, kernel, 3, ((void *)0), global_work_size, ((void *)0),
										0, ((void *)0), &event);
	if (status != 0)
		report_ocl_error(status, "clEnqueueNDRangeKernel");
	clWaitForEvents(1, &event);
}

int	main(int ac, char **av)
{
	cl_mem dest;
	float *h_copy, *ptr;
	int i,j;
	cl_int status;
	size_t siz= sizeof(float)*N_TEST_COLUMNS*N_TEST_ROWS ;

	// initialize
	if( init_ocl_platforms() < 0 )
		error1("No OpenCL platforms found!?");

	// allocate memory
	dest = clCreateBuffer(the_context, CL_MEM_READ_WRITE, siz, NULL, &status);
	if( status != CL_SUCCESS ){
		report_ocl_error(status,"clCreateBuffer");
		error1("Couldn't allocate memory!?");
	}
	h_copy = malloc( siz );

	// compute
	h_ocl_slow_sp_vramp2d(dest,the_dev_id,the_context,the_queue);

	// transfer data
	status = clEnqueueReadBuffer( the_queue,
			dest,		// cl_mem
			CL_TRUE,	// blocking_read
			0,
			siz,
			h_copy,
			0,
			NULL,
			NULL);

	// display result
	ptr = h_copy;
	for(i=0;i<N_TEST_ROWS;i++){
		for(j=0;j<N_TEST_COLUMNS;j++)
			printf("\t%g",*ptr++);
		printf("\n");
	}

	// free memory

}


