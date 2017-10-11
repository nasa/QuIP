
/* jbm's interface to opencl devices */

/* This file contains the menu-callable functions, which in turn call
 * host functions which are typed and take an oap argument.
 * These host functions then call the gpu kernel functions...
 */

#include "quip_config.h"

#ifdef HAVE_STRING_H
#include <string.h>
#endif // HAVE_STRING_H

#ifdef HAVE_OPENCL

#define BUILD_FOR_OPENCL

#include "quip_prot.h"
#include "my_ocl.h"
#include "ocl_platform.h"
#include "platform.h"
#include "veclib/ocl_veclib_prot.h"
#include "veclib_api.h"

//#include "veclib/ocl_veclib_prot.h"

// When we build in Xcode, we need to precede these with OpenGL:
#include <OpenGL/OpenGL.h>		// apple only?
#include <OpenGL/gl.h>			// apple only?
#include "gl_info.h"

#include "../opengl/glx_supp.h"

// global var
int max_threads_per_block;	// BUG should we have this?

static const char *default_ocl_dev_name=NULL;
static const char *first_ocl_dev_name=NULL;
static int default_ocl_dev_found=0;

// We get a crash if we specify an openGL window
// *after* we initialize OpenCL, so if we want to render
// to a window we must specify a window first.  But for compute
// only programs, it is not necessarily an error not to have
// a window.  So if we initialize OpenCL before a window has
// been specified, we prohibit later associations with windows.
// There ought to be a better way!?

static int opengl_prohibited=0;

static void prohibit_opengl(void)
{
	opengl_prohibited=1;
}


// Where does this comment go?

/* On the host 1L<<33 gets us bit 33 - but 1<<33 does not,
 * because, by default, ints are 32 bits.  We don't know
 * how nvcc treats 1L...  but we can try it...
 */

// make these C so we can link from other C files...

// We treat the device as a server, so "upload" transfers from host to device

static void ocl_mem_upload(QSP_ARG_DECL  void *dst, void *src, size_t siz, index_t offset, Platform_Device *pdp )
{
	cl_int status;

	// BUG need to check here
	//INSURE_ODP

	// copy the memory from host to device
	// CL_TRUE means blocking write/read

	if( curr_pdp == NULL ) return;

	status = clEnqueueWriteBuffer(OCLDEV_QUEUE(pdp),
			dst,		// device mem address (cl_mem)
			CL_TRUE,	// blocking write
			offset,		// offset
			siz,
			src,		// host mem address
			0,		// num events in wait list
			NULL,		// event wait list
			NULL);		// event - id for use if asynchronous
 
	if( status != CL_SUCCESS ){
		report_ocl_error(QSP_ARG  status, "clEnqueueWriteBuffer");
	}
}

static void ocl_mem_dnload(QSP_ARG_DECL  void *dst, void *src, size_t siz, index_t offset, Platform_Device *pdp )
{
	cl_int status;

	//INSURE_ODP

	if( curr_pdp == NULL ) return;

	//copy memory from device to host

//fprintf(stderr,"ocl_mem_dnload:  device = %s, src = 0x%lx, siz = %ld, dst = 0x%lx\n",
//PFDEV_NAME(pdp),(long)src,siz,(long)dst);
	status = clEnqueueReadBuffer( OCLDEV_QUEUE(pdp),
			src,		// cl_mem
			CL_TRUE,	// blocking_read
			/* 0 */ offset,	// offset
			siz,
			dst,
			0,		// num_events_in_wait_list
			NULL,		// event_wait_list
			NULL);		// event
 
	if( status != CL_SUCCESS ){
		report_ocl_error(QSP_ARG  status, "clEnqueueReadBuffer");
	}
}

#define MAX_OCL_DEVICES 5
#define MAX_DIGIT_CHARS	12	// overkill

static const char * available_ocl_device_name(QSP_ARG_DECL  const char *name,char *scratch_string, int scratch_len)
{
	Platform_Device *pdp;
	const char *s;
	int n=1;

	s=name;
	// Why should we care how many devices there are?
	// Why have statically-allocated structures?
	while(n<=MAX_OCL_DEVICES){
		pdp = pfdev_of(QSP_ARG  s);
		if( pdp == NULL ) return(s);

		// This name is in use
		n++;

		if( strlen(name)+1+MAX_DIGIT_CHARS+1 > scratch_len )
			ERROR1("available_ocl_device_name:  size of scratch_string is insufficient!?");

		sprintf(scratch_string,"%s_%d",name,n);
		s=scratch_string;
	}
	sprintf(ERROR_STRING,"Number of %s OpenCL devices exceed configured maximum %d!?",
		name,MAX_OCL_DEVICES);
	WARN(ERROR_STRING);
	ERROR1(ERROR_STRING);
	return(NULL);	// NOTREACHED - quiet compiler
}

#define MAX_AREA_NAME_LEN	80

static void init_ocl_dev_memory(QSP_ARG_DECL  Platform_Device *pdp)
{
	char area_name[MAX_AREA_NAME_LEN+1];
	Data_Area *ap;

	//strcpy(area_name,PFDEV_NAME(pdp));
	// make sure names will fit - longest name is %s.%s_host_mapped
	if( strlen(PLATFORM_NAME(PFDEV_PLATFORM(pdp)))+strlen(PFDEV_NAME(pdp))+strlen("._host_mapped") > MAX_AREA_NAME_LEN )
		ERROR1("init_ocl_dev_memory:  area name too large for buffer, increase MAX_AREA_NAME_LEN!?");

	sprintf(area_name,"%s.%s",
		PLATFORM_NAME(PFDEV_PLATFORM(pdp)),PFDEV_NAME(pdp));

	// what should the name for the memory area be???

	// address set to NULL says use custom allocator - see dobj/makedobj.c

	ap = pf_area_init(QSP_ARG  area_name,NULL,0, MAX_OCL_GLOBAL_OBJECTS,DA_OCL_GLOBAL,pdp);
	if( ap == NULL ){
		sprintf(ERROR_STRING,
	"init_ocl_dev_memory:  error creating global data area %s",area_name);
		WARN(ERROR_STRING);
	}
	// g++ won't take this line!?
	SET_AREA_PFDEV(ap,pdp);

	// BUG should be per-device, not global table...
	pdp->pd_ap[PF_GLOBAL_AREA_INDEX] = ap;

	/* We used to declare a heap for constant memory here,
	 * but there wasn't much of a point because:
	 * Constant memory can't be allocated, rather it is declared
	 * in the .cu code, and placed by the compiler as it sees fit.
	 * To have objects use this, we would have to declare a heap and
	 * manage it ourselves...
	 * There's only 64k, so we should be sparing...
	 * We'll try this later...
	 */


	/* Make up another area for the host memory
	 * which is locked and mappable to the device.
	 * We don't allocate a pool here, but do it as needed...
	 */

	//strcat(cname,"_host");
	sprintf(area_name,"%s.%s_host",
		PLATFORM_NAME(PFDEV_PLATFORM(pdp)),PFDEV_NAME(pdp));

	ap = pf_area_init(QSP_ARG  area_name,(u_char *)NULL,0,MAX_OCL_MAPPED_OBJECTS,
							DA_OCL_HOST,pdp);
	if( ap == NULL ){
		sprintf(ERROR_STRING,
	"init_ocl_dev_memory:  error creating host data area %s",area_name);
		ERROR1(ERROR_STRING);
	}
	SET_AREA_PFDEV(ap, pdp);
	pdp->pd_ap[PF_HOST_AREA_INDEX] = ap;

	/* Make up another psuedo-area for the mapped host memory;
	 * This is the same memory as above, but mapped to the device.
	 * In the current implementation, we create objects in the host
	 * area, and then automatically create an alias on the device side.
	 * There is a BUG in that by having this psuedo area in the data
	 * area name space, a user could select it as the data area and
	 * then try to create an object.  We will detect this in make_dobj,
	 * and complain.
	 */

	//strcpy(cname,dname);
	//strcat(cname,"_host_mapped");
	sprintf(area_name,"%s.%s_host_mapped",
		PLATFORM_NAME(PFDEV_PLATFORM(pdp)),PFDEV_NAME(pdp));

	ap = pf_area_init(QSP_ARG  area_name,(u_char *)NULL,0,MAX_OCL_MAPPED_OBJECTS,
						DA_OCL_HOST_MAPPED,pdp);
	if( ap == NULL ){
		sprintf(ERROR_STRING,
	"init_ocl_dev_memory:  error creating host-mapped data area %s",area_name);
		ERROR1(ERROR_STRING);
	}
	SET_AREA_PFDEV(ap,pdp);
	pdp->pd_ap[PF_HOST_MAPPED_AREA_INDEX] = ap;

	if( verbose ){
		sprintf(ERROR_STRING,"init_ocl_dev_memory DONE");
		advise(ERROR_STRING);
	}
}

#define EXTENSIONS_PREFIX	"Extensions:  "

static void ocl_dev_info(QSP_ARG_DECL  Platform_Device *pdp)
{
	sprintf(MSG_STR,"%s:",PFDEV_NAME(pdp));
	prt_msg(MSG_STR);
	prt_msg("Sorry, no OpenCL-specific device info yet.");
}

static void ocl_info(QSP_ARG_DECL  Compute_Platform *cpp)
{
	int s;

	sprintf(MSG_STR,"Vendor:  %s",OCLPF_VENDOR(cpp));
	prt_msg(MSG_STR);
	sprintf(MSG_STR,"Version:  %s",OCLPF_VERSION(cpp));
	prt_msg(MSG_STR);
	sprintf(MSG_STR,"Profile:  %s",OCLPF_PROFILE(cpp));
	prt_msg(MSG_STR);

	// The extensions can be long...
	s = (int) strlen(OCLPF_EXTENSIONS(cpp))+strlen(EXTENSIONS_PREFIX)+2;
	if( s > sb_size(QS_SCRATCH) )
		enlarge_buffer( QS_SCRATCH, s );
	sprintf(sb_buffer(QS_SCRATCH),"%s%s\n",EXTENSIONS_PREFIX,OCLPF_EXTENSIONS(cpp));
	prt_msg(sb_buffer(QS_SCRATCH));
}

static int extension_supported( Platform_Device *pdp, const char *ext_str )
{
	char *s;
	s = strstr( OCLDEV_EXTENSIONS(pdp), ext_str );
	return s==NULL ? 0 : 1;
}

static char *get_ocl_device_name(QSP_ARG_DECL  cl_device_id dev_id)
{
	size_t psize;
	char *name;
	cl_int status;

	// Find out space required
	status = clGetDeviceInfo(dev_id,CL_DEVICE_NAME,0, NULL,&psize);
	OCL_STATUS_CHECK_WITH_RETURN(status,clGetDeviceInfo,NULL)
	name=getbuf(psize+1);
	status = clGetDeviceInfo(dev_id,CL_DEVICE_NAME,psize+1, name,&psize);
	OCL_STATUS_CHECK_WITH_RETURN(status,clGetDeviceInfo,NULL)
	return name;
}

static void replace_spaces(char *s, int c)
{
	/* change spaces to underscores */
	while(*s){
		if( *s==' ' ) *s=c;
		s++;
	}
}

static Platform_Device * create_ocl_device(QSP_ARG_DECL  cl_device_id dev_id, Compute_Platform *cpp)
{
	char *name;
	//size_t psize;
	const char *name_p;
#define SCRATCH_LEN	128
	char scratch[SCRATCH_LEN];
	Platform_Device *pdp;

	name = get_ocl_device_name(QSP_ARG  dev_id);
	if( name == NULL ) return NULL;
	replace_spaces(name,'_');

	/* We might have two of the same devices installed in a single system.
	 * In this case, we can't use the device name twice, because there will
	 * be a conflict.  The first one gets the name, then we have to check and
	 * make sure that the name is not in use already.  If it is, then we append
	 * a number to the string...
	 */
	name_p = available_ocl_device_name(QSP_ARG  name,scratch,SCRATCH_LEN);	// use cname as scratch string
	pdp = new_pfdev(QSP_ARG  name_p);

	givbuf(name);

	// initialize all the fields?

	assert( pdp != NULL );

	if( pdp != NULL ){
		SET_PFDEV_PLATFORM(pdp,cpp);
		// allocate the memory for the platform-specific data
		SET_PFDEV_ODI(pdp,getbuf(sizeof(*PFDEV_ODI(pdp))));
		SET_PFDEV_OCL_DEV_ID(pdp,dev_id);
	}

	return pdp;
}

static void get_extensions(QSP_ARG_DECL  Platform_Device *pdp)
{
	cl_int status;
	size_t psize;
	char *extensions;

	// Check for other properties...
	// find out how much space required...
	status = clGetDeviceInfo(PFDEV_OCL_DEV_ID(pdp),CL_DEVICE_EXTENSIONS,0,NULL,&psize);
	OCL_STATUS_CHECK(status,clGetDeviceInfo)
	extensions = getbuf(psize+1);
	status = clGetDeviceInfo(PFDEV_OCL_DEV_ID(pdp),CL_DEVICE_EXTENSIONS,psize+1,extensions,&psize);
	OCL_STATUS_CHECK(status,clGetDeviceInfo)
	// change spaces to newlines for easier reading...
	replace_spaces(extensions,'\n');
	SET_OCLDEV_EXTENSIONS(pdp,extensions);
}


#define MAX_PARAM_SIZE	128


static void init_ocl_device(QSP_ARG_DECL  cl_device_id dev_id,
							Compute_Platform *cpp)
{
	cl_int status;
	//long param_data[MAX_PARAM_SIZE/sizeof(long)];	// force alignment
	//char name[LLEN];
	static int n_ocl_devs=0;
	Platform_Device *pdp;
	CGLContextObj cgl_ctx=NULL;
	cl_context context;
	cl_command_queue command_queue; //"stream" in CUDA
	cl_context_properties props[3]={
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
		0,	// need to put cgl_ctx here
		0
		};

	pdp = create_ocl_device(QSP_ARG  dev_id, cpp);
	if( pdp == NULL ) return;

	/* Remember this name in case the default is not found */
	if( first_ocl_dev_name == NULL )
		first_ocl_dev_name = PFDEV_NAME(pdp);

	/* Compare this name against the default name set in
	 * the environment, if it exists...
	 */
	if( default_ocl_dev_name != NULL && ! default_ocl_dev_found ){
		if( !strcmp(PFDEV_NAME(pdp),default_ocl_dev_name) )
			default_ocl_dev_found=1;
	}

	get_extensions(QSP_ARG  pdp);
	SET_OCLDEV_DEV_ID(pdp,dev_id);
	SET_PFDEV_PLATFORM(pdp,cpp);
	if( n_ocl_devs >= MAX_OPENCL_DEVICES ){
		sprintf(ERROR_STRING,"More than %d OpenCL devices found;"
			"need to increase MAX_OPENCL_DEVICES and recompile",
			MAX_OPENCL_DEVICES);
		ERROR1(ERROR_STRING);
	}
	SET_OCLDEV_IDX(pdp,n_ocl_devs++);

	SET_PFDEV_MAX_DIMS(pdp,DEFAULT_PFDEV_MAX_DIMS);

	// On the new MacBook Pro, with two devices, the Iris_Pro
	// throws an error at clCreateCommandQueue *iff* we set
	// the share group property here...  Presumably because
	// that device doesn't handle the display?
	// We insert a hack below by excluding that device name,
	// but maybe there is another model where that would be
	// inappropriate?

	if( extension_supported(pdp,"cl_APPLE_gl_sharing") &&
			strcmp(PFDEV_NAME(pdp),"Iris_Pro")){

		CGLShareGroupObj share_group;
	
		cgl_ctx = CGLGetCurrentContext();
		if( cgl_ctx != NULL){
			// This means that we have an OpenGL window available...
			share_group = CGLGetShareGroup(cgl_ctx);
			assert( share_group != NULL );
			props[1] = (cl_context_properties) share_group;
		} else {
			// If we let this go, it sometimes causes a seg fault
			// when we try to set the GL window afterwards!?
			//
			// But it should not be an error, because we don't know
			// for sure that we will ever attempt it.
			// We need to set a flag to prohibit it later...
			advise("init_ocl_device:  OpenCL initialized without an OpenGL context;");
			advise("init_ocl_device:  Prohibiting OpenGL operations.");
			prohibit_opengl();
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
		context = clCreateContext(
			NULL,		// cl_context_properties *properties
			1,		// num_devices
			&dev_id,	// devices
			NULL,		// void *pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data )
			NULL,		// void *user_data
			&status		// cl_int *errcode_ret
		);
	} else {
		context = clCreateContext(
			props,		// cl_context_properties *properties
			0,		// num_devices
			NULL,	// devices
			clLogMessagesToStdoutAPPLE,	// void *pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data )
			NULL,		// void *user_data
			&status		// cl_int *errcode_ret
		);
	}
	if( status != CL_SUCCESS ){
		report_ocl_error(QSP_ARG  status, "clCreateContext");
		SET_OCLDEV_CTX(pdp,NULL);
		//return;
	}
	// BUG check return value for error

	SET_OCLDEV_CTX(pdp,context);

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
	command_queue = clCreateCommandQueue(context, dev_id, 0, &status);
	if( status != CL_SUCCESS ){
		report_ocl_error(QSP_ARG  status, "clCreateCommandQueue");
		SET_OCLDEV_QUEUE(pdp,NULL);
		//return;
	} else {
		SET_OCLDEV_QUEUE(pdp,command_queue);
	}
	// set a ready flag?

	init_ocl_dev_memory(QSP_ARG  pdp);

	curr_pdp = pdp;
}

static void init_ocl_devices(QSP_ARG_DECL  Compute_Platform *cpp )
{
	cl_int	status;
	cl_uint		n_devs;
	int i;
	cl_device_id dev_tbl[MAX_OPENCL_DEVICES];

	if( cpp == NULL ) return;	// print warning?

	//get the device info
	status = clGetDeviceIDs( OCLPF_ID(cpp), CL_DEVICE_TYPE_DEFAULT,
		MAX_OPENCL_DEVICES, dev_tbl, &n_devs);

	if( status != CL_SUCCESS ){
		report_ocl_error(QSP_ARG  status, "clGetDeviceIDs");
		return;
		// BUG make sure to return cleanly...
	}
//fprintf(stderr,"init_ocl_devices:  %d device%s found\n",n_devs,n_devs==1?"":"s");

	if( verbose ){
		sprintf(ERROR_STRING,"%d OpenCL device%s found...",n_devs,
			n_devs==1?"":"s");
		advise(ERROR_STRING);
	}

	//default_ocl_dev_name = getenv(DEFAULT_OCL_DEV_VAR);
	/* may be null */

	for(i=0;i<n_devs;i++){
		init_ocl_device(QSP_ARG  dev_tbl[i],cpp);
	}

	//SET_PF_DISPATCH_FUNC( cpp, ocl_dispatch );
} // end init_ocl_devices

static void *ocl_mem_alloc(QSP_ARG_DECL  Platform_Device *pdp, dimension_t size, int align )
{
	cl_int status;
	void *ptr;

	// clCreateBuffer returns cl_mem...
	ptr = clCreateBuffer(OCLDEV_CTX(pdp), CL_MEM_READ_WRITE, size, NULL, &status);

	if( status != CL_SUCCESS ){
		report_ocl_error(QSP_ARG  status,"clCreateBuffer");
		sprintf(ERROR_STRING,"ocl_mem_alloc:  Attempting to allocate %d bytes.",size);
		advise(ERROR_STRING);
		return NULL;
	}
//fprintf(stderr,"ocl_mem_alloc %d:  returning 0x%lx\n",size,(long)ptr);
	return ptr;
}

static int ocl_obj_alloc(QSP_ARG_DECL  Data_Obj *dp, dimension_t size, int align )
{
	OBJ_DATA_PTR(dp) = ocl_mem_alloc(QSP_ARG  OBJ_PFDEV(dp), size, align );
	if( OBJ_DATA_PTR(dp) == NULL ){
		sprintf(ERROR_STRING,"ocl_obj_alloc:  error allocating memory for object %s!?",OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return -1;
	}
	return 0;
}


static void ocl_mem_free(QSP_ARG_DECL  void *ptr)
{
	cl_int		ret;

	ret = clReleaseMemObject( (cl_mem) ptr ); //free memory on device
	// BUG check return value
}

static void ocl_obj_free(QSP_ARG_DECL  Data_Obj *dp)
{
	cl_int		ret;

	ret = clReleaseMemObject( (cl_mem) OBJ_DATA_PTR(dp) ); //free memory on device
	// BUG check return value
}

//void *TMPVEC_NAME(Platform_Device *pdp, size_t size,size_t len,const char *whence)
void *ocl_tmp_vec(Platform_Device *pdp, size_t size,size_t len,const char *whence)
{
	void *ptr;

	ptr = ocl_mem_alloc(DEFAULT_QSP_ARG  pdp, len*size, 0 /* alignment arg not used? */ );
	// Nice to zero it for testing???
	return ptr;
}

//void FREETMP_NAME(void *ptr,const char *whence)
void ocl_free_tmp(void *ptr,const char *whence)
{
	ocl_mem_free(DEFAULT_QSP_ARG  ptr);
}

static void ocl_update_offset(QSP_ARG_DECL  Data_Obj *dp )
{
	ERROR1("ocl_update_offset not implemented!?");
}

#ifdef USE_OPENCL_SUBREGION
static cl_mem find_parent_buf(QSP_ARG_DECL  Data_Obj *dp, int *offset_p )
{
	int offset=0;

	while( ! OWNS_DATA(dp) ){
//fprintf(stderr,"%s does not own its data...\n",OBJ_NAME(dp));
		offset += OBJ_OFFSET(dp);	// Do we need to multiply?
//fprintf(stderr,"offset = %d\n",offset);
		dp = OBJ_PARENT(dp);
	}
//fprintf(stderr,"returning offset = %d\n",offset);
	*offset_p = offset;
//fprintf(stderr,"returning %s data ptr at 0x%lx\n",OBJ_NAME(dp),(u_long)OBJ_DATA_PTR(dp));
	return OBJ_DATA_PTR(dp);
}
#endif // USE_OPENCL_SUBREGION

/*
 * BUG - if we create a subregion for the offset area, then
 * things fail if we have multiple overlapping subregions!?
 * Better solution to keep the offset relative to the parent
 * buffer.
 */

static void ocl_offset_data(QSP_ARG_DECL  Data_Obj *dp, index_t offset)
{
#ifndef USE_OPENCL_SUBREGION
	/* The original code used subBuffers, but overlapping subregions
	 * don't work...
	 * So instead we use a common memory buffer, but keep track
	 * of the starting offset (in elements).  This offset has
	 * to be passed to the kernels.
	 */

//fprintf(stderr,"ocl_offset_data:  obj %s, offset = %d\n",OBJ_NAME(dp),offset);
//fprintf(stderr,"\tparent obj %s, parent offset = %d\n",OBJ_NAME(OBJ_PARENT(dp)),
//OBJ_OFFSET(OBJ_PARENT(dp)));

	if( IS_COMPLEX(dp) ){
		assert( (offset & 1) == 0 );
		offset /= 2;
//fprintf(stderr,"Adjusted offset (%d) for complex object %s\n",offset,OBJ_NAME(dp));
	} else if( IS_QUAT(dp) ){
		assert( (offset & 3) == 0 );
		offset /= 4;
	}

	SET_OBJ_DATA_PTR(dp,OBJ_DATA_PTR(OBJ_PARENT(dp)));
	SET_OBJ_OFFSET( dp, OBJ_OFFSET(OBJ_PARENT(dp)) + offset );

#else // USE_OPENCL_SUBREGION
	cl_mem buf;
	cl_mem parent_buf;
	cl_buffer_region reg;
	cl_int status;
	int extra_offset;

	parent_buf = find_parent_buf(QSP_ARG  OBJ_PARENT(dp),&extra_offset);
	assert( parent_buf != NULL );

	reg.origin = (offset+extra_offset) * ELEMENT_SIZE(dp);

	// No - the region has to be big enough for all of the elements.
	// The safest thing is to include everything from the start
	// of the subregion to the end of the parent.  Note that this
	// cannot handle negative increments!?
	// reg.size = OBJ_N_MACH_ELTS(dp) * ELEMENT_SIZE(dp);

	//   p p p p p p p
	//   p p c c c p p
	//   p p p p p p p
	//   p p c c c p p

	reg.size =	  OBJ_SEQ_INC(dp)*(OBJ_SEQS(dp)-1)
			+ OBJ_FRM_INC(dp)*(OBJ_FRAMES(dp)-1)
			+ OBJ_ROW_INC(dp)*(OBJ_ROWS(dp)-1)
			+ OBJ_PXL_INC(dp)*(OBJ_COLS(dp)-1)
			+ OBJ_COMP_INC(dp)*(OBJ_COMPS(dp)-1)
			+ 1;
	reg.size *= ELEMENT_SIZE(dp);
//fprintf(stderr,"requesting subregion of %ld bytes at offset %ld\n",
//reg.size,reg.origin);

	buf = clCreateSubBuffer ( parent_buf,
				CL_MEM_READ_WRITE,
				CL_BUFFER_CREATE_TYPE_REGION,
		&reg,
			&status);
	if( status != CL_SUCCESS ){
		report_ocl_error(QSP_ARG  status, "clCreateSubBuffer");
		SET_OBJ_DATA_PTR(dp,OBJ_DATA_PTR(OBJ_PARENT(dp)));
	} else {
		SET_OBJ_DATA_PTR(dp,buf);
	}
	// BUG - Because this object doesn't "own" the data, the sub-buffer
	// won't be released when the object is destroyed, a possible memory
	// leak...
	// We need to add a special case, or make data releasing a
	// platform-specific function...
#endif // USE_OPENCL_SUBREGION
}

// use register_buf for interoperability with OpenGL...

static int ocl_register_buf(QSP_ARG_DECL  Data_Obj *dp)
{
	if( opengl_prohibited )
		ERROR1("ocl_register_buf:  Need to specify GL window BEFORE initializing OpenCL!?");

#ifdef HAVE_OPENGL
	cl_mem img;
	cl_int status;


	// Texture2D deprecated on Apple
//fprintf(stderr,"obj %s has texture id %d\n",OBJ_NAME(dp),OBJ_TEX_ID(dp));
//fprintf(stderr,"obj %s has platform device %s\n",OBJ_NAME(dp),PFDEV_NAME(OBJ_PFDEV(dp)));

//advise("ocl_register_buf calling clCreateFromGLBuffer");
//longlist(QSP_ARG  dp);
	// Used to call clCreateFromGLTexture, but this works:
	img = clCreateFromGLBuffer(
				OCLDEV_CTX( OBJ_PFDEV(dp) ),	// OCL context
				CL_MEM_READ_WRITE,		// flags
				OBJ_TEX_ID(dp),			// from glBufferData?
				&status);

	if( status != CL_SUCCESS ){
		report_ocl_error(QSP_ARG  status, "clCreateFromGLTexture");
		return -1;
	} else {
		SET_OBJ_DATA_PTR(dp,img);
	}

	// dp is a special buffer object...
	//cl_mem memobj;

	//cl_mem = clCreate
	return 0;
#else // ! HAVE_OPENGL
	WARN("ocl_register_buf:  Sorry, no OpenGL support in this build!?");
	return -1;
#endif // ! HAVE_OPENGL
}

// map_buf makes an opengl buffer object usable by OpenCL?

static int ocl_map_buf(QSP_ARG_DECL  Data_Obj *dp)
{
	cl_int status;

	glFlush();

	// Acquire ownership of GL texture for OpenCL Image
	status = clEnqueueAcquireGLObjects(//cl_cmd_queue,
			OCLDEV_QUEUE(OBJ_PFDEV(dp)),
			1,		// num_images
			(const cl_mem *)(& OBJ_DATA_PTR(dp)),
			0,
			0,
			0);

	if( status != CL_SUCCESS ){
		report_ocl_error(QSP_ARG  status, "clEnqueueAcquireGLObjects");
		return -1;
	}

	// Now ready to execute kernel or other OpenCL operations ... ?
	return 0;
}

static int ocl_unmap_buf(QSP_ARG_DECL  Data_Obj *dp)
{
#ifdef HAVE_OPENGL
	cl_int status;

	// Release ownership of GL texture for OpenCL Image
	status = clEnqueueReleaseGLObjects(//cl_cmd_queue,
			OCLDEV_QUEUE(OBJ_PFDEV(dp)),
			1,	// num objects
			(const cl_mem *)(& OBJ_DATA_PTR(dp)),	// cl_mem *mem_objects
			0,		// num_events_in_wait_list
			NULL,		// const cl_event *wait_list
			NULL		// cl_event *event
			);
	if( status != CL_SUCCESS ){
		report_ocl_error(QSP_ARG  status, "clEnqueueReleaseGLObjects");
		return -1;
	}
	// Force pending CL commands to get executed
	status = clFlush( OCLDEV_QUEUE(OBJ_PFDEV(dp)) );
	if( status != CL_SUCCESS ){
		report_ocl_error(QSP_ARG  status, "clFlush");
	}
	
	// Bind GL texture and use for rendering
	glBindTexture( //gl_texture_target,
			GL_TEXTURE_2D,
			//gl_texture_id
			OBJ_TEX_ID(dp)

			);
	return 0;
#else // ! HAVE_OPENGL
	WARN("ocl_unmap_buf:  Sorry, no OpenGL support in this build!?");
	return -1;
#endif // ! HAVE_OPENGL
}

static const char *ocl_kernel_string(QSP_ARG_DECL  Platform_Kernel_String_ID which )
{
	const char *s;

	switch(which){
		case PKS_KERNEL_QUALIFIER:
			s="__kernel";
			break;
		case PKS_ARG_QUALIFIER:
			s="__global";
			break;
		case N_PLATFORM_KERNEL_STRINGS:
		default:
			ERROR1("invalid platform string ID");
			s=NULL;
			break;
	}
	return s;
}

// Can't be static because used by ocl_rand

/*cl_kernel*/ void *ocl_make_kernel(QSP_ARG_DECL  const char *ksrc,const char *kernel_name,Platform_Device *pdp)
{
	cl_program program;
	static cl_kernel kernel;	// is this really a pointer???
fprintf(stderr,"sizeof(cl_kernel) = %ld\n",sizeof(cl_kernel));

	program = ocl_create_program(ksrc,pdp);
	if( program == NULL )
		ERROR1("program creation failure!?");

	kernel = ocl_create_kernel(program, kernel_name, pdp);
	if( kernel == NULL ){
		ADVISE("Source code of failed program:");
		ADVISE(ksrc);
		ERROR1("kernel creation failure!?");
	}
	assert( sizeof(cl_kernel) == sizeof(void *) );

	return (void *) kernel;
}

static void ocl_store_kernel(QSP_ARG_DECL  Kernel_Info_Ptr *kip_p, void *kp, Platform_Device *pdp)
{
	Kernel_Info_Ptr kip;
	int idx;

	if( (*kip_p).ocl_kernel_info_p == NULL ){
		kip.ocl_kernel_info_p = getbuf( sizeof(OpenCL_Kernel_Info) );
		*kip_p = kip;
fprintf(stderr,"ocl_store_kernel:  allocated kernel info at 0x%lx\n",(long)kip.ocl_kernel_info_p);
	} else {
		kip = (*kip_p);
fprintf(stderr,"ocl_store_kernel:  using previously allocated kernel info at 0x%lx\n",(long)kip.ocl_kernel_info_p);
	}

	idx = PFDEV_SERIAL(pdp);
	assert( idx >=0 && idx < MAX_OPENCL_DEVICES );
fprintf(stderr,"stored kernel 0x%lx\n",(long)kp);
	SET_OCL_KI_KERNEL( kip, idx, kp ); 
}

static void * ocl_fetch_kernel(QSP_ARG_DECL  Kernel_Info_Ptr kip, Platform_Device *pdp)
{
	int idx;
	void *kp;

	idx = PFDEV_SERIAL(pdp);
	assert( idx >=0 && idx < MAX_OPENCL_DEVICES );
	assert(kip.any_kernel_info_p != NULL);
	kp = OCL_KI_KERNEL( kip, idx ); 
fprintf(stderr,"returning fetched kernel 0x%lx\n",(long)kp);
	return kp;
}

/* possible values for code:
 * CL_PLATFORM_PROFILE
 * CL_PLATFORM_VERSION
 * CL_PLATFORM_NAME
 * CL_PLATFORM_VENDOR
 * CL_PLATFORM_EXTENSIONS
 */

#define GET_PLATFORM_STRING(code)					\
	/* First figure out the required size */			\
	status = clGetPlatformInfo(platform_id,code,			\
		0,NULL,&ret_size);					\
	if( status != CL_SUCCESS ){					\
		report_ocl_error(QSP_ARG  status, "clGetPlatformInfo");	\
		return;							\
		/* BUG make sure to return cleanly... */		\
	}								\
	platform_str = getbuf(ret_size+1);				\
	status = clGetPlatformInfo(platform_id,code,			\
		ret_size+1,platform_str,&ret_size);			\
	if( status != CL_SUCCESS ){					\
		report_ocl_error(QSP_ARG  status, "clGetPlatformInfo");	\
		return;							\
		/* BUG make sure to return cleanly... */		\
	}

static void init_ocl_platform(QSP_ARG_DECL  cl_platform_id platform_id)
{
	Compute_Platform *cpp;
	cl_int status;
	//char param_data[MAX_PARAM_SIZE];
	char *platform_str;
	size_t ret_size;

	GET_PLATFORM_STRING(CL_PLATFORM_NAME)

	cpp = creat_platform(QSP_ARG  platform_str, PLATFORM_OPENCL);
	givbuf(platform_str);

	GET_PLATFORM_STRING(CL_PLATFORM_PROFILE)
	SET_OCLPF_PROFILE(cpp,platform_str);

	GET_PLATFORM_STRING(CL_PLATFORM_VERSION)
	SET_OCLPF_VERSION(cpp,platform_str);

	GET_PLATFORM_STRING(CL_PLATFORM_VENDOR)
	SET_OCLPF_VENDOR(cpp,platform_str);

	GET_PLATFORM_STRING(CL_PLATFORM_EXTENSIONS)
	SET_OCLPF_EXTENSIONS(cpp,platform_str);

	SET_OCLPF_ID(cpp,platform_id);

	SET_PLATFORM_FUNCTIONS(cpp,ocl)

	SET_PF_FUNC_TBL(cpp,ocl_vfa_tbl);

	// BUG need to set vfa_tbl here too!

	//icp = create_item_context(QSP_ARG  pfdev_itp, PLATFORM_NAME(cpp) );
	//push_item_context(QSP_ARG  pfdev_itp, icp );
	push_pfdev_context(QSP_ARG  PF_CONTEXT(cpp) );
	init_ocl_devices(QSP_ARG  cpp);
	if( pop_pfdev_context(SINGLE_QSP_ARG) == NULL )
		ERROR1("init_ocl_platform:  Failed to pop platform device context!?");
}

//In general Intel CPU and NV/AMD's GPU are in different platforms
//But in Mac OSX, all the OpenCL devices are in the platform "Apple"

#define MAX_CL_PLATFORMS	3

static int init_ocl_platforms(SINGLE_QSP_ARG_DECL)
{
	cl_platform_id	platform_ids[MAX_CL_PLATFORMS];
	cl_uint		num_platforms;
	cl_int		ret;
	int		i;

	// BUG need to add error checking on the return values...

	ret = clGetPlatformIDs(MAX_CL_PLATFORMS, platform_ids, &num_platforms);
//fprintf(stderr,"init_ocl_platform:  %d platform%s found\n",num_platforms,num_platforms==1?"":"s");

	for(i=0;i<num_platforms;i++)
		init_ocl_platform(QSP_ARG  platform_ids[i]);

	return num_platforms;
}

// this "platform" is OpenCL, not an OpenCL "platform" ...

void ocl_init_platform(SINGLE_QSP_ARG_DECL)
{
	init_ocl_platforms(SINGLE_QSP_ARG);

	check_ocl_vfa_tbl(SINGLE_QSP_ARG);
}

void show_gpu_vector(QSP_ARG_DECL  Platform_Device *pdp, void *ptr, int len )
{
	// BUG we assume float type!?
	float *buf;
	size_t siz;
	int i;

	siz= len*sizeof(float);
	buf=malloc(siz);
	if( buf==NULL ) NERROR1("show_gpu_vector:  error allocating buffer!?");

	fprintf(stderr,"show_gpu_vector:  src = 0x%lx\n",(long)ptr);
	// now do the memory transfer
	(*PF_MEM_DNLOAD_FN(PFDEV_PLATFORM(pdp)))(QSP_ARG  buf, ptr, siz, 0, pdp );
	for(i=0;i<len;i++){
		fprintf(stderr,"%d\t%g\n",i,buf[i]);
	}
	free(buf);
}

#endif /* HAVE_OPENCL */

