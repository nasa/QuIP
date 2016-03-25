#include "quip_config.h"

#include "my_cu2.h"
#include "quip_prot.h"
#include "veclib_api.h"
#include "veclib/cu2_port.h"

//#ifdef HAVE_OPENCL

//#define MEM_SIZE (16)//suppose we have a vector with 128 elements
#define MAX_SOURCE_SIZE (0x100000)

//In general Intel CPU and NV/AMD's GPU are in different platforms
//But in Mac OSX, all the OpenCL devices are in the platform "Apple"

#define MAX_PARAM_SIZE	128

//static const char *default_cu2_dev_name=NULL;
//static const char *first_cu2_dev_name=NULL;
//static int default_cu2_dev_found=0;

#define ERROR_CASE(code,string)	case code: msg = string; break;

#ifdef CAUTIOUS
#define INSURE_CURR_ODP(whence)					\
	if( curr_pdp == NULL ){					\
		sprintf(ERROR_STRING,"CAUTIOUS:  %s:  curr_pdp is null!?",#whence);	\
		WARN(ERROR_STRING);				\
	}
#else // ! CAUTIOUS
#define INSURE_CURR_ODP(whence)
#endif // ! CAUTIOUS


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

/*
PF_COMMAND_FUNC( dev_info )
{
	Platform_Device *pdp;

	pdp = PICK_PFDEV((char *)"device");
	if( pdp == NO_PFDEV ) return;

	//print_cudev_info_short(QSP_ARG  pdp);
	WARN("do_cudev_info not implemented!?");
}
*/

#ifdef FOOBAR
// merge with opencl and put in platforms.c ???  with ifdefs if necessary?

void PF_FUNC_NAME(init_dev_memory)(QSP_ARG_DECL  Platform_Device *pdp)
{
	char cname[LLEN];
	char dname[LLEN];
	Data_Area *ap;

	strcpy(dname,PFDEV_NAME(pdp));
	// what should the name for the memory area be???

	// address set to NULL says use custom allocator - see dobj/makedobj.c

	ap = area_init(QSP_ARG  dname,NULL,0, MAX_CUDA_GLOBAL_OBJECTS,DA_CUDA_GLOBAL);
	if( ap == NO_AREA ){
		sprintf(ERROR_STRING,
	"init_dev_memory:  error creating global data area %s",dname);
		WARN(ERROR_STRING);
	}
	// g++ won't take this line!?
//fprintf(stderr,"initializing data areas for device %s\n",
//PFDEV_NAME(pdp) );

	SET_AREA_PFDEV(ap,pdp);
	//set_device_for_area(ap,pdp);

	// BUG should be per-device, not global table...
	SET_PFDEV_AREA(pdp,PF_GLOBAL_AREA_INDEX,ap);


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

	strcat(cname,"_host");
	ap = area_init(QSP_ARG  cname,(u_char *)NULL,0,MAX_CUDA_MAPPED_OBJECTS,
								DA_CUDA_HOST);
	if( ap == NO_AREA ){
		sprintf(ERROR_STRING,
	"init_dev_memory:  error creating host data area %s",cname);
		ERROR1(ERROR_STRING);
	}
	SET_AREA_PFDEV(ap, pdp);
	SET_PFDEV_AREA(pdp,PF_HOST_AREA_INDEX,ap);

	/* Make up another psuedo-area for the mapped host memory;
	 * This is the same memory as above, but mapped to the device.
	 * In the current implementation, we create objects in the host
	 * area, and then automatically create an alias on the device side.
	 * There is a BUG in that by having this psuedo area in the data
	 * area name space, a user could select it as the data area and
	 * then try to create an object.  We will detect this in make_dobj,
	 * and complain.
	 */

	strcpy(cname,dname);
	strcat(cname,"_host_mapped");
	ap = area_init(QSP_ARG  cname,(u_char *)NULL,0,MAX_CUDA_MAPPED_OBJECTS,
							DA_CUDA_HOST_MAPPED);
	if( ap == NO_AREA ){
		sprintf(ERROR_STRING,
	"init_dev_memory:  error creating host-mapped data area %s",cname);
		ERROR1(ERROR_STRING);
	}
	SET_AREA_PFDEV(ap,pdp);
	SET_PFDEV_AREA(pdp,PF_HOST_MAPPED_AREA_INDEX,ap);

	if( verbose ){
		sprintf(ERROR_STRING,"init_dev_memory DONE");
		advise(ERROR_STRING);
	}
} // init_dev_memory

void PF_FUNC_NAME(shutdown)(void)
{
	//cl_int status;

	/*
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	*/
	NWARN("shutdown_cu2_platform NOT implemented!?");

	// Need to iterate over all devices...
}

void cu2_alloc_data(QSP_ARG_DECL  Data_Obj *dp, dimension_t size)
{
	WARN("cu2_alloc_data not implemented!?");
}

void PF_FUNC_NAME(sync)(SINGLE_QSP_ARG_DECL)
{
	WARN("sync_cu2:  not implemented!?");
}
#endif // FOOBAR

//#endif // HAVE_OPENCL
