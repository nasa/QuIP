#include "quip_config.h"
#include "veclib/cu2_veclib_prot.h"
#include "cuda_supp.h"
//#include "quip_prot.h"
//#include "data_obj.h"

/* cu2_port.m4 BEGIN */
/* gen_port.m4 BEGIN */
#include "quip_prot.h"
#include "shape_bits.h"

/* NOT Suppressing ! */


/* gen_port.m4 DONE */



#define BUILD_FOR_CUDA
#define BUILD_FOR_GPU


/* Suppressing ! */

/* NOT Suppressing ! */


extern void *cu2_tmp_vec(Platform_Device *pdp, size_t size, size_t len, const char *whence);
extern void cu2_free_tmp(void *a, const char *whence);

/* cu2_port.m4 DONE */





/* NOT Suppressing ! */

// BEGIN INCLUDED FILE _cu2_utils.c
#include "quip_config.h"

#include "my_cu2.h"
#include "quip_prot.h"
#include "veclib_api.h"
#include "veclib/cu2_veclib_prot.h"
#include "platform.h"

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
COMMAND_FUNC( do_cu2_dev_info  )
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

void cu2_init_dev_memory(QSP_ARG_DECL  Platform_Device *pdp)
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

void cu2_shutdown(void)
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

void cu2_sync(SINGLE_QSP_ARG_DECL)
{
	WARN("sync_cu2:  not implemented!?");
}
#endif // FOOBAR


void cu2_set_device( QSP_ARG_DECL  Platform_Device *pdp )
{
#ifdef HAVE_CUDA
	cudaError_t e;
#endif // HAVE_CUDA

	if( curr_pdp == pdp ){
		sprintf(DEFAULT_ERROR_STRING,"%s:  current device is already %s!?",
			STRINGIFY(h_cu2_set_device),PFDEV_NAME(pdp));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( PFDEV_PLATFORM_TYPE(pdp) != PLATFORM_CUDA ){
		sprintf(ERROR_STRING,"%s:  device %s is not a CUDA device!?",
			STRINGIFY(h_cu2_set_device),PFDEV_NAME(pdp));
		WARN(ERROR_STRING);
		return;
	}

#ifdef HAVE_CUDA
	e = cudaSetDevice( PFDEV_CUDA_DEV_INDEX(pdp) );
	if( e != cudaSuccess )
		describe_cuda_driver_error2(STRINGIFY(h_cu2_set_device),"cudaSetDevice",e);
	else
		curr_pdp = pdp;
#else // ! HAVE_CUDA
	NO_CUDA_MSG(set_device)
#endif // ! HAVE_CUDA
}

COMMAND_FUNC( do_cu2_list_devs  )
{
	list_pfdevs(QSP_ARG  tell_msgfile(SINGLE_QSP_ARG));
}

void insure_cu2_device( QSP_ARG_DECL  Data_Obj *dp )
{
	Platform_Device *pdp;

	if( AREA_FLAGS(OBJ_AREA(dp)) & DA_RAM ){
		sprintf(DEFAULT_ERROR_STRING,
	"insure_cu2_device:  Object %s is a host RAM object!?",OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	pdp = AREA_PFDEV(OBJ_AREA(dp));

#ifdef CAUTIOUS
	if( pdp == NULL )
		NERROR1("CAUTIOUS:  null cuda device ptr in data area!?");
#endif /* CAUTIOUS */

	if( curr_pdp != pdp ){
sprintf(DEFAULT_ERROR_STRING,"insure_cu2_device:  curr_pdp = 0x%lx  pdp = 0x%lx",
(int_for_addr)curr_pdp,(int_for_addr)pdp);
NADVISE(DEFAULT_ERROR_STRING);

sprintf(DEFAULT_ERROR_STRING,"insure_cu2_device:  current device is %s, want %s",
PFDEV_NAME(curr_pdp),PFDEV_NAME(pdp));
NADVISE(DEFAULT_ERROR_STRING);
		cu2_set_device(QSP_ARG  pdp);
	}

}

void *cu2_tmp_vec (Platform_Device *pdp, size_t size,size_t len,const char *whence)
{
	// Why is this commented out???
/*
	void *cuda_mem;
	cudaError_t drv_err;

	drv_err = cudaMalloc(&cuda_mem, size * len );
	if( drv_err != cudaSuccess ){
		sprintf(DEFAULT_MSG_STR,"tmpvec (%s)",whence);
		describe_cuda_driver_error2(DEFAULT_MSG_STR,"cudaMalloc",drv_err);
		NERROR1("CUDA memory allocation error");
	}

//sprintf(ERROR_STRING,"tmpvec:  %d bytes allocated at 0x%lx",len,(int_for_addr)cuda_mem);
//advise(ERROR_STRING);

//sprintf(ERROR_STRING,"tmpvec %s:  0x%lx",whence,(int_for_addr)cuda_mem);
//advise(ERROR_STRING);
	return(cuda_mem);
	*/
	return NULL;
}

void cu2_free_tmp (void *ptr,const char *whence)
{
	/*
	cudaError_t drv_err;

//sprintf(ERROR_STRING,"freetmp %s:  0x%lx",whence,(int_for_addr)ptr);
//advise(ERROR_STRING);
	drv_err=cudaFree(ptr);
	if( drv_err != cudaSuccess ){
		sprintf(DEFAULT_MSG_STR,"freetmp (%s)",whence);
		describe_cuda_driver_error2(DEFAULT_MSG_STR,"cudaFree",drv_err);
	}
	*/
}

typedef struct {
	const char *	ckpt_tag;
#ifdef FOOBAR
	cudaEvent_t	ckpt_event;
#endif // FOOBAR
} CU2_Checkpoint;

static CU2_Checkpoint *ckpt_tbl=NULL;
//static int max_cu2_ckpts=0;	// size of checkpoit table
static int n_cu2_ckpts=0;	// number of placements

static void init_cu2_ckpts(int n)
{
	/*
	//CUresult e;
	cudaError_t drv_err;
	int i;

	if( max_cu2_ckpts > 0 ){
		sprintf(DEFAULT_ERROR_STRING,
"init_cu2_ckpts (%d):  already initialized with %d checpoints",
			n,max_cu2_ckpts);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	ckpt_tbl = (Cuda_Checkpoint *) getbuf( n * sizeof(*ckpt_tbl) );
	if( ckpt_tbl == NULL ) NERROR1("failed to allocate checkpoint table");

	max_cu2_ckpts = n;

	for(i=0;i<max_cu2_ckpts;i++){
		drv_err=cudaEventCreate(&ckpt_tbl[i].ckpt_event);
		if( drv_err != cudaSuccess ){
			describe_cuda_driver_error2("init_cu2_ckpts",
				"cudaEventCreate",drv_err);
			NERROR1("failed to initialize checkpoint table");
		}
		ckpt_tbl[i].ckpt_tag=NULL;
	}
	*/
}


COMMAND_FUNC( do_cu2_init_ckpts  )
{
	int n;

	n = HOW_MANY("maximum number of checkpoints");
	init_cu2_ckpts(n);
}


COMMAND_FUNC( do_cu2_set_ckpt  )
{
	/*
	//cudaError_t e;
	cudaError_t drv_err;
	const char *s;

	s = NAMEOF("tag for this checkpoint");

	if( max_cu2_ckpts == 0 ){
		NWARN("do_place_ckpt:  checkpoint table not initialized, setting to default size");
		init_cu2_ckpts(256);
	}

	if( n_cu2_ckpts >= max_cu2_ckpts ){
		sprintf(ERROR_STRING,
	"do_place_ckpt:  Sorry, all %d checkpoints have already been placed",
			max_cu2_ckpts);
		WARN(ERROR_STRING);
		return;
	}

	ckpt_tbl[n_cu2_ckpts].ckpt_tag = savestr(s);

	// use default stream (0) for now, but will want to introduce
	// more streams later?
	drv_err = cudaEventRecord( ckpt_tbl[n_cu2_ckpts++].ckpt_event, 0 );
	CUDA_DRIVER_ERROR_RETURN( "do_place_ckpt","cudaEventRecord")
	*/
}

COMMAND_FUNC( do_cu2_show_ckpts  )
{
	/*
	CUresult e;
	cudaError_t drv_err;
	float msec, cum_msec;
	int i;

	if( n_cu2_ckpts <= 0 ){
		NWARN("do_show_cu2_ckpts:  no checkpoints placed!?");
		return;
	}

	drv_err = cudaEventSynchronize(ckpt_tbl[n_cu2_ckpts-1].ckpt_event);
	CUDA_DRIVER_ERROR_RETURN("do_show_cu2_ckpts", "cudaEventSynchronize")

	drv_err = cudaEventElapsedTime( &msec, ckpt_tbl[0].ckpt_event, ckpt_tbl[n_cu2_ckpts-1].ckpt_event);
	CUDA_DRIVER_ERROR_RETURN("do_show_cu2_ckpts", "cudaEventElapsedTime")
	sprintf(msg_str,"Total GPU time:\t%g msec",msec);
	prt_msg(msg_str);

	// show the start tag
	sprintf(msg_str,"GPU  %3d  %12.3f  %12.3f  %s",1,0.0,0.0,
		ckpt_tbl[0].ckpt_tag);
	prt_msg(msg_str);
	cum_msec =0.0;
	for(i=1;i<n_cu2_ckpts;i++){
		drv_err = cudaEventElapsedTime( &msec, ckpt_tbl[i-1].ckpt_event,
			ckpt_tbl[i].ckpt_event);
		CUDA_DRIVER_ERROR_RETURN("do_show_cu2_ckpts", "cudaEventElapsedTime")

		cum_msec += msec;
		sprintf(msg_str,"GPU  %3d  %12.3f  %12.3f  %s",i+1,msec,
			cum_msec, ckpt_tbl[i].ckpt_tag);
		prt_msg(msg_str);
	}
	*/
}

COMMAND_FUNC( do_cu2_clear_ckpts  )
{
	int i;

	for(i=0;i<n_cu2_ckpts;i++){
		rls_str(ckpt_tbl[i].ckpt_tag);
		ckpt_tbl[i].ckpt_tag=NULL;
	}
	n_cu2_ckpts=0;
}



#include "cu2.c"

//#endif // HAVE_OPENCL



/* NOT Suppressing ! */

// END INCLUDED FILE _cu2_utils.c


