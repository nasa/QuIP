
/* Use threefry parallel random number generator from Random123 package */

#include "quip_config.h"

#ifdef HAVE_OPENCL

#include "quip_prot.h"
#include "my_ocl.h"
#include "veclib/ocl_veclib_prot.h"
#include "ocl_kern_args.h"
#include "ocl_platform.h"
#include <string.h>	// strcmp - for debugging
#include <sys/time.h>	// gettimeofday

#include "vuni.i"	// kernel source string

// The generator has no random seed currently, so it
// will always produce the same "random" data!?
static int ocl_sp_vuni_counter=0;
static int ocl_sp_vuni_seeded=0;
static int ocl_sp_vuni_inited=0;
static cl_kernel ocl_sp_vuni_kernel = NULL;

// BUG the kernel is created for the device,
// so this code is broken in the case that we have multiple devices

static int ocl_sp_vuni_init(Platform_Device *pdp)
{
	struct timeval tv;

	ocl_sp_vuni_inited=1;
	// BUG need to be able to name the source string!?
	if( (ocl_sp_vuni_kernel=ocl_make_kernel(opencl_src,"g_ocl_fast_sp_vuni",pdp))
			== NULL ){
		NWARN("Error creating kernel for g_ocl_fast_sp_vuni!?");
		return -1;
	}
	if( !ocl_sp_vuni_seeded ){
		// "randomize" the seed so that we don't get the same numbers every time
		if( gettimeofday(&tv,NULL) < 0 ){
			NWARN("ocl_sp_vuni_init:  error calling gettimeofday, not setting random seed!?");
		} else {
			ocl_sp_vuni_counter = 100*(tv.tv_usec / 100);
			ocl_sp_vuni_counter += tv.tv_sec % 100;
		}
	}
	return 0;
}

void h_ocl_set_seed(int seed)
{
	ocl_sp_vuni_counter = seed;
	ocl_sp_vuni_seeded = 1;
}

void h_ocl_sp_vuni(HOST_CALL_ARG_DECLS)
{
	size_t global_work_size[3] = {1, 1, 1};
	int ki_idx=0;

	//NWARN("Sorry, sp_vuni not implemented for OpenCL");

	// BUG should make sure destination is contiguous!

	cl_int status;

	if( ! ocl_sp_vuni_inited ){
		if( ocl_sp_vuni_init( OBJ_PFDEV(OA_DEST(oap)) ) < 0 ) return;
	}

	global_work_size[0] = OBJ_N_MACH_ELTS(OA_DEST(oap));

	_SET_KERNEL_ARG( ocl_sp_vuni_kernel, void *, &(OBJ_DATA_PTR( OA_DEST(oap))) )
	_SET_KERNEL_ARG( ocl_sp_vuni_kernel, int, &ocl_sp_vuni_counter )
	ocl_sp_vuni_counter ++;

	status = clEnqueueNDRangeKernel(				
		OCLDEV_QUEUE( OBJ_PFDEV(OA_DEST(oap)) ),
		ocl_sp_vuni_kernel,
		1,	/* work_dim, 1-3 */
		NULL,
		global_work_size,
		/*local_work_size*/ NULL,
		0,	/* num_events_in_wait_list */
		NULL,	/* event_wait_list */
		NULL	/* event */
		);
	if( status != CL_SUCCESS )
		report_ocl_error(DEFAULT_QSP_ARG  status, "clEnqueueNDRangeKernel" );
}

void h_ocl_dp_vuni(HOST_CALL_ARG_DECLS)
{
	NWARN("Sorry, dp_vuni not implemented for OpenCL");
}

#endif // HAVE_OPENCL
