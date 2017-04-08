/* ocl_port.m4 BEGIN */


/* gen_port.m4 BEGIN */
#include "quip_prot.h"
#include "shape_bits.h"

/* NOT Suppressing ! */


/* gen_port.m4 DONE */




/* Suppressing ! */

/* NOT Suppressing ! */


#include "platform.h"
extern void *ocl_tmp_vec(Platform_Device *pdp, size_t size, size_t len, const char *whence);
extern void ocl_free_tmp(void *a, const char *whence);
extern int get_max_threads_per_block(Data_Obj *odp);
extern int max_threads_per_block;

/* ocl_port.m4 END */





/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/ocl_veclib_prot.m4
/* ocl_veclib_prot.m4 BEGIN */
/* ocl_port.m4 BEGIN */


/* gen_port.m4 BEGIN */
#include "quip_prot.h"
#include "shape_bits.h"

/* NOT Suppressing ! */


/* gen_port.m4 DONE */




/* Suppressing ! */

/* NOT Suppressing ! */


#include "platform.h"
extern void *ocl_tmp_vec(Platform_Device *pdp, size_t size, size_t len, const char *whence);
extern void ocl_free_tmp(void *a, const char *whence);
extern int get_max_threads_per_block(Data_Obj *odp);
extern int max_threads_per_block;

/* ocl_port.m4 END */





/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/vecgen.m4
/* vecgen.m4 BEGIN */

/* defns shared by veclib & warlib */







// Why are these called link funcs?  Maybe because they can be chained?
// Kind of a legacy from the old skywarrior library code...
// A vector arg used to just have a length and a stride, but now
// with gpus we have three-dimensional lengths.  But in principle
// there's no reason why we couldn't have full shapes passed...



/* vecgen.m4 DONE */




/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/vecgen.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/ocl_func_prot.m4

extern void h_ocl_vuni(const int vf_code,  /*const*/ Vec_Obj_Args *oap);
extern void h_ocl_sp_vuni(const int vf_code,  /*const*/ Vec_Obj_Args *oap);
extern void h_ocl_dp_vuni(const int vf_code,  /*const*/ Vec_Obj_Args *oap);


extern void h_ocl_fft2d(const int vf_code,  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_ift2d(const int vf_code,  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_fftrows(const int vf_code,  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_iftrows(const int vf_code,  Data_Obj *dst_dp, Data_Obj *src_dp);





/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/ocl_func_prot.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/platform_funcs.m4



extern COMMAND_FUNC( do_ocl_obj_dnload );
extern COMMAND_FUNC( do_ocl_obj_upload );
extern COMMAND_FUNC( do_ocl_list_devs );
extern COMMAND_FUNC( do_ocl_set_device );

extern void ocl_set_device(QSP_ARG_DECL  Platform_Device *cdp);
extern Platform_Stream * ocl_new_stream(QSP_ARG_DECL  const char *name);

extern void ocl_list_streams(SINGLE_QSP_ARG_DECL);
extern COMMAND_FUNC( do_ocl_new_stream );
extern COMMAND_FUNC( do_ocl_list_streams );
extern COMMAND_FUNC( do_ocl_stream_info );
extern COMMAND_FUNC( do_ocl_sync_stream );
extern COMMAND_FUNC( do_ocl_init_ckpts );
extern COMMAND_FUNC( do_ocl_set_ckpt );
extern COMMAND_FUNC( do_ocl_clear_ckpts );
extern COMMAND_FUNC( do_ocl_show_ckpts );
extern void ocl_shutdown(void);
extern void ocl_sync(SINGLE_QSP_ARG_DECL);
extern void ocl_init_dev_memory(QSP_ARG_DECL  Platform_Device *pdp);
extern void ocl_insure_device(QSP_ARG_DECL  Data_Obj *dp);

	


extern void ocl_init_platform(SINGLE_QSP_ARG_DECL);
extern void ocl_init(SINGLE_QSP_ARG_DECL);
extern void ocl_alloc_data(QSP_ARG_DECL  Data_Obj *dp, dimension_t size);





/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/platform_funcs.m4

/* ocl_veclib_prot.m4 DONE */





/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/ocl_veclib_prot.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ocl_kern_args.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/gen_kern_args.m4

/* Suppressing ! */

/* NOT Suppressing ! */





/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/gen_kern_args.m4





















































// BUG - need to make sure consistent with expected args???
























































































































































































































































































// All eqsp args







// All flen args




































/* NOT Suppressing ! */

// END INCLUDED FILE ocl_kern_args.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE _ocl_rand.c

/* Use threefry parallel random number generator from Random123 package */

#include "quip_config.h"

#ifdef HAVE_OPENCL

#include "quip_prot.h"
#include "my_ocl.h"
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

void h_ocl_sp_vuni(const int vf_code,  /*const*/ Vec_Obj_Args *oap)
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

	
	status = clSetKernelArg(ocl_sp_vuni_kernel,	ki_idx++, sizeof(void *), &(OBJ_DATA_PTR( OA_DEST(oap))) );
	if( status != CL_SUCCESS )
		report_ocl_error(DEFAULT_QSP_ARG  status, "clSetKernelArg" );

	
	status = clSetKernelArg(ocl_sp_vuni_kernel,	ki_idx++, sizeof(int), &ocl_sp_vuni_counter );
	if( status != CL_SUCCESS )
		report_ocl_error(DEFAULT_QSP_ARG  status, "clSetKernelArg" );

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

void h_ocl_dp_vuni(const int vf_code,  /*const*/ Vec_Obj_Args *oap)
{
	NWARN("Sorry, dp_vuni not implemented for OpenCL");
}

#endif // HAVE_OPENCL



/* NOT Suppressing ! */

// END INCLUDED FILE _ocl_rand.c


