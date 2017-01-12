
#ifndef _OCL_HOST_CALL_DEFS_H_
#define _OCL_HOST_CALL_DEFS_H_ 

// BUG - merge this file!!!

#include "ocl_kern_args.h"
#include "veclib/slow_len.h"
#include "veclib/slow_incs.h"
#include "veclib/eqsp_incs.h"
#include "veclib/slow_vars.h"

// this is 80 columns
//345678901234567890123456789012345678901234567890123456789012345678901234567890

#define CLEAR_GPU_ERROR(s)		/* nop */
#define CHECK_GPU_ERROR(s)		/* nop */

#define BLOCK_VARS_DECLS		// in MM_NOCC, may eliminate?
#define SET_BLOCK_COUNT			// nop?

#define SET_MAX_THREADS_FROM_OBJ(dp)	// nop

#ifdef MOVED

// can be shared with CUDA, should be moved?
// moved back to veclib/xfer_args.h, with BUILD_FOR_GPU guard...

#define XFER_DBM_GPU_INFO	if( BITMAP_OBJ_GPU_INFO_HOST_PTR(bitmap_dst_dp) == NULL ){				\
					/* only for gpu objects! */							\
					init_bitmap_gpu_info(bitmap_dst_dp);						\
				}											\
				SET_VA_DBM_GPU_INFO_PTR(vap, BITMAP_OBJ_GPU_INFO_DEV_PTR(bitmap_dst_dp));

#define XFER_EQSP_DBM_GPU_INFO	XFER_DBM_GPU_INFO									\
				SET_VA_ITERATION_TOTAL(vap,BMI_N_WORDS( BITMAP_OBJ_GPU_INFO_HOST_PTR(bitmap_dst_dp)));	\
fprintf(stderr,"XFER_EQSP_DBM_GPU_INFO:  iteration total = %d\n",VA_ITERATION_TOTAL(vap));

#define XFER_SLOW_DBM_GPU_INFO	XFER_DBM_GPU_INFO									\
				SET_VA_ITERATION_TOTAL(vap,BMI_N_WORDS( BITMAP_OBJ_GPU_INFO_HOST_PTR(bitmap_dst_dp)));
				// BUG?  need to set all sizes?

#endif // MOVED


// PORT - insure_gpu_device ???

#define GET_MAX_THREADS( dp )						\
									\
	/* insure_cuda_device( dp ); */					\
	insure_ocl_device( dp );					\
	max_threads_per_block = get_max_threads_per_block(dp);
	// BUG OpenCL doesn't use max_threads_per_block!?

#ifdef MORE_DEBUG

#define REPORT_KERNEL_CALL(name)				\
fprintf(stderr,"Calling kernel %s\n",STRINGIFY(HOST_FAST_CALL_NAME(name)));

#define REPORT_THREAD_INFO					\
								\
sprintf(DEFAULT_ERROR_STRING,"Blocks:  %d x %d x %d    Threads:  %d x %d x %d",	\
n_blocks.x,n_blocks.y,n_blocks.z,	\
n_threads_per_block.x,n_threads_per_block.y,n_threads_per_block.z);\
NADVISE(DEFAULT_ERROR_STRING);						\
sprintf(DEFAULT_ERROR_STRING,"Length:  %d x %d x %d    Extra:  %d x %d x %d",	\
VA_LEN_X(vap),VA_LEN_Y(vap),VA_LEN_Z(vap),extra.x,extra.y,extra.z);				\
NADVISE(DEFAULT_ERROR_STRING);

#define REPORT_THREAD_INFO2					\
								\
sprintf(DEFAULT_ERROR_STRING,"Blocks:  %d x %d    Threads:  %d x %d",	\
n_blocks.x,n_blocks.y,n_threads_per_block.x,n_threads_per_block.y);\
NADVISE(DEFAULT_ERROR_STRING);						\
sprintf(DEFAULT_ERROR_STRING,"Len1:  %ld   Len2:  %ld   Extra:  %d x %d",	\
len1,len2,extra.x,extra.y);					\
NADVISE(DEFAULT_ERROR_STRING);

#define REPORT_KERNEL_ENQUEUE(_n_dims)					\
									\
	if( verbose ){							\
		int i;							\
		fprintf(stderr,"FINISH_KERNEL_CALL, n_dims = %d\n",_n_dims); \
		for (i=0;i<_n_dims;i++)					\
			fprintf(stderr,"global_work_size[%d] = %ld\n",	\
				i,global_work_size[i]);			\
		fflush(stderr);						\
	}
									\
#else /* ! MORE_DEBUG */

#define REPORT_THREAD_INFO
#define REPORT_THREAD_INFO2
#define REPORT_KERNEL_CALL(name)
//#define REPORT_KERNEL_ENQUEUE
#define REPORT_KERNEL_ENQUEUE(n_dims)					\
									\
	if( verbose ){							\
		int i;							\
		fprintf(stderr,"FINISH_KERNEL_CALL, n_dims = %d\n",n_dims); \
		for (i=0;i<n_dims;i++)					\
			fprintf(stderr,"global_work_size[%d] = %ld\n",	\
				i,global_work_size[i]);			\
		fflush(stderr);						\
	}
									\

#endif /* ! MORE_DEBUG */


#define DEFAULT_YZ	// for MM_NOCC, should be cleaned up...

//#define REPORT_INCS(incs)					\
//	fprintf(stderr,"%s:  %d %d %d\n",#incs,incs.x,incs.y,incs.z);

// above stuff was cribbed from cuda code...
// How much can we use?

// The kernels need to be per-device...

#define PF_GPU_FAST_CALL(name,bitmap,typ,scalars,vectors)		\
	_PF_GPU_FAST_CALL(name,bitmap,typ,scalars,vectors)

#define _PF_GPU_FAST_CALL(name,bitmap,typ,scalars,vectors)		\
fprintf(stderr,"Need to implement PF_GPU_FAST_CALL (name = %s, bitmap = \"%s\", typ = \"%s\", scalars = \"%s\", vectors = \"%s\"\n", \
#name,#bitmap,#typ,#scalars,#vectors);

// Make this a nop if not waiting for kernels
#define DECLARE_OCL_EVENT	cl_event event;

// BUG we need a different OCL kernel for every device...
// We could 
// BUG - the initialization here needs to be changed if we change MAX_OPENCL_DEVICES
// But - we are probably safe, because the compiler will set un-specified
// elements to 0...

#define DECLARE_OCL_VARS						\
	static cl_kernel kernel[MAX_OPENCL_DEVICES] = {NULL,NULL,NULL,NULL};	\
	DECLARE_OCL_COMMON_VARS

#define DECLARE_OCL_COMMON_VARS						\
									\
	static cl_program program = NULL;				\
	cl_int status;							\
	DECLARE_OCL_EVENT						\
	int ki_idx=0;							\
	int pd_idx; /* need to set! */					\
	const char *ksrc;						\
	/* define the global size and local size			\
	 * (grid size and block size in CUDA) */			\
	size_t global_work_size[3] = {1, 1, 1};				\
	/* size_t local_work_size[3]  = {0, 0, 0}; */

// two different kernels used in one call (e.g. vmaxg - nocc_setup, nocc_helper

#define DECLARE_OCL_VARS_2							\
										\
	static cl_kernel kernel1[MAX_OPENCL_DEVICES] = {NULL,NULL,NULL,NULL};	\
	static cl_kernel kernel2[MAX_OPENCL_DEVICES] = {NULL,NULL,NULL,NULL};	\
	DECLARE_OCL_COMMON_VARS

#define CHECK_NOSPEED_KERNEL(name)						\
	CHECK_KERNEL(name,,GPU_CALL_NAME(name))

#define CHECK_NOSPEED_KERNEL_1(name)						\
	CHECK_KERNEL_1(name,,GPU_CALL_NAME(name))

#define CHECK_FAST_KERNEL_1(name)						\
	CHECK_KERNEL_1(name,fast,GPU_CALL_NAME(name))

#define CHECK_NOSPEED_KERNEL_2(name)						\
	CHECK_KERNEL_2(name,,GPU_CALL_NAME(name))

#define CHECK_FAST_KERNEL_2(name)						\
	CHECK_KERNEL_2(name,fast,GPU_CALL_NAME(name))

#define CHECK_FAST_KERNEL(name)						\
	CHECK_KERNEL(name,fast,GPU_FAST_CALL_NAME(name))

#define CHECK_EQSP_KERNEL(name)						\
	CHECK_KERNEL(name,eqsp,GPU_EQSP_CALL_NAME(name))

#define CHECK_SLOW_KERNEL(name)						\
	CHECK_KERNEL(name,slow,GPU_SLOW_CALL_NAME(name))

#define CHECK_KERNEL(name,ktyp,kname)					\
	_CHECK_KERNEL(kernel,name,ktyp,kname)

#define CHECK_KERNEL_1(name,ktyp,kname)					\
	_CHECK_KERNEL(kernel1,name,ktyp,kname)

#define CHECK_KERNEL_2(name,ktyp,kname)					\
	_CHECK_KERNEL(kernel2,name,ktyp,kname)

#define _CHECK_KERNEL(k,name,ktyp,kname)				\
	pd_idx = OCLDEV_IDX(VA_PFDEV(vap));				\
	if( k[pd_idx] == NULL ){	/* one-time initialization */	\
		ksrc = KERN_SOURCE_NAME(name,ktyp);			\
		program = ocl_create_program(ksrc,VA_PFDEV(vap));	\
		if( program == NULL ) 					\
			NERROR1("program creation failure!?");		\
									\
		k[pd_idx] = ocl_create_kernel(program, #kname, VA_PFDEV(vap));\
		if( k[pd_idx] == NULL ){ 					\
			NADVISE("Source code of failed program:");	\
			NADVISE(ksrc);					\
			NERROR1("kernel creation failure!?");		\
		}							\
	}


#define SETUP_FAST_BLOCKS_						\
									\
/*fprintf(stderr,"SETUP_FAST_BLOCKS_:  %d\n",VA_LENGTH(vap));*/\
	global_work_size[0] = VA_LENGTH(vap);

#define SETUP_FAST_BLOCKS_SBM_	SETUP_FAST_BLOCKS_

// BUG - the number of words we need to process depends on bit0 !?

#define SETUP_FAST_BLOCKS_DBM_						\
									\
	global_work_size[0] = VA_LENGTH( vap )/BITS_PER_BITMAP_WORD;

#define SETUP_EQSP_BLOCKS_DBM_						\
									\
	global_work_size[0] = VA_ITERATION_TOTAL( vap );


#define SETUP_FAST_BLOCKS_DBM_SBM	SETUP_FAST_BLOCKS_DBM_

#define SETUP_EQSP_BLOCKS_	SETUP_FAST_BLOCKS_
#define SETUP_EQSP_BLOCKS_SBM_	SETUP_FAST_BLOCKS_SBM_
// BUG?  do we need both???
#define SETUP_EQSP_BLOCKS_DBM_SBM_	SETUP_FAST_BLOCKS_DBM_SBM_
#define SETUP_EQSP_BLOCKS_DBM_SBM	SETUP_FAST_BLOCKS_DBM_SBM

// This looks like the non-bitmap version
#define SETUP_SLOW_BLOCKS_						\
									\
/*fprintf(stderr,"SETUP_SLOW_BLOCKS_:  %d %d %d\n",VA_LEN_X(vap),VA_LEN_Y(vap),VA_LEN_Z(vap));*/\
	global_work_size[0] = VA_ITERATION_TOTAL(vap);					\
	global_work_size[1] = 1;					\
	global_work_size[2] = 1;

#define SETUP_SLOW_BLOCKS_SBM_	SETUP_SLOW_BLOCKS_


// BUG - need to consider bit0
// BUG - also need to consider line rounding up???
// This is going to be a problem!?
// We need to have one thread per bitmap word...
// But we can have a fractional number of words per line, so we need to round up!
// Basically, we cannot use VA_ITERATION_TOTAL, we have to compute from the dimensions...
//
// We need to have one thread per bitmap word, but if we have an increment other than 1 then
// it won't just depend on the total number of bits?

#define SETUP_SLOW_BLOCKS_DBM_								\
											\
	global_work_size[0] = VA_ITERATION_TOTAL( vap );

#define SETUP_SLOW_BLOCKS_DBM_SBM	SETUP_SLOW_BLOCKS_DBM_


#define CALL_FAST_KERNEL(name,bitmap,typ,scalars,vectors)		\
									\
	/* BUG - check limit: CL_DEVICE_ADDRESS_BITS */			\
	FINISH_KERNEL_CALL(kernel,1)

#define CALL_FAST_KERNEL_1(name,bitmap,typ,scalars,vectors)		\
									\
	/* BUG - check limit: CL_DEVICE_ADDRESS_BITS */			\
	FINISH_KERNEL_CALL(kernel1,1)

#define CALL_FAST_KERNEL_2(name,bitmap,typ,scalars,vectors)		\
									\
	/* BUG - check limit: CL_DEVICE_ADDRESS_BITS */			\
	FINISH_KERNEL_CALL(kernel2,1)

// fast and eqsp only differ in args passed...
#define CALL_EQSP_KERNEL(name,bitmap,typ,scalars,vectors)		\
	CALL_FAST_KERNEL(name,bitmap,typ,scalars,vectors)

#define CALL_FAST_CONV_KERNEL(name,bitmap,typ,type)                     \
	FINISH_KERNEL_CALL(kernel,1)

#define CALL_EQSP_CONV_KERNEL(name,bitmap,typ,type)                     \
	CALL_FAST_CONV_KERNEL(name,bitmap,typ,type)

#define CALL_SLOW_CONV_KERNEL(name,bitmap,typ,type)                     \
	CALL_FAST_CONV_KERNEL(name,bitmap,typ,type)

#define CALL_GPU_FAST_NOCC_SETUP_FUNC(name)					\
									\
/*fprintf(stderr,"checking for nocc_setup kernel\n");*/\
	CHECK_FAST_KERNEL_1(name##_nocc_setup)			\
/*fprintf(stderr,"setting kernel args for nocc_setup\n");*/\
	SET_KERNEL_ARGS_FAST_NOCC_SETUP					\
/*fprintf(stderr,"setting up blocks for nocc_setup\n");*/\
	/*SETUP_FAST_BLOCKS_*/ /* uses VA_LENGTH */			\
	global_work_size[0] = len1;					\
/*fprintf(stderr,"calling fast setup kernel for %s, n_threads = %d\n",#name,len1);*/\
	CALL_FAST_KERNEL_1(name,,,,)


#define CALL_GPU_FAST_NOCC_HELPER_FUNC(name)					\
									\
/*fprintf(stderr,"checking for nocc_helper kernel\n");*/\
	CHECK_FAST_KERNEL_2(name##_nocc_helper)			\
/*fprintf(stderr,"setting kernel args for nocc_helper\n");*/\
	ki_idx=0;							\
	SET_KERNEL_ARGS_FAST_NOCC_HELPER					\
/*fprintf(stderr,"setting up blocks for nocc_helper\n");*/\
	global_work_size[0] = len1;					\
/*fprintf(stderr,"calling fast helper kernel for %s, n_threads = %d\n",#name,len1);*/\
	CALL_FAST_KERNEL_2(name,,,,)

#define CALL_GPU_FAST_PROJ_2V_SETUP_FUNC(name)					\
fprintf(stderr,"CALL_GPU_FAST_PROJ_2V_SETUP_FUNC(%s)\n",#name);\
	CHECK_FAST_KERNEL(name)					\
	SET_KERNEL_ARGS_FAST_PROJ_2V_SETUP						\
	CALL_FAST_KERNEL(name##_setup,,,,)

#define CALL_GPU_FAST_PROJ_2V_FUNC(name)					\
fprintf(stderr,"CALL_GPU_FAST_PROJ_2V_FUNC(%s)\n",#name);\
	CHECK_FAST_KERNEL(name)					\
	SET_KERNEL_ARGS_FAST_PROJ_2V						\
	CALL_FAST_KERNEL(name,,,,)

#define CALL_GPU_FAST_PROJ_3V_FUNC(name)					\
	CHECK_FAST_KERNEL(name)					\
	SET_KERNEL_ARGS_FAST_PROJ_3V						\
	CALL_FAST_KERNEL(name,,,,)

#define CALL_GPU_FAST_INDEX_SETUP_FUNC(name)					\
	CHECK_FAST_KERNEL_1(name)					\
	SET_KERNEL_ARGS_FAST_INDEX_SETUP					\
	CALL_FAST_KERNEL_1(name,,,,)

#define CALL_GPU_FAST_INDEX_HELPER_FUNC(name)					\
	CHECK_FAST_KERNEL_2(name)					\
	SET_KERNEL_ARGS_FAST_INDEX_HELPER					\
	CALL_FAST_KERNEL_2(name,,,,)

// Slow kernel - we set the sizes from the increments,
// but how do we know how many args we have???

// BUG we should be able to specify how many dims to use!

#define CALL_SLOW_KERNEL(name,bitmap,typ,scalars,vectors)		\
									\
/*show_vec_args(vap);*/\
	FINISH_KERNEL_CALL(kernel, /*3*/ 1 )

// Normally we don't want to wait
// So we can define this to be a nop
#define WAIT_FOR_KERNEL	clWaitForEvents(1,&event);
// define this to NULL if don't care
#define KERNEL_FINISH_EVENT	&event


#define FINISH_KERNEL_CALL(k,n_dims)					\
									\
	REPORT_KERNEL_ENQUEUE(n_dims)					\
	status = clEnqueueNDRangeKernel(				\
		OCLDEV_QUEUE( VA_PFDEV(vap) ),				\
		k[pd_idx],							\
		n_dims,	/* work_dim, 1-3 */				\
		NULL,							\
		global_work_size,					\
		/*local_work_size*/ NULL,				\
		0,	/* num_events_in_wait_list */			\
		NULL,	/* event_wait_list */				\
		KERNEL_FINISH_EVENT	/* event */					\
		);							\
	if( status != CL_SUCCESS )					\
		report_ocl_error(DEFAULT_QSP_ARG  status, "clEnqueueNDRangeKernel" );	\
	WAIT_FOR_KERNEL

#define DECLARE_FAST_VARS_3	// nop
#define DECLARE_EQSP_VARS_3	// nop
#define DECLARE_FAST_VARS_2	// nop
#define DECLARE_EQSP_VARS_2	// nop

#define DECLARE_PLATFORM_VARS		DECLARE_OCL_VARS
#define DECLARE_PLATFORM_VARS_2		DECLARE_OCL_VARS_2


#define DECLARE_PLATFORM_FAST_VARS	DECLARE_PLATFORM_VARS
#define DECLARE_PLATFORM_SLOW_VARS	DECLARE_PLATFORM_VARS

// OpenCL

#define SETUP_KERNEL_FAST_CALL(name,bitmap,typ,scalars,vectors)		\
									\
	CHECK_FAST_KERNEL(name)						\
	SET_KERNEL_ARGS_FAST_##bitmap##typ##scalars##vectors		\
	REPORT_KERNEL_CALL(name)					\
	SETUP_FAST_BLOCKS_##bitmap					\
	REPORT_FAST_ARGS_##bitmap##typ##scalars##vectors


#define SETUP_KERNEL_FAST_CALL_CONV(name,bitmap,typ,type)		\
									\
	CHECK_FAST_KERNEL(name)						\
	SET_KERNEL_ARGS_FAST_CONV(type)					\
	REPORT_KERNEL_CALL(name)					\
	SETUP_FAST_BLOCKS_##bitmap					\
	REPORT_FAST_ARGS_##bitmap##typ##2


#define SETUP_KERNEL_EQSP_CALL(name,bitmap,typ,scalars,vectors)		\
									\
	CHECK_EQSP_KERNEL(name)						\
	SET_KERNEL_ARGS_EQSP_##bitmap##typ##scalars##vectors		\
	REPORT_KERNEL_CALL(name)					\
	SETUP_EQSP_BLOCKS_##bitmap					\
	REPORT_EQSP_ARGS_##bitmap##typ##scalars##vectors


#define SETUP_KERNEL_EQSP_CALL_CONV(name,bitmap,typ,type)		\
									\
	CHECK_EQSP_KERNEL(name)						\
	SET_KERNEL_ARGS_EQSP_CONV(type)					\
	REPORT_KERNEL_CALL(name)					\
	SETUP_EQSP_BLOCKS_##bitmap					\
	REPORT_EQSP_ARGS_##bitmap##typ##2



#define SETUP_KERNEL_SLOW_CALL(name,bitmap,typ,scalars,vectors)		\
									\
	CHECK_SLOW_KERNEL(name)						\
/*fprintf(stderr,"%s:  setting slow args\n",#name);*/\
	SET_KERNEL_ARGS_SLOW_##bitmap##typ##scalars##vectors		\
	REPORT_KERNEL_CALL(name)					\
	SETUP_SLOW_BLOCKS_##bitmap					\
	REPORT_SLOW_ARGS_##bitmap##typ##scalars##vectors


#define SETUP_KERNEL_SLOW_CALL_CONV(name,bitmap,typ,type)		\
									\
	CHECK_SLOW_KERNEL(name)						\
	SET_KERNEL_ARGS_SLOW_CONV(type)					\
	REPORT_KERNEL_CALL(name)					\
	SETUP_SLOW_BLOCKS_##bitmap					\
	REPORT_SLOW_ARGS_##bitmap##typ##2

#define HELPER_FUNC_PRELUDE		DECLARE_OCL_VARS

// BUG Need to put things here for MM_NOCC etc!

#endif /* _OCL_HOST_CALL_DEFS_H_ */

