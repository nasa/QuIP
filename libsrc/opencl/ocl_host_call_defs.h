
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




// PORT - insure_gpu_device ???

#define GET_MAX_THREADS( dp )						\
									\
	/* insure_cuda_device( dp ); */					\
	insure_ocl_device( dp );					\
	max_threads_per_block = get_max_threads_per_block(dp);
	// BUG OpenCL doesn't use max_threads_per_block!?

#define SETUP_BLOCKS_XYZ_	SETUP_BLOCKS_XYZ
#define SETUP_BLOCKS_XYZ_SBM_	SETUP_BLOCKS_XYZ

#define SETUP_BLOCKS_XYZ					\
								\
	SETUP_BLOCKS_X(VA_LEN_X(vap))					\
	SETUP_BLOCKS_Y						\
	SETUP_BLOCKS_Z


/* If we have a destination bitmap, we handle all the bits in one word
 * in a single thread.
 *
 * BUG - here we ignore bit0 ???
 *
 * MAX_THREADS_PER_ROW is 32...  for a 512 pixel wide image, the nmber
 * of bitmap words is either 8 (64 bit words) or 16 (32 bit words).
 * So we need to 
 */


#define SETUP_BLOCKS_XYZ_DBM_					\
								\
	SETUP_BLOCKS_X( N_BITMAP_WORDS(VA_LEN_X(vap)) )			\
	SETUP_BLOCKS_Y						\
	SETUP_BLOCKS_Z


#define SETUP_BLOCKS_X(w)					\
								\
/*sprintf(DEFAULT_ERROR_STRING,"SETUP_BLOCKS_X:  len = %d",w);\
NADVISE(DEFAULT_ERROR_STRING);*/\
	if( (w) < MAX_THREADS_PER_ROW ) {			\
		n_threads_per_block.x = w;			\
		n_blocks.x = 1;					\
		extra.x = 0;					\
	} else {						\
		n_blocks.x = (w) / MAX_THREADS_PER_ROW;		\
		n_threads_per_block.x = MAX_THREADS_PER_ROW;	\
		extra.x = (w) % MAX_THREADS_PER_ROW;		\
	}


#define SETUP_BLOCKS_Y						\
								\
	n_threads_per_block.y = max_threads_per_block /		\
					n_threads_per_block.x;	\
	if( VA_LEN_Y(vap) < n_threads_per_block.y ){			\
		n_threads_per_block.y = VA_LEN_Y(vap);			\
		n_blocks.y = 1;					\
		extra.y = 0;					\
	} else {						\
		n_blocks.y = VA_LEN_Y(vap) / n_threads_per_block.y;	\
		extra.y = VA_LEN_Y(vap) % n_threads_per_block.y;	\
	}							\
	if( extra.x > 0 ) n_blocks.x++;				\
	if( extra.y > 0 ) n_blocks.y++;

#define SETUP_BLOCKS_Z						\
								\
	n_threads_per_block.z = max_threads_per_block /		\
		(n_threads_per_block.x*n_threads_per_block.y);	\
	if( VA_LEN_Z(vap) < n_threads_per_block.z ){			\
		n_threads_per_block.z = VA_LEN_Z(vap);			\
		n_blocks.z = 1;					\
		extra.z = 0;					\
	} else {						\
		n_blocks.z = VA_LEN_Z(vap) / n_threads_per_block.z;	\
		extra.z = VA_LEN_Z(vap) % n_threads_per_block.z;	\
	}							\
	if( extra.z > 0 ) n_blocks.z++;

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

#define REPORT_INCS(incs)					\
	fprintf(stderr,"%s:  %d %d %d\n",#incs,incs.x,incs.y,incs.z);

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
	static cl_program program = NULL;				\
	static cl_kernel kernel[MAX_OPENCL_DEVICES] = {NULL,NULL,NULL,NULL};	\
	cl_int status;							\
	DECLARE_OCL_EVENT						\
	int ki_idx=0;							\
	int pd_idx; /* need to set! */					\
	const char *ksrc;						\
	/* define the global size and local size			\
	 * (grid size and block size in CUDA) */			\
	size_t global_work_size[3] = {1, 1, 1};				\
	/* size_t local_work_size[3]  = {0, 0, 0}; */

#define CHECK_NOSPEED_KERNEL(name)						\
	CHECK_KERNEL(name,,GPU_CALL_NAME(name))

#define CHECK_FAST_KERNEL(name)						\
	CHECK_KERNEL(name,fast,GPU_FAST_CALL_NAME(name))

#define CHECK_EQSP_KERNEL(name)						\
	CHECK_KERNEL(name,eqsp,GPU_EQSP_CALL_NAME(name))

#define CHECK_SLOW_KERNEL(name)						\
	CHECK_KERNEL(name,slow,GPU_SLOW_CALL_NAME(name))

#define CHECK_KERNEL(name,ktyp,kname)					\
	_CHECK_KERNEL(name,ktyp,kname)

#define _CHECK_KERNEL(name,ktyp,kname)					\
	pd_idx = OCLDEV_IDX(VA_PFDEV(vap));				\
	if( kernel[pd_idx] == NULL ){	/* one-time initialization */	\
		ksrc = KERN_SOURCE_NAME(name,ktyp);			\
		program = ocl_create_program(ksrc,VA_PFDEV(vap));	\
		if( program == NULL ) 					\
			NERROR1("program creation failure!?");		\
									\
		kernel[pd_idx] = ocl_create_kernel(program, #kname, VA_PFDEV(vap));\
		if( kernel[pd_idx] == NULL ){ 					\
			NADVISE("Source code of failed program:");	\
			NADVISE(ksrc);					\
			NERROR1("kernel creation failure!?");		\
		}							\
	}


// fast and eqsp only differ in args passed...
#define CALL_EQSP_KERNEL(name,bitmap,typ,scalars,vectors)		\
	CALL_FAST_KERNEL(name,bitmap,typ,scalars,vectors)

#define SETUP_FAST_BLOCKS_						\
									\
/*fprintf(stderr,"SETUP_FAST_BLOCKS_:  %d\n",VA_LENGTH(vap));*/\
	global_work_size[0] = VA_LENGTH(vap);

#define SETUP_FAST_BLOCKS_SBM_	SETUP_FAST_BLOCKS_

// BUG - the number of words we need to process depends on bit0 !?

#define SETUP_FAST_BLOCKS_DBM_						\
									\
/*fprintf(stderr,"SETUP_FAST_BLOCKS_:  %ld\n",N_BITMAP_WORDS(VA_LENGTH(vap)));*/\
	global_work_size[0] = N_BITMAP_WORDS(VA_LENGTH(vap));

#define SETUP_FAST_BLOCKS_DBM_SBM	SETUP_FAST_BLOCKS_DBM_

#define SETUP_EQSP_BLOCKS_	SETUP_FAST_BLOCKS_
#define SETUP_EQSP_BLOCKS_SBM_	SETUP_FAST_BLOCKS_SBM_
#define SETUP_EQSP_BLOCKS_DBM_	SETUP_FAST_BLOCKS_DBM_
// BUG?  do we need both???
#define SETUP_EQSP_BLOCKS_DBM_SBM_	SETUP_FAST_BLOCKS_DBM_SBM_
#define SETUP_EQSP_BLOCKS_DBM_SBM	SETUP_FAST_BLOCKS_DBM_SBM

#define SETUP_SLOW_BLOCKS_						\
									\
/*fprintf(stderr,"SETUP_SLOW_BLOCKS_:  %d %d %d\n",VA_LEN_X(vap),VA_LEN_Y(vap),VA_LEN_Z(vap));*/\
	global_work_size[0] = VA_LEN_X(vap);					\
	global_work_size[1] = VA_LEN_Y(vap);					\
	global_work_size[2] = VA_LEN_Z(vap);

#define SETUP_SLOW_BLOCKS_SBM_	SETUP_SLOW_BLOCKS_


// BUG - need to consider bit0
#define SETUP_SLOW_BLOCKS_DBM_						\
									\
/*fprintf(stderr,"SETUP_SLOW_BLOCKS_DBM_:  %ld %d %d\n",N_BITMAP_WORDS(VA_LEN_X(vap)),VA_LEN_Y(vap),VA_LEN_Z(vap));*/\
	global_work_size[0] = N_BITMAP_WORDS(VA_LEN_X(vap));			\
	global_work_size[1] = VA_LEN_Y(vap);					\
	global_work_size[2] = VA_LEN_Z(vap);

#define SETUP_SLOW_BLOCKS_DBM_SBM	SETUP_SLOW_BLOCKS_DBM_


#define CALL_FAST_KERNEL(name,bitmap,typ,scalars,vectors)		\
									\
	/* BUG - check limit: CL_DEVICE_ADDRESS_BITS */			\
/*show_vec_args(vap);*/\
/*fprintf(stderr,"global_work_size = %ld %ld %ld\n",*/\
/*global_work_size[0],global_work_size[1],global_work_size[2]);*/\
	FINISH_KERNEL_CALL(1)

#define CALL_FAST_CONV_KERNEL(name,bitmap,typ,type)                     \
	FINISH_KERNEL_CALL(1)

#define CALL_EQSP_CONV_KERNEL(name,bitmap,typ,type)                     \
	CALL_FAST_CONV_KERNEL(name,bitmap,typ,type)

#define CALL_SLOW_CONV_KERNEL(name,bitmap,typ,type)                     \
	CALL_FAST_CONV_KERNEL(name,bitmap,typ,type)

#define CALL_GPU_NOCC_SETUP_FUNC(name)					\
									\
	CHECK_NOSPEED_KERNEL(name##_nocc_setup)				\
	SET_KERNEL_ARGS_NOCC_SETUP					\
	SETUP_FAST_BLOCKS_						\
	CALL_FAST_KERNEL(name,,,,)
	/*GPU_CALL_NAME(name##_nocc_setup)(dst_values, dst_counts, src_values, indices, len1, len2); */


#define CALL_GPU_NOCC_HELPER_FUNC(name)					\
									\
	CHECK_NOSPEED_KERNEL(name##_nocc_helper)			\
	SET_KERNEL_ARGS_NOCC_HELPER					\
	SETUP_FAST_BLOCKS_						\
	CALL_FAST_KERNEL(name,,,,)
	/*(GPU_CALL_NAME(name##_nocc_helper) (dst_values, dst_counts, src_values, src_counts, indices, len1, len2, stride); */

#define CALL_GPU_PROJ_2V_FUNC(name)					\
fprintf(stderr,"CALL_GPU_PROJ_2V_FUNC(%s)\n",#name);\
	CHECK_NOSPEED_KERNEL(name)					\
	SET_KERNEL_ARGS_PROJ_2V						\
	CALL_FAST_KERNEL(name,,,,)
	/* GPU_CALL_NAME(name)arg1 , s1 , len1 , len2 ); */

#define CALL_GPU_PROJ_3V_FUNC(name)					\
	CHECK_NOSPEED_KERNEL(name)					\
	SET_KERNEL_ARGS_PROJ_3V						\
	CALL_FAST_KERNEL(name,,,,)
	/* GPU_CALL_NAME(name)arg1 , s1 , len1 , len2 ); */

#define CALL_GPU_INDEX_SETUP_FUNC(name)					\
	CHECK_NOSPEED_KERNEL(name)					\
	SET_KERNEL_ARGS_INDEX_SETUP					\
	CALL_FAST_KERNEL(name,,,,)

#define CALL_GPU_INDEX_HELPER_FUNC(name)					\
	CHECK_NOSPEED_KERNEL(name)					\
	SET_KERNEL_ARGS_INDEX_HELPER					\
	CALL_FAST_KERNEL(name,,,,)

// Slow kernel - we set the sizes from the increments,
// but how do we know how many args we have???

// BUG we should be able to specify how many dims to use!

#define CALL_SLOW_KERNEL(name,bitmap,typ,scalars,vectors)		\
									\
/*show_vec_args(vap);\
fprintf(stderr,"global_work_size = %ld %ld %ld\n",\
global_work_size[0],global_work_size[1],global_work_size[2]);*/\
	FINISH_KERNEL_CALL(3)

// Normally we don't want to wait
// So we can define this to be a nop
#define WAIT_FOR_KERNEL	clWaitForEvents(1,&event);
// define this to NULL if don't care
#define KERNEL_FINISH_EVENT	&event


#define FINISH_KERNEL_CALL(n_dims)					\
									\
	REPORT_KERNEL_ENQUEUE(n_dims)					\
	status = clEnqueueNDRangeKernel(				\
		OCLDEV_QUEUE( VA_PFDEV(vap) ),				\
		kernel[pd_idx],							\
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

