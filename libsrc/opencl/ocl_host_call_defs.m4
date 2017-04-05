
dnl BUG - merge this file!!! (but with what???)

dnl	define(`MORE_DEBUG',`x')	dnl	print extra debugging

include(`ocl_kern_args.m4')
include(`../../include/veclib/slow_len.m4')
include(`../../include/veclib/slow_incs.m4')
include(`../../include/veclib/eqsp_incs.m4')
include(`../../include/veclib/slow_vars.m4')

// this is 80 columns
//345678901234567890123456789012345678901234567890123456789012345678901234567890

define(`CLEAR_GPU_ERROR',`')		/* nop */
define(`CHECK_GPU_ERROR',`')		/* nop */

define(`BLOCK_VARS_DECLS',`')		// in MM_NOCC, may eliminate?
define(`SET_BLOCK_COUNT',`')			// nop?

define(`SET_MAX_THREADS_FROM_OBJ',`')	// nop

dnl PORT - insure_gpu_device ???

dnl GET_MAX_THREADS(dp)

define(`GET_MAX_THREADS',`

	/* insure_cuda_device( $1 ); */
	insure_ocl_device( $1 );
	max_threads_per_block = get_max_threads_per_block($1);
	// BUG OpenCL does not use max_threads_per_block!?
')

ifdef(`MORE_DEBUG',`

dnl REPORT_KERNEL_CALL(name)
dnl BUG - this seems to still use dim3!?

define(`REPORT_KERNEL_CALL',`fprintf(stderr,"Calling kernel %s\n","HOST_FAST_CALL_NAME($1)");')

define(`REPORT_THREAD_INFO',`

sprintf(DEFAULT_ERROR_STRING,"Blocks:  %d x %d x %d    Threads:  %d x %d x %d",
n_blocks.x,n_blocks.y,n_blocks.z,
n_threads_per_block.x,n_threads_per_block.y,n_threads_per_block.z);
NADVISE(DEFAULT_ERROR_STRING);
sprintf(DEFAULT_ERROR_STRING,"Length:  %d x %d x %d    Extra:  %d x %d x %d",
VA_LEN_X(vap),VA_LEN_Y(vap),VA_LEN_Z(vap),extra.x,extra.y,extra.z);
NADVISE(DEFAULT_ERROR_STRING);
')

define(`REPORT_THREAD_INFO2',`

sprintf(DEFAULT_ERROR_STRING,"Blocks:  %d x %d    Threads:  %d x %d",
n_blocks.x,n_blocks.y,n_threads_per_block.x,n_threads_per_block.y);
NADVISE(DEFAULT_ERROR_STRING);
sprintf(DEFAULT_ERROR_STRING,"Len1:  %ld   Len2:  %ld   Extra:  %d x %d",
len1,len2,extra.x,extra.y);
NADVISE(DEFAULT_ERROR_STRING);
')

dnl REPORT_KERNEL_ENQUEUE(_n_dims)

define(`REPORT_KERNEL_ENQUEUE',`

	if( verbose ){
		int i;
		fprintf(stderr,"finish_kernel_enqueue:  n_dims = %d\n",$1); 
		for (i=0;i<$1;i++)
			fprintf(stderr,"global_work_size[%d] = %ld\n",
				i,global_work_size[i]);
		fflush(stderr);
	}
')

',`	dnl else /* ! MORE_DEBUG */

define(`REPORT_THREAD_INFO',`')
define(`REPORT_THREAD_INFO2',`')
define(`REPORT_KERNEL_CALL',`')
define(`REPORT_KERNEL_ENQUEUE',`')

')	dnl endif ! MORE_DEBUG


define(`DEFAULT_YZ',`')	dnl	// for MM_NOCC, should be cleaned up...

dnl PF_GPU_FAST_CALL(name,bitmap,typ,scalars,vectors)

define(`PF_GPU_FAST_CALL',`fprintf(stderr,
"Need to implement PF_GPU_FAST_CALL (name = %s, bitmap = \"%s\", typ = \"%s\", scalars = \"%s\", vectors = \"%s\")\n", 
"$1","$2","$3","$4","$5");')


// Make this a nop if not waiting for kernels
define(`DECLARE_OCL_EVENT',`cl_event event;')

dnl BUG we need a different OCL kernel for every device...
dnl We could 
dnl BUG - the initialization here needs to be changed if we change MAX_OPENCL_DEVICES
dnl But - we are probably safe, because the compiler will set un-specified
dnl elements to 0...

define(`DECLARE_OCL_VARS',`
	static cl_kernel kernel[MAX_OPENCL_DEVICES] = {NULL,NULL,NULL,NULL};
	DECLARE_OCL_COMMON_VARS
')

define(`DECLARE_OCL_COMMON_VARS',`

	static cl_program program = NULL;
	cl_int status;
	DECLARE_OCL_EVENT
	int ki_idx=0;
	int pd_idx; /* need to set! */
	const char *ksrc;
	/* define the global size and local size
	 * (grid size and block size in CUDA) */
	size_t global_work_size[3] = {1, 1, 1};
	/* size_t local_work_size[3]  = {0, 0, 0}; */
')

// two different kernels used in one call (e.g. vmaxg - nocc_setup, nocc_helper

define(`DECLARE_OCL_VARS_2',`

	static cl_kernel kernel1[MAX_OPENCL_DEVICES] = {NULL,NULL,NULL,NULL};
	static cl_kernel kernel2[MAX_OPENCL_DEVICES] = {NULL,NULL,NULL,NULL};
	DECLARE_OCL_COMMON_VARS
')

define(`CHECK_NOSPEED_KERNEL',`CHECK_KERNEL($1,,GPU_CALL_NAME($1))')
define(`CHECK_NOSPEED_KERNEL_1',`CHECK_KERNEL_1($1,,GPU_CALL_NAME($1))')
define(`CHECK_FAST_KERNEL_1',`CHECK_KERNEL_1($1,fast,GPU_FAST_CALL_NAME($1))')
define(`CHECK_NOSPEED_KERNEL_2',`CHECK_KERNEL_2($1,,GPU_CALL_NAME($1))')
define(`CHECK_FAST_KERNEL_2',`CHECK_KERNEL_2($1,fast,GPU_FAST_CALL_NAME($1))')
define(`CHECK_FAST_KERNEL',`CHECK_KERNEL($1,fast,GPU_FAST_CALL_NAME($1))')
define(`CHECK_EQSP_KERNEL',`CHECK_KERNEL($1,eqsp,GPU_EQSP_CALL_NAME($1))')
define(`CHECK_SLOW_KERNEL',`CHECK_KERNEL($1,slow,GPU_SLOW_CALL_NAME($1))')

dnl CHECK_KERNAL(name,ktyp,kname)

define(`CHECK_KERNEL',`_CHECK_KERNEL(kernel,$1,$2,$3)')
define(`CHECK_KERNEL_1',`_CHECK_KERNEL(kernel1,$1,$2,$3)')
define(`CHECK_KERNEL_2',`_CHECK_KERNEL(kernel2,$1,$2,$3)')

dnl _CHECK_KERNEL(k,name,ktyp,kname)

define(`_CHECK_KERNEL',`
	/* _check_kernel $1 $2 $3 $4 */
	pd_idx = OCLDEV_IDX(VA_PFDEV(vap));
dnl fprintf(stderr,"_check_kernel $2:  pd_idx = %d\n",pd_idx);
	if( $1[pd_idx] == NULL ){	/* one-time initialization */
		ksrc = KERN_SOURCE_NAME($2,$3);
dnl fprintf(stderr,"_check_kernel $2:  creating kernel\n");
		program = ocl_create_program(ksrc,VA_PFDEV(vap));
		if( program == NULL ) 
			NERROR1("program creation failure!?");

		$1[pd_idx] = ocl_create_kernel(program, "$4", VA_PFDEV(vap));
		if( $1[pd_idx] == NULL ){ 
			NADVISE("Source code of failed program:");
			NADVISE(ksrc);
			NERROR1("kernel creation failure!?");
		}
	}
')


define(`SETUP_FAST_BLOCKS_',`/*sfb*/global_work_size[0] = VA_LENGTH(vap);/*sfb*/')
define(`SETUP_FAST_BLOCKS_SBM_',`SETUP_FAST_BLOCKS_')

dnl BUG - the number of words we need to process depends on bit0 !?

define(`SETUP_FAST_BLOCKS_DBM_',`global_work_size[0] = VA_LENGTH( vap )/BITS_PER_BITMAP_WORD;')

define(`SETUP_EQSP_BLOCKS_DBM_',`global_work_size[0] = VA_ITERATION_TOTAL( vap );')

dnl	BUG	don't we need to consider source bitmaps, especially for slow ops?
define(`SETUP_FAST_BLOCKS_DBM_SBM',`SETUP_FAST_BLOCKS_DBM_')
define(`SETUP_FAST_BLOCKS_DBM_2SBM',`SETUP_FAST_BLOCKS_DBM_')
define(`SETUP_FAST_BLOCKS_DBM_1SBM',`SETUP_FAST_BLOCKS_DBM_')

define(`SETUP_EQSP_BLOCKS_',`SETUP_FAST_BLOCKS_')
define(`SETUP_EQSP_BLOCKS_SBM_',`SETUP_FAST_BLOCKS_SBM_')
dnl BUG?  do we need both???
define(`SETUP_EQSP_BLOCKS_DBM_SBM_',`SETUP_FAST_BLOCKS_DBM_SBM_')
define(`SETUP_EQSP_BLOCKS_DBM_SBM',`SETUP_FAST_BLOCKS_DBM_SBM')
define(`SETUP_EQSP_BLOCKS_DBM_2SBM',`SETUP_FAST_BLOCKS_DBM_2SBM')
define(`SETUP_EQSP_BLOCKS_DBM_1SBM',`SETUP_FAST_BLOCKS_DBM_1SBM')

dnl SETUP_SLOW_BLOCKS(bitmap)
define(`SETUP_SLOW_BLOCKS',`/*ssb $1*/SETUP_SLOW_BLOCKS_$1/*ssb $1*/')
define(`SETUP_EQSP_BLOCKS',`/*seb $1*/SETUP_EQSP_BLOCKS_$1/*seb$1*/')
define(`SETUP_FAST_BLOCKS',`/*sfb $1*/SETUP_FAST_BLOCKS_$1/*sfb $1*/')

dnl This looks like the non-bitmap version
define(`SETUP_SLOW_BLOCKS_',`

	global_work_size[0] = VA_ITERATION_TOTAL(vap);
	global_work_size[1] = 1;
	global_work_size[2] = 1;
	')

define(`SETUP_SLOW_BLOCKS_SBM_',`SETUP_SLOW_BLOCKS_')


define(`SETUP_SLOW_BLOCKS_DBM_',`global_work_size[0] = VA_ITERATION_TOTAL( vap );')

define(`SETUP_SLOW_BLOCKS_DBM_SBM',`SETUP_SLOW_BLOCKS_DBM_')
define(`SETUP_SLOW_BLOCKS_DBM_2SBM',`SETUP_SLOW_BLOCKS_DBM_')
define(`SETUP_SLOW_BLOCKS_DBM_1SBM',`SETUP_SLOW_BLOCKS_DBM_')


dnl CALL_FAST_KERNEL(name,bitmap,typ,scalars,vectors)

dnl	/* BUG - check limit: CL_DEVICE_ADDRESS_BITS */
define(`CALL_FAST_KERNEL',`FINISH_KERNEL_CALL(kernel,1)')
define(`CALL_FAST_KERNEL_1',`FINISH_KERNEL_CALL(kernel1,1)')
define(`CALL_FAST_KERNEL_2',`FINISH_KERNEL_CALL(kernel2,1)')

dnl fast and eqsp only differ in args passed...
define(`CALL_EQSP_KERNEL',`CALL_FAST_KERNEL')

define(`CALL_FAST_CONV_KERNEL',`FINISH_KERNEL_CALL(kernel,1)')
define(`CALL_EQSP_CONV_KERNEL',`CALL_FAST_CONV_KERNEL')
define(`CALL_SLOW_CONV_KERNEL',`CALL_FAST_CONV_KERNEL')

dnl CALL_GPU_FAST_NOCC_SETUP_FUNC(name)

define(`CALL_GPU_FAST_NOCC_SETUP_FUNC',`

	CHECK_FAST_KERNEL_1($1`_setup')
	/* setting kernel args */
	SET_KERNEL_ARGS_FAST_NOCC_SETUP
	/* setting global work size */
	global_work_size[0] = len1;
	/* calling fast kernel */
	CALL_FAST_KERNEL_1($1,,,,)
')


dnl CALL_GPU_FAST_NOCC_HELPER_FUNC(name)

define(`CALL_GPU_FAST_NOCC_HELPER_FUNC',`

	CHECK_FAST_KERNEL_2($1`_nocc_helper')
	SET_KERNEL_ARGS_FAST_NOCC_HELPER
	global_work_size[0] = len1;
	CALL_FAST_KERNEL_2($1,,,,)
')

dnl CALL_GPU_FAST_PROJ_2V_SETUP_FUNC(name)

define(`CALL_GPU_FAST_PROJ_2V_SETUP_FUNC',`
	CHECK_FAST_KERNEL_1($1`_setup')
	SET_KERNEL_ARGS_FAST_PROJ_2V_SETUP
	global_work_size[0] = len1;
	CALL_FAST_KERNEL_1($1`_setup',,,,)
')

define(`CALL_GPU_FAST_PROJ_2V_HELPER_FUNC',`
	CHECK_FAST_KERNEL_2($1`_helper')
	SET_KERNEL_ARGS_FAST_PROJ_2V_HELPER
	global_work_size[0] = len1;
	CALL_FAST_KERNEL_2($1`_helper',,,,)
')

define(`CALL_GPU_FAST_PROJ_3V_SETUP_FUNC',`
	CHECK_FAST_KERNEL_1($1`_setup')
	SET_KERNEL_ARGS_FAST_PROJ_3V_SETUP
	/* BUG?  set global_work_size ??? */
	CALL_FAST_KERNEL_1($1`_setup',,,,)
')

define(`CALL_GPU_FAST_PROJ_3V_HELPER_FUNC',`
	CHECK_FAST_KERNEL_2($1`_helper')
dnl fprintf(stderr,"setting helper kernel args...\n");
	SET_KERNEL_ARGS_FAST_PROJ_3V_HELPER
dnl fprintf(stderr,"DONE setting helper kernel args...\n");
	/* BUG?  set global_work_size ??? */
	CALL_FAST_KERNEL_2($1`_helper',,,,)
')

define(`CALL_GPU_FAST_INDEX_SETUP_FUNC',`
	CHECK_FAST_KERNEL_1($1)
	SET_KERNEL_ARGS_FAST_INDEX_SETUP
	/* BUG?  set global_work_size ??? */
	CALL_FAST_KERNEL_1($1,,,,)
')

define(`CALL_GPU_FAST_INDEX_HELPER_FUNC',`
	CHECK_FAST_KERNEL_2($1)
	SET_KERNEL_ARGS_FAST_INDEX_HELPER
	/* BUG?  set global_work_size ??? */
	CALL_FAST_KERNEL_2($1,,,,)
')

// Slow kernel - we set the sizes from the increments,
// but how do we know how many args we have???

// BUG we should be able to specify how many dims to use!

define(`CALL_SLOW_KERNEL',`

/*show_vec_args(vap);*/
	FINISH_KERNEL_CALL(kernel, /*3*/ 1 )
')

dnl Normally we don't want to wait
dnl So we can define this to be a nop
define(`WAIT_FOR_KERNEL',`clWaitForEvents(1,&event);')
dnl define this to NULL if don't care
define(`KERNEL_FINISH_EVENT',`&event')


dnl  FINISH_KERNEL_CALL(k,n_dims)

define(`FINISH_KERNEL_CALL',`

	REPORT_KERNEL_ENQUEUE($2)
	status = clEnqueueNDRangeKernel(
		OCLDEV_QUEUE( VA_PFDEV(vap) ),
		$1[pd_idx],
		$2,	/* work_dim, 1-3 */
		NULL,
		global_work_size,
		/*local_work_size*/ NULL,
		0,	/* num_events_in_wait_list */
		NULL,	/* event_wait_list */
		KERNEL_FINISH_EVENT	/* event */
		);
	if( status != CL_SUCCESS )
		report_ocl_error(DEFAULT_QSP_ARG  status, "clEnqueueNDRangeKernel" );
	WAIT_FOR_KERNEL
')

define(`DECLARE_FAST_VARS_3',`')
define(`DECLARE_EQSP_VARS_3',`')
define(`DECLARE_FAST_VARS_2',`')
define(`DECLARE_EQSP_VARS_2',`')

define(`DECLARE_PLATFORM_VARS',`DECLARE_OCL_VARS')
define(`DECLARE_PLATFORM_VARS_2',`DECLARE_OCL_VARS_2')


define(`DECLARE_PLATFORM_FAST_VARS',`DECLARE_PLATFORM_VARS')
define(`DECLARE_PLATFORM_SLOW_VARS',`DECLARE_PLATFORM_VARS')

// OpenCL

dnl SETUP_KERNEL_FAST_CALL(name,bitmap,typ,scalars,vectors)
define(`SETUP_KERNEL_FAST_CALL',`

	CHECK_FAST_KERNEL($1)
	SET_KERNEL_ARGS_FAST($2,$3,$4,$5)
	REPORT_KERNEL_CALL($1)
	dnl `SETUP_FAST_BLOCKS_'$2
	SETUP_FAST_BLOCKS($2)
	REPORT_FAST_ARGS($2,$3,$4,$5)
')


dnl SETUP_KERNEL_FAST_CALL_CONV( , , , dest_type )
define(`SETUP_KERNEL_FAST_CALL_CONV',`

	/* setup_kernel_fast_call_conv "$1"  "$2"  "$3"  "$4" */
	CHECK_FAST_KERNEL($1)
	SET_KERNEL_ARGS_FAST_CONV($4)
	REPORT_KERNEL_CALL($1)
	dnl `SETUP_FAST_BLOCKS_'$2
	SETUP_FAST_BLOCKS($2)
	REPORT_FAST_ARGS($2,$3,`',`2')
')


define(`SETUP_KERNEL_EQSP_CALL',`

	CHECK_EQSP_KERNEL($1)
	SET_KERNEL_ARGS_EQSP($2,$3,$4,$5)
	REPORT_KERNEL_CALL($1)
	SETUP_EQSP_BLOCKS($2)
	REPORT_EQSP_ARGS($2,$3,$4,$5)
')


define(`SETUP_KERNEL_EQSP_CALL_CONV',`

	CHECK_EQSP_KERNEL($1)
	SET_KERNEL_ARGS_EQSP_CONV($4)
	REPORT_KERNEL_CALL($1)
	SETUP_EQSP_BLOCKS($2)
	REPORT_EQSP_ARGS($2,$3,`',`2')
')


dnl SETUP_KERNEL_SLOW_CALL(name,bitmap,typ,scalars,vectors)

define(`SETUP_KERNEL_SLOW_CALL',`

	CHECK_SLOW_KERNEL($1)
	SET_KERNEL_ARGS_SLOW($2,$3,$4,$5)
	REPORT_KERNEL_CALL($1)
	dnl `SETUP_SLOW_BLOCKS_'$2
	SETUP_SLOW_BLOCKS($2)
	dnl `REPORT_SLOW_ARGS_'$2$3$4$5
	REPORT_SLOW_ARGS($2,$3,$4,$5)
')


dnl  SETUP_KERNEL_SLOW_CALL_CONV(name,bitmap,typ,type)

define(`SETUP_KERNEL_SLOW_CALL_CONV',`

	CHECK_SLOW_KERNEL($1)
	SET_KERNEL_ARGS_SLOW_CONV($4)
	REPORT_KERNEL_CALL($1)
	dnl `SETUP_SLOW_BLOCKS_'$2
	SETUP_SLOW_BLOCKS($2)
	dnl `REPORT_SLOW_ARGS_'$2$3`2'
	REPORT_SLOW_ARGS($2,$3,`',2)
')

define(`HELPER_FUNC_PRELUDE',`DECLARE_OCL_VARS')

dnl BUG Need to put things here for MM_NOCC etc!

