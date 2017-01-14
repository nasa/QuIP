#include "veclib/slow_len.h"
#include "veclib/slow_incs.h"
#include "veclib/eqsp_incs.h"
#include "veclib/slow_vars.h"

// cuda uses dim3...

#include "veclib/dim3.h"


#if CUDA_VERSION >= 5000
// CUDA 5
//#define CUDA_ERROR_CHECK(string)	getLastCudaError(string);
#else
// CUDA 4
//#define CUDA_ERROR_CHECK(string)	cutilCheckMsg(string);
#endif

// cudaGetLastError not available before 5.0 ...
#define CHECK_CUDA_ERROR(whence)					\
	e = cudaGetLastError();						\
	if( e != cudaSuccess ){						\
		describe_cuda_driver_error(#whence,e);			\
	}


#define CLEAR_CUDA_ERROR(name)	_CLEAR_CUDA_ERROR(name)

#define _CLEAR_CUDA_ERROR(name)					\
	e = cudaGetLastError();						\
	if( e != cudaSuccess ){						\
		describe_cuda_driver_error(#name,e);			\
	}

#define CLEAR_CUDA_ERROR2(name)					\
	e = cudaGetLastError();						\
	if( e != cudaSuccess ){						\
		describe_cuda_driver_error2(STRINGIFY(HOST_CALL_NAME(name)),\
			STRINGIFY(GPU_CALL_NAME(name)),e);		\
	}

#define HELPER_FUNC_PRELUDE				\
	DECLARE_PLATFORM_VARS

// What is the point of this - where does it occur?
#define SET_MAX_THREADS_FROM_OBJ(dp)					\
	/*max_threads_per_block = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(OBJ_PFDEV(dp));*/


#define BLOCK_VARS_DECLS						\
									\
	DIM3 n_blocks, n_threads_per_block;				\
	DIM3 extra;

#define DECLARE_PLATFORM_VARS_2						\
	DECLARE_PLATFORM_VARS

#define DECLARE_PLATFORM_VARS						\
	cudaError_t e;							\
	BLOCK_VARS_DECLS						\

#define DECLARE_PLATFORM_FAST_VARS					\
									\
	DECLARE_PLATFORM_VARS						\
	/*dimension_t len;*/


#define DECLARE_PLATFORM_SLOW_VARS					\
									\
	DECLARE_PLATFORM_VARS						\
	/*DIM3 xyz_len;*/


#define SETUP_KERNEL_FAST_CALL(name,bitmap,typ,scalars,vectors)	// nop
#define SETUP_KERNEL_FAST_CALL_CONV(name,bitmap,typ,type)	// nop

#define SETUP_KERNEL_EQSP_CALL(name,bitmap,typ,scalars,vectors)	// nop
#define SETUP_KERNEL_EQSP_CALL_CONV(name,bitmap,typ,type)	// nop

#define SETUP_KERNEL_SLOW_CALL(name,bitmap,typ,scalars,vectors)	// nop
#define SETUP_KERNEL_SLOW_CALL_CONV(name,bitmap,typ,type)	// nop

#define CALL_FAST_KERNEL(name,bitmap,typ,scalars,vectors)		\
									\
/*fprintf(stderr,"CALL_FAST_KERNEL %s, bitmap='%s', scalars='%s', vectors='%s'\n",\
#name,#bitmap,#scalars,#vectors);*/\
	CLEAR_CUDA_ERROR(GPU_FAST_CALL_NAME(name))			\
/*fprintf(stderr,"VA_LENGTH = %d\n",VA_LENGTH(vap));*/\
	SETUP_BLOCKS_X(VA_LENGTH(vap))					\
	GET_THREADS_PER_##bitmap##BLOCK(VA_PFDEV(vap),VA_LENGTH(vap))	\
/*REPORT_FAST_ARGS_##bitmap##typ##scalars##vectors*/			\
	if (extra.x != 0) {						\
		n_blocks.x++;						\
		REPORT_THREAD_INFO					\
		GPU_FLEN_CALL_NAME(name)<<< NN_GPU >>> 			\
			(KERN_ARGS_FLEN_##bitmap##typ##scalars##vectors );\
    		CHECK_CUDA_ERROR(flen name:  kernel launch failure);	\
	} else {							\
		REPORT_THREAD_INFO					\
		GPU_FAST_CALL_NAME(name)<<< NN_GPU >>>			\
			(KERN_ARGS_FAST_##bitmap##typ##scalars##vectors );\
		/* BUG?  should we put this check everywhere? */	\
    		CHECK_CUDA_ERROR(fast name:  kernel launch failure);	\
	}								\



#define CALL_FAST_CONV_KERNEL(name,bitmap,typ,type)			\
									\
	CLEAR_CUDA_ERROR(GPU_FAST_CALL_NAME(name))			\
	SETUP_BLOCKS_X(VA_LENGTH(vap))					\
	GET_THREADS_PER_##bitmap##BLOCK(VA_PFDEV(vap),VA_LENGTH(vap))	\
/*REPORT_FAST_ARGS_##bitmap##typ##2*/					\
	if (extra.x != 0) {						\
		n_blocks.x++;						\
		REPORT_THREAD_INFO					\
		GPU_FLEN_CALL_NAME(name)<<< NN_GPU >>> 			\
			(KERN_ARGS_FLEN_CONV(type) );			\
    		CHECK_CUDA_ERROR(flen name:  kernel launch failure);	\
	} else {							\
		REPORT_THREAD_INFO					\
		GPU_FAST_CALL_NAME(name)<<< NN_GPU >>>			\
			(KERN_ARGS_FAST_CONV(type));			\
		/* BUG?  should we put this check everywhere? */	\
    		CHECK_CUDA_ERROR(fast name:  kernel launch failure);	\
	}								\



#define CALL_EQSP_KERNEL(name,bitmap,typ,scalars,vectors)		\
									\
	CLEAR_CUDA_ERROR(GPU_EQSP_CALL_NAME(name))			\
	GET_THREADS_PER_##bitmap##BLOCK(VA_PFDEV(vap),VA_LENGTH(vap))	\
	SETUP_BLOCKS_X(VA_LENGTH(vap))					\
	/* BUG? shoudl this be commented out??? */			\
	/*SETUP_SIMPLE_INCS_##vectors*/					\
	/*GET_EQSP_INCR_##bitmap##vectors*/				\
	if (extra.x != 0) {						\
		n_blocks.x++;						\
		REPORT_THREAD_INFO					\
		GPU_ELEN_CALL_NAME(name)<<< NN_GPU >>> 		\
			(KERN_ARGS_ELEN_##bitmap##typ##scalars##vectors );\
    		CHECK_CUDA_ERROR(elen name:  kernel launch failure);		\
	} else {							\
		REPORT_THREAD_INFO					\
		GPU_EQSP_CALL_NAME(name)<<< NN_GPU >>>			\
			(KERN_ARGS_EQSP_##bitmap##typ##scalars##vectors );\
		/* BUG?  should we put this check everywhere? */	\
    		CHECK_CUDA_ERROR(eqsp name:  kernel launch failure);		\
	}								\


#define CALL_EQSP_CONV_KERNEL(name,bitmap,typ,type)			\
									\
	CLEAR_CUDA_ERROR(GPU_EQSP_CALL_NAME(name))			\
	GET_THREADS_PER_##bitmap##BLOCK(VA_PFDEV(vap),VA_LENGTH(vap))	\
	SETUP_BLOCKS_X(VA_LENGTH(vap))					\
	/* BUG? shoudl this be commented out??? */			\
	/*SETUP_SIMPLE_INCS_##vectors*/					\
	/*GET_EQSP_INCR_##bitmap##vectors*/				\
	if (extra.x != 0) {						\
		n_blocks.x++;						\
		REPORT_THREAD_INFO					\
		GPU_ELEN_CALL_NAME(name)<<< NN_GPU >>> 			\
			( KERN_ARGS_ELEN_CONV(type) );			\
    		CHECK_CUDA_ERROR(elen name:  kernel launch failure);	\
	} else {							\
		REPORT_THREAD_INFO					\
		GPU_EQSP_CALL_NAME(name)<<< NN_GPU >>>			\
			( KERN_ARGS_EQSP_CONV(type) );			\
		/* BUG?  should we put this check everywhere? */	\
    		CHECK_CUDA_ERROR(eqsp name:  kernel launch failure);	\
	}								\



#define CALL_SLOW_KERNEL(name,bitmap,typ,scalars,vectors)		\
									\
	CLEAR_CUDA_ERROR(GPU_SLOW_CALL_NAME(name))			\
	/*SETUP_SLOW_LEN_##typ##vectors*/					\
/*fprintf(stderr,"CALL_SLOW_KERNEL  %s bitmap='%s' scalars='%s' vectors='%s'\n",\
#name,#bitmap,#scalars,#vectors);*/\
	SETUP_BLOCKS_XYZ_##bitmap(VA_PFDEV(vap))	/* using len - was _XY */	\
	/*SETUP_SLOW_INCRS_##bitmap##vectors*/				\
/*REPORT_THREAD_INFO*/						\
	if( extra.x > 0 || extra.y > 0 || extra.z > 0 ){		\
/*REPORT_SLEN_ARGS_##bitmap##typ##scalars##vectors*/			\
		GPU_SLEN_CALL_NAME(name)<<< NN_GPU >>>			\
			(KERN_ARGS_SLEN_##bitmap##typ##scalars##vectors );	\
		CHECK_CUDA_ERROR(slen name:  kernel launch failure);			\
	} else {							\
/*REPORT_SLOW_ARGS_##bitmap##typ##scalars##vectors*/			\
		GPU_SLOW_CALL_NAME(name)<<< NN_GPU >>>				\
			(KERN_ARGS_SLOW_##bitmap##typ##scalars##vectors );	\
		CHECK_CUDA_ERROR(slow name:  kernel launch failure);			\
	}									\



#define CALL_SLOW_CONV_KERNEL(name,bitmap,typ,type)			\
									\
	CLEAR_CUDA_ERROR(GPU_SLOW_CALL_NAME(name))			\
	/*SETUP_SLOW_LEN_##typ##2*/					\
	SETUP_BLOCKS_XYZ_##bitmap(VA_PFDEV(vap))			\
	/*SETUP_SLOW_INCRS_##bitmap##2*/					\
	REPORT_THREAD_INFO						\
/*REPORT_ARGS_##bitmap##typ##scalars##vectors*/				\
	if( extra.x > 0 || extra.y > 0 || extra.z > 0 ){		\
		GPU_SLEN_CALL_NAME(name)<<< NN_GPU >>>			\
			( KERN_ARGS_SLEN_CONV(type) );			\
		CHECK_CUDA_ERROR(slen name:  kernel launch failure);	\
	} else {							\
		GPU_SLOW_CALL_NAME(name)<<< NN_GPU >>>			\
			( KERN_ARGS_SLOW_CONV(type) );			\
		CHECK_CUDA_ERROR(slow name:  kernel launch failure);	\
	}								\


/* For slow loops, we currently only iterate over two dimensions (x and y),
 * although in principle we should be able to handle 3...
 * We need to determine which 2 by examining the dimensions of the vectors.
 */

#ifdef FOOBAR
#define SETUP_SLOW_INCRS(var,isp)			\
	SETUP_INC_IF(var.x,isp,0)		\
	SETUP_INC_IF(var.y,isp,1)		\
	SETUP_INC_IF(var.z,isp,2)

#define SETUP_INC_IF( var ,isp, which_index )	\
	if( VA_DIM_INDEX(vap,which_index) < 0 )		\
		var = 0;				\
	else						\
		var = INCREMENT(isp,VA_DIM_INDEX(vap,which_index));

/*
sprintf(DEFAULT_ERROR_STRING,"SETUP_INC_IF:  %s = %d, dim_indices[%d] = %d",	\
#var,var,which_index,VA_DIM_INDEX(vap,which_index));				\
NADVISE(DEFAULT_ERROR_STRING);
*/


#define SETUP_SLOW_INCRS_1		SETUP_SLOW_INCRS(dst_xyz_incr,VA_DEST_INCSET(vap))
#define SETUP_SLOW_INCRS_SRC1		SETUP_SLOW_INCRS(s1_xyz_incr,VA_SRC1_INCSET(vap))
#define SETUP_SLOW_INCRS_SRC2		SETUP_SLOW_INCRS(s2_xyz_incr,VA_SRC2_INCSET(vap))
#define SETUP_SLOW_INCRS_SRC3		SETUP_SLOW_INCRS(s2_xyz_incr,VA_SRC3_INCSET(vap))
#define SETUP_SLOW_INCRS_SRC4		SETUP_SLOW_INCRS(s2_xyz_incr,VA_SRC4_INCSET(vap))
#define SETUP_SLOW_INCRS_SBM		SETUP_SLOW_INCRS(sbm_xyz_incr,VA_SRC5_INCSET(vap))
#define SETUP_SLOW_INCRS_DBM		SETUP_SLOW_INCRS(dbm_xyz_incr,VA_DEST_INCSET(vap))
#define SETUP_SLOW_INCRS_DBM_SBM 	SETUP_SLOW_INCRS_DBM SETUP_SLOW_INCRS_SBM

#define SETUP_SLOW_INCRS_CONV		SETUP_SLOW_INCRS_1 SETUP_SLOW_INCRS_SRC1
#define SETUP_SLOW_INCRS_2		SETUP_SLOW_INCRS_1 SETUP_SLOW_INCRS_SRC1
#define SETUP_SLOW_INCRS_3		SETUP_SLOW_INCRS_2 SETUP_SLOW_INCRS_SRC2
#define SETUP_SLOW_INCRS_4		SETUP_SLOW_INCRS_3 SETUP_SLOW_INCRS_SRC3
#define SETUP_SLOW_INCRS_5		SETUP_SLOW_INCRS_4 SETUP_SLOW_INCRS_SRC4
#define SETUP_SLOW_INCRS_2SRCS		SETUP_SLOW_INCRS_SRC1 SETUP_SLOW_INCRS_SRC2
#define SETUP_SLOW_INCRS_1SRC		SETUP_SLOW_INCRS_SRC1
#define SETUP_SLOW_INCRS_SBM_1		SETUP_SLOW_INCRS_1 SETUP_SLOW_INCRS_SBM
#define SETUP_SLOW_INCRS_SBM_2		SETUP_SLOW_INCRS_2 SETUP_SLOW_INCRS_SBM
#define SETUP_SLOW_INCRS_SBM_3		SETUP_SLOW_INCRS_3 SETUP_SLOW_INCRS_SBM
#define SETUP_SLOW_INCRS_DBM_		SETUP_SLOW_INCRS_DBM
#define SETUP_SLOW_INCRS_DBM_1SRC	SETUP_SLOW_INCRS_1SRC SETUP_SLOW_INCRS_DBM
#define SETUP_SLOW_INCRS_DBM_2SRCS	SETUP_SLOW_INCRS_2SRCS SETUP_SLOW_INCRS_DBM

#endif // FOOBAR

/* Moved to obj_args.h */
/*
#define VA_SVAL1(vap)			(vap)->va_sval[0]
#define VA_SVAL2(vap)			(vap)->va_sval[1]
#define VA_SVAL3(vap)			(vap)->va_sval[2]
*/

#define SVAL_FLOAT(svp)			(svp)->u_f
#define SVAL_STD(svp)			(svp)->std_scalar
#define SVAL_STDCPX(svp)		(svp)->std_cpx_scalar
#define SVAL_STDQUAT(svp)		(svp)->std_quat_scalar
#define SVAL_BM(svp)			(svp)->bitmap_scalar



// this is 80 columns
//345678901234567890123456789012345678901234567890123456789012345678901234567890


#define NN_GPU		n_blocks, n_threads_per_block


// PORT - insure_gpu_device ???

// We should have an intelligent way of designing blocks...
// On wheatstone, the card can have a maximum of 512 threads/block.
// But this doesn't divide typical video image dimensions
// of 640x480.  But 640 is 128x5, so we could have blocks
// that are 128x4 and not have any leftover.  Using our dumb
// strategy, we have blocks that are 16x32... 480/32=15,
// so we are OK here.
//
// We found a bug doing interlaces of 640/480, the 240 line half-frames
// came out to 7.5 32 row blocks in height.  In that case it's better
// to have the blocks wider and less tall.
//
// It might be nice to adjust things like this automatically, but
// it's kind of tricky!?

// This is kind of kludgy, but let's try it
#define MAX_THREADS_PER_ROW	32

#define SETUP_BLOCKS_XYZ_(pdp)		SETUP_BLOCKS_XYZ(pdp)
#define SETUP_BLOCKS_XYZ_SBM_(pdp)	SETUP_BLOCKS_XYZ(pdp)

#define SETUP_BLOCKS_XYZ(pdp)					\
								\
/*fprintf(stderr,"SETUP_BLOCKS_XYZ_\n");*/			\
	SETUP_BLOCKS_X(VA_ITERATION_TOTAL(vap))			\
	SETUP_BLOCKS_Y(pdp)					\
	SETUP_BLOCKS_Z(pdp)


/* If we have a destination bitmap, we handle all the bits in one word
 * in a single thread.
 *
 * BUG - here we ignore bit0 ???
 *
 * MAX_THREADS_PER_ROW is 32...  for a 512 pixel wide image, the nmber
 * of bitmap words is either 8 (64 bit words) or 16 (32 bit words).
 * So we need to 
 */



#define SETUP_BLOCKS_XYZ_DBM_(pdp)				\
								\
/*fprintf(stderr,"SETUP_BLOCKS_XYZ_DBM_\n");*/			\
	SETUP_BLOCKS_X( N_BITMAP_WORDS(VA_ITERATION_TOTAL(vap)) )		\
	SETUP_BLOCKS_Y(pdp)					\
	SETUP_BLOCKS_Z(pdp)

#define SETUP_BLOCKS_XYZ_DBM_SBM(pdp)				\
								\
	SETUP_BLOCKS_XYZ_DBM_(pdp)

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


#define SETUP_BLOCKS_Y(pdp)						\
								\
	n_threads_per_block.y = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(pdp) \
				/ n_threads_per_block.x;	\
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

#define SETUP_BLOCKS_Z(pdp)						\
								\
	n_threads_per_block.z = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(pdp) \
		/ (n_threads_per_block.x*n_threads_per_block.y);	\
	if( VA_LEN_Z(vap) < n_threads_per_block.z ){			\
		n_threads_per_block.z = VA_LEN_Z(vap);			\
		n_blocks.z = 1;					\
		extra.z = 0;					\
	} else {						\
		n_blocks.z = VA_LEN_Z(vap) / n_threads_per_block.z;	\
		extra.z = VA_LEN_Z(vap) % n_threads_per_block.z;	\
	}							\
	if( extra.z > 0 ) n_blocks.z++;

//#define MORE_DEBUG

#ifdef MORE_DEBUG

#define REPORT_THREAD_INFO					\
								\
sprintf(DEFAULT_ERROR_STRING,"Blocks:  %d x %d x %d    Threads:  %d x %d x %d",	\
n_blocks.x,n_blocks.y,n_blocks.z,	\
n_threads_per_block.x,n_threads_per_block.y,n_threads_per_block.z);\
NADVISE(DEFAULT_ERROR_STRING);						\
sprintf(DEFAULT_ERROR_STRING,"Length:  %d x %d x %d    Extra:  %d x %d x %d",	\
VA_ITERATION_TOTAL(vap),VA_LEN_Y(vap),VA_LEN_Z(vap),extra.x,extra.y,extra.z);				\
NADVISE(DEFAULT_ERROR_STRING);

#define REPORT_THREAD_INFO2					\
								\
sprintf(DEFAULT_ERROR_STRING,"Blocks:  %d x %d    Threads:  %d x %d",	\
n_blocks.x,n_blocks.y,n_threads_per_block.x,n_threads_per_block.y);\
NADVISE(DEFAULT_ERROR_STRING);						\
sprintf(DEFAULT_ERROR_STRING,"Len1:  %ld   Len2:  %ld   Extra:  %d x %d",	\
len1,len2,extra.x,extra.y);					\
NADVISE(DEFAULT_ERROR_STRING);

#else /* ! MORE_DEBUG */

#define REPORT_THREAD_INFO
#define REPORT_THREAD_INFO2

#endif /* ! MORE_DEBUG */


#define DEFAULT_YZ						\
								\
	n_threads_per_block.y = n_threads_per_block.z =		\
	n_blocks.y = n_blocks.z = 1;				\
	extra.y = extra.z = 0;

#define GET_THREADS_PER_BLOCK(pdp,len_var)			\
								\
	DEFAULT_YZ						\
	SET_BLOCKS_FROM_LEN(pdp,len_var)

#define SET_BLOCKS_FROM_LEN( pdp, n_tot )			\
								\
	if( n_tot < PFDEV_CUDA_MAX_THREADS_PER_BLOCK(pdp) ) {	\
		n_threads_per_block.x = n_tot;			\
		n_blocks.x = 1;					\
		extra.x = 0;					\
	} else {						\
		n_blocks.x = n_tot / PFDEV_CUDA_MAX_THREADS_PER_BLOCK(pdp);	\
		n_threads_per_block.x = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(pdp);	\
		extra.x = n_tot % PFDEV_CUDA_MAX_THREADS_PER_BLOCK(pdp);	\
	}

#define GET_THREADS_PER_SBM_BLOCK(pdp,len_var)	GET_THREADS_PER_BLOCK(pdp,len_var)

// This used to be called GET_THREADS_PER_BITMAP_BLOCK

#define GET_THREADS_PER_DBM_BLOCK(pdp,len_var)				\
								\
/*fprintf(stderr,"GET_THREADS_PER_DBM_BLOCK:  len = %d\n",len_var);*/\
	DEFAULT_YZ						\
	if( (VA_DBM_BIT0(vap)+len_var) < BITS_PER_BITMAP_WORD ) {	\
		n_threads_per_block.x = 1;			\
		n_blocks.x = 1;					\
	} else {						\
		int nw;						\
		nw = N_BITMAP_WORDS(VA_DBM_BIT0(vap)+len_var);	\
		SET_BLOCKS_FROM_LEN(pdp,nw)				\
	}


#define GET_THREADS_PER_DBM_SBMBLOCK(pdp,len_var)				\
								\
	DEFAULT_YZ						\
	if( (VA_DBM_BIT0(vap)+len_var) < BITS_PER_BITMAP_WORD ) {	\
		n_threads_per_block.x = 1;			\
		n_blocks.x = 1;					\
	} else {						\
		int nw;						\
		nw = N_BITMAP_WORDS(VA_DBM_BIT0(vap)+len_var);	\
		SET_BLOCKS_FROM_LEN(pdp,nw)				\
	}




#define MAX_THREADS_PER_BITMAP_ROW	MIN(MAX_THREADS_PER_ROW,BITS_PER_BITMAP_WORD)

#define INSIST_LENGTH( n , msg , name )					\
									\
		if( (n) == 1 ){						\
			sprintf(DEFAULT_ERROR_STRING,				\
	"Oops, kind of silly to do %s of 1-len vector %s!?",msg,name);	\
			NWARN(DEFAULT_ERROR_STRING);				\
			return;						\
		}


#ifdef MORE_DEBUG

#define REPORT_VECTORIZATION1( host_func_name )				\
									\
	/*sprintf(DEFAULT_ERROR_STRING,						\
"%s:  ready to vectorize:\txyz_len.x = %ld, inc1.x = %ld, inc1.y = %ld",	\
		#host_func_name,VA_ITERATION_TOTAL(vap),inc1.x,inc1.y);			\
	NADVISE(DEFAULT_ERROR_STRING);*/

#define REPORT_VECTORIZATION2( host_func_name )				\
									\
	REPORT_VECTORIZATION1( host_func_name )				\
	sprintf(DEFAULT_ERROR_STRING, "\t\t\t\tinc2.x = %ld, inc2.y = %ld",	\
		inc2.x,inc2.y);						\
	NADVISE(DEFAULT_ERROR_STRING);

#define REPORT_VECTORIZATION3( host_func_name )				\
									\
	REPORT_VECTORIZATION2( host_func_name )				\
	sprintf(DEFAULT_ERROR_STRING, "\t\t\t\tinc3.x = %ld, inc3.y = %ld",	\
		inc3.x,inc3.y);						\
	NADVISE(DEFAULT_ERROR_STRING);

#define REPORT_VECTORIZATION4( host_func_name )				\
									\
	REPORT_VECTORIZATION3( host_func_name )				\
	sprintf(DEFAULT_ERROR_STRING, "\t\t\t\tinc4.x = %ld, inc4.y = %ld",	\
		inc4.x,inc4.y);						\
	NADVISE(DEFAULT_ERROR_STRING);

#define REPORT_VECTORIZATION5( host_func_name )				\
									\
	REPORT_VECTORIZATION4( host_func_name )				\
	sprintf(DEFAULT_ERROR_STRING, "\t\t\t\tinc5.x = %ld, inc5.y = %ld",	\
		inc5.x,inc5.y);						\
	NADVISE(DEFAULT_ERROR_STRING);

#else /* ! MORE_DEBUG */

#define REPORT_VECTORIZATION1( host_func_name )
#define REPORT_VECTORIZATION2( host_func_name )
#define REPORT_VECTORIZATION3( host_func_name )
#define REPORT_VECTORIZATION4( host_func_name )
#define REPORT_VECTORIZATION5( host_func_name )

#endif /* ! MORE_DEBUG */





// MM_IND vmaxi etc

// CUDA definitions
// BUG we probably want the passed vap to have constant data...

// BUG use symbolic constant for kernel args!
#define CALL_GPU_FAST_NOCC_SETUP_FUNC(name)					\
	CLEAR_GPU_ERROR(name##_nocc_setup)					\
sprintf(DEFAULT_ERROR_STRING,"calling %s_nocc_setup...",#name);			\
NADVISE(DEFAULT_ERROR_STRING);							\
	REPORT_THREAD_INFO2							\
	GPU_FAST_CALL_NAME(name##_nocc_setup)<<< NN_GPU >>>				\
		(dst_values, dst_counts, orig_src_values, indices, len1, len2);	\
	CHECK_GPU_ERROR(name##_nocc_setup)


// BUG use symbolic constant for kernel args!
#define CALL_GPU_FAST_NOCC_HELPER_FUNC(name)					\
	CLEAR_GPU_ERROR(name##_nocc_helper)					\
sprintf(DEFAULT_ERROR_STRING,"calling %s_nocc_helper...",#name);		\
NADVISE(DEFAULT_ERROR_STRING);							\
	REPORT_THREAD_INFO2							\
	GPU_FAST_CALL_NAME(name##_nocc_helper)<<< NN_GPU >>>				\
		(dst_values, dst_counts,src_values,src_counts, indices,len1,len2,stride); \
	CHECK_GPU_ERROR(name##_nocc_helper)



// CUDA only!
#define CALL_GPU_FAST_PROJ_2V_SETUP_FUNC(name) /* CUDA only */			\
	CLEAR_GPU_ERROR(name)						\
	REPORT_THREAD_INFO2						\
fprintf(stderr,"CALL_GPU_FAST_PROJ_2V_SETUP_FUNC(%s):  dst_values = 0x%lx, orig_src_values = 0x%lx, len1 = %d, len2 = %d\n",\
#name,(long)dst_values,(long)orig_src_values,len1,len2);\
	GPU_FAST_CALL_NAME(name##_setup)<<< NN_GPU >>>( dst_values, orig_src_values, len1, len2 );	\
	CHECK_GPU_ERROR(name)

#define CALL_GPU_FAST_PROJ_2V_HELPER_FUNC(name) /* CUDA only */			\
	CLEAR_GPU_ERROR(name)						\
	REPORT_THREAD_INFO2						\
fprintf(stderr,"CALL_GPU_FAST_PROJ_2V_HELPER_FUNC(%s):  dst_values = 0x%lx, src_values = 0x%lx, len1 = %d, len2 = %d\n",\
#name,(long)dst_values,(long)src_values,len1,len2);\
	GPU_FAST_CALL_NAME(name##_helper)<<< NN_GPU >>>( dst_values, src_values, len1, len2 );	\
	CHECK_GPU_ERROR(name)

#define CALL_GPU_FAST_PROJ_3V_FUNC(name)					\
	CLEAR_GPU_ERROR(name)						\
	REPORT_THREAD_INFO2						\
	GPU_FAST_CALL_NAME(name)<<< NN_GPU >>>				\
		( dst_values, src1_values, src2_values, len1, len2 );	\
	CHECK_GPU_ERROR(name)


#define CALL_GPU_FAST_INDEX_SETUP_FUNC(name)					\
	CLEAR_GPU_ERROR(name##_fast_index_setup)				\
	REPORT_THREAD_INFO2						\
	GPU_FAST_CALL_NAME(name##_index_setup)<<< NN_GPU >>>			\
		(indices,src1_values,src2_values,len1,len2);		\
	CHECK_GPU_ERROR(name##_index_setup)


#define CALL_GPU_FAST_INDEX_HELPER_FUNC(name)				\
	CLEAR_GPU_ERROR(name##_fast_index_helper)				\
	REPORT_THREAD_INFO2						\
	GPU_FAST_CALL_NAME(name##_index_helper)<<< NN_GPU >>>		\
		(indices,idx1_values,idx2_values,orig_src_values,len1,len2);	\
	CHECK_GPU_ERROR(name##_index_helper)


#ifdef DUPLICATED_CODE
// this is 80 columns
//345678901234567890123456789012345678901234567890123456789012345678901234567890


#define NN_GPU		n_blocks, n_threads_per_block


// PORT - insure_gpu_device ???

// We should have an intelligent way of designing blocks...
// On wheatstone, the card can have a maximum of 512 threads/block.
// But this doesn't divide typical video image dimensions
// of 640x480.  But 640 is 128x5, so we could have blocks
// that are 128x4 and not have any leftover.  Using our dumb
// strategy, we have blocks that are 16x32... 480/32=15,
// so we are OK here.
//
// We found a bug doing interlaces of 640/480, the 240 line half-frames
// came out to 7.5 32 row blocks in height.  In that case it's better
// to have the blocks wider and less tall.
//
// It might be nice to adjust things like this automatically, but
// it's kind of tricky!?

// This is kind of kludgy, but let's try it
#define MAX_THREADS_PER_ROW	32

#define SETUP_BLOCKS_XYZ_(pdp)		SETUP_BLOCKS_XYZ(pdp)
#define SETUP_BLOCKS_XYZ_SBM_(pdp)	SETUP_BLOCKS_XYZ(pdp)

#define SETUP_BLOCKS_XYZ(pdp)					\
								\
/*fprintf(stderr,"SETUP_BLOCKS_XYZ_\n");*/			\
	SETUP_BLOCKS_X(VA_ITERATION_TOTAL(vap))					\
	SETUP_BLOCKS_Y(pdp)					\
	SETUP_BLOCKS_Z(pdp)


/* If we have a destination bitmap, we handle all the bits in one word
 * in a single thread.
 *
 * BUG - here we ignore bit0 ???
 *
 * MAX_THREADS_PER_ROW is 32...  for a 512 pixel wide image, the nmber
 * of bitmap words is either 8 (64 bit words) or 16 (32 bit words).
 * So we need to 
 */



#define SETUP_BLOCKS_XYZ_DBM_(pdp)				\
								\
/*fprintf(stderr,"SETUP_BLOCKS_XYZ_DBM_\n");*/			\
	SETUP_BLOCKS_X( N_BITMAP_WORDS(VA_ITERATION_TOTAL(vap)) )		\
	SETUP_BLOCKS_Y(pdp)					\
	SETUP_BLOCKS_Z(pdp)

#define SETUP_BLOCKS_XYZ_DBM_SBM(pdp)				\
								\
	SETUP_BLOCKS_XYZ_DBM_(pdp)

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


#define SETUP_BLOCKS_Y(pdp)						\
								\
	n_threads_per_block.y = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(pdp) \
				/ n_threads_per_block.x;	\
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

#define SETUP_BLOCKS_Z(pdp)						\
								\
	n_threads_per_block.z = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(pdp) \
		/ (n_threads_per_block.x*n_threads_per_block.y);	\
	if( VA_LEN_Z(vap) < n_threads_per_block.z ){			\
		n_threads_per_block.z = VA_LEN_Z(vap);			\
		n_blocks.z = 1;					\
		extra.z = 0;					\
	} else {						\
		n_blocks.z = VA_LEN_Z(vap) / n_threads_per_block.z;	\
		extra.z = VA_LEN_Z(vap) % n_threads_per_block.z;	\
	}							\
	if( extra.z > 0 ) n_blocks.z++;

//#define MORE_DEBUG

#ifdef MORE_DEBUG

#define REPORT_THREAD_INFO					\
								\
sprintf(DEFAULT_ERROR_STRING,"Blocks:  %d x %d x %d    Threads:  %d x %d x %d",	\
n_blocks.x,n_blocks.y,n_blocks.z,	\
n_threads_per_block.x,n_threads_per_block.y,n_threads_per_block.z);\
NADVISE(DEFAULT_ERROR_STRING);						\
sprintf(DEFAULT_ERROR_STRING,"Length:  %d x %d x %d    Extra:  %d x %d x %d",	\
VA_ITERATION_TOTAL(vap),VA_LEN_Y(vap),VA_LEN_Z(vap),extra.x,extra.y,extra.z);				\
NADVISE(DEFAULT_ERROR_STRING);

#define REPORT_THREAD_INFO2					\
								\
sprintf(DEFAULT_ERROR_STRING,"Blocks:  %d x %d    Threads:  %d x %d",	\
n_blocks.x,n_blocks.y,n_threads_per_block.x,n_threads_per_block.y);\
NADVISE(DEFAULT_ERROR_STRING);						\
sprintf(DEFAULT_ERROR_STRING,"Len1:  %ld   Len2:  %ld   Extra:  %d x %d",	\
len1,len2,extra.x,extra.y);					\
NADVISE(DEFAULT_ERROR_STRING);

#else /* ! MORE_DEBUG */

#define REPORT_THREAD_INFO
#define REPORT_THREAD_INFO2

#endif /* ! MORE_DEBUG */


#define DEFAULT_YZ						\
								\
	n_threads_per_block.y = n_threads_per_block.z =		\
	n_blocks.y = n_blocks.z = 1;				\
	extra.y = extra.z = 0;

#define GET_THREADS_PER_BLOCK(pdp,len_var)			\
								\
	DEFAULT_YZ						\
	SET_BLOCKS_FROM_LEN(pdp,len_var)

#define SET_BLOCKS_FROM_LEN( pdp, n_tot )			\
								\
	if( n_tot < PFDEV_CUDA_MAX_THREADS_PER_BLOCK(pdp) ) {	\
		n_threads_per_block.x = n_tot;			\
		n_blocks.x = 1;					\
		extra.x = 0;					\
	} else {						\
		n_blocks.x = n_tot / PFDEV_CUDA_MAX_THREADS_PER_BLOCK(pdp);	\
		n_threads_per_block.x = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(pdp);	\
		extra.x = n_tot % PFDEV_CUDA_MAX_THREADS_PER_BLOCK(pdp);	\
	}

#define GET_THREADS_PER_SBM_BLOCK(pdp,len_var)	GET_THREADS_PER_BLOCK(pdp,len_var)

// This used to be called GET_THREADS_PER_BITMAP_BLOCK

#define GET_THREADS_PER_DBM_BLOCK(pdp,len_var)				\
								\
/*fprintf(stderr,"GET_THREADS_PER_DBM_BLOCK:  len = %d\n",len_var);*/\
	DEFAULT_YZ						\
	if( (VA_DBM_BIT0(vap)+len_var) < BITS_PER_BITMAP_WORD ) {	\
		n_threads_per_block.x = 1;			\
		n_blocks.x = 1;					\
	} else {						\
		int nw;						\
		nw = N_BITMAP_WORDS(VA_DBM_BIT0(vap)+len_var);	\
		SET_BLOCKS_FROM_LEN(pdp,nw)				\
	}


#define GET_THREADS_PER_DBM_SBMBLOCK(pdp,len_var)				\
								\
	DEFAULT_YZ						\
	if( (VA_DBM_BIT0(vap)+len_var) < BITS_PER_BITMAP_WORD ) {	\
		n_threads_per_block.x = 1;			\
		n_blocks.x = 1;					\
	} else {						\
		int nw;						\
		nw = N_BITMAP_WORDS(VA_DBM_BIT0(vap)+len_var);	\
		SET_BLOCKS_FROM_LEN(pdp,nw)				\
	}




#define MAX_THREADS_PER_BITMAP_ROW	MIN(MAX_THREADS_PER_ROW,BITS_PER_BITMAP_WORD)

#define INSIST_LENGTH( n , msg , name )					\
									\
		if( (n) == 1 ){						\
			sprintf(DEFAULT_ERROR_STRING,				\
	"Oops, kind of silly to do %s of 1-len vector %s!?",msg,name);	\
			NWARN(DEFAULT_ERROR_STRING);				\
			return;						\
		}


#ifdef MORE_DEBUG

#define REPORT_VECTORIZATION1( host_func_name )				\
									\
	/*sprintf(DEFAULT_ERROR_STRING,						\
"%s:  ready to vectorize:\txyz_len.x = %ld, inc1.x = %ld, inc1.y = %ld",	\
		#host_func_name,VA_ITERATION_TOTAL(vap),inc1.x,inc1.y);			\
	NADVISE(DEFAULT_ERROR_STRING);*/

#define REPORT_VECTORIZATION2( host_func_name )				\
									\
	REPORT_VECTORIZATION1( host_func_name )				\
	sprintf(DEFAULT_ERROR_STRING, "\t\t\t\tinc2.x = %ld, inc2.y = %ld",	\
		inc2.x,inc2.y);						\
	NADVISE(DEFAULT_ERROR_STRING);

#define REPORT_VECTORIZATION3( host_func_name )				\
									\
	REPORT_VECTORIZATION2( host_func_name )				\
	sprintf(DEFAULT_ERROR_STRING, "\t\t\t\tinc3.x = %ld, inc3.y = %ld",	\
		inc3.x,inc3.y);						\
	NADVISE(DEFAULT_ERROR_STRING);

#define REPORT_VECTORIZATION4( host_func_name )				\
									\
	REPORT_VECTORIZATION3( host_func_name )				\
	sprintf(DEFAULT_ERROR_STRING, "\t\t\t\tinc4.x = %ld, inc4.y = %ld",	\
		inc4.x,inc4.y);						\
	NADVISE(DEFAULT_ERROR_STRING);

#define REPORT_VECTORIZATION5( host_func_name )				\
									\
	REPORT_VECTORIZATION4( host_func_name )				\
	sprintf(DEFAULT_ERROR_STRING, "\t\t\t\tinc5.x = %ld, inc5.y = %ld",	\
		inc5.x,inc5.y);						\
	NADVISE(DEFAULT_ERROR_STRING);

#else /* ! MORE_DEBUG */

#define REPORT_VECTORIZATION1( host_func_name )
#define REPORT_VECTORIZATION2( host_func_name )
#define REPORT_VECTORIZATION3( host_func_name )
#define REPORT_VECTORIZATION4( host_func_name )
#define REPORT_VECTORIZATION5( host_func_name )

#endif /* ! MORE_DEBUG */



#endif // DUPLICATED_CODE

