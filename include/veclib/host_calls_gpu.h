#include "veclib/report_args.h"


#define DECLARE_HOST_FAST_CALL_VARS(bitmap,vectors)			\
									\
	DECLARE_PLATFORM_FAST_VARS

#define DECLARE_HOST_EQSP_CALL_VARS(bitmap,vectors)			\
									\
	DECLARE_HOST_FAST_CALL_VARS(bitmap,vectors)			\
	DECL_EQSP_INCRS_##bitmap##vectors


#define DECLARE_HOST_SLOW_CALL_VARS(bitmap,vectors)			\
									\
	DECLARE_PLATFORM_SLOW_VARS					\
	DECL_SLOW_INCRS_##bitmap##vectors



/* SETUP_KERNEL is a nop for CUDA */

#define GENERIC_HOST_FAST_CALL(name,bitmap,typ,scalars,vectors)		\
									\
static void HOST_FAST_CALL_NAME(name)(LINK_FUNC_ARG_DECLS)		\
{									\
	DECLARE_HOST_FAST_CALL_VARS(bitmap,vectors)			\
	SETUP_KERNEL_FAST_CALL(name,bitmap,typ,scalars,vectors)		\
	CALL_FAST_KERNEL(name,bitmap,typ,scalars,vectors)		\
}

#define GENERIC_HOST_FAST_CONV(name,bitmap,typ,type)			\
/* Declare the host function here to call the kernel */			\
									\
static void HOST_FAST_CALL_NAME(name)					\
( LINK_FUNC_ARG_DECLS )							\
{									\
	DECLARE_HOST_FAST_CALL_VARS(bitmap,2)				\
	SETUP_KERNEL_FAST_CALL_CONV(name,bitmap,typ,type)		\
	CALL_FAST_CONV_KERNEL(name,bitmap,typ,type)			\
}


#define GENERIC_HOST_EQSP_CALL(name,bitmap,typ,scalars,vectors)		\
									\
static void HOST_EQSP_CALL_NAME(name)(LINK_FUNC_ARG_DECLS)		\
{									\
	DECLARE_HOST_EQSP_CALL_VARS(bitmap,vectors)			\
	SETUP_KERNEL_EQSP_CALL(name,bitmap,typ,scalars,vectors)		\
	CALL_EQSP_KERNEL(name,bitmap,typ,scalars,vectors)		\
}

#define GENERIC_HOST_EQSP_CONV(name,bitmap,typ,type)			\
/* Declare the host function here to call the kernel */			\
									\
static void HOST_EQSP_CALL_NAME(name)					\
( LINK_FUNC_ARG_DECLS )							\
{									\
	DECLARE_HOST_EQSP_CALL_VARS(bitmap,2)				\
	SETUP_KERNEL_EQSP_CALL_CONV(name,bitmap,typ,type)		\
	CALL_EQSP_CONV_KERNEL(name,bitmap,typ,type)				\
}

/* setup_slow_len must be called outside of here, because the
 * lengths are inside vap...
 */

#define GENERIC_HOST_SLOW_CALL(name,bitmap,typ,scalars,vectors)		\
									\
static void HOST_SLOW_CALL_NAME(name)(LINK_FUNC_ARG_DECLS)		\
{									\
	DECLARE_HOST_SLOW_CALL_VARS(bitmap,vectors)			\
	/*SETUP_SLOW_LEN_##bitmap##vectors*/				\
	SETUP_SLOW_INCS_##bitmap##vectors				\
	SETUP_KERNEL_SLOW_CALL(name,bitmap,typ,scalars,vectors)		\
	CALL_SLOW_KERNEL(name,bitmap,typ,scalars,vectors)		\
}

#define GENERIC_HOST_SLOW_CONV(name,bitmap,typ,type)			\
/* Declare the host function here to call the kernel */			\
									\
static void HOST_SLOW_CALL_NAME(name)					\
( LINK_FUNC_ARG_DECLS )							\
{									\
	DECLARE_HOST_SLOW_CALL_VARS(bitmap,2)				\
	/*SETUP_SLOW_LEN_##bitmap##2*/					\
	SETUP_SLOW_INCS_##bitmap##2					\
	SETUP_KERNEL_SLOW_CALL_CONV(name,bitmap,typ,type)		\
	CALL_SLOW_CONV_KERNEL(name,bitmap,typ,type)			\
}


#ifdef MOVE_TO_CUDA
// cuda version?
#define HELPER_FUNC_PRELUDE						\
									\
	BLOCK_VARS_DECLS						\
	SET_BLOCK_COUNT



#define SET_BLOCK_COUNT							\
									\
	DEFAULT_YZ							\
									\
	if( len1 < max_threads_per_block ) {				\
		n_threads_per_block.x = len1;				\
		n_blocks.x = 1;						\
		extra.x = 0; /* quiet compiler */			\
	} else {							\
		n_blocks.x = len1 / max_threads_per_block;		\
		n_threads_per_block.x = max_threads_per_block;		\
		extra.x = len1 % max_threads_per_block;			\
		if (extra.x != 0) n_blocks.x++;				\
	}
#endif // MOVE_TO_CUDA


// MM_IND vmaxi etc

#ifdef BUILD_FOR_CUDA

// BUG - move to cu2 folder file...
// CUDA definitions
// BUG we probably want the passed vap to have constant data...

#define CALL_GPU_NOCC_SETUP_FUNC(name)						\
	CLEAR_GPU_ERROR(name##_nocc_setup)					\
sprintf(DEFAULT_ERROR_STRING,"calling %s_nocc_setup...",#name);			\
NADVISE(DEFAULT_ERROR_STRING);							\
	REPORT_THREAD_INFO2							\
	GPU_CALL_NAME(name##_nocc_setup)<<< NN_GPU >>>				\
		(dst_values, dst_counts, src_values, indices, len1, len2);	\
	CHECK_GPU_ERROR(name##_nocc_setup)


#define CALL_GPU_NOCC_HELPER_FUNC(name)						\
	CLEAR_GPU_ERROR(name##_nocc_helper)					\
sprintf(DEFAULT_ERROR_STRING,"calling %s_nocc_helper...",#name);		\
NADVISE(DEFAULT_ERROR_STRING);							\
	REPORT_THREAD_INFO2							\
	GPU_CALL_NAME(name##_nocc_helper)<<< NN_GPU >>>				\
		(dst_values, dst_counts,src_values,src_counts, indices,len1,len2,stride); \
	CHECK_GPU_ERROR(name##_nocc_helper)



#define CALL_GPU_PROJ_2V_FUNC(name)					\
	CLEAR_GPU_ERROR(name)						\
	REPORT_THREAD_INFO2						\
	GPU_CALL_NAME(name)<<< NN_GPU >>>( dst_values, src_values, len1, len2 );	\
	CHECK_GPU_ERROR(name)

#define CALL_GPU_PROJ_3V_FUNC(name)					\
	CLEAR_GPU_ERROR(name)						\
	REPORT_THREAD_INFO2						\
	GPU_CALL_NAME(name)<<< NN_GPU >>>				\
		( dst_values, src1_values, src2_values, len1, len2 );	\
	CHECK_GPU_ERROR(name)


#define CALL_GPU_INDEX_SETUP_FUNC(name)					\
	CLEAR_GPU_ERROR(name##_index_setup)				\
	REPORT_THREAD_INFO2						\
	GPU_CALL_NAME(name##_index_setup)<<< NN_GPU >>>			\
		(indices,src1_values,src2_values,len1,len2);		\
	CHECK_GPU_ERROR(name##_index_setup)


#define CALL_GPU_INDEX_HELPER_FUNC(name)				\
	CLEAR_GPU_ERROR(name##_index_helper)				\
	REPORT_THREAD_INFO2						\
	GPU_CALL_NAME(name##_index_helper)<<< NN_GPU >>>		\
		(indices,idx1_values,idx2_values,orig_src_values,len1,len2);	\
	CHECK_GPU_ERROR(name##_index_helper)

#endif // BUILD_FOR_CUDA



#define FINISH_MM_ITERATION(whence)					\
									\
	/* Each temp vector gets used twice,				\
	 * first as result, then as source */				\
	if( src_to_free != NULL ){					\
		FREETMP_NAME(src_to_free,#whence);			\
		src_to_free=NULL;					\
	}								\
									\
	/* Now roll things over... */					\
	idx1_values = indices;						\
	len = len1;							\
	src_to_free = dst_to_free;					\
	dst_to_free = NULL;

#define SETUP_IDX_ITERATION(a2,a3,whence)				\
									\
	len1 = (len+1)/2;						\
	len2 = len - len1;						\
									\
	a3 = a2 + len1;							\
									\
	if( len1 == 1 ){						\
		indices = (index_type *) VA_DEST_PTR(vap);		\
		dst_to_free = NULL;					\
	} else {							\
		indices = (index_type *) TMPVEC_NAME(sizeof(index_type),len1,#whence);	\
		dst_to_free = indices;					\
	}



#ifdef FOOBAR
// args are std_type instead of index_type - ?

#define SETUP_MM_ITERATION2(a2,a3,whence)				\
									\
	len1 = (len+1)/2;						\
	len2 = len - len1;						\
									\
	a3 = a2 + len1;							\
									\
	if( len1 == 1 ){						\
		arg1 = (std_type *) OBJ_DATA_PTR(OA_DEST(oap));			\
		dst_to_free = NULL;					\
	} else {							\
		arg1 = (std_type *) TMPVEC_NAME(sizeof(std_type),len1,#whence);	\
		dst_to_free = arg1;					\
	}

#endif // FOOBAR



#define FINISH_NOCC_ITERATION(whence)					\
									\
	/* Each temp vector gets used twice,				\
	 * first as result, then as source */				\
	if( src_vals_to_free != NULL ){					\
		FREETMP_NAME(src_vals_to_free,#whence);			\
		src_vals_to_free=NULL;					\
	}								\
	if( src_counts_to_free != NULL ){				\
		FREETMP_NAME(src_counts_to_free,#whence);		\
		src_counts_to_free=NULL;				\
	}								\
									\
	/* Now roll things over... */					\
	src_values = dst_values;					\
	src_counts = dst_counts;					\
	len = len1;							\
	src_vals_to_free = dst_vals_to_free;				\
	src_counts_to_free = dst_counts_to_free;			\
	dst_vals_to_free = NULL;					\
	dst_counts_to_free = NULL;

/* For vmaxg, the temp arrays don't double for the destination vector...
 * oa_sdp[0] is the extreme value, oa_sdp[1] is the count.
 */

#define SETUP_NOCC_ITERATION(whence)							\
											\
	len1 = (len+1)/2;								\
	len2 = len - len1;								\
											\
	if( len1 == 1 ){								\
		dst_values = (std_type *) VA_SVAL1(vap);				\
		dst_vals_to_free = NULL;						\
		dst_counts = (index_type *) VA_SVAL2(vap);				\
		dst_counts_to_free = NULL;						\
	} else {									\
		dst_values = (std_type *) TMPVEC_NAME(sizeof(std_type),len1,#whence);	\
		dst_vals_to_free = (std_type *) dst_values;				\
		dst_counts = (index_type *) TMPVEC_NAME(sizeof(index_type),len1,#whence); \
		dst_counts_to_free = dst_counts;					\
	}


#ifdef OLD
#define INIT_MM(src_ptr)						\
									\
	len = OBJ_N_TYPE_ELTS(oap->oa_dp[0]);				\
	src_ptr = (std_type *)OBJ_DATA_PTR(oap->oa_dp[0]);		\
									\
	INSIST_CONTIG( oap->oa_dp[0] , "min/max" )			\
	INSIST_LENGTH( len, "min/max" , OBJ_NAME(oap->oa_dp[0]) )
#endif // OLD

#define INIT_MM(src_ptr)						\
									\
	len = OBJ_N_TYPE_ELTS(oap->oa_dp[0]);				\
	src_ptr = (std_type *)OBJ_DATA_PTR(oap->oa_dp[0]);		\
									\
	CHECK_MM(min/max)

#define CHECK_MM(name)							\
									\
	INSIST_CONTIG( oap->oa_dp[0] , #name )			\
	INSIST_LENGTH( OBJ_N_TYPE_ELTS(oap->oa_dp[0]), #name , OBJ_NAME(oap->oa_dp[0]) )


// The index functions wants to return the index of the max/min.
// After the first pass, we will have a half-length array with the
// max indices.  The subsequent passes need to use these indices to
// lookup data to make the comparison...

//#define H_CALL_PROJ_2V_IDX(name)	SLOW_HOST_CALL(name,,,,2)
//#define H_CALL_MM_IND(name)	


#define H_CALL_PROJ_2V_IDX(name)						\
										\
static void HOST_SLOW_CALL_NAME(name)(LINK_FUNC_ARG_DECLS)			\
{										\
	index_type *indices;							\
	std_type *src1_values;							\
	std_type *src2_values;							\
	std_type *orig_src_values;						\
	index_type *idx1_values,*idx2_values;					\
	index_type *dst_to_free=NULL;						\
	index_type *src_to_free=NULL;						\
	u_long len, len1, len2;							\
	DECLARE_PLATFORM_VARS							\
										\
	src1_values = (std_type *) VA_SRC1_PTR(vap);						\
	len = VA_SRC1_LEN(vap);							\
	SETUP_IDX_ITERATION(src1_values,src2_values,name)			\
	orig_src_values = src1_values;						\
	/*max_threads_per_block = OBJ_MAX_THREADS_PER_BLOCK(OA_DEST(oap));*/	\
	CALL_GPU_INDEX_SETUP_FUNC(name)						\
	FINISH_MM_ITERATION(name)						\
										\
	while( len > 1 ){							\
		SETUP_IDX_ITERATION(idx1_values,idx2_values,name)		\
		CALL_GPU_INDEX_HELPER_FUNC(name)				\
	/*(indices,idx1_values,idx2_values,orig_src_values,len1,len2)*/		\
		FINISH_MM_ITERATION(name)					\
	}									\
	if( src_to_free != NULL ){						\
		FREETMP_NAME(src_to_free,#name);				\
		src_to_free=NULL;						\
	}									\
}										\
										\
static void HOST_TYPED_CALL_NAME(name,type_code)(HOST_CALL_ARG_DECLS )		\
{										\
	Vector_Args va1, *vap=(&va1);						\
										\
	if( OBJ_MACH_PREC(OA_DEST(oap)) != INDEX_PREC ){			\
		sprintf(DEFAULT_ERROR_STRING,					\
"%s:  destination index %s has %s precision, should be %s",			\
			STRINGIFY(HOST_TYPED_CALL_NAME(name,type_code)),	\
			OBJ_NAME(OA_DEST(oap)),					\
			PREC_NAME(OBJ_MACH_PREC_PTR(OA_DEST(oap))),		\
			PREC_NAME(PREC_FOR_CODE(INDEX_PREC)) );			\
		NWARN(DEFAULT_ERROR_STRING);					\
		return;								\
	}									\
										\
	CHECK_MM(name)								\
	/* BUG need to xfer args to vap */					\
	SET_MAX_THREADS_FROM_OBJ(OA_DEST(oap))					\
										\
	HOST_SLOW_CALL_NAME(name)(LINK_FUNC_ARGS);				\
}


/* Can we use the recursive strategy for vmaxg?
 *
 * We need a vector of occurrences for each thread...
 * 
 * Let's assume that the destination vector (index array) has the same dimension
 * as the input.  So we can use portions of it for the temporary storage.
 * First do all the pairs.  We need an occurrence count array equal in size to
 * the number of pairs.  The first pass sets this - the number of occurrences
 * will be either 1 or 2.
 * The second pass will compare the values accessed indirectly (as in vmaxi).
 *
 * Unlike vmaxi, we don't compare far-away pairs, we do adjacent pairs.
 *
 *
 * How do we do the merge in general?  Let's try an example with 8 elements [ 6 6 3 6 5 6 7 8 ]
 * After the first pass:
 *	index_list	[ 0 1 3 . 5 . 7 . ]
 *	n_tmp		[ 2 1 1 1 ]
 *	v_tmp		[ 6 6 6 8 ]
 *
 * The nocc_helper function has a thread for each pair of values in v_tmp;
 * it compares the values, and then updates the new result n and v accordingly.
 * The hard part is copying the indices (if necessary).
 * We have to pass the addresses of the source and destination n and v arrays,
 * plus the address of the index_list.  As len gets smaller, so do n_tmp and
 * v_tmp, but the index_list is always the same length, so we need to pass
 * a multiplier to get the offset...
 *
 * The setup function is like the helper function, but it uses the original
 * input instead of the temp array for the source of the comparisons.
 * AND it doesn't have to worry about doing any merging.
 *
 * After the second pass:
 *	index_list	[ 0 1 3 . 7 . . . ]
 *	n_tmp		[ 3 1 ]
 *	v_tmp		[ 6 8 ]
 *
 * What if we initialize like this before any passes:
 * 	index_list	[ 0 1 2 3 4 5 6 7 ]
 * 	n_tmp		[ 1 1 1 1 1 1 1 1 ]
 * 	v_tmp		[ 6 6 3 6 5 6 7 8 ]
 *
 * Testing with [ 1 2 3 4 5 6 7 8 ]
 *
 * index_list:	[ 1 . 3 . 5 . 7 . ]
 * 		[ 3 . . . 7 . . . ]
 * 		[ 7 . . . . . . . ]
 *
 * max_list:	[ 2 4 6 8 ]
 * 		[ 4 8 ]
 * 		[ 8 ]
 *
 * n_list	[ 1 1 1 1 ]
 * 		[ 1 1 ]
 * 		[ 1 ]
 *
 * What about with 9 elements?
 * input:	[ 1 2 3 4 5 6 7 8 9 ]
 *
 * index_list:	[ 1 . 3 . 5 . 7 . 8 ]
 * 		[ 3 . . . 7 . . . 8 ]
 * 		[ 7 . . . . . . . 8 ]
 * 		[ 8 . . . . . . . . ]
 *
 *
 * max_list:	[ 2 4 6 8 9 ]
 * 		[ 4 8 9 ]
 * 		[ 8 9 ]
 * 		[ 9 ]
 *
 *
 * What about with 7?
 *
 * input:	[ 1 2 3 4 5 6 7 ]
 *
 * max_v:	[ 2 4 6 7 ]
 * 		[ 4 7 ]
 * 		[ 7 ]
 *
 * indices:	[ 1 . 3 . 5 . 6 ]
 * 		[ 3 . . . 6 . . ]
 * 		[ 6 . . . . . . ]
 */

// The g functions (vmaxg etc)  want to return an array of indices, along
// with the value and the occurrence count.
// After the first pass, we will have a full-length array with the pairwise
// max indices, and two half-length temp arrays with the occurrence counts
// and the values.  (Although we don't necessarily need to store the values
// separately, because they can be obtained from the original input array.)
// The subsequent passes need to use temp max values (or use the indices
// to lookup) and then coalesce the indices.

// BUG - where do the gpu tests get used???

// This one takes an oap argument...  we need a vap!?

// These are unusual because the output array (indices) is a linear array,
// regardless of the shape of the input...
// Actually, that not need be true - this could be a form of a projection operator,
// where we could operate on all the rows of an image.  But let's defer that
// complication for the time being.
//
// The input array can be any shape, and shouldn't have to be contiguous...
//
// Here we insist that the index array have the same number of elements as the input.
// We could insist that it have the same shape?  Most of the time the number of occurrences
// will be 1...  Insisting on the same shape would simplify generalizing to projection,
// but might complicate other things...

#define H_CALL_MM_NOCC(name)							\
										\
										\
static void HOST_SLOW_CALL_NAME(name)(LINK_FUNC_ARG_DECLS)			\
{										\
	/*DECLARE_HOST_SLOW_CALL_VARS(,2)*/			\
	DECLARE_PLATFORM_VARS							\
	/*SETUP_KERNEL_SLOW_CALL(name,,typ,scalars,2)	\
	CALL_SLOW_KERNEL(name,bitmap,typ,scalars,vectors)*/			\
										\
	std_type *dst_values;							\
	index_type *dst_counts;							\
	std_type *src_values;							\
	index_type *src_counts;							\
	index_type *indices;							\
	std_type *dst_vals_to_free=NULL;					\
	std_type *src_vals_to_free=NULL;					\
	index_type *dst_counts_to_free=NULL;					\
	index_type *src_counts_to_free=NULL;					\
	u_long stride;								\
	u_long len, len1, len2;							\
	/*int max_threads_per_block;*/						\
										\
	/* BUG need to set things from vap */					\
	len = VA_SRC1_LEN(vap);							\
	src_values = (std_type *) VA_SRC1_PTR(vap);						\
										\
	/* Set len; sets std_arg2 to the input */				\
	/*INIT_MM(src_values)*/							\
										\
	/*max_threads_per_block = OBJ_MAX_THREADS_PER_BLOCK( OA_DEST(oap) );*/	\
	/* Set len1, len2 */							\
	/* sets std_arg3 to second half of input data */			\
	/* Set dst_values, dst_counts to temp vectors */			\
	SETUP_NOCC_ITERATION(name)						\
	dst_values = (std_type *) VA_DEST_PTR(vap);				\
	indices = (index_type *) TMPVEC_NAME(sizeof(index_type),len1,#name);	\
	CALL_GPU_NOCC_SETUP_FUNC(name)						\
/*(dst_values,dst_counts, std_arg,indices,len1,len2,max_threads_per_block); */	\
	FINISH_NOCC_ITERATION(name) 						\
										\
	stride = 4;								\
	while( len > 1 ){							\
		SETUP_NOCC_ITERATION(name)					\
		CALL_GPU_NOCC_HELPER_FUNC(name)					\
	/*(dst_values, dst_counts,src_values,src_counts, indices,len1,len2,stride,max_threads_per_block); */ \
		FINISH_NOCC_ITERATION(name)					\
		stride = 2*stride;						\
	}									\
	if( src_vals_to_free != NULL ){						\
		FREETMP_NAME(src_vals_to_free,#name);				\
		src_vals_to_free=NULL;						\
	}									\
	if( src_counts_to_free != NULL ){					\
		FREETMP_NAME(src_counts_to_free,#name);				\
		src_counts_to_free=NULL;					\
	}									\
	FREETMP_NAME(indices,#name);						\
	/* BUG need to copy final index to destination scalar */		\
}										\
										\
static void HOST_TYPED_CALL_NAME(name,type_code)(HOST_CALL_ARG_DECLS )		\
{										\
	Vector_Args va1, *vap=(&va1);						\
										\
	if( OBJ_MACH_PREC(OA_DEST(oap)) != INDEX_PREC ){			\
		sprintf(DEFAULT_ERROR_STRING,					\
	"%s:  destination index %s has %s precision, should be %s",		\
	#name,OBJ_NAME(OA_DEST(oap)),						\
	PREC_NAME(OBJ_MACH_PREC_PTR(OA_DEST(oap))),				\
			PREC_NAME(PREC_FOR_CODE(INDEX_PREC)) );			\
		NWARN(DEFAULT_ERROR_STRING);					\
		return;								\
	}									\
										\
	if( OBJ_N_TYPE_ELTS(OA_DEST(oap)) !=					\
		OBJ_N_TYPE_ELTS(oap->oa_dp[0]) ){				\
		sprintf(DEFAULT_ERROR_STRING,					\
"%s:  number of elements of index array %s (%d) must match source %s (%d)",	\
			#name, OBJ_NAME(OA_DEST(oap)),				\
			OBJ_N_TYPE_ELTS(OA_DEST(oap)),				\
			OBJ_NAME(oap->oa_dp[0]),				\
			OBJ_N_TYPE_ELTS(oap->oa_dp[0]));			\
		NWARN(DEFAULT_ERROR_STRING);					\
		return;								\
	}									\
	/* BUG need to set vap entries from oap */				\
	SET_MAX_THREADS_FROM_OBJ(OA_DEST(oap))					\
	CHECK_MM(name)								\
	HOST_SLOW_CALL_NAME(name)(LINK_FUNC_ARGS);				\
}


// vmaxv, vsum, vdot, etc
//
// Called "projection" because on host can compute entire sum if dest
// is a scalar, or row sums if dest is a column, etc...
//
// To compute the sum to a scalar, we have to divide and conquer
// as we did with vmaxv...

// BUG? gpu_expr? used?

#define SETUP_PROJ_ITERATION(type,name)						\
										\
	len1 = (len+1)/2;							\
	len2 = len - len1;							\
										\
	if( len1 == 1 ){							\
		dst_values = (type *) VA_DEST_PTR(vap);				\
		dst_to_free = NULL;						\
	} else {								\
		dst_values = (type *) TMPVEC_NAME(sizeof(type),len1,#name);	\
		dst_to_free = dst_values;					\
	}

// We can't have vmaxv for complex, no ordering relation!?
// But we do have cvsum!


// This started as the cuda version...

#define H_CALL_PROJ_2V( name, type )					\
									\
static void HOST_SLOW_CALL_NAME(name)(LINK_FUNC_ARG_DECLS)			\
{										\
	u_long len, len1, len2;						\
	type *src_values, *dst_values;					\
	type *src_to_free, *dst_to_free;				\
	DECLARE_PLATFORM_VARS						\
									\
	/*len = OBJ_N_TYPE_ELTS(oap->oa_dp[0]);*/			\
	len = VARG_LEN( VA_SRC(vap,0) );					\
	src_values = (type *) VA_SRC_PTR(vap,0);			\
									\
	/*max_threads_per_block = OBJ_MAX_THREADS_PER_BLOCK(oap->oa_dp[0]);*/\
	SET_MAX_THREADS_FROM_OBJ(oap->oa_dp[0])				\
	src_to_free=NULL;						\
	while( len > 1 ){						\
		SETUP_PROJ_ITERATION(type,name)				\
		CALL_GPU_PROJ_2V_FUNC(name)				\
		len=len1;						\
		src_values = dst_values;				\
		/* Each temp vector gets used twice,			\
		 * first as result, then as source */			\
		if( src_to_free != NULL ){				\
			FREETMP_NAME(src_to_free,#name);		\
			src_to_free=NULL;				\
		}							\
		src_to_free = dst_to_free;				\
		dst_to_free = NULL;					\
	}								\
	if( src_to_free != NULL ){					\
		FREETMP_NAME(src_to_free,#name);			\
		src_to_free=NULL;					\
	}								\
}									\
									\
static void HOST_TYPED_CALL_NAME(name,type_code)( HOST_CALL_ARG_DECLS )	\
{									\
	Vector_Args va1, *vap=(&va1);						\
										\
	CHECK_MM(name)							\
									\
	/* BUG need to set vap entries from oap */				\
	SET_MAX_THREADS_FROM_OBJ(OA_DEST(oap))					\
	HOST_SLOW_CALL_NAME(name)(LINK_FUNC_ARGS);				\
}


// vdot, cvdot, etc

#define H_CALL_PROJ_3V( name, type )					\
									\
static void HOST_SLOW_CALL_NAME(name)(LINK_FUNC_ARG_DECLS)		\
{									\
	type *dst_values;						\
	type *src1_values, *src2_values;				\
	u_long len, len1, len2;						\
	type *src_to_free, *dst_to_free;				\
	DECLARE_PLATFORM_VARS						\
									\
	len = VA_SRC1_LEN(vap);						\
	src1_values = (type *) VA_SRC1_PTR(vap);			\
	src2_values = (type *) VA_SRC2_PTR(vap);			\
									\
	/*max_threads_per_block = OBJ_MAX_THREADS_PER_BLOCK(oap->oa_dp[0]);*/\
	src_to_free=NULL;						\
	while( len > 1 ){						\
		SETUP_PROJ_ITERATION(type,name)				\
		CALL_GPU_PROJ_3V_FUNC(name)				\
		len = len1;						\
		src1_values = dst_values;				\
		/* Each temp vector gets used twice,			\
		 * first as result, then as source */			\
		if( src_to_free != NULL ){				\
			FREETMP_NAME(src_to_free,#name);		\
			src_to_free=NULL;				\
		}							\
		src_to_free = dst_to_free;				\
		dst_to_free = NULL;					\
	}								\
	if( src_to_free != NULL ){					\
		FREETMP_NAME(src_to_free,#name);			\
		src_to_free=NULL;					\
	}								\
}									\
									\
static void HOST_TYPED_CALL_NAME(name,type_code)( HOST_CALL_ARG_DECLS )	\
{									\
	Vector_Args va1, *vap=(&va1);						\
										\
	CHECK_MM(name)							\
									\
	/* BUG need to set vap entries from oap */				\
	/*SET_MAX_THREADS_FROM_OBJ(OA_DEST(oap))*/					\
	SET_MAX_THREADS_FROM_OBJ(oap->oa_dp[0])				\
	HOST_SLOW_CALL_NAME(name)(LINK_FUNC_ARGS);				\
}


