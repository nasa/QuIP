suppress_no
/* gpu_host_call_defs.m4 BEGIN */
suppress_if

my_include(`veclib/report_args.m4')


define(`DECLARE_HOST_FAST_CALL_VARS',`DECLARE_PLATFORM_FAST_VARS')

dnl DECLARE_HOST_EQSP_CALL_VARS(bitmap,vectors)
define(`DECLARE_HOST_EQSP_CALL_VARS',`

	DECLARE_HOST_FAST_CALL_VARS($1,$2)
	dnl `DECL_EQSP_INCRS_'$1$2
	DECL_EQSP_INCRS($1,$2)
')


dnl DECLARE_HOST_SLOW_CALL_VARS(bitmap,vectors)
define(`DECLARE_HOST_SLOW_CALL_VARS',`

	DECLARE_PLATFORM_SLOW_VARS
	/* declare_host_slow_call_vars $1 $2 */
	dnl `DECL_SLOW_INCRS_'$1$2
	DECL_SLOW_INCRS($1,$2)
')



/* SETUP_KERNEL is a nop for CUDA */


dnl GENERIC_HOST_FAST_CALL(name,bitmap,typ,scalars,vectors)
define(`GENERIC_HOST_FAST_CALL',`

static void HOST_FAST_CALL_NAME($1)(LINK_FUNC_ARG_DECLS)
{
DECLARE_HOST_FAST_CALL_VARS($2,$5)
SETUP_KERNEL_FAST_CALL($1,$2,$3,$4,$5)
CALL_FAST_KERNEL($1,$2,$3,$4,$5)
}
')

dnl GENERIC_HOST_FAST_CONV(name,bitmap,typ,type)
define(`GENERIC_HOST_FAST_CONV',`
/* Declare the host function here to call the kernel */

static void HOST_FAST_CALL_NAME($1)
( LINK_FUNC_ARG_DECLS )
{
	DECLARE_HOST_FAST_CALL_VARS($2,2)
	/* setup_kernel_fast_call_conv */
	SETUP_KERNEL_FAST_CALL_CONV($1,$2,$3,$4)
	/* call_fast_conv_kernel */
	CALL_FAST_CONV_KERNEL($1,$2,$3,$4)
}
')


dnl GENERIC_HOST_EQSP_CALL(name,bitmap,typ,scalars,vectors)
define(`GENERIC_HOST_EQSP_CALL',`

static void HOST_EQSP_CALL_NAME($1)(LINK_FUNC_ARG_DECLS)
{
	DECLARE_HOST_EQSP_CALL_VARS($2,$5)
	SETUP_KERNEL_EQSP_CALL($1,$2,$3,$4,$5)
	CALL_EQSP_KERNEL($1,$2,$3,$4,$5)
}
')

dnl GENERIC_HOST_EQSP_CONV(name,bitmap,typ,type)
define(`GENERIC_HOST_EQSP_CONV',`
/* Declare the host function here to call the kernel */

static void HOST_EQSP_CALL_NAME($1)
( LINK_FUNC_ARG_DECLS )
{
	DECLARE_HOST_EQSP_CALL_VARS($2,2)
	SETUP_KERNEL_EQSP_CALL_CONV($1,$2,$3,$4)
	CALL_EQSP_CONV_KERNEL($1,$2,$3,$4)
}
')

/* setup_slow_len must be called outside of here, because the
 * lengths are inside vap...
 */

dnl GENERIC_HOST_SLOW_CALL(name,bitmap,typ,scalars,vectors)
define(`GENERIC_HOST_SLOW_CALL',`

static void HOST_SLOW_CALL_NAME($1)(LINK_FUNC_ARG_DECLS)
{
	DECLARE_HOST_SLOW_CALL_VARS($2,$5)
	dnl `SETUP_SLOW_INCS_'$2$5
	SETUP_SLOW_INCS($2,$5)
	SETUP_KERNEL_SLOW_CALL($1,$2,$3,$4,$5)
	CALL_SLOW_KERNEL($1,$2,$3,$4,$5)
}
')

dnl GENERIC_HOST_SLOW_CONV(name,bitmap,typ,type)
define(`GENERIC_HOST_SLOW_CONV',`
/* Declare the host function here to call the kernel */

static void HOST_SLOW_CALL_NAME($1)
( LINK_FUNC_ARG_DECLS )
{
	DECLARE_HOST_SLOW_CALL_VARS($2,2)
	dnl `SETUP_SLOW_INCS_'$22
	SETUP_SLOW_INCS($2,2)
	SETUP_KERNEL_SLOW_CALL_CONV($1,$2,$3,$4)
	CALL_SLOW_CONV_KERNEL($1,$2,$3,$4)
}
')


// is this used?

dnl FINISH_MM_ITERATION(whence)
define(`FINISH_MM_ITERATION',`

	/* Each temp vector gets used twice,
	 * first as result, then as source */
	if( src_to_free != (src_type *)NULL ){
		FREETMP_NAME`(src_to_free,"$1");'
		src_to_free=(src_type *)NULL;
	}

	/* Now roll things over... */
	idx1_values = indices;
	len = len1;
	src_to_free = dst_to_free;
	dst_to_free = (dst_type *)NULL;
')

dnl FINISH_IDX_MM_ITERATION(whence)
define(`FINISH_IDX_MM_ITERATION',`

	/* Each temp vector gets used twice,
	 * first as result, then as source */
	if( src_to_free != (index_type *)NULL ){
		FREETMP_NAME`(src_to_free,"$1");'
		src_to_free=(index_type *)NULL;
	}

	/* Now roll things over... */
	idx1_values = indices;
	len = len1;
	src_to_free = dst_to_free;
	dst_to_free = (index_type *)NULL;
')

dnl SETUP_IDX_ITERATION(a2,a3,whence)
define(`SETUP_IDX_ITERATION',`

	len1 = (len+1)/2;
	len2 = len - len1;

	$2 = $1 + len1;

	if( len1 == 1 ){
		indices = (index_type *) VA_DEST_PTR(vap);
		dst_to_free = (index_type *)NULL;
	} else {
		indices = (index_type *) TMPVEC_NAME`(VA_PFDEV(vap),sizeof(index_type),len1,"$3")';
		dst_to_free = indices;
	}
')




dnl expands dest_type but not index_type!?

dnl FINISH_NOCC_ITERATION(whence)
define(`FINISH_NOCC_ITERATION',`

	/* Each temp vector gets used twice,
	 * first as result, then as source */
	if( src_vals_to_free != (dest_type *)NULL ){
		FREETMP_NAME`(src_vals_to_free,"$1");'
		src_vals_to_free=(dest_type *)NULL;
	}
	if( src_counts_to_free != (index_type *)NULL ){
		FREETMP_NAME`(src_counts_to_free,"$1");'
		src_counts_to_free=(index_type *)NULL;
	}

	/* Now roll things over... */
	src_values = dst_values;
	src_counts = dst_counts;
	len = len1;
	src_vals_to_free = dst_vals_to_free;
	src_counts_to_free = dst_counts_to_free;
	dst_vals_to_free = (dest_type *)NULL;
	dst_counts_to_free = (index_type *)NULL;
')

/* For vmaxg, the temp arrays don't double for the destination vector...
 * oa_sdp[0] is the extreme value, oa_sdp[1] is the count.
 */

dnl SETUP_NOCC_ITERATION(whence)
define(`SETUP_NOCC_ITERATION',`

	len1 = (len+1)/2;
	len2 = len - len1;

	if( len1 == 1 ){
		dst_values = (dest_type *) VA_SVAL1(vap);
		dst_vals_to_free = (dest_type *)NULL;
		dst_counts = (index_type *) VA_SVAL2(vap);
		dst_counts_to_free = (index_type *)NULL;
	} else {
		dst_values = (dest_type *) TMPVEC_NAME`('VA_PFDEV(vap),sizeof(dest_type),len1,"$1");
		dst_vals_to_free = (dest_type *) dst_values;
		dst_counts = (index_type *) TMPVEC_NAME`('VA_PFDEV(vap),sizeof(index_type),len1,"$1");
		dst_counts_to_free = dst_counts;
	}
')


dnl INIT_MM is not used?
dnl // the current implementations seem to require contiguity of all objects?

dnl INIT_MM(src_ptr)
dnl define(`INIT_MM(src_ptr)
dnl
dnl 	len = OBJ_N_TYPE_ELTS(oap->oa_dp[0]);
dnl 	orig_src_ptr = (std_type *)OBJ_DATA_PTR(oap->oa_dp[0]);
dnl
dnl 	CHECK_MM(min/max)
dnl
dnl CHECK_MM(name)
define(`CHECK_MM',`
 	INSIST_CONTIG( oap->oa_dp[0] , "$1" )
 	INSIST_LENGTH( OBJ_N_TYPE_ELTS(oap->oa_dp[0]), "$1" , OBJ_NAME(oap->oa_dp[0]) )
')


// The index functions wants to return the index of the max/min.
// After the first pass, we will have a half-length array with the
// max indices.  The subsequent passes need to use these indices to
// lookup data to make the comparison...


define(`H_CALL_PROJ_2V_IDX',`

static void HOST_FAST_CALL_NAME($1)(LINK_FUNC_ARG_DECLS)
{
	index_type *indices;
	std_type *src1_values;
	std_type *src2_values;
	std_type *orig_src_values;
	index_type *idx1_values,*idx2_values;
	index_type *dst_to_free=(index_type *)NULL;
	index_type *src_to_free=(index_type *)NULL;
	uint32_t len, len1, len2;
	DECLARE_PLATFORM_VARS_2

	src1_values = (std_type *) VA_SRC1_PTR(vap);
	len = VA_SRC1_LEN(vap);
	SETUP_IDX_ITERATION(src1_values,src2_values,$1)
	orig_src_values = src1_values;
	/*max_threads_per_block = OBJ_MAX_THREADS_PER_BLOCK(OA_DEST(oap));*/
	CALL_GPU_FAST_INDEX_SETUP_FUNC($1)
	FINISH_IDX_MM_ITERATION($1)

	while( len > 1 ){
		SETUP_IDX_ITERATION(idx1_values,idx2_values,$1)
		CALL_GPU_FAST_INDEX_HELPER_FUNC($1)
	/*(indices,idx1_values,idx2_values,orig_src_values,len1,len2)*/
		FINISH_IDX_MM_ITERATION($1)
	}
	if( src_to_free != (index_type *)NULL ){
		FREETMP_NAME`(src_to_free,"$1");'
		src_to_free=(index_type *)NULL;
	}
}

static void HOST_TYPED_CALL_NAME($1,type_code)(HOST_CALL_ARG_DECLS )
{
	Vector_Args va1, *vap=(&va1);

	if( OBJ_MACH_PREC(OA_DEST(oap)) != INDEX_PREC ){
		sprintf(DEFAULT_ERROR_STRING,
"%s:  destination index %s has %s precision, should be %s",
			STRINGIFY(HOST_TYPED_CALL_NAME($1,type_code)),
			OBJ_NAME(OA_DEST(oap)),
			PREC_NAME(OBJ_MACH_PREC_PTR(OA_DEST(oap))),
			PREC_NAME(PREC_FOR_CODE(INDEX_PREC)) );
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	CHECK_MM($1)
	/* BUG need to xfer args to vap - proj_2v_idx */
	SET_MAX_THREADS_FROM_OBJ(OA_DEST(oap))
fprintf(stderr,"h_call_proj_2v_idx %s:  need to implement speed switch!?\n","$1");

	HOST_FAST_CALL_NAME($1)(LINK_FUNC_ARGS);
}
')


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
//
// The index array *should* have the same shape as the input, with the indices accumulating
// in the dimensions(s) that are collapsed in the output extremum target.  (What about when multiple
// dimensions are collapsed?)

// The basic idea is recursive subdivision - each kernel compares two inputs, so the initial
// number of threads is half the number of inputs (rounded up).  We use temp objects for the max's...
//
// example:
//
// input:	10   11   12   13   14   15   16   17
//
// itr. 1	11       13       15       17		indices:  1   3   5   7		count:  1   1   1   1
// itr. 2	13                17         		indices:  3   7			count:  1   1
// itr. 3	17                           		indices:  7			count:  1

// BUG len is declared uint32_t, but later treated as index_type???

// What is stride???

dnl H_CALL_MM_NOCC(name)
define(`H_CALL_MM_NOCC',`


static void HOST_FAST_CALL_NAME($1)(LINK_FUNC_ARG_DECLS)
{
	DECLARE_PLATFORM_VARS_2

	dest_type *dst_values;
	index_type *dst_counts;
	std_type *orig_src_values;
	dest_type *src_values;
	index_type *src_counts;
	index_type *indices;
	dest_type *dst_vals_to_free=(dest_type *)NULL;
	dest_type *src_vals_to_free=(dest_type *)NULL;
	index_type *dst_counts_to_free=(int32_t *)NULL;
	index_type *src_counts_to_free=(int32_t *)NULL;
	uint32_t stride;
	uint32_t len, len1, len2;
	/*int max_threads_per_block;*/

	/* set things from vap */
	/*len = VA_SRC1_LEN(vap);*/
	len = VA_LENGTH(vap);
	orig_src_values = (std_type *) VA_SRC1_PTR(vap);

	/*max_threads_per_block = OBJ_MAX_THREADS_PER_BLOCK( OA_DEST(oap) );*/
	/* Set len1, len2 */
	/* sets std_arg3 to second half of input data */
	/* Set dst_values, dst_counts to temp vectors */
	SETUP_NOCC_ITERATION($1)
	indices = (index_type *) VA_DEST_PTR(vap);
	CALL_GPU_FAST_NOCC_SETUP_FUNC($1)
	/* finish nocc iteration */
	FINISH_NOCC_ITERATION($1)

	stride = 4;
	while( len > 1 ){
		SETUP_NOCC_ITERATION($1)
		/* call fast helper func */
		CALL_GPU_FAST_NOCC_HELPER_FUNC($1)
		/* finish nocc iteration */
		FINISH_NOCC_ITERATION($1)
		stride = 2*stride;
	}
	if( src_vals_to_free != (dest_type *)NULL ){
		FREETMP_NAME`(src_vals_to_free,"$1");'
		src_vals_to_free=(dest_type *)NULL;
	}
	if( src_counts_to_free != (int32_t *)NULL ){
		FREETMP_NAME`(src_counts_to_free,"$1");'
		src_counts_to_free=(int32_t *)NULL;
	}
	/*FREETMP_NAME`(indices,"$1");'*/
}

static void HOST_TYPED_CALL_NAME($1,type_code)(HOST_CALL_ARG_DECLS )
{
	Vector_Args va1, *vap=(&va1);

	/* BUG? - isnt precision check done elsewhere??? */
	if( OBJ_MACH_PREC(OA_DEST(oap)) != INDEX_PREC ){
		sprintf(DEFAULT_ERROR_STRING,
	"%s:  destination index %s has %s precision, should be %s",
	"$1",OBJ_NAME(OA_DEST(oap)),
	PREC_NAME(OBJ_MACH_PREC_PTR(OA_DEST(oap))),
			PREC_NAME(PREC_FOR_CODE(INDEX_PREC)) );
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	if( OBJ_N_TYPE_ELTS(OA_DEST(oap)) !=
		OBJ_N_TYPE_ELTS(oap->oa_dp[0]) ){
		sprintf(DEFAULT_ERROR_STRING,
"%s:  number of elements of index array %s (%d) must match source %s (%d)",
			"$1", OBJ_NAME(OA_DEST(oap)),
			OBJ_N_TYPE_ELTS(OA_DEST(oap)),
			OBJ_NAME(oap->oa_dp[0]),
			OBJ_N_TYPE_ELTS(oap->oa_dp[0]));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	/* BUG need to set vap entries from oap - mm_nocc */
	SET_MAX_THREADS_FROM_OBJ(OA_DEST(oap))
	CHECK_MM($1)
	memset(vap,0,sizeof(*vap)); /* needed only if will show? */
	SET_VA_PFDEV(vap,OA_PFDEV(oap));
	XFER_FAST_ARGS_NOCC
	SET_VA_LENGTH(vap,OBJ_N_TYPE_ELTS(OA_SRC1(oap)));
	HOST_FAST_CALL_NAME($1)(LINK_FUNC_ARGS);
}
')


// vmaxv, vsum, vdot, etc
//
// Called "projection" because on host can compute entire sum if dest
// is a scalar, or row sums if dest is a column, etc...
//
// To compute the sum to a scalar, we have to divide and conquer
// as we did with vmaxv...
//
// This looks wrong - the destination does not have to be a scalar!?
//
// We can have a fast version when the source is contiguous and the destination is a scalar,
// we don't need anything except lengths...
//
// For non-scalar outputs (projections) it is more complicated...
// We do the same things, but with more complicated indexing calculations.
//
// i11 i12 i13 i14 i15	->	max(i11,i21)	max(i12,i22)	max(i13,i23)	max(i14,i24)	max(i15,i25)
// i21 i22 i23 i24 i25		max(i31,i41)	max(i32,i42)	max(i33,i43)	max(i34,i44)	max(i35,i45)
// i31 i32 i33 i34 i35
// i41 i42 i43 i44 i45
//
//				d11		d12		d13		d14		d15
//				d21		d22		d23		d24		d25
//
// We could collapse dimensions one by one...

// BUG? gpu_expr? used?

dnl SETUP_PROJ_ITERATION(type,name)
define(`SETUP_PROJ_ITERATION',`

	len1 = (len+1)/2;
	len2 = len - len1;

	if( len1 == 1 ){
		dst_values = ($1 *) VA_DEST_PTR(vap);
		dst_to_free = (dst_type *)NULL;
	} else {
dnl fprintf(stderr,"setup_proj_iteration $2:  allocating temp vec\n");
		dst_values = ($1 *) TMPVEC_NAME`(VA_PFDEV(vap),sizeof($1),len1,"$2")';
dnl fprintf(stderr,"setup_proj_iteration $2:  DONE allocating temp vec\n");
		dst_to_free = dst_values;
	}
')

// We can't have vmaxv for complex, no ordering relation!?
// But we do have cvsum!


// This started as the cuda version...
// where do we set dst_values for the first iteration???  ans:  in `SETUP_PROJ_ITERATION'...
//
// The idea of the original implementation was this:  we can't find the max in parallel, so instead
// we divide and conquer:  we split the input data in two, and then store the pairwise maxima
// to a temporary array.  We repeat on the temporary array, until we have a single value.
//
// That works fine if the final target is a scalar, but what if we are projecting an image
// to a row or column?  Let's analyze the case of projecting an image to a row...  Following the
// analogy, we would divide the image into the top and bottom halves, and then store the pairwise
// maxima to a temporary half-size image, and so on...  We have a problem because the Vector_Args
// struct doesn't contain shape information!?

// We need setup and helper functions in order to support mixed-precision versions (taking sum
// to higher precision).

dnl H_CALL_PROJ_2V(name,dest_type,src_type)

define(`H_CALL_PROJ_2V',`

/* h_call_proj_2v - $1   dtype = $2   stype = $3 */

static void HOST_FAST_CALL_NAME($1)(LINK_FUNC_ARG_DECLS)
{
	uint32_t len, len1, len2;
	$2 *src_values, *dst_values;
	$2 *src_to_free, *dst_to_free;
	$3 *orig_src_values;
	DECLARE_PLATFORM_VARS_2

	len = VARG_LEN( VA_SRC(vap,0) );
dnl	show_vec_args(vap);
	orig_src_values = ($3 *) VA_SRC_PTR(vap,0);

	/*max_threads_per_block = OBJ_MAX_THREADS_PER_BLOCK(oap->oa_dp[0]);*/
	SET_MAX_THREADS_FROM_OBJ(oap->oa_dp[0])
	src_to_free=($2 *)NULL;
	SETUP_PROJ_ITERATION($2,$1)
	CALL_GPU_FAST_PROJ_2V_SETUP_FUNC($1)
dnl	fprintf(stderr,"after setup:\n");
dnl	show_gpu_vector(DEFAULT_QSP_ARG  VA_PFDEV(vap), dst_values,len1);
	len=len1;
	src_values = dst_values;
	while( len > 1 ){
		SETUP_PROJ_ITERATION($2,$1)
		CALL_GPU_FAST_PROJ_2V_HELPER_FUNC($1)
dnl	fprintf(stderr,"after helper:\n");
dnl	show_gpu_vector(DEFAULT_QSP_ARG  VA_PFDEV(vap), dst_values,len1);
		len=len1;
		src_values = dst_values;
		/* Each temp vector gets used twice,
		 * first as result, then as source */
		if( src_to_free != ($2 *)NULL ){
			FREETMP_NAME`(src_to_free,"$1");'
			src_to_free=($2 *)NULL;
		}
		src_to_free = dst_to_free;
		dst_to_free = ($2 *)NULL;
	}
	if( src_to_free != ($2 *)NULL ){
		FREETMP_NAME`(src_to_free,"$1");'
		src_to_free=($2 *)NULL;
	}
}

static void HOST_TYPED_CALL_NAME($1,type_code)( HOST_CALL_ARG_DECLS )
{
	Vector_Args va1, *vap=(&va1);

	CHECK_MM($1)

	CLEAR_VEC_ARGS(vap) /* mostly for debugging */
	SET_MAX_THREADS_FROM_OBJ(OA_DEST(oap))
	/* BUG need to set vap entries from oap - proj_2v */
	SET_VA_PFDEV(vap,OA_PFDEV(oap));
dnl	show_obj_args(DEFAULT_QSP_ARG  oap);
	/* why slow args? */
	XFER_SLOW_ARGS_2
	SETUP_SLOW_LEN_2
	/* BUG need to have a speed switch!? */
dnl	fprintf(stderr,"h_call_proj_2v %s:  need to implement speed switch, calling fast function!?\n","$1");
	HOST_FAST_CALL_NAME($1)(LINK_FUNC_ARGS);
}
')

// vdot, cvdot, etc
// The following is messed up!
// The "setup" function should to the element-wise products to a temp area (which may be large),
// then the "helper" function should sum those.  Best approach would be to use vmul and vsum...

dnl	 H_CALL_PROJ_3V( name, dtype, stype, mul_func, sum_func )
define(`H_CALL_PROJ_3V',`

dnl	static void HOST_FAST_CALL_NAME($1)(LINK_FUNC_ARG_DECLS)
dnl	{
dnl		$2 *dst_values;
dnl		$3 *orig_src1_values, *orig_src2_values;
dnl		$2 *src1_values, *src2_values;
dnl		uint32_t len, len1, len2;
dnl		$2 *src_to_free;
dnl		$2 *dst_to_free;
dnl		DECLARE_PLATFORM_VARS_2
dnl	
dnl	fprintf(stderr,"h_call_proj_3v fast $1:  BEGIN, vap = 0x%lx\n",(long)vap);
dnl		//len = VA_SRC1_LEN(vap);
dnl		len = VA_LENGTH(vap);
dnl	fprintf(stderr,"h_call_proj_3v fast $1:  BEGIN, len = 0x%x\n",len);
dnl		orig_src1_values = ($3 *) VA_SRC1_PTR(vap);
dnl		orig_src2_values = ($3 *) VA_SRC2_PTR(vap);
dnl	fprintf(stderr,"h_call_proj_3v $1:  args set\n");
dnl	
dnl		/*max_threads_per_block = OBJ_MAX_THREADS_PER_BLOCK(oap->oa_dp[0]);*/
dnl		/* first iteration may be mixed types... */
dnl		SETUP_PROJ_ITERATION($2,$1)
dnl	fprintf(stderr,"h_call_proj_3v $1:  calling setup func, len = %d\n",len);
dnl		CALL_GPU_FAST_PROJ_3V_SETUP_FUNC($1)
dnl		len = len1;
dnl		src_to_free=(dst_type *)NULL;
dnl		src1_values = dst_values;
dnl		while( len > 1 ){
dnl			SETUP_PROJ_ITERATION($2,$1)
dnl	fprintf(stderr,"h_call_proj_3v $1:  calling helper func, len = %d\n",len);
dnl			CALL_GPU_FAST_PROJ_3V_HELPER_FUNC($1)
dnl			src1_values = dst_values;
dnl			/* Each temp vector gets used twice,
dnl			 * first as result, then as source */
dnl			if( src_to_free != (dst_type *)NULL ){
dnl				FREETMP_NAME`(src_to_free,"$1");'
dnl				src_to_free=(dst_type *)NULL;
dnl			}
dnl			src_to_free = dst_to_free;
dnl			dst_to_free = (dst_type *)NULL;
dnl		}
dnl		if( src_to_free != (dst_type *)NULL ){
dnl			FREETMP_NAME`(src_to_free,"$1");'
dnl			src_to_free=(dst_type *)NULL;
dnl		}
dnl	}

static void HOST_TYPED_CALL_NAME($1,type_code)( HOST_CALL_ARG_DECLS )
{
dnl	Vector_Args va1, *vap=(&va1);
	Shape_Info *shpp;
	Data_Obj *prod_dp;
dnl	Data_Obj *disp_dp;	// for debugging
	Vec_Obj_Args oa1, *prod_oap=(&oa1);

	shpp = make_outer_shape(DEFAULT_QSP_ARG  OBJ_SHAPE(OA_SRC1(oap)), OBJ_SHAPE(OA_SRC2(oap)));
	if( shpp == NULL ){
		NWARN("$1:  incompatible operand shapes!?");
		return;
	}

	/* BUG - to make this thread-safe, we need to append the thread index to the name! */
	prod_dp = make_dobj(DEFAULT_QSP_ARG   "prod_tmp",SHP_TYPE_DIMS(shpp),SHP_PREC_PTR(shpp));
	if( prod_dp == NULL ){
		NWARN("$1:  error creating temp object for products!?");
		return;
	}

	/*
	clear_oargs(prod_oap);
	SET_OA_PFDEV(prod_oap,OA_PFDEV(oap));
	SET_OA_DEST(prod_oap,prod_dp);
	SET_OA_SRC1(prod_oap,OA_SRC1(oap));
	SET_OA_SRC2(prod_oap,OA_SRC2(oap));
	*/

	*prod_oap = *oap;
	SET_OA_DEST(prod_oap,prod_dp);
	HOST_TYPED_CALL_NAME($4,type_code)(FVMUL,prod_oap);

dnl	disp_dp = insure_ram_obj(prod_dp);
dnl	assert(disp_dp!=NULL);
dnl	fprintf(stderr,"displaying intermediate result:\n");
dnl	pntvec(DEFAULT_QSP_ARG  disp_dp,stderr);
dnl	fflush(stderr);
dnl	_delvec(DEFAULT_QSP_ARG  disp_dp);

	*prod_oap = *oap;
	SET_OA_SRC1(prod_oap,prod_dp);
	SET_OA_SRC2(prod_oap,NULL);
	HOST_TYPED_CALL_NAME($5,type_code)(FVSUM,prod_oap);

	_delvec(DEFAULT_QSP_ARG  prod_dp);

dnl		CHECK_MM($1)
dnl	
dnl		/* BUG need to set vap entries from oap - proj_3v */
dnl		/*SET_MAX_THREADS_FROM_OBJ(OA_DEST(oap))*/
dnl		SET_MAX_THREADS_FROM_OBJ(oap->oa_dp[0])
dnl	fprintf(stderr,"h_call_proj_3v %s:  need to implement speed switch!?\n","$1");
dnl		/* Speed switch should determine the args to copy!? */
dnl		SET_VA_PFDEV(vap,OA_PFDEV(oap));
dnl	fprintf(stderr,"h_call_proj_3v $1:  pfdev set\n");
dnl		XFER_DEST_PTR
dnl	fprintf(stderr,"h_call_proj_3v $1:  dest ptr set\n");
dnl		XFER_FAST_ARGS_2SRCS
dnl	fprintf(stderr,"h_call_proj_3v $1:  src ptrs set\n");
dnl		XFER_FAST_COUNT(SRC1_DP)
dnl	fprintf(stderr,"h_call_proj_3v $1:  count set\n");
dnl	
dnl		HOST_FAST_CALL_NAME($1)(LINK_FUNC_ARGS);

}
')



suppress_no
/* gpu_host_call_defs.m4 END */

