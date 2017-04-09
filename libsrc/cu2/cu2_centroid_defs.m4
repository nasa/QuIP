
dnl	HCF( t )
define(`HCF',`HOST_CENT_FUNC($1)')


dnl	HOST_CENT_FUNC( typ )

define(`HOST_CENT_FUNC',`

void $1`_cent'(LINK_FUNC_ARG_DECLS)
{
	DECLARE_PLATFORM_VARS
	/*BLOCK_VARS_DECLS*/
	dim3 len;
	/*int max_threads_per_block;*/
	DECL_SLOW_INCRS_3

	/*max_threads_per_block =	curr_cdp->cudev_prop.maxThreadsPerBlock;*/
	CLEAR_CUDA_ERROR(type_code`_slow_cent_helper')
	/*XFER_SLOW_LEN_3*/
	/*SETUP_BLOCKS_XYZ_(VA_PFDEV(vap))*/
	SETUP_SLOW_INCS_3
	REPORT_THREAD_INFO
/*REPORT_ARGS_3*/
	$1`_slow_cent_helper'<<< NN_GPU >>>
		(KERN_ARGS_SLEN_3 );
    	CUDA_ERROR_CHECK("kernel launch failure");
}

/* Now the entry point */

void $1`_cuda_centroid'(HOST_CALL_ARG_DECLS)
{
	Vector_Args va1;
	Vector_Args *vap=(&va1);
	/*Spacing_Info spi1;*/
	/*Size_Info szi1;*/

	/*SET_VA_SPACING(vap,&spi1);*/
	/*SET_VA_SIZE_INFO(vap,&szi1);*/
	insure_cuda_device( oap->oa_dest );
	XFER_SLOW_ARGS_3
	SETUP_SLOW_LEN_3
	CHAIN_CHECK( $1`_cent' )
	if( is_chaining ){
		if( insure_static(oap) < 0 ) return;
		add_link( & $1`_cent' , LINK_FUNC_ARGS );
		return;
	} else {
		$1`_cent'(LINK_FUNC_ARGS);
		SET_OBJ_FLAG_BITS(OA_DEST(oap), DT_ASSIGNED);
		/* WHY set assigned flag on a source obj??? */
		/* Maybe because its really a second destination? */
		SET_OBJ_FLAG_BITS(OA_SRC1(oap), DT_ASSIGNED);
	}
}
')

dnl		kernel macros

dnl	 CENT_KERNEL( typ )
define(`CENT_KERNEL',`

KERNEL_FUNC_QUALIFIER void $1`_slow_cent_helper'
( /*std_type *x_array, dim3 inc1, std_type *y_array, dim3 inc2,
	std_type *input, dim3 inc3, dim3 len*/ DECLARE_KERN_ARGS_SLEN_3 )

{
	/*dim3 index;*/
	/*uint32_t offset1, offset2, offset3;*/
	DECL_INDICES_3
	std_type p;

	/*index.x = blockIdx.x * blockDim.x + threadIdx.x;*/
	/*index.y = blockIdx.y * blockDim.y + threadIdx.y;*/
	SET_INDICES_3

	/*offset1 = index.y * inc1.x + index.x;*/
	/*offset2 = index.y * inc2.x + index.x;*/
	/*offset3 = index.y * inc3.x + index.x;*/

	/*p = *( c + offset3);*/
	p = slow_src2;	/* third arg, no first source */
	slow_dst1 = p * index3.d5_dim[1]; /* x */
	slow_dst2 = p * index3.d5_dim[2]; /* y */
	/* *(a+offset1) = p * index.x; */
	/* *(b+offset2) = p * index.y; */
}
')

dnl	 CK( typ )
define(`CK',`CENT_KERNEL($1)')

