#ifndef _GPU_CALL_UTILS_H_
#define _GPU_CALL_UTILS_H_

// This file contains macros that are useful for writing kernels...
// Because the kernels are static, the names don't need to include the
// platform name...

/******************** VFUNC_XXXX_NAME **********************/

#define VFUNC_FAST_NAME(name)	_VFUNC_FAST_NAME(name,pf_str,type_code)
#define _VFUNC_FAST_NAME(n,pf,ty)	__VFUNC_FAST_NAME(n,pf,ty)
#define __VFUNC_FAST_NAME(n,pf,ty)	g_##pf##_fast_##ty##_##n

#define VFUNC_EQSP_NAME(name)	_VFUNC_EQSP_NAME(name,pf_str,type_code)
#define _VFUNC_EQSP_NAME(n,pf,ty)	__VFUNC_EQSP_NAME(n,pf,ty)
#define __VFUNC_EQSP_NAME(n,pf,ty)	g_##pf##_eqsp_##ty##_##n

#define VFUNC_SLOW_NAME(name)	_VFUNC_SLOW_NAME(name,pf_str,type_code)
#define _VFUNC_SLOW_NAME(n,pf,ty)	__VFUNC_SLOW_NAME(n,pf,ty)
#define __VFUNC_SLOW_NAME(n,pf,ty)	g_##pf##_slow_##ty##_##n

#define VFUNC_FLEN_NAME(name)	_VFUNC_FLEN_NAME(name,pf_str,type_code)
#define _VFUNC_FLEN_NAME(n,pf,ty)	__VFUNC_FLEN_NAME(n,pf,ty)
#define __VFUNC_FLEN_NAME(n,pf,ty)	g_##pf##_flen_##ty##_##n

#define VFUNC_ELEN_NAME(name)	_VFUNC_ELEN_NAME(name,pf_str,type_code)
#define _VFUNC_ELEN_NAME(n,pf,ty)	__VFUNC_ELEN_NAME(n,pf,ty)
#define __VFUNC_ELEN_NAME(n,pf,ty)	g_##pf##_elen_##ty##_##n

#define VFUNC_SLEN_NAME(name)	_VFUNC_SLEN_NAME(name,pf_str,type_code)
#define _VFUNC_SLEN_NAME(n,pf,ty)	__VFUNC_SLEN_NAME(n,pf,ty)
#define __VFUNC_SLEN_NAME(n,pf,ty)	g_##pf##_slen_##ty##_##n

#define VFUNC_NOCC_SETUP_NAME(func_name)	_VFUNC_NOCC_SETUP_NAME(func_name,pf_str,type_code)
#define _VFUNC_NOCC_SETUP_NAME(n,pf,ty)	__VFUNC_NOCC_SETUP_NAME(n,pf,ty)
#define __VFUNC_NOCC_SETUP_NAME(n,pf,ty)	g_##pf##_##ty##_##n##_nocc_setup

#define VFUNC_NOCC_HELPER_NAME(func_name)	_VFUNC_NOCC_HELPER_NAME(func_name,pf_str,type_code)
#define _VFUNC_NOCC_HELPER_NAME(n,pf,ty)	__VFUNC_NOCC_HELPER_NAME(n,pf,ty)
#define __VFUNC_NOCC_HELPER_NAME(n,pf,ty)	g_##pf##_##ty##_##n##_nocc_helper

#define VFUNC_SIMPLE_NAME(func_name)		_VFUNC_SIMPLE_NAME(func_name,pf_str,type_code)
#define _VFUNC_SIMPLE_NAME(n,pf,ty)		__VFUNC_SIMPLE_NAME(n,pf,ty)
#define __VFUNC_SIMPLE_NAME(n,pf,ty)		g_##pf##_##ty##_##n

#define VFUNC_HELPER_NAME(func_name)		_VFUNC_HELPER_NAME(func_name,pf_str,type_code)
#define _VFUNC_HELPER_NAME(n,pf,ty)		__VFUNC_HELPER_NAME(n,pf,ty)
#define __VFUNC_HELPER_NAME(n,pf,ty)		g_##pf##_##ty##_##n##_helper

#define VFUNC_IDX_SETUP_NAME(func_name)		_VFUNC_IDX_SETUP_NAME(func_name,pf_str,type_code)
#define _VFUNC_IDX_SETUP_NAME(n,pf,ty)		__VFUNC_IDX_SETUP_NAME(n,pf,ty)
#define __VFUNC_IDX_SETUP_NAME(n,pf,ty)		g_##pf##_##ty##_##n##_index_setup

#define VFUNC_IDX_HELPER_NAME(func_name)	_VFUNC_IDX_HELPER_NAME(func_name,pf_str,type_code)
#define _VFUNC_IDX_HELPER_NAME(n,pf,ty)	__VFUNC_IDX_HELPER_NAME(n,pf,ty)
#define __VFUNC_IDX_HELPER_NAME(n,pf,ty)	g_##pf##_##ty##_##n##_index_helper



/****************** DECL_INDICES ***********************/

#ifdef BUILD_FOR_CUDA
#define GPU_INDEX_TYPE	DIM3
#endif // BUILD_FOR_CUDA

#ifdef BUILD_FOR_OPENCL
#define GPU_INDEX_TYPE	DIM3
#endif // BUILD_FOR_OPENCL

#define DECL_INDICES_1		GPU_INDEX_TYPE index1;
#define DECL_INDICES_SRC1	GPU_INDEX_TYPE index2;
#define DECL_INDICES_SRC2	GPU_INDEX_TYPE index3;
#define DECL_INDICES_SRC3	GPU_INDEX_TYPE index4;
#define DECL_INDICES_SRC4	GPU_INDEX_TYPE index5;
#define DECL_INDICES_SBM	GPU_INDEX_TYPE sbmi;

#define DECL_INDICES_DBM	GPU_INDEX_TYPE dbmi; int i_dbm_bit;	\
				int i_dbm_word; bitmap_word dbm_bit;

#define DECL_INDICES_2		DECL_INDICES_1 DECL_INDICES_SRC1
#define DECL_INDICES_3		DECL_INDICES_2 DECL_INDICES_SRC2
#define DECL_INDICES_4		DECL_INDICES_3 DECL_INDICES_SRC3
#define DECL_INDICES_5		DECL_INDICES_4 DECL_INDICES_SRC4
#define DECL_INDICES_1SRC	DECL_INDICES_SRC1
#define DECL_INDICES_2SRCS	DECL_INDICES_SRC1 DECL_INDICES_SRC2
#define DECL_INDICES_SBM_1	DECL_INDICES_1 DECL_INDICES_SBM
#define DECL_INDICES_SBM_2	DECL_INDICES_2 DECL_INDICES_SBM
#define DECL_INDICES_SBM_3	DECL_INDICES_3 DECL_INDICES_SBM

#define DECL_INDICES_DBM_	DECL_INDICES_DBM
#define DECL_INDICES_DBM_1SRC	DECL_INDICES_1SRC DECL_INDICES_DBM
#define DECL_INDICES_DBM_2SRCS	DECL_INDICES_2SRCS DECL_INDICES_DBM
#define DECL_INDICES_DBM_SBM	DECL_INDICES_SBM DECL_INDICES_DBM


#define DECL_EXTRA_
#define DECL_EXTRA_T1	std_type r; std_type theta; std_type arg;
//#define DECL_EXTRA_T2	std_type std_tmp;
#define DECL_EXTRA_T2	std_cpx tmpc;
//#define DECL_EXTRA_T3	std_type std_tmp, tmp_denom;
#define DECL_EXTRA_T3	std_cpx tmpc; std_type tmp_denom;

// quaternion helpers
#define DECL_EXTRA_T4	std_quat tmpq; /*std_type tmp_denom;*/
#define DECL_EXTRA_T5	std_quat tmpq; std_type tmp_denom;

/*********************** INIT_INDICES *****************/

#define INIT_INDICES_1		DECL_INDICES_1 SET_INDICES_1
#define INIT_INDICES_2		DECL_INDICES_2 SET_INDICES_2
#define INIT_INDICES_2		DECL_INDICES_2 SET_INDICES_2
#define INIT_INDICES_3		DECL_INDICES_3 SET_INDICES_3
#define INIT_INDICES_4		DECL_INDICES_4 SET_INDICES_4
#define INIT_INDICES_5		DECL_INDICES_5 SET_INDICES_5

#define INIT_INDICES_2SRCS	DECL_INDICES_2SRCS SET_INDICES_2SRCS
#define INIT_INDICES_SBM_1	DECL_INDICES_SBM_1 SET_INDICES_SBM_1
#define INIT_INDICES_SBM_2	DECL_INDICES_SBM_2 SET_INDICES_SBM_2
#define INIT_INDICES_SBM_3	DECL_INDICES_SBM_3 SET_INDICES_SBM_3

#define INIT_INDICES_DBM_	DECL_INDICES_DBM_ SET_INDICES_DBM_
#define INIT_INDICES_DBM_2SRCS	DECL_INDICES_DBM_2SRCS SET_INDICES_DBM_2SRCS
#define INIT_INDICES_DBM_1SRC	DECL_INDICES_DBM_1SRC SET_INDICES_DBM_1SRC
#define INIT_INDICES_DBM_SBM	DECL_INDICES_DBM_SBM SET_INDICES_DBM_SBM

#define INIT_INDICES_XYZ_1	DECL_INDICES_1 SET_INDICES_XYZ_1
#define INIT_INDICES_XYZ_2	DECL_INDICES_2 SET_INDICES_XYZ_2
#define INIT_INDICES_XYZ_3	DECL_INDICES_3 SET_INDICES_XYZ_3
#define INIT_INDICES_XYZ_4	DECL_INDICES_4 SET_INDICES_XYZ_4
#define INIT_INDICES_XYZ_5	DECL_INDICES_5 SET_INDICES_XYZ_5

#define INIT_INDICES_XYZ_SBM_1	DECL_INDICES_SBM_1 SET_INDICES_XYZ_SBM_1
#define INIT_INDICES_XYZ_SBM_2	DECL_INDICES_SBM_2 SET_INDICES_XYZ_SBM_2
#define INIT_INDICES_XYZ_SBM_3	DECL_INDICES_SBM_3 SET_INDICES_XYZ_SBM_3

#define INIT_INDICES_XYZ_DBM_		DECL_INDICES_DBM_ SET_INDICES_XYZ_DBM_
#define INIT_INDICES_XYZ_DBM_1SRC	DECL_INDICES_DBM_1SRC SET_INDICES_XYZ_DBM_1SRC
#define INIT_INDICES_XYZ_DBM_2SRCS	DECL_INDICES_DBM_2SRCS SET_INDICES_XYZ_DBM_2SRCS
#define INIT_INDICES_XYZ_DBM_SBM	DECL_INDICES_DBM_SBM SET_INDICES_XYZ_DBM_SBM


/******************** SET_INDICES ***************************/

#ifdef BUILD_FOR_OPENCL

#define SET_INDEX( which_index )					\
									\
	which_index.x = get_global_id(0);				\
	which_index.y = which_index.z = 0;

#endif // BUILD_FOR_OPENCL

#ifdef BUILD_FOR_CUDA

#define SET_INDEX( which_index )					\
									\
		which_index.x = blockIdx.x * blockDim.x + threadIdx.x;	\
		which_index.y = which_index.z = 0;

#endif // BUILD_FOR_CUDA

#define SET_INDICES_1		SET_INDEX( index1 )
#define SET_INDICES_SRC1(dst_idx)	index2 = dst_idx;
#define SET_INDICES_SRC2	index3 = index2;
#define SET_INDICES_SRC3	index4 = index3;
#define SET_INDICES_SRC4	index5 = index4;
#define SET_INDICES_SBM		sbmi = index1;

#define SET_INDICES_2		SET_INDICES_1 SET_INDICES_SRC1(index1)
#define SET_INDICES_3		SET_INDICES_2 SET_INDICES_SRC2
#define SET_INDICES_4		SET_INDICES_3 SET_INDICES_SRC3
#define SET_INDICES_5		SET_INDICES_4 SET_INDICES_SRC4
#define SET_INDICES_2SRCS	SET_INDEX(index2) SET_INDICES_SRC2


#define SET_INDICES_SBM_1	SET_INDICES_1 SET_INDICES_SBM
#define SET_INDICES_SBM_2	SET_INDICES_2 SET_INDICES_SBM
#define SET_INDICES_SBM_3	SET_INDICES_3 SET_INDICES_SBM

#define SET_INDICES_DBM		SET_INDEX(dbmi)				\
				i_dbm_word = dbmi.x;			\
				dbmi.x *= BITS_PER_BITMAP_WORD;

#define SET_INDICES_DBM_	SET_INDICES_DBM

// BUG?  this looks wrong!?
// 1SRC is only used with dbm?
#define SET_INDICES_1SRC	index2 = dbmi;

#define SET_INDICES_DBM_1SRC	SET_INDICES_DBM SET_INDICES_1SRC
#define SET_INDICES_DBM_2SRCS	SET_INDICES_DBM_1SRC SET_INDICES_SRC2
// Can't use SET_INDICES_SBM here...
#define SET_INDICES_DBM_SBM	SET_INDICES_DBM sbmi = dbmi;

/**************************** SET_INDICES_XYZ ******************************/


#ifdef BUILD_FOR_CUDA

#if CUDA_COMP_CAP < 20

#define SET_INDEX_XYZ( which_index )					\
									\
	which_index.x = blockIdx.x * blockDim.x + threadIdx.x;		\
	which_index.y = blockIdx.y * blockDim.y + threadIdx.y;		\
	which_index.z = 0;


#else /* CUDA_COMP_CAP >= 20 */

#define SET_INDEX_XYZ( which_index )					\
									\
	which_index.x = blockIdx.x * blockDim.x + threadIdx.x;		\
	which_index.y = blockIdx.y * blockDim.y + threadIdx.y;		\
	which_index.z = blockIdx.z * blockDim.z + threadIdx.z;

#endif /* CUDA_COMP_CAP >= 20 */

#endif // BUILD_FOR_CUDA


#ifdef BUILD_FOR_OPENCL

#define SET_INDEX_XYZ( which_index )					\
									\
	which_index.x = get_global_id(0);				\
	which_index.y = get_global_id(1);				\
	which_index.z = get_global_id(2);

#endif // BUILD_FOR_OPENCL

#define SET_INDICES_XYZ_1	SET_INDEX_XYZ(index1)
#define SET_INDICES_XYZ_SRC1(dst_idx)	index2 = dst_idx;
#define SET_INDICES_XYZ_SRC2	index3 = index2;
#define SET_INDICES_XYZ_SRC3	index4 = index1;
#define SET_INDICES_XYZ_SRC4	index5 = index1;
#define SET_INDICES_XYZ_2	SET_INDICES_XYZ_1 SET_INDICES_XYZ_SRC1(index1)
#define SET_INDICES_XYZ_3	SET_INDICES_XYZ_2 SET_INDICES_XYZ_SRC2
#define SET_INDICES_XYZ_4	SET_INDICES_XYZ_3 SET_INDICES_XYZ_SRC3
#define SET_INDICES_XYZ_5	SET_INDICES_XYZ_4 SET_INDICES_XYZ_SRC4
#define SET_INDICES_XYZ_1SRC	SET_INDEX_XYZ(index2)
#define SET_INDICES_XYZ_2SRCS	SET_INDICES_XYZ_1SRC SET_INDICES_XYZ_SRC2
#define SET_INDICES_XYZ_SBM_1	SET_INDICES_XYZ_1 SET_INDICES_XYZ_SBM
#define SET_INDICES_XYZ_SBM_2	SET_INDICES_XYZ_2 SET_INDICES_XYZ_SBM
#define SET_INDICES_XYZ_SBM_3	SET_INDICES_XYZ_3 SET_INDICES_XYZ_SBM
#define SET_INDICES_XYZ_DBM_	SET_INDICES_XYZ_DBM
// This looks wrong:
//#define SET_INDICES_XYZ_DBM_1SRC	SET_INDICES_XYZ_DBM index2=bmi;
// Maybe correct?  BUG?:
#define SET_INDICES_XYZ_DBM_1SRC	SET_INDICES_XYZ_DBM SET_INDICES_XYZ_SRC1(dbmi)
#define SET_INDICES_XYZ_DBM_2SRCS	SET_INDICES_XYZ_DBM_1SRC SET_INDICES_XYZ_SRC2
#define SET_INDICES_XYZ_DBM_SBM	SET_INDICES_XYZ_DBM sbmi = dbmi;

/* BUG? is bmi set correctly? Is len.x the divided length?  or all the pixels? */
#define SET_INDICES_XYZ_SBM	sbmi = index1;

#define SET_INDICES_XYZ_DBM	SET_INDEX_XYZ(dbmi)	\
				i_dbm_word = dbmi.x;	\
				dbmi.x *= BITS_PER_BITMAP_WORD;

/**************** SCALE_INDICES_ ********************/

#define SCALE_INDICES_1		index1.x *= inc1;
#define SCALE_INDICES_SRC1	index2.x *= inc2;
#define SCALE_INDICES_SRC2	index3.x *= inc3;
#define SCALE_INDICES_SRC3	index4.x *= inc4;
#define SCALE_INDICES_SRC4	index5.x *= inc5;
#define SCALE_INDICES_SBM	sbmi.x *= sbm_inc;
#define SCALE_INDICES_DBM	dbmi.x *= dbm_inc;

#define SCALE_INDICES_2		SCALE_INDICES_1 SCALE_INDICES_SRC1
#define SCALE_INDICES_3		SCALE_INDICES_2 SCALE_INDICES_SRC2
#define SCALE_INDICES_4		SCALE_INDICES_3 SCALE_INDICES_SRC3
#define SCALE_INDICES_5		SCALE_INDICES_4 SCALE_INDICES_SRC4



#define SCALE_XYZ(n)	index##n.x *= inc##n.x;		\
			index##n.y *= inc##n.y;		\
			index##n.z *= inc##n.z;

#define SCALE_INDICES_XYZ_1	SCALE_XYZ(1)
#define SCALE_INDICES_XYZ_2	SCALE_INDICES_XYZ_1 SCALE_XYZ(2)
#define SCALE_INDICES_XYZ_3	SCALE_INDICES_XYZ_2 SCALE_XYZ(3)
#define SCALE_INDICES_XYZ_4	SCALE_INDICES_XYZ_3 SCALE_XYZ(4)
#define SCALE_INDICES_XYZ_5	SCALE_INDICES_XYZ_4 SCALE_XYZ(5)

#define SCALE_INDICES_XYZ_1_LEN	SCALE_INDICES_XYZ_1
#define SCALE_INDICES_XYZ_2_LEN	SCALE_INDICES_XYZ_2
#define SCALE_INDICES_XYZ_3_LEN	SCALE_INDICES_XYZ_3
#define SCALE_INDICES_XYZ_4_LEN	SCALE_INDICES_XYZ_4
#define SCALE_INDICES_XYZ_5_LEN	SCALE_INDICES_XYZ_5

// BUG do any checking here???
#define SCALE_INDICES_XYZ_SBM_LEN	SCALE_INDICES_XYZ_SBM	// anything with len?

#define SCALE_INDICES_XYZ_DBM_LEN	dbmi.x *= dbm_inc.x;		\
					if( dbmi.y >= len.y ) return;	\
					dbmi.y *= dbm_inc.y;		\
					if( dbmi.z >= len.z ) return;	\
					dbmi.z += dbm_inc.z;

#define SCALE_INDICES_XYZ_2SRCS		SCALE_XYZ(2) SCALE_XYZ(3)

#define SCALE_INDICES_XYZ_DBM_		SCALE_INDICES_XYZ_DBM
#define SCALE_INDICES_XYZ_DBM_1SRC	SCALE_INDICES_XYZ_DBM SCALE_XYZ(2)
#define SCALE_INDICES_XYZ_DBM_2SRCS	SCALE_INDICES_XYZ_DBM		\
					SCALE_INDICES_XYZ_2SRCS
#define SCALE_INDICES_XYZ_DBM_SBM	SCALE_INDICES_XYZ_DBM SCALE_INDICES_XYZ_SBM

#define SCALE_INDICES_XYZ_DBM__LEN	SCALE_INDICES_XYZ_DBM_LEN
#define SCALE_INDICES_XYZ_DBM_1SRC_LEN	SCALE_INDICES_XYZ_DBM_LEN SCALE_XYZ(2)
#define SCALE_INDICES_XYZ_DBM_2SRCS_LEN	SCALE_INDICES_XYZ_DBM_LEN		\
					SCALE_INDICES_XYZ_2SRCS
#define SCALE_INDICES_XYZ_DBM_SBM_LEN	SCALE_INDICES_XYZ_DBM_LEN \
					SCALE_INDICES_XYZ_SBM_LEN

#define SCALE_INDICES_XYZ_SBM_1		SCALE_INDICES_XYZ_SBM SCALE_INDICES_XYZ_1
#define SCALE_INDICES_XYZ_SBM_2		SCALE_INDICES_XYZ_SBM SCALE_INDICES_XYZ_2
#define SCALE_INDICES_XYZ_SBM_3		SCALE_INDICES_XYZ_SBM SCALE_INDICES_XYZ_3

#define SCALE_INDICES_XYZ_SBM_1_LEN		SCALE_INDICES_XYZ_SBM_1
#define SCALE_INDICES_XYZ_SBM_2_LEN		SCALE_INDICES_XYZ_SBM_2
#define SCALE_INDICES_XYZ_SBM_3_LEN		SCALE_INDICES_XYZ_SBM_3


/* These are used in DBM kernels, where we need to scale the bitmap index
 * even in fast loops
 */

#define SCALE_INDICES_FAST_1	/* nop */
#define SCALE_INDICES_FAST_2	/* nop */
#define SCALE_INDICES_FAST_3	/* nop */
#define SCALE_INDICES_FAST_4	/* nop */
#define SCALE_INDICES_FAST_5	/* nop */

#define SCALE_INDICES_EQSP_1	SCALE_INDICES_1
#define SCALE_INDICES_EQSP_2	SCALE_INDICES_2
#define SCALE_INDICES_EQSP_3	SCALE_INDICES_3
#define SCALE_INDICES_EQSP_4	SCALE_INDICES_4
#define SCALE_INDICES_EQSP_5	SCALE_INDICES_5
#define SCALE_INDICES_EQSP_1SRC	SCALE_INDICES_SRC1
#define SCALE_INDICES_EQSP_2SRCS	SCALE_INDICES_SRC1 SCALE_INDICES_SRC2
#define SCALE_INDICES_EQSP_SBM	SCALE_INDICES_SBM
#define SCALE_INDICES_EQSP_DBM	SCALE_INDICES_DBM

#define SCALE_INDICES_EQSP_SBM_1	SCALE_INDICES_EQSP_1 SCALE_INDICES_EQSP_SBM
#define SCALE_INDICES_EQSP_SBM_2	SCALE_INDICES_EQSP_2 SCALE_INDICES_EQSP_SBM
#define SCALE_INDICES_EQSP_SBM_3	SCALE_INDICES_EQSP_3 SCALE_INDICES_EQSP_SBM
#define SCALE_INDICES_EQSP_DBM_		SCALE_INDICES_EQSP_DBM
#define SCALE_INDICES_EQSP_DBM_1SRC	SCALE_INDICES_EQSP_1SRC SCALE_INDICES_EQSP_DBM
#define SCALE_INDICES_EQSP_DBM_2SRCS	SCALE_INDICES_EQSP_2SRCS SCALE_INDICES_EQSP_DBM
#define SCALE_INDICES_EQSP_DBM_SBM	SCALE_INDICES_EQSP_DBM SCALE_INDICES_EQSP_SBM

/*************************************************************/

#ifdef BUILD_FOR_OPENCL
#define OFFSET_A	+ a_offset
#define OFFSET_B	+ b_offset
#define OFFSET_C	+ c_offset
#define OFFSET_D	+ d_offset
#define OFFSET_E	+ e_offset
#else // ! BUILD_FOR_OPENCL
#define OFFSET_A
#define OFFSET_B
#define OFFSET_C
#define OFFSET_D
#define OFFSET_E
#endif // ! BUILD_FOR_OPENCL

#if CUDA_COMP_CAP < 20

#define eqsp_dst	a[(index1.x+index1.y)*inc1	OFFSET_A ]
#define eqsp_src1	b[(index2.x+index2.y)*inc2	OFFSET_B ]

#define dst	a[index1.x+index1.y	OFFSET_A ]
#define src1	b[index2.x+index2.y	OFFSET_B ]
#define src2	c[index3.x+index3.y	OFFSET_C ]
#define src3	d[index4.x+index4.y	OFFSET_D ]
#define src4	e[index5.x+index5.y	OFFSET_E ]

#define srcbit	(sbm[(sbmi.x+sbmi.y+sbm_bit0)>>LOG2_BITS_PER_BITMAP_WORD] & \
			NUMBERED_BIT((sbmi.x+sbmi.y+sbm_bit0)&(BITS_PER_BITMAP_WORD-1)))

#define cdst	a[index1.x+index1.y	OFFSET_A ]
#define csrc1	b[index2.x+index2.y	OFFSET_B ]
#define csrc2	c[index3.x+index3.y	OFFSET_C ]
#define csrc3	d[index4.x+index4.y	OFFSET_D ]
#define csrc4	e[index5.x+index5.y	OFFSET_E ]

#define qdst	a[index1.x+index1.y	OFFSET_A ]
#define qsrc1	b[index2.x+index2.y	OFFSET_B ]
#define qsrc2	c[index3.x+index3.y	OFFSET_C ]
#define qsrc3	d[index4.x+index4.y	OFFSET_D ]
#define qsrc4	e[index5.x+index5.y	OFFSET_E ]

#else /* CUDA_COMP_CAP >= 20 */

// doing this now to fix cuda, but may not be the right fix...
#define eqsp_dst	a[(index1.x+index1.y+index1.z)*inc1	OFFSET_A ]
#define eqsp_src1	b[(index2.x+index2.y+index2.z)*inc2	OFFSET_B ]

#define dst	a[index1.x+index1.y+index1.z	OFFSET_A ]
#define src1	b[index2.x+index2.y+index2.z	OFFSET_B ]
#define src2	c[index3.x+index3.y+index3.z	OFFSET_C ]
#define src3	d[index4.x+index4.y+index4.z	OFFSET_D ]
#define src4	e[index5.x+index5.y+index5.z	OFFSET_E ]

#define srcbit	(sbm[(sbmi.x+sbmi.y+sbmi.z+sbm_bit0)>>LOG2_BITS_PER_BITMAP_WORD] & \
		NUMBERED_BIT((sbmi.x+sbmi.y+sbmi.z+sbm_bit0)&(BITS_PER_BITMAP_WORD-1)))

#define cdst	a[index1.x+index1.y+index1.z	OFFSET_A ]
#define csrc1	b[index2.x+index2.y+index2.z	OFFSET_B ]
#define csrc2	c[index3.x+index3.y+index3.z	OFFSET_C ]
#define csrc3	d[index4.x+index4.y+index4.z	OFFSET_D ]
#define csrc4	e[index5.x+index5.y+index5.z	OFFSET_E ]

#define qdst	a[index1.x+index1.y+index1.z	OFFSET_A ]
#define qsrc1	b[index2.x+index2.y+index2.z	OFFSET_B ]
#define qsrc2	c[index3.x+index3.y+index3.z	OFFSET_C ]
#define qsrc3	d[index4.x+index4.y+index4.z	OFFSET_D ]
#define qsrc4	e[index5.x+index5.y+index5.z	OFFSET_E ]

#endif /* CUDA_COMP_CAP >= 20 */

/* Even if we can't do XYZ indexing, we don't do much harm by multiplying the z */

#define SCALE_INDICES_XYZ_SBM	sbmi.x *= sbm_inc.x;	\
				sbmi.y *= sbm_inc.y;	\
				sbmi.z *= sbm_inc.z;

#define SCALE_INDICES_XYZ_DBM	dbmi.x *= dbm_inc.x;		\
				dbmi.y *= dbm_inc.y;		\
				dbmi.z *= dbm_inc.z;


#define SET_DBM_BIT(cond)	if( cond ) dbm[i_dbm_word] |= dbm_bit; else dbm[i_dbm_word] &= ~dbm_bit;

// moved to this file to be shared:
//#include "veclib/both_call_utils.h"



#endif // _GPU_CALL_UTILS_H_

