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

#define VFUNC_FAST_NOCC_SETUP_NAME(func_name)	_VFUNC_FAST_NOCC_SETUP_NAME(func_name,pf_str,type_code)
#define _VFUNC_FAST_NOCC_SETUP_NAME(n,pf,ty)	__VFUNC_FAST_NOCC_SETUP_NAME(n,pf,ty)
#define __VFUNC_FAST_NOCC_SETUP_NAME(n,pf,ty)	g_##pf##_fast_##ty##_##n##_nocc_setup

#define VFUNC_FAST_NOCC_HELPER_NAME(func_name)	_VFUNC_FAST_NOCC_HELPER_NAME(func_name,pf_str,type_code)
#define _VFUNC_FAST_NOCC_HELPER_NAME(n,pf,ty)	__VFUNC_FAST_NOCC_HELPER_NAME(n,pf,ty)
#define __VFUNC_FAST_NOCC_HELPER_NAME(n,pf,ty)	g_##pf##_fast_##ty##_##n##_nocc_helper

#define VFUNC_SIMPLE_NAME(func_name)		_VFUNC_SIMPLE_NAME(func_name,pf_str,type_code)
#define _VFUNC_SIMPLE_NAME(n,pf,ty)		__VFUNC_SIMPLE_NAME(n,pf,ty)
#define __VFUNC_SIMPLE_NAME(n,pf,ty)		g_##pf##_##ty##_##n

#define VFUNC_HELPER_NAME(func_name)		_VFUNC_HELPER_NAME(func_name,pf_str,type_code)
#define _VFUNC_HELPER_NAME(n,pf,ty)		__VFUNC_HELPER_NAME(n,pf,ty)
#define __VFUNC_HELPER_NAME(n,pf,ty)		g_##pf##_##ty##_##n##_helper

#define VFUNC_FAST_IDX_SETUP_NAME(func_name)		_VFUNC_IDX_SETUP_NAME(func_name,fast,pf_str,type_code)
#define _VFUNC_IDX_SETUP_NAME(n,speed,pf,ty)		__VFUNC_IDX_SETUP_NAME(n,speed,pf,ty)
#define __VFUNC_IDX_SETUP_NAME(n,s,pf,ty)		g_##pf##_##s##_##ty##_##n##_index_setup

#define VFUNC_FAST_IDX_HELPER_NAME(func_name)	_VFUNC_IDX_HELPER_NAME(func_name,fast,pf_str,type_code)
#define _VFUNC_IDX_HELPER_NAME(n,s,pf,ty)	__VFUNC_IDX_HELPER_NAME(n,s,pf,ty)
#define __VFUNC_IDX_HELPER_NAME(n,s,pf,ty)	g_##pf##_##s##_##ty##_##n##_index_helper



/****************** DECL_INDICES ***********************/

#define SLOW_GPU_INDEX_TYPE	DIM5

#define DECL_INDICES_1		GPU_INDEX_TYPE index1;
#define DECL_INDICES_SRC1	GPU_INDEX_TYPE index2;
#define DECL_INDICES_SRC2	GPU_INDEX_TYPE index3;
#define DECL_INDICES_SRC3	GPU_INDEX_TYPE index4;
#define DECL_INDICES_SRC4	GPU_INDEX_TYPE index5;
#define DECL_INDICES_SBM	GPU_INDEX_TYPE sbmi;

// dbmi indexes the bit - from it, we have to compute the index of the word, and the bit mask
// We have an integral number of words per row.

/* different for fast and slow! */
/*
#define DECL_BASIC_INDICES_DBM	int i_dbm_bit;				\
				int i_dbm_word; bitmap_word dbm_bit;	\
				int tbl_idx;
				*/

#define DECL_INDICES_DBM	GPU_INDEX_TYPE dbmi; 			\
				DECL_BASIC_INDICES_DBM

#define DECL_INDICES_2		DECL_INDICES_1 DECL_INDICES_SRC1
#define DECL_INDICES_3		DECL_INDICES_2 DECL_INDICES_SRC2
#define DECL_INDICES_4		DECL_INDICES_3 DECL_INDICES_SRC3
#define DECL_INDICES_5		DECL_INDICES_4 DECL_INDICES_SRC4
#define DECL_INDICES_1SRC	DECL_INDICES_SRC1
#define DECL_INDICES_2SRCS	DECL_INDICES_SRC1 DECL_INDICES_SRC2
#define DECL_INDICES_SBM_1	DECL_INDICES_1 DECL_INDICES_SBM
#define DECL_INDICES_SBM_2	DECL_INDICES_2 DECL_INDICES_SBM
#define DECL_INDICES_SBM_3	DECL_INDICES_3 DECL_INDICES_SBM

#define DECL_INDICES_DBM_1S	DECL_BASIC_INDICES_DBM
#define DECL_INDICES_DBM_1S_	DECL_INDICES_DBM_1S
#define DECL_INDICES_DBM_	DECL_INDICES_DBM
#define DECL_INDICES_DBM_1SRC	DECL_INDICES_1SRC DECL_INDICES_DBM
#define DECL_INDICES_DBM_1S_1SRC	DECL_INDICES_DBM_1SRC
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
#define INIT_INDICES_DBM_1S_	DECL_INDICES_DBM_1S_ SET_INDICES_DBM_1S_
#define INIT_INDICES_DBM_1S_1SRC	INIT_INDICES_DBM_1SRC
#define INIT_INDICES_DBM_SBM	DECL_INDICES_DBM_SBM SET_INDICES_DBM_SBM


/******************** SET_INDICES ***************************/

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

#define SET_INDICES_DBM_	SET_INDICES_DBM
// this one is speed-sensitive
//#define SET_INDICES_DBM_1S_	SET_INDICES_DBM

// BUG?  this looks wrong!?
// 1SRC is only used with dbm?
#define SET_INDICES_1SRC	index2 = dbmi;

#define SET_INDICES_DBM_1SRC	SET_INDICES_DBM SET_INDICES_1SRC
#define SET_INDICES_DBM_1S_1SRC	SET_INDICES_DBM_1SRC
#define SET_INDICES_DBM_2SRCS	SET_INDICES_DBM_1SRC SET_INDICES_SRC2
// Can't use SET_INDICES_SBM here...
#define SET_INDICES_DBM_SBM	SET_INDICES_DBM sbmi = dbmi;

/**************************** SET_INDICES_XYZ ******************************/

#ifdef BUILD_FOR_OPENCL
#define THREAD_INDEX_X		get_global_id(0)
#endif // BUILD_FOR_OPENCL

#ifdef BUILD_FOR_CUDA
#define THREAD_INDEX_X		blockIdx.x * blockDim.x + threadIdx.x
#endif // BUILD_FOR_CUDA

// For bitmaps, the thread index is the word index...

/**************** SCALE_INDICES_ ********************/

#define SCALE_INDICES_1		SCALE_INDEX(index1,inc1)	// index1.d5_dim[0] *= inc1;
#define SCALE_INDICES_SRC1	SCALE_INDEX(index2,inc2)	// index2.d5_dim[0] *= inc2;
#define SCALE_INDICES_SRC2	SCALE_INDEX(index3,inc3)	// index3.d5_dim[0] *= inc3;
#define SCALE_INDICES_SRC3	SCALE_INDEX(index4,inc4)	// index4.d5_dim[0] *= inc4;
#define SCALE_INDICES_SRC4	SCALE_INDEX(index5,inc5)	// index5.d5_dim[0] *= inc5;
//#define SCALE_INDICES_SBM	SCALE_INDEX(sbmi,sbm_inc)	// sbmi.d5_dim[0] *= sbm_inc;
//#define SCALE_INDICES_DBM	SCALE_INDEX(dbmi,dbm_inc)	// dbmi.d5_dim[0] *= dbm_inc;
#define SCALE_INDICES_DBM	/* nop */
#define SCALE_INDICES_SBM	/* nop */

#define SCALE_INDICES_DBM_	SCALE_INDICES_DBM
#define SCALE_INDICES_DBM_SBM	SCALE_INDICES_DBM SCALE_INDICES_SBM

#define SCALE_INDICES_2SRCS	SCALE_INDICES_SRC1 SCALE_INDICES_SRC2

#define SCALE_INDICES_DBM_1SRC	SCALE_INDICES_DBM SCALE_INDICES_SRC1
#define SCALE_INDICES_DBM_1S_1SRC	SCALE_INDICES_DBM_1SRC
#define SCALE_INDICES_DBM_2SRCS	SCALE_INDICES_DBM SCALE_INDICES_2SRCS

#define SCALE_INDICES_SBM_1	SCALE_INDICES_SBM SCALE_INDICES_1
#define SCALE_INDICES_SBM_2	SCALE_INDICES_SBM SCALE_INDICES_2
#define SCALE_INDICES_SBM_3	SCALE_INDICES_SBM SCALE_INDICES_3
#define SCALE_INDICES_SBM_4	SCALE_INDICES_SBM SCALE_INDICES_4

#define SCALE_INDICES_2		SCALE_INDICES_1 SCALE_INDICES_SRC1
#define SCALE_INDICES_3		SCALE_INDICES_2 SCALE_INDICES_SRC2
#define SCALE_INDICES_4		SCALE_INDICES_3 SCALE_INDICES_SRC3
#define SCALE_INDICES_5		SCALE_INDICES_4 SCALE_INDICES_SRC4


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
#define SCALE_INDICES_EQSP_DBM_1S_1SRC	SCALE_INDICES_EQSP_DBM_1SRC
#define SCALE_INDICES_EQSP_DBM_1S_	/* nop */
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


// This used to be x+y+z ... (for dim3 indices)
#define INDEX5_SUM(idx)	idx.d5_dim[0]+idx.d5_dim[1]+idx.d5_dim[2]+idx.d5_dim[3]+idx.d5_dim[4]

#define fast_dst	a[index1	OFFSET_A ]
#define fast_src1	b[index2	OFFSET_B ]
#define fast_src2	c[index3	OFFSET_C ]
#define fast_src3	d[index4	OFFSET_D ]
#define fast_src4	e[index5	OFFSET_E ]

// Indices are scaled in the function prelude
#define eqsp_dst	a[index1	OFFSET_A ]
#define eqsp_src1	b[index2	OFFSET_B ]
#define eqsp_src2	c[index3	OFFSET_C ]
#define eqsp_src3	d[index4	OFFSET_D ]
#define eqsp_src4	e[index5	OFFSET_E ]

#define INDEX_SUM(idx)	(idx.d5_dim[0]+idx.d5_dim[1]+idx.d5_dim[2]+idx.d5_dim[3]+idx.d5_dim[4])

#define slow_dst	a[INDEX_SUM(index1)	OFFSET_A ]
#define slow_src1	b[INDEX_SUM(index2)	OFFSET_B ]
#define slow_src2	c[INDEX_SUM(index3)	OFFSET_C ]
#define slow_src3	d[INDEX_SUM(index4)	OFFSET_D ]
#define slow_src4	e[INDEX_SUM(index5)	OFFSET_E ]

#define slow_dst1	slow_dst	// for gpu_cent (centroid) with two outputs
#define slow_dst2	slow_src1	// for gpu_cent (centroid) with two outputs

//#define srcbit	(sbm[(INDEX_SUM(sbmi)+sbm_bit0)>>LOG2_BITS_PER_BITMAP_WORD] & 
//		NUMBERED_BIT((INDEX_SUM(sbmi)+sbm_bit0)&(BITS_PER_BITMAP_WORD-1)))

#define fast_cdst	a[index1	OFFSET_A ]
#define fast_csrc1	b[index2	OFFSET_B ]
#define fast_csrc2	c[index3	OFFSET_C ]
#define fast_csrc3	d[index4	OFFSET_D ]
#define fast_csrc4	e[index5	OFFSET_E ]

#define eqsp_cdst	a[index1*inc1	OFFSET_A ]
#define eqsp_csrc1	b[index2*inc2	OFFSET_B ]
#define eqsp_csrc2	c[index3*inc3	OFFSET_C ]
#define eqsp_csrc3	d[index4*inc4	OFFSET_D ]
#define eqsp_csrc4	e[index5*inc5	OFFSET_E ]

#define slow_cdst	a[INDEX_SUM(index1)	OFFSET_A ]
#define slow_csrc1	b[INDEX_SUM(index2)	OFFSET_B ]
#define slow_csrc2	c[INDEX_SUM(index3)	OFFSET_C ]
#define slow_csrc3	d[INDEX_SUM(index4)	OFFSET_D ]
#define slow_csrc4	e[INDEX_SUM(index5)	OFFSET_E ]


#define fast_qdst	a[index1	OFFSET_A ]
#define fast_qsrc1	b[index2	OFFSET_B ]
#define fast_qsrc2	c[index3	OFFSET_C ]
#define fast_qsrc3	d[index4	OFFSET_D ]
#define fast_qsrc4	e[index5	OFFSET_E ]

#define eqsp_qdst	a[index1*inc1	OFFSET_A ]
#define eqsp_qsrc1	b[index2*inc2	OFFSET_B ]
#define eqsp_qsrc2	c[index3*inc3	OFFSET_C ]
#define eqsp_qsrc3	d[index4*inc4	OFFSET_D ]
#define eqsp_qsrc4	e[index5*inc5	OFFSET_E ]

#define slow_qdst	a[INDEX_SUM(index1)	OFFSET_A ]
#define slow_qsrc1	b[INDEX_SUM(index2)	OFFSET_B ]
#define slow_qsrc2	c[INDEX_SUM(index3)	OFFSET_C ]
#define slow_qsrc3	d[INDEX_SUM(index4)	OFFSET_D ]
#define slow_qsrc4	e[INDEX_SUM(index5)	OFFSET_E ]



#define SET_DBM_BIT(cond)	if( cond ) dbm[i_dbm_word] |= dbm_bit; else dbm[i_dbm_word] &= ~dbm_bit;

// moved to this file to be shared:
//#include "veclib/both_call_utils.h"



#endif // _GPU_CALL_UTILS_H_

