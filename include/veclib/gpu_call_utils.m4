/* gpu_call_utils.m4 BEGIN */

// This file contains macros that are useful for writing kernels...
// Because the kernels are static, the names don't need to include the
// platform name...

/******************** VFUNC_XXXX_NAME **********************/

define(`VFUNC_NAME',`````g_'pf_str`_'$2`_'''type_code```_'$1'''')
define(`VFUNC_NAME_WITH_SUFFIX',VFUNC_NAME($1,$2)$3)

define(`VFUNC_FAST_NAME',VFUNC_NAME($1,`fast'))
define(`VFUNC_EQSP_NAME',VFUNC_NAME($1,`eqsp'))
define(`VFUNC_SLOW_NAME',VFUNC_NAME($1,`slow'))
define(`VFUNC_FLEN_NAME',VFUNC_NAME($1,`flen'))
define(`VFUNC_ELEN_NAME',VFUNC_NAME($1,`elen'))
define(`VFUNC_SLEN_NAME',VFUNC_NAME($1,`slen'))

define(`VFUNC_FAST_NOCC_SETUP_NAME',`VFUNC_FAST_NAME($1)`_setup'')
define(`VFUNC_FAST_NOCC_HELPER_NAME',`VFUNC_FAST_NAME($1)`_helper'')

define(`VFUNC_SIMPLE_NAME',`g_'pf_str`_'type_code`_'$1)
define(`VFUNC_HELPER_NAME',VFUNC_SIMPLE_NAME($1)`_helper')

define(`VFUNC_FAST_IDX_SETUP_NAME',VFUNC_FAST_NAME($1)`_setup')
define(`VFUNC_FAST_IDX_HELPER_NAME',VFUNC_FAST_NAME($1)`_helper')



/****************** DECL_INDICES ***********************/

define(`SLOW_GPU_INDEX_TYPE',	dim5)

define(`DECL_INDICES_1',	GPU_INDEX_TYPE index1;)
define(`DECL_INDICES_SRC1',	GPU_INDEX_TYPE index2;)
define(`DECL_INDICES_SRC2',	GPU_INDEX_TYPE index3;)
define(`DECL_INDICES_SRC3',	GPU_INDEX_TYPE index4;)
define(`DECL_INDICES_SRC4',	GPU_INDEX_TYPE index5;)
define(`DECL_INDICES_SBM',	GPU_INDEX_TYPE sbmi;)

// dbmi indexes the bit - from it, we have to compute the index of the word, and the bit mask
// We have an integral number of words per row.

/* different for fast and slow! */

define(`DECL_INDICES_DBM',	/* decl_indices_dbm */ GPU_INDEX_TYPE dbmi; DECL_BASIC_INDICES_DBM)

/* DECL_INDICES */
define(`DECL_INDICES_2',	DECL_INDICES_1 DECL_INDICES_SRC1)
define(`DECL_INDICES_3',	DECL_INDICES_2 DECL_INDICES_SRC2)
define(`DECL_INDICES_4',	DECL_INDICES_3 DECL_INDICES_SRC3)
define(`DECL_INDICES_5',	DECL_INDICES_4 DECL_INDICES_SRC4)
define(`DECL_INDICES_1SRC',	DECL_INDICES_SRC1)
define(`DECL_INDICES_2SRCS',	DECL_INDICES_SRC1 DECL_INDICES_SRC2)
define(`DECL_INDICES_SBM_1',	DECL_INDICES_1 DECL_INDICES_SBM)
define(`DECL_INDICES_SBM_2',	DECL_INDICES_2 DECL_INDICES_SBM)
define(`DECL_INDICES_SBM_3',	DECL_INDICES_3 DECL_INDICES_SBM)

/* `DECL_INDICES_DBM' */
define(`DECL_INDICES_DBM_1S',DECL_BASIC_INDICES_DBM)
define(`DECL_INDICES_DBM_1S_',DECL_INDICES_DBM_1S)
define(`DECL_INDICES_DBM_',DECL_INDICES_DBM)
define(`DECL_INDICES_DBM_1SRC',DECL_INDICES_1SRC DECL_INDICES_DBM)
define(`DECL_INDICES_DBM_1S_1SRC',DECL_INDICES_DBM_1SRC)
define(`DECL_INDICES_DBM_2SRCS',DECL_INDICES_2SRCS DECL_INDICES_DBM)
define(`DECL_INDICES_DBM_SBM',DECL_INDICES_SBM DECL_INDICES_DBM)


/* DECL_EXTRA */
define(`DECL_EXTRA',`DECL_EXTRA_$1')
define(`DECL_EXTRA_',`')
define(`DECL_EXTRA_T1',std_type r; std_type theta; std_type arg;)
define(`DECL_EXTRA_T2',std_cpx tmpc;)
define(`DECL_EXTRA_T3',std_cpx tmpc; std_type tmp_denom;)

// quaternion helpers
define(`DECL_EXTRA_T4',std_quat tmpq;)
define(`DECL_EXTRA_T5',std_quat tmpq; std_type tmp_denom;)

/*********************** INIT_INDICES *****************/

dnl	INIT_INDICES(bitmaps,vectors)
define(`INIT_INDICES',`/* init_indices /$1/ /$2/ */ INIT_INDICES_$1$2')

define(`INIT_INDICES_1',`DECL_INDICES_1 SET_INDICES_1')
define(`INIT_INDICES_2',`DECL_INDICES_2 SET_INDICES_2')
define(`INIT_INDICES_2',`DECL_INDICES_2 SET_INDICES_2')
define(`INIT_INDICES_3',`DECL_INDICES_3 SET_INDICES_3')
define(`INIT_INDICES_4',`DECL_INDICES_4 SET_INDICES_4')
define(`INIT_INDICES_5',`DECL_INDICES_5 SET_INDICES_5')

define(`INIT_INDICES_2SRCS',`DECL_INDICES_2SRCS SET_INDICES_2SRCS')
define(`INIT_INDICES_SBM_1',`DECL_INDICES_SBM_1 SET_INDICES_SBM_1')
define(`INIT_INDICES_SBM_2',`DECL_INDICES_SBM_2 SET_INDICES_SBM_2')
define(`INIT_INDICES_SBM_3',`DECL_INDICES_SBM_3 SET_INDICES_SBM_3')

define(`INIT_INDICES_DBM_',`DECL_INDICES_DBM_ SET_INDICES_DBM_')
define(`INIT_INDICES_DBM_2SRCS',`DECL_INDICES_DBM_2SRCS SET_INDICES_DBM_2SRCS')
define(`INIT_INDICES_DBM_1SRC',`DECL_INDICES_DBM_1SRC SET_INDICES_DBM_1SRC')
dnl	define(`INIT_INDICES_DBM_1S_',DECL_INDICES_DBM_1S_ SET_INDICES_DBM_1S_)
dnl	define(`INIT_INDICES_DBM_1S_1SRC',INIT_INDICES_DBM_1SRC)
define(`INIT_INDICES_DBM_SBM',`DECL_INDICES_DBM_SBM SET_INDICES_DBM_SBM')


/******************** SET_INDICES ***************************/

define(`SET_INDICES_1',SET_INDEX( index1 ))
define(`SET_INDICES_SRC1',index2 = dst_idx;)
define(`SET_INDICES_SRC2',index3 = index2;)
define(`SET_INDICES_SRC3',index4 = index3;)
define(`SET_INDICES_SRC4',index5 = index4;)
define(`SET_INDICES_SBM',sbmi = index1;)

define(`SET_INDICES_2',SET_INDICES_1 SET_INDICES_SRC1(index1))
define(`SET_INDICES_3',SET_INDICES_2 SET_INDICES_SRC2)
define(`SET_INDICES_4',SET_INDICES_3 SET_INDICES_SRC3)
define(`SET_INDICES_5',SET_INDICES_4 SET_INDICES_SRC4)
define(`SET_INDICES_2SRCS',SET_INDEX(index2) SET_INDICES_SRC2)


define(`SET_INDICES_SBM_1',SET_INDICES_1 SET_INDICES_SBM)
define(`SET_INDICES_SBM_2',SET_INDICES_2 SET_INDICES_SBM)
define(`SET_INDICES_SBM_3',SET_INDICES_3 SET_INDICES_SBM)

define(`SET_INDICES_DBM_',SET_INDICES_DBM)
// this one is speed-sensitive
//define(`SET_INDICES_DBM_1S_',SET_INDICES_DBM)

// BUG?  this looks wrong!?
// 1SRC is only used with dbm?
define(`SET_INDICES_1SRC',index2 = dbmi;)

define(`SET_INDICES_DBM_1SRC',SET_INDICES_DBM SET_INDICES_1SRC)
define(`SET_INDICES_DBM_1S_1SRC',SET_INDICES_DBM_1SRC)
define(`SET_INDICES_DBM_2SRCS',SET_INDICES_DBM_1SRC SET_INDICES_SRC2)
// Can't use SET_INDICES_SBM here...
define(`SET_INDICES_DBM_SBM',SET_INDICES_DBM sbmi = dbmi;)

/**************** SCALE_INDICES_ ********************/

dnl SCALE_INDICES(bitmap,vectors)
define(`SCALE_INDICES',`SCALE_INDICES_$1$2')

define(`SCALE_INDICES_1',SCALE_INDEX(index1,inc1))
define(`SCALE_INDICES_SRC1',SCALE_INDEX(index2,inc2))
define(`SCALE_INDICES_SRC2',SCALE_INDEX(index3,inc3))
define(`SCALE_INDICES_SRC3',SCALE_INDEX(index4,inc4))
define(`SCALE_INDICES_SRC4',SCALE_INDEX(index5,inc5))
dnl define(`SCALE_INDICES_SBM',SCALE_INDEX(sbmi,sbm_inc))
dnl define(`SCALE_INDICES_DBM',SCALE_INDEX(dbmi,dbm_inc))
define(`SCALE_INDICES_DBM',`')
define(`SCALE_INDICES_SBM',`')

define(`SCALE_INDICES_DBM_',SCALE_INDICES_DBM)
define(`SCALE_INDICES_DBM_SBM',SCALE_INDICES_DBM SCALE_INDICES_SBM)

define(`SCALE_INDICES_2SRCS',SCALE_INDICES_SRC1 SCALE_INDICES_SRC2)

define(`SCALE_INDICES_DBM_1SRC',SCALE_INDICES_DBM SCALE_INDICES_SRC1)
define(`SCALE_INDICES_DBM_1S_1SRC',SCALE_INDICES_DBM_1SRC)
define(`SCALE_INDICES_DBM_2SRCS',SCALE_INDICES_DBM SCALE_INDICES_2SRCS)

define(`SCALE_INDICES_SBM_1',SCALE_INDICES_SBM SCALE_INDICES_1)
define(`SCALE_INDICES_SBM_2',SCALE_INDICES_SBM SCALE_INDICES_2)
define(`SCALE_INDICES_SBM_3',SCALE_INDICES_SBM SCALE_INDICES_3)
define(`SCALE_INDICES_SBM_4',SCALE_INDICES_SBM SCALE_INDICES_4)

define(`SCALE_INDICES_2',SCALE_INDICES_1 SCALE_INDICES_SRC1)
define(`SCALE_INDICES_3',SCALE_INDICES_2 SCALE_INDICES_SRC2)
define(`SCALE_INDICES_4',SCALE_INDICES_3 SCALE_INDICES_SRC3)
define(`SCALE_INDICES_5',SCALE_INDICES_4 SCALE_INDICES_SRC4)


/* These are used in DBM kernels, where we need to scale the bitmap index
 * even in fast loops
 */

dnl define(`SCALE_INDICES_FAST_1',`')
dnl define(`SCALE_INDICES_FAST_2',`')
dnl define(`SCALE_INDICES_FAST_3',`')
dnl define(`SCALE_INDICES_FAST_4',`')
dnl define(`SCALE_INDICES_FAST_5',`')

dnl define(`SCALE_INDICES_EQSP_1',SCALE_INDICES_1)
dnl define(`SCALE_INDICES_EQSP_2',SCALE_INDICES_2)
dnl define(`SCALE_INDICES_EQSP_3',SCALE_INDICES_3)
dnl define(`SCALE_INDICES_EQSP_4',SCALE_INDICES_4)
dnl define(`SCALE_INDICES_EQSP_5',SCALE_INDICES_5)
dnl define(`SCALE_INDICES_EQSP_1SRC',SCALE_INDICES_SRC1)
dnl define(`SCALE_INDICES_EQSP_2SRCS',SCALE_INDICES_SRC1 SCALE_INDICES_SRC2)
dnl define(`SCALE_INDICES_EQSP_SBM',SCALE_INDICES_SBM)
dnl define(`SCALE_INDICES_EQSP_DBM',SCALE_INDICES_DBM)

dnl define(`SCALE_INDICES_EQSP_SBM_1',SCALE_INDICES_EQSP_1 SCALE_INDICES_EQSP_SBM)
dnl define(`SCALE_INDICES_EQSP_SBM_2',SCALE_INDICES_EQSP_2 SCALE_INDICES_EQSP_SBM)
dnl define(`SCALE_INDICES_EQSP_SBM_3',SCALE_INDICES_EQSP_3 SCALE_INDICES_EQSP_SBM)
dnl define(`SCALE_INDICES_EQSP_DBM_',SCALE_INDICES_EQSP_DBM)
dnl define(`SCALE_INDICES_EQSP_DBM_1SRC',SCALE_INDICES_EQSP_1SRC SCALE_INDICES_EQSP_DBM)
dnl define(`SCALE_INDICES_EQSP_DBM_1S_1SRC',SCALE_INDICES_EQSP_DBM_1SRC)
dnl define(`SCALE_INDICES_EQSP_DBM_1S_',`')
dnl define(`SCALE_INDICES_EQSP_DBM_2SRCS',SCALE_INDICES_EQSP_2SRCS SCALE_INDICES_EQSP_DBM)
dnl define(`SCALE_INDICES_EQSP_DBM_SBM',SCALE_INDICES_EQSP_DBM SCALE_INDICES_EQSP_SBM)

/*************************************************************/

dnl   #define INDEX_SUM(idx)	(idx.d5_dim[0]+idx.d5_dim[1]+idx.d5_dim[2]+idx.d5_dim[3]+idx.d5_dim[4])
// This used to be x+y+z ... (for dim3 indices)
define(`INDEX_SUM',`INDEX5_SUM($1)')
define(`INDEX5_SUM',$1.d5_dim[0]+$1.d5_dim[1]+$1.d5_dim[2]+$1.d5_dim[3]+$1.d5_dim[4])

define(`fast_dst',a[index1 OFFSET_A ])
define(`fast_src1',b[index2 OFFSET_B ])
define(`fast_src2',c[index3 OFFSET_C ])
define(`fast_src3',d[index4 OFFSET_D ])
define(`fast_src4',e[index5 OFFSET_E ])

// Indices are scaled in the function prelude
define(`eqsp_dst',a[index1 OFFSET_A ])
define(`eqsp_src1',b[index2 OFFSET_B ])
define(`eqsp_src2',c[index3 OFFSET_C ])
define(`eqsp_src3',d[index4 OFFSET_D ])
define(`eqsp_src4',e[index5 OFFSET_E ])

define(`slow_dst',a[INDEX_SUM(index1) OFFSET_A ])
define(`slow_src1',b[INDEX_SUM(index2) OFFSET_B ])
define(`slow_src2',c[INDEX_SUM(index3) OFFSET_C ])
define(`slow_src3',d[INDEX_SUM(index4) OFFSET_D ])
define(`slow_src4',e[INDEX_SUM(index5) OFFSET_E ])

define(`slow_dst1',slow_dst)
define(`slow_dst2',slow_src1)

define(`fast_cdst',a[index1 OFFSET_A ])
define(`fast_csrc1',b[index2 OFFSET_B ])
define(`fast_csrc2',c[index3 OFFSET_C ])
define(`fast_csrc3',d[index4 OFFSET_D ])
define(`fast_csrc4',e[index5 OFFSET_E ])

define(`eqsp_cdst',a[index1*inc1 OFFSET_A ])
define(`eqsp_csrc1',b[index2*inc2 OFFSET_B ])
define(`eqsp_csrc2',c[index3*inc3 OFFSET_C ])
define(`eqsp_csrc3',d[index4*inc4 OFFSET_D ])
define(`eqsp_csrc4',e[index5*inc5 OFFSET_E ])

define(`slow_cdst',a[INDEX_SUM(index1) OFFSET_A ])
define(`slow_csrc1',b[INDEX_SUM(index2) OFFSET_B ])
define(`slow_csrc2',c[INDEX_SUM(index3) OFFSET_C ])
define(`slow_csrc3',d[INDEX_SUM(index4) OFFSET_D ])
define(`slow_csrc4',e[INDEX_SUM(index5) OFFSET_E ])


define(`fast_qdst',a[index1 OFFSET_A ])
define(`fast_qsrc1',b[index2 OFFSET_B ])
define(`fast_qsrc2',c[index3 OFFSET_C ])
define(`fast_qsrc3',d[index4 OFFSET_D ])
define(`fast_qsrc4',e[index5 OFFSET_E ])

define(`eqsp_qdst',a[index1*inc1 OFFSET_A ])
define(`eqsp_qsrc1',b[index2*inc2 OFFSET_B ])
define(`eqsp_qsrc2',c[index3*inc3 OFFSET_C ])
define(`eqsp_qsrc3',d[index4*inc4 OFFSET_D ])
define(`eqsp_qsrc4',e[index5*inc5 OFFSET_E ])

define(`slow_qdst',a[INDEX_SUM(index1) OFFSET_A ])
define(`slow_qsrc1',b[INDEX_SUM(index2) OFFSET_B ])
define(`slow_qsrc2',c[INDEX_SUM(index3) OFFSET_C ])
define(`slow_qsrc3',d[INDEX_SUM(index4) OFFSET_D ])
define(`slow_qsrc4',e[INDEX_SUM(index5) OFFSET_E ])



define(`SET_DBM_BIT',if( $1 ) dbm[i_dbm_word] |= dbm_bit; else dbm[i_dbm_word] &= ~dbm_bit;)







/* gpu_call_utils.m4 END */

