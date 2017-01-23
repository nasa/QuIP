
define(`dst',`eqsp_dst')
define(`src1',`eqsp_src1')
define(`src2',`eqsp_src2')
define(`src3',`eqsp_src3')
define(`src4',`eqsp_src4')

define(`cdst',`eqsp_cdst')
define(`csrc1',`eqsp_csrc1')
define(`csrc2',`eqsp_csrc2')
define(`csrc3',`eqsp_csrc3')
define(`csrc4',`eqsp_csrc4')

define(`qdst',`eqsp_qdst')
define(`qsrc1',`eqsp_qsrc1')
define(`qsrc2',`eqsp_qsrc2')
define(`qsrc3',`eqsp_qsrc3')
define(`qsrc4',`eqsp_qsrc4')

define(`_VEC_FUNC_MM_NOCC',`')
define(`_VEC_FUNC_2V_PROJ',`')
define(`_VEC_FUNC_CPX_2V_PROJ',`')
define(`_VEC_FUNC_QUAT_2V_PROJ',`')
define(`_VEC_FUNC_2V_PROJ_IDX',`')
define(`_VEC_FUNC_3V_PROJ',`')
define(`_VEC_FUNC_CPX_3V_PROJ',`')

include(`../../include/veclib/fast_eqsp_defs.m4')

// slow defn - almost
define(`SET_INDICES_DBM',`SET_DBM_TBL_INDEX SET_DBM_INDEX_ARRAY')
dnl define(`DECLARE_KERN_ARGS_DBM',`KERNEL_ARG_QUALIFIER bitmap_word *dbm, DECLARE_KERN_ARGS_DBM_GPU_INFO')
define(`SET_INDICES_1SRC',`index2 = tbl_idx;')

dnl define(`SET_INDICES_DBM_1S_',SET_DBM_TBL_INDEX)

define(`SET_DBM_TBL_INDEX',`						\
	tbl_idx = THREAD_INDEX_X;					\
  	i_dbm_word = dbm_info_p->word_tbl[tbl_idx].word_offset;					\
')

dnl define(`SET_DBM_INDEX_ARRAY', dbmi = dbm_info_p->word_tbl[tbl_idx].first_bit_num;)
define(`SET_DBM_INDEX_ARRAY',`')

define(`DBM_EQSP_LEN_TEST', dbmi >= dbm_bit0  && dbmi < dbm_bit0+len)

define(`DECL_BASIC_INDICES_DBM',`			\
	unsigned int i_dbm_bit;				\
	int i_dbm_word; bitmap_word dbm_bit;		\
	int tbl_idx;					\
')


ifdef(`BUILD_FOR_CUDA',`

define(`GENERIC_GPU_FUNC_CALL',`
GENERIC_EQSP_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)
GENERIC_ELEN_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)
')

define(`SLOW_GPU_FUNC_CALL',`')

define(`GENERIC_VEC_FUNC_DBM',`
GENERIC_EQSP_VEC_FUNC_DBM($1,$2,$3,$4,$5)
GENERIC_ELEN_VEC_FUNC_DBM($1,$2,$3,$4,$5)
')

define(`_VEC_FUNC_2V_CONV',`
_GENERIC_EQSP_CONV_FUNC($1,std_type,$2)
_GENERIC_ELEN_CONV_FUNC($1,std_type,$2)
')

',` dnl else // ! BUILD_FOR_CUDA

// Why is it that only CUDA needs the len versions???

define(`GENERIC_GPU_FUNC_CALL',`
GENERIC_EQSP_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)
')

define(`SLOW_GPU_FUNC_CALL',`')

define(`GENERIC_VEC_FUNC_DBM',`
GENERIC_EQSP_VEC_FUNC_DBM($1,$2,$3,$4,$5)
')

define(`_VEC_FUNC_2V_CONV',`
_GENERIC_EQSP_CONV_FUNC($1,std_type,$2)
')

') dnl endif // ! BUILD_FOR_CUDA

