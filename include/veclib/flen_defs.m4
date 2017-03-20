
define(`dst',`fast_dst')
define(`src1',`fast_src1')
define(`src2',`fast_src2')
define(`src3',`fast_src3')
define(`src4',`fast_src4')

define(`cdst',`fast_cdst')
define(`csrc1',`fast_csrc1')
define(`csrc2',`fast_csrc2')
define(`csrc3',`fast_csrc3')
define(`csrc4',`fast_csrc4')

define(`qdst',`fast_qdst')
define(`qsrc1',`fast_qsrc1')
define(`qsrc2',`fast_qsrc2')
define(`qsrc3',`fast_qsrc3')
define(`qsrc4',`fast_qsrc4')

define(`GENERIC_GPU_FUNC_CALL',`GENERIC_FLEN_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')

define(`SLOW_GPU_FUNC_CALL',`')

define(`GENERIC_VEC_FUNC_DBM',`GENERIC_FLEN_VEC_FUNC_DBM($1,$2,$3,$4,$5)')

define(`_VEC_FUNC_2V_CONV',`_GENERIC_FLEN_CONV_FUNC($1,std_type,$2)')




define(`DECL_BASIC_INDICES_DBM',`						\
	unsigned int i_dbm_bit;				\
	int i_dbm_word; bitmap_word dbm_bit;		\
	int tbl_idx;					\
')


// slow defn - almost
define(`SET_INDICES_DBM',`				\
	SET_DBM_TBL_INDEX				\
dnl	SET_DBM_INDEX_ARRAY				\
')

// checkpoint_one
dnl define(`SET_INDICES_DBM_1S_',SET_DBM_TBL_INDEX)

// checkpoint_two
define(`SET_DBM_TBL_INDEX',`							\
	tbl_idx = THREAD_INDEX_X;						\
  	i_dbm_word = dbm_info_p->word_tbl[tbl_idx].word_offset;			\
')

// checkpoint_three
dnl	define(`SET_DBM_INDEX_ARRAY',`							\
dnl		dbmi = dbm_info_p->word_tbl[tbl_idx].first_bit_num;			\
dnl	')
define(`SET_DBM_INDEX_ARRAY',`')

// Because the fast version has lengths, we don't need flen version...
// No-ops
define(`_VEC_FUNC_MM_NOCC',`')
define(`_VEC_FUNC_2V_PROJ',`')
define(`_VEC_FUNC_CPX_2V_PROJ',`')
define(`_VEC_FUNC_QUAT_2V_PROJ',`')
define(`_VEC_FUNC_2V_PROJ_IDX',`')
define(`_VEC_FUNC_3V_PROJ',`')
define(`_VEC_FUNC_CPX_3V_PROJ',`')

include(`../../include/veclib/fast_eqsp_defs.m4')

define(`DECLARE_DBM_INDEX',`')
define(`SET_INDICES_1SRC',`index2 = tbl_idx;')


