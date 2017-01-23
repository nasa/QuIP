// Are these definitions only for gpu, or used by cpu also?

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


define(`SET_INDICES_DBM',`			\
	SET_INDEX(dbmi)			\
	i_dbm_word = dbmi;			\
	dbmi *= BITS_PER_BITMAP_WORD;			\
')

define(`DECL_BASIC_INDICES_DBM',`			\
	unsigned int i_dbm_bit;			\
	int i_dbm_word; bitmap_word dbm_bit;			\
')

define(`SET_INDICES_DBM_1S_',`			\
	i_dbm_word = THREAD_INDEX_X;			\
')

include(`../../include/veclib/fast_eqsp_defs.m4')

define(`DECLARE_DBM_INDEX',`GPU_INDEX_TYPE dbmi;')
define(`DECLARE_KERN_ARGS_DBM',`KERNEL_ARG_QUALIFIER bitmap_word *dbm')
define(`SET_INDICES_1SRC',`index2 = dbmi;')

// vmaxg etc - require contiguous, fast only

define(`_VEC_FUNC_MM_NOCC',`__VEC_FUNC_FAST_MM_NOCC($1,$5,$6)')

// vmaxv, vminv, vsum

// on gpu only fast version, but on cpu only slow version!?

define(`_VEC_FUNC_2V_PROJ',`
	__VEC_FUNC_FAST_2V_PROJ_SETUP($1,$4)
	__VEC_FUNC_FAST_2V_PROJ_HELPER($1,$4)
')

define(`_VEC_FUNC_CPX_2V_PROJ',`
	__VEC_FUNC_CPX_FAST_2V_PROJ_SETUP($1,$4,$5) 
	__VEC_FUNC_CPX_FAST_2V_PROJ_HELPER($1,$4,$5)
')

define(`_VEC_FUNC_QUAT_2V_PROJ',`
	__VEC_FUNC_QUAT_FAST_2V_PROJ_SETUP($1,$4,$5,$6,$7) 
	__VEC_FUNC_QUAT_FAST_2V_PROJ_HELPER($1,$4,$5,$6,$7)
')

define(`_VEC_FUNC_2V_PROJ_IDX',`__VEC_FUNC_FAST_2V_PROJ_IDX( $1, $4, $5 )')

// There is only one function - rvdot, cvdot - so it is implemented in a non-general way.
// Therefore, we don't have to pass the statements or expressions...

define(`_VEC_FUNC_3V_PROJ',`__VEC_FUNC_FAST_3V_PROJ( $1 )')

define(`_VEC_FUNC_CPX_3V_PROJ',`__VEC_FUNC_CPX_FAST_3V_PROJ( $1 )')


// Why is it that only CUDA needs the len versions???

define(`GENERIC_GPU_FUNC_CALL',`GENERIC_FAST_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')

define(`SLOW_GPU_FUNC_CALL',`')

define(`GENERIC_VEC_FUNC_DBM',`GENERIC_FAST_VEC_FUNC_DBM($1,$2,$3,$4,$5)')


define(`_VEC_FUNC_2V_CONV',`_GENERIC_FAST_CONV_FUNC($1,std_type,$2)')

