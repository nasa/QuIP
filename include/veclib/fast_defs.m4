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

define(`this_sbm_bit',`(sbm_bit_offset)')

dnl Need backslashes for quoting (OpenCL)

define(`SET_INDICES_DBM',`			\
	SET_INDEX(dbmi)				\
	i_dbm_word = dbmi;			\
	dbmi *= BITS_PER_BITMAP_WORD;		\
')

define(`DECL_INDICES_SBM1',`GPU_INDEX_TYPE sbm1_bit_offset;')
define(`DECL_INDICES_SBM2',`GPU_INDEX_TYPE sbm2_bit_offset;')

define(`SET_INDICES_1SBM',`							\
	sbm1_bit_offset = i_dbm_word * BITS_PER_BITMAP_WORD;			\
')

define(`SET_INDICES_2SBM',`							\
	sbm2_bit_offset = i_dbm_word * BITS_PER_BITMAP_WORD;			\
')

define(`DECL_BASIC_INDICES_DBM',`		\
	unsigned int i_dbm_bit;			\
	int i_dbm_word; bitmap_word dbm_bit;	\
')

define(`SET_INDICES_DBM__1S',`			\
	i_dbm_word = THREAD_INDEX_X;		\
')

dnl	Are these just deferred, or are they not needed???
define(`_VEC_FUNC_MM_NOCC',`_VEC_FUNC_FAST_MM_NOCC($1,$2,$3,$4,$5,$6)')
define(`_VEC_FUNC_2V_PROJ',`_VEC_FUNC_FAST_2V_PROJ($1,$2,$3,$4)')
define(`_VEC_FUNC_CPX_2V_PROJ',`_VEC_FUNC_FAST_CPX_2V_PROJ($1,$2,$3,$4,$5)')

dnl	_VEC_FUNC_QUAT_2V_PROJ( name, init_stat, loop_stat, gpu_e1, gpu_e2, gpu_e3, gpu_e4 )
define(`_VEC_FUNC_QUAT_2V_PROJ',`_VEC_FUNC_FAST_QUAT_2V_PROJ($1,$2,$3,$4,$5,$6,$7)')

define(`_VEC_FUNC_2V_PROJ_IDX',`_VEC_FUNC_FAST_2V_PROJ_IDX($1,$2,$3,$4,$5)')
define(`_VEC_FUNC_3V_PROJ',`_VEC_FUNC_FAST_3V_PROJ($1,`',$2,$3,$4,$5)')
define(`_VEC_FUNC_CPX_3V_PROJ',`_VEC_FUNC_FAST_3V_PROJ($1,`CPX_',$2,$3,$4,$5)')

dnl GPU-only stuff???

my_include(`veclib/fast_eqsp_defs.m4')

define(`DECLARE_DBM_INDEX',`GPU_INDEX_TYPE dbmi;')
define(`DECLARE_KERN_ARGS_DBM',`KERNEL_ARG_QUALIFIER bitmap_word *dbm')
define(`SET_INDICES_1SRC',`index2 = dbmi;')
define(`SET_INDICES_2SRCS',`SET_INDICES_1SRC index3 = index2;')

dnl	GENERIC_GPU_FUNC_CALL(name,statement,xxx3,xxx4,scalars,vectors,xxx7)
define(`GENERIC_GPU_FUNC_CALL',`GENERIC_FAST_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')

define(`SLOW_GPU_FUNC_CALL',`')

define(`GENERIC_VEC_FUNC_DBM',`GENERIC_FAST_VEC_FUNC_DBM($1,$2,$3,$4,$5)')


dnl	_VEC_FUNC_2V_CONV( name , dest_type )
define(`_VEC_FUNC_2V_CONV',`_GENERIC_FAST_CONV_FUNC($1,$2)')

define(`GENERIC_FUNC_DECLS',`
GENERIC_FF_DECL($1,$2,$3,$4,$5,$6,$7)
')

define(`_VEC_FUNC_1V_3SCAL',`')	dnl   No fast vramp2d

dnl	Do we need separate GPU definitions?
define(`srcbit',`((*(sbm_ptr + (sbm_bit_offset/BITS_PER_BITMAP_WORD))) & NUMBERED_BIT(sbm_bit_offset))')
define(`srcbit1',`((*(sbm1_ptr + (sbm1_bit_offset/BITS_PER_BITMAP_WORD))) & NUMBERED_BIT(sbm1_bit_offset))')
define(`srcbit2',`((*(sbm2_ptr + (sbm2_bit_offset/BITS_PER_BITMAP_WORD))) & NUMBERED_BIT(sbm2_bit_offset))')

