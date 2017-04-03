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

define(`this_sbm_bit',`(sbm_bit_idx)')

dnl Need backslashes for quoting (OpenCL)

define(`SET_INDICES_DBM',`			\
	SET_INDEX(dbmi)				\
	i_dbm_word = dbmi;			\
	dbmi *= BITS_PER_BITMAP_WORD;		\
')

define(`DECL_BASIC_INDICES_DBM',`		\
	unsigned int i_dbm_bit;			\
	int i_dbm_word; bitmap_word dbm_bit;	\
')

define(`SET_INDICES_DBM_1S_',`			\
	i_dbm_word = THREAD_INDEX_X;		\
')

dnl	Are these just deferred, or are they not needed???
define(`_VEC_FUNC_MM_NOCC',`_VEC_FUNC_FAST_MM_NOCC($1,$2,$3,$4,$5,$6)')
define(`_VEC_FUNC_2V_PROJ',`')
define(`_VEC_FUNC_CPX_2V_PROJ',`')
define(`_VEC_FUNC_QUAT_2V_PROJ',`')
define(`_VEC_FUNC_2V_PROJ_IDX',`')
define(`_VEC_FUNC_3V_PROJ',`')
define(`_VEC_FUNC_CPX_3V_PROJ',`')

include(`../../include/veclib/fast_eqsp_defs.m4')

define(`DECLARE_DBM_INDEX',`GPU_INDEX_TYPE dbmi;')
define(`DECLARE_KERN_ARGS_DBM',`KERNEL_ARG_QUALIFIER bitmap_word *dbm')
define(`SET_INDICES_1SRC',`index2 = dbmi;')

dnl	// vmaxg etc - require contiguous, fast only
dnl	BUG?  true for gpu only???
dnl	should this really be here?

dnl	move to gpu_special_defs.m4 also...
dnl	define(`_VEC_FUNC_MM_NOCC',`__VEC_FUNC_FAST_MM_NOCC($1,$5,$6)')

dnl	Moved to gpu_special_defs.m4

dnl	// vmaxv, vminv, vsum

dnl	// on gpu only fast version, but on cpu only slow version!?

dnl	define(`_VEC_FUNC_2V_PROJ',`
dnl		__VEC_FUNC_FAST_2V_PROJ_SETUP($1,$4)
dnl		__VEC_FUNC_FAST_2V_PROJ_HELPER($1,$4)
dnl	')

dnl	define(`_VEC_FUNC_CPX_2V_PROJ',`
dnl		__VEC_FUNC_CPX_FAST_2V_PROJ_SETUP($1,$4,$5) 
dnl		__VEC_FUNC_CPX_FAST_2V_PROJ_HELPER($1,$4,$5)
dnl	')

dnl	define(`_VEC_FUNC_QUAT_2V_PROJ',`
dnl		__VEC_FUNC_QUAT_FAST_2V_PROJ_SETUP($1,$4,$5,$6,$7) 
dnl		__VEC_FUNC_QUAT_FAST_2V_PROJ_HELPER($1,$4,$5,$6,$7)
dnl	')

dnl  No more kernels?
dnl	dnl	_VEC_FUNC_3V_PROJ( func_name, s1, s2, gpu_expr1, gpu_expr2 )
dnl	dnl define(`_VEC_FUNC_3V_PROJ',`__VEC_FUNC_FAST_3V_PROJ($1)')
dnl	define(`_VEC_FUNC_3V_PROJ',`
dnl	__VEC_FUNC_FAST_3V_PROJ_SETUP($1)
dnl	__VEC_FUNC_FAST_3V_PROJ_HELPER($1)
dnl	')
dnl	
dnl	dnl	 _VEC_FUNC_CPX_3V_PROJ( func_name, s1, s2, gpu_r1, gpu_i1, gpu_r2, gpu_i2 )
dnl	dnl define(`_VEC_FUNC_CPX_3V_PROJ',`__VEC_FUNC_CPX_FAST_3V_PROJ($1)')
dnl	define(`_VEC_FUNC_CPX_3V_PROJ',`
dnl	__VEC_FUNC_CPX_FAST_3V_PROJ_SETUP($1)
dnl	__VEC_FUNC_CPX_FAST_3V_PROJ_HELPER($1)
dnl	')

dnl	define(`_VEC_FUNC_3V_PROJ',`')
dnl	define(`_VEC_FUNC_CPX_3V_PROJ',`')
dnl	define(`_VEC_FUNC_2V_PROJ_IDX',`__VEC_FUNC_FAST_2V_PROJ_IDX($1,$4,$5)')

// There is only one function - rvdot, cvdot - so it is implemented in a non-general way.
// Therefore, we don't have to pass the statements or expressions...




// Why is it that only CUDA needs the len versions???

define(`GENERIC_GPU_FUNC_CALL',`GENERIC_FAST_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')

define(`SLOW_GPU_FUNC_CALL',`')

define(`GENERIC_VEC_FUNC_DBM',`GENERIC_FAST_VEC_FUNC_DBM($1,$2,$3,$4,$5)')


define(`_VEC_FUNC_2V_CONV',`_GENERIC_FAST_CONV_FUNC($1,$2)')

define(`GENERIC_FUNC_DECLS',`
/* generic_func_decls /$1/ /$2/ /$3/ /$4/ /$5/ /$6/ /$7/  BEGIN */
/* generic_func_decls calling generic_ff_decl */
GENERIC_FF_DECL($1,$2,$3,$4,$5,$6,$7)
/* generic_func_decls back from generic_ff_decl */
/* generic_func_decls /$1/ /$2/ /$3/ /$4/ /$5/ /$6/ /$7/  DONE */
')

define(`_VEC_FUNC_1V_3SCAL',`')	dnl   No fast vramp2d

