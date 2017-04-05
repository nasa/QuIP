include(`../../include/veclib/gen_gpu_calls.m4')
include(`../../include/veclib/gpu_special_defs.m4')

define(`KERNEL_ARG_QUALIFIER',`')
define(`KERNEL_FUNC_QUALIFIER',`__global__')

define(`KERNEL_FUNC_PRELUDE',`')

define(`_VEC_FUNC_SLOW_1V_3SCAL',`SLOW_GPU_FUNC_CALL($1,$4,,,_3S,1,)')

dnl	// These can't be defined generically, because for openCL we define them
dnl	// to declare a string variable containing the kernel source code.

dnl	_GENERIC_FAST_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)
define(`_GENERIC_FAST_VEC_FUNC',`__GENERIC_FAST_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')
define(`_GENERIC_EQSP_VEC_FUNC',`__GENERIC_EQSP_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')
define(`_GENERIC_SLOW_VEC_FUNC',`__GENERIC_SLOW_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')
define(`_GENERIC_FLEN_VEC_FUNC',`__GENERIC_FLEN_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')
define(`_GENERIC_ELEN_VEC_FUNC',`__GENERIC_ELEN_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')
define(`_GENERIC_SLEN_VEC_FUNC',`__GENERIC_SLEN_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')

/************** conversions **************/

dnl	_GENERIC_FAST_CONV_FUNC(name,dst_type)
define(`_GENERIC_FAST_CONV_FUNC',`__GENERIC_FAST_CONV_FUNC($1,$2)')
define(`_GENERIC_EQSP_CONV_FUNC',`__GENERIC_EQSP_CONV_FUNC($1,$2)')
define(`_GENERIC_SLOW_CONV_FUNC',`__GENERIC_SLOW_CONV_FUNC($1,$2)')
define(`_GENERIC_FLEN_CONV_FUNC',`__GENERIC_FLEN_CONV_FUNC($1,$2)')
define(`_GENERIC_ELEN_CONV_FUNC',`__GENERIC_ELEN_CONV_FUNC($1,$2)')
define(`_GENERIC_SLEN_CONV_FUNC',`__GENERIC_SLEN_CONV_FUNC($1,$2)')

dnl	// cu2_kern_call_defs testing slow_conv_func
dnl	#ifdef FOOBAR
dnl	// testing 2
dnl	_GENERIC_SLOW_CONV_FUNC(tconv2,char,u_char)
dnl	#endif // FOOBAR

dnl	_GENERIC_FAST_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)

define(`_GENERIC_FAST_VEC_FUNC_DBM',`__GENERIC_FAST_VEC_FUNC_DBM($1,$2,$3,$4,$5)')
define(`_GENERIC_EQSP_VEC_FUNC_DBM',`__GENERIC_EQSP_VEC_FUNC_DBM($1,$2,$3,$4,$5)')
define(`_GENERIC_FLEN_VEC_FUNC_DBM',`__GENERIC_FLEN_VEC_FUNC_DBM($1,$2,$3,$4,$5)')
define(`_GENERIC_ELEN_VEC_FUNC_DBM',`__GENERIC_ELEN_VEC_FUNC_DBM($1,$2,$3,$4,$5)')
define(`_GENERIC_SLOW_VEC_FUNC_DBM',`__GENERIC_SLOW_VEC_FUNC_DBM($1,$2,$3,$4,$5)')
define(`_GENERIC_SLEN_VEC_FUNC_DBM',`__GENERIC_SLEN_VEC_FUNC_DBM($1,$2,$3,$4,$5)')

dnl	__VEC_FUNC_FAST_3V_PROJ_SETUP( func_name )
dnl  just rvdot!?
define(`__VEC_FUNC_FAST_3V_PROJ_SETUP',`
___VEC_FUNC_FAST_3V_PROJ_SETUP($1)
')

define(`__VEC_FUNC_FAST_3V_PROJ_HELPER',`
___VEC_FUNC_FAST_3V_PROJ_HELPER($1)
')


dnl	__VEC_FUNC_CPX_FAST_3V_PROJ_SETUP( func_name )
dnl  just cvdot!?
define(`__VEC_FUNC_CPX_FAST_3V_PROJ_SETUP',`
___VEC_FUNC_CPX_FAST_3V_PROJ_SETUP($1)
')

define(`__VEC_FUNC_CPX_FAST_3V_PROJ_HELPER',`
___VEC_FUNC_CPX_FAST_3V_PROJ_HELPER($1)
')


dnl	__VEC_FUNC_FAST_2V_PROJ_SETUP( func_name, gpu_expr )
define(`__VEC_FUNC_FAST_2V_PROJ_SETUP',`
___VEC_FUNC_FAST_2V_PROJ_SETUP($1,$2)
')

define(`__VEC_FUNC_FAST_2V_PROJ_HELPER',`
___VEC_FUNC_FAST_2V_PROJ_HELPER($1,$2)
')

dnl	__VEC_FUNC_CPX_FAST_2V_PROJ_SETUP( func_name, re_expr, im_expr )
define(`__VEC_FUNC_CPX_FAST_2V_PROJ_SETUP',`
___VEC_FUNC_CPX_FAST_2V_PROJ_SETUP($1,$2,$3)
')

define(`__VEC_FUNC_CPX_FAST_2V_PROJ_HELPER',`
___VEC_FUNC_CPX_FAST_2V_PROJ_HELPER($1,$2,$3)
')

dnl	__VEC_FUNC_QUAT_FAST_2V_PROJ_SETUP( func_name, re_expr, im_expr1, im_expr2, im_expr3 )
define(`__VEC_FUNC_QUAT_FAST_2V_PROJ_SETUP',`
___VEC_FUNC_QUAT_FAST_2V_PROJ_SETUP($1,$2,$3,$4,$5)		
')

define(`__VEC_FUNC_QUAT_FAST_2V_PROJ_HELPER',`
___VEC_FUNC_QUAT_FAST_2V_PROJ_HELPER($1,$2,$3,$4,$5)		
')



dnl	__VEC_FUNC_FAST_2V_PROJ_IDX( func_name, statement1, statement2 )
define(`__VEC_FUNC_FAST_2V_PROJ_IDX',`
___VEC_FUNC_FAST_2V_PROJ_IDX($1,$2,$3)
')

dnl	__VEC_FUNC_FAST_MM_NOCC( func_name, test1, test2 )
define(`__VEC_FUNC_FAST_MM_NOCC',`
	___VEC_FUNC_FAST_MM_NOCC_SETUP($1,$2,$3)
	___VEC_FUNC_FAST_MM_NOCC_HELPER($1,$2,$3)
')

define(`_VEC_FUNC_EQSP_MM_NOCC',`')

dnl	Defns for bitmap ops...

dnl	indices are different for fast and slow!?
define(`srcbit',`((*(sbm_ptr + (sbm_bit_idx/BITS_PER_BITMAP_WORD))) & NUMBERED_BIT(sbm_bit_idx))')
define(`srcbit1',`((*(sbm1_ptr + (sbm1_bit_idx/BITS_PER_BITMAP_WORD))) & NUMBERED_BIT(sbm1_bit_idx))')
define(`srcbit2',`((*(sbm2_ptr + (sbm2_bit_idx/BITS_PER_BITMAP_WORD))) & NUMBERED_BIT(sbm2_bit_idx))')

