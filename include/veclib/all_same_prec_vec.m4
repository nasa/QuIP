dnl	Functions for which mixed precision makes no sense - rvset and rvmov

dnl	// bit precision handled in int_bit_vec.m4
ifdef(`BIT_PRECISION',`',`
_VEC_FUNC_1V_SCAL( rvset, dst = scalar1_val )
_VEC_FUNC_2V( rvmov, dst = src1 )
')

