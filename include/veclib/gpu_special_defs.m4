
// vmaxg etc - require contiguous, fast only

define(`_VEC_FUNC_FAST_MM_NOCC',`__VEC_FUNC_FAST_MM_NOCC($1,$5,$6)')

// vmaxv, vminv, vsum

// on gpu only fast version, but on cpu only slow version!?

define(`_VEC_FUNC_FAST_2V_PROJ',`
	__VEC_FUNC_FAST_2V_PROJ_SETUP($1,$4)
	__VEC_FUNC_FAST_2V_PROJ_HELPER($1,$4)
')

define(`_VEC_FUNC_FAST_CPX_2V_PROJ',`
	__VEC_FUNC_CPX_FAST_2V_PROJ_SETUP($1,$4,$5) 
	__VEC_FUNC_CPX_FAST_2V_PROJ_HELPER($1,$4,$5)
')

define(`_VEC_FUNC_FAST_QUAT_2V_PROJ',`
	__VEC_FUNC_QUAT_FAST_2V_PROJ_SETUP($1,$4,$5,$6,$7) 
	__VEC_FUNC_QUAT_FAST_2V_PROJ_HELPER($1,$4,$5,$6,$7)
')

define(`_VEC_FUNC_FAST_3V_PROJ',`')
define(`_VEC_FUNC_FAST_CPX_3V_PROJ',`')
define(`_VEC_FUNC_FAST_2V_PROJ_IDX',`__VEC_FUNC_FAST_2V_PROJ_IDX($1,$4,$5)')

