
define(`TMPVEC_NAME',`_TMPVEC_NAME(pf_str)')
define(`_TMPVEC_NAME',$1`_tmp_vec')

define(`FREETMP_NAME',`_FREETMP_NAME(pf_str)')
define(`_FREETMP_NAME',$1`_free_tmp')

define(`PF_COMMAND_FUNC',`COMMAND_FUNC( MENU_FUNC_NAME($1) )')
define(`PF_FUNC_NAME',`PLATFORM_SYMBOL_NAME($1)')
define(`PLATFORM_SYMBOL_NAME',`pf_str`_$1')

define(`MENU_FUNC_NAME',`do_'pf_str`_'$1)


// this is really a host call...
define(`HOST_CALL_NAME',`_XXX_CALL_NAME(h,$1)')
define(`HOST_CALL_NAME_REAL',`_XXX_CALL_NAME_REAL(h,$1)')
define(`HOST_CALL_NAME_CPX',`_XXX_CALL_NAME_CPX(h,$1)')
define(`HOST_CALL_NAME_QUAT',`_XXX_CALL_NAME_QUAT(h,$1)')
define(`HOST_CALL_NAME_MIXED',`_XXX_CALL_NAME_MIXED(h,$1)')
define(`HOST_CALL_NAME_QMIXD',`_XXX_CALL_NAME_QMIXD(h,$1)')

// XXX_CALL

define(`_XXX_CALL_NAME',`$1`_'pf_str`_'$2')
define(`_XXX_CALL_NAME_REAL',`$1`_'pf_str`_r'$2')
define(`_XXX_CALL_NAME_CPX',`$1`_'pf_str`_c'$2')
define(`_XXX_CALL_NAME_QUAT',`$1`_'pf_str`_q'$2')
define(`_XXX_CALL_NAME_MIXED',`$1`_'pf_str`_m'$2')
define(`_XXX_CALL_NAME_QMIXD',`$1`_'pf_str`_p'$2')

define(`_XXX_TYPED_CALL_NAME',`_XXX_CALL_NAME($1,$3`_'$2)')

// CPU_CALL
 
define(`CPU_CALL_NAME',`_XXX_CALL_NAME(c,$1)')
define(`CPU_CALL_NAME_REAL',`_XXX_CALL_NAME_REAL(c,$1)')
define(`CPU_CALL_NAME_CPX',`_XXX_CALL_NAME_CPX(c,$1)')
define(`CPU_CALL_NAME_QUAT',`_XXX_CALL_NAME_QUAT(c,$1)')
define(`CPU_CALL_NAME_MIXED',`_XXX_CALL_NAME_MIXED(c,$1)')
define(`CPU_CALL_NAME_QMIXD',`_XXX_CALL_NAME_QMIXD(c,$1)')

// GPU_CALL
define(`GPU_CALL_NAME',`GPU_TYPED_CALL_NAME($1,type_code)')
define(`GPU_TYPED_CALL_NAME',`_XXX_TYPED_CALL_NAME(g,$1,$2)')

// PF_TYPED

define(`PF_TYPED_CALL_NAME',``h_'pf_str`_'type_code`_'$1'')
define(`PF_FFT_CALL_NAME',`pf_str`_fft_'type_code`_'$1'')

define(`PF_TYPED_CALL_NAME_CPX',`pf_str`_'$2`_c'$1')
define(`PF_TYPED_CALL_NAME_REAL',`pf_str`_'$2`_r'$1')
define(`_XXX_TYPED_CALL_NAME',`$1`_'pf_str`_'$3`_'$2')

// HOST_TYPED

define(`HOST_TYPED_CALL_NAME',`_XXX_TYPED_CALL_NAME(h,$1,$2)')
define(`HOST_TYPED_CALL_NAME_REAL',`_XXX_TYPED_CALL_NAME(h,`r'$1,$2)')
define(`HOST_TYPED_CALL_NAME_CPX',`_XXX_TYPED_CALL_NAME(h,`c'$1,$2)')
define(`HOST_TYPED_CALL_NAME_MIXED',`_XXX_TYPED_CALL_NAME(h,`m'$1,$2)')
define(`HOST_TYPED_CALL_NAME_QUAT',`_XXX_TYPED_CALL_NAME(h,`q'$1,$2)')
define(`HOST_TYPED_CALL_NAME_QMIXD',`_XXX_TYPED_CALL_NAME(h,`p'$1,$2)')

define(`NAME_WITH_SUFFIX',``h_'pf_str`_'type_code`_'$1$2')
define(`SETUP_NAME',`NAME_WITH_SUFFIX($1,`_setup')')
define(`HELPER_NAME',`NAME_WITH_SUFFIX($1,`_helper')')
define(`INDEX_SETUP_NAME',`SETUP_NAME($1)')
define(`INDEX_HELPER_NAME',`HELPER_NAME($1)')

define(`MM_HELPER_NAME',`HELPER_NAME($1)')

define(`NOCC_SETUP_NAME',`SETUP_NAME($1)')
define(`NOCC_HELPER_NAME',`HELPER_NAME($1)')

define(`CONV_FUNC_NAME',``h_'pf_str`_v'$2`2'$3')

define(`_XXX_SPEED_CALL_NAME',`$1`_'pf_str`_'$3`_'type_code`_'$2')
define(`_XXX_FAST_CALL_NAME',`_XXX_SPEED_CALL_NAME($1,$2,fast)')
define(`_XXX_EQSP_CALL_NAME',`_XXX_SPEED_CALL_NAME($1,$2,eqsp)')
define(`_XXX_SLOW_CALL_NAME',`_XXX_SPEED_CALL_NAME($1,$2,slow)')
define(`_XXX_FLEN_CALL_NAME',`_XXX_SPEED_CALL_NAME($1,$2,flen)')
define(`_XXX_ELEN_CALL_NAME',`_XXX_SPEED_CALL_NAME($1,$2,elen)')
define(`_XXX_SLEN_CALL_NAME',`_XXX_SPEED_CALL_NAME($1,$2,slen)')

// these are the kernel names
define(`GPU_FAST_CALL_NAME',`_XXX_FAST_CALL_NAME(g,$1)')
define(`GPU_EQSP_CALL_NAME',`_XXX_EQSP_CALL_NAME(g,$1)')
define(`GPU_SLOW_CALL_NAME',`_XXX_SLOW_CALL_NAME(g,$1)')
define(`GPU_FLEN_CALL_NAME',`_XXX_FLEN_CALL_NAME(g,$1)')
define(`GPU_ELEN_CALL_NAME',`_XXX_ELEN_CALL_NAME(g,$1)')
define(`GPU_SLEN_CALL_NAME',`_XXX_SLEN_CALL_NAME(g,$1)')

define(`HOST_FAST_CALL_NAME',`_XXX_FAST_CALL_NAME(h,$1)')
define(`HOST_EQSP_CALL_NAME',`_XXX_EQSP_CALL_NAME(h,$1)')
define(`HOST_SLOW_CALL_NAME',`_XXX_SLOW_CALL_NAME(h,$1)')
define(`HOST_FLEN_CALL_NAME',`_XXX_FLEN_CALL_NAME(h,$1)')
define(`HOST_ELEN_CALL_NAME',`_XXX_ELEN_CALL_NAME(h,$1)')
define(`HOST_SLEN_CALL_NAME',`_XXX_SLEN_CALL_NAME(h,$1)')

define(`CPU_FAST_CALL_NAME',`_XXX_FAST_CALL_NAME(c,$1)')
define(`CPU_EQSP_CALL_NAME',`_XXX_EQSP_CALL_NAME(c,$1)')
define(`CPU_SLOW_CALL_NAME',`_XXX_SLOW_CALL_NAME(c,$1)')
define(`CPU_FLEN_CALL_NAME',`_XXX_FLEN_CALL_NAME(c,$1)')
define(`CPU_ELEN_CALL_NAME',`_XXX_ELEN_CALL_NAME(c,$1)')
define(`CPU_SLEN_CALL_NAME',`_XXX_SLEN_CALL_NAME(c,$1)')

