
define(`absfunc',`fabs($1)')
define(`std_type',`float')
define(`std_cpx',`SP_Complex')
define(`std_quat',`SP_Quaternion')
define(`dest_type',`double')
define(`dest_cpx',`DP_Complex')
define(`dest_quat',`DP_Quaternion')
define(`type_code',`spdp')
define(`std_scalar',`u_f')
define(`std_cpx_scalar',`u_spc')
define(`std_quat_scalar',`u_spq')

define(`ASSIGN_CPX',`$1.re = $2.re; $1.im = $2.im')

define(`ASSIGN_QUAT',$1.re = $2.re; $1._i = $2._i; $1._j = $2._j; $1._k = $2._k)

// float input and double output -
include(`../../include/veclib/dp_func_defs.m4')

