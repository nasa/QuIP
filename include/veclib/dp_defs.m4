
define(`absfunc',`fabs($1)')
define(`std_type',`double')
define(`std_cpx',`DP_Complex')
define(`std_quat',`DP_Quaternion')
define(`dest_type',`double')
define(`dest_cpx',`DP_Complex')
define(`dest_quat',`DP_Quaternion')
define(`type_code',`dp')
define(`std_scalar',`u_d')
define(`std_cpx_scalar',`u_dpc')
define(`std_quat_scalar',`u_dpq')

define(`ASSIGN_CPX',$1 = $2)
define(`ASSIGN_QUAT',$1 = $2)

define(`REQUIRED_DST_PREC',`PREC_DP')
define(`REQUIRED_SRC_PREC',`PREC_DP')

define(`MY_CLFFT_PRECISION',`CLFFT_DOUBLE')

my_include(`../../include/veclib/dp_func_defs.m4')

