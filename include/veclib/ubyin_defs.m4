/* ubyin_defs.m4 BEGIN */
define(`std_type',`u_char')
define(`std_scalar',`u_ub')
define(`std_signed',`char')
define(`dest_type',`short')
define(`ALL_ONES',`0xffff')
// abs Broken on CUDA 6?
define(`absfunc',($1<0?(-$1):$1))
define(`type_code',`ubyin')
/* ubyin_defs.m4 END */
