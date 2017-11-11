
define(`std_type',`short')
define(`std_scalar',`u_s')
define(`dest_type',`short')
define(`ALL_ONES',`0xffff')
dnl	// abs Broken on CUDA 6?
dnl	define(`absfunc',`($1<0?(-$1):$1)')
define(`absfunc',`abs((int)$1)')
define(`type_code',`in')

