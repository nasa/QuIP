
define(`type_code',`by')
define(`std_type',`char')
define(`std_scalar',`u_b')
define(`dest_type',`char')
define(`ALL_ONES',`0xff')

dnl	// Broken on CUDA 6?
dnl	// Not broken, but needs cast to avoid using host function (C++)
define(`absfunc',`abs((int)$1)')


