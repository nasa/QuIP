
divert(-1)	dnl suppress output

dnl define(`suppressing',`')

ifdef(`suppressing',`
/* Suppressing ! */
define(`suppress_if',`divert(-1)')
',`
/* NOT Suppressing ! */
define(`suppress_if',`divert(0)')
')

suppress_if dnl suppresss output

/* `suppress_if' = suppress_if */
/* cu2_veclib.m4 BEGIN */

define(`BUILD_FOR_GPU',`')

include(`../../include/veclib/vecgen.m4')

define(`BUILDING_KERNELS',`')
include(`cu2_kernels.m4')
undefine(`BUILDING_KERNELS')

// That declares the kernels - now the host-side functions

include(`cu2_host_funcs.m4')

include(`cu2_typtbl.m4')

/* cu2_veclib.m4 END */

