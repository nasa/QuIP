
divert(-1)	dnl suppress output

dnl define(`suppressing',`')

ifdef(`suppressing',`
/* Suppressing ! */
define(`suppress_if',`divert(-1)')
',`
/* NOT Suppressing ! */
define(`suppress_if',`divert(0)')
')

include(`../../include/veclib/vl2_port.m4')
include(`../../include/veclib/vl2_veclib_prot.m4')

include(`vl2_kernels.m4')

// That declares the kernels - now the host-side functions
// But for cpu, the kernels ARE the host functions...
// So we only need the untyped functions...

include(`vl2_host_funcs.m4')

include(`vl2_typtbl.m4')


