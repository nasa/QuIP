
divert(-1)	dnl suppress output

include(`veclib/vl2_port.m4')

dnl	USE_SSE should not be needed any more, because the modern
dnl	compilers do this on their own, no need to provide special
dnl	code for this...
define(`USE_SSE',`')	dnl BUG how to import this from config.h?

my_include(`veclib/vl2_veclib_prot.m4')

my_include(`vl2_kernels.m4')

// That declares the kernels - now the host-side functions
// But for cpu, the kernels ARE the host functions...
// So we only need the untyped functions...

my_include(`vl2_host_funcs.m4')

my_include(`vl2_typtbl.m4')


