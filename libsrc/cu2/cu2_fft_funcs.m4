divert(-1)	dnl suppress output

include(`veclib/cu2_port.m4')

my_include(`veclib/cu2_veclib_prot.m4')

suppress_no
#include "veclib/fftsupp.h"
#include "veclib/cu2_veclib_prot.h"
suppress_if

dnl first include the kernels...
dnl cu2_kernels.m4 includes veclib/gen_kernel_calls.m4...

dnl my_include(`cpu_call_defs.m4')

dnl // That declares the kernels - now the host-side functions
dnl // But for cpu, the kernels ARE the host functions...
dnl // So we only need the untyped functions...
dnl 
dnl my_include(`cu2_host_funcs.m4')
dnl 
dnl my_include(`cu2_typtbl.m4')

my_include(`cu2_kern_call_defs.m4')

define(`BUILDING_KERNELS',`')

my_include(`veclib/sp_defs.m4')

suppress_no
/* cu2_fft_funcs.m4:  including veclib/fft_funcs.m4 */
dnl my_include(`veclib/fft_funcs.m4')
my_include(`cu2_typed_fft_funcs.m4')

my_include(`veclib/dp_defs.m4')
my_include(`cu2_typed_fft_funcs.m4')

undefine(`BUILDING_KERNELS')

my_include(`cu2_host_call_defs.m4')

my_include(`veclib/sp_defs.m4')
my_include(`cu2_typed_fft_funcs.m4')
my_include(`veclib/dp_defs.m4')
my_include(`cu2_typed_fft_funcs.m4')

