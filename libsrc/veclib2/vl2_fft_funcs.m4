divert(-1)	dnl suppress output

include(`veclib/vl2_port.m4')

my_include(`veclib/vl2_veclib_prot.m4')

suppress_no
#include "veclib/fftsupp.h"
#include "veclib/vl2_veclib_prot.h"
suppress_if

my_include(`cpu_call_defs.m4')

dnl // That declares the kernels - now the host-side functions
dnl // But for cpu, the kernels ARE the host functions...
dnl // So we only need the untyped functions...

define(`BUILDING_KERNELS',`')

suppress_no

my_include(`veclib/sp_defs.m4')
my_include(`vl2_typed_fft_funcs.m4')
my_include(`veclib/dp_defs.m4')
my_include(`vl2_typed_fft_funcs.m4')

undefine(`BUILDING_KERNELS')

my_include(`veclib/sp_defs.m4')
my_include(`vl2_typed_fft_funcs.m4')
my_include(`veclib/dp_defs.m4')
my_include(`vl2_typed_fft_funcs.m4')

