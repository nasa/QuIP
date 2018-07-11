divert(-1)	dnl suppress output

include(`veclib/ocl_port.m4')

my_include(`veclib/ocl_veclib_prot.m4')

suppress_no
#include "veclib/fftsupp.h"
#include "veclib/ocl_veclib_prot.h"
suppress_if

dnl first include the kernels...


define(`BUILDING_KERNELS',`')


suppress_no

my_include(`ocl_kern_call_defs.m4')

my_include(`veclib/sp_defs.m4')
my_include(`ocl_typed_fft_funcs.m4')
my_include(`veclib/dp_defs.m4')
my_include(`ocl_typed_fft_funcs.m4')

undefine(`BUILDING_KERNELS')

my_include(`ocl_host_call_defs.m4')

my_include(`veclib/sp_defs.m4')
my_include(`ocl_typed_fft_funcs.m4')
my_include(`veclib/dp_defs.m4')
my_include(`ocl_typed_fft_funcs.m4')

