dnl	gen_float_calls declares all of the float funcs...

my_include(`../../include/veclib/fast_defs.m4')

dnl	no eqsp or slow fft funcs...
define(`INCLUDE_FFT_FUNCS',`')
suppress_no
// all_float_kernels.m4:  CALLING gen_float_calls.m4 with fast defs
my_include(`../../include/veclib/gen_float_calls.m4')
undefine(`INCLUDE_FFT_FUNCS')

ifdef(`BUILD_FOR_CUDA',`
my_include(`../../include/veclib/flen_defs.m4')
// all_float_kernels.m4:  CALLING gen_float_calls.m4 with flen defs
my_include(`../../include/veclib/gen_float_calls.m4')
',`')

my_include(`../../include/veclib/eqsp_defs.m4')
suppress_no
// all_float_kernels.m4:  CALLING gen_float_calls.m4 with eqsp defs
my_include(`../../include/veclib/gen_float_calls.m4')

my_include(`../../include/veclib/slow_defs.m4')
suppress_no
// all_float_kernels.m4:  CALLING gen_float_calls.m4 with slow defs
my_include(`../../include/veclib/gen_float_calls.m4')

