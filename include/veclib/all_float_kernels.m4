dnl	gen_float_calls declares all of the float funcs...

include(`../../include/veclib/fast_defs.m4')
// all_float_kernels.m4:  CALLING gen_float_calls.m4 with fast defs
define(`INCLUDE_FFT_FUNCS',`')
include(`../../include/veclib/gen_float_calls.m4')
undefine(`INCLUDE_FFT_FUNCS')

ifdef(`BUILD_FOR_CUDA',`
include(`../../include/veclib/flen_defs.m4')
// all_float_kernels.m4:  CALLING gen_float_calls.m4 with flen defs
include(`../../include/veclib/gen_float_calls.m4')
',`')

include(`../../include/veclib/eqsp_defs.m4')
// all_float_kernels.m4:  CALLING gen_float_calls.m4 with eqsp defs
include(`../../include/veclib/gen_float_calls.m4')

include(`../../include/veclib/slow_defs.m4')
// all_float_kernels.m4:  CALLING gen_float_calls.m4 with slow defs
include(`../../include/veclib/gen_float_calls.m4')

