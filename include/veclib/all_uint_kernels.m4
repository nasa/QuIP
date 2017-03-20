
include(`../../include/veclib/fast_defs.m4')
include(`../../include/veclib/gen_uint_calls.m4')

ifdef(`BUILD_FOR_CUDA',`
include(`../../include/veclib/flen_defs.m4')
include(`../../include/veclib/gen_uint_calls.m4')
')

include(`../../include/veclib/eqsp_defs.m4')
include(`../../include/veclib/gen_uint_calls.m4')

include(`../../include/veclib/slow_defs.m4')
include(`../../include/veclib/gen_uint_calls.m4')

