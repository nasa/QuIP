

my_include(`../../include/veclib/fast_defs.m4')
my_include(`../../include/veclib/gen_mixed_uint_calls.m4')

ifdef(`BUILD_FOR_CUDA',`
my_include(`../../include/veclib/flen_defs.m4')
my_include(`../../include/veclib/gen_mixed_uint_calls.m4')
')

my_include(`../../include/veclib/eqsp_defs.m4')
my_include(`../../include/veclib/gen_mixed_uint_calls.m4')

my_include(`../../include/veclib/slow_defs.m4')
my_include(`../../include/veclib/gen_mixed_uint_calls.m4')

