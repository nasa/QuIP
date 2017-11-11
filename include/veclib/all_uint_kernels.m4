
my_include(`veclib/fast_defs.m4')
my_include(`veclib/gen_uint_calls.m4')

ifdef(`BUILD_FOR_CUDA',`
my_include(`veclib/flen_defs.m4')
my_include(`veclib/gen_uint_calls.m4')
')

my_include(`veclib/eqsp_defs.m4')
my_include(`veclib/gen_uint_calls.m4')

my_include(`veclib/slow_defs.m4')
my_include(`veclib/gen_uint_calls.m4')

