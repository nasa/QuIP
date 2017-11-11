

define(`BUILD_FOR_HOST',`')

my_include(`vl2_host_call_defs.m4')
dnl	First the typed functions...
my_include(`veclib/host_typed_call_defs.m4')
my_include(`veclib/gen_host_calls.m4')

my_include(`vl2_host_untyped_call_defs.m4')
my_include(`veclib/host_fft_funcs.m4')

