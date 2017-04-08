

define(`BUILD_FOR_HOST',`')

include(`vl2_host_call_defs.m4')
dnl	First the typed functions...
include(`../../include/veclib/host_typed_call_defs.m4')
include(`../../include/veclib/gen_host_calls.m4')

include(`vl2_host_untyped_call_defs.m4')
include(`../../include/veclib/host_fft_funcs.m4')

