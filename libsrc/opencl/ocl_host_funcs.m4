
include(`ocl_host_call_defs.m4')

// First the typed functions...

include(`../../include/veclib/host_typed_call_defs.m4')

divert(0) dnl enable output
include(`../../include/veclib/gen_host_calls.m4')
suppress_if

