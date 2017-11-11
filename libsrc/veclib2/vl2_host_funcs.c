
// First the typed functions...

#undef BUILD_FOR_CPU
#define BUILD_FOR_HOST

#include "vl2_host_call_defs.h"
#include "veclib/host_typed_call_defs.h"
#include "veclib/gen_host_calls.c"
#include "vl2_host_untyped_call_defs.h"
#include "veclib/host_fft_funcs.c"

