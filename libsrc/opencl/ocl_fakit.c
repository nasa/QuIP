// This file is only used when we don't have the nVidia compiler nvcc.
// It allows us to test other parts of the code.

#include "quip_config.h"

#ifndef HAVE_CUDA

#include "quip_prot.h"
#include "veclib_api.h"
#include "my_vector_functions.h"
#include "host_calls.h"
//#include "conversions.h"
#include "undefs.h"
#include "all_funcs.h"

#endif /* ! HAVE_CUDA */

