
#ifndef _CU2_KERNELS_C_
#define _CU2_KERNELS_C_

#include "veclib/gen_gpu_calls.h"
#include "veclib/gpu_args.h"

// Setting BUILDING_KERNELS inhibits the mixed-precision routines...
// Mixed precision complex structure assignment does not work...

//#define BUILDING_KERNELS
#include "veclib/gen_kernel_calls.c"	// not just host calls...
//#undef BUILDING_KERNELS

#include "veclib/bit_defs.h"
#include "veclib/bitmap_ops.h"
#include "veclib/bitmap_ops.c"
#include "veclib/type_undefs.h"

#endif // ! _CU2_KERNELS_C_

