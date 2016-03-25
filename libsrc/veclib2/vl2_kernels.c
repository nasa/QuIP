

#include "veclib/vl2_veclib_prot.h"	// declare all the prototypes for the host
#include "veclib/fftsupp.h"	// prototypes for some external helpers

#include "cpu_call_defs.h"

// Why define "host" calls in kernel defs?
#include "veclib/gen_host_calls.c"	// all the precisions

// include fft stuff?
// Now include the special cases...

#include "veclib/sp_defs.h"
#include "vfft.c"
#include "linear.c"
#include "veclib/type_undefs.h"

#include "veclib/dp_defs.h"
#include "vfft.c"
#include "linear.c"
#include "veclib/type_undefs.h"

#include "veclib/bit_defs.h"
#include "veclib/bitmap_ops.c"
#include "veclib/type_undefs.h"

#include "veclib/method_undefs.h"

