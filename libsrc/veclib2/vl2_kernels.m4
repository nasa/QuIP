divert(0)
#ifdef HAVE_MATH_H
#include <math.h>
#endif // HAVE_MATH_H
suppress_if

define(`BUILDING_KERNELS',`')

include(`../../include/veclib/vl2_veclib_prot.m4')	// declare all the prototypes for the host

include(`../../include/veclib/fftsupp.m4')	// prototypes for some external helpers

include(`cpu_call_defs.m4')

// Why define "host" calls in kernel defs?
include(`../../include/veclib/gen_host_calls.m4')	// all the precisions

// include fft stuff?
// Now include the special cases...

include(`../../include/veclib/sp_defs.m4')
dnl include(`linear.m4')
include(`linear.m4')

include(`../../include/veclib/dp_defs.m4')
dnl include(`linear.m4')
include(`linear.m4')


undefine(`BUILDING_KERNELS')

