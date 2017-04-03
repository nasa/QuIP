divert(0)
#ifdef HAVE_MATH_H
#include <math.h>
#endif // HAVE_MATH_H
suppress_if

define(`BUILDING_KERNELS',`')

include(`../../include/veclib/vl2_veclib_prot.m4')	// declare all the prototypes for the host

include(`../../include/veclib/fftsupp.m4')	// prototypes for some external helpers

include(`cpu_call_defs.m4')

dnl // Why define "host" calls in kernel defs?
dnl include(`../../include/veclib/gen_host_calls.m4')	// all the precisions

// vl2_kernels.m4:  including gen_kernel_calls.m4
include(`../../include/veclib/gen_kernel_calls.m4')
// vl2_kernels.m4:  back from gen_kernel_calls.m4

// include fft stuff?
// Now include the special cases...

include(`../../include/veclib/sp_defs.m4')
dnl include(`vfft.m4')
include(`linear.m4')

include(`../../include/veclib/dp_defs.m4')
dnl include(`vfft.m4')
include(`linear.m4')


undefine(`BUILDING_KERNELS')

