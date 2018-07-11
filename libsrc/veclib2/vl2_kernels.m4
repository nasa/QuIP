suppress_no
#ifdef HAVE_MATH_H
#include <math.h>
#endif // HAVE_MATH_H
suppress_if

define(`BUILDING_KERNELS',`')

suppress_no
#include "veclib/fftsupp.h"
#include "veclib/vl2_veclib_prot.h"
suppress_if

my_include(`cpu_call_defs.m4')

dnl // Why define "host" calls in kernel defs?
dnl include(`../../include/veclib/gen_host_calls.m4')	// all the precisions

// vl2_kernels.m4:  including gen_kernel_calls.m4
my_include(`veclib/gen_kernel_calls.m4')
// vl2_kernels.m4:  back from gen_kernel_calls.m4

// Now include the special cases...

my_include(`veclib/sp_defs.m4')
my_include(`linear.m4')

my_include(`veclib/dp_defs.m4')
my_include(`linear.m4')


undefine(`BUILDING_KERNELS')

