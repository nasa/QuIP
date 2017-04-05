
define(`QUATERNION_SUPPORT',`')
dnl #undef USE_SSE
dnl #undef BUILD_FOR_GPU

// PORT put things here...
// PORT - This is something that is different between cuda and opencl!

define(`pf_str',`vl2')

include(`../../include/veclib/gen_port.m4')

dnl	INSURE_PLATFORM_DEVICE(dp)	/* nop */
define(`INSURE_PLATFORM_DEVICE',`')
define(`GLOBAL_QUALIFIER',`')
dnl	GET_MAX_THREADS( dp )		/* nop */
define(`GET_MAX_THREADS',`')

extern void *TMPVEC_NAME(Platform_Device *pdp, size_t size, size_t len, const char *whence);
extern void FREETMP_NAME(void *a, const char *whence);

#include <math.h>	// isinf etc

dnl  include these lines if we use functions from libgsl?
dnl #ifdef HAVE_GSL
dnl #include "gsl/gsl_sf_gamma.h"
dnl #endif // HAVE_GSL


