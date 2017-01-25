#ifndef _VL2_PORT_H_
#define _VL2_PORT_H_

#define QUATERNION_SUPPORT
//#undef USE_SSE
#undef BUILD_FOR_GPU

// PORT put things here...
// PORT - This is something that is different between cuda and opencl!

#ifdef pf_str
#undef pf_str
#endif // pf_str
#define pf_str			vl2

#include "veclib/gen_port.h"

#define INSURE_PLATFORM_DEVICE(dp)	/* nop */
#define GLOBAL_QUALIFIER
#define GET_MAX_THREADS( dp )		/* nop */
extern void *TMPVEC_NAME(Platform_Device *pdp, size_t size, size_t len, const char *whence);
extern void FREETMP_NAME(void *a, const char *whence);

#include <math.h>	// isinf etc

// include these lines if we use functions from libgsl?
//#ifdef HAVE_GSL
//#include "gsl/gsl_sf_gamma.h"
//#endif // HAVE_GSL

#endif // ! _VL2_PORT_H_

