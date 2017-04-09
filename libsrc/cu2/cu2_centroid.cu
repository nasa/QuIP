#include "quip_config.h"

#ifdef HAVE_CUDA

//#include "quip_version.h"
#define BUILD_FOR_CUDA

#include <cuda.h>
#include <curand.h>
#if CUDA_VERSION >= 5000
// but gone on 7.5...
//#include <helper_cuda.h>
#else
#include <cutil.h>
#include <cutil_inline.h>
#endif

#include "quip_prot.h"
#include "my_cuda.h"
#include "cuda_supp.h"

#include "cu2_centroid_expanded.c"

#endif /* HAVE_CUDA */

