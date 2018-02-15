#include "quip_config.h"

#ifdef HAVE_CUDA

//#include "quip_version.h"
#define BUILD_FOR_CUDA

#include <cuda.h>
#include <curand.h>
#if CUDA_VERSION >= 5000

// but gone on 7.5...
#if CUDA_VERSION < 7050
#include <helper_cuda.h>
#endif // CUDA_VERSION < 7050

#else // CUDA_VERSION < 5000
#include <cutil.h>
#include <cutil_inline.h>
#endif

#include "quip_prot.h"
#include "my_cu2.h"
#include "cuda_supp.h"

#include "cu2_centroid_expanded.c"

#endif /* HAVE_CUDA */

