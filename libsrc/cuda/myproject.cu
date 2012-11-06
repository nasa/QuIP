
#include "quip_config.h"

#ifdef HAVE_CUDA

char VersionId_cuda_myproject[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include <cutil.h>
#include <cutil_inline.h>

#include "my_cuda.h"
#include "cuda_supp.h"	// decribe_cuda_error()
#include "vecgen.h"

// gpu kernels
#include "myproject_kernel.cu"

#include "cuda.h"
#include "cufft.h"

#include "host_calls.h"

#include "undefs.h"
#include "all_funcs.h"


#endif /* HAVE_CUDA */


