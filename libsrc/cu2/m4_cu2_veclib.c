#include "quip_config.h"

#ifdef HAVE_CUDA

// added to try to resolve wierd compilation issues...
#ifndef __has_extension
#define __has_extension(x) 0
#endif
// add these lines because gcc does not support blocks
#define vImage_Utilities_h
#define vImage_CVUtilities_h

//#include <stdio.h>
#include "quip_prot.h"	// where does this need to go?

#define BUILD_FOR_CUDA

#include "my_cu2.h"
#include "cuda_supp.h"
#include "veclib_api.h"
#include "veclib/cu2_veclib_prot.h"	// declare all the prototypes for the host

#include "cu2_veclib_expanded.c"

#endif /* HAVE_CUDA */


