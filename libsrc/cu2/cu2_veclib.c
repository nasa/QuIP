#include "quip_config.h"

#ifdef HAVE_CUDA

// added to try to resolve wierd compilation issues...
#ifndef __has_extension
#define __has_extension(x) 0
#endif
// add these lines because gcc does not support blocks
#define vImage_Utilities_h
#define vImage_CVUtilities_h

#include <stdio.h>

#define BUILD_FOR_CUDA

#include "my_cu2.h"
#include "cuda_supp.h"
#include "veclib_api.h"
#include "veclib/cu2_port.h"
#include "veclib/cu2_veclib_prot.h"	// declare all the prototypes for the host
// We used to unpack the macros, to get better compiler error messages,
// BUT with kernels being defines as strings, now the simple-minded algorithm
// for breaking lines at braces and semicolons breaks the code;
// The functions that are now put with quotes HAVE to be on one line!

#include "quip_prot.h"	// where does this need to go?

//#include "foo.nosearch"

#include "cu2_kern_call_defs.h"
#include "cu2_kernels.c"

// That declares the kernels - now the host-side functions

#include "cu2_host_call_defs.h"
#include "cu2_host_funcs.c"

#include "cu2_typtbl.c"

#endif /* HAVE_CUDA */


