#include "quip_config.h"

#ifdef HAVE_OPENCL

#include <stdio.h>
#include <string.h>
#include "quip_prot.h"
#include "my_ocl.h"
#include "veclib/ocl_port.h"
#include "veclib_api.h"
#include "veclib/ocl_veclib_prot.h"
#include "ocl_platform.h"

// We used to unpack the macros, to get better compiler error messages,
// BUT with kernels being defines as strings, now the simple-minded algorithm
// for breaking lines at braces and semicolons breaks the code;
// The functions that are now put with quotes HAVE to be on one line!
#include "ocl_kernels.c"

// That declares the kernels - now the host-side functions

void insure_ocl_device(Data_Obj *dp)
{
	NWARN("insure_ocl_device:  not implemented!?");
}

#include "ocl_host_funcs.c"

#include "ocl_typtbl.c"

#endif /* HAVE_OPENCL */


