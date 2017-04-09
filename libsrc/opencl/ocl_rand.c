/* Use threefry parallel random number generator from Random123 package */

#include "quip_config.h"

#ifdef HAVE_OPENCL

#include "quip_prot.h"
#include "my_ocl.h"
#include "ocl_platform.h"
#include <string.h>	// strcmp - for debugging
#include <sys/time.h>	// gettimeofday

#include "vuni.i"	// kernel source string

// The generator has no random seed currently, so it
// will always produce the same "random" data!?
static int ocl_sp_vuni_counter=0;
static int ocl_sp_vuni_seeded=0;
static int ocl_sp_vuni_inited=0;
static cl_kernel ocl_sp_vuni_kernel = NULL;

// BUG the kernel is created for the device,
// so this code is broken in the case that we have multiple devices

static int ocl_sp_vuni_init(Platform_Device *pdp)
{
	struct timeval tv;

	ocl_sp_vuni_inited=1;
	// BUG need to be able to name the source string!?
	if( (ocl_sp_vuni_kernel=ocl_make_kernel(opencl_src,"g_ocl_fast_sp_vuni",pdp))
			== NULL ){
		NWARN("Error creating kernel for g_ocl_fast_sp_vuni!?");
		return -1;
	}
	if( !ocl_sp_vuni_seeded ){
		// "randomize" the seed so that we don't get the same numbers every time
		if( gettimeofday(&tv,NULL) < 0 ){
			NWARN("ocl_sp_vuni_init:  error calling gettimeofday, not setting random seed!?");
		} else {
			ocl_sp_vuni_counter = 100*(tv.tv_usec / 100);
			ocl_sp_vuni_counter += tv.tv_sec % 100;
		}
	}
	return 0;
}

void h_ocl_set_seed(int seed)
{
	ocl_sp_vuni_counter = seed;
	ocl_sp_vuni_seeded = 1;
}

#include "ocl_rand_expanded.c"

#endif // HAVE_OPENCL

