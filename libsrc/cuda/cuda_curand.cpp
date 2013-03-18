
/* interface to nVidia curand random number library */

#include "quip_config.h"

char VersionId_cuda_cuda_curand[] = QUIP_VERSION_STRING;

#ifdef HAVE_CUDA

#include <curand.h>

#include "vecgen.h"


#ifdef FOOBAR /* find out what we really need */

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include <cutil_inline.h>

#include "my_cuda.h"

#ifdef HAVE_NPP_H
#include <npp.h>
#endif /* HAVE_NPP_H */

#include "items.h"
#include "debug.h"		/* verbose */
#include "cuda_menu.h"
#include "cuda_error.h"
#include "fileck.h"
#include "nvf.h"		/* is_ram() */
#include "my_vector_functions.h"
#include "menu_calls.h"

#endif /* FOOBAR */

//Cuda_Device *curr_cdp=NO_CUDA_DEVICE;



#ifdef HAVE_LIBCURAND

#define NO_GENERATOR	((curandGenerator_t)NULL)

static curandGenerator_t f_gen=NO_GENERATOR;
/* Does the lib support double? */
/* static curandGenerator_t d_gen=NO_GENERATOR; */

static void init_curand(void)
{
	curandStatus_t s;

	s = curandCreateGenerator(&f_gen,CURAND_RNG_PSEUDO_DEFAULT);
	if( s != CURAND_STATUS_SUCCESS ){
		NWARN("curand error creating generator");
		return;
	}
	// BUG should allow user to seed generator
	s=curandSetPseudoRandomGeneratorSeed(f_gen,1234ULL);
	if( s != CURAND_STATUS_SUCCESS ){
		NWARN("curand error seeding generator");
		return;
	}
}

void g_sp_vuni(Data_Obj *dp)
{
	curandStatus_t s;

	if( f_gen == NO_GENERATOR ){
		init_curand();
	}
	if( is_contiguous(dp) ){
		s = curandGenerateUniform(f_gen,(float *)dp->dt_data, dp->dt_n_type_elts);
		if( s != CURAND_STATUS_SUCCESS ){
			NWARN("curand error generating uniform floats");
			return;
		}
	} else {
		/* Hopefully the rows are contiguous? */
		NWARN("Sorry, need to write code for non-contig g_sp_vuni");
	}
}
#endif /* HAVE_LIBCURAND */


// We use the curand implementation of the random number generator...
void g_vuni( Vec_Obj_Args *oap )
{
#ifdef HAVE_LIBCURAND
	Data_Obj *dp;

	dp = oap->oa_dest;
	switch( dp->dt_prec & MACH_PREC_MASK ){
		case PREC_SP:
			g_sp_vuni(dp);
			break;
			/*
		case PREC_DP:
			NWARN("Sorry, no curand double precision uniform numbers at this time.");
			break;
			*/
		default:
			sprintf(DEFAULT_ERROR_STRING,
	"g_vuni:  Object %s (%s) should have float precision",
				dp->dt_name,name_for_prec(dp->dt_prec));
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}
#else /* ! HAVE_LIBCURAND */
	NWARN("g_vuni:  Sorry, this program was not configured with libcurand.");
#endif /* ! HAVE_LIBCURAND */
}

#endif /* HAVE_CUDA */
