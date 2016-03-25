
/* interface to nVidia curand random number library */

#include "quip_config.h"

#ifdef HAVE_CUDA
#define BUILD_FOR_CUDA
#include <curand.h>

#include "quip_prot.h"
#include "veclib/vecgen.h"
#include "veclib/obj_args.h"
//#include "my_vector_functions.h"


#ifdef FOOBAR /* find out what we really need */

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include <cutil_inline.h>

#include "my_cuda.h"

#ifdef HAVE_NPP_H
#include <npp.h>
#endif /* HAVE_NPP_H */

#include "quip_prot.h"
#include "cuda_menu.h"
#include "cuda_error.h"
#include "veclib.h"		/* is_ram() */
#include "menu_call_defs.h"

#endif /* FOOBAR */

//Cuda_Device *curr_cdp=NO_CUDA_DEVICE;



#ifdef HAVE_LIBCURAND

#define NO_GENERATOR	((curandGenerator_t)NULL)

static curandGenerator_t f_gen=NO_GENERATOR;

// local prototypes
static void init_curand(void);


/* Does the lib support double? */
/* static curandGenerator_t d_gen=NO_GENERATOR; */

static void g_sp_vuni(Data_Obj *dp)
{
	curandStatus_t s;

	if( f_gen == NO_GENERATOR ){
		init_curand();
	}
	if( is_contiguous(DEFAULT_QSP_ARG  dp) ){
		s = curandGenerateUniform(f_gen,(float *)OBJ_DATA_PTR(dp), OBJ_N_TYPE_ELTS(dp) );
		if( s != CURAND_STATUS_SUCCESS ){
			NWARN("curand error generating uniform floats");
			return;
		}
	} else {
		/* Hopefully the rows are contiguous? */
		NWARN("Sorry, need to write code for non-contig g_sp_vuni");
	}
}

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

#endif /* HAVE_LIBCURAND */


// We use the curand implementation of the random number generator...
void g_vuni( Vec_Obj_Args *oap )
{
#ifdef HAVE_LIBCURAND
	Data_Obj *dp;

	dp = oap->oa_dest;
	switch( OBJ_MACH_PREC(dp) ){
		case PREC_SP:
			g_sp_vuni(dp);
			break;
		case PREC_BY:
		case PREC_IN:
		case PREC_DI:
		case PREC_LI:
		case PREC_UBY:
		case PREC_UIN:
		case PREC_UDI:
		case PREC_ULI:
		case PREC_DP:
			sprintf(DEFAULT_ERROR_STRING,
	"g_vuni:  Object %s (%s) should have float precision",
				OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)));
			NWARN(DEFAULT_ERROR_STRING);
			break;
		case PREC_NONE:
		case PREC_INVALID:
		case N_MACHINE_PRECS:
		default:
			NWARN("CAUTIOUS:  g_vuni:  unexpected invalid prec code!?");
			break;
	}
#else /* ! HAVE_LIBCURAND */
	NWARN("g_vuni:  Sorry, this program was not configured with libcurand.");
#endif /* ! HAVE_LIBCURAND */
}

#endif /* HAVE_CUDA */
