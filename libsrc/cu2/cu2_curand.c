
/* interface to nVidia curand random number library */

#include "quip_config.h"

#ifdef HAVE_CUDA
#define BUILD_FOR_CUDA

#include <curand.h>

#include "quip_prot.h"
#include "veclib/vecgen.h"
#include "veclib/cu2_veclib_prot.h"
#include "veclib/obj_args.h"
#include "platform.h"


#ifdef HAVE_LIBCURAND

static void init_curand_generator(Platform_Device *pdp)
{
	curandStatus_t s;
	curandGenerator_t f_gen;

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

	SET_PFDEV_CUDA_RNGEN(pdp,f_gen);
}


/* Does the lib support double? */
/* static curandGenerator_t d_gen=NULL; */

void h_cu2_sp_vuni(HOST_CALL_ARG_DECLS)
{
	Data_Obj *dp;
	curandStatus_t s;

//fprintf(stderr,"h_cu2_sp_vuni BEGIN\n");
	dp = OA_DEST(oap);
	// BUG?  validate that it's a cuda object?
	if( PFDEV_CUDA_RNGEN( OBJ_PFDEV(dp) ) == NULL ){
		init_curand_generator( OBJ_PFDEV(dp) );
	}
	if( is_contiguous(dp) ){
		s = curandGenerateUniform(PFDEV_CUDA_RNGEN( OBJ_PFDEV(dp) ),
			(float *)OBJ_DATA_PTR(dp), OBJ_N_TYPE_ELTS(dp) );
		if( s != CURAND_STATUS_SUCCESS ){
			NWARN("curand error generating uniform floats");
			return;
		}
	} else {
		/* Hopefully the rows are contiguous? */
		NWARN("Sorry, need to write code for non-contig g_sp_vuni");
	}
}

void h_cu2_dp_vuni(HOST_CALL_ARG_DECLS)
{
	NWARN("Sorry, no CUDA support for double-precision random numbers");
}

#endif /* HAVE_LIBCURAND */


#ifdef FOOBAR
// This is implemented in entries.c...

// We use the curand implementation of the random number generator...
void h_cu2_vuni( HOST_CALL_ARG_DECLS )
{
#ifdef HAVE_LIBCURAND
	switch( OBJ_MACH_PREC( OA_DEST(oap) ) ){
		case PREC_SP:
			h_cu2_sp_vuni(oap);
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
	"h_cu2_vuni:  Object %s (%s) should have float precision",
				OBJ_NAME(OA_DEST(oap)),
				PREC_NAME(OBJ_PREC_PTR(OA_DEST(oap))));
			NWARN(DEFAULT_ERROR_STRING);
			break;
		case PREC_NONE:
		case PREC_INVALID:
		case N_MACHINE_PRECS:
		default:
			NWARN("CAUTIOUS:  h_cu2_vuni:  unexpected invalid prec code!?");
			break;
	}
#else /* ! HAVE_LIBCURAND */
	NWARN("h_cu2_vuni:  Sorry, this program was not configured with libcurand.");
#endif /* ! HAVE_LIBCURAND */
}
#endif // FOOBAR

#endif /* HAVE_CUDA */
