
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

#define init_curand_generator(pdp) _init_curand_generator(QSP_ARG  pdp)

static void _init_curand_generator(QSP_ARG_DECL  Platform_Device *pdp)
{
	curandStatus_t s;
	curandGenerator_t f_gen;

	s = curandCreateGenerator(&f_gen,CURAND_RNG_PSEUDO_DEFAULT);
	if( s != CURAND_STATUS_SUCCESS ){
		warn("curand error creating generator");
		return;
	}
	// BUG should allow user to seed generator
	s=curandSetPseudoRandomGeneratorSeed(f_gen,1234ULL);
	if( s != CURAND_STATUS_SUCCESS ){
		warn("curand error seeding generator");
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
			warn("curand error generating uniform floats");
			return;
		}
	} else {
		/* Hopefully the rows are contiguous? */
		warn("Sorry, need to write code for non-contig g_sp_vuni");
	}
}

void h_cu2_dp_vuni(HOST_CALL_ARG_DECLS)
{
	warn("Sorry, no CUDA support for double-precision random numbers");
}

#endif /* HAVE_LIBCURAND */

#endif /* HAVE_CUDA */
