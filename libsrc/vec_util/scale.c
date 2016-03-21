
#include "quip_config.h"

#include <math.h>
#include "vec_util.h"
#include "veclib_api.h"
#include "debug.h"
#include "quip_prot.h"
#include "getbuf.h"

#ifdef HAVE_IEEEFP_H
#include <ieeefp.h>
#endif

#define BIG_FLOAT	(3.40282e+38)
#define LIL_FLOAT	(-3.40282e+38)

#if defined(SOLARIS)||defined(SGI)

/* clip extrema */

static float exclip(QSP_ARG_DECL  Data_Obj *dp,Data_Obj *val_sp,
	double clipval,double newval,Vec_Func_Code ifunc,Vec_Func_Code vfunc)
{
	float *fltp,extremum,newex;
	Data_Obj *index_p;
	int i;
	Vec_Obj_Args oargs;

	/* first clip vector */
	/* this is done because FVMAXI doesn't work
		with NaN & Inf */

	*((float *)OBJ_DATA_PTR(val_sp)) = clipval;
	setvarg2(&oargs,dp,dp);
	oargs.oa_s1 = val_sp;


	perf_vfunc(QSP_ARG  FVCLIP,&oargs);
	extremum = clipval;

	index_p=mk_scalar("___index",PREC_DI);
	if( index_p == NO_OBJ ) return(-1.0);

	while( extremum == clipval ){
		/* substitute newval for clipval */

		setvarg1(&oargs,dp);
		oargs.oa_s1 = index_p;
		perf_vfunc(QSP_ARG  ifunc,&oargs);

		/* now we have the index ??  */
		i = *((long *)OBJ_DATA_PTR(index_p));
		i--;		/* routine returns fortran index */
		fltp=(float *)OBJ_DATA_PTR(dp);
		newex = fltp[i];
		if( newex != extremum ){
			sprintf(ERROR_STRING,
				"new extremum %g, old extremum %g",
				newex,extremum);
			ADVISE(ERROR_STRING);
			ERROR1("ifunc extremum disagrees with vfunc");
		}

		/* now reset this value */

		fltp[i]=newval;

		/* get the new extremum to see if we're done */
		oargs.oa_s1 = val_sp;
		perf_vfunc(QSP_ARG  vfunc,&oargs);
		extremum = *((float *)OBJ_DATA_PTR(val_sp));
	}
	delvec(index_p);
	return(extremum);
}
#endif /* SOLARIS or SGI */

void scale(QSP_ARG_DECL  Data_Obj *dp,double desmin,double desmax)		/* scale an image (to byte range?) */
{
	double omn,omx,rf,offset;
	Vec_Obj_Args oa1, *oap=&oa1;
	Data_Obj *scratch_scalar_dp;
	Scalar_Value scratch_scalar_val;

	clear_obj_args(oap);
	SET_OA_ARGSTYPE(oap, REAL_ARGS );	/* BUG? should we check type of input? */
	SET_OA_ARGSPREC(oap, ARGSET_PREC(OBJ_PREC(dp)) );
	SET_OA_FUNCTYPE(oap, FUNCTYPE_FOR(OA_ARGSPREC(oap),OA_ARGSTYPE(oap)) );
	SET_OA_SRC_OBJ(oap,0,dp);
	SET_OA_PFDEV(oap, OBJ_PFDEV(dp) );

	scratch_scalar_dp = area_scalar( QSP_ARG  OBJ_AREA(dp) );

	SET_OBJ_PREC_PTR(scratch_scalar_dp,OBJ_PREC_PTR(dp) );
	/* this used to be oa_sdp[0], but now with "projection" the destination
	 * doesn't have to be a scalar.
	 */
	OA_DEST(oap) = scratch_scalar_dp;

	perf_vfunc(QSP_ARG  FVMINV, oap);


#ifndef HAVE_CUDA
	extract_scalar_value(QSP_ARG  &scratch_scalar_val, scratch_scalar_dp);
	// The will fail for a cuda scalar...
#else 	// HAVE_CUDA
	
	if( ! OBJ_IS_RAM(scratch_scalar_dp) ){
		WARN("OOPS - can't extract scalar value from CUDA object!?");
		return;
	} else {
		extract_scalar_value(QSP_ARG  &scratch_scalar_val,
							scratch_scalar_dp);
	}
#endif // HAVE_CUDA
	omn = cast_from_scalar_value(QSP_ARG  &scratch_scalar_val,OBJ_PREC_PTR(dp));

	perf_vfunc(QSP_ARG  FVMAXV, oap);

	extract_scalar_value(QSP_ARG  &scratch_scalar_val, scratch_scalar_dp);
	omx = cast_from_scalar_value(QSP_ARG  &scratch_scalar_val,OBJ_PREC_PTR(dp));

	/*	y = ( x - omn ) * (mx-mn)/(omx-omn) + mn
	 *	  = x * rf + mn - omn*rf
	 */

	if( omx == omn ){
		if( verbose ){
			sprintf(ERROR_STRING,
		"scale:  object %s has constant value %g",OBJ_NAME(dp),omn);
			ADVISE(ERROR_STRING);
		}
		rf = 1;
	} else {
		if( verbose ) {
			sprintf(msg_str,"Range of %s before scaling:  %g - %g",OBJ_NAME(dp),omn,omx);
			prt_msg(msg_str);
		}
		rf = (desmax-desmin)/(omx-omn);
	}
	SET_OA_SVAL(oap,0,&scratch_scalar_val);
	cast_to_scalar_value(QSP_ARG  &scratch_scalar_val,OBJ_PREC_PTR(dp),rf);

	OA_DEST(oap) = dp;
	perf_vfunc(QSP_ARG  FVSMUL, oap);

	offset = desmin - omn*rf;
	if( offset != 0 ){
		cast_to_scalar_value(QSP_ARG  &scratch_scalar_val,OBJ_PREC_PTR(dp),offset);
		perf_vfunc(QSP_ARG  FVSADD, oap);
	}
}

