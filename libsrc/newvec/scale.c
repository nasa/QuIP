

#include "quip_config.h"

char VersionId_newvec_scale[] = QUIP_VERSION_STRING;

#include <math.h>
#include "nvf.h"
#include "debug.h"

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

	*((float *)val_sp->dt_data) = clipval;
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
		i = *((long *)index_p->dt_data);
		i--;		/* routine returns fortran index */
		fltp=(float *)dp->dt_data;
		newex = fltp[i];
		if( newex != extremum ){
			sprintf(error_string,
				"new extremum %g, old extremum %g",
				newex,extremum);
			ADVISE(error_string);
			ERROR1("ifunc extremum disagrees with vfunc");
		}

		/* now reset this value */

		fltp[i]=newval;

		/* get the new extremum to see if we're done */
		oargs.oa_s1 = val_sp;
		perf_vfunc(QSP_ARG  vfunc,&oargs);
		extremum = *((float *)val_sp->dt_data);
	}
	delvec(index_p);
	return(extremum);
}
#endif /* SOLARIS or SGI */


static Data_Obj _scratch_scalar_obj;
static Scalar_Value _scratch_scalar_val;
static int scratch_scalar_inited=0;

/* init_scratch_scalar - make up a scalar object to get the return value of vmaxv & vminv */

static void init_scratch_scalar(void)
{
	int i;

#ifdef CAUTIOUS
	if( scratch_scalar_inited ){
		NWARN("CAUTIOUS:  init_scratch_scalar:  already initialized");
		return;
	}
#endif /* CAUTIOUS */

	_scratch_scalar_obj.dt_name = savestr("_unhashed_scratch_scalar_obj");	/* not hashed */
	/* can't set all the flags, because the type can vary... */


	_scratch_scalar_obj.dt_shape.si_prec = (-1);	/* an invalid value */
	for(i=0;i<N_DIMENSIONS;i++){
		_scratch_scalar_obj.dt_shape.si_mach_dimset.ds_dimension[i] = 1;
		_scratch_scalar_obj.dt_shape.si_type_dimset.ds_dimension[i] = 1;
	}
	_scratch_scalar_obj.dt_shape.si_n_mach_elts = 1;
	_scratch_scalar_obj.dt_shape.si_n_type_elts = 1;
	_scratch_scalar_obj.dt_shape.si_maxdim = 0;
	_scratch_scalar_obj.dt_shape.si_mindim = 0;
	_scratch_scalar_obj.dt_shape.si_flags = DT_SCALAR;
	_scratch_scalar_obj.dt_shape.si_last_subi = -1;

	_scratch_scalar_obj.dt_extra = NULL;
	_scratch_scalar_obj.dt_data = &_scratch_scalar_val;	/* big enough for any type */
	_scratch_scalar_obj.dt_unaligned_data = NULL;	/* never to be freed */

	_scratch_scalar_obj.dt_ap = NULL;	/* don't need? */
	_scratch_scalar_obj.dt_parent = NO_OBJ;	/* don't need? */
	_scratch_scalar_obj.dt_children = NO_LIST;	/* don't need? */

	for(i=0;i<N_DIMENSIONS;i++){
		_scratch_scalar_obj.dt_mach_inc[i] = 0;	/* don't need? */
		_scratch_scalar_obj.dt_type_inc[i] = 0;	/* don't need? */
	}
	_scratch_scalar_obj.dt_offset = 0;	/* don't need? */
	_scratch_scalar_obj.dt_refcount = 0;	/* don't need? */
	_scratch_scalar_obj.dt_declfile = NULL;	/* don't need? */

	scratch_scalar_inited=1;
}

void scale(QSP_ARG_DECL  Data_Obj *dp,double desmin,double desmax)		/* scale an image (to byte range?) */
{
	double omn,omx,rf,offset;
	Vec_Obj_Args oa;

	clear_obj_args(&oa);
	oa.oa_argstype = REAL_ARGS;	/* BUG? should we check type of input? */
	oa.oa_argsprec = ARGSET_PREC(dp->dt_prec);
	oa.oa_functype = FUNCTYPE_FOR(oa.oa_argsprec,oa.oa_argstype);
	oa.oa_dp[0] = dp;

	if( ! scratch_scalar_inited ) init_scratch_scalar();

	_scratch_scalar_obj.dt_prec = dp->dt_prec;
	/* this used to be oa_sdp[0], but now with "projection" the destination
	 * doesn't have to be a scalar.
	 */
	oa.oa_dest = &_scratch_scalar_obj;

	vminv(&oa);

	omn = cast_from_scalar_value(QSP_ARG  &_scratch_scalar_val,dp->dt_prec);

	vmaxv(&oa);
	omx = cast_from_scalar_value(QSP_ARG  &_scratch_scalar_val,dp->dt_prec);

	/*	y = ( x - omn ) * (mx-mn)/(omx-omn) + mn
	 *	  = x * rf + mn - omn*rf
	 */

	if( omx == omn ){
		if( verbose ){
			sprintf(error_string,
		"scale:  object %s has constant value %g",dp->dt_name,omn);
			ADVISE(error_string);
		}
		rf = 1;
	} else {
		if( verbose ) {
			sprintf(msg_str,"Range of %s before scaling:  %g - %g",dp->dt_name,omn,omx);
			prt_msg(msg_str);
		}
		rf = (desmax-desmin)/(omx-omn);
	}
	oa.oa_svp[0] = &_scratch_scalar_val;
	cast_to_scalar_value(QSP_ARG  &_scratch_scalar_val,dp->dt_prec,rf);

	oa.oa_dest = dp;
	vsmul(&oa);

	offset = desmin - omn*rf;
	if( offset != 0 ){
		cast_to_scalar_value(QSP_ARG  &_scratch_scalar_val,dp->dt_prec,offset);
		vsadd(&oa);
	}
}

