
#include "quip_config.h"

char VersionId_newvec_vec_args[] = QUIP_VERSION_STRING;

#include <stdio.h>

#include "nvf.h"
#include "new_chains.h"
#include "debug.h"

/* local prototypes */

static Data_Obj *getascal(QSP_ARG_DECL const char *pmpt,prec_t prec);
static int getscal(QSP_ARG_DECL Vec_Obj_Args *oap,Vec_Func *vfp);
static int get_bitmap(QSP_ARG_DECL Vec_Obj_Args *oap);
static int get_args(QSP_ARG_DECL Vec_Obj_Args *oap, Vec_Func *vfp);


/* globals */
int insist_real=0, insist_cpx=0, insist_quat=0;

#define SCALAR_IMMEDIATE	1
#define SCALAR_INDIRECT		2

static int scalar_mode=SCALAR_IMMEDIATE;

#define SCAL1_NAME	"scal1_op"
#define SCAL2_NAME	"scal2_op"


int get_dst(QSP_ARG_DECL Vec_Obj_Args *oap)
{
	oap->oa_dest=PICK_OBJ( "destination vector" );
	if( oap->oa_dest==NO_OBJ )
		return(-1);
	return(0);
}

int get_src1(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	oap->oa_dp[0]=PICK_OBJ( "first source vector" );
	if( oap->oa_dp[0]==NO_OBJ )
		return(-1);
	return(0);
}

int get_src2(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	oap->oa_dp[1]=PICK_OBJ( "second source vector" );
	if( oap->oa_dp[1]==NO_OBJ )
		return(-1);
	return(0);
}

int get_src3(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	oap->oa_dp[2]=PICK_OBJ( "third source vector" );
	if( oap->oa_dp[2]==NO_OBJ )
		return(-1);
	return(0);
}


int get_src4(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	oap->oa_dp[3]=PICK_OBJ( "fourth source vector" );
	if( oap->oa_dp[3]==NO_OBJ )
		return(-1);
	return(0);
}

void show_data_vector(Data_Vector *dvp)
{
	sprintf(msg_str,"data vec 0x%lx   inc %d   count %d   prec %s (%d)",
		(int_for_addr)dvp->dv_vec,dvp->dv_inc,dvp->dv_count,
		name_for_prec(dvp->dv_prec),
		dvp->dv_prec);
	prt_msg(msg_str);
}

/* Convert from data objects to warlib-style addresses and increments.
 * For non evenly-spaced objects, we just do the largest vectorizable
 * chunk, as previously determined and stored in max_vectorizable.
 * Note that this may have been constrained by some other object than the
 * one we are working on now.
 *
 *
 * Each data object has an array of dimensions and increments
 * For an evenly spaced object, the increment at a given level is
 * equal to the dimension increment product from the next lowest level.
 *
 * Now, we are operating on objects whose dimensions are assumed to match.
 * We would like to vectorize over as many dimensions as possible;
 * We use a variable max_vectorizable to record the max number of vectorizable
 * dimensions.
 *
 *
 * extract_vec_params() is used to set the warlib-style args from data object args.
 * This really shouldn't be necessary, it would be more efficient to simply have
 * a flag that says to take the data ptr and increment directly from the data object.
 * This might cause some problems w/ complex types, but in general it ought to work
 * fine.
 *
 * When we do need to have our own vector args is when we are not operating
 * on the entire object...  In this case, it is probably more efficient to
 * compute the offsets ourselves than to use one of the object indexing functions
 *
 * This routine does not assume that the whole object
 * is vectorizable, so it uses the global max_vectorizable to determine the length
 * (and increment) of the largest vectorizable chunk.
 */

void get_bitmap_inc(Data_Obj *dp)
{
}



void extract_vec_params(Data_Vector *dvp, Data_Obj *dp)
{
	int i;
	incr_t inc,n;
	int need_inc;
	int start_dim;

	if( dp == NO_OBJ ){
		dvp->dv_vec = NULL;
		dvp->dv_inc = 0;
		dvp->dv_count = 0;
		return;
	}

	/* find the increment and run length for the vectorizable chunks */
	n=1;
	need_inc=1;

	if( IS_COMPLEX(dp) || IS_QUAT(dp) )
		start_dim=1;
	else
		start_dim=0;

	inc=(-4);	// pointless initialization to quiet compiler

	for(i=start_dim;i<=max_vectorizable;i++){
		n *= dp->dt_type_dim[i];
		if( need_inc && (dp->dt_type_dim[i] > 1) ){
			inc = dp->dt_type_inc[i];
			need_inc=0;
		}
	}
	if( need_inc ) {
		inc=1;
		/* We used to think that if we got here, the object
		 * must be a scalar; this is not true however:  one example
		 * is a column vector taken from an image, or any object
		 * with a single column combined with an object which constrains
		 * max_vectorizable to 1 (such as a bitmap).
		 */
	}
	if( IS_COMPLEX(dp) )
		inc/=2;
	else if( IS_QUAT(dp) )
		inc/=4;

	dvp->dv_vec	= dp->dt_data;
	dvp->dv_inc	= inc;
	dvp->dv_count	= n;
	dvp->dv_prec	= dp->dt_prec;	/* BUG? should be MACHINE_PREC??? */
	dvp->dv_bit0	= dp->dt_bit0;
	if( IS_BITMAP(dp) )
		dvp->dv_flags |= DV_BITMAP;

#ifdef DEBUG
if( debug & veclib_debug ){
/* LONGLIST(dp); */
sprintf(DEFAULT_ERROR_STRING,"extract_vec_params:  obj %s, prec %s, max_vect = %d,  n = %d, inc = %d",
dp->dt_name,name_for_prec(dp->dt_prec),max_vectorizable,dvp->dv_count,dvp->dv_inc);
NADVISE(DEFAULT_ERROR_STRING);
show_data_vector(dvp);
}
#endif /* DEBUG */
}

void show_vf(Vec_Func *vfp)
{
	sprintf(DEFAULT_ERROR_STRING,"function %s, flags = 0x%x",vfp->vf_name,vfp->vf_flags);
	NADVISE(DEFAULT_ERROR_STRING);
	/*
	sprintf(DEFAULT_ERROR_STRING,"V_INPLACE = 0x%x",V_INPLACE);
	NADVISE(DEFAULT_ERROR_STRING);
	*/
}

static int get_args(QSP_ARG_DECL  Vec_Obj_Args *oap,Vec_Func *vfp)
{
	int oops=0;

	clear_obj_args(oap);

	if( vfp->vf_flags & BITMAP_SRC )
		oops|=get_bitmap(QSP_ARG  oap);

	if( vfp->vf_flags & BITMAP_DST )
		oops|=get_bitmap(QSP_ARG  oap);

	if( vfp->vf_flags & DST_VEC )
		oops|=get_dst(QSP_ARG  oap);

	if( vfp->vf_flags & SRC1_VEC )
		oops|=get_src1(QSP_ARG  oap);
	if( vfp->vf_flags & SRC2_VEC )
		oops|=get_src2(QSP_ARG  oap);
	if( vfp->vf_flags & SRC3_VEC )
		oops|=get_src3(QSP_ARG  oap);
	if( vfp->vf_flags & SRC4_VEC )
		oops|=get_src4(QSP_ARG  oap);

	/* BUG?  We should sort out passing a value vs. receiving a value in a scalar object... */
	if( vfp->vf_flags & (SRC_SCALAR1|SRC_SCALAR2|TWO_SCALAR_RESULTS) ){
		if( getscal(QSP_ARG  oap,vfp) == (-1) ) oops|=1;
#ifdef PROBABLY_NOT_NEEDED
		/* Here we point the scalar value to the object data - why? */
		for(i=0;i<MAX_RETSCAL_ARGS;i++){
			if( oap->oa_sdp[i] != NO_OBJ )
				oap->oa_svp[i] = (Scalar_Value *)oap->oa_sdp[i]->dt_data;
		}
#endif /* PROBABLY_NOT_NEEDED */
	}

	if( oops) return(-1);		/* if a requested object doesn't exist... */

	/* getscal() gets (or makes) objects? */

	return(0);
} /* end get_args */

void scalar_immediate()
{
	scalar_mode = SCALAR_IMMEDIATE;
}

void scalar_indirect()
{
	scalar_mode = SCALAR_INDIRECT;
}

int prompt_scalar_value(QSP_ARG_DECL  Data_Obj *dp, const char *pmpt, prec_t prec)
{
	if( prec==PREC_DI ){
		int32_t lvalue;
		lvalue=(int32_t)HOW_MANY(pmpt);
		*((int32_t *)dp->dt_data) = lvalue;
	} else if( prec==PREC_UDI ){
		uint32_t lvalue;
		lvalue=(uint32_t) HOW_MANY(pmpt);
		*((uint32_t *)dp->dt_data) = lvalue;
	} else if( prec==PREC_LI ){
		int64_t lvalue;
		lvalue=(int64_t)HOW_MANY(pmpt);
		*((int64_t *)dp->dt_data)=lvalue;
	} else if( prec==PREC_ULI ){
		uint64_t lvalue;
		lvalue=(uint64_t)HOW_MANY(pmpt);
		*((uint64_t *)dp->dt_data)=lvalue;
	} else if( prec==PREC_BY ){
		char cvalue;
		cvalue=(char)HOW_MANY(pmpt);
		*((char *)dp->dt_data)=cvalue;
	} else if( prec==PREC_UBY ){
		u_char cvalue;
		cvalue=(u_char)HOW_MANY(pmpt);
		*((u_char *)dp->dt_data)=cvalue;
	} else if( prec==PREC_DP ){
		double value;
		value=HOW_MUCH(pmpt);
		*((double *)dp->dt_data)=value;
	} else if( prec==PREC_SP ){
		float value;
		value=(float)HOW_MUCH(pmpt);
		*((float *)dp->dt_data)=value;
	} else if( prec==PREC_IN ){
		short svalue;
		svalue=(short)HOW_MANY(pmpt);
		*((short *)dp->dt_data) = svalue;
	} else if( prec==PREC_UIN ){
		u_short svalue;
		svalue=(u_short)HOW_MANY(pmpt);
		*((u_short *)dp->dt_data) = svalue;
	}

#ifdef CAUTIOUS
	else {
		sprintf(ERROR_STRING,
	"CAUTIOUS:  prompt_scalar_value:  unsupported precision \"%s\" (0x%x)",
			name_for_prec(prec),prec);
		WARN(ERROR_STRING);
		return(-1);
	}
#endif /* CAUTIOUS */
	return(0);
}

static Scalar_Value *get_sv(prec_t prec)
{
	Scalar_Value *svp;

	/* We pass the precision so that we don't have to allocate the huge Scalar_Value union for just one float... */
	/* But we're not taking advantage of that here... */

	svp = (Scalar_Value *)getbuf( sizeof(Scalar_Value) );
	/* BUG?  should we initialize value? */
	return(svp);			/* BUG possible memory leak? */
}

#ifdef HAVE_CUDA

/* Set the current data area to match whatever we are using,
 * so that automatically created objects (e.g. scalars) are in the
 * correct space.  (What about setting the value of scalars on the gpu?)
 */

Data_Area * set_arg_data_area(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	Data_Area *ap;

	if( oap->oa_dest != NO_OBJ ){
		push_data_area(ap=oap->oa_dest->dt_ap);
	} else {
		int i;
		ap=NO_AREA;	// quiet compiler
		for(i=0;i<MAX_N_ARGS;i++){
			if( oap->oa_dp[i] != NO_OBJ ){
				push_data_area(ap=oap->oa_dp[i]->dt_ap);
				i=MAX_N_ARGS+5;	// force exit & flag found
			}
		}
		if( i == MAX_N_ARGS ){
			/* This was originally written as a CAUTIOUS
			 * check, but we land here for example
			 * if we call ramp2d and specify a non-existant
			 * destination image.  The oops flag is set,
			 * but we still want to eat up args to avoid
			 * a parsing error.
			 */
			ap = default_data_area(SINGLE_QSP_ARG);
			push_data_area(ap);
		}
	}
	return(ap);
}

#endif /* HAVE_CUDA */

#define SET_PREC							\
									\
	if( oap->oa_dest!=NO_OBJ ) prec=oap->oa_dest->dt_prec;		\
	else prec=PREC_SP;

/* BUG - getscal gets a scalar object, but for an arg we don't need to create
 * the object.  We only need to do that to return a scalar value.
 *
 * Now that we are supporting CUDA (multiple data areas), we have to
 * insure that the newly created scalar is in the correct data area.
 * We assume that oa_dest has already been set...
 *
 * It looks like now we use scalar values for passing to, and only
 * use scalar objects for return values.  Because of the successful
 * implementation of projection for vmaxv etc, the only ops that now
 * return any scalars are vmaxg etc which return the extreme value
 * and the number of occurrences.
 *
 * BUG this should be broken into two functions?
 */

static int getscal(QSP_ARG_DECL Vec_Obj_Args *oap, Vec_Func *vfp)
{
	int ir, ic, iq;
	prec_t prec;
	int retval=0;
#ifdef HAVE_CUDA
	Data_Area *ap;
#endif /* HAVE_CUDA */

	/* We use the insist_xxx flags to force a mixed operation,
	 * e.g. real x complex.  If the scalar and source vector
	 * precisions differ, we need to make sure that the destination
	 * vector has the precision with higher type dimension.
	 */

	ir=insist_real;
	ic=insist_cpx;
	iq=insist_quat;

	insist_real=0;
	insist_cpx=0;
	insist_quat=0;

#ifdef HAVE_CUDA
	ap=set_arg_data_area(QSP_ARG  oap);
#endif /* HAVE_CUDA */

	if( vfp->vf_flags & SRC_SCALAR3 /* HAS_3_SCALARS */ ){
		const char *p1,*p2,*p3;

		SET_PREC

		if( vfp->vf_code == FVRAMP2D ){
			p1="starting value";
			p2="horizontal increment";
			p3="vertical increment";
		} else if( vfp->vf_code >= FSS_VS_LT &&
				vfp->vf_code <= FSS_VS_NE ){
			p1="result value if condition satisfied";
			p2="result value if condition not satisfied";
			p3="scalar value for comparison";
		}
#ifdef CAUTIOUS
		  else {
			// quiet compiler
			p1=p2=p3="";
			ERROR1("CAUTIOUS:  unexpected 3 scalar function!?");
		}
#endif /* CAUTIOUS */
		oap->oa_svp[0] = get_sv(prec);
		oap->oa_svp[1] = get_sv(prec);
		oap->oa_svp[2] = get_sv(prec);
		cast_to_scalar_value(QSP_ARG  oap->oa_svp[0], prec, HOW_MUCH(p1) );
		cast_to_scalar_value(QSP_ARG  oap->oa_svp[1], prec, HOW_MUCH(p2) );
		cast_to_scalar_value(QSP_ARG  oap->oa_svp[2], prec, HOW_MUCH(p3) );
	} else if( vfp->vf_flags & SRC_SCALAR2 /* HAS_2_SCALARS */ ){
		const char *p1,*p2;

		SET_PREC


		if( COMPLEX_PRECISION(prec) ){
			/* this should not happen!? */
			/* Does the function permit complex? */
			if( (vfp->vf_typemask & (C|M)) == 0 ){
				sprintf(ERROR_STRING,
	"getscal:  function %s does not permit operations with complex targets (%s)",
					vfp->vf_name,oap->oa_dest->dt_name);
				WARN(ERROR_STRING);
				retval=(-1);
			}
		}

		if( QUAT_PRECISION(prec) ){
			/* this should not happen!? */
			/* Does the function permit quaternions? */
			if( (vfp->vf_typemask & (Q)) == 0 ){
				sprintf(ERROR_STRING,
	"getscal:  function %s does not permit operations with quaternion targets (%s)",
					vfp->vf_name,oap->oa_dest->dt_name);
				WARN(ERROR_STRING);
				retval=(-1);
			}
		}

		if( vfp->vf_code == FVRAMP1D ){
			p1="starting value";
			p2="ramp increment";
		} else if( vfp->vf_code == FVSSSLCT ){
			p1="scalar1";
			p2="scalar2";
		} else if( vfp->vf_code >= FVS_VS_LT &&
				vfp->vf_code <= FVS_VS_NE ){
			p1="result value for test false";
			p2="scalar value for comparison";
		} else if( vfp->vf_code >= FSS_VV_LT &&
				vfp->vf_code <= FSS_VV_NE ){
			p1="result value for test true";
			p2="result value for test false";
		}
#ifdef CAUTIOUS
		else {
			WARN("CAUTIOUS:  unhandled case in getscal");
			p1=p2="dummy value";
			retval=(-1);
		}
#endif /* CAUTIOUS */
		oap->oa_svp[0] = get_sv(prec);
		oap->oa_svp[1] = get_sv(prec);
		cast_to_scalar_value(QSP_ARG  oap->oa_svp[0], prec, HOW_MUCH(p1) );
		cast_to_scalar_value(QSP_ARG  oap->oa_svp[1], prec, HOW_MUCH(p2) );
	}

	else if( vfp->vf_flags & SRC_SCALAR1 ){
		if( vfp->vf_flags == VS_TEST ){	/* vsm_lt etc. */
			if( oap->oa_1==NO_OBJ ){
				goto get_dummy;
			}
			oap->oa_svp[0] = get_sv(oap->oa_1->dt_prec);
			cast_to_scalar_value(QSP_ARG  oap->oa_svp[0], oap->oa_1->dt_prec,
				HOW_MUCH("source scalar value") );
		} else if( oap->oa_dest == NO_OBJ ){	/* error condition */
			double d;
			d=HOW_MUCH("dummy value");
			retval=(-1);
		} else if( IS_REAL(oap->oa_dest) || ir ){
			if( ic ) WARN("Multiplication by a complex scalar with a real target");
			if( iq ) WARN("Multiplication by a quaternion scalar with a real target");
			/* BUG we can't use destination for precision
			 * in the mixed precision ops...
			 */
			oap->oa_svp[0]=get_sv(MACHINE_PREC(oap->oa_dest));
			cast_to_scalar_value(QSP_ARG  oap->oa_svp[0], MACHINE_PREC(oap->oa_dest),
				HOW_MUCH("source real scalar value") );
		} else if( (IS_COMPLEX(oap->oa_dest) && !ir) || ic ) {
			oap->oa_svp[0] = get_sv(MACHINE_PREC(oap->oa_dest));
			cast_to_cpx_scalar(QSP_ARG  0,oap->oa_svp[0], MACHINE_PREC(oap->oa_dest),
				HOW_MUCH("source scalar value real part") );
			cast_to_cpx_scalar(QSP_ARG  1,oap->oa_svp[0], MACHINE_PREC(oap->oa_dest),
				HOW_MUCH("source scalar value imaginary part") );
		} else if( (IS_QUAT(oap->oa_dest) && !ir) || iq ) {
			oap->oa_svp[0] = get_sv(MACHINE_PREC(oap->oa_dest));
			cast_to_quat_scalar(QSP_ARG  0,oap->oa_svp[0], MACHINE_PREC(oap->oa_dest),
				HOW_MUCH("source scalar value real part") );
			cast_to_quat_scalar(QSP_ARG  1,oap->oa_svp[0], MACHINE_PREC(oap->oa_dest),
				HOW_MUCH("source scalar value i part") );
			cast_to_quat_scalar(QSP_ARG  2,oap->oa_svp[0], MACHINE_PREC(oap->oa_dest),
				HOW_MUCH("source scalar value j part") );
			cast_to_quat_scalar(QSP_ARG  3,oap->oa_svp[0], MACHINE_PREC(oap->oa_dest),
				HOW_MUCH("source scalar value k part") );
		} else {
			/* use a single scalar for all components */
			oap->oa_svp[0] = get_sv(MACHINE_PREC(oap->oa_dest));
			cast_to_scalar_value(QSP_ARG  oap->oa_svp[0], MACHINE_PREC(oap->oa_dest),
				HOW_MUCH("source scalar value") );
		}
		if( oap->oa_svp[0]==NO_SCALAR_VALUE ){
			retval=(-1);
		}
	}

	if( vfp->vf_flags & TWO_SCALAR_RESULTS ){
		if( oap->oa_1 == NO_OBJ ){
			sprintf(ERROR_STRING,
	"getscal (%s):  no argument to use for precision prototype!?",
				vfp->vf_name);
			WARN(ERROR_STRING);
			retval=(-1);
		}
		oap->oa_s1=getascal(QSP_ARG "name of scalar for extreme value",
			oap->oa_1->dt_prec);
		oap->oa_s2=getascal(QSP_ARG "name of scalar for # of occurrences",PREC_DI);
	}

#ifdef HAVE_CUDA
	pop_data_area();
#endif /* HAVE_CUDA */

	return(retval);

get_dummy:
	{
		/* avoid a parsing error */
		float dummy;

		dummy = HOW_MUCH("dummy scalar value");
		return(-1);
	}
}

/* 
 * Get a scalar object that the user specifies.
 */

static Data_Obj * getascal(QSP_ARG_DECL const char *pmpt,prec_t prec)
{
	Data_Obj *dp;

	/* which data area does PICK_OBJ use??? */
	dp=PICK_OBJ( pmpt );
	if( dp==NO_OBJ ) return(NO_OBJ);
	if( !IS_SCALAR(dp) ){
		sprintf(ERROR_STRING,
			"getascal:  %s is not a scalar",dp->dt_name);
		WARN(ERROR_STRING);
		return(NO_OBJ);
	}
	if( dp->dt_prec != prec ){
		sprintf(ERROR_STRING,
			"getascal:  %s scalar %s should have precision %s",
			prec_name[dp->dt_prec],dp->dt_name,prec_name[prec]);
		WARN(ERROR_STRING);
		return(NO_OBJ);
	}
#ifdef DEBUG
if( debug & veclib_debug ){
sprintf(ERROR_STRING,"getascal:  returning %s scalar %s",prec_name[MACHINE_PREC(dp)],dp->dt_name);
NADVISE(ERROR_STRING);
}
#endif /* DEBUG */
	return(dp);
}

static int get_bitmap(QSP_ARG_DECL Vec_Obj_Args *oap)
{
	oap->oa_bmap=PICK_OBJ( "bitmap data object" );
	if( oap->oa_bmap==NO_OBJ ) return(-1);
	if( oap->oa_bmap->dt_prec != PREC_BIT ){
		sprintf(ERROR_STRING,
			"bitmap \"%s\" (%s,0x%x) must have bit precision (0x%x)",
			oap->oa_bmap->dt_name,
			name_for_prec(oap->oa_bmap->dt_prec), oap->oa_bmap->dt_prec,PREC_BIT);
		WARN(ERROR_STRING);
		return(-1);
	}
	return(0);
}

/* We have a problem introduced by trying to use the Data_Obj framework for images in nVidia CUDA:
 * We can't mix-and-match, one solution might be to have separate name spaces for gpu and ram objects.
 * For now, we just check that all objects are from the ram data area.
 */

#ifdef HAVE_CUDA

static int check_one_obj_loc( Data_Obj *dp )
{
	if( dp == NO_OBJ ) return(0);
	if( IS_RAM(dp ) ) return(OARGS_RAM);
	return(OARGS_GPU);
}

#define CHECK_LOC( dp )							\
									\
	s = check_one_obj_loc( dp );					\
	if( s == OARGS_RAM ) all_gpu=0;					\
	if( s == OARGS_GPU ) all_ram=0;

static void check_obj_locations( Vec_Obj_Args *oap )
{
	int s, all_ram=1, all_gpu=1;
	int i;

	if( HAS_CHECKED_ARGS(oap) ){
		return;
	}

	for(i=0;i<MAX_N_ARGS;i++){
		CHECK_LOC( oap->oa_dp[i] )
	}
	CHECK_LOC( oap->oa_dest )
	CHECK_LOC( oap->oa_sdp[0] )
	CHECK_LOC( oap->oa_sdp[1] )

	oap->oa_flags = OARGS_CHECKED;
	if( all_ram ) oap->oa_flags |= OARGS_RAM;
	if( all_gpu ) oap->oa_flags |= OARGS_GPU;
}

int are_gpu_args( Vec_Obj_Args *oap )
{
	check_obj_locations(oap);
	return( HAS_GPU_ARGS(oap) );
}

int are_ram_args( Vec_Obj_Args *oap )
{
	check_obj_locations(oap);
	return( HAS_RAM_ARGS(oap) );
}

#define REPORT_LOC( str, dp )						\
									\
	if( dp != NO_OBJ ){						\
		if( IS_RAM(dp) ){					\
			sprintf(ERROR_STRING,"\t%s\tRAM:\t%s",str,dp->dt_name);	\
			NADVISE(ERROR_STRING);				\
		} else {						\
			sprintf(ERROR_STRING,"\t%s\tGPU:\t%s",str,dp->dt_name);	\
			NADVISE(ERROR_STRING);				\
		}							\
	}

void mixed_location_error(QSP_ARG_DECL  Vec_Func *vfp, Vec_Obj_Args *oap)
{
	int i;

	sprintf(ERROR_STRING,"%s:  arguments must either be all in RAM or all on a single GPU.",vfp->vf_name);
	WARN(ERROR_STRING);
	for(i=0;i<MAX_N_ARGS;i++){
		sprintf(msg_str,"oa_dp[%d]:",i);
		REPORT_LOC( msg_str, oap->oa_dp[i] )
	}
	REPORT_LOC( "oa_dest:",oap->oa_dest )
	REPORT_LOC( "oa_sdp[0]:",oap->oa_sdp[0] )
	REPORT_LOC( "oa_sdp[1]:",oap->oa_sdp[1] )
}

#endif /* HAVE_CUDA */

#define CLEAR_OARGS(oap)					\
								\
	(oap)->oa_bmap = NO_OBJ;					\
	for(i=0;i<MAX_SRCSCAL_ARGS;i++)				\
		(oap)->oa_svp[i]=NULL;				\
	for(i=0;i<MAX_RETSCAL_ARGS;i++)				\
		(oap)->oa_sdp[i] = NO_OBJ;			\
	(oap)->oa_flags = 0;


void do_vfunc( QSP_ARG_DECL  Vec_Func *vfp )
{
	Vec_Obj_Args oargs;
	int i;

	CLEAR_OARGS(&oargs)

	if( get_args(QSP_ARG  &oargs,vfp) < 0 ){
		sprintf(ERROR_STRING,"Error getting arguments for function %s",
			vfp->vf_name);
		WARN(ERROR_STRING);
		return;
	}

if( oargs.oa_dest == NO_OBJ ){
/* BUG?  is this really an error?  The bitmap destination might be here... */
/*
sprintf(ERROR_STRING,"%s:  Null destination!?!?", vfp->vf_name);
NWARN(ERROR_STRING);
*/
}
	call_vfunc(QSP_ARG  vfp,&oargs);

	/* Now free the scalars (if any) */

	/* Perhaps the code would be more efficient if
	 * oargs contained the scalar value struct
	 * itself instead of a pointer to dynamically
	 * allocated memory...  (is getbuf/malloc thread-safe?)
	 */
	for(i=0;i<MAX_SRCSCAL_ARGS;i++){
		if( oargs.oa_svp[i] != NO_SCALAR_VALUE )
			givbuf( oargs.oa_svp[i] );
	}
}

void setvarg1(Vec_Obj_Args *oap, Data_Obj *dp)
{
	int i;

	oap->oa_1=oap->oa_2=oap->oa_3=oap->oa_4=NO_OBJ;
	CLEAR_OARGS(oap)
	oap->oa_dest = dp;
	set_obj_arg_flags(oap);
}

void setvarg2(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *srcv)
{
	int i;

	oap->oa_2=oap->oa_3=oap->oa_4=NO_OBJ;
	CLEAR_OARGS(oap)
	oap->oa_dest /* =oap->oa_2 */ =dstv;
	oap->oa_1=srcv;
	set_obj_arg_flags(oap);
}

void setvarg3(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *src1,Data_Obj *src2)
{
	int i;

	CLEAR_OARGS(oap)
	oap->oa_dest=dstv;
	oap->oa_2=src2;
	oap->oa_1=src1;
	oap->oa_3=oap->oa_4=NO_OBJ;
	set_obj_arg_flags(oap);
}

void setvarg4(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *src1,Data_Obj *src2,Data_Obj *src3)
{
	int i;

	CLEAR_OARGS(oap)
	oap->oa_dest=dstv;
	oap->oa_1=src1;
	oap->oa_2=src2;
	oap->oa_3=src3;
	oap->oa_4=NO_OBJ;
	set_obj_arg_flags(oap);
}

void setvarg5(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *src1,Data_Obj *src2,Data_Obj *src3, Data_Obj *src4)
{
	int i;

	CLEAR_OARGS(oap)
	oap->oa_dest=dstv;
	oap->oa_1=src1;
	oap->oa_2=src2;
	oap->oa_3=src3;
	oap->oa_4=src4;
	set_obj_arg_flags(oap);
}


