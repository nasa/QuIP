
#include "quip_config.h"
#include "quip_prot.h"
#include "veclib_prot.h"

#include <stdio.h>

#include "nvf.h"
//#include "new_chains.h"
#include "debug.h"
//#include "warn.h"
//#include "getbuf.h"

/* globals */
int insist_real=0, insist_cpx=0, insist_quat=0;

#define SCALAR_IMMEDIATE	1
#define SCALAR_INDIRECT		2

//static int scalar_mode=SCALAR_IMMEDIATE;

#define SCAL1_NAME	"scal1_op"
#define SCAL2_NAME	"scal2_op"


static int get_dst(QSP_ARG_DECL Vec_Obj_Args *oap)
{
	SET_OA_DEST(oap, PICK_OBJ( "destination vector" ) );
	if( OA_DEST(oap) ==NO_OBJ )
		return(-1);
	return(0);
}

static int get_src1(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	SET_OA_SRC1(oap,PICK_OBJ( "first source vector" ) );
	if( OA_SRC1(oap)==NO_OBJ )
		return(-1);
	return(0);
}

static int get_src2(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	SET_OA_SRC2(oap,PICK_OBJ( "second source vector" ));
	if( OA_SRC2(oap)==NO_OBJ )
		return(-1);
	return(0);
}

static int get_src3(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	SET_OA_SRC3(oap,PICK_OBJ( "third source vector" ));
	if( OA_SRC3(oap)==NO_OBJ )
		return(-1);
	return(0);
}


static int get_src4(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	SET_OA_SRC4(oap,PICK_OBJ( "fourth source vector" ));
	if( OA_SRC4(oap)==NO_OBJ )
		return(-1);
	return(0);
}

#ifdef NOT_USED
static void show_data_vector(QSP_ARG_DECL  Data_Vector *dvp)
{
	sprintf(msg_str,"data vec 0x%lx   inc %d   count %d   prec %s (%d)",
		(int_for_addr)DV_VEC(dvp),DV_INC(dvp),DV_COUNT(dvp),
		NAME_FOR_PREC_CODE(DV_PREC(dvp) ),
		DV_PREC(dvp) );
	prt_msg(msg_str);
}
#endif /* NOT_USED */

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


#ifdef NOT_USED
static void extract_vec_params(Data_Vector *dvp, Data_Obj *dp)
{
	int i;
	incr_t inc,n;
	int need_inc;
	int start_dim;

	if( dp == NO_OBJ ){
		SET_DV_VEC(dvp, NULL);
		SET_DV_INC(dvp, 0);
		SET_DV_COUNT(dvp, 0);
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
		n *= OBJ_TYPE_DIM(dp,i);
		if( need_inc && (OBJ_TYPE_DIM(dp,i) > 1) ){
			inc = OBJ_TYPE_INC(dp,i);
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

	SET_DV_VEC(dvp, OBJ_DATA_PTR(dp) );
	SET_DV_INC(dvp, inc);
	SET_DV_COUNT(dvp, n);
	SET_DV_PREC(dvp, OBJ_PREC(dp));	/* BUG? should be OBJ_MACH_PREC??? */
	SET_DV_BIT0(dvp, OBJ_BIT0(dp) );
	if( IS_BITMAP(dp) )
		SET_DV_FLAG_BITS(dvp, DV_BITMAP);

#ifdef QUIP_DEBUG
if( debug & veclib_debug ){
/* LONGLIST(dp); */
sprintf(DEFAULT_ERROR_STRING,"extract_vec_params:  obj %s, prec %s, max_vect = %d,  n = %d, inc = %d",
OBJ_NAME(dp),OBJ_PREC_NAME(dp),max_vectorizable,DV_COUNT(dvp),DV_INC(dvp));
NADVISE(DEFAULT_ERROR_STRING);
show_data_vector(DEFAULT_QSP_ARG  dvp);
}
#endif /* QUIP_DEBUG */
}
#endif /* NOT_USED */

#ifdef NOT_YET
void show_vf(Vector_Function *vfp)
{
	sprintf(DEFAULT_ERROR_STRING,"function %s, flags = 0x%x",VF_NAME(vfp),VF_FLAGS(vfp) );
	NADVISE(DEFAULT_ERROR_STRING);
	/*
	sprintf(DEFAULT_ERROR_STRING,"V_INPLACE = 0x%x",V_INPLACE);
	NADVISE(DEFAULT_ERROR_STRING);
	*/
}
#endif /* NOT_YET */

static int get_src_bitmap(QSP_ARG_DECL Vec_Obj_Args *oap)
{
	SET_OA_SBM( oap, PICK_OBJ( "source bitmap object" ) );
	if( OA_SBM(oap) ==NO_OBJ ) return(-1);
	if( OBJ_PREC( OA_SBM(oap) ) != PREC_BIT ){
		sprintf(ERROR_STRING,
			"bitmap \"%s\" (%s,0x%x) must have bit precision (0x%x)",
			OBJ_NAME(OA_SBM(oap) ),
			OBJ_PREC_NAME( OA_SBM(oap) ), OBJ_PREC( OA_SBM(oap) ),PREC_BIT);
		WARN(ERROR_STRING);
		return(-1);
	}
	return(0);
}

static int get_dst_bitmap(QSP_ARG_DECL Vec_Obj_Args *oap)
{
	SET_OA_DBM( oap, PICK_OBJ( "destination bitmap object" ) );
	if( OA_DBM(oap) ==NO_OBJ ) return(-1);
	if( OBJ_PREC( OA_DBM(oap) ) != PREC_BIT ){
		sprintf(ERROR_STRING,
			"bitmap \"%s\" (%s,0x%x) must have bit precision (0x%x)",
			OBJ_NAME(OA_DBM(oap) ),
			OBJ_PREC_NAME( OA_DBM(oap) ), OBJ_PREC( OA_DBM(oap) ),PREC_BIT);
		WARN(ERROR_STRING);
		return(-1);
	}
	return(0);
}

#ifdef FOOBAR
static Scalar_Value *sv_array=NULL;
static int sv_idx=0;
#define N_SV_ARRAY 8	// 3 might be enough?
#endif // FOOBAR

static Scalar_Value *get_sv(Precision * prec_p)
{
#ifndef FOOBAR
	Scalar_Value *svp;

	/* We pass the precision so that we don't have
	 * to allocate the huge Scalar_Value union for just one float...
	 */

	/* But we're not taking advantage of that here... */

	svp = (Scalar_Value *)getbuf( sizeof(Scalar_Value) );
	/* BUG?  should we initialize value? */
	return(svp);			/* BUG possible memory leak? */
	// BUT we free these later???
#else // FOOBAR
	Scalar_Value *svp;

	if( sv_array == NULL ){
		sv_array = getbuf( N_SV_ARRAY * sizeof(Scalar_Value) );
	}

	svp = &sv_array[sv_idx++];
	sv_idx %= N_SV_ARRAY;
	return svp;
#endif // FOOBAR
}

/*
 * Get a scalar object that the user specifies.
 */

static Data_Obj * getascal(QSP_ARG_DECL const char *pmpt,Precision *prec_p)
{
	Data_Obj *dp;

	/* which data area does PICK_OBJ use??? */
	dp=PICK_OBJ( pmpt );
	if( dp==NO_OBJ ) return(NO_OBJ);
	if( !IS_SCALAR(dp) ){
		sprintf(ERROR_STRING,
			"getascal:  %s is not a scalar",OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return(NO_OBJ);
	}
	if( OBJ_PREC_PTR( dp) != prec_p ){
		sprintf(ERROR_STRING,
			"getascal:  %s scalar %s should have precision %s",
			OBJ_PREC_NAME(dp),OBJ_NAME(dp),PREC_NAME(prec_p));
		WARN(ERROR_STRING);
		return(NO_OBJ);
	}
#ifdef QUIP_DEBUG
if( debug & veclib_debug ){
sprintf(ERROR_STRING,"getascal:  returning %s scalar %s",OBJ_MACH_PREC_NAME(dp),OBJ_NAME(dp));
NADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	return(dp);
}

#ifdef HAVE_ANY_GPU

// This could be obsolete now that we are using platforms???

/* Set the current data area to match whatever we are using,
 * so that automatically created objects (e.g. scalars) are in the
 * correct space.  (What about setting the value of scalars on the gpu?)
 */

static Data_Area * set_arg_data_area(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	Data_Area *ap;

	if( OA_DEST(oap) != NO_OBJ ){
		push_data_area(ap=OA_DEST(oap) ->dt_ap);
	} else {
		int i;
		ap=NO_AREA;	// quiet compiler
		for(i=0;i<MAX_N_ARGS;i++){
			if( OA_SRC_OBJ(oap,i) != NO_OBJ ){
				push_data_area(ap=OA_SRC_OBJ(oap,i)->dt_ap);
				i=MAX_N_ARGS+5;	// force exit & flag found
			}
		}
		if( i == MAX_N_ARGS ){
			/* This was originally written as a CAUTIOUS //
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

#endif // HAVE_ANY_GPU

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
 *
 * BUG - source scalars get their precision from OA_DEST, but this is wrong
 * for mixed precision ops like spdp_rvsmul... or at least it was when
 * the kernel code assumed that all the source ops have the same precision.
 * The easiest fix is probably to change the cast in the kernel code...
 */

#define SET_PREC							\
									\
	if( OA_DEST(oap) !=NO_OBJ ) prec_p=OBJ_PREC_PTR( OA_DEST(oap) );\
	else prec_p=prec_for_code(PREC_SP);

static int getscal(QSP_ARG_DECL Vec_Obj_Args *oap, Vector_Function *vfp)
{
	int ir, ic, iq;
	Precision * prec_p;
	int retval=0;
#ifdef HAVE_ANY_GPU
	Data_Area *ap;
#endif /* HAVE_ANY_GPU */

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

#ifdef HAVE_ANY_GPU

	ap=set_arg_data_area(QSP_ARG  oap);

	// suppress compiler warning by checking return value
	// This never should happen...
	if( ap == NO_AREA ) WARN("bad return value from set_arg_data_area");

#endif /* HAVE_ANY_GPU */

	if( VF_FLAGS(vfp) & SRC_SCALAR3 /* HAS_3_SCALARS */ ){
		const char *p1,*p2,*p3;

		SET_PREC

		if( VF_CODE(vfp) == FVRAMP2D ){
			p1="starting value";
			p2="horizontal increment";
			p3="vertical increment";
		} else if( VF_CODE(vfp) >= FSS_VS_LT &&
				VF_CODE(vfp) <= FSS_VS_NE ){
			p1="result value if condition satisfied";
			p2="result value if condition not satisfied";
			p3="scalar value for comparison";
		}
//#ifdef CAUTIOUS
		  else {
			// quiet compiler
			//p1=p2=p3="";
		// not necessary when using noreturn attribute!
//			ERROR1("CAUTIOUS:  unexpected 3 scalar function!?");
//			  return -1;	// iOS
			assert( AERROR("unexpected 3 scalar function!?") );
		}
//#endif /* CAUTIOUS */
		SET_OA_SVAL(oap,0, get_sv(prec_p) );
		SET_OA_SVAL(oap,1, get_sv(prec_p) );
		SET_OA_SVAL(oap,2, get_sv(prec_p) );
		cast_to_scalar_value(QSP_ARG  OA_SVAL(oap,0), prec_p, HOW_MUCH(p1) );
		cast_to_scalar_value(QSP_ARG  OA_SVAL(oap,1), prec_p, HOW_MUCH(p2) );
		cast_to_scalar_value(QSP_ARG  OA_SVAL(oap,2), prec_p, HOW_MUCH(p3) );
	} else if( VF_FLAGS(vfp) & SRC_SCALAR2 /* HAS_2_SCALARS */ ){
		const char *p1,*p2;

		SET_PREC

		if( COMPLEX_PRECISION(PREC_CODE(prec_p)) ){
			/* this should not happen!? */
			/* Does the function permit complex? */
			if( (VF_TYPEMASK(vfp) &
					(CPX_ARG_MASK|MIXED_ARG_MASK)) == 0 ){
				// BUG??? can OA_DEST be null here???
				// We used to print the name of the
				// destination obj here, but in case
				// it could be null...
				if( OA_DEST(oap) != NO_OBJ )
					sprintf(ERROR_STRING,
"getscal:  function %s does not permit operations with complex targets (%s)",
		VF_NAME(vfp),OBJ_NAME(OA_DEST(oap) ));
				else
					sprintf(ERROR_STRING,
"getscal:  function %s does not permit operations with complex targets",
						VF_NAME(vfp) );
				WARN(ERROR_STRING);
				retval=(-1);
			}
		}

		if( QUAT_PRECISION(PREC_CODE(prec_p)) ){
			/* this should not happen!? */
			/* Does the function permit quaternions? */
			if( (VF_TYPEMASK(vfp) & (QUAT_ARG_MASK)) == 0 ){
				sprintf(ERROR_STRING,
	"getscal:  function %s does not permit operations with quaternion targets (%s)",
					VF_NAME(vfp),
					OA_DEST(oap)==NULL ?
					"(null destination)" :
					OBJ_NAME(OA_DEST(oap) ));
				WARN(ERROR_STRING);
				retval=(-1);
			}
		}

		if( VF_CODE(vfp) == FVRAMP1D ){
			p1="starting value";
			p2="ramp increment";
		} else if( VF_CODE(vfp) == FVSSSLCT ){
			p1="scalar1";
			p2="scalar2";
		} else if( VF_CODE(vfp) >= FVS_VS_LT &&
				VF_CODE(vfp) <= FVS_VS_NE ){
			p1="result value for test false";
			p2="scalar value for comparison";
		} else if( VF_CODE(vfp) >= FSS_VV_LT &&
				VF_CODE(vfp) <= FSS_VV_NE ){
			p1="result value for test true";
			p2="result value for test false";
		}
//#ifdef CAUTIOUS
		else {
//			WARN("CAUTIOUS:  unhandled case in getscal");
//			p1=p2="dummy value";
//			retval=(-1);
			assert( AERROR("unhandled case in getscal") );
		}
//#endif /* CAUTIOUS */
		SET_OA_SVAL(oap,0, get_sv(prec_p) );
		SET_OA_SVAL(oap,1, get_sv(prec_p) );
		cast_to_scalar_value(QSP_ARG  OA_SVAL(oap,0), prec_p, HOW_MUCH(p1) );
		cast_to_scalar_value(QSP_ARG  OA_SVAL(oap,1), prec_p, HOW_MUCH(p2) );
	}

	else if( VF_FLAGS(vfp) & SRC_SCALAR1 ){
		if( VF_FLAGS(vfp) == VS_TEST ){	/* vsm_lt etc. */
			if( OA_SRC1(oap) ==NO_OBJ ){
				goto get_dummy;
			}
			SET_OA_SVAL(oap,0, get_sv(OBJ_PREC_PTR( OA_SRC1(oap) )) );
			cast_to_scalar_value(QSP_ARG  OA_SVAL(oap,0), OBJ_PREC_PTR( OA_SRC1(oap) ),
				HOW_MUCH("source scalar value") );
		} else if( OA_DEST(oap) == NO_OBJ ){	/* error condition */
			/*double d;
			d=*/HOW_MUCH("dummy value");
			retval=(-1);
		} else if( IS_REAL(OA_DEST(oap) ) || ir ){
			if( ic ) WARN("Multiplication by a complex scalar with a real target");
			if( iq ) WARN("Multiplication by a quaternion scalar with a real target");
			/* BUG we can't use destination for precision
			 * in the mixed precision ops...
			 */
			SET_OA_SVAL(oap,0,get_sv(OBJ_MACH_PREC_PTR(OA_DEST(oap) )) );
			cast_to_scalar_value(QSP_ARG  OA_SVAL(oap,0), OBJ_MACH_PREC_PTR(OA_DEST(oap) ),
				HOW_MUCH("source real scalar value") );
		} else if( (IS_COMPLEX(OA_DEST(oap) ) && !ir) || ic ) {
			SET_OA_SVAL(oap,0, get_sv(OBJ_MACH_PREC_PTR(OA_DEST(oap) )) );
			cast_to_cpx_scalar(QSP_ARG  0,OA_SVAL(oap,0), OBJ_MACH_PREC_PTR(OA_DEST(oap) ),
				HOW_MUCH("source scalar value real part") );
			cast_to_cpx_scalar(QSP_ARG  1,OA_SVAL(oap,0), OBJ_MACH_PREC_PTR(OA_DEST(oap) ),
				HOW_MUCH("source scalar value imaginary part") );
		} else if( (IS_QUAT(OA_DEST(oap) ) && !ir) || iq ) {
			SET_OA_SVAL(oap,0, get_sv(OBJ_MACH_PREC_PTR(OA_DEST(oap) )) );
			cast_to_quat_scalar(QSP_ARG  0,OA_SVAL(oap,0), OBJ_MACH_PREC_PTR(OA_DEST(oap) ),
				HOW_MUCH("source scalar value real part") );
			cast_to_quat_scalar(QSP_ARG  1,OA_SVAL(oap,0), OBJ_MACH_PREC_PTR(OA_DEST(oap) ),
				HOW_MUCH("source scalar value i part") );
			cast_to_quat_scalar(QSP_ARG  2,OA_SVAL(oap,0), OBJ_MACH_PREC_PTR(OA_DEST(oap) ),
				HOW_MUCH("source scalar value j part") );
			cast_to_quat_scalar(QSP_ARG  3,OA_SVAL(oap,0), OBJ_MACH_PREC_PTR(OA_DEST(oap) ),
				HOW_MUCH("source scalar value k part") );
		} else {
			/* use a single scalar for all components */
			SET_OA_SVAL(oap,0, get_sv(OBJ_MACH_PREC_PTR(OA_DEST(oap) )) );
			cast_to_scalar_value(QSP_ARG  OA_SVAL(oap,0), OBJ_MACH_PREC_PTR(OA_DEST(oap) ),
				HOW_MUCH("source scalar value") );
		}
		if( OA_SVAL(oap,0) == NO_SCALAR_VALUE ){
			retval=(-1);
		}
	}

	if( VF_FLAGS(vfp) & TWO_SCALAR_RESULTS ){
		Data_Obj *_dp1, *_dp2;
		if( OA_SRC1(oap) == NO_OBJ ){
			sprintf(ERROR_STRING,
	"getscal (%s):  no argument to use for precision prototype!?",
				VF_NAME(vfp));
			WARN(ERROR_STRING);
			retval=(-1);
		} else {
			_dp1 = getascal(QSP_ARG
				"name of scalar for extreme value",
				OBJ_PREC_PTR( OA_SRC1(oap) ));
			_dp2 = getascal(QSP_ARG
				"name of scalar for # of occurrences",
				prec_for_code(PREC_DI));
			if( _dp1 == NO_OBJ || _dp2 == NO_OBJ )
				retval=(-1);
			SET_OA_SCLR1(oap,_dp1);
			SET_OA_SCLR2(oap,_dp2);
		}
	}

#ifdef HAVE_ANY_GPU
	pop_data_area();
#endif /* HAVE_ANY_GPU */

	return(retval);

get_dummy:
	{
		/* avoid a parsing error */
		/*float dummy;

	 dummy = (float)*/
	//const char *fn;
	//int value;

	/*fn=*/ NAMEOF("flag name");
 HOW_MUCH("dummy scalar value");
		return(-1);
	}
}

static int get_args(QSP_ARG_DECL  Vec_Obj_Args *oap,Vector_Function *vfp)
{
	int oops=0;

	clear_obj_args(oap);

	if( VF_FLAGS(vfp) & BITMAP_SRC )
		oops|=get_src_bitmap(QSP_ARG  oap);

	if( VF_FLAGS(vfp) & BITMAP_DST )
		oops|=get_dst_bitmap(QSP_ARG  oap);

	if( VF_FLAGS(vfp) & DST_VEC )
		oops|=get_dst(QSP_ARG  oap);

	if( VF_FLAGS(vfp) & SRC1_VEC )
		oops|=get_src1(QSP_ARG  oap);
	if( VF_FLAGS(vfp) & SRC2_VEC )
		oops|=get_src2(QSP_ARG  oap);
	if( VF_FLAGS(vfp) & SRC3_VEC )
		oops|=get_src3(QSP_ARG  oap);
	if( VF_FLAGS(vfp) & SRC4_VEC )
		oops|=get_src4(QSP_ARG  oap);

	// Now, for vmov and bitmaps, the destination bitmap
	// is the same oap member as the normal dest, but a source
	// bitmap is expected to be src5, not src1...
	if( VF_CODE(vfp) == FVMOV && OA_SRC1(oap) != NO_OBJ ){
		if( IS_BITMAP(OA_SRC1(oap)) ){
//fprintf(stderr,"Copying src1 to sbm...\n");
			SET_OA_SBM(oap,OA_SRC1(oap) );
		}
//else {
//fprintf(stderr,"Source is not a bitmap...\n");
//}
	}
		
	/* BUG?  We should sort out passing a value vs. receiving a value in a scalar object... */
	if( VF_FLAGS(vfp) & (SRC_SCALAR1|SRC_SCALAR2|TWO_SCALAR_RESULTS) ){
		if( getscal(QSP_ARG  oap, vfp) == (-1) ) oops|=1;
#ifdef PROBABLY_NOT_NEEDED
		/* Here we point the scalar value to the object data - why? */
		for(i=0;i<MAX_RETSCAL_ARGS;i++){
			if( OA_SCLR_OBJ(oap,i) != NO_OBJ )
				SET_OA_SVAL(oap,i, (Scalar_Value *)OBJ_DATA_PTR( OA_SCLR_OBJ(oap,i) ) );
		}
#endif /* PROBABLY_NOT_NEEDED */
	}

	if( oops) return(-1);		/* if a requested object doesn't exist... */

	/* getscal() gets (or makes) objects? */

	return(0);
} /* end get_args */

#ifdef FOOBAR
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
		*((int32_t *)OBJ_DATA_PTR( dp )) = lvalue;
	} else if( prec==PREC_UDI ){
		uint32_t lvalue;
		lvalue=(uint32_t) HOW_MANY(pmpt);
		*((uint32_t *)OBJ_DATA_PTR( dp )) = lvalue;
	} else if( prec==PREC_LI ){
		int64_t lvalue;
		lvalue=(int64_t)HOW_MANY(pmpt);
		*((int64_t *)OBJ_DATA_PTR( dp ))=lvalue;
	} else if( prec==PREC_ULI ){
		uint64_t lvalue;
		lvalue=(uint64_t)HOW_MANY(pmpt);
		*((uint64_t *)OBJ_DATA_PTR( dp ))=lvalue;
	} else if( prec==PREC_BY ){
		char cvalue;
		cvalue=(char)HOW_MANY(pmpt);
		*((char *)OBJ_DATA_PTR( dp ))=cvalue;
	} else if( prec==PREC_UBY ){
		u_char cvalue;
		cvalue=(u_char)HOW_MANY(pmpt);
		*((u_char *)OBJ_DATA_PTR( dp ))=cvalue;
	} else if( prec==PREC_DP ){
		double value;
		value=HOW_MUCH(pmpt);
		*((double *)OBJ_DATA_PTR( dp ))=value;
	} else if( prec==PREC_SP ){
		float value;
		value=(float)HOW_MUCH(pmpt);
		*((float *)OBJ_DATA_PTR( dp ))=value;
	} else if( prec==PREC_IN ){
		short svalue;
		svalue=(short)HOW_MANY(pmpt);
		*((short *)OBJ_DATA_PTR( dp )) = svalue;
	} else if( prec==PREC_UIN ){
		u_short svalue;
		svalue=(u_short)HOW_MANY(pmpt);
		*((u_short *)OBJ_DATA_PTR( dp )) = svalue;
	}

//#ifdef CAUTIOUS
	else {
//		sprintf(ERROR_STRING,
//	"CAUTIOUS:  prompt_scalar_value:  unsupported precision \"%s\" (0x%x)",
//			NAME_FOR_PREC_CODE(prec),prec);
//		WARN(ERROR_STRING);
//		return(-1);
		assert( AERROR("prompt_scalar_value:  unsupported precision!?") );
	}
//#endif /* CAUTIOUS */
	return(0);
}
#endif /* FOOBAR */

#ifdef FOOBAR
/* We have a problem introduced by trying to use the Data_Obj framework for images in nVidia CUDA:
 * We can't mix-and-match, one solution might be to have separate name spaces for gpu and ram objects.
 * For now, we just check that all objects are from the ram data area.
 */


static int check_one_obj_loc( Data_Obj *dp )
{
	if( dp == NO_OBJ ) return(0);
	if( OBJ_IS_RAM(dp) ) return(OARGS_RAM);
	return(OARGS_GPU);
}
#endif // FOOBAR

// We used to have just two possible locations (ram/gpu)
// That was a little brain-damaged, because with CUDA there
// is the possibility of more than one GPU.  But now we can also
// have OpenCL devices - more than one.  We really need to insure
// that all objects are on the same device.

#ifdef FOOBAR
#define CHECK_LOC( dp )							\
									\
	s = check_one_obj_loc( dp );					\
	if( s == OARGS_RAM ) all_gpu=0;					\
	if( s == OARGS_GPU ) all_ram=0;
#endif // FOOBAR

static int check_obj_device(Data_Obj *dp, Vec_Obj_Args *oap)
{
	if( dp == NO_OBJ ) return 0;	// not an error
	if( OA_PFDEV(oap) == NULL ){
		SET_OA_PFDEV(oap,OBJ_PFDEV(dp));
if( OA_PFDEV(oap) == NULL ) NWARN("Null platform device!?");
		return 0;
	}
	if( OA_PFDEV(oap) != OBJ_PFDEV(dp) ){
		sprintf(DEFAULT_ERROR_STRING,
"check_obj_device:  object %s device %s does not match expected device %s!?",
			OBJ_NAME(dp),
			PFDEV_NAME( OBJ_PFDEV(dp) ),
			PFDEV_NAME( OA_PFDEV(oap) ) );
		NWARN(DEFAULT_ERROR_STRING);
		return -1;
	}
	return 0;
}

int check_obj_devices( Vec_Obj_Args *oap )
{
	//int s, all_ram=1, all_gpu=1;
	int i;

	if( HAS_CHECKED_ARGS(oap) ){
		return 0;
	}

	if( check_obj_device(OA_DEST(oap),oap) < 0 ){
		NWARN("check_obj_devices:  bad destination!?");
		return -1;
	}

	for(i=0;i<MAX_N_ARGS;i++){
		if( check_obj_device(OA_SRC_OBJ(oap,i),oap) < 0 ){
			// BUG give a better error message
			NWARN("platform device mismatch!?");
			return -1;
		}
	}
	// Source scalars don't need to be checked...
	// But these should only be used for scalar results...
	if( check_obj_device(OA_SCLR1(oap),oap) < 0 )
		return -1;
	if( check_obj_device(OA_SCLR2(oap),oap) < 0 )
		return -1;

	SET_OA_FLAGS(oap, OARGS_CHECKED);
	return 0;
}

static int non_ram_obj(Data_Obj *dp)
{
	if( dp == NO_OBJ || OBJ_IS_RAM( dp ) ) return 0;
	return 1;
}

int are_ram_args( Vec_Obj_Args *oap )
{
	int i;

	// This is kind of inefficient, as we look at everything...
	// It might be better to use a flag that we set when the first obj
	// is set, and then check for additional objects...

	if( non_ram_obj( OA_DEST(oap) ) ) return 0;
	for(i=0;i<MAX_N_ARGS;i++)
		if( non_ram_obj( OA_SRC_OBJ(oap,i) ) ) return 0;
	for(i=0;i<MAX_RETSCAL_ARGS;i++)
		if( non_ram_obj( OA_SCLR_OBJ(oap,i) ) ) return 0;
	return 1;
}

static void report_obj_device(QSP_ARG_DECL  Data_Obj *dp)
{
	if( dp == NO_OBJ ) return;
	sprintf(ERROR_STRING,"\t%s:\t%s",OBJ_NAME(dp),PFDEV_NAME( OBJ_PFDEV(dp) ) );
	advise(ERROR_STRING);
}

void mixed_location_error(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap)
{
	int i;

	sprintf(ERROR_STRING,"%s:  arguments must reside on a single device.",VF_NAME(vfp));
	WARN(ERROR_STRING);

	report_obj_device(QSP_ARG  OA_DEST(oap) );
	for(i=0;i<MAX_N_ARGS;i++)
		report_obj_device( QSP_ARG  OA_SRC_OBJ(oap,i) );
	for(i=0;i<MAX_RETSCAL_ARGS;i++)
		report_obj_device( QSP_ARG  OA_SCLR_OBJ(oap,i) );
}

void do_vfunc( QSP_ARG_DECL  Vector_Function *vfp )
{
	Vec_Obj_Args oa1, *oap=&oa1;
	int i;

	clear_obj_args(oap);

	if( get_args(QSP_ARG  oap, vfp) < 0 ){
		sprintf(ERROR_STRING,"Error getting arguments for function %s",
			VF_NAME(vfp));
		WARN(ERROR_STRING);
		return;
	}

if( OA_DEST(oap) == NO_OBJ ){
/* BUG?  is this really an error?  The bitmap destination might be here... */
sprintf(ERROR_STRING,"%s:  Null destination!?!?", VF_NAME(vfp));
WARN(ERROR_STRING);
}

//fprintf(stderr,"Calling %s...\n",VF_NAME(vfp));
	call_vfunc(QSP_ARG  vfp,oap);

	/* Now free the scalars (if any) */

	/* Perhaps the code would be more efficient if
	 * oargs contained the scalar value struct
	 * itself instead of a pointer to dynamically
	 * allocated memory...  (is getbuf/malloc thread-safe?)
	 */
	for(i=0;i<MAX_SRCSCAL_ARGS;i++){
		if( OA_SVAL(oap,i) != NO_SCALAR_VALUE )
			givbuf( OA_SVAL(oap,i) );
	}
}

void setvarg1(Vec_Obj_Args *oap, Data_Obj *dp)
{
	clear_obj_args(oap);
	SET_OA_DEST(oap, dp);
	set_obj_arg_flags(oap);
}

void setvarg2(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *srcv)
{
	clear_obj_args(oap);
	SET_OA_DEST(oap,dstv);
	SET_OA_SRC1(oap,srcv);
	set_obj_arg_flags(oap);
}

void setvarg3(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *src1,Data_Obj *src2)
{
	clear_obj_args(oap);
	SET_OA_DEST(oap,dstv);
	SET_OA_SRC2(oap,src2);
	SET_OA_SRC1(oap,src1);
	SET_OA_SRC3(oap,NO_OBJ);
	SET_OA_SRC4(oap,NO_OBJ);
	set_obj_arg_flags(oap);
}

void setvarg4(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *src1,Data_Obj *src2,Data_Obj *src3)
{
	clear_obj_args(oap);
	SET_OA_DEST(oap,dstv);
	SET_OA_SRC1(oap,src1);
	SET_OA_SRC2(oap,src2);
	SET_OA_SRC3(oap,src3);
	SET_OA_SRC4(oap,NO_OBJ);
	set_obj_arg_flags(oap);
}

void setvarg5(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *src1,Data_Obj *src2,Data_Obj *src3, Data_Obj *src4)
{
	clear_obj_args(oap);
	SET_OA_DEST(oap,dstv);
	SET_OA_SRC1(oap,src1);
	SET_OA_SRC2(oap,src2);
	SET_OA_SRC3(oap,src3);
	SET_OA_SRC4(oap,src4);
	set_obj_arg_flags(oap);
}

/* Not sure where this should go... */

static void show_increments(Increment_Set *isp)
{
	if( isp == NULL ) return;

	fprintf(stderr,"\t%d, %d, %d, %d, %d\n",
		isp->is_increment[4],
		isp->is_increment[3],
		isp->is_increment[2],
		isp->is_increment[1],
		isp->is_increment[0] );
}

void show_vec_args(Vector_Args *vap)
{
	int i;

	fprintf(stderr,"show_vec_args 0x%lx:\n\tdst = 0x%lx\n",
		(long)vap,(long)VA_DEST_PTR(vap));

	for(i=0;i<MAX_N_ARGS;i++){
		if( VA_SRC_PTR(vap,i) != NULL )
			fprintf(stderr,"\tsrc[%d]\t0x%lx\n",i,(long)VA_SRC_PTR(vap,i) );
	}
	if( vap->va_sval[0] != NULL ){
		fprintf(stderr,"\tscalars:\n");
		for(i=0;i<3;i++){
			if( vap->va_sval[i] != NULL )
				fprintf(stderr,"\tsval[%d]\t0x%lx\n",i,(long)vap->va_sval[i]);
		}
	}

	if( ARE_SLOW_ARGS(vap) ){
		fprintf(stderr,"\tslow increments:\n");
#define SHOW_INC_IF( incset )	if( incset != NULL ) show_increments(incset);
		SHOW_INC_IF( VA_DEST_INCSET( vap ) )
		SHOW_INC_IF( VA_SRC1_INCSET( vap ) )
		SHOW_INC_IF( VA_SRC2_INCSET( vap ) )
		SHOW_INC_IF( VA_SRC3_INCSET( vap ) )
		SHOW_INC_IF( VA_SRC4_INCSET( vap ) )
		SHOW_INC_IF( VA_SRC5_INCSET( vap ) )
		/* show sizes? */
	} else if( ARE_EQSP_ARGS(vap) ){
		fprintf(stderr,"\teqsp increments:\n");
		fprintf(stderr,"\tdst_inc = %d\n",VA_DEST_INC( vap ) );
		fprintf(stderr,"\tsrc1_inc = %d\n",VA_SRC1_INC( vap ) );
		fprintf(stderr,"\tsrc2_inc = %d\n",VA_SRC2_INC( vap ) );
		fprintf(stderr,"\tsrc3_inc = %d\n",VA_SRC3_INC( vap ) );
		fprintf(stderr,"\tsrc4_inc = %d\n",VA_SRC4_INC( vap ) );
		fprintf(stderr,"\tsrc5_inc = %d\n",VA_SRC5_INC( vap ) );
	}
}

