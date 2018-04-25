
#include "quip_config.h"
#include "quip_prot.h"
#include "veclib_prot.h"
#include "platform.h"

#include <stdio.h>

#include "nvf.h"
//#include "new_chains.h"
#include "debug.h"
//#include "warn.h"
//#include "getbuf.h"

// bitnum_t is unsigned, so we can't use (-1) as an invalid value...
#define INVALID_WORD_TBL_IDX	INVALID_DIMENSION
#define INVALID_WORD_IDX	INVALID_DIMENSION

#define JUST_FOR_DEBUGGING	// extra debugging

static int get_dst(QSP_ARG_DECL Vec_Obj_Args *oap)
{
	SET_OA_DEST(oap, pick_obj( "destination vector" ) );
	if( OA_DEST(oap) == NULL )
		return(-1);
	return(0);
}

static int get_src1(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	SET_OA_SRC1(oap,pick_obj( "first source vector" ) );
	if( OA_SRC1(oap) == NULL )
		return(-1);
	return(0);
}

static int get_src2(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	SET_OA_SRC2(oap,pick_obj( "second source vector" ));
	if( OA_SRC2(oap) == NULL )
		return(-1);
	return(0);
}

static int get_src3(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	SET_OA_SRC3(oap,pick_obj( "third source vector" ));
	if( OA_SRC3(oap) == NULL )
		return(-1);
	return(0);
}


static int get_src4(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	SET_OA_SRC4(oap,pick_obj( "fourth source vector" ));
	if( OA_SRC4(oap) == NULL )
		return(-1);
	return(0);
}

#ifdef NOT_YET
void show_vf(Vector_Function *vfp)
{
	sprintf(ERROR_STRING,"function %s, flags = 0x%x",VF_NAME(vfp),VF_FLAGS(vfp) );
	advise(ERROR_STRING);
	/*
	sprintf(ERROR_STRING,"V_INPLACE = 0x%x",V_INPLACE);
	advise(ERROR_STRING);
	*/
}
#endif /* NOT_YET */

static int get_src_bitmap(QSP_ARG_DECL Vec_Obj_Args *oap)
{
	SET_OA_SBM( oap, pick_obj( "source bitmap object" ) );
	if( OA_SBM(oap) == NULL ) return(-1);
	if( OBJ_PREC( OA_SBM(oap) ) != PREC_BIT ){
		sprintf(ERROR_STRING,
			"get_src_bitmap:  bitmap \"%s\" (%s,0x%x) must have bit precision (0x%x)",
			OBJ_NAME(OA_SBM(oap) ),
			OBJ_PREC_NAME( OA_SBM(oap) ), OBJ_PREC( OA_SBM(oap) ),PREC_BIT);
		warn(ERROR_STRING);
		return(-1);
	}
	return(0);
}

static int get_dst_bitmap(QSP_ARG_DECL Vec_Obj_Args *oap)
{
	SET_OA_DBM( oap, pick_obj( "destination bitmap object" ) );
	if( OA_DBM(oap) == NULL ) return(-1);
	if( OBJ_PREC( OA_DBM(oap) ) != PREC_BIT ){
		sprintf(ERROR_STRING,
			"get_dst_bitmap:  bitmap \"%s\" (%s,0x%x) must have bit precision (0x%x)",
			OBJ_NAME(OA_DBM(oap) ),
			OBJ_PREC_NAME( OA_DBM(oap) ), OBJ_PREC( OA_DBM(oap) ),PREC_BIT);
		warn(ERROR_STRING);
		return(-1);
	}
	return(0);
}

static List *free_sval_lp=NULL;

/* We pass the precision so that we don't have
 * to allocate the huge Scalar_Value union for just one float...
 */
/* But we're not taking advantage of that here... */

static Scalar_Value *alloc_sval(Precision * prec_p)
{
	Scalar_Value *svp;

	if( free_sval_lp == NULL ) free_sval_lp = new_list();

	if( QLIST_HEAD(free_sval_lp) != NULL ){
		Node *np;
		np = remHead(free_sval_lp);
		svp = NODE_DATA(np);
		rls_node(np);
		return svp;
	}

	// We could get a block of svp's and add to the list here!  BUG
	svp = (Scalar_Value *)getbuf( sizeof(Scalar_Value) );
	/* BUG?  should we initialize value? */
	return(svp);			/* BUG possible memory leak? */
	// BUT we free these later???
	// where do we free?  memory leak?
}

static void rls_sval(Scalar_Value *svp)
{
	Node *np;

	np = mk_node(svp);

	assert( free_sval_lp != NULL );
	addHead( free_sval_lp, np );
}

/*
 * Get a scalar object that the user specifies.
 */

static Data_Obj * get_return_scalar(QSP_ARG_DECL const char *pmpt,Precision *prec_p)
{
	Data_Obj *dp;

	/* which data area does pick_obj use??? */
	dp=pick_obj( pmpt );
	if( dp == NULL ) return(NULL);
	if( !IS_SCALAR(dp) ){
		sprintf(ERROR_STRING,
			"get_return_scalar:  %s is not a scalar",OBJ_NAME(dp));
		warn(ERROR_STRING);
		return(NULL);
	}
	if( OBJ_PREC_PTR( dp) != prec_p ){
		sprintf(ERROR_STRING,
			"get_return_scalar:  %s scalar %s should have precision %s",
			OBJ_PREC_NAME(dp),OBJ_NAME(dp),PREC_NAME(prec_p));
		warn(ERROR_STRING);
		return(NULL);
	}
#ifdef QUIP_DEBUG
if( debug & veclib_debug ){
sprintf(ERROR_STRING,"get_return_scalar:  returning %s scalar %s",OBJ_MACH_PREC_NAME(dp),OBJ_NAME(dp));
advise(ERROR_STRING);
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

	if( OA_DEST(oap) != NULL ){
		push_data_area(ap=OA_DEST(oap) ->dt_ap);
	} else {
		int i;
		ap = NULL;	// quiet compiler
		for(i=0;i<MAX_N_ARGS;i++){
			if( OA_SRC_OBJ(oap,i) != NULL ){
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
			ap = default_data_area();
			push_data_area(ap);
		}
	}
	return(ap);
}

#endif // HAVE_ANY_GPU

/* BUG - get_scalar_args gets a scalar object, but for an arg we don't need to create
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

#define SET_MACH_PREC_FROM_OBJ( prec_ptr, dp )				\
									\
	{								\
		assert( dp != NULL );					\
		prec_ptr = OBJ_MACH_PREC_PTR( dp );			\
	}

#define SET_PREC_FROM_OBJ( prec_ptr, dp )				\
									\
	{								\
		assert( dp != NULL );					\
		prec_ptr = OBJ_PREC_PTR( dp );				\
	}

#ifdef FOOBAR
#define SET_PREC							\
									\
	if( OA_DEST(oap) != NULL ) prec_p=OBJ_PREC_PTR( OA_DEST(oap) );\
	else prec_p=prec_for_code(PREC_SP);
#endif // FOOBAR

static int get_scalar_args(QSP_ARG_DECL Vec_Obj_Args *oap, Vector_Function *vfp)
{
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

#ifdef HAVE_ANY_GPU

	ap = set_arg_data_area(QSP_ARG  oap);

	// suppress compiler warning by checking return value
	// This never should happen...
	//if( ap == NULL ) warn("bad return value from set_arg_data_area");
	assert( ap != NULL );

#endif /* HAVE_ANY_GPU */

	if( VF_FLAGS(vfp) & SRC_SCALAR3 /* HAS_3_SCALARS */ ){
		const char *p1,*p2,*p3;

		// Ramp2D or 3 scalar conditional...
		//SET_PREC

		if( VF_CODE(vfp) == FVRAMP2D ){
			SET_PREC_FROM_OBJ( prec_p, OA_DEST(oap) );
			p1="starting value";
			p2="horizontal increment";
			p3="vertical increment";
		} else if( VF_CODE(vfp) >= FSS_VS_LT &&
				VF_CODE(vfp) <= FSS_VS_NE ){
			// Here the result scalars should match OA_DEST,
			// but the source scalar should match OA_SRC1
			SET_PREC_FROM_OBJ( prec_p, OA_DEST(oap) );
			p1="result value if condition satisfied";
			p2="result value if condition not satisfied";
			p3="scalar value for comparison";
		}
		  else {
			assert( AERROR("unexpected 3 scalar function!?") );
		}
		SET_OA_SVAL(oap,0, alloc_sval(prec_p) );
		SET_OA_SVAL(oap,1, alloc_sval(prec_p) );
		SET_OA_SVAL(oap,2, alloc_sval(prec_p) );
		// BUG - could have an array of prec_p's ???
		cast_dbl_to_scalar_value(OA_SVAL(oap,0), prec_p, HOW_MUCH(p1) );
		cast_dbl_to_scalar_value(OA_SVAL(oap,1), prec_p, HOW_MUCH(p2) );
		cast_dbl_to_scalar_value(OA_SVAL(oap,2), prec_p, HOW_MUCH(p3) );
	} else if( VF_FLAGS(vfp) & SRC_SCALAR2 /* HAS_2_SCALARS */ ){
		const char *p1,*p2;

		// RAMP1D or ???
		//SET_PREC
		// dest can be null if arg error
		if( OA_DEST(oap) != NULL ){
			SET_PREC_FROM_OBJ( prec_p, OA_DEST(oap) );
		} else {
			prec_p=prec_for_code(PREC_BY);	// quiet compiler
			retval = -1;
		}
		
		if( COMPLEX_PRECISION(PREC_CODE(prec_p)) ){
			/* this should not happen!? */
			/* Does the function permit complex? */
			if( (VF_TYPEMASK(vfp) & (CPX_ARG_MASK|MIXED_ARG_MASK)) == 0 ){
				// BUG??? can OA_DEST be null here???
				// We used to print the name of the
				// destination obj here, but in case
				// it could be null...
				if( OA_DEST(oap) != NULL )
					sprintf(ERROR_STRING,
"get_scalar_args:  function %s does not permit operations with complex targets (%s)",
		VF_NAME(vfp),OBJ_NAME(OA_DEST(oap) ));
				else
					sprintf(ERROR_STRING,
"get_scalar_args:  function %s does not permit operations with complex targets",
						VF_NAME(vfp) );
				warn(ERROR_STRING);
				retval=(-1);
			}
		}

		if( QUAT_PRECISION(PREC_CODE(prec_p)) ){
			/* this should not happen!? */
			/* Does the function permit quaternions? */
			if( (VF_TYPEMASK(vfp) & (QUAT_ARG_MASK)) == 0 ){
				sprintf(ERROR_STRING,
	"get_scalar_args:  function %s does not permit operations with quaternion targets (%s)",
					VF_NAME(vfp),
					OA_DEST(oap) == NULL ?
					"(null destination)" :
					OBJ_NAME(OA_DEST(oap) ));
				warn(ERROR_STRING);
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
		else {
			assert( AERROR("unhandled case in get_scalar_args") );
		}
		SET_OA_SVAL(oap,0, alloc_sval(prec_p) );
		SET_OA_SVAL(oap,1, alloc_sval(prec_p) );
		if( retval == 0 ){
			cast_dbl_to_scalar_value(OA_SVAL(oap,0), prec_p, HOW_MUCH(p1) );
			cast_dbl_to_scalar_value(OA_SVAL(oap,1), prec_p, HOW_MUCH(p2) );
		}
	}

	else if( VF_FLAGS(vfp) & SRC_SCALAR1 ){
		if( VF_FLAGS(vfp) == VS_TEST ){	/* vsm_lt etc. */
			if( OA_SRC1(oap) == NULL ){
				goto get_dummy;
			}
			SET_PREC_FROM_OBJ( prec_p, OA_SRC1(oap) );
			SET_OA_SVAL(oap,0, alloc_sval( prec_p ) );
			cast_dbl_to_scalar_value(OA_SVAL(oap,0), prec_p,
				HOW_MUCH("source scalar value") );
		} else if( OA_DEST(oap) == NULL ){	/* error condition */
			/*double d;
			d=*/HOW_MUCH("dummy value");
			retval=(-1);
		} else if( IS_REAL(OA_DEST(oap) ) ){
			/* BUG we can't use destination for precision
			 * in the mixed precision ops...
			 * Well we can, but it breaks vsatan2 with cuda...
			 */
			if( VF_FLAGS(vfp) & SRC1_VEC ){
				if( OA_SRC1(oap) == NULL ){	// input error
					HOW_MUCH("dummy value");
					retval=(-1);
				} else {
					SET_MACH_PREC_FROM_OBJ( prec_p, OA_SRC1(oap) );
				}
			} else {
				// We assume OA_DEST can't be null, because it is real
				SET_MACH_PREC_FROM_OBJ( prec_p, OA_DEST(oap) );
			}
			if( retval >= 0 ){	// no error yet?
				SET_OA_SVAL(oap,0,alloc_sval( prec_p ) );
				cast_dbl_to_scalar_value(OA_SVAL(oap,0), prec_p,
					HOW_MUCH("source real scalar value") );
			}
		} else if( IS_COMPLEX(OA_DEST(oap)) ) {
			prec_p = OBJ_PREC_PTR(OA_DEST(oap));
			SET_OA_SVAL(oap,0, alloc_sval(prec_p) );
			cast_dbl_to_cpx_scalar(0,OA_SVAL(oap,0), prec_p,
				HOW_MUCH("source scalar value real part") );
			cast_dbl_to_cpx_scalar(1,OA_SVAL(oap,0), prec_p,
				HOW_MUCH("source scalar value imaginary part") );
		} else if( IS_QUAT(OA_DEST(oap)) ) {
			prec_p = OBJ_PREC_PTR(OA_DEST(oap));
			SET_OA_SVAL(oap,0, alloc_sval( prec_p ));
			cast_dbl_to_quat_scalar(0,OA_SVAL(oap,0),  prec_p,
				HOW_MUCH("source scalar value real part") );
			cast_dbl_to_quat_scalar(1,OA_SVAL(oap,0),  prec_p,
				HOW_MUCH("source scalar value i part") );
			cast_dbl_to_quat_scalar(2,OA_SVAL(oap,0),  prec_p,
				HOW_MUCH("source scalar value j part") );
			cast_dbl_to_quat_scalar(3,OA_SVAL(oap,0),  prec_p,
				HOW_MUCH("source scalar value k part") );
		} else {
			// what precision is this?
			/* use a single scalar for all components */
			SET_MACH_PREC_FROM_OBJ( prec_p, OA_DEST(oap) );	// why dest?
			SET_OA_SVAL(oap,0, alloc_sval( prec_p ) );
			cast_dbl_to_scalar_value(OA_SVAL(oap,0),  prec_p,
				HOW_MUCH("source scalar value") );
		}
		if( OA_SVAL(oap,0) == NULL ){
			retval=(-1);
		}
	}

	if( VF_FLAGS(vfp) & TWO_SCALAR_RESULTS ){
		Data_Obj *_dp1, *_dp2;
		if( OA_SRC1(oap) == NULL ){
			sprintf(ERROR_STRING,
	"get_scalar_args (%s):  no argument to use for precision prototype!?",
				VF_NAME(vfp));
			warn(ERROR_STRING);
			retval=(-1);
		} else {
			_dp1 = get_return_scalar(QSP_ARG
				"name of scalar for extreme value",
				OBJ_PREC_PTR( OA_SRC1(oap) ));
			_dp2 = get_return_scalar(QSP_ARG
				"name of scalar for # of occurrences",
				prec_for_code(PREC_DI));
			if( _dp1 == NULL || _dp2 == NULL )
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
	if( VF_CODE(vfp) == FVMOV && OA_SRC1(oap) != NULL ){
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
		if( get_scalar_args(QSP_ARG  oap, vfp) == (-1) ) oops|=1;
#ifdef PROBABLY_NOT_NEEDED
		/* Here we point the scalar value to the object data - why? */
		for(i=0;i<MAX_RETSCAL_ARGS;i++){
			if( OA_SCLR_OBJ(oap,i) != NULL )
				SET_OA_SVAL(oap,i, (Scalar_Value *)OBJ_DATA_PTR( OA_SCLR_OBJ(oap,i) ) );
		}
#endif /* PROBABLY_NOT_NEEDED */
	}

	if( oops) return(-1);		/* if a requested object doesn't exist... */

	/* get_scalar_args() gets (or makes) objects? */

	return(0);
} /* end get_args */

#ifdef FOOBAR
/* We have a problem introduced by trying to use the Data_Obj framework for images in nVidia CUDA:
 * We can't mix-and-match, one solution might be to have separate name spaces for gpu and ram objects.
 * For now, we just check that all objects are from the ram data area.
 */


static int check_one_obj_loc( Data_Obj *dp )
{
	if( dp == NULL ) return(0);
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

#define check_obj_device(dp,oap) _check_obj_device(QSP_ARG  dp,oap)

static int _check_obj_device(QSP_ARG_DECL  Data_Obj *dp, Vec_Obj_Args *oap)
{
	if( dp == NULL ) return 0;	// not an error
	if( OA_PFDEV(oap) == NULL ){
		SET_OA_PFDEV(oap,OBJ_PFDEV(dp));
if( OA_PFDEV(oap) == NULL ) warn("Null platform device!?");
		return 0;
	}
	if( OA_PFDEV(oap) != OBJ_PFDEV(dp) ){
		sprintf(ERROR_STRING,
"check_obj_device:  object %s device %s does not match expected device %s!?",
			OBJ_NAME(dp),
			PFDEV_NAME( OBJ_PFDEV(dp) ),
			PFDEV_NAME( OA_PFDEV(oap) ) );
		warn(ERROR_STRING);
		return -1;
	}
	return 0;
}

int _check_obj_devices(QSP_ARG_DECL  Vec_Obj_Args *oap )
{
	//int s, all_ram=1, all_gpu=1;
	int i;

	if( HAS_CHECKED_ARGS(oap) ){
		return 0;
	}

	if( check_obj_device(OA_DEST(oap),oap) < 0 ){
		warn("check_obj_devices:  bad destination!?");
		return -1;
	}

	for(i=0;i<MAX_N_ARGS;i++){
		if( check_obj_device(OA_SRC_OBJ(oap,i),oap) < 0 ){
			// BUG give a better error message
			warn("platform device mismatch!?");
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
	if( dp == NULL || OBJ_IS_RAM( dp ) ) return 0;
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
	if( dp == NULL ) return;
	sprintf(ERROR_STRING,"\t%s:\t%s",OBJ_NAME(dp),PFDEV_NAME( OBJ_PFDEV(dp) ) );
	advise(ERROR_STRING);
}

void mixed_location_error(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap)
{
	int i;

	sprintf(ERROR_STRING,"%s:  arguments must reside on a single device.",VF_NAME(vfp));
	warn(ERROR_STRING);

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
		warn(ERROR_STRING);
		return;
	}

if( OA_DEST(oap) == NULL ){
/* BUG?  is this really an error?  The bitmap destination might be here... */
sprintf(ERROR_STRING,"%s:  Null destination!?!?", VF_NAME(vfp));
warn(ERROR_STRING);
}

	call_vfunc(QSP_ARG  vfp,oap);

	/* Now free the scalars (if any) */

	/* Perhaps the code would be more efficient if
	 * oargs contained the scalar value struct
	 * itself instead of a pointer to dynamically
	 * allocated memory...  (is getbuf/malloc thread-safe?)
	 */
	for(i=0;i<MAX_SRCSCAL_ARGS;i++){
		if( OA_SVAL(oap,i) != NULL )
			rls_sval( OA_SVAL(oap,i) );	// free scalars allocated by alloc_sval()
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
	SET_OA_SRC3(oap,NULL);
	SET_OA_SRC4(oap,NULL);
	set_obj_arg_flags(oap);
}

void setvarg4(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *src1,Data_Obj *src2,Data_Obj *src3)
{
	clear_obj_args(oap);
	SET_OA_DEST(oap,dstv);
	SET_OA_SRC1(oap,src1);
	SET_OA_SRC2(oap,src2);
	SET_OA_SRC3(oap,src3);
	SET_OA_SRC4(oap,NULL);
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
	if( isp == NULL ){
		fprintf(stderr,"\t\t(null increment set)\n");
		return;
	}

	fprintf(stderr,"\t\t%d, %d, %d, %d, %d\n",
		isp->is_increment[4],
		isp->is_increment[3],
		isp->is_increment[2],
		isp->is_increment[1],
		isp->is_increment[0] );
}

static void show_dimensions(Dimension_Set *dsp)
{
	if( dsp == NULL ){
		fprintf(stderr,"\t\t(null dimension set)\n");
		return;
	}

	fprintf(stderr,"\t\t%d, %d, %d, %d, %d\n",
		dsp->ds_dimension[4],
		dsp->ds_dimension[3],
		dsp->ds_dimension[2],
		dsp->ds_dimension[1],
		dsp->ds_dimension[0] );
}

static void show_vector_arg(const char *name, const Vector_Arg *varg_p)
{
	fprintf(stderr,"\t%s = 0x%lx\n",name,(long)VARG_PTR(*varg_p));
	show_dimensions(VARG_DIMSET(*varg_p));
	show_increments(VARG_INCSET(*varg_p));
}

void show_vec_args(const Vector_Args *vap)
{
	int i;

	fprintf(stderr,"show_vec_args 0x%lx:\n", (long)vap);

	fprintf(stderr,"platform_device = 0x%lx\n",(long)VA_PFDEV(vap));
	fprintf(stderr,"platform_device = %s\n",PFDEV_NAME(VA_PFDEV(vap)));
	show_vector_arg( "dest", & VA_DEST(vap) );

	for(i=0;i<MAX_N_ARGS;i++){
		if( VA_SRC_PTR(vap,i) != NULL ){
			char name[8];
			sprintf(name,"src%d",i+1);
			show_vector_arg( name, & VA_SRC(vap,i) );
		}
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
		fprintf(stderr,"\tdst_inc = %d\n",VA_DEST_EQSP_INC( vap ) );
		fprintf(stderr,"\tsrc1_inc = %d\n",VA_SRC1_EQSP_INC( vap ) );
		fprintf(stderr,"\tsrc2_inc = %d\n",VA_SRC2_EQSP_INC( vap ) );
		fprintf(stderr,"\tsrc3_inc = %d\n",VA_SRC3_EQSP_INC( vap ) );
		fprintf(stderr,"\tsrc4_inc = %d\n",VA_SRC4_EQSP_INC( vap ) );
		fprintf(stderr,"\tsrc5_inc = %d\n",VA_SRC5_EQSP_INC( vap ) );
	}
#ifdef FOOBAR
#ifdef HAVE_ANY_GPU
	fprintf(stderr,"iteration sizes:  %d %d %d %d %d\n",
		vap->va_iteration_size.ds_dimension[0],
		vap->va_iteration_size.ds_dimension[1],
		vap->va_iteration_size.ds_dimension[2],
		vap->va_iteration_size.ds_dimension[3],
		vap->va_iteration_size.ds_dimension[4]);
	fprintf(stderr,"iteration total:  %d\n",
		vap->va_iteration_size.ds_n_elts);
#endif // HAVE_ANY_GPU
#endif // FOOBAR
	
}

#ifdef PAD_MINDIM
static dimension_t slow_bitmap_word_count( Dimension_Set *dsp, Increment_Set *isp, bitnum_t bit0 )
{
	dimension_t bits_per_row, words_per_row;
	dimension_t n;

	assert( isp != NULL );
// BUG llu format assumes bit_count_t is 64 bits
fprintf(stderr,"varg_n_bitmap_bits (1):  dim[0] = %d, dim[1] = %d, inc[1] = %d, offset = %llu\n",DS_DIM(dsp,0),
DS_DIM(dsp,1),INCREMENT(isp,1),bit0);

	bits_per_row = 1 + (DS_DIM(dsp,0) * DS_DIM(dsp,1) - 1 ) * INCREMENT(isp,1);

	// The number of words per row may need to be incremented, depending on the value of bit0
	bits_per_row += bit0 % BITS_PER_BITMAP_WORD;

	words_per_row = N_BITMAP_WORDS(bits_per_row);
	n = words_per_row * DS_DIM(dsp,2) * DS_DIM(dsp,3) * DS_DIM(dsp,4) ;
	return n;
}

static dimension_t eqsp_bitmap_word_count( Dimension_Set *dsp, incr_t eqsp_incr, bit_count_t bit0 )
{
	dimension_t bits_per_row, words_per_row;
	dimension_t n;

	assert(eqsp_incr>0);

	bits_per_row = 1 + (DS_DIM(dsp,0) * DS_DIM(dsp,1) - 1 ) * eqsp_incr;
	bits_per_row += bit0 % BITS_PER_BITMAP_WORD;

	words_per_row = N_BITMAP_WORDS(bits_per_row);
	n = words_per_row * DS_DIM(dsp,2) * DS_DIM(dsp,3) * DS_DIM(dsp,4) ;
	return n;
}

static dimension_t fast_bitmap_word_count( Dimension_Set *dsp, bit_count_t offset )
{
	dimension_t n;
	n = N_BITMAP_WORDS( DS_N_ELTS(dsp) + (offset%BITS_PER_BITMAP_WORD) );
	return n;
}
#endif // ! PAD_MINDIM

#ifdef FOOBAR
// For bitmaps with an increment other than 1, we have to include the gaps within the rows
// to determine the proper number of words

dimension_t varg_bitmap_word_count(const Vector_Arg *varg_p)
{
#ifdef PAD_MINDIM
	if(  VARG_INCSET(*varg_p) != NULL )
		return slow_bitmap_word_count(VARG_DIMSET(*varg_p),VARG_INCSET(*varg_p),VARG_OFFSET(*varg_p));
	else if ( VARG_EQSP_INC(*varg_p) > 0 )
		return eqsp_bitmap_word_count(VARG_DIMSET(*varg_p),VARG_EQSP_INC(*varg_p),VARG_OFFSET(*varg_p));
	else
		return fast_bitmap_word_count(VARG_DIMSET(*varg_p),VARG_OFFSET(*varg_p));
#else // ! PAD_MINDIM
	// This can be very tricky because the word boundaries can fall anywhere...
	// One case is where the gaps are in every case less than the word size, in that
	// case all of the words are used.
	//
	// We have the offset (bit0), and dimensions and increments
	dimension_t bits_per_dim, n_words;
	dimension_t n;

	assert( isp != NULL );

	for(i=0;i<N_DIMENSIONS;i++){
		dimension_t gap;
		gap = INCREMENT(isp,i);
	bits_per_row = 1 + (DS_DIM(dsp,0) * DS_DIM(dsp,1) - 1 ) * INCREMENT(isp,1);

	// The number of words per row may need to be incremented, depending on the value of bit0
	bits_per_row += bit0 % BITS_PER_BITMAP_WORD;

	words_per_row = N_BITMAP_WORDS(bits_per_row);
	n = words_per_row * DS_DIM(dsp,2) * DS_DIM(dsp,3) * DS_DIM(dsp,4) ;
	return n;

#endif // ! PAD_MINDIM
}
#endif // FOOBAR

static void traverse_bitmap(Data_Obj *dp, dimension_t word_idx_arg, bitnum_t (*func)(Data_Obj *dp, bitnum_t bit_num, dimension_t word_idx) )
{
	dimension_t i,j,k,l,m;
	bitnum_t bit_number;	// even though dimension_t is 32 bits, the bit number can have 6 more bits
	bitnum_t seq_base,frame_base, row_base, col_base;

	seq_base = OBJ_BIT0(dp);
	for(i=0;i<OBJ_SEQS(dp);i++){
		frame_base = seq_base;
		for(j=0;j<OBJ_FRAMES(dp);j++){
			row_base = frame_base;
			for(k=0;k<OBJ_ROWS(dp);k++){
				col_base = row_base;
				for(l=0;l<OBJ_COLS(dp);l++){
					bit_number = col_base;
					for(m=0;m<OBJ_COMPS(dp);m++){
						word_idx_arg = (*func)(dp,bit_number,word_idx_arg);
						bit_number += OBJ_COMP_INC(dp);
					}
					col_base += OBJ_PXL_INC(dp);
				}
				row_base += OBJ_ROW_INC(dp);
			}
			frame_base += OBJ_FRM_INC(dp);
		}
		seq_base += OBJ_SEQ_INC(dp);
	}
}

// Given how simple this is, we probably should use a macro.
// Previously, we constrained all dimension boundaries to align with word boundaries,
// in that case this needs to be more comples.  This needs to be studies to determine
// what results in the highest-performing GPU implementation...

static bitnum_t word_for_bit( bitnum_t bit_number )
{
	// We don't need to do anything more complicated, because we count all the bits (even the unused ones),
	// letting the increments take care of things correctly...

	return bit_number / BITS_PER_BITMAP_WORD;
}

// We call this to determine how many words we need to use

static dimension_t count_bitmap_word(Data_Obj *dp, bitnum_t bit_number, dimension_t last_word_idx)
{
	Bitmap_GPU_Info *bmi_p;
	bitnum_t new_word_idx;

	bmi_p = BITMAP_OBJ_GPU_INFO_HOST_PTR(dp);

	new_word_idx = word_for_bit(bit_number);
	if( new_word_idx != last_word_idx ){
		SET_BMI_N_WORDS(bmi_p,BMI_N_WORDS(bmi_p)+1);	// increment index
	}
	return new_word_idx;
}

bitnum_t bitmap_obj_word_count(Data_Obj *dp)
{
	dimension_t n_words;
	Bitmap_GPU_Info *bmi_p;

	// New method, word boundaries not aligned with dimensions boundaries

	if( BITMAP_OBJ_GPU_INFO_HOST_PTR(dp) != NULL ){
		return BMI_N_WORDS( BITMAP_OBJ_GPU_INFO_HOST_PTR(dp) );
	}

	// No info, we have to count

	// First, make a temporary struct
	bmi_p = getbuf( sizeof(*bmi_p) );
	SET_BITMAP_OBJ_GPU_INFO_HOST_PTR(dp,bmi_p);

	SET_BMI_N_WORDS(bmi_p,0);
	SET_BMI_STRUCT_SIZE( bmi_p, sizeof(*bmi_p) );

	SET_BMI_N_WORDS( bmi_p, 0 );

	traverse_bitmap(dp,INVALID_WORD_IDX,count_bitmap_word);
	n_words = BMI_N_WORDS( BITMAP_OBJ_GPU_INFO_HOST_PTR(dp) );

	SET_BITMAP_OBJ_GPU_INFO_HOST_PTR(dp,NULL);
	givbuf(bmi_p);

	return n_words;
}

//	return bitmap_word_count(VARG_DIMSET(*varg_p),VARG_INCSET(*varg_p),VARG_EQSP_INC(*varg_p),);

#ifdef HAVE_ANY_GPU

#ifdef JUST_FOR_DEBUGGING
// This function is just for testing, to make sure that we got the correct data onto the device
static void verify_gpu_bitmap_info(Data_Obj *dp)
{
	Bitmap_GPU_Info * bmi_p;
	dimension_t n_words_expected;
	int i;

	n_words_expected = bitmap_obj_word_count(dp);
	bmi_p = getbuf( BITMAP_GPU_INFO_SIZE(n_words_expected));

	(*PF_MEM_DNLOAD_FN( OBJ_PLATFORM(dp) ))(DEFAULT_QSP_ARG  bmi_p, BITMAP_OBJ_GPU_INFO_DEV_PTR(dp), BITMAP_GPU_INFO_SIZE(n_words_expected), 0 /* offset */, OBJ_PFDEV(dp) );
	for(i=0;i<n_words_expected;i++){
		// definitions of PRIx64 etc in inttypes.h
		fprintf(stderr,"word %d   offset %d   indices %d %d %d %d %d   valid_bits 0x%"PRIx64"\n",
			i,bmi_p->word_tbl[i].word_offset,
			bmi_p->word_tbl[i].first_index[0],
			bmi_p->word_tbl[i].first_index[1],
			bmi_p->word_tbl[i].first_index[2],
			bmi_p->word_tbl[i].first_index[3],
			bmi_p->word_tbl[i].first_index[4],
			bmi_p->word_tbl[i].valid_bits
			);
	}
	givbuf(bmi_p);
}
#endif // JUST_FOR_DEBUGGING

static void get_indices_for_bit(dimension_t idx_tbl[N_DIMENSIONS], Data_Obj *dp, bitnum_t bit_number)
{
	dimension_t denom=1;
	int i;

	for(i=0;i<N_DIMENSIONS;i++){
		idx_tbl[i] = (bit_number/denom) % OBJ_TYPE_DIM(dp,i);
		denom *= OBJ_TYPE_DIM(dp,i);
	}

//fprintf(stderr,"get_indices_for bit %llu:  %d %d %d %d %d\n",
//bit_number,idx_tbl[0], idx_tbl[1], idx_tbl[2], idx_tbl[3], idx_tbl[4]);

//	idx_tbl[0] = bit_number % OBJ_COMPS(dp);
//	idx_tbl[1] = (bit_number / OBJ_COMPS(dp)) % OBJ_COLS(dp);
}

static void init_bitmap_word_info(Bitmap_GPU_Word_Info *bmwi_p, dimension_t this_word_idx, bitnum_t bit_number, Data_Obj *dp)
{
	dimension_t idx_tbl[N_DIMENSIONS];
	int i;

	SET_BMWI_OFFSET(bmwi_p,this_word_idx);
	SET_BMWI_FIRST_BIT_NUM(bmwi_p,bit_number);
	get_indices_for_bit(idx_tbl,dp,bit_number);
	for(i=0;i<N_DIMENSIONS;i++){
		SET_BMWI_FIRST_INDEX(bmwi_p,i,idx_tbl[i]);
	}
	SET_BMWI_VALID_BITS(bmwi_p,0);	// to be safe - but where do we set for real?
}

static dimension_t tabulate_bitmap_word(Data_Obj *dp, bitnum_t bit_number, dimension_t word_tbl_idx)
{
	Bitmap_GPU_Word_Info *bmwi_p;
	Bitmap_GPU_Info *bmi_p;
	dimension_t this_word_idx;

	bmi_p = BITMAP_OBJ_GPU_INFO_HOST_PTR(dp);

	this_word_idx = word_for_bit(bit_number);

	if( word_tbl_idx == INVALID_WORD_TBL_IDX ){	// first time
		word_tbl_idx = 0;
		bmwi_p = BMI_WORD_INFO_P(bmi_p,word_tbl_idx);
		init_bitmap_word_info(bmwi_p,this_word_idx,bit_number,dp);
	} else {
		bmwi_p = BMI_WORD_INFO_P(bmi_p,word_tbl_idx);
	}

	if( this_word_idx != BMWI_OFFSET(bmwi_p) ){
		word_tbl_idx ++;
		assert( word_tbl_idx < BMI_N_WORDS(bmi_p) );
		bmwi_p = BMI_WORD_INFO_P(bmi_p,word_tbl_idx);
		init_bitmap_word_info(bmwi_p,this_word_idx,bit_number,dp);
	}
	SET_BMWI_VALID_BIT(bmwi_p,1L<<(bit_number % BITS_PER_BITMAP_WORD));

	return word_tbl_idx;
}

#ifdef JUST_FOR_DEBUGGING
// This functions may be useful for debugging...

#define show_bitmap_word_info(bmwi_p, tbl_idx) _show_bitmap_word_info(QSP_ARG  bmwi_p, tbl_idx)

static void _show_bitmap_word_info(QSP_ARG_DECL  Bitmap_GPU_Word_Info *bmwi_p, int tbl_idx)
{
	// BUG - get correct format!
	sprintf(MSG_STR,"word %3d   offset %d   first bit %"PRId64"  seq %4d  frame %4d   row %4d   col %4d   comp %4d   mask = 0x%"PRIx64,
		tbl_idx,BMWI_OFFSET(bmwi_p),
		BMWI_FIRST_BIT_NUM(bmwi_p),
		BMWI_FIRST_INDEX(bmwi_p,4),
		BMWI_FIRST_INDEX(bmwi_p,3),
		BMWI_FIRST_INDEX(bmwi_p,2),
		BMWI_FIRST_INDEX(bmwi_p,1),
		BMWI_FIRST_INDEX(bmwi_p,0),
		BMWI_VALID_BITS(bmwi_p)
		);
	prt_msg(MSG_STR);
}

static void show_bitmap_gpu_info(QSP_ARG_DECL  Bitmap_GPU_Info *bmi_p)
{
	dimension_t i;
	Bitmap_GPU_Word_Info * bmwi_p;

	sprintf(MSG_STR,"Bitmap_GPU_Info at 0x%lx",(long)bmi_p);
	prt_msg(MSG_STR);

	sprintf(MSG_STR,"\t%d total bytes",BMI_STRUCT_SIZE(bmi_p));
	prt_msg(MSG_STR);

	sprintf(MSG_STR,"\tobj dimensions:  %d  %d  %d  %d  %d",
		BMI_DIMENSION(bmi_p,4),
		BMI_DIMENSION(bmi_p,3),
		BMI_DIMENSION(bmi_p,2),
		BMI_DIMENSION(bmi_p,1),
		BMI_DIMENSION(bmi_p,0) );
	prt_msg(MSG_STR);

	sprintf(MSG_STR,"\t%d words:",BMI_N_WORDS(bmi_p));
	prt_msg(MSG_STR);

	for(i=0;i<BMI_N_WORDS(bmi_p);i++){
		bmwi_p = BMI_WORD_INFO_P(bmi_p,i);
		show_bitmap_word_info(bmwi_p,i);
	}
}

#endif // JUST_FOR_DEBUGGING

static Data_Obj *oldest_ancestor(Data_Obj *dp)
{
	if( OBJ_PARENT(dp) == NULL ) return dp;
	return oldest_ancestor( OBJ_PARENT(dp) );
}

// Populate the structure used to help gpu kernel threads know which bits to twiddle
//
// Basically, we have a copy of the bitmap, with 1's in bit positions where there is valid
// data, and 0's elsewhere.  For each word, we store the indices.  The strategy for populating
// is to execute the "slow" loops over all dimensions, keeping track of the bit number in the parent
// object.

void init_bitmap_gpu_info(Data_Obj *dp)
{
	dimension_t n_words_expected;
	Bitmap_GPU_Info *bmi_p;
	void *ptr;
	int tot_siz;
	Data_Obj *ancestor;	// oldest ancestor, for underlying dimensions
	int i;

fprintf(stderr,"init_bitmap_gpu_info BEGIN\n");
	n_words_expected = bitmap_obj_word_count(dp);
	tot_siz = BITMAP_GPU_INFO_SIZE(n_words_expected);
	bmi_p = getbuf(tot_siz);
fprintf(stderr,"init_bitmap_gpu_info allocated host copy at 0x%lx\n",(long)bmi_p);
	SET_BITMAP_OBJ_GPU_INFO_HOST_PTR(dp,bmi_p);

	SET_BMI_N_WORDS(bmi_p,n_words_expected);
	SET_BMI_STRUCT_SIZE(bmi_p,tot_siz);

	ancestor = oldest_ancestor(dp);
	for(i=0;i<N_DIMENSIONS;i++){
		SET_BMI_DIMENSION(bmi_p,i,OBJ_DIMENSION(ancestor,i));
	}

	// BUG? should zero the block just to be safe?

	// initialize the word index from bit0
	//SET_BMI_NEXT_WORD_IDX( bmi_p, 0 );
	//SET_BMI_LAST_WORD_IDX( bmi_p, -1 );
	traverse_bitmap(dp,INVALID_WORD_TBL_IDX,tabulate_bitmap_word);

#ifdef JUST_FOR_DEBUGGING
show_bitmap_gpu_info(DEFAULT_QSP_ARG  BITMAP_OBJ_GPU_INFO_HOST_PTR(dp) );
#endif // JUST_FOR_DEBUGGING

	// Now the struct has been completed - we need to allocate some memory on the gpu, and copy the data,
	// so that it will be available for use by kernels

	ptr = (*PF_MEM_ALLOC_FN( OBJ_PLATFORM(dp) ))(DEFAULT_QSP_ARG  OBJ_PFDEV(dp), BMI_STRUCT_SIZE(bmi_p), 0 );
	assert( ptr != NULL );
	SET_BITMAP_OBJ_GPU_INFO_DEV_PTR(dp,ptr);
fprintf(stderr,"allocated platform mem at 0x%lx\n",(long)ptr);

	// now copy to device
	(*PF_MEM_UPLOAD_FN( OBJ_PLATFORM(dp) ))(DEFAULT_QSP_ARG
		ptr, bmi_p, BMI_STRUCT_SIZE(bmi_p), 0, OBJ_PFDEV(dp) );

#ifdef JUST_FOR_DEBUGGING
verify_gpu_bitmap_info(dp);
#endif // JUST_FOR_DEBUGGING
} // init_bitmap_gpu_info

#endif // HAVE_ANY_GPU

