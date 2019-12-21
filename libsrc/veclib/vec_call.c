#include "quip_config.h"

#include <string.h>
#include "quip_prot.h"
#include "nvf.h"
#include "debug.h"
#include "platform.h"

/* BUG - this is only correct when the order of words in this table corresponds
 * to the ordering of the corresponding constants.  Better to have a software initialization
 * of the table.
 */

const char *argset_type_name[N_ARGSET_TYPES]={
	"unknown",
	"real",
	"complex",
	"mixed (complex/real)",
	"quaternion",
	"mixed (quaternion/real)"
};

#define shape_error(vfp, dp) _shape_error(QSP_ARG  vfp, dp)

static void _shape_error(QSP_ARG_DECL  Vector_Function *vfp, Data_Obj *dp)
{
	sprintf(ERROR_STRING,"shape_error:  Vector function %s:  argument %s has unknown shape!?",
		VF_NAME(vfp),OBJ_NAME(dp));
	warn(ERROR_STRING);
}


#define chk_uk(vfp, oap) _chk_uk(QSP_ARG  vfp, oap)

static int _chk_uk(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap)
{
	int i;

	if( OA_DEST(oap)  != NULL && UNKNOWN_OBJ_SHAPE(OA_DEST(oap)) ){
		shape_error(vfp,OA_DEST(oap) );
		return -1;
	}
	for(i=0;i<MAX_N_ARGS;i++){
		if( OA_SRC_OBJ(oap,i) != NULL && UNKNOWN_OBJ_SHAPE( OA_SRC_OBJ(oap,i) ) ){
			shape_error(vfp,OA_SRC_OBJ(oap,i));
			return -1;
		}
	}
	if( OA_SBM(oap) != NULL && UNKNOWN_OBJ_SHAPE(OA_SBM(oap)) ){
		shape_error(vfp,OA_SBM(oap) );
		return -1;
	}
	/* BUG check the scalar objects too? */
	return 0;
}

static const char *name_for_type(Data_Obj *dp)
{
	if( IS_REAL(dp) ) return("real");
	else if( IS_COMPLEX(dp) ) return("complex");
	else if( IS_QUAT(dp) ) return("quaternion");
	else {
		assert( AERROR("name_for_type:  unexpected type code!?") );
	}
}

#define IS_FWD_FFT(vfp)		(VF_CODE(vfp)==FVFFT || \
				 VF_CODE(vfp)==FVFFT2D || \
				 VF_CODE(vfp)==FVFFTROWS )

#define chktyp_fft(vfp,oap) _chktyp_fft(QSP_ARG  vfp,oap)

static int _chktyp_fft(QSP_ARG_DECL  Vector_Function *vfp,Vec_Obj_Args *oap)
{
	assert( OA_SRC1(oap) != NULL );

	if( IS_FWD_FFT(vfp) ){
		/* source vector can be real or complex */
		if( !IS_COMPLEX(OA_DEST(oap) ) ){
			warn("chktyp_fft:  destination must be complex for fft");
			return -1;
		}

		if( IS_COMPLEX( OA_SRC1(oap) ) )
			SET_OA_ARGSTYPE(oap,COMPLEX_ARGS);
		else if( IS_QUAT( OA_SRC1(oap) ) ){
			warn("chktyp:  Can't compute FFT of a quaternion input");
			return -1;
		} else
			SET_OA_ARGSTYPE(oap,REAL_ARGS);
	} else {	// inverse fft
		/* destination vector can be real or complex */
		if( !IS_COMPLEX( OA_SRC1(oap) ) ){
			warn("chktyp:  source must be complex for inverse fft");
			return -1;
		}
		if( IS_COMPLEX(OA_DEST(oap) ) )
			SET_OA_ARGSTYPE(oap,COMPLEX_ARGS);
		else if( IS_QUAT(OA_DEST(oap) ) ){
			warn("chktyp:  Can't compute inverse FFT to a quaternion target");
			return -1;
		} else
			SET_OA_ARGSTYPE(oap,REAL_ARGS);
	}
	return 0;
}

#define IS_FFT_FUNC(vfp)	(VF_CODE(vfp)==FVFFT || VF_CODE(vfp)==FVIFT || \
				 VF_CODE(vfp)==FVFFT2D || VF_CODE(vfp)==FVIFT2D || \
				 VF_CODE(vfp)==FVFFTROWS || VF_CODE(vfp)==FVIFTROWS )

/* The "type" is real, complex, quaternion, or mixed...
 * independent of "precision" (byte/short/float etc)
 */

#define chktyp(vfp,oap) _chktyp(QSP_ARG  vfp,oap)

static int _chktyp(QSP_ARG_DECL  Vector_Function *vfp,Vec_Obj_Args *oap)
{
	SET_OA_ARGSTYPE(oap, UNKNOWN_ARGS);

	if( IS_FFT_FUNC(vfp) ){
		return chktyp_fft(vfp, oap);
	}

	/* Set the type based on the destination vector */
	/* destv is highest numbered arg */
	if( IS_REAL(OA_DEST(oap) ) ){
		if( OA_SRC2(oap)  != NULL ){	/* two source operands */
			if( IS_REAL( OA_SRC1(oap) ) ){
				if( IS_REAL(OA_SRC2(oap) ) ){
					SET_OA_ARGSTYPE(oap, REAL_ARGS);
				} else if( IS_COMPLEX(OA_SRC2(oap) ) ){
					SET_OA_ARGSTYPE(oap, MIXED_ARGS);
				} else if( IS_QUAT(OA_SRC2(oap) ) ){
					SET_OA_ARGSTYPE(oap, QMIXED_ARGS);
				} else {
					/* error - type mismatch */
					goto type_mismatch23;
				}
			} else if( IS_COMPLEX( OA_SRC1(oap) ) ){
				if( IS_REAL(OA_SRC2(oap) ) ){
					SET_OA_ARGSTYPE(oap, MIXED_ARGS);
				} else if( IS_COMPLEX(OA_SRC2(oap) ) ){
					SET_OA_ARGSTYPE(oap, MIXED_ARGS);
				} else {
					/* error - type mismatch */
					goto type_mismatch23;
				}
			} else if( IS_QUAT( OA_SRC1(oap) ) ){
				if( IS_REAL(OA_SRC2(oap) ) ){
					SET_OA_ARGSTYPE(oap, QMIXED_ARGS);
				} else if( IS_QUAT(OA_SRC2(oap) ) ){
					SET_OA_ARGSTYPE(oap, QMIXED_ARGS);
				} else {
					/* error - type mismatch */
					goto type_mismatch23;
				}
			}
			// Why was this CAUTIOUS when other goto's to type_mismatch13 are not???
			  else {
				/* OA_SRC1 is not real or complex, must be a type mismatch */
				goto type_mismatch13;
			}
		} else if(  OA_SRC1(oap)  != NULL ){	/* one source operand */
			if( IS_REAL( OA_SRC1(oap) ) ){
				SET_OA_ARGSTYPE(oap, REAL_ARGS);
			} else if( IS_COMPLEX( OA_SRC1(oap) ) ){
				SET_OA_ARGSTYPE(oap, MIXED_ARGS);
			} else if( IS_QUAT( OA_SRC1(oap) ) ){
				SET_OA_ARGSTYPE(oap, QMIXED_ARGS);
			} else {
				/* OA_SRC1 is not real or complex, must be a type mismatch */
				goto type_mismatch01;
			}
		} else {				/* only 1 operand */
			SET_OA_ARGSTYPE(oap, REAL_ARGS);
		}
	} else if( IS_COMPLEX(OA_DEST(oap) ) ){
		if( OA_SRC2(oap)  != NULL ){	/* two source operands */
			if( IS_COMPLEX( OA_SRC1(oap) ) ){
				if( IS_COMPLEX( OA_SRC2(oap) ) ){
					SET_OA_ARGSTYPE(oap, COMPLEX_ARGS);
				} else if( IS_REAL( OA_SRC2(oap) ) ){
					SET_OA_ARGSTYPE(oap, MIXED_ARGS);
				} else {
					/* error - type mismatch */
					goto type_mismatch23;
				}
			} else if( IS_REAL( OA_SRC1(oap) ) ){
				if( IS_COMPLEX( OA_SRC2(oap) ) ){
					SET_OA_ARGSTYPE(oap, MIXED_ARGS);
				/* Should we check for real-real with complex result??? */
				} else {
					/* error - type mismatch */
					goto type_mismatch23;
				}
			} else {
				/* OA_SRC1 is not real or complex, must be a type mismatch */
				goto type_mismatch13;
			}
		} else if( OA_SRC1(oap) != NULL ){	/* one source operand */
			if( IS_COMPLEX( OA_SRC1(oap) ) ){
				SET_OA_ARGSTYPE(oap, COMPLEX_ARGS);
			} else if( IS_REAL( OA_SRC1(oap) ) ){
				// This may be correct for vsadd, etc.
				// but NOT for fft!?
				if( vfp )
				SET_OA_ARGSTYPE(oap, MIXED_ARGS);
			} else {
				/* OA_SRC1 is not real or complex, must be a type mismatch */
				goto type_mismatch01;
			}
		} else {				/* only 1 operand */
			SET_OA_ARGSTYPE(oap, COMPLEX_ARGS);
		}
	} else if( IS_QUAT(OA_DEST(oap) ) ){
		if( OA_SRC2(oap)  != NULL ){	/* two source operands */
			if( IS_QUAT( OA_SRC1(oap) ) ){
				if( IS_QUAT( OA_SRC2(oap) ) ){
					SET_OA_ARGSTYPE(oap, QUATERNION_ARGS);
				} else if( IS_REAL( OA_SRC2(oap) ) ){
					SET_OA_ARGSTYPE(oap, QMIXED_ARGS);
				} else {
					/* error - type mismatch */
					goto type_mismatch23;
				}
			} else if( IS_REAL( OA_SRC1(oap) ) ){
				if( IS_QUAT( OA_SRC2(oap) ) ){
					SET_OA_ARGSTYPE(oap, QMIXED_ARGS);
				/* Should we check for real-real with complex result??? */
				} else {
					/* error - type mismatch */
					goto type_mismatch23;
				}
			} else {
				/* OA_SRC1 is not real or complex, must be a type mismatch */
				goto type_mismatch13;
			}
		} else if(  OA_SRC1(oap)  != NULL ){	/* one source operand */
			if( IS_QUAT( OA_SRC1(oap) ) ){
				SET_OA_ARGSTYPE(oap, QUATERNION_ARGS);
			} else if( IS_REAL( OA_SRC1(oap) ) ){
				SET_OA_ARGSTYPE(oap, QMIXED_ARGS);
			} else {
				/* OA_SRC1 is not real or complex, must be a type mismatch */
				goto type_mismatch01;
			}
		} else {				/* only 1 operand */
			SET_OA_ARGSTYPE(oap, QUATERNION_ARGS);
		}
	} else {
		sprintf(ERROR_STRING,"chktyp:  can't categorize destination object %s!?",OBJ_NAME(OA_DEST(oap) ) );
		warn(ERROR_STRING);
	}

	/* now the type field has been set - make sure it's legal */
	/* But first check a couple of special cases */

	/* make sure that function doesn't require mixed types */

	if( VF_FLAGS(vfp) & CPX_2_REAL ){
		// For inverse fourier transform, the destination can be real
		// but does not have to be!
		assert( OA_SRC1(oap) != NULL );

		if( ! IS_COMPLEX( OA_SRC1(oap) ) ){
			sprintf(ERROR_STRING,"chktyp:  source vector %s (%s) must be complex with function %s",
				OBJ_NAME( OA_SRC1(oap) ) ,OBJ_PREC_NAME( OA_DEST(oap) ),VF_NAME(vfp) );
			warn(ERROR_STRING);
			list_dobj(OA_SRC1(oap) );
			return -1;
		}
		if( (VF_FLAGS(vfp) & INV_FT)==0 ){
			if( ! IS_REAL(OA_DEST(oap)) ){
				sprintf(ERROR_STRING,"chktyp:  destination vector %s (%s) must be real with function %s",
					OBJ_NAME(OA_DEST(oap) ) ,OBJ_PREC_NAME( OA_DEST(oap) ),VF_NAME(vfp) );
				warn(ERROR_STRING);
				list_dobj(OA_DEST(oap) );
				return -1;
			}
			SET_OA_ARGSTYPE(oap, REAL_ARGS);
		} else {	// inverse Fourier transform
			if( IS_REAL(OA_DEST(oap)) ){
				SET_OA_ARGSTYPE(oap, REAL_ARGS);
			} else {
				SET_OA_ARGSTYPE(oap, COMPLEX_ARGS);
			}
		}
		return 0;
	}


	/* now the type field has been set - make sure it's legal */
	assert( OA_ARGSTYPE(oap) != UNKNOWN_ARGS );
	if( (VF_TYPEMASK(vfp) & VL_TYPE_MASK(OA_ARGSTYPE(oap) ) )==0 ){
		sprintf(ERROR_STRING,
	"chktyp:  Arguments of type %s are not permitted with function %s",
			argset_type_name[OA_ARGSTYPE(oap) ],VF_NAME(vfp) );
		warn(ERROR_STRING);
		return -1;
	}

/*
sprintf(ERROR_STRING,"function %s:  oa_argstype = %s",VF_NAME(vfp) ,argset_type_name[OA_ARGSTYPE(oap) ]);
advise(ERROR_STRING);
*/

	/* if we get to here then it wasn't a special function */
	/* should be a function which allows same type or mixed */

	/* the first and second stuff is reversed
	 * because the interpreter prompts for them
	 * in the opposite order that they're passed
	 */

	/* most legal mixed functions have one real and one complex operand, and
	 * a complex target...  But with the addition of quaternions, we also
	 * can have quaternions mixed with real...
	 */

	if( HAS_MIXED_ARGS(oap) && ! IS_COMPLEX(OA_DEST(oap)) ){
		sprintf(ERROR_STRING,"chktyp:  destination vector %s must be complex when mixing types with function %s",
			OBJ_NAME(OA_DEST(oap) ) ,VF_NAME(vfp) );
		warn(ERROR_STRING);
		return -1;
	}

	if( HAS_QMIXED_ARGS(oap)  && ! IS_QUAT(OA_DEST(oap) ) ){
		sprintf(ERROR_STRING,"chktyp:  destination vector %s must be quaternion when mixing types with function %s",
			OBJ_NAME(OA_DEST(oap) ) ,VF_NAME(vfp) );
		warn(ERROR_STRING);
		return -1;
	}

#define USES_REAL_SCALAR(code)					\
								\
	( code == FVSMUL || code == FVSADD ||			\
	  code == FVSDIV || code == FVSSUB )

#define USES_COMPLEX_SCALAR(code)				\
								\
	( code == FVCSMUL || code == FVCSADD ||			\
	  code == FVCSDIV || code == FVCSSUB )

#define USES_QUAT_SCALAR(code)					\
								\
	( code == FVQSMUL || code == FVQSADD ||			\
	  code == FVQSDIV || code == FVQSSUB )


	/* BUG for things like vmul, it would be nice if the code
	 * could just figure out which is real and which is complex
	 * and swap around accordingly!
	 */
	if( HAS_MIXED_ARGS(oap) ){
	
		/*
		show_obj_args(oap);
sprintf(ERROR_STRING,"Function %s.",VF_NAME(vfp) );
advise(ERROR_STRING);
		error1("chktyp:  Sorry, not sure how to deal with this situation...");
		*/

		if( ! IS_COMPLEX( OA_SRC1(oap) ) ){
			sprintf(ERROR_STRING,
"chktyp:  first source vector (%s,%s) must be complex when mixing types with function %s",
				OBJ_NAME( OA_SRC1(oap) ) ,
				name_for_type( OA_SRC1(oap) ),
				VF_NAME(vfp) );
			warn(ERROR_STRING);
			return -1;
		}
		// Mixed-arg fuctions have to have two sources, but the analyzer
		// doesn't know that...
		assert( OA_SRC2(oap) != NULL );

		if( ! IS_REAL(OA_SRC2(oap) ) ){
			sprintf(ERROR_STRING,
"second source vector (%s,%s) must be real when mixing types with function %s",
				OBJ_NAME(OA_SRC2(oap) ) ,
				name_for_type(OA_SRC2(oap) ),
				VF_NAME(vfp) );
			warn(ERROR_STRING);
			return -1;
		}
		/* Should the destination be complex??? */
	} else if( HAS_QMIXED_ARGS(oap) ){
		error1("FIXME:  need to add code for quaternions, vec_call.c");
		/* BUG add the same check as above for quaternions */
		return -1;
	}

	return 0;

type_mismatch13:
	sprintf(ERROR_STRING,"Type mismatch between objects %s (%s) and %s (%s), function %s",
		OBJ_NAME( OA_SRC1(oap) ) ,name_for_type( OA_SRC1(oap) ),
		OBJ_NAME( OA_SRC3(oap) ) ,name_for_type( OA_SRC3(oap) ),
		VF_NAME(vfp) );
	warn(ERROR_STRING);
	return -1;
    
type_mismatch01:
	sprintf(ERROR_STRING,"Type mismatch between objects %s (%s) and %s (%s), function %s",
            OBJ_NAME( OA_SRC1(oap) ) ,name_for_type( OA_SRC1(oap) ),
            OBJ_NAME(OA_DEST(oap) ) ,name_for_type(OA_DEST(oap) ),
            VF_NAME(vfp) );
	warn(ERROR_STRING);
	return -1;
    
    
type_mismatch23:
	sprintf(ERROR_STRING,"Type mismatch between objects %s (%s) and %s (%s), function %s",
		OBJ_NAME(OA_SRC2(oap) ) ,name_for_type(OA_SRC2(oap) ),
		OBJ_NAME( OA_SRC3(oap) ) ,name_for_type( OA_SRC3(oap) ),
		VF_NAME(vfp) );
	warn(ERROR_STRING);
	return -1;
} /* end chktyp() */

#define show_legal_precisions(mask) _show_legal_precisions(QSP_ARG  mask)

static void _show_legal_precisions(QSP_ARG_DECL  uint32_t mask)
{
	uint32_t bit=1;
	prec_t prec;

	advise("legal precisions are:");
	for( prec = 0; prec < 32 ; prec ++ ){
		bit = 1 << prec ;
		if( mask & bit ){
			sprintf(ERROR_STRING,"\t%s",NAME_FOR_PREC_CODE(prec));
			advise(ERROR_STRING);
		}
	}
}

#define NEW_PREC_ERROR_MSG( prec )					\
									\
	sprintf(ERROR_STRING,						\
"chkprec:  %s:  input %s (%s) should have %s or %s precision with target %s (%s)",	\
VF_NAME(vfp) ,OBJ_NAME( OA_SRC1(oap) ) ,OBJ_PREC_NAME( OA_SRC1(oap) ),	\
NAME_FOR_PREC_CODE( prec ),NAME_FOR_PREC_CODE(dst_prec),		\
OBJ_NAME(OA_DEST(oap) ) ,OBJ_PREC_NAME( OA_DEST(oap) ));		\
	warn(ERROR_STRING);						\
	return -1;


#define CHECK_MAP_INDEX_PREC(func_code,prec_code,dp)					\
											\
	if( VF_CODE(vfp) == func_code ){						\
		if( OBJ_MACH_PREC(dp) != prec_code ){					\
			sprintf(ERROR_STRING,						\
		"Source object %s (%s) must have %s precision for function %s",		\
		OBJ_NAME(dp),OBJ_PREC_NAME(dp),						\
		NAME_FOR_PREC_CODE(prec_code),VF_NAME(vfp));				\
			warn(ERROR_STRING);						\
			return -1;							\
		}									\
	}

// Make sure that the object precision is legal with this function

#define CHECK_SOURCE_PREC_LEGAL_FOR_FUNC(dp,vfp)				\
	CHECK_OBJ_PREC_LEGAL_FOR_FUNC(dp,vfp,source)

#define CHECK_DEST_PREC_LEGAL_FOR_FUNC(dp,vfp)					\
	CHECK_OBJ_PREC_LEGAL_FOR_FUNC(dp,vfp,destination)

#define CHECK_OBJ_PREC_LEGAL_FOR_FUNC(dp,vfp,role)					\
											\
	if( ( VF_PRECMASK(vfp) & (1<<OBJ_MACH_PREC(dp))) == 0 ){			\
		sprintf(ERROR_STRING,							\
"chkprec:  %s precision %s (obj %s) cannot be used with function %s",			\
			#role,OBJ_PREC_NAME(dp),OBJ_NAME( dp ) ,VF_NAME(vfp) );		\
		warn(ERROR_STRING);							\
		show_legal_precisions( VF_PRECMASK(vfp));				\
		return -1;								\
	}


#define check_sources_precisions_for_func(vfp, oap ) _check_sources_precisions_for_func(QSP_ARG  vfp, oap )

static int _check_sources_precisions_for_func(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap )
{
	CHECK_DEST_PREC_LEGAL_FOR_FUNC(OA_DEST(oap),vfp)

	int n_srcs = 0;
	if(  OA_SRC1(oap) != NULL ){
		CHECK_SOURCE_PREC_LEGAL_FOR_FUNC(OA_SRC1(oap),vfp)
		n_srcs++;
		if( OA_SRC2(oap)  != NULL ){
			CHECK_SOURCE_PREC_LEGAL_FOR_FUNC(OA_SRC2(oap),vfp)
			n_srcs++;
			if( OA_SRC3(oap) != NULL ){
				CHECK_SOURCE_PREC_LEGAL_FOR_FUNC(OA_SRC3(oap),vfp)
				n_srcs++;
				if( OA_SRC4(oap) != NULL ){
					CHECK_SOURCE_PREC_LEGAL_FOR_FUNC(OA_SRC4(oap),vfp)
					n_srcs++;
					// Can there be more than 4 sources???
					// I guess not...
				}
			}
		}
	}
	return n_srcs;
}

#define REPORT_DEST_SRC_MISMATCH_ERROR(vfp,dp1,dp2)					\
											\
	sprintf(ERROR_STRING,								\
"chkprec:  %s: destination %s (%s) and source %s (%s) should have the same precision",	\
		VF_NAME(vfp),								\
		OBJ_NAME( dp1 ),							\
		OBJ_PREC_NAME( dp1 ),							\
		OBJ_NAME( dp2 ),							\
		OBJ_PREC_NAME( dp2 ) );							\
	warn(ERROR_STRING);


#define CHECK_DEST_SRC_MATCHING_PRECISIONS(vfp,dp1,dp2)					\
											\
	if( OBJ_MACH_PREC(dp1) != OBJ_MACH_PREC(dp2) ) {				\
		REPORT_DEST_SRC_MISMATCH_ERROR(vfp,dp1,dp2)				\
		return -1;								\
	}


#define REPORT_SRC_SRC_MISMATCH_ERROR(vfp,dp1,dp2)					\
											\
	sprintf(ERROR_STRING,								\
"chkprec:  %s operands %s (%s) and %s (%s) should have the same precision",		\
		VF_NAME(vfp),								\
		OBJ_NAME( dp1 ),							\
		OBJ_PREC_NAME( dp1 ),							\
		OBJ_NAME( dp2 ),							\
		OBJ_PREC_NAME( dp2 ) );							\
	warn(ERROR_STRING);


#define CHECK_MATCHING_SOURCES(vfp,dp1,dp2)						\
											\
	if( OBJ_MACH_PREC(dp1) != OBJ_MACH_PREC(dp2) ) {				\
		REPORT_SRC_SRC_MISMATCH_ERROR(vfp,dp1,dp2)				\
		return -1;								\
	}

#define check_first_two_sources(vfp, oap ) _check_first_two_sources(QSP_ARG  vfp, oap )

static int _check_first_two_sources(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap )
{
	/* First make sure that the two source operands match */
	// BUT only if not a mapping func...
	if( IS_LUTMAP_FUNC(vfp) ){
		// The mapping functions require a specific index (first source) precision...
		CHECK_MAP_INDEX_PREC(FVLUTMAPB,PREC_UBY,OA_SRC1(oap))
		CHECK_MAP_INDEX_PREC(FVLUTMAPS,PREC_UIN,OA_SRC1(oap))

		CHECK_DEST_SRC_MATCHING_PRECISIONS(vfp,OA_DEST(oap),OA_SRC2(oap))
	} else {
		CHECK_MATCHING_SOURCES(vfp,OA_SRC1(oap),OA_SRC2(oap))
	}

	/* if the precision is long, make sure that
	 * none (or all) are bitmaps
	 */
	if( OBJ_MACH_PREC( OA_SRC1(oap) ) == BITMAP_MACH_PREC ){
		if( (IS_BITMAP( OA_SRC1(oap) ) && ! IS_BITMAP(OA_SRC2(oap) )) ||
		    ( ! IS_BITMAP( OA_SRC1(oap) ) && IS_BITMAP(OA_SRC2(oap) )) ){
			REPORT_SRC_SRC_MISMATCH_ERROR(vfp,OA_SRC1(oap),OA_SRC2(oap))
			return -1;
		}
	}
	return 0;
}


#define srcp1		OBJ_MACH_PREC(OA_SRC1(oap))
#define dst_prec	OBJ_MACH_PREC(OA_DEST(oap))

static int dest_src1_precisions_match( Vec_Obj_Args *oap )
{
	if( srcp1 == dst_prec ){
		if( srcp1 == BITMAP_MACH_PREC ){
			if( IS_BITMAP(OA_DEST(oap) ) && !IS_BITMAP( OA_SRC1(oap) ) )
				return 0;
			if( IS_BITMAP( OA_SRC1(oap) ) && !IS_BITMAP(OA_DEST(oap) ) )
				return 0;
		}
		return 1;
	}
	return 0;
}

// check_special_projections called for vmaxg vmaxi etc
// Here the destination is an array of indices of where the occurrences were

#define check_special_projections(vfp, oap) _check_special_projections(QSP_ARG  vfp, oap)

static int _check_special_projections(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap)
{
	/* We assme that this is an index array and
	 * not a bitmap.
	 */
	if( OBJ_PREC( OA_DEST(oap) ) != PREC_DI ){
		sprintf(ERROR_STRING,
"chkprec:  %s:  destination vector %s (%s) should have %s precision",
			VF_NAME(vfp) ,OBJ_NAME(OA_DEST(oap) ) ,
			OBJ_PREC_NAME( OA_DEST(oap) ),
			NAME_FOR_PREC_CODE(PREC_DI) );
		warn(ERROR_STRING);
		return -1;
	}
	assert( OA_SRC1(oap) != NULL );
	SET_OA_ARGSPREC_CODE(oap, ARGSET_PREC(OBJ_PREC( OA_SRC1(oap) ) ));
	return 0;
}

// This is called when the function requires a bitmap destination...

#define check_bitmap_dest(vfp, oap) _check_bitmap_dest(QSP_ARG  vfp, oap)

static int _check_bitmap_dest(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap)
{
	if( OBJ_PREC( OA_DEST(oap) ) != PREC_BIT ){
		sprintf(ERROR_STRING,
	"%s:  result vector %s (%s) should have %s precision",
			VF_NAME(vfp) ,OBJ_NAME(OA_DEST(oap) ) ,
			OBJ_PREC_NAME( OA_DEST(oap) ),
			NAME_FOR_PREC_CODE(PREC_BIT));
		warn(ERROR_STRING);
		return -1;
	}
	/* use the precision from the source */
	SET_OA_ARGSPREC_CODE(oap, ARGSET_PREC(  OBJ_PREC( OA_SRC1(oap) )  ));
	return 0;
}

/* chkprec sets two flags:
 * oa_argsprec (tells machine prec),
 * and argstype (real/complex etc)
 *
 * But isn't argstype set by chktyp???
 *
 * The conditional operators don't have the same requirements
 * for matching...  That is, they *shouldn't*, but because we have
 * different function calls for each operand type, it would not be practical
 * to have all different types...
 */

#define chkprec(vfp,oap) _chkprec(QSP_ARG  vfp,oap)

static int _chkprec(QSP_ARG_DECL  Vector_Function *vfp,Vec_Obj_Args *oap)
{
	int n_srcs;
	if( IS_CONVERSION(vfp) ){
		// New conversions specify the destination precision
		// e.g. vconv2by, and have sub-functions for all possible source precs
		// QUESTION:  does that include the SAME precision???

		// We should have a check here to insure that the destination prec
		// is appropriate for each function code.

		// No support for bitmaps yet
		return 0;
	}

	/* BUG? could be bitmap destination??? */
	/* need to find out which prec to test... */

	/* First we make sure that all arg precisions
	 * are legal with this function
	 */

	n_srcs = check_sources_precisions_for_func(vfp,oap);
	if( n_srcs < 0 ) return -1;

	/* Figure out what type of function to call based on the arguments... */

	if( n_srcs == 0 ){
		/* oa_argstype is Function_Type...
		 * is this right? just a null setting?
		 */

		/* we used to use dst_prec here, but that
		 * is only the machine precision!?
		 */
		SET_OA_ARGSPREC_CODE(oap, ARGSET_PREC(OBJ_PREC( OA_DEST(oap) ) ));
		return 0;
	}

	if( n_srcs >= 2 ){
		if( check_first_two_sources(vfp,oap) < 0 )
			return -1;
	}

	// 3 or 4 inputs are the selection functions...  In principle the test operands
	// could have different types from the selection types - the latter have to match the destination,
	// while the former only have to match each other.  But that would lead to an unreasonable proliferation
	// in the number of function types, so we don't allow it.
	if( n_srcs >= 3 ){
		CHECK_MATCHING_SOURCES(vfp,OA_SRC1(oap),OA_SRC3(oap))
	}
	if( n_srcs >= 4 ){
		CHECK_MATCHING_SOURCES(vfp,OA_SRC1(oap),OA_SRC4(oap))
	}

	/* Before proceeding, make sure that this destination precision is legal with this function
	 * There are a few special cases for which the destination has a different precision than the source.
	 */

	                  /* vmaxg etc */                    /* vmaxi etc */
	if( VF_FLAGS(vfp) == V_SCALRET2 || VF_FLAGS(vfp) == V_INT_PROJECTION)
		return check_special_projections(vfp,oap);


	/* Now we know that there are 1 or 2 inputs in addition to the target,
	 * and that if there are two they match.  Therefore we only have to
	 * consider the first one.
	 * dst_prec is the machine precision of the destination -
	 * but doesn't include the pseudo-precision for bitmaps?
	 */
	/* This test can succeed when the input is the same as bitmap_word */

	if( dest_src1_precisions_match(oap) ){
		SET_OA_ARGSPREC_CODE(oap, ARGSET_PREC(OBJ_PREC( OA_DEST(oap) ) ));
		return 0;
	}

	/* Now we know that this is a mixed precision case.
	 * Make sure it is one of the legal ones.
	 * First we check the special cases (bitmaps, indices).
	 */

	if( IS_LUTMAP_FUNC(vfp) ){		/* */
		/* use the precision from the map */
		SET_OA_ARGSPREC_CODE(oap, ARGSET_PREC(  OBJ_PREC( OA_SRC2(oap) )  ));
		return 0;
	}

	if( VF_FLAGS(vfp) & BITMAP_DST ){		/* vcmp, vcmpm */
		return check_bitmap_dest(vfp,oap);
		// BUG?  functype gets set at the bottom of this function, so how can we return?
		// Do we also set it elsewhere???
	}

	/* don't insist on a precision match if result is an index */
	if( VF_FLAGS(vfp) & INDEX_RESULT ){
		/* We assume that we check the result precision elsewhere? */
		return 0;
	}

	// Finally, check for the known allowed mixed precisions

	switch( dst_prec ){
		case PREC_IN:
			if( srcp1==PREC_UBY ){
				SET_OA_ARGSPREC_CODE(oap, BYIN_ARGS);
				return 0;
			}
			NEW_PREC_ERROR_MSG(PREC_UBY);
			break;
		case PREC_DP:
			if( srcp1==PREC_SP ){
				SET_OA_ARGSPREC_CODE(oap, SPDP_ARGS);
				return 0;
			}
			NEW_PREC_ERROR_MSG(PREC_SP);
			break;
		case PREC_DI:
			if( srcp1==PREC_UIN ){
				SET_OA_ARGSPREC_CODE(oap, INDI_ARGS);
				return 0;
			}
			NEW_PREC_ERROR_MSG(PREC_UIN);
			break;
		case PREC_BY:
			if( srcp1==PREC_IN ){
				SET_OA_ARGSPREC_CODE(oap, INBY_ARGS);
				return 0;
			}
			NEW_PREC_ERROR_MSG(PREC_IN);
			break;
		default:
			sprintf(ERROR_STRING,
"chkprec:  %s:  target '%s' (%s) cannot be used with mixed prec source '%s' (%s)",
				VF_NAME(vfp) ,OBJ_NAME(OA_DEST(oap) ) ,
				OBJ_PREC_NAME( OA_DEST(oap) ),
				OBJ_NAME( OA_SRC1(oap) ) ,NAME_FOR_PREC_CODE(srcp1));
			warn(ERROR_STRING);
			return -1;
	}

	// It's an error if we've gotten here...
	SET_OA_FUNCTYPE( oap, FUNCTYPE_FOR( OA_ARGSPREC_CODE(oap) ,OA_ARGSTYPE(oap) ) );
} /* end chkprec() */

#define check_size_match(vfp, dp1, dp2 ) _check_size_match(QSP_ARG  vfp, dp1, dp2 )

static int _check_size_match(QSP_ARG_DECL  Vector_Function *vfp, Data_Obj *dp1, Data_Obj *dp2 )
{
	int status;

	if( dp1 == NULL ) return 0;
	if( dp2 == NULL ) return 0;

	if( (status=cksiz(VF_FLAGS(vfp), dp1 ,dp2 )) == (-1) ){
		sprintf(ERROR_STRING,
	"check_size_match:  Size mismatch between objects %s and %s, function %s",
			OBJ_NAME( dp1 ) ,OBJ_NAME( dp2 ), VF_NAME(vfp) );
		advise(ERROR_STRING);	// why not warning???
		return -1;
	}
	assert( status == 0 );
	return 0;
}

#define chksiz(vfp,oap) _chksiz(QSP_ARG  vfp,oap)

static int _chksiz(QSP_ARG_DECL  Vector_Function *vfp,Vec_Obj_Args *oap)	/* check for argument size match */
{
	int status=0;

	/* If the operation has a bitmap, then check the bitmap against the first source
	 * (if bitmap is destination), or the destination (if bitmap is a source).
	 *
	 * An exception is conversion routines, where the bitmap *is* the first source...
	 * The conversion routines do not have the BITMAP flag set...
	 *
	 * We should allow the bitmap to have additional dimensions above what the source
	 * has, like a projection loop...
	 */

	if( OA_SBM(oap) != NULL ){
		if( VF_FLAGS(vfp) & BITMAP_SRC ){	// redundant?
			/* We used to require that the bitmap size matched the destination,
			 * but that is not necessary...
			 */

			if( (status=cksiz(VF_FLAGS(vfp),OA_SBM(oap) ,OA_DEST(oap) ))
				== (-1) )
			{
				sprintf(ERROR_STRING,
			"chksiz:  bitmap arg func size error, function %s",VF_NAME(vfp) );
				advise(ERROR_STRING);
				return -1;
			}
		}
		else {
			assert( IS_CONVERSION(vfp) || VF_CODE(vfp) == FVMOV );
		}
		assert( status == 0 );

	}

	if(  OA_SRC1(oap)  == NULL ){
		/* nothing to check!? */
		return 0;
	}
#ifdef QUIP_DEBUG
if( debug & veclib_debug ){
sprintf(ERROR_STRING,"chksiz:  destv %s (%s)  arg1 %s (%s)",
OBJ_NAME(OA_DEST(oap) ), AREA_NAME(OBJ_AREA(OA_DEST(oap))),
OBJ_NAME( OA_SRC1(oap) ), AREA_NAME(OBJ_AREA(OA_SRC1(oap))) );
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

#ifdef FVDOT
	/* We check the sizes of the args against the destination object - but in the case of ops like vdot,
	 * (or any other scalar-returning projection op like vmax etc)
	 * this may not match...
	 */
	if( VF_CODE(vfp) == FVDOT ){
		if( (status=cksiz(VF_FLAGS(vfp), OA_SRC1(oap) ,OA_SRC2(oap) )) == (-1) ){
			sprintf(ERROR_STRING,"chksiz:  Size mismatch between arg1 (%s) and arg2 (%s), function %s",
				OBJ_NAME( OA_SRC1(oap) ) ,OBJ_NAME(OA_SRC2(oap) ) ,VF_NAME(vfp) );
			advise(ERROR_STRING);
			return -1;
		}
		return 0;
	}
#endif // FVDOT

	if( check_size_match(vfp, OA_SRC1(oap), OA_DEST(oap) ) < 0 )
		return -1;

	if( IS_LUTMAP_FUNC(vfp) ){
		// second source should be a map w/ 256 entries
		// For FVLUTMAPS, we pass the table size as a scalar arg.
		if( VF_CODE(vfp) == FVLUTMAPB ){
			if( OBJ_N_MACH_ELTS(OA_SRC2(oap)) != 256 ){
				sprintf(ERROR_STRING,
			"chksiz:  byte-indexed map object %s (%d) should have 256 elements!?",
			OBJ_NAME(OA_SRC2(oap)),OBJ_N_MACH_ELTS(OA_SRC2(oap)));
				warn(ERROR_STRING);
				return -1;
			}
		}
		if( ! IS_CONTIGUOUS(OA_SRC2(oap)) ){
			sprintf(ERROR_STRING,"chksiz:  map object %s must be contiguous!?",
				OBJ_NAME(OA_SRC2(oap)));
			warn(ERROR_STRING);
			return -1;
		}
	} else {
		if( check_size_match(vfp, OA_SRC2(oap), OA_DEST(oap) ) < 0 )
			return -1;
	}

	if( check_size_match(vfp, OA_SRC3(oap), OA_DEST(oap) ) < 0 )
		return -1;

	if( check_size_match(vfp, OA_SRC4(oap), OA_DEST(oap) ) < 0 )
		return -1;

	// SRC5 ???



	/* BUG what about bitmaps?? */

	return 0;
} /* end chksiz() */

/* check that all of the arguments match (when they should) */


#define chkargs(vfp, oap) _chkargs( QSP_ARG  vfp, oap)

static int _chkargs( QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap)
{
	assert( OA_DEST(oap) != NULL || (VF_FLAGS(vfp) & BITMAP_DST)==0 );

	if( chk_uk(vfp,oap) == (-1) ) return -1;
	if( chktyp(vfp,oap) == (-1) ) return -1;
	if( chkprec(vfp,oap) == (-1) ) return -1;
	if( chksiz(vfp,oap) == (-1) ) return -1;

	/* Now we have to set the function type */

	return 0;
}


/* chkprec() now has the job of figuring out mixed precision op's */

/* Instead of using the prec_mask from the table, we can figure out what
 * function we want and see if it is not equal to nullf...
 *
 */

#define PREC_ERROR_MSG( prec )						\
									\
	sprintf(ERROR_STRING,						\
"chkprec:  %s:  input %s (%s) should have %s or %s precision with target %s (%s)",	\
VF_NAME(vfp) ,OBJ_NAME( OA_SRC1(oap) ) ,OBJ_PREC_NAME( OA_SRC1(oap) ),	\
NAME_FOR_PREC_CODE( prec ),NAME_FOR_PREC_CODE(dst_prec),		\
OBJ_NAME(OA_DEST(oap) ) ,OBJ_PREC_NAME( OA_DEST(oap) ));		\
	warn(ERROR_STRING);						\
	return -1;


#ifdef FOOBAR
int cktype(Data_Obj *dp1,Data_Obj *dp2)
{
	if( dp1->dt_tdim != dp2->dt_tdim ) return -1;
	else return 0;
}

void wacky_arg(Data_Obj *dp)
{
	sprintf(ERROR_STRING, "%s:  inc = %d, cols = %d",
		OBJ_NAME(dp) , dp->dt_inc, dp->dt_cols );
	warn(ERROR_STRING);
	list_dobj(dp);
	error1("wacky_arg:  can't happen #1");
}

static char *remove_brackets(char *name)
{
	static char clean_name[LLEN];
	char *s,*t;

	/* if the name has no brackets, we don't need to do anything */
	if( strstr(name,"[") == NULL && strstr(name,"{") == NULL ) return(name);

	/* BUG we don't check for the name overflowing LLEN */
	s=name;
	t=clean_name;
	while( *s ){
		if( *s == '[' ){
			*t++ = 'S';
			*t++ = 'O';
		} else if( *s == ']' ){
			*t++ = 'S';
			*t++ = 'C';
		} else if( *s == '{' ){
			*t++ = 'C';
			*t++ = 'O';
		} else if( *s == '}' ){
			*t++ = 'C';
			*t++ = 'C';
		} else {
			*t++ = *s;
		}
		s++;
	}
	*t=0;
	return(clean_name);
}
#endif /* FOOBAR */

#ifdef FOOBAR
static int make_arg_evenly_spaced(Vec_Obj_Args *oap,int index)
{
	Data_Obj *new_dp,*arg_dp;
	char tmp_name[LLEN];

	arg_dp = OA_SRC_OBJ(oap,index) ;

	if( arg_dp == NULL ) return 0;
	if( IS_EVENLY_SPACED(arg_dp) ) return 0;

	/* If the object is subscripted, the brackets will break the name */
	sprintf(tmp_name,"%s.dup",remove_brackets(OBJ_NAME(arg_dp) ));
	new_dp = dup_obj(arg_dp,tmp_name);
	dp_copy(new_dp,arg_dp);	/* BUG use vmov */
	if( OA_DEST(oap)  == arg_dp )
		OA_DEST(oap)  = new_dp;
	SET_OA_SRC_OBJ(oap,index) = new_dp;

	return(1);
}
#endif /* FOOBAR */

int _perf_vfunc(QSP_ARG_DECL  Vec_Func_Code code, Vec_Obj_Args *oap)
{
	return( call_vfunc(FIND_VEC_FUNC(code), oap) );
}

#ifdef HAVE_ANY_GPU
// BUG???  is this redundant now that we have platforms?

static int _default_gpu_dispatch(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap)
{
	sprintf(ERROR_STRING,"No GPU dispatch function specified, can't call %s",VF_NAME(vfp) );
	warn(ERROR_STRING);
	advise("Please call set_gpu_dispatch_func().");
	return -1;
}

static int (*gpu_dispatch_func)(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap)=_default_gpu_dispatch;

void set_gpu_dispatch_func( int (*func)(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap) )
{
//sprintf(ERROR_STRING,"Setting gpu dispatch func (0x%"PRIxPTR")",(uintptr_t)func);
//advise(ERROR_STRING);
	gpu_dispatch_func = func;
}

#endif /* HAVE_ANY_GPU */

int _call_vfunc( QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap )
{
	int retval;

	/* Set the default function type.
	 * Why do we use src1 in preference to oa_dest?
	 *
	 * One answer is bitmap result functions...
	 */
	if(  OA_SRC1(oap)  != NULL ){
		SET_OA_ARGSPREC_CODE(oap, ARGSET_PREC(  OBJ_PREC( OA_SRC1(oap) )  ));
	} else if( OA_DEST(oap)  != NULL ){
		SET_OA_ARGSPREC_CODE(oap, ARGSET_PREC( OBJ_PREC( OA_DEST(oap) )  ));
	} else {
		sprintf(ERROR_STRING,"call_vfunc %s:",VF_NAME(vfp) );
		advise(ERROR_STRING);
		error1("call_vfunc:  no prototype vector!?");
	}

	/* check for precision, type, size matches */
	if( chkargs(vfp,oap) == (-1) ) return -1;	/* make set vslct_fake */

	/* argstype has been set from within chkargs */
	SET_OA_FUNCTYPE( oap, FUNCTYPE_FOR( OA_ARGSPREC_CODE(oap) ,OA_ARGSTYPE(oap) ) );

	/* We don't worry here about vectorization on CUDA... */

	// Here we should call the platform-specific dispatch function...
	if( check_obj_devices(oap) < 0 )
		return -1;

	assert( OA_PFDEV(oap) != NULL );

	retval = platform_dispatch( PFDEV_PLATFORM(OA_PFDEV(oap)), vfp, oap );

	return retval;
} // call_vfunc

