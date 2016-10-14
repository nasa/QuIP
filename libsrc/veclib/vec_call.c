#include "quip_config.h"

#include <string.h>
#include "quip_prot.h"
#include "nvf.h"
#include "debug.h"

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

static void shape_error(QSP_ARG_DECL  Vector_Function *vfp, Data_Obj *dp)
{
	sprintf(ERROR_STRING,"shape_error:  Vector function %s:  argument %s has unknown shape!?",
		VF_NAME(vfp),OBJ_NAME(dp));
	WARN(ERROR_STRING);
}


static int chk_uk(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap)
{
	int i;

	if( OA_DEST(oap)  != NO_OBJ && UNKNOWN_OBJ_SHAPE(OA_DEST(oap)) ){
		shape_error(QSP_ARG  vfp,OA_DEST(oap) );
		return(-1);
	}
	for(i=0;i<MAX_N_ARGS;i++){
		if( OA_SRC_OBJ(oap,i) != NO_OBJ && UNKNOWN_OBJ_SHAPE( OA_SRC_OBJ(oap,i) ) ){
			shape_error(QSP_ARG  vfp,OA_SRC_OBJ(oap,i));
			return(-1);
		}
	}
	if( OA_SBM(oap) != NO_OBJ && UNKNOWN_OBJ_SHAPE(OA_SBM(oap)) ){
		shape_error(QSP_ARG  vfp,OA_SBM(oap) );
		return(-1);
	}
	/* BUG check the scalar objects too? */
	return(0);
}

static const char *name_for_type(Data_Obj *dp)
{
	if( IS_REAL(dp) ) return("real");
	else if( IS_COMPLEX(dp) ) return("complex");
	else if( IS_QUAT(dp) ) return("quaternion");
//#ifdef CAUTIOUS
	else {
//		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  name_for_type:  type of object %s is unknown",OBJ_NAME(dp) );
//		NWARN(DEFAULT_ERROR_STRING);
//		return("unknown");
		assert( AERROR("name_for_type:  unexpected type code!?") );
	}
//#endif /* CAUTIOUS */
}

/* The "type" is real, complex, quaternion, or mixed...
 * independent of "precision" (byte/short/float etc)
 */

static int chktyp(QSP_ARG_DECL  Vector_Function *vfp,Vec_Obj_Args *oap)
{
	SET_OA_ARGSTYPE(oap, UNKNOWN_ARGS);

	/* Set the type based on the destination vector */
	/* destv is highest numbered arg */
	if( IS_REAL(OA_DEST(oap) ) ){
		if( OA_SRC2(oap)  != NO_OBJ ){	/* two source operands */
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
//#ifdef CAUTIOUS
			// Why was this CAUTIOUS when other goto's to type_mismatch13 are not???
			  else {
				/* OA_SRC1 is not real or complex, must be a type mismatch */
				goto type_mismatch13;
			}
//#endif /* CAUTIOUS */
		} else if(  OA_SRC1(oap)  != NO_OBJ ){	/* one source operand */
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
		if( OA_SRC2(oap)  != NO_OBJ ){	/* two source operands */
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
		} else if(  OA_SRC1(oap)  != NO_OBJ ){	/* one source operand */
			if( IS_COMPLEX( OA_SRC1(oap) ) ){
				SET_OA_ARGSTYPE(oap, COMPLEX_ARGS);
			} else if( IS_REAL( OA_SRC1(oap) ) ){
				SET_OA_ARGSTYPE(oap, MIXED_ARGS);
			} else {
				/* OA_SRC1 is not real or complex, must be a type mismatch */
				goto type_mismatch01;
			}
		} else {				/* only 1 operand */
			SET_OA_ARGSTYPE(oap, COMPLEX_ARGS);
		}
	} else if( IS_QUAT(OA_DEST(oap) ) ){
		if( OA_SRC2(oap)  != NO_OBJ ){	/* two source operands */
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
		} else if(  OA_SRC1(oap)  != NO_OBJ ){	/* one source operand */
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
		WARN(ERROR_STRING);
	}

	/* now the type field has been set - make sure it's legal */
	/* But first check a couple of special cases */

	/* make sure that function doesn't require mixed types */

	if( VF_FLAGS(vfp) & CPX_2_REAL ){
//#ifdef CAUTIOUS
//		// quiet compiler
//		if( OA_SRC1(oap) == NULL ){
//			WARN("CAUITOUS:  get_scal:  Unexpected null source operand!?");
//			return -1;
//		}
//#endif // CAUTIOUS
		assert( OA_SRC1(oap) != NULL );

		if( ! IS_COMPLEX( OA_SRC1(oap) ) ){
			sprintf(ERROR_STRING,"source vector %s (%s) must be complex with function %s",
				OBJ_NAME( OA_SRC1(oap) ) ,OBJ_PREC_NAME( OA_DEST(oap) ),VF_NAME(vfp) );
			WARN(ERROR_STRING);
			list_dobj(QSP_ARG  OA_SRC1(oap) );
			return(-1);
		}
		if( ! IS_REAL(OA_DEST(oap) ) ){
			sprintf(ERROR_STRING,"destination vector %s (%s) must be real with function %s",
				OBJ_NAME(OA_DEST(oap) ) ,OBJ_PREC_NAME( OA_DEST(oap) ),VF_NAME(vfp) );
			WARN(ERROR_STRING);
			list_dobj(QSP_ARG OA_DEST(oap) );
			return(-1);
		}
		SET_OA_ARGSTYPE(oap, REAL_ARGS);
		return(0);
	}

	if( VF_CODE(vfp) == FVFFT ){
		/* source vector can be real or complex */
		if( !IS_COMPLEX(OA_DEST(oap) ) ){
			WARN("destination must be complex for fft");
			return(-1);
		}
//#ifdef CAUTIOUS
//		// quiet analyzer
//		if( OA_SRC1(oap) == NULL ){
//			WARN("CAUTIOUS:  Unexpected null src1 with fft!?");
//			return -1;
//		}
//#endif // CAUTIOUS
		assert( OA_SRC1(oap) != NULL );

		if( IS_COMPLEX( OA_SRC1(oap) ) )
			SET_OA_ARGSTYPE(oap,COMPLEX_ARGS);
		else if( IS_QUAT( OA_SRC1(oap) ) ){
			WARN("Can't compute FFT of a quaternion input");
			return(-1);
		} else
			SET_OA_ARGSTYPE(oap,REAL_ARGS);

		return(0);
	}

	if( VF_CODE(vfp) == FVIFT ){
//#ifdef CAUTIOUS
//		// quiet analyzer
//		if( OA_SRC1(oap) == NULL ){
//			WARN("CAUTIOUS:  Unexpected null src1 with fft!?");
//			return -1;
//		}
//#endif // CAUTIOUS
		assert( OA_SRC1(oap) != NULL );

		/* destination vector can be real or complex */
		if( !IS_COMPLEX( OA_SRC1(oap) ) ){
			WARN("source must be complex for inverse fft");
			return(-1);
		}
		if( IS_COMPLEX(OA_DEST(oap) ) )
			SET_OA_ARGSTYPE(oap,COMPLEX_ARGS);
		else if( IS_QUAT(OA_DEST(oap) ) ){
			WARN("Can't compute inverse FFT to a quaternion target");
			return(-1);
		} else
			SET_OA_ARGSTYPE(oap,REAL_ARGS);

		return(0);
	}

	/* now the type field has been set - make sure it's legal */
	if( (VF_TYPEMASK(vfp) & VL_TYPE_MASK(OA_ARGSTYPE(oap) ) )==0 ){
		sprintf(ERROR_STRING,
	"chktyp:  Arguments of type %s are not permitted with function %s",
			argset_type_name[OA_ARGSTYPE(oap) ],VF_NAME(vfp) );
		WARN(ERROR_STRING);
		return(-1);
	}

/*
sprintf(ERROR_STRING,"function %s:  oa_argstype = %s",VF_NAME(vfp) ,argset_type_name[OA_ARGSTYPE(oap) ]);
ADVISE(ERROR_STRING);
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
		WARN(ERROR_STRING);
		return(-1);
	}

	if( HAS_QMIXED_ARGS(oap)  && ! IS_QUAT(OA_DEST(oap) ) ){
		sprintf(ERROR_STRING,"chktyp:  destination vector %s must be quaternion when mixing types with function %s",
			OBJ_NAME(OA_DEST(oap) ) ,VF_NAME(vfp) );
		WARN(ERROR_STRING);
		return(-1);
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
#ifdef FOOBAR
		if( USES_REAL_SCALAR(VF_CODE(vfp)) ){
			if( ! IS_COMPLEX( OA_SRC2(oap) ) ){
				WARN("destination vector must be complex when mixing types with vsmul");
				return(-1);
			}
			if( ! IS_REAL( OA_SRC1(oap) ) ){
				WARN("source vector must be real when mixing types with vsmul");
				return(-1);
			}
		} else if( USES_COMPLEX_SCALAR(VF_CODE(vfp)) ){
			if( ! IS_COMPLEX( OA_SRC2(oap) ) ){
				WARN("destination vector must be complex when mixing types with vcsmul");
				return(-1);
			}
			if( ! IS_REAL( OA_SRC1(oap) ) ){
				WARN("source vector must be real when mixing types with vcsmul");
				return(-1);
			}
		} else if( USES_QUAT_SCALAR(VF_CODE(vfp)) ){
			if( ! IS_QUAT( OA_SRC2(oap) ) ){
				WARN("destination vector must be quaternion when mixing types with vqsmul");
				return(-1);
			}
			if( ! IS_REAL( OA_SRC1(oap) ) ){
				WARN("source vector must be real when mixing types with vqsmul");
				return(-1);
			}
		}
#endif /* FOOBAR */
	
		/*
		show_obj_args(oap);
sprintf(ERROR_STRING,"Function %s.",VF_NAME(vfp) );
ADVISE(ERROR_STRING);
		ERROR1("chktyp:  Sorry, not sure how to deal with this situation...");
		*/

		if( ! IS_COMPLEX( OA_SRC1(oap) ) ){
			sprintf(ERROR_STRING,
"first source vector (%s,%s) must be complex when mixing types with function %s",
				OBJ_NAME( OA_SRC1(oap) ) ,
				name_for_type( OA_SRC1(oap) ),
				VF_NAME(vfp) );
			WARN(ERROR_STRING);
			return(-1);
		}
		// Mixed-arg fuctions have to have two sources, but the analyzer
		// doesn't know that...
//#ifdef CAUTIOUS
//		if( OA_SRC2(oap) == NULL ){
//			WARN("CAUTIOUS:  Null src2 with mixed-arg function!?");
//			return -1;
//		}
//#endif // CAUTIOUS
		assert( OA_SRC2(oap) != NULL );

		if( ! IS_REAL(OA_SRC2(oap) ) ){
			sprintf(ERROR_STRING,
"second source vector (%s,%s) must be real when mixing types with function %s",
				OBJ_NAME(OA_SRC2(oap) ) ,
				name_for_type(OA_SRC2(oap) ),
				VF_NAME(vfp) );
			WARN(ERROR_STRING);
			return(-1);
		}
		/* Should the destination be complex??? */
	} else if( HAS_QMIXED_ARGS(oap) ){
		ERROR1("FIXME:  need to add code for quaternions, vec_call.c");
		/* BUG add the same check as above for quaternions */
		return -1;
	}

	return(0);

type_mismatch13:
	sprintf(ERROR_STRING,"Type mismatch between objects %s (%s) and %s (%s), function %s",
		OBJ_NAME( OA_SRC1(oap) ) ,name_for_type( OA_SRC1(oap) ),
		OBJ_NAME( OA_SRC3(oap) ) ,name_for_type( OA_SRC3(oap) ),
		VF_NAME(vfp) );
	WARN(ERROR_STRING);
	return(-1);
    
type_mismatch01:
	sprintf(ERROR_STRING,"Type mismatch between objects %s (%s) and %s (%s), function %s",
            OBJ_NAME( OA_SRC1(oap) ) ,name_for_type( OA_SRC1(oap) ),
            OBJ_NAME(OA_DEST(oap) ) ,name_for_type(OA_DEST(oap) ),
            VF_NAME(vfp) );
	WARN(ERROR_STRING);
	return(-1);
    
    
type_mismatch23:
	sprintf(ERROR_STRING,"Type mismatch between objects %s (%s) and %s (%s), function %s",
		OBJ_NAME(OA_SRC2(oap) ) ,name_for_type(OA_SRC2(oap) ),
		OBJ_NAME( OA_SRC3(oap) ) ,name_for_type( OA_SRC3(oap) ),
		VF_NAME(vfp) );
	WARN(ERROR_STRING);
	return(-1);
} /* end chktyp() */

static void show_legal_precisions(uint32_t mask)
{
	uint32_t bit=1;
	prec_t prec;

	NADVISE("legal precisions are:");
	for( prec = 0; prec < 32 ; prec ++ ){
		bit = 1 << prec ;
		if( mask & bit ){
			sprintf(DEFAULT_ERROR_STRING,"\t%s",NAME_FOR_PREC_CODE(prec));
			NADVISE(DEFAULT_ERROR_STRING);
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
	WARN(ERROR_STRING);						\
	return(-1);



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

 /* end chkprec */

static int chkprec(QSP_ARG_DECL  Vector_Function *vfp,Vec_Obj_Args *oap)
{
	prec_t srcp1,srcp2,dst_prec;
	int n_srcs=0;
	if( IS_NEW_CONVERSION(vfp) ){
		// New conversions specify the destination precision
		// e.g. vconv2by, and have sub-functions for all possible source precs

		// We should have a check here to insure that the destination prec
		// is appropriate for each function code.

		// No support for bitmaps yet
		return(0);
	}
	if( IS_CONVERSION(vfp) ){
		/* Conversions support all the rpecisions, so the checks
		 * after this block are irrelevant.
		 */

		/* the default function type is set using OA_SRC1 (the source),
		 * but if the target is a bitmap we want to set it to bitmap...
		 */
		if( IS_BITMAP(OA_DEST(oap) ) ){
/*
ADVISE("chkprec:  Setting argstype to R_BIT_ARGS!?");
			SET_OA_ARGSTYPE(oap, R_BIT_ARGS);
*/
			/* R_BIT_ARGS was a functype - not an argset type??? */
			SET_OA_ARGSPREC(oap, BIT_ARGS);
		} else if( IS_BITMAP( OA_SRC1(oap) ) ){
			/* this is necessary because bitmaps handled with kludgy hacks */
			SET_OA_SBM(oap,OA_SRC1(oap) );
		}
		return(0);
	}

	dst_prec=OBJ_MACH_PREC(OA_DEST(oap) );
	/* BUG? could be bitmap destination??? */
	/* need to find out which prec to test... */

	/* First we make sure that all arg precisions
	 * are legal with this function
	 */

	if( ( VF_PRECMASK(vfp) & (1<<dst_prec)) == 0  ){
		sprintf(ERROR_STRING,
"chkprec:  dest. precision %s (obj %s) cannot be used with function %s",
			NAME_FOR_PREC_CODE(dst_prec),OBJ_NAME(OA_DEST(oap) ) ,VF_NAME(vfp) );
		WARN(ERROR_STRING);
		show_legal_precisions( VF_PRECMASK(vfp));
		return(-1);
	}

	if(  OA_SRC1(oap)  != NO_OBJ ){
		srcp1=OBJ_MACH_PREC( OA_SRC1(oap) );
		if( ( VF_PRECMASK(vfp) & (1<<srcp1)) == 0 ){
			sprintf(ERROR_STRING,
"chkprec:  src precision %s (obj %s) cannot be used with function %s",
		NAME_FOR_PREC_CODE(srcp1),OBJ_NAME( OA_SRC1(oap) ) ,VF_NAME(vfp) );
			WARN(ERROR_STRING);
			show_legal_precisions( VF_PRECMASK(vfp));
			return(-1);
		}
		n_srcs++;
		if( OA_SRC2(oap)  != NO_OBJ ){
			srcp2=OBJ_MACH_PREC(OA_SRC2(oap) );
			if( ( VF_PRECMASK(vfp) & (1<<srcp2)) == 0 ){
				sprintf(ERROR_STRING,
"chkprec:  src precision %s (obj %s) cannot be used with function %s",
			NAME_FOR_PREC_CODE(srcp2),OBJ_NAME(OA_SRC2(oap) ) ,VF_NAME(vfp) );
				WARN(ERROR_STRING);
				show_legal_precisions( VF_PRECMASK(vfp));
				return(-1);
			}
			n_srcs++;
		}
		// Can there be more than 3 sources???
	}

	/* Figure out what type of function to call based on the arguments... */

	if( n_srcs == 0 ){
		/* oa_argstype is Function_Type...
		 * is this right? just a null setting?
		 */

		/* we used to use dst_prec here, but that
		 * is only the machine precision!?
		 */
		SET_OA_ARGSPREC(oap, ARGSET_PREC(OBJ_PREC( OA_DEST(oap) ) ));
		return(0);
	} else if( n_srcs == 2 ){
		/* First make sure that the two source operands match */
		if( srcp1 != srcp2 ) {
source_mismatch_error:
			sprintf(ERROR_STRING,
"chkprec:  %s operands %s (%s) and %s (%s) should have the same precision",
				VF_NAME(vfp) ,OBJ_NAME( OA_SRC1(oap) ) ,
				OBJ_PREC_NAME( OA_SRC1(oap) ),
				OBJ_NAME(OA_SRC2(oap) ) ,
				OBJ_PREC_NAME( OA_SRC2(oap) ) );
			WARN(ERROR_STRING);
			return(-1);
		}
		/* if the precision is long, make sure that
		 * none (or all) are bitmaps
		 */
		if( srcp1 == BITMAP_MACH_PREC ){
			if( (IS_BITMAP( OA_SRC1(oap) ) && ! IS_BITMAP(OA_SRC2(oap) )) ||
			    ( ! IS_BITMAP( OA_SRC1(oap) ) && IS_BITMAP(OA_SRC2(oap) )) )
				goto source_mismatch_error;
		}
	}
	/* Now we know that there are 1 or 2 inputs in addition to the target,
	 * and that if there are two they match.  Therefore we only have to
	 * consider the first one.
	 * dst_prec is the machine precision of the destination -
	 * but doesn't include the pseudo-precision for bitmaps?
	 */
	/* This test can succeed when the input is the same as bitmap_word */
	if( srcp1 == dst_prec ){
		if( srcp1 == BITMAP_MACH_PREC ){
			if( IS_BITMAP(OA_DEST(oap) ) && !IS_BITMAP( OA_SRC1(oap) ) )
				goto next1;
			if( IS_BITMAP( OA_SRC1(oap) ) && !IS_BITMAP(OA_DEST(oap) ) )
				goto next1;
		}
		/* Can't use dst_prec here because might be bitmap */
		SET_OA_ARGSPREC(oap, ARGSET_PREC(OBJ_PREC( OA_DEST(oap) ) ));
		return(0);
	}
next1:

	/* Now we know that this is a mixed precision case.
	 * Make sure it is one of the legal ones.
	 * First we check the special cases (bitmaps, indices).
	 */
	if( VF_FLAGS(vfp) & BITMAP_DST ){		/* vcmp, vcmpm */
		/* Is dest vector set too??? */
		if( OBJ_PREC( OA_DEST(oap) )  != PREC_BIT ){
			sprintf(ERROR_STRING,
		"%s:  result vector %s (%s) should have %s precision",
				VF_NAME(vfp) ,OBJ_NAME(OA_DEST(oap) ) ,
				OBJ_PREC_NAME( OA_DEST(oap) ),
				NAME_FOR_PREC_CODE(PREC_BIT));
			WARN(ERROR_STRING);
			return(-1);
		}
		/* use the precision from the source */
		SET_OA_ARGSPREC(oap, ARGSET_PREC(  OBJ_PREC( OA_SRC1(oap) )  ));
		return(0);
	}
	if( VF_FLAGS(vfp) == V_SCALRET2 ){ /* vmaxg etc */
		/* We assme that this is an index array and
		 * not a bitmap.
		 */
		if( OBJ_PREC( OA_DEST(oap) )  != PREC_DI ){
			sprintf(ERROR_STRING,
"chkprec:  %s:  destination vector %s (%s) should have %s precision",
				VF_NAME(vfp) ,OBJ_NAME(OA_DEST(oap) ) ,
				OBJ_PREC_NAME( OA_DEST(oap) ),
				NAME_FOR_PREC_CODE(PREC_DI) );
			WARN(ERROR_STRING);
			return(-1);
		}
		/* If the destination is long, don't worry about
		 * a match with the arg...
		 */
		return(0);
	}

	/* don't insist on a precision match if result is an index */
	if( VF_FLAGS(vfp) & INDEX_RESULT ){
		/* We assume that we check the result precision elsewhere? */
		return(0);
	}

	switch( dst_prec ){
		case PREC_IN:
			if( srcp1==PREC_UBY ){
				SET_OA_ARGSPREC(oap, BYIN_ARGS);
				return(0);
			}
			NEW_PREC_ERROR_MSG(PREC_UBY);
			break;
		case PREC_DP:
			if( srcp1==PREC_SP ){
				SET_OA_ARGSPREC(oap, SPDP_ARGS);
				return(0);
			}
			NEW_PREC_ERROR_MSG(PREC_SP);
			break;
		case PREC_DI:
			if( srcp1==PREC_UIN ){
				SET_OA_ARGSPREC(oap, INDI_ARGS);
				return(0);
			}
			NEW_PREC_ERROR_MSG(PREC_UIN);
			break;
		case PREC_BY:
			if( srcp1==PREC_IN ){
				SET_OA_ARGSPREC(oap, INBY_ARGS);
				return(0);
			}
			NEW_PREC_ERROR_MSG(PREC_IN);
			break;
		default:
			sprintf(ERROR_STRING,
"chkprec:  %s:  target %s (%s) cannot be used with mixed prec source %s (%s)",
				VF_NAME(vfp) ,OBJ_NAME(OA_DEST(oap) ) ,
				OBJ_PREC_NAME( OA_DEST(oap) ),
				OBJ_NAME( OA_SRC1(oap) ) ,NAME_FOR_PREC_CODE(srcp1));
			WARN(ERROR_STRING);
			return(-1);
	}
	SET_OA_FUNCTYPE( oap, FUNCTYPE_FOR( OA_ARGSPREC(oap) ,OA_ARGSTYPE(oap) ) );
//TELL_FUNCTYPE( OA_ARGSPREC(oap) ,OA_ARGSTYPE(oap) )
} /* end chkprec() */

static int chksiz(QSP_ARG_DECL  Vector_Function *vfp,Vec_Obj_Args *oap)	/* check for argument size match */
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

	if( OA_SBM(oap) != NO_OBJ ){
		if( VF_FLAGS(vfp) & BITMAP_SRC ){	// redundant?
			/* We used to require that the bitmap size matched the destination,
			 * but that is not necessary...
			 */

			/*
			if( (status=old_cksiz(VF_FLAGS(vfp),OA_SBM(oap) ,OA_DEST(oap) ))
					== (-1) )
			*/
			if( (status=cksiz(QSP_ARG  VF_FLAGS(vfp),OA_SBM(oap) ,OA_DEST(oap) ))
				== (-1) )
			{
				sprintf(ERROR_STRING,
			"chksiz:  bitmap arg func size error, function %s",VF_NAME(vfp) );
				ADVISE(ERROR_STRING);
				return(-1);
			}
		}
//#ifdef CAUTIOUS
//		  else if( ! IS_CONVERSION(vfp) && VF_CODE(vfp) != FVMOV ){
//			sprintf(ERROR_STRING,
//		"CAUTIOUS:  chksiz %s:  obj args bitmap is non-null, but function has no bitmap flag!?",
//				VF_NAME(vfp) );
//			ERROR1(ERROR_STRING);
//		}
		else {
			assert( IS_CONVERSION(vfp) || VF_CODE(vfp) == FVMOV );
		}
//		if( status != 0 ){
//			sprintf(ERROR_STRING,"CAUTIOUS:  chksiz %s:  old_cksiz returned status=%d!?",VF_NAME(vfp) ,status);
//			NWARN(ERROR_STRING);
//		}
//#endif /* CAUTIOUS */

		assert( status == 0 );

	}

	if(  OA_SRC1(oap)  == NO_OBJ ){
		/* nothing to check!? */
		return(0);
	}
#ifdef QUIP_DEBUG
if( debug & veclib_debug ){
sprintf(ERROR_STRING,"chksiz:  destv %s (%s)  arg1 %s (%s)",
OBJ_NAME(OA_DEST(oap) ), AREA_NAME(OBJ_AREA(OA_DEST(oap))),
OBJ_NAME( OA_SRC1(oap) ), AREA_NAME(OBJ_AREA(OA_SRC1(oap))) );
ADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	/* We check the sizes of the args against the destination object - but in the case of ops like vdot,
	 * (or any other scalar-returning projection op like vmax etc)
	 * this may not match...
	 */
	if( VF_CODE(vfp) == FVDOT ){
		if( (status=cksiz(QSP_ARG  VF_FLAGS(vfp), OA_SRC1(oap) ,OA_SRC2(oap) )) == (-1) ){
			sprintf(ERROR_STRING,"chksiz:  Size mismatch between arg1 (%s) and arg2 (%s), function %s",
				OBJ_NAME( OA_SRC1(oap) ) ,OBJ_NAME(OA_SRC2(oap) ) ,VF_NAME(vfp) );
			ADVISE(ERROR_STRING);
			return(-1);
		}
		return(0);
	}

	if( (status=cksiz(QSP_ARG  VF_FLAGS(vfp), OA_SRC1(oap) ,OA_DEST(oap) )) == (-1) ){
		sprintf(ERROR_STRING,"chksiz:  Size mismatch between arg1 (%s) and destination (%s), function %s",
			OBJ_NAME( OA_SRC1(oap) ) ,OBJ_NAME(OA_DEST(oap) ) ,VF_NAME(vfp) );
		ADVISE(ERROR_STRING);
		return(-1);
	}
//#ifdef CAUTIOUS
//	if( status != 0 ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  chksiz %s:  cksiz returned status=%d!?",VF_NAME(vfp) ,status);
//		NWARN(ERROR_STRING);
//	}
//#endif /* CAUTIOUS */
	assert( status == 0 );

	if( OA_SRC2(oap)  == NO_OBJ ) return(0);
#ifdef QUIP_DEBUG
if( debug & veclib_debug ){
sprintf(ERROR_STRING,"chksiz:  destv %s (%s)  arg2 %s (%s)",
OBJ_NAME(OA_DEST(oap) ), AREA_NAME(OBJ_AREA(OA_DEST(oap))),
OBJ_NAME(OA_SRC2(oap) ), AREA_NAME(OBJ_AREA(OA_SRC2(oap))) );
ADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( (status=cksiz(QSP_ARG  VF_FLAGS(vfp),OA_SRC2(oap) ,OA_DEST(oap) )) == (-1) ){
		sprintf(ERROR_STRING,"chksiz:  Size mismatch between arg2 (%s) and destination (%s), function %s",
			OBJ_NAME(OA_SRC2(oap) ) ,OBJ_NAME(OA_DEST(oap) ) ,VF_NAME(vfp) );
		ADVISE(ERROR_STRING);
		return(-1);
	}

//#ifdef CAUTIOUS
//	if( status != 0 ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  chksiz %s:  cksiz returned status=%d!?",VF_NAME(vfp) ,status);
//		NWARN(ERROR_STRING);
//		return(-1);
//	}
//#endif /* CAUTIOUS */
	assert( status == 0 );

	/* BUG what about bitmaps?? */
	return(0);
} /* end chksiz() */

/* check that all of the arguments match (when they should) */


static int chkargs( QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap)
{
//#ifdef CAUTIOUS
//	if( OA_DEST(oap)  == NO_OBJ && VF_FLAGS(vfp) & BITMAP_DST ){
////		OA_DEST(oap)  = OA_BMAP(oap) ;
//		ERROR1("CAUTIOUS:  chkargs:  OA_DEST is null, expected a bitmap!?");
//	}
//#endif // CAUTIOUS

	assert( OA_DEST(oap) != NO_OBJ || (VF_FLAGS(vfp) & BITMAP_DST)==0 );

	if( chk_uk(QSP_ARG  vfp,oap) == (-1) ) return(-1);
	if( chktyp(QSP_ARG  vfp,oap) == (-1) ) return(-1);
	if( chkprec(QSP_ARG  vfp,oap) == (-1) ) return(-1);
	if( chksiz(QSP_ARG  vfp,oap) == (-1) ) return(-1);

	/* Now we have to set the function type */

	return(0);
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
	WARN(ERROR_STRING);						\
	return(-1);


#ifdef FOOBAR
int cktype(Data_Obj *dp1,Data_Obj *dp2)
{
	if( dp1->dt_tdim != dp2->dt_tdim ) return(-1);
	else return(0);
}

void wacky_arg(Data_Obj *dp)
{
	sprintf(ERROR_STRING, "%s:  inc = %d, cols = %d",
		OBJ_NAME(dp) , dp->dt_inc, dp->dt_cols );
	NWARN(ERROR_STRING);
	list_dobj(QSP_ARG dp);
	ERROR1("wacky_arg:  can't happen #1");
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

	if( arg_dp == NO_OBJ ) return(0);
	if( IS_EVENLY_SPACED(arg_dp) ) return(0);

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

int perf_vfunc(QSP_ARG_DECL  Vec_Func_Code code, Vec_Obj_Args *oap)
{
	return( call_vfunc(QSP_ARG  FIND_VEC_FUNC(code), oap) );
}

#ifdef HAVE_ANY_GPU
// BUG???  is this redundant now that we have platforms?

static int default_gpu_dispatch(Vector_Function *vfp, Vec_Obj_Args *oap)
{
	sprintf(DEFAULT_ERROR_STRING,"No GPU dispatch function specified, can't call %s",VF_NAME(vfp) );
	NWARN(DEFAULT_ERROR_STRING);
	NADVISE("Please call set_gpu_dispatch_func().");
	return(-1);
}

static int (*gpu_dispatch_func)(Vector_Function *vfp, Vec_Obj_Args *oap)=default_gpu_dispatch;

void set_gpu_dispatch_func( int (*func)(Vector_Function *vfp, Vec_Obj_Args *oap) )
{
//sprintf(ERROR_STRING,"Setting gpu dispatch func (0x%lx)",(int_for_addr)func);
//ADVISE(ERROR_STRING);
	gpu_dispatch_func = func;
}

#endif /* HAVE_ANY_GPU */

int call_vfunc( QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap )
{
	/* Set the default function type.
	 * Why do we use src1 in preference to oa_dest?
	 *
	 * One answer is bitmap result functions...
	 */
	if(  OA_SRC1(oap)  != NO_OBJ ){
		SET_OA_ARGSPREC(oap, ARGSET_PREC(  OBJ_PREC( OA_SRC1(oap) )  ));
	} else if( OA_DEST(oap)  != NO_OBJ ){
		SET_OA_ARGSPREC(oap, ARGSET_PREC( OBJ_PREC( OA_DEST(oap) )  ));
	} else {
		sprintf(ERROR_STRING,"call_vfunc %s:",VF_NAME(vfp) );
		ADVISE(ERROR_STRING);
		ERROR1("call_vfunc:  no prototype vector!?");
	}

//sprintf(ERROR_STRING,"call_vfunc:  function %s",VF_NAME(vfp));
//advise(ERROR_STRING);
//show_obj_args(QSP_ARG  oap);
	/* If we are performing a conversion, we assume that the proper
	 * conversion function has already been selected.
	 * We want to do this efficiently...
	 */
	/* if( IS_CONVERSION(vfp) ) return(0); */

	/* check for precision, type, size matches */
	if( chkargs(QSP_ARG  vfp,oap) == (-1) ) return(-1);	/* make set vslct_fake */

	/* argstype has been set from within chkargs */
	SET_OA_FUNCTYPE( oap, FUNCTYPE_FOR( OA_ARGSPREC(oap) ,OA_ARGSTYPE(oap) ) );
//TELL_FUNCTYPE( OA_ARGSPREC(oap) ,OA_ARGSTYPE(oap) )

	/* We don't worry here about vectorization on CUDA... */

	// Here we should call the platform-specific dispatch function...
	if( check_obj_devices(oap) < 0 )
		return -1;

	assert( OA_PFDEV(oap) != NULL );
/*
fprintf(stderr,"call_vfunc:  oap = 0x%lx  vfp = 0x%lx\n",
(long)oap,(long)vfp );
fprintf(stderr,"call_vfunc:  func at 0x%lx\n",(long)OA_DISPATCH_FUNC(oap));
*/
	//return (* OA_DISPATCH_FUNC( oap ) )(QSP_ARG  vfp,oap);
	return platform_dispatch( QSP_ARG  PFDEV_PLATFORM(OA_PFDEV(oap)), vfp,oap);
} // call_vfunc

