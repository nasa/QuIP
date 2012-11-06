

#include "quip_config.h"

char VersionId_newvec_vec_call[] = QUIP_VERSION_STRING;

#include "nvf.h"
#include <string.h>
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

/* local prototypes */
static int chkprec(QSP_ARG_DECL Vec_Func *vfp,Vec_Obj_Args *argp);
static int chktyp(QSP_ARG_DECL Vec_Func *vfp,Vec_Obj_Args *argp);
static int chksiz(QSP_ARG_DECL  Vec_Func *vfp, Vec_Obj_Args *argp);
static int chkargs(QSP_ARG_DECL Vec_Func *vfp,Vec_Obj_Args *argp);


void shape_error(QSP_ARG_DECL  Vec_Func *vfp, Data_Obj *dp)
{
	sprintf(error_string,"shape_error:  Vector function %s:  argument %s has unknown shape!?",
		vfp->vf_name,dp->dt_name);
	WARN(error_string);
}


static int chk_uk(QSP_ARG_DECL  Vec_Func *vfp, Vec_Obj_Args *oap)
{
	int i;

	if( oap->oa_dest != NO_OBJ && UNKNOWN_SHAPE(&oap->oa_dest->dt_shape) ){
		shape_error(QSP_ARG  vfp,oap->oa_dest);
		return(-1);
	}
	for(i=0;i<MAX_N_ARGS;i++){
		if( oap->oa_dp[i] != NO_OBJ && UNKNOWN_SHAPE(&oap->oa_dp[i]->dt_shape) ){
			shape_error(QSP_ARG  vfp,oap->oa_dp[i]);
			return(-1);
		}
	}
	if( oap->oa_bmap != NO_OBJ && UNKNOWN_SHAPE(&oap->oa_bmap->dt_shape) ){
		shape_error(QSP_ARG  vfp,oap->oa_bmap);
		return(-1);
	}
	/* BUG check the scalar objects too? */
	return(0);
}

/* check that all of the arguments match (when they should) */


static int chkargs( QSP_ARG_DECL  Vec_Func *vfp, Vec_Obj_Args *oap)
{
	if( oap->oa_dest == NO_OBJ && vfp->vf_flags & BITMAP_DST ){
		oap->oa_dest = oap->oa_bmap;
	}

	if( chk_uk(QSP_ARG  vfp,oap) == (-1) ) return(-1);
	if( chktyp(QSP_ARG  vfp,oap) == (-1) ) return(-1);
	if( chkprec(QSP_ARG  vfp,oap) == (-1) ) return(-1);
	if( chksiz(QSP_ARG  vfp,oap) == (-1) ) return(-1);

	/* Now we have to set the function type */

	return(0);
}

static const char *name_for_type(Data_Obj *dp)
{
	if( IS_REAL(dp) ) return("real");
	else if( IS_COMPLEX(dp) ) return("complex");
	else if( IS_QUAT(dp) ) return("quaternion");
#ifdef CAUTIOUS
	else {
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  name_for_type:  type of object %s is unknown",dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return("unknown");
	}
#endif /* CAUTIOUS */
}

/* The "type" is real, complex, quaternion, or mixed...
 * independent of "precision" (byte/short/float etc)
 */

static int chktyp(QSP_ARG_DECL  Vec_Func *vfp,Vec_Obj_Args *oap)
{
	oap->oa_argstype = UNKNOWN_ARGS;

	/* Set the type based on the destination vector */
	/* destv is highest numbered arg */
	if( IS_REAL(oap->oa_dest) ){
		if( oap->oa_2 != NO_OBJ ){	/* two source operands */
			if( IS_REAL(oap->oa_1) ){
				if( IS_REAL(oap->oa_2) ){
					oap->oa_argstype = REAL_ARGS;
				} else if( IS_COMPLEX(oap->oa_2) ){
					oap->oa_argstype = MIXED_ARGS;
				} else if( IS_QUAT(oap->oa_2) ){
					oap->oa_argstype = QMIXED_ARGS;
				} else {
					/* error - type mismatch */
					goto type_mismatch23;
				}
			} else if( IS_COMPLEX(oap->oa_1) ){
				if( IS_REAL(oap->oa_2) ){
					oap->oa_argstype = MIXED_ARGS;
				} else if( IS_COMPLEX(oap->oa_2) ){
					oap->oa_argstype = MIXED_ARGS;
				} else {
					/* error - type mismatch */
					goto type_mismatch23;
				}
			} else if( IS_QUAT(oap->oa_1) ){
				if( IS_REAL(oap->oa_2) ){
					oap->oa_argstype = QMIXED_ARGS;
				} else if( IS_QUAT(oap->oa_2) ){
					oap->oa_argstype = QMIXED_ARGS;
				} else {
					/* error - type mismatch */
					goto type_mismatch23;
				}
			}
#ifdef CAUTIOUS
			  else {
				/* oa_1 is not real or complex, must be a type mismatch */
				goto type_mismatch13;
			}
#endif /* CAUTIOUS */
		} else if( oap->oa_1 != NO_OBJ ){	/* one source operand */
			if( IS_REAL(oap->oa_1) ){
				oap->oa_argstype = REAL_ARGS;
			} else if( IS_COMPLEX(oap->oa_1) ){
				oap->oa_argstype = MIXED_ARGS;
			} else if( IS_QUAT(oap->oa_1) ){
				oap->oa_argstype = QMIXED_ARGS;
			} else {
				/* oa_1 is not real or complex, must be a type mismatch */
				goto type_mismatch12;
			}
		} else {				/* only 1 operand */
			oap->oa_argstype = REAL_ARGS;
		}
	} else if( IS_COMPLEX(oap->oa_dest) ){
		if( oap->oa_2 != NO_OBJ ){	/* two source operands */
			if( IS_COMPLEX(oap->oa_1) ){
				if( IS_COMPLEX(oap->oa_2) ){
					oap->oa_argstype = COMPLEX_ARGS;
				} else if( IS_REAL(oap->oa_2) ){
					oap->oa_argstype = MIXED_ARGS;
				} else {
					/* error - type mismatch */
					goto type_mismatch23;
				}
			} else if( IS_REAL(oap->oa_1) ){
				if( IS_COMPLEX(oap->oa_2) ){
					oap->oa_argstype = MIXED_ARGS;
				/* Should we check for real-real with complex result??? */
				} else {
					/* error - type mismatch */
					goto type_mismatch23;
				}
			} else {
				/* oa_1 is not real or complex, must be a type mismatch */
				goto type_mismatch13;
			}
		} else if( oap->oa_1 != NO_OBJ ){	/* one source operand */
			if( IS_COMPLEX(oap->oa_1) ){
				oap->oa_argstype = COMPLEX_ARGS;
			} else if( IS_REAL(oap->oa_1) ){
				oap->oa_argstype = MIXED_ARGS;
			} else {
				/* oa_1 is not real or complex, must be a type mismatch */
				goto type_mismatch12;
			}
		} else {				/* only 1 operand */
			oap->oa_argstype = COMPLEX_ARGS;
		}
	} else if( IS_QUAT(oap->oa_dest) ){
		if( oap->oa_2 != NO_OBJ ){	/* two source operands */
			if( IS_QUAT(oap->oa_1) ){
				if( IS_QUAT(oap->oa_2) ){
					oap->oa_argstype = QUATERNION_ARGS;
				} else if( IS_REAL(oap->oa_2) ){
					oap->oa_argstype = QMIXED_ARGS;
				} else {
					/* error - type mismatch */
					goto type_mismatch23;
				}
			} else if( IS_REAL(oap->oa_1) ){
				if( IS_QUAT(oap->oa_2) ){
					oap->oa_argstype = QMIXED_ARGS;
				/* Should we check for real-real with complex result??? */
				} else {
					/* error - type mismatch */
					goto type_mismatch23;
				}
			} else {
				/* oa_1 is not real or complex, must be a type mismatch */
				goto type_mismatch13;
			}
		} else if( oap->oa_1 != NO_OBJ ){	/* one source operand */
			if( IS_QUAT(oap->oa_1) ){
				oap->oa_argstype = QUATERNION_ARGS;
			} else if( IS_REAL(oap->oa_1) ){
				oap->oa_argstype = QMIXED_ARGS;
			} else {
				/* oa_1 is not real or complex, must be a type mismatch */
				goto type_mismatch12;
			}
		} else {				/* only 1 operand */
			oap->oa_argstype = QUATERNION_ARGS;
		}
	} else {
		sprintf(error_string,"chktyp:  can't categorize destination object %s!?",oap->oa_dest->dt_name);
		WARN(error_string);
	}

	/* now the type field has been set - make sure it's legal */
	/* But first check a couple of special cases */

	/* make sure that function doesn't require mixed types */

	if( vfp->vf_flags & CPX_2_REAL ){
		if( ! IS_COMPLEX(oap->oa_1) ){
			sprintf(error_string,"source vector %s (%s) must be complex with function %s",
				oap->oa_1->dt_name,name_for_prec(oap->oa_dest->dt_prec),vfp->vf_name);
			WARN(error_string);
			listone(oap->oa_1);
			return(-1);
		}
		if( ! IS_REAL(oap->oa_dest) ){
			sprintf(error_string,"destination vector %s (%s) must be real with function %s",
				oap->oa_dest->dt_name,name_for_prec(oap->oa_dest->dt_prec),vfp->vf_name);
			WARN(error_string);
			listone(oap->oa_dest);
			return(-1);
		}
		oap->oa_argstype = REAL_ARGS;
		return(0);
	}

	if( vfp->vf_code == FVFFT ){
		/* source vector can be real or complex */
		if( !IS_COMPLEX(oap->oa_dest) ){
			WARN("destination must be complex for fft");
			return(-1);
		}
		if( IS_COMPLEX(oap->oa_1) )
			oap->oa_argstype=COMPLEX_ARGS;
		else if( IS_QUAT(oap->oa_1) ){
			WARN("Can't compute FFT of a quaternion input");
			return(-1);
		} else
			oap->oa_argstype=REAL_ARGS;

		return(0);
	}

	if( vfp->vf_code == FVIFT ){
		/* destination vector can be real or complex */
		if( !IS_COMPLEX(oap->oa_1) ){
			WARN("source must be complex for inverse fft");
			return(-1);
		}
		if( IS_COMPLEX(oap->oa_dest) )
			oap->oa_argstype=COMPLEX_ARGS;
		else if( IS_QUAT(oap->oa_dest) ){
			WARN("Can't compute inverse FFT to a quaternion target");
			return(-1);
		} else
			oap->oa_argstype=REAL_ARGS;

		return(0);
	}

	/* now the type field has been set - make sure it's legal */
	if( (vfp->vf_typemask & VL_TYPE_MASK(oap->oa_argstype) )==0 ){
		sprintf(error_string,
	"chktyp:  Arguments of type %s are not permitted with function %s",
			argset_type_name[oap->oa_argstype],vfp->vf_name);
		WARN(error_string);
		return(-1);
	}

/*
sprintf(error_string,"function %s:  oa_argstype = %s",vfp->vf_name,argset_type_name[oap->oa_argstype]);
ADVISE(error_string);
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

	if( HAS_MIXED_ARGS(oap) && ! IS_COMPLEX(oap->oa_dest) ){
		sprintf(error_string,"chktyp:  destination vector %s must be complex when mixing types with function %s",
			oap->oa_dest->dt_name,vfp->vf_name);
		WARN(error_string);
		return(-1);
	}

	if( HAS_QMIXED_ARGS(oap) == QMIXED_ARGS && ! IS_QUAT(oap->oa_dest) ){
		sprintf(error_string,"chktyp:  destination vector %s must be quaternion when mixing types with function %s",
			oap->oa_dest->dt_name,vfp->vf_name);
		WARN(error_string);
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
		if( USES_REAL_SCALAR(vfp->vf_code) ){
			if( ! IS_COMPLEX(oap->oa_2) ){
				WARN("destination vector must be complex when mixing types with vsmul");
				return(-1);
			}
			if( ! IS_REAL(oap->oa_1) ){
				WARN("source vector must be real when mixing types with vsmul");
				return(-1);
			}
		} else if( USES_COMPLEX_SCALAR(vfp->vf_code) ){
			if( ! IS_COMPLEX(oap->oa_2) ){
				WARN("destination vector must be complex when mixing types with vcsmul");
				return(-1);
			}
			if( ! IS_REAL(oap->oa_1) ){
				WARN("source vector must be real when mixing types with vcsmul");
				return(-1);
			}
		} else if( USES_QUAT_SCALAR(vfp->vf_code) ){
			if( ! IS_QUAT(oap->oa_2) ){
				WARN("destination vector must be quaternion when mixing types with vqsmul");
				return(-1);
			}
			if( ! IS_REAL(oap->oa_1) ){
				WARN("source vector must be real when mixing types with vqsmul");
				return(-1);
			}
		}
#endif /* FOOBAR */
	
		/*
		show_obj_args(oap);
sprintf(error_string,"Function %s.",vfp->vf_name);
ADVISE(error_string);
		ERROR1("chktyp:  Sorry, not sure how to deal with this situation...");
		*/

		if( ! IS_COMPLEX(oap->oa_1) ){
			sprintf(error_string,
"first source vector (%s,%s) must be complex when mixing types with function %s",
				oap->oa_1->dt_name,
				name_for_type(oap->oa_1),
				vfp->vf_name);
			WARN(error_string);
			return(-1);
		}
		if( ! IS_REAL(oap->oa_2) ){
			sprintf(error_string,
"second source vector (%s,%s) must be real when mixing types with function %s",
				oap->oa_2->dt_name,
				name_for_type(oap->oa_2),
				vfp->vf_name);
			WARN(error_string);
			return(-1);
		}
		/* Should the destination be complex??? */
	} else if( HAS_QMIXED_ARGS(oap) ){
		ERROR1("FIXME:  need to add code for quaternions, vec_call.c");
		/* BUG add the same check as above for quaternions */
	}

	return(0);

type_mismatch13:
	sprintf(error_string,"Type mismatch between objects %s (%s) and %s (%s), function %s",
		oap->oa_1->dt_name,name_for_type(oap->oa_1),
		oap->oa_3->dt_name,name_for_type(oap->oa_3),
		vfp->vf_name);
	WARN(error_string);
	return(-1);

type_mismatch12:
	sprintf(error_string,"Type mismatch between objects %s (%s) and %s (%s), function %s",
		oap->oa_1->dt_name,name_for_type(oap->oa_1),
		oap->oa_2->dt_name,name_for_type(oap->oa_2),
		vfp->vf_name);
	WARN(error_string);
	return(-1);

type_mismatch23:
	sprintf(error_string,"Type mismatch between objects %s (%s) and %s (%s), function %s",
		oap->oa_2->dt_name,name_for_type(oap->oa_2),
		oap->oa_3->dt_name,name_for_type(oap->oa_3),
		vfp->vf_name);
	WARN(error_string);
	return(-1);
} /* end chktyp() */


/* chkprec() now has the job of figuring out mixed precision op's */

/* Instead of using the prec_mask from the table, we can figure out what
 * function we want and see if it is not equal to nullf...
 *
 */

#define PREC_ERROR_MSG( prec )								\
											\
	sprintf(error_string,								\
	"chkprec:  %s:  input %s (%s) should have %s or %s precision with target %s (%s)",	\
		vfp->vf_name,oap->oa_1->dt_name,name_for_prec(oap->oa_1->dt_prec),	\
		prec_name[ prec ],prec_name[dst_prec],					\
		oap->oa_dest->dt_name,name_for_prec(oap->oa_dest->dt_prec));		\
	WARN(error_string);								\
	return(-1);

#define NEW_PREC_ERROR_MSG( prec )								\
											\
	sprintf(error_string,								\
	"chkprec:  %s:  input %s (%s) should have %s or %s precision with target %s (%s)",	\
		vfp->vf_name,oap->oa_1->dt_name,name_for_prec(oap->oa_1->dt_prec),	\
		prec_name[ prec ],prec_name[dst_prec],					\
		oap->oa_dest->dt_name,name_for_prec(oap->oa_dest->dt_prec));		\
	WARN(error_string);								\
	return(-1);

static void show_legal_precisions(uint32_t mask)
{
	uint32_t bit=1;
	prec_t prec;

	NADVISE("legal precisions are:");
	for( prec = 0; prec < 32 ; prec ++ ){
		bit = 1 << prec ;
		if( mask & bit ){
			sprintf(DEFAULT_ERROR_STRING,"\t%s",name_for_prec(prec));
			NADVISE(DEFAULT_ERROR_STRING);
		}
	}
}

/* chkprec sets two flags:
 * oa_argsprec (tells machine prec),
 * and argstype (real/complex etc)
 *
 * But isn't argstype set by chktyp???
 */

 /* end chkprec */

static int chkprec(QSP_ARG_DECL  Vec_Func *vfp,Vec_Obj_Args *oap)
{
	prec_t srcp1,srcp2,dst_prec;
	int n_srcs=0;
	if( IS_CONVERSION(vfp) ){
		/* Conversions support all the rpecisions, so the checks
		 * after this block are irrelevant.
		 */

		/* the default function type is set using oa_1 (the source),
		 * but if the target is a bitmap we want to set it to bitmap...
		 */
		if( IS_BITMAP(oap->oa_dest) ){
/*
ADVISE("chkprec:  Setting argstype to R_BIT_ARGS!?");
			oap->oa_argstype = R_BIT_ARGS;
*/
			/* R_BIT_ARGS was a functype - not an argset type??? */
			oap->oa_argsprec = BIT_ARGS;
		} else if( IS_BITMAP(oap->oa_1) ){
			/* this is necessary because bitmaps handled with kludgy hacks */
			oap->oa_bmap = oap->oa_1;
		}
		return(0);
	}

	dst_prec=MACHINE_PREC(oap->oa_dest);
	/* BUG? could be bitmap destination??? */
	/* need to find out which prec to test... */

	/* First we make sure that all arg precisions
	 * are legal with this function
	 */

	if( (vfp->vf_precmask & (1<<dst_prec)) == 0  ){
		sprintf(error_string,
"chkprec:  dest. precision %s (obj %s) cannot be used with function %s",
			prec_name[dst_prec],oap->oa_dest->dt_name,vfp->vf_name);
		WARN(error_string);
		show_legal_precisions(vfp->vf_precmask);
		return(-1);
	}

	if( oap->oa_1 != NO_OBJ ){
		srcp1=MACHINE_PREC(oap->oa_1);
		if( (vfp->vf_precmask & (1<<srcp1)) == 0 ){
			sprintf(error_string,
"chkprec:  src precision %s (obj %s) cannot be used with function %s",
		prec_name[srcp1],oap->oa_1->dt_name,vfp->vf_name);
			WARN(error_string);
			show_legal_precisions(vfp->vf_precmask);
			return(-1);
		}
		n_srcs++;
		if( oap->oa_2 != NO_OBJ ){
			srcp2=MACHINE_PREC(oap->oa_2);
			if( (vfp->vf_precmask & (1<<srcp2)) == 0 ){
				sprintf(error_string,
"chkprec:  src precision %s (obj %s) cannot be used with function %s",
			prec_name[srcp2],oap->oa_2->dt_name,vfp->vf_name);
				WARN(error_string);
				show_legal_precisions(vfp->vf_precmask);
				return(-1);
			}
			n_srcs++;
		}
	}

	/* Figure out what type of function to call based on the arguments... */

	if( n_srcs == 0 ){
		/* oa_argstype is Function_Type...
		 * is this right? just a null setting?
		 */

		/* we used to use dst_prec here, but that
		 * is only the machine precision!?
		 */
		oap->oa_argsprec = ARGSET_PREC(oap->oa_dest->dt_prec);
		return(0);
	} else if( n_srcs == 2 ){
		/* First make sure that the two source operands match */
		if( srcp1 != srcp2 ) {
source_mismatch_error:
			sprintf(error_string,
"chkprec:  %s operands %s (%s) and %s (%s) should have the same precision",
				vfp->vf_name,oap->oa_1->dt_name,
				name_for_prec(oap->oa_1->dt_prec),
				oap->oa_2->dt_name,
				name_for_prec(oap->oa_2->dt_prec) );
			WARN(error_string);
			return(-1);
		}
		/* if the precision is long, make sure that
		 * none (or all) are bitmaps
		 */
		if( srcp1 == BITMAP_MACH_PREC ){
			if( (IS_BITMAP(oap->oa_1) && ! IS_BITMAP(oap->oa_2)) ||
			    ( ! IS_BITMAP(oap->oa_1) && IS_BITMAP(oap->oa_2)) )
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
			if( IS_BITMAP(oap->oa_dest) && !IS_BITMAP(oap->oa_1) )
				goto next1;
			if( IS_BITMAP(oap->oa_1) && !IS_BITMAP(oap->oa_dest) )
				goto next1;
		}
		/* Can't use dst_prec here because might be bitmap */
		oap->oa_argsprec = ARGSET_PREC(oap->oa_dest->dt_prec);
		return(0);
	}
next1:

	/* Now we know that this is a mixed precision case.
	 * Make sure it is one of the legal ones.
	 * First we check the special cases (bitmaps, indices).
	 */
	if( vfp->vf_flags & BITMAP_DST ){		/* vcmp, vcmpm */
		/* Is dest vector set too??? */
		if( oap->oa_dest->dt_prec != PREC_BIT ){
			sprintf(error_string,
		"%s:  result vector %s (%s) should have %s precision",
				vfp->vf_name,oap->oa_dest->dt_name,
				name_for_prec(oap->oa_dest->dt_prec),
				prec_name[N_MACHINE_PRECS+PP_BIT]);
			WARN(error_string);
			return(-1);
		}
		/* use the precision from the source */
		oap->oa_argsprec = ARGSET_PREC( oap->oa_1->dt_prec );
		return(0);
	}
	if( vfp->vf_flags == V_SCALRET2 ){ /* vmaxg etc */
		/* We assme that this is an index array and
		 * not a bitmap.
		 */
		if( oap->oa_dest->dt_prec != PREC_DI ){
			sprintf(error_string,
"chkprec:  %s:  destination vector %s (%s) should have %s precision",
				vfp->vf_name,oap->oa_dest->dt_name,
				name_for_prec(oap->oa_dest->dt_prec),
				prec_name[PREC_DI] );
			WARN(error_string);
			return(-1);
		}
		/* If the destination is long, don't worry about
		 * a match with the arg...
		 */
		return(0);
	}

	/* don't insist on a precision match if result is an index */
	if( vfp->vf_flags & INDEX_RESULT ){
		/* We assume that we check the result precision elsewhere? */
		return(0);
	}

	switch( dst_prec ){
		case PREC_IN:
			if( srcp1==PREC_UBY ){
				oap->oa_argsprec = BYIN_ARGS;
				return(0);
			}
			NEW_PREC_ERROR_MSG(PREC_UBY);
			break;
		case PREC_DP:
			if( srcp1==PREC_SP ){
				oap->oa_argsprec = SPDP_ARGS;
				return(0);
			}
			NEW_PREC_ERROR_MSG(PREC_SP);
			break;
		case PREC_DI:
			if( srcp1==PREC_UIN ){
				oap->oa_argsprec = INDI_ARGS;
				return(0);
			}
			NEW_PREC_ERROR_MSG(PREC_UIN);
			break;
		case PREC_BY:
			if( srcp1==PREC_IN ){
				oap->oa_argsprec = INBY_ARGS;
				return(0);
			}
			NEW_PREC_ERROR_MSG(PREC_IN);
			break;
		default:
			sprintf(error_string,
"chkprec:  %s:  target %s (%s) cannot be used with mixed prec source %s (%s)",
				vfp->vf_name,oap->oa_dest->dt_name,
				name_for_prec(oap->oa_dest->dt_prec),
				oap->oa_1->dt_name,prec_name[srcp1]);
			WARN(error_string);
			return(-1);
	}
	oap->oa_functype = FUNCTYPE_FOR(oap->oa_argsprec,oap->oa_argstype);
//TELL_FUNCTYPE(oap->oa_argsprec,oap->oa_argstype)
} /* end chkprec() */



static int chksiz(QSP_ARG_DECL  Vec_Func *vfp,Vec_Obj_Args *oap)	/* check for argument size match */
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

	if( oap->oa_bmap != NO_OBJ ){
		if( vfp->vf_flags & BITMAP_DST ){
#ifdef FOOBAR
			if( (status=old_cksiz(QSP_ARG  vfp->vf_flags,oap->oa_bmap,oap->oa_1)) == (-1) ){
				sprintf(error_string,"chksiz:  bitmap result func size error, function %s",vfp->vf_name);
				ADVISE(error_string);
				return(-1);
			}
#endif /* FOOBAR */
			/* We used to require an exact size match for bitmap destinations, but no longer */
			/* proceed to the new tests further down... */
		} else if( vfp->vf_flags & BITMAP_SRC ){
			/* We used to require that the bitmap size matched the destination,
			 * but that is not necessary...
			 */

			/*
			if( (status=old_cksiz(vfp->vf_flags,oap->oa_bmap,oap->oa_dest))
					== (-1) )
			*/
			if( (status=cksiz(QSP_ARG  vfp->vf_flags,oap->oa_bmap,oap->oa_dest))
				== (-1) )
			{
				sprintf(error_string,
			"chksiz:  bitmap arg func size error, function %s",vfp->vf_name);
				ADVISE(error_string);
				return(-1);
			}
		}
#ifdef CAUTIOUS
		  else if( ! IS_CONVERSION(vfp) ){
			sprintf(error_string,
		"CAUTIOUS:  chksiz %s:  obj args bitmap is non-null, but function has no bitmap flag!?",
				vfp->vf_name);
			ERROR1(error_string);
		}
		if( status != 0 ){
			sprintf(error_string,"CAUTIOUS:  chksiz %s:  old_cksiz returned status=%d!?",vfp->vf_name,status);
			NWARN(error_string);
		}
#endif /* CAUTIOUS */

	}

	if( oap->oa_1 == NO_OBJ ){
		/* nothing to check!? */
		return(0);
	}
#ifdef DEBUG
if( debug & veclib_debug ){
sprintf(error_string,"chksiz:  destv %s  arg1 %s",
oap->oa_dest->dt_name,oap->oa_1->dt_name);
ADVISE(error_string);
}
#endif /* DEBUG */

	/* We check the sizes of the args against the destination object - but in the case of ops like vdot,
	 * (or any other scalar-returning projection op like vmax etc)
	 * this may not match...
	 */
	if( vfp->vf_code == FVDOT ){
		if( (status=cksiz(QSP_ARG  vfp->vf_flags,oap->oa_1,oap->oa_2)) == (-1) ){
			sprintf(error_string,"chksiz:  Size mismatch between arg1 (%s) and arg2 (%s), function %s",
				oap->oa_1->dt_name,oap->oa_2->dt_name,vfp->vf_name);
			ADVISE(error_string);
			return(-1);
		}
		return(0);
	}

	if( (status=cksiz(QSP_ARG  vfp->vf_flags,oap->oa_1,oap->oa_dest)) == (-1) ){
		sprintf(error_string,"chksiz:  Size mismatch between arg1 (%s) and destination (%s), function %s",
			oap->oa_1->dt_name,oap->oa_dest->dt_name,vfp->vf_name);
		ADVISE(error_string);
		return(-1);
	}
#ifdef CAUTIOUS
	if( status != 0 ){
		sprintf(error_string,"CAUTIOUS:  chksiz %s:  cksiz returned status=%d!?",vfp->vf_name,status);
		NWARN(error_string);
	}
#endif /* CAUTIOUS */
	if( oap->oa_2 == NO_OBJ ) return(0);
#ifdef DEBUG
if( debug & veclib_debug ){
sprintf(error_string,"chksiz:  destv %s  arg2 %s",
oap->oa_dest->dt_name,oap->oa_2->dt_name);
ADVISE(error_string);
}
#endif /* DEBUG */

	if( (status=cksiz(QSP_ARG  vfp->vf_flags,oap->oa_2,oap->oa_dest)) == (-1) ){
		sprintf(error_string,"chksiz:  Size mismatch between arg2 (%s) and destination (%s), function %s",
			oap->oa_2->dt_name,oap->oa_dest->dt_name,vfp->vf_name);
		ADVISE(error_string);
		return(-1);
	}

#ifdef CAUTIOUS
	if( status != 0 ){
		sprintf(error_string,"CAUTIOUS:  chksiz %s:  cksiz returned status=%d!?",vfp->vf_name,status);
		NWARN(error_string);
		return(-1);
	}
#endif /* CAUTIOUS */
	/* BUG what about bitmaps?? */
	return(0);
} /* end chksiz() */


#ifdef FOOBAR
int cktype(Data_Obj *dp1,Data_Obj *dp2)
{
	if( dp1->dt_tdim != dp2->dt_tdim ) return(-1);
	else return(0);
}

void wacky_arg(Data_Obj *dp)
{
	sprintf(error_string, "%s:  inc = %d, cols = %d",
		dp->dt_name, dp->dt_inc, dp->dt_cols );
	NWARN(error_string);
	listone(dp);
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

	arg_dp = oap->oa_dp[index] ;

	if( arg_dp == NO_OBJ ) return(0);
	if( IS_EVENLY_SPACED(arg_dp) ) return(0);

	/* If the object is subscripted, the brackets will break the name */
	sprintf(tmp_name,"%s.dup",remove_brackets(arg_dp->dt_name));
	new_dp = dup_obj(arg_dp,tmp_name);
	dp_copy(new_dp,arg_dp);	/* BUG use vmov */
	if( oap->oa_dest == arg_dp )
		oap->oa_dest = new_dp;
	oap->oa_dp[index] = new_dp;

	return(1);
}
#endif /* FOOBAR */

int perf_vfunc(QSP_ARG_DECL  Vec_Func_Code code, Vec_Obj_Args *oap)
{
	return( call_vfunc(QSP_ARG  &vec_func_tbl[code],oap) );
}

#ifdef HAVE_CUDA

int default_gpu_dispatch(Vec_Func *vfp, Vec_Obj_Args *oap)
{
	sprintf(DEFAULT_ERROR_STRING,"No GPU dispatch function specified, can't call %s",vfp->vf_name);
	NWARN(DEFAULT_ERROR_STRING);
	ADVISE("Please call set_gpu_dispatch_func().");
	return(-1);
}

static int (*gpu_dispatch_func)(Vec_Func *vfp, Vec_Obj_Args *oap)=default_gpu_dispatch;

void set_gpu_dispatch_func( int (*func)(Vec_Func *vfp, Vec_Obj_Args *oap) )
{
//sprintf(error_string,"Setting gpu dispatch func (0x%lx)",(int_for_addr)func);
//ADVISE(error_string);
	gpu_dispatch_func = func;
}

#endif /* HAVE_CUDA */

int call_vfunc( QSP_ARG_DECL  Vec_Func *vfp, Vec_Obj_Args *oap )
{
	/* Set the default function type.
	 * Why do we use src1 in preference to oa_dest?
	 *
	 * One answer is bitmap result functions...
	 */
	if( oap->oa_1 != NO_OBJ ){
		oap->oa_argsprec = ARGSET_PREC( oap->oa_1->dt_prec );
	} else if( oap->oa_dest != NO_OBJ ){
		oap->oa_argsprec = ARGSET_PREC( oap->oa_dest->dt_prec );
	} else {
		sprintf(error_string,"call_vfunc %s:",vfp->vf_name);
		ADVISE(error_string);
		ERROR1("call_vfunc:  no prototype vector!?");
	}

	/* If we are performing a conversion, we assume that the proper
	 * conversion function has already been selected.
	 * We want to do this efficiently...
	 */
	/* if( IS_CONVERSION(vfp) ) return(0); */

	/* check for precision, type, size matches */
	if( chkargs(QSP_ARG  vfp,oap) == (-1) ) return(-1);	/* make set vslct_fake */

	/* argstype has been set from within chkargs */
	oap->oa_functype = FUNCTYPE_FOR(oap->oa_argsprec,oap->oa_argstype);
//TELL_FUNCTYPE(oap->oa_argsprec,oap->oa_argstype)

	/* We don't worry here about vectorization on CUDA... */
#ifdef HAVE_CUDA
	if( are_gpu_args(oap) ){
		int status;
		status = (*gpu_dispatch_func)(vfp,oap);
		return(status);
	}
#endif /* HAVE_CUDA */

	if( vfp->vf_code == FVSET && IS_BITMAP(oap->oa_dest) )
		oap->oa_bmap = oap->oa_dest;

#ifdef HAVE_CUDA
	if( ! are_ram_args(oap) ){		// Make sure all objs in RAM
		mixed_location_error(QSP_ARG  vfp,oap);
		return(-1);
	}
#endif /* HAVE_CUDA */

	vec_dispatch(QSP_ARG  vfp,oap);

	/* This should be done in the object method instead? */
	if( oap->oa_dest != NO_OBJ )
		oap->oa_dest->dt_flags |= DT_ASSIGNED;
	
	/* This should be done in the object method? */
	if( vfp->vf_flags & TWO_SCALAR_RESULTS ){
		oap->oa_s1->dt_flags |= DT_ASSIGNED;
		oap->oa_s2->dt_flags |= DT_ASSIGNED;
	}

	return(0);
}


