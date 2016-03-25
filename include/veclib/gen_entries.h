#ifndef _ENTRIES_H_ 
#define _ENTRIES_H_ 

// these macros should go into host_call_utils.h or something...

/* We used to do a bunch of switches here...
 * But we'd like to replace that with a table jump.
 * Then we can "compile" the table indices, which is not a great big
 * speedup, but it may simplify things...
 */

/*
 *	SIMPLE_PREC_SWITCH	all machine precs, switches on src1
 *	FLOAT_PREC_SWITCH	sp, dp
 *	SWITCH2			sp, dp, switches on dest
 *	RAMP_SWITCH		looks a lot like SIMPLE_PREC_SWITCH,
 *				but switches on dest
 *	RCQM_SWITCH		real/cpx/quat/mixed
 *	RC_SIGNED_SWITCH	not sure what this is for?
 *	RC_SWITCH		what is this for?
 *	RC_SWITCH_SSE
 *	RCQ_SWITCH_SSE
 *	CPX_SWITCH		only cpx, two precisions
 *	QUAT_SWITCH		what is this for?
 *	RCQ_SWITCH		not sure what this is for...
 *	INT_SWITCH
 */

/* SIMPLE_PREC_SWITCH
 * All precisions, but only one type (implicit)
 * OLD COMMENT:  some real, some complex. - WHAT DOES THAT MEAN?
 */

#define SIMPLE_PREC_SWITCH(func,prot_dp)					\
									\
switch( OBJ_MACH_PREC(prot_dp) ){				\
	case PREC_BY:  HOST_TYPED_CALL_NAME(func,by)(HOST_CALL_ARGS); break;	\
	case PREC_IN:  HOST_TYPED_CALL_NAME(func,in)(HOST_CALL_ARGS); break;	\
	case PREC_DI:  HOST_TYPED_CALL_NAME(func,di)(HOST_CALL_ARGS); break;	\
	case PREC_LI:  HOST_TYPED_CALL_NAME(func,li)(HOST_CALL_ARGS); break;	\
	case PREC_SP:  HOST_TYPED_CALL_NAME(func,sp)(HOST_CALL_ARGS); break;	\
	case PREC_DP:  HOST_TYPED_CALL_NAME(func,dp)(HOST_CALL_ARGS); break;	\
	case PREC_UBY:  HOST_TYPED_CALL_NAME(func,uby)(HOST_CALL_ARGS); break;	\
	case PREC_UIN:  HOST_TYPED_CALL_NAME(func,uin)(HOST_CALL_ARGS); break;	\
	case PREC_UDI:  HOST_TYPED_CALL_NAME(func,udi)(HOST_CALL_ARGS); break;	\
	case PREC_ULI:  HOST_TYPED_CALL_NAME(func,uli)(HOST_CALL_ARGS); break;	\
	default:							\
		sprintf(DEFAULT_ERROR_STRING,				\
	"SIMPLE_PREC_SWITCH (%s):  missing precision case",#func);	\
		NWARN(DEFAULT_ERROR_STRING);				\
		break;							\
}

/* REAL_SWITCH - like SIMPLE_PREC_SWITCH, but calls real func...
 *
 * The get bit_rvset to work properly, we have to know which machine precision
 * is used for the bitmap words.  On the cpu we want to use the biggest word
 * supported by the machine, while on the gpu it might be more efficient to
 * use a 32 bit word.  For now we assume that ULI is the bitmap word size.
 * Perhaps we should put a CAUTIOUS check on that here?
 */

#define REAL_SWITCH(func,prot_dp)							\
											\
/*fprintf(stderr,"REAL_SWITCH %s, prot_dp = %s, mach_prec = %s, prec = %s\n",\
#func,OBJ_NAME(prot_dp),PREC_NAME(PREC_FOR_CODE(OBJ_MACH_PREC(prot_dp))),\
PREC_NAME(OBJ_PREC_PTR(prot_dp)));*/\
											\
switch( OBJ_MACH_PREC(prot_dp) ){							\
	case PREC_BY:  HOST_TYPED_CALL_NAME_REAL(func,by)(HOST_CALL_ARGS); break;	\
	case PREC_IN:  HOST_TYPED_CALL_NAME_REAL(func,in)(HOST_CALL_ARGS); break;	\
	case PREC_DI:  HOST_TYPED_CALL_NAME_REAL(func,di)(HOST_CALL_ARGS); break;	\
	case PREC_LI:  HOST_TYPED_CALL_NAME_REAL(func,li)(HOST_CALL_ARGS); break;	\
	case PREC_SP:  HOST_TYPED_CALL_NAME_REAL(func,sp)(HOST_CALL_ARGS); break;	\
	case PREC_DP:  HOST_TYPED_CALL_NAME_REAL(func,dp)(HOST_CALL_ARGS); break;	\
	case PREC_UBY:  HOST_TYPED_CALL_NAME_REAL(func,uby)(HOST_CALL_ARGS); break;	\
	case PREC_UIN:  HOST_TYPED_CALL_NAME_REAL(func,uin)(HOST_CALL_ARGS); break;	\
	case PREC_UDI:  HOST_TYPED_CALL_NAME_REAL(func,udi)(HOST_CALL_ARGS); break;	\
	case PREC_ULI:	HOST_TYPED_CALL_NAME_REAL(func,uli)(HOST_CALL_ARGS); break;	\
	default:									\
		sprintf(DEFAULT_ERROR_STRING,						\
	"REAL_SWITCH (%s):  missing precision case (obj %s, prec %s)",			\
		#func, OBJ_NAME(prot_dp), OBJ_MACH_PREC_NAME(prot_dp) );		\
		NWARN(DEFAULT_ERROR_STRING);						\
		break;									\
}


#define REAL_SWITCH_B(func,prot_dp)							\
											\
/*fprintf(stderr,"REAL_SWITCH_B %s, prot_dp = %s, mach_prec = %s, prec = %s\n",	\
#func,OBJ_NAME(prot_dp),PREC_NAME(PREC_FOR_CODE(OBJ_MACH_PREC(prot_dp))),	\
PREC_NAME(OBJ_PREC_PTR(prot_dp)));*/						\
											\
switch( OBJ_MACH_PREC(prot_dp) ){							\
	case PREC_BY:  HOST_TYPED_CALL_NAME_REAL(func,by)(HOST_CALL_ARGS); break;	\
	case PREC_IN:  HOST_TYPED_CALL_NAME_REAL(func,in)(HOST_CALL_ARGS); break;	\
	case PREC_DI:  HOST_TYPED_CALL_NAME_REAL(func,di)(HOST_CALL_ARGS); break;	\
	case PREC_LI:  HOST_TYPED_CALL_NAME_REAL(func,li)(HOST_CALL_ARGS); break;	\
	case PREC_SP:  HOST_TYPED_CALL_NAME_REAL(func,sp)(HOST_CALL_ARGS); break;	\
	case PREC_DP:  HOST_TYPED_CALL_NAME_REAL(func,dp)(HOST_CALL_ARGS); break;	\
	case PREC_UBY:  HOST_TYPED_CALL_NAME_REAL(func,uby)(HOST_CALL_ARGS); break;	\
	case PREC_UIN:  HOST_TYPED_CALL_NAME_REAL(func,uin)(HOST_CALL_ARGS); break;	\
	case PREC_UDI:  HOST_TYPED_CALL_NAME_REAL(func,udi)(HOST_CALL_ARGS); break;	\
	case PREC_ULI:									\
		if( PREC_CODE(OBJ_PREC_PTR(prot_dp)) == PREC_BIT )			\
			HOST_TYPED_CALL_NAME_REAL(func,bit)(HOST_CALL_ARGS);		\
		else									\
			HOST_TYPED_CALL_NAME_REAL(func,uli)(HOST_CALL_ARGS);		\
		break;									\
	default:							\
		sprintf(DEFAULT_ERROR_STRING,				\
	"REAL_SWITCH_B (%s):  missing precision case (obj %s, prec %s)",	\
		#func, OBJ_NAME(prot_dp), OBJ_MACH_PREC_NAME(prot_dp) );	\
		NWARN(DEFAULT_ERROR_STRING);				\
		break;							\
}


#define REAL_FLOAT_SWITCH(func,prot_dp)						\
									\
switch( OBJ_MACH_PREC(prot_dp) ){						\
	case PREC_SP:  HOST_TYPED_CALL_NAME_REAL(func,sp)(HOST_CALL_ARGS); break;	\
	case PREC_DP:  HOST_TYPED_CALL_NAME_REAL(func,dp)(HOST_CALL_ARGS); break;	\
	default:							\
		sprintf(DEFAULT_ERROR_STRING,				\
	"REAL_FLOAT_SWITCH (%s):  missing precision case (obj %s, prec %s)",	\
		#func, OBJ_NAME(prot_dp), OBJ_MACH_PREC_NAME(prot_dp) );	\
		NWARN(DEFAULT_ERROR_STRING);				\
		break;							\
}

#define REAL_SWITCH_SSE(func,prot_dp)						\
										\
switch( OBJ_MACH_PREC(prot_dp) ){							\
	case PREC_BY:	HOST_TYPED_CALL_NAME_REAL(func,by)(HOST_CALL_ARGS); break;		\
	case PREC_IN:	HOST_TYPED_CALL_NAME_REAL(func,in)(HOST_CALL_ARGS); break;		\
	case PREC_DI:	HOST_TYPED_CALL_NAME_REAL(func,di)(HOST_CALL_ARGS); break;		\
	case PREC_LI:	HOST_TYPED_CALL_NAME_REAL(func,li)(HOST_CALL_ARGS); break;		\
	case PREC_SP:	SP_SSE_SWITCH(func) break;				\
	case PREC_DP:	HOST_TYPED_CALL_NAME_REAL(func,dp)(HOST_CALL_ARGS); break;		\
	case PREC_UBY:	HOST_TYPED_CALL_NAME_REAL(func,uby)(HOST_CALL_ARGS); break;	\
	case PREC_UIN:	HOST_TYPED_CALL_NAME_REAL(func,uin)(HOST_CALL_ARGS); break;	\
	case PREC_UDI:	HOST_TYPED_CALL_NAME_REAL(func,udi)(HOST_CALL_ARGS); break;	\
	case PREC_ULI:	HOST_TYPED_CALL_NAME_REAL(func,uli)(HOST_CALL_ARGS); break;	\
	default:								\
		sprintf(DEFAULT_ERROR_STRING,					\
	"REAL_SWITCH (%s):  missing precision case (obj %s, prec %s)",		\
		#func, OBJ_NAME(prot_dp), OBJ_MACH_PREC_NAME(prot_dp) );		\
		NWARN(DEFAULT_ERROR_STRING);					\
		break;								\
}

#define FLOAT_PREC_SWITCH(func,prot_dp)					\
									\
switch( OBJ_MACH_PREC( prot_dp ) ){						\
	case PREC_SP:  HOST_TYPED_CALL_NAME(func,sp)(HOST_CALL_ARGS); break;	\
	case PREC_DP:  HOST_TYPED_CALL_NAME(func,dp)(HOST_CALL_ARGS); break;	\
	default:							\
		sprintf(DEFAULT_ERROR_STRING,				\
	"FLOAT_PREC_SWITCH (%s):  missing precision case",#func);	\
		NWARN(DEFAULT_ERROR_STRING);				\
		break;							\
}

/*
void vmscp(Vec_Obj_Args *oap){ SWITCH5("vmscp",byvmscp,invmscp,divmscp,spvmscp,dpvmscp ) } 
void vmscm(Vec_Obj_Args *oap){ SWITCH5("vmscm",byvmscm,invmscm,divmscm,spvmscm,dpvmscm ) } 
*/

/* vector comparisons with bitmap result */

/* We use NORM_DECL for functions that have implicit type (real or complex) */

#define NORM_DECL( func )				\
void  HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS)		\
{ SIMPLE_PREC_SWITCH(func,OA_SRC_OBJ(oap,0) ) }

#define FLOAT_DECL( func )				\
void  HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS)		\
{ FLOAT_PREC_SWITCH(func,OA_SRC_OBJ(oap,0)) }

/* RCQM_SWITCH:  5 types (real/complex/mixed/quat/qmixd)
 * by appropriate precisions
 */

#define RCQM_SWITCH(func)							\
										\
/*fprintf(stderr,"RCQM_SWITCH %s\n",#func);*/\
switch( OA_ARGSTYPE(oap) ){							\
	case REAL_ARGS:		REAL_SWITCH(func,OA_SRC_OBJ(oap,0))	break;	\
	case COMPLEX_ARGS:	CPX_SWITCH(func,OA_SRC_OBJ(oap,0))	break;	\
	case MIXED_ARGS:	MIXED_SWITCH(func,OA_SRC_OBJ(oap,0))	break;	\
	case QUATERNION_ARGS:	QUAT_SWITCH(func,OA_SRC_OBJ(oap,0))	break;	\
	case QMIXED_ARGS:	QMIXD_SWITCH(func,OA_SRC_OBJ(oap,0))	break;	\
	default:								\
		NWARN("RCQM_SWITCH:  unexpected argset type!?");		\
		break;								\
}


// BUG?  what is this used for?

/* RC_SWITCH:  2 types (real/complex) by 5 precisions */
/*		oops, now 8 precisions with unsigned... */
/*		oops, now 10 precisions with int64... */
/* Now we only support complex for floating pt types */

#define RC_SWITCH(func)								\
										\
switch( OA_ARGSTYPE(oap) ){							\
	case REAL_ARGS:		REAL_SWITCH(func,OA_SRC_OBJ(oap,0))	break;	\
	case COMPLEX_ARGS:	CPX_SWITCH(func,OA_SRC_OBJ(oap,0))	break;	\
	default:								\
		sprintf(DEFAULT_ERROR_STRING,					\
"RC_SWITCH (%s):  unhandled type code (%d 0x%x) (expected REAL_ARGS or COMPLEX_ARGS)",	\
#func, OA_ARGSTYPE(oap) , OA_ARGSTYPE(oap) );					\
		NWARN(DEFAULT_ERROR_STRING);					\
		break;								\
}

/* CPX_SWITCH:  complex type, by 2 precisions */

#define CPX_SWITCH(func,prot_dp)						\
									\
switch( OBJ_MACH_PREC( prot_dp ) ){						\
	case PREC_SP:  HOST_TYPED_CALL_NAME_CPX(func,sp)(HOST_CALL_ARGS); break;	\
	case PREC_DP:  HOST_TYPED_CALL_NAME_CPX(func,dp)(HOST_CALL_ARGS); break;	\
	default:							\
		sprintf(DEFAULT_ERROR_STRING,				\
	"CPX_SWITCH (%s):  missing complex precision case (%d 0x%x)",	\
			#func, OBJ_MACH_PREC( prot_dp ) ,			\
		OBJ_MACH_PREC( prot_dp ) );					\
		NWARN(DEFAULT_ERROR_STRING);				\
		break;							\
}

#define MIXED_SWITCH(func,prot_dp)						\
									\
switch( OBJ_MACH_PREC( prot_dp ) ){						\
	case PREC_SP:  HOST_TYPED_CALL_NAME_MIXED(func,sp)(HOST_CALL_ARGS); break;	\
	case PREC_DP:  HOST_TYPED_CALL_NAME_MIXED(func,dp)(HOST_CALL_ARGS); break;	\
	default:							\
		sprintf(DEFAULT_ERROR_STRING,				\
"MIXED_SWITCH (%s):  missing complex mixed precision case (%d 0x%x)",	\
			#func, OBJ_MACH_PREC( prot_dp ) ,			\
		OBJ_MACH_PREC( prot_dp ) );					\
		NWARN(DEFAULT_ERROR_STRING);				\
		break;							\
}

/* QUAT_SWITCH:  quaternion type, by 2 precisions */
// we do not support quaternions with integer precisions.

#define QUAT_SWITCH(func,prot_dp)						\
									\
switch( OBJ_MACH_PREC( prot_dp ) ){						\
	case PREC_SP:  HOST_TYPED_CALL_NAME_QUAT(func,sp)(HOST_CALL_ARGS); break;	\
	case PREC_DP:  HOST_TYPED_CALL_NAME_QUAT(func,dp)(HOST_CALL_ARGS); break;	\
	default:							\
		sprintf(DEFAULT_ERROR_STRING,				\
"QUAT_SWITCH (%s):  missing quaternion precision case (%d 0x%x)",	\
#func, OBJ_MACH_PREC( OA_SRC_OBJ(oap,0) ) ,				\
OBJ_MACH_PREC( OA_SRC_OBJ(oap,0) ) );					\
		NWARN(DEFAULT_ERROR_STRING);				\
		break;							\
}

#define QMIXD_SWITCH(func,prot_dp)						\
									\
switch( OBJ_MACH_PREC( prot_dp ) ){						\
	case PREC_SP:  HOST_TYPED_CALL_NAME_QMIXD(func,sp)(HOST_CALL_ARGS); break;	\
	case PREC_DP:  HOST_TYPED_CALL_NAME_QMIXD(func,dp)(HOST_CALL_ARGS); break;	\
	default:					\
		sprintf(DEFAULT_ERROR_STRING,				\
"QMIXD_SWITCH (%s):  missing mixed quaternion precision case (%d 0x%x)",	\
			#func, OBJ_MACH_PREC( prot_dp ) ,			\
		OBJ_MACH_PREC( prot_dp ) );					\
		NWARN(DEFAULT_ERROR_STRING);				\
		break;							\
}

#define RCQ_SWITCH(func)						\
									\
switch(  OA_ARGSTYPE(oap)  ){						\
	case REAL_ARGS:		REAL_SWITCH(func,OA_DEST(oap))	break;	\
	case COMPLEX_ARGS:	CPX_SWITCH(func,OA_DEST(oap))	break;	\
	case QUATERNION_ARGS:	QUAT_SWITCH(func,OA_DEST(oap))	break;	\
	default:							\
		sprintf(DEFAULT_ERROR_STRING,				\
"RCQ_SWITCH (%s):  unhandled type code (%d 0x%x)",			\
#func,  OA_ARGSTYPE(oap)  ,  OA_ARGSTYPE(oap)  );			\
		NWARN(DEFAULT_ERROR_STRING);				\
		break;							\
}

#define RCQB_SWITCH(func)						\
									\
switch(  OA_ARGSTYPE(oap)  ){						\
	case REAL_ARGS:		REAL_SWITCH_B(func,OA_DEST(oap))	break;	\
	case COMPLEX_ARGS:	CPX_SWITCH(func,OA_DEST(oap))	break;	\
	case QUATERNION_ARGS:	QUAT_SWITCH(func,OA_DEST(oap))	break;	\
	default:							\
		sprintf(DEFAULT_ERROR_STRING,				\
"RCQB_SWITCH (%s):  unhandled type code (%d 0x%x)",			\
#func,  OA_ARGSTYPE(oap)  ,  OA_ARGSTYPE(oap)  );			\
		NWARN(DEFAULT_ERROR_STRING);				\
		break;							\
}

/* BUG can't call simd_func here unless all checks are performed!? */

#ifdef USE_SSE

#define SP_SSE_SWITCH(func)					\
								\
	if( use_sse_extensions )				\
		simd_obj_r##func(HOST_CALL_ARGS);				\
	else							\
		HOST_TYPED_CALL_NAME_REAL(func,sp)(HOST_CALL_ARGS);


#else /* ! USE_SSE */

#define SP_SSE_SWITCH(func)	HOST_TYPED_CALL_NAME_REAL(func,sp)(HOST_CALL_ARGS);

#endif /* ! USE_SSE */


/* RC_SWITCH_SSE:  like RC_SWITCH, but for single precision float have mmx version...  */
// BUG SSE is only for cpu!?

#define RC_SWITCH_SSE(func)							\
										\
switch(  OA_ARGSTYPE(oap)  ){							\
	case REAL_ARGS:		REAL_SWITCH_SSE(func,OA_SRC_OBJ(oap,0))	break;	\
	case COMPLEX_ARGS:	CPX_SWITCH(func,OA_SRC_OBJ(oap,0))	break;	\
	default:								\
		sprintf(DEFAULT_ERROR_STRING,					\
"RC_SWITCH (%s):  unhandled type code (%d 0x%x)",				\
#func,  OA_ARGSTYPE(oap)  ,  OA_ARGSTYPE(oap)  );				\
		NWARN(DEFAULT_ERROR_STRING);					\
		break;								\
}


#define RCQ_SWITCH_SSE(func)							\
										\
switch(  OA_ARGSTYPE(oap)  ){							\
	case REAL_ARGS:		REAL_SWITCH_SSE(func,OA_SRC_OBJ(oap,0))	break;	\
	case COMPLEX_ARGS:	CPX_SWITCH(func,OA_SRC_OBJ(oap,0))	break;	\
	case QUATERNION_ARGS:	QUAT_SWITCH(func,OA_SRC_OBJ(oap,0))	break;	\
	default:								\
		sprintf(DEFAULT_ERROR_STRING,					\
"RCQ_SWITCH_SSE (%s):  unhandled type code (%d 0x%x)",				\
#func,  OA_ARGSTYPE(oap)  ,  OA_ARGSTYPE(oap)  );				\
		NWARN(DEFAULT_ERROR_STRING);					\
		break;								\
}

#define RC_ALL_FUNC( func )					\
void HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS)			\
{ RC_SWITCH(func) }

#define C_ALL_FUNC( func )					\
void HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS)			\
{ CPX_SWITCH(func, OA_SRC_OBJ(oap,0) ) }

#define Q_ALL_FUNC( func )					\
void HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS)			\
{	QUAT_SWITCH(func,OA_SRC_OBJ(oap,0) )	}

// The real/cpx/quat functions are not called from the switch, but they
// are used in the vfa_tbl...

#define RCQ_ALL_FUNC( func )						\
									\
void HOST_CALL_NAME_REAL(func)(HOST_CALL_ARG_DECLS)			\
{	/*REAL_CHECK*/							\
	REAL_SWITCH(func, OA_SRC_OBJ(oap,0) )	}			\
									\
void HOST_CALL_NAME_CPX(func)(HOST_CALL_ARG_DECLS)			\
{	/*CPX_CHECK*/							\
	CPX_SWITCH(func, OA_SRC_OBJ(oap,0) )	}			\
									\
void HOST_CALL_NAME_QUAT(func)(HOST_CALL_ARG_DECLS)			\
{	/*QUAT_CHECK*/							\
	QUAT_SWITCH(func,OA_SRC_OBJ(oap,0) )	}			\
									\
void HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS)				\
{ RCQ_SWITCH(func) }

#define RCQB_ALL_FUNC( func )						\
									\
void HOST_CALL_NAME_REAL(func)(HOST_CALL_ARG_DECLS)			\
{	/*REAL_CHECK*/							\
	REAL_SWITCH_B(func, OA_SRC_OBJ(oap,0) )	}			\
									\
void HOST_CALL_NAME_CPX(func)(HOST_CALL_ARG_DECLS)			\
{	/*CPX_CHECK*/							\
	CPX_SWITCH(func, OA_SRC_OBJ(oap,0) )	}			\
									\
void HOST_CALL_NAME_QUAT(func)(HOST_CALL_ARG_DECLS)			\
{	/*QUAT_CHECK*/							\
	QUAT_SWITCH(func,OA_SRC_OBJ(oap,0) )	}			\
									\
void HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS)				\
{ RCQB_SWITCH(func) }

#define RCQM_ALL_FUNC( func )						\
									\
void HOST_CALL_NAME_REAL(func) (HOST_CALL_ARG_DECLS)			\
{ \
	REAL_SWITCH(func,OA_SRC_OBJ(oap,0))				\
}									\
									\
void HOST_CALL_NAME_CPX(func) (HOST_CALL_ARG_DECLS)			\
{ \
	CPX_SWITCH(func,OA_SRC_OBJ(oap,0))				\
}									\
									\
void HOST_CALL_NAME_QUAT(func) (HOST_CALL_ARG_DECLS)			\
{ \
	QUAT_SWITCH(func,OA_SRC_OBJ(oap,0))				\
}									\
									\
void HOST_CALL_NAME_MIXED(func) (HOST_CALL_ARG_DECLS)			\
{ \
	MIXED_SWITCH(func,OA_SRC_OBJ(oap,0))				\
}									\
									\
void HOST_CALL_NAME_QMIXD(func) (HOST_CALL_ARG_DECLS)			\
{ \
	QMIXD_SWITCH(func,OA_SRC_OBJ(oap,0))				\
}									\
									\
void HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS)				\
{ RCQM_SWITCH(func) }

#ifdef USE_SSE

/* BUG need to add SSE hooks! */
#define C_ALL_FUNC_SSE( func )			\
void HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS)	\
{	CPX_CHECK				\
	CPX_SWITCH(func, OA_SRC_OBJ(oap,0) )	}

/* BUG need to add SSE hooks! */
#define Q_ALL_FUNC_SSE( func )				\
void HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS)		\
{	QUAT_CHECK					\
	QUAT_SWITCH(func,OA_SRC_OBJ(oap,0) )	}

#define RC_ALL_FUNC_SSE( func )				\
void HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS)		\
{ RC_SWITCH_SSE(func) }

#define RCQ_ALL_FUNC_SSE( func )			\
void HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS)		\
{ RCQ_SWITCH_SSE(func) }

#define RCQM_ALL_FUNC_SSE( func )					\
									\
void HOST_CALL_NAME_REAL(func) (HOST_CALL_ARG_DECLS)			\
{									\
	REAL_SWITCH_SSE(func,OA_SRC_OBJ(oap,0))				\
}									\
									\
void HOST_CALL_NAME_CPX(func) (HOST_CALL_ARG_DECLS)			\
{									\
	CPX_SWITCH(func,OA_SRC_OBJ(oap,0))				\
}									\
									\
void HOST_CALL_NAME_QUAT(func) (HOST_CALL_ARG_DECLS)			\
{									\
	QUAT_SWITCH(func,OA_SRC_OBJ(oap,0))				\
}									\
									\
void HOST_CALL_NAME_MIXED(func) (HOST_CALL_ARG_DECLS)			\
{									\
	MIXED_SWITCH(func,OA_SRC_OBJ(oap,0))				\
}									\
									\
void HOST_CALL_NAME_QMIXD(func) (HOST_CALL_ARG_DECLS)			\
{									\
	QMIXD_SWITCH(func,OA_SRC_OBJ(oap,0))				\
}									\
									\
void HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS)				\
{ RCQM_SWITCH_SSE(func) }

#else /* ! USE_SSE */

#define Q_ALL_FUNC_SSE( func )		Q_ALL_FUNC( func )
#define C_ALL_FUNC_SSE( func )		C_ALL_FUNC( func )
#define RC_ALL_FUNC_SSE( func )		RC_ALL_FUNC( func )
#define RCQ_ALL_FUNC_SSE( func )	RCQ_ALL_FUNC( func )
#define RCQM_ALL_FUNC_SSE( func )	RCQM_ALL_FUNC( func )

#endif /* ! USE_SSE */

#define FLT_FUNC( func )				\
							\
void HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS ) {	\
	FLOAT_PREC_SWITCH( func, OA_DEST(oap) )		\
} 


/* SWITCH4
 * float, double
 * real, complex
 */

#define RC_FLOAT_SWITCH( func )							\
switch(  OA_ARGSTYPE(oap)  ){							\
	case REAL_ARGS:	REAL_FLOAT_SWITCH(func,OA_SRC_OBJ(oap,0)) break;	\
	case COMPLEX_ARGS:	CPX_SWITCH(func,OA_SRC_OBJ(oap,0)) break;	\
	default:								\
		NWARN("SWITCH4:  bad type");					\
		break;								\
}


/* INT_SWITCH
 * switch on integer precision
 */

#define INT_SWITCH(func)						\
									\
switch( OBJ_MACH_PREC( OA_SRC_OBJ(oap,0) ) ){				\
	case PREC_BY:  HOST_TYPED_CALL_NAME(func,by)(HOST_CALL_ARGS); break;	\
	case PREC_IN:  HOST_TYPED_CALL_NAME(func,in)(HOST_CALL_ARGS); break;	\
	case PREC_DI:  HOST_TYPED_CALL_NAME(func,di)(HOST_CALL_ARGS); break;	\
	case PREC_LI:  HOST_TYPED_CALL_NAME(func,li)(HOST_CALL_ARGS); break;	\
	case PREC_UBY:  HOST_TYPED_CALL_NAME(func,uby)(HOST_CALL_ARGS); break;	\
	case PREC_UIN:  HOST_TYPED_CALL_NAME(func,uin)(HOST_CALL_ARGS); break;	\
	case PREC_UDI:  HOST_TYPED_CALL_NAME(func,udi)(HOST_CALL_ARGS); break;	\
	case PREC_ULI:  HOST_TYPED_CALL_NAME(func,uli)(HOST_CALL_ARGS); break;	\
	default:							\
		NWARN("INT_SWITCH:  missing precision case");		\
		break;							\
}

#define INT_FUNC( func )						\
void HOST_CALL_NAME(func) (HOST_CALL_ARG_DECLS){INT_SWITCH( func ) } 

// DSWITCH2 is used just for real fft functions...

#define DSWITCH2( funcname, spf, dpf )					\
	switch( OBJ_MACH_PREC(srcdp) ){					\
		case PREC_SP:	spf(HOST_CALL_ARGS);	break;			\
		case PREC_DP:	dpf(HOST_CALL_ARGS);	break;			\
		default:	sprintf(DEFAULT_ERROR_STRING,			\
	"DSWITCH2 (%s):  object %s has bad machine precision (%s)",	\
		funcname,OBJ_NAME(srcdp),OBJ_MACH_PREC_NAME(srcdp) );\
				NWARN(DEFAULT_ERROR_STRING);		\
							break;		\
	}

// FFT_SWITCH is used for complex fft functions, that have the inverse flag...
#define FFT_SWITCH( func, _is_inv )					\
									\
switch( OBJ_MACH_PREC(srcdp) ){						\
	case PREC_SP:							\
		HOST_TYPED_CALL_NAME_CPX(func,sp)(HOST_CALL_ARGS, _is_inv);	\
		break;							\
	case PREC_DP:							\
		HOST_TYPED_CALL_NAME_CPX(func,dp)(HOST_CALL_ARGS, _is_inv);	\
		break;							\
	default:	sprintf(DEFAULT_ERROR_STRING,			\
"FFT_SWITCH (%s):  object %s has bad machine precision %s",		\
#func,OBJ_NAME(srcdp),OBJ_MACH_PREC_NAME(srcdp) );			\
		NWARN(DEFAULT_ERROR_STRING);				\
		break;							\
}

#endif // ! _ENTRIES_H_ 
