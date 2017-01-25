
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

dnl	SIMPLE_PREC_SWITCH(func,prot_dp)
define(`SIMPLE_PREC_SWITCH',`

switch( OBJ_MACH_PREC($2) ){
	case PREC_BY:  HOST_TYPED_CALL_NAME($1,by)(HOST_CALL_ARGS); break;
	case PREC_IN:  HOST_TYPED_CALL_NAME($1,in)(HOST_CALL_ARGS); break;
	case PREC_DI:  HOST_TYPED_CALL_NAME($1,di)(HOST_CALL_ARGS); break;
	case PREC_LI:  HOST_TYPED_CALL_NAME($1,li)(HOST_CALL_ARGS); break;
	case PREC_SP:  HOST_TYPED_CALL_NAME($1,sp)(HOST_CALL_ARGS); break;
	case PREC_DP:  HOST_TYPED_CALL_NAME($1,dp)(HOST_CALL_ARGS); break;
	case PREC_UBY:  HOST_TYPED_CALL_NAME($1,uby)(HOST_CALL_ARGS); break;
	case PREC_UIN:  HOST_TYPED_CALL_NAME($1,uin)(HOST_CALL_ARGS); break;
	case PREC_UDI:  HOST_TYPED_CALL_NAME($1,udi)(HOST_CALL_ARGS); break;
	case PREC_ULI:  HOST_TYPED_CALL_NAME($1,uli)(HOST_CALL_ARGS); break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
	"simple_prec_switch (%s):  missing precision case","$1");
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

/* REAL_SWITCH - like SIMPLE_PREC_SWITCH, but calls real func...
 *
 * The get bit_rvset to work properly, we have to know which machine precision
 * is used for the bitmap words.  On the cpu we want to use the biggest word
 * supported by the machine, while on the gpu it might be more efficient to
 * use a 32 bit word.  For now we assume that ULI is the bitmap word size.
 * Perhaps we should put a CAUTIOUS check on that here?
 */

dnl	REAL_SWITCH(func,prot_dp)
define(`REAL_SWITCH',`

/*fprintf(stderr,"real_switch %s, prot_dp = %s, mach_prec = %s, prec = %sn",\
"$1",OBJ_NAME($2),PREC_NAME(PREC_FOR_CODE(OBJ_MACH_PREC($2))),
PREC_NAME(OBJ_PREC_PTR($2)));*/

switch( OBJ_MACH_PREC($2) ){
	case PREC_BY:  HOST_TYPED_CALL_NAME_REAL($1,by)(HOST_CALL_ARGS); break;
	case PREC_IN:  HOST_TYPED_CALL_NAME_REAL($1,in)(HOST_CALL_ARGS); break;
	case PREC_DI:  HOST_TYPED_CALL_NAME_REAL($1,di)(HOST_CALL_ARGS); break;
	case PREC_LI:  HOST_TYPED_CALL_NAME_REAL($1,li)(HOST_CALL_ARGS); break;
	case PREC_SP:  HOST_TYPED_CALL_NAME_REAL($1,sp)(HOST_CALL_ARGS); break;
	case PREC_DP:  HOST_TYPED_CALL_NAME_REAL($1,dp)(HOST_CALL_ARGS); break;
	case PREC_UBY:  HOST_TYPED_CALL_NAME_REAL($1,uby)(HOST_CALL_ARGS); break;
	case PREC_UIN:  HOST_TYPED_CALL_NAME_REAL($1,uin)(HOST_CALL_ARGS); break;
	case PREC_UDI:  HOST_TYPED_CALL_NAME_REAL($1,udi)(HOST_CALL_ARGS); break;
	case PREC_ULI:	HOST_TYPED_CALL_NAME_REAL($1,uli)(HOST_CALL_ARGS); break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
	"real_switch (%s):  missing precision case (obj %s, prec %s)",
		"$1", OBJ_NAME($2), OBJ_MACH_PREC_NAME($2) );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')


dnl	REAL_SWITCH_B(func,prot_dp)
define(`REAL_SWITCH_B',`

/*fprintf(stderr,"real_switch_b %s, prot_dp = %s, mach_prec = %s, prec = %sn",	\
"$1",OBJ_NAME($2),PREC_NAME(PREC_FOR_CODE(OBJ_MACH_PREC($2))),
PREC_NAME(OBJ_PREC_PTR($2)));*/

switch( OBJ_MACH_PREC($2) ){
	case PREC_BY:  HOST_TYPED_CALL_NAME_REAL($1,by)(HOST_CALL_ARGS); break;
	case PREC_IN:  HOST_TYPED_CALL_NAME_REAL($1,in)(HOST_CALL_ARGS); break;
	case PREC_DI:  HOST_TYPED_CALL_NAME_REAL($1,di)(HOST_CALL_ARGS); break;
	case PREC_LI:  HOST_TYPED_CALL_NAME_REAL($1,li)(HOST_CALL_ARGS); break;
	case PREC_SP:  HOST_TYPED_CALL_NAME_REAL($1,sp)(HOST_CALL_ARGS); break;
	case PREC_DP:  HOST_TYPED_CALL_NAME_REAL($1,dp)(HOST_CALL_ARGS); break;
	case PREC_UBY:  HOST_TYPED_CALL_NAME_REAL($1,uby)(HOST_CALL_ARGS); break;
	case PREC_UIN:  HOST_TYPED_CALL_NAME_REAL($1,uin)(HOST_CALL_ARGS); break;
	case PREC_UDI:  HOST_TYPED_CALL_NAME_REAL($1,udi)(HOST_CALL_ARGS); break;
	case PREC_ULI:
		if( PREC_CODE(OBJ_PREC_PTR($2)) == PREC_BIT )
			HOST_TYPED_CALL_NAME_REAL($1,bit)(HOST_CALL_ARGS);
		else
			HOST_TYPED_CALL_NAME_REAL($1,uli)(HOST_CALL_ARGS);
		break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
	"real_switch_b (%s):  missing precision case (obj %s, prec %s)",
		"$1", OBJ_NAME($2), OBJ_MACH_PREC_NAME($2) );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')


dnl	REAL_FLOAT_SWITCH(func,prot_dp)
define(`REAL_FLOAT_SWITCH',`

switch( OBJ_MACH_PREC($2) ){
	case PREC_SP:  HOST_TYPED_CALL_NAME_REAL($1,sp)(HOST_CALL_ARGS); break;
	case PREC_DP:  HOST_TYPED_CALL_NAME_REAL($1,dp)(HOST_CALL_ARGS); break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
	"real_float_switch (%s):  missing precision case (obj %s, prec %s)",
		"$1", OBJ_NAME($2), OBJ_MACH_PREC_NAME($2) );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

dnl	REAL_SWITCH_SSE(func,prot_dp)
define(`REAL_SWITCH_SSE',`

switch( OBJ_MACH_PREC($2) ){
	case PREC_BY:	HOST_TYPED_CALL_NAME_REAL($1,by)(HOST_CALL_ARGS); break;
	case PREC_IN:	HOST_TYPED_CALL_NAME_REAL($1,in)(HOST_CALL_ARGS); break;
	case PREC_DI:	HOST_TYPED_CALL_NAME_REAL($1,di)(HOST_CALL_ARGS); break;
	case PREC_LI:	HOST_TYPED_CALL_NAME_REAL($1,li)(HOST_CALL_ARGS); break;
	case PREC_SP:	SP_SSE_SWITCH($1) break;
	case PREC_DP:	HOST_TYPED_CALL_NAME_REAL($1,dp)(HOST_CALL_ARGS); break;
	case PREC_UBY:	HOST_TYPED_CALL_NAME_REAL($1,uby)(HOST_CALL_ARGS); break;
	case PREC_UIN:	HOST_TYPED_CALL_NAME_REAL($1,uin)(HOST_CALL_ARGS); break;
	case PREC_UDI:	HOST_TYPED_CALL_NAME_REAL($1,udi)(HOST_CALL_ARGS); break;
	case PREC_ULI:	HOST_TYPED_CALL_NAME_REAL($1,uli)(HOST_CALL_ARGS); break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
	"real_switch_sse (%s):  missing precision case (obj %s, prec %s)",
		"$1", OBJ_NAME($2), OBJ_MACH_PREC_NAME($2) );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

dnl	FLOAT_PREC_SWITCH(func,prot_dp)
define(`FLOAT_PREC_SWITCH',`

switch( OBJ_MACH_PREC( $2 ) ){
	case PREC_SP:  HOST_TYPED_CALL_NAME($1,sp)(HOST_CALL_ARGS); break;
	case PREC_DP:  HOST_TYPED_CALL_NAME($1,dp)(HOST_CALL_ARGS); break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
	"float_prec_switch (%s):  missing precision case","$1");
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

/*
void vmscp(Vec_Obj_Args *oap){ SWITCH5("vmscp",byvmscp,invmscp,divmscp,spvmscp,dpvmscp ) } 
void vmscm(Vec_Obj_Args *oap){ SWITCH5("vmscm",byvmscm,invmscm,divmscm,spvmscm,dpvmscm ) } 
*/

/* vector comparisons with bitmap result */

/* We use NORM_DECL for functions that have implicit type (real or complex) */

define(`NORM_DECL',`
void  HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS)
{ SIMPLE_PREC_SWITCH($1,OA_SRC_OBJ(oap,0) ) }
')

define(`FLOAT_DECL',`
void  HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS)
{ FLOAT_PREC_SWITCH($1,OA_SRC_OBJ(oap,0)) }
')

/* RCQM_SWITCH:  5 types (real/complex/mixed/quat/qmixd)
 * by appropriate precisions
 */

dnl	RCQM_SWITCH(func)
define(`RCQM_SWITCH',`

/*fprintf(stderr,"rcqm_switch %sn","$1");*/\
switch( OA_ARGSTYPE(oap) ){
	case REAL_ARGS:		REAL_SWITCH($1,OA_SRC_OBJ(oap,0))	break;
	case COMPLEX_ARGS:	CPX_SWITCH($1,OA_SRC_OBJ(oap,0))	break;
	case MIXED_ARGS:	MIXED_SWITCH($1,OA_SRC_OBJ(oap,0))	break;
	case QUATERNION_ARGS:	QUAT_SWITCH($1,OA_SRC_OBJ(oap,0))	break;
	case QMIXED_ARGS:	QMIXD_SWITCH($1,OA_SRC_OBJ(oap,0))	break;
	default:
		NWARN("RCQM_SWITCH:  unexpected argset type!?");
		break;
}
')


// BUG?  what is this used for?

/* RC_SWITCH:  2 types (real/complex) by 5 precisions */
/*		oops, now 8 precisions with unsigned... */
/*		oops, now 10 precisions with int64... */
/* Now we only support complex for floating pt types */

dnl	RC_SWITCH(func)
define(`RC_SWITCH',`

switch( OA_ARGSTYPE(oap) ){
	case REAL_ARGS:		REAL_SWITCH($1,OA_SRC_OBJ(oap,0))	break;
	case COMPLEX_ARGS:	CPX_SWITCH($1,OA_SRC_OBJ(oap,0))	break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
"rc_switch (%s):  unhandled type code (%d 0x%x) (expected REAL_ARGS or COMPLEX_ARGS)",
"$1", OA_ARGSTYPE(oap) , OA_ARGSTYPE(oap) );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

/* CPX_SWITCH:  complex type, by 2 precisions */

dnl	CPX_SWITCH(func,prot_dp)
define(`CPX_SWITCH',`

switch( OBJ_MACH_PREC( $2 ) ){
	case PREC_SP:  HOST_TYPED_CALL_NAME_CPX($1,sp)(HOST_CALL_ARGS); break;
	case PREC_DP:  HOST_TYPED_CALL_NAME_CPX($1,dp)(HOST_CALL_ARGS); break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
	"cpx_switch (%s):  missing complex precision case (%d 0x%x)",
			"$1", OBJ_MACH_PREC( $2 ) ,
		OBJ_MACH_PREC( $2 ) );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

dnl	MIXED_SWITCH(func,prot_dp)
define(`MIXED_SWITCH',`

switch( OBJ_MACH_PREC( $2 ) ){
	case PREC_SP:  HOST_TYPED_CALL_NAME_MIXED($1,sp)(HOST_CALL_ARGS); break;
	case PREC_DP:  HOST_TYPED_CALL_NAME_MIXED($1,dp)(HOST_CALL_ARGS); break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
"mixed_switch (%s):  missing complex mixed precision case (%d 0x%x)",
			"$1", OBJ_MACH_PREC( $2 ) ,
		OBJ_MACH_PREC( $2 ) );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

/* QUAT_SWITCH:  quaternion type, by 2 precisions */
// we do not support quaternions with integer precisions.

dnl	QUAT_SWITCH(func,prot_dp)
define(`QUAT_SWITCH',`

switch( OBJ_MACH_PREC( $2 ) ){
	case PREC_SP:  HOST_TYPED_CALL_NAME_QUAT($1,sp)(HOST_CALL_ARGS); break;
	case PREC_DP:  HOST_TYPED_CALL_NAME_QUAT($1,dp)(HOST_CALL_ARGS); break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
"quat_switch (%s):  missing quaternion precision case (%d 0x%x)",
"$1", OBJ_MACH_PREC( OA_SRC_OBJ(oap,0) ) ,
OBJ_MACH_PREC( OA_SRC_OBJ(oap,0) ) );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

dnl	QMIXD_SWITCH(func,prot_dp)
define(`QMIXD_SWITCH',`

switch( OBJ_MACH_PREC( $2 ) ){
	case PREC_SP:  HOST_TYPED_CALL_NAME_QMIXD($1,sp)(HOST_CALL_ARGS); break;
	case PREC_DP:  HOST_TYPED_CALL_NAME_QMIXD($1,dp)(HOST_CALL_ARGS); break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
"qmixd_switch (%s):  missing mixed quaternion precision case (%d 0x%x)",
			"$1", OBJ_MACH_PREC( $2 ) ,
		OBJ_MACH_PREC( $2 ) );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

dnl	RCQ_SWITCH(func)
define(`RCQ_SWITCH',`

switch(  OA_ARGSTYPE(oap)  ){
	case REAL_ARGS:		REAL_SWITCH($1,OA_DEST(oap))	break;
	case COMPLEX_ARGS:	CPX_SWITCH($1,OA_DEST(oap))	break;
	case QUATERNION_ARGS:	QUAT_SWITCH($1,OA_DEST(oap))	break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
"rcq_switch (%s):  unhandled type code (%d 0x%x)",
"$1",  OA_ARGSTYPE(oap)  ,  OA_ARGSTYPE(oap)  );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

dnl	RCQB_SWITCH(func)
define(`RCQB_SWITCH',`

switch(  OA_ARGSTYPE(oap)  ){
	case REAL_ARGS:		REAL_SWITCH_B($1,OA_DEST(oap))	break;
	case COMPLEX_ARGS:	CPX_SWITCH($1,OA_DEST(oap))	break;
	case QUATERNION_ARGS:	QUAT_SWITCH($1,OA_DEST(oap))	break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
"rcqb_switch (%s):  unhandled type code (%d 0x%x)",
"$1",  OA_ARGSTYPE(oap)  ,  OA_ARGSTYPE(oap)  );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

/* BUG can't call simd_func here unless all checks are performed!? */

ifdef(`USE_SSE',`

dnl	SP_SSE_SWITCH(func)
define(`SP_SSE_SWITCH',`

	if( use_sse_extensions )
		simd_obj_r#"$1"(HOST_CALL_ARGS);
	else
		HOST_TYPED_CALL_NAME_REAL($1,sp)(HOST_CALL_ARGS);
')


',` dnl else /* ! USE_SSE */

define(`SP_SSE_SWITCH',`HOST_TYPED_CALL_NAME_REAL($1,sp)(HOST_CALL_ARGS);')

') dnl endif /* ! USE_SSE */


dnl /* RC_SWITCH_SSE:  like RC_SWITCH, but for single precision float have mmx version...  */
dnl // BUG SSE is only for cpu!?

dnl	RC_SWITCH_SSE(func)
define(`RC_SWITCH_SSE',`

switch(  OA_ARGSTYPE(oap)  ){
	case REAL_ARGS:		REAL_SWITCH_SSE($1,OA_SRC_OBJ(oap,0))	break;
	case COMPLEX_ARGS:	CPX_SWITCH($1,OA_SRC_OBJ(oap,0))	break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
"rc_switch (%s):  unhandled type code (%d 0x%x)",
"$1",  OA_ARGSTYPE(oap)  ,  OA_ARGSTYPE(oap)  );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')


dnl	RCQ_SWITCH_SSE(func)
define(`RCQ_SWITCH_SSE',`

switch(  OA_ARGSTYPE(oap)  ){
	case REAL_ARGS:		REAL_SWITCH_SSE($1,OA_SRC_OBJ(oap,0))	break;
	case COMPLEX_ARGS:	CPX_SWITCH($1,OA_SRC_OBJ(oap,0))	break;
	case QUATERNION_ARGS:	QUAT_SWITCH($1,OA_SRC_OBJ(oap,0))	break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
"rcq_switch_sse (%s):  unhandled type code (%d 0x%x)",
"$1",  OA_ARGSTYPE(oap)  ,  OA_ARGSTYPE(oap)  );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

dnl	RC_ALL_FUNC( func )
define(`RC_ALL_FUNC',`
void HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS)
{ RC_SWITCH($1) }
')

define(`C_ALL_FUNC',`
void HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS)
{ CPX_SWITCH($1, OA_SRC_OBJ(oap,0) ) }
')

define(`Q_ALL_FUNC',`
void HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS)
{	QUAT_SWITCH($1,OA_SRC_OBJ(oap,0) )	}
')

// The real/cpx/quat functions are not called from the switch, but they
// are used in the vfa_tbl...

define(`RCQ_ALL_FUNC',`

void HOST_CALL_NAME_REAL($1)(HOST_CALL_ARG_DECLS)
{	/*REAL_CHECK*/
	REAL_SWITCH($1, OA_SRC_OBJ(oap,0) )	}

void HOST_CALL_NAME_CPX($1)(HOST_CALL_ARG_DECLS)
{	/*CPX_CHECK*/
	CPX_SWITCH($1, OA_SRC_OBJ(oap,0) )	}

void HOST_CALL_NAME_QUAT($1)(HOST_CALL_ARG_DECLS)
{	/*QUAT_CHECK*/
	QUAT_SWITCH($1,OA_SRC_OBJ(oap,0) )	}

void HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS)
{ RCQ_SWITCH($1) }
')

define(`RCQB_ALL_FUNC',`

void HOST_CALL_NAME_REAL($1)(HOST_CALL_ARG_DECLS)
{	/*REAL_CHECK*/
	REAL_SWITCH_B($1, OA_SRC_OBJ(oap,0) )	}

void HOST_CALL_NAME_CPX($1)(HOST_CALL_ARG_DECLS)
{	/*CPX_CHECK*/
	CPX_SWITCH($1, OA_SRC_OBJ(oap,0) )	}

void HOST_CALL_NAME_QUAT($1)(HOST_CALL_ARG_DECLS)
{	/*QUAT_CHECK*/
	QUAT_SWITCH($1,OA_SRC_OBJ(oap,0) )	}

void HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS)
{ RCQB_SWITCH($1) }
')

define(`RCQM_ALL_FUNC',`

void HOST_CALL_NAME_REAL($1) (HOST_CALL_ARG_DECLS)
{ 
	REAL_SWITCH($1,OA_SRC_OBJ(oap,0))
}

void HOST_CALL_NAME_CPX($1) (HOST_CALL_ARG_DECLS)
{ 
	CPX_SWITCH($1,OA_SRC_OBJ(oap,0))
}

void HOST_CALL_NAME_QUAT($1) (HOST_CALL_ARG_DECLS)
{ 
	QUAT_SWITCH($1,OA_SRC_OBJ(oap,0))
}

void HOST_CALL_NAME_MIXED($1) (HOST_CALL_ARG_DECLS)
{ 
	MIXED_SWITCH($1,OA_SRC_OBJ(oap,0))
}

void HOST_CALL_NAME_QMIXD($1) (HOST_CALL_ARG_DECLS)
{ 
	QMIXD_SWITCH($1,OA_SRC_OBJ(oap,0))
}

void HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS)
{ RCQM_SWITCH($1) }
')

ifdef(`USE_SSE',`

/* BUG need to add SSE hooks! */
dnl	C_ALL_FUNC_SSE( func )
define(`C_ALL_FUNC_SSE',`
void HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS)
{	CPX_CHECK
	CPX_SWITCH($1, OA_SRC_OBJ(oap,0) )
}
')

/* BUG need to add SSE hooks! */
dnl	Q_ALL_FUNC_SSE( func )
define(`Q_ALL_FUNC_SSE',`
void HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS)
{	QUAT_CHECK
	QUAT_SWITCH($1,OA_SRC_OBJ(oap,0) )	}
')

define(`RC_ALL_FUNC_SSE',`
void HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS)
{ RC_SWITCH_SSE($1) }
')

define(`RCQ_ALL_FUNC_SSE',`
void HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS)
{ RCQ_SWITCH_SSE($1) }
')

define(`RCQM_ALL_FUNC_SSE',`

void HOST_CALL_NAME_REAL($1) (HOST_CALL_ARG_DECLS)
{
	REAL_SWITCH_SSE($1,OA_SRC_OBJ(oap,0))
}

void HOST_CALL_NAME_CPX($1) (HOST_CALL_ARG_DECLS)
{
	CPX_SWITCH($1,OA_SRC_OBJ(oap,0))
}

void HOST_CALL_NAME_QUAT($1) (HOST_CALL_ARG_DECLS)
{
	QUAT_SWITCH($1,OA_SRC_OBJ(oap,0))
}

void HOST_CALL_NAME_MIXED($1) (HOST_CALL_ARG_DECLS)
{
	MIXED_SWITCH($1,OA_SRC_OBJ(oap,0))
}

void HOST_CALL_NAME_QMIXD($1) (HOST_CALL_ARG_DECLS)
{
	QMIXD_SWITCH($1,OA_SRC_OBJ(oap,0))
}

void HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS)
{ RCQM_SWITCH_SSE($1) }
')

',` dnl else /* ! USE_SSE */

define(`Q_ALL_FUNC_SSE',`Q_ALL_FUNC($1)')
define(`C_ALL_FUNC_SSE',`C_ALL_FUNC($1)')
define(`RC_ALL_FUNC_SSE',`RC_ALL_FUNC($1)')
define(`RCQ_ALL_FUNC_SSE',`RCQ_ALL_FUNC($1)')
define(`RCQM_ALL_FUNC_SSE',`RCQM_ALL_FUNC($1)')

') dnl endif /* ! USE_SSE */

define(`FLT_FUNC',`

void HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS ) {
	FLOAT_PREC_SWITCH( func, OA_DEST(oap) )
} 
')


/* SWITCH4
 * float, double
 * real, complex
 */

define(`RC_FLOAT_SWITCH',`
switch(  OA_ARGSTYPE(oap)  ){
	case REAL_ARGS:	REAL_FLOAT_SWITCH($1,OA_SRC_OBJ(oap,0)) break;
	case COMPLEX_ARGS:	CPX_SWITCH($1,OA_SRC_OBJ(oap,0)) break;
	default:
		NWARN("SWITCH4:  bad type");
		break;
}
')


/* INT_SWITCH
 * switch on integer precision
 */

dnl	INT_SWITCH(func)
define(`INT_SWITCH',`

switch( OBJ_MACH_PREC( OA_SRC_OBJ(oap,0) ) ){
	case PREC_BY:  HOST_TYPED_CALL_NAME($1,by)(HOST_CALL_ARGS); break;
	case PREC_IN:  HOST_TYPED_CALL_NAME($1,in)(HOST_CALL_ARGS); break;
	case PREC_DI:  HOST_TYPED_CALL_NAME($1,di)(HOST_CALL_ARGS); break;
	case PREC_LI:  HOST_TYPED_CALL_NAME($1,li)(HOST_CALL_ARGS); break;
	case PREC_UBY:  HOST_TYPED_CALL_NAME($1,uby)(HOST_CALL_ARGS); break;
	case PREC_UIN:  HOST_TYPED_CALL_NAME($1,uin)(HOST_CALL_ARGS); break;
	case PREC_UDI:  HOST_TYPED_CALL_NAME($1,udi)(HOST_CALL_ARGS); break;
	case PREC_ULI:  HOST_TYPED_CALL_NAME($1,uli)(HOST_CALL_ARGS); break;
	default:
		NWARN("INT_SWITCH:  missing precision case");
		break;
}
')

define(`INT_FUNC',`
void HOST_CALL_NAME($1) (HOST_CALL_ARG_DECLS){INT_SWITCH($1) } 
')

// DSWITCH2 is used just for real fft functions...

define(`DSWITCH2( funcname, spf, dpf )
	switch( OBJ_MACH_PREC(srcdp) ){
		case PREC_SP:	spf(HOST_CALL_ARGS);	break;
		case PREC_DP:	dpf(HOST_CALL_ARGS);	break;
		default:	sprintf(DEFAULT_ERROR_STRING,
	"dswitch2 (%s):  object %s has bad machine precision (%s)",
		funcname,OBJ_NAME(srcdp),OBJ_MACH_PREC_NAME(srcdp) );
				NWARN(DEFAULT_ERROR_STRING);
							break;
	}
')

// FFT_SWITCH is used for complex fft functions, that have the inverse flag...
dnl	FFT_SWITCH( func, _is_inv )
define(`FFT_SWITCH',`

switch( OBJ_MACH_PREC(srcdp) ){
	case PREC_SP:
		HOST_TYPED_CALL_NAME_CPX($1,sp)(HOST_CALL_ARGS, $2);
		break;
	case PREC_DP:
		HOST_TYPED_CALL_NAME_CPX($1,dp)(HOST_CALL_ARGS, $2);
		break;
	default:	sprintf(DEFAULT_ERROR_STRING,
"fft_switch (%s):  object %s has bad machine precision %s",
"$1",OBJ_NAME(srcdp),OBJ_MACH_PREC_NAME(srcdp) );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

