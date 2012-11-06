#include "quip_config.h"

char VersionId_newvec_entries[] = QUIP_VERSION_STRING;


#include "nvf.h"

#include "by_prot.h"
#include "in_prot.h"
#include "di_prot.h"
#include "li_prot.h"
#include "sp_prot.h"
#include "dp_prot.h"
#include "uby_prot.h"
#include "uin_prot.h"
#include "udi_prot.h"
#include "uli_prot.h"

/* We used to do a bunch of switches here...
 * But we'd like to replace that with a table jump.
 * Then we can "compile" the table indices, which is not a great big
 * speedup, but it may simplify things...
 */


/* SWITCH5
 * All precisions, but only one type (implicit), some real, some complex.
 */

#define SIMPLE_PREC_SWITCH(funcname,func)					\
										\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){				\
			case PREC_BY:  by_obj_##func(oap); break;		\
			case PREC_IN:  in_obj_##func(oap); break;		\
			case PREC_DI:  di_obj_##func(oap); break;		\
			case PREC_SP:  sp_obj_##func(oap); break;		\
			case PREC_DP:  dp_obj_##func(oap); break;		\
			case PREC_UBY:  uby_obj_##func(oap); break;		\
			case PREC_UIN:  uin_obj_##func(oap); break;		\
			case PREC_UDI:  udi_obj_##func(oap); break;		\
			default:						\
				sprintf(DEFAULT_ERROR_STRING,				\
		"SIMPLE_PREC_SWITCH (%s):  missing precision case",funcname);	\
				NWARN(DEFAULT_ERROR_STRING);				\
				break;						\
		}

#define FLOAT_PREC_SWITCH(funcname,func)					\
										\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){				\
			case PREC_SP:  sp_obj_##func(oap); break;		\
			case PREC_DP:  dp_obj_##func(oap); break;		\
			default:						\
				sprintf(DEFAULT_ERROR_STRING,				\
		"SIMPLE_PREC_SWITCH (%s):  missing precision case",funcname);	\
				NWARN(DEFAULT_ERROR_STRING);				\
				break;						\
		}


/*
void vmscp(Vec_Obj_Args *oap){ SWITCH5("vmscp",byvmscp,invmscp,divmscp,spvmscp,dpvmscp ) } 
void vmscm(Vec_Obj_Args *oap){ SWITCH5("vmscm",byvmscm,invmscm,divmscm,spvmscm,dpvmscm ) } 
*/

/* vector comparisons with bitmap result */

/* We use NORM_DECL for functions that have implicit type (real or complex) */

#define NORM_DECL( func )						\
void  func (Vec_Obj_Args *oap)						\
{ SIMPLE_PREC_SWITCH(#func,func) }

#define FLOAT_DECL( func )						\
void  func (Vec_Obj_Args *oap)						\
{ FLOAT_PREC_SWITCH(#func,func) }

NORM_DECL(vsm_lt)
NORM_DECL(vsm_gt)
NORM_DECL(vsm_le)
NORM_DECL(vsm_ge)
NORM_DECL(vsm_ne)
NORM_DECL(vsm_eq)

NORM_DECL(vvm_lt)
NORM_DECL(vvm_gt)
NORM_DECL(vvm_le)
NORM_DECL(vvm_ge)
NORM_DECL(vvm_ne)
NORM_DECL(vvm_eq)

NORM_DECL( vv_vv_lt )
NORM_DECL( vv_vv_gt )
NORM_DECL( vv_vv_le )
NORM_DECL( vv_vv_ge )
NORM_DECL( vv_vv_eq )
NORM_DECL( vv_vv_ne )

NORM_DECL( vv_vs_lt )
NORM_DECL( vv_vs_gt )
NORM_DECL( vv_vs_le )
NORM_DECL( vv_vs_ge )
NORM_DECL( vv_vs_eq )
NORM_DECL( vv_vs_ne )

NORM_DECL( vs_vv_lt )
NORM_DECL( vs_vv_gt )
NORM_DECL( vs_vv_le )
NORM_DECL( vs_vv_ge )
NORM_DECL( vs_vv_eq )
NORM_DECL( vs_vv_ne )

NORM_DECL( vs_vs_lt )
NORM_DECL( vs_vs_gt )
NORM_DECL( vs_vs_le )
NORM_DECL( vs_vs_ge )
NORM_DECL( vs_vs_eq )
NORM_DECL( vs_vs_ne )

NORM_DECL( ss_vv_lt )
NORM_DECL( ss_vv_gt )
NORM_DECL( ss_vv_le )
NORM_DECL( ss_vv_ge )
NORM_DECL( ss_vv_eq )
NORM_DECL( ss_vv_ne )

NORM_DECL( ss_vs_lt )
NORM_DECL( ss_vs_gt )
NORM_DECL( ss_vs_le )
NORM_DECL( ss_vs_ge )
NORM_DECL( ss_vs_eq )
NORM_DECL( ss_vs_ne )

NORM_DECL(vclip)
NORM_DECL(viclp)
FLOAT_DECL(vscml)

/* Obsolete bitmap funcs, delete */

/* void vscmm(oap){ SWITCH5("vscmm",byvscmm,invscmm,divscmm,spvscmm,dpvscmm ) } */

NORM_DECL( vsign )
NORM_DECL( vabs )
FLOAT_DECL( vconj )
NORM_DECL( vmin  )
NORM_DECL( vmax  )
NORM_DECL( vming )
NORM_DECL( vmaxg )
NORM_DECL( vibnd )
NORM_DECL( vcmp  )
NORM_DECL( vsmax )
NORM_DECL( vscmp )
NORM_DECL( vscmp2 )
NORM_DECL( vsmin )
NORM_DECL( vminv )
NORM_DECL( vmaxv )
NORM_DECL( vmini )
NORM_DECL( vmaxi )
NORM_DECL( vmnmi )
NORM_DECL( vmxmi )

/* BUG obsolete bitmap funcs, delete */
/*
void vmcmp(Vec_Obj_Args *oap){ SWITCH5("vmcmp",byvmcmp,invmcmp,divmcmp,spvmcmp,dpvmcmp) }
void vmcmm(Vec_Obj_Args *oap){ SWITCH5("vmcmm",byvmcmm,invmcmm,divmcmm,spvmcmm,dpvmcmm) }
*/


#define RSWITCH5(funcname,func)						\
									\
		switch(oap->oa_dp[0]->dt_prec){				\
			case PREC_BY:  by_obj_##func(oap); break;	\
			case PREC_IN:  in_obj_##func(oap); break;	\
			case PREC_DI:  di_obj_##func(oap); break;	\
			case PREC_LI:  li_obj_##func(oap); break;	\
			case PREC_SP:  sp_obj_##func(oap); break;	\
			case PREC_DP:  dp_obj_##func(oap); break;	\
			case PREC_UBY: uby_obj_##func(oap); break;	\
			case PREC_UIN: uin_obj_##func(oap); break;	\
			case PREC_UDI: udi_obj_##func(oap); break;	\
			case PREC_ULI: uli_obj_##func(oap); break;	\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
	"RSWITCH5 (%s):  missing case for precision %s (object %s)",	\
		funcname,name_for_prec(oap->oa_dp[0]->dt_prec),oap->oa_dp[0]->dt_name);			\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}


void vramp2d(Vec_Obj_Args *oap )
{
	/* vramp2d isn't called like the tabled functions,
	 * so we have to do some checking here.
	 *
	 * Why ISN'T it called like the others???
	 */

#ifdef HAVE_CUDA
	if( are_gpu_args(oap) ){
		NWARN("Sorry, vramp2d not implemented yet on the GPU.");
		return;
	}
#endif /* HAVE_CUDA */

	if( oap->oa_dp[0]->dt_mach_dim[0] != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"vramp2d:  target %s has type dimension %d, should be 1",
			oap->oa_dp[0]->dt_name,oap->oa_dp[0]->dt_mach_dim[0]);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( oap->oa_dp[0]->dt_frames != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"vramp2d:  target %s has nframes %d, should be 1",
			oap->oa_dp[0]->dt_name,oap->oa_dp[0]->dt_frames);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( oap->oa_dp[0]->dt_seqs != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"vramp2d:  target %s has nseqs %d, should be 1",
			oap->oa_dp[0]->dt_name,oap->oa_dp[0]->dt_seqs);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	RSWITCH5( "vramp2d", vramp2d ) }

void vramp1d(Vec_Obj_Args *oap )
{
	if( oap->oa_dp[0]->dt_mach_dim[0] != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"ramp1d:  target %s has type dimension %d, should be 1",
			oap->oa_dp[0]->dt_name,oap->oa_dp[0]->dt_mach_dim[0]);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( oap->oa_dp[0]->dt_frames != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"ramp1d:  target %s has nframes %d, should be 1",
			oap->oa_dp[0]->dt_name,oap->oa_dp[0]->dt_frames);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( oap->oa_dp[0]->dt_seqs != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"ramp1d:  target %s has nseqs %d, should be 1",
			oap->oa_dp[0]->dt_name,oap->oa_dp[0]->dt_seqs);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	RSWITCH5( "ramp1d", vramp1d ) }

/* SWITCH20:  4 types (real/complex/mixed/quat) by 5 precisions */

#define SWITCH20(func)							\
									\
	switch(oap->oa_argstype){					\
		case REAL_ARGS:						\
		switch(MACHINE_PREC(oap->oa_dp[0])){			\
			case PREC_BY:  by_obj_r##func(oap); break;	\
			case PREC_IN:  in_obj_r##func(oap); break;	\
			case PREC_DI:  di_obj_r##func(oap); break;	\
			case PREC_LI:  li_obj_r##func(oap); break;	\
			case PREC_SP:  sp_obj_r##func(oap); break;	\
			case PREC_DP:  dp_obj_r##func(oap); break;	\
			default:					\
				NWARN("SWITCH20:  missing precision case, real type");	\
				break;					\
		}							\
		break;							\
									\
		case COMPLEX_ARGS:					\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_SP:  sp_obj_c##func(oap); break;	\
			case PREC_DP:  dp_obj_c##func(oap); break;	\
			default:					\
				NWARN("SWITCH20:  complex operations permitted only for float or double");	\
				break;					\
		}							\
		break;							\
									\
		/* case QUATERNION_ARGS:				\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_BY:  by_obj_q##func(oap); break;	\
			case PREC_IN:  in_obj_q##func(oap); break;	\
			case PREC_DI:  di_obj_q##func(oap); break;	\
			case PREC_SP:  sp_obj_q##func(oap); break;	\
			case PREC_DP:  dp_obj_q##func(oap); break;	\
			default:					\
				NWARN("SWITCH20:  missing precision case, quaternion type");	\
				break;					\
		} */							\
									\
									\
		case MIXED_ARGS:					\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_SP:  sp_obj_m##func(oap); break;	\
			case PREC_DP:  dp_obj_m##func(oap); break;	\
			default:					\
				NWARN("SWITCH20:  mixed complex/real operations permitted only for float or double");	\
				break;					\
		}							\
		break;							\
									\
		default: /* N_TYPES */ break;				\
	}

/* SWITCH15:  3 types (real/complex/mixed) by 5 precisions */

#define SWITCH15(func)\
									\
	switch( oap->oa_argstype ){					\
		case REAL_ARGS:						\
		switch(MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_BY:  by_obj_r##func(oap); break;	\
			case PREC_IN:  in_obj_r##func(oap); break;	\
			case PREC_DI:  di_obj_r##func(oap); break;	\
			case PREC_SP:  sp_obj_r##func(oap); break;	\
			case PREC_DP:  dp_obj_r##func(oap); break;	\
			case PREC_UBY:  uby_obj_r##func(oap); break;	\
			case PREC_UIN:  uin_obj_r##func(oap); break;	\
			case PREC_UDI:  udi_obj_r##func(oap); break;	\
			default:					\
				NWARN("SWITCH15:  missing precision case, real type");	\
				break;					\
		}							\
		break;							\
									\
		case COMPLEX_ARGS:					\
		switch(MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_SP:  sp_obj_c##func(oap); break;	\
			case PREC_DP:  dp_obj_c##func(oap); break;	\
			default:					\
				NWARN("SWITCH15:  complex only supported for float and double");	\
				break;					\
		}							\
		break;							\
									\
									\
		case MIXED_ARGS:					\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_SP:  sp_obj_m##func(oap); break;	\
			case PREC_DP:  dp_obj_m##func(oap); break;	\
			default:					\
				NWARN("SWITCH15:  mixed real/complex only supported for float and double");	\
				break;					\
		}							\
		break;							\
									\
		default: /* N_TYPES */ break;				\
	}


void vmul(Vec_Obj_Args *oap)
{
SWITCH20(vmul)
}


/* SWITCH10:  2 types (real/complex) by 5 precisions */
/*		oops, now 8 precisions with unsigned... */
/*		oops, now 10 precisions with int64... */
/* Now we only support complex for floating pt types */

#define SWITCH10(name,func)						\
									\
	switch(oap->oa_argstype){					\
		case REAL_ARGS:						\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_BY:  by_obj_r##func(oap); break;	\
			case PREC_IN:  in_obj_r##func(oap); break;	\
			case PREC_DI:  di_obj_r##func(oap); break;	\
			case PREC_LI:  li_obj_r##func(oap); break;	\
			case PREC_SP:  sp_obj_r##func(oap); break;	\
			case PREC_DP:  dp_obj_r##func(oap); break;	\
			case PREC_UBY: uby_obj_r##func(oap); break;	\
			case PREC_UIN: uin_obj_r##func(oap); break;	\
			case PREC_UDI: udi_obj_r##func(oap); break;	\
			case PREC_ULI: uli_obj_r##func(oap); break;	\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
		"SWITCH10 (%s):  missing real precision case (%d 0x%x)",\
			name,MACHINE_PREC(oap->oa_dp[0]) ,		\
			     MACHINE_PREC(oap->oa_dp[0]) );		\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}							\
		break;							\
									\
		case COMPLEX_ARGS:					\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_SP:  sp_obj_c##func(oap); break;	\
			case PREC_DP:  dp_obj_c##func(oap); break;	\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
	"SWITCH10 (%s):  missing complex precision case (%d 0x%x)",	\
			name,MACHINE_PREC(oap->oa_dp[0]) ,		\
			     MACHINE_PREC(oap->oa_dp[0]) );		\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}							\
		break;							\
									\
		default:						\
			sprintf(DEFAULT_ERROR_STRING,				\
		"SWITCH10 (%s):  unhandled type code (%d 0x%x) (expected REAL_ARGS or COMPLEX_ARGS)",	\
				name,oap->oa_argstype,oap->oa_argstype);	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}

/* CSWITCH8:  complex type, by 8 precisions */

#define CSWITCH8(name,func)	\
									\
	switch( oap->oa_argstype ){					\
		case COMPLEX_ARGS:					\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_SP:  sp_obj_c##func(oap); break;	\
			case PREC_DP:  dp_obj_c##func(oap); break;	\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
	"CSWITCH8 (%s):  missing complex precision case (%d 0x%x)",	\
			name, MACHINE_PREC(oap->oa_dp[0]) ,		\
			      MACHINE_PREC(oap->oa_dp[0]) );		\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}							\
		break;							\
									\
		default:						\
			sprintf(DEFAULT_ERROR_STRING,				\
		"CSWITCH8 (%s):  unhandled type code (%d 0x%x)",	\
				name, oap->oa_argstype , oap->oa_argstype );	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}

/* QSWITCH8:  complex type, by 8 precisions */

#define QSWITCH8(name,byqf,inqf,diqf,spqf,dpqf,ubyqf,uinqf,udiqf)	\
									\
	switch( oap->oa_argstype ){					\
		case QUATERNION_ARGS:					\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_BY:  byqf(oap); break;		\
			case PREC_IN:  inqf(oap); break;		\
			case PREC_DI:  diqf(oap); break;		\
			case PREC_SP:  spqf(oap); break;		\
			case PREC_DP:  dpqf(oap); break;		\
			case PREC_UBY: ubyqf(oap); break;		\
			case PREC_UIN: uinqf(oap); break;		\
			case PREC_UDI: udiqf(oap); break;		\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
	"QSWITCH8 (%s):  missing quaternion precision case (%d 0x%x)",	\
			name, MACHINE_PREC(oap->oa_dp[0]) ,		\
			      MACHINE_PREC(oap->oa_dp[0]) );		\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}							\
		break;							\
									\
		default:						\
			sprintf(DEFAULT_ERROR_STRING,				\
		"QSWITCH8 (%s):  unhandled type code (%d 0x%x)",	\
				name, oap->oa_argstype , oap->oa_argstype );	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}


#define QSWITCH10(name,func)						\
									\
	switch( oap->oa_argstype ){					\
		case REAL_ARGS:						\
		switch( MACHINE_PREC(oap->oa_dest) ){			\
			case PREC_BY:  by_obj_r##func(oap); break;	\
			case PREC_IN:  in_obj_r##func(oap); break;	\
			case PREC_DI:  di_obj_r##func(oap); break;	\
			case PREC_LI:  li_obj_r##func(oap); break;	\
			case PREC_SP:  sp_obj_r##func(oap); break;	\
			case PREC_DP:  dp_obj_r##func(oap); break;	\
			case PREC_UBY: uby_obj_r##func(oap); break;	\
			case PREC_UIN: uin_obj_r##func(oap); break;	\
			case PREC_UDI: udi_obj_r##func(oap); break;	\
			case PREC_ULI: uli_obj_r##func(oap); break;	\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
		"SWITCH10 (%s):  missing real precision case (%d 0x%x)",\
			name, MACHINE_PREC(oap->oa_dest) ,		\
			      MACHINE_PREC(oap->oa_dest) );		\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}							\
		break;							\
									\
		case COMPLEX_ARGS:					\
		switch( MACHINE_PREC(oap->oa_dest) ){			\
			case PREC_SP:  sp_obj_c##func(oap); break;	\
			case PREC_DP:  dp_obj_c##func(oap); break;	\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
	"SWITCH10 (%s):  missing complex precision case (%d 0x%x)",	\
			name, MACHINE_PREC(oap->oa_dest) ,		\
			      MACHINE_PREC(oap->oa_dest) );		\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}							\
		break;							\
									\
		case QUATERNION_ARGS:					\
		switch( MACHINE_PREC(oap->oa_dest) ){			\
			case PREC_SP:  sp_obj_q##func(oap); break;	\
			case PREC_DP:  dp_obj_q##func(oap); break;	\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
	"QSWITCH10 (%s):  missing quaternion precision case (%d 0x%x)",	\
			name, MACHINE_PREC(oap->oa_dest) ,		\
			      MACHINE_PREC(oap->oa_dest) );		\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}							\
		break;							\
									\
		default:						\
			sprintf(DEFAULT_ERROR_STRING,				\
		"QSWITCH10 (%s):  unhandled type code (%d 0x%x)",	\
				name, oap->oa_argstype , oap->oa_argstype );	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}



#define QSWITCH_SIGNED(name,func)					\
									\
	switch( oap->oa_argstype ){					\
		case REAL_ARGS:						\
		switch( MACHINE_PREC(oap->oa_dest) ){			\
			case PREC_BY:  by_obj_r##func(oap); break;	\
			case PREC_IN:  in_obj_r##func(oap); break;	\
			case PREC_DI:  di_obj_r##func(oap); break;	\
			case PREC_LI:  li_obj_r##func(oap); break;	\
			case PREC_SP:  sp_obj_r##func(oap); break;	\
			case PREC_DP:  dp_obj_r##func(oap); break;	\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
		"QSWITCH_SIGNED (%s):  missing real precision case (%d 0x%x)",\
			name, MACHINE_PREC(oap->oa_dest) ,		\
			      MACHINE_PREC(oap->oa_dest) );		\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}							\
		break;							\
									\
		case COMPLEX_ARGS:					\
		switch( MACHINE_PREC(oap->oa_dest) ){			\
			case PREC_SP:  sp_obj_c##func(oap); break;	\
			case PREC_DP:  dp_obj_c##func(oap); break;	\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
	"QSWITCH_SIGNED (%s):  missing complex precision case (%d 0x%x)",	\
			name, MACHINE_PREC(oap->oa_dest) ,		\
			      MACHINE_PREC(oap->oa_dest) );		\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}							\
		break;							\
									\
		case QUATERNION_ARGS:					\
		switch( MACHINE_PREC(oap->oa_dest) ){			\
			case PREC_SP:  sp_obj_q##func(oap); break;	\
			case PREC_DP:  dp_obj_q##func(oap); break;	\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
	"QSWITCH_SIGNED (%s):  missing quaternion precision case (%d 0x%x)",	\
			name, MACHINE_PREC(oap->oa_dest) ,		\
			      MACHINE_PREC(oap->oa_dest) );		\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}							\
		break;							\
									\
		default:						\
			sprintf(DEFAULT_ERROR_STRING,				\
		"QSWITCH_SIGNED (%s):  unhandled type code (%d 0x%x)",	\
				name, oap->oa_argstype , oap->oa_argstype );	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}


/* BUG can't call simd_func here unless all checks are performed!? */

#ifdef USE_SSE

#define SP_SSE_SWITCH(func)						\
									\
				if( use_sse_extensions )		\
					simd_obj_r##func(oap);		\
				else					\
					sp_obj_r##func(oap);


#else /* ! USE_SSE */

#define SP_SSE_SWITCH(func)	sp_obj_r##func(oap);

#endif /* ! USE_SSE */


/* SWITCH10_SSE:  like SWITCH10, but for single precision float have mmx version...  */

#define SWITCH10_SSE(name,func)\
									\
	switch( oap->oa_argstype ){					\
		case REAL_ARGS:						\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_BY:  by_obj_r##func(oap); break;	\
			case PREC_IN:  in_obj_r##func(oap); break;	\
			case PREC_DI:  di_obj_r##func(oap); break;	\
			case PREC_SP:					\
				SP_SSE_SWITCH(func)			\
				break;					\
			case PREC_DP:  dp_obj_r##func(oap); break;	\
			case PREC_UBY: uby_obj_r##func(oap); break;	\
			case PREC_UIN: uin_obj_r##func(oap); break;	\
			case PREC_UDI: udi_obj_r##func(oap); break;	\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
		"SWITCH10 (%s):  missing real precision case (%d 0x%x)",\
			name, MACHINE_PREC(oap->oa_dp[0]) ,		\
			      MACHINE_PREC(oap->oa_dp[0]) );		\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}							\
		break;							\
									\
		case COMPLEX_ARGS:					\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_SP:					\
				sp_obj_c##func(oap);			\
				break;					\
			case PREC_DP:  dp_obj_c##func(oap); break;	\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
	"SWITCH10 (%s):  missing complex precision case (%d 0x%x)",	\
			name, MACHINE_PREC(oap->oa_dp[0]) ,		\
			      MACHINE_PREC(oap->oa_dp[0]) );		\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}							\
		break;							\
									\
		default:						\
			sprintf(DEFAULT_ERROR_STRING,				\
		"SWITCH10 (%s):  unhandled type code (%d 0x%x)",	\
				name, oap->oa_argstype , oap->oa_argstype );	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}


#define QSWITCH10_SSE(name,func)					\
									\
	switch( oap->oa_argstype ){					\
		case REAL_ARGS:						\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_BY:  by_obj_r##func(oap); break;	\
			case PREC_IN:  in_obj_r##func(oap); break;	\
			case PREC_DI:  di_obj_r##func(oap); break;	\
			case PREC_SP:	SP_SSE_SWITCH(func)		\
				break;					\
			case PREC_DP:  dp_obj_r##func(oap); break;	\
			case PREC_UBY: uby_obj_r##func(oap); break;	\
			case PREC_UIN: uin_obj_r##func(oap); break;	\
			case PREC_UDI: udi_obj_r##func(oap); break;	\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
		"SWITCH10 (%s):  missing real precision case (%d 0x%x)",\
			name, MACHINE_PREC(oap->oa_dp[0]) ,		\
			      MACHINE_PREC(oap->oa_dp[0]) );		\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}							\
		break;							\
									\
		case COMPLEX_ARGS:					\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_SP:					\
				sp_obj_c##func(oap);			\
				break;					\
			case PREC_DP:  dp_obj_c##func(oap); break;	\
			default:					\
				sprintf(DEFAULT_ERROR_STRING,			\
	"SWITCH10 (%s):  missing complex precision case (%d 0x%x)",	\
			name, MACHINE_PREC(oap->oa_dp[0]) ,		\
			      MACHINE_PREC(oap->oa_dp[0]) );		\
				NWARN(DEFAULT_ERROR_STRING);			\
				break;					\
		}							\
		break;							\
									\
		default:						\
			sprintf(DEFAULT_ERROR_STRING,				\
		"SWITCH10 (%s):  unhandled type code (%d 0x%x)",	\
				name, oap->oa_argstype , oap->oa_argstype );	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}

#define RC_ALL_FUNC( func )						\
void func (Vec_Obj_Args *oap) 						\
{ SWITCH10(#func,func) }

#define C_ALL_FUNC( func )						\
void func (Vec_Obj_Args *oap)						\
{ CSWITCH8(#func,func) }

#define Q_ALL_FUNC( func )						\
void func (Vec_Obj_Args *oap)						\
{ QSWITCH8(#func,byq##func,inq##func,diq##func,spq##func,dpq##func,	\
		  ubyq##func,uinq##func,udiq##func			\
		  ) }

#define RCQ_ALL_FUNC( func )						\
void func (Vec_Obj_Args *oap)						\
{ QSWITCH10(#func,func) }

#define RCQ_SIGNED_FUNC( func )						\
void func (Vec_Obj_Args *oap)						\
{ QSWITCH_SIGNED(#func,func) }

#ifdef USE_SSE

/* BUG need to add SSE hooks! */
#define C_ALL_FUNC_SSE( func )						\
void func (Vec_Obj_Args *oap)\
{ CSWITCH8(#func,func) }

/* BUG need to add SSE hooks! */
#define Q_ALL_FUNC_SSE( func )						\
void func (Vec_Obj_Args *oap)\
{ QSWITCH8(#func,byq##func,inq##func,diq##func,spq##func,dpq##func,	\
		  ubyq##func,uinq##func,udiq##func			\
		  ) }

#define RC_ALL_FUNC_SSE( func )						\
void func (Vec_Obj_Args *oap)\
{ SWITCH10_SSE(#func,func) }

#define RCQ_ALL_FUNC_SSE( func )					\
void func (Vec_Obj_Args *oap)\
{ QSWITCH10_SSE(#func,func) }

#else /* ! USE_SSE */

#define Q_ALL_FUNC_SSE( func )						\
					Q_ALL_FUNC( func )

#define C_ALL_FUNC_SSE( func )						\
					C_ALL_FUNC( func )

#define RC_ALL_FUNC_SSE( func )						\
					RC_ALL_FUNC( func )

#define RCQ_ALL_FUNC_SSE( func )					\
					RCQ_ALL_FUNC( func )

#endif /* ! USE_SSE */


RCQ_ALL_FUNC( vsum )
RC_ALL_FUNC( vdot )
RC_ALL_FUNC( vrand )

RCQ_ALL_FUNC( vvv_slct )
RCQ_ALL_FUNC( vss_slct )
RCQ_ALL_FUNC( vvs_slct )

RCQ_ALL_FUNC( vset )		/* sse version! */
RCQ_ALL_FUNC( vsqr )		/* sse version! */
RCQ_ALL_FUNC( vneg )		/* sse version! */

RCQ_ALL_FUNC_SSE( vadd )		/* sse version! */
RCQ_ALL_FUNC_SSE( vsub )		/* sse version! */
RC_ALL_FUNC_SSE( vdiv )		/* sse version! */

RCQ_ALL_FUNC_SSE( vsadd )	/* sse version! */
RCQ_ALL_FUNC_SSE( vsmul )	/* sse version! */
RCQ_ALL_FUNC_SSE( vssub )	/* sse version! */
RC_ALL_FUNC_SSE( vsdiv )	/* sse version! */
/* C_ALL_FUNC_SSE( vcsmul ) */	/* sse version! */
/* Q_ALL_FUNC_SSE( vqsmul ) */	/* sse version! */

RC_ALL_FUNC( vsdiv2 )	/* sse version! */

void vmov(Vec_Obj_Args *oap )
{
#ifdef USE_SSE
	if( oap->oa_argsprec == SP_ARGS ){
		int n_per_128;

		n_per_128 = 16 / siztbl[  MACHINE_PREC(oap->oa_dest)  ];
		if(  oap->oa_argstype  == COMPLEX_ARGS )
			n_per_128 >>= 1;


		if( use_sse_extensions && IS_CONTIGUOUS(oap->oa_dest) && IS_CONTIGUOUS(oap->oa_1) &&
			(oap->oa_dest->dt_n_mach_elts % n_per_128)==0 ){

			if( (((u_long) oap->oa_dest->dt_data) & 15) ||
			    (((u_long) oap->oa_1->dt_data) & 15) ){
		NWARN("data vectors must be aligned on 16 byte boundary for SSE acceleration");
			} else {
				simd_vec_rvmov(oap->oa_dest->dt_data,oap->oa_1->dt_data,oap->oa_dest->dt_n_mach_elts/n_per_128);
				return;
			}
		}
	}

#endif /* USE_SSE */

	/* can't (or don't want to) use SSE - just call normal function */
	SWITCH10("vmov",vmov);
} /* end vmov */

/* SWITCH2
 * just float or double...
 */

#define SWITCH2( func )							\
	switch( MACHINE_PREC(oap->oa_dp[0]) ){				\
		case PREC_SP:						\
			sp_obj_##func(oap);				\
			break;						\
		case PREC_DP:						\
			dp_obj_##func(oap);				\
			break;						\
		default:						\
			NWARN("SWITCH2:  bad precision");		\
			break;						\
	}

#define FLT_FUNC( func )							\
void func (Vec_Obj_Args *oap ) { SWITCH2( func ) } 

FLT_FUNC( vrint )
FLT_FUNC( vfloor )
FLT_FUNC( vround )
FLT_FUNC( vceil  )
FLT_FUNC( vminm  )
FLT_FUNC( vmaxm  )
FLT_FUNC( vmnmv  )
FLT_FUNC( vmxmv  )
FLT_FUNC( vsmxm  )
FLT_FUNC( vsmnm  )
FLT_FUNC( vmxmg  )
FLT_FUNC( vmnmg  )
FLT_FUNC( vasin  )
FLT_FUNC( vacos  )
FLT_FUNC( vj0    )
FLT_FUNC( vj1    )
FLT_FUNC( vsqrt  )
FLT_FUNC( vbnd   )
FLT_FUNC( vcos   )
FLT_FUNC( verf   )
FLT_FUNC( vsin   )
FLT_FUNC( vtan   )
FLT_FUNC( vatan  )
FLT_FUNC( vatan2 )
FLT_FUNC( vatn2  )
FLT_FUNC( vexp   )
FLT_FUNC( vuni   )
FLT_FUNC( vlog   )
FLT_FUNC( rvpow   )
FLT_FUNC( vlog10   )
FLT_FUNC( vsatan2 )
FLT_FUNC( vsatan22 )
FLT_FUNC( vspow )
FLT_FUNC( vspow2 )
FLT_FUNC( vcmul )
FLT_FUNC( vmgsq )



/* SWITCH4
 * float, double
 * real, complex
 */

#define SWITCH4( func )							\
	switch( oap->oa_argstype ){					\
		case REAL_ARGS:						\
			switch( MACHINE_PREC(oap->oa_dp[0]) ){		\
				case PREC_SP:				\
					sp_obj_r##func(oap);		\
					break;				\
				case PREC_DP:				\
					dp_obj_r##func(oap);		\
					break;				\
				default:				\
					NWARN("SWITCH4:  bad real prec");\
					break;				\
			}						\
			break;						\
		case COMPLEX_ARGS:					\
			switch( MACHINE_PREC(oap->oa_dp[0]) ){		\
				case PREC_SP:				\
					sp_obj_c##func(oap);		\
					break;				\
				case PREC_DP:				\
					dp_obj_c##func(oap);		\
					break;				\
				default:				\
					NWARN("SWITCH4:  bad cpx prec");	\
					break;				\
			}						\
			break;						\
		default:						\
			NWARN("SWITCH4:  bad type");			\
			break;						\
	}


/* void vpow(Vec_Obj_Args *oap) { SWITCH4( vpow ) } */
void vpow(Vec_Obj_Args *oap) { SWITCH2( rvpow ) }
void vfft(Vec_Obj_Args *oap) { SWITCH4( vfft ) }
void vift(Vec_Obj_Args *oap) { SWITCH4( vift ) }

#ifdef FOOBAR
/* T_SWITCH2:
 * switch on type:  real or complex
 * These are the conversion functions, precisions are implicit.
 *
 * Note that we have implemented integer complex types, but
 * that we really do not have a way of creating these objects yet!?
 * Probably we don't need them...
 */

#define T_SWITCH2( func )						\
	switch( oap->oa_argstype ){					\
		case REAL_ARGS:  (c##func)(oap); break;			\
		case COMPLEX_ARGS:  (c##func)(oap); break;		\
		default:						\
	NWARN("T_SWITCH2:  bad type");					\
			break;						\
	}

#define _CONVERSION( func )						\
void func (Vec_Obj_Args *oap) { T_SWITCH2( func ); } 

#define CONVERSION( key1, key2 )					\
					_CONVERSION( v##key1##2##key2 )

#define ALL_CONVERSIONS( key )						\
									\
	CONVERSION( key, by  )						\
	CONVERSION( key, in  )						\
	CONVERSION( key, di  )						\
	CONVERSION( key, sp )						\
	CONVERSION( key, dp )						\
	CONVERSION( key, uby )						\
	CONVERSION( key, uin )						\
	CONVERSION( key, udi )

ALL_CONVERSIONS( by )
ALL_CONVERSIONS( in )
ALL_CONVERSIONS( di )
ALL_CONVERSIONS( sp )
ALL_CONVERSIONS( dp )
ALL_CONVERSIONS( uby )
ALL_CONVERSIONS( uin )
ALL_CONVERSIONS( udi )

#endif /* FOOBAR */

/* SWITCH3
 * switch on integer precision
 */

#define SWITCH3(func)							\
									\
		switch( MACHINE_PREC(oap->oa_dp[0]) ){			\
			case PREC_BY:  by_obj_##func(oap); break;	\
			case PREC_IN:  in_obj_##func(oap); break;	\
			case PREC_DI:  di_obj_##func(oap); break;	\
			case PREC_UBY:  uby_obj_##func(oap); break;	\
			case PREC_UIN:  uin_obj_##func(oap); break;	\
			case PREC_UDI:  udi_obj_##func(oap); break;	\
			default:					\
				NWARN("SWITCH3:  missing precision case");\
				break;					\
		}

#define INT_FUNC( func )						\
void func (Vec_Obj_Args *oap){SWITCH3( func ) } 

INT_FUNC( vand   )
INT_FUNC( vnand  )
INT_FUNC( vor    )
INT_FUNC( vxor   )
INT_FUNC( vnot   )
INT_FUNC( vsand  )
INT_FUNC( vshr  )
INT_FUNC( vcomp  )
INT_FUNC( vsshr  )
INT_FUNC( vsshr2  )
INT_FUNC( vshl  )
INT_FUNC( vsshl  )
INT_FUNC( vsshl2  )
INT_FUNC( vsor   )
INT_FUNC( vsxor  )
INT_FUNC( vsnand )
INT_FUNC( vmod   )
INT_FUNC( vsmod  )
INT_FUNC( vsmod2 )

#define DSWITCH2( funcname, spf, dpf )					\
	switch( MACHINE_PREC(srcdp) ){					\
		case PREC_SP:	spf(&oargs);	break;			\
		case PREC_DP:	dpf(&oargs);	break;			\
		default: 	sprintf(DEFAULT_ERROR_STRING,			\
	"DSWITCH2 (%s):  object %s has bad machine precision (%s)",	\
		funcname,srcdp->dt_name,prec_name[MACHINE_PREC(srcdp)]);\
				NWARN(DEFAULT_ERROR_STRING);		\
							break;		\
	}

#define ISWITCH2( funcname, spf, dpf, _is_inv )				\
	switch( MACHINE_PREC(srcdp) ){					\
		case PREC_SP:	spf(&oargs, _is_inv);	break;		\
		case PREC_DP:	dpf(&oargs, _is_inv);	break;		\
		default: 	sprintf(DEFAULT_ERROR_STRING,			\
	"ISWITCH2 (%s):  object %s has bad machine precision %s",	\
		funcname,srcdp->dt_name,prec_name[MACHINE_PREC(srcdp)]);\
				NWARN(DEFAULT_ERROR_STRING);		\
							break;		\
	}

void fft2d(Data_Obj *dstdp,Data_Obj *srcdp)
{
	Vec_Obj_Args oargs;

	setvarg2(&oargs,dstdp,srcdp);
	if( IS_COMPLEX( srcdp ) ){
		ISWITCH2( "fft2d", sp_obj_c_2dfft, dp_obj_c_2dfft, FWD_FFT )
	} else {
		DSWITCH2( "fft2d", sp_obj_r_2dfft, dp_obj_r_2dfft )
	}
}

void fftrows(Data_Obj *dstdp,Data_Obj *srcdp)
{
	Vec_Obj_Args oargs;

	setvarg2(&oargs,dstdp,srcdp);
	if( IS_COMPLEX( srcdp ) ){
		ISWITCH2( "fftrows", sp_obj_c_rowfft, dp_obj_c_rowfft, FWD_FFT )
	} else {
		DSWITCH2( "fftrows", sp_obj_r_rowfft, dp_obj_r_rowfft )
	}
}

void ift2d(Data_Obj *dstdp,Data_Obj *srcdp)
{
	Vec_Obj_Args oargs;

	oargs.oa_dest = dstdp;
	oargs.oa_1 = srcdp;
	oargs.oa_2 = oargs.oa_3 = oargs.oa_s1 = oargs.oa_s2 = NO_OBJ;

	if( IS_COMPLEX( dstdp ) ){
		ISWITCH2( "fft2d", sp_obj_c_2dfft, dp_obj_c_2dfft, INV_FFT )
	} else {
		DSWITCH2( "ift2d", sp_obj_r_2dift, dp_obj_r_2dift )
	}
}

void iftrows(Data_Obj *dstdp,Data_Obj *srcdp)
{
	Vec_Obj_Args oargs;

	oargs.oa_dest = dstdp;
	oargs.oa_1 = srcdp;
	oargs.oa_2 = oargs.oa_3 = oargs.oa_s1 = oargs.oa_s2 = NO_OBJ;

	if( IS_COMPLEX( dstdp ) ){
		ISWITCH2( "fftrows", sp_obj_c_rowfft, dp_obj_c_rowfft, INV_FFT )
	} else {
		DSWITCH2( "ift2d", sp_obj_r_rowift, dp_obj_r_rowift )
	}
}

