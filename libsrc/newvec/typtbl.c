#include "quip_config.h"

char VersionId_newvec_typtbl[] = QUIP_VERSION_STRING;

#include <stdlib.h>	/* abort */

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
#include "ubyin_prot.h"
#include "inby_prot.h"
#include "uindi_prot.h"
#include "spdp_prot.h"
#include "bm_prot.h"
//#include "convert.h"
#include "bitmap.h"

/* Now 12 is 13 for bitmap */

/* We have a table of functions that take the following arg types:
 *
 *	byte, short, long, float, double, u_byte,
 *	u_short, u_long, u_byte/short, short/byte, u_short/long, float/double, bitmap
 *
 * This is one row of the table - 
 *  then the table has separate rows for:
 *		real
 *		complex
 *		mixed (r/c)
 *		quaternion
 *		mixed (r/q)
 *
 */

#define XNAME(typ,stem)		typ##_obj_##stem
#define RNAME(typ,stem)		typ##_obj_r##stem
#define CNAME(typ,stem)		typ##_obj_c##stem
#define MNAME(typ,stem)		typ##_obj_m##stem
#define QNAME(typ,stem)		typ##_obj_q##stem
#define _PNAME(typ,stem)	typ##_obj_p##stem

#ifdef USE_SSE
#define SIMD_NAME(stem)		simd_##stem
#else /* ! USE_SSE */
#define SIMD_NAME(stem)		nullobjf
#endif /* ! USE_SSE */

#define NULL_5							\
								\
          nullobjf, nullobjf, nullobjf, nullobjf, nullobjf

#define NULL_4							\
								\
          nullobjf, nullobjf, nullobjf, nullobjf

#define NULL_3							\
								\
          nullobjf, nullobjf, nullobjf

#define NULL_2							\
								\
          nullobjf, nullobjf

#define DUP_4(stem)						\
								\
	stem, stem, stem, stem
	
#define DUP_2(stem)						\
								\
	stem, stem

#define DUP_3(stem)						\
								\
	stem, stem, stem

#define ALL_NULL						\
								\
	  NULL_4,						\
	  NULL_2,						\
	  NULL_4,						\
	  NULL_5

#define DUP_ALL(stem,bmfunc)					\
								\
	DUP_4(stem),						\
	DUP_2(stem),						\
	DUP_4(stem),						\
	DUP_4(stem),						\
	bmfunc

#define FLT_ALL( stem )						\
								\
	NULL_4,							\
        sp_obj_##stem, dp_obj_##stem,				\
	NULL_4,							\
	NULL_3,							\
	spdp_obj_##stem, nullobjf

#define RFLT_ALL(stem)						\
								\
	NULL_4,							\
	sp_obj_r##stem, dp_obj_r##stem,				\
	NULL_4,							\
	NULL_3,							\
	spdp_obj_r##stem, nullobjf

#define RFLT_NO_MIXED(stem)					\
								\
	NULL_4,							\
	sp_obj_r##stem, dp_obj_r##stem,				\
	NULL_4,							\
	NULL_5

#define CFLT_ALL(stem)						\
								\
	NULL_4,							\
	sp_obj_c##stem, dp_obj_c##stem,				\
	NULL_4,							\
	NULL_3,							\
	spdp_obj_c##stem, nullobjf

#define CFLT_NO_MIXED(stem)					\
								\
	NULL_4,							\
	sp_obj_c##stem, dp_obj_c##stem,				\
	NULL_4,							\
	NULL_5

#define MFLT_ALL(stem)						\
								\
	NULL_4,							\
	sp_obj_m##stem, dp_obj_m##stem,				\
	NULL_4,							\
	NULL_3,							\
	spdp_obj_m##stem, nullobjf

#define INT_ALL(stem)						\
								\
	XNAME(by,stem),						\
	XNAME(in,stem),						\
	XNAME(di,stem),						\
	XNAME(li,stem),						\
	nullobjf,						\
	nullobjf,						\
	XNAME(uby,stem),					\
	XNAME(uin,stem),					\
	XNAME(udi,stem),					\
	XNAME(uli,stem),					\
	XNAME(ubyin,stem),					\
	XNAME(inby,stem),					\
	XNAME(uindi,stem),					\
	nullobjf,						\
	XNAME(bitmap,stem)


/* this is for vneg */

#define SIGNED_ALL_REAL(stem)					\
								\
	RNAME(by,stem),						\
	RNAME(in,stem),						\
	RNAME(di,stem),						\
	RNAME(li,stem),						\
	RNAME(sp,stem),						\
	RNAME(dp,stem),						\
	nullobjf,	/* uby */				\
	nullobjf,	/* uin */				\
	nullobjf,	/* udi */				\
	nullobjf,	/* uli */				\
	nullobjf,	/* ubyin */				\
	RNAME(inby,stem),					\
	nullobjf,	/* uindi */				\
	RNAME(spdp,stem),					\
	nullobjf	/* bit */

/* 12B means no Bitmap */

#define INT_ALL_NO_BITMAP(stem)					\
								\
	XNAME(by,stem),						\
	XNAME(in,stem),						\
	XNAME(di,stem),						\
	XNAME(li,stem),						\
	nullobjf,						\
	nullobjf,						\
	XNAME(uby,stem),					\
	XNAME(uin,stem),					\
	XNAME(udi,stem),					\
	XNAME(uli,stem),					\
	XNAME(ubyin,stem),					\
	XNAME(inby,stem),					\
	XNAME(uindi,stem),					\
	nullobjf,						\
	nullobjf


#define REAL_INT_ALL(stem)					\
								\
	byr##stem,  inr##stem,  dir##stem,  lir##stem,		\
	nullobjf, nullobjf,					\
	ubyr##stem, uinr##stem, udir##stem, uli##stem,		\
	ubyinr##stem, inbyr##stem, uindir##stem, nullobjf, nullobjf


#define ALL_NO_BITMAP(stem)					\
								\
	by_obj_##stem,  in_obj_##stem,  di_obj_##stem,  li_obj_##stem,	\
	sp_obj_##stem, dp_obj_##stem,				\
	uby_obj_##stem, uin_obj_##stem, udi_obj_##stem, uli_obj_##stem,	\
	ubyin_obj_##stem, inby_obj_##stem, uindi_obj_##stem, spdp_obj_##stem, nullobjf

#define ALL_REAL_NO_BITMAP_SSE(stem)				\
								\
	RNAME(by,stem),						\
	RNAME(in,stem),						\
	RNAME(di,stem),						\
	RNAME(li,stem),						\
	stem,							\
	/* SIMD_NAME(stem), */					\
	/* RNAME(sp,stem), */					\
	RNAME(dp,stem),						\
	RNAME(uby,stem),					\
	RNAME(uin,stem),					\
	RNAME(udi,stem),					\
	RNAME(uli,stem),					\
	RNAME(ubyin,stem),					\
	RNAME(inby,stem),					\
	RNAME(uindi,stem),					\
	RNAME(spdp,stem),					\
	nullobjf

#define ALL_REAL_SSE(stem)					\
								\
	RNAME(by,stem),						\
	RNAME(in,stem),						\
	RNAME(di,stem),						\
	RNAME(li,stem),						\
	stem,							\
	/* SIMD_NAME(stem), */					\
	/* RNAME(sp,stem), */					\
	RNAME(dp,stem),						\
	RNAME(uby,stem),					\
	RNAME(uin,stem),					\
	RNAME(udi,stem),					\
	RNAME(uli,stem),					\
	RNAME(ubyin,stem),					\
	RNAME(inby,stem),					\
	RNAME(uindi,stem),					\
	RNAME(spdp,stem),					\
	RNAME(bm,stem)

/* really 13 now... */

#define ALL_REAL(stem)						\
								\
	RNAME(by,stem),						\
	RNAME(in,stem),						\
	RNAME(di,stem),						\
	RNAME(li,stem),						\
	RNAME(sp,stem),						\
	RNAME(dp,stem),						\
	RNAME(uby,stem),					\
	RNAME(uin,stem),					\
	RNAME(udi,stem),					\
	RNAME(uli,stem),					\
	RNAME(ubyin,stem),					\
	RNAME(inby,stem),					\
	RNAME(uindi,stem),					\
	RNAME(spdp,stem),					\
	RNAME(bm,stem)

#define ALL_REAL_NO_BITMAP(stem)				\
								\
	RNAME(by,stem),						\
	RNAME(in,stem),						\
	RNAME(di,stem),						\
	RNAME(li,stem),						\
	RNAME(sp,stem),						\
	RNAME(dp,stem),						\
	RNAME(uby,stem),					\
	RNAME(uin,stem),					\
	RNAME(udi,stem),					\
	RNAME(uli,stem),					\
	RNAME(ubyin,stem),					\
	RNAME(inby,stem),					\
	RNAME(uindi,stem),					\
	RNAME(spdp,stem),					\
	nullobjf

#define ALL_COMPLEX(stem)						\
								\
	NULL_4,							\
	CNAME(sp,stem),						\
	CNAME(dp,stem),						\
	NULL_4,							\
	NULL_3,							\
	CNAME(spdp,stem),					\
	nullobjf


#define ALL_MIXED(stem)						\
								\
	NULL_4,							\
	MNAME(sp,stem),						\
	MNAME(dp,stem),						\
	NULL_4,							\
	NULL_3,							\
	MNAME(spdp,stem),					\
	nullobjf

#define ALL_QUAT(stem)						\
								\
	NULL_4,							\
	QNAME(sp,stem),						\
	QNAME(dp,stem),						\
	NULL_4,							\
	NULL_3,							\
	QNAME(spdp,stem),					\
	nullobjf

#define ALL_QMIXD(stem)						\
								\
	NULL_4,							\
	_PNAME(sp,stem),					\
	_PNAME(dp,stem),					\
	NULL_4,							\
	NULL_3,							\
	_PNAME(spdp,stem),					\
	nullobjf


#define NULL_ARR( stem, code )						\
	{ code, { ALL_NULL , ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }

#define CONV_ARR( stem, code, bmfunc )					\
	{ code, { DUP_ALL(stem,bmfunc), DUP_ALL(stem,bmfunc), ALL_NULL, DUP_ALL(stem,bmfunc), ALL_NULL } }

#define CFLT_ARR( stem, code )						\
	{ code, { ALL_NULL, FLT_ALL(stem), ALL_NULL, ALL_NULL, ALL_NULL } }

#define RFLT_ARR( stem, code )						\
	{ code, { FLT_ALL(stem), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }

#define RCFLT_ARR2( stem, code )					\
	{ code, { RFLT_NO_MIXED(stem), CFLT_NO_MIXED(stem), ALL_NULL, ALL_NULL, ALL_NULL } }

#define RCFLT_ARR( stem, code )						\
	{ code, { RFLT_ALL(stem), CFLT_ALL(stem), ALL_NULL, ALL_NULL, ALL_NULL } }

#define FLT_ARR( stem, code )						\
	{ code, { RFLT_ALL(stem), CFLT_ALL(stem), MFLT_ALL(stem), ALL_NULL, ALL_NULL } }

#define ALL_ARR( stem, code )						\
	{ code, { ALL_REAL_NO_BITMAP(stem), ALL_COMPLEX(stem), ALL_MIXED(stem), ALL_NULL, ALL_NULL } }

#ifdef FOOBAR
/* used to have this for vneg, no more. */
#define SIGNED_ARR( stem, code )						\
	{ code, { SIGNED_ALL_REAL(stem), ALL_COMPLEX(stem), ALL_NULL, ALL_QUAT(stem), ALL_NULL } }
#endif /* FOOBAR */

#define RALL_ARR( stem, code )						\
	{ code, { ALL_NO_BITMAP(stem), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL	} }

#define RCALL_ARR( stem, code )						\
	{ code, { ALL_REAL_NO_BITMAP(stem), ALL_COMPLEX(stem), ALL_NULL, ALL_NULL, ALL_NULL } }

#define QALL_ARR( stem, code )						\
	{ code, { ALL_REAL_NO_BITMAP(stem), ALL_COMPLEX(stem), ALL_MIXED(stem), ALL_QUAT(stem), ALL_NULL } }

#define RCQALL_ARR( stem, code )					\
	{ code, { ALL_REAL_NO_BITMAP(stem), ALL_COMPLEX(stem), ALL_NULL, ALL_QUAT(stem), ALL_NULL } }

#define RCQPALL_ARR( stem, code )					\
	{ code, { ALL_REAL_NO_BITMAP(stem), ALL_COMPLEX(stem), ALL_NULL, ALL_QUAT(stem), ALL_QMIXD(stem) } }

#define CMALL_ARR( stem, code )						\
	{ code, { ALL_NULL, ALL_COMPLEX(stem), ALL_MIXED(stem), ALL_NULL, ALL_NULL } }

#define QPALL_ARR( stem, code )						\
	{ code, { ALL_NULL, ALL_NULL, ALL_NULL, ALL_QUAT(stem), ALL_QMIXD(stem) } }

#define RCMQPALL_ARR( stem, code )					\
	{ code, { ALL_REAL_NO_BITMAP(stem), ALL_COMPLEX(stem), ALL_MIXED(stem), ALL_QUAT(stem), ALL_QMIXD(stem) } }

#define RC_FIXED_ARR( stem, code, bm_func )				\
	{ code, { DUP_ALL(stem,bm_func), DUP_ALL(stem,nullobjf), ALL_NULL, ALL_NULL, ALL_NULL } }


#ifdef USE_SSE

#define ALL_ARR_SSE( stem, code )					\
	{ code, { ALL_REAL_NO_BITMAP_SSE(stem), ALL_COMPLEX(stem), ALL_MIXED(stem), ALL_NULL, ALL_NULL } }
#ifdef FOOBAR
/* This is not used anywhere??? */
#define RCALL_ARR_SSE( stem, code )					\
	{ code, { /* ALL_REAL_NO_BITMAP_SSE */ALL_REAL_SSE(stem), ALL_COMPLEX(stem), ALL_NULL, ALL_NULL, ALL_NULL } }
#endif

#define RCQALL_ARR_SSE( stem, code )					\
	{ code, { ALL_REAL_NO_BITMAP_SSE(stem), ALL_COMPLEX(stem), ALL_NULL, ALL_QUAT(stem), ALL_NULL } }
#define QALL_ARR_SSE( stem, code )					\
	{ code, { ALL_REAL_NO_BITMAP_SSE(stem), ALL_COMPLEX(stem), ALL_MIXED(stem), ALL_QUAT(stem), ALL_NULL } }
#define RCMQPALL_ARR_SSE( stem, code )					\
	{ code, { ALL_REAL_NO_BITMAP_SSE(stem), ALL_COMPLEX(stem), ALL_MIXED(stem), ALL_QUAT(stem), ALL_QMIXD(stem) } }

#else /* ! USE_SSE */

#ifdef FOOBAR
#define RCALL_ARR_SSE( stem, code )		RCALL_ARR( stem, code )
#endif

#define RCQALL_ARR_SSE( stem, code )		RCQALL_ARR( stem, code )
#define ALL_ARR_SSE( stem, code )		ALL_ARR( stem, code ) 
#define QALL_ARR_SSE( stem, code )		QALL_ARR( stem, code )
#define RCMQPALL_ARR_SSE( stem, code )		RCMQPALL_ARR( stem, code )

#endif /* ! USE_SSE */


#define INT_ARR( stem, code )						\
	{ code, { REAL_INT_ALL(stem), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }

#define REAL_INT_ARR( stem, code )					\
	{ code, { INT_ALL(stem), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }

#define REAL_INT_ARR_NO_BITMAP( stem, code )				\
	{ code, { INT_ALL_NO_BITMAP(stem), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }

static void nullobjf(Vec_Obj_Args *oap)
{
	advise("nullobjf:");
	sprintf(DEFAULT_ERROR_STRING,
		"Oops, function %s has not been implemented for %s %s precision (functype = %d)",
		this_vfp->vf_name, type_strings[oap->oa_functype%N_ARGSET_PRECISIONS],
		argset_type_name[(oap->oa_functype/N_ARGSET_PRECISIONS)+1],oap->oa_functype);
	NWARN(DEFAULT_ERROR_STRING);
	advise("Need to add better error checking!");
	abort();
}


Vec_Func_Array vfa_tbl[N_VEC_FUNCS]={

/* RCQALL_ARR_SSE(		vset,		FVSET ), */
/* ALL_REAL instead of ALL_REAL_NO_BITMAP */
	{ FVSET, { ALL_REAL(vset), ALL_COMPLEX(vset), ALL_NULL, ALL_QUAT(vset), ALL_NULL } },

RCQALL_ARR_SSE(		vmov,		FVMOV ),
RCMQPALL_ARR_SSE(	vadd,		FVADD ),
RCMQPALL_ARR_SSE(	vsub,		FVSUB ),
RCMQPALL_ARR_SSE(	vmul,		FVMUL ),
RCMQPALL_ARR_SSE(	vdiv,		FVDIV ),
RCQALL_ARR(		vneg,		FVNEG ),
RCQALL_ARR(		vsqr,		FVSQR ),
RALL_ARR(		vramp1d,	FVRAMP1D ),
RALL_ARR(		vramp2d,	FVRAMP2D ),
RCMQPALL_ARR(		vsadd,		FVSADD ),
RCMQPALL_ARR(		vssub,		FVSSUB ),
RCMQPALL_ARR(		vsmul,		FVSMUL ),
RCMQPALL_ARR(		vsdiv,		FVSDIV ),
RCMQPALL_ARR(		vsdiv2,		FVSDIV2 ),
RALL_ARR(		vabs,		FVABS ),
RALL_ARR(		vsign,		FVSIGN ),

RFLT_ARR(		vsqrt,		FVSQRT ),		/* 0 */
RFLT_ARR(		vsin,		FVSIN ),
RFLT_ARR(		vcos,		FVCOS ),
RFLT_ARR(		vtan,		FVTAN ),
RFLT_ARR(		vatan,		FVATAN ),
RFLT_ARR(		vatan2,		FVATAN2 ),
RFLT_ARR(		vsatan2,	FVSATAN2 ),
RFLT_ARR(		vsatan22,	FVSATAN22 ),
RFLT_ARR(		vlog,		FVLOG ),
RFLT_ARR(		vlog10,		FVLOG10 ),
RFLT_ARR(		vexp,		FVEXP ),
RFLT_ARR(		verf,		FVERF ),
RFLT_ARR(		rvpow,		FVPOW ),
RFLT_ARR(		vspow,		FVSPOW ),
RFLT_ARR(		vspow2,		FVSPOW2 ),

RALL_ARR(		vmin,		FVMIN ),			/* 10 */
RALL_ARR(		vmax,		FVMAX ),
RALL_ARR(		vminm,		FVMINM ),
RALL_ARR(		vmaxm,		FVMAXM ),
RALL_ARR(		vsmin,		FVSMIN ),
RALL_ARR(		vsmax,		FVSMAX ),
RALL_ARR(		vsmnm,		FVSMNM ),
RALL_ARR(		vsmxm,		FVSMXM ),

RALL_ARR(		vminv,		FVMINV ),
RALL_ARR(		vmaxv,		FVMAXV ),
RALL_ARR(		vmnmv,		FVMNMV ),
RALL_ARR(		vmxmv,		FVMXMV ),
RALL_ARR(		vmini,		FVMINI ),
RALL_ARR(		vmaxi,		FVMAXI ),
RALL_ARR(		vmnmi,		FVMNMI ),
RALL_ARR(		vmxmi,		FVMXMI ),
RALL_ARR(		vming,		FVMING ),
RALL_ARR(		vmaxg,		FVMAXG ),
RALL_ARR(		vmnmg,		FVMNMG ),
RALL_ARR(		vmxmg,		FVMXMG ),

RFLT_ARR(		vfloor,		FVFLOOR ),
RFLT_ARR(		vround,		FVROUND ),
RFLT_ARR(		vceil,		FVCEIL ),
RFLT_ARR(		vrint,		FVRINT ),
RFLT_ARR(		vj0,		FVJ0 ),
RFLT_ARR(		vj1,		FVJ1 ),
RFLT_ARR(		vacos,		FVACOS ),
RFLT_ARR(		vasin,		FVASIN ),
RFLT_ARR(		vatn2,		FVATN2 ),
RFLT_ARR(		vuni,		FVUNI ),

/* bitwise operators,	integer only math */
REAL_INT_ARR(		vand,		FVAND ),
REAL_INT_ARR(		vnand,		FVNAND ),
REAL_INT_ARR(		vor,		FVOR ),
REAL_INT_ARR(		vxor,		FVXOR ),
REAL_INT_ARR(		vnot,		FVNOT ),
REAL_INT_ARR(		vcomp,		FVCOMP ),

REAL_INT_ARR(		vmod,		FVMOD ),
REAL_INT_ARR(		vsmod,		FVSMOD ),
REAL_INT_ARR(		vsmod2,		FVSMOD2 ),

REAL_INT_ARR(		vsand,		FVSAND ),
REAL_INT_ARR(		vsor,		FVSOR ),
REAL_INT_ARR(		vsxor,		FVSXOR ),
REAL_INT_ARR(		vshr,		FVSHR ),
REAL_INT_ARR(		vsshr,		FVSSHR ),
REAL_INT_ARR(		vsshr2,		FVSSHR2 ),
REAL_INT_ARR(		vshl,		FVSHL ),
REAL_INT_ARR(		vsshl,		FVSSHL ),
REAL_INT_ARR(		vsshl2,		FVSSHL2 ),

/* RC_FIXED_ARR(	vmov,	FVMOV, bmvmov ), */

RCQALL_ARR(		vsum,		FVSUM ),
RCALL_ARR(		vdot,		FVDOT ),
RCALL_ARR(		vrand,		FVRAND ),

RALL_ARR(		vsm_lt,		FVSMLT ),
RALL_ARR(		vsm_gt,		FVSMGT ),
RALL_ARR(		vsm_le,		FVSMLE ),
RALL_ARR(		vsm_ge,		FVSMGE ),
RALL_ARR(		vsm_ne,		FVSMNE ),
RALL_ARR(		vsm_eq,		FVSMEQ ),

RALL_ARR(		vvm_lt,		FVVMLT ),
RALL_ARR(		vvm_gt,		FVVMGT ),
RALL_ARR(		vvm_le,		FVVMLE ),
RALL_ARR(		vvm_ge,		FVVMGE ),
RALL_ARR(		vvm_ne,		FVVMNE ),
RALL_ARR(		vvm_eq,		FVVMEQ ),

RCQALL_ARR(		vvv_slct,	FVVVSLCT ),
RCQALL_ARR(		vvs_slct,	FVVSSLCT ),
RCQALL_ARR(		vss_slct,	FVSSSLCT ),

RALL_ARR(		vv_vv_lt,	FVV_VV_LT ),
RALL_ARR(		vv_vv_gt,	FVV_VV_GT ),
RALL_ARR(		vv_vv_le,	FVV_VV_LE ),
RALL_ARR(		vv_vv_ge,	FVV_VV_GE ),
RALL_ARR(		vv_vv_eq,	FVV_VV_EQ ),
RALL_ARR(		vv_vv_ne,	FVV_VV_NE ),

RALL_ARR(		vv_vs_lt,	FVV_VS_LT ),
RALL_ARR(		vv_vs_gt,	FVV_VS_GT ),
RALL_ARR(		vv_vs_le,	FVV_VS_LE ),
RALL_ARR(		vv_vs_ge,	FVV_VS_GE ),
RALL_ARR(		vv_vs_eq,	FVV_VS_EQ ),
RALL_ARR(		vv_vs_ne,	FVV_VS_NE ),

RALL_ARR(		vs_vv_lt,	FVS_VV_LT ),
RALL_ARR(		vs_vv_gt,	FVS_VV_GT ),
RALL_ARR(		vs_vv_le,	FVS_VV_LE ),
RALL_ARR(		vs_vv_ge,	FVS_VV_GE ),
RALL_ARR(		vs_vv_eq,	FVS_VV_EQ ),
RALL_ARR(		vs_vv_ne,	FVS_VV_NE ),

RALL_ARR(		vs_vs_lt,	FVS_VS_LT ),
RALL_ARR(		vs_vs_gt,	FVS_VS_GT ),
RALL_ARR(		vs_vs_le,	FVS_VS_LE ),
RALL_ARR(		vs_vs_ge,	FVS_VS_GE ),
RALL_ARR(		vs_vs_eq,	FVS_VS_EQ ),
RALL_ARR(		vs_vs_ne,	FVS_VS_NE ),

RALL_ARR(		ss_vv_lt,	FSS_VV_LT ),
RALL_ARR(		ss_vv_gt,	FSS_VV_GT ),
RALL_ARR(		ss_vv_le,	FSS_VV_LE ),
RALL_ARR(		ss_vv_ge,	FSS_VV_GE ),
RALL_ARR(		ss_vv_eq,	FSS_VV_EQ ),
RALL_ARR(		ss_vv_ne,	FSS_VV_NE ),

RALL_ARR(		ss_vs_lt,	FSS_VS_LT ),
RALL_ARR(		ss_vs_gt,	FSS_VS_GT ),
RALL_ARR(		ss_vs_le,	FSS_VS_LE ),
RALL_ARR(		ss_vs_ge,	FSS_VS_GE ),
RALL_ARR(		ss_vs_eq,	FSS_VS_EQ ),
RALL_ARR(		ss_vs_ne,	FSS_VS_NE ),

/*
FUNC_ARR(	vscmm,	FVSCMM ),
FUNC_ARR(	vmcmm,	FVMCMM ),
FUNC_ARR(	vmcmp,	FVMCMP ),
*/

/* complex stuff */

RFLT_ARR(		vmgsq,		FVMGSQ ),
CFLT_ARR(		vcmul,		FVCMUL ),
RFLT_ARR(		vscml,		FVSCML ),
CFLT_ARR(		vconj,		FVCONJ ),
RCFLT_ARR2(		vfft,		FVFFT ),
RCFLT_ARR2(		vift,		FVIFT ),

RALL_ARR(		vbnd,		FVBND ),
RALL_ARR(		vibnd,		FVIBND ),
RALL_ARR(		vclip,		FVCLIP ),
RALL_ARR(		viclp,		FVICLP ),
RALL_ARR(		vcmp,		FVCMP ),
RALL_ARR(		vscmp,		FVSCMP ),
RALL_ARR(		vscmp2,		FVSCMP2 ),

/* Type conversions
 *
 * For now, bitmaps are constrained to be a single unsigned type,
 * determined at compile time.  But here the conversion/unconversion
 * functions are installed for all unsigned types, regardless of which
 * one is actually used for bitmaps.  This should be safe, because
 * these are only called when one object is a bitmap, and that should
 * never be the wrong type...
 */

CONV_ARR(		vby2in,		FVB2I,		nullobjf ),
CONV_ARR(		vby2di,		FVB2L,		nullobjf ),
CONV_ARR(		vby2li,		FVB2LL,		nullobjf ),
CONV_ARR(		vby2sp,		FVB2SP,		nullobjf ),
CONV_ARR(		vby2dp,		FVB2DP,		nullobjf ),
CONV_ARR(		vby2uby,	FVB2UB,		by_obj_vconv_to_bit ),
CONV_ARR(		vby2uin,	FVB2UI,		by_obj_vconv_to_bit ),
CONV_ARR(		vby2udi,	FVB2UL,  	by_obj_vconv_to_bit ),
CONV_ARR(		vby2uli,	FVB2ULL,	by_obj_vconv_to_bit ),

CONV_ARR(		vin2by,		FVI2B,		nullobjf ),
CONV_ARR(		vin2di,		FVI2L,		nullobjf ),
CONV_ARR(		vin2li,		FVI2LL,		nullobjf ),
CONV_ARR(		vin2sp,		FVI2SP,		nullobjf ),
CONV_ARR(		vin2dp,		FVI2DP,		nullobjf ),
CONV_ARR(		vin2uby,	FVI2UB,		in_obj_vconv_to_bit ),
CONV_ARR(		vin2uin,	FVI2UI,		in_obj_vconv_to_bit ),
CONV_ARR(		vin2udi,	FVI2UL,  	in_obj_vconv_to_bit ),
CONV_ARR(		vin2uli,	FVI2ULL,	in_obj_vconv_to_bit ),

CONV_ARR(		vdi2by,		FVL2B,		nullobjf ),
CONV_ARR(		vdi2in,		FVL2I,		nullobjf ),
CONV_ARR(		vdi2li,		FVL2LL,		nullobjf ),
CONV_ARR(		vdi2sp,		FVL2SP,		nullobjf ),
CONV_ARR(		vdi2dp,		FVL2DP,		nullobjf ),
CONV_ARR(		vdi2uby,	FVL2UB,		di_obj_vconv_to_bit ),
CONV_ARR(		vdi2uin,	FVL2UI,		di_obj_vconv_to_bit ),
CONV_ARR(		vdi2udi,	FVL2UL,  	di_obj_vconv_to_bit ),
CONV_ARR(		vdi2uli,	FVL2ULL,	di_obj_vconv_to_bit ),

CONV_ARR(		vli2by,		FVLL2B,		nullobjf ),
CONV_ARR(		vli2in,		FVLL2I,		nullobjf ),
CONV_ARR(		vli2di,		FVLL2L,		nullobjf ),
CONV_ARR(		vli2sp,		FVLL2SP,	nullobjf ),
CONV_ARR(		vli2dp,		FVLL2DP,	li_obj_vconv_to_bit ),
CONV_ARR(		vli2uby,	FVLL2UB,	li_obj_vconv_to_bit ),
CONV_ARR(		vli2uin,	FVLL2UI,	li_obj_vconv_to_bit ),
CONV_ARR(		vli2udi,	FVLL2UL,	li_obj_vconv_to_bit ),
CONV_ARR(		vli2uli,	FVLL2ULL,	li_obj_vconv_to_bit ),

CONV_ARR(		vsp2by,		FVSP2B,		nullobjf ),
CONV_ARR(		vsp2in,		FVSP2I,		nullobjf ),
CONV_ARR(		vsp2di,		FVSP2L,		nullobjf ),
CONV_ARR(		vsp2li,		FVSP2LL,	nullobjf ),
CONV_ARR(		vsp2dp,		FVSPDP,		nullobjf ),
CONV_ARR(		vsp2uby,	FVSP2UB,	sp_obj_vconv_to_bit ),
CONV_ARR(		vsp2uin,	FVSP2UI,	sp_obj_vconv_to_bit ),
CONV_ARR(		vsp2udi,	FVSP2UL,	sp_obj_vconv_to_bit ),
CONV_ARR(		vsp2uli,	FVSP2ULL,	sp_obj_vconv_to_bit ),

CONV_ARR(		vdp2by,		FVDP2B,		nullobjf ),
CONV_ARR(		vdp2in,		FVDP2I,		nullobjf ),
CONV_ARR(		vdp2di,		FVDP2L,		nullobjf ),
CONV_ARR(		vdp2li,		FVDP2LL,	nullobjf ),
CONV_ARR(		vdp2sp,		FVDPSP,		nullobjf ),
CONV_ARR(		vdp2uby,	FVDP2UB,	dp_obj_vconv_to_bit ),
CONV_ARR(		vdp2uin,	FVDP2UI,	dp_obj_vconv_to_bit ),
CONV_ARR(		vdp2udi,	FVDP2UL,	dp_obj_vconv_to_bit ),
CONV_ARR(		vdp2uli,	FVDP2ULL,	dp_obj_vconv_to_bit ),

/* unsigned conversions */

CONV_ARR(		vuby2by,	FVUB2B,		by_obj_vconv_from_bit ),
CONV_ARR(		vuby2in,	FVUB2I,		in_obj_vconv_from_bit ),
CONV_ARR(		vuby2di,	FVUB2L,		di_obj_vconv_from_bit ),
CONV_ARR(		vuby2li,	FVUB2LL,	li_obj_vconv_from_bit ),
CONV_ARR(		vuby2sp,	FVUB2SP,	sp_obj_vconv_from_bit ),
CONV_ARR(		vuby2dp,	FVUB2DP,	dp_obj_vconv_from_bit ),
CONV_ARR(		vuby2uin,	FVUB2UI,	nullobjf ),
CONV_ARR(		vuby2udi,	FVUB2UL,	nullobjf ),
CONV_ARR(		vuby2uli,	FVUB2ULL,	nullobjf ),

CONV_ARR(		vuin2by,	FVUI2B,		by_obj_vconv_from_bit ),
CONV_ARR(		vuin2in,	FVUI2I,		in_obj_vconv_from_bit ),
CONV_ARR(		vuin2di,	FVUI2L,		di_obj_vconv_from_bit ),
CONV_ARR(		vuin2li,	FVUI2LL,	li_obj_vconv_from_bit ),
CONV_ARR(		vuin2sp,	FVUI2SP,	sp_obj_vconv_from_bit ),
CONV_ARR(		vuin2dp,	FVUI2DP,	dp_obj_vconv_from_bit ),
CONV_ARR(		vuin2uby,	FVUI2UB,	nullobjf ),
CONV_ARR(		vuin2udi,	FVUI2UL,	nullobjf ),
CONV_ARR(		vuin2uli,	FVUI2ULL,	nullobjf ),

CONV_ARR(		vudi2by,	FVUL2B,		by_obj_vconv_from_bit ),
CONV_ARR(		vudi2in,	FVUL2I,		in_obj_vconv_from_bit ),
CONV_ARR(		vudi2di,	FVUL2L,		di_obj_vconv_from_bit ),
CONV_ARR(		vudi2li,	FVUL2LL,	li_obj_vconv_from_bit ),
CONV_ARR(		vudi2sp,	FVUL2SP,	sp_obj_vconv_from_bit ),
CONV_ARR(		vudi2dp,	FVUL2DP,	dp_obj_vconv_from_bit ),
							/* should these be from or to bit??? */
CONV_ARR(		vudi2uby,	FVUL2UB,	nullobjf ),
CONV_ARR(		vudi2uin,	FVUL2UI,	nullobjf ),
CONV_ARR(		vudi2uli,	FVUL2ULL,	nullobjf ),

CONV_ARR(		vuli2by,	FVULL2B,	by_obj_vconv_from_bit ),
CONV_ARR(		vuli2in,	FVULL2I,	in_obj_vconv_from_bit ),
CONV_ARR(		vuli2di,	FVULL2L,	di_obj_vconv_from_bit ),
CONV_ARR(		vuli2li,	FVULL2LL,	li_obj_vconv_from_bit ),
CONV_ARR(		vuli2sp,	FVULL2SP,	sp_obj_vconv_from_bit ),
CONV_ARR(		vuli2dp,	FVULL2DP,	dp_obj_vconv_from_bit ),
CONV_ARR(		vuli2uby,	FVULL2UB,	nullobjf ),
CONV_ARR(		vuli2uin,	FVULL2UI,	nullobjf ),
CONV_ARR(		vuli2udi,	FVULL2UL,	nullobjf ),

};

