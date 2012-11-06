
/* defns shared by veclib & warlib */

#ifndef _VECGEN_H_
#define _VECGEN_H_


#include <stdio.h>
#include "typedefs.h"
#include "data_obj.h"

#define FWD_FFT		(-1)
#define INV_FFT		1

/* These aren't really types of numbers - the "mixed" types refer to operations... */
typedef enum {
	UNKNOWN_TYPE,
	TYPE_REAL,
	TYPE_COMPLEX,
	TYPE_QUATERNION,
	N_NUMBER_TYPES		/* must be last! */
} number_type;

typedef enum {
	UNKNOWN_ARGS,
	REAL_ARGS,
	COMPLEX_ARGS,
	MIXED_ARGS,		/* real/complex */
	QUATERNION_ARGS,
	QMIXED_ARGS,		/* real/quaternion */
	N_ARGSET_TYPES,
	INVALID_ARGSET_TYPE
} argset_type;	/* argstype */


#define VL_TYPE_MASK(code)		(1<<((code)-1))

/* Not the machine precisions, because we include psuedo precisions (bit) and some mixed combos */

typedef enum {
	BY_ARGS,	/* 0 */
	IN_ARGS,	/* 1 */
	DI_ARGS,	/* 2 */
	LI_ARGS,	/* 3 */
	SP_ARGS,	/* 4 */
	DP_ARGS,	/* 5 */
	UBY_ARGS,	/* 6 */
	UIN_ARGS,	/* 7 */
	UDI_ARGS,	/* 8 */
	ULI_ARGS,	/* 9 */
	BYIN_ARGS,	/* 10 */		/* uby args, short result */
	INBY_ARGS,	/* 11 */		/* what is this for??? */
	INDI_ARGS,	/* 12 */		/* uin args, int32 result */
	SPDP_ARGS,	/* 13 */		/* sp args, dp result */
	BIT_ARGS,	/* 14 */
	N_ARGSET_PRECISIONS,/* 15 */
	INVALID_ARGSET_PREC
} argset_prec;	/* argsprec */

#define ARGPREC_UNSPECIFIED	(-1)

#define FUNCTYPE_FOR( argsprec, arg_type )				\
		( argsprec + (arg_type-1) * N_ARGSET_PRECISIONS )

#define TELL_FUNCTYPE(argsprec,arg_type)				\
sprintf(error_string,"functype = %d, argsprec = %d ,arg_type = %d",\
( argsprec + (arg_type-1) * N_ARGSET_PRECISIONS ),argsprec,arg_type);\
advise(error_string);



#define	N_FUNCTION_TYPES	(N_ARGSET_PRECISIONS * N_ARGSET_TYPES)



#define ARGSET_PREC( prec )						\
									\
	( (argset_prec)( PSEUDO_PREC_INDEX(prec) == PP_BIT?		\
					BIT_ARGS :			\
					( ((prec)&MACH_PREC_MASK) - PREC_BY ) ))

/* PREC_FOR_ARGSET returns a normal precision code associated with an argset */
/* BUG we punt here if the argset prec is not a simple machine precision FIXME */
#define PREC_FOR_ARGSET( argsprec )					\
	( (argsprec) <= ULI_ARGS ? (argsprec + PREC_BY) : PREC_NONE )

/* MAX_N_ARGS originally was 3.
 * Increased to 4 to accomodate bitmaps.
 * Increased to 5 when new conditional-assignment ops were added,
 * so that the bitmap arg didn't have to share space with src4.
 */

#define MAX_SRC_OBJECTS	4
#define MAX_N_ARGS	(MAX_SRC_OBJECTS+1)	// up to 4 sources, bitmap arg doesn't share
#define MAX_RETSCAL_ARGS	2		// objects used for scalar returns
#define MAX_SRCSCAL_ARGS	3		// values used for scalar operands


typedef struct data_vec {
	void *		dv_vec;
	dimension_t	dv_count;
	incr_t		dv_inc;
	prec_t		dv_prec;
	int		dv_flags;
	int		dv_bit0;
} Data_Vector;

#define NO_DATA_VECTOR	((Data_Vector *)NULL)

/* what are the Data_Vector flags?? */
#define DV_BITMAP			1

#define DATA_VEC_IS_BITMAP(dvp)		((dvp)->dv_flags & DV_BITMAP)

typedef struct vec_objargs {
	Data_Obj *	oa_dp[MAX_N_ARGS];
	Data_Obj *	oa_dest;
	Data_Obj *	oa_sdp[MAX_RETSCAL_ARGS];	/* scalars */
	Scalar_Value *	oa_svp[MAX_SRCSCAL_ARGS];	/* scalar data */
	argset_type	oa_argstype;		/* real/complex/mixed etc. */
	argset_prec	oa_argsprec;		/* machine precision group */
	int		oa_functype;
	int		oa_flags;		/* introduced to support CUDA args */
} Vec_Obj_Args;

#define NO_VEC_OBJ_ARGS	((Vec_Obj_Args *) NULL)

/* Flag values */
#define OARGS_CHECKED	1
#define OARGS_RAM	2
#define OARGS_GPU	4

#define HAS_CHECKED_ARGS(oap)	( (oap)->oa_flags & OARGS_CHECKED )
#define HAS_RAM_ARGS(oap)	( (oap)->oa_flags & OARGS_CHECKED ? (oap)->oa_flags & OARGS_RAM : are_ram_args(oap) )
#define HAS_GPU_ARGS(oap)	( (oap)->oa_flags & OARGS_CHECKED ? (oap)->oa_flags & OARGS_GPU : are_gpu_args(oap) )

#define HAS_MIXED_ARGS(oap)	( (oap)->oa_argstype == MIXED_ARGS )
#define HAS_QMIXED_ARGS(oap)	( (oap)->oa_argstype == QMIXED_ARGS )

#define oa_1		oa_dp[0]
#define oa_2		oa_dp[1]
#define oa_3		oa_dp[2]
#define oa_4		oa_dp[3]
#define oa_bmap		oa_dp[4]
#define oa_s1		oa_sdp[0]
#define oa_s2		oa_sdp[1]

/* Data pointer args for speeded-up execution */

typedef struct spacing_info {
	incr_t *	spi_dst_incr;
	incr_t *	spi_src_incr[MAX_N_ARGS];
} Spacing_Info;

typedef struct size_info {
	dimension_t *	szi_dst_dim;
	dimension_t *	szi_src_dim[MAX_N_ARGS];
} Size_Info;

typedef struct vector_args {
	void *		va_dst_vp;
	void *		va_src_vp[MAX_N_ARGS];
	Spacing_Info *	va_spi_p;
	Size_Info *	va_szi_p;
	Scalar_Value	va_sval[3];		/* scalar data */
	dimension_t	va_bit0;		/* for bitmaps */
	dimension_t	va_len;			/* for fast args */

	/* do we need these? */
	argset_type	va_argstype;		/* real/complex/mixed etc. */
	argset_prec	va_argsprec;		/* machine precision group */
	int		va_functype;
	int		va_flags;
} Vector_Args;
	
#define va_dest		va_dst_vp
#define va_dinc		va_spi_p->spi_dst_incr
#define va_count	va_szi_p->szi_dst_dim

#define va_src1		va_src_vp[0]
#define va_src1_inc	va_spi_p->spi_src_incr[0]
#define va_src1_cnt	va_szi_p->szi_src_dim[0]

#define va_src2		va_src_vp[1]
#define va_src2_inc	va_spi_p->spi_src_incr[1]
#define va_src2_cnt	va_szi_p->szi_src_dim[1]

#define va_src3		va_src_vp[2]
#define va_src3_inc	va_spi_p->spi_src_incr[2]
#define va_src3_cnt	va_szi_p->szi_src_dim[2]

#define va_src4		va_src_vp[3]
#define va_src4_inc	va_spi_p->spi_src_incr[3]
#define va_src4_cnt	va_szi_p->szi_src_dim[3]

#define va_sbm_p	va_src_vp[4]
#define va_sbm_inc	va_spi_p->spi_src_incr[4]
#define va_sbm_count	va_szi_p->szi_src_dim[4]

#define scalar_val1	va_sval[0].std_scalar
#define scalar_val2	va_sval[1].std_scalar
#define scalar_val3	va_sval[2].std_scalar

#define cpx_scalar_val1	va_sval[0].std_cpx_scalar
#define cpx_scalar_val2	va_sval[1].std_cpx_scalar
#define cpx_scalar_val3	va_sval[2].std_cpx_scalar

#define quat_scalar_val1	va_sval[0].std_quat_scalar
#define quat_scalar_val2	va_sval[1].std_quat_scalar
#define quat_scalar_val3	va_sval[2].std_quat_scalar


#define CHAIN_CHECK( func )					\
								\
	if( is_chaining ){					\
		if( insure_static(oap) < 0 ) return;		\
		add_link( & func , &va1 );			\
		return;						\
	} else {						\
		func(&va1);					\
		oap->oa_dest->dt_flags |= DT_ASSIGNED;		\
	}


/* Now we subtract 1 because the 0 code is "unknown" */

#define R	VL_TYPE_MASK(REAL_ARGS)
#define C	VL_TYPE_MASK(COMPLEX_ARGS)
#define M	VL_TYPE_MASK(MIXED_ARGS)
#define Q	VL_TYPE_MASK(QUATERNION_ARGS)
#define P	VL_TYPE_MASK(QMIXED_ARGS)	/* lmnop, Q, R ...   P!  (well, what else?) */

#define RC	(R|C)
#define RCQ	(R|C|Q)
#define RCM	(R|C|M)
#define CM	(C|M)
#define RCMQ	(R|C|M|Q)
#define RCMQP	(R|C|M|Q|P)
#define QP	(Q|P)

/* masks for the allowable machine precisions */
#define M_BY	(1<<PREC_BY)
#define M_IN	(1<<PREC_IN)
#define M_DI	(1<<PREC_DI)
#define M_LI	(1<<PREC_LI)
#define M_SP	(1<<PREC_SP)
#define M_DP	(1<<PREC_DP)
#define M_UBY	(1<<PREC_UBY)
#define M_UIN	(1<<PREC_UIN)
#define M_UDI	(1<<PREC_UDI)
#define M_ULI	(1<<PREC_ULI)
#define M_MM	(1<<N_MACHINE_PRECS)

#define M_BP	(M_SP|M_DP)
#define M_ALL	(M_BY|M_IN|M_DI|M_LI|M_SP|M_DP|M_UBY|M_UIN|M_UDI|M_ULI)
#define M_ALLMM	(M_ALL|M_MM)
#define M_BPDI	(M_SP|M_DP|M_DI)
#define M_AI	(M_BY|M_IN|M_DI|M_LI|M_UBY|M_UIN|M_UDI|M_ULI)
#define M_BPMM	(M_BP|M_MM)
#define NULLARG	0

/* some flags defining what types of scalars are used */

typedef enum {
	SCAL_OP,	/* single scalar argument */
	SCAL_2OP,	/* 2 scalar arguments */
	SCAL_RET,	/* returns 1 scalar */
	SCAL_2RET	/* returns 2 scalars */
} Scalar_Arg_Type;

#define M_SCAL_OP	(1<<SCAL_OP)
#define M_SCAL_2OP	(1<<SCAL_2OP)
#define M_SCAL_RET	(1<<SCAL_RET)
#define M_SCAL_2RET	(1<<SCAL_2RET)

typedef struct c_fft_args {
	void *		src_addr;
	void *		dst_addr;
	incr_t		src_inc;
	incr_t		dst_inc;
	dimension_t	len;
	int		isi;
} FFT_Args;

/* end of vecgen.h */

typedef enum {
	FVSET,				/* 0 */
	FVMOV,				/* 1 */
	FVADD,				/* 2 */
	FVSUB,				/* 3 */
	FVMUL,				/* 4 */
	FVDIV,				/* 5 */
	FVNEG,				/* 6 */
	FVSQR,				/* 7 */
	FVRAMP1D,			/* 8 */
	FVRAMP2D,			/* 9 */

	FVSADD,				/* 10 */
	FVSSUB,				/* 11 */
	FVSMUL,				/* 12 */
	FVSDIV,				/* 13 */
	FVSDIV2,			/* 14 */
	FVABS,				/* 15 */
	FVSIGN,				/* 16 */

	FVSQRT,				/* 17 */
	FVSIN,				/* 18 */
	FVCOS,				/* 19 */
	FVTAN,				/* 20 */
	FVATAN,				/* 21 */
	FVATAN2,			/* 22 */
	FVSATAN2,			/* 23 */
	FVSATAN22,			/* 24 */
	FVLOG,				/* 25 */
	FVLOG10,			/* 26 */
	FVEXP,				/* 27 */
	FVERF,				/* 28 */
	FVPOW,				/* 29 */
	FVSPOW,				/* 30 */
	FVSPOW2,			/* 31 */

	FVMIN,				/* 32 */
	FVMAX,				/* 33 */
	FVMINM,				/* 34 */
	FVMAXM,				/* 35 */
	FVSMIN,				/* 36 */
	FVSMAX,				/* 37 */
	FVSMNM,				/* 38 */
	FVSMXM,				/* 39 */

	FVMINV,				/* 40 */
	FVMAXV,				/* 41 */
	FVMNMV,				/* 42 */
	FVMXMV,				/* 43 */
	FVMINI,				/* 44 */
	FVMAXI,				/* 45 */
	FVMNMI,				/* 46 */
	FVMXMI,				/* 47 */
	FVMING,				/* 48 */
	FVMAXG,				/* 49 */
	FVMNMG,				/* 50 */
	FVMXMG,				/* 51 */

	FVFLOOR,			/* 52 */
	FVROUND,			/* 53 */
	FVCEIL,				/* 54 */
	FVRINT,				/* 55 */
	FVJ0,				/* 56 */
	FVJ1,				/* 57 */
	FVACOS,				/* 58 */
	FVASIN,				/* 59 */
	FVATN2,				/* 60 */
	FVUNI,				/* 61 */

	/* bitwise operators */
	FVAND,				/* 62 */
	FVNAND,				/* 63 */
	FVOR,				/* 64 */
	FVXOR,				/* 65 */
	FVNOT,				/* 66 */
	FVCOMP,				/* 67 */
	FVMOD,
	FVSMOD,
	FVSMOD2,

	FVSAND, /* morebitwise operators */				/* 69 */
	FVSOR,
	FVSXOR,
	FVSHR,
	FVSSHR,
	FVSSHR2,
	FVSHL,
	FVSSHL,
	FVSSHL2,


	FVSUM,
	FVDOT,
	FVRAND,

	FVSMLT,
	FVSMGT,
	FVSMLE,
	FVSMGE,
	FVSMNE,
	FVSMEQ,
	FVVMLT,
	FVVMGT,
	FVVMLE,
	FVVMGE,
	FVVMNE,
	FVVMEQ,

	FVVVSLCT,
	FVVSSLCT,
	FVSSSLCT,

	FVMGSQ,
	FVCMUL, /* complex stuff */
	FVSCML,
	FVCONJ,
	FVFFT,
	FVIFT,

	FVBND, /* comparison operators */
	FVIBND,
	FVCLIP,
	FVICLP,
	FVCMP,
	FVSCMP,
	FVSCMP2,

	/* New conditional ops
	 *
	 * assign to any type -
	 * select from vector or scalar
	 * compare vectors, scalars, or v/s
	 *
	 * First group after the F indicates the sources for
	 * the assignment; second group indicates the sources
	 * for the test.  We don't bother implementing SS tests,
	 * as that can easily be done in the script.
	 */

	FVV_VV_LT,
	FVV_VV_GT,
	FVV_VV_LE,
	FVV_VV_GE,
	FVV_VV_EQ,
	FVV_VV_NE,

	FVV_VS_LT,
	FVV_VS_GT,
	FVV_VS_LE,
	FVV_VS_GE,
	FVV_VS_EQ,
	FVV_VS_NE,

	FVS_VV_LT,
	FVS_VV_GT,
	FVS_VV_LE,
	FVS_VV_GE,
	FVS_VV_EQ,
	FVS_VV_NE,

	FVS_VS_LT,
	FVS_VS_GT,
	FVS_VS_LE,
	FVS_VS_GE,
	FVS_VS_EQ,
	FVS_VS_NE,

	FSS_VV_LT,
	FSS_VV_GT,
	FSS_VV_LE,
	FSS_VV_GE,
	FSS_VV_EQ,
	FSS_VV_NE,

	FSS_VS_LT,
	FSS_VS_GT,
	FSS_VS_LE,
	FSS_VS_GE,
	FSS_VS_EQ,
	FSS_VS_NE,


#define FIRST_TYPE_CONVERSION_CODE	FVB2I
	/* Type conversions */
	FVB2I,
	FVB2L,
	FVB2LL,
	FVB2SP,
	FVB2DP,
	FVB2UB,
	FVB2UI,
	FVB2UL,
	FVB2ULL,

	FVI2B,
	FVI2L,
	FVI2LL,
	FVI2SP,
	FVI2DP,
	FVI2UB,
	FVI2UI,
	FVI2UL,
	FVI2ULL,

	FVL2B,
	FVL2I,
	FVL2LL,
	FVL2SP,
	FVL2DP,
	FVL2UB,
	FVL2UI,
	FVL2UL,
	FVL2ULL,

	FVSP2B, /* misc */ /* conversions */	/* 90 */
	FVSP2I,
	FVSP2L,
	FVSP2LL,
	FVSPDP,
	FVSP2UB,
	FVSP2UI,
	FVSP2UL,
	FVSP2ULL,

	FVDP2B,
	FVDP2I,
	FVDP2L,
	FVDP2LL,
	FVDPSP,
	FVDP2UB,
	FVDP2UI,
	FVDP2UL,
	FVDP2ULL,

	FVUB2B,
	FVUB2I,
	FVUB2L,
	FVUB2LL,
	FVUB2SP,
	FVUB2DP,
	FVUB2UI,
	FVUB2UL,
	FVUB2ULL,


	FVLL2B,
	FVLL2I,
	FVLL2L,
	FVLL2UB,
	FVLL2UI,
	FVLL2UL,
	FVLL2ULL,
	FVLL2SP,
	FVLL2DP,

	FVUI2B,
	FVUI2I,
	FVUI2L,
	FVUI2LL,
	FVUI2SP,
	FVUI2DP,
	FVUI2UB,
	FVUI2UL,
	FVUI2ULL,

	FVUL2B,
	FVUL2I,
	FVUL2L,
	FVUL2SP,
	FVUL2DP,
	FVUL2UB,
	FVUL2UI,
	FVUL2LL,
	FVUL2ULL,

	FVULL2B,
	FVULL2I,
	FVULL2L,
	FVULL2LL,
	FVULL2SP,
	FVULL2DP,
	FVULL2UB,
	FVULL2UI,
	FVULL2UL,
#define LAST_TYPE_CONVERSION_CODE	FVULL2UL

	/* Don't put any new codes below this line, reserved for warrior obsolete */
#ifdef SKY_WARRIOR_OBSOLETE
	/* unimplemented sky warrior funcs? */



	FVPOLY, /* warrior only codes! */

	FVCMPM,
	FVMSCM,
	FVMSCP,
	FVCDOT,
	FVSCMM,

	FVSWAP,
	FVD2SP,
	FVF2SP,
	FVDFSP,
	FVSP2F,
	FVSPDF,
	FVSP2D,
	FVCONV,
	FVSFFT,
	FVSIFT,
	FVXPWY,
	FVMCMM,
	FVMCMP,
	FVEXPE,
	FVLOGD,
	FVLOGE,
	FVFRAC,
	FVINT,
	FVSPIV,
	FVCPIV,
	FVSLCT,
	FVPIV,
	FVWAIT,
	FVDONE,
	FVADDR,
	FVOPCNT,
	FVCONT,
	FVINCR,
	FVFIDLE,
	FVSKYID,
	FVRDPTR,
	FVFSTAT,
	FVCRDP,
	FVCWRP,
#endif /* SKY_WARRIOR_OBSOLETE */

	N_VEC_FUNCS,		/* must be next-to-last! */
	INVALID_VFC		/* must be last! */

	/* FVSCMM, */ /* FVMSCM, FVMSCP, */
	/* FVCMPM, */


} Vec_Func_Code;


#define VV_ALLPREC_FUNC_CASES						\
									\
	case FVMIN:							\
	case FVMAX:							\
	case FVADD:							\
	case FVSUB:							\
	case FVMUL:							\
	case FVCMUL:							\
	case FVDIV:

#define VS_ALLPREC_FUNC_CASES						\
									\
	case FVSMIN:							\
	case FVSMAX:							\
	case FVSADD:							\
	case FVSSUB:							\
	case FVSMUL:							\
	case FVSDIV:							\
	case FVSDIV2:

#define VS_INTONLY_FUNC_CASES						\
									\
	case FVSSHR:							\
	case FVSSHR2:							\
	case FVSSHL:							\
	case FVSSHL2:							\
	case FVSMOD:							\
	case FVSMOD2:							\
	case FVSOR:							\
	case FVSAND:							\
	case FVSXOR:


#define IS_CONVERSION( vfp )						\
	( CODE_BETWEEN( vfp, FIRST_TYPE_CONVERSION_CODE,		\
				LAST_TYPE_CONVERSION_CODE ) )

#define CODE_BETWEEN( vfp, code1, code2 )				\
	( (vfp)->vf_code >= ( code1 ) && (vfp)->vf_code <= ( code2 ) )


/* globals */
//extern int for_real;		/* in nvf.h */

// BUG not thread-safe
extern int is_chaining;

#ifdef FOOBAR
extern int this_functype;		// not needed now?
#endif /* FOOBAR */

#ifdef DEBUG
extern debug_flag_t veclib_debug;
#endif /* DEBUG */

/* prototypes */

extern void setvarg1(Vec_Obj_Args *oap,Data_Obj *);
extern void setvarg2(Vec_Obj_Args *oap,Data_Obj *,Data_Obj *);
extern void setvarg3(Vec_Obj_Args *oap,Data_Obj *,Data_Obj *,Data_Obj *);
extern void setvarg4(Vec_Obj_Args *oap,Data_Obj *,Data_Obj *,Data_Obj *,Data_Obj *);
extern void setvarg5(Vec_Obj_Args *oap,Data_Obj *,Data_Obj *,Data_Obj *,Data_Obj *,Data_Obj *);

/* vec_chn.c */
extern int insure_static(Vec_Obj_Args *oap);
extern void add_link(void (*func)(Vector_Args *), Vector_Args *vap);

#endif  /* ! _VECGEN_H_ */

