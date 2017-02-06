
/* defns shared by veclib & warlib */

#ifndef _VECGEN_H_
#define _VECGEN_H_

#ifdef __cplusplus
extern "C" {
#endif


#include <stdio.h>
#include "typedefs.h"
//#include "data_obj.h"
// Does this need to be a separate file?
#include "veclib/obj_args.h"

//typedef uint32_t	index_type;
typedef uint32_t	count_type;

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

extern const char *name_for_argsprec(argset_prec i);
extern const char *name_for_argtype(argset_type i);

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

/* what are the Data_Vector flags?? */
#define DV_BITMAP			1

#define DATA_VEC_IS_BITMAP(dvp)		((dvp)->dv_flags & DV_BITMAP)

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
	FVMOD,				/* 68 */
	FVSMOD,				/* 69 */
	FVSMOD2,			/* 70 */

	/* morebitwise operators */
	FVSAND,				/* 71 */
	FVSOR,				/* 72 */
	FVSXOR,				/* 73 */
	FVSHR,				/* 74 */
	FVSSHR,				/* 75 */
	FVSSHR2,			/* 76 */
	FVSHL,				/* 77 */
	FVSSHL,				/* 78 */
	FVSSHL2,			/* 79 */

	FVSUM,				/* 80 */
	/* to implement on GPU, better to write vdot as composition of VMUL and VSUM */
	FVDOT,				/* 81 */
#define HAVE_FVDOT
	FVRAND,				/* 82 */

	FVSMLT,				/* 83 */
	FVSMGT,				/* 84 */
	FVSMLE,				/* 85 */
	FVSMGE,				/* 86 */
	FVSMNE,				/* 87 */
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
//#ifdef NOT_YET
	FVFFT,
	FVIFT,
//#else // !NOT_YET
//#define FVFFT	(-3)
//#define FVIFT	(-5)
//#endif // NOT_YET

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

	FVISNAN,
	FVISINF,
	FVISNORM,

	FVTOLOWER,
	FVTOUPPER,

	FVISLOWER,
	FVISUPPER,
	FVISALNUM,
	FVISALPHA,
	FVISDIGIT,
	FVISSPACE,
	FVISBLANK,
	FVISCNTRL,

// OLD:  one code for each conversion (pair of distinct precisions)
// eg. vby2in
//
// NEW:  one code for each destination precision, and typed subfunctions
// for each possible source precision.  This is more in line with the
// approach taken for the other functions...

#define FIRST_NEW_CONVERSION_CODE	FVCONV2BY
	FVCONV2BY,
	FVCONV2IN,
	FVCONV2DI,
	FVCONV2LI,
	FVCONV2UBY,
	FVCONV2UIN,
	FVCONV2UDI,
	FVCONV2ULI,
	FVCONV2SP,
	FVCONV2DP,
#define LAST_NEW_CONVERSION_CODE	FVCONV2DP

#define IS_CONVERSION( vfp )	IS_NEW_CONVERSION(vfp)

	/* Don't put any new codes below this line, reserved for warrior obsolete */

	FVTRUNC,			/* would add after FVFLOOR, but that would mess up numbers in comments */
	FVERFINV,			/* ditto... */

	FVGAMMA,		// new funcs from libgsl...
	FVLNGAMMA,

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


#define IS_NEW_CONVERSION( vfp )					\
	( CODE_BETWEEN( vfp, FIRST_NEW_CONVERSION_CODE,		\
				LAST_NEW_CONVERSION_CODE ) )

#define CODE_BETWEEN( vfp, code1, code2 )				\
	( VF_CODE(vfp) >= ( code1 ) && VF_CODE(vfp) <= ( code2 ) )


/* globals */
//extern int for_real;		/* in nvf.h */

// BUG not thread-safe
extern int is_chaining;

#ifdef QUIP_DEBUG
extern debug_flag_t veclib_debug;
#endif /* QUIP_DEBUG */

struct vector_function;

//#define VFPTR_ARG		vfp,
//#define VFPTR_ARG_DECL		Vector_Function *vfp,
#define VFCODE_ARG		vf_code,
#define VFCODE_ARG_DECL		const int vf_code,
#define HOST_CALL_ARGS		VFCODE_ARG  oap
#define HOST_CALL_ARG_DECLS	VFCODE_ARG_DECL  /*const*/ Vec_Obj_Args *oap

// Why are these called link funcs?  Maybe because they can be chained?
// Kind of a legacy from the old skywarrior library code...
// A vector arg used to just have a length and a stride, but now
// with gpu's we have three-dimensional lengths.  But in principle
// there's no reason why we couldn't have full shapes passed...
#define LINK_FUNC_ARGS		VFCODE_ARG  vap
#define LINK_FUNC_ARG_DECLS	VFCODE_ARG_DECL  const Vector_Args *vap


typedef struct vec_func_array {
	Vec_Func_Code	vfa_code;
	void		(*vfa_func[N_FUNCTION_TYPES])
				(HOST_CALL_ARG_DECLS);
} Vec_Func_Array;

extern Vec_Func_Array vl2_vfa_tbl[];	// BUG put in platform-specific file

extern void check_vfa_tbl(QSP_ARG_DECL  Vec_Func_Array vfa_tbl[], int size );

// BUG - these should be part of the platform code?
extern void check_vl2_vfa_tbl(SINGLE_QSP_ARG_DECL);
#ifdef HAVE_OPENCL
extern void check_ocl_vfa_tbl(SINGLE_QSP_ARG_DECL);
#endif // HAVE_OPENCL


#ifdef __cplusplus
}
#endif

#endif  /* ! _VECGEN_H_ */

