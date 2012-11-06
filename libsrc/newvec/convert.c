#include "quip_config.h"

char VersionId_newvec_convert[] = QUIP_VERSION_STRING;

#include "nvf.h"
#include "debug.h"		/* verbose */
#define std_type float
#define dest_type float
#include "conv_prot.h"
#include "new_ops.h"

#define PREC_SWITCH( bcode , scode, lcode, llcode, fcode, dcode, ubcode, uscode, ulcode, ullcode )	\
	switch( MACHINE_PREC(dp_to) ){							\
		case PREC_BY:  code = bcode; break;					\
		case PREC_IN:  code = scode; break;					\
		case PREC_DI:  code = lcode; break;					\
		case PREC_LI:  code = llcode; break;					\
		case PREC_SP:  code = fcode; break;					\
		case PREC_DP:  code = dcode; break;					\
		case PREC_UBY:  code = ubcode; break;					\
		case PREC_UIN:  code = uscode; break;					\
		case PREC_UDI:  code = ulcode; break;					\
		case PREC_ULI:  code = ullcode; break;					\
		default:  WARN("convert:  PREC_SWITCH:  bad prec"); return;		\
	}

void convert(QSP_ARG_DECL  Data_Obj *dp_to,Data_Obj *dp_fr)
{
	Vec_Func_Code code;
	Vec_Obj_Args oargs;
	mach_prec mp;

	/* sizes are checked in call_wfunc() */

	mp = MACHINE_PREC(dp_fr);
	switch(mp){
		case PREC_BY:  PREC_SWITCH( FVMOV,   FVB2I,   FVB2L,   FVB2LL,   FVB2SP,   FVB2DP,   FVB2UB,   FVB2UI,   FVB2UL,   FVB2ULL  ) break;
		case PREC_IN:  PREC_SWITCH( FVI2B,   FVMOV,   FVI2L,   FVI2LL,   FVI2SP,   FVI2DP,   FVI2UB,   FVI2UI,   FVI2UL,   FVI2ULL  ) break;
		case PREC_DI:  PREC_SWITCH( FVL2B,   FVL2I,   FVMOV,   FVL2LL,   FVL2SP,   FVL2DP,   FVL2UB,   FVL2UI,   FVL2UL,   FVL2ULL  ) break;
		case PREC_LI:  PREC_SWITCH( FVLL2B,  FVLL2I,  FVLL2L,  FVMOV,    FVLL2SP,  FVLL2DP,  FVLL2UB,  FVLL2UI,  FVLL2UL,  FVLL2ULL ) break;
		case PREC_SP:  PREC_SWITCH( FVSP2B,  FVSP2I,  FVSP2L,  FVSP2LL,  FVMOV,    FVSPDP,   FVSP2UB,  FVSP2UI,  FVSP2UL,  FVSP2ULL ) break;
		case PREC_DP:  PREC_SWITCH( FVDP2B,  FVDP2I,  FVDP2L,  FVDP2LL,  FVDPSP,   FVMOV,    FVDP2UB,  FVDP2UI,  FVDP2UL,  FVDP2ULL ) break;
		case PREC_UBY: PREC_SWITCH( FVUB2B,  FVUB2I,  FVUB2L,  FVUB2LL,  FVUB2SP,  FVUB2DP,  FVMOV,    FVUB2UI,  FVUB2UL,  FVUB2ULL ) break;
		case PREC_UIN: PREC_SWITCH( FVUI2B,  FVUI2I,  FVUI2L,  FVUI2LL,  FVUI2SP,  FVUI2DP,  FVUI2UB,  FVMOV,    FVUI2UL,  FVUI2ULL ) break;
		case PREC_UDI: PREC_SWITCH( FVUL2B,  FVUL2I,  FVUL2L,  FVUL2LL,  FVUL2SP,  FVUL2DP,  FVUL2UB,  FVUL2UI,  FVMOV,    FVUL2ULL ) break;
		case PREC_ULI: PREC_SWITCH( FVULL2B, FVULL2I, FVULL2L, FVULL2LL, FVULL2SP, FVULL2DP, FVULL2UB, FVULL2UI, FVULL2UL, FVMOV    ) break;

#ifdef CAUTIOUS
		case PREC_NONE:
		case N_MACHINE_PRECS:
		default:
			ERROR1("CAUTIOUS:  bad case in convert()");
			return;	/* NOTREACHED */
			break;
#endif /* CAUTIOUS */
	}
	/* What if we are converting to a bitmap?
	 * The dispatch switch seems to be based on the source type...
	 */

	setvarg2(&oargs,dp_to,dp_fr);
	perf_vfunc(QSP_ARG  code,&oargs);
} /* end convert() */

ALL_UNSIGNED_CONVERSIONS( by, char )
ALL_FLOAT_CONVERSIONS( by, char )
REAL_CONVERSION( by, char, in,  short   )			\
REAL_CONVERSION( by, char, di,  int32_t    )			\
REAL_CONVERSION( by, char, li,  int64_t    )

ALL_UNSIGNED_CONVERSIONS( in, short )
ALL_FLOAT_CONVERSIONS( in, short )
REAL_CONVERSION( in, short, by,  char   )			\
REAL_CONVERSION( in, short, di,  int32_t    )			\
REAL_CONVERSION( in, short, li,  int64_t    )

ALL_UNSIGNED_CONVERSIONS( di, int32_t )
ALL_FLOAT_CONVERSIONS( di, int32_t )
REAL_CONVERSION( di, int32_t, in,  short   )			\
REAL_CONVERSION( di, int32_t, by,  char    )			\
REAL_CONVERSION( di, int32_t, li,  int64_t    )

ALL_UNSIGNED_CONVERSIONS( li, int64_t )
ALL_FLOAT_CONVERSIONS( li, int64_t )
REAL_CONVERSION( li, int64_t, in,  short   )			\
REAL_CONVERSION( li, int64_t, di,  int32_t    )			\
REAL_CONVERSION( li, int64_t, by,  char    )

ALL_SIGNED_CONVERSIONS( uby, u_char )
ALL_FLOAT_CONVERSIONS( uby, u_char )
REAL_CONVERSION( uby, u_char, uin,  u_short   )			\
REAL_CONVERSION( uby, u_char, udi,  uint32_t    )			\
REAL_CONVERSION( uby, u_char, uli,  uint64_t    )

ALL_SIGNED_CONVERSIONS( uin, u_short )
ALL_FLOAT_CONVERSIONS( uin, u_short )
REAL_CONVERSION( uin, u_short, uby,  u_char   )			\
REAL_CONVERSION( uin, u_short, udi,  uint32_t    )			\
REAL_CONVERSION( uin, u_short, uli,  uint64_t    )

ALL_SIGNED_CONVERSIONS( udi, uint32_t )
ALL_FLOAT_CONVERSIONS( udi, uint32_t )
REAL_CONVERSION( udi, uint32_t, uin,  u_short   )			\
REAL_CONVERSION( udi, uint32_t, uby,  u_char    )			\
REAL_CONVERSION( udi, uint32_t, uli,  uint64_t    )

ALL_SIGNED_CONVERSIONS( uli, uint64_t )
ALL_FLOAT_CONVERSIONS( uli, uint64_t )
REAL_CONVERSION( uli, uint64_t, uin,  u_short   )			\
REAL_CONVERSION( uli, uint64_t, udi,  uint32_t    )			\
REAL_CONVERSION( uli, uint64_t, uby,  u_char    )

ALL_UNSIGNED_CONVERSIONS( sp, float )
ALL_SIGNED_CONVERSIONS( sp, float )
REAL_CONVERSION( sp, float, dp, double   )			\

ALL_UNSIGNED_CONVERSIONS( dp, double )
ALL_SIGNED_CONVERSIONS( dp, double )
REAL_CONVERSION( dp, double, sp, float   )			\

#define _CPX_CONVERSION(name,type1,type2)								\
void name(Vec_Args *argp){LDECLS2;type1 *v1;type2 *v2;SETUP2;V2LOOP(v2a.re=v1a.re;v2a.im=v1a.im;) }

#define CPX_CONVERSION( key1, type1, key2, type2 )							\
_CPX_CONVERSION( cv##key1##2##key2, type1, type2 )

#define ALL_CPX_CONVERSIONS( key, type )			\
CPX_CONVERSION( key, type, b,  By_Cpx )				\
CPX_CONVERSION( key, type, i,  In_Cpx )				\
CPX_CONVERSION( key, type, l,  Di_Cpx )				\
CPX_CONVERSION( key, type, sp, Sp_Cpx )				\
CPX_CONVERSION( key, type, dp, Dp_Cpx )				\
CPX_CONVERSION( key, type, ub, UBy_Cpx )			\
CPX_CONVERSION( key, type, ui, UIn_Cpx )			\
CPX_CONVERSION( key, type, ul, UDi_Cpx )

#ifdef FOOBAR
ALL_CPX_CONVERSIONS( b, By_Cpx )
ALL_CPX_CONVERSIONS( i, In_Cpx )
ALL_CPX_CONVERSIONS( l, Di_Cpx )
ALL_CPX_CONVERSIONS( sp, Sp_Cpx )
ALL_CPX_CONVERSIONS( dp, Dp_Cpx )
ALL_CPX_CONVERSIONS( ub, UBy_Cpx )
ALL_CPX_CONVERSIONS( ui, UIn_Cpx )
ALL_CPX_CONVERSIONS( ul, UDi_Cpx )
#endif

