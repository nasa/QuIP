
#include "quip_config.h"

char VersionId_newvec_vectbl[] = QUIP_VERSION_STRING;

/* this table defines all of the functions.
 * It gives their name, a mask that tells the supported precisions,
 * and an entry point...
 */

#include <stdlib.h>	/* qsort */
#include "nvf.h"
#include "items.h"
#include "version.h"
#include "rn.h"		/* set_random_seed */
#include "debug.h"

//#include "convert.h"

debug_flag_t veclib_debug=0;



/* local prototypes */

ITEM_INTERFACE_DECLARATIONS(Vec_Func,vf)

Vec_Func vec_func_tbl[N_VEC_FUNCS]={
{ "vset",	FVSET,		S_UNARY,	vset,	M_ALL,	RCQ	},
{ "vmov",	FVMOV,		V_UNARY,	vmov,	M_ALL,	RCQ	},
{ "vadd",	FVADD,		VV_BINARY,	vadd,	M_ALLMM,RCMQP	},
{ "vsub",	FVSUB,		VV_BINARY,	vsub,	M_ALL,	RCMQP	},
{ "vmul",	FVMUL,		VV_BINARY,	vmul,	M_ALL,	RCMQP	},
{ "vdiv",	FVDIV,		VV_BINARY,	vdiv,	M_ALL,	RCMQP	},
{ "vneg",	FVNEG,		V_UNARY,	vneg,	M_ALL,	RCQ	},
{ "vsqr",	FVSQR,		V_UNARY,	vsqr,	M_ALL,	RCQ	},
{ "vramp1d",	FVRAMP1D,	SS_RAMP,	vramp1d,M_ALL,	R	},
{ "vramp2d",	FVRAMP2D,	SSS_RAMP,	vramp2d,M_ALL,	R	},
{ "vsadd",	FVSADD,		VS_BINARY,	vsadd,	M_ALL,	RCMQP	},
{ "vssub",	FVSSUB,		VS_BINARY,	vssub,	M_ALL,	RCMQP	},
{ "vsmul",	FVSMUL,		VS_BINARY,	vsmul,	M_ALL,	RCMQP	},
{ "vsdiv",	FVSDIV,		VS_BINARY,	vsdiv,	M_ALL,	RCMQP	},
{ "vsdiv2",	FVSDIV2,	VS_BINARY,	vsdiv2,	M_ALL,	RCMQP	},
{ "vabs",	FVABS,		V_UNARY,	vabs,	M_ALL,	R	},
{ "vsign",	FVSIGN,		V_UNARY,	vsign,	M_ALL,	R	},

{ "vsqrt",	FVSQRT,		V_UNARY,	vsqrt,	M_BP,	R	},
{ "vsin",	FVSIN,		V_UNARY,	vsin,	M_BP,	R	},
{ "vcos",	FVCOS,		V_UNARY,	vcos,	M_BP,	R	},
{ "vatan",	FVATAN,		V_UNARY,	vatan,	M_BP,	R	},
{ "vtan",	FVTAN,		V_UNARY,	vtan,	M_BP,	R	},
{ "vatan2",	FVATAN2,	VV_BINARY,	vatan2,	M_BP,	R	},
{ "vsatan2",	FVSATAN2,	VS_BINARY,	vsatan2,	M_BP,	R	},
{ "vsatan22",	FVSATAN22,	VS_BINARY,	vsatan22,	M_BP,	R	},
{ "vlog",	FVLOG,		V_UNARY,	vlog,	M_BP,	R	},
{ "vlog10",	FVLOG10,	V_UNARY,	vlog10,	M_BP,	R	},
{ "vexp",	FVEXP,		V_UNARY,	vexp,	M_BP,	R	},
{ "verf",	FVERF,		V_UNARY,	verf,	M_BP,	R	},
{ "vpow",	FVPOW,		VV_BINARY,	vpow,	M_BP,	RC	},
{ "vspow",	FVSPOW,		VS_BINARY,	vspow,	M_BP,	R	},
{ "vspow2",	FVSPOW2,	VS_BINARY,	vspow2,	M_BP,	R	},

{ "vmin",	FVMIN,		VV_BINARY,	vmin,	M_ALL,	R	},
{ "vmax",	FVMAX,		VV_BINARY,	vmax,	M_ALL,	R	},
{ "vminm",	FVMINM,		VV_BINARY,	vminm,	M_ALL,	R	},
{ "vmaxm",	FVMAXM,		VV_BINARY,	vmaxm,	M_ALL,	R	},
{ "vsmin",	FVSMIN,		VS_BINARY,	vsmin,	M_ALL,	R	},
{ "vsmax",	FVSMAX,		VS_BINARY,	vsmax,	M_ALL,	R	},
{ "vsmnm",	FVSMNM,		VS_BINARY,	vsmnm,	M_ALL,	R	},
{ "vsmxm",	FVSMXM,		VS_BINARY,	vsmxm,	M_ALL,	R	},

{ "vminv",	FVMINV,		V_PROJECTION,	vminv,	M_ALL,	R	},
{ "vmaxv",	FVMAXV,		V_PROJECTION,	vmaxv,	M_ALL,	R	},
{ "vmnmv",	FVMNMV,		V_PROJECTION,	vmnmv,	M_ALL,	R	},
{ "vmxmv",	FVMXMV,		V_PROJECTION,	vmxmv,	M_ALL,	R	},
{ "vmini",	FVMINI,		V_INT_PROJECTION,	vmini,	M_ALL,	R	},
{ "vmaxi",	FVMAXI,		V_INT_PROJECTION,	vmaxi,	M_ALL,	R	},
{ "vmnmi",	FVMNMI,		V_INT_PROJECTION,	vmnmi,	M_ALL,	R	},
{ "vmxmi",	FVMXMI,		V_INT_PROJECTION,	vmxmi,	M_ALL,	R	},
{ "vming",	FVMING,		V_SCALRET2,	vming,	M_ALL,	R	},
{ "vmaxg",	FVMAXG,		V_SCALRET2,	vmaxg,	M_ALL,	R	},
{ "vmnmg",	FVMNMG,		V_SCALRET2,	vmnmg,	M_ALL,	R	},
{ "vmxmg",	FVMXMG,		V_SCALRET2,	vmxmg,	M_ALL,	R	},

{ "vfloor",	FVFLOOR,	V_UNARY,	vfloor,	M_BP,	R	},
{ "vround",	FVROUND,	V_UNARY,	vround,	M_BP,	R	},
{ "vceil",	FVCEIL,		V_UNARY,	vceil,	M_BP,	R	},
{ "vrint",	FVRINT,		V_UNARY,	vrint,	M_BP,	R	},
{ "vj0",	FVJ0,		V_UNARY,	vj0,	M_BP,	R	},
{ "vj1",	FVJ1,		V_UNARY,	vj1,	M_BP,	R	},
{ "vacos",	FVACOS,		V_UNARY,	vacos,	M_BP,	R	},
{ "vasin",	FVASIN,		V_UNARY,	vasin,	M_BP,	R	},
{ "vatn2",	FVATN2,		V_UNARY_C,	vatn2,	M_BP,	RC	},
{ "vuni",	FVUNI,		V_NO_ARGS,	vuni,	M_BP,	R	},

{ "vand",	FVAND,		VV_BINARY,	vand,	M_AI,	R	},
{ "vnand",	FVNAND,		VV_BINARY,	vnand,	M_AI,	R	},
{ "vor",	FVOR,		VV_BINARY,	vor,	M_AI,	R	},
{ "vxor",	FVXOR,		VV_BINARY,	vxor,	M_AI,	R	},
{ "vnot",	FVNOT,		V_UNARY,	vnot,	M_AI,	R	},
{ "vmod",	FVMOD,		VV_BINARY,	vmod,	M_AI,	R	},
{ "vsmod",	FVSMOD,		VS_BINARY,	vsmod,	M_AI,	R	},
{ "vsmod2",	FVSMOD2,	VS_BINARY,	vsmod2,	M_AI,	R	},

{ "vsand",	FVSAND,		VS_BINARY,	vsand,	M_AI,	R	},
{ "vsor",	FVSOR,		VS_BINARY,	vsor,	M_AI,	R	},
{ "vsxor",	FVSXOR,		VS_BINARY,	vsxor,	M_AI,	R	},
{ "vshr",	FVSHR,		VV_BINARY,	vshr,	M_AI,	R	},
{ "vsshr",	FVSSHR,		VS_BINARY,	vsshr,	M_AI,	R	},
{ "vsshr2",	FVSSHR2,	VS_BINARY,	vsshr2,	M_AI,	R	},
{ "vshl",	FVSHL,		VV_BINARY,	vshl,	M_AI,	R	},
{ "vsshl",	FVSSHL,		VS_BINARY,	vsshl,	M_AI,	R	},
{ "vsshl2",	FVSSHL2,	VS_BINARY,	vsshl2,	M_AI,	R	},

{ "vby2in",	FVB2I,		V_UNARY,	vby2in,	0,	R	},
{ "vby2di",	FVB2L,		V_UNARY,	vby2di,	0,	R	},
{ "vby2sp",	FVB2SP,		V_UNARY,	vby2sp,	0,	R	},
{ "vby2dp",	FVB2DP,		V_UNARY,	vby2dp,	0,	R	},
{ "vin2by",	FVI2B,		V_UNARY,	vin2by,	0,	R	},
{ "vin2di",	FVI2L,		V_UNARY,	vin2di,	0,	R	},
{ "vin2sp",	FVI2SP,		V_UNARY,	vin2sp,	0,	R	},
{ "vin2dp",	FVI2DP,		V_UNARY,	vin2dp,	0,	R	},
{ "vdi2by",	FVL2B,		V_UNARY,	vdi2by,	0,	R	},
{ "vdi2in",	FVL2I,		V_UNARY,	vdi2in,	0,	R	},
{ "vdi2sp",	FVL2SP,		V_UNARY,	vdi2sp,	0,	R	},
{ "vdi2dp",	FVL2DP,		V_UNARY,	vdi2dp,	0,	R	},
{ "vsp2by",	FVSP2B,		V_UNARY,	vsp2by,	0,	R	},
{ "vsp2in",	FVSP2I,		V_UNARY,	vsp2in,	0,	R	},
{ "vsp2di",	FVSP2L,		V_UNARY,	vsp2di,	0,	R	},
{ "vsp2dp",	FVSPDP,		V_UNARY,	vsp2dp,	0,	RCQ	},
{ "vdp2by",	FVDP2B,		V_UNARY,	vdp2by,	0,	R	},
{ "vdp2in",	FVDP2I,		V_UNARY,	vdp2in,	0,	R	},
{ "vdp2di",	FVDP2L,		V_UNARY,	vdp2di,	0,	R	},
{ "vdp2sp",	FVDPSP,		V_UNARY,	vdp2sp,	0,	RCQ	},

/* unsigned conversions */
{ "vuby2by",	FVUB2B,		V_UNARY,	vuby2by, 0,	R	},
{ "vuby2in",	FVUB2I,		V_UNARY,	vuby2in, 0,	R	},
{ "vuby2di",	FVUB2L,		V_UNARY,	vuby2di, 0,	R	},
{ "vuby2sp",	FVUB2SP,	V_UNARY,	vuby2sp, 0,	R	},
{ "vuby2dp",	FVUB2DP,	V_UNARY,	vuby2dp, 0,	R	},
{ "vuby2uin",	FVUB2UI,	V_UNARY,	vuby2uin, 0,	R	},
{ "vuby2udi",	FVUB2UL,	V_UNARY,	vuby2udi, 0,	R	},
{ "vuin2by",	FVUI2B,		V_UNARY,	vuin2by, 0,	R	},
{ "vuin2in",	FVUI2I,		V_UNARY,	vuin2in, 0,	R	},
{ "vuin2di",	FVUI2L,		V_UNARY,	vuin2di, 0,	R	},
{ "vuin2sp",	FVUI2SP,	V_UNARY,	vuin2sp, 0,	R	},
{ "vuin2dp",	FVUI2DP,	V_UNARY,	vuin2dp, 0,	R	},
{ "vuin2uby",	FVUI2UB,	V_UNARY,	vuin2uby, 0,	R	},
{ "vuin2udi",	FVUI2UL,	V_UNARY,	vuin2udi, 0,	R	},
{ "vudi2by",	FVUL2B,		V_UNARY,	vudi2by, 0,	R	},
{ "vudi2in",	FVUL2I,		V_UNARY,	vudi2in, 0,	R	},
{ "vudi2di",	FVUL2L,		V_UNARY,	vudi2di, 0,	R	},
{ "vudi2sp",	FVUL2SP,	V_UNARY,	vudi2sp, 0,	R	},
{ "vudi2dp",	FVUL2DP,	V_UNARY,	vudi2dp, 0,	R	},
{ "vudi2uby",	FVUL2UB,	V_UNARY,	vudi2uby, 0,	R	},
{ "vudi2uin",	FVUL2UI,	V_UNARY,	vudi2uin, 0,	R	},
{ "vby2uby",	FVB2UB,		V_UNARY,	vby2uby, 0,	R	},
{ "vin2uby",	FVI2UB,		V_UNARY,	vin2uby, 0,	R	},
{ "vdi2uby",	FVL2UB,		V_UNARY,	vdi2uby, 0,	R	},
{ "vsp2uby",	FVSP2UB,	V_UNARY,	vsp2uby, 0,	R	},
{ "vdp2uby",	FVDP2UB,	V_UNARY,	vdp2uby, 0,	R	},
{ "vby2uin",	FVB2UI,		V_UNARY,	vby2uin, 0,	R	},
{ "vin2uin",	FVI2UI,		V_UNARY,	vin2uin, 0,	R	},
{ "vdi2uin",	FVL2UI,		V_UNARY,	vdi2uin, 0,	R	},
{ "vsp2uin",	FVSP2UI,	V_UNARY,	vsp2uin, 0,	R	},
{ "vdp2uin",	FVDP2UI,	V_UNARY,	vdp2uin, 0,	R	},
{ "vby2udi",	FVB2UL,		V_UNARY,	vby2udi, 0,	R	},
{ "vin2udi",	FVI2UL,		V_UNARY,	vin2udi, 0,	R	},
{ "vdi2udi",	FVL2UL,		V_UNARY,	vdi2udi, 0,	R	},
{ "vsp2udi",	FVSP2UL,	V_UNARY,	vsp2udi, 0,	R	},
{ "vdp2udi",	FVDP2UL,	V_UNARY,	vdp2udi, 0,	R	},
/* long long conversions */
{ "vby2li",	FVB2LL,		V_UNARY,	vby2li,	0,	R	},
{ "vin2li",	FVI2LL,		V_UNARY,	vin2li,	0,	R	},
{ "vdi2li",	FVL2LL,		V_UNARY,	vdi2li,	0,	R	},
{ "vsp2li",	FVSP2LL,	V_UNARY,	vsp2li,	0,	R	},
{ "vdp2li",	FVDP2LL,	V_UNARY,	vdp2li,	0,	R	},
{ "vuby2li",	FVUB2LL,	V_UNARY,	vuby2li, 0,	R	},
{ "vuin2li",	FVUI2LL,	V_UNARY,	vuin2li, 0,	R	},
{ "vudi2li",	FVUL2LL,	V_UNARY,	vudi2li, 0,	R	},
{ "vuli2li",	FVULL2LL,	V_UNARY,	vuli2li, 0,	R	},

{ "vby2uli",	FVB2ULL,	V_UNARY,	vby2uli, 0,	R	},
{ "vin2uli",	FVI2ULL,	V_UNARY,	vin2uli, 0,	R	},
{ "vdi2uli",	FVL2ULL,	V_UNARY,	vdi2uli, 0,	R	},
{ "vli2uli",	FVLL2ULL,	V_UNARY,	vli2uli, 0,	R	},
{ "vsp2uli",	FVSP2ULL,	V_UNARY,	vsp2uli, 0,	R	},
{ "vdp2uli",	FVDP2ULL,	V_UNARY,	vdp2uli, 0,	R	},
{ "vuby2uli",	FVUB2ULL,	V_UNARY,	vuby2uli, 0,	R	},
{ "vuin2uli",	FVUI2ULL,	V_UNARY,	vuin2uli, 0,	R	},
{ "vudi2uli",	FVUL2ULL,	V_UNARY,	vudi2uli, 0,	R	},

{ "vli2by",	FVLL2B,		V_UNARY,	vli2by,	0,	R	},
{ "vli2in",	FVLL2I,		V_UNARY,	vli2in,	0,	R	},
{ "vli2di",	FVLL2L,		V_UNARY,	vli2di,	0,	R	},
{ "vli2sp",	FVLL2SP,	V_UNARY,	vli2sp,	0,	R	},
{ "vli2dp",	FVLL2DP,	V_UNARY,	vli2dp,	0,	R	},
{ "vli2uby",	FVLL2UB,	V_UNARY,	vli2uby, 0,	R	},
{ "vli2uin",	FVLL2UI,	V_UNARY,	vli2uin, 0,	R	},
{ "vli2udi",	FVLL2UL,	V_UNARY,	vli2udi, 0,	R	},

{ "vuli2by",	FVULL2B,	V_UNARY,	vuli2by, 0,	R	},
{ "vuli2in",	FVULL2I,	V_UNARY,	vuli2in, 0,	R	},
{ "vuli2di",	FVULL2L,	V_UNARY,	vuli2di, 0,	R	},
{ "vuli2sp",	FVULL2SP,	V_UNARY,	vuli2sp, 0,	R	},
{ "vuli2dp",	FVULL2DP,	V_UNARY,	vuli2dp, 0,	R	},
{ "vuli2uby",	FVULL2UB,	V_UNARY,	vuli2uby, 0,	R	},
{ "vuli2uin",	FVULL2UI,	V_UNARY,	vuli2uin, 0,	R	},
{ "vuli2udi",	FVULL2UL,	V_UNARY,	vuli2udi, 0,	R	},



/* max mag,	min max changed to M_ALL to allow long destination... */
{ "vsum",	FVSUM,		V_PROJECTION,	vsum,	M_ALL,	RCQ	},
{ "vdot",	FVDOT,		V_PROJECTION2,	vdot,	M_ALL,	RC	},
{ "vrand",	FVRAND,		V_UNARY,	vrand,	M_ALL,	RC	},

{ "vsm_lt",	FVSMLT,		VS_TEST,	vsm_lt,	M_ALL,	R	},
{ "vsm_gt",	FVSMGT,		VS_TEST,	vsm_gt,	M_ALL,	R	},
{ "vsm_le",	FVSMLE,		VS_TEST,	vsm_le,	M_ALL,	R	},
{ "vsm_ge",	FVSMGE,		VS_TEST,	vsm_ge,	M_ALL,	R	},
{ "vsm_ne",	FVSMNE,		VS_TEST,	vsm_ne,	M_ALL,	R	},
{ "vsm_eq",	FVSMEQ,		VS_TEST,	vsm_eq,	M_ALL,	R	},
{ "vvm_lt",	FVVMLT,		VV_TEST,	vvm_lt,	M_ALL,	R	},
{ "vvm_gt",	FVVMGT,		VV_TEST,	vvm_gt,	M_ALL,	R	},
{ "vvm_le",	FVVMLE,		VV_TEST,	vvm_le,	M_ALL,	R	},
{ "vvm_ge",	FVVMGE,		VV_TEST,	vvm_ge,	M_ALL,	R	},
{ "vvm_ne",	FVVMNE,		VV_TEST,	vvm_ne,	M_ALL,	R	},
{ "vvm_eq",	FVVMEQ,		VV_TEST,	vvm_eq,	M_ALL,	R	},

{ "vvv_slct",	FVVVSLCT,	VV_SELECT,	vvv_slct, M_ALL, R	},
{ "vvs_slct",	FVVSSLCT,	VS_SELECT,	vvs_slct, M_ALL, R	},
{ "vss_slct",	FVSSSLCT,	SS_SELECT,	vss_slct, M_ALL, R	},

/* New conditionals */
{ "vv_vv_lt",	FVV_VV_LT,	VVVVCA,		vv_vv_lt, M_ALL, R	},
{ "vv_vv_gt",	FVV_VV_GT,	VVVVCA,		vv_vv_gt, M_ALL, R	},
{ "vv_vv_le",	FVV_VV_LE,	VVVVCA,		vv_vv_le, M_ALL, R	},
{ "vv_vv_ge",	FVV_VV_GE,	VVVVCA,		vv_vv_ge, M_ALL, R	},
{ "vv_vv_eq",	FVV_VV_EQ,	VVVVCA,		vv_vv_eq, M_ALL, R	},
{ "vv_vv_ne",	FVV_VV_NE,	VVVVCA,		vv_vv_ne, M_ALL, R	},

{ "vv_vs_lt",	FVV_VS_LT,	VVVSCA,		vv_vs_lt, M_ALL, R	},
{ "vv_vs_gt",	FVV_VS_GT,	VVVSCA,		vv_vs_gt, M_ALL, R	},
{ "vv_vs_le",	FVV_VS_LE,	VVVSCA,		vv_vs_le, M_ALL, R	},
{ "vv_vs_ge",	FVV_VS_GE,	VVVSCA,		vv_vs_ge, M_ALL, R	},
{ "vv_vs_eq",	FVV_VS_EQ,	VVVSCA,		vv_vs_eq, M_ALL, R	},
{ "vv_vs_ne",	FVV_VS_NE,	VVVSCA,		vv_vs_ne, M_ALL, R	},

{ "vs_vv_lt",	FVS_VV_LT,	VSVVCA,		vs_vv_lt, M_ALL, R	},
{ "vs_vv_gt",	FVS_VV_GT,	VSVVCA,		vs_vv_gt, M_ALL, R	},
{ "vs_vv_le",	FVS_VV_LE,	VSVVCA,		vs_vv_le, M_ALL, R	},
{ "vs_vv_ge",	FVS_VV_GE,	VSVVCA,		vs_vv_ge, M_ALL, R	},
{ "vs_vv_eq",	FVS_VV_EQ,	VSVVCA,		vs_vv_eq, M_ALL, R	},
{ "vs_vv_ne",	FVS_VV_NE,	VSVVCA,		vs_vv_ne, M_ALL, R	},

{ "vs_vs_lt",	FVS_VS_LT,	VSVSCA,		vs_vs_lt, M_ALL, R	},
{ "vs_vs_gt",	FVS_VS_GT,	VSVSCA,		vs_vs_gt, M_ALL, R	},
{ "vs_vs_le",	FVS_VS_LE,	VSVSCA,		vs_vs_le, M_ALL, R	},
{ "vs_vs_ge",	FVS_VS_GE,	VSVSCA,		vs_vs_ge, M_ALL, R	},
{ "vs_vs_eq",	FVS_VS_EQ,	VSVSCA,		vs_vs_eq, M_ALL, R	},
{ "vs_vs_ne",	FVS_VS_NE,	VSVSCA,		vs_vs_ne, M_ALL, R	},

{ "ss_vv_lt",	FSS_VV_LT,	SSVVCA,		ss_vv_lt, M_ALL, R	},
{ "ss_vv_gt",	FSS_VV_GT,	SSVVCA,		ss_vv_gt, M_ALL, R	},
{ "ss_vv_le",	FSS_VV_LE,	SSVVCA,		ss_vv_le, M_ALL, R	},
{ "ss_vv_ge",	FSS_VV_GE,	SSVVCA,		ss_vv_ge, M_ALL, R	},
{ "ss_vv_eq",	FSS_VV_EQ,	SSVVCA,		ss_vv_eq, M_ALL, R	},
{ "ss_vv_ne",	FSS_VV_NE,	SSVVCA,		ss_vv_ne, M_ALL, R	},

{ "ss_vs_lt",	FSS_VS_LT,	SSVSCA,		ss_vs_lt, M_ALL, R	},
{ "ss_vs_gt",	FSS_VS_GT,	SSVSCA,		ss_vs_gt, M_ALL, R	},
{ "ss_vs_le",	FSS_VS_LE,	SSVSCA,		ss_vs_le, M_ALL, R	},
{ "ss_vs_ge",	FSS_VS_GE,	SSVSCA,		ss_vs_ge, M_ALL, R	},
{ "ss_vs_eq",	FSS_VS_EQ,	SSVSCA,		ss_vs_eq, M_ALL, R	},
{ "ss_vs_ne",	FSS_VS_NE,	SSVSCA,		ss_vs_ne, M_ALL, R	},

{ "vmgsq",	FVMGSQ,		V_UNARY_C,	vmgsq,	M_ALL,	C	},
{ "vcmul",	FVCMUL,		VV_BINARY,	vcmul,	M_ALL,	C	},
{ "vscml",	FVSCML,		VS_BINARY,	vscml,	M_BP,	C	},
{ "vconj",	FVCONJ,		V_UNARY,	vconj,	M_ALL,	C	},
{ "vfft",	FVFFT,		V_FWD_FFT,	vfft,	M_BP,	RCM	},
{ "vift",	FVIFT,		V_INV_FFT,	vift,	M_BP,	RCM	},

{ "vbnd",	FVBND,		VV_BINARY,	vbnd,	M_ALL,	R	},
{ "vibnd",	FVIBND,		VV_BINARY,	vibnd,	M_ALL,	R	},
{ "vclip",	FVCLIP,		VS_BINARY,	vclip,	M_ALL,	R	},
{ "viclp",	FVICLP,		VS_BINARY,	viclp,	M_ALL,	R	},
{ "vcmp",	FVCMP,		VV_BINARY,	vcmp,	M_ALL,	R	},
{ "vscmp",	FVSCMP,		VS_BINARY,	vscmp,	M_ALL,	R	},
{ "vscmp2",	FVSCMP2,	VS_BINARY,	vscmp2,	M_ALL,	R	},


{ "vcomp",	FVCOMP,		V_UNARY,	vcomp,	M_AI,	R	},


#ifdef NOT_YET
{ "vcsadd",	FVCSADD,	VS_BINARY,	vsadd,	M_ALL,	CM	},
{ "vcssub",	FVCSSUB,	VS_BINARY,	vssub,	M_ALL,	CM	},
{ "vcsmul",	FVCSMUL,	VS_BINARY,	vcsmul,	M_ALL,	CM	},
{ "vcsdiv",	FVCSDIV,	VS_BINARY,	vsdiv,	M_ALL,	CM	},
{ "vcsdiv2",	FVCSDIV2,	VS_BINARY,	vsdiv2,	M_ALL,	CM	},
#endif
/*
{ "vqsadd",	FVQSADD,	VS_BINARY,	vsadd,	M_ALL,	QP	},
{ "vqssub",	FVQSSUB,	VS_BINARY,	vssub,	M_ALL,	QP	},
{ "vqsmul",	FVQSMUL,	VS_BINARY,	vqsmul,	M_ALL,	QP	},
{ "vqsdiv",	FVQSDIV,	VS_BINARY,	vsdiv,	M_ALL,	QP	},
{ "vqsdiv2",	FVQSDIV2,	VS_BINARY,	vsdiv2,	M_ALL,	QP	},
*/

#ifdef NOT_YET
{ "vd2sp",	FVD2SP,		V_UNARY,	no_func, 0,	0	},
{ "vsp2d",	FVSP2D,		V_UNARY,	no_func, 0,	0	},
#endif

#ifdef FOOBAR
{ "vsfft",	FVSFFT,		V_UNARY,	no_func, 0,	0	},
{ "vsift",	FVSIFT,		V_UNARY,	no_func, 0,	0	},
{ "vcdot",	FVCDOT,		V_PROJECTION2,	no_func,	M_BP,	C	},
{ "vf2sp",	FVF2SP,		V_UNARY,	no_func, 0,	0	},
{ "vdfsp",	FVDFSP,		V_UNARY,	no_func, 0,	0	},
{ "vsp2f",	FVSP2F,		V_UNARY,	no_func, 0,	0	},
{ "vspdf",	FVSPDF,		V_UNARY,	no_func, 0,	0	},

{ "vcmpm",	FVCMPM,		0,		no_func, 0,	0	},
{ "vmcmm",	FVMCMM,		0,		no_func, 0,	0	},
{ "vmcmp",	FVMCMP,		0,		no_func, 0,	0	},
{ "vexpe",	FVEXPE,		0,		no_func, 0,	0	},
{ "vloge",	FVLOGE,		0,		no_func, 0,	0	},
{ "vfrac",	FVFRAC,		0,		no_func, 0,	0	},
{ "vint",	FVINT,		0,		no_func, 0,	0	},
{ "vmscm",	FVMSCM,		0,		no_func, 0,	0	},
{ "vmscp",	FVMSCP,		0,		no_func, 0,	0	},
{ "vscmm",	FVSCMM,		0,		no_func, 0,	0	},
{ "vxpwy",	FVXPWY,		0,		no_func, 0,	0	},
{ "vpoly",	FVPOLY,		0,		no_func, 0,	0	},	/* warrior only stuff... */
{ "vswap",	FVSWAP,		0,		no_func, 0,	0	},
{ "vconv",	FVCONV,		N23ARG,		no_func, M_BP,	RC	},
{ "vspiv",	FVSPIV,		SST2A,		no_func, M_BP,	RC	},
{ "vcpiv",	FVCPIV,		S3ARG,		no_func, M_BP,	C	},
{ "vpiv",	FVPIV,		0,		no_func, 0,	0	},
{ "vwait",	FVWAIT,		0,		no_func, 0,	0	},
{ "vdone",	FVDONE,		0,		no_func, 0,	0	},
{ "vaddr",	FVADDR,		0,		no_func, 0,	0	},
{ "vopcnt",	FVOPCNT,	0,		no_func, 0,	0	},
{ "vcont",	FVCONT,		0,		no_func, 0,	0	},
{ "vincr",	FVINCR,		0,		no_func, 0,	0	},
{ "vidle",	FVFIDLE,	0,		no_func, 0,	0	},
{ "vskyid",	FVSKYID,	0,		no_func, 0,	0	},
{ "vrdptr",	FVRDPTR,	0,		no_func, 0,	0	},
{ "vfstat",	FVFSTAT,	0,		no_func, 0,	0	},
{ "vcrdp",	FVCRDP,		0,		no_func, 0,	0	},
{ "vcwrp",	FVCWRP,		0,		no_func, 0,	0	},



/* {	"vcmpm",	FVCMPM,		N3AJRG,	vcmpm,	M_ALL,	R	},	*/
/*
{ "vscmm",	FVSCMM,		VS_BINARY,	vscmm,	M_BP,	R	},
{ "vmcmm",	FVMCMM,		VV_TEST,	vmcmm,	M_ALL,	R	},
{ "vmcmp",	FVMCMP,		VV_TEST,	vmcmp,	M_ALL,	R	},
*/
#endif /* FOOBAR */

};

#define N_NVFS		(sizeof(vec_func_tbl)/sizeof(Vec_Func))

static void create_vfs(SINGLE_QSP_ARG_DECL)
{
	Vec_Func *vfp;

	u_int i;
	for(i=0;i<N_NVFS;i++){
		vfp=new_vf(QSP_ARG  vec_func_tbl[i].vf_name);
		if( vfp == NO_NEW_VEC_FUNC )
			ERROR1("error creating item");
		vfp->vf_code = vec_func_tbl[i].vf_code;
		vfp->vf_func = vec_func_tbl[i].vf_func;
		vfp->vf_flags = vec_func_tbl[i].vf_flags;
		vfp->vf_precmask = vec_func_tbl[i].vf_precmask;
		vfp->vf_typemask = vec_func_tbl[i].vf_typemask;
	}
}
		

static int vf_cmp(CONST void *vfp1,CONST void *vfp2)
{
	if( ((CONST Vec_Func *)vfp1)->vf_code > ((CONST Vec_Func *)vfp2)->vf_code ) return(1);
	else return(-1);
}

static int vfa_cmp(CONST void *vfp1,CONST void *vfp2)
{
	if( ((CONST Vec_Func_Array *)vfp1)->vfa_code > ((CONST Vec_Func_Array *)vfp2)->vfa_code ) return(1);
	else return(-1);
}


void vl_init(SINGLE_QSP_ARG_DECL)
{
	int i;
	static int inited=0;

	if( inited ){
		/*warn("vl_init:  already initialized"); */
		return;
	}

	veclib_debug = add_debug_module(QSP_ARG  "veclib");

	/* sort the table so that each entry is at the location of its code */

	qsort(vec_func_tbl,N_VEC_FUNCS,sizeof(Vec_Func),vf_cmp);
	qsort(vfa_tbl,N_VEC_FUNCS,sizeof(Vec_Func_Array),vfa_cmp);

	/* make sure the table is complete */
	for(i=0;i<N_VEC_FUNCS;i++){
if( verbose ){
sprintf(error_string,"vl_init:  entry %d (%s)     code %d (%s)",i,
vec_func_tbl[i].vf_name,vec_func_tbl[i].vf_code, vec_func_tbl[ vec_func_tbl[i].vf_code ].vf_name);
ADVISE(error_string);
}
		if( vec_func_tbl[i].vf_code != i ){
			sprintf(error_string,
	"vl_init:  Vec_Func table entry %d (%s) has code %d (%s)!?",i,
		vec_func_tbl[i].vf_name,
		vec_func_tbl[i].vf_code,
		vec_func_tbl[ vec_func_tbl[i].vf_code ].vf_name );
			ERROR1(error_string);
		}
		if( vfa_tbl[i].vfa_code != i ){
			sprintf(error_string,
	"vl_init:  Vec_Func_Array table entry %d (%s) has code %d (%s)!?",
		i,vec_func_tbl[i].vf_name,
		vfa_tbl[i].vfa_code,vec_func_tbl[ vfa_tbl[i].vfa_code ].vf_name);
			ERROR1(error_string);
		}
	}

	/* now create some items */

if( !inited ){
	create_vfs(SINGLE_QSP_ARG);

	set_random_seed();	/* use low order bits of microsecond clock */
}

	auto_version(QSP_ARG  "NEWVEC","VersionId_newvec");

	inited++;
}

