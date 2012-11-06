
#include "quip_config.h"

char VersionId_cuda_cuda_functbl[] = QUIP_VERSION_STRING;

#ifdef HAVE_CUDA

#include <stdlib.h>	/* qsort */
#include "nvf_api.h"
#include "debug.h"

#include "my_vector_functions.h"

// just for testing
extern void g_vramp2d( Vec_Obj_Args * );

typedef struct gpu_func {
	int	gf_code;
	void	(*gf_func)(Vec_Obj_Args *);
} GPU_Func;

#define GPU_UNIMP	((void (*)(Vec_Obj_Args *))NULL)

GPU_Func gpu_func_tbl[N_VEC_FUNCS]={
{	FVSET,		g_vset		},
{	FVMOV,		g_rvmov		},
{	FVADD,		g_vadd		},
{	FVSUB,		g_rvsub		},
{	FVMUL,		g_rvmul		},
{	FVDIV,		g_rvdiv		},
{	FVNEG,		g_rvneg		},
{	FVSQR,		g_rvsqr		},
{	FVRAMP1D,	g_vramp1d	},
{	FVRAMP2D,	g_vramp2d	},
{	FVSADD,		g_rvsadd	},
{	FVSSUB,		g_rvssub	},
{	FVSMUL,		g_rvsmul	},
{	FVSDIV,		g_rvsdiv	},
{	FVSDIV2,	g_rvsdiv2	},
{	FVABS,		g_vabs		},
{	FVSIGN,		g_vsign		},

{	FVSQRT,		g_vsqrt		},
{	FVSIN,		g_vsin		},
{	FVCOS,		g_vcos		},
{	FVATAN,		g_vatan		},
{	FVTAN,		g_vtan		},
{	FVATAN2,	g_vatan2	},
{	FVSATAN2,	g_vsatan2	},
{	FVSATAN22,	g_vsatan22	},
{	FVLOG,		g_vlog		},
{	FVLOG10,	g_vlog10	},
{	FVEXP,		g_vexp		},
{	FVERF,		g_verf		},
{	FVPOW,		g_rvpow		},
{	FVSPOW,		g_vspow		},
{	FVSPOW2,	g_vspow2	},

{	FVMIN,		g_vmin		},
{	FVMAX,		g_vmax		},
{	FVMINM,		g_vminm		},
{	FVMAXM,		g_vmaxm		},
{	FVSMIN,		g_vsmin		},
{	FVSMAX,		g_vsmax		},
{	FVSMNM,		g_vsmnm		},
{	FVSMXM,		g_vsmxm		},

{	FVMINV,		g_vminv		},
{	FVMAXV,		g_vmaxv		},
{	FVMNMV,		g_vmnmv		},
{	FVMXMV,		g_vmxmv		},
{	FVMINI,		g_vmini		},
{	FVMAXI,		g_vmaxi		},
{	FVMNMI,		g_vmnmi		},
{	FVMXMI,		g_vmxmi		},
{	FVMING,		g_vming		},
{	FVMAXG,		g_vmaxg		},
{	FVMNMG,		g_vmnmg		},
{	FVMXMG,		g_vmxmg		},

{	FVFLOOR,	g_vfloor	},

{	FVROUND,	/* g_vround */ GPU_UNIMP	},
{	FVCEIL,		/* g_vceil */ GPU_UNIMP	},
{	FVRINT,		/* g_vrint */ GPU_UNIMP	},
{	FVJ0,		/* g_vj0	 */ GPU_UNIMP	},
{	FVJ1,		/* g_vj1 */ GPU_UNIMP		},
{	FVACOS,		g_vacos	},
{	FVASIN,		g_vasin	},
{	FVATN2,		/* g_vatn2 */ GPU_UNIMP	},
{	FVUNI,		g_vuni		},

{	FVAND,		g_vand		},
{	FVNAND,		g_vnand		},
{	FVOR,		g_vor		},
{	FVXOR,		g_vxor		},
{	FVNOT,		g_vnot		},
{	FVCOMP,		g_vcomp		},
{	FVMOD,		g_vmod		},
{	FVSMOD,		g_vsmod		},
{	FVSMOD2,	g_vsmod2	},

{	FVSAND,		g_vsand		},
{	FVSOR,		g_vsor		},
{	FVSXOR,		g_vsxor		},
{	FVSHR,		g_vshr		},
{	FVSSHR,		g_vsshr		},
{	FVSSHR2,	g_vsshr2	},
{	FVSHL,		g_vshl		},
{	FVSSHL,		g_vsshl		},
{	FVSSHL2,	g_vsshl2	},

{	FVSUM,		g_vsum		},
{	FVDOT,		g_vdot		},
{	FVRAND,		/* g_vrand */ GPU_UNIMP	},

{	FVSMLT,		g_vsm_lt	},
{	FVSMGT,		g_vsm_gt	},
{	FVSMLE,		g_vsm_le	},
{	FVSMGE,		g_vsm_ge	},
{	FVSMNE,		g_vsm_ne	},
{	FVSMEQ,		g_vsm_eq	},

{	FVVMLT,		g_vvm_lt	},
{	FVVMGT,		g_vvm_gt	},
{	FVVMLE,		g_vvm_le	},
{	FVVMGE,		g_vvm_ge	},
{	FVVMNE,		g_vvm_ne	},
{	FVVMEQ,		g_vvm_eq	},

{	FVVVSLCT,	g_vvv_slct	},
{	FVVSSLCT,	g_vvs_slct	},
{	FVSSSLCT,	g_vss_slct	},

{	FVV_VV_LT,	g_vv_vv_lt	},
{	FVV_VV_GT,	g_vv_vv_gt	},
{	FVV_VV_LE,	g_vv_vv_le	},
{	FVV_VV_GE,	g_vv_vv_ge	},
{	FVV_VV_EQ,	g_vv_vv_eq	},
{	FVV_VV_NE,	g_vv_vv_ne	},

{	FVV_VS_LT,	g_vv_vs_lt	},
{	FVV_VS_GT,	g_vv_vs_gt	},
{	FVV_VS_LE,	g_vv_vs_le	},
{	FVV_VS_GE,	g_vv_vs_ge	},
{	FVV_VS_EQ,	g_vv_vs_eq	},
{	FVV_VS_NE,	g_vv_vs_ne	},

{	FVS_VV_LT,	g_vs_vv_lt	},
{	FVS_VV_GT,	g_vs_vv_gt	},
{	FVS_VV_LE,	g_vs_vv_le	},
{	FVS_VV_GE,	g_vs_vv_ge	},
{	FVS_VV_EQ,	g_vs_vv_eq	},
{	FVS_VV_NE,	g_vs_vv_ne	},

{	FVS_VS_LT,	g_vs_vs_lt	},
{	FVS_VS_GT,	g_vs_vs_gt	},
{	FVS_VS_LE,	g_vs_vs_le	},
{	FVS_VS_GE,	g_vs_vs_ge	},
{	FVS_VS_EQ,	g_vs_vs_eq	},
{	FVS_VS_NE,	g_vs_vs_ne	},

{	FSS_VV_LT,	g_ss_vv_lt	},
{	FSS_VV_GT,	g_ss_vv_gt	},
{	FSS_VV_LE,	g_ss_vv_le	},
{	FSS_VV_GE,	g_ss_vv_ge	},
{	FSS_VV_EQ,	g_ss_vv_eq	},
{	FSS_VV_NE,	g_ss_vv_ne	},

{	FSS_VS_LT,	g_ss_vs_lt	},
{	FSS_VS_GT,	g_ss_vs_gt	},
{	FSS_VS_LE,	g_ss_vs_le	},
{	FSS_VS_GE,	g_ss_vs_ge	},
{	FSS_VS_EQ,	g_ss_vs_eq	},
{	FSS_VS_NE,	g_ss_vs_ne	},

{	FVMGSQ,		g_vmgsq		},
{	FVCMUL,		/* g_vcmul */ GPU_UNIMP	},
{	FVSCML,		/* g_vscml */ GPU_UNIMP	},
{	FVCONJ,		/* g_vconj */ GPU_UNIMP	},
{	FVFFT,		/* g_vfft */ GPU_UNIMP	},
{	FVIFT,		/* g_vift */ GPU_UNIMP	},

{	FVBND,		/* g_vbnd */ GPU_UNIMP	},
{	FVIBND,		/* g_vibnd */ GPU_UNIMP	},
{	FVCLIP,		/* g_vclip */ GPU_UNIMP	},
{	FVICLP,		/* g_viclp */ GPU_UNIMP	},
{	FVCMP,		/* g_vcmp */ GPU_UNIMP	},
{	FVSCMP,		/* g_vscmp */ GPU_UNIMP	},
{	FVSCMP2,	/* g_vscmp2 */ GPU_UNIMP	},

{	FVB2I,		h_by2in		},
{	FVB2L,		h_by2di		},
{	FVB2LL,		h_by2li		},
{	FVB2SP,		h_by2sp		},
{	FVB2DP,		h_by2dp		},
{	FVB2UB,		h_by2uby	},
{	FVB2UI,		h_by2uin	},
{	FVB2UL,		h_by2udi	},
{	FVB2ULL,	h_by2uli	},

{	FVI2B,		h_in2by		},
{	FVI2L,		h_in2di		},
{	FVI2LL,		h_in2li		},
{	FVI2SP,		h_in2sp		},
{	FVI2DP,		h_in2dp		},
{	FVI2UB,		h_in2uby	},
{	FVI2UI,		h_in2uin	},
{	FVI2UL,		h_in2udi	},
{	FVI2ULL,	h_in2uli	},

{	FVL2B,		h_di2by		},
{	FVL2I,		h_di2in		},
{	FVL2LL,		h_di2li		},
{	FVL2SP,		h_di2sp		},
{	FVL2DP,		h_di2dp		},
{	FVL2UB,		h_di2uby	},
{	FVL2UI,		h_di2uin	},
{	FVL2UL,		h_di2udi	},
{	FVL2ULL,	h_di2uli	},

{	FVLL2B,		h_li2by		},
{	FVLL2I,		h_li2in		},
{	FVLL2L,		h_li2di		},
{	FVLL2SP,	h_li2sp		},
{	FVLL2DP,	h_li2dp		},
{	FVLL2UB,	h_li2uby	},
{	FVLL2UI,	h_li2uin	},
{	FVLL2UL,	h_li2udi	},
{	FVLL2ULL,	h_li2uli	},

{	FVSP2B,		h_sp2by		},
{	FVSP2I,		h_sp2in		},
{	FVSP2L,		h_sp2di		},
{	FVSP2LL,	h_sp2li		},
{	FVSPDP,		h_sp2dp		},
{	FVSP2UB,	h_sp2uby	},
{	FVSP2UI,	h_sp2uin	},
{	FVSP2UL,	h_sp2udi	},
{	FVSP2ULL,	h_sp2uli	},

{	FVDP2B,		h_dp2by		},
{	FVDP2I,		h_dp2in		},
{	FVDP2L,		h_dp2di		},
{	FVDP2LL,	h_dp2li		},
{	FVDPSP,		h_dp2sp		},
{	FVDP2UB,	h_dp2uby	},
{	FVDP2UI,	h_dp2uin	},
{	FVDP2UL,	h_dp2udi	},
{	FVDP2ULL,	h_dp2uli	},

{	FVUB2B,		h_uby2by	},
{	FVUB2I,		h_uby2in	},
{	FVUB2L,		h_uby2di	},
{	FVUB2LL,	h_uby2li	},
{	FVUB2SP,	h_uby2sp	},
{	FVUB2DP,	h_uby2dp	},
{	FVUB2UI,	h_uby2uin	},
{	FVUB2UL,	h_uby2udi	},
{	FVUB2ULL,	h_uby2uli	},

{	FVUI2B,		h_uin2by	},
{	FVUI2I,		h_uin2in	},
{	FVUI2L,		h_uin2di	},
{	FVUI2LL,	h_uin2li	},
{	FVUI2SP,	h_uin2sp	},
{	FVUI2DP,	h_uin2dp	},
{	FVUI2UB,	h_uin2uby	},
{	FVUI2UL,	h_uin2udi	},
{	FVUI2ULL,	h_uin2uli	},

{	FVUL2B,		h_udi2by	},
{	FVUL2I,		h_udi2in	},
{	FVUL2L,		h_udi2di	},
{	FVUL2LL,	h_udi2li	},
{	FVUL2SP,	h_udi2sp	},
{	FVUL2DP,	h_udi2dp	},
{	FVUL2UB,	h_udi2uby	},
{	FVUL2UI,	h_udi2uin	},
{	FVUL2ULL,	h_udi2uli	},

{	FVULL2B,	h_uli2by	},
{	FVULL2I,	h_uli2in	},
{	FVULL2L,	h_uli2di	},
{	FVULL2LL,	h_uli2li	},
{	FVULL2SP,	h_uli2sp	},
{	FVULL2DP,	h_uli2dp	},
{	FVULL2UB,	h_uli2uby	},
{	FVULL2UI,	h_uli2uin	},
{	FVULL2UL,	h_uli2udi	},

};

static int gf_cmp(CONST void *gfp1,CONST void *gfp2)
{
	if( ((CONST GPU_Func *)gfp1)->gf_code > ((CONST GPU_Func *)gfp2)->gf_code ) return(1);
	else return(-1);
}

static int gtbl_inited=0;

void gtbl_init(void)
{
	int i;

	if( gtbl_inited ){
		NWARN("gtbl_init:  already initialized");
		return;
	}

	/* sort the table so that each entry is at the location of its code */

	qsort(gpu_func_tbl,N_VEC_FUNCS,sizeof(GPU_Func),gf_cmp);

	/* make sure the table is complete */
	for(i=0;i<N_VEC_FUNCS;i++){
		if( gpu_func_tbl[i].gf_code != i ){
			int j;

			sprintf(DEFAULT_ERROR_STRING,
	"gtbl_init:  GPU_Func table entry %d has code %d!?",i, gpu_func_tbl[i].gf_code);
			NWARN(DEFAULT_ERROR_STRING);
			sprintf(DEFAULT_ERROR_STRING,"Expected function %s.",vec_func_tbl[i].vf_name);
			advise(DEFAULT_ERROR_STRING);
			/* Dump all the entries to facilitate locating the missing entry */
			for(j=0;j<N_VEC_FUNCS;j++){
				sprintf(DEFAULT_ERROR_STRING,"\tentry %d holds code %d",j,gpu_func_tbl[j].gf_code);
				advise(DEFAULT_ERROR_STRING);
			}
			NERROR1("Please correct error in file cuda_func_tbl.cpp");
		}
	}
	gtbl_inited=1;
}

int gpu_dispatch( Vec_Func *vfp, Vec_Obj_Args *oap )
{
	int i;

	if( !gtbl_inited ) gtbl_init();
	
	i = vfp->vf_code;
	if( gpu_func_tbl[i].gf_func == GPU_UNIMP ){
		sprintf(DEFAULT_ERROR_STRING,"Sorry, function %s has not yet been implemented for the GPU.",vfp->vf_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	(*gpu_func_tbl[i].gf_func)(oap);
	return(0);
}

#endif /* HAVE_CUDA */

