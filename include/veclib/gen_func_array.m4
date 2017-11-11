/* gen_func_array.m4 BEGIN */

RCQALL_SAME_PREC_ARR(vset,FVSET),
RCQALL_SAME_PREC_ARR(vmov,FVMOV),
RALL_SAME_PREC_ARR(vconv2by,FVCONV2BY),
RALL_SAME_PREC_ARR(vconv2in,FVCONV2IN),
RALL_SAME_PREC_ARR(vconv2di,FVCONV2DI),
RALL_SAME_PREC_ARR(vconv2li,FVCONV2LI),
RALL_SAME_PREC_ARR(vconv2uby,FVCONV2UBY),
RALL_SAME_PREC_ARR(vconv2uin,FVCONV2UIN),
RALL_SAME_PREC_ARR(vconv2udi,FVCONV2UDI),
RALL_SAME_PREC_ARR(vconv2uli,FVCONV2ULI),
RALL_SAME_PREC_ARR(vconv2sp,FVCONV2SP),
RALL_SAME_PREC_ARR(vconv2dp,FVCONV2DP),
RCMQPALL_ARR(vadd,FVADD),
RCMQPALL_ARR(vsub,FVSUB),
RCMQPALL_ARR(vmul,FVMUL),
RCMQPALL_ARR(vdiv,FVDIV),
dnl //SIGNED_ARR(vneg,FVNEG),
dnl //RALL_ARR(rvneg,FVNEG),
RCQALL_ARR(vneg,FVNEG),
RCQALL_ARR(vsqr,FVSQR),
RALL_ARR(vramp1d,FVRAMP1D),
RALL_ARR(vramp2d,FVRAMP2D),
RCMQPALL_ARR(vsadd,FVSADD),
RCMQPALL_ARR(vssub,FVSSUB),
RCMQPALL_ARR(vssub2,FVSSUB2),
RCMQPALL_ARR(vsmul,FVSMUL),
RCMQPALL_ARR(vsdiv,FVSDIV),
RCMQPALL_ARR(vsdiv2,FVSDIV2),
RALL_ARR(vabs,FVABS),
RALL_ARR(vsign,FVSIGN),
RCFLT_ARR(vexp,FVEXP),

RFLT_ARR(vsqrt,FVSQRT),
RFLT_ARR(vsin,FVSIN),
RFLT_ARR(vcos,FVCOS),
RFLT_ARR(vtan,FVTAN),
RFLT_ARR(vatan,FVATAN),
RFLT_ARR(vatan2,FVATAN2),
RFLT_ARR(vsatan2,FVSATAN2),
RFLT_ARR(vsatan22,FVSATAN22),
RFLT_ARR(vlog,FVLOG),
RFLT_ARR(vlog10,FVLOG10),
dnl //RFLT_ARR(rvexp,FVEXP),
RFLT_ARR(verf,FVERF),
RFLT_ARR(verfinv,FVERFINV),
dnl //RFLT_ARR(rvpow,FVPOW),
RCFLT_ARR(vpow,FVPOW),
RFLT_ARR(vspow,FVSPOW),
RFLT_ARR(vspow2,FVSPOW2),

RALL_ARR(vmin,FVMIN),
RALL_ARR(vmax,FVMAX),
RALL_ARR(vminm,FVMINM),
RALL_ARR(vmaxm,FVMAXM),
RALL_ARR(vsmin,FVSMIN),
RALL_ARR(vsmax,FVSMAX),
RALL_ARR(vsmnm,FVSMNM),
RALL_ARR(vsmxm,FVSMXM),

RALL_ARR(vminv,FVMINV),
RALL_ARR(vmaxv,FVMAXV),
RALL_ARR(vmnmv,FVMNMV),
RALL_ARR(vmxmv,FVMXMV),
RALL_ARR(vmini,FVMINI),
RALL_ARR(vmaxi,FVMAXI),
RALL_ARR(vmnmi,FVMNMI),
RALL_ARR(vmxmi,FVMXMI),
RALL_ARR(vming,FVMING),
RALL_ARR(vmaxg,FVMAXG),
RALL_ARR(vmnmg,FVMNMG),
RALL_ARR(vmxmg,FVMXMG),

RFLT_ARR(vfloor,FVFLOOR),
RFLT_ARR(vtrunc,FVTRUNC),
RFLT_ARR(vround,FVROUND),
RFLT_ARR(vceil,FVCEIL),
RFLT_ARR(vrint,FVRINT),

ifdef(`BUILD_FOR_GPU',`
dnl // put null entries here for these funcs with no GPU implementation
ifdef(`BUILD_FOR_CUDA',`
NULL_ARR(vuni,FVUNI),
',`
RFLT_SAME_PREC_ARR(vuni,FVUNI),
')
NULL_ARR(vj0,FVJ0),
NULL_ARR(vj1,FVJ1),
NULL_ARR(vgamma,FVGAMMA),
NULL_ARR(vlngamma,FVLNGAMMA),
NULL_ARR(visnan,FVISNAN),
NULL_ARR(visinf,FVISINF),
NULL_ARR(visnorm,FVISNORM),
NULL_ARR(vrand,FVRAND),
',` dnl else // ! BUILD_FOR_GPU
RFLT_SAME_PREC_ARR(vuni,FVUNI),
RFLT_ARR(vj0,FVJ0),
RFLT_ARR(vj1,FVJ1),
RFLT_ARR(vgamma,FVGAMMA),
RFLT_ARR(vlngamma,FVLNGAMMA),
RFLT_ARR(visnan,FVISNAN),
RFLT_ARR(visinf,FVISINF),
RFLT_ARR(visnorm,FVISNORM),
RCALL_ARR(vrand,FVRAND),
') dnl endif // ! BUILD_FOR_GPU

RCFLT_ARR2(vfft,FVFFT),
RCFLT_ARR2(vift,FVIFT),

RCFLT_ARR2(fft2d,FVFFT2D),
RCFLT_ARR2(ift2d,FVIFT2D),
RCFLT_ARR2(fftrows,FVFFTROWS),
RCFLT_ARR2(iftrows,FVIFTROWS),

RFLT_ARR(vacos,FVACOS),
RFLT_ARR(vasin,FVASIN),
RFLT_ARR(vatn2,FVATN2),

/* bitwise operators, integer only math */
REAL_INT_ARR(vand,FVAND),
REAL_INT_ARR(vnand,FVNAND),
REAL_INT_ARR(vor,FVOR),
REAL_INT_ARR(vxor,FVXOR),
REAL_INT_ARR(vnot,FVNOT),
REAL_INT_ARR(vcomp,FVCOMP),
REAL_INT_ARR(vsand,FVSAND),
REAL_INT_ARR(vsor,FVSOR),
REAL_INT_ARR(vsxor,FVSXOR),

/*
REAL_INT_ARR(vmod,FVMOD),
REAL_INT_ARR(vsmod,FVSMOD),
REAL_INT_ARR(vsmod2,FVSMOD2),

REAL_INT_ARR(vshr,FVSHR),
REAL_INT_ARR(vsshr,FVSSHR),
REAL_INT_ARR(vsshr2,FVSSHR2),
REAL_INT_ARR(vshl,FVSHL),
REAL_INT_ARR(vsshl,FVSSHL),
REAL_INT_ARR(vsshl2,FVSSHL2),
*/
REAL_INT_ARR_NO_BITMAP(vmod,FVMOD),
REAL_INT_ARR_NO_BITMAP(vsmod,FVSMOD),
REAL_INT_ARR_NO_BITMAP(vsmod2,FVSMOD2),

REAL_INT_ARR_NO_BITMAP(vshr,FVSHR),
REAL_INT_ARR_NO_BITMAP(vsshr,FVSSHR),
REAL_INT_ARR_NO_BITMAP(vsshr2,FVSSHR2),
REAL_INT_ARR_NO_BITMAP(vshl,FVSHL),
REAL_INT_ARR_NO_BITMAP(vsshl,FVSSHL),
REAL_INT_ARR_NO_BITMAP(vsshl2,FVSSHL2),

REAL_INT_ARR_NO_BITMAP(vtolower,FVTOLOWER),
REAL_INT_ARR_NO_BITMAP(vtoupper,FVTOUPPER),
REAL_INT_ARR_NO_BITMAP(vislower,FVISLOWER),
REAL_INT_ARR_NO_BITMAP(visupper,FVISUPPER),
REAL_INT_ARR_NO_BITMAP(visalpha,FVISALPHA),
REAL_INT_ARR_NO_BITMAP(visalnum,FVISALNUM),
REAL_INT_ARR_NO_BITMAP(visdigit,FVISDIGIT),
REAL_INT_ARR_NO_BITMAP(visspace,FVISSPACE),
REAL_INT_ARR_NO_BITMAP(viscntrl,FVISCNTRL),
REAL_INT_ARR_NO_BITMAP(visblank,FVISBLANK),

dnl /* RC_FIXED_ARR(vmov,FVMOV, bmvmov), */

RCQALL_ARR(vsum,FVSUM),
dnl // BUG - need to deal with vdot
RCALL_ARR(vdot,FVDOT),

RALL_ARR(vsm_lt,FVSMLT),
RALL_ARR(vsm_gt,FVSMGT),
RALL_ARR(vsm_le,FVSMLE),
RALL_ARR(vsm_ge,FVSMGE),
RALL_ARR(vsm_ne,FVSMNE),
RALL_ARR(vsm_eq,FVSMEQ),

RALL_ARR(vvm_lt,FVVMLT),
RALL_ARR(vvm_gt,FVVMGT),
RALL_ARR(vvm_le,FVVMLE),
RALL_ARR(vvm_ge,FVVMGE),
RALL_ARR(vvm_ne,FVVMNE),
RALL_ARR(vvm_eq,FVVMEQ),

RCQALL_ARR(vvv_slct,FVVVSLCT),
RCQALL_ARR(vvs_slct,FVVSSLCT),
RCQALL_ARR(vss_slct,FVSSSLCT),

RALL_ARR(vv_vv_lt,FVV_VV_LT),
RALL_ARR(vv_vv_gt,FVV_VV_GT),
RALL_ARR(vv_vv_le,FVV_VV_LE),
RALL_ARR(vv_vv_ge,FVV_VV_GE),
RALL_ARR(vv_vv_eq,FVV_VV_EQ),
RALL_ARR(vv_vv_ne,FVV_VV_NE),

RALL_ARR(vv_vs_lt,FVV_VS_LT),
RALL_ARR(vv_vs_gt,FVV_VS_GT),
RALL_ARR(vv_vs_le,FVV_VS_LE),
RALL_ARR(vv_vs_ge,FVV_VS_GE),
RALL_ARR(vv_vs_eq,FVV_VS_EQ),
RALL_ARR(vv_vs_ne,FVV_VS_NE),

RALL_ARR(vs_vv_lt,FVS_VV_LT),
RALL_ARR(vs_vv_gt,FVS_VV_GT),
RALL_ARR(vs_vv_le,FVS_VV_LE),
RALL_ARR(vs_vv_ge,FVS_VV_GE),
RALL_ARR(vs_vv_eq,FVS_VV_EQ),
RALL_ARR(vs_vv_ne,FVS_VV_NE),

RALL_ARR(vs_vs_lt,FVS_VS_LT),
RALL_ARR(vs_vs_gt,FVS_VS_GT),
RALL_ARR(vs_vs_le,FVS_VS_LE),
RALL_ARR(vs_vs_ge,FVS_VS_GE),
RALL_ARR(vs_vs_eq,FVS_VS_EQ),
RALL_ARR(vs_vs_ne,FVS_VS_NE),

RALL_ARR(ss_vv_lt,FSS_VV_LT),
RALL_ARR(ss_vv_gt,FSS_VV_GT),
RALL_ARR(ss_vv_le,FSS_VV_LE),
RALL_ARR(ss_vv_ge,FSS_VV_GE),
RALL_ARR(ss_vv_eq,FSS_VV_EQ),
RALL_ARR(ss_vv_ne,FSS_VV_NE),

RALL_ARR(ss_vs_lt,FSS_VS_LT),
RALL_ARR(ss_vs_gt,FSS_VS_GT),
RALL_ARR(ss_vs_le,FSS_VS_LE),
RALL_ARR(ss_vs_ge,FSS_VS_GE),
RALL_ARR(ss_vs_eq,FSS_VS_EQ),
RALL_ARR(ss_vs_ne,FSS_VS_NE),

dnl /*
dnl FUNC_ARR(vscmm,FVSCMM),
dnl FUNC_ARR(vmcmm,FVMCMM),
dnl FUNC_ARR(vmcmp,FVMCMP),
dnl */

/* complex stuff */

RFLT_ARR(vmgsq,FVMGSQ),
CFLT_ARR(vcmul,FVCMUL),
RFLT_ARR(vscml,FVSCML),
CFLT_ARR(vconj,FVCONJ),

RALL_ARR(vbnd,FVBND),
RALL_ARR(vibnd,FVIBND),
RALL_ARR(vclip,FVCLIP),
RALL_ARR(viclp,FVICLP),
RALL_ARR(vcmp,FVCMP),
RALL_ARR(vscmp,FVSCMP),
RALL_ARR(vscmp2,FVSCMP2),

dnl	Mapping complex and quaternions should be straightforward,
dnl	but not immediately necessary...
dnl
dnl RCQALL_ARR(vlutmapb,FVLUTMAPB),
RALL_ARR(vlutmapb,FVLUTMAPB),
RALL_ARR(vlutmaps,FVLUTMAPS),

dnl	/* Type conversions
dnl	 *
dnl	 * For now, bitmaps are constrained to be a single unsigned type,
dnl	 * determined at compile time.  But here the conversion/unconversion
dnl	 * functions are installed for all unsigned types, regardless of which
dnl	 * one is actually used for bitmaps.  This should be safe, because
dnl	 * these are only called when one object is a bitmap, and that should
dnl	 * never be the wrong type...
dnl	 */

/* gen_func_array.m4 END */

