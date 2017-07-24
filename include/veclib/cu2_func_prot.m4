
/* cu2_func_prot.m4 BEGIN */

// these are special cases...
extern void h_cu2_sp_vuni(HOST_CALL_ARG_DECLS);
extern void h_cu2_dp_vuni(HOST_CALL_ARG_DECLS);
extern void h_cu2_vuni(HOST_CALL_ARG_DECLS);

// BUG special case needs to be handled!?
extern void h_cu2_vshl(HOST_CALL_ARG_DECLS);
extern void h_cu2_vsshl(HOST_CALL_ARG_DECLS);
extern void h_cu2_vsshl2(HOST_CALL_ARG_DECLS);

extern void g_cu2_vfft(QSP_ARG_DECL  Data_Obj *dpto, Data_Obj *dpfr);

dnl extern void HOST_CALL_NAME(fft2d)( FFT_FUNC_ARG_DECLS );
dnl extern void HOST_CALL_NAME(ift2d)( FFT_FUNC_ARG_DECLS );
dnl extern void HOST_CALL_NAME(fftrows)( FFT_FUNC_ARG_DECLS );
dnl extern void HOST_CALL_NAME(iftrows)( FFT_FUNC_ARG_DECLS );

/* cu2_func_prot.m4 DONE */
