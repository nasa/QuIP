
extern void h_ocl_vuni(HOST_CALL_ARG_DECLS);
extern void h_ocl_sp_vuni(HOST_CALL_ARG_DECLS);
extern void h_ocl_dp_vuni(HOST_CALL_ARG_DECLS);

dnl	extern void ocl_fft_shutdown(SINGLE_QSP_ARG_DECL);

extern void h_ocl_fft2d(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_ift2d(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_fftrows(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_iftrows(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);


