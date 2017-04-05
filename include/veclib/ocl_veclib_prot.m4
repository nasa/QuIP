
my_include(`../../include/veclib/ocl_port.m4')

extern void h_ocl_vuni(HOST_CALL_ARG_DECLS);
extern void h_ocl_sp_vuni(HOST_CALL_ARG_DECLS);
extern void h_ocl_dp_vuni(HOST_CALL_ARG_DECLS);

extern void h_ocl_fft2d(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_ift2d(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_fftrows(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_iftrows(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);


