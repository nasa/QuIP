

//extern void h_vl2_vuni(HOST_CALL_ARG_DECLS);
//extern void h_vl2_sp_vuni(HOST_CALL_ARG_DECLS);
//extern void h_vl2_dp_vuni(HOST_CALL_ARG_DECLS);

extern void h_vl2_fft2d(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_vl2_ift2d(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_vl2_fftrows(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_vl2_iftrows(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);

extern void h_vl2_xform_list(const int code, Vec_Obj_Args *oap);
extern void h_vl2_vec_xform(const int code, Vec_Obj_Args *oap);
extern void h_vl2_homog_xform(HOST_CALL_ARG_DECLS);
extern void h_vl2_determinant(HOST_CALL_ARG_DECLS);

extern int xform_chk(Data_Obj *dpto, Data_Obj *dpfr, Data_Obj *xform );
