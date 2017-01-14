#ifndef _VL2_VECLIB_PROT_H_
#define _VL2_VECLIB_PROT_H_

#include "veclib/vecgen.h"
#include "veclib/vl2_port.h"


extern void h_vl2_fft2d(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_vl2_ift2d(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_vl2_fftrows(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_vl2_iftrows(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);

extern void vl2_convert(QSP_ARG_DECL  Data_Obj *dpto, Data_Obj *dpfr);


// other specia stuff

//extern void h_vl2_vec_xform(QSP_ARG_DECL  Data_Obj *dpto, Data_Obj *dpfr, Data_Obj *xform );
//extern void h_vl2_homog_xform(QSP_ARG_DECL  Data_Obj *dpto, Data_Obj *dpfr, Data_Obj *xform );
//extern double h_vl2_determinant(Data_Obj *dpfr );

extern void h_vl2_xform_list(HOST_CALL_ARG_DECLS);
extern void h_vl2_vec_xform(HOST_CALL_ARG_DECLS);
extern void h_vl2_homog_xform(HOST_CALL_ARG_DECLS);
extern void h_vl2_determinant(HOST_CALL_ARG_DECLS);


extern int xform_chk(Data_Obj *dpto, Data_Obj *dpfr, Data_Obj *xform );

#endif // ! _VL2_VECLIB_PROT_H_

