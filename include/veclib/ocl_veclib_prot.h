#ifndef _OCL_VECLIB_PROT_H_
#define _OCL_VECLIB_PROT_H_

#include "veclib/ocl_port.h"
//#include "veclib/gen_veclib_prot.h"

///* Here are the conversions */
//#include "host_conv_prot.h"

extern void h_ocl_vuni(HOST_CALL_ARG_DECLS);
extern void h_ocl_sp_vuni(HOST_CALL_ARG_DECLS);
extern void h_ocl_dp_vuni(HOST_CALL_ARG_DECLS);

extern void h_ocl_fft2d(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_ift2d(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_fftrows(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_iftrows(VFCODE_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);

#endif // ! _OCL_VECLIB_PROT_H_

