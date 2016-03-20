#ifndef _CU2_VECLIB_PROT_H_
#define _CU2_VECLIB_PROT_H_

#include "veclib/cu2_port.h"
#include "veclib/gen_veclib_prot.h"

// these are special cases...
extern void h_cu2_sp_vuni(HOST_CALL_ARG_DECLS);
extern void h_cu2_dp_vuni(HOST_CALL_ARG_DECLS);
extern void h_cu2_vuni(HOST_CALL_ARG_DECLS);

// BUG special case needs to be handled!?
extern void h_cu2_vshl(HOST_CALL_ARG_DECLS);
extern void h_cu2_vsshl(HOST_CALL_ARG_DECLS);
extern void h_cu2_vsshl2(HOST_CALL_ARG_DECLS);

extern void g_cu2_vfft(QSP_ARG_DECL  Data_Obj *dpto, Data_Obj *dpfr);
extern void HOST_CALL_NAME(fft2d)(HOST_CALL_ARG_DECLS);
extern void HOST_CALL_NAME(ift2d)(HOST_CALL_ARG_DECLS);
extern void HOST_CALL_NAME(fftrows)(HOST_CALL_ARG_DECLS);
extern void HOST_CALL_NAME(iftrows)(HOST_CALL_ARG_DECLS);

#endif // ! _CU2_VECLIB_PROT_H_

