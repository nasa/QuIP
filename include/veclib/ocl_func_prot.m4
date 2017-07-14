
extern void h_ocl_vuni(HOST_CALL_ARG_DECLS);
extern void h_ocl_sp_vuni(HOST_CALL_ARG_DECLS);
extern void h_ocl_dp_vuni(HOST_CALL_ARG_DECLS);

dnl	extern void ocl_fft_shutdown(SINGLE_QSP_ARG_DECL);

#include "platform.h"
#
extern void h_ocl_fft2d( FFT_FUNC_ARG_DECLS );
extern void h_ocl_ift2d( FFT_FUNC_ARG_DECLS );
extern void h_ocl_fftrows( FFT_FUNC_ARG_DECLS );
extern void h_ocl_iftrows( FFT_FUNC_ARG_DECLS );


