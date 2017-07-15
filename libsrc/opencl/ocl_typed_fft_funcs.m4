
#ifdef HAVE_CLFFT
#include <clFFT.h>
#endif // HAVE_CLFFT

ifdef(`BUILDING_KERNELS',`',`

/* cu2_fft_funcs.m4 `type_code' = type_code */
static void HOST_TYPED_CALL_NAME(rvfft,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(rvfft,type_code) not implemented!?"); }

static void HOST_TYPED_CALL_NAME(rvift,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(rvift,type_code) not implemented!?"); }

static void HOST_TYPED_CALL_NAME(cvfft,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(cvfft,type_code) not implemented!?"); }

static void HOST_TYPED_CALL_NAME(cvift,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(cvift,type_code) not implemented!?"); }

static void HOST_TYPED_CALL_NAME(rfft2d,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(rfft2d,type_code) not implemented!?"); }

static void HOST_TYPED_CALL_NAME(rift2d,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(rift2d,type_code) not implemented!?"); }

static void HOST_TYPED_CALL_NAME(cfft2d,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(cfft2d,type_code) not implemented!?"); }

static void HOST_TYPED_CALL_NAME(cift2d,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(cift2d,type_code) not implemented!?"); }

static void HOST_TYPED_CALL_NAME(rfftrows,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(rfftrows,type_code) not implemented!?"); }

static void HOST_TYPED_CALL_NAME(riftrows,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(riftrows,type_code) not implemented!?"); }

static void HOST_TYPED_CALL_NAME(cfftrows,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(cfftrows,type_code) not implemented!?"); }

static void HOST_TYPED_CALL_NAME(ciftrows,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(ciftrows,type_code) not implemented!?"); }

') dnl endif // ! BUILDING_KERNELS




