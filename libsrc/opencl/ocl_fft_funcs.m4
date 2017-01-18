dnl nothing here yet


dnl BUG? use macros to generate these???

ifdef(`BUILDING_KERNELS',`',`

/* ocl_fft_funcs.m4 `type_code' = type_code */
static void HOST_TYPED_CALL_NAME(rvfft,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(rvfft,type_code) not implemented!?"); }

static void HOST_TYPED_CALL_NAME(cvfft,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(cvfft,type_code) not implemented!?"); }

static void HOST_TYPED_CALL_NAME(rvift,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(rvift,type_code) not implemented!?"); }

static void HOST_TYPED_CALL_NAME(cvift,type_code)(HOST_CALL_ARG_DECLS)
{ NWARN("HOST_TYPED_CALL_NAME(cvift,type_code) not implemented!?"); }

') dnl endif // ! BUILDING_KERNELS




