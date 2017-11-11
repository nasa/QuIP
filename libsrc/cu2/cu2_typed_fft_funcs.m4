// cu2_fft_funcs.m4 BEGIN


dnl	CUDA_CPX_FFT_BODY(cuda_func,cuda_type,direction)

define(`CUDA_CPX_FFT_BODY',`
	enum cufftResult_t status;
	cufftHandle plan;

	// Create plan for FFT

	status = cufftPlan1d(&plan,
		OBJ_N_TYPE_ELTS( OA_DEST(oap) ),
		CUFFT_C2C, 1 /* BATCH */ );	// why called batch?
	if (status != CUFFT_SUCCESS) {
		sprintf(DEFAULT_ERROR_STRING, "Error in cufftPlan1d: %s\\n", getCUFFTError(status));
		NWARN(DEFAULT_ERROR_STRING);
fprintf(stderr,"requested size:  %d\\n",OBJ_N_TYPE_ELTS(OA_DEST(oap)));
		return;
	}

fprintf(stderr,"cufft complex, direction %d\\n",$3);
	status = $1(plan, ($2 *) OBJ_DATA_PTR( OA_SRC1(oap) ),
			($2 *) OBJ_DATA_PTR( OA_DEST(oap) ), $3);
	if (status != CUFFT_SUCCESS) {
		sprintf(DEFAULT_ERROR_STRING, "Error in $1: %s\\n", getCUFFTError(status));
		NWARN(DEFAULT_ERROR_STRING);
	}

	// BUG?  should we cache the plan?
	cufftDestroy(plan);
')

dnl	CUDA_REAL_FFT_BODY(cuda_exec_func,cuda_xform_type,cuda_dest_type,cuda_src_type,len)

define(`CUDA_REAL_FFT_BODY',`
	enum cufftResult_t status;
	cufftHandle plan;

	// Create plan for FFT

	status = cufftPlan1d(&plan,
		/*OBJ_N_TYPE_ELTS( OA_DEST(oap) )*/ $5,
		$2, 1 /* BATCH */ );	// why called batch?
	if (status != CUFFT_SUCCESS) {
		sprintf(DEFAULT_ERROR_STRING, "Error in cufftPlan1d: %s\\n", getCUFFTError(status));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	status = $1(plan, ($4 *) OBJ_DATA_PTR( OA_SRC1(oap) ),
			($3 *) OBJ_DATA_PTR( OA_DEST(oap) ) );
	if (status != CUFFT_SUCCESS) {
		sprintf(DEFAULT_ERROR_STRING, "Error in $1: %s\\n", getCUFFTError(status));
		NWARN(DEFAULT_ERROR_STRING);
	}

	// BUG?  should we cache the plan?
	cufftDestroy(plan);
')

dnl	Instead of ifdef, better to have separate file for kernels and host funcs?

ifdef(`BUILDING_KERNELS',`
',` dnl else ! BUILDING_KERNELS

// vl2_fft_funcs.m4 buiding_kernels is NOT SET

static void HOST_TYPED_CALL_NAME_CPX(vfft,type_code)( HOST_CALL_ARG_DECLS )
{
CUDA_CPX_FFT_BODY(cuda_cpx_fft_func,cuda_cpx_fft_type,CUFFT_FORWARD)
}

dnl	BUG - too much duplicated code?

static void HOST_TYPED_CALL_NAME_CPX(vift,type_code)( HOST_CALL_ARG_DECLS )
{
CUDA_CPX_FFT_BODY(cuda_cpx_fft_func,cuda_cpx_fft_type,CUFFT_INVERSE)
}

static void HOST_TYPED_CALL_NAME_REAL(vfft,type_code)( HOST_CALL_ARG_DECLS )
{
CUDA_REAL_FFT_BODY(cuda_real_fft_func,CUFFT_R2C,cuda_cpx_fft_type,cuda_real_fft_type,OBJ_N_TYPE_ELTS(OA_SRC1(oap)))
}

static void HOST_TYPED_CALL_NAME_REAL(vift,type_code)( HOST_CALL_ARG_DECLS )
{
CUDA_REAL_FFT_BODY(cuda_real_ift_func,CUFFT_C2R,cuda_real_fft_type,cuda_cpx_fft_type,OBJ_N_TYPE_ELTS(OA_DEST(oap)))
}

dnl static void HOST_TYPED_CALL_NAME(rvfft,type_code)(HOST_CALL_ARG_DECLS)
dnl { NWARN("HOST_TYPED_CALL_NAME(rvfft,type_code) not implemented!?"); }

dnl static void HOST_TYPED_CALL_NAME(rvift,type_code)(HOST_CALL_ARG_DECLS)
dnl { NWARN("HOST_TYPED_CALL_NAME(rvift,type_code) not implemented!?"); }

dnl static void HOST_TYPED_CALL_NAME(cvfft,type_code)(HOST_CALL_ARG_DECLS)
dnl { NWARN("HOST_TYPED_CALL_NAME(cvfft,type_code) not implemented!?"); }

dnl static void HOST_TYPED_CALL_NAME(cvift,type_code)(HOST_CALL_ARG_DECLS)
dnl { NWARN("HOST_TYPED_CALL_NAME(cvift,type_code) not implemented!?"); }

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

') dnl endif ! BUILDING_KERNELS

// cu2_fft_funcs.m4 DONE

