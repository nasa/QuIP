dnl	Put fft stuff here??

suppress_no


static void ocl_fft_shutdown(SINGLE_QSP_ARG_DECL)
{
#ifdef HAVE_CLFFT
	cl_int err;
	err=clfftTeardown();
	CLFFT_RESULT_CHECK(clfftTeardown)
#endif // HAVE_CLFFT
}

static int ocl_fft_inited=FALSE;

define(`OCL_FFT_INIT',`
	if( ! ocl_fft_inited ){
		ocl_fft_inited = TRUE;
fprintf(stderr,"ocl_fft_init setting exit function\n");
		do_on_exit(ocl_fft_shutdown);
	}
')


dnl	 FP_PREC_SWITCH(sw_dp,func)
define(`FP_PREC_SWITCH',`

	switch( OBJ_MACH_PREC($1) ){
		case PREC_SP:
			HOST_TYPED_CALL_NAME($2,sp)(HOST_CALL_ARGS);
			break;
		case PREC_DP:
			HOST_TYPED_CALL_NAME($2,dp)(HOST_CALL_ARGS);
			break;
		default:
			fprintf(stderr,"Unexpected destination precision!?\n");
			break;
	}
')

dnl	FP_PREC_SWITCH_ISINV(sw_dp,func,isinv)
define(`FP_PREC_SWITCH_ISINV',`

	switch( OBJ_MACH_PREC($1) ){
		case PREC_SP:
			HOST_TYPED_CALL_NAME($2,sp)(HOST_CALL_ARGS,$3);
			break;
		case PREC_DP:
			HOST_TYPED_CALL_NAME($2,dp)(HOST_CALL_ARGS,$3);
			break;
		default:
			fprintf(stderr,"Unexpected destination precision!?\n");
			break;
	}
')

dnl	 RC_SWITCH(sw_dp,rfunc,cfunc,isinv)
define(`RC_SWITCH',`
	Vec_Obj_Args oa1, *oap=(&oa1);
fprintf(stderr,"%s BEGIN, oap = 0x%lx\n",STRINGIFY(HOST_CALL_NAME($2)),(u_long)oap);
	setvarg2(oap,_dst_dp,src_dp);
	SET_OA_PFDEV(oap,OBJ_PFDEV(OA_DEST(oap)));
	if( IS_REAL($1) ){
		FP_PREC_SWITCH(_dst_dp,`r'$2)
	} else {
		FP_PREC_SWITCH_ISINV(_dst_dp,`c'$3,$4)
	}
')

void HOST_CALL_NAME(fft2d)( VFCODE_ARG_DECL  Data_Obj *_dst_dp, Data_Obj *src_dp )
{
	OCL_FFT_INIT
	RC_SWITCH(src_dp,fft2d,fft2d,0)
}

void HOST_CALL_NAME(ift2d)( VFCODE_ARG_DECL  Data_Obj *_dst_dp, Data_Obj *src_dp )
{
	OCL_FFT_INIT
	RC_SWITCH(_dst_dp,ift2d,fft2d,1)
}

void HOST_CALL_NAME(fftrows)( VFCODE_ARG_DECL  Data_Obj *_dst_dp, Data_Obj *src_dp )
{
	OCL_FFT_INIT
	RC_SWITCH(src_dp,fftrows,fftrows,0)
}

void HOST_CALL_NAME(iftrows)( VFCODE_ARG_DECL  Data_Obj *_dst_dp, Data_Obj *src_dp )
{
	OCL_FFT_INIT
	RC_SWITCH(_dst_dp,iftrows,fftrows,0)
}

