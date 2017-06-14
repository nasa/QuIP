dnl	 This file is kind of the opposite of the type_tbl approach...
dnl	 here we have a bunch of switches and if/then's...

dnl	This file defines the untyped functions
dnl	BUT it doesn't seem to be included???

dnl	my_include(`veclib/gen_entries.m4')

ifdef(`BUILD_FOR_GPU',`

dnl	Need GPU definitions here!!!

',` dnl ifndef BUILD_FOR_GPU


dnl	REAL_FLOAT_SWITCH(func,prot_dp)
define(`REAL_FLOAT_SWITCH',`

switch( OBJ_MACH_PREC($2) ){
	case PREC_SP:  HOST_TYPED_CALL_NAME_REAL($1,sp)(HOST_CALL_ARGS); break;
	case PREC_DP:  HOST_TYPED_CALL_NAME_REAL($1,dp)(HOST_CALL_ARGS); break;
	default:
		sprintf(DEFAULT_ERROR_STRING,
	"real_float_switch (%s):  missing precision case (obj %s, prec %s)",
		"$1", OBJ_NAME($2), OBJ_MACH_PREC_NAME($2) );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

// FFT_SWITCH is used for complex fft functions, that have the inverse flag...
dnl	FFT_SWITCH( func, _is_inv )
define(`FFT_SWITCH',`

switch( OBJ_MACH_PREC(srcdp) ){
	case PREC_SP:
		HOST_TYPED_CALL_NAME_CPX($1,sp)(HOST_CALL_ARGS, $2);
		break;
	case PREC_DP:
		HOST_TYPED_CALL_NAME_CPX($1,dp)(HOST_CALL_ARGS, $2);
		break;
	default:	sprintf(DEFAULT_ERROR_STRING,
"fft_switch (%s):  object %s has bad machine precision %s",
"$1",OBJ_NAME(srcdp),OBJ_MACH_PREC_NAME(srcdp) );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

dnl	These are the un-typed calls...

void HOST_CALL_NAME(xform_list)(HOST_CALL_ARG_DECLS) { REAL_FLOAT_SWITCH( xform_list, OA_DEST(oap) ) }
void HOST_CALL_NAME(vec_xform)(HOST_CALL_ARG_DECLS) { REAL_FLOAT_SWITCH( vec_xform, OA_DEST(oap) ) }

void HOST_CALL_NAME(fft2d)(VFCODE_ARG_DECL  Data_Obj *dstdp,Data_Obj *srcdp)
{
	Vec_Obj_Args oa1, *oap=(&oa1);

	setvarg2(oap,dstdp,srcdp);
	SET_OA_PFDEV(oap,OBJ_PFDEV(dstdp));
	if( IS_COMPLEX( srcdp ) ){
		FFT_SWITCH( fft2d, FWD_FFT )
	} else {
		/* now we have two kinds of real fft... */
		/* FIXME */
		REAL_FLOAT_SWITCH(fft2d,srcdp)
		/*
		DSWITCH2( "fft2d", HOST_TYPED_CALL_NAME_REAL(fft2d,sp),
				HOST_TYPED_CALL_NAME_REAL(fft2d,dp) )
				*/
	}
}

void HOST_CALL_NAME(fftrows)(VFCODE_ARG_DECL  Data_Obj *dstdp,Data_Obj *srcdp)
{
	Vec_Obj_Args oa1, *oap=(&oa1);

	setvarg2(oap,dstdp,srcdp);
	SET_OA_PFDEV(oap,OBJ_PFDEV(dstdp));
	if( IS_COMPLEX( srcdp ) ){
		FFT_SWITCH( fftrows, FWD_FFT )
	} else {
		REAL_FLOAT_SWITCH(fftrows,srcdp)
		/*
		DSWITCH2( "fftrows", HOST_TYPED_CALL_NAME_REAL(fftrows,sp),
				HOST_TYPED_CALL_NAME_REAL(fftrows,dp) )
				*/
	}
}

void HOST_CALL_NAME(ift2d)(VFCODE_ARG_DECL  Data_Obj *dstdp,Data_Obj *srcdp)
{
	Vec_Obj_Args oa1, *oap=(&oa1);

	SET_OA_DEST(oap,dstdp);
	SET_OA_SRC1(oap,srcdp);
	SET_OA_SRC2(oap,NULL);
	SET_OA_SRC3(oap,NULL);
	SET_OA_SCLR1(oap,NULL);
	SET_OA_SCLR2(oap,NULL);

	if( IS_COMPLEX( dstdp ) ){
		FFT_SWITCH( fft2d, INV_FFT )
	} else {
		REAL_FLOAT_SWITCH(ift2d,srcdp)
		/*
		DSWITCH2( "ift2d", HOST_TYPED_CALL_NAME_REAL(ift2d,sp),
				HOST_TYPED_CALL_NAME_REAL(ift2d,dp) )
				*/
	}
}

void HOST_CALL_NAME(iftrows)(VFCODE_ARG_DECL  Data_Obj *dstdp,Data_Obj *srcdp)
{
	Vec_Obj_Args oa1, *oap=(&oa1);

	SET_OA_DEST(oap,dstdp);
	SET_OA_SRC1(oap,srcdp);
	SET_OA_SRC2(oap,NULL);
	SET_OA_SRC3(oap,NULL);
	SET_OA_SCLR1(oap,NULL);
	SET_OA_SCLR2(oap,NULL);

	if( IS_COMPLEX( dstdp ) ){
		FFT_SWITCH( iftrows, INV_FFT )
	} else {
		REAL_FLOAT_SWITCH(iftrows,srcdp)
		/*
		DSWITCH2( "iftrows", HOST_TYPED_CALL_NAME_REAL(iftrows,sp),
				HOST_TYPED_CALL_NAME_REAL(iftrows,dp) )
				*/
	}
}

') dnl endif // ! BUILD_FOR_GPU

