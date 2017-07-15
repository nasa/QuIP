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

switch( OBJ_MACH_PREC(OA_SRC1(oap)) ){
	case PREC_SP:
		HOST_TYPED_CALL_NAME_CPX($1,sp)(HOST_CALL_ARGS, $2);
		break;
	case PREC_DP:
		HOST_TYPED_CALL_NAME_CPX($1,dp)(HOST_CALL_ARGS, $2);
		break;
	default:	sprintf(DEFAULT_ERROR_STRING,
"fft_switch (%s):  object %s has bad machine precision %s",
"$1",OBJ_NAME(OA_SRC1(oap)),OBJ_MACH_PREC_NAME(OA_SRC1(oap)) );
		NWARN(DEFAULT_ERROR_STRING);
		break;
}
')

dnl	These are the un-typed calls...
dnl THESE NEXT TWO ARE NOT FFT FUNCS!?!?

void HOST_CALL_NAME(xform_list)(HOST_CALL_ARG_DECLS) { REAL_FLOAT_SWITCH( xform_list, OA_DEST(oap) ) }
void HOST_CALL_NAME(vec_xform)(HOST_CALL_ARG_DECLS) { REAL_FLOAT_SWITCH( vec_xform, OA_DEST(oap) ) }

ifdef(`FOOBAR',`

void HOST_CALL_NAME(fft2d)( HOST_CALL_ARG_DECLS )
{
	//Vec_Obj_Args oa1, *oap=(&oa1);

	setvarg2(oap,OA_DEST(oap),OA_SRC1(oap));
	SET_OA_PFDEV(oap,OBJ_PFDEV(OA_DEST(oap)));
	if( IS_COMPLEX( OA_SRC1(oap) ) ){
		FFT_SWITCH( fft2d, FWD_FFT )
	} else {
		/* now we have two kinds of real fft, based on shape of the transform... */
		/* FIXME */
		REAL_FLOAT_SWITCH(fft2d,OA_SRC1(oap))
	}
}

void HOST_CALL_NAME(fftrows)( HOST_CALL_ARG_DECLS )
{
	//Vec_Obj_Args oa1, *oap=(&oa1);

	setvarg2(oap,OA_DEST(oap),OA_SRC1(oap));
	SET_OA_PFDEV(oap,OBJ_PFDEV(OA_DEST(oap)));
	if( IS_COMPLEX( OA_SRC1(oap) ) ){
		FFT_SWITCH( fftrows, FWD_FFT )
	} else {
		REAL_FLOAT_SWITCH(fftrows,OA_SRC1(oap))
	}
}

void HOST_CALL_NAME(ift2d)( HOST_CALL_ARG_DECLS )
{
	if( IS_COMPLEX( OA_DEST(oap) ) ){
		// For complex, inverse is passed in flag
		FFT_SWITCH( fft2d, INV_FFT )
	} else {
		REAL_FLOAT_SWITCH(ift2d,OA_SRC1(oap))
	}
}

void HOST_CALL_NAME(iftrows)( HOST_CALL_ARG_DECLS )
{
	/*
	Vec_Obj_Args oa1, *oap=(&oa1);

	SET_OA_DEST(oap,dstdp);
	SET_OA_SRC1(oap,srcdp);
	SET_OA_SRC2(oap,NULL);
	SET_OA_SRC3(oap,NULL);
	SET_OA_SCLR1(oap,NULL);
	SET_OA_SCLR2(oap,NULL);
	*/

	if( IS_COMPLEX( OA_DEST(oap) ) ){
		FFT_SWITCH( iftrows, INV_FFT )
	} else {
		REAL_FLOAT_SWITCH(iftrows,OA_SRC1(oap))
	}
}

',`')	dnl end ifdef FOOBAR

') dnl endif // ! BUILD_FOR_GPU

