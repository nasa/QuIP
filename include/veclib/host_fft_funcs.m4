dnl	 This file is kind of the opposite of the type_tbl approach...
dnl	 here we have a bunch of switches and if/then's...

dnl	This file defines the untyped functions
dnl	BUT it doesn't seem to be included???

my_include(`veclib/gen_entries.m4')

ifdef(`BUILD_FOR_GPU',`

dnl	Need GPU definitions here!!!

',` dnl ifndef BUILD_FOR_GPU

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

