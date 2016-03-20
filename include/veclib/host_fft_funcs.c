// This file is kind of the opposite of the type_tbl approach...
// here we have a bunch of switches and if/then's...

#include "veclib/gen_entries.h"

#ifdef FOOBAR
void HOST_CALL_NAME(vramp2d)(HOST_CALL_ARG_DECLS )
{
	Data_Obj *dp;

	/* vramp2d isn't called like the tabled functions,
	 * so we have to do some checking here.
	 *
	 * Why ISN'T it called like the others???
	 */

	dp = OA_DEST(oap);
	if( OBJ_MACH_DIM( dp ,0) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"vramp2d:  target %s has type dimension %d, should be 1",
			OBJ_NAME( dp ),OBJ_MACH_DIM( dp ,0));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( OBJ_FRAMES( dp ) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"vramp2d:  target %s has nframes %d, should be 1",
			OBJ_NAME( dp ),OBJ_FRAMES( dp ));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( OBJ_SEQS( dp ) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"vramp2d:  target %s has nseqs %d, should be 1",
			OBJ_NAME( dp ),OBJ_SEQS( dp ));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	SIMPLE_PREC_SWITCH( vramp2d, OA_DEST(oap) ) }

void HOST_CALL_NAME(vramp1d)(HOST_CALL_ARG_DECLS )
{
	Data_Obj *dp;

	dp = OA_DEST(oap);

	if( OBJ_MACH_DIM( dp, 0 ) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"ramp1d:  target %s has type dimension %d, should be 1",
			OBJ_NAME( dp ),OBJ_MACH_DIM( dp ,0));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( OBJ_FRAMES( dp ) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"ramp1d:  target %s has nframes %d, should be 1",
			OBJ_NAME( dp ),OBJ_FRAMES( dp ));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( OBJ_SEQS( dp ) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"ramp1d:  target %s has nseqs %d, should be 1",
			OBJ_NAME( dp ),OBJ_SEQS( dp ));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	SIMPLE_PREC_SWITCH( vramp1d, OA_DEST(oap) ) }


RCQM_ALL_FUNC_SSE(vmul)

#ifndef BUILD_FOR_GPU
RC_ALL_FUNC( vrand )
#endif // BUILD_FOR_GPU

void HOST_CALL_NAME(convert)(HOST_CALL_ARG_DECLS)
{
	/*SET_OA_PFDEV( oap, OBJ_PFDEV(OA_DEST(oap)) );*/	// do this upstairs?
	switch( OBJ_MACH_PREC( OA_DEST(oap) ) ){
		case PREC_SP:  HOST_CALL_NAME(vconv2sp)(HOST_CALL_ARGS); break;
		case PREC_DP:  HOST_CALL_NAME(vconv2dp)(HOST_CALL_ARGS); break;
		case PREC_BY:  HOST_CALL_NAME(vconv2by)(HOST_CALL_ARGS); break;
		case PREC_IN:  HOST_CALL_NAME(vconv2in)(HOST_CALL_ARGS); break;
		case PREC_DI:  HOST_CALL_NAME(vconv2di)(HOST_CALL_ARGS); break;
		case PREC_LI:  HOST_CALL_NAME(vconv2li)(HOST_CALL_ARGS); break;
		case PREC_UBY:  HOST_CALL_NAME(vconv2uby)(HOST_CALL_ARGS); break;
		case PREC_UIN:  HOST_CALL_NAME(vconv2uin)(HOST_CALL_ARGS); break;
		case PREC_UDI:  HOST_CALL_NAME(vconv2udi)(HOST_CALL_ARGS); break;
		case PREC_ULI:  HOST_CALL_NAME(vconv2uli)(HOST_CALL_ARGS); break;
		default:  sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  CALL_CONVERSION:  Bad destination precision!?");
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}
}

#endif // FOOBAR

#ifndef BUILD_FOR_GPU
void HOST_CALL_NAME(xform_list)(HOST_CALL_ARG_DECLS) { REAL_FLOAT_SWITCH( xform_list, OA_DEST(oap) ) }

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
	SET_OA_SRC2(oap,NO_OBJ);
	SET_OA_SRC3(oap,NO_OBJ);
	SET_OA_SCLR1(oap,NO_OBJ);
	SET_OA_SCLR2(oap,NO_OBJ);

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
	SET_OA_SRC2(oap,NO_OBJ);
	SET_OA_SRC3(oap,NO_OBJ);
	SET_OA_SCLR1(oap,NO_OBJ);
	SET_OA_SCLR2(oap,NO_OBJ);

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

#endif // ! BUILD_FOR_GPU

