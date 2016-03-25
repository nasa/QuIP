// This file is kind of the opposite of the type_tbl approach...
// here we have a bunch of switches and if/then's...

#include "veclib/gen_entries.h"

NORM_DECL(vconv2by)
NORM_DECL(vconv2in)
NORM_DECL(vconv2di)
NORM_DECL(vconv2li)
NORM_DECL(vconv2uby)
NORM_DECL(vconv2uin)
NORM_DECL(vconv2udi)
NORM_DECL(vconv2uli)
NORM_DECL(vconv2sp)
NORM_DECL(vconv2dp)

NORM_DECL(vsm_lt)
NORM_DECL(vsm_gt)
NORM_DECL(vsm_le)
NORM_DECL(vsm_ge)
NORM_DECL(vsm_ne)
NORM_DECL(vsm_eq)

NORM_DECL(vvm_lt)
NORM_DECL(vvm_gt)
NORM_DECL(vvm_le)
NORM_DECL(vvm_ge)
NORM_DECL(vvm_ne)
NORM_DECL(vvm_eq)

NORM_DECL( vv_vv_lt )
NORM_DECL( vv_vv_gt )
NORM_DECL( vv_vv_le )
NORM_DECL( vv_vv_ge )
NORM_DECL( vv_vv_eq )
NORM_DECL( vv_vv_ne )

NORM_DECL( vv_vs_lt )
NORM_DECL( vv_vs_gt )
NORM_DECL( vv_vs_le )
NORM_DECL( vv_vs_ge )
NORM_DECL( vv_vs_eq )
NORM_DECL( vv_vs_ne )

NORM_DECL( vs_vv_lt )
NORM_DECL( vs_vv_gt )
NORM_DECL( vs_vv_le )
NORM_DECL( vs_vv_ge )
NORM_DECL( vs_vv_eq )
NORM_DECL( vs_vv_ne )

NORM_DECL( vs_vs_lt )
NORM_DECL( vs_vs_gt )
NORM_DECL( vs_vs_le )
NORM_DECL( vs_vs_ge )
NORM_DECL( vs_vs_eq )
NORM_DECL( vs_vs_ne )

NORM_DECL( ss_vv_lt )
NORM_DECL( ss_vv_gt )
NORM_DECL( ss_vv_le )
NORM_DECL( ss_vv_ge )
NORM_DECL( ss_vv_eq )
NORM_DECL( ss_vv_ne )

NORM_DECL( ss_vs_lt )
NORM_DECL( ss_vs_gt )
NORM_DECL( ss_vs_le )
NORM_DECL( ss_vs_ge )
NORM_DECL( ss_vs_eq )
NORM_DECL( ss_vs_ne )

NORM_DECL(vclip)
NORM_DECL(viclp)
FLOAT_DECL(vscml)

NORM_DECL( vsign )
NORM_DECL( vabs )
FLOAT_DECL( vconj )
NORM_DECL( vmin  )
NORM_DECL( vmax  )
NORM_DECL( vming )
NORM_DECL( vmaxg )
NORM_DECL( vibnd )
NORM_DECL( vcmp  )
NORM_DECL( vsmax )
NORM_DECL( vscmp )
NORM_DECL( vscmp2 )
NORM_DECL( vsmin )
NORM_DECL( vminv )
NORM_DECL( vmaxv )
NORM_DECL( vmini )
NORM_DECL( vmaxi )
NORM_DECL( vmnmi )
NORM_DECL( vmxmi )

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

RCQ_ALL_FUNC( vsum )
RC_ALL_FUNC( vdot )
#ifndef BUILD_FOR_GPU
RC_ALL_FUNC( vrand )
#endif // BUILD_FOR_GPU

RCQ_ALL_FUNC( vvv_slct )
RCQ_ALL_FUNC( vss_slct )
RCQ_ALL_FUNC( vvs_slct )

RCQB_ALL_FUNC( vset )		/* sse version! */
RCQ_ALL_FUNC( vsqr )		/* sse version! */
//RCQ_SIGNED_FUNC( vneg )		/* sse version! */
RCQ_ALL_FUNC( vneg )		/* sse version! */

RCQM_ALL_FUNC_SSE( vadd )		/* sse version! */
RCQM_ALL_FUNC_SSE( vsub )		/* sse version! */
RCQM_ALL_FUNC_SSE( vdiv )		/* sse version! */

RCQM_ALL_FUNC_SSE( vsadd )	/* sse version! */
RCQM_ALL_FUNC_SSE( vsmul )	/* sse version! */
//RCQM_ALL_FUNC_SSE( vsmul2 )	/* only here because quaternions are non-commutative... */
RCQM_ALL_FUNC_SSE( vssub )	/* sse version! */
RCQM_ALL_FUNC_SSE( vsdiv )	/* sse version! */
/* C_ALL_FUNC_SSE( vcsmul ) */	/* sse version! */
/* Q_ALL_FUNC_SSE( vqsmul ) */	/* sse version! */

RCQM_ALL_FUNC( vsdiv2 )	/* sse version! */

#ifdef OLD_SSE_IMPLEMENTATION
// Worry about SSE integration later...
// We should be able to use the C compiler to get SSE acceleration...
void vmov(HOST_CALL_ARG_DECLS )
{
#ifdef USE_SSE
	if( oap->oa_argsprec == SP_ARGS ){
		int n_per_128;

		n_per_128 = 16 / PREC_SIZE( OBJ_MACH_PREC_PTR( OA_DEST(oap) )  );
		if(   OA_ARGSTYPE(oap)   == COMPLEX_ARGS )
			n_per_128 >>= 1;


		if( use_sse_extensions && N_IS_CONTIGUOUS( OA_DEST(oap) ) &&
				N_IS_CONTIGUOUS(OA_SRC1(oap)) &&
			( OBJ_N_MACH_ELTS(OA_DEST(oap) ) % n_per_128)==0 ){

			if( (((u_long)  OBJ_DATA_PTR(OA_DEST(oap) )) & 15) ||
			    (((u_long) OBJ_DATA_PTR(OA_SRC1(oap))) & 15) ){
		NWARN("data vectors must be aligned on 16 byte boundary for SSE acceleration");
			} else {
				simd_vec_rvmov( OBJ_DATA_PTR(OA_DEST(oap) ),OBJ_DATA_PTR(OA_SRC1(oap)), OBJ_N_MACH_ELTS(OA_DEST(oap) )/n_per_128);
				return;
			}
		}
	}

#endif /* USE_SSE */

	/* can't (or don't want to) use SSE - just call normal function */
	RCQ_SWITCH(vmov);
} /* end vmov */
#endif // OLD_SSE_IMPLEMENTATION

RCQ_ALL_FUNC_SSE(vmov)

/* SWITCH2
 * just float or double...
 */

FLT_FUNC( vrint )
FLT_FUNC( vfloor )
FLT_FUNC( vtrunc )
FLT_FUNC( vround )
FLT_FUNC( vceil  )

NORM_DECL( vminm  )
NORM_DECL( vmaxm  )
NORM_DECL( vmnmv  )
NORM_DECL( vmxmv  )
NORM_DECL( vsmxm  )
NORM_DECL( vsmnm  )
NORM_DECL( vmxmg  )
NORM_DECL( vmnmg  )
NORM_DECL( vbnd   )

FLT_FUNC( vasin  )
FLT_FUNC( vacos  )
FLT_FUNC( vsqrt  )
FLT_FUNC( vcos   )
FLT_FUNC( verf   )
FLT_FUNC( verfinv   )
FLT_FUNC( vsin   )
FLT_FUNC( vtan   )
FLT_FUNC( vatan  )
FLT_FUNC( vatan2 )
FLT_FUNC( vatn2  )
FLT_FUNC( rvexp   )
FLT_FUNC( vlog   )
FLT_FUNC( rvpow   )
FLT_FUNC( vlog10   )
FLT_FUNC( vsatan2 )
FLT_FUNC( vsatan22 )
FLT_FUNC( vspow )
FLT_FUNC( vspow2 )
FLT_FUNC( vcmul )
FLT_FUNC( vmgsq )

//#ifndef BUILD_FOR_GPU
FLT_FUNC( vuni   )
#ifndef BUILD_FOR_GPU
FLT_FUNC( vj0    )
FLT_FUNC( vj1    )
FLT_FUNC( visnan   )
FLT_FUNC( visinf   )
FLT_FUNC( visnorm   )
#endif // ! BUILD_FOR_GPU


void HOST_CALL_NAME(vpow)(HOST_CALL_ARG_DECLS) { RC_FLOAT_SWITCH( vpow ) }
void HOST_CALL_NAME(vexp)(HOST_CALL_ARG_DECLS) { RC_FLOAT_SWITCH( vexp ) }

// BUG - how do we do this?  CUDA has a built-in library, what about OpenCL?
void HOST_CALL_NAME(vfft)(HOST_CALL_ARG_DECLS) { RC_FLOAT_SWITCH( vfft ) }
void HOST_CALL_NAME(vift)(HOST_CALL_ARG_DECLS) { RC_FLOAT_SWITCH( vift ) }

//void HOST_CALL_NAME(xform_list)(HOST_CALL_ARG_DECLS) { RC_FLOAT_SWITCH( xform_list ) }

#ifndef BUILD_FOR_GPU
void HOST_CALL_NAME(xform_list)(HOST_CALL_ARG_DECLS) { REAL_FLOAT_SWITCH( xform_list, OA_DEST(oap) ) }
#endif // ! BUILD_FOR_GPU

INT_FUNC( vand   )
INT_FUNC( vnand  )
INT_FUNC( vor    )
INT_FUNC( vxor   )
INT_FUNC( vnot   )
INT_FUNC( vsand  )
INT_FUNC( vshr  )
INT_FUNC( vcomp  )
INT_FUNC( vsshr  )
INT_FUNC( vsshr2  )
INT_FUNC( vshl  )
INT_FUNC( vsshl  )
INT_FUNC( vsshl2  )
INT_FUNC( vsor   )
INT_FUNC( vsxor  )
/* INT_FUNC( vsnand ) */
INT_FUNC( vmod   )
INT_FUNC( vsmod  )
INT_FUNC( vsmod2 )

INT_FUNC( vtolower )
INT_FUNC( vtoupper )
INT_FUNC( vislower )
INT_FUNC( visupper )
INT_FUNC( visdigit )
INT_FUNC( visalpha )
INT_FUNC( visalnum )
INT_FUNC( visspace )
INT_FUNC( visblank )
INT_FUNC( viscntrl )


#ifndef BUILD_FOR_GPU
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

