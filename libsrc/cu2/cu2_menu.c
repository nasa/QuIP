
#include "quip_config.h"


// includes, system
#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#ifdef HAVE_CUDA
#include "my_cu2.h"

#include "quip_prot.h"

#include "veclib/cu2_menu_prot.h"
//#include "veclib/menu_decls.c"

#ifdef FOOBAR
#define ADD_CMD(s,f,h)	ADD_COMMAND(unary_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN(unary)
ADD_CMD( set,		vset,	set to a constant )
#ifdef NOT_YET
ADD_CMD( convert,	convert,	convert to from one type to another )
ADD_CMD( abs, 		vabs, 	convert to absolute value )
ADD_CMD( mov, 		rvmov,	copy data )
ADD_CMD( neg, 		rvneg,	convert to negative )
ADD_CMD( sign, 		vsign,	take sign of vector )
ADD_CMD( round,		vround,
				round vector elements to nearest integer using round() )
ADD_CMD( floor, 	vfloor,	take floor of vector )
ADD_CMD( ceil, 		vceil,	take ceiling of vector )
//ADD_CMD( conj, 	,		 "convert to complex conjugate )
//ADD_CMD( find, 	, 		return indeces of non-zero elements )
//ADD_CMD( dimsum, 	, 		return sum along comumns (rows) )
//ADD_CMD( sum, 		vsum,	get sum of vector )
//ADD_CMD( flip, 	,		copy a vector, reversing the order of the elements )
//ADD_CMD( flipall, 	,		copy an object, reversing order of all dimensions )
//ADD_CMD( vrint, 	vrint,	round vector elements to nearest integer using rint() )
//ADD_CMD( uni, 		,		"uniform random numbers )
#endif // NOT_YET
MENU_END(unary)

static PF_COMMAND_FUNC( unary )
{
	PUSH_MENU(unary);
}

#ifdef NOT_YET
//OpenCL TRIG COMMAND TABLE - ALL KNOWN FUNCTIONS ACCOUNTED FOR
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(trig_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN(trig)
ADD_CMD( atan, 		vatan, 	compute arc tangent )
ADD_CMD( atan2, 	vatan2,	compute arc tangent (two real args) )
ADD_CMD( cos,	 	vcos, 	compute cosine )
ADD_CMD( erf, 		verf, 	compute error function (erf) )
ADD_CMD( exp, 		rvexp, 	exponentiate (base e) )
ADD_CMD( log, 		vlog, 	natural logarithm )
ADD_CMD( log10, 	vlog10, 	logarithm base 10 )
ADD_CMD( sin,		vsin, 	sompute sine )
ADD_CMD( square, 	rvsqr, 	compute square )
ADD_CMD( sqrt, 		vsqrt, 	computer square root )
ADD_CMD( tan, 		vtan, 	compute tangent )
ADD_CMD( pow, 		rvpow, 	raise to a power )
ADD_CMD( acos, 		vacos, 	compute inverse cosine )
ADD_CMD( asin, 		vasin, 	compute inverse sine )
//ADD_CMD( magsq, 	,		convert to magnitude squared )
//ADD_CMD( atn2, 	,		compute arc tangent (single xomplex arg) )
//ADD_CMD( j0, 		vj0, 	compute bessel function j0 )
//ADD_CMD( j1, 		vj1, 	compute bessel function j1 )
MENU_END(trig)

static PF_COMMAND_FUNC( trig )
{
	PUSH_MENU(trig);
}

//OpenCL LOGICAL COMMAND TABLE - ALL KNOWN FUNCTIONS ACCOUNTED FOR

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(logical_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN(logical)
ADD_CMD( and, 	vand, 	logical AND )
ADD_CMD( nand, 	vnand, 	logical NAND )
ADD_CMD( not, 	vnot, 	logical NOT )
ADD_CMD( or, 	vor, 	logical OR )
ADD_CMD( xor, 	vxor, 	logical XOR )
ADD_CMD( sand, 	vsand, 	logical AND with scalar )
ADD_CMD( sor, 	vsor, 	logical OR with scalar )
ADD_CMD( sxor, 	vsxor, 	logical XOR with scalar )
ADD_CMD( shr, 	vshr, 	right shift )
ADD_CMD( shl, 	vshl, 	left shift )
ADD_CMD( sshr, 	vsshr, 	right shift by a constant )
ADD_CMD( sshl, 	vsshl, 	left shift by a constant )
ADD_CMD( sshr2,	vsshr2, 	right shift a constant )
ADD_CMD( sshl2,	vsshl2, 	left shift a constant )
//ADD_CMD( cmp, 		, 		bitwise complement )
MENU_END(logical)

static PF_COMMAND_FUNC( logic )
{
	PUSH_MENU(logical);
}

#endif // NOT_YET

//OpenCL VVECTOR COMMAND TABLE - ALL KNOWN FUNCTIONS ACCOUNTED FOR
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(vvector_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN(vvector)
ADD_CMD( add, 	vadd, 	vector addition )
#ifdef NOT_YET
ADD_CMD( div, 	rvdiv, 	element by element division )
ADD_CMD( mul, 	rvmul, 	element by element multiplication )
ADD_CMD( sub, 	rvsub, 	vector subtraction )
//	cmul, , 		multiply by complex conjugation )
#endif // NOT_YET
MENU_END(vvector)

static PF_COMMAND_FUNC( vv )
{
	PUSH_MENU(vvector);
}

#ifdef NOT_YET
//OpenCL RSV COMMAND TABLE - ALL KNOWN FUNCTIONS ACCOUNTED FOR
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(svector_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN(svector)
ADD_CMD( add, 	rvsadd, 	add scalar to elements of a vector )
ADD_CMD( sub, 	rvssub, 	subtract elements of a vecotr from a scalar )
ADD_CMD( div, 	rvsdiv, 	divide a scalar by the elements of a vector )
ADD_CMD( div2, 	rvsdiv2,	divide elements of a vector by a scalar )
ADD_CMD( mul, 	rvsmul, 	multiply a vector by a real scalar )
ADD_CMD( mod, 	vsmod, 	integer modulo of a vector by a real scalar )
ADD_CMD( mod2,	vsmod2, 	integer modulo of a real scalar by a vector )
ADD_CMD( pow, 	vspow, 	raise the elements of a vector to a scalar power )
ADD_CMD( pow2, 	vspow2, 	raise a scalar to powers given by the elements of a vector )
ADD_CMD( atan2, 	vsatan2,	compute 4-quadrant arc tangent of vector and scalar )
ADD_CMD( atan22,	vsatan22,	compute 4-quadrant arc tangent of a scalar and vector )
//ADD_CMD( and, 	,		bitwise and of scalar and vector )
//ADD_CMD( or, 	,		bitwise or of scalar and vector )
//ADD_CMD( xor, 	,		bitwise xor of scalar and vector )
MENU_END(svector)

static PF_COMMAND_FUNC( rvs )
{
	PUSH_MENU(svector);
}

//OpenCL CSV COMMAND TABLE
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(csvector_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN(csvector)
//ADD_CMD( add, 	, 		add complex scalar to elements of a vector )
//ADD_CMD( mul, , 		multiply a vector by a complex scalar )
//ADD_CMD( sub, , 		subtract elements of a vector from a complex scalar )
//ADD_CMD( sub2, , 		subtract elements of a vector from a complex scalar )
//ADD_CMD( div, , 		divide a complex scalar by the elements of a vector )
//ADD_CMD( div2, , 		divide elements of a vector by a complex scalar )
//ADD_CMD( conjmul, ,	 	multiply vector conjugation by a complex scalar )
MENU_END(csvector)

static PF_COMMAND_FUNC( cvs )
{
	PUSH_MENU(csvector);
}

//OpenCL QSV COMMAND TABLE
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(qsvector_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN(qsvector)
//ADD_CMD( add, 	, 		add quaternion scalar to elements of a vector )
//ADD_CMD( mul, , 		multiply a vector by a quaternion scalar )
//ADD_CMD( sub, , 		subtract elements of a vector from a quaternion scalar )
//ADD_CMD( sub2, , 		subtract elements of a vector from a quaternion scalar )
//ADD_CMD( div, , 		divide a quaternion scalar by the elements of a vector )
//ADD_CMD( div2, , 		divide elements of a vector by a quaternion scalar )
MENU_END(qsvector)

static PF_COMMAND_FUNC( qvs )
{
	PUSH_MENU(qsvector);
}

//OpenCL MINMAX COMMAND TABLE - ALL KNOWN FUNCTIONS ACCOUNTED FOR
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(minmax_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN(minmax)
ADD_CMD( max, 			vmax,	find maximum value )
ADD_CMD( min, 			vmin,	find minimum value )
ADD_CMD( max_mag, 		vmaxm,	find maximum absolute value )
ADD_CMD( min_mag, 		vminm,	find minimum absolute value )
//ADD_CMD( max_index,		,		find index of maximum value )
//ADD_CMD( min_index,		,		find index of minimum value )
//ADD_CMD( max_mag_index, 	, 		find index of maximum absolute value )
//ADD_CMD( min_mag_index, 	, 		find index of minimum absolute value )
//ADD_CMD( max_times, 		,		find index of maximum & # of occurrences )
//ADD_CMD( min_times, 		, 		find index of minimum & # of occurrences )
//ADD_CMD( max_mag_times, 	, 		find index of max. mag. & # of occurrences )
//ADD_CMD( min_mag_times, 	, 		find index of min. mag. & # of occurrences )
MENU_END(minmax)

static PF_COMMAND_FUNC( minmax )
{
	PUSH_MENU(minmax);
}

//OpenCL CMP COMMAND TABLE
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(compare_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN(compare)
ADD_CMD( max, 		vmax, 	take the max of two vectors )
ADD_CMD( min, 		vmin, 	take the min of two vectors )
ADD_CMD( max_mag, 	vmaxm, 	take the max mag of two vectors )
ADD_CMD( min_mag, 	vminm,	take the min mag of two vectors )
ADD_CMD( clip, 	vclip, 	clip elements of a vector )
ADD_CMD( iclip, 	viclp, 	inverted clip )
ADD_CMD( vscmp, 	vscmp, 	vector-scalar comparison (>=) )
ADD_CMD( vscmp2, 	vscmp2, 	vector-scalar comparison (<=) )
ADD_CMD( bound, 	vbnd, 	bound elements of a vector )
ADD_CMD( ibound, 	vibnd, 	inverted bound )
ADD_CMD( cmp, 		vcmp, 	vector-vector comparison )
ADD_CMD( vsmax, 	vsmax, 	scalar-vector maximum )
ADD_CMD( vsmxm, 	vsmxm, 	scalar-vector maximum magnitude )
ADD_CMD( vsmin, 	vsmin, 	scalar-vector minimum )
ADD_CMD( vsmnm, 	vsmnm, 	scalar-vector minimum magnitude )
//ADD_CMD( vmscm, 	, 		bit-map scalar-vector mag. comparison )
//ADD_CMD( vmscp, 	, 		bit-map scalar-vector comparison )
//ADD_CMD( vcmpm, 	, 		vector-vector magnitude comparison )
//ADD_CMD( vscmm, 	, 		scalar-vector magnitude comparison )
//ADD_CMD( vvm_lt, 	, 		bit-map vector comparison (<) )
//ADD_CMD( vvm_gt, 	, 		bit-map vector comparison (>) )
//ADD_CMD( vvm_le, 	, 		bit-map vector comparison (<=) )
//ADD_CMD( vvm_ge, 	, 		bit-map vector comparison (>=) )
//ADD_CMD( vvm_ne, 	,		bit-map vector comparison (!=) )
//ADD_CMD( vvm_eq, 	, 		bit-map vector comparison (==) )
//ADD_CMD( vsm_lt, 	, 		bit-map vector/scalar comparison (<) )
//ADD_CMD( vsm_gt, 	, 		bit-map vector/scalar comparison (>) )
//ADD_CMD( vsm_ge, 	, 		bit-map vector/scalar comparison (<=) )
//ADD_CMD( vsm_ne, 	, 		bit-map vector/scalar comparison (>=) )
//ADD_CMD( vsm_eq, 	, 		bit-map vector/scalar comparison (==) )
//ADD_CMD( vmcmp, 	,		bit-map vector comarison )
//ADD_CMD( select, 	,		vector/vector selection based on bit-map )
//ADD_CMD( vv_select, 	, 		vector/vector selection based on bit-map )
//ADD_CMD( vs_select, 	, 		vector/scalar selection based on bit-map )
//ADD_CMD( ss_select, 	, 		scalar/scalar selection based on bit-map )
MENU_END(compare)

static PF_COMMAND_FUNC( docmp )
{
	PUSH_MENU(compare);
}

//CUFFT COMMAND TABLE
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(fft_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN( fft )
ADD_CMD( fft, 		fwdfft, 		forward complex Fourier Transform )
//ADD_CMD( newfft, 	newfft,		test new chainable complex fft )
/*
ADD_CMD( row_fft, 	fwdrowfft,	forward complex Fourier Transform of rows only  )
ADD_CMD( rfft, 		fwdrfft,		forward Fourier Transform, real input )
ADD_CMD( row_rfft,	fwdrowrfft,	forward Fourier transform of rows only, real input )
ADD_CMD( irfft, 	invrfft,		inverse Fourier Transform, real output )
ADD_CMD( row_irfft, 	invrowrfft,	inverse Fourier Transform of rows only, real output )
ADD_CMD( invfft, 	invfft,		inverse complex Fourier Transform )
ADD_CMD( row_invfft, 	invrowfft,	inverse complex Fourier Transform of rows only )
ADD_CMD( radavg, 	radavg,		compute radial average )
ADD_CMD( oriavg, 	oriavg,		compute orientation average )
ADD_CMD( wrap, 		wrap,		wrap DFT iamge )
ADD_CMD( wrap3d, 	wrap3d,		wrap 3-D DFT )
ADD_CMD( scroll, 	scroll,		scroll image )
ADD_CMD( dct, 		dct,			compute blocked discrete cosine xform )
ADD_CMD( odct, 		odct,		compute DCT using old method )
ADD_CMD( idct, 		idct,		compute inverse descrete cosine xform )
*/
MENU_END( fft )


static PF_COMMAND_FUNC( fft )
{
	PUSH_MENU(fft);
}

// NOTE:  NPP is cuda-only... - but this is 
//COMMAND TABLE FOR Cuda NPP LIBRARY

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(npp_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN( npp )
ADD_CMD( image,		npp_malloc,		declare new image )
ADD_CMD( vadd,		npp_vadd,		add two images )
ADD_CMD( erode,		npp_erosion,		erosion )
ADD_CMD( dilate,	npp_dilation,	dilation )
ADD_CMD( filter,	npp_filter,		space-domain filtering )
ADD_CMD( sum,		npp_sum,		compute sum )
ADD_CMD( sum_scratch,	npp_sum_scratch,	allocate scratch space for sum )
ADD_CMD( i_vmul,	nppi_vmul,		image/image multiplication )
ADD_CMD( s_vmul,	npps_vmul,		in-place signel multiplication )
ADD_CMD( version,	report_npp_version,	report NPP library version )
MENU_END( npp )

static PF_COMMAND_FUNC( npp_menu )
{
	PUSH_MENU(npp);
}


static PF_COMMAND_FUNC( fill )
{
	Data_Obj *dp;
	int x,y;
	float fill_val, tol;

	dp = PICK_OBJ("image");
	x=HOW_MANY("seed point x");
	y=HOW_MANY("seed point y");
	fill_val=HOW_MUCH("fill value");
	tol = HOW_MUCH("tolerance");

	h_sp_ifl(dp,x,y,tol,fill_val);
}

static PF_COMMAND_FUNC( fill2 )
{
	Data_Obj *dp;
	int x,y;
	float fill_val, tol;

	dp = PICK_OBJ("image");
	x=HOW_MANY("seed point x");
	y=HOW_MANY("seed point y");
	fill_val=HOW_MUCH("fill value");
	tol = HOW_MUCH("tolerance");

	h_sp_ifl2(dp,x,y,tol,fill_val);
}

static PF_COMMAND_FUNC( yuv2rgb )
{
	Data_Obj *rgb_dp;
	Data_Obj *yuv_dp;

	rgb_dp = PICK_OBJ("RGB image");
	yuv_dp = PICK_OBJ("YUV image");

	//  BUG do all checks:
	// pixel types, mating sizes
	cu2_yuv422_to_rgb24(rgb_dp,yuv_dp);
}

static PF_COMMAND_FUNC( centroid )
{
	Data_Obj *dst1_dp;
	Data_Obj *dst2_dp;
	Data_Obj *src_dp;
	Vec_Obj_Args oargs;

	dst1_dp = PICK_OBJ("x scratch image");
	dst2_dp = PICK_OBJ("y scratch image");
	src_dp = PICK_OBJ("source image");

	if( OBJ_MACH_PREC(src_dp) != PREC_SP && OBJ_MACH_PREC(src_dp) != PREC_DP ){
		sprintf(ERROR_STRING,"Object %s (%s) must have %s or %s precision for centroid helper",
			OBJ_NAME(src_dp),PREC_NAME(OBJ_PREC_PTR(src_dp)),PREC_NAME(PREC_FOR_CODE(PREC_SP)),
			PREC_NAME(PREC_FOR_CODE(PREC_DP)));
		WARN(ERROR_STRING);
		return;
	}

	// BUG - do more checking here sizes must match, precisions must match.

	setvarg3(&oargs,dst1_dp,dst2_dp,src_dp);	/* abusing this a little */

	if( OBJ_PREC(src_dp) == PREC_SP )
		sp_cu2_centroid(&oargs);
	else if( OBJ_PREC(src_dp) == PREC_DP )
		dp_cu2_centroid(&oargs);
#ifdef CAUTIOUS
	else ERROR1("CAUTIOUS:  centroid:  unexpected source precision!?");
#endif /* CAUTIOUS */
}

//COMMAND TABLE FOR OpenCL NPP LIBRARY
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(misc_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN( misc )
ADD_CMD( fill,		fill,		flood fill )
ADD_CMD( fill2,		fill2,		flood fill version 2 )
ADD_CMD( yuv2rgb,	yuv2rgb,	YUV to RGB conversion )
ADD_CMD( centroid,	centroid,	centroid helper function )
MENU_END( misc )

static PF_COMMAND_FUNC( misc )
{
	PUSH_MENU(misc);
}

#endif // NOT_YET

//OpenCL WAR COMMAND TABLE
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(compute_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN( compute )
ADD_CMD( unary,		unary,	unary operations on data )
ADD_CMD( vvector,	vv,		vector-vector operations )
#ifdef NOT_YET
ADD_CMD( trig,		trig,	trigonometric operations )
ADD_CMD( logical,	logic,	logical operations on data )
ADD_CMD( svector,	rvs,		real scalar-vector operations )
ADD_CMD( csvector,	cvs,		complex scalar-vector operations )
ADD_CMD( Qsvector,	qvs,		quaternion scalar-vector operations )
ADD_CMD( minmax,	minmax,	minimum/maximum routines )
ADD_CMD( compare,	comp,		comparison routines )
ADD_CMD( fft,		fft, 	fft )
ADD_CMD( misc,		misc,	miscellaneous platform functions )
#endif // NOT_YET
MENU_END( compute )

static PF_COMMAND_FUNC(func_menu)
{
	/* Do platform-specific init here? */
	PUSH_MENU(compute);
}

#ifdef HAVE_OPENGL
#ifdef HAVE_GLUT

//OpenCL GL COMMAND TABLE
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(pf_gl_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN( pf_gl )
ADD_CMD( buffer,	new_gl_buffer,	create a named GL buffer )
#ifdef FOOBAR
ADD_CMD( test,		gl_test,		gl test function )
ADD_CMD( display,	gl_disp,		update display window )
#endif /* FOOBAR */
ADD_CMD( viewer,	new_vwr,	create a new image viewer )
ADD_CMD( load,		load_vwr,	write an image to a viewer )
MENU_END( pf_gl )

static PF_COMMAND_FUNC(pf_gl_menu)
{
	/* Do platform-specific init here? */
	PUSH_MENU(pf_gl);
}

#endif /* HAVE_GLUT */
#endif /* HAVE_OPENGL */


//OpenCL EVENT CHECKPOINTING COMMAND TABLE
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(event_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN( event )
ADD_CMD( max_checkpoints,	init_ckpts,	set maximum number of checkpoints )
ADD_CMD( set_checkpoint,	set_ckpt,	set a checkpoint )
ADD_CMD( reset,			clear_ckpts,	clear all checkpoints )
ADD_CMD( show,			show_ckpts,	show checkpoint times )
MENU_END( event )

static PF_COMMAND_FUNC( event_menu )
{
	PUSH_MENU(event);
}


//OpenCL EVENT CHECKPOINTING COMMAND TABLE
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(stream_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN( stream )
ADD_CMD( stream,	new_stream,	create a new stream )
ADD_CMD( list,		list_streams,	list all streams )
ADD_CMD( info,		stream_info,	report information about a stream )
ADD_CMD( sync,		sync_stream,	synchronize host execution with a stream )
MENU_END( stream )

static PF_COMMAND_FUNC( stream_menu )
{
	PUSH_MENU(stream);
}

static PF_COMMAND_FUNC( prt_cap )
{
#ifdef HAVE_OpenCL
#ifdef OpenCL_COMP_CAP
	sprintf(MSG_STR,"Compiled for compute capability %d.%d",
		OpenCL_COMP_CAP/10,OpenCL_COMP_CAP%10);
#else	// ! OpenCL_COMP_CAP
	ERROR1("CAUTIOUS:  HAVE_OpenCL is defined, but OpenCL_COMP_CAP is not!?!?");
#endif	// ! OpenCL_COMP_CAP
#else	// ! HAVE_OpenCL
	sprintf(MSG_STR,"No OpenCL support in this build");
#endif	// ! HAVE_OpenCL

	prt_msg(MSG_STR);
}

static PF_COMMAND_FUNC( about_platform )
{
#ifdef HAVE_OpenCL
	sprintf(MSG_STR,"OpenCL version:  %d.%d",
		OpenCL_VERSION/1000,(OpenCL_VERSION%100)/10);
	prt_msg(MSG_STR);
#else // ! HAVE_OpenCL
	prt_msg("No OpenCL support in this build");
#endif // ! HAVE_OpenCL
#ifdef FOOBAR
	report_npp_version(SINGLE_QSP_ARG);
#endif // FOOBAR
}

static PF_COMMAND_FUNC( select_device )
{
	Platform_Device *pdp;

	pdp = PICK_PFDEV((char *)"device");
	if( pdp == NULL ) return;

	curr_pdp = pdp;
}


//OpenCL MAIN MENU
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(platform_menu,s,MENU_FUNC_NAME(f),h)

MENU_BEGIN( platform )
//ADD_CMD( devices,	query_devices,	list all platform devices )
ADD_CMD( capability,	prt_cap,		print GPU capabilities )
ADD_CMD( about,		about_platform,		print software versions)
//ADD_CMD( test_blas,	test_blas,		simple test of OpenCL BLAS )
ADD_CMD( list,		list_devs,	list all GPU devices )
ADD_CMD( info,		dev_info,		print information about a device )
ADD_CMD( select,	select_device,		select device for operations )
ADD_CMD( upload,	obj_upload,	upload data to a GPU )
ADD_CMD( dnload,	obj_dnload,	download data from a GPU )
ADD_CMD( compute,	func_menu,	compute function submenu )
#ifdef FOOBAR
ADD_CMD( npp,		npp_menu,		NPP library submenu )
#endif // FOOBAR
#ifdef HAVE_OPENGL
#ifdef HAVE_GLUT
ADD_CMD( gl,	pf_gl_menu,		openCL GL submenu )
#endif /* HAVE_GLUT */
#endif /* HAVE_OPENGL */
ADD_CMD( streams,	stream_menu,	OpenCL stream submenu )
ADD_CMD( events,	event_menu,	event  submenu )
MENU_END( platform )

PF_COMMAND_FUNC( menu )
{
	static int inited=0;

	if( ! inited ){
		PF_FUNC_NAME(init_platform)(SINGLE_QSP_ARG);
		inited=1;
	}

	PUSH_MENU(platform);
}

#endif // FOOBAR

#else // ! HAVE_CUDA

#ifdef FOOBAR
COMMAND_FUNC( do_cu2_menu )
{
	WARN("No CUDA support in this build!?");
}
#endif // FOOBAR

#endif // ! HAVE_CUDA
