
#include "quip_config.h"

/* #define REQUANT_COLOR */
/* #define REQUANT_ACHROM */
// define HAVE_MORPH in quip_config.h...
//#define HAVE_MORPH		/* why is this a flag??? */

#include <stdio.h>
#include "quip_prot.h"
#include "nvf.h"
#include "data_obj.h"
#include "vec_util.h"
#include "quip_menu.h"
#include "veclib/vl2_veclib_prot.h"
#include "platform.h"

#define TEMP_UNIMP(func)			\
						\
	warn("Sorry, function " #func " is temporarily unimplemented.");

#define DO_VCODE(code)	do_vcode(QSP_ARG  code)

static COMMAND_FUNC( getabs )	{ DO_VCODE(FVABS); }
static COMMAND_FUNC( getconj )	{ DO_VCODE(FVCONJ); }
static COMMAND_FUNC( domov )	{ DO_VCODE(FVMOV); }
static COMMAND_FUNC( getneg )	{ DO_VCODE(FVNEG); }
static COMMAND_FUNC( do_vset )	{ DO_VCODE(FVSET); }
static COMMAND_FUNC( do_uni )	{ DO_VCODE(FVUNI); }

static COMMAND_FUNC( getsum )	{ DO_VCODE(FVSUM); }
/* static COMMAND_FUNC( getsum )	{ do_projection(FVSUM); } */

static COMMAND_FUNC( do_convert )
{
	Data_Obj *dst,*src;
	Vec_Obj_Args oa1;
	Vec_Func_Code code;

	dst= pick_obj( "destination" );
	src= pick_obj( "source" );

	if( dst==NULL || src == NULL ) return;

	//convert(QSP_ARG  dst,src);
	//h_vl2_convert(QSP_ARG  dst,src);
	// BUG need to choose platform based on args!
	if( IS_BITMAP(dst) ){
		warn("do_convert:  Sorry, forgot how to convert to bit!?");
		return;
	}
	if( IS_BITMAP(src) ){
		warn("do_convert:  Sorry, forgot how to convert from bit!?");
		return;
	}

	setvarg2(&oa1,dst,src);
	switch( PREC_CODE( OBJ_MACH_PREC_PTR(dst) ) ){
		case PREC_SP:	code = FVCONV2SP; break;
		case PREC_DP:	code = FVCONV2DP; break;
		case PREC_BY:	code = FVCONV2BY; break;
		case PREC_IN:	code = FVCONV2IN; break;
		case PREC_DI:	code = FVCONV2DI; break;
		case PREC_LI:	code = FVCONV2LI; break;
		case PREC_UBY:	code = FVCONV2UBY; break;
		case PREC_UIN:	code = FVCONV2UIN; break;
		case PREC_UDI:	code = FVCONV2UDI; break;
		case PREC_ULI:	code = FVCONV2ULI; break;
		default:
			assert( AERROR("do_convert:  unhandled machine precision!?") );
			return;
	}

	perf_vfunc(QSP_ARG  code,&oa1);
}

static COMMAND_FUNC( do_ceil ) { DO_VCODE(FVCEIL); }
static COMMAND_FUNC( do_floor ) { DO_VCODE(FVFLOOR); }
static COMMAND_FUNC( do_round ) { DO_VCODE(FVROUND); }
static COMMAND_FUNC( do_rint ) { DO_VCODE(FVRINT); }
static COMMAND_FUNC( do_sign ) { DO_VCODE(FVSIGN); }

#ifdef NOT_YET
COMMAND_FUNC( dowheel )
{
	Data_Obj *dp;
	float arg;
	int n;

	dp=pick_obj( "target float image" );
	n=HOW_MANY("number of spokes");
	arg=HOW_MUCH("spoke phase");
	if( dp==NULL ) return;

	mkwheel(dp,n,arg);
}

COMMAND_FUNC( doaxle )
{
	Data_Obj *dp;

	dp=pick_obj( "target float image" );
	if( dp==NULL ) return;

	make_axle(dp);
}
#endif /* NOT_YET */

/*
 * This function will return the sum
 * of columns (rows) of an input image
 * based on a flag (dim) passed to it
 */
static COMMAND_FUNC( getDimSum )
{
	Data_Obj *dp_src, *dp_dst;
	int dim;
	int sw, sh;
	int dw, dh;
	int i, j;
	float* src_data;
	float* dst_data;

	if ((dp_dst = pick_obj( "destination object" )) == NULL) return;
	if ((dp_src = pick_obj( "source object" )) == NULL) return;

	if (OBJ_PREC(dp_dst) != PREC_SP || OBJ_PREC(dp_src) != PREC_SP) {
		sprintf(ERROR_STRING,"Object types must be float");
		advise(ERROR_STRING);
		return;
	}

	dim = (int) HOW_MANY("summing dimension");

	if (dim > 2 || dim < 1) {
		sprintf(ERROR_STRING,"Dimension can only be 1 (rows) or 2 (cols)");
		advise(ERROR_STRING);
		return;
	}

	dw = OBJ_COLS(dp_dst);
	dh = OBJ_ROWS(dp_dst);

	sw = OBJ_COLS(dp_src);
	sh = OBJ_ROWS(dp_src);

	src_data = (float *)(OBJ_DATA_PTR(dp_src));
	dst_data = (float *)(OBJ_DATA_PTR(dp_dst));

	if (dim == 1) {
		if (dh == sh && dw == 1) {
			for (i = 0; i < sh; i++) {
				dst_data[i] = 0;
				for (j = 0; j < sw; j++)
					dst_data[i] += src_data[i * sw + j];
			}

		} else {
			sprintf(ERROR_STRING,"Check destination object %s dims.", OBJ_NAME(dp_dst));
			warn(ERROR_STRING);
			return;
		}
	} else {
		if (dw == sw && dh == 1) {
			for (i = 0; i < sw; i++) {
				dst_data[i] = 0;
				for (j = 0; j < sh; j++)
					dst_data[i] += src_data[i + j * sw];
			}
		} else {
			sprintf(ERROR_STRING,"Check destination object %s dims.", OBJ_NAME(dp_dst));
			warn(ERROR_STRING);
			return;
		}
	}
}

static COMMAND_FUNC( dofind )
{
	Data_Obj *dp_src, *dp_dst;
	int sw, sh;
	int dw, dh;
	int i, j, k;
	float* src_data;
	float* dst_data;
	int thres;

	if ((dp_dst = pick_obj( "destination object" )) == NULL) return;
	if ((dp_src = pick_obj( "source object" )) == NULL) return;

	if (OBJ_PREC(dp_dst) != PREC_SP || OBJ_PREC(dp_src) != PREC_SP) {
		sprintf(ERROR_STRING,"Object types must be float");
		advise(ERROR_STRING);
		return;
	}

	if ((dw = OBJ_COLS(dp_dst)) != 2) {
		sprintf(ERROR_STRING,"Destination object must have exactly 2 columns");
		advise(ERROR_STRING);
		return;
	}

	thres = (int)HOW_MANY("threshold");
	dh = OBJ_ROWS(dp_dst);
	sw = OBJ_COLS(dp_src);
	sh = OBJ_ROWS(dp_src);

	k = 0;
	src_data = (float *)(OBJ_DATA_PTR(dp_src));
	dst_data = (float *)(OBJ_DATA_PTR(dp_dst));

	for (i = 0; i < sh; i++) {
		for (j = 0; j < sw; j++) {
			if  (src_data[j + i * sw] > thres && k < dh) {
				dst_data[2 * k] = i;
				dst_data[2 * k + 1] = j;
				k ++ ;
			}
		}
	}

	if (k != dh) {
		sprintf(ERROR_STRING,"Destination object rows were too long");
		advise(ERROR_STRING);
	}

}

#define ADD_CMD(s,f,h)	ADD_COMMAND(unary_menu,s,f,h)

MENU_BEGIN(unary)
ADD_CMD( abs,		getabs,		convert to absolute value	)
ADD_CMD( conj,		getconj,	convert to complex conjugate	)
ADD_CMD( find,		dofind,		return indeces of non-zero elements	)
ADD_CMD( dimsum,	getDimSum,	return sum along columns (rows)	)
ADD_CMD( mov,		domov,		copy data	)
ADD_CMD( neg,		getneg,		convert to negative	)
ADD_CMD( set,		do_vset,	set vector to a constant value	)
ADD_CMD( sum,		getsum,		get sum of vector	)
ADD_CMD( sign,		do_sign,	take sign of vector	)
ADD_CMD( rint,		do_rint,	round nearest integer using rint()	)
ADD_CMD( round,		do_round,	round nearest integer using round()	)
ADD_CMD( floor,		do_floor,	take floor of vector	)
ADD_CMD( ceil,		do_ceil,	take ceil of vector	)
ADD_CMD( convert,	do_convert,	convert vectors	)
ADD_CMD( uni,		do_uni,		uniform random numbers	)
MENU_END(unary)

static COMMAND_FUNC( do_unary )
{
	CHECK_AND_PUSH_MENU(unary);
}

static COMMAND_FUNC( doatan ){	DO_VCODE(FVATAN); }
static COMMAND_FUNC( doatn2 ){	DO_VCODE(FVATN2); }
static COMMAND_FUNC( doatan2 ){	DO_VCODE(FVATAN2); }
static COMMAND_FUNC( getmagsq ){	DO_VCODE(FVMGSQ); }
static COMMAND_FUNC( docos ){	DO_VCODE(FVCOS); }
static COMMAND_FUNC( doerf ){	DO_VCODE(FVERF); }
static COMMAND_FUNC( doerfinv ){DO_VCODE(FVERFINV); }
static COMMAND_FUNC( doexp ){	DO_VCODE(FVEXP); }
static COMMAND_FUNC( dolog ){	DO_VCODE(FVLOG); }
static COMMAND_FUNC( dolog10 ){	DO_VCODE(FVLOG10); }
static COMMAND_FUNC( dosin ){	DO_VCODE(FVSIN); }
static COMMAND_FUNC( dosqr ){	DO_VCODE(FVSQR); }
static COMMAND_FUNC( dosqrt ){	DO_VCODE(FVSQRT); }
static COMMAND_FUNC( dotan ){	DO_VCODE(FVTAN); }
static COMMAND_FUNC( doacos ){	DO_VCODE(FVACOS); }
static COMMAND_FUNC( dopow ){	DO_VCODE(FVPOW); }
static COMMAND_FUNC( doasin ){	DO_VCODE(FVASIN); }
// BUG - need to add configure tests for these?
//#ifdef HAVE_BESSEL
static COMMAND_FUNC( do_j0 ){	DO_VCODE(FVJ0); }
static COMMAND_FUNC( do_j1 ){	DO_VCODE(FVJ1); }
static COMMAND_FUNC( do_gamma ){	DO_VCODE(FVGAMMA); }
static COMMAND_FUNC( do_lngamma ){	DO_VCODE(FVLNGAMMA); }
//#endif /* HAVE_BESSEL */

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(trig_menu,s,f,h)

MENU_BEGIN(trig)
ADD_CMD( atan,	doatan,		compute arc tangent	)
ADD_CMD( atn2,	doatn2,		compute arc tangent (single complex arg)	)
ADD_CMD( atan2,	doatan2,	compute arc tangent (two real args)	)
ADD_CMD( magsq,	getmagsq,	convert to magnitude squared	)
ADD_CMD( cos,	docos,		compute cosine	)
ADD_CMD( erf,	doerf,		compute error function (erf)	)
ADD_CMD( erfinv, doerfinv,	compute inverse error function (erfinv)	)
ADD_CMD( exp,	doexp,		exponentiate (base e)	)
ADD_CMD( log,	dolog,		natural logarithm	)
ADD_CMD( log10,	dolog10,	logarithm base 10	)
ADD_CMD( sin,	dosin,		compute sine	)
ADD_CMD( square,	dosqr,		compute square	)
ADD_CMD( sqrt,	dosqrt,		compute square root	)
ADD_CMD( tan,	dotan,		compute tangent	)
ADD_CMD( pow,	dopow,		raise to a power	)
ADD_CMD( acos,	doacos,		compute inverse cosine	)
ADD_CMD( asin,	doasin,		compute inverse sine	)
//#ifdef HAVE_BESSEL
ADD_CMD( j0,	do_j0,		compute bessel function J0	)
ADD_CMD( j1,	do_j1,		compute bessel function J1	)
ADD_CMD( gamma,	do_gamma,	compute gamma function		)
ADD_CMD( lngamma, do_lngamma,	compute log gamma function	)
//#endif /* HAVE_BESSEL */

MENU_END(trig)

static COMMAND_FUNC( do_trig )
{
	CHECK_AND_PUSH_MENU(trig);
}

static COMMAND_FUNC( do_and ){	DO_VCODE(FVAND); }
static COMMAND_FUNC( do_nand ){	DO_VCODE(FVNAND); }
static COMMAND_FUNC( do_not ){	DO_VCODE(FVNOT); }
static COMMAND_FUNC( do_or ){	DO_VCODE(FVOR); }
static COMMAND_FUNC( do_xor ){	DO_VCODE(FVXOR); }
static COMMAND_FUNC( do_sand ){	DO_VCODE(FVSAND); }
static COMMAND_FUNC( do_sor ){	DO_VCODE(FVSOR); }
static COMMAND_FUNC( do_sxor ){	DO_VCODE(FVSXOR); }
static COMMAND_FUNC( do_shr ){	DO_VCODE(FVSHR); }
static COMMAND_FUNC( do_shl ){	DO_VCODE(FVSHL); }
static COMMAND_FUNC( do_sshr ){	DO_VCODE(FVSSHR); }
static COMMAND_FUNC( do_sshl ){	DO_VCODE(FVSSHL); }
static COMMAND_FUNC( do_sshr2 ){	DO_VCODE(FVSSHR2); }
static COMMAND_FUNC( do_sshl2 ){	DO_VCODE(FVSSHL2); }
static COMMAND_FUNC( do_comp ){	DO_VCODE(FVCOMP); }

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(logical_menu,s,f,h)

MENU_BEGIN(logical)

ADD_CMD( comp,	do_comp,	bitwise complement	)
ADD_CMD( and,	do_and,		logical AND	)
ADD_CMD( nand,	do_nand,	logical NAND	)
ADD_CMD( not,	do_not,		logical NOT	)
ADD_CMD( or,	do_or,		logical OR	)
ADD_CMD( xor,	do_xor,		logical XOR	)
ADD_CMD( sand,	do_sand,	logical AND with scalar	)
ADD_CMD( sor,	do_sor,		logical OR with scalar	)
ADD_CMD( sxor,	do_sxor,	logical XOR with scalar	)
ADD_CMD( shr,	do_shr,		right shift	)
ADD_CMD( shl,	do_shl,		left shift	)
ADD_CMD( sshr,	do_sshr,	right shift by a constant	)
ADD_CMD( sshl,	do_sshl,	left shift by a constant	)
ADD_CMD( sshr2,	do_sshr2,	right shift a constant	)
ADD_CMD( sshl2,	do_sshl2,	left shift a constant	)

MENU_END(logical)


static COMMAND_FUNC( do_logic )
{
	CHECK_AND_PUSH_MENU(logical);
}

static COMMAND_FUNC( do_vvadd ){	DO_VCODE(FVADD); }
static COMMAND_FUNC( do_vvcmul ){	DO_VCODE(FVCMUL); }
static COMMAND_FUNC( do_vvdiv ){	DO_VCODE(FVDIV); }
static COMMAND_FUNC( do_vvmul ){	DO_VCODE(FVMUL); }
static COMMAND_FUNC( do_vvsub ){	DO_VCODE(FVSUB); }

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(vvector_menu,s,f,h)

MENU_BEGIN(vvector)

ADD_CMD( add,	do_vvadd,	vector addition	)
ADD_CMD( cmul,	do_vvcmul,	multiply by complex conjugate	)
ADD_CMD( div,	do_vvdiv,	element by element division	)
ADD_CMD( mul,	do_vvmul,	element by element multiplication	)
ADD_CMD( sub,	do_vvsub,	vector subtraction	)

MENU_END(vvector)

static COMMAND_FUNC( do_vv )
{
	CHECK_AND_PUSH_MENU(vvector);
}

static COMMAND_FUNC( do_vsadd )		{	DO_VCODE(FVSADD); }
static COMMAND_FUNC( do_vssub )		{	DO_VCODE(FVSSUB); }
static COMMAND_FUNC( do_vsmul )		{	DO_VCODE(FVSMUL); }
static COMMAND_FUNC( do_vsdiv )		{	DO_VCODE(FVSDIV); }
static COMMAND_FUNC( do_vsdiv2 )	{	DO_VCODE(FVSDIV2); }

/* static COMMAND_FUNC( do_vscml )	{	DO_VCODE(FVSCML); } */

static COMMAND_FUNC( do_vsmod )		{	DO_VCODE(FVSMOD); }
static COMMAND_FUNC( do_vsmod2 )	{	DO_VCODE(FVSMOD2); }

static COMMAND_FUNC( do_vspow )		{	DO_VCODE(FVSPOW); }
static COMMAND_FUNC( do_vspow2 )	{	DO_VCODE(FVSPOW2); }
static COMMAND_FUNC( do_vsatan2 )	{	DO_VCODE(FVSATAN2); }
static COMMAND_FUNC( do_vsatan22 )	{	DO_VCODE(FVSATAN22); }
static COMMAND_FUNC( do_vsand )		{	DO_VCODE(FVSAND); }
static COMMAND_FUNC( do_vsor )		{	DO_VCODE(FVSOR); }
static COMMAND_FUNC( do_vsxor )		{	DO_VCODE(FVSXOR); }

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(svector_menu,s,f,h)

MENU_BEGIN(svector)

ADD_CMD( add,	do_vsadd,	add scalar to elements of a vector	)
ADD_CMD( sub,	do_vssub,	subtract elts. of a vector from a scalar	)
ADD_CMD( div,	do_vsdiv,	divide a scalar by the elements of a vector	)
ADD_CMD( div2,	do_vsdiv2,	divide elements of a vector by a scalar	)
ADD_CMD( mul,	do_vsmul,	multiply a vector by a real scalar	)
ADD_CMD( mod,	do_vsmod,	integer modulo of a vector by a real scalar	)
ADD_CMD( mod2,	do_vsmod2,	integer modulo of a real scalar by a vector	)
ADD_CMD( pow,	do_vspow,	raise the elements of a vector to a scalar power	)
ADD_CMD( pow2,	do_vspow2,	raise a scalar to powers given by the elements of a vector	)
ADD_CMD( atan2,	do_vsatan2,	compute 4-quadrant arc tangent of vector and scalar	)
ADD_CMD( atan22,	do_vsatan22,	compute 4-quadrant arc tangent of scalar and vector	)
ADD_CMD( and,	do_vsand,	bitwise and of scalar and vector	)
ADD_CMD( or,	do_vsor,	bitwise or of scalar and vector	)
ADD_CMD( xor,	do_vsxor,	bitwise xor of scalar and vector	)

MENU_END(svector)


static COMMAND_FUNC( do_vs ) { CHECK_AND_PUSH_MENU(svector); }

/* These return a single scalar, and can be used as projection operators */
static COMMAND_FUNC( domaxv ){	DO_VCODE(FVMAXV); }
static COMMAND_FUNC( dominv ){	DO_VCODE(FVMINV); }
static COMMAND_FUNC( domxmv ){	DO_VCODE(FVMXMV); }
static COMMAND_FUNC( domnmv ){	DO_VCODE(FVMNMV); }

static COMMAND_FUNC( domaxi ){	DO_VCODE(FVMAXI); }
static COMMAND_FUNC( domini ){	DO_VCODE(FVMINI); }
static COMMAND_FUNC( domnmi ){	DO_VCODE(FVMNMI); }
static COMMAND_FUNC( domxmi ){	DO_VCODE(FVMXMI); }

static COMMAND_FUNC( domaxg ){	DO_VCODE(FVMAXG); }
static COMMAND_FUNC( doming ){	DO_VCODE(FVMING); }
static COMMAND_FUNC( domnmg ){	DO_VCODE(FVMNMG); }
static COMMAND_FUNC( domxmg ){	DO_VCODE(FVMXMG); }

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(minmax_menu,s,f,h)

MENU_BEGIN(minmax)

ADD_CMD( max,		domaxv,	find maximum value	)
ADD_CMD( min,		dominv,	find minimum value	)
ADD_CMD( max_mag,	domxmv,	find maximum absolute value	)
ADD_CMD( min_mag,	domnmv,	find minimum absolute value	)
ADD_CMD( max_index,	domaxi,	find index of maximum value	)
ADD_CMD( min_index,	domini,	find index of minimum value	)
ADD_CMD( max_mag_index,	domxmi,	find index of maximum absolute value	)
ADD_CMD( min_mag_index,	domnmi,	find index of minimum absolute value	)
ADD_CMD( max_times,	domaxg,	find index of maximum & # of occurrences	)
ADD_CMD( min_times,	doming,	find index of minimum & # of occurrences	)
ADD_CMD( max_mag_times,	domxmg,	find index of max. mag. & # of occurrences	)
ADD_CMD( min_mag_times,	domnmg,	find index of min. mag. & # of occurrences	)

MENU_END(minmax)

static COMMAND_FUNC( do_minmax )
{
	CHECK_AND_PUSH_MENU(minmax);
}

static COMMAND_FUNC( do_cumsum )
{
	Data_Obj *dp_to,*dp_fr;

	dp_to=pick_obj( "destination vector" );
	dp_fr=pick_obj( "source vector" );

	if( dp_to==NULL || dp_fr==NULL ) return;

	war_cumsum(QSP_ARG  dp_to,dp_fr);
}

static COMMAND_FUNC( do_reduce )
{
	Data_Obj *dp,*dp2;

	dp2=pick_obj( "destination image" );
	dp=pick_obj( "source image" );
	if( dp2==NULL || dp == NULL ) return;
	reduce(QSP_ARG  dp2,dp);
}

static COMMAND_FUNC( do_enlarge )
{
	Data_Obj *dp,*dp2;

	dp2=pick_obj( "destination image" );
	dp=pick_obj( "source image" );
	if( dp2==NULL || dp == NULL ) return;
	enlarge(QSP_ARG  dp2,dp);
}

static COMMAND_FUNC( do_fwdfft )
{
	/*
	Data_Obj *dp;
	int vf_code=(-1);

	dp=pick_obj("complex image");
	if( dp == NULL ) return;

	(*PF_FFT2D_FN(PFDEV_PLATFORM(OBJ_PFDEV(dp))))(VFCODE_ARG  dp,dp);
	*/
	DO_VCODE(FVFFT2D);
}

static COMMAND_FUNC( do_fwdrowfft )
{
	/*
	Data_Obj *dp;
	int vf_code=(-1);

	dp=pick_obj("complex image");
	if( dp == NULL ) return;

	(*PF_FFTROWS_FN(PFDEV_PLATFORM(OBJ_PFDEV(dp))))(VFCODE_ARG  dp,dp);
	*/
	DO_VCODE(FVFFTROWS);
}

static COMMAND_FUNC( do_invfft )
{
	/*
	Data_Obj *dp;
	int vf_code=(-1);

	dp=pick_obj("complex image");
	if( dp == NULL ) return;
	(*PF_IFT2D_FN(PFDEV_PLATFORM(OBJ_PFDEV(dp))))(VFCODE_ARG  dp,dp);
	*/
	DO_VCODE(FVIFT2D);
}

static COMMAND_FUNC( do_invrowfft )
{
	/*
	Data_Obj *dp;
	int vf_code=(-1);

	dp=pick_obj("complex image");
	if( dp == NULL ) return;
	(*PF_IFTROWS_FN(PFDEV_PLATFORM(OBJ_PFDEV(dp))))(VFCODE_ARG  dp,dp);
	*/
	DO_VCODE(FVIFTROWS);
}

#ifdef FOOBAR

static COMMAND_FUNC( do_invrfft )
{
	Data_Obj *src,*targ;
	int vf_code=(-1);

	targ = pick_obj("real target");
	src = pick_obj("complex source");
	if( targ == NULL || src == NULL ) return;

	//h_vl2_ift2d(VFCODE_ARG  targ,src);
	(*PF_IFT2D_FN(PFDEV_PLATFORM(OBJ_PFDEV(targ))))(VFCODE_ARG  targ,src);
}

static COMMAND_FUNC( do_invrowrfft )
{
	Data_Obj *src,*targ;
	int vf_code=(-1);

	targ = pick_obj("real target");
	src = pick_obj("complex source");
	if( targ == NULL || src == NULL ) return;

	//h_vl2_iftrows(VFCODE_ARG  targ,src);
	(*PF_IFTROWS_FN(PFDEV_PLATFORM(OBJ_PFDEV(targ))))(VFCODE_ARG  targ,src);
}

/* There is no need now for separate r and c fft commands, because the selection
 * is done automatically - The commands are retained for backwards compatibility.
 */

static COMMAND_FUNC( do_fwdrfft )
{
	Data_Obj *src,*targ;
	int vf_code=(-1);

	targ = pick_obj("complex target");
	src = pick_obj("real source");
	if( targ == NULL || src == NULL ) return;

	//h_vl2_fft2d(VFCODE_ARG  targ,src);
	(*PF_FFT2D_FN(PFDEV_PLATFORM(OBJ_PFDEV(targ))))(VFCODE_ARG  targ,src);
}

static COMMAND_FUNC( do_fwdrowrfft )
{
	Data_Obj *src,*targ;
	int vf_code=(-1);

	targ = pick_obj("complex target");
	src = pick_obj("real source");
	if( targ == NULL || src == NULL ) return;

	//h_vl2_fftrows(VFCODE_ARG  targ,src);
	(*PF_FFTROWS_FN(PFDEV_PLATFORM(OBJ_PFDEV(targ))))(VFCODE_ARG  targ,src);
}
#endif // FOOBAR

#ifdef REQUANT_ACHROM

static COMMAND_FUNC( do_scan2_requant )
{
	int n;

	n=HOW_MANY("number of passes");
	scan2_requant(n);
}

static COMMAND_FUNC( do_scan_requant )
{
	int n;

	n=HOW_MANY("number of passes");
	scan_requant(n);
}

static COMMAND_FUNC( do_anneal )
{
	double temp;
	int n;

	temp = HOW_MUCH("temperature");
	n=HOW_MANY("number of passes");
	scan_anneal(temp,n);
}

static char *scanlist_choices[]={"raster","scattered","random"};

static COMMAND_FUNC( do_pickscan )
{
	switch( WHICH_ONE("type of scanning pattern",3,scanlist_choices) ){
		case 0: scan_func=get_xy_raster_point; break;
		case 1: scan_func=get_xy_scattered_point; break;
		case 2: scan_func=get_xy_random_point; break;
	}
}

static COMMAND_FUNC( do_set_input )
{
	Data_Obj *gdp;

	gdp = pick_obj( "source image" );
	if( gdp == NULL ) return;
	set_grayscale(gdp);
}


static COMMAND_FUNC( do_set_output )
{
	Data_Obj *hdp;

	hdp = pick_obj( "output image" );
	if( hdp == NULL ) return;
	set_halftone(hdp);
}

static COMMAND_FUNC( do_set_filter )
{
	Data_Obj *fdp;

	fdp = pick_obj( "filter image" );
	if( fdp == NULL ) return;
	set_filter(fdp);
}

static COMMAND_FUNC( do_init_images )
{
	if( setup_requantize() == -1 )
		warn("error setting up images");
}

static COMMAND_FUNC( do_init_requant )
{
	if( setup_requantize() == -1 ) return;
	init_requant();
}

static COMMAND_FUNC( do_qt_dither )
{
	Data_Obj *dpto, *dpfr;

	dpto = pick_obj( "target image" );
	dpfr = pick_obj( "source image" );

	if( dpto == NULL || dpfr == NULL ) return;

	qt_dither(dpto,dpfr);
}

static COMMAND_FUNC( do_tweak )
{
	int x,y;

	x=HOW_MANY("x coord");
	y=HOW_MANY("y coord");

	redo_two_pixels(x,y);
}

Command req_menu[]={
{ "set_input",	do_set_input,		"specify input grayscale image"	},
{ "set_filter",	do_set_filter,		"specify error filter image"	},
{ "set_output",	do_set_output,		"specify output halftone image"	},
{ "anneal",	do_anneal,		"requant image at specified temp"},
{ "descend",	do_scan_requant,	"requantize entire image"	},
{ "migrate",	do_scan2_requant,	"migrate pixels"		},
{ "tweak",	do_tweak,		"tweak at one pixel"		},
{ "quadtree",	do_qt_dither,		"quadtree dither algorithm"	},
{ "scan",	do_pickscan,		"select scanning pattern"	},
{ "setup_error",do_init_requant,	"initialize error & filtered error"},
{ "setup_images",do_init_images,	"setup internal images"		},
#ifndef MAC
{ "quit",	popcmd,			"exit submenu"			},
#endif
{ NULL_COMMAND								}
};

#endif /* REQUANT_ACHROM */


#ifdef REQUANT_COLOR

static COMMAND_FUNC( do_init_requant )
{
	init_requant();
}

static COMMAND_FUNC( do_set_input )
{
	Data_Obj *lumdp, *rgdp, *bydp;

	lumdp = pick_obj( "luminance image" );
	rgdp = pick_obj( "red-green image" );
	bydp = pick_obj( "blue-yellow image" );

	if( lumdp==NULL || rgdp==NULL || bydp==NULL ) return;

	set_rgb_input(lumdp,rgdp,bydp);
}

static COMMAND_FUNC( do_set_output )
{
	Data_Obj *hdp;

	hdp = pick_obj( "byte image for composite halftone" );
	if( hdp==NULL ) return;
	set_rgb_output(hdp);
}

static COMMAND_FUNC( do_set_filter )
{
	Data_Obj *rdp, *gdp, *bdp;

	rdp = pick_obj( "red filter image" );
	gdp = pick_obj( "green filter image" );
	bdp = pick_obj( "blue filter image" );

	if( rdp==NULL || gdp==NULL || bdp==NULL ) return;

	set_rgb_filter(rdp,gdp,bdp);
}

static COMMAND_FUNC( do_redo_pixel )
{
	int x,y;

	x=HOW_MANY("x coordinate");
	y=HOW_MANY("y coordinate");
	redo_pixel(x,y);
}

static char *scanlist_choices[]={"raster","scattered","random"};

static COMMAND_FUNC( do_pickscan )
{
	switch( WHICH_ONE("type of scanning pattern",3,scanlist_choices) ){
		case 0: scan_func=get_raster_point; break;
		case 1: scan_func=get_scattered_point; break;
		case 2: scan_func=get_random_point; break;
	}
}

static COMMAND_FUNC( do_setxform )
{
	Data_Obj *matrix;

	matrix = pick_obj( "transformation matrix" );
	if( matrix == NULL ) return;
	set_xform(matrix);
}

static COMMAND_FUNC( do_descend )
{
	scan_requant(1);
}

Command req_menu[]={
{ "set_input",	do_set_input,		"specify input grayscale image"	},
{ "set_output",	do_set_output,		"specify output halftone image"	},
{ "set_filter",	do_set_filter,		"specify error filter image"	},
{ "redo_pixel",	do_redo_pixel,		"redo a particular pixel"	},
{ "descend",	do_descend,		"scan image & reduce error SOS"	},
{ "scan",	do_pickscan,		"select scanning pattern"	},
{ "matrix",	do_setxform,		"specify color transformation matrix"},
{ "initialize",	do_init_requant,	"initialize error images"	},
{ "sos",	tell_sos,		"report SOS's"			},
{ "tell",	cspread_tell,		"info for all internal data images"},
#ifndef MAC
{ "quit",	popcmd,			"exit submenu"			},
#endif
{ NULL_COMMAND								}
};


#endif /* REQUANT_COLOR */

static COMMAND_FUNC( do_scale )
{
	Data_Obj *dp;
	double mn,mx;

	dp=pick_obj( "float image" );
	mn=HOW_MUCH("desired min value");
	mx=HOW_MUCH("desired max value");
	if( dp==NULL ) return;

	scale(QSP_ARG  dp,mn,mx);
}

static COMMAND_FUNC( do_dither )
{
	Data_Obj *dp;
	int size;

	dp=pick_obj( "float image" );
	size=(int)HOW_MANY("size of dither matrix");
	if( dp==NULL ) return;
	odither(QSP_ARG  dp,size);
}

static COMMAND_FUNC( do_1dramp ) { DO_VCODE(FVRAMP1D); }
static COMMAND_FUNC( do_2dramp ) { DO_VCODE(FVRAMP2D); }

#ifdef FOOBAR
static COMMAND_FUNC( do_2dramp )
{
	double s,dx,dy;
	Vec_Obj_Args oargs;

	oargs.oa_dp[0]=oargs.oa_dest = pick_obj("target image");
	oargs.oa_dp[1]=
	oargs.oa_dp[2]=NULL;

	if( oargs.oa_dest==NULL ) return;
	s=HOW_MUCH("start value");
	dx=HOW_MUCH("x ramp increment");
	dy=HOW_MUCH("y ramp increment");

	/* BUG the scalar type should match the target */
#define SETUP_SCALARS(std_type)						\
	{								\
	static std_type _s,_dx,_dy;					\
	_s = s;								\
	_dx = dx;							\
	_dy = dy;							\
	oargs.oa_svp[0] = (Scalar_Value *)(&_s) ;			\
	oargs.oa_svp[1] = (Scalar_Value *)(&_dx) ;			\
	oargs.oa_svp[2] = (Scalar_Value *)(&_dy) ;			\
	}

	switch(MACHINE_PREC(oargs.oa_dest)){
		case PREC_SP:  SETUP_SCALARS(float); break;
		case PREC_DP:  SETUP_SCALARS(double); break;
		case PREC_BY:  SETUP_SCALARS(char); break;
		case PREC_IN:  SETUP_SCALARS(short); break;
		case PREC_DI:  SETUP_SCALARS(long); break;
		case PREC_UBY:  SETUP_SCALARS(u_char); break;
		case PREC_UIN:  SETUP_SCALARS(u_short); break;
		case PREC_UDI:  SETUP_SCALARS(u_long); break;
		default:  warn("missing prec case in ramp2d"); break;
	}

	ramp2d(&oargs);
}
#endif /* FOOBAR */

static COMMAND_FUNC( do_wrap )
{
	Data_Obj *dpto,*dpfr;

	dpto=pick_obj("destination image");
	dpfr=pick_obj("source image");
	if( dpto==NULL || dpfr==NULL ) return;
	wrap(QSP_ARG  dpto,dpfr);
}

static COMMAND_FUNC( do_wrap3d )
{
	Data_Obj *dpto,*dpfr;

	dpto=pick_obj("destination image");
	dpfr=pick_obj("source image");
	if( dpto==NULL || dpfr==NULL ) return;
#ifdef FOOBAR
	wrap3d(QSP_ARG  dpto,dpfr);
#else
	warn("no wrap3d yet");
#endif
}

static COMMAND_FUNC( do_scroll )
{
	Data_Obj *dpto,*dpfr;
	int dx,dy;

	dpto=pick_obj("destination image");
	dpfr=pick_obj("source image");
	dx=(int)HOW_MANY("x displacement");
	dy=(int)HOW_MANY("y displacement");
	if( dpto==NULL || dpfr==NULL ) return;
	dp_scroll(QSP_ARG  dpto,dpfr,dx,dy);
}

#ifdef NOT_YET

static COMMAND_FUNC( doodd )
{
	Data_Obj *dp;

	dp=pick_obj( "name of image" );
	if( dp == NULL ) return;
	mkodd(dp);
}

static COMMAND_FUNC( doeven )
{
	Data_Obj *dp;

	dp=pick_obj( "name of image" );
	if( dp == NULL ) return;
	mkeven(dp);
}

#endif /* NOT_YET */

static COMMAND_FUNC( do_lutmap_b )
{ DO_VCODE(FVLUTMAPB); }

static COMMAND_FUNC( do_lutmap_s )
{ DO_VCODE(FVLUTMAPS); }

#ifdef FOOBAR
{
	Data_Obj *dst, *src, *map;

	dst=pick_obj( "destination image" );
	src=pick_obj( "source byte image" );
	map=pick_obj( "lut vector" );
	if( dst==NULL || src==NULL || map==NULL )
		return;
	if( lutmap(QSP_ARG  dst,src,map) == (-1) )
		warn("mapping failed");
}
#endif // FOOBAR

#define MAX_FS_LEVELS	128

static COMMAND_FUNC( do_fsdither )
{
	Data_Obj *dpto, *dpfr;
	int n;
	float lvl[MAX_FS_LEVELS];
	int i;

	dpto=pick_obj( "target byte image" );
	dpfr=pick_obj( "source image" );
	n=(int)HOW_MANY("number of quantization levels");
	if( n<2 || n > MAX_FS_LEVELS ){
		warn("bad number of halftone levels");
		return;
	}
	for(i=0;i<n;i++)
		lvl[i] = (float)HOW_MUCH("level value");

	if( dpto == NULL || dpfr == NULL )
		return;
	dp_halftone(QSP_ARG  dpto,dpfr,n,lvl);
}

static COMMAND_FUNC( do_udither )		/* uniform quantization */
{
	Data_Obj *dpto, *dpfr;
	int n;
	float minlvl,maxlvl,lvl[MAX_FS_LEVELS];
	int i;

	dpto=pick_obj( "target byte image" );
	dpfr=pick_obj( "source image" );
	n=(int)HOW_MANY("number of quantization levels");
	minlvl = (float)HOW_MUCH("minimum level value");
	maxlvl = (float)HOW_MUCH("maximum level value");
	if( n<2 || n > MAX_FS_LEVELS ){
		warn("bad number of halftone levels");
		return;
	}
	for(i=0;i<n;i++){
		lvl[i] = ((n-1)-i)*minlvl + i*maxlvl;
		lvl[i] /= (n-1);
	}

	if( dpto == NULL || dpfr == NULL )
		return;
	dp_halftone(QSP_ARG  dpto,dpfr,n,lvl);
}

static COMMAND_FUNC( do_resample )
{
	Data_Obj *dpto, *dpfr, *dpwarp;

	dpto=pick_obj( "target float image" );
	dpfr=pick_obj( "source float image" );
	dpwarp=pick_obj( "complex control image" );
	if( dpto==NULL || dpfr==NULL || dpwarp==NULL )
		return;

	resample(QSP_ARG  dpto,dpfr,dpwarp);
}

static COMMAND_FUNC( do_bilinear )
{
	Data_Obj *dpto, *dpfr, *dpwarp;

	dpto=pick_obj( "target float image" );
	dpfr=pick_obj( "source float image" );
	dpwarp=pick_obj( "complex control image" );
	if( dpto==NULL || dpfr==NULL || dpwarp==NULL )
		return;

	bilinear_warp(QSP_ARG  dpto,dpfr,dpwarp);
}

static COMMAND_FUNC( do_new_bilinear )
{
	Data_Obj *dpto, *dpfr, *dpwarp;

	dpto=pick_obj( "target float image" );
	dpfr=pick_obj( "source float image" );
	dpwarp=pick_obj( "complex control image" );
	if( dpto==NULL || dpfr==NULL || dpwarp==NULL )
		return;

	new_bilinear_warp(QSP_ARG  dpto,dpfr,dpwarp);
}

static COMMAND_FUNC( do_iconv )
{
	Data_Obj *dpto, *dpfr, *dpfilt;

	dpto = pick_obj( "target image" );
	dpfr = pick_obj( "source image" );
	dpfilt = pick_obj( "filter image" );

	if( dpto==NULL || dpfr==NULL || dpfilt==NULL )
		return;

	convolve(QSP_ARG  dpto,dpfr,dpfilt);
}

#ifdef NOT_YET
static COMMAND_FUNC( do_iconv3d )
{
	Data_Obj *dpto, *dpfr, *dpfilt;

	dpto = pick_obj( "target image" );
	dpfr = pick_obj( "source image" );
	dpfilt = pick_obj( "filter image" );

	if( dpto==NULL || dpfr==NULL || dpfilt==NULL )
		return;

	convolve3d(QSP_ARG  dpto,dpfr,dpfilt);
}
#endif /* NOT_YET */

static COMMAND_FUNC( do_histo )
{
	Data_Obj *dp, *hdp;
	double bw, minbin;

	hdp = pick_obj( "vector for histogram data" );
	dp = pick_obj( "source data object" );
	minbin = HOW_MUCH("minimum bin center");
	bw = HOW_MUCH("bin width");

	if( hdp == NULL || dp == NULL ) return;

	compute_histo(QSP_ARG  hdp,dp,bw,minbin);
}

static COMMAND_FUNC( do_integral )
{
	Data_Obj *dst, *src;

	dst = pick_obj( "destination image" );
	src = pick_obj( "source image" );

	if( dst == NULL || src == NULL ) return;

	cum_sum(QSP_ARG  dst,src);
}

static COMMAND_FUNC( do_hough )
{
	Data_Obj *dst, *src;
	float thresh,x0,y0;

	dst = pick_obj("destination image for transform");
	src = pick_obj("source image");
	thresh = (float)HOW_MUCH("threshold");
	x0 = (float) HOW_MUCH("x origin");
	y0 = (float) HOW_MUCH("y origin");

	if( dst == NULL || src == NULL ) return;

	hough(QSP_ARG  dst,src,thresh,x0,y0);
}

static COMMAND_FUNC( do_local_max )
{
	Data_Obj *val_dp, *coord_dp, *src;
	long n;

	val_dp = pick_obj("destination vector for local maximum values");
	coord_dp = pick_obj("destination vector for coordinates");
	src = pick_obj("source image");

	if( val_dp != NULL && coord_dp != NULL && src != NULL )
		n = local_maxima(QSP_ARG  val_dp,coord_dp,src);
	else
		n = 0;

	sprintf(msg_str,"%ld",n);
	assign_var("n_maxima",msg_str);
}


#define MAX_DIMENSIONS	(N_DIMENSIONS-1)

static COMMAND_FUNC( do_mhisto )
{
	Data_Obj *dp, *hdp;
	float bw[MAX_DIMENSIONS], minbin[MAX_DIMENSIONS];
	dimension_t i;

	hdp = pick_obj( "target histogram data object" );
	dp = pick_obj( "source data object" );
	if( dp == NULL ) return;
	for(i=0;i<OBJ_COMPS(dp);i++){
		minbin[i] = (float) HOW_MUCH("minimum bin center");
		bw[i] = (float) HOW_MUCH("bin width");
	}

	if( hdp == NULL || dp == NULL ) return;

	multivariate_histo(QSP_ARG  hdp,dp,bw,minbin);
}

#ifdef HAVE_MORPH

static COMMAND_FUNC( do_fill )
{
	Data_Obj *dp;
	dimension_t x,y;
	double val;
	double tol;

	dp=pick_obj("image");
	x= (dimension_t) HOW_MANY("seed x");
	y= (dimension_t) HOW_MANY("seed y");
	val=HOW_MUCH("value");
	tol=HOW_MUCH("tolerance");

	if( dp == NULL ) return;
	if( tol < 0 ){
		warn("tolerance must be non-negative");
		return;
	}

	ifl(QSP_ARG  dp,x,y,val,tol);
}
#endif /* HAVE_MORPH */


#ifdef NOT_YET

static COMMAND_FUNC( do_quads )
{
	Data_Obj *src, *dst;

	dst=pick_obj( "destination 4-tuple list" );
	src=pick_obj( "source image" );
	if( dst == NULL || src == NULL ) return;

	make_all_quads(QSP_ARG  dst,src);
}

static COMMAND_FUNC( do_ext_paths )
{
	Data_Obj *src, *dst;

	dst=pick_obj( "destination matrix" );
	src=pick_obj( "source matrix" );
	if( dst == NULL || src == NULL ) return;

	extend_shortest_paths(QSP_ARG  dst,src);
}

#ifdef FOOBAR
static COMMAND_FUNC( do_vstitch )
{
	Data_Obj *to,*fr,*co;
	int n;

	to=pick_obj( "target float vector" );
	fr=pick_obj( "source float vector" );
	co=pick_obj( "control float vector" );
	if( to==NULL || fr==NULL || co==NULL ) return;
	n=vstitch(QSP_ARG  to,fr,co);
	if( verbose ){
		sprintf(ERROR_STRING,"%d elements copied",n);
		advise(ERROR_STRING);
	}
}
#endif /* FOOBAR */

#endif /* NOT_YET */

static COMMAND_FUNC( do_vinterp )
{
	Data_Obj *to,*fr,*co;

	to=pick_obj( "target float vector" );
	fr=pick_obj( "source float vector" );
	co=pick_obj( "control float vector" );
	if( to==NULL || fr==NULL || co==NULL ) return;
	vinterp(QSP_ARG  to,fr,co);
}

static COMMAND_FUNC( do_median )
{
	Data_Obj *to, *fr;

	to=pick_obj("target");
	fr=pick_obj("source");
	if( to==NULL || fr==NULL ) return;
	median(QSP_ARG  to,fr);
}

static COMMAND_FUNC( do_median_clip )
{
	Data_Obj *to, *fr;

	to=pick_obj("target");
	fr=pick_obj("source");
	if( to==NULL || fr==NULL ) return;
	median_clip(QSP_ARG  to,fr);
}

static COMMAND_FUNC( do_median_1D )
{
	Data_Obj *to, *fr;
	int rad;

	to=pick_obj("target");
	fr=pick_obj("source");
	rad = (int) HOW_MANY("radius");

	if( to==NULL || fr==NULL ) return;

	median_1D(QSP_ARG  to,fr,rad);
}

static COMMAND_FUNC( do_krast )
{
	Data_Obj *dp;

	dp=pick_obj("coord list");
	if( dp != NULL )
		mk_krast(QSP_ARG  dp);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(image_menu,s,f,h)

MENU_BEGIN(image)

ADD_CMD( ramp1d,	do_1dramp,	make a 1-D ramp	)
ADD_CMD( ramp2d,	do_2dramp,	make a ramp image	)
ADD_CMD( diffuse,	do_fsdither,	dither image using arbitrary levels	)
ADD_CMD( udiffuse,	do_udither,	dither image using uniform levels	)
#ifdef NOT_YET
ADD_CMD( product,	do_prod,	make product image	)
ADD_CMD( wheel,		dowheel,	make a wheel image	)
ADD_CMD( axle,		doaxle,		put a 1 at 0 freq	)
ADD_CMD( odd,		doodd,		make image odd	)
ADD_CMD( even,		doeven,		make image even	)
#endif /* NOT_YET */

MENU_END(image)


static COMMAND_FUNC( do_imgsyn )
{
	CHECK_AND_PUSH_MENU(image);
}

static COMMAND_FUNC( do_sort )
{
	Data_Obj *dp;

	dp=pick_obj("");
	if( dp == NULL ) return;

	sort_data(QSP_ARG  dp);
}

static COMMAND_FUNC( do_sort_indices )
{
	Data_Obj *dp1,*dp2;

	dp1=pick_obj("array of indices");
	dp2=pick_obj("data array");
	if( dp1 == NULL || dp2 == NULL ) return;
	sort_indices(QSP_ARG  dp1,dp2);
}

static COMMAND_FUNC( do_scramble )
{
	Data_Obj *dp;

	dp = pick_obj("");
	if( dp == NULL ) return;

	dp_scramble(QSP_ARG  dp);
}

COMMAND_FUNC( do_yuv2rgb )
{
	Data_Obj *dst_dp, *src_dp;

	dst_dp = pick_obj("destination rgb image");
	src_dp = pick_obj("source yuv image");

	if( dst_dp == NULL || src_dp == NULL ) return;

	yuv422_to_rgb24(QSP_ARG   dst_dp, src_dp );
}

static COMMAND_FUNC( do_yuv2gray )
{
	Data_Obj *dst_dp, *src_dp;

	dst_dp = pick_obj("destination grayscale image");
	src_dp = pick_obj("source yuv image");

	if( dst_dp == NULL || src_dp == NULL ) return;

	yuv422_to_gray(QSP_ARG   dst_dp, src_dp );
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(misc_menu,s,f,h)

MENU_BEGIN(misc)

ADD_CMD( krast,		do_krast,	compute coords of space-filling curve	)
ADD_CMD( cumsum,		do_cumsum,	computes cumulative sum of a vector	)
ADD_CMD( reduce,		do_reduce,	reduce an image	)
ADD_CMD( enlarge,	do_enlarge,	enlarge an image	)
ADD_CMD( scale,		do_scale,	scale float image to byte range	)
ADD_CMD( histogram,	do_histo,	compute histogram of data	)
ADD_CMD( integral,	do_integral,	compute integral image (cumulative sum)	)
ADD_CMD( hough,		do_hough,	compute Hough transform	)
ADD_CMD( local_maxima,	do_local_max,	find local maxima	)
ADD_CMD( multivariate,	do_mhisto,	multivariate histogram of data	)
ADD_CMD( dither,		do_dither,	initialize ordered dither matrix	)
ADD_CMD( median,		do_median,	apply median filter to image	)
ADD_CMD( median_clip,	do_median_clip,	apply median filter to bright pixels	)
ADD_CMD( median_1D,	do_median_1D,	apply median filter to vector	)
ADD_CMD( map_b,		do_lutmap_b,	map through a byte-indexed lookup table	)
ADD_CMD( map_s,		do_lutmap_s,	map through a short-indexed lookup table	)
ADD_CMD( resample,	do_resample,	warp image using control image	)
ADD_CMD( bilinear,	do_bilinear,	warp image w/ bilinear inter.	)
ADD_CMD( new_bilinear,	do_new_bilinear,warp image w/ bilinear inter.	)
ADD_CMD( interpolate,	do_vinterp,	interpolate gaps in a vector	)

#ifdef HAVE_MORPH
ADD_CMD( fill,		do_fill,	flood fill from seed point	)
#endif

ADD_CMD( sort,		do_sort,	sort elements of a vector in-place	)
ADD_CMD( sort_indices,	do_sort_indices,sort indices of data array	)
ADD_CMD( scramble,	do_scramble,	permute elements of a real data object	)

#ifdef NOT_YET
#ifdef FOOBAR
ADD_CMD( project,	do_project,	project columns of an image	)
ADD_CMD( shrink,		do_shrink,	shrink image in place by half	)
ADD_CMD( stitch,		do_vstitch,	cut out unwanted bits of a vector	)
ADD_CMD( accumulate,	do_accumulate,	compute cumulative sum of a vector	)
ADD_CMD( finite,		do_nanchk,	verify finite values	)
ADD_CMD( transform,	cmxform,	matrix transformation of cpx data	)
#endif /* FOOBAR */


ADD_CMD( convolve3d,	do_iconv3d,	convolve two sequences	)

ADD_CMD( Quads,		do_quads,	list of all 4-tuples from an image	)
ADD_CMD( extend_paths,	do_ext_paths,	matrix "squaring" for shortest-path algorithm	)

#endif /* NOT_YET */

ADD_CMD( convolve,	do_iconv,	convolve two images	)
ADD_CMD( yuv2rgb,	do_yuv2rgb,	convert yuv422 to rgb	)
ADD_CMD( yuv2gray,	do_yuv2gray,	convert yuv422 to grayscale	)

MENU_END(misc)


static COMMAND_FUNC( do_misc )
{
	CHECK_AND_PUSH_MENU(misc);
}

#define EXCHANGE( dp1, dp2 )				\
							\
	{						\
		Data_Obj *temp;				\
		temp = dp1;				\
		dp1 = dp2;				\
		dp2 = temp;				\
	}


	/*
	 * Because we do an equal number of erosions and dilations, we always
	 * do an even number of passes regardless of whether size is even or odd.
	 * Therefore, the final data ends up in the original input buffer, and
	 * because we want it to end up in the other buffer, a copy is required.
	 * We could eliminate this copy by creating a third buffer and using it
	 * appropriately, but we will defer that more efficient but complicated
	 * approach for the time being because we are too lazy to figure out the
	 * correct logic.
	 */

#define CYCLE_OPS( op1, op2 )				\
							\
	int n;						\
							\
	n=size;						\
	while(n--){					\
		op1(QSP_ARG  to,fr);			\
		EXCHANGE(to,fr);			\
	}						\
	n=size;						\
	while(n--){					\
		op2(QSP_ARG  to,fr);			\
		EXCHANGE(to,fr);			\
	}						\
	dp_copy(to,fr);

#ifdef HAVE_MORPH

static void image_close(QSP_ARG_DECL  Data_Obj *to,Data_Obj *fr,int size)
{
	CYCLE_OPS( dilate, erode )
}

static void image_open(QSP_ARG_DECL  Data_Obj *to,Data_Obj *fr,int size)
{
	CYCLE_OPS( erode, dilate )
}


static COMMAND_FUNC( do_closing )
{
	Data_Obj *to,*fr;
	int size;

	to=pick_obj("target");
	fr=pick_obj("source");
	size=(int)HOW_MANY("size of the closing");

	if( to==NULL || fr==NULL ) return;

	if( size <= 0 ){
		warn("size for closing operator must be positive");
		return;
	}

	image_close(QSP_ARG  to,fr,size);
}

static COMMAND_FUNC( do_opening )   /* Should choose the size of the opening */
{
	Data_Obj *to,*fr;
	int size;

	to=pick_obj("target");
	fr=pick_obj("source");
	size=(int)HOW_MANY("size of the opening");

	if( to==NULL || fr==NULL ) return;

	if( size <= 0 ){
		warn("size for opening operator must be positive");
		return;
	}

	image_open(QSP_ARG  to,fr,size);
}

static COMMAND_FUNC( do_dilate )
{
	Data_Obj *to,*fr;

	to=pick_obj("target");
	fr=pick_obj("source");
	if( to==NULL || fr==NULL ) return;

	dilate(QSP_ARG  to,fr);
}

static COMMAND_FUNC( do_erode )
{
	Data_Obj *to,*fr;

	to=pick_obj("target");
	fr=pick_obj("source");
	if( to==NULL || fr==NULL ) return;

	erode(QSP_ARG  to,fr);
}

static COMMAND_FUNC( gen_morph )
{
	Data_Obj *to,*fr,*tbl;

	to=pick_obj("target");
	fr=pick_obj("source");
	tbl=pick_obj("function look-up table");
	if( to==NULL || fr==NULL || tbl == NULL ) return;

	morph_process(QSP_ARG  to,fr,tbl);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(morph_menu,s,f,h)

MENU_BEGIN(morph)

ADD_CMD( dilate,	do_dilate,	dilatation of a binary image	)
ADD_CMD( erode,		do_erode,	erosion of a binary image	)
ADD_CMD( closing,	do_closing,	closing of a binary image	)
ADD_CMD( opening,	do_opening,	opening of a binary image	)
ADD_CMD( morph,		gen_morph,	apply table-defined morphological operator	)

MENU_END(morph)


static COMMAND_FUNC( do_morph )
{
	CHECK_AND_PUSH_MENU(morph);
}

#endif /* HAVE_MORPH */

#ifdef NOT_YET
static COMMAND_FUNC( do_radavg )
{
	Data_Obj *m_dp, *v_dp, *c_dp, *i_dp;

	m_dp = pick_obj( "data vector for mean" );
	v_dp = pick_obj( "data vector for variance" );
	c_dp = pick_obj( "data vector for counts" );
	i_dp = pick_obj( "source image" );

	if( m_dp == NULL || v_dp == NULL || c_dp == NULL || i_dp == NULL )
		return;

	rad_avg(QSP_ARG  m_dp,v_dp,c_dp,i_dp);
}

static COMMAND_FUNC( do_oriavg )
{
	Data_Obj *m_dp, *v_dp, *c_dp, *i_dp;

	m_dp = pick_obj( "data vector for mean" );
	v_dp = pick_obj( "data vector for variance" );
	c_dp = pick_obj( "data vector for counts" );
	i_dp = pick_obj( "source image" );

	if( m_dp == NULL || v_dp == NULL || c_dp == NULL || i_dp == NULL )
		return;

	ori_avg(QSP_ARG  m_dp,v_dp,c_dp,i_dp);
}
#endif /* NOT_YET */

#include "dct8.h"

static COMMAND_FUNC( do_dct )
{
	Data_Obj *dp;

	dp = pick_obj("");
	if( dp==NULL ) return;

	compute_dct(QSP_ARG  dp,FWD_DCT);
}

static COMMAND_FUNC( do_idct )
{
	Data_Obj *dp;

	dp = pick_obj("");
	if( dp==NULL ) return;

	compute_dct(QSP_ARG  dp,INV_DCT);
}

static COMMAND_FUNC( do_odct )
{
	Data_Obj *dp;

	dp = pick_obj("");
	if( dp==NULL ) return;

	compute_dct(QSP_ARG  dp,OLD_DCT);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(fft_menu,s,f,h)

MENU_BEGIN(fft)


#ifdef FOOBAR
ADD_CMD( newfft,	do_newfft,	test new chainable complex fft )
#endif /* FOOBAR */

ADD_CMD( fft,		do_fwdfft,	forward Fourier transform )
ADD_CMD( invfft,	do_invfft,	inverse Fourier transform )
ADD_CMD( row_fft,	do_fwdrowfft,	forward Fourier transform of rows only )
ADD_CMD( row_invfft,	do_invrowfft,	inverse Fourier transform of rows only )

#ifdef FOOBAR
ADD_CMD( fft,		do_fwdfft,	forward complex Fourier transform )
ADD_CMD( invfft,	do_invfft,	inverse complex Fourier transform )
ADD_CMD( row_fft,	do_fwdrowfft,	forward complex Fourier transform of rows only )
ADD_CMD( row_invfft,	do_invrowfft,	inverse complex Fourier transform of rows only )
ADD_CMD( rfft,		do_fwdrfft,	forward Fourier transform w/ real input )
ADD_CMD( irfft,		do_invrfft,	inverse Fourier transform w/ real output )
ADD_CMD( row_rfft,	do_fwdrowrfft,	forward Fourier transform of rows only w/ real input )
ADD_CMD( row_irfft,	do_invrowrfft,	inverse Fourier transform of rows only w/ real output )
#endif // FOOBAR

ADD_CMD( wrap,		do_wrap,	wrap DFT image )
ADD_CMD( wrap3d,	do_wrap3d,	wrap 3-D DFT )
ADD_CMD( scroll,	do_scroll,	scroll image )
#ifdef NOT_YET
ADD_CMD( radavg,	do_radavg,	compute radial average )
ADD_CMD( oriavg,	do_oriavg,	compute orientation average )
#endif /* NOT_YET */
ADD_CMD( dct,		do_dct,		compute blocked discrete cosine xform )
ADD_CMD( odct,		do_odct,	compute DCT using old method )
ADD_CMD( idct,		do_idct,	compute inverse discrete cosine xform )

MENU_END(fft)



static COMMAND_FUNC( do_fft )
{
	CHECK_AND_PUSH_MENU(fft);
}


static COMMAND_FUNC( do_clip ){	DO_VCODE(FVCLIP);	}
static COMMAND_FUNC( do_iclip ){	DO_VCODE(FVICLP);	}
static COMMAND_FUNC( do_vscmp ){	DO_VCODE(FVSCMP);	}
static COMMAND_FUNC( do_vscmp2 ){DO_VCODE(FVSCMP2);	}

static COMMAND_FUNC( do_bnd ){	DO_VCODE(FVBND);	}
static COMMAND_FUNC( do_ibnd ){	DO_VCODE(FVIBND);	}
static COMMAND_FUNC( do_vcmp ){	DO_VCODE(FVCMP);	}

/* static COMMAND_FUNC( do_vcmpm ){	DO_VCODE(FVCMPM);	} */
/* static COMMAND_FUNC( do_vscmm ){	DO_VCODE(FVSCMM);	} */

static COMMAND_FUNC( do_vsmax ){	DO_VCODE(FVSMAX);	}
static COMMAND_FUNC( do_vsmxm ){	DO_VCODE(FVSMXM);	}
static COMMAND_FUNC( do_vsmin ){	DO_VCODE(FVSMIN);	}
static COMMAND_FUNC( do_vsmnm ){	DO_VCODE(FVSMNM);	}

/* static COMMAND_FUNC( do_vmcmm ){	DO_VCODE(FVMCMM);	} */

static COMMAND_FUNC( do_vsm_lt ){	DO_VCODE(FVSMLT);	}
static COMMAND_FUNC( do_vsm_gt ){	DO_VCODE(FVSMGT);	}
static COMMAND_FUNC( do_vsm_le ){	DO_VCODE(FVSMLE);	}
static COMMAND_FUNC( do_vsm_ge ){	DO_VCODE(FVSMGE);	}
static COMMAND_FUNC( do_vsm_ne ){	DO_VCODE(FVSMNE);	}
static COMMAND_FUNC( do_vsm_eq ){	DO_VCODE(FVSMEQ);	}

static COMMAND_FUNC( do_vvm_lt ){	DO_VCODE(FVVMLT);	}
static COMMAND_FUNC( do_vvm_gt ){	DO_VCODE(FVVMGT);	}
static COMMAND_FUNC( do_vvm_le ){	DO_VCODE(FVVMLE);	}
static COMMAND_FUNC( do_vvm_ge ){	DO_VCODE(FVVMGE);	}
static COMMAND_FUNC( do_vvm_ne ){	DO_VCODE(FVVMNE);	}
static COMMAND_FUNC( do_vvm_eq ){	DO_VCODE(FVVMEQ);	}

/* static COMMAND_FUNC( do_vmcmp ){	DO_VCODE(FVMCMP);	} */

#ifdef FOOBAR
static COMMAND_FUNC( do_vmscm ){	DO_VCODE(FVMSCM);	}
static COMMAND_FUNC( do_vmscp ){	DO_VCODE(FVMSCP);	}
#endif /* FOOBAR */

static COMMAND_FUNC( do_vvvslct ){ DO_VCODE(FVVVSLCT);	}
static COMMAND_FUNC( do_vvsslct ){ DO_VCODE(FVVSSLCT);	}
static COMMAND_FUNC( do_vssslct ){ DO_VCODE(FVSSSLCT);	}

static COMMAND_FUNC( do_vmax ){		DO_VCODE(FVMAX);	}
static COMMAND_FUNC( do_vmin ){		DO_VCODE(FVMIN);	}
static COMMAND_FUNC( do_vmaxm ){	DO_VCODE(FVMAXM);	}
static COMMAND_FUNC( do_vminm ){	DO_VCODE(FVMINM);	}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(compare_menu,s,f,h)

MENU_BEGIN(compare)

ADD_CMD( max,		do_vmax,	take the max of two vectors	)
ADD_CMD( min,		do_vmin,	take the min of two vectors	)
ADD_CMD( max_mag,	do_vmaxm,	take the max mag of two vectors	)
ADD_CMD( min_mag,	do_vminm,	take the min mag of two vectors	)
ADD_CMD( vmscp,		do_vsm_gt,	bit-map scalar-vector comparision	)
ADD_CMD( clip,		do_clip,	clip elements of a vector	)
ADD_CMD( iclip,		do_iclip,	inverted clip	)
ADD_CMD( vscmp,		do_vscmp,	vector-scalar comparison (>=)	)
ADD_CMD( vscmp2,	do_vscmp2,	vector-scalar comparison (<=)	)
ADD_CMD( bound,		do_bnd,		bound elements of a vector	)
ADD_CMD( ibound,	do_ibnd,	inverted bound	)
ADD_CMD( vcmp,		do_vcmp,	vector-vector comparison	)
ADD_CMD( vsmax,		do_vsmax,	scalar-vector maximum	)
ADD_CMD( vsmxm,		do_vsmxm,	scalar-vector maximum magnitude	)
ADD_CMD( vsmin,		do_vsmin,	scalar-vector minimum	)
ADD_CMD( vsmnm,		do_vsmnm,	scalar-vector minimum magnitude	)
ADD_CMD( vvm_lt,	do_vvm_lt,	bit-map vector comparision (<)	)
ADD_CMD( vvm_gt,	do_vvm_gt,	bit-map vector comparision (>)	)
ADD_CMD( vvm_le,	do_vvm_le,	bit-map vector comparision (<=)	)
ADD_CMD( vvm_ge,	do_vvm_ge,	bit-map vector comparision (>=)	)
ADD_CMD( vvm_ne,	do_vvm_ne,	bit-map vector comparision (!=)	)
ADD_CMD( vvm_eq,	do_vvm_eq,	bit-map vector comparision (==)	)
ADD_CMD( vsm_lt,	do_vsm_lt,	bit-map vector/scalar comparision (<)	)
ADD_CMD( vsm_gt,	do_vsm_gt,	bit-map vector/scalar comparision (>)	)
ADD_CMD( vsm_le,	do_vsm_le,	bit-map vector/scalar comparision (<=)	)
ADD_CMD( vsm_ge,	do_vsm_ge,	bit-map vector/scalar comparision (>=)	)
ADD_CMD( vsm_ne,	do_vsm_ne,	bit-map vector/scalar comparision (!=)	)
ADD_CMD( vsm_eq,	do_vsm_eq,	bit-map vector/scalar comparision (==)	)

/* for iview compatibility */
ADD_CMD( select,	do_vvvslct,	vector/vector selection based on bit-map	)
ADD_CMD( vv_select,	do_vvvslct,	vector/vector selection based on bit-map	)
ADD_CMD( vs_select,	do_vvsslct,	vector/scalar selection based on bit-map	)
ADD_CMD( ss_select,	do_vssslct,	scalar/scalar selection based on bit-map	)

#ifdef FOOBAR
/* put back in for iview compatibility */
ADD_CMD( vmscm,		do_vmscm,	bit-map scalar-vector mag. comparision	)
ADD_CMD( vcmpm,		do_vcmpm,	vector-vector magnitude comparison	)
ADD_CMD( vscmm,		do_vscmm,	scalar-vector magnitude comparison	)
ADD_CMD( vmcmp,		do_vmcmp,	bit-map vector comparision	)
#endif /* FOOBAR */

MENU_END(compare)


static COMMAND_FUNC( docmp )
{
	CHECK_AND_PUSH_MENU(compare);
}

/* static COMMAND_FUNC( do_corr ) { DO_VCODE(FVCONV); } */
static COMMAND_FUNC( do_dot )
{
	DO_VCODE(FVDOT);
}
/* static COMMAND_FUNC( do_cdot ) { DO_VCODE(FVCDOT); } */

static COMMAND_FUNC( do_xpose )
{
	Data_Obj *dpto, *dpfr;

	dpto=pick_obj( "target image" );
	dpfr=pick_obj( "source image" );
	if( dpto == NULL || dpfr == NULL ) return;
	transpose(QSP_ARG  dpto,dpfr);

	//TEMP_UNIMP(transpose)
}

static COMMAND_FUNC( do_ginvert )
{
	Data_Obj *dp;

	dp=pick_obj( "target matrix" );
	if( dp == NULL ) return;

	/* BUG? is complete error checking done here??? */

	if( ! IS_CONTIGUOUS(dp) ){
		warn("matrix must be contiguous for G-J inversion");
		return;
	}

	if( OBJ_PREC(dp) == PREC_SP )
		gauss_jordan( (float *)OBJ_DATA_PTR(dp), OBJ_COLS(dp) );
	else if( OBJ_PREC(dp) == PREC_DP )
		dp_gauss_jordan( (double *)OBJ_DATA_PTR(dp), OBJ_COLS(dp) );
	else {
		sprintf(ERROR_STRING,
	"Matrix %s should have float or double precision for Gauss-Jordan inversion",
			OBJ_NAME(dp));
		warn(ERROR_STRING);
	}
}


/* inner (matrix) product - implemented in newvec? */

static COMMAND_FUNC(do_inner)
{
	Data_Obj *target, *v1, *v2;

	target=pick_obj( "target data object" );
	v1=pick_obj( "first operand" );
	v2=pick_obj( "second operand" );

	if( target==NULL || v1==NULL || v2==NULL )
		return;

	inner(target,v1,v2);
}

static COMMAND_FUNC( do_invert )
{
	Data_Obj *dp;
	double det;

	dp=pick_obj( "matrix" );
	if( dp == NULL ) return;
	det=dt_invert(dp);
	if( det == 0.0 ) warn("ZERO DETERMINANT!!!");
	else if( verbose ) {
		sprintf(msg_str,"determinant:  %g",det);
		prt_msg(msg_str);
	}
}

#ifdef FOOBAR
static COMMAND_FUNC( do_corr_mtrx )
{
	Data_Obj *dpto;
	Data_Obj *dpfr;

	dpto = pick_obj("target corrlation matrix");
	dpfr = pick_obj("source vector array");

	if( dpto == NULL || dpfr == NULL ) return;

	corr_matrix(dpto,dpfr);
}
#endif /* FOOBAR */

static COMMAND_FUNC( do_determinant )
{
	Data_Obj *dp,*scal_dp;
	double d;

	scal_dp = pick_obj("scalar for determinant");
	dp = pick_obj("matrix");

	if( scal_dp == NULL || dp==NULL ) return;

	//d = determinant(dp);
	TEMP_UNIMP(determinant) d=0.0;

	if( OBJ_PREC(dp) == PREC_SP ){
		*((float *)OBJ_DATA_PTR(scal_dp)) = (float)d;
	} else {
		sprintf(ERROR_STRING,"determinant:  bad scalar precision (%s) for object %s!?",
			PREC_NAME(OBJ_PREC_PTR(dp)),OBJ_NAME(dp));
		warn(ERROR_STRING);
	}
	
}

static COMMAND_FUNC(do_xform_list )
{
	Data_Obj *dpto, *dpfr, *xform;
	Vec_Obj_Args oa1;

	dpto=pick_obj( "target data object" );
	dpfr=pick_obj( "source data object" );
	xform=pick_obj( "transformation matrix" );

	if( dpto==NULL || dpfr==NULL || xform==NULL ) return;

	clear_obj_args(&oa1);
	SET_OA_DEST(&oa1,dpto);
	SET_OA_SRC1(&oa1,dpfr);
	SET_OA_SRC2(&oa1,xform);
	set_obj_arg_flags(&oa1);

	// BUG need to make this a platform function!?
	h_vl2_xform_list(QSP_ARG  -1,  &oa1);
	//TEMP_UNIMP(xform_list)
}

static COMMAND_FUNC( do_vec_xform )
{
	Data_Obj *dpto, *dpfr, *xform;

	dpto=pick_obj( "target data object" );
	dpfr=pick_obj( "source data object" );
	xform=pick_obj( "transformation matrix" );

	if( dpto==NULL || dpfr==NULL || xform==NULL ) return;

	//vec_xform(QSP_ARG  dpto,dpfr,xform);
	TEMP_UNIMP(vec_xform)
}

static COMMAND_FUNC( do_homog_xform )
{
	Data_Obj *dpto, *dpfr, *xform;

	dpto=pick_obj( "target data object" );
	dpfr=pick_obj( "source data object" );
	xform=pick_obj( "transformation matrix" );

	if( dpto==NULL || dpfr==NULL || xform==NULL ) return;

	//homog_xform(QSP_ARG  dpto,dpfr,xform);
	TEMP_UNIMP(homog_xform)
}

#ifdef FOOBAR
COMMAND_FUNC( do_outer )
{
	warn("sorry, outer product not yet implemented");
}
#endif /* FOOBAR */

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(linear_menu,s,f,h)

MENU_BEGIN(linear)

ADD_CMD( dot,		do_dot,		vector dot product	)
ADD_CMD( transpose,	do_xpose,	transpose an image	)
ADD_CMD( xform,		do_vec_xform,	transform (vectorizing each element)	)
ADD_CMD( lxform,	do_xform_list,	transform (vectorizing over the list)	)
ADD_CMD( homogenous,	do_homog_xform,	transform homogenous coords (vectorizes over list)	)
ADD_CMD( oinvert,	do_invert,	invert a square matrix (old way)	)
ADD_CMD( invert,	do_ginvert,	invert a square matrix (Gauss-Jordan)	)
ADD_CMD( determinant,	do_determinant,	compute matrix determinant	)

ADD_CMD( inner_prod,	do_inner,	compute inner product	)
#ifdef NOT_YET
ADD_CMD( outer_prod,	do_outer,	compute outer product	)

#ifdef FOOBAR
ADD_CMD( correlate,	do_corr,	correlate two vectors	)
ADD_CMD( corr_mtrx,	do_corr_mtrx,	compute correlation matrix of row vectors	)
ADD_CMD( cdot,		do_cdot,	conjugate dot product	)
ADD_CMD( convolve,	do_conv,	convolve two vectors	)
#endif /* FOOBAR */
#endif /* NOT_YET */

MENU_END(linear)


static COMMAND_FUNC( do_lin )
{
	CHECK_AND_PUSH_MENU(linear);
}

#ifdef VECEXP
static COMMAND_FUNC( do_fileparse )
{
	FILE *fp;
	char *s;

	/* disable_lookahead(); */
	s=NAMEOF("expression file");
	if( strcmp(s,"-") ){
		fp=try_open( s, "r" );
		if( !fp ) return;
		push_input_file(s);
		redir(fp);
	}
	expr_file();
	/* enable_lookahead(); */
}

static COMMAND_FUNC( do_parse )
{
	expr_file();
}


Command expr_menu[]={
{	"parse",	do_parse,	"parse single expression"		},
{	"read",	do_fileparse,	"parse contents of a file"		},
{	"list",	list_subrts,	"list defined subroutines"		},
{	"run",	do_run_subrt,	"run a subroutine"			},
{	"scan",	do_scan_subrt,	"scan a subroutine tree"		},
{	"optimize",	do_opt_subrt,	"optimize a subroutine tree"		},
{	"info",	do_subrt_info,	"print subroutine info"			},
{	"dump",	do_dump_subrt,	"run a subroutine"			},
#ifndef MAC
{	"quit",	popcmd,		"exit submenu"				},
#endif
{	NULL_COMMAND								}
};

 COMMAND_FUNC(do_exprs )
{
	warm_init();
	expr_init();
	CHECK_AND_PUSH_MENU(expr_menu);
}
#endif /* VECEXP */

void set_perf(int flag)
{
	for_real=flag;
}

static COMMAND_FUNC( do_set_perf )
{
	set_perf( ASKIF("really perform vector computations") );
}

static COMMAND_FUNC( do_report_status )
{
	if( for_real )
		prt_msg("Executing all vector computations");
	else
		prt_msg("Parsing but not executing all vector computations");

	sprintf(msg_str,"Using %d processor%s (%d max on this machine)",
			n_processors,n_processors==1?"":"s",N_PROCESSORS);
	prt_msg(msg_str);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(control_menu,s,f,h)

MENU_BEGIN(control)

ADD_CMD( perform,	do_set_perf,		set/clear execute flag )
ADD_CMD( n_processors,	set_n_processors,	set number of processors to use )
ADD_CMD( status,	do_report_status,	report current computation modes )
#ifdef FOOBAR
ADD_CMD( cpuinfo,	get_cpu_info,		report cpu info )
#endif /* FOOBAR */

MENU_END(control)

static COMMAND_FUNC(do_ctl )
{
	CHECK_AND_PUSH_MENU(control);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(compute_menu,s,f,h)

MENU_BEGIN(compute)

ADD_CMD(	trig,		do_trig,	trigonometric operations	)
ADD_CMD(	unary,		do_unary,	unary operations on data	)
ADD_CMD(	logical,	do_logic,	logical operations on data	)
ADD_CMD(	vvector,	do_vv,		vector-vector operations	)
ADD_CMD(	svector,	do_vs,		scalar-vector operations	)
ADD_CMD(	minmax,		do_minmax,	minimum/maximum routines	)
ADD_CMD(	compare,	docmp,		comparision routines	)
ADD_CMD(	fft,		do_fft,		FFT submenu	)
ADD_CMD(	linear,		do_lin,		linear algebra functions	)
ADD_CMD(	misc,		do_misc,	miscellaneous functions	)
ADD_CMD(	image,		do_imgsyn,	image synthesis functions	)
ADD_CMD(	control,	do_ctl,		warrior control functions	)
ADD_CMD(	sample,		do_samp_menu,	image sampling submenu	)
#ifdef HAVE_MORPH
ADD_CMD(	morph,		do_morph,	morphological operators	)
#endif /* HAVE_MORPH */
ADD_CMD(	requantize,	do_requant,	requantization (dithering) submenu	)

#ifdef FOOBAR
ADD_CMD(	polynomial,	polymenu,	polynomial manipulations	)
#endif /* FOOBAR */

MENU_END(compute)

COMMAND_FUNC(do_comp_menu )
{
	static int inited=0;

	if( ! inited ){
		vl_init(SINGLE_QSP_ARG);
#ifdef FOOBAR
		init_unary_menu();
		init_trig_menu();
		init_log_menu();
		init_vv_menu();
		init_rvs_menu();
		init_min_menu();
		init_imgsyn_menu();
		init_misc_menu();
#ifdef NOT_YET
		init_morph_menu();
#endif /* NOT_YET */
		init_fft_menu();
		init_cmp_menu();
		init_lin_menu();
		init_vec_ctl_menu();
		init_comp_menu();
#endif /* FOOBAR */

		inited=1;
	}

	CHECK_AND_PUSH_MENU(compute);
}

#define VFUNC_FOR_CODE(code)		(&vec_func_tbl[code])

void do_vcode(QSP_ARG_DECL  Vec_Func_Code code)
{
	do_vfunc(QSP_ARG  VFUNC_FOR_CODE(code) );
}

