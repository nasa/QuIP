
#include "quip_config.h"

char VersionId_newvec_warmenu[] = QUIP_VERSION_STRING;

/* #define REQUANT_COLOR */
/* #define REQUANT_ACHROM */
#define HAVE_MORPH		/* why is this a flag??? */

#include <stdio.h>
#include "nvf.h"
#include "data_obj.h"
#include "debug.h"
#include "menuname.h"
#include "version.h"
#include "vec_util.h"
#include "submenus.h"
#include "my_cpuid.h"
#include "query.h"		/* assign_var */

#define DO_VCODE(code)	do_vcode(QSP_ARG  code)

/* local prototypes */

static COMMAND_FUNC( getabs );
static COMMAND_FUNC( getconj );
static COMMAND_FUNC( getmagsq );
static COMMAND_FUNC( domov );
static COMMAND_FUNC( getneg );
static COMMAND_FUNC( do_vset );
static COMMAND_FUNC( getsum );
static COMMAND_FUNC( do_convert );
static COMMAND_FUNC( do_unary );
static COMMAND_FUNC( doatan );
static COMMAND_FUNC( doatn2 );
static COMMAND_FUNC( doatan2 );
static COMMAND_FUNC( docos );
static COMMAND_FUNC( doerf );
static COMMAND_FUNC( doacos );
static COMMAND_FUNC( dopow );
static COMMAND_FUNC( doexp );
static COMMAND_FUNC( dolog );
static COMMAND_FUNC( dolog10 );
static COMMAND_FUNC( dosin );
static COMMAND_FUNC( doasin );
static COMMAND_FUNC( dosqr );
static COMMAND_FUNC( dosqrt );
static COMMAND_FUNC( dotan );
static COMMAND_FUNC( do_trig );
static COMMAND_FUNC( do_and );
static COMMAND_FUNC( do_nand );
static COMMAND_FUNC( do_not );
static COMMAND_FUNC( do_or );
static COMMAND_FUNC( do_xor );
static COMMAND_FUNC( do_logic );
static COMMAND_FUNC( do_vvadd );
static COMMAND_FUNC( do_vvcmul );
static COMMAND_FUNC( do_vvdiv );
static COMMAND_FUNC( do_vvmul );
static COMMAND_FUNC( do_vvsub );
static COMMAND_FUNC( do_vv );

static COMMAND_FUNC( do_vsadd );
static COMMAND_FUNC( do_vssub );
static COMMAND_FUNC( do_vsmul );
static COMMAND_FUNC( do_vsdiv );
static COMMAND_FUNC( do_vsdiv2 );

#ifdef FOOBAR
static COMMAND_FUNC( do_vcsadd );
static COMMAND_FUNC( do_vcssub );
static COMMAND_FUNC( do_vcsmul );
static COMMAND_FUNC( do_vcsdiv );
static COMMAND_FUNC( do_vcsdiv2 );

static COMMAND_FUNC( do_vqsadd );
static COMMAND_FUNC( do_vqssub );
static COMMAND_FUNC( do_vqsmul );
static COMMAND_FUNC( do_vqsdiv );
static COMMAND_FUNC( do_vqsdiv2 );
#endif /* FOOBAR */

#ifdef FOOBAR
static COMMAND_FUNC( do_vscml );
#endif /* FOOBAR */

static COMMAND_FUNC( do_vsmod );
static COMMAND_FUNC( do_vsmod2 );
static COMMAND_FUNC( do_rvs );
#ifdef FOOBAR
static COMMAND_FUNC( do_cvs );
static COMMAND_FUNC( do_qvs );
#endif
static COMMAND_FUNC( domaxg );
static COMMAND_FUNC( domaxi );
static COMMAND_FUNC( domaxv );
static COMMAND_FUNC( doming );
static COMMAND_FUNC( domini );
static COMMAND_FUNC( dominv );
static COMMAND_FUNC( domnmg );
static COMMAND_FUNC( domnmi );
static COMMAND_FUNC( domnmv );
static COMMAND_FUNC( domxmg );
static COMMAND_FUNC( domxmi );
static COMMAND_FUNC( domxmv );
static COMMAND_FUNC( do_minmax );
static COMMAND_FUNC( do_cumsum );

static COMMAND_FUNC( do_reduce );
static COMMAND_FUNC( do_enlarge );
static COMMAND_FUNC( do_prod );
static COMMAND_FUNC( do_fwdfft );
static COMMAND_FUNC( do_fwdrowfft );
static COMMAND_FUNC( do_fwdrfft );
static COMMAND_FUNC( do_fwdrowrfft );
static COMMAND_FUNC( do_invrfft );
static COMMAND_FUNC( do_invrowrfft );
static COMMAND_FUNC( do_invfft );


static COMMAND_FUNC( do_scale );
static COMMAND_FUNC( dolutmap );
static COMMAND_FUNC( do_dither );
static COMMAND_FUNC( doodd );
static COMMAND_FUNC( doeven );
static COMMAND_FUNC( do_fsdither );
static COMMAND_FUNC( do_udither );
static COMMAND_FUNC( do_resample );
static COMMAND_FUNC( do_bilinear );
static COMMAND_FUNC( do_new_bilinear );
static COMMAND_FUNC( do_1dramp );
static COMMAND_FUNC( do_2dramp );
static COMMAND_FUNC( do_wrap );
static COMMAND_FUNC( do_wrap3d );
static COMMAND_FUNC( do_scroll );

static COMMAND_FUNC( do_iconv );

static COMMAND_FUNC( do_histo );
static COMMAND_FUNC( do_integral );
static COMMAND_FUNC( do_mhisto );
static COMMAND_FUNC( do_misc );
static COMMAND_FUNC( do_quads );
static COMMAND_FUNC( do_median );
static COMMAND_FUNC( do_median_1D );
static COMMAND_FUNC( do_median_clip );
static COMMAND_FUNC( do_radavg );
static COMMAND_FUNC( do_oriavg );
static COMMAND_FUNC( do_vinterp );
#ifdef FOOBAR
static COMMAND_FUNC( do_morph );
static COMMAND_FUNC( do_vstitch );
#endif /* FOOBAR */
static COMMAND_FUNC( do_vscmp );
static COMMAND_FUNC( do_vscmp2 );
static COMMAND_FUNC( do_clip );
static COMMAND_FUNC( do_iclip );
static COMMAND_FUNC( do_fft );
static COMMAND_FUNC( do_bnd );
static COMMAND_FUNC( do_ibnd );
static COMMAND_FUNC( do_vcmp );
/* static COMMAND_FUNC( do_vscmm ); */
static COMMAND_FUNC( do_vsmax );
static COMMAND_FUNC( do_vsmxm );
static COMMAND_FUNC( do_vsmin );
static COMMAND_FUNC( do_vsmnm );
/* static COMMAND_FUNC( do_vmcmm ); */
/* static COMMAND_FUNC( do_vmcmp ); */
static COMMAND_FUNC( do_vvvslct );
static COMMAND_FUNC( do_vvsslct );
static COMMAND_FUNC( do_vssslct );
static COMMAND_FUNC( do_vmax );
static COMMAND_FUNC( do_vmin );
static COMMAND_FUNC( do_vmaxm );
static COMMAND_FUNC( do_vminm );
static COMMAND_FUNC( docmp );

#ifdef FOOBAR
static COMMAND_FUNC( do_vmscm );
static COMMAND_FUNC( do_vmscp );
static COMMAND_FUNC( do_corr );
static COMMAND_FUNC( do_cdot );
#endif /* FOOBAR */

static COMMAND_FUNC( do_dot );
static COMMAND_FUNC( do_xpose );
static COMMAND_FUNC( do_invert );
static COMMAND_FUNC( do_ginvert );
static COMMAND_FUNC( do_lin );

#ifdef VECEXP
static COMMAND_FUNC( do_fileparse );
static COMMAND_FUNC( do_parse );
#endif /* VECEXP */


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

	dst= PICK_OBJ( "destination" );
	src= PICK_OBJ( "source" );

	if( dst==NO_OBJ || src == NO_OBJ ) return;

	convert(QSP_ARG  dst,src);
}

COMMAND_FUNC( do_ceil ) { DO_VCODE(FVCEIL); }
COMMAND_FUNC( do_floor ) { DO_VCODE(FVFLOOR); }
COMMAND_FUNC( do_round ) { DO_VCODE(FVROUND); }
COMMAND_FUNC( do_rint ) { DO_VCODE(FVRINT); }
COMMAND_FUNC( do_sign ) { DO_VCODE(FVSIGN); }

COMMAND_FUNC( dowheel )
{
	Data_Obj *dp;
	float arg;
	int n;

	dp=PICK_OBJ( "target float image" );
	n=HOW_MANY("number of spokes");
	arg=HOW_MUCH("spoke phase");
	if( dp==NO_OBJ ) return;

	mkwheel(dp,n,arg);
}

COMMAND_FUNC( doaxle )
{
	Data_Obj *dp;

	dp=PICK_OBJ( "target float image" );
	if( dp==NO_OBJ ) return;

	make_axle(dp);
}

#ifdef FOO
static COMMAND_FUNC( getfind2 )
{
	Data_Obj *dp_src;
	Dimension_Set dimset;
	char *s;
	int sw, sh;
	float* src_data;
	int** ind;
	int i, j, k;
	int thres;

	s = NAMEOF("destination object name");
	if ((dp_src = PICK_OBJ( "source object" )) == NO_OBJ) return;

	thres = HOW_MANY("threshold");

	sw = dp_src->dt_cols;
	sh = dp_src->dt_rows;

	k = 0;
	src_data = (float *)(dp_src->dt_data);
	ind = (int **) malloc(sw * sh * sizeof(int *));
	for (i = 0; i < sw * sh; i++)
		ind[i] = (int *) malloc(2 * sizeof(int));

}
#endif /* FOO */

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

	if ((dp_dst = PICK_OBJ( "destination object" )) == NO_OBJ) return;
	if ((dp_src = PICK_OBJ( "source object" )) == NO_OBJ) return;

	if (dp_dst->dt_prec != PREC_SP || dp_src->dt_prec != PREC_SP) {
		sprintf(ERROR_STRING,"Object types must be float");
		ADVISE(ERROR_STRING);
		return;
	}

	dim = HOW_MANY("summing dimension");

	if (dim > 2 || dim < 1) {
		sprintf(ERROR_STRING,"Dimension can only be 1 (rows) or 2 (cols)");
		ADVISE(ERROR_STRING);
		return;
	}

	dw = dp_dst->dt_cols;
	dh = dp_dst->dt_rows;

	sw = dp_src->dt_cols;
	sh = dp_src->dt_rows;

	src_data = (float *)(dp_src->dt_data);
	dst_data = (float *)(dp_dst->dt_data);

	if (dim == 1) {
		if (dh == sh && dw == 1) {
			for (i = 0; i < sh; i++) {
				dst_data[i] = 0;
				for (j = 0; j < sw; j++)
					dst_data[i] += src_data[i * sw + j];
			}

		} else {
			sprintf(ERROR_STRING,"Check destination object %s dims.", dp_dst->dt_name);
			WARN(ERROR_STRING);
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
			sprintf(ERROR_STRING,"Check destination object %s dims.", dp_dst->dt_name);
			WARN(ERROR_STRING);
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

	if ((dp_dst = PICK_OBJ( "destination object" )) == NO_OBJ) return;
	if ((dp_src = PICK_OBJ( "source object" )) == NO_OBJ) return;

	if (dp_dst->dt_prec != PREC_SP || dp_src->dt_prec != PREC_SP) {
		sprintf(ERROR_STRING,"Object types must be float");
		ADVISE(ERROR_STRING);
		return;
	}

	if ((dw = dp_dst->dt_cols) != 2) {
		sprintf(ERROR_STRING,"Destination object must have exactly 2 columns");
		ADVISE(ERROR_STRING);
		return;
	}

	thres = HOW_MANY("threshold");
	dh = dp_dst->dt_rows;
	sw = dp_src->dt_cols;
	sh = dp_src->dt_rows;

	k = 0;
	src_data = (float *)(dp_src->dt_data);
	dst_data = (float *)(dp_dst->dt_data);

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
		ADVISE(ERROR_STRING);
	}

}


Command un_ctbl[]={			/* unary op's */
{ "abs",	getabs,		"convert to absolute value"		},
{ "conj",	getconj,	"convert to complex conjugate"		},
{ "find",	dofind,		"return indeces of non-zero elements"	},
{ "dimsum",	getDimSum,	"return sum along columns (rows)"	},
{ "mov",	domov,		"copy data"				},
{ "neg",	getneg,		"convert to negative"			},
{ "set",	do_vset,	"set vector to a constant value"	},
{ "sum",	getsum,		"get sum of vector"			},
{ "sign",	do_sign,	"take sign of vector"			},
{ "rint",	do_rint,	"round nearest integer using rint()"	},
{ "round",	do_round,	"round nearest integer using round()"	},
{ "floor",	do_floor,	"take floor of vector"			},
{ "ceil",	do_ceil,	"take ceil of vector"			},
{ "convert",	do_convert,	"convert vectors"			},
{ "uni",	do_uni,		"uniform random numbers"		},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"				},
#endif
{ NULL_COMMAND								}
};

static COMMAND_FUNC( do_unary )
{
	PUSHCMD(un_ctbl,UNARY_MENU_NAME);
}

static COMMAND_FUNC( doatan ){	DO_VCODE(FVATAN); }
static COMMAND_FUNC( doatn2 ){	DO_VCODE(FVATN2); }
static COMMAND_FUNC( doatan2 ){	DO_VCODE(FVATAN2); }
static COMMAND_FUNC( getmagsq ){	DO_VCODE(FVMGSQ); }
static COMMAND_FUNC( docos ){	DO_VCODE(FVCOS); }
static COMMAND_FUNC( doerf ){	DO_VCODE(FVERF); }
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
static COMMAND_FUNC( do_j0 ){	DO_VCODE(FVJ0); }
static COMMAND_FUNC( do_j1 ){	DO_VCODE(FVJ1); }

Command trig_ctbl[]={
{ "atan",	doatan,		"compute arc tangent"			},
{ "atn2",	doatn2,		"compute arc tangent (single complex arg)"	},
{ "atan2",	doatan2,	"compute arc tangent (two real args)"	},
{ "magsq",	getmagsq,	"convert to magnitude squared"		},
{ "cos",	docos,		"compute cosine"			},
{ "erf",	doerf,		"compute error function (erf)"			},
{ "exp",	doexp,		"exponentiate (base e)"			},
{ "log",	dolog,		"natural logarithm"			},
{ "log10",	dolog10,	"logarithm base 10"			},
{ "sin",	dosin,		"compute sine"				},
{ "square",	dosqr,		"compute square"			},
{ "sqrt",	dosqrt,		"compute square root"			},
{ "tan",	dotan,		"compute tangent"			},
{ "pow",	dopow,		"raise to a power"			},
{ "acos",	doacos,		"compute inverse cosine"		},
{ "asin",	doasin,		"compute inverse sine"			},
#ifndef MAC
{ "j0",		do_j0,		"compute bessel function J0"		},
{ "j1",		do_j1,		"compute bessel function J1"		},
/* { "bessel",	do_j0,		"compute J0"				},	*/
{ "quit",	popcmd,		"exit submenu"				},
#endif
{ NULL_COMMAND								}
};

static COMMAND_FUNC( do_trig )
{
	PUSHCMD(trig_ctbl,TRIG_MENU_NAME);
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

Command log_ctbl[]={
{ "comp",	do_comp,	"bitwise complement"		},
{ "and",	do_and,		"logical AND"			},
{ "nand",	do_nand,	"logical NAND"			},
{ "not",	do_not,		"logical NOT"			},
{ "or",		do_or,		"logical OR"			},
{ "xor",	do_xor,		"logical XOR"			},
{ "sand",	do_sand,	"logical AND with scalar"	},
{ "sor",	do_sor,		"logical OR with scalar"	},
{ "sxor",	do_sxor,	"logical XOR with scalar"	},
{ "shr",	do_shr,		"right shift"			},
{ "shl",	do_shl,		"left shift"			},
{ "sshr",	do_sshr,	"right shift by a constant"	},
{ "sshl",	do_sshl,	"left shift by a constant"	},
{ "sshr2",	do_sshr2,	"right shift a constant"	},
{ "sshl2",	do_sshl2,	"left shift a constant"		},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"			},
#endif
{ NULL_COMMAND							}
};

static COMMAND_FUNC( do_logic )
{
	PUSHCMD(log_ctbl,LOG_MENU_NAME);
}

static COMMAND_FUNC( do_vvadd ){	DO_VCODE(FVADD); }
static COMMAND_FUNC( do_vvcmul ){	DO_VCODE(FVCMUL); }
static COMMAND_FUNC( do_vvdiv ){	DO_VCODE(FVDIV); }
static COMMAND_FUNC( do_vvmul ){	DO_VCODE(FVMUL); }
static COMMAND_FUNC( do_vvsub ){	DO_VCODE(FVSUB); }

Command vv_ctbl[]={
{ "add",	do_vvadd,	"vector addition"			},
{ "cmul",	do_vvcmul,	"multiply by complex conjugate"		},
{ "div",	do_vvdiv,	"element by element division"		},
{ "mul",	do_vvmul,	"element by element multiplication"	},
{ "sub",	do_vvsub,	"vector subtraction"			},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"				},
#endif
{ NULL_COMMAND								}
};

static COMMAND_FUNC( do_vv )
{
	PUSHCMD(vv_ctbl,VV_MENU_NAME);
}

static COMMAND_FUNC( do_vsadd )		{	insist_real=1; DO_VCODE(FVSADD); }
static COMMAND_FUNC( do_vssub )		{	insist_real=1; DO_VCODE(FVSSUB); }
static COMMAND_FUNC( do_vsmul )		{	insist_real=1; DO_VCODE(FVSMUL); }
static COMMAND_FUNC( do_vsdiv )		{	insist_real=1; DO_VCODE(FVSDIV); }
static COMMAND_FUNC( do_vsdiv2 )		{	insist_real=1; DO_VCODE(FVSDIV2); }

/* static COMMAND_FUNC( do_vscml )	{	DO_VCODE(FVSCML); } */

static COMMAND_FUNC( do_vsmod )		{	DO_VCODE(FVSMOD); }
static COMMAND_FUNC( do_vsmod2 )		{	DO_VCODE(FVSMOD2); }

#ifdef FOOBAR
static COMMAND_FUNC( do_vcsadd )	{	insist_cpx=1;  DO_VCODE(FVCSADD); }
static COMMAND_FUNC( do_vcssub )	{	insist_cpx=1;  DO_VCODE(FVCSSUB); }
static COMMAND_FUNC( do_vcsmul )	{	insist_cpx=1;  DO_VCODE(FVCSMUL); }
static COMMAND_FUNC( do_vcsdiv )	{	insist_cpx=1;  DO_VCODE(FVCSDIV); }
static COMMAND_FUNC( do_vcsdiv2 )	{	insist_cpx=1;  DO_VCODE(FVCSDIV2); }

static COMMAND_FUNC( do_vqsadd )	{	insist_quat=1;  DO_VCODE(FVQSADD); }
static COMMAND_FUNC( do_vqssub )	{	insist_quat=1;  DO_VCODE(FVQSSUB); }
static COMMAND_FUNC( do_vqsmul )	{	insist_quat=1;  DO_VCODE(FVQSMUL); }
static COMMAND_FUNC( do_vqsdiv )	{	insist_quat=1;  DO_VCODE(FVQSDIV); }
static COMMAND_FUNC( do_vqsdiv2 )	{	insist_quat=1;  DO_VCODE(FVQSDIV2); }
#endif /* FOOBAR */

static COMMAND_FUNC( do_vspow )		{	DO_VCODE(FVSPOW); }
static COMMAND_FUNC( do_vspow2 )	{	DO_VCODE(FVSPOW2); }
static COMMAND_FUNC( do_vsatan2 )	{	DO_VCODE(FVSATAN2); }
static COMMAND_FUNC( do_vsatan22 )	{	DO_VCODE(FVSATAN22); }
static COMMAND_FUNC( do_vsand )		{	DO_VCODE(FVSAND); }
static COMMAND_FUNC( do_vsor )		{	DO_VCODE(FVSOR); }
static COMMAND_FUNC( do_vsxor )		{	DO_VCODE(FVSXOR); }

Command rvs_ctbl[]={
{ "add",	do_vsadd,	"add scalar to elements of a vector"	},
{ "sub",	do_vssub,	"subtract elts. of a vector from a scalar"},
{ "div",	do_vsdiv,	"divide a scalar by the elements of a vector"},
{ "div2",	do_vsdiv2,	"divide elements of a vector by a scalar"},
{ "mul",	do_vsmul,	"multiply a vector by a real scalar"	},
{ "mod",	do_vsmod,	"integer modulo of a vector by a real scalar"	},
{ "mod2",	do_vsmod2,	"integer modulo of a real scalar by a vector"	},
{ "pow",	do_vspow,	"raise the elements of a vector to a scalar power"	},
{ "pow2",	do_vspow2,	"raise a scalar to powers given by the elements of a vector" },
{ "atan2",	do_vsatan2,	"compute 4-quadrant arc tangent of vector and scalar"	},
{ "atan22",	do_vsatan22,	"compute 4-quadrant arc tangent of scalar and vector"	},
{ "and",	do_vsand,	"bitwise and of scalar and vector"	},
{ "or",		do_vsor,	"bitwise or of scalar and vector"	},
{ "xor",	do_vsxor,	"bitwise xor of scalar and vector"	},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"				},
#endif
{ NULL_COMMAND								}
};

static COMMAND_FUNC( do_rvs ) { PUSHCMD(rvs_ctbl,SV_MENU_NAME); }

#ifdef FOOBAR
Command cvs_ctbl[]={
{ "add",	do_vcsadd,	"add complex scalar to elements of a vector"	},
{ "div",	do_vcsdiv,	"divide a complex scalar by the elements of a vector"},
{ "div2",	do_vcsdiv2,	"divide elements of a vector by a complex scalar"},
{ "mul",	do_vcsmul,	"multiply a vector by a complex scalar"	},
/* { "conjmul",	do_vscml,	"multiply vector conj. by a complex scalar"},	*/
{ "sub",	do_vcssub,	"subtract elements of a vector from a complex scalar"},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"				},
#endif
{ NULL_COMMAND								}
};

Command qvs_ctbl[]={
{ "add",	do_vqsadd,	"add quaternion scalar to elements of a vector"	},
{ "div",	do_vqsdiv,	"divide a quaternion scalar by the elements of a vector"},
{ "div2",	do_vqsdiv2,	"divide elements of a vector by a quaternion scalar"},
{ "mul",	do_vqsmul,	"multiply a vector by a quaternion scalar"	},
/* { "conjmul",	do_vscml,	"multiply vector conj. by a quaternion scalar"},	*/
{ "sub",	do_vqssub,	"subtract elements of a vector from a quaternion scalar"},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"				},
#endif
{ NULL_COMMAND								}
};

static COMMAND_FUNC( do_cvs ) { PUSHCMD(cvs_ctbl,CSV_MENU_NAME); }
static COMMAND_FUNC( do_qvs ) { PUSHCMD(qvs_ctbl,QSV_MENU_NAME); }
#endif	/* FOOBAR */

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

Command min_ctbl[]={
{ "max",	domaxv,		"find maximum value"			},
{ "min",	dominv,		"find minimum value"			},
{ "max_mag",	domxmv,		"find maximum absolute value"		},
{ "min_mag",	domnmv,		"find minimum absolute value"		},
{ "max_index",	domaxi,		"find index of maximum value"		},
{ "min_index",	domini,		"find index of minimum value"		},
{ "max_mag_index",domxmi,	"find index of maximum absolute value"	},
{ "min_mag_index",domnmi,	"find index of minimum absolute value"	},
{ "max_times",	domaxg,		"find index of maximum & # of occurrences"},
{ "min_times",	doming,		"find index of minimum & # of occurrences"},
{ "max_mag_times",domxmg,	"find index of max. mag. & # of occurrences"},
{ "min_mag_times",domnmg,	"find index of min. mag. & # of occurrences"},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"				},
#endif
{ NULL_COMMAND								}
};

static COMMAND_FUNC( do_minmax )
{
	PUSHCMD( min_ctbl,MINMAX_MENU_NAME);
}

#ifdef FOOBAR
static COMMAND_FUNC( do_accumulate )
{
	Data_Obj *dp_to,*dp_fr;

	dp_to=PICK_OBJ( "destination vector" );
	dp_fr=PICK_OBJ( "source vector" );

	if( dp_to==NO_OBJ || dp_fr==NO_OBJ ) return;

	war_accumulate(dp_to,dp_fr);
}

static COMMAND_FUNC( do_project )
{
	Data_Obj *dp_to,*dp_fr;

	dp_to=PICK_OBJ( "destination vector" );
	dp_fr=PICK_OBJ( "source image" );

	if( dp_to==NO_OBJ || dp_fr==NO_OBJ ) return;

	war_project(dp_to,dp_fr);
}
#endif /* FOOBAR */


static COMMAND_FUNC( do_cumsum )
{
	Data_Obj *dp_to,*dp_fr;

	dp_to=PICK_OBJ( "destination vector" );
	dp_fr=PICK_OBJ( "source vector" );

	if( dp_to==NO_OBJ || dp_fr==NO_OBJ ) return;

	war_cumsum(QSP_ARG  dp_to,dp_fr);
}

static COMMAND_FUNC( do_reduce )
{
	Data_Obj *dp,*dp2;

	dp2=PICK_OBJ( "destination image" );
	dp=PICK_OBJ( "source image" );
	if( dp2==NO_OBJ || dp == NO_OBJ ) return;
	reduce(QSP_ARG  dp2,dp);
}

static COMMAND_FUNC( do_enlarge )
{
	Data_Obj *dp,*dp2;

	dp2=PICK_OBJ( "destination image" );
	dp=PICK_OBJ( "source image" );
	if( dp2==NO_OBJ || dp == NO_OBJ ) return;
	enlarge(QSP_ARG  dp2,dp);
}

#ifdef FOOBAR
static COMMAND_FUNC( do_reduce )
{ WARN("no reduce"); }

static COMMAND_FUNC( do_enlarge )
{ WARN("no enlarge"); }
#endif /* FOOBAR */



static COMMAND_FUNC( do_prod )
{
	Data_Obj *target, *rowobj, *colobj;
	Vec_Obj_Args oargs;


	target=PICK_OBJ( "target image" );
	rowobj=PICK_OBJ("row vector");
	colobj=PICK_OBJ("column vector");

	if( target == NO_OBJ || rowobj == NO_OBJ || colobj == NO_OBJ )
		return;

	oargs.oa_dest = target;
	oargs.oa_dp[0]=rowobj;
	oargs.oa_dp[1]=colobj;
	oargs.oa_argstype = REAL_ARGS;	/* BUG check source for real or complex? */

	vmul(&oargs);
}

#ifdef FOOBAR
static COMMAND_FUNC( do_nanchk )
{
	Data_Obj *dp;

	dp=PICK_OBJ("image");
	if( dp == NO_OBJ ) return;
	nan_chk(dp);
}
#endif /* FOOBAR */

static COMMAND_FUNC( do_fwdfft )
{
	Data_Obj *dp;

	dp=PICK_OBJ("complex image");
	if( dp == NO_OBJ ) return;

	fft2d(dp,dp);
}

static COMMAND_FUNC( do_fwdrowfft )
{
	Data_Obj *dp;

	dp=PICK_OBJ("complex image");
	if( dp == NO_OBJ ) return;

	fftrows(dp,dp);
}

static COMMAND_FUNC( do_invfft )
{
	Data_Obj *dp;

	dp=PICK_OBJ("complex image");
	if( dp == NO_OBJ ) return;
	ift2d(dp,dp);
}

static COMMAND_FUNC( do_invrowfft )
{
	Data_Obj *dp;

	dp=PICK_OBJ("complex image");
	if( dp == NO_OBJ ) return;
	iftrows(dp,dp);
}

static COMMAND_FUNC( do_invrfft )
{
	Data_Obj *src,*targ;

	targ = PICK_OBJ("real target");
	src = PICK_OBJ("complex source");
	if( targ == NO_OBJ || src == NO_OBJ ) return;

	ift2d(targ,src);
}

static COMMAND_FUNC( do_invrowrfft )
{
	Data_Obj *src,*targ;

	targ = PICK_OBJ("real target");
	src = PICK_OBJ("complex source");
	if( targ == NO_OBJ || src == NO_OBJ ) return;

	iftrows(targ,src);
}

/* There is no need now for separate r and c fft commands, because the selection
 * is done automatically - The commands are retained for backwards compatibility.
 */

static COMMAND_FUNC( do_fwdrfft )
{
	Data_Obj *src,*targ;

	targ = PICK_OBJ("complex target");
	src = PICK_OBJ("real source");
	if( targ == NO_OBJ || src == NO_OBJ ) return;

	fft2d(targ,src);
}

static COMMAND_FUNC( do_fwdrowrfft )
{
	Data_Obj *src,*targ;

	targ = PICK_OBJ("complex target");
	src = PICK_OBJ("real source");
	if( targ == NO_OBJ || src == NO_OBJ ) return;

	fftrows(targ,src);
}


#ifdef FOOBAR
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

	gdp = PICK_OBJ( "source image" );
	if( gdp == NO_OBJ ) return;
	set_grayscale(gdp);
}


static COMMAND_FUNC( do_set_output )
{
	Data_Obj *hdp;

	hdp = PICK_OBJ( "output image" );
	if( hdp == NO_OBJ ) return;
	set_halftone(hdp);
}

static COMMAND_FUNC( do_set_filter )
{
	Data_Obj *fdp;

	fdp = PICK_OBJ( "filter image" );
	if( fdp == NO_OBJ ) return;
	set_filter(fdp);
}

static COMMAND_FUNC( do_init_images )
{
	if( setup_requantize() == -1 )
		WARN("error setting up images");
}

static COMMAND_FUNC( do_init_requant )
{
	if( setup_requantize() == -1 ) return;
	init_requant();
}

static COMMAND_FUNC( do_qt_dither )
{
	Data_Obj *dpto, *dpfr;

	dpto = PICK_OBJ( "target image" );
	dpfr = PICK_OBJ( "source image" );

	if( dpto == NO_OBJ || dpfr == NO_OBJ ) return;

	qt_dither(dpto,dpfr);
}

static COMMAND_FUNC( do_tweak )
{
	int x,y;

	x=HOW_MANY("x coord");
	y=HOW_MANY("y coord");

	redo_two_pixels(x,y);
}

Command req_ctbl[]={
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

#endif /* FOOBAR */

#endif /* REQUANT_ACHROM */


#ifdef REQUANT_COLOR

static COMMAND_FUNC( do_init_requant )
{
	init_requant();
}

static COMMAND_FUNC( do_set_input )
{
	Data_Obj *lumdp, *rgdp, *bydp;

	lumdp = PICK_OBJ( "luminance image" );
	rgdp = PICK_OBJ( "red-green image" );
	bydp = PICK_OBJ( "blue-yellow image" );

	if( lumdp==NO_OBJ || rgdp==NO_OBJ || bydp==NO_OBJ ) return;

	set_rgb_input(lumdp,rgdp,bydp);
}

static COMMAND_FUNC( do_set_output )
{
	Data_Obj *hdp;

	hdp = PICK_OBJ( "byte image for composite halftone" );
	if( hdp==NO_OBJ ) return;
	set_rgb_output(hdp);
}

static COMMAND_FUNC( do_set_filter )
{
	Data_Obj *rdp, *gdp, *bdp;

	rdp = PICK_OBJ( "red filter image" );
	gdp = PICK_OBJ( "green filter image" );
	bdp = PICK_OBJ( "blue filter image" );

	if( rdp==NO_OBJ || gdp==NO_OBJ || bdp==NO_OBJ ) return;

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

	matrix = PICK_OBJ( "transformation matrix" );
	if( matrix == NO_OBJ ) return;
	set_xform(matrix);
}

static COMMAND_FUNC( do_descend )
{
	scan_requant(1);
}

Command req_ctbl[]={
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

	dp=PICK_OBJ( "float image" );
	mn=HOW_MUCH("desired min value");
	mx=HOW_MUCH("desired max value");
	if( dp==NO_OBJ ) return;

	scale(QSP_ARG  dp,mn,mx);
}

static COMMAND_FUNC( do_dither )
{
	Data_Obj *dp;
	int size;

	dp=PICK_OBJ( "float image" );
	size=HOW_MANY("size of dither matrix");
	if( dp==NO_OBJ ) return;
	odither(dp,size);
}

static COMMAND_FUNC( do_1dramp ) { DO_VCODE(FVRAMP1D); }
static COMMAND_FUNC( do_2dramp ) { DO_VCODE(FVRAMP2D); }

#ifdef FOOBAR
static COMMAND_FUNC( do_2dramp )
{
	double s,dx,dy;
	Vec_Obj_Args oargs;

	oargs.oa_dp[0]=oargs.oa_dest = PICK_OBJ("target image");
	oargs.oa_dp[1]=
	oargs.oa_dp[2]=NO_OBJ;

	if( oargs.oa_dest==NO_OBJ ) return;
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
		default:  WARN("missing prec case in ramp2d"); break;
	}

	ramp2d(&oargs);
}
#endif /* FOOBAR */

static COMMAND_FUNC( do_wrap )
{
	Data_Obj *dpto,*dpfr;

	dpto=PICK_OBJ("destination image");
	dpfr=PICK_OBJ("source image");
	if( dpto==NO_OBJ || dpfr==NO_OBJ ) return;
	wrap(QSP_ARG  dpto,dpfr);
}

static COMMAND_FUNC( do_wrap3d )
{
	Data_Obj *dpto,*dpfr;

	dpto=PICK_OBJ("destination image");
	dpfr=PICK_OBJ("source image");
	if( dpto==NO_OBJ || dpfr==NO_OBJ ) return;
#ifdef FOOBAR
	wrap3d(QSP_ARG  dpto,dpfr);
#else
	WARN("no wrap3d yet");
#endif
}

static COMMAND_FUNC( do_scroll )
{
	Data_Obj *dpto,*dpfr;
	int dx,dy;

	dpto=PICK_OBJ("destination image");
	dpfr=PICK_OBJ("source image");
	dx=HOW_MANY("x displacement");
	dy=HOW_MANY("y displacement");
	if( dpto==NO_OBJ || dpfr==NO_OBJ ) return;
	dp_scroll(QSP_ARG  dpto,dpfr,dx,dy);
}


static COMMAND_FUNC( doodd )
{
	Data_Obj *dp;

	dp=PICK_OBJ( "name of image" );
	if( dp == NO_OBJ ) return;
	mkodd(dp);
}

static COMMAND_FUNC( doeven )
{
	Data_Obj *dp;

	dp=PICK_OBJ( "name of image" );
	if( dp == NO_OBJ ) return;
	mkeven(dp);
}

static COMMAND_FUNC( dolutmap )
{
	Data_Obj *dst, *src, *map;

	dst=PICK_OBJ( "destination image" );
	src=PICK_OBJ( "source byte image" );
	map=PICK_OBJ( "lut vector" );
	if( dst==NO_OBJ || src==NO_OBJ || map==NO_OBJ )
		return;
	if( lutmap(QSP_ARG  dst,src,map) == (-1) )
		WARN("mapping failed");
}

#define MAX_FS_LEVELS	128

static COMMAND_FUNC( do_fsdither )
{
	Data_Obj *dpto, *dpfr;
	int n;
	float lvl[MAX_FS_LEVELS];
	int i;

	dpto=PICK_OBJ( "target byte image" );
	dpfr=PICK_OBJ( "source image" );
	n=HOW_MANY("number of quantization levels");
	if( n<2 || n > MAX_FS_LEVELS ){
		WARN("bad number of halftone levels");
		return;
	}
	for(i=0;i<n;i++)
		lvl[i] = HOW_MUCH("level value");

	if( dpto == NO_OBJ || dpfr == NO_OBJ )
		return;
	dp_halftone(QSP_ARG  dpto,dpfr,n,lvl);
}

static COMMAND_FUNC( do_udither )		/* uniform quantization */
{
	Data_Obj *dpto, *dpfr;
	int n;
	float minlvl,maxlvl,lvl[MAX_FS_LEVELS];
	int i;

	dpto=PICK_OBJ( "target byte image" );
	dpfr=PICK_OBJ( "source image" );
	n=HOW_MANY("number of quantization levels");
	minlvl = HOW_MUCH("minimum level value");
	maxlvl = HOW_MUCH("maximum level value");
	if( n<2 || n > MAX_FS_LEVELS ){
		WARN("bad number of halftone levels");
		return;
	}
	for(i=0;i<n;i++){
		lvl[i] = ((n-1)-i)*minlvl + i*maxlvl;
		lvl[i] /= (n-1);
	}

	if( dpto == NO_OBJ || dpfr == NO_OBJ )
		return;
	dp_halftone(QSP_ARG  dpto,dpfr,n,lvl);
}

static COMMAND_FUNC( do_resample )
{
	Data_Obj *dpto, *dpfr, *dpwarp;

	dpto=PICK_OBJ( "target float image" );
	dpfr=PICK_OBJ( "source float image" );
	dpwarp=PICK_OBJ( "complex control image" );
	if( dpto==NO_OBJ || dpfr==NO_OBJ || dpwarp==NO_OBJ )
		return;

	resample(QSP_ARG  dpto,dpfr,dpwarp);
}

static COMMAND_FUNC( do_bilinear )
{
	Data_Obj *dpto, *dpfr, *dpwarp;

	dpto=PICK_OBJ( "target float image" );
	dpfr=PICK_OBJ( "source float image" );
	dpwarp=PICK_OBJ( "complex control image" );
	if( dpto==NO_OBJ || dpfr==NO_OBJ || dpwarp==NO_OBJ )
		return;

	bilinear_warp(QSP_ARG  dpto,dpfr,dpwarp);
}

static COMMAND_FUNC( do_new_bilinear )
{
	Data_Obj *dpto, *dpfr, *dpwarp;

	dpto=PICK_OBJ( "target float image" );
	dpfr=PICK_OBJ( "source float image" );
	dpwarp=PICK_OBJ( "complex control image" );
	if( dpto==NO_OBJ || dpfr==NO_OBJ || dpwarp==NO_OBJ )
		return;

	new_bilinear_warp(QSP_ARG  dpto,dpfr,dpwarp);
}

static COMMAND_FUNC( do_iconv )
{
	Data_Obj *dpto, *dpfr, *dpfilt;

	dpto = PICK_OBJ( "target image" );
	dpfr = PICK_OBJ( "source image" );
	dpfilt = PICK_OBJ( "filter image" );

	if( dpto==NO_OBJ || dpfr==NO_OBJ || dpfilt==NO_OBJ )
		return;

	convolve(QSP_ARG  dpto,dpfr,dpfilt);
}

static COMMAND_FUNC( do_iconv3d )
{
	Data_Obj *dpto, *dpfr, *dpfilt;

	dpto = PICK_OBJ( "target image" );
	dpfr = PICK_OBJ( "source image" );
	dpfilt = PICK_OBJ( "filter image" );

	if( dpto==NO_OBJ || dpfr==NO_OBJ || dpfilt==NO_OBJ )
		return;

	convolve3d(QSP_ARG  dpto,dpfr,dpfilt);
}

static COMMAND_FUNC( do_histo )
{
	Data_Obj *dp, *hdp;
	double bw, minbin;

	hdp = PICK_OBJ( "vector for histogram data" );
	dp = PICK_OBJ( "source data object" );
	minbin = HOW_MUCH("minimum bin center");
	bw = HOW_MUCH("bin width");

	if( hdp == NO_OBJ || dp == NO_OBJ ) return;

	compute_histo(QSP_ARG  hdp,dp,bw,minbin);
}

static COMMAND_FUNC( do_integral )
{
	Data_Obj *dst, *src;

	dst = PICK_OBJ( "destination image" );
	src = PICK_OBJ( "source image" );

	if( dst == NO_OBJ || src == NO_OBJ ) return;

	cum_sum(QSP_ARG  dst,src);
}

static COMMAND_FUNC( do_hough )
{
	Data_Obj *dst, *src;
	float thresh,x0,y0;

	dst = PICK_OBJ("destination image for transform");
	src = PICK_OBJ("source image");
	thresh = HOW_MUCH("threshold");
	x0 = HOW_MUCH("x origin");
	y0 = HOW_MUCH("y origin");

	hough(QSP_ARG  dst,src,thresh,x0,y0);
}

static COMMAND_FUNC( do_local_max )
{
	Data_Obj *val_dp, *coord_dp, *src;
	long n;

	val_dp = PICK_OBJ("destination vector for local maximum values");
	coord_dp = PICK_OBJ("destination vector for coordinates");
	src = PICK_OBJ("source image");

	n = local_maxima(QSP_ARG  val_dp,coord_dp,src);

	sprintf(msg_str,"%ld",n);
	ASSIGN_VAR("n_maxima",msg_str);
}


#define MAX_DIMENSIONS	(N_DIMENSIONS-1)

static COMMAND_FUNC( do_mhisto )
{
	Data_Obj *dp, *hdp;
	float bw[MAX_DIMENSIONS], minbin[MAX_DIMENSIONS];
	dimension_t i;

	hdp = PICK_OBJ( "target histogram data object" );
	dp = PICK_OBJ( "source data object" );
	if( dp == NO_OBJ ) return;
	for(i=0;i<dp->dt_comps;i++){
		minbin[i] = HOW_MUCH("minimum bin center");
		bw[i] = HOW_MUCH("bin width");
	}

	if( hdp == NO_OBJ || dp == NO_OBJ ) return;

	multivariate_histo(QSP_ARG  hdp,dp,bw,minbin);
}

#ifdef HAVE_MORPH

static COMMAND_FUNC( do_fill )
{
	Data_Obj *dp;
	dimension_t x,y;
	double val;
	double tol;

	dp=PICK_OBJ("image");
	x=HOW_MANY("seed x");
	y=HOW_MANY("seed y");
	val=HOW_MUCH("value");
	tol=HOW_MUCH("tolerance");

	if( dp == NO_OBJ ) return;
	if( tol < 0 ){
		WARN("tolerance must be non-negative");
		return;
	}

	ifl(QSP_ARG  dp,x,y,val,tol);
}
#endif /* HAVE_MORPH */

static COMMAND_FUNC( do_quads )
{
	Data_Obj *src, *dst;

	dst=PICK_OBJ( "destination 4-tuple list" );
	src=PICK_OBJ( "source image" );
	if( dst == NO_OBJ || src == NO_OBJ ) return;

	make_all_quads(QSP_ARG  dst,src);
}

static COMMAND_FUNC( do_ext_paths )
{
	Data_Obj *src, *dst;

	dst=PICK_OBJ( "destination matrix" );
	src=PICK_OBJ( "source matrix" );
	if( dst == NO_OBJ || src == NO_OBJ ) return;

	extend_shortest_paths(QSP_ARG  dst,src);
}

#ifdef FOOBAR
static COMMAND_FUNC( do_vstitch )
{
	Data_Obj *to,*fr,*co;
	int n;

	to=PICK_OBJ( "target float vector" );
	fr=PICK_OBJ( "source float vector" );
	co=PICK_OBJ( "control float vector" );
	if( to==NO_OBJ || fr==NO_OBJ || co==NO_OBJ ) return;
	n=vstitch(QSP_ARG  to,fr,co);
	if( verbose ){
		sprintf(ERROR_STRING,"%d elements copied",n);
		ADVISE(ERROR_STRING);
	}
}
#endif /* FOOBAR */

static COMMAND_FUNC( do_vinterp )
{
	Data_Obj *to,*fr,*co;

	to=PICK_OBJ( "target float vector" );
	fr=PICK_OBJ( "source float vector" );
	co=PICK_OBJ( "control float vector" );
	if( to==NO_OBJ || fr==NO_OBJ || co==NO_OBJ ) return;
	vinterp(QSP_ARG  to,fr,co);
}

static COMMAND_FUNC( do_median )
{
	Data_Obj *to, *fr;

	to=PICK_OBJ("target");
	fr=PICK_OBJ("source");
	if( to==NO_OBJ || fr==NO_OBJ ) return;
	median(QSP_ARG  to,fr);
}

static COMMAND_FUNC( do_median_clip )
{
	Data_Obj *to, *fr;

	to=PICK_OBJ("target");
	fr=PICK_OBJ("source");
	if( to==NO_OBJ || fr==NO_OBJ ) return;
	median_clip(QSP_ARG  to,fr);
}

static COMMAND_FUNC( do_median_1D )
{
	Data_Obj *to, *fr;
	int rad;

	to=PICK_OBJ("target");
	fr=PICK_OBJ("source");
	rad = HOW_MANY("radius");

	if( to==NO_OBJ || fr==NO_OBJ ) return;

	median_1D(QSP_ARG  to,fr,rad);
}

static COMMAND_FUNC( do_krast )
{
	Data_Obj *dp;

	dp=PICK_OBJ("coord list");
	if( dp != NO_OBJ )
		mk_krast(QSP_ARG  dp);
}

Command imgsyn_ctbl[]={
{ "ramp1d",	do_1dramp,		"make a 1-D ramp"		},
{ "ramp2d",	do_2dramp,		"make a ramp image"		},
{ "product",	do_prod,		"make product image"		},
{ "diffuse",	do_fsdither,		"dither image, arbitrary levels"},
{ "udiffuse",	do_udither,		"dither image, uniform levels"	},
{ "wheel",	dowheel,		"make a wheel image"		},
{ "axle",	doaxle,			"put a 1 at 0 freq"		},
{ "odd",	doodd,			"make image odd"		},
{ "even",	doeven,			"make image even"		},
#ifndef MAC
{ "quit",	popcmd,			"exit submenu"			},
#endif
{ NULL_COMMAND								}
};

static COMMAND_FUNC( do_imgsyn )
{
	PUSHCMD(imgsyn_ctbl,IMGSYN_MENU_NAME);
}

static COMMAND_FUNC( do_sort )
{
	Data_Obj *dp;

	dp=PICK_OBJ("");
	if( dp == NO_OBJ ) return;

	sort_data(QSP_ARG  dp);
}

static COMMAND_FUNC( do_sort_indices )
{
	Data_Obj *dp1,*dp2;

	dp1=PICK_OBJ("array of indices");
	dp2=PICK_OBJ("data array");
	if( dp1 == NO_OBJ || dp2 == NO_OBJ ) return;
	sort_indices(QSP_ARG  dp1,dp2);
}

static COMMAND_FUNC( do_scramble )
{
	Data_Obj *dp;

	dp = PICK_OBJ("");
	if( dp == NO_OBJ ) return;

	dp_scramble(QSP_ARG  dp);
}

static COMMAND_FUNC( do_yuv2rgb )
{
	Data_Obj *dst_dp, *src_dp;

	dst_dp = PICK_OBJ("destination rgb image");
	src_dp = PICK_OBJ("source yuv image");

	if( dst_dp == NO_OBJ || src_dp == NO_OBJ ) return;

	yuv422_to_rgb24(dst_dp, src_dp );
}

static COMMAND_FUNC( do_yuv2gray )
{
	Data_Obj *dst_dp, *src_dp;

	dst_dp = PICK_OBJ("destination grayscale image");
	src_dp = PICK_OBJ("source yuv image");

	if( dst_dp == NO_OBJ || src_dp == NO_OBJ ) return;

	yuv422_to_gray(dst_dp, src_dp );
}


Command misc_ctbl[]={
{ "krast",	do_krast,		"compute coords of space-filling curve"	},
{ "cumsum",	do_cumsum,		"computes cumulative sum of a vector"	},
{ "reduce",	do_reduce,		"reduce an image"		},
{ "enlarge",	do_enlarge,		"enlarge an image"		},
{ "interpolate",do_vinterp,		"interpolate gaps in a vector"	},
{ "sort",	do_sort,		"sort elements of a vector in-place"	},
{ "sort_indices",do_sort_indices,	"sort indices of data array"		},
{ "scramble",	do_scramble,		"permute elements of an (unsigned short) object"	},
#ifdef FOOBAR
{ "project",	do_project,		"project columns of an image"	},
{ "shrink",	do_shrink,		"shrink image in place by half"	},
{ "stitch",	do_vstitch,		"cut out unwanted bits of a vector"},
{ "accumulate",	do_accumulate,		"compute cumulative sum of a vector"	},
{ "finite",	do_nanchk,		"verify finite values"		},
#endif /* FOOBAR */
	/*
{ "transform",	cmxform,		"matrix transformation of cpx data"},
	*/
{ "scale",	do_scale,		"scale float image to byte range"},
{ "histogram",	do_histo,		"compute histogram of data"	},
{ "integral",	do_integral,		"compute integral image (cumulative sum)"	},
{ "hough",	do_hough,		"compute Hough transform"	},
{ "local_maxima",	do_local_max,		"find local maxima"	},
{ "multivariate",do_mhisto,		"multivariate histogram of data"},
{ "dither",	do_dither,		"initialize ordered dither matrix"},
{ "median",	do_median,		"apply median filter to image"	},
{ "median_clip",do_median_clip,		"apply median filter to bright pixels" },
{ "median_1D",	do_median_1D,		"apply median filter to vector"	},

{ "convolve",	do_iconv,		"convolve two images"		},
{ "convolve3d",	do_iconv3d,		"convolve two sequences"	},

#ifdef HAVE_MORPH
{ "fill",	do_fill,		"flood fill from seed point"	},
#endif

{ "map",	dolutmap,		"map through a lookup table"	},
{ "resample",	do_resample,		"warp image using control image"},
{ "bilinear",	do_bilinear,		"warp image w/ bilinear inter."	},
{ "new_bilinear",do_new_bilinear,	"warp image w/ bilinear inter."	},
{ "Quads",	do_quads,		"list of all 4-tuples from an image"},
{ "extend_paths",do_ext_paths,		"matrix \"squaring\" for shortest-path algorithm"},
{ "yuv2rgb",	do_yuv2rgb,		"convert yuv422 to rgb"		},
{ "yuv2gray",	do_yuv2gray,		"convert yuv422 to grayscale"	},
#ifndef MAC
{ "quit",	popcmd,			"exit submenu"			},
#endif
{ NULL_COMMAND								}
};

static COMMAND_FUNC( do_misc )
{
	PUSHCMD(misc_ctbl,MISC_MENU_NAME);
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
	dp_copy(QSP_ARG  to,fr);

#ifdef HAVE_MORPH

void image_close(QSP_ARG_DECL  Data_Obj *to,Data_Obj *fr,int size)
{
	CYCLE_OPS( dilate, erode )
}

void image_open(QSP_ARG_DECL  Data_Obj *to,Data_Obj *fr,int size)
{
	CYCLE_OPS( erode, dilate )
}


static COMMAND_FUNC( do_closing )
{
	Data_Obj *to,*fr;
	int size;

	to=PICK_OBJ("target");
	fr=PICK_OBJ("source");
	size=HOW_MUCH("size of the closing");

	if( to==NO_OBJ || fr==NO_OBJ ) return;

	if( size <= 0 ){
		WARN("size for closing operator must be positive");
		return;
	}

	image_close(QSP_ARG  to,fr,size);
}

static COMMAND_FUNC( do_opening )   /* Should choose the size of the opening */
{
	Data_Obj *to,*fr;
	int size;

	to=PICK_OBJ("target");
	fr=PICK_OBJ("source");
	size=HOW_MUCH("size of the opening");

	if( to==NO_OBJ || fr==NO_OBJ ) return;

	if( size <= 0 ){
		WARN("size for opening operator must be positive");
		return;
	}

	image_open(QSP_ARG  to,fr,size);
}

static COMMAND_FUNC( do_dilate )
{
	Data_Obj *to,*fr;

	to=PICK_OBJ("target");
	fr=PICK_OBJ("source");
	if( to==NO_OBJ || fr==NO_OBJ ) return;

	dilate(QSP_ARG  to,fr);
}

static COMMAND_FUNC( do_erode )
{
	Data_Obj *to,*fr;

	to=PICK_OBJ("target");
	fr=PICK_OBJ("source");
	if( to==NO_OBJ || fr==NO_OBJ ) return;

	erode(QSP_ARG  to,fr);
}

static COMMAND_FUNC( gen_morph )
{
	Data_Obj *to,*fr,*tbl;

	to=PICK_OBJ("target");
	fr=PICK_OBJ("source");
	tbl=PICK_OBJ("function look-up table");
	if( to==NO_OBJ || fr==NO_OBJ || tbl == NO_OBJ ) return;

	morph_process(QSP_ARG  to,fr,tbl);
}

Command morph_ctbl[]={
{	"dilate",	do_dilate,	"dilatation of a binary image"	},
{	"erode",	do_erode,	"erosion of a binary image"	},
{	"closing",	do_closing,	"closing of a binary image"	},
{	"opening",	do_opening,	"opening of a binary image"	},
{	"morph",	gen_morph,	"apply table-defined morphological operator"	},
#ifndef MAC
{	"quit",	popcmd,			"exit submenu"			},
#endif
{	NULL_COMMAND							}
};

static COMMAND_FUNC( do_morph )
{
	PUSHCMD(morph_ctbl,MORPH_MENU_NAME);
}

#endif /* HAVE_MORPH */

static COMMAND_FUNC( do_radavg )
{
	Data_Obj *m_dp, *v_dp, *c_dp, *i_dp;

	m_dp = PICK_OBJ( "data vector for mean" );
	v_dp = PICK_OBJ( "data vector for variance" );
	c_dp = PICK_OBJ( "data vector for counts" );
	i_dp = PICK_OBJ( "source image" );

	if( m_dp == NO_OBJ || v_dp == NO_OBJ || c_dp == NO_OBJ || i_dp == NO_OBJ )
		return;

	rad_avg(QSP_ARG  m_dp,v_dp,c_dp,i_dp);
}

static COMMAND_FUNC( do_oriavg )
{
	Data_Obj *m_dp, *v_dp, *c_dp, *i_dp;

	m_dp = PICK_OBJ( "data vector for mean" );
	v_dp = PICK_OBJ( "data vector for variance" );
	c_dp = PICK_OBJ( "data vector for counts" );
	i_dp = PICK_OBJ( "source image" );

	if( m_dp == NO_OBJ || v_dp == NO_OBJ || c_dp == NO_OBJ || i_dp == NO_OBJ )
		return;

	ori_avg(QSP_ARG  m_dp,v_dp,c_dp,i_dp);
}

#include "dct8.h"

static COMMAND_FUNC( do_dct )
{
	Data_Obj *dp;

	dp = PICK_OBJ("");
	if( dp==NO_OBJ ) return;

	compute_dct(QSP_ARG  dp,FWD_DCT);
}

static COMMAND_FUNC( do_idct )
{
	Data_Obj *dp;

	dp = PICK_OBJ("");
	if( dp==NO_OBJ ) return;

	compute_dct(QSP_ARG  dp,INV_DCT);
}

static COMMAND_FUNC( do_odct )
{
	Data_Obj *dp;

	dp = PICK_OBJ("");
	if( dp==NO_OBJ ) return;

	compute_dct(QSP_ARG  dp,OLD_DCT);
}

Command fft_ctbl[]={
#ifdef FOOBAR
{	"newfft",	do_newfft,	"test new chainable complex fft"	},
#endif /* FOOBAR */
{	"fft",		do_fwdfft,	"forward complex Fourier transform"	},
{	"row_fft",	do_fwdrowfft,	"forward complex Fourier transform of rows only"	},
{	"rfft",		do_fwdrfft,	"forward Fourier transform, real input"	},
{	"row_rfft",	do_fwdrowrfft,	"forward Fourier transform of rows only, real input"	},
{	"irfft",	do_invrfft,	"inverse Fourier transform, real output"},
{	"row_irfft",	do_invrowrfft,	"inverse Fourier transform of rows only, real output"},
{	"invfft",	do_invfft,	"inverse complex Fourier transform"	},
{	"row_invfft",	do_invrowfft,	"inverse complex Fourier transform of rows only"	},
{	"radavg",	do_radavg,	"compute radial average"		},
{	"oriavg",	do_oriavg,	"compute orientation average"		},
{	"wrap",	do_wrap,	"wrap DFT image"			},
{	"wrap3d",	do_wrap3d,	"wrap 3-D DFT"				},
{	"scroll",	do_scroll,	"scroll image"				},
{	"dct",	do_dct,		"compute blocked discrete cosine xform"	},
{	"odct",	do_odct,	"compute DCT using old method"		},
{	"idct",	do_idct,	"compute inverse discrete cosine xform"	},
#ifndef MAC
{	"quit",	popcmd,	"exit submenu"					},
#endif
{	NULL_COMMAND								}
};

static COMMAND_FUNC( do_fft )
{
	PUSHCMD(fft_ctbl,FFT_MENU_NAME);
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

Command cmp_ctbl[]={
{	"max",	do_vmax,	"take the max of two vectors"		},
{	"min",	do_vmin,	"take the min of two vectors"		},
{	"max_mag",	do_vmaxm,	"take the max mag of two vectors"	},
{	"min_mag",	do_vminm,	"take the min mag of two vectors"	},
/* {	"vmscm",	do_vmscm,	"bit-map scalar-vector mag. comparision"}, */
/* put back in for iview compatibility */
{	"vmscp",	do_vsm_gt,	"bit-map scalar-vector comparision"	},
{	"clip",	do_clip,	"clip elements of a vector"		},
{	"iclip",	do_iclip,	"inverted clip"				},
{	"vscmp",	do_vscmp,	"vector-scalar comparison (>=)"		},
{	"vscmp2",	do_vscmp2,	"vector-scalar comparison (<=)"		},
{	"bound",	do_bnd,	"bound elements of a vector"			},
{	"ibound",	do_ibnd,	"inverted bound"			},
{	"vcmp",	do_vcmp,	"vector-vector comparison"		},
/* {	"vcmpm",	do_vcmpm,	"vector-vector magnitude comparison"	}, */
/* {	"vscmm",	do_vscmm,	"scalar-vector magnitude comparison"	}, */
{	"vsmax",	do_vsmax,	"scalar-vector maximum"			},
{	"vsmxm",	do_vsmxm,	"scalar-vector maximum magnitude"	},
{	"vsmin",	do_vsmin,	"scalar-vector minimum"			},
{	"vsmnm",	do_vsmnm,	"scalar-vector minimum magnitude"	},
{	"vvm_lt",	do_vvm_lt,	"bit-map vector comparision (<)"	},
{	"vvm_gt",	do_vvm_gt,	"bit-map vector comparision (>)"	},
{	"vvm_le",	do_vvm_le,	"bit-map vector comparision (<=)"	},
{	"vvm_ge",	do_vvm_ge,	"bit-map vector comparision (>=)"	},
{	"vvm_ne",	do_vvm_ne,	"bit-map vector comparision (!=)"	},
{	"vvm_eq",	do_vvm_eq,	"bit-map vector comparision (==)"	},
{	"vsm_lt",	do_vsm_lt,	"bit-map vector/scalar comparision (<)"	},
{	"vsm_gt",	do_vsm_gt,	"bit-map vector/scalar comparision (>)"	},
{	"vsm_le",	do_vsm_le,	"bit-map vector/scalar comparision (<=)" },
{	"vsm_ge",	do_vsm_ge,	"bit-map vector/scalar comparision (>=)" },
{	"vsm_ne",	do_vsm_ne,	"bit-map vector/scalar comparision (!=)" },
{	"vsm_eq",	do_vsm_eq,	"bit-map vector/scalar comparision (==)" },

/* {	"vmcmp",	do_vmcmp,	"bit-map vector comparision"		}, */
/* for iview compatibility */
{	"select",	do_vvvslct,	"vector/vector selection based on bit-map"	},
{	"vv_select",	do_vvvslct,	"vector/vector selection based on bit-map"	},
{	"vs_select",	do_vvsslct,	"vector/scalar selection based on bit-map"	},
{	"ss_select",	do_vssslct,	"scalar/scalar selection based on bit-map"	},
#ifndef MAC
{	"quit",	popcmd,		"exit submenu"				},
#endif
{	NULL_COMMAND								}
};

static COMMAND_FUNC( docmp )
{
	PUSHCMD(cmp_ctbl,COMP_MENU_NAME);
}

/* static COMMAND_FUNC( do_corr ) { DO_VCODE(FVCONV); } */
static COMMAND_FUNC( do_dot ) { DO_VCODE(FVDOT); }
/* static COMMAND_FUNC( do_cdot ) { DO_VCODE(FVCDOT); } */

static COMMAND_FUNC( do_xpose )
{
	Data_Obj *dpto, *dpfr;

	dpto=PICK_OBJ( "target image" );
	dpfr=PICK_OBJ( "source image" );
	if( dpto == NO_OBJ || dpfr == NO_OBJ ) return;
	transpose(QSP_ARG  dpto,dpfr);
}

static COMMAND_FUNC( do_ginvert )
{
	Data_Obj *dp;

	dp=PICK_OBJ( "target matrix" );
	if( dp == NO_OBJ ) return;

	/* BUG? is complete error checking done here??? */

	if( ! IS_CONTIGUOUS(dp) ){
		WARN("matrix must be contiguous for G-J inversion");
		return;
	}

	if( dp->dt_prec == PREC_SP )
		gauss_jordan( (float *)dp->dt_data, dp->dt_cols );
	else if( dp->dt_prec == PREC_DP )
		dp_gauss_jordan( (double *)dp->dt_data, dp->dt_cols );
	else {
		sprintf(ERROR_STRING,
	"Matrix %s should have float or double precision for Gauss-Jordan inversion",
			dp->dt_name);
		WARN(ERROR_STRING);
	}
}

static COMMAND_FUNC( do_invert )
{
	Data_Obj *dp;
	double det;

	dp=PICK_OBJ( "matrix" );
	if( dp == NO_OBJ ) return;
	det=dt_invert(QSP_ARG  dp);
	if( det == 0.0 ) WARN("ZERO DETERMINANT!!!");
	else if( verbose ) {
		sprintf(msg_str,"determinant:  %g",det);
		prt_msg(msg_str);
	}
}

/* inner (matrix) product - implemented in newvec? */

 COMMAND_FUNC(do_inner)
{
	Data_Obj *target, *v1, *v2;

	target=PICK_OBJ( "target data object" );
	v1=PICK_OBJ( "first operand" );
	v2=PICK_OBJ( "second operand" );

	if( target==NO_OBJ || v1==NO_OBJ || v2==NO_OBJ )
		return;

	inner(QSP_ARG  target,v1,v2);
}

#ifdef FOOBAR
static COMMAND_FUNC( do_corr_mtrx )
{
	Data_Obj *dpto;
	Data_Obj *dpfr;

	dpto = PICK_OBJ("target corrlation matrix");
	dpfr = PICK_OBJ("source vector array");

	if( dpto == NO_OBJ || dpfr == NO_OBJ ) return;

	corr_matrix(dpto,dpfr);
}
#endif /* FOOBAR */

static COMMAND_FUNC( do_determinant )
{
	Data_Obj *dp,*scal_dp;
	double d;

	scal_dp = PICK_OBJ("scalar for determinant");
	dp = PICK_OBJ("matrix");

	if( scal_dp == NO_OBJ || dp==NO_OBJ ) return;

	d = determinant(dp);
	/* BUG check typing on scalar */
	*((float *)scal_dp->dt_data) = d;
}

 COMMAND_FUNC(do_xform_list )
{
	Data_Obj *dpto, *dpfr, *xform;

	dpto=PICK_OBJ( "target data object" );
	dpfr=PICK_OBJ( "source data object" );
	xform=PICK_OBJ( "transformation matrix" );

	if( dpto==NO_OBJ || dpfr==NO_OBJ || xform==NO_OBJ ) return;

	xform_list(QSP_ARG  dpto,dpfr,xform);
}

COMMAND_FUNC( do_vec_xform )
{
	Data_Obj *dpto, *dpfr, *xform;

	dpto=PICK_OBJ( "target data object" );
	dpfr=PICK_OBJ( "source data object" );
	xform=PICK_OBJ( "transformation matrix" );

	if( dpto==NO_OBJ || dpfr==NO_OBJ || xform==NO_OBJ ) return;

	vec_xform(QSP_ARG  dpto,dpfr,xform);
}

COMMAND_FUNC( do_homog_xform )
{
	Data_Obj *dpto, *dpfr, *xform;

	dpto=PICK_OBJ( "target data object" );
	dpfr=PICK_OBJ( "source data object" );
	xform=PICK_OBJ( "transformation matrix" );

	if( dpto==NO_OBJ || dpfr==NO_OBJ || xform==NO_OBJ ) return;

	homog_xform(QSP_ARG  dpto,dpfr,xform);
}


COMMAND_FUNC( do_outer )
{
	WARN("sorry, outer product not yet implemented");
}

Command lin_ctbl[]={
#ifdef FOOBAR
{	"correlate",	do_corr,	"correlate two vectors"			},
{	"corr_mtrx",	do_corr_mtrx,	"compute correlation matrix of row vectors"},
#endif /* FOOBAR */
{	"dot",	do_dot,		"vector dot product"			},
/* {	"cdot",	do_cdot,	"conjugate dot product"			}, */
{	"oinvert",	do_invert,	"invert a square matrix (old way)"	},
{	"invert",	do_ginvert,	"invert a square matrix (Gauss-Jordan)"	},
{	"transpose",	do_xpose,	"transpose an image"			},
/* { "convolve",	do_conv,	"convolve two vectors"			}, */
{	"inner_prod",	do_inner,	"compute inner product"			},
{	"outer_prod",	do_outer,	"compute outer product"			},
{	"xform",	do_vec_xform,	"transform, vectorizing each element"	},
{	"lxform",	do_xform_list,	"transform, vectorizing over the list"	},
{	"homogenous",	do_homog_xform,	"transform homogenous coords (vectorizes over list)"},
{	"determinant",do_determinant,	"compute matrix determinant"		},
#ifndef MAC
{	"quit",	popcmd,		"exit submenu"				},
#endif
{	NULL_COMMAND								}
};

static COMMAND_FUNC( do_lin )
{
	PUSHCMD(lin_ctbl,LIN_MENU_NAME);
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


Command expr_ctbl[]={
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
	PUSHCMD(expr_ctbl,EXPR_MENU_NAME);
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

#ifdef USE_SSE
	if( use_sse_extensions )
		prt_msg("Using SSE extensions when possible");
	else
		prt_msg("Never using SSE extensions");
#else /* ! USE_SSE */
	prt_msg("No SSE support (compile with USE_SSE defined)");
#endif /* ! USE_SSE */

	sprintf(msg_str,"Using %d processors (%d max on this machine",
			n_processors,N_PROCESSORS);
	prt_msg(msg_str);
}

static Command ctl_ctbl[]={
{	"perform",	do_set_perf,		"set/clear execute flag"			},
{	"n_processors",	set_n_processors,	"set number of processors to use"		},
{	"use_sse",	set_use_sse,		"enable/disable use of Pentium SSE extensions"	},
{	"status",	do_report_status,	"report current computation modes"		},
{	"cpuinfo",	get_cpu_info,		"report cpu info"				},
#ifndef MAC
{	"quit",	popcmd,		"exit submenu"							},
#endif
{	NULL_COMMAND										}
};

 COMMAND_FUNC(do_ctl )
{
	PUSHCMD(ctl_ctbl,"vec_ctl");
}

Command war_ctbl[]={
{	"trig",		do_trig,	"trigonometric operations"		},
{	"unary",	do_unary,	"unary operations on data"		},
{	"logical",	do_logic,	"logical operations on data"		},
{	"vvector",	do_vv,		"vector-vector operations"		},
{	"svector",	do_rvs,		"real_scalar-vector operations"		},
#ifdef FOOBAR
{	"csvector",	do_cvs,		"complex_scalar-vector operations"	},
{	"Qsvector",	do_qvs,		"quaternion_scalar-vector operations"	},
#endif /* FOOBAR */
{	"minmax",	do_minmax,	"minimum/maximum routines"		},
{	"compare",	docmp,		"comparision routines"			},
{	"fft",		do_fft,		"FFT submenu"				},
{	"linear",	do_lin,		"linear algebra functions"		},
{	"misc",		do_misc,	"miscellaneous functions"		},
{	"image",	do_imgsyn,	"image synthesis functions"		},
{	"control",	do_ctl,		"warrior control functions"		},
{	"sample",	sampmenu,	"image sampling submenu"		},
#ifdef HAVE_MORPH
{	"morph",	do_morph,	"morphological operators"		},
#endif /* HAVE_MORPH */
{	"requant",	do_requant,	"requantization (dithering) submenu"	},

#ifdef FOOBAR
{	"polynomial",	polymenu,	"polynomial manipulations"		},
#endif /* FOOBAR */

#ifndef MAC
{	"quit",		popcmd,		"exit submenu"				},
#endif
{	NULL_COMMAND								}
};

COMMAND_FUNC(warmenu )
{
	vl_init(SINGLE_QSP_ARG);

#ifdef MAC
	do_trig();
	do_unary();
	do_logic();
	do_vv();
	do_vs();
	do_minmax();
	docmp();
	do_fft();
	do_lin();
	do_misc();
	do_morph();
	do_ctl();
	sampmenu();

#else /* ! MAC */
	PUSHCMD(war_ctbl,"compute");
#endif /* ! MAC */
}


void do_vcode(QSP_ARG_DECL  Vec_Func_Code code)
{
	do_vfunc(QSP_ARG  &vec_func_tbl[code]);
}

