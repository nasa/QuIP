
#include "quip_config.h"

char VersionId_cuda_cuda_menu[] = QUIP_VERSION_STRING;

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

#ifdef HAVE_OPENGL
#ifdef HAVE_GLUT
#include "cuda_viewer.h"
#endif /* HAVE_GLUT */
#endif /* HAVE_OPENGL */

#include "menuname.h"
#include "query.h"
#include "version.h"
#include "my_cuda.h"
#include "cuda_supp.h"
#include "submenus.h"		// prototype for top level menu func.

#ifdef HAVE_OPENGL
#ifdef HAVE_GLUT
#include "cuda_viewer.h"
#endif /* HAVE_GLUT */
#endif /* HAVE_OPENGL */

// includes, project
//#include <cutil_inline.h>

// includes, kernels
//#include <myproject_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

//static int gl_argc;
//static char ** gl_argv;

/*
static COMMAND_FUNC( test_cuda )
{
	runTest( gl_argc, gl_argv );
}

static COMMAND_FUNC( exit_cuda )
{
	//advise("exit_cuda calling cutilExit");
	//cutilExit(gl_argc, gl_argv);
	nice_exit(0);
}
*/

static COMMAND_FUNC( do_test_blas )
{
#ifdef HAVE_CUDA
	if( test_cublas() != EXIT_SUCCESS )
		NWARN("BLAS test failed");
#endif
}


/*
static COMMAND_FUNC(do_gpu_sub)
{
	VecSub();
}
*/

//CUDA UNIARY COMMAND TABLE - ALL KNOWN FUNCTIONS ACCOUNTED FOR
Command cuda_unary_ctbl[]={		
{	"set",		do_gpu_vset,	"set to a constant"					},
{	"convert",	do_gpu_convert,	"convert to from one type to another"			},
{	"abs", 		do_gpu_vabs, 	"convert to absolute value"				},
//{	"conj", 	,		 "convert to complex conjugate"				},
//{	"find", 	, 		"return indeces of non-zero elements"			},
//{	"dimsum", 	, 		"return sum along comumns (rows)"			},
{	"mov", 		do_gpu_rvmov,	"copy data"						},
{	"neg", 		do_gpu_rvneg,	"convert to negative"					},
//{	"sum", 		do_gpu_vsum,	"get sum of vector"					},
//{	"flip", 	,		"copy a vector, reversing the order of the elements"	},
//{	"flipall", 	,		"copy an object, reversing order of all dimensions"	},
{	"sign", 	do_gpu_vsign,	"take sign of vector"					},
//{	"vrint", 	do_gpu_vrint,	"round vector elements to nearest integer using rint()"	},
{	"round",	do_gpu_vround,	"round vector elements to nearest integer using round()"},
{	"floor", 	do_gpu_vfloor,	"take floor of vector"					},
{	"ceil", 	do_gpu_vceil,	"take ceiling of vector"				},
//{	"uni", 		,		"uniform random numbers"				},
{	"quit", 	popcmd, 		"exit submenu"					},
{ 	NULL_COMMAND 										}
};

static COMMAND_FUNC( do_unary )
{
	PUSHCMD(cuda_unary_ctbl, UNARY_MENU_NAME);
}

//CUDA TRIG COMMAND TABLE - ALL KNOWN FUNCTIONS ACCOUNTED FOR
Command cuda_trig_ctbl[]={
{	"atan", 	do_gpu_vatan, 	"compute arc tangent"					},
//{	"atn2", 	,		"compute arc tangent (single xomplex arg)"		},
{	"atan2", 	do_gpu_vatan2,	"compute arc tangent (two real args)"			},
//{	"magsq", 	,		"convert to magnitude squared"				},
{	"cos",	 	do_gpu_vcos, 	"compute cosine"					},
{	"erf", 		do_gpu_verf, 	"compute error function (erf)"				},
{	"exp", 		do_gpu_vexp, 	"exponentiate (base e)"					},
{	"log", 		do_gpu_vlog, 	"natural logarithm"					},
{	"log10", 	do_gpu_vlog10, 	"logarithm base 10"					},
{	"sin",		do_gpu_vsin, 	"sompute sine"						},
{	"square", 	do_gpu_rvsqr, 	"compute square"					},
{	"sqrt", 	do_gpu_vsqrt, 	"computer square root"					},
{	"tan", 		do_gpu_vtan, 	"compute tangent"					},
{	"pow", 		do_gpu_rvpow, 	"raise to a power"					},
{	"acos", 	do_gpu_vacos, 	"compute inverse cosine"				},
{	"asin", 	do_gpu_vasin, 	"compute inverse sine"					},
//{	"j0", 		do_gpu_vj0, 	"compute bessel function j0"				},
//{	"j1", 		do_gpu_vj1, 	"compute bessel function j1"				},
{ 	"quit", 	popcmd, 	"exit submenu"						},
{ NULL_COMMAND 											}
};

static COMMAND_FUNC( do_trig )
{
	PUSHCMD(cuda_trig_ctbl, TRIG_MENU_NAME);
}

//CUDA LOGICAL COMMAND TABLE - ALL KNOWN FUNCTIONS ACCOUNTED FOR
Command cuda_logical_ctbl[]={
//{	"cmp", 		, 		"bitwise complement"			},
{	"and", 		do_gpu_vand, 	"logical AND"				},
{	"nand", 	do_gpu_vnand, 	"logical NAND"				},
{	"not", 		do_gpu_vnot, 	"logical NOT"				},
{	"or", 		do_gpu_vor, 	"logical OR"				},
{	"xor", 		do_gpu_vxor, 	"logical XOR"				},
{	"sand", 	do_gpu_vsand, 	"logical AND with scalar"		},
{	"sor", 		do_gpu_vsor, 	"logical OR with scalar"		},
{	"sxor", 	do_gpu_vsxor, 	"logical XOR with scalar"		},
{	"shr", 		do_gpu_vshr, 	"right shift"				},
{	"shl", 		do_gpu_vshl, 	"left shift"				},
{	"sshr", 	do_gpu_vsshr, 	"right shift by a constant"		},
{	"sshl", 	do_gpu_vsshl, 	"left shift by a constant"		},
{	"sshr2", 	do_gpu_vsshr2, 	"right shift a constant"		},
{	"sshl2", 	do_gpu_vsshl2, 	"left shift a constant"			},
{ 	"quit", 	popcmd, 	"exit submenu"				},
{ NULL_COMMAND 									}
};

static COMMAND_FUNC( do_logic )
{
	PUSHCMD(cuda_logical_ctbl, LOG_MENU_NAME);
}

//CUDA VVECTOR COMMAND TABLE - ALL KNOWN FUNCTIONS ACCOUNTED FOR
static Command cuda_vvector_ctbl[]={
{	"add", 	do_gpu_vadd, 	"vector addition"			},
//	"cmul", , 		"multiply by complex conjugation"	},
{	"div", 	do_gpu_rvdiv, 	"element by element division"		},
{	"mul", 	do_gpu_rvmul, 	"element by element multiplication"	},
{	"sub", 	do_gpu_rvsub, 	"vector subtraction"			},
{	"quit", popcmd, 	"exit submenu"				},
{ 	NULL_COMMAND 							}
};

static COMMAND_FUNC( do_vv )
{
	PUSHCMD(cuda_vvector_ctbl, VV_MENU_NAME);
}

//CUDA RSV COMMAND TABLE - ALL KNOWN FUNCTIONS ACCOUNTED FOR
static Command cuda_rvs_ctbl[]={
{ "add", 	do_gpu_rvsadd, 	"add scalar to elements of a vector"				},
{ "sub", 	do_gpu_rvssub, 	"subtract elements of a vecotr from a scalar"			},
{ "div", 	do_gpu_rvsdiv, 	"divide a scalar by the elements of a vector"			},
{ "div2", 	do_gpu_rvsdiv2,	"divide elements of a vector by a scalar"			},
{ "mul", 	do_gpu_rvsmul, 	"multiply a vector by a real scalar"				},
{ "mod", 	do_gpu_vsmod, 	"integer modulo of a vector by a real scalar"			},
{ "mod2",	do_gpu_vsmod2, 	"integer modulo of a real scalar by a vector"			},
{ "pow", 	do_gpu_vspow, 	"raise the elements of a vector to a scalar power"		},
{ "pow2", 	do_gpu_vspow2, 	"raise a scalar to powers given by the elements of a vector"	},
{ "atan2", 	do_gpu_vsatan2, "compute 4-quadrant arc tangent of vector and scalar"		},
{ "atan22",	do_gpu_vsatan22,"compute 4-quadrant arc tangent of a scalar and vector"		},
//{ "and", 	,		"bitwise and of scalar and vector"				},
//{ "or", 	,		"bitwise or of scalar and vector"				},
//{ "xor", 	,		"bitwise xor of scalar and vector"				},
{ "quit", 	popcmd,		"exit submenu"							},
{ NULL_COMMAND											}
};

static COMMAND_FUNC( do_rvs )
{
	PUSHCMD(cuda_rvs_ctbl, SV_MENU_NAME);
}

//CUDA CSV COMMAND TABLE
static Command cuda_cvs_ctbl[]={
//{"add", 	, 		"add complex scalar to elements of a vector"			},
//{"div", , 		"divide a complex scalar by the elements of a vector"	},
//{"div2", , 		"divide elements of a vector by a complex scalar"		},
//{"mul", , 		"multiply a vector by a complex scalar"					},
//{"conjmul", ,	 	"multiply vector conjugation by a complex scalar"		},
//{"sub", , 		"subtract elements of a vector from a complex scalar"	},
{"quit",	popcmd,	"exit submenu"											},
{ NULL_COMMAND 																}
};

static COMMAND_FUNC( do_cvs )
{
	PUSHCMD(cuda_cvs_ctbl, CSV_MENU_NAME);
}

//CUDA QSV COMMAND TABLE
static Command cuda_qvs_ctbl[]={
//{"add", , 		"add complex scalar to elements of a vector"			},
//{"div", , 		"divide a complex scalar by the elements of a vector"	},
//{"div2", , 		"divide elements of a vector by a complex scalar"		},
//{"mul", , 		"multiply a vector by a complex scalar"					},
//{"conjmul", , 	"multiply vector conjugation by a complex scalar"		},
//{"sub", , 		"subtract elements of a vector from a complex scalar"	},
{"quit", 	popcmd, "quit submenu"											},
{ NULL_COMMAND 																}
};

static COMMAND_FUNC( do_qvs )
{
	PUSHCMD(cuda_qvs_ctbl, QSV_MENU_NAME);
}

//CUDA MINMAX COMMAND TABLE - ALL KNOWN FUNCTIONS ACCOUNTED FOR
static Command cuda_minmax_ctbl[]={
{	"max", 			do_gpu_vmax,	"find maximum value"					},
{	"min", 			do_gpu_vmin,	"find minimum value"					},
{	"max_mag", 		do_gpu_vmaxm,	"find maximum absolute value"			},
{	"min_mag", 		do_gpu_vminm,	"find minimum absolute value"			},
//{	"max_index",		,		"find index of maximum value"				},
//{	"min_index",		,		"find index of minimum value"				},
//{	"max_mag_index", 	, 		"find index of maximum absolute value"		},
//{	"min_mag_index", 	, 		"find index of minimum absolute value"		},
//{	"max_times", 		,		"find index of maximum & # of occurrences"	},
//{	"min_times", 		, 		"find index of minimum & # of occurrences"	},
//{	"max_mag_times", 	, 		"find index of max. mag. & # of occurrences"},
//{	"min_mag_times", 	, 		"find index of min. mag. & # of occurrences"},
{	"quit", 		popcmd, 	"exit submenu"								},
{ 	NULL_COMMAND 															}
};

static COMMAND_FUNC( do_minmax )
{
	PUSHCMD(cuda_minmax_ctbl, MINMAX_MENU_NAME);
}

//CUDA CMP COMMAND TABLE
static Command cuda_cmp_ctbl[]={
{	"max", 		do_gpu_vmax, 	"take the max of two vectors"				},
{	"min", 		do_gpu_vmin, 	"take the min of two vectors"				},
{	"max_mag", 	do_gpu_vmaxm, 	"take the max mag of two vectors"			},
{	"min_mag", 	do_gpu_vminm,	"take the min mag of two vectors"			},
//{	"vmscm", 	, 				"bit-map scalar-vector mag. comparison"		},
//{	"vmscp", 	, 				"bit-map scalar-vector comparison"			},
{	"clip", 	do_gpu_vclip, 	"clip elements of a vector"					},
{	"iclip", 	do_gpu_viclp, 	"inverted clip"								},
{	"vscmp", 	do_gpu_vscmp, 	"vector-scalar comparison (>=)"				},
{	"vscmp2", 	do_gpu_vscmp2, 	"vector-scalar comparison (<=)"				},
{	"bound", 	do_gpu_vbnd, 	"bound elements of a vector"				},
{	"ibound", 	do_gpu_vibnd, 	"inverted bound"							},
{	"cmp", 		do_gpu_vcmp, 	"vector-vector comparison"					},
//{	"vcmpm", 	, 				"vector-vector magnitude comparison"		},
//{	"vscmm", 	, 				"scalar-vector magnitude comparison"		},
{	"vsmax", 	do_gpu_vsmax, 	"scalar-vector maximum"						},
{	"vsmxm", 	do_gpu_vsmxm, 	"scalar-vector maximum magnitude"			},
{	"vsmin", 	do_gpu_vsmin, 	"scalar-vector minimum"						},
{	"vsmnm", 	do_gpu_vsmnm, 	"scalar-vector minimum magnitude"			},
//{	"vvm_lt", 	, 				"bit-map vector comparison (<)"				},
//{	"vvm_gt", 	, 				"bit-map vector comparison (>)"				},
//{	"vvm_le", 	, 				"bit-map vector comparison (<=)"			},
//{	"vvm_ge", 	,		 		"bit-map vector comparison (>=)"			},
//{	"vvm_ne", 	, 				"bit-map vector comparison (!=)"			},
//{	"vvm_eq", 	, 				"bit-map vector comparison (==)"			},
//{	"vsm_lt", 	, 				"bit-map vector/scalar comparison (<)"		},
//{	"vsm_gt", 	, 				"bit-map vector/scalar comparison (>)"		},
//{	"vsm_ge", 	, 				"bit-map vector/scalar comparison (<=)"		},
//{	"vsm_ne", 	, 				"bit-map vector/scalar comparison (>=)"		},
//{	"vsm_eq", 	, 				"bit-map vector/scalar comparison (==)"		},
//{	"vmcmp", 	,				"bit-map vector comarison"					},
//{	"select", 	,				"vector/vector selection based on bit-map"	},
//{	"vv_select", 	, 			"vector/vector selection based on bit-map"	},
//{	"vs_select", 	, 			"vector/scalar selection based on bit-map"	},
//{	"ss_select", 	, 			"scalar/scalar selection based on bit-map"	},
{	"quit", 	popcmd, 		"exit submenu"								},
{ NULL_COMMAND 																}
};

static COMMAND_FUNC( docmp )
{
	PUSHCMD(cuda_cmp_ctbl, COMP_MENU_NAME);
}

//CUFFT COMMAND TABLE
static Command cuda_fft_ctbl[]={
//{"newfft", 	do_newfft,			"test new chainable complex fft"},
{"fft", 		do_gpu_fwdfft, 		"forward complex Fourier Transform"},
/*{"row_fft", 	do_gpu_fwdrowfft,	"forward complex Fourier Transform of rows only "},
{"rfft", 		do_gpu_fwdrfft,		"forward Fourier Transform, real input"},
{"row_rfft",	do_gpu_fwdrowrfft,	"forward Fourier transform of rows only, real input"},
{"irfft", 		do_gpu_invrfft,		"inverse Fourier Transform, real output"},
{"row_irfft", 	do_gpu_invrowrfft,	"inverse Fourier Transform of rows only, real output"},
{"invfft", 		do_gpu_invfft,		"inverse complex Fourier Transform"},
{"row_invfft", 	do_gpu_invrowfft,	"inverse complex Fourier Transform of rows only"},
{"radavg", 		do_gpu_radavg,		"compute radial average"},
{"oriavg", 		do_gpu_oriavg,		"compute orientation average"},
{"wrap", 		do_gpu_wrap,		"wrap DFT iamge"},
{"wrap3d", 		do_gpu_wrap3d,		"wrap 3-D DFT"},
{"scroll", 		do_gpu_scroll,		"scroll image"},
{"dct", 		do_gpu_dct,			"compute blocked discrete cosine xform"},
{"odct", 		do_gpu_odct,		"compute DCT using old method"},
{"idct", 		do_gpu_idct,		"compute inverse descrete cosine xform"},*/
{"quit", 		popcmd,				"exit submenu"},
{NULL_COMMAND}
};

static COMMAND_FUNC( do_fft )
{
	PUSHCMD(cuda_fft_ctbl, FFT_MENU_NAME);
}

#ifdef HAVE_LIBNPP

static Command cuda_npp_ctbl[]={
{ "image",	do_npp_malloc,		"declare new image"		},
{ "vadd",	do_npp_vadd,		"add two images"		},
{ "erode",	do_npp_erosion,		"erosion"			},
{ "dilate",	do_npp_dilation,	"dilation"			},
{ "filter",	do_npp_filter,		"space-domain filtering"	},
{ "sum",	do_npp_sum,		"compute sum"			},
{ "sum_scratch",do_npp_sum_scratch,	"allocate scratch space for sum"},
{ "i_vmul",	do_nppi_vmul,		"image/image multiplication"	},
{ "s_vmul",	do_npps_vmul,		"in-place signel multiplication"},
{ "quit",	popcmd,			"exit submenu"			},
{ NULL_COMMAND								}
};

static COMMAND_FUNC( npp_menu )
{
	PUSHCMD(cuda_npp_ctbl,NPP_MENU_NAME);
}

#endif /* HAVE_LIBNPP */

COMMAND_FUNC( do_cuda_fill )
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

COMMAND_FUNC( do_cuda_fill2 )
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

static COMMAND_FUNC( do_cuda_yuv2rgb )
{
	Data_Obj *rgb_dp;
	Data_Obj *yuv_dp;

	rgb_dp = PICK_OBJ("RGB image");
	yuv_dp = PICK_OBJ("YUV image");

	//  BUG do all checks:
	// pixel types, mating sizes
	cuda_yuv422_to_rgb24(rgb_dp,yuv_dp);
}

static COMMAND_FUNC( do_cuda_centroid )
{
	Data_Obj *dst1_dp;
	Data_Obj *dst2_dp;
	Data_Obj *src_dp;
	Vec_Obj_Args oargs;

	dst1_dp = PICK_OBJ("x scratch image");
	dst2_dp = PICK_OBJ("y scratch image");
	src_dp = PICK_OBJ("source image");

	if( MACHINE_PREC(src_dp) != PREC_SP && MACHINE_PREC(src_dp) != PREC_DP ){
		sprintf(error_string,"Object %s (%s) must have %s or %s precision for centroid helper",
			src_dp->dt_name,name_for_prec(src_dp->dt_prec),name_for_prec(PREC_SP),
			name_for_prec(PREC_DP));
		WARN(error_string);
		return;
	}

	// BUG - do more checking here sizes must match, precisions must match.

	setvarg3(&oargs,dst1_dp,dst2_dp,src_dp);	/* abusing this a little */

	if( src_dp->dt_prec == PREC_SP )
		sp_cuda_centroid(&oargs);
	else if( src_dp->dt_prec == PREC_DP )
		dp_cuda_centroid(&oargs);
#ifdef CAUTIOUS
	else ERROR1("CAUTIOUS:  do_cuda_centroid:  unexpected source precision!?");
#endif /* CAUTIOUS */
}

Command cuda_misc_ctbl[]={
{ "fill",	do_cuda_fill,	"flood fill"		},
{ "fill2",	do_cuda_fill2,	"flood fill version 2"	},
{ "yuv2rgb",	do_cuda_yuv2rgb,"YUV to RGB conversion"	},
{ "centroid",	do_cuda_centroid,	"centroid helper function"	},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL_COMMAND 								}
};

COMMAND_FUNC( do_cuda_misc )
{
	PUSHCMD(cuda_misc_ctbl, "cuda_misc");
}

//CUDA WAR COMMAND TABLE
Command cuda_war_ctbl[]={
{ "trig",	do_trig,	"trigonometric operations"		},
{ "unary",	do_unary,	"unary operations on data"		},
{ "logical",	do_logic,	"logical operations on data"		},
{ "vvector",	do_vv,		"vector-vector operations"		},
{ "svector",	do_rvs,		"real scalar-vector operations"		},
{ "csvector",	do_cvs,		"complex scalar-vector operations"	},
{ "Qsvector",	do_qvs,		"quaternion scalar-vector operations"	},
{ "minmax",	do_minmax,	"minimum/maximum routines"		},
{ "compare",	docmp,		"comparison routines"			},
{ "fft",	do_fft, 	"fft"					},
{ "misc",	do_cuda_misc,	"miscellaneous cuda functions"		},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL_COMMAND 								}
};

COMMAND_FUNC(cuda_func_menu)
{
	/* Do cuda-specific init here? */
	PUSHCMD(cuda_war_ctbl, "cuda_compute");
}

#ifdef HAVE_OPENGL
#ifdef HAVE_GLUT
static Command cuda_gl_ctbl[]={
{	"buffer",	do_new_gl_buffer,	"create a named GL buffer"	},
#ifdef FOOBAR
{	"test",		gl_test,		"gl test function"	},
{	"display",	gl_disp,		"update display window"	},
#endif /* FOOBAR */
{	"viewer",	do_new_cuda_vwr,	"create a new cuda image viewer"	},
{	"load",		do_load_cuda_vwr,	"write an image to a cuda viewer"	},
{	"quit",		popcmd,			"exit submenu"		},
{	NULL_COMMAND							}
};

COMMAND_FUNC(cuda_gl_menu)
{
	/* Do cuda-specific init here? */
	PUSHCMD(cuda_gl_ctbl, "cuda_gl");
}

#endif /* HAVE_GLUT */
#endif /* HAVE_OPENGL */

static Command cuda_event_ctbl[]={
{ "max_checkpoints",	do_init_checkpoints,	"set maximum number of checkpoints"				},
{ "set_checkpoint",	do_set_checkpoint,	"set a checkpoint"				},
{ "reset",	do_clear_checkpoints,	"clear all checkpoints"				},
{ "show",	do_show_checkpoints,	"show checkpoint times"				},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL_COMMAND								}
};

static COMMAND_FUNC( cuda_event_menu )
{
	PUSHCMD(cuda_event_ctbl,"cuda_events");
}

static Command stream_ctbl[]={
{ "stream",	do_new_stream,	"create a new stream"			},
{ "sync",	do_sync_stream,	"synchronize host execution with a stream"	},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL_COMMAND								}
};

static COMMAND_FUNC( stream_menu )
{
	PUSHCMD(stream_ctbl,"streams");
}

static COMMAND_FUNC( do_prt_cap )
{
}


static Command cuda_ctbl[]={
{ "devices",	query_cuda_devices,	"list all cuda devices"		},
{ "capability",	do_prt_cap,	"print GPU compute capability"		},
{ "test_blas",	do_test_blas,	"simple test of CUDA BLAS"		},
{ "list",	do_list_cudevs,	"list all cuda devices"			},
{ "info",	do_cudev_info,	"print information about a device"	},
{ "upload",	do_gpu_upload,	"upload data to a GPU"			},
{ "dnload",	do_gpu_dnload,	"download data from a GPU"		},
{ "compute",	cuda_func_menu,	"cuda compute function submenu"		},
#ifdef HAVE_LIBNPP
{ "npp",	npp_menu,	"NPP library submenu"			},
#endif /* HAVE_LIBNPP */
#ifdef HAVE_OPENGL
#ifdef HAVE_GLUT
{ "cuda_gl",	cuda_gl_menu,	"cuda GL submenu"			},
#endif /* HAVE_GLUT */
#endif /* HAVE_OPENGL */
{ "streams",	stream_menu,	"CUDA stream submenu"			},
{ "events",	cuda_event_menu,"cuda event  submenu"			},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL_COMMAND								}
};

extern "C" {

COMMAND_FUNC( cuda_menu )
{
	static int cuda_inited=0;

	if( ! cuda_inited ){
		init_cuda_devices(SINGLE_QSP_ARG);
		auto_version(QSP_ARG  "CUDA","VersionId_cuda");
		cuda_inited=1;
	}

	PUSHCMD(cuda_ctbl,"cuda");
}

}

