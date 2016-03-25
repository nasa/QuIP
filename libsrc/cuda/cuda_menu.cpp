
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

//#include "quip_prot.h"	// can this come after cuda includes?

#ifdef HAVE_CUDA
#define BUILD_FOR_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>


#ifdef FOOBAR
// BUG - why include this twice???
#ifdef HAVE_OPENGL
#ifdef HAVE_GLUT
#include "cuda_viewer.h"
#endif /* HAVE_GLUT */
#endif /* HAVE_OPENGL */
#endif // FOOBAR

#endif // HAVE_CUDA

#include "quip_prot.h"	// can this come after cuda includes?

#include "cuda_api.h"
#include "my_cuda.h"
#include "cuda_supp.h"

#ifdef FOOBAR
// BUG - why include this twice???
#ifdef HAVE_CUDA
#ifdef HAVE_OPENGL
#ifdef HAVE_GLUT
#include "cuda_viewer.h"
#endif /* HAVE_GLUT */
#endif /* HAVE_OPENGL */
#endif // HAVE_CUDA
#endif // FOOBAR

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

#ifdef FOOBAR
static COMMAND_FUNC( do_test_blas )
{
#ifdef HAVE_CUDA
	if( test_cublas() != EXIT_SUCCESS )
		NWARN("BLAS test failed");
#endif
}
#endif /* FOOBAR */


//COMMAND TABLE FOR CUDA NPP LIBRARY
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(npp_menu,s,f,h)

MENU_BEGIN( npp )
ADD_CMD( image,		do_npp_malloc,		declare new image )
ADD_CMD( vadd,		do_npp_vadd,		add two images )
ADD_CMD( erode,		do_npp_erosion,		erosion )
ADD_CMD( dilate,	do_npp_dilation,	dilation )
ADD_CMD( filter,	do_npp_filter,		space-domain filtering )
ADD_CMD( sum,		do_npp_sum,		compute sum )
ADD_CMD( sum_scratch,	do_npp_sum_scratch,	allocate scratch space for sum )
ADD_CMD( i_vmul,	do_nppi_vmul,		image/image multiplication )
ADD_CMD( s_vmul,	do_npps_vmul,		in-place signel multiplication )
ADD_CMD( version,	do_report_npp_version,	report NPP library version )
MENU_END( npp )

static COMMAND_FUNC( do_npp_menu )
{
	PUSH_MENU(npp);
}

#ifdef NOT_YET	// not obsolete, but not ready to integrate...

static COMMAND_FUNC( do_cuda_fill )
{
	Data_Obj *dp;
	int x,y;
	float fill_val, tol;

	dp = PICK_OBJ("image");
	x=HOW_MANY("seed point x");
	y=HOW_MANY("seed point y");
	fill_val=HOW_MUCH("fill value");
	tol = HOW_MUCH("tolerance");

#ifdef HAVE_CUDA
	h_sp_ifl(dp,x,y,tol,fill_val);
#endif // HAVE_CUDA
}

static COMMAND_FUNC( do_cuda_fill2 )
{
	Data_Obj *dp;
	int x,y;
	float fill_val, tol;

	dp = PICK_OBJ("image");
	x=HOW_MANY("seed point x");
	y=HOW_MANY("seed point y");
	fill_val=HOW_MUCH("fill value");
	tol = HOW_MUCH("tolerance");

#ifdef HAVE_CUDA
	h_sp_ifl2(dp,x,y,tol,fill_val);
#endif // HAVE_CUDA
}

static COMMAND_FUNC( do_cuda_yuv2rgb )
{
	Data_Obj *rgb_dp;
	Data_Obj *yuv_dp;

	rgb_dp = PICK_OBJ("RGB image");
	yuv_dp = PICK_OBJ("YUV image");

	//  BUG do all checks:
	// pixel types, mating sizes
#ifdef HAVE_CUDA
	cuda_yuv422_to_rgb24(rgb_dp,yuv_dp);
#endif // HAVE_CUDA
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

	if( OBJ_MACH_PREC(src_dp) != PREC_SP && OBJ_MACH_PREC(src_dp) != PREC_DP ){
		sprintf(ERROR_STRING,"Object %s (%s) must have %s or %s precision for centroid helper",
			OBJ_NAME(src_dp),PREC_NAME(OBJ_PREC_PTR(src_dp)),PREC_NAME(PREC_FOR_CODE(PREC_SP)),
			PREC_NAME(PREC_FOR_CODE(PREC_DP)));
		WARN(ERROR_STRING);
		return;
	}

	// BUG - do more checking here sizes must match, precisions must match.

	setvarg3(&oargs,dst1_dp,dst2_dp,src_dp);	/* abusing this a little */

#ifdef HAVE_CUDA
	if( OBJ_PREC(src_dp) == PREC_SP )
		sp_cuda_centroid(&oargs);
	else if( OBJ_PREC(src_dp) == PREC_DP )
		dp_cuda_centroid(&oargs);
#ifdef CAUTIOUS
	else ERROR1("CAUTIOUS:  do_cuda_centroid:  unexpected source precision!?");
#endif /* CAUTIOUS */
#endif // HAVE_CUDA
}

#endif // NOT_YET

#ifdef FOOBAR	// obsolete

#ifdef HAVE_OPENGL
#ifdef HAVE_GLUT

//CUDA GL COMMAND TABLE
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(cuda_gl_menu,s,f,h)

MENU_BEGIN( cuda_gl )
#ifdef FOOBAR
ADD_CMD( buffer,	do_new_gl_buffer,	create a named GL buffer )
ADD_CMD( test,		gl_test,		gl test function )
ADD_CMD( display,	gl_disp,		update display window )
#endif /* FOOBAR */
ADD_CMD( viewer,	do_new_cuda_vwr,	create a new cuda image viewer )
ADD_CMD( load,		do_load_cuda_vwr,	write an image to a cuda viewer )
ADD_CMD( map,		do_map_cuda_vwr,	map an image to a cuda viewer )
MENU_END( cuda_gl )

static COMMAND_FUNC(do_cuda_gl_menu)
{
	/* Do cuda-specific init here? */
	PUSH_MENU(cuda_gl);
}

#endif /* HAVE_GLUT */
#endif /* HAVE_OPENGL */

#endif // FOOBAR

//CUDA EVENT CHECKPOINTING COMMAND TABLE
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(cuda_event_menu,s,f,h)

MENU_BEGIN( cuda_event )
ADD_CMD( max_checkpoints,	do_init_checkpoints,	set maximum number of checkpoints )
ADD_CMD( set_checkpoint,	do_set_checkpoint,	set a checkpoint )
ADD_CMD( reset,			do_clear_checkpoints,	clear all checkpoints )
ADD_CMD( show,			do_show_checkpoints,	show checkpoint times )
MENU_END( cuda_event )

static COMMAND_FUNC( do_cuda_event_menu )
{
	PUSH_MENU(cuda_event);
}


//CUDA EVENT CHECKPOINTING COMMAND TABLE
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(cuda_stream_menu,s,f,h)

MENU_BEGIN( cuda_stream )
ADD_CMD( stream,	do_new_stream,		create a new stream )
ADD_CMD( list,		do_list_cuda_streams,	list all streams )
ADD_CMD( info,		do_cuda_stream_info,	report information about a stream )
ADD_CMD( sync,		do_sync_stream,		synchronize host execution with a stream )
MENU_END( cuda_stream )

static COMMAND_FUNC( do_cuda_stream_menu )
{
	PUSH_MENU(cuda_stream);
}

static COMMAND_FUNC( do_prt_cap )
{
#ifdef HAVE_CUDA
#ifdef CUDA_COMP_CAP
	sprintf(MSG_STR,"Compiled for compute capability %d.%d",
		CUDA_COMP_CAP/10,CUDA_COMP_CAP%10);
#else	// ! CUDA_COMP_CAP
	ERROR1("CAUTIOUS:  HAVE_CUDA is defined, but CUDA_COMP_CAP is not!?!?");
#endif	// ! CUDA_COMP_CAP
#else	// ! HAVE_CUDA
	sprintf(MSG_STR,"No CUDA support in this build");
#endif	// ! HAVE_CUDA

	prt_msg(MSG_STR);
}

static COMMAND_FUNC( do_about_cuda )
{
#ifdef HAVE_CUDA
	sprintf(MSG_STR,"CUDA version:  %d.%d",
		CUDA_VERSION/1000,(CUDA_VERSION%100)/10);
	prt_msg(MSG_STR);
#else // ! HAVE_CUDA
	prt_msg("No CUDA support in this build");
#endif // ! HAVE_CUDA
	do_report_npp_version(SINGLE_QSP_ARG);
}


//CUDA MAIN MENU
#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(cuda_menu,s,f,h)

MENU_BEGIN( cuda )
ADD_CMD( devices,	query_cuda_devices,	list all cuda devices )
ADD_CMD( capability,	do_prt_cap,		print GPU compute capability )
ADD_CMD( about_cuda,	do_about_cuda,		print CUDA software versions)
//ADD_CMD( test_blas,	do_test_blas,		simple test of CUDA BLAS )
ADD_CMD( list,		do_list_cudevs,		list all cuda devices )
ADD_CMD( info,		do_cudev_info,		print information about a device )
//ADD_CMD( upload,	do_gpu_obj_upload,		upload data to a GPU )
//ADD_CMD( dnload,	do_gpu_obj_dnload,		download data from a GPU )
//ADD_CMD( compute,	do_cuda_func_menu,	cuda compute function submenu )
ADD_CMD( npp,		do_npp_menu,		NPP library submenu )

#ifdef FOOBAR	// obsolete with new platform stuff...
#ifdef HAVE_OPENGL
#ifdef HAVE_GLUT
ADD_CMD( cuda_gl,	do_cuda_gl_menu,	cuda GL submenu )
#endif /* HAVE_GLUT */
#endif /* HAVE_OPENGL */
#endif // FOOBAR	// obsolete with new platform stuff...

ADD_CMD( streams,	do_cuda_stream_menu,	CUDA stream submenu )
ADD_CMD( events,	do_cuda_event_menu,	cuda event  submenu )
MENU_END( cuda )

// make this callable from regular C instead of just cplusplus...
extern "C" {

//#include "veclib/cu2_menu_prot.h"	// cu2_init_platform()
extern void cu2_init_platform(SINGLE_QSP_ARG_DECL);	// BUG include file conflicts with old macros...

void init_cuda_devices(SINGLE_QSP_ARG_DECL)
{
#ifdef HAVE_CUDA
#ifdef FOOBAR
	_init_cuda_devices(SINGLE_QSP_ARG);
#endif // FOOBAR
	cu2_init_platform(SINGLE_QSP_ARG);
#else // ! HAVE_CUDA
	WARN("init_cuda_devices:  no CUDA support in this build!?");
#endif // ! HAVE_CUDA
}

COMMAND_FUNC( do_cuda_menu )
{
	static int cuda_inited=0;

	if( ! cuda_inited ){
//		init_cuda_devices(SINGLE_QSP_ARG);
		cu2_init_platform(SINGLE_QSP_ARG);
		//auto_version(QSP_ARG  "CUDA","VersionId_cuda");
		cuda_inited=1;
	}

	PUSH_MENU(cuda);
}

}

