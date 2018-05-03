
#include "quip_config.h"

//#include "query_prot.h"
#include "quip_prot.h"

#include "data_obj.h"
#include "veclib_api.h"
#include "vt_api.h"
#include "camera_api.h"
#include "view_cmds.h"
#include "server.h"	// do_http_menu
#include "polh_menu.h"	// do_polh
#include "query_stack.h"	// BUG?  elim dependency...

#ifdef HAVE_CUDA
#include "cuda_api.h"
#endif /* HAVE_CUDA */

#ifdef BUILD_FOR_MACOS
//#define MINIMAL_BUILD
#endif // BUILD_FOR_MACOS

#define ADD_CMD(s,f,h)	ADD_COMMAND(quip_menu,s,f,h)

MENU_BEGIN(quip)
ADD_CMD(	data,		do_dobj_menu,	data object submenu )
ADD_CMD(	ports,		do_port_menu,	network port submenu )

// define MINIMAL_BUILD to make a stripped-down version for testing...
#ifndef MINIMAL_BUILD

#ifdef HAVE_POLHEMUS
ADD_CMD(	polhemus,	do_polh,	polhemus fasttrak submenu )
#endif // HAVE_POLHEMUS

ADD_CMD(	compute,	do_comp_menu,	computation submenu )
ADD_CMD(	veclib,		do_vl_menu,	vector function submenu )
ADD_CMD(	expressions,	do_exprs,	vector expression language submenu )
ADD_CMD(	fileio,		do_fio_menu,	file I/O submenu )
ADD_CMD(	http,		do_http_menu,	http server submenu )

#ifdef VIEWERS
ADD_CMD(	view,		do_view_menu,	image viewer submenu )
ADD_CMD(	genwin,		do_genwin_menu,	viewer/panel submenu )
#endif /* VIEWERS */

ADD_CMD(	cameras,	do_cam_menu,	camera submenu )

ADD_CMD(	movie,		do_movie_menu,	movie submenu )
ADD_CMD(	mseq,		do_mseq_menu,	M-sequence submenu )
//ADD_CMD(	staircases,	do_staircase_menu,	staircase submenu )
ADD_CMD(	experiments,	do_exp_menu,	psychophysical experiment submenu )
ADD_CMD(	sound,		do_sound_menu,	sound submenu )
ADD_CMD(	requantize,	do_requant,	dithering submenu )

// commented out to build in Xcode - how to configure?
//ADD_CMD(	fann,		do_fann_menu,	neural network submenu )

#ifndef BUILD_FOR_OBJC

// include menus even when hardware not present
// to avoid parsing errors.

ADD_CMD(	rawvol,		do_rv_menu,	raw disk volume submenu )
ADD_CMD(	aio,		do_aio_menu,	analog I/O submenu )
ADD_CMD(	knox,		do_knox_menu,	Knox RS8x8HB routing switcher )
ADD_CMD(	visca,		do_visca_menu,	Sony VISCA camera control protocol )

#endif // ! BUILD_FOR_OBJC

// or ifdef HAVE_GUI_INTERFACE???
//#ifdef HAVE_MOTIF
ADD_CMD(	interface,	do_protomenu,	user interface submenu)
//#endif /* HAVE_MOTIF */

#ifdef HAVE_OPENGL
ADD_CMD(	gl,		do_gl_menu,	OpenGL submenu )
#endif /* HAVE_OPENGL */

#ifndef BUILD_FOR_IOS
ADD_CMD(	platforms,	do_platform_menu,	compute platform menu )
#endif // BUILD_FOR_IOS

#ifdef HAVE_CUDA
ADD_CMD(	cuda,		do_cuda_menu,	nVidia CUDA submenu)
#endif /* HAVE_CUDA */

#ifdef HAVE_PIC
ADD_CMD(	pic,		do_pic_menu,	Microchip PIC submenu )
#endif /* HAVE_PIC */

#ifdef STEPIT
ADD_CMD(	stepit,		do_step_menu,	stepit submenu )
#endif /* STEPIT */

#ifdef HAVE_NUMREC
// Apparently there are situations where we have numrec but don't want to use it?
//#ifdef USE_NUMREC
ADD_CMD(	numrec,		do_nr_menu,	numerical recipes submenu )
//#endif // USE_NUMREC
#endif /* HAVE_NUMREC */

#ifndef BUILD_FOR_OBJC
ADD_CMD(	v4l2,		do_v4l2_menu,	V4L2 submenu )
#endif // ! BUILD_FOR_OBJC

#ifdef HAVE_METEOR
ADD_CMD(	meteor,		do_meteor_menu,	Matrox Meteor I frame grabber )
#endif

#ifdef HAVE_OPENCV
ADD_CMD(	opencv,		do_ocv_menu,	OpenCV submenu )
#endif

#ifdef HAVE_LIBDV
ADD_CMD(	dv,		do_dv_menu,	ieee1394 camera submenu )
#endif

#ifdef HAVE_GSL
ADD_CMD(	gsl,		do_gsl_menu,	GNU scientific library submenu )
#endif /* HAVE_GSL */

#ifdef HAVE_LIBDC1394
ADD_CMD(	pgr,		do_pgr_menu,	PGR camera submenu )
#endif /* HAVE_LIBDC1394 */

#ifdef HAVE_LIBFLYCAP
ADD_CMD(	fly,		do_fly_menu,	PGR camera submenu )
#endif /* HAVE_LIBFLYCAP */

#ifdef HAVE_LIBSPINNAKER
ADD_CMD(	spinnaker,	do_spink_menu,	FLIR/PGR Spinnaker camera submenu )
#endif /* HAVE_LIBFLYCAP */

#endif // ! MINIMAL_BUILD

MENU_END(quip)

// added for threads menu
//
//void push_quip_menu(Query_Stack *qsp)
//{
//	PUSH_MENU(quip);
//}

// start_quip executes on the main thread...

int main(int argc,char *argv[])
{
#ifdef BUILD_FOR_MACOS
    @autoreleasepool {
#endif // BUILD_FOR_MACOS
	input_on_stdin();
	//CHECK_MENU(quip);
	if( quip_menu == NULL ){
		init_quip_menu(SGL_DEFAULT_QSP_ARG);
	}
	start_quip_with_menu(argc,argv,quip_menu);
	while( QS_LEVEL(DEFAULT_QSP) >= 0 ){
		qs_do_cmd(DEFAULT_QSP);
	}
#ifdef BUILD_FOR_MACOS
    } // closing brace for autoreleasepool
#endif // BUILD_FOR_MACOS
	return 0;
}
//#import <Foundation/Foundation.h>


