
#include "quip_config.h"
#include "query_prot.h"
#include "data_obj.h"
#include "veclib_api.h"
#include "vt_api.h"
#include "camera_api.h"
#include "view_cmds.h"
#include "server.h"	// do_http_menu
#include "polh_menu.h"	// do_polh
#ifdef BUILD_FOR_IOS
#include "camera_api.h"
#endif // BUILD_FOR_IOS

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

#ifndef BUILD_FOR_OBJC

// include menus even when hardware not present
// to avoid parsing errors.

ADD_CMD(	rawvol,		do_rv_menu,	raw disk volume submenu )
ADD_CMD(	aio,		do_aio_menu,	analog I/O submenu )
ADD_CMD(	knox,		do_knox_menu,	Knox RS8x8HB routing switcher )
ADD_CMD(	visca,		do_visca_menu,	Sony VISCA camera control protocol )

#endif // ! BUILD_FOR_OBJC

#ifdef HAVE_GPS
ADD_CMD(	gps,		do_gps_menu,	communicate w/ GPS receiver )
#endif /* HAVE_GPS */


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
ADD_CMD(	numrec,		do_nr_menu,	numerical recipes submenu )
#endif /* HAVE_NUMREC */

//#ifdef HAVE_V4L2
#ifndef BUILD_FOR_OBJC
ADD_CMD(	v4l2,		do_v4l2_menu,	V4L2 submenu )
#endif // ! BUILD_FOR_OBJC
//#endif /* HAVE_V4L2 */


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

#ifdef HAVE_X11
//ADD_CMD(	atc,		do_atc_menu,	ATC submenu )
#endif /* HAVE_X11 */

#endif // ! MINIMAL_BUILD

MENU_END(quip)


// added for threads menu

void push_quip_menu(Query_Stack *qsp)
{
	PUSH_MENU(quip);
}

// start_quip executes on the main thread...

void start_quip(int argc, char **argv)
{
	CHECK_MENU(quip);
	start_quip_with_menu(argc,argv,quip_menu);
}


#ifdef MOVED_TO_NEW_FILE

void start_quip_with_menu(int argc, char **argv, Menu *initial_menu_p )
{
	Query_Stack *qsp;

	set_progname(argv[0]);

	//debug |= CTX_DEBUG_MASK;
	//debug |= GETBUF_DEBUG_MASK;

	qsp=init_first_query_stack();	// reads stdin?

	init_builtins();
	init_variables(SINGLE_QSP_ARG);	// specify dynamic variables
	declare_functions(SINGLE_QSP_ARG);

	//PUSH_MENU(quip);
	PUSH_MENU_PTR(initial_menu_p);

	set_args(QSP_ARG  argc,argv);
	rcfile(qsp,argv[0]);

	// If we have commands to create a widget in the startup file,
	// we get an error, so don't call exec_quip until after the appDelegate
	// has started...
	
} // end start_quip_with_menu

// This seems redundant with exec_pending_cmds ???

static void exec_qs_cmds( void *_qsp )
{
	Query_Stack *qsp=(Query_Stack *)_qsp;

	while( lookahead_til(QSP_ARG  0) ){
		while( QS_HAS_SOMETHING(qsp) && ! IS_HALTING(qsp) ){
			QS_DO_CMD(qsp);
		}
	}
}

void exec_this_level(SINGLE_QSP_ARG_DECL)
{
	exec_at_level(QSP_ARG  QLEVEL);
}

// BUG if we push a macro which contains only comments, then we could
// end up presenting an interactive prompt instead of popping back...

void exec_at_level(QSP_ARG_DECL  int level)
{
//#ifdef CAUTIOUS
//	if( level < 0 ){
//		fprintf(stderr,
//	"CAUTIOUS:  exec_at_level:  bad level %d!?\n",level);
//		abort();
//	}
//#endif // CAUTIOUS

	assert( level >= 0 );

	// We thought a lookahead here might help, but it didn't, probably
	// because lookahead does not skip comments?

	//lookahead(SINGLE_QSP_ARG);	// in case an empty macro was pushed?
	while( QLEVEL >= level ){
		qs_do_cmd(THIS_QSP);
		if( IS_HALTING(THIS_QSP) ){
			return;
		}

		/* The command may have disabled lookahead;
		 * We need to lookahead here to make sure
		 * the end of the text is properly detected.
		 */

		lookahead(SINGLE_QSP_ARG);
	}

	// BUG?  what happens if we halt execution when an alert is delivered?
}

// exec_quip executes commands as long as there is something to do.
// We might want to execute this in a queue other than the main queue
// in order to catch events while we are executing?
// This is important for display updates, but could cause problems for
// touch events which call exec_quip again?

void exec_quip(SINGLE_QSP_ARG_DECL)
{
#ifdef USE_QS_QUEUE

	// This is iOS only!

	//dispatch_async_f(QS_QUEUE(THIS_QSP),qsp,exec_qs_cmds);

	if( QS_QUEUE(THIS_QSP) == dispatch_get_current_queue() ){
		exec_qs_cmds(THIS_QSP);
	} else {
		dispatch_async_f(QS_QUEUE(THIS_QSP),THIS_QSP,exec_qs_cmds);
	}

#else /* ! USE_QS_QUEUE */

	exec_qs_cmds(THIS_QSP);

#endif /* ! USE_QS_QUEUE */
}

#endif // MOVED_TO_NEW_FILE

