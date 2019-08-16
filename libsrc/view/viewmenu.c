#include "quip_config.h"

#include "quip_menu.h"
#ifndef BUILD_FOR_IOS
#include "pf_viewer.h"
#endif // BUILD_FOR_IOS

#include <stdio.h>
#include "quip_prot.h"
#include "viewer.h"
#include "view_cmds.h"
#include "view_prot.h"
#include "view_util.h"
#include "xsupp.h"
#include "lut_cmds.h"

char *display_name=NULL;

#include "get_viewer.h"

#ifdef HAVE_X11
void insure_x11_server(SINGLE_QSP_ARG_DECL)
{
	static int have_server=0;
	if( have_server ) return;

	window_sys_init(SINGLE_QSP_ARG);
	have_server=1;
}
#endif /* HAVE_X11 */

COMMAND_FUNC( do_show_viewer )
{
	Viewer *vp;

	GET_VIEWER("do_show_viewer")
	if( vp == NULL ) return;

	INSURE_X11_SERVER

	show_viewer(vp);
}

COMMAND_FUNC( do_delete_viewer )
{
	Viewer *vp;

	GET_VIEWER("do_delete_viewer")
	if( vp == NULL ) return;

	delete_viewer(vp);
}

COMMAND_FUNC( do_unshow_viewer )
{
	Viewer *vp;


	GET_VIEWER("do_unshow_viewer")
	if( vp == NULL ) return;

	INSURE_X11_SERVER
	SET_VW_FLAG_BITS(vp, VIEW_UNSHOWN);	/* in case not already mapped */
	unshow_viewer(vp);
}

COMMAND_FUNC( do_posn_viewer )
{
	Viewer *vp;
	int x,y;

	GET_VIEWER("do_posn_viewer")

	x = (int)how_many("x");
	y = (int)how_many("y");

warn("do_posn_viewer is deprecated, please use position function from genwin menu");

	if( vp == NULL ) return;

	INSURE_X11_SERVER
	posn_viewer(vp,x,y);
}

COMMAND_FUNC( do_xsync )
{
#ifdef HAVE_X11
	int yesno;
    
	INSURE_X11_SERVER
    
	yesno = askif("synchronize Xlib execution");
    
	if( yesno ) x_sync_on();
	else        x_sync_off();
#else /* ! HAVE_X11 */
	askif("synchronize Xlib execution (ineffective!)");
#endif /* ! HAVE_X11 */
}

COMMAND_FUNC( do_relabel )
{
	Viewer *vp;
	const char *s;

	GET_VIEWER("do_relabel")
	s=nameof("new label");
	if( vp == NULL ) return;

	INSURE_X11_SERVER
	relabel_viewer(vp,s);
}

COMMAND_FUNC( do_track )
{
	Viewer *vp;

	GET_VIEWER("do_track")
	if( vp == NULL ) return;
	if( !IS_ADJUSTER(vp) ){
		sprintf(ERROR_STRING,
			"viewer %s is not an adjuster",VW_NAME(vp));
		warn(ERROR_STRING);
		return;
	}
	SET_VW_FLAG_BITS(vp, VIEW_TRACK);
}

COMMAND_FUNC( do_geom )
{
	Viewer *vp;

	GET_VIEWER("do_geom")
	if( vp == NULL ) return;
#ifdef HAVE_X11
	INSURE_X11_SERVER
	show_geom(vp);
#endif /* HAVE_X11 */
}

COMMAND_FUNC( do_info_viewer )
{
	Viewer *vp;

	GET_VIEWER("do_info_viewer")
	if( vp==NULL ) return;

	info_viewer(vp);
}

#ifdef HAVE_X11_EXT

static COMMAND_FUNC( do_shm_setup )
{
	Viewer *vp;

	GET_VIEWER("do_shm_setup")
	if( vp==NULL ) return;

	shm_setup(vp);
}

static COMMAND_FUNC( do_shm_update )
{
	Viewer *vp;
	Data_Obj *dp;
	int x0,y0;

	GET_VIEWER("do_shm_update")
	dp=pick_obj("");
	x0 = how_many("x location");
	y0 = how_many("y location");

	if( vp == NULL || dp == NULL ) return;

	/* BUG should confirm sizes... */
	update_shm_viewer(vp,(char *)OBJ_DATA_PTR(dp),(int)OBJ_PXL_INC(dp),(int)OBJ_COMP_INC(dp),(int)OBJ_COLS(dp),(int)OBJ_ROWS(dp),x0,y0);
}

#endif /* HAVE_X11_EXT */

static COMMAND_FUNC( do_list_viewers )
{ list_vwrs(tell_msgfile()); }

#define ADD_CMD(s,f,h)	ADD_COMMAND(viewers_menu,s,f,h)

MENU_BEGIN(viewers)
#ifndef BUILD_FOR_IOS
ADD_CMD( platform_viewer,	do_new_pf_vwr,		create a new platform viewer )
#endif // BUILD_FOR_IOS
ADD_CMD( new,			mk_viewer,		create new image viewer )
ADD_CMD( plotter,		mk_plotter,		create new plot viewer )
ADD_CMD( pixmap,		mk_pixmap,		create an off-screen drawable )
ADD_CMD( adjuster,		mk_2d_adjuster,		create new image with adjuster )
#ifdef SGI_GL
ADD_CMD( glwindow,		mk_gl_window,		create new GL window )
#endif /* SGI_GL */
ADD_CMD( buttons,		mk_button_arena,	create a window to read mouse buttons )
ADD_CMD( actions,		reset_button_funcs,	redefine button actions )
ADD_CMD( event_action,		do_set_event_action,	define action for an event )
ADD_CMD( dragscape,		mk_dragscape,		create a viewer with draggable objects )
ADD_CMD( mousescape,		mk_mousescape,		create a viewer that intercepts mouse movement )
ADD_CMD( reset_window_text,	reset_window_text,	set window action text )
ADD_CMD( list,			do_list_viewers,	list all viewers and adjusters )
ADD_CMD( info,			do_info_viewer,		give info about a viewer )
#ifdef HAVE_X11_EXT
ADD_CMD( shm_setup,		do_shm_setup,		set up viewer for shared memory access )
ADD_CMD( shm_update,		do_shm_update,		update shared memory window )
#endif /* HAVE_X11_EXT */
ADD_CMD( delete,		do_delete_viewer,	delete viewer or adjuster )

MENU_END(viewers)



COMMAND_FUNC( do_viewer_menu )
{
	INSURE_X11_SERVER
	CHECK_AND_PUSH_MENU(viewers);
}

COMMAND_FUNC( do_select_vp )
{
	Viewer *vp;

	GET_VIEWER("do_select_vp")
	if( vp==NULL ) return;

	INSURE_X11_SERVER
	select_viewer(vp);
}

#ifndef HAVE_VBL
static int no_vbl_warning_given=0;
#endif /* ! HAVE_VBL */

static COMMAND_FUNC( do_vblank )
{
#ifdef HAVE_VBL
	int n;

	n=(int)how_many("number of fields to wait");
	
	INSURE_X11_SERVER
	
	while(n--)
		vbl_wait();

#else /* ! HAVE_VBL */

	how_many("number of fields to wait (ineffective)");
	if( ! no_vbl_warning_given ) {
		warn("Sorry, no vblank support in this build.");
		no_vbl_warning_given = 1;
	}

#endif /* ! HAVE_VBL */
}


static COMMAND_FUNC( do_wait )
{
	Viewer *vp;

	GET_VIEWER("do_wait")
	if( vp == NULL ) return;
#ifdef HAVE_X11
	INSURE_X11_SERVER
	wait_for_mapped(vp,10);
#endif /* HAVE_X11 */
}

static COMMAND_FUNC( do_discard_images )
{
	Viewer *vp;

	GET_VIEWER("do_discard_images");
	if( vp == NULL ) return;

#ifdef BUILD_FOR_IOS
	[ VW_IMAGES(vp) discard_subviews];
#else
	warn("stored images not implemented for this platform");
#endif
}

static COMMAND_FUNC( do_cycle_viewer )
{
	Viewer *vp;

	GET_VIEWER("do_cycle_viewer");
	if( vp == NULL ) return;

#ifdef BUILD_FOR_IOS
	if( VW_IMAGES(vp) == NULL ){
		sprintf(ERROR_STRING,
			"do_cycle_viewer:  viewer %s does not have an associated image viewer!?",
			VW_NAME(vp));
		warn(ERROR_STRING);
		return;
	}
	[ VW_IMAGES(vp) cycle_images];
#else
	warn("refresh event handling not implemented for this platform");
#endif

}

static COMMAND_FUNC( do_bring_fwd )
{
	Viewer *vp;
	Data_Obj *dp;

	GET_VIEWER("do_bring_fwd");
	dp = pick_obj("image");

	if( vp == NULL ) return;
	if( dp == NULL ) return;

#ifdef BUILD_FOR_IOS
	if( VW_IMAGES(vp) == NULL ){
		sprintf(ERROR_STRING,
			"do_bring_fwd:  viewer %s does not have an associated image viewer!?",
			VW_NAME(vp));
		warn(ERROR_STRING);
		return;
	}
	[ VW_IMAGES(vp) bring_to_front:dp ];
#else
	warn("do_bring_fwd:  image stacking not implemented for this platform");
#endif

}

static COMMAND_FUNC( do_send_back )
{
	Viewer *vp;
	Data_Obj *dp;

	GET_VIEWER("do_send_back");
	dp = pick_obj("image");

	if( vp == NULL ) return;
	if( dp == NULL ) return;

#ifdef BUILD_FOR_IOS
	if( VW_IMAGES(vp) == NULL ){
		sprintf(ERROR_STRING,
			"do_send_back:  viewer %s does not have an associated image viewer!?",
			VW_NAME(vp));
		warn(ERROR_STRING);
		return;
	}
	[ VW_IMAGES(vp) send_to_back:dp ];
#else
	warn("do_send_back:  image stacking not implemented for this platform");
#endif

}

static COMMAND_FUNC( do_hide_imgs )
{
	Viewer *vp;

	GET_VIEWER("do_send_back");

	if( vp == NULL ) return;

#ifdef BUILD_FOR_IOS
	if( VW_IMAGES(vp) == NULL ){
		sprintf(ERROR_STRING,
			"do_send_back:  viewer %s does not have an associated image viewer!?",
			VW_NAME(vp));
		warn(ERROR_STRING);
		return;
	}
	[ VW_IMAGES(vp) hide ];
#else
	//warn("do_hide_imgs:  image stacking not implemented for this platform");
#endif

}

// I forget what this does - VW_IMAGES is a canvas viewer in iOS???

static COMMAND_FUNC( do_reveal_imgs )
{
	Viewer *vp;

	GET_VIEWER("do_send_back");

	if( vp == NULL ) return;

#ifdef BUILD_FOR_IOS
	INSIST_IMAGE_VIEWER(reveal_images)

	if( VW_IMAGES(vp) == NULL ){
		sprintf(ERROR_STRING,
			"do_send_back:  viewer %s does not have an associated image viewer!?",
			VW_NAME(vp));
		warn(ERROR_STRING);
		return;
	}
	[ VW_IMAGES(vp) reveal ];
#else
	//warn("do_reveal_imgs:  image stacking not implemented for this platform");
#endif

}

static COMMAND_FUNC( do_cycle_func )
{
	const char *s;
	Viewer *vp;

	GET_VIEWER("do_cycle_func");
	s=nameof("text to interpret at next image flip");

	if( vp == NULL ) return;

#ifdef BUILD_FOR_IOS
	if( VW_IMAGES(vp) == NULL ){
		sprintf(ERROR_STRING,
			"do_cycle_viewer:  viewer %s does not have an associated image viewer!?",
			VW_NAME(vp));
		warn(ERROR_STRING);
		return;
	}
	[ VW_IMAGES(vp) set_cycle_func:s];
#else
	warn("image cycle functions not implemented for this platform");
	// suppress warning
	sprintf(ERROR_STRING,"Not interpreting \"%s\" in viewer 0x%lx",
		s,(long)vp);
	advise(ERROR_STRING);
#endif

}

static COMMAND_FUNC( do_cycle_done_func )
{
	const char *s;
	Viewer *vp;

	GET_VIEWER("do_cycle_done_func");
	s=nameof("text to interpret at end of animation cycle");

	if( vp == NULL ) return;

#ifdef BUILD_FOR_IOS
	if( VW_IMAGES(vp) == NULL ){
		sprintf(ERROR_STRING,
			"do_cycle_viewer:  viewer %s does not have an associated image viewer!?",
			VW_NAME(vp));
		warn(ERROR_STRING);
		return;
	}
	[ VW_IMAGES(vp) set_cycle_done_func:s];
#else
	warn("image cycle functions not implemented for this platform");
	// suppress warning
	sprintf(ERROR_STRING,"Not interpreting \"%s\" in viewer 0x%lx",
		s,(long)vp);
	advise(ERROR_STRING);
#endif

}

// another command developed to support animation on iOS...
//
// We have a stack of images that get cycled at the frame interrupts?
// duration is given in refreshes...
// see ../ios/quipImages.m, ../ios/quipImageView.m

static COMMAND_FUNC( do_animate_viewer )
{
	Viewer *vp;
	int frame_duration, n_repeats;

	GET_VIEWER("do_animate_viewer");
	frame_duration = (int)how_many("Number of refreshes per frame (<=0 to disable)");
	n_repeats = (int)how_many("Number of repetitions (0 for free-run)");
	if( vp == NULL ) return;

#ifdef BUILD_FOR_OBJC
	INSIST_IMAGE_VIEWER(refresh)
	set_viewer_animation(vp,frame_duration,n_repeats);
#else // ! BUILD_FOR_OBJC
	//warn("do_animate_viewer:  refresh event handling not implemented for this platform");

	// The goal of this is to cycle the loaded images every refresh...
	// For unix we ought to give a duration?
	// Can we get an event at the refresh???

	cycle_viewer_images(vp, frame_duration);
#endif // ! BUILD_FOR_OBJC
}

static COMMAND_FUNC( do_stop_animation )
{
	Viewer *vp;

	GET_VIEWER("do_stop_animation");
	if( vp == NULL ) return;
#ifdef BUILD_FOR_OBJC
	stop_viewer_animation(vp);
#endif // BUILD_FOR_OBJC
}

static COMMAND_FUNC( do_after_animation )
{
	Viewer *vp;
	char *s;

	GET_VIEWER("do_stop_animation");
	s = nameof("text to interpret when animation finishes");
	if( vp == NULL ) return;
#ifdef BUILD_FOR_OBJC
	exec_after_animation(vp,s);
#endif // BUILD_FOR_OBJC
}

static COMMAND_FUNC( do_lock_orientation )
{
	Viewer *vp;

	vp = pick_vwr("");
	if( vp == NULL ) return;
#ifdef BUILD_FOR_IOS
	[ VW_QVC(vp) setQvc_flags:
		VW_QVC(vp).qvc_flags & ~(QVC_ALLOWS_AUTOROTATION) ];
#endif /* BUILD_FOR_IOS */
}

// Following function implemented by GJS
static COMMAND_FUNC( do_set_backlight )
{
        float level;

        level = (float) how_much("backlight (0.0-1.0)");

        if( level < 0.0 || level > 1.0 ){
		sprintf(ERROR_STRING,"Backlight level (%g) must be between 0 and 1",level);
		warn(ERROR_STRING);
		return;
        }
#ifdef BUILD_FOR_IOS
        set_backlight((CGFloat)level);
#else // ! BUILD_FOR_IOS
        warn("Oops, can't set backlight, not an iOS device!?");
#endif // ! BUILD_FOR_IOS
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(view_menu,s,f,h)

MENU_BEGIN(view)
#ifndef BUILD_FOR_OBJC
ADD_CMD( displays,	dpymenu,	display object submenu )
#endif /* BUILD_FOR_OBJC */
ADD_CMD( level,         do_set_backlight,       set display backlight level )
ADD_CMD( viewers,	do_viewer_menu,	viewer object submenu)
ADD_CMD( fb,		do_fb_menu,	frame buffer device submenu )
ADD_CMD( select,	do_select_vp,	select viewer for implicit operations )
/* genwin support */
ADD_CMD( genwin,	do_genwin_menu,	general window operations submenu )
ADD_CMD( show,		do_show_viewer,	display viewing window )
ADD_CMD( unshow,	do_unshow_viewer,	hide viewing window )
ADD_CMD( animate,	do_animate_viewer,	start viewer animation with parameters)
ADD_CMD( after_animation,	do_after_animation,	specify script to execute after animation ends)
ADD_CMD( stop_animation,	do_stop_animation,	stop viewer animation)
ADD_CMD( cycle,		do_cycle_viewer,	cycle images associated with a viewer)
ADD_CMD( queue_frame,	do_queue_frame,		queue an animation frame for display in a viewer )
ADD_CMD( clear_queue,	do_clear_queue,		empty the animation queue )
ADD_CMD( forget_frame,	do_forget_frame,	forget OS-specific image copy )
ADD_CMD( cycle_func,	do_cycle_func,		specify script to run at next image cycling)
ADD_CMD( cycle_done_func,	do_cycle_done_func,	specify script to run at end of animation cycle)
ADD_CMD( bring_to_front,	do_bring_fwd,	bring an image to the front of a viewer)
ADD_CMD( send_to_back,	do_send_back,	send an image to the back of a viewer)
ADD_CMD( hide_images,	do_hide_imgs,	hide all images in a viewer)
ADD_CMD( reveal_images,	do_reveal_imgs,	reveal images in a viewer)
ADD_CMD( discard_images,	do_discard_images,	release stored images)
ADD_CMD( lock_orientation,	do_lock_orientation,	prevent auto-rotation (handheld device only) )
ADD_CMD( load,		do_load_viewer,	display image in a viewer )
#ifndef BUILD_FOR_IOS
ADD_CMD( platform_load,	do_load_pf_vwr,	display image in a platform viewer )
#endif // BUILD_FOR_IOS
ADD_CMD( embed,		do_embed_image,	embed an image in a viewer )
ADD_CMD( extract,	do_unembed_image,	extract an image from a viewer )
ADD_CMD( position,	do_posn_viewer,	position viewer )
ADD_CMD( vblank,	do_vblank,	wait for vertical blanking )
ADD_CMD( wait,		do_wait,	wait for viewer to be mapped on-screen )
ADD_CMD( label,		do_relabel,	relabel viewer )
ADD_CMD( luts,		do_lut_menu,	color map submenu )
ADD_CMD( redraw,	do_redraw,	redraw a viewer )
ADD_CMD( track,		do_track,	track motion events in adjuster )
#ifndef BUILD_FOR_OBJC
ADD_CMD( loop,		do_loop,	process window system events )
ADD_CMD( redir,		event_redir,	ignore keyboard )
ADD_CMD( unredir,	event_unredir,	attend to keyboard )
ADD_CMD( cursors,	do_cursors,	cursor submenu )
ADD_CMD( end_loop,	stop_loop,	cease processing window system events )
#endif /* ! BUILD_FOR_OBJC */
	/*
ADD_CMD( pixrect,	pr_menu,	pixrect submenu )
	*/
ADD_CMD( plot,		do_xp_menu,	plotting submenu )
ADD_CMD( draw,		do_draw_menu,	drawing submenu )
ADD_CMD( dragg,		draggmenu,	draggable object submenu )
ADD_CMD( xsync,		do_xsync,	enable or disable Xlib synchronization )
ADD_CMD( geometry,	do_geom,	show geometry of a window )
MENU_END(view)

static double viewer_exists(QSP_ARG_DECL  const char *name)
{
	Viewer *vp;

	vp=vwr_of(name);
	if( vp==NULL ) return(0.0);
	else return(1.0);
}

COMMAND_FUNC( do_view_menu )
{
	static int inited=0;

	if( !inited ){
		DECLARE_STR1_FUNCTION(	viewer_exists,	viewer_exists )

		/* We used to call window_sys_init() here, which is generally
		 * not a bad thing to do, but it caused problems when recompiling
		 * on client machines, because after the rebuild we want to update
		 * the version info file, which requires running the program and
		 * entering this menu (to trigger the calls to auto_version() above).
		 * The problem arises when the client machine itself has no display,
		 * and for one reason or another connection to the normal server
		 * cannot be made.  therefore, at the expense of a few extra instructions,
		 * we move this check to each of the commands in the main view menu.
		 */

		/* window_sys_init(); */	/* See comment directly above */

		/* Now with git, we have a version for the whole source collection,
		 * instead of version numbers on individual files.
		 * So perhaps the objection given above is no longer relevant?
		 */

		window_sys_init(SINGLE_QSP_ARG);

		/* genwin support */
		init_viewer_genwin();	

		inited=1;
	}

	CHECK_AND_PUSH_MENU(view);
}

//#else /* ! HAVE_X11 */
//
//COMMAND_FUNC( do_view_menu )
//{
//	warn("Program was not configured with X11 support.");
//}

//#endif /* ! HAVE_X11 */

