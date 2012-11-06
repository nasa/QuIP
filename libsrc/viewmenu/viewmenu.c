#include "quip_config.h"

char VersionId_viewmenu_viewmenu[] = QUIP_VERSION_STRING;

#include "query.h"

#ifdef HAVE_X11

#include <stdio.h>

#include "node.h"
#include "viewer.h"
#include "view_cmds.h"
#include "version.h"
#include "debug.h"
#include "submenus.h"
#include "function.h"	/* setdatafunc() */
#include "view_util.h"
#include "xsupp.h"

char *display_name=NULL;

#include "get_viewer.h"

void insure_x11_server(SINGLE_QSP_ARG_DECL)
{
	static int have_server=0;
	if( have_server ) return;

	window_sys_init(SINGLE_QSP_ARG);
	have_server=1;
}

COMMAND_FUNC( do_show_viewer )
{
	Viewer *vp;

	GET_VIEWER("do_show_viewer")
	if( vp == NO_VIEWER ) return;

	insure_x11_server(SINGLE_QSP_ARG);

	show_viewer(QSP_ARG  vp);
}

COMMAND_FUNC( do_delete_viewer )
{
	Viewer *vp;

	GET_VIEWER("do_delete_viewer")
	if( vp == NO_VIEWER ) return;

	delete_viewer(QSP_ARG  vp);
}

COMMAND_FUNC( do_unshow_viewer )
{
	Viewer *vp;


	GET_VIEWER("do_unshow_viewer")
	if( vp == NO_VIEWER ) return;

	insure_x11_server(SINGLE_QSP_ARG);
	vp->vw_flags |= VIEW_UNSHOWN;	/* in case not already mapped */
	unshow_viewer(QSP_ARG  vp);
}

COMMAND_FUNC( do_posn_viewer )
{
	Viewer *vp;
	int x,y;

	GET_VIEWER("do_posn_viewer")

	x = HOW_MANY("x");
	y = HOW_MANY("y");

WARN("do_posn_viewer is deprecated, please use position function from genwin menu");

	if( vp == NO_VIEWER ) return;

	insure_x11_server(SINGLE_QSP_ARG);
	posn_viewer(vp,x,y);
}

COMMAND_FUNC( do_xsync )
{
	insure_x11_server(SINGLE_QSP_ARG);
	if( ASKIF("synchronize Xlib execution") )
		x_sync_on();
	else x_sync_off();
}

COMMAND_FUNC( do_relabel )
{
	Viewer *vp;
	const char *s;

	GET_VIEWER("do_relabel")
	s=NAMEOF("new label");
	if( vp == NO_VIEWER ) return;

	insure_x11_server(SINGLE_QSP_ARG);
	relabel_viewer(vp,s);
}

COMMAND_FUNC( do_track )
{
	Viewer *vp;

	GET_VIEWER("do_track")
	if( vp == NO_VIEWER ) return;
	if( !IS_ADJUSTER(vp) ){
		sprintf(ERROR_STRING,
			"viewer %s is not an adjuster",vp->vw_name);
		WARN(ERROR_STRING);
		return;
	}
	vp->vw_flags |= VIEW_TRACK;
}

COMMAND_FUNC( do_geom )
{
	Viewer *vp;

	GET_VIEWER("do_geom")
	if( vp == NO_VIEWER ) return;

	insure_x11_server(SINGLE_QSP_ARG);
	show_geom(vp);
}

COMMAND_FUNC( do_info_viewer )
{
	Viewer *vp;

	GET_VIEWER("do_info_viewer")
	if( vp==NO_VIEWER ) return;

	info_viewer(vp);
}

#ifdef HAVE_X11_EXT

COMMAND_FUNC( do_shm_setup )
{
	Viewer *vp;

	GET_VIEWER("do_shm_setup")
	if( vp==NO_VIEWER ) return;

	shm_setup(vp);
}

static COMMAND_FUNC( do_shm_update )
{
	Viewer *vp;
	Data_Obj *dp;
	int x0,y0;

	GET_VIEWER("do_shm_update")
	dp=PICK_OBJ("");
	x0 = HOW_MANY("x location");
	y0 = HOW_MANY("y location");

	if( vp == NO_VIEWER || dp == NO_OBJ ) return;

	/* BUG should confirm sizes... */
	update_shm_viewer(vp,(char *)dp->dt_data,(int)dp->dt_pinc,(int)dp->dt_cinc,(int)dp->dt_cols,(int)dp->dt_rows,x0,y0);
}

#endif /* HAVE_X11_EXT */

static COMMAND_FUNC( do_list_viewers )
{ list_vwrs(SINGLE_QSP_ARG); }

Command viewer_ctbl[]={
{ "new",	mk_viewer,	"create new image viewer"		},
{ "adjuster",	mk_2d_adjuster,	"create new image with adjuster"	},
#ifdef SGI_GL
{ "glwindow",	mk_gl_window,	"create new GL window"			},
#endif /* SGI_GL */
{ "buttons",	mk_button_arena,"create a window to read mouse buttons"	},
{ "actions",	reset_button_funcs,"redefine button actions"	},
{ "dragscape",	mk_dragscape,	"create a viewer with draggable objects"},
{ "mousescape",	mk_mousescape,	"create a viewer that intercepts mouse movement"},
{ "reset_window_text",	reset_window_text,	"set window action text"	},
{ "list",	do_list_viewers,	"list all viewers and adjusters"	},
{ "info",	do_info_viewer,	"give info about a viewer"		},
#ifdef HAVE_X11_EXT
{ "shm_setup",	do_shm_setup,	"set up viewer for shared memory access"},
{ "shm_update",	do_shm_update,	"update shared memory window"		},
#endif /* HAVE_X11_EXT */
{ "delete",	do_delete_viewer,"delete viewer or adjuster"		},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL,		NULL,		NULL					}
};

COMMAND_FUNC( viewer_menu )
{
	insure_x11_server(SINGLE_QSP_ARG);
	PUSHCMD(viewer_ctbl,"viewers");
}

COMMAND_FUNC( do_select_vp )
{
	Viewer *vp;

	GET_VIEWER("do_select_vp")
	if( vp==NO_VIEWER ) return;

	insure_x11_server(SINGLE_QSP_ARG);
	select_viewer(QSP_ARG  vp);
}

#ifdef HAVE_VBL

COMMAND_FUNC( do_vblank )
{
	int n;

	n=HOW_MANY("number of fields to wait");

	insure_x11_server(SINGLE_QSP_ARG);

	while(n--)
		vbl_wait();
}

#endif /* HAVE_VBL */

static COMMAND_FUNC( do_wait )
{
	Viewer *vp;

	GET_VIEWER("do_wait")
	if( vp == NO_VIEWER ) return;

	insure_x11_server(SINGLE_QSP_ARG);
	wait_for_mapped(QSP_ARG  vp,10);
}

Command view_ctbl[]={
{ "displays",	dpymenu,	"display object submenu"		},
{ "viewers",	viewer_menu,	"viewer object submenu"			},
#ifdef HAVE_FB_DEV
{ "fb",		fb_menu,	"frame buffer device submenu"		},
#endif /* HAVE_FB_DEV */
{ "select",	do_select_vp,	"select viewer for implicit operations"	},
/* genwin support */
{ "genwin",	genwin_menu,	"general window operations submenu"	},
{ "show",	do_show_viewer,	"display viewing window"		},
{ "unshow",	do_unshow_viewer,"hide viewing window"			},
{ "load",	do_load_viewer,	"display image in a viewer"		},
{ "embed",	do_embed_image,	"embed an image in a viewer"		},
{ "extract",	do_unembed_image,"extract an image from a viewer"	},
{ "position",	do_posn_viewer,	"position viewer"			},
#ifdef HAVE_VBL
{ "vblank",	do_vblank,	"wait for vertical blanking"		},
#endif /* HAVE_VBL */
{ "wait",	do_wait,	"wait for viewer to be mapped on-screen"	},
{ "label",	do_relabel,	"relabel viewer"			},
{ "luts",	lutmenu,	"color map submenu"			},
{ "redraw",	do_redraw,	"redraw a viewer"			},
{ "cursors",	do_cursors,	"cursor submenu"			},
{ "loop",	do_loop,	"process window system events"		},
{ "redir",	event_redir,	"ignore keyboard"			},
{ "unredir",	event_unredir,	"attend to keyboard"			},
{ "track",	do_track,	"track motion events in adjuster"	},
{ "end_loop",	stop_loop,	"cease processing window system events"	},
	/*
{ "pixrect",	pr_menu,	"pixrect submenu"			},
	*/
{ "plot",	xp_menu,	"plotting submenu"			},
{ "draw",	drawmenu,	"drawing submenu"			},
{ "dragg",	draggmenu,	"draggable object submenu"		},
{ "xsync",	do_xsync,	"enable or disable Xlib synchronization" },
{ "geometry",	do_geom,	"show geometry of a window"		},
{ "quit",	popcmd,	"exit program"					},
{ NULL,		NULL,		NULL					}
};

COMMAND_FUNC( viewmenu )
{
	static int inited=0;

	if( !inited ){
		auto_version(QSP_ARG  "VIEWER","VersionId_viewer");
		auto_version(QSP_ARG  "VIEWMENU","VersionId_viewmenu");
		auto_version(QSP_ARG  "XSUPP","VersionId_xsupp");
		setstrfunc("viewer_exists",viewer_exists);

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

		/* genwin support */
		init_viewer_genwin(SINGLE_QSP_ARG);	
		inited=1;
	}

	PUSHCMD(view_ctbl,"view");
}

#else /* ! HAVE_X11 */

COMMAND_FUNC( viewmenu )
{
	WARN("Program was not configured with X11 support.");
}

#endif /* ! HAVE_X11 */

