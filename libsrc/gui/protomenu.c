#include "quip_config.h"

char VersionId_gui_protomenu[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif /* HAVE_STRING_H */

#include "gui.h"
#include "gui_cmds.h"
#include "debug.h"
#include "submenus.h"		/* do_xsync() */
#include "cmaps.h"		/* set_colormap() */
#include "xsupp.h"		/* which_display() */
#include "my_motif.h"
#include "function.h"		/* setdatafunc */

#ifdef HAVE_MOTIF
static COMMAND_FUNC( do_panel_cmap )
{
	Panel_Obj *po;
	Data_Obj *cm_dp;

	po = PICK_PANEL("");
	cm_dp = PICK_OBJ("colormap object");

	if( po == NO_PANEL_OBJ || cm_dp == NO_OBJ )
		return;
	panel_cmap(po,cm_dp);
}

static COMMAND_FUNC( end_decorate )
{
	if( curr_panel != NULL ){
		pop_item_context(QSP_ARG  scrnobj_itp);
	}
	popcmd(SINGLE_QSP_ARG);
}

Command creat_ctbl[]={
{ "position",	mk_position,	"set create position"			},
{ "button",	mk_button,	"create new button"			},
{ "toggle",	mk_toggle,	"create new toggle button"		},
#ifdef FOOBAR
{ "menu",	mk_menu_button,	"create new menu"			},
#endif /* FOOBAR */
{ "scroller",	mk_scroller,	"create new scrolling list"		},
{ "gauge",	mk_gauge,	"create new gauge"			},
{ "slider",	mk_slider,	"create new slider"			},
{ "slider_w",	mk_slider_w,	"create new slider with width spec"	},
{ "adjuster",	mk_adjuster,	"create new adjuster"			},
{ "adjuster_w",	mk_adjuster_w,	"create new adjuster with width spec."	},
{ "message",	mk_message,	"create new message"			},
{ "text",	mk_text,	"create new text input"			},
{ "edit_box",	mk_edit_box,	"create new edit box"			},
{ "cmap",	do_panel_cmap,	"set custion LUT for panel"		},
{ "chooser",	do_chooser,	"create new chooser"			},
{ "items",	do_set_scroller,"set scroller items"			},
{ "file_list",	do_file_scroller,"make a list of items from a file"	},
{ "get_text",	assign_text,	"assign text to script variable"	},
{ "get_position",do_get_posn_object,"assign obj posn to script variable"},
{ "set_message",do_set_message,	"update message"			},
{ "set_edit_text",do_set_edit_text,"update text"			},
{ "set_text_field",do_set_text_field,"update text field"		},
{ "set_text_prompt",do_set_prompt,"update text prompt"			},
{ "set_scale",	do_set_gauge,	"update gauge/slider/adjuster "		},
{ "set_position",do_set_posn_object,"move object"			},
{ "set_toggle",	do_set_toggle,	"set state of toggle"			},
{ "set_choice",	do_set_choice,	"set state of chooser"			},
{ "range",	set_new_range,	"reset slider range"			},
{ "slide_pos",	set_new_pos,	"set slider position"			},
{ "label_window",set_panel_label,"set panel window label"		},
{ "quit",	end_decorate,	"exit submenu"				},
{ NULL_COMMAND								}
};
#endif /* HAVE_MOTIF */


COMMAND_FUNC( creat_menu )
{
#ifdef HAVE_MOTIF
	Panel_Obj *po;

	po=PICK_PANEL( "" );
	if( po != NO_PANEL_OBJ ){
		curr_panel=po;
		PUSH_ITEM_CONTEXT(scrnobj_itp,po->po_icp);

#ifdef PO_XWIN
		set_curr_win( po->po_xwin );
		colormap( po->po_cm_dp );
#endif /* PO_XWIN */
	} else {
		curr_panel = NULL;
	}


	PUSHCMD(creat_ctbl,"create");

#else /* ! HAVE_MOTIF */
	ERROR1("creat_menu:  Program not configured with Motif.");
#endif /* ! HAVE_MOTIF */
}

/*
COMMAND_FUNC( do_xsync )
{
	if( askif("synchronize Xlib execution") )
		x_sync_on();
	else x_sync_off();
}
*/

#ifdef HAVE_MOTIF
Command control_ctbl[]={
/*{ "clear",	clear_screen,	"delete a panel"			}, */
{ "show",	do_show,	"display a panel or viewer"		},
{ "unshow",	do_unshow,	"stop displaying panel or viewer"	},
{ "dispatch",	do_dispatch,	"start implicit dispatching"		},
{ "position",	do_pposn,	"position a panel or viewer"		},
{ "xsync",	do_xsync,	"enable/disable Xlib synchronization"	},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL,		NULL,		NULL					},
};

COMMAND_FUNC( control_menu )
{
	PUSHCMD(control_ctbl,"control");
}

static COMMAND_FUNC( do_list_panels ){ list_panel_objs(SINGLE_QSP_ARG); }

Command so_ctbl[]={
{ "panels",	do_list_panels,	"list panels"				},
{ "objects",	do_list_panel_objs,"list objects belonging to a panel"	},
{ "info",	do_so_info,	"give object info"			},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL,		NULL,		NULL					}
};

COMMAND_FUNC( so_menu )
{
	PUSHCMD(so_ctbl,"objects");
}
#endif /* HAVE_MOTIF */

#ifdef MALLOC_DEBUG
COMMAND_FUNC( do_mallocverify )
{
	malloc_verify();
}

COMMAND_FUNC( do_mallocdebug )
{
	int n;

	n=how_many("malloc debugging level");
	if( n>= 0 && n <= 2 ) malloc_debug(n);
	else warn("bad level");
}
#endif /* MALLOC_DEBUG */

#ifdef HAVE_MOTIF
static COMMAND_FUNC( do_sel_panel )
{
	Panel_Obj *po;

	po=PICK_PANEL( "" );
	set_curr_win( po->po_xwin );
	set_colormap( po->po_cm_dp );
}
#endif /* HAVE_MOTIF */

Command proto_ctbl[]={
#ifdef HAVE_MOTIF
{ "panel",	mk_panel,	"create new control panel"		},
{ "decorate",	creat_menu,	"add objects to a panel"		},
{ "select",	do_sel_panel,	"select current panel for colormap ops"	},
{ "control",	control_menu,	"window system control"			},
{ "objects",	so_menu,	"object database submenu"		},
{ "delete",	do_delete,	"delete a canvas or panel"		},
{ "notice",	do_notice,	"give a notice"				},
#endif /* HAVE_MOTIF */
/* support for genwin */
{ "genwin",	genwin_menu,	"general window operations submenu"	},
{ "luts",	lutmenu,	"color map submenu"			},
#ifdef MALLOC_DEBUG
{ "malloc_debug",do_mallocdebug,"set malloc debugging level"		},
{ "malloc_verify",do_mallocverify,"verify memory heap"			},
#endif
{ "quit",	popcmd,		"exit program"				},
{ NULL,		NULL,		NULL					}
};

/* If SGI window manager, then popups are supported */

COMMAND_FUNC( protomenu )
{
	static int inited=0;

	if( ! inited ){
		static char word[32];
		char *str;
		const char *display_name;

		display_name = which_display();

		str=word;
		strcpy(str,"protomenu");
#ifdef HAVE_MOTIF
		so_init(QSP_ARG  1,&str);		/* screen_object init - why? */
		setstrfunc("panel_exists",panel_exists);
#endif /* HAVE_MOTIF */

		inited=1;
	}

	PUSHCMD(proto_ctbl,"proto");
}

