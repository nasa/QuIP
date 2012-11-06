#include "quip_config.h"

char VersionId_viewmenu_dpymenu[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

/* manipulate displays */


#include <stdio.h>

#include "data_obj.h"
#include "getbuf.h"
#include "viewer.h"
#include "view_cmds.h"
#include "view_util.h"
#include "items.h"
#include "xsupp.h"

#include "dispobj.h"
#include "debug.h"	/* verbose */


/* local prototypes */
static COMMAND_FUNC( do_new_do );
static COMMAND_FUNC( do_open_do );
static COMMAND_FUNC( set_do );
static COMMAND_FUNC( do_tell_dpy );
static COMMAND_FUNC( do_info_do );


static COMMAND_FUNC( do_new_do )
{
	const char *s;

	s=NAMEOF("display");
	if( open_display(QSP_ARG  s,8) == NO_DISP_OBJ ){
		sprintf(error_string,"unable to open %s",s);
		WARN(error_string);
	}
}

static COMMAND_FUNC( do_open_do )
{
	const char *s;
	int d;

	s=NAMEOF("display");
	d=HOW_MANY("desired bit depth");
	if( open_display(QSP_ARG  s,d) == NO_DISP_OBJ ){
		sprintf(error_string,"unable to open %s",s);
		WARN(error_string);
	}
}

static COMMAND_FUNC( set_do )
{
	Disp_Obj *dop;

	dop = PICK_DISP_OBJ("");
	if( dop == NO_DISP_OBJ ) return;

	set_display(dop);
}

static COMMAND_FUNC( do_tell_dpy )
{
	Disp_Obj *dop;
	const char *s;

	s=NAMEOF("name of variable in which to deposit name of current display");

	dop = curr_dop();

#ifdef CAUTIOUS
	if( dop == NO_DISP_OBJ ){
		WARN("CAUTIOUS:  do_tell_dpy:  no current display!?");
		return;
	}
#endif /* CAUTIOUS */

	ASSIGN_VAR(s,dop->do_name);
}

static COMMAND_FUNC( do_info_do )
{
	Disp_Obj *dop;

	dop= PICK_DISP_OBJ("");
	if( dop == NO_DISP_OBJ ) return;
	info_do(dop);
}

static COMMAND_FUNC( do_list_dos )
{ list_disp_objs(SINGLE_QSP_ARG); }

Command doctbl[]={
{ "new",	do_new_do,	"open new display"			},
{ "open",	do_open_do,	"open new display w/ specified depth"	},
{ "info",	do_info_do,	"give information about a display"	},
{ "list",	do_list_dos,	"list displays"				},
{ "set",	set_do,		"select display for succeeding operations" },
{ "tell",	do_tell_dpy,	"report current default display"	},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL,		NULL,		NULL					}
};

COMMAND_FUNC( dpymenu )
{
	insure_x11_server(SINGLE_QSP_ARG);

	PUSHCMD(doctbl,"displays");
}

#endif /* HAVE_X11 */

