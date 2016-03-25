#include "quip_config.h"

/* manipulate displays */


#include <stdio.h>

#include "data_obj.h"
#include "getbuf.h"
#include "viewer.h"
#include "view_cmds.h"
#include "view_util.h"
#include "item_type.h"
#include "quip_menu.h"
#include "xsupp.h"

#include "dispobj.h"
#include "debug.h"	/* verbose */

static COMMAND_FUNC( do_new_do )
{
	const char *s;

	s=NAMEOF("display");
#ifdef HAVE_X11
	if( open_display(QSP_ARG  s,8) == NO_DISP_OBJ ){
		sprintf(ERROR_STRING,"unable to open %s",s);
		WARN(ERROR_STRING);
	}
#endif /* HAVE_X11 */
}

static COMMAND_FUNC( do_open_do )
{
	const char *s;
	int d;

	s=NAMEOF("display");
	d=HOW_MANY("desired bit depth");
#ifdef HAVE_X11
	if( open_display(QSP_ARG  s,d) == NO_DISP_OBJ ){
		sprintf(ERROR_STRING,"unable to open %s",s);
		WARN(ERROR_STRING);
	}
#endif /* HAVE_X11 */
}

static COMMAND_FUNC( set_do )
{
	Disp_Obj *dop;

	dop = PICK_DISP_OBJ("");
	if( dop == NO_DISP_OBJ ) return;
#ifdef HAVE_X11
	set_display(dop);
#endif /* HAVE_X11 */
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
#ifdef HAVE_X11
	info_do(dop);
#endif /* HAVE_X11 */
}

static COMMAND_FUNC( do_list_dos )
{ list_disp_objs(SINGLE_QSP_ARG); }



#define ADD_CMD(s,f,h)	ADD_COMMAND(displays_menu,s,f,h)

MENU_BEGIN(displays)
ADD_CMD( new,	do_new_do,		open new display )
ADD_CMD( open,	do_open_do,		open new display w/ specified depth )
ADD_CMD( info,	do_info_do,		give information about a display )
ADD_CMD( list,	do_list_dos,		list displays )
ADD_CMD( set,	set_do,			select display for succeeding operations )
ADD_CMD( tell,	do_tell_dpy,		report current default display )
MENU_END(displays)

COMMAND_FUNC( dpymenu )
{
#ifdef HAVE_X11
	insure_x11_server(SINGLE_QSP_ARG);
#endif /* HAVE_X11 */

	PUSH_MENU(displays);
}

