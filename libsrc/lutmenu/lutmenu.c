#include "quip_config.h"

char VersionId_lutmenu_lutmenu[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "data_obj.h"
#include "debug.h"
#include "lut_cmds.h"
#include "menuname.h"
#include "cmaps.h"
#include "xsupp.h"		/* x_dump_lut() */
#include "gen_win.h"		/* x_dump_lut() */

//Dpyable *current_dpyp;

static COMMAND_FUNC( do_newlut );
static COMMAND_FUNC( do_setlut );
static COMMAND_FUNC( show_current_lb );
static COMMAND_FUNC( do_dumplut );

Data_Obj *pick_lb(QSP_ARG_DECL  const char *pmpt)
{
	Data_Obj *dp;

	dp=PICK_OBJ(pmpt);
	/* here we ought to check for size, type etc. */
	return(dp);
}

static COMMAND_FUNC( do_newlut )
{
	const char *name;

	name = NAMEOF("Name of new colormap");
	if( new_colormap(QSP_ARG  name) == NO_OBJ )
		WARN("error making new colormap");
}
 
static COMMAND_FUNC( do_setlut )
{
	Data_Obj *dp;
	
	dp=PICK_LB("");
	if( dp==NO_OBJ ) return;

	set_colormap(dp);
}

static COMMAND_FUNC( show_current_lb )
{
#ifdef HAVE_X11
	if( current_dpyp->c_cm_dp == NO_OBJ )
		advise("no current colormap");
	else {
		sprintf(error_string,"current colormap is \"%s\"",
			current_dpyp->c_cm_dp->dt_name);
		advise(error_string);
	}
#endif
}

static COMMAND_FUNC( do_dumplut )
{
#ifdef PC
	do_ldlut();	/* stage table gets the args for us */
			/* and this will work right in deferred mode */
#else
	Data_Obj *dp;

	dp = PICK_LB("");
	if( dp == NO_OBJ ) return;

	/* dump_lut(dp); */

	/* BUG check for proper type and dimension here */

#ifdef HAVE_X11
	current_dpyp->c_cm_dp = dp;
	x_dump_lut(current_dpyp);
#endif /* HAVE_X11 */

#endif
}

Command lutbuf_ctbl[]={
{ "newlut",	do_newlut,	"Allocates LUT memory"			},
{ "setlut",	do_setlut,	"Sets the current LUT"			},
{ "dump",	do_dumplut,	"Dump named lut to HW"			},
{ "current",	show_current_lb,"show currently selected lut buffer"	},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"				},
#endif /* ! MAC */
{ NULL,		NULL,		NULL					}
};

COMMAND_FUNC( do_lutbufs )
{
	PUSHCMD(lutbuf_ctbl,LUTBUF_MENU_NAME);
}

