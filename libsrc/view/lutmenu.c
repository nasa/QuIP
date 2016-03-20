#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "data_obj.h"
#include "lut_cmds.h"
#include "view_util.h"
#include "cmaps.h"
#include "xsupp.h"		/* x_dump_lut() */
#include "gen_win.h"		/* x_dump_lut() */

//Dpyable *current_dpyp;

static COMMAND_FUNC( do_newlut );
static COMMAND_FUNC( do_setlut );
static COMMAND_FUNC( show_current_lb );
static COMMAND_FUNC( do_dumplut );

static Data_Obj *pick_lb(QSP_ARG_DECL  const char *pmpt)
{
	Data_Obj *dp;

	dp=PICK_OBJ(pmpt);
	/* BUG here we ought to check for size, type etc. */
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
	if( DPA_CMAP_OBJ(current_dpyp) == NO_OBJ )
		advise("no current colormap");
	else {
		sprintf(ERROR_STRING,"current colormap is \"%s\"",
			OBJ_NAME(DPA_CMAP_OBJ(current_dpyp)));
		advise(ERROR_STRING);
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
	DPA_CMAP_OBJ(current_dpyp) = dp;
	x_dump_lut(current_dpyp);
#endif /* HAVE_X11 */

#endif
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(lutbufs_menu,s,f,h)

MENU_BEGIN(lutbufs)
ADD_CMD( newlut,	do_newlut,		Allocates LUT memory )
ADD_CMD( setlut,	do_setlut,		Sets the current LUT )
ADD_CMD( dump,		do_dumplut,		Dump named lut to HW )
ADD_CMD( current,	show_current_lb,	show currently selected lut buffer )
MENU_END(lutbufs)

COMMAND_FUNC( do_lutbufs )
{
	PUSH_MENU(lutbufs);
}

