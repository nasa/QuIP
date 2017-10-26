#include "quip_config.h"

/* cursor package */

#include <string.h>
#include "quip_prot.h"
#include "viewer.h"
#include "view_cmds.h"
#include "view_util.h"

COMMAND_FUNC( do_make_cursor )
{
	Data_Obj *dp;
	char cname[64];
	int x,y;

	strcpy(cname,NAMEOF("cursor name"));
	dp = pick_obj( "bitmap image" );
	x=HOW_MANY("x coordinate of hot spot");
	y=HOW_MANY("y coordinate of hot spot");

	if( dp == NULL ) return;

	/* BUG should verify that dp is the right kind of image here */

	/* BUG should verify that x and y are within range */
	make_cursor(QSP_ARG  cname,dp,x,y);
}

COMMAND_FUNC( do_assign_cursor )
{
	View_Cursor *vcp;
	Viewer *vp;

	vp = pick_vwr("");
	vcp = pick_cursor( "cursor" );

	if( vp == NULL || vcp==NULL ) return;
	assign_cursor(vp,vcp);
}

COMMAND_FUNC( do_root_cursor )
{
	View_Cursor *vcp;

	vcp = pick_cursor( "cursor" );

	if( vcp==NULL ) return;

	root_cursor(vcp);
}

static COMMAND_FUNC( do_list_cursors ){ list_cursors(tell_msgfile()); }

#define ADD_CMD(s,f,h)	ADD_COMMAND(cursors_menu,s,f,h)

MENU_BEGIN(cursors)
ADD_CMD( make_cursor,	do_make_cursor,		create new cursor )
ADD_CMD( root_cursor,	do_root_cursor,		assign cursor to root window )
ADD_CMD( set_cursor,	do_assign_cursor,	assign cursor to viewer )
ADD_CMD( list,		do_list_cursors,	list all available cursors )

MENU_END(cursors)

COMMAND_FUNC( do_cursors )
{
	INSURE_X11_SERVER
	PUSH_MENU(cursors);
}

