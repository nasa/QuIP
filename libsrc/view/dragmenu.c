#include "quip_config.h"

/* draggable objects for xlib !? */

#include <string.h>
#include "quip_prot.h"
#include "data_obj.h"
#include "node.h"
#include "viewer.h"
#include "view_util.h"
#include "view_cmds.h"

COMMAND_FUNC( do_make_dragg )
{
	Data_Obj *bm;
	Data_Obj *dp;
	char s[256];

	strcpy(s,NAMEOF("name for draggable"));
	bm=PICK_OBJ( "mask bitmap" );
	dp=PICK_OBJ( "image" );

	if( bm == NO_OBJ || dp == NO_OBJ ) return;

	INSIST_RAM_OBJ(bm,"make_dragg")
	INSIST_RAM_OBJ(dp,"make_dragg")

	make_dragg(QSP_ARG  s,bm,dp);
}

COMMAND_FUNC( do_embed_draggable )
{
	Viewer *vp;
	Draggable *dgp;
	int x,y;

	dgp = PICK_DRAGG("");
	vp = PICK_VWR("");

	x=(int)HOW_MANY("x position");
	y=(int)HOW_MANY("y position");

	if( dgp == NO_DRAGG || vp == NO_VIEWER ) return;

	dgp->dg_x=x;
	dgp->dg_y=y;
	dgp->dg_rx=0;
	dgp->dg_ry=0;

	embed_draggable(VW_OBJ(vp),dgp);

	addTail(VW_DRAG_LIST(vp),dgp->dg_np);
}

static COMMAND_FUNC( do_list_draggs ){ list_draggs(QSP_ARG  tell_msgfile(SINGLE_QSP_ARG)); }

#define ADD_CMD(s,f,h)	ADD_COMMAND(dragg_menu,s,f,h)

MENU_BEGIN(dragg)
ADD_CMD( draggable,	do_make_dragg,		create new draggable object )
ADD_CMD( list,		do_list_draggs,		list all draggables )
ADD_CMD( embed,		do_embed_draggable,	place a draggable in an image )
MENU_END(dragg)

COMMAND_FUNC( draggmenu )
{
	INSURE_X11_SERVER
	PUSH_MENU(dragg);
}

