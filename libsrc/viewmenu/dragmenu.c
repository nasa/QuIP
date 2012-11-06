#include "quip_config.h"

char VersionId_viewmenu_dragmenu[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

/* draggable objects for xlib !? */

#include <string.h>
#include "data_obj.h"
#include "savestr.h"
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

	INSIST_RAM(bm,"make_dragg")
	INSIST_RAM(dp,"make_dragg")

	make_dragg(QSP_ARG  s,bm,dp);
}

COMMAND_FUNC( do_embed_draggable )
{
	Viewer *vp;
	Draggable *dgp;
	int x,y;

	dgp = PICK_DRAGG("");
	vp = PICK_VWR("");

	x=HOW_MANY("x position");
	y=HOW_MANY("y position");

	if( dgp == NO_DRAGG || vp == NO_VIEWER ) return;

	dgp->dg_x=x;
	dgp->dg_y=y;
	dgp->dg_rx=0;
	dgp->dg_ry=0;

	embed_draggable(vp->vw_dp,dgp);

	addTail(vp->vw_draglist,dgp->dg_np);
}

static COMMAND_FUNC( do_list_draggs ){ list_draggs(SINGLE_QSP_ARG); }

Command dragg_ctbl[]={
{ "draggable",	do_make_dragg,		"create new draggable object"	},
{ "list",	do_list_draggs,		"list all draggables"		},
{ "embed",	do_embed_draggable,	"place a draggable in an image"	},
{ "quit",	popcmd,			"exit submenu"			},
{ NULL,		NULL,			NULL				}
};

COMMAND_FUNC( draggmenu )
{
	insure_x11_server(SINGLE_QSP_ARG);
	PUSHCMD(dragg_ctbl,"dragg");
}

#endif /* HAVE_X11 */

