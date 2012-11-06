#include "quip_config.h"

char VersionId_viewmenu_canmenu[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "savestr.h"
#include "debug.h"
#include "viewer.h"
#include "view_cmds.h"
#include "view_util.h"
#include "xsupp.h"


/* local prototypes */
static int name_in_use(QSP_ARG_DECL const char *s);

static int name_in_use(QSP_ARG_DECL const char *s)
{
	Viewer *vp;

	vp = VWR_OF(s);
	if( vp != NO_VIEWER){
		sprintf(error_string,"viewer name \"%s\" in use",s);
		WARN(error_string);
		return(1);
	}
	return(0);
}

static Viewer * mk_new_viewer(QSP_ARG_DECL int viewer_type)
{
	const char *s;
	char name[256];
	int dx,dy;
	Viewer *vp;

	s=NAMEOF("viewer name");
	strcpy(name,s);
	dx=HOW_MANY("width");
	dy=HOW_MANY("height");
	if( name_in_use(QSP_ARG name) ) return NO_VIEWER;
	if( dx <= 0 || dy <= 0 ){
		WARN("viewer sizes must be positive");
		return NO_VIEWER;
	}
	vp = viewer_init(QSP_ARG  name,dx,dy,viewer_type);

	if( vp == NO_VIEWER ) return NO_VIEWER;

	default_cmap(&vp->vw_top);
	show_viewer(QSP_ARG  vp);	/* default state is to be shown */
	select_viewer(QSP_ARG  vp);
	return vp;
}

COMMAND_FUNC( mk_viewer )
{
	Viewer *vp;
	vp=mk_new_viewer(QSP_ARG 0);
}

COMMAND_FUNC( mk_2d_adjuster )
{
	Viewer *vp;
	const char *s;

	vp=mk_new_viewer(QSP_ARG VIEW_ADJUSTER);
	s=NAMEOF("action text");
	if( vp == NO_VIEWER ) return;
	vp->vw_text = savestr(s);
}

COMMAND_FUNC( mk_gl_window )
{
	Viewer *vp;
	vp=mk_new_viewer(QSP_ARG VIEW_GL);
}

COMMAND_FUNC( mk_button_arena )
{
	Viewer *vp;
	char b1[LLEN],b2[LLEN],b3[LLEN];

	vp=mk_new_viewer(QSP_ARG VIEW_BUTTON_ARENA);
	strcpy(b1,NAMEOF("left button text"));
	strcpy(b2,NAMEOF("middle button text"));
	strcpy(b3,NAMEOF("right button text"));
	if( vp == NO_VIEWER ) return;
	vp->vw_text1 = savestr(b1);
	vp->vw_text2 = savestr(b2);
	vp->vw_text3 = savestr(b3);
}

COMMAND_FUNC( reset_button_funcs )
{
	Viewer *vp;
	char b1[LLEN],b2[LLEN],b3[LLEN];

	vp = PICK_VWR("");

	strcpy(b1,NAMEOF("left button text"));
	strcpy(b2,NAMEOF("middle button text"));
	strcpy(b3,NAMEOF("right button text"));

	if( vp == NO_VIEWER ) return;

	rls_str((char *)vp->vw_text1);
	rls_str((char *)vp->vw_text2);
	rls_str((char *)vp->vw_text3);

	vp->vw_text1 = savestr(b1);
	vp->vw_text2 = savestr(b2);
	vp->vw_text3 = savestr(b3);
}

COMMAND_FUNC( mk_mousescape )
{
	Viewer *vp;
	const char *s;

	vp=mk_new_viewer(QSP_ARG VIEW_MOUSESCAPE);
	s=NAMEOF("action text");
	if( vp == NO_VIEWER ) return;
	vp->vw_text = savestr(s);
}

COMMAND_FUNC( reset_window_text )
{
	const char *s;
	Viewer *vp;

	vp=PICK_VWR("");
	s=NAMEOF("window action text");

	if( vp == NO_VIEWER) return;
	if( vp->vw_text != NULL ) rls_str((char *)vp->vw_text);

	vp->vw_text = savestr(s);
}

COMMAND_FUNC( mk_dragscape )
{
	Viewer *vp;
	vp=mk_new_viewer(QSP_ARG VIEW_DRAGSCAPE);
}

COMMAND_FUNC( do_redraw )
{
	Viewer *vp;

	vp=PICK_VWR("");
	if( vp == NO_VIEWER) return;

	insure_x11_server(SINGLE_QSP_ARG);
	redraw_viewer(QSP_ARG  vp);
	select_viewer(QSP_ARG  vp);
}

COMMAND_FUNC( do_embed_image )
{
	Viewer *vp;
	Data_Obj *dp;
	int x,y;

	vp = PICK_VWR("");
	dp = PICK_OBJ("image");
	x=HOW_MANY("x position");
	y=HOW_MANY("y position");

	if( vp == NO_VIEWER || dp == NO_OBJ ){
		WARN("can't embed image");
		return;
	}

	INSIST_RAM(dp,"embed_image");

	insure_x11_server(SINGLE_QSP_ARG);
	add_image(vp,dp,x,y);
	embed_image(QSP_ARG  vp,dp,x,y);
}

COMMAND_FUNC( do_unembed_image )
{
	Viewer *vp;
	Data_Obj *dp;
	int x,y;

	dp = PICK_OBJ("image");
	vp = PICK_VWR("");
	x=HOW_MANY("x position");
	y=HOW_MANY("y position");

	if( vp == NO_VIEWER || dp == NO_OBJ ){
		WARN("can't unembed image");
		return;
	}

	INSIST_RAM(dp,"unembed_image");

	insure_x11_server(SINGLE_QSP_ARG);
	unembed_image(QSP_ARG  vp,dp,x,y);
}

COMMAND_FUNC( do_load_viewer )
{
	Viewer *vp;
	Data_Obj *dp;

	vp = PICK_VWR("");
	dp = PICK_OBJ("image");

	if( vp == NO_VIEWER || dp == NO_OBJ ) return;

	INSIST_RAM(dp,"load_viewer");

	insure_x11_server(SINGLE_QSP_ARG);
	load_viewer(QSP_ARG  vp,dp);
	select_viewer(QSP_ARG  vp);
}

#endif /* HAVE_X11 */

