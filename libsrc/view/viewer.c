#include "quip_config.h"

#include <stdio.h>

#include "quip_prot.h"
#include "debug.h"
#include "function.h"		/* prototype for add_sizable */
//#include "gen_win.h"		/* add_genwin */
#include "linear.h"		/* set_lintbl */
#include "viewer.h"
#include "item_type.h"

// BUG not thread-safe
Viewer *curr_vp=NULL;

static int siz_done=0;

//ITEM_INTERFACE_DECLARATIONS(Viewer,vwr)

static IOS_Item_Type *vwr_itp=NULL;

// This is almost all of them!?
#ifdef BUILD_FOR_OBJC

IOS_ITEM_INIT_FUNC(Viewer,vwr,0)

#else /* ! BUILD_FOR_OBJC */

// override init func to call declare_canvas_events
// For IOS, this is done in initClass

void _init_vwrs(SINGLE_QSP_ARG_DECL)
{
	vwr_itp = new_item_type("Viewer", DEFAULT_CONTAINER_TYPE);
	declare_canvas_events(SINGLE_QSP_ARG);
}

#endif /* ! BUILD_FOR_OBJC */

IOS_ITEM_NEW_FUNC(Viewer,vwr)
IOS_ITEM_CHECK_FUNC(Viewer,vwr)
IOS_ITEM_GET_FUNC(Viewer,vwr)
IOS_ITEM_PICK_FUNC(Viewer,vwr)
IOS_ITEM_LIST_FUNC(Viewer,vwr)
IOS_ITEM_DEL_FUNC(Viewer,vwr)


static IOS_Item_Type *canvas_event_itp=NULL;
IOS_ITEM_INIT_FUNC(Canvas_Event,canvas_event,0)
IOS_ITEM_NEW_FUNC(Canvas_Event,canvas_event)
IOS_ITEM_CHECK_FUNC(Canvas_Event,canvas_event)
IOS_ITEM_PICK_FUNC(Canvas_Event,canvas_event)
IOS_ITEM_LIST_FUNC(Canvas_Event,canvas_event)

#ifndef BUILD_FOR_OBJC


static void rls_vw_lists(Viewer *);

#else	/* ! BUILD_FOR_IOS */

@implementation Canvas_Event

@synthesize event_code;

+(void) initClass
{
	canvas_event_itp = new_ios_item_type(DEFAULT_QSP_ARG  "Canvas_Event");
}

@end

@implementation Viewer
@synthesize bgcolor;
@synthesize fgcolor;
@synthesize frameno;
@synthesize depth;
@synthesize event_mask;
@synthesize image_lp;
@synthesize label;
@synthesize text;
@synthesize text1;
@synthesize text2;
@synthesize text3;
@synthesize vw_dp;
@synthesize drag_lp;
@synthesize draw_lp;
@synthesize vw_time;
@synthesize xmin;
@synthesize ymin;
@synthesize xdel;
@synthesize ydel;
@synthesize ctx_ref;
@synthesize gwp;

+(void) initClass
{
	vwr_itp = new_ios_item_type(DEFAULT_QSP_ARG  "Viewer");

	// Somewhere we need to declare the event types!
	declare_canvas_events(SGL_DEFAULT_QSP_ARG);
}

@end

void set_action_for_event(Viewer *vp,Canvas_Event *cep, const char *s)
{
	if( VW_EVENT_TBL(vp) == NULL ){
		SET_VW_EVENT_TBL(vp, [NSMutableArray arrayWithCapacity:N_CANVAS_EVENTS] );
		// We fill the array with empty strings,
		// so that we can replace at any index.
		for(int i=0;i<N_CANVAS_EVENTS;i++)
			[VW_EVENT_TBL(vp) insertObject: @"" atIndex:i];
	}

	[VW_EVENT_TBL(vp) replaceObjectAtIndex:cep.event_code
		withObject: STRINGOBJ(s) ];
}

#endif /* BUILD_FOR_OBJC */

#ifndef BUILD_FOR_OBJC

static void init_viewer_event_table(Viewer *vp)
{
	int i;
	SET_VW_EVENT_TBL(vp, (const char **) getbuf( sizeof(const char *) * N_CANVAS_EVENTS ) );
	for(i=0;i<N_CANVAS_EVENTS;i++){
		SET_VW_EVENT_ACTION(vp,i,NULL);
	}
}

void set_action_for_event(Viewer *vp,Canvas_Event *cep, const char *s)
{
	if( VW_EVENT_TBL(vp) == NULL )
		init_viewer_event_table(vp);

	if( VW_EVENT_ACTION(vp,CE_CODE(cep)) != NULL )
		rls_str(VW_EVENT_ACTION(vp,CE_CODE(cep)));

	SET_VW_EVENT_ACTION(vp,CE_CODE(cep),savestr(s));
}

static void rls_vw_lists(Viewer *vp)
{
	rls_list(vp->vw_image_list);
	rls_list(vp->vw_draglist);
	if( vp->vw_drawlist != NULL )
		rls_list(vp->vw_drawlist);
}

#endif /* ! BUILD_FOR_OBJC */

// We would like to be able to select a drawable (pixmap or viewer)
// rather than just a viewer...

void select_viewer(QSP_ARG_DECL  Viewer *vp)
{
	if( vp == NULL ) return;


	curr_vp = vp;		// BUG should phase out curr_vp...
    
#ifdef HAVE_X11
    
	/* when creating the viewer this hasn't been done yet... */
	if( VW_CMAP_OBJ(vp) != NULL ){
		select_cmap_display( VW_DPYABLE(vp) );
		set_colormap(VW_CMAP_OBJ(vp));
	} else {
sprintf(ERROR_STRING,"select_viewer %s:  no colormap",VW_NAME(vp));
advise(ERROR_STRING);
	}
#endif /* HAVE_X11 */

#ifndef BUILD_FOR_OBJC
	if( VW_LINTBL_OBJ(vp) != NULL )
		set_lintbl(QSP_ARG  VW_LINTBL_OBJ(vp));
#endif /* ! BUILD_FOR_OBJC */
}

void release_image(QSP_ARG_DECL  Data_Obj *dp)
{
	dp->dt_refcount --;
//sprintf(ERROR_STRING,"release_image %s:  refcount = %d",OBJ_NAME(dp),dp->dt_refcount);
//advise(ERROR_STRING);
	if( dp->dt_refcount <= 0 /* && IS_ZOMBIE(dp) */ )
		delvec(dp);
}

void delete_viewer(QSP_ARG_DECL  Viewer *vp)
{
	zap_viewer(vp);		/* window sys specific, no calls to givbuf... */
#ifndef BUILD_FOR_OBJC
	rls_list_nodes(vp->vw_image_list);	// necessary???
	rls_vw_lists(vp);	/* release list heads */
#endif /* BUILD_FOR_OBJC */

	if( VW_OBJ(vp) != NULL )
		release_image(QSP_ARG  VW_OBJ(vp) );
#ifndef BUILD_FOR_OBJC
	if( VW_CMAP_OBJ(vp) != NULL )
		release_image(QSP_ARG  VW_CMAP_OBJ(vp));
//}
	/* BUG the linearization table is owned by the display, not the viewer */
	/*
	if( VW_LINTBL_OBJ(vp) != NULL )
		release_image(QSP_ARG  VW_LINTBL_OBJ(vp));
	*/

#endif /* ! BUILD_FOR_OBJC */

	if( VW_LABEL(vp) != NULL )
		rls_str((char *)VW_LABEL(vp));

	del_vwr(vp);	// releases the name
	select_viewer(QSP_ARG  NULL);
}

// For ios viewers, see genwin.c

static double get_vw_size(QSP_ARG_DECL  IOS_Item *ip,int index)
{
	double d;
	Viewer *vp;

	vp = (Viewer *)ip;
	switch(index){
		case 0: d = VW_DEPTH(vp)/8; break;
		case 1:	d = VW_WIDTH(vp); break;
		case 2:	d = VW_HEIGHT(vp); break;
		default: d=1.0; break;
	}
	return(d);
}

static double get_vw_posn(QSP_ARG_DECL  IOS_Item *ip, int index )
{
	double d=(-1);
	Viewer *vp;

	vp = (Viewer *)ip;
	switch(index){
		case 0: d = VW_X_REQUESTED(vp); break;
		case 1: d = VW_Y_REQUESTED(vp); break;
#ifdef CAUTIOUS
		default:
			error1("CAUTIOUS:  get_vw_posn:  bad index!?");
			break;
#endif // CAUTIOUS
	}
	return(d);
}

static IOS_Size_Functions view_sf={
	get_vw_size,
	(const char *(*)(QSP_ARG_DECL  IOS_Item *))default_prec_name
};

static IOS_Position_Functions view_pf={
	get_vw_posn
};

Viewer *viewer_init(QSP_ARG_DECL  const char *name,int dx,int dy,int flags)
{
	Viewer *vp;
#ifdef HAVE_X11
	char str[256];
#endif /* HAVE_X11 */
	int stat;

	if( dx <= 0 || dy <= 0 ){
		sprintf(ERROR_STRING,
	"Dimensions for viewer %s (%d,%d) must be positive",
			name,dx,dy);
		WARN(ERROR_STRING);
		return(NULL);
	}

	vp=new_vwr(name);
	if( vp == NULL ) return(vp);

	/* this might be better done in a global init routine... */
	if( !siz_done ){
		/* Can we handle mixed Item_Types and IOS_Item_Types??? */
		add_ios_sizable(vwr_itp,&view_sf, NULL );
#ifndef BUILD_FOR_OBJC
		add_positionable(vwr_itp,&view_pf,NULL );
#endif // BUILD_FOR_OBJC
		siz_done=1;
	}

	SET_VW_LABEL(vp, NULL);		/* use name if null */
	SET_VW_TIME(vp, (time_t) 0 );

	SET_VW_WIDTH(vp,dx);
	SET_VW_HEIGHT(vp,dy);

	SET_VW_XMIN(vp, 0);
	SET_VW_YMIN(vp, 0);
	SET_VW_XDEL(vp, 0);
	SET_VW_YDEL(vp, 0);

	/* should adopt a consistent policy re initializing these lists!? */
	SET_VW_IMAGE_LIST(vp, new_list());
	SET_VW_DRAG_LIST(vp, new_list());
	SET_VW_DRAW_LIST(vp, NULL);		/* BUG should be in view_xlib.c */

#ifdef HAVE_X11
	vp->vw_ip = (XImage *) NULL;
	vp->vw_ip2 = (XImage *) NULL;

	// see XSetLineAttributes
	SET_VW_LINE_WIDTH(vp,1);
	SET_VW_LINE_STYLE(vp,LineSolid);
	SET_VW_CAP_STYLE(vp,CapRound);
	SET_VW_JOIN_STYLE(vp,JoinRound);
#endif /* HAVE_X11 */
	SET_VW_FRAMENO(vp, 0);
	SET_VW_TEXT(vp, NULL);
	// old-style event handling...
	SET_VW_TEXT1(vp, NULL);
	SET_VW_TEXT2(vp, NULL);
	SET_VW_TEXT3(vp, NULL);

	/* we used to do this only if the display depth was 8, but now we simulate
	 * lutbuffers on 24 bit displays...
	 *
	 * BUG we need to have some way to do this only when requested...
	 */
#ifdef HAVE_X11
	window_sys_init(SINGLE_QSP_ARG);/* make sure that the_dop is not NULL */
	set_viewer_display(QSP_ARG  vp);		/* sets vw_dop... */
	cmap_setup(vp);		/* refers to vw_dop, but that's not set until later? */

	install_default_lintbl(QSP_ARG  VW_DPYABLE(vp) );
	sprintf(str,"colormap.%s",name);

	VW_CMAP_OBJ(vp) = new_colormap(QSP_ARG  str);

	/* new_colormap() used to call set_colormap(),
	 * but that generated a warning because the window
	 * doesn't actually exist yet.
	 * Better to do it in select_viewer().
	 */

	if( VW_CMAP_OBJ(vp) != NULL )
		VW_CMAP_OBJ(vp)->dt_refcount++;

#endif /* HAVE_X11 */


	/* do window system specific stuff */

	if( flags & VIEW_ADJUSTER )
		stat=make_2d_adjuster(vp,dx,dy);
	else if( flags & VIEW_DRAGSCAPE )
		stat=make_dragscape(vp,dx,dy);
	else if( flags & VIEW_MOUSESCAPE )
		stat=make_mousescape(vp,dx,dy);
	else if( flags & (VIEW_BUTTON_ARENA|VIEW_PLOTTER) ){
		stat=make_button_arena(vp,dx,dy);
	}
#ifdef SGI_GL
	else if( flags & VIEW_GL )
		stat=make_gl_window(vp,dx,dy);
#endif /* SGI_GL */
	else
		stat=make_viewer(vp,dx,dy);

	if( stat < 0 ){		/* probably can't open DISPLAY */
#ifndef BUILD_FOR_OBJC
		rls_vw_lists(vp);
#endif /* BUILD_FOR_OBJC */
		del_vwr(vp);
		return(NULL);
	}

	// in ios, these properties are part of the associated Gen_Win,
	// and so must be set after make_viewer is called...
	SET_VW_X(vp, 0);
	SET_VW_Y(vp, 0);
	SET_VW_FLAGS(vp, flags);
	SET_VW_FLAG_BITS(vp, VIEW_LUT_SIM);
	SET_VW_EVENT_TBL(vp, NULL);

#ifdef BUILD_FOR_IOS
	left_justify(vp);
#endif /* BUILD_FOR_IOS */


	SET_VW_DEPTH(vp, display_depth() );

	SET_VW_OBJ(vp, NULL);

#ifdef HAVE_OPENGL
	SET_VW_OGL_CTX(vp,NULL);
#endif /* HAVE_OPENGL */

	SET_GW_TYPE( VW_GW(vp), GW_VIEWER );

	select_viewer(QSP_ARG  vp);

	return(vp);
} // end viewer_init


IOS_Node *first_viewer_node(SINGLE_QSP_ARG_DECL)
{
	IOS_List *lp;

	lp=ios_item_list(vwr_itp);
	if( lp==NULL ) return(NULL);
	else return(IOS_LIST_HEAD(lp));
}

IOS_List *viewer_list(SINGLE_QSP_ARG_DECL)
{
	return( ios_item_list(vwr_itp) );
}

void info_viewer(QSP_ARG_DECL  Viewer *vp)
{
	const char *vtyp;

	if( IS_ADJUSTER(vp) ) vtyp="Adjuster";
	else if( IS_DRAGSCAPE(vp) ) vtyp="Dragscape";
	else if( IS_TRACKING(vp) ) vtyp="Tracking_Viewer";
	else if( IS_BUTTON_ARENA(vp) ) vtyp="Click_Viewer";
	else vtyp="Viewer";

	sprintf(msg_str,"%s \"%s\", %d rows, %d columns, at %d %d",vtyp,VW_NAME(vp),
		VW_HEIGHT(vp),VW_WIDTH(vp),VW_X(vp),VW_Y(vp));
	prt_msg(msg_str);

	if( VW_OBJ(vp) != NULL ){
		sprintf(msg_str,
			"\tassociated data object:  %s",OBJ_NAME(VW_OBJ(vp)));
		prt_msg(msg_str);
	} else prt_msg("\tNo associated image");

#ifdef HAVE_X11
	if( SIMULATING_LUTS(vp) ){
		prt_msg("\tSimulating LUT color mapping");
		sprintf(msg_str,"\t\tcolormap object:  %s",OBJ_NAME(VW_CMAP_OBJ(vp)));
		prt_msg(msg_str);
		if( VW_LINTBL_OBJ(vp) != NULL ){
			sprintf(msg_str,"\t\tlinearization table object:  %s",OBJ_NAME(VW_LINTBL_OBJ(vp)));
			prt_msg(msg_str);
		}
	}
#endif /* HAVE_X11 */

	extra_viewer_info(vp);
}

/* genwin support */

/* These don't seem like the right place to put these,
 * but we need the item_type in order for
 * add_genwin() to work...
 */

#ifndef BUILD_FOR_OBJC

static void genwin_viewer_show(QSP_ARG_DECL  const char *s)
{
	Viewer *vp;

	vp=get_vwr(s);
	if( vp == NULL ) return;
	show_viewer(vp);
	return;
}

static void genwin_viewer_unshow(QSP_ARG_DECL  const char *s)
{
	Viewer *vp;

	vp=get_vwr(s);
	if( vp == NULL ) return;
	unshow_viewer(vp);
	return;
}

static void genwin_viewer_posn(QSP_ARG_DECL  const char *s, int x, int y)
{
	Viewer *vp;

	vp=get_vwr(s);
	if( vp == NULL ) return;
	posn_viewer(vp, x, y);
	return;
}

static void genwin_viewer_delete(QSP_ARG_DECL  const char *s)
{
	Viewer *vp;

	vp=get_vwr(s);
	if( vp == NULL ) return;
	delete_viewer(QSP_ARG  vp);
	return;
}

static Genwin_Functions gwfp={
	(void (*)(QSP_ARG_DECL  const char *, int , int))genwin_viewer_posn,
	(void (*)(QSP_ARG_DECL  const char *))genwin_viewer_show,
	(void (*)(QSP_ARG_DECL  const char *))genwin_viewer_unshow,
	(void (*)(QSP_ARG_DECL  const char *))genwin_viewer_delete
};
#endif /* ! BUILD_FOR_OBJC */

void init_viewer_genwin(SINGLE_QSP_ARG_DECL)
{
	if( vwr_itp == NULL ) init_vwrs();
#ifndef BUILD_FOR_OBJC
	add_genwin(QSP_ARG  vwr_itp, &gwfp, NULL);
#endif /* ! BUILD_FOR_OBJC */
	return;
}

#define DECLARE_CANVAS_EVENT(name,code)			\
							\
	cep = new_canvas_event(#name);			\
	SET_CE_CODE(cep,code);

void declare_canvas_events(SINGLE_QSP_ARG_DECL)
{
	Canvas_Event *cep;

	// We treat the three buttons like three touches?
	DECLARE_CANVAS_EVENT( touch_up,			CE_TOUCH_UP )
	DECLARE_CANVAS_EVENT( touch_down,		CE_TOUCH_DOWN )
	DECLARE_CANVAS_EVENT( touch_move,		CE_TOUCH_MOVE )

	DECLARE_CANVAS_EVENT( left_button_up,		CE_LEFT_BUTTON_UP )
	DECLARE_CANVAS_EVENT( left_button_down,		CE_LEFT_BUTTON_DOWN )
	DECLARE_CANVAS_EVENT( middle_button_up,		CE_MIDDLE_BUTTON_UP )
	DECLARE_CANVAS_EVENT( middle_button_down,	CE_MIDDLE_BUTTON_DOWN )
	DECLARE_CANVAS_EVENT( right_button_up,		CE_RIGHT_BUTTON_UP )
	DECLARE_CANVAS_EVENT( right_button_down,	CE_RIGHT_BUTTON_DOWN )

	DECLARE_CANVAS_EVENT( mouse_move,		CE_MOUSE_MOVE )
}


