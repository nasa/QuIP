#include "quip_config.h"

char VersionId_xsupp_dpy[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11


/* manipulate displays */

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* atoi() */
#endif

#include "items.h"
#include "debug.h"
#include "xsupp.h"
//#include "../interpreter/callback.h"		/* add_event_func() */
#include "callback_api.h"		/* add_event_func() */
#include "function.h"	/* sizables */

/* local prototypes */
static Visual *Get15BitVisual(Disp_Obj *dop);
static Visual *Get16BitVisual(Disp_Obj *dop);
static Visual *GetSpecifiedVisual(Disp_Obj *dop,int depth);

static int dop_setup(Disp_Obj *dop,int desired_depth);
static int dop_open(QSP_ARG_DECL  Disp_Obj *dop);
static XVisualInfo *get_depth_list(Disp_Obj *dop,int depth,int *np);
static int find_visual(XVisualInfo *list,int n,int cl, int depth);
static XVisualInfo *get_vis_list(Disp_Obj *dop,int *np);

static int window_sys_inited=0;

ITEM_INTERFACE_DECLARATIONS(Disp_Obj,disp_obj)

#if defined(__cplusplus) || defined(c_plusplus)
#define xvi_class	c_class
#else
#define xvi_class	class
#endif

static Disp_Obj *current_dop=NO_DISP_OBJ;

void set_display( Disp_Obj *dop )
{
	current_dop = dop;
}

Disp_Obj *curr_dop(void)
{
	return(current_dop);
}

List *displays_list(SINGLE_QSP_ARG_DECL)
{
	if( disp_obj_itp == NO_ITEM_TYPE ) return(NO_LIST);
	return( item_list(QSP_ARG  disp_obj_itp) );
}

void info_do( Disp_Obj *dop )
{
	printf("Display %s:\n",dop->do_name);
	printf("\tdpy = 0x%lx\n",(u_long)dop->do_dpy);
	printf("\tvis = 0x%lx\n",(u_long)dop->do_visual);
	printf("\tgc  = 0x%lx\n",(u_long)dop->do_gc);
	printf("\tscreen = %d\n",dop->do_screen);
	printf("\trootw = %ld (0x%lx)\n",dop->do_rootw,dop->do_rootw);
	printf("\twidth = %d\n",dop->do_width);
	printf("\theight = %d\n",dop->do_height);
	printf("\tdepth = %d\n",dop->do_depth);
}



static int dop_open( QSP_ARG_DECL  Disp_Obj *dop )
{
	/* BUG - on Solaris, when we have DISPLAY set to :0,
	 * but are on a remote server, this call hangs...
	 * (e.g., run iview on vision from xterm on stiles,
	 * w/ DISPLAY erroneously set to :0 instead of stiles:0)
	 * We ought to put a watchdog timer here...
	 */

	if ( (dop->do_dpy=XOpenDisplay(dop->do_name)) == NULL) {
		sprintf(ERROR_STRING,
			"dop_open:  Can't open display \"%s\"\n",dop->do_name);
		NWARN(ERROR_STRING);
		/* remove the object */
		del_disp_obj(QSP_ARG  dop->do_name);
		rls_str((char *)dop->do_name);
		return(-1);
	}
	return(0);
}

static XVisualInfo *get_vis_list( Disp_Obj * dop, int *np )
{
	XVisualInfo *		visualList;
	XVisualInfo		vTemplate;

	vTemplate.screen=dop->do_screen;

	visualList = XGetVisualInfo(dop->do_dpy,VisualScreenMask,
		&vTemplate,np);

#ifdef DEBUG
if( debug & xdebug ){
sprintf(DEFAULT_ERROR_STRING,"%d visuals found",*np);
advise(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */

	return(visualList);
}

static XVisualInfo *get_depth_list( Disp_Obj * dop, int depth, int *np )
{
	XVisualInfo *		visualList;
	XVisualInfo		vTemplate;

	/* try to get an 8 bit psuedocolor visual */
	/* taken from Xlib prog manual p. 215 */

	vTemplate.depth=depth;
	vTemplate.screen=dop->do_screen;

	visualList = XGetVisualInfo(dop->do_dpy,VisualScreenMask|VisualDepthMask,
		&vTemplate,np);
	if( visualList == NULL ){
		sprintf(DEFAULT_ERROR_STRING,
			"get_depth_list(%d) got NULL from XGetVisualInfo!?",depth);
		NWARN(DEFAULT_ERROR_STRING);
	}
#ifdef DEBUG
if( debug & xdebug ){
sprintf(DEFAULT_ERROR_STRING,"%d visuals found with depth %d",*np,depth);
advise(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */

	return(visualList);
}

static int find_visual_by_id( XVisualInfo *list, int n,unsigned int id )
{
	int i;

	/* find one which of the matching class */
	for(i=0;i<n;i++)
		if( list[i].visualid == id  ) return(i);
	return(-1);
}

static int find_visual( XVisualInfo *list, int n, int cl, int depth )
{
	int i;

	/* find one which of the matching class */
	for(i=0;i<n;i++)
		if( list[i].xvi_class == cl  && list[i].depth >= depth) return(i);
	return(-1);
}

Visual *GetEightBitVisual( Disp_Obj * dop)
{
	static XVisualInfo *	visualList = 0;
	int			visualsMatched;
Visual *vis;
int i;

	if( visualList == 0)
		visualList = get_depth_list(dop,8,&visualsMatched);
	if( visualsMatched == 0 ) return(NULL);

	i=find_visual(visualList,visualsMatched,PseudoColor,8);
	if( i < 0 ){
		NWARN("no pseudocolor visual found!?");
		return(visualList[0].visual);
	}
#ifdef DEBUG
if( debug & xdebug ){
sprintf(DEFAULT_ERROR_STRING,"using visual %ld",visualList[i].visualid);
advise(DEFAULT_ERROR_STRING);
vis=DefaultVisual(dop->do_dpy,dop->do_screen);
sprintf(DEFAULT_ERROR_STRING,"default visual is %ld",vis->visualid);
advise(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */

	return(visualList[i].visual);
}

#define PREFERRED_MODE TrueColor
#define ALTERNATE_MODE DirectColor
#define PREFERRED_NAME "TrueColor"
#define ALTERNATE_NAME "DirectColor"


static Visual *GetSpecifiedVisual( Disp_Obj * dop, int depth )
{
	XVisualInfo *	visualList = 0;
	int			visualsMatched;
	int i;
	const char *name;

	if( visualList == 0){
		visualList = get_depth_list(dop,depth,&visualsMatched);
	}
	if( visualsMatched == 0 ) return(NULL);

	/* We need something more here, for openGL we have multiple visuals w/ 24 bit truecolor,
	 * but not all have a Z buffer...
	 * As a hack, we pass the desired visualID through the environment...
	 */
	{
		char *s;

		s=getenv("PREFERRED_VISUAL_ID");
		if( s != NULL ){
			int preferred_id;
sprintf(DEFAULT_ERROR_STRING,"Checking for PREFERRED_VISUAL_ID %s",s);
advise(DEFAULT_ERROR_STRING);
			preferred_id = atoi(s);	/* BUG decimal only, should parse hex too */
			i=find_visual_by_id(visualList,visualsMatched,preferred_id);
			if( i >= 0 ){
sprintf(DEFAULT_ERROR_STRING,"preferred visual id %d FOUND at index %d",preferred_id,i);
advise(DEFAULT_ERROR_STRING);
				return(visualList[i].visual);
			}
			sprintf(DEFAULT_ERROR_STRING,"Unable to find requested visual id %d",preferred_id);
			NWARN(DEFAULT_ERROR_STRING);
		}
	}
	i=find_visual(visualList,visualsMatched,PREFERRED_MODE,depth);
	if( i < 0 ){
		sprintf(DEFAULT_ERROR_STRING,"no %s visual found with depth %d!?",
			PREFERRED_NAME,depth);
		NWARN(DEFAULT_ERROR_STRING);
		i=find_visual(visualList,visualsMatched,ALTERNATE_MODE,depth);
		if( i < 0 ){
			sprintf(DEFAULT_ERROR_STRING,"no %s visual found with depth %d!?",
				ALTERNATE_NAME,depth);
			NWARN(DEFAULT_ERROR_STRING);
			return(visualList[0].visual);
		} else {
			name = ALTERNATE_NAME;
		}
	} else {
		name = PREFERRED_NAME;
	}

if( verbose ){
sprintf(DEFAULT_ERROR_STRING,"i=%d, using visual %ld (%s, depth = %d)",
i, visualList[i].visualid,name,depth);
advise(DEFAULT_ERROR_STRING);
}
	return(visualList[i].visual);
}

static Visual *Get16BitVisual( Disp_Obj * dop)
{
	return( GetSpecifiedVisual(dop,16) );
}

Visual *Get24BitVisual( Disp_Obj * dop)
{
	return( GetSpecifiedVisual(dop,24) );
}

/* powerbook display */

static Visual *Get15BitVisual( Disp_Obj * dop)
{
	return( GetSpecifiedVisual(dop,15) );
}


static int dop_setup( Disp_Obj *dop, int desired_depth)
{
	XVisualInfo vinfo, *list;
	int n;

	dop->do_screen 	= DefaultScreen(dop->do_dpy);
	dop->do_rootw	= RootWindow(dop->do_dpy,dop->do_screen);
	dop->do_currw	= RootWindow(dop->do_dpy,dop->do_screen);

#ifdef HAVE_OPENGL
	dop->do_ctx = NULL;
#endif /* HAVE_OPENGL */

#ifdef DEBUG
if( debug & xdebug ){
XWindowAttributes wa;
XGetWindowAttributes(dop->do_dpy,dop->do_rootw,&wa);
sprintf(DEFAULT_ERROR_STRING,"depth of root window = %d", wa.depth);
prt_msg(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */
	dop->do_gc	= DefaultGC(dop->do_dpy,dop->do_screen);


	if( verbose ){
sprintf(DEFAULT_ERROR_STRING,"desired depth is %d",desired_depth);
advise(DEFAULT_ERROR_STRING);
	}

	if( desired_depth == 8 ){
		dop->do_visual	= GetEightBitVisual(dop);
	} else if( desired_depth == 24 ){
		dop->do_visual	= Get24BitVisual(dop);
	} else if( desired_depth == 16 ){
		dop->do_visual = Get16BitVisual(dop);
	} else {
		dop->do_visual	= DefaultVisual(dop->do_dpy,dop->do_screen);
	}

	if( dop->do_visual == 0 ){
		if( verbose )
			advise("initial attempt to get a visual failed");
	}

	/* BUG? can't we do something better here? */
	if( dop->do_visual == 0 && desired_depth == 8 ){
		/* this works on powerbook */
		dop->do_visual = Get15BitVisual(dop);
	}
	if( dop->do_visual == 0 && desired_depth == 8 ){
		/* this works on durer... */
		dop->do_visual = Get16BitVisual(dop);
	}
	if( dop->do_visual == 0 ){	/* couldn't find anything !? */
		if( verbose ){
			int nvis,i;
			XVisualInfo *	visualList;

			visualList=get_vis_list(dop,&nvis);

			sprintf(DEFAULT_ERROR_STRING,"%d visuals found:",nvis);
			advise(DEFAULT_ERROR_STRING);

			for(i=0;i<nvis;i++){
				sprintf(DEFAULT_ERROR_STRING,"class %d   depth %d",
			visualList[i].xvi_class,visualList[i].depth);
				advise(DEFAULT_ERROR_STRING);
			}
		}

		return(-1);
	}

	/* remember the depth of this visual - do we still need to do this? */
	vinfo.visualid = XVisualIDFromVisual(dop->do_visual);
	list = XGetVisualInfo(dop->do_dpy,VisualIDMask,&vinfo,&n);
	if( n != 1 ){
		NWARN("more than one visual with specified ID!?");
		dop->do_depth = 8;
	} else {
		dop->do_depth = list[0].depth;
	}
	XFree(list);

	dop->do_width	= DisplayWidth(dop->do_dpy,dop->do_screen);
	dop->do_height	= DisplayHeight(dop->do_dpy,dop->do_screen);

#ifdef DEBUG
if( debug & xdebug ){
sprintf(msg_str,"display %s, %d by %d, %d bits deep",
dop->do_name,dop->do_width,dop->do_height,dop->do_depth);
prt_msg(msg_str);
}
#endif /* DEBUG */

	return(0);
}

static double get_dpy_size( Item *ip, int index)
{
	Disp_Obj *dop;
	dop=(Disp_Obj *)ip;
	switch( index ){
		case 0: return( (double) current_dop->do_depth );
		case 1: return( (double) current_dop->do_width );
		case 2: return( (double) current_dop->do_height );
		default:
			NWARN("unsupported display size function");
			return(0.0);
	}
}

static Size_Functions dpy_sf={
	/*(double (*)(Item *,int))*/		get_dpy_size,
	/*(Item * (*)(Item *,index_t))*/	NULL,
	/*(Item * (*)(Item *,index_t))*/	NULL,
	/*(double (*)(Item *))*/		NULL,
};



/*
 * Open the named display
 */

Disp_Obj *open_display(QSP_ARG_DECL  const char *name,int desired_depth)
{
	Disp_Obj *dop;
	static int siz_done=0;

	dop = new_disp_obj(QSP_ARG  name);
	if( dop == NO_DISP_OBJ ){
		sprintf(ERROR_STRING, "Couldn't create object for display %s",
					name);
		NWARN(ERROR_STRING);
		return(NO_DISP_OBJ);
	}

	if( dop_open(QSP_ARG  dop) < 0 ){
		return(NO_DISP_OBJ);
	}

	if( dop_setup(dop,desired_depth) < 0 ){
		/* Bug - XCloseDisplay?? */
		/* need to destroy object here */
		del_disp_obj(QSP_ARG  dop->do_name);
		rls_str((char *)dop->do_name);
		return(NO_DISP_OBJ);
	}
	set_display(dop);

	if( ! siz_done ){
		siz_done++;
		add_sizable(QSP_ARG  disp_obj_itp,&dpy_sf, NULL );
	}

	return(dop);
}

void window_sys_init(SINGLE_QSP_ARG_DECL)
{
	if( window_sys_inited ) return;

#ifdef DEBUG
	xdebug = add_debug_module(QSP_ARG  "xsupp");
#endif /* DEBUG */

	add_event_func(i_loop);
	set_discard_func( discard_events );

	if( current_dop == NO_DISP_OBJ ){
		current_dop = default_x_display(SINGLE_QSP_ARG);
		if( current_dop == NO_DISP_OBJ ){
			NWARN("Couldn't open default display!?");
			return;
		}
	}

	window_sys_inited=1;
}

#define DEFAULT_DISPLAY_DEPTH	24
#define ALTERNATE_DISPLAY_DEPTH	8

#define MAX_DISPLAY_DEPTHS	4
static int possible_depths[MAX_DISPLAY_DEPTHS]={24,8,16,15};

Disp_Obj * default_x_display(SINGLE_QSP_ARG_DECL)
{
	const char *dname;
	Disp_Obj *dop;
	int which_depth;
	char *s;

	dname = check_display();

	/* these two lines added so this can be called more than once */
	dop = disp_obj_of(QSP_ARG  dname);
	if( dop != NO_DISP_OBJ ) return(dop);

	s=getenv("DESIRED_DEPTH");
	if( s != NULL ){
		int desired_depth;

		desired_depth=atoi(s);

sprintf(ERROR_STRING,"Desired depth %d obtained from environment",desired_depth);
advise(ERROR_STRING);
		dop = open_display(QSP_ARG  dname,desired_depth);
		if( dop != NO_DISP_OBJ ) return(dop);

		sprintf(ERROR_STRING,"Unable to open display %s with $DESIRED_DEPTH (%d)",
			dname,desired_depth);
		NWARN(ERROR_STRING);
	}
			

	for(which_depth=0;which_depth<MAX_DISPLAY_DEPTHS;which_depth++){
		dop = open_display(QSP_ARG  dname,possible_depths[which_depth]);
		if( dop != NO_DISP_OBJ ){
			if( verbose ){
				sprintf(ERROR_STRING,
					"Using depth %d on display %s",
					possible_depths[which_depth],dname);
				advise(ERROR_STRING);
			}
			return(dop);
		} else {
			if( verbose && which_depth<(MAX_DISPLAY_DEPTHS-1) ){
				sprintf(ERROR_STRING,
			"Couldn't get %d bit visual on device %s, trying %d",
					possible_depths[which_depth],dname,
					possible_depths[which_depth+1]);
				advise(ERROR_STRING);
			}
		}
	}
	if( verbose ){
		sprintf(ERROR_STRING,
	"Couldn't get %d bit visual on device %s, giving up",
			possible_depths[MAX_DISPLAY_DEPTHS-1],dname);
		advise(ERROR_STRING);
	}
	return(dop);
} /* end default_x_display */

int display_depth(SINGLE_QSP_ARG_DECL)
{
	if( current_dop == NO_DISP_OBJ )
		current_dop = default_x_display(SINGLE_QSP_ARG);

	if( current_dop == NO_DISP_OBJ )
		return(0);

	return( current_dop->do_depth );
}

#endif /* HAVE_X11 */

