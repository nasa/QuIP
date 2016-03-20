#include "quip_config.h"

#include "quip_prot.h"
#include "viewer.h"
#include "xsupp.h"

static Disp_Obj *current_dop=NO_DISP_OBJ;

ITEM_INTERFACE_DECLARATIONS(Disp_Obj,disp_obj)

#ifdef HAVE_X11

#include "xsupp_prot.h"

/* manipulate displays */

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* atoi() */
#endif

static int window_sys_inited=0;
static XVisualInfo *	visualList = NULL;

#if defined(__cplusplus) || defined(c_plusplus)
#define xvi_class	c_class
#else
#define xvi_class	class
#endif


void set_display( Disp_Obj *dop )
{
	current_dop = dop;
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
		del_disp_obj(QSP_ARG  dop);
		rls_str((char *)dop->do_name);
		return(-1);
	}
	return(0);
}

static XVisualInfo *get_vis_list( Disp_Obj * dop, int *np )
{
	XVisualInfo		vTemplate;

	vTemplate.screen=dop->do_screen;

	if( visualList != NULL ) XFree(visualList);
	visualList = XGetVisualInfo(dop->do_dpy,VisualScreenMask,
		&vTemplate,np);

#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(DEFAULT_ERROR_STRING,"%d visuals found",*np);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	return(visualList);
}

static XVisualInfo *get_depth_list( Disp_Obj * dop, int depth, int *np )
{
	XVisualInfo		vTemplate;

	/* try to get an 8 bit psuedocolor visual */
	/* taken from Xlib prog manual p. 215 */

	vTemplate.depth=depth;
	vTemplate.screen=dop->do_screen;

	if( visualList != NULL ) XFree(visualList);
	visualList = XGetVisualInfo(dop->do_dpy,VisualScreenMask|VisualDepthMask,
		&vTemplate,np);
	if( visualList == NULL ){
		sprintf(DEFAULT_ERROR_STRING,
			"get_depth_list(%d) got NULL from XGetVisualInfo!?",depth);
		NWARN(DEFAULT_ERROR_STRING);
	}
#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(DEFAULT_ERROR_STRING,"%d visuals found with depth %d",*np,depth);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

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

static Visual *GetEightBitVisual( Disp_Obj * dop)
{
	int			visualsMatched=0;
	Visual *vis;
	XVisualInfo *vip;
	int i;

	vip = get_depth_list(dop,8,&visualsMatched);

	if( visualsMatched == 0 ) return(NULL);

	i=find_visual(vip,visualsMatched,PseudoColor,8);
	if( i < 0 ){
		NWARN("no pseudocolor visual found!?");
		return(vip[0].visual);
	}
#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(DEFAULT_ERROR_STRING,"using visual %ld",vip[i].visualid);
NADVISE(DEFAULT_ERROR_STRING);
vis=DefaultVisual(dop->do_dpy,dop->do_screen);
sprintf(DEFAULT_ERROR_STRING,"default visual is %ld",vis->visualid);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	return(vip[i].visual);
}

#define PREFERRED_MODE TrueColor
#define ALTERNATE_MODE DirectColor
#define PREFERRED_NAME "TrueColor"
#define ALTERNATE_NAME "DirectColor"


static Visual *GetSpecifiedVisual( Disp_Obj * dop, int depth )
{
	XVisualInfo *	vip;
	int		visualsMatched;
	int i;
	const char *	name;

	vip = get_depth_list(dop,depth,&visualsMatched);
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
NADVISE(DEFAULT_ERROR_STRING);
			preferred_id = atoi(s);	/* BUG decimal only, should parse hex too */
			i=find_visual_by_id(vip,visualsMatched,preferred_id);
			if( i >= 0 ){
sprintf(DEFAULT_ERROR_STRING,"preferred visual id %d FOUND at index %d",preferred_id,i);
NADVISE(DEFAULT_ERROR_STRING);
				return(vip[i].visual);
			}
			sprintf(DEFAULT_ERROR_STRING,"Unable to find requested visual id %d",preferred_id);
			NWARN(DEFAULT_ERROR_STRING);
		}
	}
	i=find_visual(vip,visualsMatched,PREFERRED_MODE,depth);
	if( i < 0 ){
		sprintf(DEFAULT_ERROR_STRING,"no %s visual found with depth %d!?",
			PREFERRED_NAME,depth);
		NWARN(DEFAULT_ERROR_STRING);
		i=find_visual(vip,visualsMatched,ALTERNATE_MODE,depth);
		if( i < 0 ){
			sprintf(DEFAULT_ERROR_STRING,"no %s visual found with depth %d!?",
				ALTERNATE_NAME,depth);
			NWARN(DEFAULT_ERROR_STRING);
			return(vip[0].visual);
		} else {
			name = ALTERNATE_NAME;
		}
	} else {
		name = PREFERRED_NAME;
	}

if( verbose ){
sprintf(DEFAULT_ERROR_STRING,"i=%d, using visual %ld (%s, depth = %d)",
i, vip[i].visualid,name,depth);
NADVISE(DEFAULT_ERROR_STRING);
}
	return(vip[i].visual);
}

static Visual *Get16BitVisual( Disp_Obj * dop)
{
	return( GetSpecifiedVisual(dop,16) );
}

static Visual *Get24BitVisual( Disp_Obj * dop)
{
	return( GetSpecifiedVisual(dop,24) );
}

/* powerbook display */

static Visual *Get15BitVisual( Disp_Obj * dop)
{
	return( GetSpecifiedVisual(dop,15) );
}


static int dop_setup( QSP_ARG_DECL   Disp_Obj *dop, int desired_depth)
{
	XVisualInfo vinfo, *list;
	int n;

	dop->do_screen 	= DefaultScreen(dop->do_dpy);
	dop->do_rootw	= RootWindow(dop->do_dpy,dop->do_screen);
	dop->do_currw	= RootWindow(dop->do_dpy,dop->do_screen);

#ifdef HAVE_OPENGL
	dop->do_ctx = NULL;
#endif /* HAVE_OPENGL */

#ifdef QUIP_DEBUG
if( debug & xdebug ){
XWindowAttributes wa;
XGetWindowAttributes(dop->do_dpy,dop->do_rootw,&wa);
sprintf(DEFAULT_ERROR_STRING,"depth of root window = %d", wa.depth);
prt_msg(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	dop->do_gc	= DefaultGC(dop->do_dpy,dop->do_screen);


	if( verbose ){
sprintf(DEFAULT_ERROR_STRING,"desired depth is %d",desired_depth);
NADVISE(DEFAULT_ERROR_STRING);
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
			NADVISE("initial attempt to get a visual failed");
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
			XVisualInfo *	vlp;

			vlp=get_vis_list(dop,&nvis);

			sprintf(DEFAULT_ERROR_STRING,"%d visuals found:",nvis);
			NADVISE(DEFAULT_ERROR_STRING);

			for(i=0;i<nvis;i++){
				sprintf(DEFAULT_ERROR_STRING,"class %d   depth %d",
			vlp[i].xvi_class,vlp[i].depth);
				NADVISE(DEFAULT_ERROR_STRING);
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

#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(msg_str,"display %s, %d by %d, %d bits deep",
dop->do_name,dop->do_width,dop->do_height,dop->do_depth);
prt_msg(msg_str);
}
#endif /* QUIP_DEBUG */

	return(0);
}

static double get_dpy_size( QSP_ARG_DECL  Item *ip, int index)
{
	Disp_Obj *dop;

	dop = (Disp_Obj *)ip;

	switch( index ){
		case 0: return( (double) dop->do_depth );
		case 1: return( (double) dop->do_width );
		case 2: return( (double) dop->do_height );
		default:
			sprintf(ERROR_STRING,
	"get_dpy_size:  unsupported display size function (index = %d)",index);
			WARN(ERROR_STRING);
			return(0.0);
	}
}

static Size_Functions dpy_sf={
	get_dpy_size,
	default_prec_name
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

	if( dop_setup(QSP_ARG  dop,desired_depth) < 0 ){
		/* Bug - XCloseDisplay?? */
		/* need to destroy object here */
		del_disp_obj(QSP_ARG  dop);
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

#define MAX_DISPLAY_DEPTHS	4
static int possible_depths[MAX_DISPLAY_DEPTHS]={24,8,16,15};

static Disp_Obj * default_x_display(SINGLE_QSP_ARG_DECL)
{
	const char *dname;
	Disp_Obj *dop;
	int which_depth;
	char *s;

	dname = check_display(SINGLE_QSP_ARG);

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

void window_sys_init(SINGLE_QSP_ARG_DECL)
{
	char s[8];
	Variable *vp;

	if( window_sys_inited ) return;

#ifdef QUIP_DEBUG
	xdebug = add_debug_module(QSP_ARG  "xsupp");
#endif /* QUIP_DEBUG */

	add_event_func(QSP_ARG  i_loop);
	set_discard_func( discard_events );

	window_sys_inited=1;

	if( current_dop == NO_DISP_OBJ ){
		current_dop = default_x_display(SINGLE_QSP_ARG);
		if( current_dop == NO_DISP_OBJ ){
			NWARN("Couldn't open default display!?");
			return;
		}
	}
	// Make sure DISPLAY_WIDTH and DISPLAY_HEIGHT are set...
	// If these have been set in the environment, leave be.
	vp = var_of(QSP_ARG  "DISPLAY_WIDTH");
	if( vp == NULL ){
		sprintf(s,"%d",current_dop->do_width);
		ASSIGN_RESERVED_VAR("DISPLAY_WIDTH",s);
	}
	vp = var_of(QSP_ARG  "DISPLAY_HEIGHT");
	if( vp == NULL ){
		sprintf(s,"%d",current_dop->do_height);
		ASSIGN_RESERVED_VAR("DISPLAY_HEIGHT",s);
	}

	//window_sys_inited=1;
}

#define DEFAULT_DISPLAY_DEPTH	24
#define ALTERNATE_DISPLAY_DEPTH	8

int display_depth(SINGLE_QSP_ARG_DECL)
{
	if( current_dop == NO_DISP_OBJ )
		current_dop = default_x_display(SINGLE_QSP_ARG);

	if( current_dop == NO_DISP_OBJ )
		return(0);

	return( current_dop->do_depth );
}

#else /* !HAVE_X11 */

/* dummy functions to allow linking w/o X11 */

int display_depth(SINGLE_QSP_ARG_DECL)
{
	UNIMP_MSG(display_depth)
	return 0;
}

void window_sys_init(SINGLE_QSP_ARG_DECL)
{ UNIMP_MSG(window_sys_init) }

#endif /* !HAVE_X11 */

Disp_Obj *curr_dop(void)
{ return(current_dop); }


