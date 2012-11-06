#include "quip_config.h"

char VersionId_xsupp_event[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* getpid() */
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif

#include "viewer.h"
#include "debug.h"
#include "xsupp.h"
#include "chewtext.h"	/* CHEW_TEXT */
#include "query.h"

static int button_number, last_button;
static Draggable *carried=NO_DRAGG;
static Data_Obj *drag_image=NO_OBJ;
static XImage *drag_xim;
static XImage *bg_xim;

#define NOTHING_HAPPENED	(-1)

/* local prototypes */
static int HandleEvent(QSP_ARG_DECL  XEvent *event, int *donep);

int flush_one_display( Disp_Obj *dop )
{
	XEvent event;
	long mask;

	if( dop == NO_DISP_OBJ ) return(-1);

	/* This mask should match check_one_display */

	mask =	  ExposureMask
		| ButtonPressMask
		| ButtonReleaseMask
		| ButtonMotionMask
		| PointerMotionMask
		/* | StructureNotifyMask */
		| KeyPressMask
		;
		/* | Button1MotionMask */

	if( XCheckMaskEvent(dop->do_dpy,mask,&event) == True ){
		/* we just discard the event */
		return(0);
	} else {
		return(0);
	}
}

int check_one_display( QSP_ARG_DECL  Disp_Obj *dop )
{
	XEvent event;
	long mask;
	int retval,done;

	if( dop == NO_DISP_OBJ ) return(-1);

	/* this is not really a loop yet! */

	mask =	  ExposureMask
		| ButtonPressMask
		| ButtonReleaseMask
		| ButtonMotionMask
		| PointerMotionMask
		/* | StructureNotifyMask */
		| KeyPressMask
		;
		/* | Button1MotionMask */

	/* BUG we need to be sure that HandleEvent actually handles all masked
	 * events, otherwise we could get into a bad loop!
	 */

	if( XCheckMaskEvent(dop->do_dpy,mask,&event) == True ){
		retval = HandleEvent(QSP_ARG  &event,&done);
	} else {
		return(NOTHING_HAPPENED);
	}
		
	return(retval);
}

/* Check all displays, return after an event is detected... */

int event_loop(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;
	Disp_Obj *dop;
	int stat;

	lp = displays_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST ) return(-1);

	np=lp->l_head;
	while( np != NO_NODE ){
		dop = (Disp_Obj *)np->n_data;
		stat=check_one_display(QSP_ARG  dop);
		if( stat != NOTHING_HAPPENED ) return(stat);
		np = np->n_next;
	}
	return(NOTHING_HAPPENED);
}

/* Check all displays, flushing events */
void discard_events(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;
	Disp_Obj *dop;
	int stat;

	lp = displays_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST ) return;

	np=lp->l_head;
	while( np != NO_NODE ){
		dop = (Disp_Obj *)np->n_data;
		stat=flush_one_display(dop);
		/* if( stat != NOTHING_HAPPENED ) return(stat); */
		np = np->n_next;
	}
}


void do_enter_leave( XEvent *event )
{

	/*
	XCrossingEvent *cross_event = (XCrossingEvent *) event;

	if (cross_event->type == EnterNotify && LocalCmap && !ninstall) 
		XInstallColormap(curr_dop()->do_dpy,LocalCmap);

	if (cross_event->type == LeaveNotify && LocalCmap && !ninstall) 
		XUninstallColormap(curr_dop()->do_dpy,LocalCmap);
	*/
	return;
}

Viewer *find_viewer( QSP_ARG_DECL  Window win )
{
	Node *np;
	Viewer *vp;

	np=first_viewer_node(SINGLE_QSP_ARG);
	while( np != NO_NODE ){
		vp=(Viewer *) np->n_data;
		if( vp->vw_xwin == win ) return(vp);
		np=np->n_next;
	}
	return(NO_VIEWER);
}


Draggable *in_draggable( Viewer *vp, int x,int y )
{
	Node *np;
	Draggable *dgp;
	int rx,ry;
	WORD_TYPE *base;
	int words_per_row, wordno, bit;

	np=vp->vw_draglist->l_head;
	while(np!=NO_NODE){
		dgp=(Draggable *)np->n_data;
		/* compute coords rel. the draggable object */
		rx=x-dgp->dg_x;
		ry=y-dgp->dg_y;
		/* now see if inside bitmap */

		if( rx >= 0 && rx < dgp->dg_width &&
		    ry >= 0 && ry < dgp->dg_height ){

			words_per_row = (dgp->dg_width+WORDLEN-1)/WORDLEN;
			wordno = ry*words_per_row + rx/WORDLEN;
			bit = 1<<(rx%WORDLEN);
			base=(WORD_TYPE *)dgp->dg_bitmap->dt_data;
			if( base[wordno] & bit ){
				dgp->dg_rx=rx;
				dgp->dg_ry=ry;
				return(dgp);
			}
		}

		np=np->n_next;
	}
	return(NO_DRAGG);
}

#ifdef THREAD_SAFE_QUERY
#define INIT_EVENT_QSP			qsp = curr_qsp;
#else
#define INIT_EVENT_QSP
#endif

static int HandleEvent( QSP_ARG_DECL  XEvent *event, int *donep )
{
	int done=0, retval=0;
	Viewer *vp;
	Window win;
	char string[256];
	unsigned int x,y;

#define KEYBYTES	8
	static char keystr[KEYBYTES]; /* must be global! */
	int ks_len;
#ifdef FOOBAR
	QSP_DECL

	INIT_EVENT_QSP
#endif /* FOOBAR */

	switch (event->type) {
		case Expose:
			win = event->xexpose.window;

			if( (vp=find_viewer(QSP_ARG  win)) == NO_VIEWER ){
				NWARN("can't find viewing window for expose event");
				return(0);
			}
			vp->vw_flags |= VIEW_EXPOSED;
			/*
			 * When a viewer is brought to front, an expose event is generated
			 * by EACH window that previously overlapped it; thus, on a
			 * crowded screen, this could be called many times.  It
			 * therefore makes sense to NOT redraw for events that happen
			 * shortly after the last one...
			 *
			 * We used to check the time here, but now that is done
			 * within redraw_viewer().
			 *
			 * Well, of course the authors of XLib had thought about this, so they
			 * put more info in the event struct.  The count member tells how many
			 * more events follow in the queue, so it is safe to ignore events with
			 * non zero counts if we are going to go ahead and redraw the whole 
			 * window anyway.  (The events indicate rectangular subregions which 
			 * are exposed and need redrawing...)
			 */

			if( ((XExposeEvent *)event)->count == 0 ){
				/* We see this message printed even when
				 * a window is hidden!?
				 * Does the event signify ANY change in
				 * the exposure status?
				 */
#ifdef DEBUG
if( debug & xdebug ){
sprintf(ERROR_STRING,"redrawing viewer %s after expose event",vp->vw_name);
advise(ERROR_STRING);
}
#endif /* DEBUG */
				redraw_viewer(QSP_ARG  vp);
			}
//else advise("ignoring expose event w/ count > 0");

			return(0);

		case ButtonRelease:
		case ButtonPress:
			win = event->xbutton.window;
			if( (vp=find_viewer(QSP_ARG  win)) == NO_VIEWER ){
		NWARN("can't find viewing window for button event");
				return(0);
			}
			/*
			 * BUG !? sometimes button 1 returns
			 * a button code of 0x10000001 !?
			 */

			button_number = event->xbutton.button & 3;

			if( event->type == ButtonPress ){
				switch(button_number){
					case 1:
						ASSIGN_VAR("left_button_down","1");
						ASSIGN_VAR("left_button_up","0");
						break;
					case 2:
						ASSIGN_VAR("middle_button_down","1");
						ASSIGN_VAR("middle_button_up","0");
						break;
					case 3:
						ASSIGN_VAR("right_button_down","1");
						ASSIGN_VAR("right_button_up","0");
						break;
				}
			} else {
				switch(button_number){
					case 1:
						ASSIGN_VAR("left_button_down","0");
						ASSIGN_VAR("left_button_up","1");
						break;
					case 2:
						ASSIGN_VAR("middle_button_down","0");
						ASSIGN_VAR("middle_button_up","1");
						break;
					case 3:
						ASSIGN_VAR("right_button_down","0");
						ASSIGN_VAR("right_button_up","1");
						break;
				}
			}

			x=event->xbutton.x;
			y=event->xbutton.y;

			if( x < 0 ) x=0;
			else if( x >= vp->vw_width ) x=vp->vw_width-1;

			if( y < 0 ) {
if( verbose ){
sprintf(ERROR_STRING,"HandleEvent 2:  clipping y value %d from below at 0",y);
advise(ERROR_STRING);
}
				y=0;
			} else if( y >= vp->vw_height ){
if( verbose ){
sprintf(ERROR_STRING,"HandleEvent 2:  clipping y value %d from above at %d",y,vp->vw_height-1);
advise(ERROR_STRING);
}
				y=vp->vw_height-1;
			}
			sprintf(string,"%d",x);
			ASSIGN_VAR("view_xpos",string);
			sprintf(string,"%d",y);
			ASSIGN_VAR("view_ypos",string);

			win = event->xbutton.window;
			if( (vp=find_viewer(QSP_ARG  win)) == NO_VIEWER ){
				NWARN("can't find viewing window for button event");
				return(0);
			}

			ASSIGN_VAR("event_window",vp->vw_name);

			if( IS_BUTTON_ARENA(vp) || IS_MOUSESCAPE(vp) ){
				if( button_number == 1 )
					CHEW_TEXT(vp->vw_text1);
				else if( button_number == 2 )
					CHEW_TEXT(vp->vw_text2);
				else if( button_number == 3 )
					CHEW_TEXT(vp->vw_text3);
				else if( button_number == 0 ){
					/* this sometimes happens on the new laptop
					 * when using the touchpad... ???
					 */
					if( verbose ){
						sprintf(ERROR_STRING,"Ignoring event w/ button #0");
						advise(ERROR_STRING);
					}
				} else {
					sprintf(ERROR_STRING,
					"wacky button number %d",button_number);
					NWARN(ERROR_STRING);
				}
			} else if( IS_ADJUSTER(vp) ) goto adjust_event;

			else if( IS_DRAGSCAPE(vp) ){
				int t;

				t = event->xbutton.type;
				if( t == ButtonRelease && carried != NO_DRAGG ){
					x=event->xbutton.x;
					y=event->xbutton.y;
					put_down(QSP_ARG  x,y,vp);
				} else if( t == ButtonPress ){
					Draggable *dgp;

					x=event->xbutton.x;
					y=event->xbutton.y;

					/* see if inside any draggables */

					if( (dgp=in_draggable(vp,x,y))
						!= NO_DRAGG ){
						pickup(QSP_ARG  dgp,vp);
					}
				}
			}
			break;

		case MapNotify:
			win = event->xmapping.window;
			if( (vp=find_viewer(QSP_ARG  win)) == NO_VIEWER ){
				NWARN("can't find viewing window for map event");
				return(0);
			}
			vp->vw_flags |= VIEW_MAPPED;
#ifdef DEBUG
if( debug & xdebug ){
sprintf(ERROR_STRING,"redrawing viewer %s after MapNotify event",vp->vw_name);
advise(ERROR_STRING);
}
#endif /* DEBUG */

			redraw_viewer(QSP_ARG  vp);
			return(0);

		case MotionNotify:
			win = event->xmotion.window;
			if( (vp=find_viewer(QSP_ARG  win)) == NO_VIEWER ){
		NWARN("can't find viewing window for motion event");
				return(0);
			}
			if( !IS_TRACKING(vp) ){
				/* eat up motion events */

				while( XCheckMaskEvent(
					vp->vw_dop->do_dpy,ButtonMotionMask|PointerMotionMask,event) )
					;
			}

			if( IS_DRAGSCAPE(vp) ){
				if( carried != NO_DRAGG ){
					x = event->xmotion.x;
					y = event->xmotion.y;
					drag_to(x,y,vp);
				}
				break;
			}

			/* Possible BUG?
			 * what if we have two adjusters active at once...
			 */

			button_number=last_button;

adjust_event:
			x = event->xmotion.x;
			y = event->xmotion.y;

			if( x < 0 ) x=0;
			else if( x >= vp->vw_width ) x=vp->vw_width-1;

			/* WHY do we ever get events that are
			 * outside the window???
			 */
			if( y < 0 ) {
if( verbose ){
sprintf(ERROR_STRING,"HandleEvent:  clipping y value %d from below at 0",y);
advise(ERROR_STRING);
}
				y=0;
			} else if( y >= vp->vw_height ){
if( verbose ){
sprintf(ERROR_STRING,"HandleEvent:  clipping y value %d from above at %d",y,vp->vw_height-1);
advise(ERROR_STRING);
}
				y=vp->vw_height-1;
			}

			sprintf(string,"%d",x);
			ASSIGN_VAR("view_xpos",string);
			sprintf(string,"%d",y);
			ASSIGN_VAR("view_ypos",string);

			sprintf(string,"%d",button_number);
			ASSIGN_VAR("button",string);
			sprintf(string,"%d", event->type);
			ASSIGN_VAR("event_type",string);

			CHEW_TEXT(vp->vw_text);		/* used for motion... */

			last_button = button_number;
			break;

		case ConfigureNotify:
		case ReparentNotify:
		case CirculateNotify:
		case DestroyNotify:
		case GravityNotify:
		case UnmapNotify:
#ifdef DEBUG
if( debug & xdebug ){
sprintf(ERROR_STRING,"do-nothing event type %d",event->type);
NWARN(ERROR_STRING);
}
#endif /* DEBUG */
			break;

		/*
		case ReparentNotify: retval=do_reparent(event); break;
		*/
		case EnterNotify:
		case LeaveNotify:
#ifdef DEBUG
if( debug ){
sprintf(ERROR_STRING,"enter/leave event type %d",event->type);
NWARN(ERROR_STRING);
}
#endif /* DEBUG */
			do_enter_leave(event);
			break;
		case 14:
#ifdef DEBUG
if( debug ){
sprintf(ERROR_STRING,"mysterious event14 type %d",event->type);
NWARN(ERROR_STRING);
}
#endif /* DEBUG */
			break;		/* what is this??? */
		case KeyPress:
			ks_len = XLookupString((XKeyEvent *)event,
				keystr,KEYBYTES,NULL,NULL);
			if( ks_len > 1 ){
				sprintf(ERROR_STRING,"keystring has length %d",
					ks_len);
				NWARN(ERROR_STRING);
			} else if( ks_len == 0 ){
				/* usually because shift or control was depressed */
			} else {
				if( keystr[0] == 015 )
					keystr[0]=012;	/* map CR to LF */
				keystr[1]=0;
				if( keystr[0] == 03 ){	/* ^C */
					/* BUG should get intr char from stty */
					/* This allows us to kill the program when the focus
					 * is in the viewer, but only if we check events...
					 */
					int pid;
					pid=getpid();
					kill(pid,SIGINT);
				} else {
					simulate_typing(keystr);
				}
			}
			break;

		default: 
#ifdef DEBUG
if( debug ){
sprintf(ERROR_STRING,"uncaught event type %d",event->type);
NWARN(ERROR_STRING);
}
#endif /* DEBUG */
sprintf(ERROR_STRING,"uncaught event type %d",event->type);
NWARN(ERROR_STRING);
			break;		/* ignore unexpected events */
	}
	*donep = done;
	return(retval);
}

/* i_loop()  --  process events until there are none */

void i_loop(SINGLE_QSP_ARG_DECL)
{
	do {
		usleep(5);
	} while( event_loop(SINGLE_QSP_ARG) != -1 );
}

void pickup( QSP_ARG_DECL  Draggable *dgp, Viewer *vp )
{
#ifdef CAUTIOUS
	if( vp->vw_dp == NO_OBJ ){
		sprintf(ERROR_STRING,
			"CAUTIOUS:  pickup:  null data object for viewer %s",
			vp->vw_name);
		NWARN(ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */

	carried=dgp;

	/* remove it from the list */
	remData(vp->vw_draglist,carried);

	/* re-render background */
	update_image(vp);

	/* make the rendering image */
#ifdef CAUTIOUS
	if( drag_image != NO_OBJ )
		ERROR1("extra drag image!?");
#endif /* CAUTIOUS */

	drag_image = mk_img(QSP_ARG  "drag_image",dgp->dg_height,
		dgp->dg_width,vp->vw_depth/8,PREC_BY);

	drag_xim = XCreateImage(vp->vw_dpy,vp->vw_visual,vp->vw_depth,ZPixmap,0,
		(char *)drag_image->dt_data,
		drag_image->dt_cols,drag_image->dt_rows,8,0);

	bg_xim = XCreateImage(vp->vw_dpy,vp->vw_visual,vp->vw_depth,ZPixmap,0,
		(char *)vp->vw_dp->dt_data,
		vp->vw_dp->dt_cols,vp->vw_dp->dt_rows,8,0);
}

/* Why is this in event.c?? */

void extract_image( Data_Obj *dpto, Data_Obj *dpfr, int x, int y )
{
	/* assumes dpto is smaller */

	u_char *to, *fr;
	dimension_t i,j;

	to=(u_char *)dpto->dt_data;
	fr=(u_char *)dpfr->dt_data;

	for(j=0;j<dpto->dt_rows;j++){
		if( j+y < 0 || j+y >= dpfr->dt_rows ) continue;
		for(i=0;i<dpto->dt_cols;i++){
			if( i+x < 0 || i+x >= dpfr->dt_cols ) continue;

			*(to+j*dpto->dt_rowinc+i*dpto->dt_pinc)
			  = *(fr+(j+y)*dpfr->dt_rowinc+(i+x)*dpfr->dt_pinc);
		}
	}
}

void drag_to( int x, int y, Viewer *vp )
{
#ifdef CAUTIOUS
	if( vp->vw_dp == NO_OBJ ){
		sprintf(DEFAULT_ERROR_STRING,
			"CAUTIOUS:  drag_to:  null data object for viewer %s",
			vp->vw_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */
	/* erase at old location */

	/*
	XPutImage(vp->vw_dpy, vp->vw_xwin, vp->vw_gc, bg_xim,
		x, y, x, y, carried->dg_width, carried->dg_height);
	*/
	XPutImage(vp->vw_dpy, vp->vw_xwin, vp->vw_gc, bg_xim,
		0, 0, 0, 0, vp->vw_width, vp->vw_height);

	carried->dg_x=0;
	carried->dg_y=0;

	/* copy new background into drag image */

	/* BUG the new background may containg a piece of the dragged
	 * object, not drawing it in can make the dragged object flicker
	 */
	extract_image(drag_image,vp->vw_dp,x-carried->dg_rx,y-carried->dg_ry);

	/* paint icon over new background */
	embed_draggable(drag_image,carried);

	/* redraw thingy at new location */
	XPutImage(vp->vw_dpy, vp->vw_xwin, vp->vw_gc, drag_xim,
		0, 0, x-carried->dg_rx, y-carried->dg_ry,
		drag_image->dt_cols, drag_image->dt_rows);

	carried->dg_x=x-carried->dg_rx;
	carried->dg_y=y-carried->dg_ry;
}

void put_down( QSP_ARG_DECL  int x, int y, Viewer *vp )
{
	/* place the object */
	drag_to(x,y,vp);

	/* BUG should factor where the object was picked up, rx ry */

	addTail(vp->vw_draglist,carried->dg_np);
	carried=NO_DRAGG;

	delvec(QSP_ARG  drag_image);
	drag_image = NO_OBJ;

	drag_xim->data = (char *)NULL;	/* to avoid freeing the data */
	XDestroyImage(drag_xim);

	bg_xim->data = (char *)NULL;	/* to avoid freeing the data */
	XDestroyImage(bg_xim);
}

#endif /* HAVE_X11 */

