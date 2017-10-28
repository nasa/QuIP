#include "quip_config.h"

#include "quip_prot.h"
#include "viewer.h"

#ifdef HAVE_X11

#include "xsupp.h"
#include "xsupp_prot.h"
#include "debug.h"

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* getpid() */
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif

static int button_number, last_button;
static Draggable *carried=NULL;
static Data_Obj *drag_image=NULL;
static XImage *drag_xim;
static XImage *bg_xim;

#define NOTHING_HAPPENED	(-1)

static /* int */ void flush_one_display( Disp_Obj *dop )
{
	XEvent event;
	long mask;

	if( dop == NULL ) return; // return(-1);

	/* This mask should match check_one_display */

	mask =	  ExposureMask
		| ButtonPressMask
		| ButtonReleaseMask
		| ButtonMotionMask
		| PointerMotionMask
		| StructureNotifyMask
		| KeyPressMask
		| KeyReleaseMask
		/* | SubstructureRedirectMask */
		;
		/* | Button1MotionMask */

	/* Does XCheckMaskEvent actually flush the events???
	 */

	XCheckMaskEvent(DO_DISPLAY(dop),mask,&event);

#ifdef FOOBAR
	if( XCheckMaskEvent(DO_DISPLAY(dop),mask,&event) == True ){
		/* we just discard the event */
		return(0);
	} else {
		return(0);
	}
#endif // FOOBAR
}

static void drag_to( int x, int y, Viewer *vp )
{
	assert( vp->vw_dp != NULL );

	/* erase at old location */

	/*
	XPutImage(VW_DPY(vp), vp->vw_xwin, vp->vw_gc, bg_xim,
		x, y, x, y, carried->dg_width, carried->dg_height);
	*/
	XPutImage(VW_DPY(vp), vp->vw_xwin, vp->vw_gc, bg_xim,
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
	XPutImage(VW_DPY(vp), vp->vw_xwin, vp->vw_gc, drag_xim,
		0, 0, x-carried->dg_rx, y-carried->dg_ry,
		OBJ_COLS(drag_image), OBJ_ROWS(drag_image));

	carried->dg_x=x-carried->dg_rx;
	carried->dg_y=y-carried->dg_ry;
}

static void put_down( QSP_ARG_DECL  int x, int y, Viewer *vp )
{
	/* place the object */
	drag_to(x,y,vp);

	/* BUG should factor where the object was picked up, rx ry */

	addTail(vp->vw_draglist,carried->dg_np);
	carried=NULL;

	delvec(drag_image);
	drag_image = NULL;

	drag_xim->data = (char *)NULL;	/* to avoid freeing the data */
	XDestroyImage(drag_xim);

	bg_xim->data = (char *)NULL;	/* to avoid freeing the data */
	XDestroyImage(bg_xim);
}

static void pickup( QSP_ARG_DECL  Draggable *dgp, Viewer *vp )
{
	assert( vp->vw_dp != NULL );

	carried=dgp;

	/* remove it from the list */
	remData(vp->vw_draglist,carried);

	/* re-render background */
	update_image(vp);

	/* make the rendering image */
	assert( drag_image == NULL );

	drag_image = mk_img("drag_image",dgp->dg_height,
		dgp->dg_width,vp->vw_depth/8,PREC_FOR_CODE(PREC_BY));

	drag_xim = XCreateImage(VW_DPY(vp),VW_VISUAL(vp),vp->vw_depth,ZPixmap,0,
		(char *)OBJ_DATA_PTR(drag_image),
		OBJ_COLS(drag_image),OBJ_ROWS(drag_image),8,0);

	bg_xim = XCreateImage(VW_DPY(vp),VW_VISUAL(vp),vp->vw_depth,ZPixmap,0,
		(char *)OBJ_DATA_PTR(vp->vw_dp),
		OBJ_COLS(vp->vw_dp),OBJ_ROWS(vp->vw_dp),8,0);
}

static void do_enter_leave( XEvent *event )
{

	/*
	XCrossingEvent *cross_event = (XCrossingEvent *) event;

	if (cross_event->type == EnterNotify && LocalCmap && !ninstall) 
		XInstallColormap(DO_DISPLAY(curr_dop()),LocalCmap);

	if (cross_event->type == LeaveNotify && LocalCmap && !ninstall) 
		XUninstallColormap(DO_DISPLAY(curr_dop()),LocalCmap);
	*/
fprintf(stderr,"do_enter_leave:  doing nothing\n");
	return;
}

#ifdef FOOBAR
static void show_configure_notify( XEvent *event )
{
	fprintf(stderr,"ConfigureNotify event:\n");
	fprintf(stderr,"\tserial = %ld, send_event = %s\n",
		event->xconfigure.serial,
		event->xconfigure.send_event?"true":"false");
	fprintf(stderr,"\tdisplay = 0x%lx\n",(u_long)event->xconfigure.display);
	fprintf(stderr,"\tWindows:  event = %ld, window = %ld, above = %ld\n",
		event->xconfigure.event, event->xconfigure.window,
		event->xconfigure.above);
	fprintf(stderr,"\tx = %d, y = %d\n",
		event->xconfigure.x, event->xconfigure.y);
	fprintf(stderr,"\twidth = %d, height = %d\n",
		event->xconfigure.width, event->xconfigure.height);
	fprintf(stderr,"\tborder_width = %d\n",
		event->xconfigure.border_width);
	fflush(stderr);
}
#endif // FOOBAR

static int HandleEvent( QSP_ARG_DECL  XEvent *event, int *donep )
{
	int done=0, retval=0;
	Viewer *vp;
	Window win;
	char string[256];
	int x,y;
	Canvas_Event_Code ce_code=CE_INVALID_CODE;

#define KEYBYTES	8
	static char keystr[KEYBYTES]; /* must be global! */
	int ks_len;

	switch (event->type) {
		case Expose:
			win = event->xexpose.window;

			if( (vp=find_viewer(win)) == NULL ){
				NWARN("can't find viewing window for expose event");
				return 0;
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
#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(ERROR_STRING,"redrawing viewer %s after expose event",vp->vw_name);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
				redraw_viewer(vp);
			}
//else advise("ignoring expose event w/ count > 0");

			return 0;

		case ButtonRelease:
		case ButtonPress:
			win = event->xbutton.window;
			if( (vp=find_viewer(win)) == NULL ){
		NWARN("can't find viewing window for button event");
				return 0;
			}
			/*
			 * BUG !? sometimes button 1 returns
			 * a button code of 0x10000001 !?
			 */

			button_number = event->xbutton.button & 3;

			if( event->type == ButtonPress ){
				switch(button_number){
					// Sometimes this happens on the mac???
					case 0:
						sprintf(DEFAULT_ERROR_STRING,
				"Huh?  Button press with button number 0!?");
						advise(DEFAULT_ERROR_STRING);
						// just return for now
						return 0;
						// BUG give a value to ce_code
						break;
					case 1:
						assign_reserved_var("left_button_down","1");
						assign_reserved_var("left_button_up","0");
						ce_code = CE_LEFT_BUTTON_DOWN;
						break;
					case 2:
						assign_reserved_var("middle_button_down","2");
						assign_reserved_var("middle_button_up","0");
						ce_code = CE_MIDDLE_BUTTON_DOWN;
						break;
					case 3:
						assign_reserved_var("right_button_down","4");
						assign_reserved_var("right_button_up","0");
						ce_code = CE_RIGHT_BUTTON_DOWN;
						break;
					default:
						assert( ! "bad button code in HandleEvent" );
						break;
				}
			} else {	// ButtonRelease
				switch(button_number){
					case 0:
						sprintf(DEFAULT_ERROR_STRING,
				"Huh?  Button release with button number 0!?");
						advise(DEFAULT_ERROR_STRING);
						// just return for now
						return 0;
						// BUG give a value to ce_code
						break;
					case 1:
						assign_reserved_var("left_button_down","0");
						assign_reserved_var("left_button_up","1");
						ce_code = CE_LEFT_BUTTON_UP;
						break;
					case 2:
						assign_reserved_var("middle_button_down","0");
						assign_reserved_var("middle_button_up","2");
						ce_code = CE_MIDDLE_BUTTON_UP;
						break;
					case 3:
						assign_reserved_var("right_button_down","0");
						assign_reserved_var("right_button_up","4");
						ce_code = CE_RIGHT_BUTTON_UP;
						break;
					default:
						assert( ! "bad button release code in HandleEvent" );
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
			assign_reserved_var("view_xpos",string);
			sprintf(string,"%d",y);
			assign_reserved_var("view_ypos",string);

			win = event->xbutton.window;
			if( (vp=find_viewer(win)) == NULL ){
				NWARN("can't find viewing window for button event");
				return 0;
			}

			assign_reserved_var("event_window",vp->vw_name);

			// This is where we would like to use the new canvas_event
			// scheme instead of the old button scheme...

			if( IS_BUTTON_ARENA(vp) || IS_MOUSESCAPE(vp) || IS_PLOTTER(vp) ){
				assert( ce_code != CE_INVALID_CODE );	// always true?

				if( VW_EVENT_TBL(vp) != NULL ){
					if( VW_EVENT_ACTION(vp,ce_code) != NULL ){
						chew_text( VW_EVENT_ACTION(vp,ce_code), "(viewer event)" );
					}
					// Do nothing if the event action has not been specified
				} else {
					// Use the old system if no actions have been defined for the specific event...
					// This is mainly here to support backward-compatibility with old scripts.
					// We should print a deprecated warning when the old-style actions are set...
					if( button_number == 1 )
				chew_text(vp->vw_text1,"(button 1 event)");
					else if( button_number == 2 )
				chew_text(vp->vw_text2,"(button 2 event)");
					else if( button_number == 3 )
				chew_text(vp->vw_text3,"(button 3 event)");
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
				}
			} else if( IS_ADJUSTER(vp) ) goto adjust_event;

			else if( IS_DRAGSCAPE(vp) ){
				int t;

				t = event->xbutton.type;
				if( t == ButtonRelease && carried != NULL ){
					x=event->xbutton.x;
					y=event->xbutton.y;
					put_down(QSP_ARG  x,y,vp);
				} else if( t == ButtonPress ){
					Draggable *dgp;

					x=event->xbutton.x;
					y=event->xbutton.y;

					/* see if inside any draggables */

					if( (dgp=in_draggable(vp,x,y))
						!= NULL ){
						pickup(QSP_ARG  dgp,vp);
					}
				}
			}
			break;

		case MapNotify:
			win = event->xmapping.window;
			if( (vp=find_viewer(win)) == NULL ){
				NWARN("can't find viewing window for map event");
				return 0;
			}
			vp->vw_flags |= VIEW_MAPPED;
#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(ERROR_STRING,"redrawing viewer %s after MapNotify event",vp->vw_name);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

			redraw_viewer(vp);
			return 0;

		case MotionNotify:
			win = event->xmotion.window;
			if( (vp=find_viewer(win)) == NULL ){
		NWARN("can't find viewing window for motion event");
				return 0;
			}
			if( !IS_TRACKING(vp) ){
				/* eat up motion events */

				while( XCheckMaskEvent(
					VW_DPY(vp),ButtonMotionMask|PointerMotionMask,event) )
					;
			}

			if( IS_DRAGSCAPE(vp) ){
				if( carried != NULL ){
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
			assign_reserved_var("view_xpos",string);
			sprintf(string,"%d",y);
			assign_reserved_var("view_ypos",string);

			sprintf(string,"%d",button_number);
			assign_reserved_var("button",string);
			sprintf(string,"%d", event->type);
			assign_reserved_var("event_type",string);

			chew_text(vp->vw_text,"(button motion event)");

			last_button = button_number;
			break;

		// These all used to be no-ops...

		case ConfigureNotify:

		/* Some work was attempted here to make this work correctly
		 * with XQuartz/MacOS, but I'm giving up for the moment...
		 * I think that XQuartz may be broken...
		 * jbm, 7/30/2015
		 */

			//NADVISE("ConfigureNotify event!");
			win = event->xconfigure.window;

			if( (vp=find_viewer(win)) == NULL ){
				// Apparently we receive this event after we delete a window
		//NWARN("can't find viewer for configure event");
				return 0;
			}
//show_configure_notify(event);
			if( event->xconfigure.x != VW_X(vp)
					|| event->xconfigure.y != VW_Y(vp) ){
/*
fprintf(stderr,"window %s moved to %d %d...\n",
VW_NAME(vp),event->xconfigure.x,event->xconfigure.y);
*/
				SET_VW_X(vp,event->xconfigure.x);
				SET_VW_Y(vp,event->xconfigure.y);

				// This could be due to a programmatic move,
				// or a user action!?
				if( VW_MOVE_REQUESTED(vp) ){
/*
fprintf(stderr,"Configure event after move request:\n");
fprintf(stderr,"Reported position %d, %d, requested position %d, %d\n",
VW_X(vp),VW_Y(vp),VW_X_REQUESTED(vp),VW_Y_REQUESTED(vp));
*/
					SET_VW_Y_OFFSET(vp,VW_Y(vp)-VW_Y_REQUESTED(vp));
					CLEAR_VW_FLAG_BITS(vp,VW_PROG_MOVE_REQ);
				} else {
//fprintf(stderr,"Window move event without request???\n");
					// update requested by actual - offset?
				}
			}

			break;

		case ReparentNotify:
			// This one occurs when a window is first created...
			//NADVISE("ReparentNotify event!");
			break;

		case CirculateNotify:
			NADVISE("CirculateNotify event!");
			break;

		case DestroyNotify:
			//NADVISE("DestroyNotify event!");
			break;

		case GravityNotify:
			NADVISE("GravityNotify event!");
			break;

		case UnmapNotify:
			// happens when we "unshow"
			//NADVISE("UnmapNotify event!");
			break;

#ifdef FOOBAR
#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(ERROR_STRING,"do-nothing event type %d",event->type);
NWARN(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
#endif // FOOBAR

		/*
		case ReparentNotify: retval=do_reparent(event); break;
		*/
		case EnterNotify:
		case LeaveNotify:
fprintf(stderr,"enter/leave event type %d\n",event->type);
#ifdef QUIP_DEBUG
if( debug ){
sprintf(ERROR_STRING,"enter/leave event type %d",event->type);
NWARN(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			do_enter_leave(event);
			break;
		case 14:
#ifdef QUIP_DEBUG
if( debug ){
sprintf(ERROR_STRING,"mysterious event14 type %d",event->type);
NWARN(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			break;		/* what is this??? */
		case KeyRelease:
//fprintf(stderr,"Keyrelease event\n",ks_len,keystr[0]);
			break;
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
// On Mac, we sometimes get spurious keypress events that are
// never cleared!?
// Maybe faulty auto-repeat?  We could check KeyRelease?
//
//fprintf(stderr,"KeyPress event, ks_len = %d, keystr[0] = 0x%x\n",ks_len,keystr[0]);
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
#ifdef QUIP_DEBUG
if( debug ){
sprintf(ERROR_STRING,"uncaught event type %d",event->type);
NWARN(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
sprintf(ERROR_STRING,"uncaught event type %d",event->type);
NWARN(ERROR_STRING);
			break;		/* ignore unexpected events */
	}
	*donep = done;
	return(retval);
} // HandleEvent

static int check_one_display( QSP_ARG_DECL  Disp_Obj *dop )
{
	XEvent event;
	long mask;
	int retval,done;

	if( dop == NULL ) return(-1);

	/* this is not really a loop yet! */

	mask =	  ExposureMask
		| ButtonPressMask
		| ButtonReleaseMask
		| ButtonMotionMask
		| PointerMotionMask
		| KeyPressMask
		| KeyReleaseMask
		// what about when the user moves or resizes a window?
		| StructureNotifyMask
		// | SubstructureRedirectMask
		;
		/* | Button1MotionMask */

	/* BUG we need to be sure that HandleEvent actually handles all masked
	 * events, otherwise we could get into a bad loop!
	 */

	if( XCheckMaskEvent(DO_DISPLAY(dop),mask,&event) == True ){
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
	if( lp == NULL ) return(-1);

	np=QLIST_HEAD(lp);
	while( np != NULL ){
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

	lp = displays_list(SINGLE_QSP_ARG);
	if( lp == NULL ) return;

	np=QLIST_HEAD(lp);
	while( np != NULL ){
		dop = (Disp_Obj *)np->n_data;
		flush_one_display(dop);
		np = np->n_next;
	}
}

Viewer *_find_viewer( QSP_ARG_DECL  Window win )
{
	Node *np;
	Viewer *vp;

	np=first_viewer_node();
	while( np != NULL ){
		vp=(Viewer *) np->n_data;
		if( vp->vw_xwin == win ) return(vp);
		np=np->n_next;
	}
	return(NULL);
}


Draggable *in_draggable( Viewer *vp, int x,int y )
{
	Node *np;
	Draggable *dgp;
	int rx,ry;
	WORD_TYPE *base;
	int words_per_row, wordno, bit;

	np=QLIST_HEAD(vp->vw_draglist);
	while(np!=NULL){
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
			base=(WORD_TYPE *)OBJ_DATA_PTR(dgp->dg_bitmap);
			if( base[wordno] & bit ){
				dgp->dg_rx=rx;
				dgp->dg_ry=ry;
				return(dgp);
			}
		}

		np=np->n_next;
	}
	return(NULL);
}

#ifdef THREAD_SAFE_QUERY
#define INIT_EVENT_QSP			qsp = curr_qsp;
#else
#define INIT_EVENT_QSP
#endif

/* i_loop()  --  process events until there are none */

void i_loop(SINGLE_QSP_ARG_DECL)
{
	do {
		usleep(5);
	} while( event_loop(SINGLE_QSP_ARG) != -1 );
}

/* Why is this in event.c?? */

void extract_image( Data_Obj *dpto, Data_Obj *dpfr, int x, int y )
{
	/* assumes dpto is smaller */

	u_char *to, *fr;
	incr_t i,j;

	to=(u_char *)OBJ_DATA_PTR(dpto);
	fr=(u_char *)OBJ_DATA_PTR(dpfr);

	for(j=0;j<OBJ_ROWS(dpto);j++){
		if( j+y < 0 || j+y >= OBJ_ROWS(dpfr) ) continue;
		for(i=0;i<OBJ_COLS(dpto);i++){
			if( i+x < 0 || i+x >= OBJ_COLS(dpfr) ) continue;

			*(to+j*OBJ_ROW_INC(dpto)+i*OBJ_PXL_INC(dpto))
			  = *(fr+(j+y)*OBJ_ROW_INC(dpfr)+(i+x)*OBJ_PXL_INC(dpfr));
		}
	}
}

#else /* ! HAVE_X11 */

int event_loop(SINGLE_QSP_ARG_DECL)
{
	UNIMP_MSG(event_loop)
	return -1;
}

#endif /* ! HAVE_X11 */

