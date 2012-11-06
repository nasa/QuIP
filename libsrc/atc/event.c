#include "quip_config.h"

#ifdef HAVE_X11

char VersionId_atc_event[] = QUIP_VERSION_STRING;

#ifdef FTIME
#include <sys/timeb.h>		/* ftime() */
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#include <math.h>		/* fabs() */
#include <string.h>		/* strcpy() */
#include "viewer.h"
//#include "myerror.h"
#include "debug.h"
#include "xsupp.h"
//#include "vars.h"
#include "chewtext.h"		/* chew_text() */

#include "conflict.h"
#include "draw.h"

#define FLIP_VAR(var)	var = FLIP_X(var)

/* globals */
Flight_Path *selection_1, *selection_2, *selected_tag;
Atc_State atc_state;
int trial_aborted=0;
int allow_tag_movement=1;
int cursor_controls_visibility=0;
int timeout_seconds=(-1);


static struct timeval trial_start_tv;
static int event_overflow_warning_given;

/* local prototypes */
static void discard_pending_events(void);
static void fix_event_log_times(void);
static void process_atc_event(QSP_ARG_DECL  XEvent *event);
static int listed_position(QSP_ARG_DECL  Point *);

/* want 500 ms, but the delay function is badly miscalibrated... */
#define DEFAULT_FIXATION_DELAY	20
int calibration_fixation_duration=DEFAULT_FIXATION_DELAY;

static int exit_on_right_click=0;

/*
 *
 * Tag motion mode:
 * States:
 *	SELECTING
 *		button down:  if over a tag, select it, change state to DRAGGING
 *	DRAGGING
 *		movement:  erase and redraw tag
 *		button up:  deselect, change state to SELECTING
 */

/* static int button_number; */



/* Event logging functions */

typedef struct logged_event {
#ifdef FTIME
	time_t		le_time;
	unsigned short	le_millitm;
#else
	struct timeval	le_tv;
#endif /* FTIME */

	unsigned short	le_event_class;
#define NULL_EVENT	0
#define X_EVENT		1
#define ATC_EVENT	2
	union {
		XEvent		u_x_event;
		Atc_Event	u_atc_event;
	} le_u;
#define le_x_event	le_u.u_x_event
#define le_atc_event	le_u.u_atc_event
} Logged_Event;

#define NO_LOGGED_EVENT	((Logged_Event *)NULL)

static Logged_Event *log_buf=NO_LOGGED_EVENT;
static unsigned long n_logged_events=0, max_logged_events;
static int logging_events=0;

int get_event_key( XEvent *event )
{
#define KEYBYTES	8
	char keystr[KEYBYTES];
	int ks_len;

	ks_len = XLookupString((XKeyEvent *)event,
		keystr,KEYBYTES,NULL,NULL);
	if( ks_len > 1 ){
		sprintf(DEFAULT_ERROR_STRING,"get_event_key:  keystring has length %d",
			ks_len);
		NWARN(DEFAULT_ERROR_STRING);
	} else if( ks_len == 0 ){
		/* usually because shift or control was depressed */
		return(-1);
	}

	return( keystr[0] );
}


void enable_event_logging(unsigned long max_to_log)
{
	if( log_buf != NO_LOGGED_EVENT ){
		/* release old buffer */
		givbuf(log_buf);
		log_buf = NO_LOGGED_EVENT;
	}

	if( max_to_log == 0 ){	/* disable logging */
		n_logged_events=0;
		max_logged_events=0;
		logging_events=0;
		return;
	}

	log_buf = getbuf( max_to_log * sizeof(*log_buf) );

#ifdef CAUTIOUS
	if( log_buf == NO_LOGGED_EVENT )
		NERROR1("CAUTIOUS:  unable to create event log buffer");
#endif /* CAUTIOUS */

	n_logged_events=0;
	max_logged_events = max_to_log;
	logging_events=1;
}

static void set_event_time(Logged_Event *lep)
{
#ifdef FTIME
	{
		struct timeb tb;
		ftime(&tb);

		lep->le_time = tb.time;
		lep->le_millitm = tb.millitm;
	}
#else
	{
		struct timeval tv;
		struct timezone tz;
		if( gettimeofday(&tv,&tz) < 0 ){
			perror("gettimeofday");
			NWARN("set_event_time:  error getting time of day");
		}
		lep->le_tv = tv;
	}
#endif /* ! FTIME */
}

static Logged_Event *next_event_log_record(void)
{
	Logged_Event *lep;

	if( n_logged_events >= max_logged_events ){
		if( ! event_overflow_warning_given ){
			sprintf(DEFAULT_ERROR_STRING,"Max. # of logged events (%ld) exceeded",
				max_logged_events);
			NWARN(DEFAULT_ERROR_STRING);
			event_overflow_warning_given = 1;
		}
		n_logged_events--;
	}
	lep = &log_buf[n_logged_events++];
	set_event_time(lep);
	return lep;
}

void log_x_event(XEvent *event)
{
	Logged_Event *lep;

	lep = next_event_log_record();

	if( event != (XEvent *)NULL ){
		lep->le_x_event = *event;
		lep->le_event_class = X_EVENT;
	} else {
		lep->le_event_class = NULL_EVENT;
	}
}

void log_atc_event(Atc_Event_Code code, Flight_Path *fpp)
{
	Logged_Event *lep;

	lep = next_event_log_record();

	lep->le_event_class = ATC_EVENT;
	lep->le_atc_event.ae_code = code;
	lep->le_atc_event.ae_fpp = fpp;
	lep->le_atc_event.ae_pt = *the_ptp;
}

/* the zeroeth entry is a dummy event just there to hold the time of the start
 * of the recording period...
 */

static void fix_event_log_times()
{
	u_long i;

#ifdef FTIME
	if( log_buf[0].le_millitm==0 && log_buf[0].le_time==0 ) return;	/* nothing to do */

	for(i=1;i<n_logged_events;i++){
		if( log_buf[i].le_millitm < log_buf[0].le_millitm ){
			log_buf[i].le_millitm += 1000;
			log_buf[i].le_time -= 1;
		}
		log_buf[i].le_millitm -= log_buf[0].le_millitm;
		log_buf[i].le_time -= log_buf[0].le_time;
	}
	log_buf[0].le_millitm = 0;
	log_buf[0].le_time = 0;
#else
	if( log_buf[0].le_tv.tv_usec==0 && log_buf[0].le_tv.tv_sec==0 ) return;	/* nothing to do */

	for(i=1;i<n_logged_events;i++){
		if( log_buf[i].le_tv.tv_usec < log_buf[0].le_tv.tv_usec ){
			log_buf[i].le_tv.tv_usec += 1000000;
			log_buf[i].le_tv.tv_sec -= 1;
		}
		log_buf[i].le_tv.tv_usec -= log_buf[0].le_tv.tv_usec;
		log_buf[i].le_tv.tv_sec -= log_buf[0].le_tv.tv_sec;
	}
	log_buf[0].le_tv.tv_usec = 0;
	log_buf[0].le_tv.tv_sec = 0;
#endif
}

void replay_event_log(SINGLE_QSP_ARG_DECL)
{
#ifdef FTIME
	struct timeb start_tb,tb;
#else
	struct timeval start_tv,tv;
	struct timezone tz;
#endif
	unsigned long i;

	sprintf(ERROR_STRING,"%ld events to replay",n_logged_events);
	advise(ERROR_STRING);

	fix_event_log_times();

	/* to prevent infinite recursion when we call process_atc_event() !!! */
	logging_events=0;

#ifdef FTIME
	ftime(&start_tb);
#else
	gettimeofday(&start_tv,&tz);
#endif

	/* the 0th event should be a null event with the clock zero time */

	/* BUG we ought to record the initial state in the 0th event... */
	atc_state = SELECTING_TAG;

	for(i=1;i<n_logged_events;i++){
#ifdef FTIME
		do {
			ftime(&tb);
			if( tb.millitm < start_tb.millitm ){
				tb.millitm += 1000;
				tb.time -= 1;
			}
			tb.millitm -= start_tb.millitm;
			tb.time -= start_tb.time;
		} while ( tb.time <= log_buf[i].le_time && tb.millitm < log_buf[i].le_millitm );
#else
		do {
			gettimeofday(&tv,&tz);
			if( tv.tv_usec < start_tv.tv_usec ){
				tv.tv_usec += 1000000;
				tv.tv_sec -= 1;
			}
			tv.tv_usec -= start_tv.tv_usec;
			tv.tv_sec -= start_tv.tv_sec;
		} while ( tv.tv_sec < log_buf[i].le_tv.tv_sec ||
		( tv.tv_sec == log_buf[i].le_tv.tv_sec && tv.tv_usec < log_buf[i].le_tv.tv_usec ) );

/*
sprintf(ERROR_STRING,"current time %ld, event time %ld",
tv.tv_sec,log_buf[i].le_tv.tv_sec);
advise(ERROR_STRING);
*/

#endif
		process_atc_event(QSP_ARG  &log_buf[i].le_x_event);
	}
}

void dump_event_log()
{
	u_long i;
	char *event_name;

	if( logging_events == 0 ){
		NWARN("dump_event_log:  event logging not enabled");
		return;
	}

	if( n_logged_events == 0 ){
		NWARN("dump_event_log:  no events logged");
		return;
	}

	/* why was this commented out??? */
	fix_event_log_times();

	/* indices start at 1 because first "event" is dummy to hold start time */
	for(i=1;i<n_logged_events;i++){

#ifdef FTIME
		sprintf(msg_str,"Event at %ld sec, %d msec",log_buf[i].le_time,
			log_buf[i].le_millitm);
#else
		sprintf(msg_str,"Event at %ld sec, %ld usec",log_buf[i].le_tv.tv_sec,
			log_buf[i].le_tv.tv_usec);
#endif
		prt_msg_frag(msg_str);

		if( log_buf[i].le_event_class == X_EVENT ){
			XEvent *event;
			int key;

			event = &log_buf[i].le_x_event;

			switch(event->type){
				case Expose:
					strcpy(msg_str,"Expose");
					break;
				case MotionNotify:
					event_name="MotionNotify";
					sprintf(msg_str,"MotionNotify %d %d",event->xmotion.x,event->xmotion.y);
					break;
				case ButtonRelease:
					sprintf(msg_str,"ButtonRelease %d %d %d",
						event->xbutton.button & 3,
						event->xbutton.x,event->xbutton.y);
					break;
				case KeyPress:
					key=get_event_key(event);
					sprintf(msg_str,"KeyPress 0%o 0x%x",key,key);
					break;
				case ButtonPress:
					sprintf(msg_str,"ButtonPress %d %d %d",
						event->xbutton.button & 3,
						event->xbutton.x,event->xbutton.y);
					break;
				case MapNotify:
					strcpy(msg_str,"MapNotify");
					break;
				default:
					NWARN("dump_event_log:  unexpected event type!?");
					event_name="???";
					break;
			}
			prt_msg_frag("\t");
			prt_msg(msg_str);
		} else if( log_buf[i].le_event_class == ATC_EVENT ){
			char *event_tag;
			switch(log_buf[i].le_atc_event.ae_code){
				case ICON_REVEALED: event_tag = "icon_revealed"; break;
				case ICON_CONCEALED: event_tag = "icon_concealed"; break;
				case TAG_REVEALED: event_tag = "tag_revealed"; break;
				case TAG_CONCEALED: event_tag = "tag_concealed"; break;
#ifdef CAUTIOUS
				default:
					sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  bad atc event code in dump_event_log()");
					event_tag="";	// quiet compiler
					NERROR1(DEFAULT_ERROR_STRING);
#endif /* CAUTIOUS */
			}
			sprintf(msg_str,"%s %s %g %g",event_tag,log_buf[i].le_atc_event.ae_fpp->fp_name,
				log_buf[i].le_atc_event.ae_pt.p_x,log_buf[i].le_atc_event.ae_pt.p_y);
			prt_msg_frag("\t");
			prt_msg(msg_str);
		}
	}
} /* end dump_event_log() */



/* end of event logging stuff */

#define NOTHING_HAPPENED	(-1)

void reposition_tag(QSP_ARG_DECL  Flight_Path *fpp,Point *ptp)
{
	draw_tag(fpp,BLACK);	/* erase old tag */
	DRAW_TAG_LINE(fpp,BLACK);

	fpp->fp_tag_loc = *ptp;

	/* recalculate line */
	fpp->fp_tag_dist = dist( &fpp->fp_plane_loc, ptp );
	fpp->fp_tag_angle =
		RADIANS_TO_DEGREES( atan2( ptp->p_y - fpp->fp_plane_loc.p_y, ptp->p_x - fpp->fp_plane_loc.p_x) );

	update_tag_line(QSP_ARG  fpp);

	draw_tag(fpp,WHITE);	/* draw new tag */
	DRAW_TAG_LINE(fpp,WHITE);	/* draw new tag */
	draw_plane(fpp,WHITE);
}

void mouse_to_airspace(Point *ptp)
{
	/* this function was originally introduced when we needed to flip the y
	 * coordinate...  the plotting space was changed to match the default
	 * screen coords, but we leave this here as a placeholder for when
	 * we represent the plane locations in miles...
	 *
	 * Actually, we need it when we display on the small screen...
	 */
}

void oops(char *s)
{
	/* BUG really ought to display this mesage in the display and beep! */
	NWARN(s);
}

#define TAG_WIDTH	tag_x_offset
#define TAG_HEIGHT	tag_x_offset

Flight_Path *tag_at(QSP_ARG_DECL  Point *ptp,Viewer *vp)
{
	List *lp;
	Node *np;
	Flight_Path *fpp;
	Point tmp_pt, *tptp=&tmp_pt;

	lp = plane_list(SINGLE_QSP_ARG);
#ifdef CAUTIOUS
	if( lp == NO_LIST ){
		WARN("CAUTIOUS:  no planes!?");
		return(NO_FLIGHT_PATH);
	}
#endif /* CAUTIOUS */

	np=lp->l_head;
	while(np!=NO_NODE){
		fpp = (Flight_Path *)(np->n_data);
		if( is_stereo ){
			if( vp == eye_screen[0] )
				current_eye = LEFT_EYE;
			else
				current_eye = RIGHT_EYE;
			get_disparity(fpp);
		} else {
			this_disparity=0;
		}
		tptp->p_x = ptp->p_x - this_disparity;
		tptp->p_y = ptp->p_y;

		if( inside_tag(fpp,tptp) ) return(fpp);
		np = np->n_next;
	}
	return(NO_FLIGHT_PATH);
}


Flight_Path *obj_selected_at(QSP_ARG_DECL  Point *ptp,Viewer *vp)
{
	Flight_Path *fpp;
	Node *np;
	List *lp;
	Point tmp_pt, *tptp=&tmp_pt;


	/* this implementation is simple minded - we just return
	 * the first hit we find...  a more sophisticated version
	 * would make a list of candidate hits and return the best
	 * one! BUG
	 */

	lp=plane_list(SINGLE_QSP_ARG);
#ifdef CAUTIOUS
	if( lp == NO_LIST ){
		WARN("CAUTIOUS:  no planes!?");
		return(NO_FLIGHT_PATH);
	}
#endif /* CAUTIOUS */

	np=lp->l_head;
	while(np!=NO_NODE){
		fpp = (Flight_Path *)(np->n_data);

		if( is_stereo ){
			if( vp == eye_screen[0] )
				current_eye = LEFT_EYE;
			else
				current_eye = RIGHT_EYE;
			get_disparity(fpp);
		}

		tptp->p_x = ptp->p_x - this_disparity;
		tptp->p_y = ptp->p_y;

		if( dist(&fpp->fp_plane_loc,tptp) < PLANE_RADIUS )
			return(fpp);

		if( inside_tag(fpp,tptp) )
			return(fpp);

		np=np->n_next;
	}
	return(NO_FLIGHT_PATH);
}

static void no_thing_at(char *thing_name,Point *ptp)
{
	sprintf(DEFAULT_ERROR_STRING,"no %s at location %g %g",thing_name,ptp->p_x,ptp->p_y);
	oops(DEFAULT_ERROR_STRING);
}

static void process_atc_event(QSP_ARG_DECL  XEvent *event)
{
	Viewer *vp;
	Window win;
	/* int x,y; */
	Point mouse_point;
	time_t now_time;
	Flight_Path *fpp;
	int key;

	switch (event->type) {
		case Expose:
			win = event->xexpose.window;

			if( (vp=find_viewer(QSP_ARG  win)) == NO_VIEWER ){
				WARN("can't find viewing window for expose event");
				return;
			}
			vp->vw_flags |= VIEW_EXPOSED;
	/*
	 * When a viewer is brought to front, an expose event is generated
	 * by EACH window that previously overlapped it; thus, on a
	 * crowded screen, this could be called many times.  It
	 * therefore makes sense to NOT redraw for events that happen
	 * shortly after the last one...
	 */
			if( (now_time=time((time_t *)NULL)) == (time_t) -1 ){
				WARN("error getting time");
				redraw_viewer(QSP_ARG  vp);
			} else {
				if( (now_time - vp->vw_time) > 1 )
					redraw_viewer(QSP_ARG  vp);
				vp->vw_time = now_time;
			}
			return;

		case MotionNotify:		/* process_atc_event */
			if( mirror_reversed )
				FLIP_VAR(event->xmotion.x);

			if( logging_events ) log_x_event(event);

			if( atc_state == DRAGGING_TAG ){
				mouse_point.p_x = event->xmotion.x;
				mouse_point.p_y = event->xmotion.y;
				mouse_to_airspace(&mouse_point);

				if( allow_tag_movement && inbounds(&mouse_point) )
					reposition_tag(QSP_ARG  selected_tag,&mouse_point);
			} else if ( atc_state == SELECTING_PLANE_1 ||
					atc_state == SELECTING_PLANE_2 ||
					atc_state == WAITING_FOR_APPROVAL ) {
				if( cursor_controls_visibility ){
					mouse_point.p_x = event->xmotion.x;
					mouse_point.p_y = event->xmotion.y;
					mouse_to_airspace(&mouse_point);

					render_visible(QSP_ARG  &mouse_point);
				}
			}
			break;

		case ButtonRelease:
			if( mirror_reversed )
				FLIP_VAR(event->xbutton.x);

			if( logging_events ) log_x_event(event);

			if( atc_state == DRAGGING_TAG ){
				Flight_Path *fpp;

				mouse_point.p_x=event->xbutton.x;
				mouse_point.p_y=event->xbutton.y;
				mouse_to_airspace(&mouse_point);
				if( allow_tag_movement && inbounds(&mouse_point) )
					reposition_tag(QSP_ARG  selected_tag,&mouse_point);

				fpp = selected_tag;
				selected_tag = NO_FLIGHT_PATH;
				refresh_object(QSP_ARG  fpp);
				atc_state = SELECTING_TAG;

				/* this is using a sledgehammer to kill
				 * a fly, but it guarantees that no other
				 * planes will be munged by the tag passing
				 * over them.
				 */
				draw_objects(SINGLE_QSP_ARG);
			} else if( atc_state == WAITING_FOR_CLICK ){
				atc_state = SELECTION_DONE;
			}
			break;

		case KeyPress:

			if( logging_events ) log_x_event(event);

			key = get_event_key(event);
			if( key < 0 ) return;

			/* this was expected to be the ascii character,
			 * but on the mac KB, space gives 0x39, and
			 * other keys give strange codes...
			 */
			if( key == ' ' ){	/* space pressed */
				if( atc_state == WAITING_FOR_FIXATION ){
					/* We used to delay here, but
					 * we need to go and tell the recorder
					 * to start here
					 */
					atc_state = SUBJECT_READY;
				} else if( atc_state == SELECTING_TAG
					|| atc_state == WAITING_FOR_APPROVAL ) {
					atc_state = SELECTION_DONE;
				} else if( atc_state == SELECTING_PLANE_1 ){
					oops("no planes selected");
				} else if( atc_state == SELECTING_PLANE_2 ){
					sprintf(ERROR_STRING,
		"Only one plane currently selected:  %s",selection_1->fp_name);
					oops(ERROR_STRING);
				}
			} else if( key == 3 ){	/* ^C */
				nice_exit(0);
			} else if( key == 'a' || key == 'q' ){	/* abort trial */
				atc_state = SELECTION_DONE;
				trial_aborted=1;
			}

			break;

		case ButtonPress:
			if( exit_on_right_click && ((event->xbutton.button&3)==3) )
				exit(0);

			if( mirror_reversed )
				FLIP_VAR(event->xbutton.x);

			if( logging_events ) log_x_event(event);

			/* first figure out which window was clicked in */
			win = event->xbutton.window;

			if( (vp=find_viewer(QSP_ARG  win)) == NO_VIEWER ){
				WARN("can't find viewing window for button press event");
				return;
			}

			mouse_point.p_x=event->xbutton.x;
			mouse_point.p_y=event->xbutton.y;

			mouse_to_airspace(&mouse_point);

			if( atc_state == STEPPING ){
				Point new_fixation;

				plan_saccade( QSP_ARG   &new_fixation );
				make_fixation( QSP_ARG  &new_fixation );
				if( conflict_found(SINGLE_QSP_ARG) ){
					atc_state = SELECTION_DONE;
					report_model_rt(SINGLE_QSP_ARG);
				}
			} else if( atc_state == CLICK_FIXING ){
				make_fixation(QSP_ARG  &mouse_point);
				/* check for conflicts */
				if( conflict_found(SINGLE_QSP_ARG) ){
					atc_state = SELECTION_DONE;
					report_model_rt(SINGLE_QSP_ARG);
				}
			} else if( atc_state == WAITING_FOR_CLICK ){
				/* this used to have the click_fixing stuff here... */
				if( atc_state == WAITING_FOR_CLICK )
					atc_state = SELECTION_DONE;
			} else if( atc_state == SELECTING_TAG ){
				if( (fpp=tag_at(QSP_ARG  &mouse_point,vp)) != NO_FLIGHT_PATH ){
					selected_tag = fpp;
					refresh_object(QSP_ARG  fpp);
					atc_state=DRAGGING_TAG;
				} else no_thing_at("tag",&mouse_point);
			} else if( atc_state == SELECTING_PLANE_1 ){
				if( (fpp=obj_selected_at(QSP_ARG  &mouse_point,vp))
					!= NO_FLIGHT_PATH ){

					selection_1 = fpp;
					refresh_object(QSP_ARG  fpp);
					atc_state = SELECTING_PLANE_2;
				} else no_thing_at("plane",&mouse_point);
			} else if( atc_state == SELECTING_PLANE_2 ){
				if( (fpp=obj_selected_at(QSP_ARG  &mouse_point,vp))
					!= NO_FLIGHT_PATH ){

					if( fpp == selection_1 ){
						selection_1 = NO_FLIGHT_PATH;
						refresh_object(QSP_ARG  fpp);
						atc_state = SELECTING_PLANE_1;
					} else {
						selection_2 = fpp;
						refresh_object(QSP_ARG  fpp);
						atc_state = WAITING_FOR_APPROVAL;
					}
				} else no_thing_at("plane",&mouse_point);
			} else if( atc_state == WAITING_FOR_APPROVAL ){
				if( (fpp=obj_selected_at(QSP_ARG  &mouse_point,vp))
					!= NO_FLIGHT_PATH ){

					/* If we click a craft after we've already selected two,
					 * it ought to be because we're deselecting one of our choices...
					 */
					if( fpp == selection_1 ){
						selection_1 = selection_2;
						selection_2 = NO_FLIGHT_PATH;
						refresh_object(QSP_ARG  fpp);
						atc_state = SELECTING_PLANE_2;
					} else if( fpp == selection_2 ){
						selection_2 = NO_FLIGHT_PATH;
						refresh_object(QSP_ARG  fpp);
						atc_state = SELECTING_PLANE_2;
					} else {
						sprintf(ERROR_STRING,"must deselect %s or %s before selecting %s",
								selection_1->fp_name,selection_2->fp_name,
								fpp->fp_name);
						WARN(ERROR_STRING);
					}
				} else no_thing_at("plane",&mouse_point);
			} else if( atc_state == SELECTING_ATC_POSITION ){
				if( listed_position(QSP_ARG  &mouse_point) )
					atc_state = SELECTION_DONE;
				break;
			} else if( atc_state == SELECTING_ATC_OBJECT ){
				if( (fpp=obj_selected_at(QSP_ARG  &mouse_point,vp)) != NO_FLIGHT_PATH ){
					Flight_Path *tagp;
					Point *ptp;
					if( (tagp=tag_at(QSP_ARG  &mouse_point,vp))!=NO_FLIGHT_PATH ){
						highlight_object(QSP_ARG  tagp,1);
						sprintf(ERROR_STRING,"Object %s tag selected",
							tagp->fp_name);
						ptp = &fpp->fp_tag_loc;
						sprintf(msg_str,"%f",ptp->p_x);
						ASSIGN_VAR("selected_x",msg_str);
						sprintf(msg_str,"%f",ptp->p_y);
						ASSIGN_VAR("selected_y",msg_str);
					} else {
						highlight_object(QSP_ARG  fpp,0);
						sprintf(ERROR_STRING,"Object %s icon selected",
							fpp->fp_name);
						ptp = &fpp->fp_plane_loc;
						sprintf(msg_str,"%f",ptp->p_x);
						ASSIGN_VAR("selected_x",msg_str);
						sprintf(msg_str,"%f",ptp->p_y);
						ASSIGN_VAR("selected_y",msg_str);
					}
				} else {
					no_thing_at("object",&mouse_point);
					break;
				}
				advise(ERROR_STRING);
				atc_state = SELECTION_DONE;
			} else {
				/* what states might we be in here??? */
				sprintf(ERROR_STRING,
		"Unexpected button press in state %d",atc_state);
				advise(ERROR_STRING);
			}
			break;

		case MapNotify:
			win = event->xmapping.window;
			if( (vp=find_viewer(QSP_ARG  win)) == NO_VIEWER ){
				WARN("can't find viewing window for map event");
				return;
			}
			vp->vw_flags |= VIEW_MAPPED;
			redraw_viewer(QSP_ARG  vp);
			return;

		case ConfigureNotify:
			break;
		case ReparentNotify:
			break;
		/* ignore all of these events... */
		/*
		case CirculateNotify:
		case DestroyNotify:
		case GravityNotify:
		case UnmapNotify:
			break;
		*/

		/*
		case ReparentNotify: retval=do_reparent(event); break;
		*/
		case EnterNotify:
			break;
		/* case LeaveNotify: do_enter_leave(event); break; */
		/* case 14: break; */		/* what is this??? */
		default: 
sprintf(ERROR_STRING,"uncaught event type %d",event->type);
WARN(ERROR_STRING);

#ifdef DEBUG
/* if( debug ){ */
sprintf(ERROR_STRING,"uncaught event type %d",event->type);
WARN(ERROR_STRING);
/* } */
#endif /* DEBUG */
			break;		/* ignore unexpected events */
	}
} /* end process_atc_event() */

static void discard_pending_events(void)
{
	XEvent event;
	long mask;
	Disp_Obj *dop;

	if( (dop=curr_dop()) == NO_DISP_OBJ ) return;

	mask =	  ExposureMask
		| ButtonPressMask
		| KeyPressMask
		| ButtonReleaseMask
		| ButtonMotionMask
		| StructureNotifyMask
		| PointerMotionMask
		/* | Button1MotionMask */
		;
		/* | Button1MotionMask */

	while( XCheckMaskEvent(dop->do_dpy,mask,&event) == True )
		;
}

static void check_for_atc_event(SINGLE_QSP_ARG_DECL)
{
	XEvent event;
	long mask;
	Disp_Obj *dop;

	if( (dop=curr_dop()) == NO_DISP_OBJ ) return;

	mask =	  ExposureMask
		| ButtonPressMask
		| KeyPressMask
		| ButtonReleaseMask
		| ButtonMotionMask
		| StructureNotifyMask
		| PointerMotionMask
		;
		/* | Button1MotionMask */

	if( XCheckMaskEvent(dop->do_dpy,mask,&event) == True )
		process_atc_event(QSP_ARG  &event);
}

COMMAND_FUNC( move_tags )
{
	allow_tag_movement = ASKIF("allow tag movement");

	atc_state = SELECTING_TAG;
	selection_1 = selection_2 = NO_FLIGHT_PATH;

	discard_pending_events();
	
	/* BUG we zero the event clock here, but eventually
	 * we may want to do this in a more sensible way
	 */
	if( logging_events ){
		n_logged_events=0;
		event_overflow_warning_given = 0;
		log_x_event((XEvent *)NULL);
	}
		
	while( atc_state == SELECTING_TAG || atc_state == DRAGGING_TAG )
		check_for_atc_event(SINGLE_QSP_ARG);
}

COMMAND_FUNC( select_planes )
{
	struct timezone tz;
	struct timeval tv;

	if( trial_aborted ) return;

	atc_state = SELECTING_PLANE_1;
	selection_1 = selection_2 = NO_FLIGHT_PATH;

	discard_pending_events();

	if( timeout_seconds > 0 ){
		if( gettimeofday(&trial_start_tv,&tz) < 0 ){
			perror("gettimeofday");
			WARN("select_planes:  error getting start trial start time, disabling timeout");
			timeout_seconds=(-1);
		}
	}
	
	/*refresh_objects(); */	/* so previous selections don't display */

	while( atc_state == SELECTING_PLANE_1 || atc_state == SELECTING_PLANE_2
		|| atc_state == WAITING_FOR_APPROVAL ){

		check_for_atc_event(SINGLE_QSP_ARG);

		if( timeout_seconds > 0 ){
			if( gettimeofday(&tv,&tz) < 0 ){
				perror("gettimeofday");
				WARN("select_planes:  error getting current time, disabling timeout");
				timeout_seconds=(-1);
			} else {
				long delta_seconds, delta_usecs;

				delta_seconds = tv.tv_sec - trial_start_tv.tv_sec;
				delta_usecs = tv.tv_usec - trial_start_tv.tv_usec;
				if( delta_usecs < 0 ){
					delta_usecs += 1000000;
					delta_seconds -= 1;
				}

				if( delta_seconds >= timeout_seconds ){
					sprintf(msg_str,"TIMEOUT after %ld seconds",delta_seconds);
					prt_msg(msg_str);
					trial_aborted=1;
					atc_state = SELECTION_DONE;
				}
			}
		}
	}

	if( ! trial_aborted ){
		if( IS_IN_CONFLICT(selection_1) && IS_IN_CONFLICT(selection_2) ){
			sprintf(msg_str,"CONFLICT DETECTED : %s - %s",
				selection_1->fp_name,selection_2->fp_name);
			prt_msg(msg_str);
			if( correct_action != NULL ){
				CHEW_TEXT(correct_action);
			}
		} else {
			sprintf(msg_str,"FALSE ALARM : %s - %s",
				selection_1->fp_name,selection_2->fp_name);
			prt_msg(msg_str);
			if( incorrect_action != NULL ){
				CHEW_TEXT(incorrect_action);
			}
		}
	}
	/* BUG?  do we need these later??? */
	selection_1 = selection_2 = NO_FLIGHT_PATH;
}

COMMAND_FUNC( present_fixation )
{
	draw_fixation_screen(NULL_QSP);
	atc_state = WAITING_FOR_FIXATION;
	trial_aborted =0;

	discard_pending_events();
	
	while( atc_state == WAITING_FOR_FIXATION )
		check_for_atc_event(SINGLE_QSP_ARG);
}

static Data_Obj *_posn_list=NO_OBJ;
static float dist_thresh=10.0;
Point *selected_ptp=NO_POINT;

static int listed_position(QSP_ARG_DECL  Point *ptp)
{
	dimension_t n;
	Point *ptp2;

	n = _posn_list->dt_cols;
	ptp2 = _posn_list->dt_data;
	while(n--){
		if( DIST(ptp,ptp2) < dist_thresh ){
			selected_ptp = ptp2;
			sprintf(msg_str,"%f",ptp2->p_x);
			ASSIGN_VAR("selected_x",msg_str);
			sprintf(msg_str,"%f",ptp2->p_y);
			ASSIGN_VAR("selected_y",msg_str);
			return(1);
		}
		ptp2++;
	}
	return(0);
}

void select_posn_from_list(QSP_ARG_DECL  Data_Obj *dp )
{
	atc_state = SELECTING_ATC_POSITION;
	selected_ptp = NO_POINT;
	_posn_list = dp;
	while( atc_state == SELECTING_ATC_POSITION )
		check_for_atc_event(SINGLE_QSP_ARG);
	atc_state = SELECTION_DONE;
	/* Now we need to pass something to the user?? */
}

COMMAND_FUNC( select_any_object )
{
	atc_state = SELECTING_ATC_OBJECT;
	while( atc_state == SELECTING_ATC_OBJECT )
		check_for_atc_event(SINGLE_QSP_ARG);
	atc_state = SELECTION_DONE;
	/* Now we need to pass something to the user?? */
}

COMMAND_FUNC( sgl_click_fix )
{
	atc_state = WAITING_FOR_CLICK;
	while( atc_state == WAITING_FOR_CLICK )
		check_for_atc_event(SINGLE_QSP_ARG);
	atc_state = SELECTION_DONE;
}

COMMAND_FUNC( click_fixations )
{
	atc_state = CLICK_FIXING;
	while( atc_state == CLICK_FIXING )
		check_for_atc_event(SINGLE_QSP_ARG);
	if( trial_aborted )
		end_simulation_run(QSP_ARG  "trial aborted");
}

COMMAND_FUNC( wait_for_click )
{
	/* This routine was introduced to facilitate netscape-driven demos.
	 *
	 * We want to wait for up-clicks, so that we don't exit with
	 * the button down (after which the releasing upclick re-selects
	 * the invoking link!?)
	 */

	atc_state = WAITING_FOR_CLICK;
	while( atc_state == WAITING_FOR_CLICK )
		check_for_atc_event(SINGLE_QSP_ARG);
}

COMMAND_FUNC( step_model )
{
	atc_state = STEPPING;
	while( atc_state == STEPPING )
		check_for_atc_event(SINGLE_QSP_ARG);
	if( trial_aborted )
		end_simulation_run(QSP_ARG  "trial aborted");
}

void set_exit_on_right_click( int yesno )
{
	exit_on_right_click = yesno;
}

#endif /* HAVE_X11 */

