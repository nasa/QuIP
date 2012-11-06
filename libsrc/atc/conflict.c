
#include "quip_config.h"

char VersionId_atc_conflict[] = QUIP_VERSION_STRING;

#include <sys/types.h>
#include <sys/timeb.h>
#include <stdio.h>
#include <unistd.h>	/* sleep */
#include <math.h>	/* sqrt() */
#include <signal.h>	/* sqrt() */

#include "query.h"
#include "rn.h"			/* scramble */
#include "viewer.h"		/* pick_viewer() */
#include "submenus.h"		/* atcmenu() */

#include "conflict.h"
#include "draw.h"		/* SELECTED_PLANE_COLOR */

/* globals */

const char *correct_action=NULL, *incorrect_action=NULL;

/* these used to be part of the trial structure...
 */

unsigned long atc_flags;

int conceal_radius=0;
/* static long trialTime; */
static long updatenum;   /* # updates so far this trial*/
static int smart_tag_loc=1;

int mirror_reversed;


/* utility function to iterate over all the planes */

void apply_to_planes(QSP_ARG_DECL  void (*func)(QSP_ARG_DECL  Flight_Path *))
{
	Node *np;
	List *lp;

	lp = plane_list(SINGLE_QSP_ARG);

	if( lp==NO_LIST ){
		WARN("no planes in existence!?");
		return;
	}

	np=lp->l_head;
	while(np!=NO_NODE){
		Flight_Path *fpp;
		fpp=(Flight_Path *)(np->n_data);
		(*func)(QSP_ARG  fpp);
		np=np->n_next;
	}
}

static void update_position(Flight_Path *fpp)
{
	fpp->fp_plane_loc.p_x += fpp->fp_vel.v_x;
	fpp->fp_plane_loc.p_y += fpp->fp_vel.v_y;
}

void update_object(QSP_ARG_DECL  Flight_Path *fpp)
{
	erase_object(QSP_ARG  fpp);
	update_position(fpp);
	recompute_coords(QSP_ARG  fpp);
	draw_object(QSP_ARG  fpp);
}

static void update_objects(SINGLE_QSP_ARG_DECL)
{
	apply_to_planes(QSP_ARG  update_object);
}

/* ------------------------------------------------------------------ */
static COMMAND_FUNC( update )
{
	/* time to move planes and then redraw them */

/*
sprintf(ERROR_STRING,"trialTime = %ld, nextupdatetime = %ld",
trialTime,nextupdatetime);
advise(ERROR_STRING);
*/

	/*
	while (trialTime > nextupdatetime) {
	*/
		update_objects(SINGLE_QSP_ARG);
		updatenum++;
	/*
		nextupdatetime = UPDATE_INTERVAL * updatenum;
	}
	*/
	/* now draw all the planes again so that they appear on top */
	redraw_planes(SINGLE_QSP_ARG);
}  /* update */

int atc_overlay_mode=0;

COMMAND_FUNC( overlay_enable )
{
	atc_overlay_mode = ASKIF("draw atc graphics on previous window contents");
}

static void update_things(QSP_ARG_DECL  Flight_Path *fpp)
{
	update_wedge(fpp);
	update_tag(fpp);

	/* compute velocity from speed and heading */

	/*
	fpp->fp_vel.v_x = (fpp->fp_speed * PIXELS_PER_KNOT)
		* cos( DEGREES_TO_RADIANS(fpp->fp_theta) );
	fpp->fp_vel.v_y = (fpp->fp_speed * PIXELS_PER_KNOT)
		* sin( DEGREES_TO_RADIANS(fpp->fp_theta) );
	*/
}

static void improve_tag(QSP_ARG_DECL  Flight_Path *fpp)
{
	find_best_tag_angle(QSP_ARG  fpp);
	update_tag(fpp);
}

/* ---------------------------------------------------------------- */
/* at beginning of trial, initialize locations of planes, tags, etc */

COMMAND_FUNC( setup_object_coords )
{
	apply_to_planes( QSP_ARG  update_things );

	if( smart_tag_loc ){   /* places tags so as to minimize overwriting */
		int n_passes=2;

		while(n_passes--)
			apply_to_planes( QSP_ARG  improve_tag );
	}

	apply_to_planes( QSP_ARG  update_tag_line );
}

static COMMAND_FUNC( set_altcolor )
{
	if( ASKIF("use color coding for altitude") )
		atc_flags |= ALT_COLOR_BIT;
	else
		atc_flags &= ~ALT_COLOR_BIT;
}

#define MAX_FIXATION_DURATION	1000

static COMMAND_FUNC( set_fixation_duration )
{
	int d;

	d = HOW_MANY("calibration fixation duration in milliseconds");
	if( d>=0 && d < MAX_FIXATION_DURATION )
		calibration_fixation_duration=d;
	else {
		sprintf(ERROR_STRING,"Requested fixation duration %d out of range (0-%d)",
			d,MAX_FIXATION_DURATION);
		WARN(ERROR_STRING);
	}
}

static COMMAND_FUNC( do_delay )
{
	int d;

	d=HOW_MANY("delay in milliseconds");
	delay(d);
}

static COMMAND_FUNC( do_scan_path )
{
	Data_Obj *dp;

	dp=PICK_OBJ("vector of fixations");

	if( dp==NO_OBJ || ! good_scanpath_vector(QSP_ARG  dp) ) return;

	draw_scan_path(dp);
}

static COMMAND_FUNC( do_log_events )
{
	long n;

	n=HOW_MANY("maximum number of events to log (0 to disable)");
	if( n < 0 ) {
		WARN("number of events to log should be positive (0 to disable)");
		return;
	}
	enable_event_logging((u_long)n);
}

static COMMAND_FUNC( do_replay_events )
{
	replay_event_log(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_dump_events )
{
	dump_event_log();
}

static COMMAND_FUNC( do_select_posn )
{
	Data_Obj *dp;

	dp = PICK_OBJ("position list");
	if( dp == NO_OBJ ) return;

	select_posn_from_list(QSP_ARG  dp);
}

static COMMAND_FUNC( do_draw_legend )
{
	int i_trl;

	i_trl = HOW_MANY("trial number");
	/* should we do error checking here? */
	draw_legend(QSP_ARG  i_trl);
}

static COMMAND_FUNC( do_set_disparities )
{
	int d[4][2];
	char prompt[LLEN];
	int i;

	for(i=0;i<4;i++){
		sprintf(prompt,"left eye disparity for altitude %d",i+1);
		d[i][0] = HOW_MANY(prompt);
		sprintf(prompt,"right eye disparity for altitude %d",i+1);
		d[i][1] = HOW_MANY(prompt);
	}
	set_disparities(d);
}

static COMMAND_FUNC( do_conceal )
{
	conceal_radius = HOW_MANY("radius of zone of de-concealment (0 or negative to disable concealment)");
	if( conceal_radius <= 0 )
		cursor_controls_visibility = 0;
	else
		cursor_controls_visibility = 1;
sprintf(ERROR_STRING,"cursor_controls_visibility = %d, radius = %d",
cursor_controls_visibility,conceal_radius);
advise(ERROR_STRING);
}

static COMMAND_FUNC( define_feedback )
{
	const char *s;

	s=NAMEOF("action string for correct feedback");
	if( correct_action != NULL )
		givbuf((char *)correct_action);
	correct_action = savestr(s);

	s=NAMEOF("action string for incorrect feedback");
	if( incorrect_action != NULL )
		givbuf((char *)incorrect_action);
	incorrect_action = savestr(s);
}

static COMMAND_FUNC( set_timeout )
{
	timeout_seconds = HOW_MANY("trial timeout delay in seconds (negative for no limit)");
}

static Command stim_ctbl[]={
{ "update",		update,			"update display"			},
{ "render",		do_render,		"render display"			},
{ "overlay",		overlay_enable,		"enable/disable graphics overlay mode"	},
{ "scan_path",		do_scan_path,		"render scan path"			},
{ "select_obj",		select_any_object,	"select an object using the mouse"	},
{ "select_posn",	do_select_posn,		"select a position using the mouse"	},
{ "setup_coords",	setup_object_coords,	"initialize object drawing coords"	},
{ "region",		draw_region,		"draw region circle"			},
{ "legend",		do_draw_legend,		"draw display legend"			},
{ "fixation",		present_fixation,	"present pretrial fixation point"	},
{ "fixation_duration",	set_fixation_duration,	"specify calibration fixation duration"	},
{ "move_tags",		move_tags,		"tag movement mode"			},
{ "select_planes",	select_planes,		"plane selection mode"			},
{ "timeout",		set_timeout,		"specify timeout delay in seconds"	},
{ "altcolor",		set_altcolor,		"enable/disable altitude color coding"	},
{ "conceal",		do_conceal,		"conceal info except near cursor"	},
{ "disparities",	do_set_disparities,	"specify disparities for stereo rendering"	},
{ "delay",		do_delay,		"delay some number of milliseconds"	},
{ "log_events",		do_log_events,		"specify number of X events to log"	},
{ "replay_events",	do_replay_events,	"replay logged events"			},
{ "dump_events",	do_dump_events,		"show event log"			},
{ "wait_for_click",	wait_for_click,		"display stimulus until mouse click"	},
{ "feedback",		define_feedback,	"define feedback actions"		},
{ "quit",		popcmd,			"exit submenu"				},
{ NULL_COMMAND										}
};

static COMMAND_FUNC( stim_menu ){ PUSHCMD(stim_ctbl,"stimuli"); }

#define NO_X11	WARN("Sorry, program not built with X11 support");

static COMMAND_FUNC( do_set_screen )
{
#ifdef HAVE_X11
	Viewer *vp;

	vp = PICK_VWR("");
	if( vp == NO_VIEWER ){
		WARN("do_set_screen:  bad viewer selection");
		return;
	}

	set_screen_viewer(QSP_ARG  vp);
#else
	NO_X11
#endif
}

static COMMAND_FUNC( do_set_stereo )
{
#ifdef HAVE_X11
	Viewer *l_vp,*r_vp;

	l_vp = PICK_VWR("viewer for left eye image");
	r_vp = PICK_VWR("viewer for right eye image");
	if( l_vp == NO_VIEWER || r_vp == NO_VIEWER ) {
		WARN("do_set_stereo:  bad viewer selection");
		return;
	}

	set_stereo_viewers(QSP_ARG  l_vp,r_vp);
advise("STEREO viewers set");
#else
	NO_X11
#endif
}

static COMMAND_FUNC( do_set_mirror )
{
	mirror_reversed = ASKIF("mirror-reverse text");
}


static COMMAND_FUNC( do_tag_enable )
{
	set_tag_visibility( ASKIF("display tags") );
}

static Command gfx_ctbl[]={
{ "init_graphics",	init_graphics,		"create default screen window"		},
{ "screen",		do_set_screen,		"specify existing window for screen"	},
{ "stereo",		do_set_stereo,		"specify a pair windows for stereo display"	},
{ "mirror",		do_set_mirror,		"enable/disable mirror reversal of text"	},
{ "tags",		do_tag_enable,		"enable/disable tag display"		},
{ "quit",		popcmd,			"exit submenu"				},
{ NULL_COMMAND										}
};

static COMMAND_FUNC( gfx_menu ) { PUSHCMD(gfx_ctbl,"gfx"); }

static COMMAND_FUNC( do_set_exit )
{
	int yn;

	yn = ASKIF("exit program on right-click");
	set_exit_on_right_click(yn);
}


static Command atc_ctbl[]={
{ "planes",		plane_menu,		"flight path submenu"			},
{ "graphics",		gfx_menu,		"graphical display submenu"		},
{ "model",		model_menu,		"search simulation submenu"		},
{ "stimuli",		stim_menu,		"stimulus presentation submenu"		},
{ "pathgen",		pathgen_menu,		"stimulus generation submenu"		},
{ "exit_on_right_click",do_set_exit,		"enable/disable exit on right-click (for demos)"},
{ "quit",		popcmd,			"exit submenu"				},
{ NULL_COMMAND										}
};

COMMAND_FUNC( atc_menu )
{
	PUSHCMD(atc_ctbl,"atc");
}

