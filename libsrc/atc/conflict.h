
#ifndef CONFLICT_H
#define CONFLICT_H

/* defns for conflict.c */

#include "typedefs.h"		/* FLOAT_ARG */
#include "data_obj.h"		/* Data_Obj */
#include "viewer.h"		/* Viewer */
#include "query.h"

typedef int boolean;
typedef float atc_type;
#define STD_ARG	double

#include "node.h"
#include "geom.h"
#include "flight_path.h"

#define TRUE	1
#define FALSE	0

typedef struct indexed_name {
	int	in_index;
	char *	in_name;
} Indexed_Name;

/* These are in kph, we need to rationalize the system of units...
 * It looks as if we don't have to worry about it here, however, because
 * the scale factors cancel out by the time we do the comparison...
 */

#define MIN_SPEED	400	/* knots per hour */
#define MAX_SPEED	500

extern double assumed_min_speed; 
extern double assumed_max_speed; 
extern int atc_overlay_mode;

#ifdef FOOBAR
typedef struct trial_data {
	long trial_id;			/* index # for this trial */
	long trialtype, angle; /* angle of incidence of the colliding planes*/
	Flight_Path planepaths[MAX_PLANES];
	atc_type contime; /* # updates til collision occurs */
	long num_planes, num_xy_foils; /* written to output file */
	boolean alt_differences; /* whether alt restrictions are imposed */
	boolean fixed; /* whether routes are fixed */
	atc_type initial_distance; /* written to output file */
	boolean large_xy_foils; /* written to output file */
	boolean large_angle; /* written to output file */
	atc_type conflict_eccentricity; /* written to output file */
} Trial_Data;
#endif /* FOOBAR */



#define UPDATE_INTERVAL		4.0	/* seconds */
#define SECONDS_PER_HOUR	( 60.0 * 60.0 )
#define UPDATES_PER_HOUR	( SECONDS_PER_HOUR / UPDATE_INTERVAL )

#define SUFFICIENT_OBJECT_SEPARATION	50

/* event stuff */

typedef enum {
	WAITING_FOR_FIXATION,		/* 0 */
	SUBJECT_READY,			/* 1 */

	SELECTING_TAG,			/* 2 */
	DRAGGING_TAG,			/* 3 */

	SELECTING_PLANE_1,		/* 4 */
	SELECTING_PLANE_2,		/* 5 */
	WAITING_FOR_APPROVAL,		/* 6 */
	SELECTION_DONE,			/* 7 */

	WAITING_FOR_CLICK,		/* 8 */
	CLICK_FIXING,			/* 9 */
	STEPPING,			/* 10 */
	SELECTING_ATC_OBJECT,		/* 11 */
	SELECTING_ATC_POSITION		/* 12 */
} Atc_State;

/* The above states allow us to implement several modes, which
 * we will identify:
 *
 * TRIAL_INITIATION MODE
 *
 * WAITING_FOR_FIXATION -> (space) -> SUBJECT_READY
 *
 * TAG MOVEMENT MODE
 *
 * SELECTING_TAG -> <down click> -> DRAGGING_TAG -> <up click> -> SELECTING_TAG
 * SELECTING_TAG -> (space) -> SELECTING_PLANE_1
 *
 *
 * PLANE SELECTION MODE
 *
 * SELECTING_PLANE1 -> (click on plane) -> SELECTING_PLANE2
 *
 * SELECTING_PLANE2 -> (click on plane1) -> SELECTING_PLANE1
 * SELECTING_PLANE2 -> (click on plane) -> WAITING_FOR_APPROVAL
 *
 * WAITING_FOR_APPROVAL -> (click on plane[12]) -> SELECTING_PLANE2
 * WAITING_FOR_APPROVAL -> (space) -> SELECTION_DONE
 *
 * We would like to do this in a general way to allow flexible
 * interactive specification of new state transition systems...
 */

#ifndef MIN
#define MIN(a,b)	( (a)<(b) ? (a) : (b) )
#endif /* MIN */

#ifndef MAX
#define MAX(a,b)	( (a)>(b) ? (a) : (b) )
#endif /* MAX */

#define ROUND(x)	floor( (x) + 0.5 )

#define BOUND(var,lower,upper)							\
										\
	var = ( (var) < (lower) ? (lower) : ( (var) > (upper) ? (upper) : (var) ) )


#include "model.h"	/* has to be after Flight_Path */

typedef enum {
	LEFT_EYE,
	RIGHT_EYE,
	N_EYES
} Eye;

typedef enum {
	ICON_REVEALED,
	ICON_CONCEALED,
	TAG_REVEALED,
	TAG_CONCEALED,
	N_ATC_EVENTS
} Atc_Event_Code;

typedef struct atc_event {
	Atc_Event_Code	ae_code;
	Flight_Path *	ae_fpp;
	Point		ae_pt;
} Atc_Event;

/* globals */

extern int timeout_seconds;
extern const char *correct_action, *incorrect_action;
extern Point *the_ptp;

extern int mirror_reversed;		/* a boolean flag */
extern int display_width;
#define FLIP_X(x)			(display_width-((x)+1))


extern int cursor_controls_visibility;
extern int conceal_radius;

#ifdef HAVE_X11
extern Viewer *eye_screen[2];
#define left_screen	eye_screen[0]
#define right_screen	eye_screen[1]
#endif /* HAVE_X11 */

extern int current_eye;

extern atc_type heading_ecc_thresh;
extern int this_disparity;
extern int is_stereo;

extern int calibration_fixation_duration;

extern Atc_State atc_state;
extern Flight_Path *selection_1, *selection_2;
extern Flight_Path *selected_tag;   /* tag currently being dragged */

extern unsigned long atc_flags;

#define FIXED_ROUTES_BIT	1
#define ALT_RESTRICTED_BIT	2
#define ALT_COLOR_BIT		4
#define REVEAL_CONFLICT_BIT	8

#define CONFINED_TO_ROUTES	(atc_flags&FIXED_ROUTES_BIT)
#define ALT_RESTRICTED		(atc_flags&ALT_RESTRICTED_BIT)
#define ALT_COLOR		(atc_flags&ALT_COLOR_BIT)
#define REVEAL_CONFLICT		(atc_flags&REVEAL_CONFLICT_BIT)


#define RADIANS_TO_DEGREES( ang )	( (ang) * 45.0 / atan(1.0) )
#define DEGREES_TO_RADIANS( ang )	( (ang) * atan(1.0) / 45.0 )


extern void display_message(char *,long x,long y,int color);

/* get_elapsed_time was gxElapsedTime */
extern long get_elapsed_time(long);


#define LEFT_BUTTON	1	/* a dummy placeholder... */


/* conflict.c */
extern void delay(int msec);
extern int boundedi(int val,int lower,int upper);
extern atc_type idist(int x1,int y1,int x2, int y2);
extern int color_of(Flight_Path *);
extern int hl_color_of(Flight_Path *);
extern int selected_flight_path(Flight_Path *);
extern void apply_to_planes(QSP_ARG_DECL  void (*func)(QSP_ARG_DECL  Flight_Path *));
extern COMMAND_FUNC( setup_object_coords );
extern COMMAND_FUNC( do_render );

/* planes.c */
extern COMMAND_FUNC( plane_menu );
extern List * plane_list(SINGLE_QSP_ARG_DECL);
extern COMMAND_FUNC( clear_all_planes );

/* draw.c */
extern void set_tag_visibility(int yn);
extern void atc_line(int x1, int y1, int x2, int y2);
extern void render_visible(QSP_ARG_DECL  Point *);
extern void get_disparity(Flight_Path *);
extern void refresh_objects(void);
extern void refresh_object(QSP_ARG_DECL  Flight_Path *);
extern void set_disparities(int d[4][2] );
extern void highlight_object(QSP_ARG_DECL  Flight_Path *,int);
extern void draw_crossing(Pair *,int);
extern void draw_model_tag(Model_Obj *,int);
#ifdef HAVE_X11
extern void set_screen_viewer(QSP_ARG_DECL  Viewer *);
extern void set_stereo_viewers(QSP_ARG_DECL  Viewer *,Viewer *);
extern int reset_screen(QSP_ARG_DECL  Viewer *);
#endif /* HAVE_X11 */
extern void draw_projections(QSP_ARG_DECL  STD_ARG time);
extern void draw_intersection(Flight_Path *,Flight_Path *,Point *,int);
extern void draw_foil(Flight_Path *,Flight_Path *,double);
extern void draw_scan_path(Data_Obj *);
extern void clear_atc_screen(void);
extern COMMAND_FUNC( draw_region );
extern void draw_legend(QSP_ARG_DECL  int);
extern void draw_tag(Flight_Path *,int color);
extern void draw_tag_loc(Flight_Path *,int color);
extern void draw_tag_line(Flight_Path *,int color);
extern void draw_plane(Flight_Path *,int color);
extern void draw_plane_loc(Flight_Path *,int color);
extern void update_wedge(Flight_Path *);
extern void update_tag(Flight_Path *);
extern void update_tag_line(QSP_ARG_DECL  Flight_Path *);
extern void find_best_tag_angle(QSP_ARG_DECL  Flight_Path *);
extern boolean inbounds(Point *ptp);
extern boolean onscreen(Point *ptp);
extern void /*re*/draw_objects(SINGLE_QSP_ARG_DECL);
extern void recompute_coords(QSP_ARG_DECL  Flight_Path *fpp);
extern void redraw_planes(SINGLE_QSP_ARG_DECL);
extern COMMAND_FUNC( init_graphics );
extern COMMAND_FUNC( draw_fixation_screen );
extern COMMAND_FUNC( erase_fixation_screen );
extern void draw_fixation_indicator(Point *);
extern void erase_fixation_indicator(Point *);
extern int inside_tag(Flight_Path *fpp,Point *ptp);
extern void init_center(void);
extern void erase_object(QSP_ARG_DECL  Flight_Path *);
extern void draw_object(QSP_ARG_DECL  Flight_Path *);
extern void redraw_object(Flight_Path *);
extern void draw_conflict(QSP_ARG_DECL  Pair *);


/* event.c */

extern void log_atc_event(Atc_Event_Code code,Flight_Path *fpp);
extern void select_posn_from_list(QSP_ARG_DECL  Data_Obj *);
extern void enable_event_logging(u_long);
extern void replay_event_log(SINGLE_QSP_ARG_DECL);
extern void dump_event_log(void);
extern COMMAND_FUNC( move_tags );
extern COMMAND_FUNC( select_planes );
extern COMMAND_FUNC( wait_for_click );
extern COMMAND_FUNC( select_any_object );
extern COMMAND_FUNC( present_fixation );
extern COMMAND_FUNC( sgl_click_fix );
extern COMMAND_FUNC( click_fixations );
extern COMMAND_FUNC( step_model );
extern void set_exit_on_right_click(int);

/* model.c */

extern int paths_are_in_conflict(Flight_Path *fpp1, Flight_Path *fpp2);
extern Point *crossing_point(Flight_Path *fpp1, Flight_Path *fpp2);
extern COMMAND_FUNC( model_menu );
extern void make_fixation(QSP_ARG_DECL  Point *);
extern void plan_saccade(QSP_ARG_DECL  Point *ptp);
extern int conflict_found(SINGLE_QSP_ARG_DECL);
extern void report_model_rt(SINGLE_QSP_ARG_DECL);
extern void end_simulation_run(QSP_ARG_DECL  char *reason);
extern boolean good_fixation_vector(QSP_ARG_DECL  Data_Obj *);
extern boolean good_scanpath_vector(QSP_ARG_DECL  Data_Obj *);


/* geom.c */
extern atc_type dist(Point *p1p, Point *p2p);

/* pathgen.c */
extern COMMAND_FUNC( pathgen_menu );

#endif /* ! CONFLICT_H */
