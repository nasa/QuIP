#include "quip_config.h"

#ifdef HAVE_X11

char VersionId_atc_draw[] = "%Z% $RCSfile: draw.c,v $ ver: $Revision: 1.45 $ $Date: 2009/08/20 00:19:46 $";

#include <math.h>
#include <string.h>

#include "conflict.h"
#include "draw.h"
#include "cmaps.h"	/* setcolor() */
#include "viewer.h"	/* xp_line() etc */
#include "xsupp.h"	/* show_viewer() */
#include "data_obj.h"

#define atc_select(color)	xp_select(color)

static void populate_flight_info(SINGLE_QSP_ARG_DECL);

int display_width;
static const char *reverse_text( const char *s );

static int using_tags=1;

/*miscellaneous options */

#define smart_tag_loc   true
/* choose data tag locations to reduce overwriting*/


/* this should move from the center to the left edge...
 * Original value was 30, but probably presumed a different font.
 * We really should get the value dynamically from the font.
 * With the 5x8 font, we use a value of 20.
 *
 * We assume here 5x8 font, say 6 pixels per char...
 * 30 should then be 5 chars, which should be enough!?
 *
 * 4/11/01 jbm:  Now we are registering these images to the scene
 * camera images of the pc screen, and the tags are too narrow.
 * Eric R. reports that the font on the PC is 8x8ROM, the closest
 * thing we can find on unix is linux8x8.  When we make this
 * change with the offset at 20, the tags are too far to the
 * right, so we will put the default back to 30.
 * We may want to make this a parameter.
 *
 * The font setting is actually done in the viewmenu menu,
 * so currently our default defn doesn't do anything,
 * but we should make it so BUG.
 */

#ifdef FOOBAR
#define DEFAULT_TAG_X_OFFSET		20	/* for font 5x8 */
#define DEFAULT_ATC_FONT		5x8
#endif /* FOOBAR */

#define DEFAULT_TAG_X_OFFSET		30	/* for font linux8x8 */

int tag_x_offset = DEFAULT_TAG_X_OFFSET;

static atc_type projection_time;

#define DEFAULT_TAG_DISTANCE	50.0
static atc_type default_tag_dist=DEFAULT_TAG_DISTANCE;

static Viewer *screen=NO_VIEWER;
Viewer *eye_screen[2];

int is_stereo=0;
static int disparities[4][N_EYES]={{0,0},{0,0},{0,0},{0,0}};		/* 4 elevations, 2 eyes */
int this_disparity=0;
int current_eye=LEFT_EYE;

#define left_screen	eye_screen[0]
#define right_screen	eye_screen[1]

#define SETUP_LEFT(fpp)					\
		select_viewer(QSP_ARG  left_screen);		\
		xp_setup(left_screen);			\
		current_eye=LEFT_EYE;			\
		if( fpp != NULL ) get_disparity(fpp)


#define SETUP_RIGHT(fpp)				\
							\
		select_viewer(QSP_ARG  right_screen);		\
		xp_setup(right_screen);			\
		current_eye=RIGHT_EYE;			\
		if( fpp != NULL ) get_disparity(fpp);

#define DEFAULT_MAX_X	1023
#define DEFAULT_MAX_Y	767

Point max_point={{DEFAULT_MAX_X,DEFAULT_MAX_Y}};


#define MOVE_PT(ptp)	atc_move((int) (ptp->p_x),(int) (ptp->p_y))


static int data_tag_spacing=DEFAULT_DATA_TAG_SPACING;
static int tag_radius = DEFAULT_TAG_X_OFFSET;

#define TAG_X_MIN	tag_x_offset
#define TAG_Y_MIN	floor(2.5 * data_tag_spacing + 0.5)
#define TAG_X_MAX	(max_x - TAG_X_MIN)
#define TAG_Y_MAX	(max_y - TAG_Y_MIN)


Point _center;
Point *center_p=NO_POINT;

#define INSURE_CENTER if( center_p == NO_POINT ) init_center();

int default_plane_color=DEFAULT_PLANE_COLOR;	/* changed in model.c */
int default_plane_highlight_color=DEFAULT_HL_COLOR;	/* changed in model.c */
static int tag_color;
static int tag_line_color;


static void atc_move(int x, int y)
{
	if( ! mirror_reversed ){
		xp_move(x,y);
	} else {
		xp_move( FLIP_X(x), y );
	}
}

static void atc_cont(int x, int y)
{
	if( ! mirror_reversed ){
		xp_cont(x,y);
	} else {
		xp_cont(FLIP_X(x),y);
	}
}

void atc_line(int x1, int y1, int x2, int y2)
{
	if( ! mirror_reversed ){
		xp_line(x1,y1,x2,y2);
	} else {
		xp_line(FLIP_X(x1),y1,FLIP_X(x2),y2);
	}
}


/* ---------------------------------------------------------------- */
COMMAND_FUNC( draw_region )
{  /* draws circle around the region to be monitored */

	float arg,arginc;
	int i,j,region_radius;
	Point _p1,_p2;
	int n_tick_samples;	/* for drawing thick tick marks */
	float tp;		/* tick posn:  sideways drawing position within the tick */
	float tick_delta;

	/* rc = grsetlinestyle(grLSolid, 1); */

	INSURE_CENTER

	this_disparity=0;

	atc_select(CIRCLE_COLOR);	/* select pen color */
	MOVE_PT(center_p);
	xp_circle(max_y / 2);

	/* This next section draws the inner circle with tick marks.
	 * This was not part of the original program, but was added
	 * by trebor to provide additional landmarks for scene
	 * camera registration.
	 */
	region_radius = floor( 0.5 + ((max_y+1)*0.75)/2 );
	xp_circle( region_radius );

	/* now draw the ticks */
	arg=0;

#define N_TICKS		8
#define TICK_WIDTH	3	/* the code below assumes this is odd */
#define TICK_LEN	20

	arginc = (8*atan(1))/N_TICKS;

	/* We have to step slowly when filling in the ticks in the diagonal directions,
	 * or else we get holes.
	 *
	 * sqrt(2) should have worked here, but we can't be bothered to make it efficient.
	 */
	n_tick_samples = floor(TICK_WIDTH * 2.0 * sqrt(2.0)) + 1;
	tick_delta = TICK_WIDTH/(n_tick_samples-1.0);

	for(i=0;i<N_TICKS;i++){
		tp = -floor(TICK_WIDTH/2.0);
		for(j=0;j<n_tick_samples;j++){
			_p1.p_x = center_p->p_x + floor( 0.5 + (region_radius) * cos(arg) - tp * sin(arg) );
			_p1.p_y = center_p->p_y + floor( 0.5 + (region_radius) * sin(arg) + tp * cos(arg) );
			_p2.p_x = center_p->p_x + floor( 0.5 + (TICK_LEN+region_radius) * cos(arg) - tp * sin(arg) );
			_p2.p_y = center_p->p_y + floor( 0.5 + (TICK_LEN+region_radius) * sin(arg) + tp * cos(arg) );
			DRAW_LINE(&_p1,&_p2);
			tp += tick_delta;
		}

		arg += arginc;
	}
}



/* ---------------------------------------------------------------- */
#ifdef FOOBAR
/* BUG need a way to specify the routes */
static void DrawPaths()
{
	/* draws dashed lines indicating routes */
	atc_select(BROWN);
	/*rc := atc_select(GRAY);*/
	/* rc = grsetlinestyle(grLwidedot, 1); */
	atc_line(423, 756, 582, 7);
	atc_line(236, 116, 787, 648);
	atc_line(128, 402, 741, 77);
	atc_line(144, 493, 859, 543);
	/* rc = grsetlinestyle(grLSolid, 1); */
}
#endif /* FOOBAR */

static void atc_text( const char *str )
{
	if( ! mirror_reversed ){
		left_justify();
		xp_text(str);
	} else {
		right_justify();
		xp_text( reverse_text(str) );
	}
}

/* ---------------------------------------------------------------- */
static void _draw_legend(int trl_number)
{
	/* draws misc info in corners of the display */
	char istr1[256], istr2[256];
	char STR1[256];

#ifdef FOOBAR
	if (CONFINED_TO_ROUTES)
		DrawPaths();
#endif /* FOOBAR */

	/* rc = grsettextstyle(grtxt8X16, gropaque); */
	atc_select(BROWN);

	atc_move(820, 10);
	sprintf(istr1, "%3d", trl_number);
	sprintf(STR1, "Trial  =%s", istr1);
	atc_text(STR1);
	atc_move(820, 40);
	atc_select(BROWN);
	if (CONFINED_TO_ROUTES) {
		atc_text("ROUTE RESTRICTIONS");
	} else {
		atc_text("NO ROUTE RESTRICTIONS");
	}
	atc_move(820, 70);
	if (ALT_RESTRICTED) {
		atc_text("ALTITUDE RESTRICTIONS");
		atc_move(820, 650);
		sprintf(istr1, "%12ld", (long)EAST_1);
		sprintf(istr2, "%12ld", (long)EAST_2);
		sprintf(STR1, "EASTBOUND:  %s, %s", istr1, istr2);
		atc_text(STR1);
		atc_move(820, 680);
		sprintf(istr1, "%12ld", (long)WEST_1);
		sprintf(istr2, "%12ld", (long)WEST_2);
		sprintf(STR1, "WESTBOUND:  %s, %s", istr1, istr2);
		atc_text(STR1);
	} else {
		atc_text("NO ALTITUDE RESTRICTIONS");
	}

	/* Draw mileage scale */

#define SCALE_Y			650
#define MAJOR_TICK_LEN		7
#define MINOR_TICK_LEN		5
#define SCALE_TEXT_OFFSET	26
#define SCALE_LEGEND_OFFSET	60
#define X_POSN_0		10
#define X_POSN_25		( X_POSN_0 + ROUND(25 * PIXELS_PER_KNOT) )
#define X_POSN_50		( X_POSN_0 + ROUND(50 * PIXELS_PER_KNOT) )

	atc_select(BROWN);
	atc_line(X_POSN_0,  SCALE_Y, (int) X_POSN_50, SCALE_Y);
	atc_line(X_POSN_0,  SCALE_Y-MAJOR_TICK_LEN, X_POSN_0,  SCALE_Y+MAJOR_TICK_LEN);
	atc_line((int) X_POSN_50, SCALE_Y-MAJOR_TICK_LEN, (int) X_POSN_50, SCALE_Y+MAJOR_TICK_LEN);
	atc_line((int) X_POSN_25, SCALE_Y-MINOR_TICK_LEN, (int) X_POSN_25, SCALE_Y+MINOR_TICK_LEN);

	center_text();

	atc_move(X_POSN_0,  SCALE_Y + SCALE_TEXT_OFFSET); atc_text("0");
	atc_move((int) X_POSN_50, SCALE_Y + SCALE_TEXT_OFFSET); atc_text("50");
	atc_move((int) X_POSN_25, SCALE_Y + SCALE_TEXT_OFFSET); atc_text("25");
	atc_move((int) X_POSN_25, SCALE_Y + SCALE_LEGEND_OFFSET); atc_text("Nautical Miles");

}

void draw_legend(QSP_ARG_DECL  int trl_number)
{
	if( is_stereo ){
		SETUP_LEFT(NULL);
		_draw_legend(trl_number);
		SETUP_RIGHT(NULL);
	}
		_draw_legend(trl_number);
}


/* Need to call this again if the screen size is changed.
 */

void init_center()
{
#ifdef CAUTIOUS
	if( center_p != NO_POINT ){
		NWARN("CAUTIOUS:  init_center:  center is already initialized!?");
		return;
	}
#endif /* CAUTIOUS */

advise("init_center() setting coords of screen center");
	_center.p_x = ROUND( max_x/2 );
	_center.p_y = ROUND( max_y/2 );
	center_p = &_center;
}


/* ---------------------------------------------------------------- */
boolean inbounds(Point *ptp)
{
	atc_type distance_from_center;

	/* determines if object is within region*/

	INSURE_CENTER

	distance_from_center = DIST(center_p,ptp);

	if( distance_from_center <= (max_y / 2) )
		return(1);
	else
		return(0);
}


/* ---------------------------------------------------------------- */
boolean onscreen(Point *ptp)
{
	if( ptp->p_x < 0 || ptp->p_y < 0 ) return(0);

	if( ptp->p_x > max_x ) return(0);
	if( ptp->p_x > max_y ) return(0);

	return(1);
}



/* ---------------------------------------------------------------- */
void update_tag(Flight_Path *fpp)
{
	atc_type angle,c,s;

	/* updates location of tag */

	angle = DEGREES_TO_RADIANS( fpp->fp_tag_angle );

	c=cos(angle);
	s=sin(angle);
	fpp->fp_tag_loc.p_x = floor(
		fpp->fp_vertices[0].p_x + c * fpp->fp_tag_dist + 0.5);
	fpp->fp_tag_loc.p_y = floor(
		fpp->fp_vertices[0].p_y + s * fpp->fp_tag_dist + 0.5);

	/* BOUND makes sure the integer variable (first parameter)
	 * falls within the lower (second parameter) and upper (third param)
	 * bounds */

	/* This is needed to avoid drawing the tag off the screen */

	BOUND(fpp->fp_tag_loc.p_x, TAG_X_MIN, TAG_X_MAX );
	BOUND(fpp->fp_tag_loc.p_y, TAG_Y_MIN, TAG_Y_MAX );
}



/* ---------------------------------------------------------------- */

/* BUG the tag line is terminated a fixed distance from the tag center.
 * It would be better to use the tag_line_angle, determine which edge of
 * the tag box is intersected, and then do the math to find the endpoint...
 */

void update_tag_line(QSP_ARG_DECL  Flight_Path *fpp)
{
	atc_type angle, c, s;
	atc_type tag_box_angle;
	static atc_type pi=0.0;

	if( pi == 0.0 ) pi=4*atan(1.0);

	angle = DEGREES_TO_RADIANS( fpp->fp_tag_angle );
	c = cos(angle);
	s = sin(angle);

	fpp->fp_tag_line = fpp->fp_vertices[0];	/* the center of the plane */

	tag_box_angle = atan2(TAG_Y_MIN,TAG_X_MIN);

	/* We assume that the tag angle in degrees is between 0 and 360... */

	if( angle >= 0 && angle <= tag_box_angle ){
		tag_radius = TAG_X_MIN / fabs(c);
	} else if( angle > tag_box_angle && angle <= pi-tag_box_angle ){
		tag_radius = TAG_Y_MIN / fabs(s);
	} else if( angle > pi-tag_box_angle && angle <= pi+tag_box_angle ){
		tag_radius = TAG_X_MIN / fabs(c);
	} else if( angle > pi+tag_box_angle && angle <= 2*pi-tag_box_angle ){
		tag_radius = TAG_Y_MIN / fabs(s);
	} else if( angle > 2*pi-tag_box_angle && angle <= 2*pi ){
		tag_radius = TAG_X_MIN / fabs(c);
	} else {
		/* BUG this happens fairly frequently, not sure why...
		 * we're commenting out this printf so that it runs
		 * quietly under netscape.
		 */

		/*
		sprintf(ERROR_STRING,"bad tag angle %g radians (%g degrees)",angle,fpp->fp_tag_angle);
		NWARN(ERROR_STRING);
		*/
		tag_radius = TAG_X_MIN;
	}

	fpp->fp_tag_line.p_x += floor(c * (fpp->fp_tag_dist - tag_radius) + 0.5);
	fpp->fp_tag_line.p_y += floor(s * (fpp->fp_tag_dist - tag_radius) + 0.5);

	fpp->fp_tag_line.p_x = BOUND( fpp->fp_tag_line.p_x, 0, max_x );
	fpp->fp_tag_line.p_y = BOUND( fpp->fp_tag_line.p_y, 0, max_y );
}



/* ---------------------------------------------------------------- */

/* the following constants determine the shape of the wedges
 * used to represent the planes
 */

#define WEDGE_DIST_1  18   /*Distance from plane to tip of wedge*/
#define WEDGE_DIST_2  9   /*Distance from plane to base of wedge*/
#define WEDGE_ANGLE     90   /*Exterior angle at trailing edge of wedge*/

void update_wedge(Flight_Path *fpp)
{
	atc_type c,s;
	atc_type angle;

	/* calculates vertices of wedge representing plane */

	fpp->fp_vertices[0].p_x = floor(fpp->fp_plane_loc.p_x + 0.5);
	fpp->fp_vertices[0].p_y = floor(fpp->fp_plane_loc.p_y + 0.5);

	angle = DEGREES_TO_RADIANS( 180 + fpp->fp_theta + WEDGE_ANGLE / 2.0 );
	c = cos(angle); s = sin(angle);

	fpp->fp_vertices[1].p_x = floor( fpp->fp_vertices[0].p_x + WEDGE_DIST_2 * c + 0.5);
	fpp->fp_vertices[1].p_y = floor( fpp->fp_vertices[0].p_y + WEDGE_DIST_2 * s + 0.5);

	angle = DEGREES_TO_RADIANS( fpp->fp_theta );
	c = cos(angle); s = sin(angle);

	fpp->fp_vertices[2].p_x = floor( fpp->fp_vertices[0].p_x + WEDGE_DIST_1 * c + 0.5);
	fpp->fp_vertices[2].p_y = floor( fpp->fp_vertices[0].p_y + WEDGE_DIST_1 * s + 0.5);

	angle = DEGREES_TO_RADIANS( 180 + fpp->fp_theta - WEDGE_ANGLE / 2.0 );
	c = cos(angle); s = sin(angle);

	fpp->fp_vertices[3].p_x = floor( fpp->fp_vertices[0].p_x + WEDGE_DIST_2 * c + 0.5);
	fpp->fp_vertices[3].p_y = floor( fpp->fp_vertices[0].p_y + WEDGE_DIST_2 * s + 0.5);

	BOUND(fpp->fp_vertices[0].p_x, 0, max_x);
	BOUND(fpp->fp_vertices[1].p_x, 0, max_x);
	BOUND(fpp->fp_vertices[2].p_x, 0, max_x);
	BOUND(fpp->fp_vertices[3].p_x, 0, max_x);

	BOUND(fpp->fp_vertices[0].p_y, 0, max_y);
	BOUND(fpp->fp_vertices[1].p_y, 0, max_y);
	BOUND(fpp->fp_vertices[2].p_y, 0, max_y);
	BOUND(fpp->fp_vertices[3].p_y, 0, max_y);
}



/* ---------------------------------------------------------------- */
void find_best_tag_angle(QSP_ARG_DECL  Flight_Path *fpp)
{
	/* finds for plane k, the location of the tag that causes the least
		overwriting of other screen objects */
	long q;
	atc_type curr_dist, max_dist, save_dist, best_angle, best_distance;

	max_dist = 0.0;
	best_angle = fpp->fp_theta + 90;
	best_distance = 45.0;
	for (q = 0; q <= 3; q++) {
		Node *np;

		if (CONFINED_TO_ROUTES) {   /* route condition */
			fpp->fp_tag_angle = fpp->fp_theta + 90;
			fpp->fp_tag_dist = q * 20.0 + 45;
		} else {  /* free flight condition */
			fpp->fp_tag_angle = fpp->fp_theta + ((float)q + 1) * 90;
			while (fpp->fp_tag_angle >= 360)
				fpp->fp_tag_angle -= 360;
			fpp->fp_tag_dist = default_tag_dist;
		}
		update_tag(fpp);
		save_dist = 100000.0;

		np = plane_list(SINGLE_QSP_ARG)->l_head;
		while(np!=NO_NODE){
			Flight_Path *fpp2;

			fpp2 = (Flight_Path *)(np->n_data);

			if ( fpp2 != fpp ) {  /* find closest plane/tag for this prospective data tag loc*/
				curr_dist = DIST(&fpp->fp_tag_loc,&fpp2->fp_tag_loc );
				if (curr_dist < save_dist)
					save_dist = curr_dist;
				curr_dist = DIST(&fpp->fp_tag_loc, &fpp2->fp_plane_loc);
				if (curr_dist > SUFFICIENT_OBJECT_SEPARATION)
					curr_dist = SUFFICIENT_OBJECT_SEPARATION;
				if (curr_dist < save_dist)
					save_dist = curr_dist;
			}
			np=np->n_next;
		}
		if ((save_dist > max_dist) && inbounds(&fpp->fp_tag_loc) ) {
			/* is this data_tag location best so far? */
			best_angle = fpp->fp_tag_angle;
			best_distance = fpp->fp_tag_dist;
			max_dist = save_dist;
		}
	}
	fpp->fp_tag_angle = best_angle;
	fpp->fp_tag_dist = best_distance;
}


static const char *reverse_text( const char *s )
{
	static char rev_str[LLEN];		/* watch out for buffer overflow BUG */
	int i,j;

	i=strlen(s);
	j=0;
	while(i)
		rev_str[j++] = s[--i];
	rev_str[j]=0;
	return(rev_str);
}

/* ------------------------------------------------------------------ */

#define TAG_OFFSET( fpp, dx , dy ) 						\
	{									\
	atc_move((int) ((fpp)->fp_tag_loc.p_x - floor((dx))),\
	(int) ((fpp)->fp_tag_loc.p_y - floor((dy))));	\
	}									\

#define LINE_HEIGHT	4		/* linux8x8 font is 8 pixels high? */




#define DRAW_TAG_NAME(fpp)							\
	{									\
	TAG_OFFSET( fpp, tag_x_offset-this_disparity, (1.5 * data_tag_spacing + 0.5 - LINE_HEIGHT) );	\
	atc_text( fpp->fp_name );				\
	}


#define DRAW_TAG_ALT(fpp)							\
										\
	{									\
	TAG_OFFSET( fpp, tag_x_offset-this_disparity, 0  - LINE_HEIGHT );					\
	sprintf(str, "%-4ld C", fpp->fp_altitude);				\
	atc_text(str);								\
	}

#define DRAW_TAG_SPEED(fpp)							\
										\
	{									\
	TAG_OFFSET( fpp, tag_x_offset-this_disparity, (-1.5 * data_tag_spacing + 0.5 - LINE_HEIGHT) );	\
	sprintf(str, "B737 %-4ld", (long)floor(fpp->fp_speed + 0.5));		\
	atc_text(str);								\
	}



void draw_tag(Flight_Path *fpp,int color)
{
	char str[256];

	atc_select(color);
	DRAW_TAG_NAME(fpp);
	DRAW_TAG_ALT(fpp);
	DRAW_TAG_SPEED(fpp);

	/* this is here for ebugging draw_tag_loc */
	/*draw_tag_loc(fpp,color); */

	/* CURRENTLY all planes are identified as B737 */
	/* this should be changed */
}

static void get_tag_corners(Point *ul_ptp,Point *lr_ptp,Flight_Path *fpp)
{
	/* We used to multiply the quantity inside floor() by correction_factor
	 * to insure that the tag offset was a constant number of screen units.
	 */

	ul_ptp->p_x  = fpp->fp_tag_loc.p_x - floor(tag_x_offset+0.5);
	lr_ptp->p_x = fpp->fp_tag_loc.p_x + floor(tag_x_offset+0.5);

	/* screen coords have y increasing down */
	ul_ptp->p_y = fpp->fp_tag_loc.p_y - floor(2.0 * data_tag_spacing + 0.5);
	lr_ptp->p_y = fpp->fp_tag_loc.p_y + floor(2.0 * data_tag_spacing + 0.5);
}

#define xleft	upper_left.p_x
#define xright	lower_right.p_x
#define ytop	upper_left.p_y
#define ybot	lower_right.p_y

#define DRAW_TAG_BOX						\
								\
	{							\
	atc_move(((int) (xleft))+this_disparity,(int) (ytop));					\
	atc_cont(((int) (xright))+this_disparity,(int) (ytop));					\
	atc_cont(((int) (xright))+this_disparity,(int) (ybot));					\
	atc_cont(((int) (xleft))+this_disparity, (int) (ybot));					\
	atc_cont(((int) (xleft))+this_disparity, (int) (ytop));					\
	}

void draw_tag_loc(Flight_Path *fpp,int color)
{
	Point upper_left,lower_right;

	get_tag_corners(&upper_left,&lower_right,fpp);

	atc_select(color);

	/* draw in the tag name - doesn't represent knowledge, but helps debug ! */
	DRAW_TAG_BOX;
	DRAW_TAG_NAME(fpp);
}

void draw_model_tag(Model_Obj *mop,int color)
{
	Point upper_left,lower_right;
	char str[256];


	get_tag_corners(&upper_left,&lower_right,mop->mo_fpp);

	atc_select(color);

	if( ! COMPLETE_TAG_INFO(mop) )
		DRAW_TAG_BOX;

	DRAW_TAG_NAME(mop->mo_fpp);

	if( SPEED_KNOWN(mop) )
		DRAW_TAG_SPEED(mop->mo_fpp);

	if( ALT_KNOWN(mop) )
		DRAW_TAG_ALT(mop->mo_fpp);
}

int inside_tag(Flight_Path *fpp,Point *ptp)
{
	Point upper_left,lower_right;

	get_tag_corners(&upper_left,&lower_right,fpp);

	if( ptp->p_x >= xleft && ptp->p_x <= xright &&
		ptp->p_y >= ytop && ptp->p_y <= ybot )
		return(1);
	else
		return(0);
}


#define CHECK_BOUNDS(string,fpp)							\
										\
	if( !inbounds(&fpp->fp_vertices[0]) ){					\
		sprintf(DEFAULT_ERROR_STRING,						\
"%s:  plane %s at %g %g (%g %g) is not in bounds!?",string,(fpp)->fp_name,			\
(fpp)->fp_plane_loc.p_x,(fpp)->fp_plane_loc.p_y,(fpp)->fp_vertices[0].p_x,(fpp)->fp_vertices[0].p_y);	\
		NWARN(DEFAULT_ERROR_STRING);						\
		return;								\
	}

/* ------------------------------------------------------------------ */
void draw_plane(Flight_Path *fpp,int color)
{
	/* do inbounds check here */
	CHECK_BOUNDS("draw_plane",fpp);

	/*var gpoints : array [0..9] of integer;    */
	atc_select(color);
	DRAW_LINE( &fpp->fp_vertices[0], &fpp->fp_vertices[1] );
	DRAW_LINE( &fpp->fp_vertices[1], &fpp->fp_vertices[2] );
	DRAW_LINE( &fpp->fp_vertices[2], &fpp->fp_vertices[3] );
	DRAW_LINE( &fpp->fp_vertices[3], &fpp->fp_vertices[0] );
}

#define PLANE_LOC_RADIUS	((WEDGE_DIST_1+WEDGE_DIST_2)/2)

void draw_plane_loc(Flight_Path *fpp,int color)
{
	Point tpt,*ptp=&tpt;
	CHECK_BOUNDS("draw_plane_loc",fpp);

	atc_select(color);
	tpt = fpp->fp_vertices[0] ;
	if( is_stereo )
		tpt.p_x += this_disparity;
	MOVE_PT(ptp);
	xp_circle(PLANE_LOC_RADIUS);
}

int selected_flight_path(Flight_Path *fpp)
{
	if( fpp == selection_1 ) return(1);
	if( fpp == selection_2 ) return(1);
	if( fpp == selected_tag ) return(1);
	return(0);
}

void get_disparity(Flight_Path *fpp)
{
	switch( fpp->fp_altitude ){
		case ALTITUDE_1: this_disparity = disparities[0][current_eye]; break;
		case ALTITUDE_2: this_disparity = disparities[1][current_eye]; break;
		case ALTITUDE_3: this_disparity = disparities[2][current_eye]; break;
		case ALTITUDE_4: this_disparity = disparities[3][current_eye]; break;
#ifdef CAUTIOUS
		default: NERROR1("CAUTIOUS:  get_disparity:  unexpected altitude!?"); break;
#endif /* CAUTIOUS */
	}
}

#define DRAW_IT									\
										\
	if( using_tags ){							\
		GET_TAG_COLOR							\
		draw_tag(fpp,tag_color);					\
		DRAW_TAG_LINE(fpp,tag_line_color);				\
	}									\
	draw_plane(fpp,color_of(fpp));

#define ERASE_IT								\
										\
	if( using_tags ){							\
		draw_tag(fpp,BLACK);						\
	}									\
	draw_plane(fpp,BLACK);


#define GET_TAG_COLOR								\
	if( !selected_flight_path(fpp) ){					\
		if( REVEAL_CONFLICT && IS_IN_CONFLICT(fpp) ){			\
			tag_color=WHITE;					\
			tag_line_color=WHITE;					\
		} else {							\
			tag_color=DATA_TAG_COLOR;				\
			tag_line_color=TAG_LINE_COLOR;				\
		}								\
	} else {								\
		tag_color=DATA_TAG_COLOR;					\
		tag_line_color=TAG_LINE_COLOR;					\
	}


void set_tag_visibility(int yn)
{
	using_tags = yn;
}

void draw_object(QSP_ARG_DECL  Flight_Path *fpp)
{

	/* draw plane, tag, tag line for plane j */
	/* BUG we don't seem to use the color arg? */
	if( !inbounds(&fpp->fp_vertices[0]) )
		return;

	if( is_stereo ){
		SETUP_LEFT(fpp);
		DRAW_IT
		SETUP_RIGHT(fpp);
	}
		DRAW_IT
}

void erase_object(QSP_ARG_DECL  Flight_Path *fpp)
{
	if( !inbounds(&fpp->fp_vertices[0]) )
		return;

	if( is_stereo ){
		SETUP_LEFT(fpp);
		ERASE_IT
		SETUP_RIGHT(fpp);
	}
		ERASE_IT
}

static void erase_location(Flight_Path *fpp)
{
	/*Point upper_left,lower_right; */

	draw_plane_loc(fpp,BLACK);
	/*get_tag_corners(&upper_left,&lower_right,fpp); */
	/*DRAW_TAG_BOX; */
	DRAW_TAG_LINE(fpp,BLACK);
}

static Flight_Path *last_highlighted=NO_FLIGHT_PATH;

void highlight_object(QSP_ARG_DECL  Flight_Path *fpp,int tag_flag)
{
	if( last_highlighted != NO_FLIGHT_PATH ){
		/*re*/draw_object(QSP_ARG  last_highlighted);
		last_highlighted = NO_FLIGHT_PATH;
	}

	/* draw plane, tag, tag line for plane j */
	/* BUG we don't seem to use the color arg? */
	/* Could be a plane or a tag ... how do we know which?? */

	if( !inbounds(&fpp->fp_vertices[0]) )
		return;

	if( tag_flag ){
		draw_tag(fpp,WHITE);
	} else {
		draw_plane(fpp,hl_color_of(fpp));
	}
	last_highlighted = fpp;
}


#ifdef UNUSED
/* ---------------------------------------------------------------- */
static void draw_pair_projection(Flight_Path *fpp1, Flight_Path *fpp2, atc_type t)
{
	/* show the paths of planes a and b up until time t */
	Point final;

	final.p_x = ROUND(fpp1->fp_plane_loc.p_x + t * fpp1->fp_vel.v_x);
	final.p_y = ROUND(fpp1->fp_plane_loc.p_y + t * fpp1->fp_vel.v_y);
	/* what color? */
	DRAW_LINE( &fpp1->fp_vertices[0], &final );
	atc_select(CIRCLE_COLOR);
	xp_circle(4);

	final.p_x = ROUND(fpp2->fp_plane_loc.p_x + t * fpp2->fp_vel.v_x);
	final.p_y = ROUND(fpp2->fp_plane_loc.p_y + t * fpp2->fp_vel.v_y);
	/* what color? */
	DRAW_LINE( &fpp2->fp_vertices[0], &final );
	atc_select(CIRCLE_COLOR);
	xp_circle(4);
}
#endif /* UNUSED */

/* #ifdef PSEUDOCOLOR */

/* Why use this cumbersome method instead of just a list of setcolor() calls?
 * This way, the compiler will warn us if we forget to define one of the
 * color offsets enumerated in conflict.h
 */

static void define_color(Color_Offset offset)
{
	switch(offset){
		case BLACK_OFFSET:	setcolor(BLACK,		  0,  0,  0);	break;
		case WHITE_OFFSET:	setcolor(WHITE,		255,255,255);	break;
		case GRAY_OFFSET:	setcolor(GRAY,		140,140,140);	break;
		case DARK_GRAY_OFFSET:	setcolor(DARK_GRAY,	 60, 60, 60);	break;
		case MAGENTA_OFFSET:	setcolor(MAGENTA,	200,  0,255);	break;
		case BLUE_OFFSET:	setcolor(BLUE,		100, 50,240);	break;
		case YELLOW_OFFSET:	setcolor(YELLOW,	240,210,  0);	break;
		case RED_OFFSET:	setcolor(RED,		220,  0, 90);	break;
		case GREEN_OFFSET:	setcolor(GREEN,		0,  190, 30);	break;
		case LIGHT_CYAN_OFFSET:	setcolor(LIGHT_CYAN,	100,200,255);	break;
		case BROWN_OFFSET:	setcolor(BROWN,		150, 50,  0);	break;

		case BRT_MAGENTA_OFFSET:setcolor(BRT_MAGENTA,	255, 50,255);	break;
		case BRT_BLUE_OFFSET:	setcolor(BRT_BLUE,	120, 80,255);	break;
		case BRT_YELLOW_OFFSET:	setcolor(BRT_YELLOW,	255,255, 50);	break;
		case BRT_RED_OFFSET:	setcolor(BRT_RED,	255, 50,140);	break;
		case BRT_GREEN_OFFSET:	setcolor(BRT_GREEN,	50, 255, 80);	break;

		/* never called, but this makes the compiler shut up */
		case N_DEFINED_COLORS	:					break;
	}
}

static void init_colors(void)
{
	int i;

	for(i=0;i<N_DEFINED_COLORS;i++)
		define_color( (Color_Offset)i);
}

/* #endif */ /* PSEUDOCOLOR */



void /*re*/draw_objects(SINGLE_QSP_ARG_DECL)
{
	apply_to_planes(QSP_ARG  draw_object);
	populate_flight_info(SINGLE_QSP_ARG);
}

void redraw_plane(QSP_ARG_DECL  Flight_Path *fpp)
{
	draw_plane(fpp, color_of(fpp) );
}

void redraw_planes(SINGLE_QSP_ARG_DECL)
{
	apply_to_planes(QSP_ARG  redraw_plane);
	populate_flight_info(SINGLE_QSP_ARG);
}


void recompute_coords(QSP_ARG_DECL  Flight_Path *fpp)
{
	update_wedge(fpp);
	update_tag(fpp);
	update_tag_line(QSP_ARG  fpp);
}

#define FIX_RAD	10	/* fixation radius (size of fixation cross) */

static void draw_fixation_cross(int color)
{
	INSURE_CENTER
	atc_select(color);
	atc_line((int)(center_p->p_x-FIX_RAD),(int)(center_p->p_y),(int)(center_p->p_x+FIX_RAD),(int)(center_p->p_y));
	atc_line((int)(center_p->p_x),(int)(center_p->p_y-FIX_RAD),(int)(center_p->p_x),(int)(center_p->p_y+FIX_RAD));
}

COMMAND_FUNC( draw_fixation_screen )
{
	if( is_stereo ){
		SETUP_LEFT(NULL);
		xp_erase();
		draw_fixation_cross(WHITE);

		SETUP_RIGHT(NULL);
		xp_erase();
		draw_fixation_cross(WHITE);
	} else {
		xp_erase();
		draw_fixation_cross(WHITE);
	}
}

void clear_atc_screen()
{
	xp_erase();
}

COMMAND_FUNC( erase_fixation_screen )
{
	draw_fixation_cross(BLACK);
}


void draw_fixation_indicator(Point *ptp)
{
	xp_fmove(ptp->p_x,ptp->p_y);
	atc_select(WHITE);
	xp_circle(heading_ecc_thresh);
}

void erase_fixation_indicator(Point *ptp)
{
	xp_fmove(ptp->p_x,ptp->p_y);
	atc_select(BLACK);
	xp_circle(heading_ecc_thresh);
}

static void draw_projection(QSP_ARG_DECL  Flight_Path *fpp)
{
	Point endpt;
	Vector delta;

	delta = fpp->fp_vel;
	SCALE_VECTOR(&delta,projection_time);
	endpt.p_x = fpp->fp_plane_loc.p_x + delta.v_x;
	endpt.p_y = fpp->fp_plane_loc.p_y + delta.v_y;
	DRAW_LINE(&fpp->fp_plane_loc,&endpt);
}

void draw_projections(QSP_ARG_DECL  STD_ARG time)
{
	projection_time = time;
	apply_to_planes(QSP_ARG  draw_projection);
}

void draw_conflict(QSP_ARG_DECL  Pair *prp)
{
	atc_select(WHITE);
	draw_object(QSP_ARG  prp->pr_mop1->mo_fpp);
	draw_object(QSP_ARG  prp->pr_mop2->mo_fpp);
	projection_time = prp->pr_conflict_time;
	draw_projection(QSP_ARG  prp->pr_mop1->mo_fpp);
	draw_projection(QSP_ARG  prp->pr_mop2->mo_fpp);
}

static void set_center(Viewer *vp)
{
	max_x = vp->vw_width;
	max_y = vp->vw_height;

	_center.p_x = ROUND( max_x/2 );
	_center.p_y = ROUND( max_y/2 );
	center_p = &_center;

	display_width = vp->vw_width;
}

static void release_old_screens()
{
	if( screen!=NO_VIEWER ){
		sprintf(DEFAULT_ERROR_STRING,"Ceasing use of previous screen viewer %s",screen->vw_name);
		advise(DEFAULT_ERROR_STRING);
		screen=NO_VIEWER;
	}
	if( left_screen!=NO_VIEWER ){
		sprintf(DEFAULT_ERROR_STRING,"Ceasing use of previous left screen viewer %s",left_screen->vw_name);
		advise(DEFAULT_ERROR_STRING);
		left_screen=NO_VIEWER;
	}
	if( right_screen!=NO_VIEWER ){
		sprintf(DEFAULT_ERROR_STRING,"Ceasing use of previous right screen viewer %s",right_screen->vw_name);
		advise(DEFAULT_ERROR_STRING);
		right_screen=NO_VIEWER;
	}
}

void set_screen_viewer(QSP_ARG_DECL  Viewer *vp)
{
	release_old_screens();

	screen = vp;

	/* We used to keep the coordinates of the planes and things in fixed 1024x768 screen
	 * coords...  eventually we might like to have these be in knots instead...
	 *
	 * But when we resize the window, we could like the tags to be readable...
	 * but the tag offset and tag line spacing are given in plotting units.
	 * So we compute the correction factors here:
	 *
	 * When stereo was added to the experiment, it seemed like a good idea
	 * to stick with the screen units.
	 */

	set_center(vp);
	reset_screen(QSP_ARG  vp);
	is_stereo=0;

	reset_screen(QSP_ARG  vp);
	default_cmap(&vp->vw_top);
	init_colors();

}	/* end set_screen_viewer */

void set_stereo_viewers(QSP_ARG_DECL  Viewer *l_vp, Viewer *r_vp)
{
	release_old_screens();
	set_center(l_vp);

	if( l_vp->vw_height != r_vp->vw_height || l_vp->vw_width != r_vp->vw_width ){
		sprintf(ERROR_STRING,"Stereo viewers %s (%dx%d) and %s (%dx%d) should have the same size",
				l_vp->vw_name,l_vp->vw_height,l_vp->vw_width,
				r_vp->vw_name,r_vp->vw_height,r_vp->vw_width);
		NWARN(ERROR_STRING);
	}

	select_viewer(QSP_ARG  l_vp);
	init_colors();
	select_viewer(QSP_ARG  r_vp);
	init_colors();

	reset_screen(QSP_ARG  l_vp);
	reset_screen(QSP_ARG  r_vp);
	is_stereo=1;
	left_screen = l_vp;
	right_screen = r_vp;

	/*enable_masked_events(l_vp,PointerMotionMask); */
	/*enable_masked_events(r_vp,PointerMotionMask); */
}

int reset_screen(QSP_ARG_DECL  Viewer *vp)
{
#ifdef CAUTIOUS
	if( vp == NO_VIEWER ){
		WARN("CAUTIOUS:  reset_screen:  null screen specified");
		return(-1);
	}
#endif /* CAUTIOUS */

	show_viewer(QSP_ARG  vp);	/* default state is to be shown */
	if( cursor_controls_visibility )
		enable_masked_events(vp,PointerMotionMask);
	select_viewer(QSP_ARG  vp);

	/* The default plotting space has 0,0 in the lower left corner...
	 * but the original PC pascal atc program assumed 0,0 in the upper
	 * left, so we explicitly set the coordinate space here...
	 *
	 * We still need to call xp_setup(), because it also makes this
	 * viewer the plot viewer.
	 */

	xp_setup(vp);		/* set default plotting space */

	/* this is the default used by xp_setup():
	 * xp_space(0,0,max_x,max_y);
	 */

	xp_space(0,(int)(max_y),(int)(max_x),0);

	xp_bgselect(BLACK);
	if( ! atc_overlay_mode ){
		xp_erase();
	}

	/* make sure the colormap is good */

	return(0);

}	/* end reset_screen */


#define SCREEN_WIDTH	1024
#define SCREEN_HEIGHT	768

COMMAND_FUNC( init_graphics )
{
	Viewer *vp;

	if( screen != NO_VIEWER ){
		sprintf(ERROR_STRING,"init_graphics:  viewer %s is already in use as screen!?",screen->vw_name);
		WARN(ERROR_STRING);
		return;
	}

	vp = viewer_init(QSP_ARG  "default_screen",SCREEN_WIDTH,SCREEN_HEIGHT,VIEW_ADJUSTER);
	if( vp == NO_VIEWER ){
		WARN("unable to create default screen viewer");
		return;
	}

	set_screen_viewer(QSP_ARG  vp);
} /* end init_graphics() */

/* should be called draw_crossing, but we already used that name
 * for the pair version...
 */

void draw_intersection(Flight_Path *fpp1, Flight_Path *fpp2, Point *ptp, int color)
{

	atc_select(color);
	DRAW_LINE(&fpp1->fp_plane_loc,ptp);
	DRAW_LINE(&fpp2->fp_plane_loc,ptp);
	xp_fmove(ptp->p_x,ptp->p_y);
	atc_select(color);
	xp_circle(5.0);
}


/* This gets called only when we know that this pair has an interesting crossing...
 */

void draw_crossing(Pair *prp,int color)
{
#ifdef CAUTIOUS
	if( color == BLACK && !CROSSING_DRAWN(prp) ){
		sprintf(DEFAULT_ERROR_STRING,
			"CAUTIOUS:  crossing for pair %s erased more than once!?",
			prp->pr_name);
		NWARN(DEFAULT_ERROR_STRING);
	}
	/* don't squawk about redrawing, we do this after we erase
	 * another crossing involving one of these aricraft.
	 */
#endif /* CAUTIOUS */

	draw_intersection(prp->pr_mop1->mo_fpp,prp->pr_mop2->mo_fpp,&prp->pr_crossing_pt,color);

	if( color == BLACK )
		prp->pr_flags &= ~DRAWN_BIT;
	else
		prp->pr_flags |= DRAWN_BIT;
}

int color_of(Flight_Path *fpp)
{
	long Result;

	/* If the REVEAL flag is set, we want to display the conflict
	 * pair in a special color - but how do we know which planes
	 * are the conflict pair?
	 */
	if( IS_IN_CONFLICT(fpp) && REVEAL_CONFLICT )
		return REVEAL_PLANE_COLOR;

	if (fpp == selection_2 || fpp == selection_1)
		return SELECTED_PLANE_COLOR;
	if (fpp == selected_tag)
		return DRAG_PLANE_COLOR;
	if (!ALT_COLOR)
		return default_plane_color;

	switch( fpp->fp_altitude ){
		case ALTITUDE_1: Result = PLANECOLOR_1; break;
		case ALTITUDE_2: Result = PLANECOLOR_2; break;
		case ALTITUDE_3: Result = PLANECOLOR_3; break;
		case ALTITUDE_4: Result = PLANECOLOR_4; break;
#ifdef CAUTIOUS
		default:
			Result=0;	// quiet compiler
			NERROR1("CAUTIOUS:  unexpected altitude!?");
			break;
#endif /* CAUTIOUS */
	}
	return Result;
}

int hl_color_of(Flight_Path *fpp)
{
	long Result;

	if (!ALT_COLOR)
		return default_plane_highlight_color;

	switch( fpp->fp_altitude ){
		case ALTITUDE_1: Result = HL_PLANECOLOR_1; break;
		case ALTITUDE_2: Result = HL_PLANECOLOR_2; break;
		case ALTITUDE_3: Result = HL_PLANECOLOR_3; break;
		case ALTITUDE_4: Result = HL_PLANECOLOR_4; break;
#ifdef CAUTIOUS
		default:
			Result=0;	// quiet compiler
			NERROR1("CAUTIOUS:  unexpected altitude!?");
			break;
#endif /* CAUTIOUS */
	}
	return Result;
}


/* this is like plotmenu's xyplot, but we check that the points are in bounds,
 * and don't draw if they are not...
 */

void draw_scan_path(Data_Obj *dp)
{
	Point *ptp;
	int new_down,pen_down;
	unsigned n;

	if( sizeof(atc_type) != sizeof(float) )
		NERROR1("draw_scan_path:  atc_type expected to be float!?");

	ptp = (Point *) (dp->dt_data);
	n=dp->dt_cols;

	atc_select(WHITE);

	pen_down=0;

#define N_BEFORE	2
#define N_AFTER		2

	while(n--){
		unsigned i;

		/* we plot this point if it and the next N_AHEAD are inbounds... */
		new_down=1;
		for(i=0;i<=N_BEFORE;i++){
			if( n>(i-1) && !inbounds(ptp+i) )
				new_down=0;
		}
		for(i=1;i<=N_AFTER;i++){
			if( (n+i)<(dp->dt_cols) && !inbounds(ptp-i) )
				new_down=0;
		}
		if( new_down ){		/* draw this point */
			if( pen_down )
				xp_fcont(ptp->p_x,ptp->p_y);
			else {
				xp_fmove(ptp->p_x,ptp->p_y);
				pen_down=1;
			}
		}
		pen_down = new_down;
		ptp++;
	}
}

void draw_foil(Flight_Path *fpp1,Flight_Path *fpp2,double time)
{
	Point end_pt;
	Vector dv;

	atc_select(WHITE);

	dv = fpp1->fp_vel;
	SCALE_VECTOR(&dv,time);	/* what are the units of time??? */
	end_pt = fpp1->fp_plane_loc;
	DISPLACE_POINT(&end_pt,&dv);
	DRAW_LINE(&fpp1->fp_plane_loc,&end_pt);
	xp_fmove(end_pt.p_x,end_pt.p_y);
	xp_circle(5.0);

	dv = fpp2->fp_vel;
	SCALE_VECTOR(&dv,time);	/* what are the units of time??? */
	end_pt = fpp2->fp_plane_loc;
	DISPLACE_POINT(&end_pt,&dv);
	DRAW_LINE(&fpp2->fp_plane_loc,&end_pt);
	xp_fmove(end_pt.p_x,end_pt.p_y);
	xp_circle(5.0);
}

/* make a data object with the geometric info for each plane */

static void populate_flight_info(SINGLE_QSP_ARG_DECL)
{
	Dimension_Set dimset;
	unsigned int i=0,  uNumPlanes=0;
	Data_Obj* pdobj;
	Node *np;

	uNumPlanes = eltcount( plane_list(SINGLE_QSP_ARG) );

	if( uNumPlanes == 0 ) return;

	np = plane_list(SINGLE_QSP_ARG)->l_head;

	dimset.ds_dimension[0] = 7;
	dimset.ds_dimension[1] = uNumPlanes;
	for (i=2; i < N_DIMENSIONS; i++)
		dimset.ds_dimension[i] = 1;

	/* Make sure no previous version exists */
	pdobj=DOBJ_OF(FLIGHT_INFO_SYSVAR_NAME);
	if( pdobj != NO_OBJ )
		delvec(QSP_ARG  pdobj);

	/* Create new variable */
	pdobj = make_dobj(QSP_ARG  FLIGHT_INFO_SYSVAR_NAME, &dimset, PREC_SP);

	np = plane_list(SINGLE_QSP_ARG)->l_head;
	for(i=0; np!=NO_NODE; i++){
		Flight_Path *fpp;
		float* data;

		fpp = (Flight_Path *) (np->n_data);
		data = (float *) (pdobj->dt_data);
		data += pdobj->dt_pinc * i;
		/* Component 0 = X-coord of plane */
		*data = fpp->fp_plane_loc.p_x;      data += pdobj->dt_cinc;
		/* Component 1 = Y-coord of plane */
		*data = fpp->fp_plane_loc.p_y;      data += pdobj->dt_cinc;
		/* Component 2 = X-coord of tag loc  */
		*data = fpp->fp_tag_loc.p_x;        data += pdobj->dt_cinc;
		/* Component 3 = Y-coord of tag loc */
		*data = fpp->fp_tag_loc.p_y;        data += pdobj->dt_cinc;
		/* Component 4 = Theta */
		*data = fpp->fp_theta;              data += pdobj->dt_cinc;
		/* Component 5 = Speed */
		*data = fpp->fp_speed;              data += pdobj->dt_cinc;
		/* Component 6 = Altitude */
		*data = (float) fpp->fp_altitude;   data += pdobj->dt_cinc;

		np = np->n_next;
	}
	/* now i is the number of planes that have been added */
#ifdef CAUTIOUS
	if( i !=  uNumPlanes ){
		sprintf(ERROR_STRING,
"CAUTIOUS:  populate_flight_info:  i (%d) does not match uNumPlanes (%d) !?!?",
			i,uNumPlanes);
		NWARN(ERROR_STRING);
	}
#endif /* CAUTIOUS */
}

/* Only render the things which are under the cursor */

/* static */ Point *the_ptp;

static void draw_visible(QSP_ARG_DECL  Flight_Path *fpp)
{
	if( !inbounds(&fpp->fp_vertices[0]) )
		return;

	if( DIST(the_ptp,&fpp->fp_plane_loc) < conceal_radius ){
		if( ( ! ICON_IS_VISIBLE(fpp) ) || ICON_NOT_RENDERED(fpp) ){
			if( is_stereo ){
				SETUP_LEFT(fpp);
				erase_location(fpp);
				draw_plane(fpp,color_of(fpp));
				SETUP_RIGHT(fpp);
				erase_location(fpp);
				draw_plane(fpp,color_of(fpp));
			} else {
				erase_location(fpp);
				draw_plane(fpp,color_of(fpp));
			}
			fpp->fp_flags |= FP_ICON_VISIBLE;
			fpp->fp_flags |= FP_ICON_RENDERED;
			log_atc_event(ICON_REVEALED,fpp);
		}
	} else {
		if( ICON_IS_VISIBLE(fpp) || ICON_NOT_RENDERED(fpp) ){
			if( is_stereo ){
				SETUP_LEFT(fpp);
				draw_plane(fpp,BLACK);
				draw_plane_loc(fpp,color_of(fpp));

				SETUP_RIGHT(fpp);
				draw_plane(fpp,BLACK);
				draw_plane_loc(fpp,color_of(fpp));
			} else {
				draw_plane(fpp,BLACK);
				draw_plane_loc(fpp,color_of(fpp));
			}
			fpp->fp_flags &= ~FP_ICON_VISIBLE;
			fpp->fp_flags |= FP_ICON_RENDERED;
			log_atc_event(ICON_CONCEALED,fpp);
		}
	}
	if( using_tags ){
		if( inside_tag(fpp,the_ptp) ){
			if( (! TAG_IS_VISIBLE(fpp)) || TAG_NOT_RENDERED(fpp) ){
				if( is_stereo ){
					SETUP_LEFT(fpp);
					draw_tag_loc(fpp,BLACK);
					draw_tag(fpp,DATA_TAG_COLOR);
					SETUP_RIGHT(fpp);
					draw_tag_loc(fpp,BLACK);
					draw_tag(fpp,DATA_TAG_COLOR);
				} else {
					draw_tag_loc(fpp,BLACK);
					draw_tag(fpp,DATA_TAG_COLOR);
				}
				fpp->fp_flags |= FP_TAG_VISIBLE;
				fpp->fp_flags |= FP_TAG_RENDERED;
				log_atc_event(TAG_REVEALED,fpp);
			}
		} else {
			if( TAG_IS_VISIBLE(fpp) || TAG_NOT_RENDERED(fpp) ){
				if( is_stereo ){
					SETUP_LEFT(fpp);
					draw_tag(fpp,BLACK);
					draw_tag_loc(fpp,DATA_TAG_COLOR);
					SETUP_RIGHT(fpp);
					draw_tag(fpp,BLACK);
					draw_tag_loc(fpp,DATA_TAG_COLOR);
				} else {
					draw_tag(fpp,BLACK);
					draw_tag_loc(fpp,DATA_TAG_COLOR);
				}
				fpp->fp_flags &= ~FP_TAG_VISIBLE;
				fpp->fp_flags |= FP_TAG_RENDERED;
				log_atc_event(TAG_CONCEALED,fpp);
			}
		}
		if( is_stereo ){
			SETUP_LEFT(fpp);
			DRAW_TAG_LINE(fpp,TAG_LINE_COLOR);
			SETUP_RIGHT(fpp);
			DRAW_TAG_LINE(fpp,TAG_LINE_COLOR);
		} else {
			DRAW_TAG_LINE(fpp,TAG_LINE_COLOR);
		}
	}
}

void render_visible(QSP_ARG_DECL  Point *fix_ptp)
{
	the_ptp = fix_ptp;
	apply_to_planes(QSP_ARG  draw_visible);
	if( is_stereo ){
		SETUP_LEFT(NULL);
		draw_region(NULL_QSP);
		SETUP_RIGHT(NULL);
	}
		draw_region(NULL_QSP);
}

COMMAND_FUNC( do_render )
{
	if( is_stereo ){
		if( reset_screen(QSP_ARG  left_screen) < 0 ) return;
		if( reset_screen(QSP_ARG  right_screen) < 0 ) return;
	} else {
		if( reset_screen(QSP_ARG  screen) < 0 ) return;
	}

	/* we used to (conditionally) erase here, but this is handled in reset_screen */

	if( cursor_controls_visibility ){
		Point far_pt={{5000,5000}};
		render_visible(QSP_ARG  &far_pt);
	} else {
		draw_objects(SINGLE_QSP_ARG);
		if( is_stereo ){
			SETUP_LEFT(NULL);
			draw_region(SINGLE_QSP_ARG);
			SETUP_RIGHT(NULL);
			draw_region(SINGLE_QSP_ARG);
		} else {
			draw_region(SINGLE_QSP_ARG);
		}
	}
}

void refresh_object(QSP_ARG_DECL Flight_Path *fpp )
{
	if( is_stereo ){
		SETUP_LEFT(fpp);
		draw_object(QSP_ARG  fpp);
		SETUP_RIGHT(fpp);
		draw_object(QSP_ARG  fpp);
	} else {
		draw_object(QSP_ARG  fpp);
	}
}

void set_disparities(int d[4][2])
{
	int i,j;

	if( ! is_stereo ){
		NWARN("Setting disparities although stereo viewers have not been specified");
	}
	for(i=0;i<4;i++)
		for(j=0;j<2;j++)
			disparities[i][j]=d[i][j];
}

#endif /* HAVE_X11 */


