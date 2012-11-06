#include "quip_config.h"

#ifdef HAVE_X11

char VersionId_atc_pathgen[] = QUIP_VERSION_STRING;

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "conflict.h"
#include "draw.h"
#include "rn.h"

/* This program generates flight paths of N planes, such that one and
 * only one pair of planes will conclude at some future time T.  Other
 * constraints are also imposed (see below). It generates several such
 * scenarios (enough for one session of an experiment) and outputs
 * all the information into a file.
 *
 * Here are the factors that we manipulate:
 *
 *	# planes
 *	conflict time
 *	# x-y foils (# of planes that fly close by in X-Y space -- ignoring altitude)
 *	angle of incidence
 *
 * Here is the algorithm used to generate the flight paths:
 *
 * FREE FLIGHT:
 *
 * 1) First a conflict pair is generated.
 *	a) choose random heading for plane 1
 *	b) choose random heading for plane 2 with the following constraints:
 *		1. it must produce planes that are both eastbound or westbound
 *			when altitude restrictions are in force.
 *		2. the angle of incidence must be greater than the minimum (this
 *			 minimum is to ensure that planes are not in conflict at the
 *			 beginning of a trials)
 *		3. the angle is chosen to be above or below the median, depending
 *			 on the trial type.
 *	c) the conflict location is chosen at random, somewhere within
 *		 the central region, but with the restriction that the planes
 *		 are on-screen at trial onset.  (note: if planes initially fail
 *		 this criterion, we try reversing the directions of both planes
 *		 by 180 degrees).
 *	d) Given this information (angles and conflict point and conflict
 *		 time) we can compute the plane's flight plath.
 * 2) Next we generate the N-2 distractor planes in the folling manner:
 *	a) a position in the region and a time are randomly chosen
 *		(currently, the time is 100 - rand (120) which means that it is
 *		possible to have a value of -20)
 *	b) the planes must be on screen at trial onset or the plane is rejected
 *	b) the plane must never come within a certain minimum distance of
 *		other planes (currently, 10 knots) or it is rejected.
 *	c) Before each trial we decide that the trial will have more or less
 *		than the median # of XY foils.  (Half the time we go high, half
 *		of the time we go low.)  This is to ensure that we don't
 *		get really bad luck and generate a bunch of trials all with more
 *		(or less) than the median.  So, basically, we reject the set of
 *		distractors if it fails to fall on the desired side of the median).
 *		NOTE: it might have been more efficient to generate a bunch of
 *		trials and then go back and select the ones with the right
 *		properties -- i.e. # xy foils above (below) the median.
 *		NOTE: the medians were established through simulation of thousands
 *		of flight path generations (see procedure medians). We don't do
 *		this every time we run the program.
 *
 * FIXED ROUTE:
 *
 * 1) First a conflict pair is generated.
 *	a) a random conflict point is selected, with the restriction
 *		that it be either above or below the median (depending
 *		upon the experimental condition) in terms of the angle
 *		of incidence.  NOTE: all of the angles of incidence for
 *		the possible conflict points are calculated in advance,
 *		in the procedure called SETUPPATHS.
 *		ALSO NOTE: when there are no altitude restrictions, we
 *		treat each physical route crossing as if it were 2 separate
 *		conflict points. This is because in this condition there
 *		are two possible angles of incidence (e.g., 40 and 140).
 *	 b) a direction along the intersectin paths is chosen for each plane
 *	 c) Given the conflict time for that trial (and the plane speeds)
 *		we can then determine the entire flight path of the conflict
 *		planes
 *	 d) if these planes are not on-screen at trial onset, we try
 *		reversing the directions of the two planes.  Because of the
 *		way the sector was set up, this is guarranteed to provide
 *		a solution.
 * 2) Next we generate the N-2 distractor planes in the folling manner:
 *	NOTE: distractor planes are chosen so that there are at least 2
 *	planes at every altitude, and at least 1 plane on every
 *	path.
 *	 NOTE: The entire process below is repeated until the # XY foils is
 *	 above or below the median (depending on the experimental
 *	 condition).
 *
 *	 a) randomly choose a path and altitude.
 *	 b) randomly choose a location on the path and the
 *		time at which the plane is to be at that location
 *		[rand(100)-20]
 *	 c) see if the plane is on-screen at trial onset;
 *		if not then reverse its direction and try again.
 *	 d) also see if the plane conflicts with any other planes.
 *	 e) if the plane does conflict, or is still off-screen, then go back to (b).
 *	 f) If this process fails to find a valid flight path after
 *		100 tries, then we give up and essentially go
 *		back to (a) [hoping that we will have better luck on a
 *		different path].
 *		This never happened during the runs used to create the
 *		trials for CON1 and CON2 experiments.
 *
 *
 */

static int conflict_path_index;
static int reporting=0;
static char report_string[LLEN];

#define PREVIEW( fpp )	recompute_coords(QSP_ARG  fpp);				\
			if( inbounds(&fpp->fp_plane_loc) )		\
				draw_plane(fpp,color_of(fpp))

#define CPREVIEW( fpp )								\
	recompute_coords(QSP_ARG  fpp);							\
	if( inbounds(&fpp->fp_plane_loc) ) draw_plane(fpp,WHITE)


/* this goes down to 80% of max speed:  500-400
 */

static int speed_var = 20;

#define RANDOM_SPEED	( MAX_SPEED * ((100.0 - rn(speed_var)) / 100))

#define ANY_PATH	(-1L)
#define ANY_ALT	(-1L)

/*miscellaneous options */

/* BUG shoudl go somewhre else ... */
#define REGION_RADIUS   ((long)floor((max_y + 1) * 0.75 / 2 + 0.5))
#define MIN_SEP_KNOTS   10
#define MIN_SEPARATION  (MIN_SEP_KNOTS * PIXELS_PER_KNOT)
#define HOURS_PER_UPDATE  ( UPDATE_INTERVAL / SECONDS_PER_HOUR )

/* East and West traffic travel at different altitude levels */

#define Init_Tag_Angle  90

/* a "foil" is any pair of of planes that comes closer than this in x-y, ignoring z. */
#define FOIL_SEPARATION  (12 * PIXELS_PER_KNOT)

#ifdef FOOBAR

#define UpdateInterval  4000   /*msec*/
/* sector_width, ave_speed, and transit_time parameters are used
 * only to determine how fast our simulated planes should travel */

#define screen_width_knots  (200 / 0.75)   /* in nautical miles */



#define REGION_RADIUS   ((long)floor((max_y + 1) * 0.75 / 2 + 0.5))
/* REGION_RADIUS is radius of region to be monitored in pixels */


#define pathwidth_knots  0

#define pathwidth       ((long)floor(pathwidth_knots * PIXELS_PER_KNOT + 0.5))
/* halfwidth of channel (in pixels)*/

#define max_paths       4
#define max_con_points  20

#define conflict_time_1  4.0
/* in minutes */
/* 4 + random(2) */
#define conflict_time_2  6.0   /* 6 + random(2) */

#define num_ang_bins    2
#define num_xy_bins     2

#endif /* FOOBAR */

/* The next 2 constants are the earliest and latest points at which
 * the controller must worry about conflicts...they define a window
 * of time that should much greater thant he window of time
 * during which conflicts actually occur during the experiment.
 * Currently, the window starts at display onset and ends at
 * twice the time at which conflict might occur.
 */
#define MIN_CONFLICT_TIME	0   /*# updates*/
#define MAX_CONFLICT_TIME   	(1800.0 / UPDATE_INTERVAL)

#define MAX_PLANES	20
#define MIN_ANGLE       20.0

/* # updates */

typedef struct fixed_route {
	double fr_b;   /* y-intercept */
	double fr_m;   /* slope */
	double fr_theta;
	/* theta is determined by slope and the direction of motion */
} Fixed_Route;

typedef struct xy_type {
	Point		xy_pt;
	atc_type	xy_heading[2];
	atc_type	xy_angle;
} XY_Type;


static int numxys, num_fixed_routes;
#define MAX_PATHS	4
static Fixed_Route fixed_routes[MAX_PATHS];
static double conflict_time;   /* the update # when conflict occurs */
static int num_planes;

/* NOTE: these variables do not have to
 * be type integer (some can be byte).
 * But at present there is little to be
 * gained from using more efficient data
 * types. */
static double angle_of_incidence;

static long altitudes[N_ALTITUDES]={
	ALTITUDE_1,
	ALTITUDE_2,
	ALTITUDE_3,
	ALTITUDE_4
};

static long west[2]={ WEST_1, WEST_2 };
static long east[2]={ EAST_1, EAST_2 };

#define MAX_CON_POINTS	20
static XY_Type path_intersection[MAX_CON_POINTS];
static long num_con_points;

static boolean small_ang;	/* if true, then conflict angle is below median for cond */
				/* For routes, we know what all the possible angle are,
				 * so there are 5 above and 5 below (or whatever number)...
				 *
				 * also a head-on condition?
				 *
				 * For free flight, we don't allow the angle to be less than
				 * 20 (or 30), and it can't be more than 180...
				 *
				 * If no alt restrictions then the distributions of angles
				 * should be uniform, so the median should be (180-30)/2
				 *
				 */

/* Number of foils depends on number of planes, and whether alt restrictions
 * are in place...
 */

static boolean small_xys;	/* if true, then number of xy-foils is below the median.
				 */

static double angle_median;

static double xy_frac[3];	/* fractional part of the median */
static long xy_median[3];	/* 3 bins, one for each Nplanes... */

				/* These are not doubly substricted because for a given
				 * run alt restrictions etc did not vary, only nplanes.
				 */



/* local prototypes (for sgi compilation) */

static double norm_angle(STD_ARG);
static double angle_under_180(STD_ARG ang);
static boolean eastbound(STD_ARG ang);
static double getyintercept(double x1, double y1, double slope);
static double reverse_angle(STD_ARG ang);
static double transformtheta(double ang, long alt);
static long getaltitude(STD_ARG ang);
static double  getdisttime(void);
static void default_medians(void);
static void check_separation(SINGLE_QSP_ARG_DECL);
static void get_free_distractors(SINGLE_QSP_ARG_DECL);
static void calculate_numxys(SINGLE_QSP_ARG_DECL);
static void get_fixed_distractors(SINGLE_QSP_ARG_DECL);
static void draw_foils(SINGLE_QSP_ARG_DECL);
static void select_paths(SINGLE_QSP_ARG_DECL);
static void setup_fixed_routes(void);

static void set_speed_var( int v )
{
	if( v < 0 ){
		sprintf(DEFAULT_ERROR_STRING,"set_speed_var:  requested value (%d) should be non-negative",v);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( v > 20 ){
		sprintf(DEFAULT_ERROR_STRING,"set_speed_var:  requested value (%d) should be less than or equal to 20",v);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	speed_var = v;
}
		

/* ---------------------------------------------------------------- */
static double norm_angle(STD_ARG ang)
{
	while (ang > 360)
		ang -= 360;
	while (ang < 0)
		ang += 360;
	return ang;
}


/* ---------------------------------------------------------------- */
static double angle_under_180(STD_ARG ang)
{  /* gets supplement (?) if angle > 180 */
	ang = norm_angle(ang);
	if (ang > 180)
		ang = 360 - ang;
	return ang;
}


/* ---------------------------------------------------------------- */
static double reverse_angle(STD_ARG ang)
{
	return (norm_angle(ang + 180));
}


/* ---------------------------------------------------------------- */
static void GetAngleOfIncidence(Flight_Path *fpp1,Flight_Path *fpp2)
{
	angle_of_incidence = fpp2->fp_theta - fpp1->fp_theta;
	angle_of_incidence = angle_under_180(angle_of_incidence);
}


/* ---------------------------------------------------------------- */
static boolean eastbound(STD_ARG ang)
{
	
	return( (ang < 90 || ang >= 270) ? TRUE : FALSE );
}


/* ---------------------------------------------------------------- */
static long getaltitude(STD_ARG ang)
{
	if (ang == -1 || ! ALT_RESTRICTED ) {
		return (altitudes[rn(N_ALTITUDES-1)] );
	} else {
		if (eastbound(ang)) {
			return( east[rn(1)] );
		} else {
			return( west[rn(1)] );
		}
	}
}


/* ---------------------------------------------------------------- */
static boolean inmiddle(Point *ptp)
{
	/* determines if plane is in the controller region */
	/* different from just being on-screen...more restrictive */

	return( DIST( ptp, center_p ) <= REGION_RADIUS);
}


/* ---------------------------------------------------------------- */
static double Getdistance(Flight_Path *fpp1, Flight_Path *fpp2, double t)
{  /* calculates distance between two planes at time t */
	Point p1,p2;
	Vector v1,v2;

	v1=fpp1->fp_vel;
	v2=fpp2->fp_vel;

	SCALE_VECTOR(&v1,t);
	SCALE_VECTOR(&v2,t);

	p1=fpp1->fp_plane_loc;
	p2=fpp2->fp_plane_loc;

	DISPLACE_POINT(&p1,&v1);
	DISPLACE_POINT(&p2,&v2);

	return( DIST(&p1,&p2) );
}


/* ---------------------------------------------------------------- */
static double getyintercept(double x1, double y1, double slope)
{
	return (y1 - slope * x1);
}



/* ---------------------------------------------------------------- */
static void check_separation(SINGLE_QSP_ARG_DECL)
{  /* check to see that planes are not in conflict at trial onset*/
	List *lp;
	Node *np;
	Flight_Path *fpp1,*fpp2;

	lp=plane_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST || (np=lp->l_head)==NO_NODE ){
		WARN("check_separation:  no planes!?");
		return;
	}
	fpp1 = (Flight_Path *)(np->n_data);
	np = np->n_next;
	if( np==NO_NODE ){
		WARN("check_separation:  no second plane!?");
		return;
	}
	fpp2 = (Flight_Path *)(np->n_data);

	if( Getdistance(fpp1,fpp2, 0.0) > MIN_SEPARATION)
		return;
	WARN("Min_Separation violation at onset!!!");
}


/* ---------------------------------------------------------------- */
static void check_on_screen(Flight_Path *fpp1,Flight_Path *fpp2)
{
	if( inbounds(&fpp1->fp_plane_loc) && inbounds(&fpp2->fp_plane_loc) )
		return;
	NWARN("Planes off screen at onset!");
}


/* ---------------------------------------------------------------- */
static void get_region_crossing(Point *left_ptp, Point *right_ptp, Flight_Path *fpp)
{
	double a, b, c, temp;
	double time_1, time_2;
	Vector rv;

	a = MAG_SQ( &fpp->fp_vel );
	rv.v_x = fpp->fp_plane_loc.p_x - center_p->p_x;
	rv.v_y = fpp->fp_plane_loc.p_y - center_p->p_y;
	b = 2 * DOT_PROD(&rv,&fpp->fp_vel);
	c = DOT_PROD(&rv,&rv) - REGION_RADIUS*REGION_RADIUS;

	/* this looks like the quadratic formula!  but what are we solving???
	 * Is it the intersection of the flight path with the boundary circle??
	 *
	 * flight path locus is p = plane_loc + k vel
	 * circle locus is DIST(p,center) = RADIUS
	 *
	 * (locx + k velx - centerx) * (locx + k velx - centerx)
	 * 	+ (locy + k vely - centery) * (locy + k vely - centery)
	 *	= r^2;
	 *
	 * Solve for k:
	 *
	 * locx^2 + 2 k(velx (locx-centerx) - 2 locx centerx + centerx^2 + k^2 velx^2 
	 * + locy^2 + 2 k(vely (locy-centery) - 2 locy centery + centery^2 + k^2 vely^2 
	 */

	temp = b * b - 4 * a * c;

	if (temp >= 0) {
		Point p1,p2;
		Vector dv;

		temp = sqrt(temp);

		dv = fpp->fp_vel;
		p1 = p2 = fpp->fp_plane_loc;

		time_1 = (temp - b) / (2 * a);
		SCALE_VECTOR(&dv,time_1);
		DISPLACE_POINT(&p1,&dv);

		time_2 = (-b - temp) / (2 * a);
		SCALE_VECTOR(&dv,time_2);
		DISPLACE_POINT(&p2,&dv);

		if( p1.p_x < p2.p_x) {
			*left_ptp = p1;
			*right_ptp = p2;
		} else {
			*left_ptp = p2;
			*right_ptp = p1;
		}
		return;
	} else {  /* no possible solutions */
		NWARN("No solutions!?");	/* can this ever happen?? */
		NWARN("No region crossing possible for this plane!!!");
	}
}


/* ---------------------------------------------------------------- */
static Point find_path_intersection(Fixed_Route *frp1, Fixed_Route *frp2)
{
	Point ipt;

	/* find point at which paths i and j intersect */
	if( frp1->fr_m == frp2->fr_m) {
		ipt.p_x = -999.0;
		ipt.p_y = -999.0;
	} else {
		ipt.p_x = (frp1->fr_b - frp2->fr_b) /
			(frp2->fr_m - frp1->fr_m);

		ipt.p_y = ipt.p_x * frp2->fr_m + frp2->fr_b;
	}
	return(ipt);
}


static int conflict_cmp( const void *p1, const void *p2 )
{
	const XY_Type *xyp1, *xyp2;

	xyp1 = (const XY_Type *)(p1);
	xyp2 = (const XY_Type *)(p2);

	if( xyp1->xy_angle < xyp2->xy_angle ) return(1);
	else if( xyp1->xy_angle > xyp2->xy_angle ) return(1);
	else return(0);
}

#define GET_SLOPE(a)		(tan( DEGREES_TO_RADIANS((a)) ))

#define SETUP_ROUTE( frp, angle, x , y )			\
								\
	{							\
	(frp)->fr_theta = angle;				\
	(frp)->fr_m = GET_SLOPE((frp)->fr_theta);		\
	(frp)->fr_b = getyintercept(x , y, (frp)->fr_m);	\
	}


/* ---------------------------------------------------------------- */
static void setup_fixed_routes()
{
	XY_Type *savep, save[MAX_CON_POINTS];
	Fixed_Route *frp1,*frp2;
	int n,m;

	frp1 = &fixed_routes[0];
	SETUP_ROUTE( frp1, 102.0, 507.0, 360.0 ) frp1++;
	SETUP_ROUTE( frp1,  44.0, 520.0, 390.0 ) frp1++;
	SETUP_ROUTE( frp1, 152.0, 490.0, 210.0 ) frp1++;
	SETUP_ROUTE( frp1, 184.0, 490.0, 517.0 )

	/*for p := 1 to num_fixed_routes do report_region_crossing(p);*/

	num_con_points = 0;
	savep = &save[0];
	frp1 = &fixed_routes[0];
	n=num_fixed_routes;
	while(n--){
		m=n;
		frp2 = frp1+1;
		while(m--){
			Point ipt;
			ipt = find_path_intersection(frp1,frp2);
			if( inmiddle( &ipt ) ){
				savep->xy_pt = ipt;
				savep->xy_heading[0] = frp1->fr_theta;
				savep->xy_heading[1] = frp2->fr_theta;
				savep++;
				num_con_points++;
	/* now make sure the conflict is possible, given
	 * the altitude restrictions... if not then reverse
	 * one of the two flight paths.   Note that we make
	 * an arbitrary choice regarding what we call the path
	 * direction...later on, when we create planes, we will
	 * randomly choose between this direction along the path
	 * and the opposite direction */

				if ( ALT_RESTRICTED  & (eastbound(frp1->fr_theta) !=
					eastbound(frp2->fr_theta)))
					savep->xy_heading[1] = reverse_angle(frp2->fr_theta);
	/* if there are no altitude restrictions then
	 * this point produce two conflicts, e.g. a 40 and a 140. */
				if( ! ALT_RESTRICTED ) {
					num_con_points++;
					save->xy_pt = ipt;
					savep->xy_heading[0] = frp1->fr_theta;
					savep->xy_heading[1] = reverse_angle(frp2->fr_theta);
					savep++;
				}
			}
			frp2++;
		}
		frp1++;
	}

	/* go over all the conflicts and calculate the angle of incidence.*/
	savep = save;
	n=num_con_points;
	while(n--){
		savep->xy_angle =
			angle_under_180(savep->xy_heading[0] - savep->xy_heading[1]);
		savep++;
	}

	/* now rank order the conflicts in order of increasing angle of
	 * incidence....this code is probably not very efficient or
	 * easy to read...though it does seem to work. */

	qsort( save, num_con_points, sizeof(save[0]), conflict_cmp );

	if( ! ALT_RESTRICTED ) {  /* add the head-on/180 degree conditions to end of list*/
		num_con_points++;
		path_intersection[num_con_points - 1].xy_pt.p_x = 0.0;
		path_intersection[num_con_points - 1].xy_pt.p_y = 0.0;
		path_intersection[num_con_points - 1].xy_heading[0] = 0.0;
		path_intersection[num_con_points - 1].xy_heading[1] = 0.0;
		num_con_points++;
		path_intersection[num_con_points - 1].xy_pt.p_x = 0.0;
		path_intersection[num_con_points - 1].xy_pt.p_y = 0.0;
		path_intersection[num_con_points - 1].xy_heading[0] = 0.0;
		path_intersection[num_con_points - 1].xy_heading[1] = 0.0;
	}
	/* we somewhat arbitrarily decided that head-ons should
	 * occur 1/6th of the time  (I think...I'll have to check
	 * these numbers)
	 */

	/* we add 2 head-on conditions,
	 * so now head-ons will occur 2/12 times, or 1/6th
	 */
}

/* ---------------------------------------------------------------- */


/* ---------------------------------------------------------------- */
/* this function calculates the time of closest approach,
 * except that if that time is in the past it returns a value of 0
 * and if that time is way into the future it returns MAX_CONFLICT_TIME
 */

/* the equation below was derived by setting the first derivative
 * of the distance function equal to zero...
 */

static double GetApproachTime(Flight_Path *fpp1, Flight_Path *fpp2)
{
	double TimeofClosestApproach, TEMP, TEMP1;

	TEMP = fpp1->fp_vel.v_x - fpp2->fp_vel.v_x;
	TEMP1 = fpp1->fp_vel.v_y - fpp2->fp_vel.v_y;
	TimeofClosestApproach = TEMP * TEMP + TEMP1 * TEMP1;
	if( TimeofClosestApproach != 0)
		TimeofClosestApproach = ((fpp1->fp_vel.v_x - fpp2->fp_vel.v_x) *
			(fpp2->fp_plane_loc.p_x - fpp1->fp_plane_loc.p_x) +
			(fpp1->fp_vel.v_y - fpp2->fp_vel.v_y) *
			(fpp2->fp_plane_loc.p_y - fpp1->fp_plane_loc.p_y))
			/ TimeofClosestApproach;

	if( TimeofClosestApproach < MIN_CONFLICT_TIME)
		TimeofClosestApproach = MIN_CONFLICT_TIME;
	if( TimeofClosestApproach > MAX_CONFLICT_TIME )
		TimeofClosestApproach = MAX_CONFLICT_TIME ;
	return TimeofClosestApproach;
}


/* ---------------------------------------------------------------- */
static boolean WillConflict(Flight_Path *fpp1, Flight_Path *fpp2)
{
	/* planes are considered to be in conflict if they come within
	 * MIN_SEPARATION of each other. */
	double ApproachTime;

	/* This expression gives the time at which the two planes are closest.
	 * It works regardless of the speeds of the airplanes.
	 * Note that it gives an answer of '0' if the planes are on parallel
	 * paths. */
	if( fpp1->fp_altitude == fpp2->fp_altitude) {
		ApproachTime = GetApproachTime(fpp1, fpp2);
		if( Getdistance(fpp1, fpp2, ApproachTime) > MIN_SEPARATION)
			return FALSE;
		else
			return TRUE;
		/* OK. At the point of closest approach the planes are
		in violation, and this happens somewhere within the
		window of time during which the controllers are
		controlling.  The only question left is whether the
		two planes crash inside or outside of the controlled
		region.  For now, however, I won't worry about this
		problem because controllers might worry about crashing
		planes even if they aren't within the controlled
		region.*/
	} else
		return FALSE;
}  /* WillConflict */


/* ---------------------------------------------------------------- */
static double transformtheta(double ang, long alt)
{
	double t;

	/* reverse theta (if necessary) so plane flies in accordance
	 * with altitude restrictions
	 */
	t = norm_angle(ang);
	if ( ALT_RESTRICTED ) {
		if (eastbound(t) != (alt == east[0] || alt == east[1]))
			t = reverse_angle(t);
	}
	return t;
}

void apply_to_plane_pairs( QSP_ARG_DECL  void (*func)(Flight_Path *,Flight_Path *) )
{
	List *lp;
	Node *np1,*np2;
	Flight_Path *fpp1;
	Flight_Path *fpp2;

	lp=plane_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST ) return;

	np1=lp->l_head;
	while( np1 != NO_NODE ){
		np2 = np1->n_next;
		fpp1 = (Flight_Path *)(np1->n_data);
		while( np2 != NO_NODE ){
			fpp2 = (Flight_Path *)(np2->n_data);
			(*func)(fpp1,fpp2);
			np2=np2->n_next;
		}
		np1=np1->n_next;
	}
}

static void check_foils(Flight_Path *fpp1,Flight_Path *fpp2,int draw_flag)
{
	double approachtime, xydist;

	if( (! ALT_RESTRICTED ) | (eastbound(fpp1->fp_theta) == eastbound(fpp2->fp_theta))) {
		approachtime = GetApproachTime(fpp1, fpp2);
		if (approachtime > MAX_CONFLICT_TIME  / 2)   /* 15 minutes */
			approachtime = MAX_CONFLICT_TIME  / 2;
		xydist = Getdistance(fpp1, fpp2, approachtime);
		/* (approachtime > MAX_CONFLICT_TIME /5)
		 * and (approachtime > MAX_CONFLICT_TIME /3) and
		 */
		if (xydist < FOIL_SEPARATION){
			/* we might like to show the foils here... */
			numxys++;
			if( draw_flag )
				draw_foil(fpp1,fpp2,approachtime);
		}
	}
}

void count_foils(Flight_Path *fpp1,Flight_Path *fpp2)
{
	check_foils(fpp1,fpp2,0);
}

void show_foils(Flight_Path *fpp1,Flight_Path *fpp2)
{
	check_foils(fpp1,fpp2,1);
}

static void draw_foils(SINGLE_QSP_ARG_DECL)
{
	numxys = 0;
	apply_to_plane_pairs( QSP_ARG  show_foils );
}

/* ---------------------------------------------------------------- */
static void calculate_numxys(SINGLE_QSP_ARG_DECL)
{
	/* for this set of planes, determine how many unique pairs come
	 * with FOIL_SEPARATION of each other in xy, ignoring altitude
	 */

	numxys = 0;
	apply_to_plane_pairs( QSP_ARG  count_foils );
}


/* ---------------------------------------------------------------- */
static void calc_params(Flight_Path *fpp, double ct)
{
	Vector dv;

	fpp->fp_vel.v_x = cos(DEGREES_TO_RADIANS(fpp->fp_theta))
		* fpp->fp_speed * PIXELS_PER_KNOT * HOURS_PER_UPDATE;
	fpp->fp_vel.v_y = sin(DEGREES_TO_RADIANS(fpp->fp_theta))
		* fpp->fp_speed * PIXELS_PER_KNOT * HOURS_PER_UPDATE;

	fpp->fp_plane_loc = fpp->fp_conf_loc;
	dv = fpp->fp_vel;
	SCALE_VECTOR(&dv,-ct);
	DISPLACE_POINT(&fpp->fp_plane_loc,&dv);
}


/* ---------------------------------------------------------------- */
static void get_random_posn(Point *ptp)
{
	Vector dv;

	/* randomly choose a location in the controller region */
	do {
		*ptp = *center_p;
		dv.v_x = (2*drand48()-1) * REGION_RADIUS ;
		dv.v_y = (2*drand48()-1) * REGION_RADIUS ;
		DISPLACE_POINT(ptp,&dv);
	} while( !inmiddle(ptp) );
}


/* ---------------------------------------------------------------- */
static void generate_free_path(Flight_Path *fpp, double ct, long alt, double ang, Point *ptp )
{
	/* generates a random flight path under the restriction that the plane
	 * must pass through the controller region during a critical period
	 * of time.
	 */
	fpp->fp_altitude = alt;
	fpp->fp_speed = RANDOM_SPEED;
	fpp->fp_theta = transformtheta(ang, alt);
	fpp->fp_conf_loc = *ptp;
	fpp->fp_tag_angle = fpp->fp_theta - Init_Tag_Angle;
	calc_params(fpp, ct);
	/* what about plane_loc? */
}

/* generate a path for plane *fpp that passes through the center region
 * at time ct, and doesn't conflict with any plane within the
 * screen boundaries
 */

static void free_nonconflict(QSP_ARG_DECL  Flight_Path *fpp, double ct, long alt, int ang)
{
	boolean nonconflict;
	long alt1;
	Point pt1;
	List *lp;
	Node *np;

	alt1 = alt;
	do {
		get_random_posn(&pt1);
		if (alt == -1)
			alt1 = getaltitude(-1.0);
		generate_free_path(fpp, ct, alt1, ang, &pt1);
		/* check for conflict with other planes */
		lp=plane_list(SINGLE_QSP_ARG);
#ifdef CAUTIOUS
		if( lp==NO_LIST ) ERROR1("CAUTIOUS:  no planes!?!?");
#endif /* CAUTIOUS */
		np = lp->l_head;
		nonconflict = TRUE;
		while( np!=NO_NODE && nonconflict ){
			Flight_Path *fpp2;
			fpp2=(Flight_Path *)(np->n_data);
			if( fpp2 != fpp ){
				if( WillConflict(fpp, fpp2) )
					nonconflict = FALSE;
			}
			np=np->n_next;
		}
	} while( !( nonconflict && inbounds(&fpp->fp_plane_loc) ) );
}

#define PLANES_INBOUNDS( fpp1, fpp2 )						\
										\
	( inbounds(&(fpp1)->fp_plane_loc) && inbounds(&(fpp2)->fp_plane_loc) )

#define ABOUT_FACE(fpp)	(fpp)->fp_theta = norm_angle( (fpp)->fp_theta + 180 )

/* ---------------------------------------------------------------- */

/* Create a conflict (in free flight mode?)
 */

static void free_conflict(QSP_ARG_DECL  Flight_Path *fpp1, Flight_Path *fpp2, double ct, double ang1, double ang2)
{
	Point pt1;

	/* generate path for plane j that collides with plane i at time ct,
	 * somewhere within the central region
	 */
	/* currently, conflicting planes are generated first, so there is no
	 * need to see that these planes do not conflict with other planes
	 */
	long alt;

	fpp1->fp_speed = RANDOM_SPEED;
	fpp2->fp_speed = RANDOM_SPEED;

	selection_1 = fpp1;
	selection_2 = fpp2;	/* so they will redraw in WHITE */

	do {
		get_random_posn(&pt1);   /* choose random position within region */

		fpp1->fp_conf_loc = pt1;
		fpp2->fp_conf_loc = pt1;

		alt = getaltitude(-1.0);
		fpp1->fp_altitude = alt;
		fpp2->fp_altitude = alt;

		fpp1->fp_theta = transformtheta(ang1, alt);
		fpp2->fp_theta = transformtheta(ang2, alt);

		calc_params(fpp1, ct);
		calc_params(fpp2, ct);

/*
sprintf(DEFAULT_ERROR_STRING,"CPREV1 %s at %g %g,  %s at %g %g",
fpp1->fp_name,fpp1->fp_plane_loc.p_x,fpp1->fp_plane_loc.p_y,
fpp2->fp_name,fpp2->fp_plane_loc.p_x,fpp2->fp_plane_loc.p_y);
advise(DEFAULT_ERROR_STRING);
*/
		/*
		CPREVIEW(fpp1);
		CPREVIEW(fpp2);
		*/
		recompute_coords(QSP_ARG  fpp1);							\
		recompute_coords(QSP_ARG  fpp2);							\

		if( ! PLANES_INBOUNDS(fpp1,fpp2) ){

			/* try again, this time flipping both directions,
			 * keeping angle constant
			 */
			if( inbounds(&fpp1->fp_plane_loc) ) draw_plane(fpp1,BLACK);
			if( inbounds(&fpp2->fp_plane_loc) ) draw_plane(fpp2,BLACK);

			ABOUT_FACE(fpp1);
			ABOUT_FACE(fpp2);
			alt = getaltitude(fpp1->fp_theta);
			fpp1->fp_altitude = alt;
			fpp2->fp_altitude = alt;
			calc_params(fpp1, ct);
			calc_params(fpp2, ct);

/*
sprintf(DEFAULT_ERROR_STRING,"CPREV2 %s at %g %g,  %s at %g %g",
fpp1->fp_name,fpp1->fp_plane_loc.p_x,fpp1->fp_plane_loc.p_y,
fpp2->fp_name,fpp2->fp_plane_loc.p_x,fpp2->fp_plane_loc.p_y);
advise(DEFAULT_ERROR_STRING);
*/
			/*
			CPREVIEW(fpp1);
			CPREVIEW(fpp2);
			*/
			recompute_coords(QSP_ARG  fpp1);							\
			recompute_coords(QSP_ARG  fpp2);							\
		}
		if( ! PLANES_INBOUNDS(fpp1,fpp2) ){
			/* BUG should this stuff test drawing flag? */
			if( inbounds(&fpp1->fp_plane_loc) ) draw_plane(fpp1,BLACK);
			if( inbounds(&fpp2->fp_plane_loc) ) draw_plane(fpp2,BLACK);
		}

	} while( ! PLANES_INBOUNDS(fpp1,fpp2) );

	fpp2->fp_tag_angle = fpp2->fp_theta - Init_Tag_Angle;
	fpp1->fp_tag_angle = fpp1->fp_theta - Init_Tag_Angle;
}


/* ---------------------------------------------------------------- */
static void generate_fixed_path(Flight_Path *fpp, double ct, long npath, long alt)
{
	/* generates a flight plan along a given path under the restriction
	 * that the plane must pass through the controller region during
	 * a critical period of time.
	 */
	double ctime;
	Point cpt;
	Point left_pt, right_pt;

	/* time at which plane is guaranteed to be
	 * crossing the controller region
	 */
	if( rn(1) ) {
		fpp->fp_theta = reverse_angle(fixed_routes[npath - 1].fr_theta);
	} else
		fpp->fp_theta = fixed_routes[npath - 1].fr_theta;

	fpp->fp_theta = transformtheta(fpp->fp_theta, alt);

	fpp->fp_speed = RANDOM_SPEED;
	fpp->fp_vel.v_x = cos( DEGREES_TO_RADIANS(fpp->fp_theta) ) * fpp->fp_speed *
			PIXELS_PER_KNOT * HOURS_PER_UPDATE;
	fpp->fp_vel.v_y = sin( DEGREES_TO_RADIANS(fpp->fp_theta) ) * fpp->fp_speed *
			PIXELS_PER_KNOT * HOURS_PER_UPDATE;

	fpp->fp_plane_loc.p_x = 0.0;

	/* had a random number added here,
	 * but pathwidth defined to 0 so had no effect!?
	 */
	fpp->fp_plane_loc.p_y = fixed_routes[npath - 1].fr_b;

	get_region_crossing(&left_pt, &right_pt,  fpp );
	cpt.p_x = drand48() * (right_pt.p_x - left_pt.p_x) + left_pt.p_x;

	/* left_x and right_x are the x coordinates of the region
	 * crossing for plane j.  Using a uniform dist choose a
	 * point between left_x and right_x.
	 */
	ctime = (cpt.p_x - fpp->fp_plane_loc.p_x) / fpp->fp_vel.v_x;
	cpt.p_y = fpp->fp_plane_loc.p_y + ctime * fpp->fp_vel.v_y;

	fpp->fp_conf_loc = cpt;
	fpp->fp_plane_loc = cpt;

	/* Why reset the plane location when we used it to compute the conflict point? */
	/*
	dv = fpp->fp_vel;
	SCALE_VECTOR(&dv,-ct);
	DISPLACE_POINT(&fpp->fp_plane_loc,&dv);
	*/

	fpp->fp_tag_angle = fpp->fp_theta - Init_Tag_Angle;
	fpp->fp_altitude = alt;
}


/* ---------------------------------------------------------------- */
static void fixed_nonconflict(QSP_ARG_DECL  Flight_Path *fpp, double ct, long npath, long alt)
{
	boolean nonconflict;
	long counter, npath1, alt1;
	List *lp;
	Node *np;

	counter = 0;
	npath1 = npath;
	alt1 = alt;
	do {
		if( npath == ANY_PATH )
			npath1 = rn(num_fixed_routes-1);
		if( alt == ANY_ALT )
			alt1 = getaltitude(-1.0);

		counter++;

		if (counter > 100) {
			/* if we are having trouble placing a plane on this path
			 * at this altitude, then lets try a different path
			 */
			generate_fixed_path(fpp, ct, rn(num_fixed_routes-1), alt1);
			/* warn the user that this has happened */
			WARN("fixed_nonconflict counter > 100");
		} else
			generate_fixed_path(fpp, ct, npath1, alt1);

		nonconflict = TRUE;
		lp=plane_list(SINGLE_QSP_ARG);
		np=lp->l_head;
		while( np!=NO_NODE && nonconflict ){
			Flight_Path *fpp2;
			/* BUG fpp2 not set!? */
			fpp2 = (Flight_Path *)np->n_data;
			if( fpp2 != fpp && WillConflict( fpp, fpp2 ) )
				nonconflict = FALSE;
			np=np->n_next;
		}
	} while( !( nonconflict && inbounds(&fpp->fp_plane_loc) ) );
}


/* ---------------------------------------------------------------- */
static void fixed_conflict(Flight_Path *fpp1, Flight_Path *fpp2, double ct, long con_num)
{
	/* generates a flight plan for fpp2 along a given path
	 * so that it crashes into fpp1.
	 */

	double left_x, right_x, ct2;
	Point cpt;
	/* time at which plane is guaranteed to be
	 * crossing the controller region
	 */
	long alt, which_route;
	Point left_pt, right_pt;
	Vector dv;

	fpp1->fp_speed = RANDOM_SPEED;
	fpp2->fp_speed = RANDOM_SPEED;
	alt = getaltitude(-1.0);
	fpp1->fp_altitude = alt;
	fpp2->fp_altitude = alt;

	if( path_intersection[con_num - 1].xy_pt.p_x == 0) {  /* HEAD-ON COLLISION */
		if ( ALT_RESTRICTED )
			return;
		do {
			which_route = rn(num_fixed_routes-1);
			fpp1->fp_theta = fixed_routes[which_route - 1].fr_theta;
			if( rn(1) )
				fpp1->fp_theta = reverse_angle(fpp1->fp_theta);
			fpp2->fp_theta = reverse_angle(fpp1->fp_theta);
			fpp2->fp_vel.v_x = cos( DEGREES_TO_RADIANS(fpp2->fp_theta) )
				* fpp2->fp_speed * PIXELS_PER_KNOT * HOURS_PER_UPDATE;
			fpp2->fp_vel.v_y = sin( DEGREES_TO_RADIANS(fpp2->fp_theta) )
				* fpp2->fp_speed * PIXELS_PER_KNOT * HOURS_PER_UPDATE;
			fpp1->fp_vel.v_x = cos( DEGREES_TO_RADIANS(fpp1->fp_theta) )
				* fpp1->fp_speed * PIXELS_PER_KNOT * HOURS_PER_UPDATE;
			fpp1->fp_vel.v_y = sin( DEGREES_TO_RADIANS(fpp1->fp_theta) )
				* fpp1->fp_speed * PIXELS_PER_KNOT * HOURS_PER_UPDATE;

			fpp2->fp_plane_loc.p_x = 0.0;
			fpp2->fp_plane_loc.p_y = fixed_routes[which_route - 1].fr_b;
			get_region_crossing(&left_pt, &right_pt, fpp2);

			/* BUG right_x not set!? */
			right_x = right_pt.p_x;
			left_x = left_pt.p_x;
			cpt.p_x = drand48() * (right_x - left_x) + left_x;

			ct2 = cpt.p_x / fpp2->fp_vel.v_x;
			cpt.p_y = fpp2->fp_plane_loc.p_y + ct2 * fpp2->fp_vel.v_y;
			fpp1->fp_conf_loc = cpt;
			fpp2->fp_conf_loc = cpt;

			fpp1->fp_plane_loc = cpt;
			dv = fpp1->fp_vel;
			SCALE_VECTOR(&dv,-ct);
			DISPLACE_POINT(&fpp1->fp_plane_loc,&dv);

			fpp2->fp_plane_loc = cpt;
			dv = fpp2->fp_vel;
			SCALE_VECTOR(&dv,-ct);
			DISPLACE_POINT(&fpp2->fp_plane_loc,&dv);

		} while( ! PLANES_INBOUNDS(fpp1,fpp2) );
	} else {
		fpp1->fp_theta = transformtheta(path_intersection[con_num - 1].xy_heading[0],
						alt);
		fpp2->fp_theta = transformtheta(path_intersection[con_num - 1].xy_heading[1],
						alt);
		if( (! ALT_RESTRICTED ) && rn(1) ){
			fpp1->fp_theta = reverse_angle(fpp1->fp_theta);
			fpp2->fp_theta = reverse_angle(fpp2->fp_theta);
		}
		cpt.p_x = path_intersection[con_num - 1].xy_pt.p_x;
		cpt.p_y = path_intersection[con_num - 1].xy_pt.p_y;
		fpp1->fp_conf_loc = cpt;
		fpp2->fp_conf_loc = cpt;
		calc_params(fpp1, ct);
		calc_params(fpp2, ct);
		if( ! PLANES_INBOUNDS(fpp1,fpp2) ){
			/* try flippling planes about conflict point...*/
			fpp1->fp_theta = reverse_angle(fpp1->fp_theta);
			fpp2->fp_theta = reverse_angle(fpp2->fp_theta);
			alt = getaltitude(fpp1->fp_theta);
			fpp1->fp_altitude = alt;
			fpp2->fp_altitude = alt;
			calc_params(fpp1, ct);
			calc_params(fpp2, ct);
		}
	}

	fpp1->fp_tag_angle = fpp1->fp_theta - Init_Tag_Angle;
	fpp2->fp_tag_angle = fpp2->fp_theta - Init_Tag_Angle;
}


/* ---------------------------------------------------------------- */
/* Unsolved BUG - we were getting wacky numbers with
 * d = (double)( 100-rn(120) );		WHY???
 */

static double  getdisttime()
{  /* time at which distractor passes through region */
	double d;
	int i;
	/* BUG?? why negative times??? */
	i = rn(120);
/*	i = ( 100 - rn(120) ); */
/*	d = (double)i; */
	d = (double) ( 100 - i );
	return( d );
}

char *new_plane_id(void)
{
	static char id_string[32];
	char *s;
	int n;

	switch( rn(4) ){
		case 0: s="UAL"; break;
		case 1: s="TWA"; break;
		case 2: s="NWA"; break;
		case 3: s="AAL"; break;
		case 4: s="SWA"; break;
#ifdef CAUTIOUS
		default:
			NERROR1("CAUTIOUS:  new_plane_id:  unexpected output from rn(4)!?");
			s="XXX";	// quiet compiler
			break;
#endif /* CAUTIOUS */

	}
	n = 100+rn(899);
	sprintf(id_string,"%s%d",s,n);
	return(id_string);
}


static Flight_Path *first_flight(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;

	lp=plane_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST ) ERROR1("no first flight!?");

	np=lp->l_head;
	if( np == NO_NODE ) ERROR1("no first flight!?!?");

	return( (Flight_Path *)(np->n_data) );
}

Flight_Path * add_plane(QSP_ARG_DECL  Flight_Path *fpp)
{
	Flight_Path *nfpp;
	const char *s;

	nfpp = new_flight_path(QSP_ARG  fpp->fp_name );
	if( nfpp == NO_FLIGHT_PATH ) ERROR1("error entering flight path into database");

	s = nfpp->fp_name;
	*nfpp = *fpp;
	nfpp->fp_name = s;
	return(nfpp);
}

Flight_Path *get_another_plane(SINGLE_QSP_ARG_DECL)
{
	Flight_Path *nfpp;
	char *s;

	/* make sure the random name is not in use */

	do {
		s = new_plane_id();
		nfpp = flight_path_of(QSP_ARG  s);
	} while( nfpp != NO_FLIGHT_PATH );

	nfpp = new_flight_path(QSP_ARG  s );
	nfpp->fp_flags = 0;

	return(nfpp);
}

/* ---------------------------------------------------------------- */
static void get_free_distractors(SINGLE_QSP_ARG_DECL)
{
	long a;
	Flight_Path *first_fpp, *fpp;
	int n;

	/* we need to make sure we have at least 2 planes at each altitude
	 * on every trial
	 */
	first_fpp=first_flight(SINGLE_QSP_ARG);

	for (a = 0; a < N_ALTITUDES; a++) {
		if( altitudes[a] != first_fpp->fp_altitude ){
			fpp = get_another_plane(SINGLE_QSP_ARG);
			free_nonconflict(QSP_ARG  fpp, getdisttime(), altitudes[a],
						rn(359) );
			/*
			PREVIEW(fpp);
			*/
			recompute_coords(QSP_ARG  fpp);							\

			fpp = get_another_plane(SINGLE_QSP_ARG);
			free_nonconflict(QSP_ARG  fpp, getdisttime(), altitudes[a],
						rn(359));
			/*
			PREVIEW(fpp);
			*/
			recompute_coords(QSP_ARG  fpp);							\
		}
	}
	/* now we have 8 planes */
	n = num_planes - 8;
	while(n--){
		fpp = get_another_plane(SINGLE_QSP_ARG);
		free_nonconflict(QSP_ARG  fpp, getdisttime(), -1L, rn(359));
		/*
		PREVIEW(fpp);
		*/
		recompute_coords(QSP_ARG  fpp);							\
	}
}



/* ---------------------------------------------------------------- */
static void get_fixed_distractors(SINGLE_QSP_ARG_DECL)
{
	Flight_Path new_fp, *fpp;
	int i,n;
	int a;

	fpp = first_flight(SINGLE_QSP_ARG);

	for (a = 0; a < N_ALTITUDES; a++) {
		/* make sure >=2 planes at each alt */
		if( altitudes[a] != fpp->fp_altitude) {
			fixed_nonconflict(QSP_ARG  &new_fp, getdisttime(), ANY_PATH, altitudes[a]);
			add_plane(QSP_ARG  &new_fp);
			fixed_nonconflict(QSP_ARG  &new_fp, getdisttime(), ANY_PATH, altitudes[a]);
			add_plane(QSP_ARG  &new_fp);
		}
	}
	/* create at least one plane on every path */
	for(i=0;i<num_fixed_routes;i++){
		if( i != conflict_path_index ){
			fixed_nonconflict(QSP_ARG  &new_fp, getdisttime(), i, ANY_ALT );
			add_plane(QSP_ARG  &new_fp);
		}
	}
	n = num_planes - ( 2 + num_fixed_routes - 1);
	while(n--){
		fixed_nonconflict(QSP_ARG  &new_fp, getdisttime(), ANY_PATH, ANY_ALT );
		add_plane(QSP_ARG  &new_fp);
	}
}

static void default_medians()
{
	/* these values were determined through simulation (see the
	 * procedure called MEDIANS) and have been written down here
	 * so we don't have to simulate each time we run the program.
	 * However, we do have to rerun the sims every time we change
	 * the parameters of the flight path generation.
	 */

	if ( ALT_RESTRICTED )   /* angle_median only used for Free flight */
		angle_median = 66.4;
	else
		angle_median = 100.0;
	if (! ALT_RESTRICTED  && ! CONFINED_TO_ROUTES ) {
		xy_median[0] = 5;
		xy_frac[0] = 0.31;
		xy_median[1] = 8;
		xy_frac[1] = 0.38;
		xy_median[2] = 12;
		xy_frac[2] = 0.38;
	}
	if (! CONFINED_TO_ROUTES ) {
		xy_median[0] = 2;
		xy_frac[0] = 0.55;
		xy_median[1] = 3;
		xy_frac[1] = 0.52;
		xy_median[2] = 4;
		xy_frac[2] = 0.74;
	}
	if (! ALT_RESTRICTED ) {
		xy_median[0] = 10;
		xy_frac[0] = 0.87;
		xy_median[1] = 16;
		xy_frac[1] = 0.92;
		xy_median[2] = 25;
		xy_frac[2] = 0.02;
	}
	if (! CONFINED_TO_ROUTES )
		return;
	xy_median[0] = 3;
	xy_frac[0] = 0.21;
	xy_median[1] = 4;
	xy_frac[1] = 0.63;
	xy_median[2] = 6;
	xy_frac[2] = 0.50;
}

#define GOOD_ANGLE		(angle_of_incidence >= MIN_ANGLE)
#define ALTS_MIGHT_CONFLICT	((! ALT_RESTRICTED ) || (eastbound(theta1) == eastbound(theta2)))


/* ---------------------------------------------------------------- */
static void select_paths(SINGLE_QSP_ARG_DECL)
{
	/* choose a set of flight paths, so that exactly
	 * one pair of the planes collide.
	 */
	long con_num;
	double theta1, theta2;
	Flight_Path *fpp1,*fpp2;

	default_medians();
	atc_flags |= ALT_COLOR_BIT; /* for visualizing */

	clear_all_planes(NULL_QSP);	/* delete any preexisting planes */

	fpp1 = get_another_plane(SINGLE_QSP_ARG);
	fpp2 = get_another_plane(SINGLE_QSP_ARG);

	fpp1->fp_flags |= FP_CONFLICT;
	fpp2->fp_flags |= FP_CONFLICT;

	if( ! CONFINED_TO_ROUTES  ) {
		theta1 = drand48() * 360;
		do {
			theta2 = drand48() * 360;
			angle_of_incidence = angle_under_180(theta1 - theta2);
		} while( !(
			( GOOD_ANGLE && ALTS_MIGHT_CONFLICT )
			&& ( angle_of_incidence < angle_median ) == small_ang));
		free_conflict(QSP_ARG  fpp1, fpp2, conflict_time, theta1, theta2);

		/* determine the flight paths */
		GetAngleOfIncidence(fpp1,fpp2);

		/*
		printf(" %12ld ", (long)floor(angle_of_incidence + 0.5));
		*/

		/* these checks should not be needed at this point, but what the heck */
		check_separation(SINGLE_QSP_ARG);
		/* make sure planes not in conflict at trial onset */
		check_on_screen(fpp1,fpp2);   /* make sure planes on-screen at trial onset */
		if( reporting ){
			if (small_xys)
				strcpy(report_string,"xy-small ");
			else
				strcpy(report_string,"xy-large ");
		}

#ifdef FOO
		if( reporting ){
			sprintf(msg_str,"%12ld ", xy_median[emod(cond, num_plane_nums) - 1]);
			strcat(report_string,msg_str);
		}

		if (small_xys) {
			do {
				/* we generate distractors, but only keep them if the # of
				 * xy-foils meets the criterion for this trial
				 */
				get_free_distractors();   /* choose distractor flight paths */
				calculate_numxys(SINGLE_QSP_ARG);
				if( reporting ){
					sprintf(msg_str,".%12ld", numxys);
					strcat(report_string,msg_str);
				}

		/* NOTE: the previous line is needed because the median
		 * # of xy-foils is not an integer.  Suppose the median is 3.4.
		 * And suppose we are looking for a # of XY foils below the median.
		 * We obviously throw out any trials with more than 4 foils and
		 * we keep anything with 3 or less.  But if it happens to have 4 foils,
		 * then we need to roll the dice so that we keep the trial 40 % of
		 * the time and discard it 60% of the time.
		 */

			} while (!((numxys < xy_median[emod(cond, num_plane_nums) - 1]) |
			((numxys == xy_median[emod(cond, num_plane_nums) - 1]) &
			(drand48() < xy_frac[emod(cond, num_plane_nums) - 1]))));
		} else {
			do {
				get_free_distractors();
				calculate_numxys(SINGLE_QSP_ARG);

				if( reporting ){
					sprintf(msg_str,".%12ld", numxys);
					strcat(report_string,msg_str);
				}

			} while (!((numxys > xy_median[emod(cond, num_plane_nums) - 1]) |
			((numxys == xy_median[emod(cond, num_plane_nums) - 1]) &
			(drand48() > xy_frac[emod(cond, num_plane_nums) - 1]))));
		}
#else
		get_free_distractors(SINGLE_QSP_ARG);
		calculate_numxys(SINGLE_QSP_ARG);
		if( reporting ){
			sprintf(msg_str," numxys = %d", numxys);
			strcat(report_string,msg_str);
		}
#endif
	} else {  /* this is the fixed path condition */
advise("select_paths:  paths are fixed");
		/* what is going on here?? */
		con_num = ( rn(num_con_points-1) + 1 ) / 2;
		if (!small_ang)
			con_num = num_con_points - con_num + 1;
		fixed_conflict(fpp1, fpp2, conflict_time, con_num);
		GetAngleOfIncidence(fpp1,fpp2);
		/*
		printf(" %12ld ", (long)floor(angle_of_incidence + 0.5));
		*/
		check_separation(SINGLE_QSP_ARG);
		check_on_screen(fpp1,fpp2);
		/*
		if (small_xys)
			printf("xy-small ");
		else
			printf("xy-large ");
		*/

#ifdef FOO
		if( reporting ){
			sprintf(msg_str,"%12ld ", xy_median[emod(cond, num_plane_nums) - 1]);
			strcat(report_string,msg_str);
		}

		if (small_xys) {
			do {
				get_fixed_distractors(SINGLE_QSP_ARG);
				calculate_numxys(SINGLE_QSP_ARG);
				/*
				printf(".%12ld", numxys);
				*/
			} while (!((numxys < xy_median[emod(cond, num_plane_nums) - 1]) |
			((numxys == xy_median[emod(cond, num_plane_nums) - 1]) &
			(drand48() < xy_frac[emod(cond, num_plane_nums) - 1]))));
		} else {
			do {
				get_fixed_distractors(SINGLE_QSP_ARG);
				calculate_numxys(SINGLE_QSP_ARG);
				/*
				printf(".%12ld", numxys);
				*/
			} while (!((numxys > xy_median[emod(cond, num_plane_nums) - 1]) |
			((numxys == xy_median[emod(cond, num_plane_nums) - 1]) &
			(drand48() > xy_frac[emod(cond, num_plane_nums) - 1]))));
		}
#else
		get_fixed_distractors(SINGLE_QSP_ARG);
		calculate_numxys(SINGLE_QSP_ARG);
#endif
	}
	/* get_plane_ids(); */
	/*
	initdrawingvars();
	*/

	/* randomly choose a conflict point, above or below the median.
	 * NOTE: conflict points are ordered according to angle of
	 * incidence.
	 */
	/* If # conflict points is odd, then the median item is used
	 * both in the above median and below median conditions, but only
	 * half as often in each case. WARNING: This is not obvious
	 * from a quick glance at the code.
	 */

	if( reporting ) prt_msg(report_string);

	setup_object_coords(NULL_QSP);	/* place tags */

	selection_1 = selection_2 = NO_FLIGHT_PATH;
}  /*select_paths*/

/* ---------------------------------------------------------------- */
static COMMAND_FUNC( report_numxys )
{
	/* display the xy foils here too */
	draw_foils(SINGLE_QSP_ARG);

	sprintf(msg_str,"Number of xy (not z) crossings = %d", numxys);
	prt_msg(msg_str);
}

static int num_crossings;

static void check_crossing(Flight_Path *fpp1, Flight_Path *fpp2)
{
	Point *ptp;

	ptp = crossing_point(fpp1,fpp2);
	if( ptp!=NO_POINT ){
		num_crossings++;
		draw_intersection(fpp1,fpp2,ptp,WHITE);
	}
}

static COMMAND_FUNC( report_crossings )
{
	/* display the crossings here */

	num_crossings=0;
	apply_to_plane_pairs( QSP_ARG  check_crossing );

	sprintf(msg_str,"%d 2-D crossings",num_crossings);
	prt_msg(msg_str);
}

/* ---------------------------------------------------------------- */

#define sample_size     1000


#ifdef FOOBAR
static void medians()
{
	/* simulates flight path generation so we can determine the
	 * distribution of angles_of_incidence and the number of XY_foils
	 * in various conditions
	 */

	/* This procedure needs only to be run once, and the resulting values
	 * recorded -- it does not need to be run each time we generate flight
	 * paths.
	 * Well, actually, it should perhaps be run several times to make
	 * sure we have a large enough sample size to get a precise
	 * estimate of the medians.
	 */
	long o, con_num, over, p;
	double median, frac;
	long xy_counts[102];
	long angle_counts[181];
	double theta1, theta2;

	for (o = 0; o <= 180; o++)
		angle_counts[o] = 0;
	for (o = 0; o <= 101; o++)
		xy_counts[o] = 0;
	printf("Calculating Medians...  ");
	if ( CONFINED_TO_ROUTES )
		printf("FIXED +");
	else
		printf("FREE +");
	if ( ALT_RESTRICTED )
		printf("AR ");
	else
		printf("NO AR");
	putchar('\n');
	conflict_time = (conflict_time_1 + drand48() * 2) * 60 * 1000 / UpdateInterval;

	for (p = 1; p <= sample_size; p++) {
		if (p % 1000 == 0)
			printf("%12ld ", p);
/* p2c: pathgen.pas, line 1309:
 * Note: Using % for possibly-negative arguments [317] */
		if (! CONFINED_TO_ROUTES ) {
			theta1 = drand48() * 360;
			do {
	theta2 = drand48() * 360;
	angle_of_incidence = angle_under_180(theta1 - theta2);
			} while (!((angle_of_incidence >= MIN_ANGLE) &
			((! ALT_RESTRICTED ) | (eastbound(theta1) == eastbound(theta2)))));
			free_conflict(QSP_ARG  2L, 1L, conflict_time, theta1, theta2);
			get_free_distractors();
		} else {
			con_num = rand(num_con_points);
/* p2c: pathgen.pas, line 1322:
 * Warning: Symbol 'RAND' is not defined [221] */
			fixed_conflict(1L, 2L, conflict_time, con_num);
			get_fixed_distractors(SINGLE_QSP_ARG);
		}
		check_separation(SINGLE_QSP_ARG);
		GetAngleOfIncidence(fpp1,fpp2);
		angle_counts[(int)((long)floor(angle_of_incidence + 0.5))]++;
		calculate_numxys(SINGLE_QSP_ARG);
		xy_counts[numxys + 1]++;
	}

	o = 0;
	do {
		o++;
		/*write (o,' ',angle_counts[o]);*/
		angle_counts[o] += angle_counts[o - 1];
		/*writeln(' ',angle_counts[o]);   */
		/*if (o+1) mod 20 = 0 then readln; */
	} while (angle_counts[o] <= sample_size / 2.0);
	frac = (sample_size / 2.0 - angle_counts[o - 1]) /
		(angle_counts[o] - angle_counts[o - 1]);
	median = o - 1 + frac + 0.5;
	printf("angle Median = %5.2f\n", median);

	o = -1;
	over = 0;
	/*writeln('#xys freq cumul_perc');       */
	do {
		o++;
		/*write (' ',o:2,' ',xy_counts[o]:6);*/
		xy_counts[o + 1] += xy_counts[o];
		/* writeln ('   ',xy_counts[o]/(sample_size div 100):5:1);*/
		/*if (o+1) mod 18 = 0 then waitreturn;*/
		if (xy_counts[o + 1] < sample_size / 2.0)
			over = o + 1;
	} while (xy_counts[o + 1] <= sample_size / 2.0 || o <= 3);
	median = over;
	frac = (sample_size / 2.0 - xy_counts[over]) /
		(xy_counts[over + 1] - xy_counts[over]);
	printf("num XYs Median = %5.0f\n", median);
	printf("        frac   = %5.2f\n", frac);
} /* end medians */


/* ---------------------------------------------------------------- */

main(int argc, char *argv[])
{
	/* ----MAIN---- */

	long o;

	pathgen_init();
	default_medians();
	randperm(condarray, (long)num_trials);

	for (trial_counter = 1 - num_prac; trial_counter <= 0; trial_counter++) {
		printf("  trial = %12ld ", trial_counter);
		cond = rand(num_conds);
		printf(" Cond= %12ld ", cond);
		trial_select();
		num_planes = num_planes_1;   /* practice trials have min # planes*/
		small_xys = TRUE;   /* practice trials have few foils*/
		select_paths(SINGLE_QSP_ARG);
		save_trial();
		putchar('\n');
	}

	for (trial_counter = 1; trial_counter <= num_trials; trial_counter++) {
		printf("  trial = %12ld ", trial_counter);
		cond = condarray[trial_counter - 1];
		printf(" Cond= %12ld ", cond);
		trial_select();
		select_paths(SINGLE_QSP_ARG);
		totalxys += numxys;   /* update total # of xyfoils */
		save_trial();   /* write trial to file */
		putchar('\n');
	}

	printf("Press RETURN to continue\n");
	scanf("%*[^\n]");
	getchar();
} /* end main */
	report_numxys();   /* print total # of xy foils - not important */
	if (outfile != NULL)
		fclose(outfile);
	exit(EXIT_SUCCESS);
}
#endif /* FOOBAR */

static COMMAND_FUNC( gen_trial )
{
	num_planes = HOW_MANY("number of planes");
	conflict_time = HOW_MANY("conflict time");
	small_ang = ASKIF("small angle");
	small_xys = ASKIF("small xys");

	if(  ASKIF("fixed routes") )
		atc_flags |= FIXED_ROUTES_BIT;
	else
		atc_flags &= ~FIXED_ROUTES_BIT;

	if(  ASKIF("no altitude restrictions") )
		atc_flags &= ~ALT_RESTRICTED_BIT;
	else
		atc_flags |= ALT_RESTRICTED_BIT;

	select_paths(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( set_min_speed )
{
	double s;

	s=HOW_MUCH("assumed minimum speed");
	if( s >= assumed_max_speed ){
		sprintf(ERROR_STRING,
	"Requested value for min_speed (%g) is not less than than current max_speed (%g)",
			s,assumed_max_speed);
		WARN(ERROR_STRING);
		return;
	}
	assumed_min_speed=s;
}

static COMMAND_FUNC( set_max_speed )
{
	double s;

	s=HOW_MUCH("assumed maximum speed");
	if( s <= assumed_min_speed ){
		sprintf(ERROR_STRING,
	"Requested value for max_speed (%g) is not greater than current min_speed (%g)",
			s,assumed_min_speed);
		WARN(ERROR_STRING);
		return;
	}
	assumed_max_speed=s;
}

static COMMAND_FUNC( do_projections )
{
	atc_type t;

	t=HOW_MUCH("projection time in seconds (or updates?)");

	draw_projections(QSP_ARG  t);
}

static void dump_one_plane(QSP_ARG_DECL  Flight_Path *fpp)
{
	sprintf(msg_str,"CRAFT %s A %ld H %g S %g X %g Y %g TX %g TY %g",
		fpp->fp_name,fpp->fp_altitude,fpp->fp_theta,fpp->fp_speed,
		fpp->fp_plane_loc.p_x,fpp->fp_plane_loc.p_y,
		fpp->fp_tag_loc.p_x,fpp->fp_tag_loc.p_y);
	prt_msg(msg_str);
}

static COMMAND_FUNC( do_dump_config )
{
	apply_to_planes(QSP_ARG  dump_one_plane);
}

static COMMAND_FUNC( do_reveal )
{
	if( ASKIF("reveal conflict pair") )
		atc_flags |= REVEAL_CONFLICT_BIT;
	else
		atc_flags &= ~REVEAL_CONFLICT_BIT;
}

static COMMAND_FUNC( do_set_speed_var )
{
	int v;

	v = HOW_MANY("max speed decrement (%)");
	set_speed_var(v);
}

static Command pathgen_ctbl[]={
{ "trial",	gen_trial,		"generate a trial"				},
{ "render",	do_render,		"display stimulus configuration"				},
{ "reveal",	do_reveal,		"reveal conflict when rendering"				},
{ "wait_for_click",	wait_for_click,	"wait for click in display window"				},
{ "numxys",	report_numxys,		"report number of 2-D approaches"		},
{ "crossings",	report_crossings,	"report number of plausible path crossings"	},
{ "min_speed",	set_min_speed,		"set assumed min speed for crossing calc"	},
{ "max_speed",	set_max_speed,		"set assumed max speed for crossing calc"	},
{ "speed_var",	do_set_speed_var,	"set max percentage variation in speed"	},
{ "projections",do_projections,		"draw projections for a given future time"	},
{ "dump",	do_dump_config,		"dump configuration information"		},
{ "quit",	popcmd,			"exit submenu"					},
{ NULL_COMMAND										}
};

COMMAND_FUNC( pathgen_menu )
{
	if( CONFINED_TO_ROUTES )
		setup_fixed_routes();

	if( center_p == NO_POINT ) init_center();

	PUSHCMD(pathgen_ctbl,"pathgen");
}


#endif /* HAVE_X11 */
