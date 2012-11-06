#ifndef NO_FLIGHT_PATH

#include "geom.h"

/* contains all the information (with redundancy) needed about a plane */

typedef long Altitude;	/* should be enum... */

typedef union {
	Point	pv_pt;
	Vector	pv_vec;
} PtVec;

typedef struct flight_path {
	const char *	fp_name;
	PtVec		fp_plane_pv;
#define	fp_plane_loc	fp_plane_pv.pv_pt
#define	fp_plane_locv	fp_plane_pv.pv_vec
	Point		fp_tag_loc;

	Point		fp_vertices[4];		/* the wedge vertices */

	atc_type	fp_theta;		/* heading of plane - need to check code
						 * to see what 0, 90, etc correspond to.
						 */
	atc_type	fp_speed;		/* in knots */
	Vector		fp_vel;			/* in pixels per update */
	Altitude	fp_altitude;

	atc_type	fp_tag_angle;	/* relative to a vector pointing towards the
						 * plane's destination.  Angles increase in
						 * the clockwise direction.
						 */
	atc_type	fp_tag_dist;
	Point		fp_tag_line;		/* termination of tag line */
	Point		fp_conf_loc;
	int		fp_flags;
} Flight_Path;

#define NO_FLIGHT_PATH	((Flight_Path *)NULL)

/* flag bits */
#define FP_CONFLICT	1
#define FP_ICON_VISIBLE	2
#define FP_TAG_VISIBLE	4
#define FP_TAG_RENDERED	8
#define FP_ICON_RENDERED	16

#define IS_IN_CONFLICT(fpp)	(fpp->fp_flags & FP_CONFLICT)
#define ICON_IS_VISIBLE(fpp)		(fpp->fp_flags & FP_ICON_VISIBLE)
#define TAG_IS_VISIBLE(fpp)		(fpp->fp_flags & FP_TAG_VISIBLE)
#define ICON_NOT_RENDERED(fpp)		((fpp->fp_flags & FP_ICON_RENDERED) == 0 )
#define TAG_NOT_RENDERED(fpp)		((fpp->fp_flags & FP_TAG_RENDERED) == 0 )


ITEM_INTERFACE_PROTOTYPES(Flight_Path,flight_path)

#define PICK_FLIGHT_PATH(pmpt)		pick_flight_path(QSP_ARG  pmpt)



#define ALTITUDE_1	310	/* West */
#define ALTITUDE_2	330	/* East */
#define ALTITUDE_3	350	/* West */
#define ALTITUDE_4	370	/* East */

#define N_ALTITUDES	4


#define WEST_1		ALTITUDE_1
#define WEST_2		ALTITUDE_3
#define EAST_1		ALTITUDE_2
#define EAST_2		ALTITUDE_4

#define UPDATE_INTERVAL		4.0	/* seconds */
#define SECONDS_PER_HOUR	( 60.0 * 60.0 )
#define UPDATES_PER_HOUR	( SECONDS_PER_HOUR / UPDATE_INTERVAL )


#define SUFFICIENT_OBJECT_SEPARATION	50

#endif /* ! NO_FLIGHT_PATH */

