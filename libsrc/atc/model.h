#ifndef NO_MODEL_OBJECT

/* the orientation of symbols within this eccentricity can be determined preattentively */
#define DEFAULT_HEADING_ECC_THRESH	150 /* should be specified in degs */

/* For each Flight_Path object, we have a mental representation.
 * This representation holds the observers current state of
 * knowledge about the object.  We assume that location and
 * color are known implicitly (preattentively), and that orientation
 * of the plane (heading) can be recovered preattentively
 * within some radius about fixation.
 */

typedef enum {
	HEADING,
	SPEED,
	ALTITUDE,
	N_INFO_TYPES
} Info_Type;

typedef long age_t;

typedef struct model_object {
	char *		mo_name;	/* not part of the representation */
	Flight_Path *	mo_fpp;		/* the source of the info */
	long		mo_flags;	/* what info is avaiable */
	atc_type	mo_plane_dist;	/* distance from current fixation */
	atc_type	mo_tag_dist;
	age_t		mo_info_age[N_INFO_TYPES];	/* number of fixations since this info was acquired */
						/* starts at 0; -1 indicates info forgotten or not present */
} Model_Obj;

#define NO_MODEL_OBJECT		((Model_Obj *)NULL)

#define SEPARATION_MINIMUM	((10)*PIXELS_PER_KNOT)	/* for now, pixels */


/* flag bits */
#define HEADING_BIT		(1<<HEADING)
#define ALTITUDE_BIT		(1<<ALTITUDE)
#define SPEED_BIT		(1<<SPEED)

#define ALL_FACTS	( HEADING_BIT | ALTITUDE_BIT | SPEED_BIT )
#define ALL_TAG_INFO	( ALTITUDE_BIT | SPEED_BIT )

#define HEADING_KNOWN(mop)	( (mop)->mo_flags & HEADING_BIT )
#define ALT_KNOWN(mop)		( (mop)->mo_flags & ALTITUDE_BIT )
#define SPEED_KNOWN(mop)	( (mop)->mo_flags & SPEED_BIT )
#define COMPLETE_TAG_INFO(mop)	( ((mop)->mo_flags & ALL_TAG_INFO) == ALL_TAG_INFO )
#define COMPLETE_KNOWLEDGE(mop)	( ((mop)->mo_flags & ALL_FACTS) == ALL_FACTS )

typedef struct pair {
	char *		pr_name;
	Model_Obj *	pr_mop1;
	Model_Obj *	pr_mop2;
	u_long		pr_flags;
	atc_type	pr_min_dist;
	atc_type	pr_conflict_time;
	Point		pr_crossing_pt;
} Pair;

#define NO_PAIR	((Pair *)NULL)

/* flag bits */

#define TRACKS_CROSS	1
#define IN_CONFLICT	2
#define DRAWN_BIT	4

#define HAS_CROSSING(prp)	( (prp)->pr_flags & TRACKS_CROSS )
#define HAS_CONFLICT(prp)	( (prp)->pr_flags & IN_CONFLICT )
#define CROSSING_DRAWN(prp)	( (prp)->pr_flags & DRAWN_BIT )

typedef enum {
	PLANE_TARGET,
	TAG_TARGET
} Target_Type;

typedef enum {
	RANDOM,
	NEAREST,
	CLOCKWISE,
	MAX_INFO,
	N_SCAN_TYPES	/* must be last */
} Scan_Type;

typedef struct strategy {
	char *		strat_name;
	Scan_Type	strat_scan;
	u_long		strat_flags;
} Strategy;

#define NO_STRATEGY 	((Strategy *)NULL)

/* flag bits */
#define STRAT_USES_CROSSINGS	1
#define STRAT_USES_ALTITUDE	2
#define STRAT_USES_SALIENCY	4

#define SEES_CROSSINGS( stratp )	( (stratp)->strat_flags & STRAT_USES_CROSSINGS )
#define USES_ALT_KNOWLEDGE( stratp )	( (stratp)->strat_flags & STRAT_USES_ALTITUDE )
#define USES_SALIENCY( stratp )	( (stratp)->strat_flags & STRAT_USES_SALIENCY )

ITEM_INTERFACE_PROTOTYPES(Model_Obj,model_object)
ITEM_INTERFACE_PROTOTYPES(Pair,pair)




#endif /* ! NO_MODEL_OBJECT */

