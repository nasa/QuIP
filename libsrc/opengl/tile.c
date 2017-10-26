/* Jeff's attempt to imitate GeoFusion's approach to dynamic LOD management */
#include "quip_config.h"

/*#include <stdlib.h> */		/* malloc() */ 

#ifdef HAVE_STRING_H
#include <string.h>		/* strcpy() */
#endif

#ifdef HAVE_MATH_H
#include <math.h>		/* sqrt() */
#endif

#include "quip_prot.h"
#include "debug.h"
#include "tile.h"
#include "gl_util.h"
#include "data_obj.h"
#include "fio_api.h"

// why would we want the Z coords to be negative?
// They are supposed to be positive elevations above sea level...
// The answer is that we are looking down on the landscape from the air,
// positive z values indicate range away from us, so altitudes are negative...
#define Z_SIGN		-

#define MAX_VERTICES	10000000

#define RELEASE_VERTEX( vp )								\
											\
			/* vp->v_nref --; */		/* free vertex */		\
			/* if( vp->v_nref == 0 ){					\
				release_vertex(vp);				\
			}*/							\
			vp = NULL;

List *free_pts_lp=NULL;

Vertex vertex_tbl[MAX_VERTICES];
int n_pts_used=0;
int texturing=1;

const char *dir_name[4];
const char *quad_name[4];

int tile_serial=1;

/* d_thresh determines when a tile will be subdivided.
 * Make it smaller to have more subdivisions, larger for fewer.
 * It is not clear here what the units are...
 */

/*#define D_THRESH	0.1 */
/*#define D_THRESH	0.3 */
/*#define D_THRESH	5.0 */
//#define DEFAULT_D_THRESH	0.6
#define DEFAULT_D_THRESH	1.6f

static float d_thresh=DEFAULT_D_THRESH;

/* how do we figure out what this should really be???
 * If we use Init_Projection (view.mac), then
 * 0.3 makes the boundary about halfway to the edge
 * from the center...
 */

#define DEFAULT_X_LIMIT	1
#define DEFAULT_Y_LIMIT	1
static float x_left_limit=(-DEFAULT_X_LIMIT);
static float x_right_limit=(DEFAULT_X_LIMIT);
static float y_bottom_limit=(-DEFAULT_Y_LIMIT);
static float y_top_limit=(DEFAULT_Y_LIMIT);
const char *dem_directory=NULL, *tex_directory=NULL;

int opposite_dir[4]={ SOUTH, WEST, NORTH, EAST };	/* relies on NESW = 0123 */

static void xdraw_tile(Tile *tp,Quad_Coords *qcp);

#ifdef HASH_TILE_NAMES

ITEM_INTERFACE_DECLARATIONS(Tile,tile_)

#endif /* HASH_TILE_NAMES */

void set_dthresh(float d)
{
	if( d_thresh <= 0 ){
		NWARN("set_dthresh:  distance threshold must be positive");
		return;
	}
	d_thresh=d;
}

void set_coord_limits(float xl, float yb, float xr, float yt )
{
	x_left_limit=xl;
	x_right_limit=xr;
	y_top_limit=yt;
	y_bottom_limit=yb;
}

#define WACKY_NUM	12345

#ifdef FOOBAR
static void release_vertex(Vertex *vp)
{
	Node *np;

//sprintf(DEFAULT_ERROR_STRING,"\t\tFreeing vertex at 0x%lx",(int_for_addr)vp);
//NADVISE(DEFAULT_ERROR_STRING);
	np = mk_node(vp);
	if( free_pts_lp == NULL )
		free_pts_lp = new_list();
	addTail(free_pts_lp,np);

#ifdef CAUTIOUS
	vp->v_x = WACKY_NUM;
	vp->v_y = WACKY_NUM;
	vp->v_z = WACKY_NUM;
#endif /* CAUTIOUS */
}
#endif /* FOOBAR */

#ifdef NOT_USED
static int vertex_index(Vertex *vp)
{
	return vp - &vertex_tbl[0];
}
#endif /* NOT_USED */

void init_dir_names()
{
	dir_name[NORTH]="north";
	dir_name[SOUTH]="south";
	dir_name[EAST]="east";
	dir_name[WEST]="west";
	quad_name[NW]="nw";
	quad_name[NE]="ne";
	quad_name[SW]="sw";
	quad_name[SE]="se";
}

static Vertex *find_free_vertex()
{
	Node *np;
	Vertex *vp;

	if( free_pts_lp == NULL ) return(NULL);
	if( QLIST_HEAD(free_pts_lp) == NULL ) return(NULL);
	np = remHead(free_pts_lp);
	vp = (Vertex *)np->n_data;
	rls_node(np);
#ifdef CAUTIOUS
if( vp->v_x != WACKY_NUM || vp->v_y != WACKY_NUM || vp->v_z != WACKY_NUM ){
sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  find_free_vertex:  supposedly free vertex at 0x%lx does not have expected wacky coord vals",(int_for_addr)vp);
NERROR1(DEFAULT_ERROR_STRING);
}
#endif /* CAUTIOUS */
	return(vp);
}

Vertex *new_vertex( float x, float y )
{
	Vertex *vp;

	if( n_pts_used >= MAX_VERTICES ){
		sprintf(DEFAULT_ERROR_STRING,"too many tile vertices requested (%d max)",MAX_VERTICES);
		NERROR1(DEFAULT_ERROR_STRING);
	}

	vp = find_free_vertex();
	if( vp==NULL ){
		vp = &vertex_tbl[n_pts_used++];
sprintf(DEFAULT_ERROR_STRING,"new_vertex %d at %g %g",n_pts_used,x,y);
NADVISE(DEFAULT_ERROR_STRING);
	}
	vp->v_x = x;
	vp->v_y = y;
	vp->v_z = 0.0;		/* maybe this is where we should load the elevation value? */
				/* BUT we don't know the level-of-detail... */
/*
sprintf(DEFAULT_ERROR_STRING,"new_vertex %g %g",x,y);
NADVISE(DEFAULT_ERROR_STRING);
*/
	vp->v_nref = 0;		/* we assume caller increments this as needed? */
	return(vp);
}

#ifdef CAUTIOUS

#define VERIFY_NEIGHBOR_IS_NULL(new_tp,new_dir,old_tp,old_dir)						\
													\
		if( new_tp->t_n[new_dir] != NULL )							\
			NERROR1("CAUTIOUS:  check_if_neighbors:  new tile neighbor is not null!?");	\
		if( old_tp->t_n[old_dir] != NULL ){							\
show_tile(QSP_ARG  old_tp,"");\
			sprintf(DEFAULT_ERROR_STRING,								\
	"CAUTIOUS:  check_if_neighbors:  existing tile %s, %s neighbor is %s (expected null)!?",	\
			old_tp->t_name,dir_name[old_dir], old_tp->t_n[old_dir]->t_name);		\
			NERROR1(DEFAULT_ERROR_STRING);								\
		}

#else /* ! CAUTIOUS */
#define VERIFY_NEIGHBOR_IS_NULL(new_mtp,new_dir,old_mtp,old_dir)
#endif /* ! CAUTIOUS */

#ifdef FOOBAR

#define SET_BOUNDARY_FLAGS(tp)										\
	{												\
		int i;											\
													\
		tp->t_flags &= ~BOUNDARY_FLAGS_MASK;							\
		for(i=0;i<4;i++){									\
			if( tp->t_n[i] == NULL )							\
				tp->t_flags |= TILE_BOUNDARY_FLAG(i);					\
		}											\
	}

/* Call FREE_VERTEX when we believe that we are the only reference to the vertex... */
 
#define FREE_VERTEX( vp )								\
											\
			vp->v_nref --;		/* free vertex */			\
			if( vp->v_nref != 0 ){						\
				sprintf(DEFAULT_ERROR_STRING,"Attempt to free multiply-referenced vertex!?");\
				NWARN(DEFAULT_ERROR_STRING);					\
				NERROR1("Check declaration order of master tiles");	\
			} else {							\
				release_vertex(vp);					\
			}								\
			vp = NULL;

#define TEST_IF_NEIGHBORS(new_tp,new_dir,new_corner1,new_corner2,old_tp,old_dir,old_corner1,old_corner2) \
													\
	if( new_tp->t_v[new_corner1]->v_y == old_tp->t_v[old_corner1]->v_y &&				\
	    new_tp->t_v[new_corner1]->v_x == old_tp->t_v[old_corner1]->v_x ){				\
		/* release the two vertices */								\
sprintf(DEFAULT_ERROR_STRING,"%s corner of tile %s (%g,%g) matches %s corner of tile %s (%g,%g)",\
quad_name[new_corner1],new_tp->t_name,new_tp->t_v[new_corner1]->v_x,new_tp->t_v[new_corner1]->v_y,\
quad_name[old_corner1],old_tp->t_name,old_tp->t_v[old_corner1]->v_x,old_tp->t_v[old_corner1]->v_y);\
NADVISE(DEFAULT_ERROR_STRING);\
sprintf(DEFAULT_ERROR_STRING,"releasing %s and %s vertices at 0x%lx and 0x%lx",\
quad_name[new_corner1],quad_name[new_corner2],(int_for_addr)new_tp->t_v[new_corner1],(int_for_addr)new_tp->t_v[new_corner2]);\
NADVISE(DEFAULT_ERROR_STRING);\
		if( new_tp->t_v[new_corner1] != old_tp->t_v[old_corner1] ){				\
			FREE_VERTEX(new_tp->t_v[new_corner1]);						\
			new_tp->t_v[new_corner1] = old_tp->t_v[old_corner1];				\
			new_tp->t_v[new_corner1]->v_nref++;						\
sprintf(DEFAULT_ERROR_STRING,"%s corner of tile %s  reset to %g,%g",\
quad_name[new_corner1],new_tp->t_name,new_tp->t_v[new_corner1]->v_x,new_tp->t_v[new_corner1]->v_y);\
NADVISE(DEFAULT_ERROR_STRING);\
		}											\
		if( new_tp->t_v[new_corner2] != old_tp->t_v[old_corner2] ){				\
			FREE_VERTEX(new_tp->t_v[new_corner2]);						\
			new_tp->t_v[new_corner2] = old_tp->t_v[old_corner2];				\
			new_tp->t_v[new_corner2]->v_nref++;						\
sprintf(DEFAULT_ERROR_STRING,"%s corner of tile %s  reset to %g,%g",\
quad_name[new_corner2],new_tp->t_name,new_tp->t_v[new_corner2]->v_x,new_tp->t_v[new_corner2]->v_y);\
NADVISE(DEFAULT_ERROR_STRING);\
		}											\
													\
		VERIFY_NEIGHBOR_IS_NULL(new_tp,new_dir,old_tp,old_dir)					\
													\
		new_tp->t_n[new_dir] = old_tp;								\
		old_tp->t_n[old_dir] = new_tp;								\
													\
		/* Now test to see if these are (still) boundary tiles */				\
		SET_BOUNDARY_FLAGS(new_tp)								\
		SET_BOUNDARY_FLAGS(old_tp)								\
	}

/* We could examine the coordinates in an intelligent way, but for now
 * we just test each of the four cases.  Because we don't do this very often,
 * efficiency is not a big concern.
 */

#define CHECK_IF_NEIGHBORS(new_tp,old_tp)				\
									\
	TEST_IF_NEIGHBORS(new_tp,NORTH,NW,NE,old_tp,SOUTH,SW,SE)	\
	TEST_IF_NEIGHBORS(new_tp,SOUTH,SW,SE,old_tp,NORTH,NW,NE)	\
	TEST_IF_NEIGHBORS(new_tp,WEST, NW,SW,old_tp,EAST, NE,SE)	\
	TEST_IF_NEIGHBORS(new_tp,EAST, NE,SE,old_tp,WEST, NW,SW)
#endif /* FOOBAR */


Master_Tile * new_master_tile(Vertex *nw, Vertex *ne, Vertex *se, Vertex *sw)
{
	Master_Tile *mtp;
	int i;

	mtp = (Master_Tile *)getbuf(sizeof(*mtp));
	mtp->mt_tp=new_tile(NULL,nw,ne,se,sw,-1);

	/* the vertices have no elevation set yet...
	 * But we can't do anything now because the DEM's haven't been read.
	 */

	mtp->mt_tp->t_mtp = mtp;
	mtp->mt_dem_name = NULL;
	mtp->mt_tex_name = NULL;
	for(i=0;i<MAX_DEM_LEVELS;i++)
		mtp->mt_dem_dp[i] = NULL;
	for(i=0;i<MAX_TEX_LEVELS;i++)
		mtp->mt_tex_dp[i] = NULL;

	/* now determine whether or not this tile is a boundary tile */
	/* When two master tiles abut, we would really like for them to share vertices.
	 * But this is not implemented yet.
	 */

	if( tile_lp == NULL )
		mtp->mt_tp->t_flags |= BOUNDARY_FLAGS_MASK;
	else {
		/* the new master tile is a boundary unless it is flanked on all 4 sides.
		 * If we find a neighbor, that neighbor needs to be rechecked because it
		 * may no longer be a boundary.
		 */
		Node *np;

		np = QLIST_HEAD(tile_lp);
/*
sprintf(DEFAULT_ERROR_STRING,"searching for neighbors of tile %s",mtp->mt_tp->t_name);
NADVISE(DEFAULT_ERROR_STRING);
*/
		while(np!= NULL){
			Master_Tile *mtp2;

			mtp2 = (Master_Tile *)np->n_data;
sprintf(DEFAULT_ERROR_STRING,"NOT checking tile %s",mtp2->mt_tp->t_name);
NADVISE(DEFAULT_ERROR_STRING);
			/* Not used?  why not? */
			/*
			CHECK_IF_NEIGHBORS(mtp->mt_tp,mtp2->mt_tp);
			*/

			np = np->n_next;
		}
	}

#ifdef FOOBAR
/* show all the tiles for debugging */
if( tile_lp!=NULL){
Node *np;
Master_Tile *mtp;
np = QLIST_HEAD(tile_lp);
while(np!= NULL){
mtp=np->n_data;
show_tile(QSP_ARG  mtp->mt_tp,"");
np = np->n_next;
}
}
#endif /* FOOBAR */

	return(mtp);
}

Tile * new_tile(Tile *parent, Vertex *nw, Vertex *ne, Vertex *se, Vertex *sw,int quadrant)
{
	Tile *tp;
	char name[128];


	if( quadrant < 0 ){
		sprintf(name,"tile%d",tile_serial++);
	} else {
		sprintf(name,"%s.%s",parent->t_name,quad_name[quadrant]);
	}

	/* Hashing the tile names simplifies lookup-by-name, but we generally won't
	 * want to do this because of the overhead added to tile creation and destruction.
	 */

#ifdef HASH_TILE_NAMES
	tp = new_tile_(name);
#else
	tp=(Tile *)getbuf(sizeof(*tp));
	tp->t_name = savestr(name);
#endif


#ifdef CAUTIOUS
	if( tp == NULL ) NERROR1("CAUTIOUS:  new_tile:  couldn't allocate new tile");
#endif /* CAUTIOUS */

	nw->v_nref ++;
	sw->v_nref ++;
	ne->v_nref ++;
	se->v_nref ++;

	/* Should we copy the vertices? NO! */

	tp->t_v[NW] = nw;
	tp->t_v[NE] = ne;
	tp->t_v[SE] = se;
	tp->t_v[SW] = sw;

	tp->t_q[0] = NULL;
	tp->t_q[1] = NULL;
	tp->t_q[2] = NULL;
	tp->t_q[3] = NULL;

#ifdef TRACK_NEIGHBORS
	tp->t_n[0] = NULL;
	tp->t_n[1] = NULL;
	tp->t_n[2] = NULL;
	tp->t_n[3] = NULL;
#endif /* TRACK_NEIGHBORS */

	tp->t_flags = 0;

	tp->t_parent = parent;
	if( parent == NULL ){
		tp->t_level=0;
		tp->t_max=0;
		tp->t_mtp = NULL;
	} else {
		tp->t_level = parent->t_level+1;
		tp->t_max = tp->t_level;
		tp->t_mtp = parent->t_mtp;
	}

	if( parent == NULL ){
		tp->t_ix = tp->t_iy = 0;
	} else {
		/* ix, iy give the coords within the master tile.
		 * As we descend a level, the range doubles.
		 * Example:
		 *				0,3	1,3		2,3	3,3
		 *	0,1	1,1		0,2	1,2		2,2	3,2
		 *
		 *			->
		 *				0,1	1,1		2,1	3,1	
		 *	0,0	1,0		0,0	1,0		2,0	3,0
		 *
		 * We can see that for each subdivision, we double the values from the parent
		 * and add a 1 (or not) based on the quadrant.
		 */
		
		tp->t_ix = 2 * parent->t_ix;
		tp->t_iy = 2 * parent->t_iy;

		if( quadrant == NW )
			tp->t_iy += 1;
		else if( quadrant == SE )
			tp->t_ix += 1;
		else if( quadrant == NE ){
			tp->t_ix += 1;
			tp->t_iy += 1;
		}
	}
	return(tp);
}

/* The max level is the max number of subdivisions in any of this tile's quadrants - right?
 */

static void set_max_level(Tile *tp,int level, int count /* for debugging */ )
{
if( count > 50 )
{
NWARN("too many recursive calls to set_max_level");
return;
}
	if( tp->t_max < level ){
		tp->t_max = level;
		if( tp->t_parent != NULL )
			set_max_level(tp->t_parent,level, count+1 );
	}
}

#ifdef TRACK_NEIGHBORS

/* debug print statements...
 *
sprintf(DEFAULT_ERROR_STRING,"using %s neighbor edge vertex for tile %s",dir_name[neighbor],tp->t_name);\
NADVISE(DEFAULT_ERROR_STRING);\
 *
 *
sprintf(DEFAULT_ERROR_STRING,"No %s neighbor for tile %s, creating new edge vertex",dir_name[neighbor],tp->t_name);\
NADVISE(DEFAULT_ERROR_STRING);\
 *
 */

/* CHECK_NEIGHBOR_EDGE
 * We need an edge center vertex - see if the neighboring tile has one we can use.
 * If is has one, we could get the vertex from either of the two neighboring
 * subtiles, we arbitrarily pick one when we call...
 * The four args are not really necessary, as they are all determined by
 * the first arg...
 *
 * We need to make sure that we link in all neighbors when we subdivide...
 * If the neighbor does not have the vertex we need, we create a new one midway
 * between corner1 and corner2.
 */

#define CHECK_NEIGHBOR_EDGE( neighbor, n_quadrant, nq_corner, corner1, corner2 )		\
												\
	if( tp->t_n[neighbor] != NULL && IS_SUBDIVIDED(tp->t_n[neighbor]) ){			\
		/* neighboring tile has already been subdivided, reuse the vertex */		\
		e_ptp[neighbor] = tp->t_n[neighbor]->t_q[n_quadrant]->t_v[nq_corner];		\
	} else {										\
		e_ptp[neighbor] = new_vertex( (tp->t_v[corner1]->v_x + tp->t_v[corner2]->v_x)/2,	\
					(tp->t_v[corner1]->v_y + tp->t_v[corner2]->v_y)/2 );	\
	}

#else

/* use this version if we don't share vertices between neighboring tiles */

#define CHECK_NEIGHBOR_EDGE( neighbor, n_quadrant, nq_corner, corner1, corner2 )		\
		e_ptp[neighbor] = new_vertex( (tp->t_v[corner1]->v_x + tp->t_v[corner2]->v_x)/2,	\
					(tp->t_v[corner1]->v_y + tp->t_v[corner2]->v_y)/2 );	\

#endif /* TRACK_NEIGHBORS */

void subdivide_tile(Tile *tp)
{
	Vertex *new_ptp;
	Vertex *e_ptp[4];
	float d_nw, d_ne, d_sw, d_se;

#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"\t %s original vertices:\n\t\t\t%g, %g, %g\n\t\t\t%g, %g, %g\n\t\t\t%g, %g, %g\n\t\t\t%g, %g, %g",tp->t_name,
tp->t_v[NW]->v_x,tp->t_v[NW]->v_y,tp->t_v[NW]->v_z,
tp->t_v[NE]->v_x,tp->t_v[NE]->v_y,tp->t_v[NE]->v_z,
tp->t_v[SE]->v_x,tp->t_v[SE]->v_y,tp->t_v[SE]->v_z,
tp->t_v[SW]->v_x,tp->t_v[SW]->v_y,tp->t_v[SW]->v_z
);
NADVISE(DEFAULT_ERROR_STRING);

sprintf(DEFAULT_ERROR_STRING,"\t %s transformed vertices:\n\t\t\t%g, %g\n\t\t\t%g, %g\n\t\t\t%g, %g\n\t\t\t%g, %g",tp->t_name,
tp->t_v[NW]->v_xf.p_x,tp->t_v[NW]->v_xf.p_y,
tp->t_v[NE]->v_xf.p_x,tp->t_v[NE]->v_xf.p_y,
tp->t_v[SE]->v_xf.p_x,tp->t_v[SE]->v_xf.p_y,
tp->t_v[SW]->v_xf.p_x,tp->t_v[SW]->v_xf.p_y
);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

#ifdef CAUTIOUS
	if( IS_SUBDIVIDED(tp) ){
		/* should this be a CAUTIOUS warning, or is the user allowed to request this? */
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  subdivide_tile:  tile %s has already been subdivided!?",tp->t_name);
		NERROR1(DEFAULT_ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */

	/* Create the center vertex */
	new_ptp = new_vertex( (tp->t_v[NW]->v_x+tp->t_v[NE]->v_x)/2, 
				(tp->t_v[NW]->v_y+tp->t_v[SW]->v_y)/2 );

	/* Upper left quadrant */

	/* Get the 4 edge vertices, reusing from neighboring tiles if possible */
	CHECK_NEIGHBOR_EDGE( NORTH, SW, SE, NW, NE )
	CHECK_NEIGHBOR_EDGE( WEST,  NE, SE, NW, SW )
	CHECK_NEIGHBOR_EDGE( EAST,  NW, SW, NE, SE )
	CHECK_NEIGHBOR_EDGE( SOUTH, NW, NE, SW, SE )

	/* We need to set the elevations for the new vertices - first we need to determine
	 * whether or not we can take them directly from a DEM image or whether we need to
	 * interpolate from the corners...  We need to know which DEM image to use, and
	 * how to index the proper location...  For simplicity for now we just interpolate...
	 *
	 * DEM resolution is 2^(level+1):  level 0 is 2x2, level 1 is 4x4, etc
	 * up to level 9 which is 1024x1024
	 *
	 * A difficulty is that  the number of vertices is not a power of two except at
	 * level zero; level zero has two vertices, when we subdivide we add one edge vertex,
	 * giving us two edges (three vertices), then four edges (five vertices).
	 * The source of this difficulty is that the elevation samples represent the values
	 * at the tile centers - thus, when we subdivide a tile, we can use a sample value
	 * for the new center vertex, but must interpolate for the new edge values.
	 *
	 * Another difficulty arises in the choice of which DEM resolution to use -
	 * in general, we would like to use a resolution appropriate for the level
	 * of tile subdivision (to avoid aliasing), but we also need to insure
	 * continuity between adjacent tiles, as well as stability as resolution changes.
	 * 
	 * We also need to have a way to keep track of the position of this tile
	 * within the master tile...  we do this with the integer coords t_ix and t_iy.
	 *
	 * New strategy:  master tile has 1 associated elevation value...
	 *		we use this for all vertices, unless there is a neighboring master tile.
	 *		In that case, we average...
	 *
	 * 	First subdivision: 2x2 dem image, with 1 value for each sub-tile
	 *		the center vertex gets the mean of all 4 values, 
	 *		each edge gets the mean of 2.
	 *
	 *	Each tile has an associated depth sample, and up to 8 neighbor depth samples:
	 *
	 *
	 *			dA		dB		dC
	 *				vNW		vNE
	 *			dD		dE		dF
	 *				vSW		vSE
	 *			dG		dH		dI
	 *
	 *
	 *			dA		dB		dC
	 *				vNW		vNE
	 *				    d1      d2
	 *			dD		dE		dF
	 * 				    d3      d4
	 *				vSW		vSE
	 *			dG		dH		dI
	 *
	 */

	if( tp->t_level < MAX_DEM_LEVELS ){
		index_t offsets[N_DIMENSIONS]={0,0,0,0,0};
		u_short *ptr;


/*
sprintf(DEFAULT_ERROR_STRING,"tile %s, level = %d, ix = %d, iy = %d",tp->t_name, tp->t_level,tp->t_ix,tp->t_iy);
NADVISE(DEFAULT_ERROR_STRING);
sprintf(DEFAULT_ERROR_STRING,"object %s, size = %ld",
OBJ_NAME(tp->t_mtp->mt_dem_dp[tp->t_level]),OBJ_COLS(tp->t_mtp->mt_dem_dp[tp->t_level]));
NADVISE(DEFAULT_ERROR_STRING);
*/

		/* Get the center depth samples for the new subtiles.
		 */
		offsets[1] = tp->t_ix;
		/* The tile offsets have the origin 0,0 in the lower left...  flip y axis.  */
		offsets[2] = (OBJ_ROWS(tp->t_mtp->mt_dem_dp[tp->t_level]) - 1) - tp->t_iy ;
		ptr = (u_short *)multiply_indexed_data(tp->t_mtp->mt_dem_dp[tp->t_level],offsets);
		d_nw = *ptr;

		offsets[1] += 1;
		ptr = (u_short *)multiply_indexed_data(tp->t_mtp->mt_dem_dp[tp->t_level],offsets);
		d_ne = *ptr;

		offsets[1] -= 1;
		offsets[2] -= 1;
		ptr = (u_short *)multiply_indexed_data(tp->t_mtp->mt_dem_dp[tp->t_level],offsets);
		d_sw = *ptr;

		offsets[1] += 1;
		ptr = (u_short *)multiply_indexed_data(tp->t_mtp->mt_dem_dp[tp->t_level],offsets);
		d_se = *ptr;
	} else {
		sprintf(DEFAULT_ERROR_STRING,"subdivide_tile:  tile %s, level %d, MAX_DEM_LEVELS = %d",
			tp->t_name,tp->t_level,MAX_DEM_LEVELS);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}


#ifdef FOOBAR
		/* We add 1 because this is the center point... */
		offsets[1] = tp->t_ix + 1;
		/* The tile offsets have the origin 0,0 in the lower left...  flip y axis.  */
		offsets[2] = (OBJ_ROWS(tp->t_mtp->mt_dem_dp[tp->t_level]) - 1) - (tp->t_iy + 1);
		ptr = multiply_indexed_data(tp->t_mtp->mt_dem_dp[tp->t_level],offsets);
		new_ptp->v_z = Z_SIGN *ptr;
		/* Now do the edges... */

		offsets[2] --;
		ptr = multiply_indexed_data(tp->t_mtp->mt_dem_dp[tp->t_level],offsets);
		e_ptp[NORTH]->v_z = Z_SIGN *ptr;
sprintf(DEFAULT_ERROR_STRING,"N edge vertex z value = %g, offsets = %ld, %ld",e_ptp[NORTH]->v_z,offsets[2],offsets[1]);
NADVISE(DEFAULT_ERROR_STRING);

		offsets[2] += 2;
		ptr = multiply_indexed_data(tp->t_mtp->mt_dem_dp[tp->t_level],offsets);
		e_ptp[SOUTH]->v_z = Z_SIGN *ptr;

		offsets[2] --;
		offsets[1] --;
		ptr = multiply_indexed_data(tp->t_mtp->mt_dem_dp[tp->t_level],offsets);
		e_ptp[WEST]->v_z = Z_SIGN *ptr;

		offsets[1] += 2;
		ptr = multiply_indexed_data(tp->t_mtp->mt_dem_dp[tp->t_level],offsets);
		e_ptp[EAST]->v_z = Z_SIGN *ptr;
	} else
	
	{
		/* interpolate elevations from adjacent vertices */
		new_ptp->v_z = (	tp->t_v[NW]->v_z + tp->t_v[NE]->v_z +
					tp->t_v[SW]->v_z + tp->t_v[SE]->v_z )/4;

		e_ptp[NORTH]->v_z = ( tp->t_v[NW]->v_z + tp->t_v[NE]->v_z )/2;
		e_ptp[SOUTH]->v_z = ( tp->t_v[SW]->v_z + tp->t_v[SE]->v_z )/2;
		e_ptp[EAST]->v_z = ( tp->t_v[NE]->v_z + tp->t_v[SE]->v_z )/2;
		e_ptp[WEST]->v_z = ( tp->t_v[NW]->v_z + tp->t_v[SW]->v_z )/2;
	}
#endif /* FOOBAR */

	/* new_tile increments the reference counts on each vertex... */
	tp->t_q[NW] = new_tile( tp, tp->t_v[NW], e_ptp[NORTH], new_ptp, e_ptp[WEST], NW );
	tp->t_q[NE] = new_tile( tp, e_ptp[NORTH], tp->t_v[NE], e_ptp[EAST], new_ptp, NE );
	tp->t_q[SE] = new_tile( tp, new_ptp, e_ptp[EAST], tp->t_v[SE], e_ptp[SOUTH], SE );
	tp->t_q[SW] = new_tile( tp, e_ptp[WEST], new_ptp, e_ptp[SOUTH], tp->t_v[SW], SW );

	/* set the flags so that the children know where they sit relative to the parent */
	tp->t_q[NW]->t_flags |= TILE_N_CHILD;
	tp->t_q[NW]->t_flags |= TILE_W_CHILD;

	tp->t_q[NE]->t_flags |= TILE_N_CHILD;

	tp->t_q[SW]->t_flags |= TILE_W_CHILD;

	/* Set the subtile z value */
	tp->t_q[NW]->t_z = Z_SIGN d_nw;
	tp->t_q[NE]->t_z = Z_SIGN d_ne;
	tp->t_q[SW]->t_z = Z_SIGN d_sw;
	tp->t_q[SE]->t_z = Z_SIGN d_se;

	/* Make sure master tile z value is set...
	 * BUG this never gets done if master tile is never subdivided...
	 */
	if( tp->t_level == 0 )
		tp->t_z = Z_SIGN (d_nw+d_ne+d_sw+d_se)/4;

	/* S and E are indicated by 0 value for the flags... */

#ifdef TRACK_NEIGHBORS
	/* now link these tiles to each other */
	tp->t_q[NW]->t_n[SOUTH] = tp->t_q[SW];
	tp->t_q[NW]->t_n[EAST] = tp->t_q[NE];

	tp->t_q[NE]->t_n[SOUTH] = tp->t_q[SE];
	tp->t_q[NE]->t_n[WEST] = tp->t_q[NW];

	tp->t_q[SE]->t_n[NORTH] = tp->t_q[NE];
	tp->t_q[SE]->t_n[WEST] = tp->t_q[SW];

	tp->t_q[SW]->t_n[NORTH] = tp->t_q[NW];
	tp->t_q[SW]->t_n[EAST] = tp->t_q[SE];
#endif /* TRACK_NEIGHBORS */

#ifdef TRACK_NEIGHBORS
	/* Now we would like to link to neighbors that are not tree siblings.
	 * We may have to ascend all the way to the root to find a sibling...
	 *
	 * We need to ascend to the parent node.  It the node we are subdividing now
	 * is a southern quadrant, then its NORTH neighbor will be one of its siblings,
	 * and so the new NW quadrant may be linked to the appropriate cousin.  If the
	 * node is a northern quadrant, on the other hand, then all of its cousins will be
	 * to the south, and we must look for a second cousin IF the parent is a southern
	 * quadrant, and so on.  We have to pay attention when we descend the other side:
	 *
	 *                              
	 *                                 ___________________GG________________
	 *                               /                                      \
	 *				WG                                      EG
	 *                             / \                                     /  \
	 *                    ww_parent   we_parent		      ew_parent    ee_parent
	 *                   / \                 /   \
	 *          www_child   wwe_child  wew_child  wee_child  eww_child ewe_child    eew_child  eee_child
	 */

#define DEBUG_COUSIN(subq,dir,cousin_q)							\
											\
	sprintf(DEFAULT_ERROR_STRING,"Tile %s:  neighbor[%d] = 0x%lx",tp->t_name,dir,(int_for_addr)tp->t_n[dir]);\
	NADVISE(DEFAULT_ERROR_STRING);\
	sprintf(DEFAULT_ERROR_STRING,"\tsubquad[%d] = 0x%lx",subq,(int_for_addr)tp->t_q[subq]);\
	NADVISE(DEFAULT_ERROR_STRING);\
	if( tp->t_n[dir]!= NULL ){\
		sprintf(DEFAULT_ERROR_STRING,"\tneighbor[%d]->quadrant[%d] = 0x%lx",dir,cousin_q,\
			(int_for_addr)tp->t_n[dir]->t_q[cousin_q]);\
		NADVISE(DEFAULT_ERROR_STRING);\
	}

	/* The solution given here only searches up one level...
	 * How can we generalize??  Is it guaranteed to work?
	 */

#define FIND_COUSIN(subq,dir,cousin_q)							\
											\
	if( tp->t_n[dir] != NULL && tp->t_n[dir]->t_q[cousin_q] != NULL ){	\
		/* link the new quadrant to its cousin */				\
		tp->t_q[subq]->t_n[dir] = tp->t_n[dir]->t_q[cousin_q];			\
		/* link the existing cousin to the new quadrant */			\
		tp->t_n[dir]->t_q[cousin_q]->t_n[ opposite_dir[dir] ] = tp->t_q[subq];	\
	}

	/* Find the neighbor to the north of the NW quadrant */
	FIND_COUSIN(NW,NORTH,SW)
	/* Find the neighbor to the west of the NW quadrant */
	FIND_COUSIN(NW,WEST,NE)

	/* Find the neighbor to the north of the NE quadrant */
	FIND_COUSIN(NE,NORTH,SE)
	/* Find the neighbor to the east of the NE quadrant */
	FIND_COUSIN(NE,EAST,NW)

	/* Find the neighbor to the south of the SW quadrant */
	FIND_COUSIN(SW,SOUTH,NW)
	/* Find the neighbor to the west of the SW quadrant */
	FIND_COUSIN(SW,WEST,SE)

	/* Find the neighbor to the south of the SE quadrant */
	FIND_COUSIN(SE,SOUTH,NE)
	/* Find the neighbor to the east of the SE quadrant */
	FIND_COUSIN(SE,EAST,SW)
#endif /* TRACK_NEIGHBORS */

	set_max_level(tp,tp->t_max+1,0);

	tp->t_flags |= TILE_SUBDIVIDED;

	/* Now we have hopefully linked all the neighbors...  Now we can set the z
	 * values on the new vertices.
	 */
	new_ptp->v_z = Z_SIGN (d_nw+d_ne+d_sw+d_se)/4;
	/* BUG we should be checking for neighbors and averaging accordingly... */
	e_ptp[WEST]->v_z = Z_SIGN (d_nw+d_sw)/2;
	e_ptp[EAST]->v_z = Z_SIGN (d_ne+d_se)/2;
	e_ptp[NORTH]->v_z = Z_SIGN (d_nw+d_ne)/2;
	e_ptp[SOUTH]->v_z = Z_SIGN (d_sw+d_se)/2;
//sprintf(DEFAULT_ERROR_STRING,"Depth samples:  %g   %g   %g   %g",d_nw,d_ne,d_sw,d_se);
//NADVISE(DEFAULT_ERROR_STRING);
}

#ifdef FOOBAR

#define EXTEND_EDGE( c1, c2 )							\
										\
	new_vertex( 2 * tp->t_v[c1]->v_x - tp->t_v[c2]->v_x,			\
				2 * tp->t_v[c1]->v_y - tp->t_v[c2]->v_y );


Tile *add_neighbor(Tile *tp, Cardinal_Direction dir)
{
	Vertex *ptp1, *ptp2;
	Tile *new_tp;

#ifdef TRACK_NEIGHBORS
	if( tp->t_n[dir] != NULL ){
		sprintf(DEFAULT_ERROR_STRING,"tile already has a neighbor to the %s",
			dir_name[dir]);
		NWARN(DEFAULT_ERROR_STRING);
		return(NULL);
	}
#endif /* TRACK_NEIGHBORS */

	switch(dir){
		case NORTH:
			ptp1 = EXTEND_EDGE(NW,SW);
			ptp2 = EXTEND_EDGE(NE,SE);
			new_tp = new_tile( NULL, ptp1, ptp2, tp->t_v[NE], tp->t_v[NW] );
			break;
		case SOUTH:
			ptp1 = EXTEND_EDGE(SW,NW);
			ptp2 = EXTEND_EDGE(SE,NE);
			new_tp = new_tile( NULL, tp->t_v[SW], tp->t_v[SE], ptp2, ptp1 );
			break;
		case WEST:
			ptp1 = EXTEND_EDGE(NW,NE);
			ptp2 = EXTEND_EDGE(SW,SE);
			new_tp = new_tile( NULL, ptp1, tp->t_v[NW], tp->t_v[SW], ptp2 );
			break;
		case EAST:
			ptp1 = EXTEND_EDGE(NE,NW);
			ptp2 = EXTEND_EDGE(SE,SW);
			new_tp = new_tile( NULL, tp->t_v[NE], ptp1, ptp2, tp->t_v[SE] );
			break;
	}
	return(new_tp);
}

#endif /* FOOBAR */

/*
static void show_pt(Vertex *ptp)
{
	sprintf(msg_str,"\t%g, %g, %g",ptp->v_x,ptp->v_y,ptp->v_z);
	prt_msg(msg_str);
}
*/

void show_tile(QSP_ARG_DECL  Tile *tp, const char *prefix)
{
	char str[256];
	int i;

if( verbose ){
sprintf(DEFAULT_ERROR_STRING,"show_tile 0x%lx",(int_for_addr)tp);
NADVISE(DEFAULT_ERROR_STRING);
}
	sprintf(msg_str,"Tile %s:",tp->t_name);
	prt_msg(msg_str);

	strcpy(str,prefix);
	strcat(str,"  ");

	sprintf(msg_str,"\t%sTile level %d, max %d",
		prefix, tp->t_level,tp->t_max);
	prt_msg(msg_str);
	/*
	sprintf(msg_str,"\tvertices:\n\t\t%g\t%g\t%g\n\t\t%g\t%g\t%g\n\t\t%g\t%g\t%g\n\t\t%g\t%g\t%g",
		tp->t_v[NW]->v_x,tp->t_v[NW]->v_y,tp->t_v[NW]->v_z,
		tp->t_v[NE]->v_x,tp->t_v[NE]->v_y,tp->t_v[NE]->v_z,
		tp->t_v[SW]->v_x,tp->t_v[SW]->v_y,tp->t_v[SW]->v_z,
		tp->t_v[SE]->v_x,tp->t_v[SE]->v_y,tp->t_v[SE]->v_z);
	prt_msg(msg_str);
	*/

	for(i=0;i<4;i++){
		sprintf(msg_str,"\t\t%s vertex (addr 0x%lx):  %g, %g, %g",quad_name[i],
			(int_for_addr)tp->t_v[i],
			tp->t_v[i]->v_x,
			tp->t_v[i]->v_y,
			tp->t_v[i]->v_z);
		prt_msg(msg_str);
	}

	if( tp->t_q[NW] != NULL ){
		for(i=0;i<4;i++){
			sprintf(msg_str,"%s%s quadrant:",prefix,quad_name[i]);
			prt_msg(msg_str);

			show_tile(QSP_ARG  tp->t_q[i],str);
		}
	}
#ifdef TRACK_NEIGHBORS
	for(i=0;i<4;i++){
		if( tp->t_n[i] != NULL ){
			sprintf(msg_str,"%s%s neighbor",prefix,dir_name[i]);
			prt_msg(msg_str);

			/* Can't call this recursively - because the master tiles
			 * are each other's neighbors, this would lead to infinte regress.
			 */
			/* show_tile(QSP_ARG  tp->t_n[i],str); */
		}
	}
#endif /* TRACK_NEIGHBORS */
}

/* draw the tile - do we use the transformed vertices, or the original vertices???
 * Here it appears we are using v_x, v_y, v_z, which are the original???
 */

void draw_tile(Tile *tp)
{
	if( tp->t_level == tp->t_max ){
		/* this is just a quadrilateral, a fan with two triangles. */
		if( debug & gl_debug ) NADVISE("glBegin");
#ifdef HAVE_OPENGL
		glBegin(GL_TRIANGLE_FAN);

		/* opengl appears to use a left-handed coordinate system, so we flip the sign on z... */
		/* OR DOES IT??? */
		if( debug & gl_debug ) NADVISE("glVertex3f (4)");
		glVertex3f(	tp->t_v[NW]->v_x,
			tp->t_v[NW]->v_y, tp->t_v[NW]->v_z );
		glVertex3f(	tp->t_v[NE]->v_x,
			tp->t_v[NE]->v_y, tp->t_v[NE]->v_z );
		glVertex3f(	tp->t_v[SE]->v_x,
			tp->t_v[SE]->v_y, tp->t_v[SE]->v_z );
		glVertex3f(	tp->t_v[SW]->v_x,
			tp->t_v[SW]->v_y, tp->t_v[SW]->v_z );
		if( debug & gl_debug ) NADVISE("glEnd");
			glEnd();
#endif /* HAVE_OPENGL */
	} else if( tp->t_level == tp->t_max - 1 ){
		/* This tile is subdivided, but it is the final subdivision.
		 * We can therefore do the whole tile as a triangle fan.
		 */
#ifdef HAVE_OPENGL
		if( debug & gl_debug ) NADVISE("glBegin GL_TRIANGLE_FAN");
		glBegin(GL_TRIANGLE_FAN);
		if( debug & gl_debug ) NADVISE("glVertex3f (9)");

		glVertex3f(	tp->t_q[NW]->t_v[SE]->v_x,
				tp->t_q[NW]->t_v[SE]->v_y, tp->t_q[NW]->t_v[SE]->v_z );
		glVertex3f(	tp->t_q[NW]->t_v[SW]->v_x,
				tp->t_q[NW]->t_v[SW]->v_y, tp->t_q[NW]->t_v[SW]->v_z );
		glVertex3f(	tp->t_q[NW]->t_v[NW]->v_x,
				tp->t_q[NW]->t_v[NW]->v_y, tp->t_q[NW]->t_v[NW]->v_z );
		glVertex3f(	tp->t_q[NW]->t_v[NE]->v_x,
				tp->t_q[NW]->t_v[NE]->v_y, tp->t_q[NW]->t_v[NE]->v_z );
		glVertex3f(	tp->t_q[NE]->t_v[NE]->v_x,
				tp->t_q[NE]->t_v[NE]->v_y, tp->t_q[NE]->t_v[NE]->v_z );
		glVertex3f(	tp->t_q[NE]->t_v[SE]->v_x,
				tp->t_q[NE]->t_v[SE]->v_y, tp->t_q[NE]->t_v[SE]->v_z );
		glVertex3f(	tp->t_q[SE]->t_v[SE]->v_x,
				tp->t_q[SE]->t_v[SE]->v_y, tp->t_q[SE]->t_v[SE]->v_z );
		glVertex3f(	tp->t_q[SE]->t_v[SW]->v_x,
				tp->t_q[SE]->t_v[SW]->v_y, tp->t_q[SE]->t_v[SW]->v_z );
		glVertex3f(	tp->t_q[SW]->t_v[SW]->v_x,
				tp->t_q[SW]->t_v[SW]->v_y, tp->t_q[SW]->t_v[SW]->v_z );
		glVertex3f(	tp->t_q[SW]->t_v[NW]->v_x,
				tp->t_q[SW]->t_v[NW]->v_y, tp->t_q[SW]->t_v[NW]->v_z );	/* same as first vertex */
		if( debug & gl_debug ) NADVISE("glEnd");
		glEnd();
#endif /* HAVE_OPENGL */
	} else {
		/* If only one of the children is subdivided, then we could make a fan of
		 * the others...  But for now we just go recursive
		 */
		draw_tile(tp->t_q[NW]);
		draw_tile(tp->t_q[NE]);
		draw_tile(tp->t_q[SE]);
		draw_tile(tp->t_q[SW]);
	}
}

static void set_tile_texture(QSP_ARG_DECL  Master_Tile *mtp)
{
	Data_Obj *tex_dp;

	tex_dp = mtp->mt_tex_dp[3];		/* which index? */
	set_texture_image(QSP_ARG  tex_dp);

	/* need to specify the texture coords... */
}

void xdraw_master( QSP_ARG_DECL  Master_Tile *mtp )
{
	Quad_Coords qc;

	if( texturing ) set_tile_texture(QSP_ARG  mtp);
	qc.qc_x0=0;
	qc.qc_y0=0;
	qc.qc_delx=1;
	qc.qc_dely=1;
	xdraw_tile(mtp->mt_tp,&qc);
}

#define VERTEX(a,b)	if( debug & gl_debug ){				\
				sprintf(DEFAULT_ERROR_STRING,"glVertex3f %s %s quadrant %s vertex    %g %g %g",\
				tp->t_name,quad_name[a],quad_name[b],	\
					tp->t_q[a]->t_v[b]->v_xf.p_x,				\
					tp->t_q[a]->t_v[b]->v_xf.p_y,				\
					tp->t_q[a]->t_v[b]->v_xf.p_z);	\
				NADVISE(DEFAULT_ERROR_STRING);			\
				sprintf(DEFAULT_ERROR_STRING,"Original pt %s %s quadrant %s vertex  at 0x%lx    %g %g %g",\
				tp->t_name,quad_name[a],quad_name[b],	\
					(int_for_addr)tp->t_q[a]->t_v[b],\
					tp->t_q[a]->t_v[b]->v_x,				\
					tp->t_q[a]->t_v[b]->v_y,				\
					tp->t_q[a]->t_v[b]->v_z);	\
				NADVISE(DEFAULT_ERROR_STRING);			\
			}						\
			glVertex3f(	tp->t_q[a]->t_v[b]->v_xf.p_x,	\
					tp->t_q[a]->t_v[b]->v_xf.p_y,	\
					tp->t_q[a]->t_v[b]->v_xf.p_z );

#define QVERTEX(b)	if( debug & gl_debug ){				\
				sprintf(DEFAULT_ERROR_STRING,"glVertex3f %d    %g %g %g",b,	\
					tp->t_v[b]->v_xf.p_x,				\
					tp->t_v[b]->v_xf.p_y,				\
					tp->t_v[b]->v_xf.p_z);	\
				NADVISE(DEFAULT_ERROR_STRING);			\
			}						\
			glVertex3f(	tp->t_v[b]->v_xf.p_x,	\
					tp->t_v[b]->v_xf.p_y,	\
					tp->t_v[b]->v_xf.p_z );


#define TEX_COORD(x,y)	if( texturing ){ 					\
				if( debug & gl_debug ){				\
					sprintf(DEFAULT_ERROR_STRING,"glTexCoord2f %g %g",x,y);	\
					NADVISE(DEFAULT_ERROR_STRING);			\
				}						\
				glTexCoord2f(x,y);				\
			}

/* Draw the tile using the transformed vertices.
 *
 * We do this so that we can determine the level of detail ourselves...
 *
 * How do we know which image goes with this tile???
 * The tile needs to refer to the image (images) and the coordinates...
 */

static void xdraw_tile(Tile *tp,Quad_Coords *qcp)
{
	float x_l,x_c,x_r;
	float y_u,y_c,y_l;

	if( NOT_VISIBLE(tp) ){
#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"xdraw_tile %s:  NOT_VISIBLE",tp->t_name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif // QUIP_DEBUG
		return;
	}
#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"xdraw_tile %s:  TILE_IS_VISIBLE",tp->t_name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif // QUIP_DEBUG

	/* compute coords for texture mapping */
	x_l=qcp->qc_x0;
	x_c=x_l+qcp->qc_delx/2.0f;
	x_r=x_l+qcp->qc_delx;

	y_l=qcp->qc_y0;
	y_c=y_l+qcp->qc_dely/2.0f;
	y_u=y_l+qcp->qc_dely;

	/* flip y coord */
	y_l = 1-y_l;
	y_c = 1-y_c;
	y_u = 1-y_u;

	if( tp->t_level == tp->t_max ){
		/* this is just a quadrilateral, a fan with two triangles. */
		/* this code draws the outline ... */
#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"xdraw_tile %s:  drawing 2-fan",tp->t_name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif // QUIP_DEBUG
		if( debug & gl_debug ) NADVISE("xdraw_tile:  glBegin (2-fan)");
#ifdef HAVE_OPENGL
		glBegin(GL_TRIANGLE_FAN);
		TEX_COORD(x_l,y_u); QVERTEX(NW)
		TEX_COORD(x_r,y_u); QVERTEX(NE)
		TEX_COORD(x_r,y_l); QVERTEX(SE)
		TEX_COORD(x_l,y_l); QVERTEX(SW)
		if( debug & gl_debug ) NADVISE("glEnd");
		glEnd();
#endif /* HAVE_OPENGL */
	} else if( tp->t_level == tp->t_max - 1 ){
		/* This tile is subdivided, but it is the final subdivision.
		 * We can therefore do the whole tile as a triangle fan.
		 */

/*		NW,NW	NW,NE		NE,NW	NE,NE
 *
 *		NW,SW	NW,SE		NE,SW	NE,SE
 *
 *
 *
 *		SW,NW	SW,NE		SE,NW	SE,NE
 *
 *		SW,SW	SW,SE		SE,SW	SE,SE
 */

		Data_Obj *dem_dp;
		Data_Obj *tex_dp;

#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"xdraw_tile %s:  drawing 8-fan",tp->t_name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif // QUIP_DEBUG
		dem_dp = tp->t_mtp->mt_dem_dp[0];		/* which index? */
		tex_dp = tp->t_mtp->mt_tex_dp[3];		/* which index? */

#ifdef CAUTIOUS
		if( dem_dp == NULL ){
			NWARN("null dem object!?");
			return;
		}
		if( texturing && tex_dp == NULL ){
			NWARN("null texture object!?");
			return;
		}
#endif /* CAUTIOUS */


if( debug & gl_debug ) NADVISE("glBegin GL_TRIANGLE_FAN");
#ifdef HAVE_OPENGL
		glBegin(GL_TRIANGLE_FAN);

		/* need to specify the texture coords before each vertex... */
/* NADVISE("texturing tile"); */

		TEX_COORD(x_c,y_c); VERTEX(NW,SE)
		TEX_COORD(x_l,y_c); VERTEX(NW,SW)
		TEX_COORD(x_l,y_u); VERTEX(NW,NW)		/* first triangle */
		TEX_COORD(x_c,y_u); VERTEX(NW,NE)		/* second triangle */
		TEX_COORD(x_r,y_u); VERTEX(NE,NE)		/* third triangle */
		TEX_COORD(x_r,y_c); VERTEX(NE,SE)		/* fourth triangle */
		TEX_COORD(x_r,y_l); VERTEX(SE,SE)		/* fifth triangle */
		TEX_COORD(x_c,y_l); VERTEX(SE,SW)		/* sixth triangle */
		TEX_COORD(x_l,y_l); VERTEX(SW,SW)		/* seventh triangle */
		TEX_COORD(x_l,y_c); VERTEX(SW,NW)		/* eigth triangle */

if( debug & gl_debug ) NADVISE("glEnd GL_TRIANGLE_FAN");
		glEnd();
#endif /* HAVE_OPENGL */

	} else {
		/* If only one of the children is subdivided, then we could make a fan of
		 * the others...  But for now we just go recursive
		 */
		Quad_Coords qc;

		qc=(*qcp);

		qc.qc_delx /= 2.0;
		qc.qc_dely /= 2.0;
		xdraw_tile(tp->t_q[SW],&qc);
		qc.qc_x0 += qc.qc_delx;
		xdraw_tile(tp->t_q[SE],&qc);
		qc.qc_y0 += qc.qc_dely;
		xdraw_tile(tp->t_q[NE],&qc);
		qc.qc_x0 -= qc.qc_delx;
		xdraw_tile(tp->t_q[NW],&qc);
	}
} /* end xdraw_tile() */

#ifdef FOOBAR
/* Fill in the z values of the tile vertices.
 * 
 * If the resolution is not high enough, then we bump the level.
 */

/* Should we elevate the tile when it is created, or wait until we are all done
 * subdividing???
 */

static void elevate_tile(Tile *tp,Master_Tile *mtp,float x1, float y1, float x2, float y2)
{
	float dx,dy;
	Data_Obj *dem_dp;
	index_t xindex,yindex;

	dx=x2-x1;
	dy=y2-y1;
	/* NW, NE, SE, SW */
	if( tp->t_q[NW] != NULL ){
		elevate_tile(tp->t_q[NW],mtp,x1     ,y1     ,x1+dx/2,y1+dy/2);
		elevate_tile(tp->t_q[NE],mtp,x1+dx/2,y1     ,x2     ,y1+dy/2);
		elevate_tile(tp->t_q[SE],mtp,x1+dx/2,y1+dy/2,x2     ,y2     );
		elevate_tile(tp->t_q[SW],mtp,x1     ,y1+dy/2,x1+dx/2,y2     );
	}

	if( tp->t_level >= MAX_DEM_LEVELS )
		dem_dp = mtp->mt_dem_dp[MAX_DEM_LEVELS-1];
	else
		dem_dp = mtp->mt_dem_dp[tp->t_level];

	if( dem_dp == NULL ){
		sprintf(DEFAULT_ERROR_STRING,"elevate_tile:  missing object at level %d",tp->t_level);
		NWARN(DEFAULT_ERROR_STRING);
	}

#define Z_SCALE_FACTOR	(-0.1)		/* should be 0.1 to convert to meters? */

#define SET_Z( ptp, x, y )							\
										\
	xindex = floor((x)*(OBJ_COLS(dem_dp)-1));					\
	yindex = floor((y)*(OBJ_COLS(dem_dp)-1));					\
	/* 0.1 because values are in 10ths of a meter? */			\
	ptp->v_z = Z_SCALE_FACTOR * ((float) *(((u_short *)OBJ_DATA_PTR(dem_dp)) + xindex*OBJ_PXL_INC(dem_dp) + yindex*OBJ_ROW_INC(dem_dp)));

	/* The short values are in 10ths of a meter??? */

	SET_Z(tp->t_v[NW],x1,y2)
	SET_Z(tp->t_v[NE],x2,y2)
	SET_Z(tp->t_v[SE],x2,y1)
	SET_Z(tp->t_v[SW],x1,y1)
#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"elevate_tile %s z values:  %g %g %g %g",tp->t_name,
tp->t_v[NW]->v_z,
tp->t_v[NE]->v_z,
tp->t_v[SE]->v_z,
tp->t_v[SW]->v_z);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
}
#endif /* FOOBAR */

#ifdef NOT_USED_NOW
static void texture_tile(Tile *tp,Master_Tile *mtp,float x1, float y1, float x2, float y2)
{
	float dx,dy;
	Data_Obj *tex_dp;

	dx=x2-x1;
	dy=y2-y1;
	/* NW, NE, SE, SW */
	if( tp->t_q[NW] != NULL ){
		texture_tile(tp->t_q[NW],mtp,x1     ,y1     ,x1+dx/2,y1+dy/2);
		texture_tile(tp->t_q[NE],mtp,x1+dx/2,y1     ,x2     ,y1+dy/2);
		texture_tile(tp->t_q[SE],mtp,x1+dx/2,y1+dy/2,x2     ,y2     );
		texture_tile(tp->t_q[SW],mtp,x1     ,y1+dy/2,x1+dx/2,y2     );
	}

	if( tp->t_level >= MAX_TEX_LEVELS )
		tex_dp = mtp->mt_tex_dp[MAX_DEM_LEVELS-1];
	else
		tex_dp = mtp->mt_tex_dp[tp->t_level];

	if( tex_dp == NULL ){
		sprintf(DEFAULT_ERROR_STRING,"texture_tile:  missing object at level %d",tp->t_level);
		NWARN(DEFAULT_ERROR_STRING);
	}
}
#endif // NOT_USED_NOW

#ifdef FOOBAR
void clear_tile_elevations(Tile *tp)
{
	int i;

	for(i=0;i<4;i++){
		tp->t_v[i]->v_z = 0;
	}
	if( tp->t_q[NW] != NULL ){
		clear_tile_elevations(tp->t_q[NW]);
		clear_tile_elevations(tp->t_q[NE]);
		clear_tile_elevations(tp->t_q[SE]);
		clear_tile_elevations(tp->t_q[SW]);
	}
}
#endif /* FOOBAR */

/* master_tile_elevate() doesn't set anything, it just reads all the DEM files
 * for later use.  Called once when the master tile is created.
 * Give default z values for the corner vertices based on the tile z value...
 */

void master_tile_elevate(QSP_ARG_DECL  Master_Tile *mtp)
{
	int i=0;
	u_short *ptr;

/*
sprintf(ERROR_STRING,"elevating master tile %s",mtp->mt_tp->t_name);
NADVISE(ERROR_STRING);
*/
	for(i=0;i<MAX_DEM_LEVELS;i++){
		if( mtp->mt_dem_dp[i] == NULL ){
			char filename[256];
			char name[256];
			Image_File *ifp;

			/* OK, lets find the DEM file */

			/*
			sprintf(ERROR_STRING,
				"master tile at 0x%lx has no level %d elevation data object!?",(int_for_addr)mtp,i);
			WARN(ERROR_STRING);
			*/
#ifdef CAUTIOUS
			if( mtp->mt_dem_name == NULL || *mtp->mt_dem_name==0 ){
				sprintf(ERROR_STRING,"CAUTIOUS:  master_tile_elevate:  Null DEM name string, master tile at 0x%lx",
					(int_for_addr)mtp);
				WARN(ERROR_STRING);
				return;
			}
#endif /* CAUTIOUS */


			if( dem_directory != NULL ){
sprintf(ERROR_STRING,"master_tile_elevate:  using DEM dir %s",dem_directory);
advise(ERROR_STRING);
				set_iofile_directory(QSP_ARG  dem_directory);
			}

			sprintf(filename,"%s.%d.tif",mtp->mt_dem_name,i+1);
sprintf(ERROR_STRING,"DEM filename = %s",filename);
advise(ERROR_STRING);
			ifp = read_image_file( QSP_ARG  filename );
			if( ifp == NULL )
				error1("error reading elevation data");
/*
sprintf(ERROR_STRING,"\tlevel %d, size is %ld x %ld",i,OBJ_ROWS(ifp->if_dp),OBJ_COLS(ifp->if_dp));
advise(ERROR_STRING);
*/

			/* now we need to create an object to hold the image */
			sprintf(name,"elevation.%s.%d",mtp->mt_dem_name,i);
			mtp->mt_dem_dp[i] = dup_obj(QSP_ARG  ifp->if_dp,name);

			read_object_from_file(QSP_ARG  mtp->mt_dem_dp[i],ifp);
		}
	}

	/* Now set the elevations of the corner vertices of the master tile.
	 * What we should do is interpolate values from the neighboring master
	 * tiles, since the vertices are shared, but for now we just use what we have
	 * easily at hand...
	 */

	/* The elevation data are u_short... */
sprintf(ERROR_STRING,"master_tile_elevate:  using object %s, size %d",OBJ_NAME(mtp->mt_dem_dp[0]),OBJ_COLS(mtp->mt_dem_dp[0]));
advise(ERROR_STRING);
	ptr = (u_short *)OBJ_DATA_PTR(mtp->mt_dem_dp[0]);		/* first pixel is upper-LEFT corner */
	mtp->mt_tp->t_v[NW]->v_z =  Z_SIGN *ptr++;
	mtp->mt_tp->t_v[NE]->v_z =  Z_SIGN *ptr++;
	mtp->mt_tp->t_v[SW]->v_z =  Z_SIGN *ptr++;
	mtp->mt_tp->t_v[SE]->v_z =  Z_SIGN *ptr;

	show_tile(QSP_ARG  mtp->mt_tp,"");
} /* end master_tile_elevate */

void master_tile_texture(QSP_ARG_DECL  Master_Tile *mtp)
{
	int i=0;

	if( ! texturing ) return;

/*
sprintf(DEFAULT_ERROR_STRING,"texturing master tile %s",mtp->mt_tp->t_name);
advise(DEFAULT_ERROR_STRING);
*/

/* level 0 is 4k x 4k, too big makes the computer thrash... */
	for(i=3;i<MAX_TEX_LEVELS;i++){
		if( mtp->mt_tex_dp[i] == NULL ){
			char filename[256];
			char name[256];
			Image_File *ifp;

			/* OK, lets find the DEM file */

#ifdef CAUTIOUS
			if( mtp->mt_tex_name == NULL || *mtp->mt_tex_name==0 ){
				sprintf(ERROR_STRING,"CAUTIOUS:  master_tile_texture:  Null TEX name string, master tile at 0x%lx",
					(int_for_addr)mtp);
				WARN(ERROR_STRING);
				return;
			}
#endif /* CAUTIOUS */

			if( tex_directory != NULL )
				set_iofile_directory(QSP_ARG  tex_directory);
			else
advise("not resetting texture directory...");

			sprintf(filename,"%s.%d.jpg",mtp->mt_tex_name,i/*+1*/);
			ifp = read_image_file( QSP_ARG  filename );
			if( ifp == NULL )
				NERROR1("unable to read texture file");

/*
sprintf(ERROR_STRING,"\tlevel %d, size is %ld x %ld",i,OBJ_ROWS(ifp->if_dp),OBJ_COLS(ifp->if_dp));
advise(ERROR_STRING);
*/
			/* now we need to create an object to hold the image */
			sprintf(name,"texture.%s.%d",mtp->mt_tex_name,i);
			mtp->mt_tex_dp[i] = dup_obj(QSP_ARG  ifp->if_dp,name);

			read_object_from_file(QSP_ARG  mtp->mt_tex_dp[i],ifp);
		}
	}

	/*
	texture_tile(mtp->mt_tp,mtp,0.0,0.0,1.0,1.0);
	*/
}

/* Apply a viewing transformation to the vertices of a tile...
 * Then check the diameter (which we define to be the max of the two diagonals).
 * If it is below a threshold, then we don't need to subdivide any more,
 * otherwise we do.
 *
 * The viewing transformation is a 4x3 matrix, then we divide by the z coordinate
 * (focal len = 1) to get the projected coordinates.
 *
 * Why not 4x4??
 *
 * BUG there is no linkage between this viewing transformation and
 * the one we use in opengl to render (a "frustrum" call in the script).
 * There we try to simulate an actual camera...
 *
 * Here we assume we work in meters...
 * We assume the elevations are given in meters, and we set the
 * scale of the map when we define the master tile.  If the focal
 * length is expressed in picture widths, things may be simplified somewhat...
 * For the case of a 1/4" sensor w/ 6mm lens, the focal len is about
 * equal to the picture size, or about twice the picture "radius".
 * So if X/Z = 0.5, that should be the edge of the picture.
 * Here we assume a picture size of 512; so a "distance" of 1 will
 * correspond to 512 pixels.  If we want to subdivide when a tile
 * takes up more than 2 pixels, we set the threshold to 2/512 = 1/256.
 */

void tile_check_subdiv(Tile *tp,Data_Obj *matp)
{
	/* BUG need to error-check *matp */
	int i;
	float *cp;	/* coefficient ptr */
	float d1,d2,dx,dy;
	float denom;
	float x,y;
	int right_of, left_of, above, below;
	int n_vertices_behind=0;
	int behind[4]={0,0,0,0};

	/* There is really no point in subdividing below the
	 * max dem level...
	 */
	if( tp->t_level >= MAX_DEM_LEVELS )
		return;


	cp=(float *)OBJ_DATA_PTR(matp);

	tp->t_flags &= ~(TILE_BEHIND_CAMERA|TILE_OUTSIDE_FOV);		/* assume visible */

#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"tile_check_subdiv %s:  BEGIN\n\n\tmatrix:",tp->t_name);
NADVISE(DEFAULT_ERROR_STRING);
sprintf(DEFAULT_ERROR_STRING,"\t%g\t%g\t%g\t%g",cp[0],cp[1],cp[2],cp[3]);
NADVISE(DEFAULT_ERROR_STRING);
sprintf(DEFAULT_ERROR_STRING,"\t%g\t%g\t%g\t%g",cp[4],cp[5],cp[6],cp[7]);
NADVISE(DEFAULT_ERROR_STRING);
sprintf(DEFAULT_ERROR_STRING,"\t%g\t%g\t%g\t%g",cp[8],cp[9],cp[10],cp[11]);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	/* transform the vertices */

	for(i=0;i<4;i++){

		x =	tp->t_v[i]->v_x * cp[0] +
			tp->t_v[i]->v_y * cp[1] +
			tp->t_v[i]->v_z * cp[2] +
					cp[3] ;
		y =	tp->t_v[i]->v_x * cp[4] +
			tp->t_v[i]->v_y * cp[5] +
			tp->t_v[i]->v_z * cp[6] +
					cp[7] ;
		denom =	tp->t_v[i]->v_x * cp[8] +
			tp->t_v[i]->v_y * cp[9] +
			tp->t_v[i]->v_z * cp[10] +
					cp[11] ;

/*
sprintf(DEFAULT_ERROR_STRING,"vertex %d:   %g %g %g",i,tp->t_v[i]->v_x,
                                               tp->t_v[i]->v_y,
                                               tp->t_v[i]->v_z);
NADVISE(DEFAULT_ERROR_STRING);
sprintf(DEFAULT_ERROR_STRING,"\txformed:  %g %g %g",x,y,denom);
NADVISE(DEFAULT_ERROR_STRING);
*/
		if( denom <= 0.0 ){
			/* tile vertex is behind the camera.  We may cull if the tile
			 * has already been subdivided a lot...
			 */
			tp->t_flags |= VERTEX_BEHIND_CAMERA;
			n_vertices_behind++;
			behind[i]=1;
		}

#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"\tx = %g\t\ty = %g\t\tdenom = %g",x,y,denom);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		x /= denom;
		y /= denom;
/*
sprintf(DEFAULT_ERROR_STRING,"vertex (%g %g %g) transformed to %g %g",
tp->t_v[i]->v_x,tp->t_v[i]->v_y,tp->t_v[i]->v_z,x,y);
NADVISE(DEFAULT_ERROR_STRING);
*/
		tp->t_v[i]->v_xf.p_x = x;
		tp->t_v[i]->v_xf.p_y = y;
		tp->t_v[i]->v_xf.p_z = 0;

/*
sprintf(DEFAULT_ERROR_STRING,"\t computed screen pos:  %g %g",	tp->t_v[i]->v_xf.p_x,
							tp->t_v[i]->v_xf.p_y);
NADVISE(DEFAULT_ERROR_STRING);
*/
	}

#define MAX_CAMERA_PLANE_SUBDIVISIONS	12

	if( n_vertices_behind == 4 ){
		tp->t_flags |= TILE_BEHIND_CAMERA;
#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"COMPLETELY_BEHIND_CAMERA %s",tp->t_name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		return;
	} else if( n_vertices_behind>0 && tp->t_level >= MAX_CAMERA_PLANE_SUBDIVISIONS ){
		tp->t_flags |= TILE_BEHIND_CAMERA;
#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"SMALL_TILE_PARTIALLY_BEHIND_CAMERA %s",tp->t_name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		return;
	}


	/* Here we check to see if all 4 vertices are outside the bounding box */
	left_of = 1;
	right_of = 1;
	above = 1;
	below = 1;
	for(i=0;i<4;i++){
		if( ! behind[i] ){
			float _x,_y;
			_x = tp->t_v[i]->v_xf.p_x;
			_y = tp->t_v[i]->v_xf.p_y;

			if( _x > x_left_limit ) left_of = 0;
			if( _x < x_right_limit ) right_of = 0;
			if( _y < y_top_limit ) above = 0;
			if( _y > y_bottom_limit ) below = 0;
		}
	}
	if( left_of || right_of || above || below ){
		tp->t_flags |= TILE_OUTSIDE_FOV;
#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"OUTSIDE_FOV %s",tp->t_name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		return;
	}


	/* Assume everything is ok - now see if we should subdivide this tile */
	/* measure the two diagonals... */
	dx = tp->t_v[NW]->v_xf.p_x - tp->t_v[SE]->v_xf.p_x;
	dy = tp->t_v[NW]->v_xf.p_y - tp->t_v[SE]->v_xf.p_y;
	d1 = sqrtf(dx*dx+dy*dy);

	dx = tp->t_v[NE]->v_xf.p_x - tp->t_v[SW]->v_xf.p_x;
	dy = tp->t_v[NE]->v_xf.p_y - tp->t_v[SW]->v_xf.p_y;
	d2 = sqrtf(dx*dx+dy*dy);

#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"CHECKING %s:  d1 = %g, d2 = %g, THRESH = %g)",
tp->t_name,d1,d2,d_thresh);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( n_vertices_behind > 0 ){
#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"tile_check_subdiv %s:  subdividing partially visible tile",tp->t_name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		subdivide_tile(tp);
		tile_check_subdiv(tp->t_q[NW],matp);
		tile_check_subdiv(tp->t_q[NE],matp);
		tile_check_subdiv(tp->t_q[SW],matp);
		tile_check_subdiv(tp->t_q[SE],matp);
	} else if( d1 > d_thresh || d2 > d_thresh ){

#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"tile_check_subdiv %s:  subdividing large tile",tp->t_name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		subdivide_tile(tp);
		tile_check_subdiv(tp->t_q[NW],matp);
		tile_check_subdiv(tp->t_q[NE],matp);
		tile_check_subdiv(tp->t_q[SW],matp);
		tile_check_subdiv(tp->t_q[SE],matp);

		/* It is possible that none of the children are visible...
		 * in that case, mark this tile as invisible also.
		 */
	} else {
#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"tile_check_subdiv %s:  READY_TO_RENDER",tp->t_name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	}
#ifdef QUIP_DEBUG
if( debug & debug_tiles ){
sprintf(DEFAULT_ERROR_STRING,"tile_check_subdiv %s:  DONE",tp->t_name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

} /* end tile_check_subdiv() */

static void free_subtile(Tile *tp)		/* garbage collection */
{
	int i;

	for(i=0;i<4;i++){
		if( tp->t_q[i] != NULL )
			free_subtile(tp->t_q[i]);
	}

//sprintf(DEFAULT_ERROR_STRING,"free_subtile %s",tp->t_name);
//NADVISE(DEFAULT_ERROR_STRING);

	/* free the vertices */
	for(i=0;i<4;i++){
//sprintf(DEFAULT_ERROR_STRING,"\tvertex %d at 0x%lx, n_ref = %d",i,(int_for_addr)tp->t_v[i],tp->t_v[i]->v_nref);
//NADVISE(DEFAULT_ERROR_STRING);
		RELEASE_VERTEX( tp->t_v[i] );
	}
#ifdef HASH_TILE_NAMES
	del_tile(tp->t_name);
#else
	givbuf((char *)tp->t_name);
	givbuf(tp);
#endif
}

void undivide_tile(Tile *tp)		/* garbage collection */
{
	int i;

	for(i=0;i<4;i++){
		if( tp->t_q[i] != NULL ){
			free_subtile(tp->t_q[i]);
			tp->t_q[i] = NULL;
			tp->t_max = tp->t_level;
		}
	}
	tp->t_flags &= ~TILE_SUBDIVIDED;
}

void tile_info(QSP_ARG_DECL  Tile *tp)
{
	int i;

	sprintf(msg_str,"Tile at 0x%lx",(int_for_addr)tp);
	prt_msg(msg_str);

	for(i=0;i<4;i++){
		sprintf(msg_str,"\tvertex %d at %g %g %g",i,tp->t_v[i]->v_x,tp->t_v[i]->v_y,tp->t_v[i]->v_z);
		prt_msg(msg_str);
	}
}

