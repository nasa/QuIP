#include <stdio.h>		/* NULL */
#include "data_obj.h"

/* Jeff's attempt to imitate GeoFusion's approach to dynamic LOD management */

/* These index the edges */

typedef enum {
	NORTH,
	EAST,
	SOUTH,
	WEST
} Cardinal_Direction;

/* These index the quadrants, and the corners... */

typedef enum {
	NW,
	NE,
	SE,
	SW
} Corner_Direction;

typedef struct tile_point {
	float p_c[3];
} Tile_Point;

typedef struct vertex {
	/* The point in object space */
	Tile_Point	v_p;
	/* The transformed point */
	Tile_Point	v_xf;
	int	v_nref;
} Vertex;

#define p_x	p_c[0]
#define p_y	p_c[1]
#define p_z	p_c[2]

#define v_x	v_p.p_x
#define v_y	v_p.p_y
#define v_z	v_p.p_z

#define NO_VERTEX	((Vertex *)NULL)
#define NO_POINT	((Tile_Point *)NULL)

#define POINT_USED(ptp)		((ptp)->v_nref > 0)
#define POINT_FREE(ptp)		((ptp)->v_nref == -1 )

#define TRACK_NEIGHBORS

typedef struct tile {
	const char *		t_name;
	struct tile *		t_parent;
	struct tile *		t_q[4];		/* quadrants (for recursive subdivision) */
	Vertex *		t_v[4];		/* the 4 vertices (shared with neighbors) */
#ifdef TRACK_NEIGHBORS
	struct tile *		t_n[4];		/* neighbors */
#endif /* TRACK_NEIGHBORS */
	int			t_level;	/* recursion depth of this tile */
	int			t_max;		/* recursion depth of deepest subtile */
	float			t_z;		/* z value associated with tile center */
	struct master_tile *	t_mtp;
	int			t_flags;	/* off-screen, or behind cam? */
	int			t_ix, t_iy;	/* integer coords relative to master tile */
} Tile;

/* flag bits */

#define TILE_BOUNDARY_FLAG(dir)		(1<<(dir))
#define TILE_IS_NORTH_BOUNDARY		TILE_BOUNDARY_FLAG(NORTH)		/* 1 */
#define TILE_IS_EAST_BOUNDARY		TILE_BOUNDARY_FLAG(EAST)		/* 2 */
#define TILE_IS_SOUTH_BOUNDARY		TILE_BOUNDARY_FLAG(SOUTH)		/* 4 */
#define TILE_IS_WEST_BOUNDARY		TILE_BOUNDARY_FLAG(WEST)		/* 8 */
#define BOUNDARY_FLAGS_MASK	(TILE_IS_NORTH_BOUNDARY	|	\
				 TILE_IS_EAST_BOUNDARY	|	\
				 TILE_IS_SOUTH_BOUNDARY	|	\
				 TILE_IS_WEST_BOUNDARY)
#define TILE_BEHIND_CAMERA	16
#define TILE_OUTSIDE_FOV	32
#define VERTEX_BEHIND_CAMERA	64
#define TILE_SUBDIVIDED		128
#define TILE_N_CHILD		256
#define TILE_W_CHILD		512

#define NOT_VISIBLE(tp)		((tp)->t_flags & (TILE_BEHIND_CAMERA|TILE_OUTSIDE_FOV))
#define IS_SUBDIVIDED(tp)	((tp)->t_flags & TILE_SUBDIVIDED)

#define IS_NORTH_CHILD(tp)	((tp)->t_flags & TILE_N_CHILD)
#define IS_SOUTH_CHILD(tp)	(((tp)->t_flags & TILE_N_CHILD)==0)
#define IS_WEST_CHILD(tp)	((tp)->t_flags & TILE_W_CHILD)
#define IS_EAST_CHILD(tp)	(((tp)->t_flags & TILE_W_CHILD)==0)

#define NO_TILE	((Tile *)NULL)

#define MAX_DEM_LEVELS		10
#define MAX_TEX_LEVELS		7

typedef struct master_tile {
	Tile *		mt_tp;
	Data_Obj *	mt_dem_dp[MAX_DEM_LEVELS];	/* the elevation data */
	Data_Obj *	mt_tex_dp[MAX_TEX_LEVELS];	/* the elevation data */
	const char *	mt_dem_name;
	const char *	mt_tex_name;
} Master_Tile;

typedef struct quad_coords {
	float qc_x0;
	float qc_y0;
	float qc_delx;
	float qc_dely;
} Quad_Coords;

/* globals */
extern List *tile_lp;
extern u_long	debug_tiles;
extern const char *quad_name[];
extern const char *dir_name[];
extern const char *dem_directory, *tex_directory;

/* prototypes */

#ifdef HASH_TILE_NAMES
extern Tile *tile_of(const char *name);
extern Tile *get_tile(const char *name);
extern void list_tiles(void);
extern Tile *del_tile(const char *name);
extern Tile *pick_tile(const char *prompt);
#endif /* HASH_TILE_NAMES */

extern Vertex *_new_vertex(QSP_ARG_DECL  float,float);
#define new_vertex(x,y) _new_vertex(QSP_ARG  x,y)

extern Master_Tile * _new_master_tile(QSP_ARG_DECL  Vertex *nw, Vertex *ne, Vertex *se, Vertex *sw);
extern Tile * _new_tile(QSP_ARG_DECL  Tile *parent, Vertex *nw, Vertex *ne, Vertex *se, Vertex *sw, int quadrant);
extern void _subdivide_tile(QSP_ARG_DECL  Tile *tp);
extern Tile *add_neighbor(Tile *tp, Cardinal_Direction dir);
extern void show_tile(QSP_ARG_DECL  Tile *tp, const char *prefix);
extern void init_dir_names(void);

#define new_tile(parent,nw,ne,se,sw,quadrant) _new_tile(QSP_ARG  parent,nw,ne,se,sw,quadrant)
#define subdivide_tile(tp) _subdivide_tile(QSP_ARG  tp)
#define new_master_tile(nw,ne,se,sw) _new_master_tile(QSP_ARG  nw,ne,se,sw)


extern COMMAND_FUNC( do_tile_menu );

extern void _draw_tile(QSP_ARG_DECL  Tile *tp);
extern void undivide_tile(Tile *tp);
extern void xdraw_master(QSP_ARG_DECL  Master_Tile *tp);
#define draw_tile(tp) _draw_tile(QSP_ARG  tp)

extern void tile_xform(Tile *tp,Data_Obj *matp);
extern void master_tile_elevate(QSP_ARG_DECL  Master_Tile *tp);
extern void master_tile_texture(QSP_ARG_DECL  Master_Tile *tp);
extern void tile_info(QSP_ARG_DECL  Tile *tp);
extern void _set_dthresh(QSP_ARG_DECL  float);
#define set_dthresh(f) _set_dthresh(QSP_ARG  f)

#ifdef FOOBAR
extern void clear_tile_elevations(Tile *tp);
#endif

extern void _tile_check_subdiv(QSP_ARG_DECL  Tile *tp,Data_Obj *matp);
#define tile_check_subdiv(tp,matp) _tile_check_subdiv(QSP_ARG  tp,matp)

extern void set_coord_limits(float xl, float yb, float xr, float yt );

