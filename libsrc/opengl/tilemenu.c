
#include "quip_config.h"

#ifdef HAVE_OPENGL

#include "quip_prot.h"
#include "tile.h"
#include "data_obj.h"

List *tile_lp=NULL;

#ifdef QUIP_DEBUG
#include "debug.h"
u_long debug_tiles=0;
#endif /* QUIP_DEBUG */

/* TILE_ITERATE executes the statement on all master tiles in the list */

#define TILE_ITERATE( statement )					\
									\
	if ( tile_lp == NULL ){						\
		NWARN("no tiles");					\
		return;							\
	}								\
									\
	np = QLIST_HEAD(tile_lp);					\
	while(np!=NULL){						\
		statement;						\
		np = np->n_next;					\
	}

#define THIS_TILE	(((Master_Tile *)np->n_data)->mt_tp)

static COMMAND_FUNC( do_add )
{
	float x,y;
	Vertex *vp[4];
	Master_Tile *mtp;
	int i;
	char prompt[128];
	Node *np;
	const char *s;

	s=NAMEOF("elevation map stem");

	for(i=0;i<4;i++){
		sprintf(prompt,"x coordinate of %s vertex",quad_name[i]);
		x = (float)HOW_MUCH(prompt);
		sprintf(prompt,"y coordinate of %s vertex",quad_name[i]);
		y = (float)HOW_MUCH(prompt);

		vp[i] = new_vertex(x,y);
	}
	mtp = new_master_tile(vp[0],vp[1],vp[2],vp[3]);

	mtp->mt_dem_name=savestr(s);	/* save the height map */

	/* the dem_name is just the quad name, so we use it for the texture name too... */
	mtp->mt_tex_name = savestr( mtp->mt_dem_name );

	if( tile_lp == NULL )
		tile_lp = new_list();
#ifdef CAUTIOUS
	if( tile_lp == NULL )
		ERROR1("CAUTIOUS:  error creating tile list");
#endif /* CAUTIOUS */
	np = mk_node(mtp);
	addTail(tile_lp,np);

advise("elevating master tile");
	master_tile_elevate(QSP_ARG  mtp);
advise("texturing master tile");
	master_tile_texture(QSP_ARG  mtp);
}

static COMMAND_FUNC( do_show )
{
	Node *np;
	TILE_ITERATE( show_tile(QSP_ARG  THIS_TILE,"") )
}

static COMMAND_FUNC( do_xdraw_tiles )
{
	Node *np;

	TILE_ITERATE( xdraw_master( QSP_ARG  ((Master_Tile *)np->n_data) ) )
	TILE_ITERATE( undivide_tile( THIS_TILE ) )
}

static COMMAND_FUNC( do_draw_tiles )
{
	Node *np;

	TILE_ITERATE( draw_tile( THIS_TILE ) )
}

static COMMAND_FUNC( do_xform_tiles )
{
	Data_Obj *dp;
	Node *np;

	dp = PICK_OBJ("transformation matrix");
	if( dp == NULL ) return;

	/* We call tile_check_subdiv to determine the subdivisions */
	TILE_ITERATE( tile_check_subdiv( THIS_TILE,dp) )

	/* These are not view-specific, and should be done when the master tiles
	 * are created...
	 */

#ifdef FOOBAR
advise("setting tile elevations");
	TILE_ITERATE( master_tile_elevate(np->n_data) )
	TILE_ITERATE( master_tile_texture(np->n_data) )
	/* set the texture for the master tiles - only needs to be done once! */
#endif /* FOOBAR */
}

static COMMAND_FUNC( do_set_bb )
{
	float xl,xr,yb,yt;

	xl=(float)HOW_MUCH("left x limit");
	yb=(float)HOW_MUCH("bottom y limit");
	xr=(float)HOW_MUCH("right x limit");
	yt=(float)HOW_MUCH("top y limit");

	set_coord_limits(xl,yb,xr,yt);
}

static COMMAND_FUNC( do_tile_info )
{
#ifdef HASH_TILE_NAMES
	Tile *tp;

	tp = PICK_TILE("tile name");
	tile_info(QSP_ARG  tp);
#else
	const char *s;

	s=NAMEOF("tile name");
	advise("Sorry, need to compile with HASH_TILE_NAMES defined to do lookup by name");
	sprintf(ERROR_STRING,"Unable to print info for tile %s",s);
	advise(ERROR_STRING);
#endif
}

static COMMAND_FUNC( do_dem_dir )
{
	dem_directory = savestr( NAMEOF("DEM directory") );
}

static COMMAND_FUNC( do_tex_dir )
{
	tex_directory = savestr( NAMEOF("texture directory") );
}

#ifndef HASH_TILE_NAMES
static COMMAND_FUNC( list_tiles )
{
	advise("Sorry, need to compile with HASH_TILE_NAMES defined to list all tile names");
}
#endif /* HASH_TILE_NAMES */

static COMMAND_FUNC( do_set_dthresh )
{
	float d;

	d=(float)HOW_MUCH("distance threshold");
	set_dthresh(d);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(tiles_menu,s,f,h)

MENU_BEGIN(tiles)
ADD_CMD( add,		do_add,		add a tile )
ADD_CMD( show,		do_show,	show all tiles )
ADD_CMD( draw,		do_draw_tiles,	draw tiles )
ADD_CMD( xdraw,		do_xdraw_tiles,	draw transformed tiles )
ADD_CMD( xform,		do_xform_tiles,	transform tiles subdividing as necessary )
ADD_CMD( threshold,	do_set_dthresh,	set threshold distance for tile subdivision )
ADD_CMD( bounding_box,	do_set_bb,	specify bounding box for drawing )
ADD_CMD( info,		do_tile_info,	print info about current tile )
ADD_CMD( list,		list_tiles,	list names of all tiles )
ADD_CMD( dem_directory,	do_dem_dir,	specify DEM directory )
ADD_CMD( tex_directory,	do_tex_dir,	specify texture directory )
MENU_END(tiles)


COMMAND_FUNC( do_tile_menu )
{
	static int inited=0;

	if( !inited ){
		init_dir_names();
		inited=1;
#ifdef DEBUG
		debug_tiles = add_debug_module(QSP_ARG  "tiles");
#endif /* DEBUG */
	}
	PUSH_MENU(tiles);
}

#endif /* HAVE_OPENGL */

