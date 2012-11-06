#include "quip_config.h"

char VersionId_atc_planes[] = QUIP_VERSION_STRING;

/* stuff to handle flight path objects */

#include <math.h>
#include <string.h>

#include "items.h"

#include "conflict.h"
#include "flight_path.h"
#include "draw.h"		/* center_p (among other things) */

static Flight_Path *the_fpp=NO_FLIGHT_PATH;	/* for edit menu */

ITEM_INTERFACE_DECLARATIONS(Flight_Path,flight_path)

List * plane_list(SINGLE_QSP_ARG_DECL)
{
	return( item_list(QSP_ARG  flight_path_itp) );
}

/* convert a structure read in from a specification file
 * to an item.
 */

static void create_flight_path(QSP_ARG_DECL  Flight_Path *fpp)
{
	Flight_Path *new_fpp;
	const char *s;

	new_fpp = new_flight_path(QSP_ARG  fpp->fp_name);

	if( new_fpp == NO_FLIGHT_PATH ){
		sprintf(ERROR_STRING,"Unable to create flight path %s",fpp->fp_name);
		WARN(ERROR_STRING);
		return;
	}

	/* now copy all the fields over */
	/* We don't want to copy the name pointer, because that has already
	 * been initialized by the call to new_flight_path().
	 * Therefore, we save the value before copying the rest of the struct,
	 * then we restore it.
	 */
	s = new_fpp->fp_name;
	*new_fpp = *fpp;
	new_fpp->fp_name = s;

	/* inherit flags from the prototype object */
	/*new_fpp->fp_flags = 0; */
}

void flight_path_info(Flight_Path *fpp)
{
	sprintf(msg_str,"%s:",fpp->fp_name);
	prt_msg(msg_str);

	sprintf(msg_str,"\tlocation:  %g %g (%g %g)",
		fpp->fp_plane_loc.p_x,fpp->fp_plane_loc.p_y,fpp->fp_vertices[0].p_x,fpp->fp_vertices[0].p_y);
	prt_msg(msg_str);

	/*
	prt_msg("\twedge coords:");
	sprintf(msg_str,"\t\t%ld %ld",fpp->fp_ix2,fpp->fp_iy2);
	prt_msg(msg_str);
	sprintf(msg_str,"\t\t%ld %ld",fpp->fp_ix3,fpp->fp_iy3);
	prt_msg(msg_str);
	sprintf(msg_str,"\t\t%ld %ld",fpp->fp_ix4,fpp->fp_iy4);
	prt_msg(msg_str);
	*/
	sprintf(msg_str,"\ttheta:  %g",fpp->fp_theta);
	prt_msg(msg_str);
	sprintf(msg_str,"\tspeed:  %g",fpp->fp_speed);
	prt_msg(msg_str);
	/*
	sprintf(msg_str,"\tx_inc, y_inc:  %g %g",fpp->fp_vel.v_x,fpp->fp_vel.v_y);
	prt_msg(msg_str);
	*/
	sprintf(msg_str,"\taltitude:  %ld",fpp->fp_altitude);
	prt_msg(msg_str);
	sprintf(msg_str,"\ttag_ang:  %g",fpp->fp_tag_angle);
	prt_msg(msg_str);
	sprintf(msg_str,"\ttag_dist:  %g",fpp->fp_tag_dist);
	prt_msg(msg_str);
	sprintf(msg_str,"\ttag_loc:  %g  %g",fpp->fp_tag_loc.p_x, fpp->fp_tag_loc.p_y);
	prt_msg(msg_str);
	sprintf(msg_str,"\ttag_line:  %g  %g",fpp->fp_tag_line.p_x, fpp->fp_tag_line.p_y);
	prt_msg(msg_str);
}

static COMMAND_FUNC( do_flight_path_info )
{
	Flight_Path *fpp;

	fpp = PICK_FLIGHT_PATH("");
	if( fpp == NO_FLIGHT_PATH ) return;

	flight_path_info(fpp);
}

void flight_path_coords(Flight_Path *fpp)
{
	int i;

	sprintf(msg_str,"%s:",fpp->fp_name);
	prt_msg(msg_str);

	prt_msg("\twedge coords:");
	for(i=0;i<4;i++){
		sprintf(msg_str,"\t\t%g %g",
			fpp->fp_vertices[i].p_x,fpp->fp_vertices[i].p_y);
		prt_msg(msg_str);
	}

	sprintf(msg_str,"\tx_inc, y_inc:  %g %g",fpp->fp_vel.v_x,fpp->fp_vel.v_y);
	prt_msg(msg_str);
}

static COMMAND_FUNC( do_flight_path_coords )
{
	Flight_Path *fpp;

	fpp = PICK_FLIGHT_PATH("");
	if( fpp == NO_FLIGHT_PATH ) return;

	flight_path_coords(fpp);
}

static void check_conflicts(QSP_ARG_DECL  Flight_Path *fpp)
{
	List *lp;
	Node *np;
	Flight_Path *fpp2;

	/* fpp is a pointer to a new flight not yet entered into the database */

	/* check for conflicts with all the existing flight paths */

	if( flight_path_itp == NO_ITEM_TYPE ) return;

	lp = item_list(QSP_ARG  flight_path_itp);
	if( lp == NO_LIST ) return;		/* the new flight might be the first? */

	np = lp->l_head;
	if( np == NO_NODE ) return;

	while(np!=NO_NODE){
		fpp2 = np->n_data;

		if( paths_are_in_conflict(fpp,fpp2) ){
			fpp->fp_flags |= FP_CONFLICT;
			fpp2->fp_flags |= FP_CONFLICT;
		}

		np = np->n_next;
	}
}


COMMAND_FUNC( do_def_plane )
{
	Flight_Path fp1;

	fp1.fp_name=NAMEOF("flight call sign");

	fp1.fp_plane_loc.p_x = HOW_MUCH("x location");
	fp1.fp_plane_loc.p_y = HOW_MUCH("y location");
	fp1.fp_theta = HOW_MUCH("heading in degrees");
	fp1.fp_speed = HOW_MUCH("speed in knots");
	fp1.fp_altitude = HOW_MANY("flight level (in 100's of feet)");
	fp1.fp_flags = 0;

	/* probably need to initialize other fields here... */

	/* what are the angle conventions??? */
	fp1.fp_vel.v_x = cos( DEGREES_TO_RADIANS(fp1.fp_theta) );
	fp1.fp_vel.v_y = sin( DEGREES_TO_RADIANS(fp1.fp_theta) );
	SCALE_VECTOR( &fp1.fp_vel, KNOTS_TO_PIXELS(fp1.fp_speed) );

	/* speed is in kts/h, we want to represent how many pixels per update */
	SCALE_VECTOR( &fp1.fp_vel, 1/UPDATES_PER_HOUR );

	/* because id[] has 7 chars... */
	if( strlen(fp1.fp_name) > 6 ){
		sprintf(ERROR_STRING,
			"flight call sign \"%s\" has too many characters",fp1.fp_name);
		WARN(ERROR_STRING);
		return;
	}

	/* check all the other existing planes for conflicts with this one */
	check_conflicts(QSP_ARG  &fp1);

	create_flight_path(QSP_ARG  &fp1);
}

COMMAND_FUNC( clear_all_planes )
{
	List *lp;

	if( flight_path_itp == NO_ITEM_TYPE ) return;

	lp = plane_list(SINGLE_QSP_ARG);
	while( lp!=NO_LIST && lp->l_head != NO_NODE ){
		Node *np;
		Flight_Path *fpp;

		np = lp->l_head;
		fpp = (Flight_Path *)(np->n_data);
		del_flight_path(QSP_ARG  fpp->fp_name);
		rls_str(fpp->fp_name);
		lp = plane_list(SINGLE_QSP_ARG);
	}
}

static COMMAND_FUNC( set_y )
{
	double y;

	y=HOW_MUCH("y location");
	if( the_fpp == NO_FLIGHT_PATH ) return;

	the_fpp->fp_plane_loc.p_y = y;
	/* BUG update iy? */
}

static COMMAND_FUNC( set_x )
{
	double x;

	x=HOW_MUCH("x location");
	if( the_fpp == NO_FLIGHT_PATH ) return;

	the_fpp->fp_plane_loc.p_x = x;
	/* BUG update ix? */
}

static Command edit_ctbl[]={
{ "x",		set_x,		"set x position"	},
{ "y",		set_y,		"set y position"	},
{ "quit",	popcmd,		"exit submenu"		},
{ NULL_COMMAND						}
};

static void edit_plane(QSP_ARG_DECL   Flight_Path *fpp)
{
	the_fpp = fpp;

	PUSHCMD(edit_ctbl,"edit_plane");
}

static COMMAND_FUNC( do_edit_plane )
{
	Flight_Path *fpp;

	fpp = PICK_FLIGHT_PATH("");
	edit_plane(QSP_ARG fpp);
}

static COMMAND_FUNC(do_list_flight_paths){ list_flight_paths(SINGLE_QSP_ARG);}

static Command fp_ctbl[]={
{ "list",	do_list_flight_paths,	"list all planes"		},
{ "info",	do_flight_path_info,	"report info about flight path"	},
{ "coords",	do_flight_path_coords,	"report coords for flight path"	},
{ "new",	do_def_plane,		"define a new flight"		},
{ "edit",	do_edit_plane,		"edit flight path"		},
{ "clear",	clear_all_planes,	"delete all planes"		},
{ "quit",	popcmd,			"exit submenu"			},
{ NULL_COMMAND								}
};


COMMAND_FUNC( plane_menu )
{
	PUSHCMD(fp_ctbl,"planes");
}

