#include "quip_config.h"

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* prototype for sync() */
#endif

#include "quip_prot.h"
#include "param_api.h"
#include "stc.h"
#include "rn.h"
#include "stack.h"	// BUG
#include "query_stack.h"	// BUG

/* This is an alternative menu for when we want to use a staircase to set levels, but
 * we want to maintain control ourselves...
 * "staircases a la carte"
 */

static COMMAND_FUNC( do_get_value )
{
	const char *s;
	Staircase *stc_p;
	char valstr[32];
	float *xv_p;

	s = nameof("name of variable for value storage");
	stc_p=pick_stair( "" );

	if( stc_p == NULL ) return;

	assert(STAIR_XVAL_OBJ(stc_p)!=NULL);
	xv_p = indexed_data(STAIR_XVAL_OBJ(stc_p),stc_p->stair_val);
	assert(xv_p!=NULL);

	sprintf(valstr,"%g",*xv_p);
	assign_var(s,valstr);
}

static COMMAND_FUNC( do_list_staircases )
{
	prt_msg("\nStaircases:\n");
	list_stairs( tell_msgfile() );
	prt_msg("");
}

static COMMAND_FUNC( do_stair_info )
{
	Staircase *stc_p;
	stc_p=pick_stair( "" );
	if( stc_p == NULL ) return;
	print_stair_info(stc_p);
}

static COMMAND_FUNC( do_reset_stair )
{
	Staircase *stc_p;
	stc_p=pick_stair( "" );
	if( stc_p == NULL ) return;
	reset_stair(stc_p);
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(staircases_menu,s,f,h)

MENU_BEGIN(staircases)
ADD_CMD( list,		do_list_staircases,	list all staircases )
ADD_CMD( info,		do_stair_info,		print info about a staircase )
ADD_CMD( reset,		do_reset_stair,		reset staircase state and clear data )
ADD_CMD( edit,		staircase_menu,		edit individual staircases )
ADD_CMD( get_value,	do_get_value,		get current level of a staircase )
MENU_END(staircases)

COMMAND_FUNC( do_staircase_menu )
{
	CHECK_AND_PUSH_MENU( staircases );
}

