#include "quip_config.h"

#ifdef HAVE_STRING_H
#include <string.h>		/* memcpy() */
#endif

#include "stc.h"
#include "param_api.h"
#include "quip_menu.h"

static Staircase s1;

static Param stair_ptbl[]={
{ "val",     "starting value",			       SHORTP,&s1.stair_val },
{ "inc",     "current increment",		       SHORTP,&s1.stair_inc },
{ "mininc",  "minimum increment (negative for ?)",     SHORTP,&s1.stair_min_inc },
{ "crctrsp", "response to count as correct",	       SHORTP,&s1.stair_crct_rsp },
{ "incrsp",  "response for staircase to consider up",  SHORTP,&s1.stair_inc_rsp },
{ "type",    "type code (1 up-dn, 2 2-to-1, 4 3-to-1", SHORTP,&s1.stair_type },
{ NULL_UPARAM	}
};

static COMMAND_FUNC( edit_stair )
{
	Staircase *stc_p;

	stc_p = pick_stair("");
	if( stc_p== NULL ) return;

	memcpy(&s1,stc_p,sizeof(*stc_p));
	chngp(QSP_ARG stair_ptbl);
	memcpy(stc_p,&s1,sizeof(*stc_p));
}

static const char *type_list[]={ "up_down","two_to_one","three_to_one" };

static int get_stair_type(SINGLE_QSP_ARG_DECL)
{
	int t;

	t=which_one("staircase feedback type",3,type_list);
	switch(t){
		case 0: return(UP_DOWN);
		case 1: return(TWO_TO_ONE);
		case 2: return(THREE_TO_ONE);
	}
	return(-1);
}
	
static COMMAND_FUNC( do_add_stair )
{
	short t;
	Trial_Class *tcp;

	/* now get the parameters in a user friendly way */

	t= (short) get_stair_type(SINGLE_QSP_ARG);
	tcp = pick_trial_class("");

	if( t < 0 ) return;
	if( tcp == NULL ) return;

	add_stair(t,tcp);
}

static COMMAND_FUNC( do_del_stair )
{
	Staircase *stc_p;

	stc_p=pick_stair( "" );
	if( stc_p == NULL ) return;
	del_stair(stc_p);
}

static COMMAND_FUNC(do_list_stairs){list_stairs(tell_msgfile());}

#ifdef FOOBAR
// redundant with one_resp in exp.c !?

static COMMAND_FUNC( do_step_stair )
{
	Staircase *stc_p;
	int resp;

	stc_p=pick_stair( "" );
	resp = get_response(stc_p,exp_p);

	if( stc_p == NULL ) return;

	save_response(resp,stc_p);
}
#endif // FOOBAR

static COMMAND_FUNC( do_del_all )
{
	delete_all_stairs();
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(staircases_menu,s,f,h)

// BUG this menu is redundant with stc_menu!!!

MENU_BEGIN(staircases)
ADD_CMD( add,		do_add_stair,	add a staircase )
ADD_CMD( list,		do_list_stairs,	list currently specified staircases )
ADD_CMD( edit,		edit_stair,	edit a particular staircase )
//ADD_CMD( step,		do_step_stair,	step a staircase )
ADD_CMD( delete,	do_del_stair,	delete a staircase )
ADD_CMD( delete_all,	do_del_all,	delete all staircases )
MENU_END(staircases)

COMMAND_FUNC( staircase_menu )
{
	CHECK_AND_PUSH_MENU(staircases);
}

