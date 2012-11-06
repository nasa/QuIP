#include "quip_config.h"

char VersionId_psych_stair_edit[] = QUIP_VERSION_STRING;

#ifdef HAVE_STRING_H
#include <string.h>		/* memcpy() */
#endif

#include "stc.h"
#include "param_api.h"

static Staircase s1;

static Param stair_ptbl[]={
{ "val",     "starting value",			       SHORTP,&s1.stc_val },
{ "inc",     "current increment",		       SHORTP,&s1.stc_inc },
{ "mininc",  "minimum increment (negative for ?)",     SHORTP,&s1.stc_mininc },
{ "crctrsp", "response to count as correct",	       SHORTP,&s1.stc_crctrsp },
{ "incrsp",  "response for staircase to consider up",  SHORTP,&s1.stc_incrsp },
{ "type",    "type code (1 up-dn, 2 2-to-1, 4 3-to-1", SHORTP,&s1.stc_type },
{ NULL_UPARAM	}
};

static COMMAND_FUNC( edit_stair )
{
	Staircase *stcp;

	stcp = PICK_STC("");
	if( stcp== NO_STAIR ) return;

	memcpy(&s1,stcp,sizeof(*stcp));
	chngp(QSP_ARG stair_ptbl);
	memcpy(stcp,&s1,sizeof(*stcp));
}

static const char *type_list[]={ "up_down","two_to_one","three_to_one" };

static int get_stair_type(SINGLE_QSP_ARG_DECL)
{
	int t;

	t=WHICH_ONE("staircase feedback type",3,type_list);
	switch(t){
		case 0: return(UP_DOWN);
		case 1: return(TWO_TO_ONE);
		case 2: return(THREE_TO_ONE);
	}
	return(-1);
}
	
static COMMAND_FUNC( do_add_stair )
{
	int t,c;

	/* now get the parameters in a user friendly way */

	t=get_stair_type(SINGLE_QSP_ARG);
	c=(int)HOW_MANY("index of associated condition");

	if( t < 0 ) return;

	add_stair(QSP_ARG  t,c);
}

static COMMAND_FUNC( do_del_stair )
{
	Staircase *stcp;

	stcp=PICK_STC( "" );
	if( stcp == NO_STAIR ) return;
	del_stair(QSP_ARG  stcp);
}

static COMMAND_FUNC(do_list_stairs){list_stcs(SINGLE_QSP_ARG);}

static COMMAND_FUNC( do_step_stair )
{
	Staircase *stcp;
	int resp;

	stcp=PICK_STC( "" );
	resp=response(QSP_ARG "response");

	if( stcp == NO_STAIR ) return;

	save_response(QSP_ARG resp,stcp);
}

static Command staircase_ctbl[]={
{ "add",	do_add_stair,	"add a staircase"			},
{ "list",	do_list_stairs,	"list currently specified staircases"	},
{ "edit",	edit_stair,	"edit a particular staircase"		},
{ "step",	do_step_stair,	"step a staircase"			},
{ "delete",	do_del_stair,	"delete a staircase"			},
{ "Delete",	del_all_stairs,	"delete all staircases"			},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL_COMMAND								}
};

COMMAND_FUNC( staircase_menu )
{
	PUSHCMD(staircase_ctbl,"staircases");
}

