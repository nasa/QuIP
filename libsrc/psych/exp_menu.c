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

/* local prototypes */

static void null_mod(QSP_ARG_DECL Trial_Class *);
static void set_rsp_word(const char **sptr,const char *s,const char *default_str);

int exp_flags=0;

Experiment expt1;	// a singleton



static int custom_keys=0;
static int get_response_from_keyboard=1;

static const char *response_list[N_RESPONSES];
static int exp_inited=0;

static void null_mod(QSP_ARG_DECL Trial_Class * tc_p){}

static COMMAND_FUNC( modify )
{
	Trial_Class *tc_p;

	tc_p = pick_trial_class("");
	if( tc_p == NULL ) return;

	// BUG?  perhaps modrt should be a class member?
	if( modrt==null_mod ) error1("pointer modrt must be defined by user");

	(*modrt)(QSP_ARG tc_p);
}

static int insure_exp_is_ready(SINGLE_QSP_ARG_DECL)	/* make sure there is something to run */
{
	if( eltcount(trial_class_list()) <= 0 ){
		warn("no conditions defined");
		return(-1);
	}
	return(0);
}

#define present_stim(stc_p) _present_stim(QSP_ARG stc_p)

static int _present_stim(QSP_ARG_DECL Staircase *stc_p)
{
	int rsp=REDO;
	Trial_Class *tc_p;

	assert(stc_p!=NULL);

	tc_p = STAIR_CLASS(stc_p);
	assert(tc_p!=NULL);

	if( insure_exp_is_ready(SINGLE_QSP_ARG) == -1 ) return(-1);

	assert( CLASS_XVAL_OBJ(tc_p) != NULL );

	(*stim_func)(QSP_ARG stc_p);

	rsp = (*response_func)(QSP_ARG  stc_p,&expt1);
	return(rsp);
}

#define present_stim_for_stair(stc_p) _present_stim_for_stair(QSP_ARG  stc_p)

static void _present_stim_for_stair(QSP_ARG_DECL  Staircase *stc_p)
{
	if( insure_exp_is_ready(SINGLE_QSP_ARG) == -1 ) return;

	(*stim_func)(QSP_ARG stc_p);
}

#define INIT_DUMMY_STAIR(st)			\
	/* make a dummy staircase */		\
	SET_STAIR_CLASS(&st, tc_p);		\
	SET_STAIR_SUMM_DTBL(&st, NULL);		\
	SET_STAIR_SEQ_DTBL(&st, NULL);		\
	SET_STAIR_INDEX(&st, 0);		\
	SET_STAIR_VAL(&st, v);			\
	SET_STAIR_CRCT_RSP(&st, YES);		\
	SET_STAIR_INC_RSP(&st, YES);		\
	SET_STAIR_TYPE(&st, UP_DOWN);		\
	SET_STAIR_INC(&st, 1);

static COMMAND_FUNC( do_one_trial )	/** present a stimulus, tally response */
{
	short v;
	int rsp;
	Trial_Class *tc_p;
	Staircase st1;

	tc_p = pick_trial_class("");
	v=(short)how_many("level");

	if( tc_p == NULL ) return;

	INIT_DUMMY_STAIR(st1)
	SET_STAIR_VAL(&st1,v);

	rsp=present_stim(&st1);


	set_recording(1);
	save_response(rsp,&st1);
}

static COMMAND_FUNC( do_one_stim )	/** present a stimulus, tally response */
{
	Staircase *stc_p;

	stc_p = pick_stair("");

	if( stc_p == NULL ) return;

	present_stim_for_stair(stc_p);
}

static COMMAND_FUNC( do_one_response )	/** give a response to a staircase and step it! */
{
	int rsp;
	Staircase *stc_p;

	stc_p = pick_stair("");

	if( stc_p == NULL ) return;

	rsp = get_response(stc_p,&expt1);
fprintf(stderr,"do_one_response will call save_response...\n");
	set_recording(1);
	save_response(rsp,stc_p);	// also updates staircase!
}

#define IS_VALID_RESPONSE(r)	( r == REDO || r == ABORT || r == YES || r == NO )

static COMMAND_FUNC( do_test_stim )		/** demo a stimulus for this experiment */
{
	int v,r;
	Trial_Class *tc_p;
	Staircase st1;

	tc_p = pick_trial_class("");
	v=(int)how_many("level");

	if( tc_p == NULL ) return;

	INIT_DUMMY_STAIR(st1)
	SET_STAIR_CLASS(&st1,tc_p);
	SET_STAIR_VAL(&st1,v);

	r=present_stim(&st1);
	assert( IS_VALID_RESPONSE(r) );
}

static COMMAND_FUNC( do_run_exp )
{
	if( insure_exp_is_ready(SINGLE_QSP_ARG) == -1 ) return;

	setup_files(&expt1);

	run_stairs(&expt1);
}

static COMMAND_FUNC( set_dribble_flag )
{
	if( askif("Record trial-by-trial data") )
		SET_EXP_FLAG(&expt1,DRIBBLING);
	else
		CLEAR_EXP_FLAG(&expt1,DRIBBLING);

	if( IS_DRIBBLING(&expt1) )
		advise("Recording trial-by-trial data");
	else
		advise("Recording only summary data");
}

static COMMAND_FUNC( do_exp_init )
{
	setup_files(&expt1);
	run_init();
	set_recording( 1 );
}

static COMMAND_FUNC( set_2afc )
{
	if( askif( "2AFC experiment" ) ){
		advise("Setting 2afc flag");
		advise("Inverting response based on $coin");
		is_fc=1;
	} else {
		advise("Clearing 2afc flag");
		advise("NOT inverting response based on $coin");
		is_fc=0;
	}
}

static COMMAND_FUNC( do_feedback )
{
	const char *s;

	s=nameof("script fragment to interpret for correct feedback");

	if( correct_feedback_string != NULL )
		rls_str(correct_feedback_string);
	correct_feedback_string = savestr(s);

	s=nameof("script fragment to interpret for incorrect feedback");

	if( incorrect_feedback_string != NULL )
		rls_str(incorrect_feedback_string);
	incorrect_feedback_string = savestr(s);

}


static void set_rsp_word(const char **sptr,const char *s,const char *default_str)
{
	/* free the old string only if different from the default */
	if( strcmp(*sptr,default_str) )
		givbuf((void *)(*sptr));

	/* save the new string only if different from the default */
	if( strcmp(s,default_str) ) *sptr=savestr(s);
	else *sptr=default_str;
}

static COMMAND_FUNC( do_use_kb )
{
	get_response_from_keyboard = askif("use keyboard for responses");
}

/* BUG
 * if we set the redo char to 'r' after we have alread run
 * (so that "redo" is in the history list, then the response
 * handler no longer accepts "redo" !?
 */

static COMMAND_FUNC( setyesno )
{
	get_rsp_word(&response_list[YES_INDEX],RSP_YES);
	get_rsp_word(&response_list[NO_INDEX],RSP_NO);
	get_rsp_word(&response_list[REDO_INDEX],RSP_REDO);

	/* now check that everything is legal! */

	if( is_a_substring(RSP_ABORT,response_list[YES_INDEX]) ||
		is_a_substring(RSP_ABORT,response_list[NO_INDEX]) ||
		is_a_substring(RSP_ABORT,response_list[REDO_INDEX]) ){

		warn("conflict with abort response");
		goto bad;
	}
	if( response_list[YES_INDEX][0] == response_list[NO_INDEX][0] ){
		warn("yes and no responses must differ in the 1st character");
		goto bad;
	}
	if( response_list[YES_INDEX][0] == response_list[REDO_INDEX][0] ){
		warn("yes and redo responses must differ in the 1st character");
		goto bad;
	}
	if( response_list[NO_INDEX][0] == response_list[REDO_INDEX][0] ){
		warn("no and redo responses must differ in the 1st character");
		goto bad;
	}
	custom_keys=1;
	return;
bad:
	/* install default responses */
	set_rsp_word(&response_list[YES_INDEX],RSP_YES,RSP_YES);
	set_rsp_word(&response_list[NO_INDEX],RSP_NO,RSP_NO);
	set_rsp_word(&response_list[REDO_INDEX],RSP_REDO,RSP_REDO);
	custom_keys=0;

	return;
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(experiment_menu,s,f,h)

MENU_BEGIN(experiment)
ADD_CMD( classes,	do_class_menu,	stimulus class submenu )
ADD_CMD( modify,	modify,		modify class parameters )
ADD_CMD( parameters,	do_exp_param_menu,		modify experiment parameters )
ADD_CMD( test,		do_test_stim,	test a condition )
ADD_CMD( use_keyboard,	do_use_kb,	enable/disable use of keyboard for responses )
ADD_CMD( init,		do_exp_init,	start new experiment )
ADD_CMD( present_trial,	do_one_trial,	present a stimulus & save data )
ADD_CMD( present_stim,	do_one_stim,	present a stimulus without collecting response)
ADD_CMD( response,	do_one_response,	specify the response for the preceding stimulus)
ADD_CMD( finish,	do_save_data,	close data files )

/*
ADD_CMD( staircase_menu,	do_staircase_menu,	staircase submenu )
ADD_CMD( edit_stairs,	staircase_menu,	edit individual staircases )
*/

ADD_CMD( staircases,	do_staircase_menu,	edit individual staircases )

ADD_CMD( run,		do_run_exp,	run experiment )
ADD_CMD( 2AFC,		set_2afc,	set forced choice flag )
ADD_CMD( keys,		setyesno,	select response keys )
ADD_CMD( xvals,		xval_menu,	x value submenu )
ADD_CMD( dribble,	set_dribble_flag,	set long/short data file format )
ADD_CMD( lookit,	lookmenu,	data analysis submenu )
ADD_CMD( feedback,	do_feedback,	specify feedback strings )
MENU_END(experiment)

COMMAND_FUNC( do_exp_menu )
{
	if( !exp_inited ){
		init_responses();
		init_experiment(&expt1);
		rninit();
		new_exp(SINGLE_QSP_ARG);	// what does that do?
		exp_inited=1;
	}
	CHECK_AND_PUSH_MENU(experiment);
}

