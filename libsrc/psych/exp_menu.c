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

int exp_flags=0;

Experiment expt1;	// a singleton

static int get_response_from_keyboard=1;

const char *response_choices[N_RESPONSES];
static int exp_inited=0;

static void null_mod(QSP_ARG_DECL Trial_Class * tc_p){}

static COMMAND_FUNC( modify )
{
	Trial_Class *tc_p;

	tc_p = pick_trial_class("");
	if( tc_p == NULL ) return;

	// BUG?  perhaps modrt should be a class member?
	if( EXPT_MOD_FUNC(&expt1)==null_mod ) error1("Modify function must be defined by user");

	(* EXPT_MOD_FUNC(&expt1) )(QSP_ARG tc_p);
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
	int rsp=REDO_INDEX;
	Trial_Class *tc_p;

	assert(stc_p!=NULL);

	tc_p = STAIR_CLASS(stc_p);
	assert(tc_p!=NULL);

	if( insure_exp_is_ready(SINGLE_QSP_ARG) == -1 ) return(-1);

	assert( CLASS_XVAL_OBJ(tc_p) != NULL );

	(* EXPT_STIM_FUNC( STAIR_EXPT(stc_p) ) )(QSP_ARG stc_p);

	rsp = (* EXPT_RSP_FUNC( STAIR_EXPT(stc_p) ) )(QSP_ARG  stc_p,&expt1);
	return(rsp);
}

#define present_stim_for_stair(stc_p) _present_stim_for_stair(QSP_ARG  stc_p)

static void _present_stim_for_stair(QSP_ARG_DECL  Staircase *stc_p)
{
	if( insure_exp_is_ready(SINGLE_QSP_ARG) == -1 ) return;

	(* EXPT_STIM_FUNC( STAIR_EXPT(stc_p) ) )(QSP_ARG stc_p);
}

#define INIT_DUMMY_STAIR(st)			\
	/* make a dummy staircase */		\
	SET_STAIR_CLASS(&st, tc_p);		\
	SET_STAIR_INDEX(&st, 0);		\
	SET_STAIR_VAL(&st, v);			\
	SET_STAIR_CRCT_RSP(&st, YES_INDEX);		\
	SET_STAIR_INC_RSP(&st, YES_INDEX);		\
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

	process_response(rsp,&st1);
}

static Staircase *get_current_stair(Experiment *exp_p)
{
	Staircase **trl_tbl;
	int idx;

	idx = EXPT_CURR_TRIAL_IDX(exp_p);
	assert( idx >= 0 && idx < EXPT_N_TOTAL_TRIALS(exp_p) );
	trl_tbl = EXPT_TRIAL_TBL(exp_p);
	assert(trl_tbl!=NULL);
	return trl_tbl[idx];
}

static COMMAND_FUNC( do_one_stim )	/** present the stimulus for the next staircase */
{
	Staircase *stc_p;

	stc_p = pick_stair("");
	if( stc_p == NULL ) return;

	present_stim_for_stair(stc_p);
}

static COMMAND_FUNC( do_next_stim )	/** present the stimulus for the next staircase */
{
	Staircase *stc_p;

	stc_p = get_current_stair(&expt1);
	assert(stc_p!=NULL);

	present_stim_for_stair(stc_p);
}

static COMMAND_FUNC( do_next_resp )
{
	Staircase *stc_p;
	int rsp;

	stc_p = get_current_stair(&expt1);
	assert(stc_p!=NULL);

	rsp = get_response(stc_p,&expt1);
	process_response(rsp,stc_p);	// updates staircase and saves data
}

static COMMAND_FUNC( do_one_response )	/** give a response to a staircase and step it! */
{
	int rsp;
	Staircase *stc_p;

	stc_p = pick_stair("");
	if( stc_p == NULL ) return;

	rsp = get_response(stc_p,&expt1);
fprintf(stderr,"do_one_response will call process_response...\n");
	process_response(rsp,stc_p);	// updates staircase and saves data
}

//#define IS_VALID_RESPONSE(r)	( r == REDO_INDEX || r == ABORT_INDEX || r == YES_INDEX || r == NO )
#define IS_VALID_RESPONSE(r)	( r >= 0 && r < N_RESPONSES )

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

	run_stairs(&expt1);
}

static COMMAND_FUNC( do_exp_init )
{
	run_init();
}

static COMMAND_FUNC( set_2afc )
{
	if( askif( "2AFC experiment" ) ){
		advise("Setting 2afc flag");
		advise("Inverting response based on $coin");
		SET_EXPT_FLAG_BITS(&expt1,EXPT_2AFC);
	} else {
		advise("Clearing 2afc flag");
		advise("NOT inverting response based on $coin");
		CLEAR_EXPT_FLAG_BITS(&expt1,EXPT_2AFC);
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

static void revert_to_default_response(Response_Index idx)
{
	if( response_words[idx].custom == NULL ){
		NWARN("CAUTIOUS:  revert_to_default_response:  no custom setting!?");
	} else {
		rls_str(response_words[idx].custom);
		response_words[idx].custom = NULL;
	}
}

static COMMAND_FUNC( do_use_kb )
{
	get_response_from_keyboard = askif("use keyboard for responses");
}

static COMMAND_FUNC( setyesno )
{
	get_rsp_word(YES_INDEX);
	get_rsp_word(NO_INDEX);
	get_rsp_word(REDO_INDEX);
	// Why not allow custom for undo also?

	/* now check that everything is legal! */

	if( check_custom_response(YES_INDEX) < 0 ){
		revert_to_default_response(YES_INDEX);
	}
	if( check_custom_response(NO_INDEX) < 0 ){
		revert_to_default_response(NO_INDEX);
	}
	if( check_custom_response(REDO_INDEX) < 0 ){
		revert_to_default_response(REDO_INDEX);
	}

	return;
}

static COMMAND_FUNC( do_init_block )
{
	init_trial_block(&expt1);
}

static COMMAND_FUNC( do_expt_info )
{
	print_expt_info(&expt1);
}

static COMMAND_FUNC( do_save_data )
{
	FILE *fp;
	const char *s;

	s = nameof("output filename for sequential data");

	fp = try_nice(s,"w");
	if( fp == NULL ) return;	// BUG - should we save somewhere else
					// so that data is not lost???

	save_data(&expt1,fp);
}

static COMMAND_FUNC( do_import_xvals )
{
	Data_Obj *dp;

	dp = pick_obj("float object for x-values");
	if( dp == NULL ) return;

	if( OBJ_PREC(dp) != PREC_SP ){
		sprintf(ERROR_STRING,"import_xvals:  object %s (%s) should have %s precision!?",
			OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)),NAME_FOR_PREC_CODE(PREC_SP));
		warn(ERROR_STRING);
		return;
	}
	if( OBJ_COMPS(dp) != 1 ){
		sprintf(ERROR_STRING,"import_xvals:  object %s should have 1 component!?", OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}
	if( OBJ_ROWS(dp) != 1 ){
		sprintf(ERROR_STRING,"import_xvals:  object %s should have 1 row!?", OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}
	if( OBJ_FRAMES(dp) != 1 ){
		sprintf(ERROR_STRING,"import_xvals:  object %s should have 1 frame!?", OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}
	if( OBJ_COLS(dp) < 2 || OBJ_COLS(dp) > MAX_X_VALUES ){
		sprintf(ERROR_STRING,"import_xvals:  object %s has %d columns, should be in range 2-%d!?",
			OBJ_NAME(dp),OBJ_COLS(dp),MAX_X_VALUES);
		warn(ERROR_STRING);
		return;
	}
	SET_EXPT_XVAL_OBJ(&expt1,dp);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(experiment_menu,s,f,h)

MENU_BEGIN(experiment)
ADD_CMD( info,		do_expt_info,	report information about the experiment )
ADD_CMD( classes,	do_class_menu,	stimulus class submenu )
ADD_CMD( modify,	modify,		modify class parameters )
ADD_CMD( parameters,	do_exp_param_menu,		modify experiment parameters )
ADD_CMD( test,		do_test_stim,	test a condition )
ADD_CMD( use_keyboard,	do_use_kb,	enable/disable use of keyboard for responses )
ADD_CMD( init,		do_exp_init,	start new experiment )
ADD_CMD( present_trial,	do_one_trial,	present a stimulus & save data )
ADD_CMD( present_stim,	do_one_stim,	present a stimulus without collecting response)
ADD_CMD( present_next,	do_next_stim,	present the stimulus for the next staircase)
ADD_CMD( respond_next,	do_next_resp,	collect the response for the next staircase)
ADD_CMD( init_block,	do_init_block,	create a randomized order of staircase trials )
ADD_CMD( response,	do_one_response,	specify the response for the preceding stimulus)
ADD_CMD( save,		do_save_data,	save sequential data to a file )
ADD_CMD( finish,	do_save_data,	close data files )

/*
ADD_CMD( staircase_menu,	do_staircase_menu,	staircase submenu )
ADD_CMD( edit_stairs,	staircase_menu,	edit individual staircases )
*/

ADD_CMD( staircases,	do_staircase_menu,	edit individual staircases )

ADD_CMD( run,		do_run_exp,	run experiment )
ADD_CMD( 2AFC,		set_2afc,	set forced choice flag )
ADD_CMD( keys,		setyesno,	select response keys )
ADD_CMD( import_xvals,	do_import_xvals,	specify data object to use for x-values )
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

