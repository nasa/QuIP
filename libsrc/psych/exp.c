#include "quip_config.h"

/*	this is a general package for running experiments
 *
 *	OLD:  this was a library that we linked with, providing our
 *	own c-callable routines for stimulus delivery and parameter
 *	modification.
 *	The responsibilities of the caller of exprmnt() :
 *
 *		define the following global fuction ptrs:
 *		int (*stmrt)(), (*modrt)();
 *		initrt points to a routine which is called before each run
 *
 *	stimrt pts to a routine called with two integer args: class, val
 *	modrt pts to a routine to modify stimulus parameters
 *
 *	NEW:  everything should be script-based.  Instead of providing a
 *	c-callable stimulus routine, for each condition we provide a macro
 *	to be called.
 */

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

static void null_init(void);
static void null_mod(QSP_ARG_DECL Trial_Class *);
static void set_rsp_word(const char **sptr,const char *s,const char *default_str);

/* these two global vars ought to be declared in a .h file ... */
void (*initrt)(void)=null_init;
void (*modrt)(QSP_ARG_DECL Trial_Class *)=null_mod;




#define RSP_YES		"yes"
#define RSP_NO		"no"
#define RSP_REDO	"redo"
#define RSP_ABORT	"abort"
#define YES_INDEX	0
#define NO_INDEX	1
#define REDO_INDEX	2
#define ABORT_INDEX	3
#define N_RESPONSES	4

static int custom_keys=0;
static int get_response_from_keyboard=1;
//static int dribbling=1;
static int exp_flags=0;
#define DRIBBLING	1
#define SET_EXP_FLAG(bit)	exp_flags |= bit
#define CLEAR_EXP_FLAG(bit)	exp_flags &= ~(bit)

#define IS_DRIBBLING	(exp_flags & DRIBBLING)

static char rsp_tbl[N_RESPONSES][64];
static const char *response_list[N_RESPONSES];
static int rsp_inited=0;
static int n_prel, n_data;

static int class_index=0;

static void null_init(void)
{
	NADVISE("null_init...");
}

static void null_mod(QSP_ARG_DECL Trial_Class * tc_p){}

#define INSIST_XVALS						\
								\
	if( CLASS_XVAL_OBJ(tc_p) == NULL ){			\
		sprintf(ERROR_STRING,				\
	"Need to specify x values for class '%s'",		\
			CLASS_NAME(tc_p));			\
		warn(ERROR_STRING);				\
		return;						\
	}

static void do_rspinit()
{
	int i;

	strcpy(rsp_tbl[YES_INDEX],RSP_YES);
	strcpy(rsp_tbl[NO_INDEX],RSP_NO);
	strcpy(rsp_tbl[REDO_INDEX],RSP_REDO);
	strcpy(rsp_tbl[ABORT_INDEX],RSP_ABORT);
	for(i=0;i<N_RESPONSES;i++) response_list[i] = rsp_tbl[i];
	rsp_inited=1;
}

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

#define present_stim(tc_p,v,stc_p) _present_stim(QSP_ARG tc_p,v,stc_p)

static int _present_stim(QSP_ARG_DECL Trial_Class *tc_p,int v,Staircase *stc_p)
{
	int rsp=REDO;

	if( insure_exp_is_ready(SINGLE_QSP_ARG) == -1 ) return(-1);

	assert( CLASS_XVAL_OBJ(tc_p) != NULL );
	assert( v >= 0 && v < OBJ_COLS( CLASS_XVAL_OBJ(tc_p) ) );

	rsp=(*stmrt)(QSP_ARG tc_p,v,stc_p);

	return(rsp);
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

	rsp=present_stim(tc_p,v,NULL);

	INIT_DUMMY_STAIR(st1)

	set_recording(1);
	save_response(rsp,&st1);
}

#define IS_VALID_RESPONSE(r)	r == REDO || r == ABORT || r == YES || r == NO

static COMMAND_FUNC( do_test_stim )		/** demo a stimulus for this experiment */
{
	int v,r;
	Trial_Class *tc_p;

	tc_p = pick_trial_class("");
	v=(int)how_many("level");

	if( tc_p == NULL ) return;

	r=present_stim(tc_p,v,NULL);
	assert( IS_VALID_RESPONSE(r) );
}

static int n_updn;	/** number of up-down stairs */
static int n_dnup;
static int n_2iup, n_2idn, n_2up, n_2dn, n_3up, n_3dn;

struct param expptbl[]={
{ "n_prelim",	"# of preliminary trials per stair (<0 for variable to criterion)",
							INT_PARAM(&n_prel) },
{ "n_data",	"# of recorded trials per stair",	INT_PARAM(&n_data) },
{ "n_updn",	"up-down stairs per cond.",		INT_PARAM(&n_updn) },
{ "n_dnup",	"down-up stairs per cond.",		INT_PARAM(&n_dnup) },
{ "n_2up",	"two-to-one stairs per cond.",		INT_PARAM(&n_2up)  },
{ "n_2dn",	"one-to-two stairs per cond.",		INT_PARAM(&n_2dn)  },
{ "n_2iup",	"inverted two-to-one stairs per cond.",	INT_PARAM(&n_2iup) },
{ "n_2idn",	"inverted one-to-two stairs per cond.",	INT_PARAM(&n_2idn) },
{ "n_3up",	"three-to-one stairs per condition",	INT_PARAM(&n_3up)  },
{ "n_3dn",	"one-to-three stairs per condition",	INT_PARAM(&n_3dn)  },
{ NULL_UPARAM }
};


/* make the staircases specified by the parameter table */

#define make_staircases() _make_staircases(SINGLE_QSP_ARG)

static void _make_staircases(SINGLE_QSP_ARG_DECL)
{
	int j;
	List *lp;
	Node *np;
	Trial_Class *tc_p;

	lp=trial_class_list();
	assert( lp != NULL );

	np=QLIST_HEAD(lp);
	while(np!=NULL){
		tc_p=(Trial_Class *)np->n_data;
		np=np->n_next;

		/* make_staircase( type, class, mininc, correct rsp, inc rsp ); */
		for( j=0;j<n_updn;j++)
			make_staircase( UP_DOWN, tc_p, 1, YES, YES );
		for( j=0;j<n_dnup;j++)
			make_staircase( UP_DOWN, tc_p, -1, YES, YES );
		for(j=0;j<n_2up;j++)
			/*
			 * 2-up increases val after 2 YES's,
			 * decreases val after 1 NO
			 * seeks 71% YES, YES decreasing with val
			 */
			make_staircase( TWO_TO_ONE, tc_p, 1, YES, YES );
		for(j=0;j<n_2dn;j++)
			/*
			 * 2-down decreases val after 2 NO's,
			 * increases val after 1 YES
			 * Seeks 71% NO, NO increasing with val
			 */
			make_staircase( TWO_TO_ONE, tc_p, -1, YES, NO );
		for(j=0;j<n_2iup;j++)
			/*
			 * 2-inverted-up decreases val after 2 YES's,
			 * increases val after 1 NO
			 * Seeks 71% YES, YES increasing with val
			 */
			make_staircase( TWO_TO_ONE, tc_p, -1, YES, YES );
		for(j=0;j<n_2idn;j++)
			/*
			 * 2-inverted-down increases val after 2 NO's,
			 * decreases val after 1 YES
			 * Seeks 71% NO, NO decreasing with val
			 */
			make_staircase( TWO_TO_ONE, tc_p, 1, YES, NO );
		for(j=0;j<n_3up;j++)
			make_staircase( THREE_TO_ONE, tc_p, 1, YES, YES );
		for(j=0;j<n_3dn;j++)
			make_staircase( THREE_TO_ONE, tc_p, -1, YES, NO );
	}
}

static COMMAND_FUNC( set_nstairs )	/** set up experiment */
{
	int d;

	/* BUG?
	 * by clearing out the actual staircases, any
	 * hand edited changes will be lost.
	 * Since the old system didn't include hand editing,
	 * this is OK (at least unlikely to break any old stuff).
	 */

	del_all_stairs(SINGLE_QSP_ARG);	/* clear out old ones */

	chngp(QSP_ARG expptbl);

	/* chngp just pushes the parameter menu...
	 * But we need to executed some routines when we are done.
	 * So we have to duplicate the loop here...
	 */
	d = STACK_DEPTH(QS_MENU_STACK(THIS_QSP));
	while( STACK_DEPTH(QS_MENU_STACK(THIS_QSP)) == d )
		qs_do_cmd(THIS_QSP);

	new_exp(SINGLE_QSP_ARG);
	make_staircases();
}

// was addcnd

static COMMAND_FUNC( do_new_class )
{
	const char *name, *cmd;
	Trial_Class *tc_p;

	name = nameof("nickname for this class");
	cmd = nameof("string to execute for this stimulus class");

	tc_p = create_named_class(name);
	if( tc_p == NULL ) return;

	SET_CLASS_CMD(tc_p,savestr(cmd));
}

static void set_class_xval_obj( Trial_Class *tc_p, Data_Obj *dp )
{
	if( CLASS_XVAL_OBJ(tc_p) != NULL )
		remove_reference(CLASS_XVAL_OBJ(tc_p));

	SET_CLASS_XVAL_OBJ(tc_p,dp);

	if( dp != NULL )
		add_reference(dp);
}

Trial_Class *_create_named_class(QSP_ARG_DECL  const char *name)
{
	Trial_Class *tc_p;
	Summary_Data_Tbl *sdt_p;

	// Make sure not in use
	tc_p = trial_class_of(name);
	if( tc_p != NULL ){
		sprintf(ERROR_STRING,"Class name \"%s\" is already in use!?",
			name);
		warn(ERROR_STRING);
		return NULL;
	}

	tc_p = new_trial_class(name );
	SET_CLASS_INDEX(tc_p,class_index++);
	SET_CLASS_N_STAIRS(tc_p,0);

	SET_CLASS_XVAL_OBJ(tc_p,NULL);			// so we don't un-reference garbage
	set_class_xval_obj(tc_p,global_xval_dp);	// may be null

	sdt_p = new_summary_data_tbl();
	init_summ_dtbl_for_class(sdt_p,tc_p);

	SET_CLASS_SEQ_DTBL(tc_p,new_sequential_data_tbl());
	SET_SEQ_DTBL_CLASS( CLASS_SEQ_DTBL(tc_p), tc_p );


	SET_CLASS_CMD(tc_p, NULL);

	assert( CLASS_SUMM_DTBL(tc_p) != NULL );
	clear_summary_data( CLASS_SUMM_DTBL(tc_p) );
	return tc_p;
}

// In the old days we only saved to files...
// Now we want to be able to import other data for fitting???

#define setup_files() _setup_files(SINGLE_QSP_ARG)

static void _setup_files(SINGLE_QSP_ARG_DECL)
{
	FILE *fp;

	set_summary_file(NULL);
	set_dribble_file(NULL);

	sync();		/* make files safe in case of crash */
			/* a relic of pdp-11 days */

	/*
	 * We don't bother to write a summary data file
	 * if we are writing a trial-by-trial dribble file,
	 * since the former can be constructed from the latter
	 */

	if( IS_DRIBBLING ){
		init_dribble_file(SINGLE_QSP_ARG);
	} else {
		while( (fp=try_nice(nameof("summary data file"),"w")) == NULL )
			;
		set_summary_file(fp);
	}
}

static COMMAND_FUNC( do_run_exp )
{
	if( insure_exp_is_ready(SINGLE_QSP_ARG) == -1 ) return;

	setup_files();

	run_stairs(n_prel,n_data);
}

COMMAND_FUNC( do_clear_all_classes )
{
	clear_all_data_tables();
}

COMMAND_FUNC( do_delete_all_classes )
{
	List *lp;
	Node *np,*next;
	Trial_Class *tc_p;

	lp=trial_class_list();
	if( lp==NULL ) return;

	np=QLIST_HEAD(lp);
	while(np!=NULL){
		tc_p=(Trial_Class *)np->n_data;
		next = np->n_next;	/* del_class messes with the nodes... */
		del_class(tc_p);
		np=next;
	}
	assign_reserved_var( "n_classes" , "0" );
}

static COMMAND_FUNC( set_dribble_flag )
{
	if( askif("Record trial-by-trial data") )
		SET_EXP_FLAG(DRIBBLING);
	else
		CLEAR_EXP_FLAG(DRIBBLING);

	if( IS_DRIBBLING )
		advise("Recording trial-by-trial data");
	else
		advise("Recording only summary data");
}

COMMAND_FUNC( do_exp_init )
{
	setup_files();
	run_init();
	set_recording( 1 );
}

COMMAND_FUNC( set_2afc )
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

// BUG!?!?  - this creates a class, not a staircase???

static COMMAND_FUNC( do_creat_stc )
{
	Trial_Class *tc_p;
	short c;

	c= (short) how_many("stimulus class");

	tc_p = new_class_for_index(c);

	assert( tc_p != NULL );
}

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

static COMMAND_FUNC( do_list_classes )
{
	list_trial_classs( tell_msgfile() );
}

static COMMAND_FUNC( do_class_info )
{
	Trial_Class *tc_p;

	tc_p = pick_trial_class("");
	if( tc_p == NULL ) return;

	advise("Sorry, don't know how to print class info yet!?");
}

static COMMAND_FUNC( do_show_class_summ )
{
	Trial_Class *tc_p;

	tc_p = pick_trial_class("");
	if( tc_p == NULL ) return;

	write_summary_data( CLASS_SUMM_DTBL(tc_p), tell_msgfile() );
}

static COMMAND_FUNC( do_show_class_seq )
{
	Trial_Class *tc_p;

	tc_p = pick_trial_class("");
	if( tc_p == NULL ) return;

	write_sequential_data( CLASS_SEQ_DTBL(tc_p), tell_msgfile() );
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(class_menu,s,f,h)

MENU_BEGIN(class)
ADD_CMD( new,		do_new_class,		add a stimulus class )
ADD_CMD( list,		do_list_classes,	list all stimulus classes )
ADD_CMD( info,		do_class_info,		print info about a class )
ADD_CMD( summary_data,	do_show_class_summ,	print summary data from a class )
ADD_CMD( sequential_data,	do_show_class_seq,	print sequential data from a class )
ADD_CMD( clear_all,	do_clear_all_classes,	clear data for all conditions )
ADD_CMD( delete_all,	do_delete_all_classes,	delete all conditions )
MENU_END(class)

static COMMAND_FUNC( do_class_menu )
{
	CHECK_AND_PUSH_MENU(class);
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

void _get_rsp_word(QSP_ARG_DECL const char **sptr,const char *def_rsp)
{
	char buf[LLEN];
	const char *s;

	sprintf(buf,"word %s response",def_rsp);
	s=nameof(buf);
	sprintf(buf,"use \"%s\" for %s response",s,def_rsp);
	if( !confirm(buf) ) return;

	set_rsp_word(sptr,s,def_rsp);
}


static void init_responses(char *target_prompt_string,const char *question_string)
{
	if( custom_keys ){
		sprintf(target_prompt_string,
	"%s? [(%c)%s (yes), (%c)%s (no), (%c)%s (redo), (a)bort] : ",
			question_string,
			response_list[YES_INDEX][0],response_list[YES_INDEX]+1,
			response_list[NO_INDEX][0],response_list[NO_INDEX]+1,
			response_list[REDO_INDEX][0],response_list[REDO_INDEX]+1
			);
	} else {
		sprintf(target_prompt_string,
	"%s? [(%c)%s, (%c)%s, (%c)%s, (a)bort] : ",
			question_string,
			response_list[YES_INDEX][0],response_list[YES_INDEX]+1,
			response_list[NO_INDEX][0],response_list[NO_INDEX]+1,
			response_list[REDO_INDEX][0],response_list[REDO_INDEX]+1
			);
	}
}

int _collect_response(QSP_ARG_DECL  const char *question_string)
{
	int n;
	char rpmtstr[128];	// BUG? possible buffer overflow?

	init_responses(rpmtstr,question_string);

	if( get_response_from_keyboard ){
#ifndef BUILD_FOR_OBJC
		redir( tfile(), "/dev/tty" );	/* get response from keyboard */
#else // BUILD_FOR_OBJC
		warn("response (exp.c):  can't get response from keyboard!?");
#endif // BUILD_FOR_OBJC
	}


	do {
		inhibit_next_prompt_format(SINGLE_QSP_ARG);	// prompt already formatted!
		n=which_one(rpmtstr,N_RESPONSES,response_list);
		enable_prompt_format(SINGLE_QSP_ARG);
	} while( n < 0 );

	if( get_response_from_keyboard )
		pop_file();		/* back to default input */

	switch(n){
		case YES_INDEX:		return(YES); break;
		case NO_INDEX:		return(NO); break;
		case REDO_INDEX:	return(REDO); break;
		case ABORT_INDEX:	Abort=1; return(REDO); break;
		default:
			assert( AERROR("response:  crazy response value") );
	}
	/* should never be reached */
	return(ABORT);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(staircases_menu,s,f,h)

MENU_BEGIN(staircases)
ADD_CMD( create,	do_creat_stc,	create a staircase )
ADD_CMD( edit,		staircase_menu,	edit individual staircases )
ADD_CMD( get_value,	do_get_value,	get current level of a staircase )
ADD_CMD( xvals,		xval_menu,	x value submenu )
MENU_END(staircases)

static COMMAND_FUNC( do_staircase_menu )
{
	if( !rsp_inited ){
		do_rspinit();
		rninit();
		new_exp(SINGLE_QSP_ARG);
	}
	CHECK_AND_PUSH_MENU( staircases );
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(experiment_menu,s,f,h)

MENU_BEGIN(experiment)
ADD_CMD( classes,	do_class_menu,	stimulus class submenu )
ADD_CMD( modify,	modify,		modify parameters )
ADD_CMD( test,		do_test_stim,	test a condition )
ADD_CMD( use_keyboard,	do_use_kb,	enable/disable use of keyboard for responses )
ADD_CMD( init,		do_exp_init,	start new experiment )
ADD_CMD( present,	do_one_trial,	present a stimulus & save data )
ADD_CMD( finish,	do_save_data,	close data files )
ADD_CMD( staircases,	set_nstairs,	set up staircases )
ADD_CMD( staircase_menu,	do_staircase_menu,	staircase submenu )
ADD_CMD( run,		do_run_exp,	run experiment )
ADD_CMD( 2AFC,		set_2afc,	set forced choice flag )
ADD_CMD( keys,		setyesno,	select response keys )
ADD_CMD( xvals,		xval_menu,	x value submenu )
ADD_CMD( dribble,	set_dribble_flag,	set long/short data file format )
ADD_CMD( edit_stairs,	staircase_menu,	edit individual staircases )
ADD_CMD( lookit,	lookmenu,	data analysis submenu )
ADD_CMD( feedback,	do_feedback,	specify feedback strings )
MENU_END(experiment)

COMMAND_FUNC( do_exp_menu )
{
	if( !rsp_inited ){
		do_rspinit();
		rninit();
		new_exp(SINGLE_QSP_ARG);
	}
	CHECK_AND_PUSH_MENU(experiment);
}

