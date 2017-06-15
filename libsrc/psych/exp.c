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
static void null_mod(QSP_ARG_DECL int);
static void set_rsp_word(const char **sptr,const char *s,const char *default_str);

/* these two global vars ought to be declared in a .h file ... */
void (*initrt)(void)=null_init;
void (*modrt)(QSP_ARG_DECL int)=null_mod;




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
static const char *rsp_list[N_RESPONSES];
static int rsp_inited=0;
static int n_prel, n_data;

static int class_index=0;

static void null_init(void)
{
	NADVISE("null_init...");
}

static void null_mod(QSP_ARG_DECL int c){}

static COMMAND_FUNC( modify );
static COMMAND_FUNC( do_trial );
static COMMAND_FUNC( demo );
static COMMAND_FUNC( set_nstairs );
static COMMAND_FUNC( set_dribble_flag );
static COMMAND_FUNC( setyesno );
static COMMAND_FUNC( use_keyboard );

//static int present_stim(QSP_ARG_DECL int c,int v,Staircase *stcp);
static void do_rspinit(void);

#define INSIST_XVALS						\
								\
	if( _nvals <= 0 ){					\
		sprintf(ERROR_STRING,				\
	"Need to initialize x values (n=%d)",_nvals);		\
		WARN(ERROR_STRING);				\
		return;						\
	}

static void do_rspinit()
{
	int i;

	strcpy(rsp_tbl[YES_INDEX],RSP_YES);
	strcpy(rsp_tbl[NO_INDEX],RSP_NO);
	strcpy(rsp_tbl[REDO_INDEX],RSP_REDO);
	strcpy(rsp_tbl[ABORT_INDEX],RSP_ABORT);
	for(i=0;i<N_RESPONSES;i++) rsp_list[i] = rsp_tbl[i];
	rsp_inited=1;
}

static COMMAND_FUNC( modify )
{
	unsigned int n;

	if( modrt==null_mod ) ERROR1("pointer modrt must be defined by user");
	n=(unsigned int)HOW_MANY("condition index");
	if( n >= eltcount(class_list(SINGLE_QSP_ARG)) ) WARN("undefined condition");
	else (*modrt)(QSP_ARG n);
}

static int insure_exp_is_ready(SINGLE_QSP_ARG_DECL)	/* make sure there is something to run */
{
	if( eltcount(class_list(SINGLE_QSP_ARG)) <= 0 ){
		WARN("no conditions defined");
		return(-1);
	}
	if( _nvals <= 0 ){
		sprintf(ERROR_STRING,"Need to initialize x values (n=%d)",_nvals);
		WARN(ERROR_STRING);
		return(-1);
	}
	return(0);
}

static int present_stim(QSP_ARG_DECL Trial_Class *tcp,int v,Staircase *stcp)
{
	int rsp=REDO;

	if( insure_exp_is_ready(SINGLE_QSP_ARG) == -1 ) return(-1);

	assert( v >= 0 && v < _nvals );

	rsp=(*stmrt)(QSP_ARG tcp,v,stcp);

	return(rsp);
}

static COMMAND_FUNC( do_trial )	/** present a stimulus, tally response */
{
	short v;
	int rsp;
	Staircase st1;
	Trial_Class *tcp;

	//c=(short)HOW_MANY("stimulus class");
	tcp = PICK_TRIAL_CLASS("");
	v=(short)HOW_MANY("level");

	if( tcp == NULL ) return;

	rsp=present_stim(QSP_ARG tcp,v,NULL);

	/* make a dummy staircase */
	SET_STAIR_CLASS(&st1, tcp);
	SET_STAIR_INDEX(&st1, 0);
	SET_STAIR_VAL(&st1, v);
	SET_STAIR_CRCT_RSP(&st1, YES);
	SET_STAIR_INC_RSP(&st1, YES);
	SET_STAIR_TYPE(&st1, UP_DOWN);
	SET_STAIR_INC(&st1, 1);

	save_response(QSP_ARG  rsp,&st1);
}

#define IS_VALID_RESPONSE(r)	r == REDO || r == ABORT || r == YES || r == NO

static COMMAND_FUNC( demo )		/** demo a stimulus for this experiment */
{
	int v,r;
	Trial_Class *tcp;

	//c=(int)HOW_MANY("stimulus class");
	tcp = PICK_TRIAL_CLASS("");
	v=(int)HOW_MANY("level");

	if( tcp == NULL ) return;

	r=present_stim(QSP_ARG tcp,v,NULL);
	assert( IS_VALID_RESPONSE(r) );
}

static COMMAND_FUNC( show_stim )	/** demo a stimulus but don't get response */
{
	int v,r;
	Trial_Class *tcp;

	//c=(int)HOW_MANY("stimulus class");
	tcp = PICK_TRIAL_CLASS("");
	v=(int)HOW_MANY("level");

	if( tcp == NULL ) return;

	INSIST_XVALS

	get_response_from_keyboard=0;
	r=present_stim(QSP_ARG tcp,v,NULL);
	assert( IS_VALID_RESPONSE(r) );

	get_response_from_keyboard=1;
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

static void make_staircases(SINGLE_QSP_ARG_DECL)
{
	int j;
	List *lp;
	Node *np;
	Trial_Class *tcp;

	lp=class_list(SINGLE_QSP_ARG);
	assert( lp != NULL );

	np=QLIST_HEAD(lp);
	while(np!=NULL){
		tcp=(Trial_Class *)np->n_data;
		np=np->n_next;

		/* makestair( type, class, mininc, correct rsp, inc rsp ); */
		for( j=0;j<n_updn;j++)
			makestair( QSP_ARG  UP_DOWN, tcp, 1, YES, YES );
		for( j=0;j<n_dnup;j++)
			makestair( QSP_ARG  UP_DOWN, tcp, -1, YES, YES );
		for(j=0;j<n_2up;j++)
			/*
			 * 2-up increases val after 2 YES's,
			 * decreases val after 1 NO
			 * seeks 71% YES, YES decreasing with val
			 */
			makestair( QSP_ARG  TWO_TO_ONE, tcp, 1, YES, YES );
		for(j=0;j<n_2dn;j++)
			/*
			 * 2-down decreases val after 2 NO's,
			 * increases val after 1 YES
			 * Seeks 71% NO, NO increasing with val
			 */
			makestair( QSP_ARG  TWO_TO_ONE, tcp, -1, YES, NO );
		for(j=0;j<n_2iup;j++)
			/*
			 * 2-inverted-up decreases val after 2 YES's,
			 * increases val after 1 NO
			 * Seeks 71% YES, YES increasing with val
			 */
			makestair( QSP_ARG  TWO_TO_ONE, tcp, -1, YES, YES );
		for(j=0;j<n_2idn;j++)
			/*
			 * 2-inverted-down increases val after 2 NO's,
			 * decreases val after 1 YES
			 * Seeks 71% NO, NO decreasing with val
			 */
			makestair( QSP_ARG  TWO_TO_ONE, tcp, 1, YES, NO );
		for(j=0;j<n_3up;j++)
			makestair( QSP_ARG  THREE_TO_ONE, tcp, 1, YES, YES );
		for(j=0;j<n_3dn;j++)
			makestair( QSP_ARG  THREE_TO_ONE, tcp, -1, YES, NO );
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
	make_staircases(SINGLE_QSP_ARG);
}

// was addcnd

static COMMAND_FUNC( do_new_class )
{
	const char *name, *cmd;
	Trial_Class *tcp;

	name = NAMEOF("nickname for this class");
	cmd = NAMEOF("string to execute for this stimulus class");

	// Make sure not in use
	tcp = trial_class_of(QSP_ARG  name);
	if( tcp != NULL ){
		sprintf(ERROR_STRING,"Class name \"%s\" is already in use!?",
			name);
		WARN(ERROR_STRING);
		return;
	}

	tcp = new_trial_class(QSP_ARG  name );
	SET_CLASS_CMD(tcp,savestr(cmd));
	SET_CLASS_INDEX(tcp,class_index++);
	SET_CLASS_DATA_TBL(tcp,NULL);
	SET_CLASS_N_STAIRS(tcp,0);

	if( _nvals > 0 ){
		alloc_data_tbl(tcp,_nvals);
	} else {
		WARN("need to specify x-values before declaring stimulus class!?");
		SET_CLASS_DATA_TBL(tcp,NULL);
	}

	//(*modrt)(QSP_ARG tcp->cl_index);
}

static void setup_files(SINGLE_QSP_ARG_DECL)
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
		while( (fp=TRYNICE(NAMEOF("summary data file"),"w"))
			== NULL ) ;
		set_summary_file(fp);
	}
advise("setup_files DONE");
}

static COMMAND_FUNC( do_run_exp )
{
	if( insure_exp_is_ready(SINGLE_QSP_ARG) == -1 ) return;

	setup_files(SINGLE_QSP_ARG);

fprintf(stderr,"calling _run_stairs %d %d\n",n_prel,n_data);
	_run_stairs(QSP_ARG  n_prel,n_data);
}

COMMAND_FUNC( do_delete_all_classes )
{
	List *lp;
	Node *np,*next;
	Trial_Class *tcp;

	lp=class_list(SINGLE_QSP_ARG);
	if( lp==NULL ) return;

	np=QLIST_HEAD(lp);
	while(np!=NULL){
		tcp=(Trial_Class *)np->n_data;
		next = np->n_next;	/* del_class messes with the nodes... */
		del_class(QSP_ARG  tcp);
		np=next;
	}
	ASSIGN_RESERVED_VAR( "n_classes" , "0" );
	/* new_exp(); */
}

static COMMAND_FUNC( set_dribble_flag )
{
	if( ASKIF("Record trial-by-trial data") )
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
	setup_files(SINGLE_QSP_ARG);
	_run_init(SINGLE_QSP_ARG);
	set_recording( 1 );
}

COMMAND_FUNC( set_2afc )
{
	if( ASKIF( "2AFC experiment" ) ){
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

	s=NAMEOF("script fragment to interpret for correct feedback");

	if( correct_feedback_string != NULL )
		rls_str(correct_feedback_string);
	correct_feedback_string = savestr(s);

	s=NAMEOF("script fragment to interpret for incorrect feedback");

	if( incorrect_feedback_string != NULL )
		rls_str(incorrect_feedback_string);
	incorrect_feedback_string = savestr(s);

}

static COMMAND_FUNC( do_creat_stc )
{
	Trial_Class *tcp;
	short c;

	c= (short) HOW_MANY("stimulus class");

	tcp = class_for(QSP_ARG  c);

	assert( tcp != NO_CLASS );
}

/* This is an alternative menu for when we want to use a staircase to set levels, but
 * we want to maintain control ourselves...
 * "staircases a la carte"
 */

static COMMAND_FUNC( do_get_value )
{
	const char *s;
	Staircase *stcp;
	char valstr[32];

	s = NAMEOF("name of variable for value storage");
	stcp=PICK_STC( "" );

	if( stcp == NO_STAIR ) return;

	sprintf(valstr,"%g",xval_array[stcp->stc_val]);
	ASSIGN_VAR(s,valstr);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(staircases_menu,s,f,h)

MENU_BEGIN(staircases)
ADD_CMD( create,	do_creat_stc,	create a staircase )
ADD_CMD( edit,		staircase_menu,	edit individual staircases )
ADD_CMD( get_value,	do_get_value,	get current level of a staircase )
ADD_CMD( xvals,		xval_menu,	x value submenu )
ADD_CMD( use_keyboard,	use_keyboard,	enable/disable use of keyboard for responses )
MENU_END(staircases)

static COMMAND_FUNC( do_staircase_menu )
{
	if( !rsp_inited ){
		do_rspinit();
		rninit(SINGLE_QSP_ARG);
		new_exp(SINGLE_QSP_ARG);
	}
	PUSH_MENU( staircases );
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(experiment_menu,s,f,h)

MENU_BEGIN(experiment)
ADD_CMD( class,		do_new_class,	add a stimulus class )
ADD_CMD( modify,	modify,		modify parameters )
ADD_CMD( test,		demo,		test a condition )
ADD_CMD( test_stim,	show_stim,	test with scripted response )
ADD_CMD( init,		do_exp_init,	start new experiment )
ADD_CMD( present,	do_trial,	present a stimulus & save data )
ADD_CMD( finish,	do_save_data,	close data files )
ADD_CMD( staircases,	set_nstairs,	set up staircases )
ADD_CMD( staircase_menu,	do_staircase_menu,	staircase submenu )
ADD_CMD( run,		do_run_exp,	run experiment )
ADD_CMD( 2AFC,		set_2afc,	set forced choice flag )
ADD_CMD( delete,	do_delete_all_classes,	delete all conditions )
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
		rninit(SINGLE_QSP_ARG);
		new_exp(SINGLE_QSP_ARG);
	}
	PUSH_MENU(experiment);
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

static COMMAND_FUNC( use_keyboard )
{
	get_response_from_keyboard = ASKIF("use keyboard for responses");
}

/* BUG
 * if we set the redo char to 'r' after we have alread run
 * (so that "redo" is in the history list, then the response
 * handler no longer accepts "redo" !?
 */

static COMMAND_FUNC( setyesno )
{
	get_rsp_word(QSP_ARG  &rsp_list[YES_INDEX],RSP_YES);
	get_rsp_word(QSP_ARG  &rsp_list[NO_INDEX],RSP_NO);
	get_rsp_word(QSP_ARG  &rsp_list[REDO_INDEX],RSP_REDO);

	/* now check that everything is legal! */

	if( is_a_substring(RSP_ABORT,rsp_list[YES_INDEX]) ||
		is_a_substring(RSP_ABORT,rsp_list[NO_INDEX]) ||
		is_a_substring(RSP_ABORT,rsp_list[REDO_INDEX]) ){

		WARN("conflict with abort response");
		goto bad;
	}
	if( rsp_list[YES_INDEX][0] == rsp_list[NO_INDEX][0] ){
		WARN("yes and no responses must differ in the 1st character");
		goto bad;
	}
	if( rsp_list[YES_INDEX][0] == rsp_list[REDO_INDEX][0] ){
		WARN("yes and redo responses must differ in the 1st character");
		goto bad;
	}
	if( rsp_list[NO_INDEX][0] == rsp_list[REDO_INDEX][0] ){
		WARN("no and redo responses must differ in the 1st character");
		goto bad;
	}
	custom_keys=1;
	return;
bad:
	/* install default responses */
	set_rsp_word(&rsp_list[YES_INDEX],RSP_YES,RSP_YES);
	set_rsp_word(&rsp_list[NO_INDEX],RSP_NO,RSP_NO);
	set_rsp_word(&rsp_list[REDO_INDEX],RSP_REDO,RSP_REDO);
	custom_keys=0;

	return;
}

void get_rsp_word(QSP_ARG_DECL const char **sptr,const char *def_rsp)
{
	char buf[LLEN];
	const char *s;

	sprintf(buf,"word %s response",def_rsp);
	s=NAMEOF(buf);
	sprintf(buf,"use \"%s\" for %s response",s,def_rsp);
	if( !CONFIRM(buf) ) return;

	set_rsp_word(sptr,s,def_rsp);
}

int response(QSP_ARG_DECL  const char *s)
{
	int n;
	char rpmtstr[128];

	init_rps(rpmtstr,s);


	if( get_response_from_keyboard ){
#ifndef BUILD_FOR_OBJC
		redir( QSP_ARG tfile(SINGLE_QSP_ARG), "/dev/tty" );	/* get response from keyboard */
#else // BUILD_FOR_OBJC
		WARN("response (exp.c):  can't get response from keyboard!?");
#endif // BUILD_FOR_OBJC
	}


	do {
		n=WHICH_ONE2(rpmtstr,N_RESPONSES,rsp_list);
	} while( n < 0 );

	if( get_response_from_keyboard )
		pop_file(SINGLE_QSP_ARG);		/* back to default input */

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


void init_rps(char *target,const char *s)
{
	if( custom_keys ){
		sprintf(target,
	"%s? [(%c)%s (yes), (%c)%s (no), (%c)%s (redo), (a)bort] : ",
			s,
			rsp_list[YES_INDEX][0],rsp_list[YES_INDEX]+1,
			rsp_list[NO_INDEX][0],rsp_list[NO_INDEX]+1,
			rsp_list[REDO_INDEX][0],rsp_list[REDO_INDEX]+1
			);
	} else {
		sprintf(target,
	"%s? [(%c)%s, (%c)%s, (%c)%s, (a)bort] : ",
			s,
			rsp_list[YES_INDEX][0],rsp_list[YES_INDEX]+1,
			rsp_list[NO_INDEX][0],rsp_list[NO_INDEX]+1,
			rsp_list[REDO_INDEX][0],rsp_list[REDO_INDEX]+1
			);
	}
}

