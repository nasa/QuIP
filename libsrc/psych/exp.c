#include "quip_config.h"

char VersionId_psych_exp[] = QUIP_VERSION_STRING;

/*	this is a general package for running experiments
 *
 *	The responsibilities of the caller of exprmnt() :
 *
 *		define the following global fuction ptrs:
 *		int (*stmrt)(), (*modrt)();
 *		initrt points to a routine which is called before each run
 *
 *	stimrt pts to a routine called with two integer args: class, val
 *	modrt pts to a routine to modify stimulus parameters
 */

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* prototype for sync() */
#endif

#include "param_api.h"
#include "stc.h"
#include "savestr.h"
#include "getbuf.h"
#include "rn.h"
#include "substr.h"
#include "query.h"
#include "version.h"

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
static int dribbling=1;
static char rsp_tbl[N_RESPONSES][64];
static const char *rsp_list[N_RESPONSES];
static int rsp_inited=0;
static int n_prel, n_data;

static void null_init(){}
static void null_mod(QSP_ARG_DECL int c){}

static COMMAND_FUNC( modify );
static COMMAND_FUNC( do_trial );
static COMMAND_FUNC( demo );
static COMMAND_FUNC( show_stim );
static COMMAND_FUNC( set_nstairs );
static COMMAND_FUNC( addcnd );
static COMMAND_FUNC( runexp );
static COMMAND_FUNC( set_dribble_flag );
static COMMAND_FUNC( setyesno );
static COMMAND_FUNC( use_keyboard );

static int chkout(SINGLE_QSP_ARG_DECL);
static void setup_files(SINGLE_QSP_ARG_DECL);
static int present_stim(QSP_ARG_DECL int c,int v,Staircase *stcp);
static void do_rspinit(void);

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

static int present_stim(QSP_ARG_DECL int c,int v,Staircase *stcp)
{
	int rsp=REDO;

	if( chkout(SINGLE_QSP_ARG) == -1 ) return(-1);

#ifdef CAUTIOUS
	if( v<0 || v>=_nvals ){
		WARN("CAUTIOUS:  level out of range");
		return(-1);
	}
#endif

	rsp=(*stmrt)(QSP_ARG c,v,stcp);

	return(rsp);
}

static COMMAND_FUNC( do_trial )	/** present a stimulus, tally response */
{
	int c,v;
	int rsp;
	Staircase st1;

	c=(int)HOW_MANY("stimulus class");
	v=(int)HOW_MANY("level");

	rsp=present_stim(QSP_ARG c,v,NULL);

	/* make a dummy staircase */
	st1.stc_clp = index_class(QSP_ARG  c);
	st1.stc_index = 0;
	st1.stc_val = v;
	st1.stc_crctrsp = YES;
	st1.stc_incrsp = YES;
	st1.stc_type = UP_DOWN;
	st1.stc_inc = 1;

	save_response(QSP_ARG  rsp,&st1);
}

static COMMAND_FUNC( demo )		/** demo a stimulus for this experiment */
{
	int c,v,r;

	c=(int)HOW_MANY("stimulus class");
	v=(int)HOW_MANY("level");

	r=present_stim(QSP_ARG c,v,NULL);
}

static COMMAND_FUNC( show_stim )	/** demo a stimulus but don't get response */
{
	int c,v,r;

	c=(int)HOW_MANY("stimulus class");
	v=(int)HOW_MANY("level");

	get_response_from_keyboard=0;
	r=present_stim(QSP_ARG c,v,NULL);
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

static COMMAND_FUNC( set_nstairs )	/** set up experiment */
{
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
	while(chngp_ing) do_cmd(SINGLE_QSP_ARG);

	new_exp(SINGLE_QSP_ARG);
	make_staircases(SINGLE_QSP_ARG);
}


static COMMAND_FUNC( addcnd )
{
	Trial_Class *clp;

	clp = new_class(SINGLE_QSP_ARG);
	(*modrt)(QSP_ARG clp->cl_index);
}

static int chkout(SINGLE_QSP_ARG_DECL)	/* make sure there is something to run */
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

static COMMAND_FUNC( runexp )
{
	if( chkout(SINGLE_QSP_ARG) == -1 ) return;

	setup_files(SINGLE_QSP_ARG);

	_run_stairs(QSP_ARG  n_prel,n_data);
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

	if( dribbling ){
		while( (fp=TRYNICE(NAMEOF("dribble data file"),"w"))
			== NULL ) ;
		set_dribble_file(fp);
		wt_top(QSP_ARG  fp);		/* write header w/ #classes & xvalues */
		markdrib(fp);		/* identify this as dribble data */
	} else {
		while( (fp=TRYNICE(NAMEOF("summary data file"),"w"))
			== NULL ) ;
		set_summary_file(fp);
	}
}

/* make the staircases specified by the parameter table */

void make_staircases(SINGLE_QSP_ARG_DECL)
{
	int j;
	List *lp;
	Node *np;
	Trial_Class *clp;

	lp=class_list(SINGLE_QSP_ARG);
#ifdef CAUTIOUS
	if( lp == NO_LIST ){
		WARN("CAUTIOUS:  no conditions in make_staircases()");
		return;
	}
#endif

	np=lp->l_head;
	while(np!=NO_NODE){
		int i;

		clp=(Trial_Class *)np->n_data;
		np=np->n_next;
		i=clp->cl_index;

		/* makestair( type, class, mininc, correct rsp, inc rsp ); */
		for( j=0;j<n_updn;j++)
			makestair( QSP_ARG  UP_DOWN, i, 1, YES, YES );
		for( j=0;j<n_dnup;j++)
			makestair( QSP_ARG  UP_DOWN, i, -1, YES, YES );
		for(j=0;j<n_2up;j++)
			/*
			 * 2-up increases val after 2 YES's,
			 * decreases val after 1 NO
			 * seeks 71% YES, YES decreasing with val
			 */
			makestair( QSP_ARG  TWO_TO_ONE, i, 1, YES, YES );
		for(j=0;j<n_2dn;j++)
			/*
			 * 2-down decreases val after 2 NO's,
			 * increases val after 1 YES
			 * Seeks 71% NO, NO increasing with val
			 */
			makestair( QSP_ARG  TWO_TO_ONE, i, -1, YES, NO );
		for(j=0;j<n_2iup;j++)
			/*
			 * 2-inverted-up decreases val after 2 YES's,
			 * increases val after 1 NO
			 * Seeks 71% YES, YES increasing with val
			 */
			makestair( QSP_ARG  TWO_TO_ONE, i, -1, YES, YES );
		for(j=0;j<n_2idn;j++)
			/*
			 * 2-inverted-down increases val after 2 NO's,
			 * decreases val after 1 YES
			 * Seeks 71% NO, NO decreasing with val
			 */
			makestair( QSP_ARG  TWO_TO_ONE, i, 1, YES, NO );
		for(j=0;j<n_3up;j++)
			makestair( QSP_ARG  THREE_TO_ONE, i, 1, YES, YES );
		for(j=0;j<n_3dn;j++)
			makestair( QSP_ARG  THREE_TO_ONE, i, -1, YES, NO );
	}
}

COMMAND_FUNC( delcnds )
{
	List *lp;
	Node *np,*next;
	Trial_Class *clp;

	lp=class_list(SINGLE_QSP_ARG);
	if( lp==NO_LIST ) return;

	np=lp->l_head;
	while(np!=NO_NODE){
		clp=(Trial_Class *)np->n_data;
		next = np->n_next;	/* del_class messes with the nodes... */
		del_class(QSP_ARG  clp);
		np=next;
	}
	/* new_exp(); */
}

static COMMAND_FUNC( set_dribble_flag )
{
	dribbling = ASKIF("Record trial-by-trial data");

	if( dribbling )
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

COMMAND_FUNC( do_creat_stc )
{
	Trial_Class *clp;
	int c;

	c=HOW_MANY("stimulus class");

	clp = class_for(QSP_ARG  c);

#ifdef CAUTIOUS
	if( clp == NO_CLASS )
		ERROR1("CAUTIOUS:  do_creat_stc:  error creating stimulus class");
#endif /* CAUTIOUS */

	
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

Command salac_ctbl[]={
{ "create",	do_creat_stc,	"create a staircase"			},
{ "edit",	staircase_menu,	"edit individual staircases"		},
{ "get_value",	do_get_value,	"get current level of a staircase"	},
{ "xvals",	xval_menu,	"x value submenu"			},
{ "use_keyboard",use_keyboard,	"enable/disable use of keyboard for responses"	},
{ "quit",	popcmd,		"quit"					},
{ NULL,		NULL,		NULL					}
};

COMMAND_FUNC( salac_menu )
{
	if( !rsp_inited ){
		do_rspinit();
		rninit(SINGLE_QSP_ARG);
		new_exp(SINGLE_QSP_ARG);
		auto_version(QSP_ARG  "PSYCH","VersionId_psych");
	}
	PUSHCMD( salac_ctbl, "staircases" );
}

Command expctbl[]={
{ "add",	addcnd,		"add a condition"			},
{ "modify",	modify,		"modify parameters"			},
{ "test",	demo,		"test a condition"			},
{ "test_stim",	show_stim,	"test with scripted response"		},
{ "init",	do_exp_init,	"start new experiment"			},
{ "present",	do_trial,	"present a stimulus & save data"	},
{ "finish",	savdat,		"close data files"			},
{ "staircases",	set_nstairs,	"set up staircases"			},
{ "run",	runexp,		"run experiment"			},
{ "2AFC",	set_2afc,	"set forced choice flag"		},
{ "delete",	delcnds,	"delete all conditions"			},
{ "keys",	setyesno,	"select response keys"			},
{ "xvals",	xval_menu,	"x value submenu"			},
{ "dribble",	set_dribble_flag,"set long/short data file format"	},
{ "edit_stairs",staircase_menu,	"edit individual staircases"		},
{ "lookit",	lookmenu,	"data analysis submenu"			},
{ "feedback",	do_feedback,	"specify feedback strings"		},
{ "quit",	popcmd,		"quit"					},
{ NULL,		NULL,		NULL					}
};

#ifdef FOOBAR
COMMAND_FUNC( exprmnt )	/** play around with an experiment */
{
	exp_menu(SINGLE_QSP_ARG);
	while(1) do_cmd(SINGLE_QSP_ARG);
}
#endif /* FOOBAR */

COMMAND_FUNC( exp_menu )
{
	if( !rsp_inited ){
		do_rspinit();
		rninit(SINGLE_QSP_ARG);
		new_exp(SINGLE_QSP_ARG);
		auto_version(QSP_ARG  "PSYCH","VersionId_psych");
	}
	PUSHCMD(expctbl,EXP_MENU_PROMPT);
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
		PUSH_INPUT_FILE("/dev/tty");
		redir( QSP_ARG tfile(SINGLE_QSP_ARG) );	/* get response from keyboard */
	}


	do {
		n=WHICH_ONE2(rpmtstr,N_RESPONSES,rsp_list);
	} while( n < 0 );

	if( get_response_from_keyboard )
		popfile(SINGLE_QSP_ARG);		/* back to default input */

	switch(n){
		case YES_INDEX:		return(YES); break;
		case NO_INDEX:		return(NO); break;
		case REDO_INDEX:	return(REDO); break;
		case ABORT_INDEX:	Abort=1; return(REDO); break;
#ifdef CAUTIOUS
		default: ERROR1("CAUTIOUS:  response:  crazy response value");
#endif /* CAUTIOUS */
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

