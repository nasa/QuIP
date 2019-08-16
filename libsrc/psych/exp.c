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

static void null_init(void);
static void null_mod(QSP_ARG_DECL Trial_Class *);
static void set_rsp_word(const char **sptr,const char *s,const char *default_str);

/* these two global vars ought to be declared in a .h file ... */
void (*initrt)(void)=null_init;
void (*modrt)(QSP_ARG_DECL Trial_Class *)=null_mod;

static char rsp_tbl[N_RESPONSES][64];

static int custom_keys=0;
//static int dribbling=1;

static const char *response_list[N_RESPONSES];

static void null_init(void)
{
	NADVISE("null_init...");
}

static void null_mod(QSP_ARG_DECL Trial_Class * tc_p){}

// In the old days we only saved to files...
// Now we want to be able to import other data for fitting???

void _setup_files(QSP_ARG_DECL  Experiment *exp_p)
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

	if( IS_DRIBBLING(exp_p) ){
		init_dribble_file(SINGLE_QSP_ARG);
	} else {
		while( (fp=try_nice(nameof("summary data file"),"w")) == NULL )
			;
		set_summary_file(fp);
	}
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


static void init_response_prompt(char *target_prompt_string,const char *question_string)
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

#define collect_response(exp_p) _collect_response(QSP_ARG  exp_p)

static int _collect_response(QSP_ARG_DECL  Experiment * exp_p)
{
	int n;
	char rpmtstr[128];	// BUG? possible buffer overflow?

	init_response_prompt(rpmtstr,EXPT_QUESTION(exp_p));

fprintf(stderr,"collect_response:  get_response_from_keyboard = %d\n",IS_USING_KEYBOARD(exp_p));
	if( IS_USING_KEYBOARD(exp_p) ){
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

	if( IS_USING_KEYBOARD(exp_p) ){
		pop_file();		/* back to default input */
	}

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

#define consider_coin(stc_p) _consider_coin(QSP_ARG  stc_p)

static void _consider_coin(QSP_ARG_DECL  Staircase *stc_p)
{
	Variable *vp;

	/* stimulus routine may have changed value of coin */
	int coin;
	vp=var_of("coin");
	if( vp == NULL ){
		warn("variable \"coin\" not set!!!");
		coin=0;
	} else {
		if( sscanf(VAR_VALUE(vp),"%d",&coin) != 1 )
		warn("error scanning integer from variable \"coin\"\n");
	}

	assert( stc_p != NULL );
        
       	// analyzer complains coin is a garbage value??? BUG?

	if( coin ){
		SET_STAIR_CRCT_RSP(stc_p,NO);
	} else {
		SET_STAIR_CRCT_RSP(stc_p,YES);
	}
}

int _get_response(QSP_ARG_DECL  Staircase *stc_p, Experiment *exp_p)
{
	Variable *vp;
	int rsp;

	if( stc_p == NULL ){
		warn("get_response passed null staircase!?");
		return 0;
	}

	// BUG instead of getting this from a variable,
	// better to set in in the experiment struct...
	vp=var_of("response_string");
	SET_EXPT_QUESTION( exp_p, VAR_VALUE(vp) );

	if( vp != NULL ){
		rsp = collect_response(exp_p);
	} else {
		static int warned=0;

		if( !warned ){
			warn("default_stim:  script variable $response_string not defined");
			warned=1;
		}
		rsp = collect_response(exp_p);
	}
	if( is_fc ){
fprintf(stderr,"get_response:  will consider coin...\n");
		consider_coin(stc_p);
	} else {
fprintf(stderr,"get_response:  will NOT consider coin...\n");
	}

	return rsp;
}

void _delete_all_trial_classes(SINGLE_QSP_ARG_DECL)
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

void _init_responses(SINGLE_QSP_ARG_DECL)
{
	static int rsp_inited=0;
	int i;

	if( rsp_inited ){
		warn("redundant call to init_responses!?");
		return;
	}

	strcpy(rsp_tbl[YES_INDEX],RSP_YES);
	strcpy(rsp_tbl[NO_INDEX],RSP_NO);
	strcpy(rsp_tbl[REDO_INDEX],RSP_REDO);
	strcpy(rsp_tbl[ABORT_INDEX],RSP_ABORT);

	// BUG why have response_list AND rsp_tbl ???
	for(i=0;i<N_RESPONSES;i++)
		response_list[i] = rsp_tbl[i];

	rsp_inited=1;
}

void init_experiment( Experiment *exp_p )
{
	exp_p->expt_flags		= 0;
	exp_p->question_string		= NULL;
	exp_p->n_preliminary_trials	= 0;
	exp_p->n_recorded_trials	= 0;

	exp_p->n_updn_stairs		= 0;
	exp_p->n_dnup_stairs		= 0;
	exp_p->n_2iup_stairs		= 0;
	exp_p->n_2idn_stairs		= 0;
	exp_p->n_2up_stairs		= 0;
	exp_p->n_2dn_stairs		= 0;
	exp_p->n_3up_stairs		= 0;
	exp_p->n_3dn_stairs		= 0;
}

