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

/* these two global vars ought to be declared in a .h file ... */
void (*initrt)(void)=null_init;
void (*modrt)(QSP_ARG_DECL Trial_Class *)=null_mod;

Response_Word response_words[N_RESPONSES];	// a global

static void null_init(void)
{
	NADVISE("null_init...");
}

static void null_mod(QSP_ARG_DECL Trial_Class * tc_p){}

// In the old days we only saved to files...
// Now we want to be able to import other data for fitting???
//
// BUG - this will prompt for a summary filename until a good name is entered!?
// It is not clear why we need to store summary data, as it can be reconstituted
// from sequential data...
//
// BUG init_dribble_file will prompt for a filename too, so this function doesn't
// take a fixed number of arguments!?

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


static void set_rsp_word(Response_Index idx, const char *s)
{
	if( response_words[idx].custom != NULL )
		rls_str(response_words[idx].custom);

	response_words[idx].custom = savestr(s);
}


void _get_rsp_word(QSP_ARG_DECL Response_Index idx)
{
	char pmpt[LLEN];
	const char *s;

	sprintf(pmpt,"custom '%s' response",response_words[idx].dflt);
	s=nameof(pmpt);

	// only confirm if interactive...
	if( intractive() ){
		sprintf(pmpt,"use \"%s\" for '%s' response",s,response_words[idx].dflt);
		if( !confirm(pmpt) ) return;
	}

	set_rsp_word(idx,s);
}

static void get_response_parts(int *ip, const char **sp, Response_Index idx)
{
	const char *s;
	s = response_words[idx].custom;
	if( s == NULL ){
		s = response_words[idx].dflt;
	}
	*ip = s[0];
	*sp = s+1;
}

static void init_response_prompt(char *target_prompt_string,const char *question_string)
{
	int c1_yes, c1_no, c1_redo;
	const char *s2_yes, *s2_no, *s2_redo;

	get_response_parts(&c1_yes,&s2_yes,YES_INDEX);
	get_response_parts(&c1_no,&s2_no,NO_INDEX);
	get_response_parts(&c1_redo,&s2_redo,REDO_INDEX);

	sprintf(target_prompt_string,
"%s? [(%c)%s (yes), (%c)%s (no), (%c)%s (redo), (a)bort, (u)ndo previous finger error] : ",
		question_string, c1_yes,s2_yes, c1_no,s2_no, c1_redo,s2_redo);
}

static const char *current_response_word(int idx)
{
	return response_words[idx].custom == NULL ?
		response_words[idx].dflt :
		response_words[idx].custom ;
}

/* BUG
 * if we set the redo char to 'r' after we have alread run
 * (so that "redo" is in the history list, then the response
 * handler no longer accepts "redo" !?
 */

int _check_custom_response(QSP_ARG_DECL  int rsp_idx)
{
	int idx;
	const char *r;

	r = current_response_word(rsp_idx);

	for(idx=0;idx<N_RESPONSES;idx++){
		if( rsp_idx != idx ){
			const char *w;
			w = current_response_word(idx);
			if( r[0] == w[0] ){
				sprintf(ERROR_STRING,
	"Responses '%s' (%d) and '%s' (%d) must differ in the 1st character!?",
					r,rsp_idx, w,idx);
				warn(ERROR_STRING);
				return -1;
			}
		}
	}
	return 0;
}

static const char ** init_response_choices(void)
{
	const char ** s_arr;
	int idx;

	s_arr = getbuf( sizeof(const char *) * N_RESPONSES );
	for(idx=0;idx<N_RESPONSES;idx++){
		s_arr[idx] = current_response_word(idx);
	}
	return s_arr;
}

#define collect_response(exp_p) _collect_response(QSP_ARG  exp_p)

static int _collect_response(QSP_ARG_DECL  Experiment * exp_p)
{
	static const char **response_choices=NULL;
	int n;
	char rpmtstr[128];	// BUG? possible buffer overflow?

	init_response_prompt(rpmtstr,EXPT_QUESTION(exp_p));

fprintf(stderr,"collect_response:  using_keyboard = %d\n",IS_USING_KEYBOARD(exp_p));
	if( IS_USING_KEYBOARD(exp_p) ){
#ifndef BUILD_FOR_OBJC
		redir( tfile(), "/dev/tty" );	/* get response from keyboard */
#else // BUILD_FOR_OBJC
		warn("response (exp.c):  can't get response from keyboard!?");
#endif // BUILD_FOR_OBJC
	}

	if( response_choices == NULL ) {
		// BUG with this as a local static var,
		// we can't ever update it!?
		response_choices = init_response_choices();
	}

	do {
		inhibit_next_prompt_format(SINGLE_QSP_ARG);	// prompt already formatted!
		n = which_one(rpmtstr,N_RESPONSES,response_choices);
		enable_prompt_format(SINGLE_QSP_ARG);
	} while( n < 0 );

	if( IS_USING_KEYBOARD(exp_p) ){
		pop_file();		/* back to default input */
	}

	switch(n){
		case YES_INDEX:		return(n); break;
		case NO_INDEX:		return(n); break;
		case REDO_INDEX:	return(n); break;
		case UNDO_INDEX:	return(n); break;
		case ABORT_INDEX:	Abort=1; return(REDO_INDEX); break;
		default:
			assert( AERROR("response:  crazy response value") );
	}
	/* should never be reached */
	return(ABORT_INDEX);
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
		SET_STAIR_CRCT_RSP(stc_p,NO_INDEX);
	} else {
		SET_STAIR_CRCT_RSP(stc_p,YES_INDEX);
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

#define INIT_RESPONSE_WORD(idx,dflt_val)				\
	response_words[idx].code = idx;					\
	response_words[idx].dflt = dflt_val;				\
	response_words[idx].custom = NULL;

void _init_responses(SINGLE_QSP_ARG_DECL)
{
	static int rsp_inited=0;

	if( rsp_inited ){
		warn("redundant call to init_responses!?");
		return;
	}

	INIT_RESPONSE_WORD(YES_INDEX,RSP_YES)
	INIT_RESPONSE_WORD(NO_INDEX,RSP_NO)
	INIT_RESPONSE_WORD(REDO_INDEX,RSP_REDO)
	INIT_RESPONSE_WORD(ABORT_INDEX,RSP_ABORT)
	INIT_RESPONSE_WORD(UNDO_INDEX,RSP_UNDO)

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

