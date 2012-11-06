#include "quip_config.h"

char VersionId_psych_stair[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include "stc.h"
#include "getbuf.h"
#include "node.h"
#include "savestr.h"
#include "debug.h"
#include "rn.h"
#include "items.h"
#include "query.h"	/* push_input_file() */

/* If CATCH_SIGS is defined, then we can use ^C to interrupt trials...
 *
 * Not sure when this was used...
 */

#ifdef CATCH_SIGS
#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif
int Redo;		/* flags a redo from a software interrupt */
static int caught;
#endif /* CATCH_SIGS */

/* local prototypes */

static int nullstim(QSP_ARG_DECL  int,int,Staircase *);
static void mk_stair_array(SINGLE_QSP_ARG_DECL);
static int prestep(SINGLE_QSP_ARG_DECL);
#define PRESTEP			prestep(SINGLE_QSP_ARG)
static void clear_data(Trial_Class *clp);

static List *stair_list(SINGLE_QSP_ARG_DECL);
static void tally(Staircase *stc,int rsp);
static int iftrans(Staircase *stc,int rsp);
static void adj_inc(Staircase *stc);
static void adj_val(Staircase *stc);
static int step(QSP_ARG_DECL  Staircase *stc);
static void stepall(SINGLE_QSP_ARG_DECL);
static void stair_trials(QSP_ARG_DECL  int np,int nt);

int (*stmrt)(QSP_ARG_DECL  int,int,Staircase *)=nullstim;		/* global var */

/* globals? */
int Abort;				/* stop the run now if set != 0 */
FILE *drib_file=NULL;		/* trial by trial dribble */
const char *correct_feedback_string=NULL, *incorrect_feedback_string=NULL;

static FILE *summ_file=NULL;		/* summary data */
static int recording;
static int prelim;
static int trialno;
static int _trans;		/* possible transition of current trial */
static u_long *order=((u_long *)NULL);	/* permutation buffer */

static short nstairs=0;

#define NO_STAIR_PTR	((Staircase **)NULL)
static Staircase **stair_tbl=NO_STAIR_PTR;

ITEM_INTERFACE_DECLARATIONS(Trial_Class,trial_class)
ITEM_INTERFACE_DECLARATIONS(Staircase,stc)

static List *stair_list(SINGLE_QSP_ARG_DECL)
{
	if( stc_itp==NO_ITEM_TYPE ) return(NO_LIST);
	return(item_list(QSP_ARG  stc_itp) );
}

static int nullstim(QSP_ARG_DECL  int c,int v,Staircase *stcp)
{
	sprintf(error_string,"no stimulus routine");
	WARN(error_string);
	return(0);
}

void new_exp(SINGLE_QSP_ARG_DECL)		/** discard old stairs */
{
	List *lp;
	Node *np;
	Staircase *stcp;

	nstairs=0;

	lp = stair_list(SINGLE_QSP_ARG);
	if( lp==NO_LIST ) return;

	/* Don't we have a routine to delete all staircases? */
	np=lp->l_head;
	while(np!=NO_NODE){
		stcp = (Staircase *)np->n_data;
		np=np->n_next;		/* must do this before freeing item! */
		del_stc(QSP_ARG  stcp->stc_name);
		givbuf((void *)stcp->stc_name);
	}
}

void set_recording(int flag)
{
	recording = flag;
}

static void tally(Staircase *stc,int rsp)			/* record new data */
{ note_trial(stc->stc_clp,stc->stc_val,rsp,stc->stc_crctrsp); }

static void adj_inc(Staircase *stc)	/* adjust the increment */
{

	if( _trans == NO_TR ) return;
	else if( _trans == stc->stc_lasttr ) return;
	stc->stc_lasttr=_trans;

	/*
	 * For two-to-one staircases, we only adjust the increment
	 * when a reversal takes us in the slow (more difficult)
	 * direction. This is done to allow a relatively speedy
	 * exit from random excursions into never-never land
	 */

	if( stc->stc_type != UP_DOWN && _trans == TR_DN ) return;

	/* negative inc means WHAT??? */

	if(stc->stc_inc>0){
		stc->stc_inc++;
		stc->stc_inc /= 2;
		if( stc->stc_inc < stc->stc_mininc )
			stc->stc_inc=stc->stc_mininc;
	} else {	/* negative increment */
		stc->stc_inc --;
		stc->stc_inc /=2;
		if( stc->stc_inc > stc->stc_mininc )
			stc->stc_inc=stc->stc_mininc;
	}
}

static void adj_val(Staircase *stc)	/* set the value for the next trial */
{
	switch( _trans ){
		case TR_UP: 
			stc->stc_val += stc->stc_inc; break;
		case TR_DN: 
			stc->stc_val -= stc->stc_inc; break;
		default: break;
	}
	if( stc->stc_val < 0 ) stc->stc_val = 0;
	if( stc->stc_val >= _nvals ) stc->stc_val = _nvals-1;
}

#define FC_RSP(stc,rsp)		(stc->stc_crctrsp==YES?rsp:(rsp^3))

static int iftrans(Staircase *stc,int rsp)	/* see if this response warrants a transition */
{
	int retval;

	if( stc->stc_type == UP_DOWN ){
		if( rsp == stc->stc_incrsp ) retval=TR_UP;
		else retval=TR_DN;
	} else if( stc->stc_type == TWO_TO_ONE ){
		if( FC_RSP(stc,rsp) == stc->stc_incrsp ){
			if( stc->stc_lstrsp == stc->stc_incrsp ){
				rsp=stc->stc_lstrsp = NO_RSP;
				retval=TR_UP;
			} else retval=NO_TR;
		} else retval=TR_DN;
	} else if( stc->stc_type == THREE_TO_ONE ){
		if( FC_RSP(stc,rsp) == stc->stc_incrsp ){
			if( stc->stc_lstrsp == stc->stc_incrsp ){
				if( stc->stc_lr3 == stc->stc_incrsp ){
					rsp=stc->stc_lstrsp = NO_RSP;
					retval=TR_UP;
				} else retval=NO_TR;
			} else retval=NO_TR;
		} else retval=TR_DN;
	} else {
		NERROR1("bad stair type");
		retval=NO_TR;		/* NOTREACHED */
	}

	stc->stc_lr3 = stc->stc_lstrsp;
	stc->stc_lstrsp = FC_RSP(stc,rsp);
	return(retval);
}

/* save_reponse not only saves the response, it also updates the staircase!?  a misnomer... */

void save_response(QSP_ARG_DECL  int rsp,Staircase *stc)
{
	/* give feedback if feedback string is set */

	if( correct_feedback_string != NULL ){
		if( rsp == stc->stc_crctrsp ){
			PUSH_INPUT_FILE("correct_feedback_string");
			interpret_text_fragment(QSP_ARG correct_feedback_string);
		}
#ifdef CAUTIOUS
		else if( incorrect_feedback_string == NULL )
			ERROR1("CAUTIOUS: save_response: correct_feedback_string is set, but incorrect_feedback string is not!?");
#endif /* CAUTIOUS */
		else if( rsp != REDO && rsp != ABORT ){
			PUSH_INPUT_FILE("incorrect_feedback_string");
			interpret_text_fragment(QSP_ARG incorrect_feedback_string);
		}

	}

	if( drib_file != NULL )
		wt_dribble(drib_file,
			stc->stc_clp,
			stc->stc_index,
			stc->stc_val,
			rsp,
			stc->stc_crctrsp);
		/*
		 * BUG would be nice to also print out coin flips here,
		 * but can't know what stmrt does!
		 */

#ifdef CATCH_SIGS
	if( Redo || Abort ) rsp=REDO;
#endif /* CATCH_SIGS */

	if( rsp != REDO ){
		if( recording ) tally(stc,rsp);
		_trans=iftrans(stc,rsp);
		if( stc->stc_inc != stc->stc_mininc ) adj_inc(stc);
		adj_val(stc);
	} else {
		sprintf(error_string,"discarding trial");
		advise(error_string);
		if( Abort ){
			sprintf(error_string,"aborting run");
			advise(error_string);
		}
	}
}

static int step(QSP_ARG_DECL Staircase *stc)
{
	int rsp;

	if( stc->stc_val == (-1) ) return(0);	/* discarded */
#ifdef CATCH_SIGS
	Redo=0;
	caught=0;
#endif /* CATCH_SIGS */
	trialno++;
	if( prelim )
		sprintf(error_string,"preliminary trial number %d",trialno);
	else
		sprintf(error_string,"trial number %d",trialno);
	advise(error_string);

	/* stimulus routines MUST call response() for proper abort & redo */
	rsp=(*stmrt)( QSP_ARG stc->stc_clp->cl_index, stc->stc_val, stc );

	save_response(QSP_ARG rsp,stc);
	return(rsp);
}

int makestair( QSP_ARG_DECL  int st, int sc, int mi, int cr, int ir )
/* staircase type */
/* staircase class */
/* mininimum increment */
/* correct response */
/* increment response */
{
	char str[64];
	Staircase *stcp;
	Trial_Class *clp;

#ifdef CAUTIOUS
	if( sc < 0 ){
		WARN("CAUTIOUS:  negative class specification");
		return(-1);
	}
#endif /* CAUTIOUS */

	clp = index_class(QSP_ARG  sc);
#ifdef CAUTIOUS
	if( clp == NO_CLASS ) ERROR1("CAUTIOUS:  missing class in makestair");
#endif

	sprintf(str,"staircase.%d.%d",sc,clp->cl_nstairs);
	stcp = new_stc(QSP_ARG  str);
	if( stcp == NO_STAIR )
		return(-1);
	stcp->stc_clp = clp;
	stcp->stc_index = clp->cl_nstairs++;

	stcp->stc_type=st;
	stcp->stc_crctrsp=cr;
	stcp->stc_incrsp=ir;

	stcp->stc_lr3=
	stcp->stc_lstrsp=REDO;
	stcp->stc_lasttr=NO_TR;

	if( _nvals <= 0 ){
		WARN("X values should be read before setting up staircases");
		advise("setting initial staircase increment to 1");
		stcp->stc_inc=1;
	} else stcp->stc_inc=_nvals/2;

	/* random initialization is ok in general, but not good
		for different types of stair on a U-shaped function! */

	/*
	stcp->stc_val = rn( _nvals-1 );
	*/

	/* normally val=0 is easiest to see (YES)
	 */

	/* n/2 is not so good for real forced choice,
		but it's probably not much worse than random */

	stcp->stc_val = _nvals/2 ;

	stcp->stc_mininc=mi;
	if( mi < 0 ) stcp->stc_inc *= (-1);

	return(nstairs++);
} /* end makestair */

COMMAND_FUNC( savdat )
{
	if( summ_file != NULL ){
		if( recording ) wtdata(QSP_ARG  summ_file);
		else WARN("closing data file without writing anything");
		fclose(summ_file);
		summ_file=NULL;
	}
	if( drib_file != NULL ) {
		fclose(drib_file);
		drib_file=NULL;
	}
	recording=0;
}

#ifdef CATCH_SIGS
void icatch()	/** when SIGINT is caught */
{
	const char *sustr="are you sure";

	if( caught ) return;
	caught=1;
	sig_set(SIGINT,icatch);	/* reset signal */
	sprintf(error_string,"\n\007\007");
	advise(error_string);
	if( askif("intentional interrupt") ){
		if( askif("redo last trial") ) Redo=1;
		/* is Redo initially zero? */
		else if( askif("abort run") ){
			if( askif(sustr) ){
				Abort=1;
				savdat();
			}
		}
		else if( askif("DIE") ){
			if( askif(sustr) ) exit(0);
		}
		else Redo=1;
	}
	else Redo=1;

	/* giveup();*/	/* suppresses repetition of response prompt */
			/* see qword.c in libjbm.a */
}
#endif /* CATCH_SIGS */

/* What does mk_stair_array do??? */

static void mk_stair_array(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;
	Staircase **stcp;
	int n;

	if( stc_itp == NO_ITEM_TYPE )
		stc_init(SINGLE_QSP_ARG);

	if( stair_tbl != NO_STAIR_PTR )
		givbuf(stair_tbl);
	if( order != ((u_long)NULL) )
		givbuf(order);

	lp = stair_list(SINGLE_QSP_ARG);
	n = eltcount(lp);
	stcp = stair_tbl = (Staircase **)getbuf( n * sizeof(stcp) );
	order = (u_long *) getbuf( n * sizeof(*order) );
	np = lp->l_head;
	while( np != NO_NODE ){
		*stcp++ = (Staircase *)np->n_data;
		np = np->n_next;
	}
}

void _run_init(SINGLE_QSP_ARG_DECL)	/* general runtime initialization */
{
	rninit(SINGLE_QSP_ARG);

#ifdef CATCH_SIGS
	sig_set(SIGINT,icatch);
#endif /* CATCH_SIGS */

	clrdat(SINGLE_QSP_ARG);
	recording=0;
	Abort=0;

	mk_stair_array(SINGLE_QSP_ARG);

	(*initrt)();
}

static void stepall(SINGLE_QSP_ARG_DECL)	/** step each stair in a random order */
{
	int i;

if( nstairs==0 ) ERROR1("stepall:  no staircases!?");

	permute(QSP_ARG  order,nstairs);
	for(i=0;i<nstairs;i++){
		if( (!Abort) && step( QSP_ARG  stair_tbl[ order[i] ] ) == REDO ){
			scramble(QSP_ARG  &order[i], nstairs-i );
			i--;
		}
	}
}

static void stair_trials(QSP_ARG_DECL  int np,int nt)
{
	int i;
	int ndone;

	recording=0;
	trialno=0;
	prelim=1;
	if( np < 0 ){
		ndone=0;
		while( !Abort && ndone<MAXPREL && PRESTEP ) ndone++;
		if( ndone>=MAXPREL ) for(i=0;i<nstairs;i++)
			if( stair_tbl[i]->stc_inc
				!= stair_tbl[i]->stc_mininc ){

				sprintf(error_string,
		"\007no preliminary threshold for stair %d",i);
				WARN(error_string);
				stair_tbl[i]->stc_val=(-1);
			}
	} else while( (!Abort) && np-- ) stepall(SINGLE_QSP_ARG);
	trialno=0;
	prelim=0;
	if( !Abort ) {
		sprintf(error_string,"data logging starting");
		advise(error_string);
		recording=1;
		while( (!Abort) && nt-- ) stepall(SINGLE_QSP_ARG);
	}
}

void _run_stairs(QSP_ARG_DECL  int np,int nt)	/** this does most everything */
{
	_run_init(SINGLE_QSP_ARG);
	stair_trials(QSP_ARG np,nt);
	savdat(SINGLE_QSP_ARG);

#ifdef CATCH_SIGS
	sig_set(SIGINT,SIG_DFL);
#endif /* CATCH_SIGS */
}

static int prestep(SINGLE_QSP_ARG_DECL)	/* step stairs below criterion in a random order */
{
	int i;
	int still=0;

	permute(QSP_ARG  order,nstairs);
	for(i=0;i<nstairs;i++){
		if( !Abort && stair_tbl[order[i]]->stc_inc
				!= stair_tbl[order[i]]->stc_mininc ){

			still=1;
			step( QSP_ARG  stair_tbl[order[i]] );
		}
	}
	return(still);
}

void set_dribble_file(FILE *fp) { drib_file=fp; }
void set_summary_file(FILE *fp) { summ_file=fp; }

void add_stair(QSP_ARG_DECL  int type,int condition)
{
	int i;

	i = makestair(QSP_ARG  type,condition,1,YES,YES);
}

void del_stair(QSP_ARG_DECL  Staircase *stcp)
{
	del_item(QSP_ARG  stc_itp,stcp);
	givbuf((void *)stcp->stc_name);
}

COMMAND_FUNC( del_all_stairs )
{
	List *lp;
	Node *np;
	Staircase *stcp;

	if( stc_itp == NO_ITEM_TYPE ) return;	/* nothing to do */

advise("deleting all staircases");

	lp=stair_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST ) return;

	np=lp->l_head;
	while( np != NO_NODE ){
		stcp = (Staircase *) np->n_data;
		del_stair(QSP_ARG  stcp);
		np=np->n_next;
	}
}

/* class functions */

Trial_Class *index_class(QSP_ARG_DECL  int index)
{
	Node *np;
	Trial_Class *clp;
	List *lp;

	lp=item_list(QSP_ARG  trial_class_itp);
#ifdef CAUTIOUS
	if( lp == NO_LIST ){
		WARN("CAUTIOUS:  index_class:  no classes defined");
		return(NO_CLASS);
	}
#endif /* CAUTIOUS */

	np=lp->l_head;
	while(np!=NO_NODE){
		clp=(Trial_Class *)np->n_data;
		if( clp->cl_index == index ) return(clp);
		np=np->n_next;
	}
	sprintf(error_string,
		"index_class:  no class with index %d",index);
	WARN(error_string);

	return(NO_CLASS);
}

void del_class(QSP_ARG_DECL  Trial_Class *clp)
{
	givbuf(clp->cl_dtp);

	/* what is this data field? */
	if( clp->cl_data != NULL ) givbuf(clp->cl_data);

	del_trial_class(QSP_ARG  clp->cl_name);
	rls_str(clp->cl_name);
}

Trial_Class *new_class(SINGLE_QSP_ARG_DECL)
{
	Trial_Class *clp;
	List *lp;
	int n;

	if( trial_class_itp == NO_ITEM_TYPE )
		trial_class_init(SINGLE_QSP_ARG);

#ifdef CAUTIOUS
	if( trial_class_itp == NO_ITEM_TYPE )
		ERROR1("CAUTIOUS:  error creating class item type");
#endif /* CAUTIOUS */

	lp=item_list(QSP_ARG  trial_class_itp);
	if( lp == NO_LIST ) n=0;
	else n=eltcount(lp);

	clp = class_for(QSP_ARG  n);
	return(clp);
}

Trial_Class *class_for( QSP_ARG_DECL  int class_index )
{
	char newname[32];
	Trial_Class *clp;

	sprintf(newname,"class%d",class_index);
	clp = trial_class_of(QSP_ARG  newname);
	if( clp != NO_CLASS )
		return(clp);

	clp = new_trial_class(QSP_ARG  newname);

#ifdef CAUTIOUS
	if( clp == NO_CLASS ){
		sprintf(error_string,"CAUTIOUS:  new_class:  error creating %s!?",newname);
		ERROR1(error_string);
	}
#endif /* CAUTIOUS */

	/* Now do the initial setup for the class structure */

	clp->cl_dtp=(Data_Tbl *)getbuf(sizeof(Data_Tbl));
	clp->cl_data = NULL;
	clp->cl_index=class_index;
	clp->cl_nstairs=0;
	clear_data(clp);

	return(clp);
}

static void clear_data(Trial_Class *clp)	/* clear data table for this class */
{
	int val;

	clp->cl_dtp->d_npts=0;
	for(val=0;val<MAXVALS;val++)
		clp->cl_dtp->d_data[val].ntotal=
		clp->cl_dtp->d_data[val].ncorr=0;
}

List *class_list(SINGLE_QSP_ARG_DECL)
{
	if( trial_class_itp == NO_ITEM_TYPE ) return(NO_LIST);
	return( item_list(QSP_ARG  trial_class_itp) );
}

