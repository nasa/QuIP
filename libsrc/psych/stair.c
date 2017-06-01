#include "quip_config.h"

#include <stdio.h>
#include "quip_prot.h"
#include "stc.h"
#include "rn.h"
#include "item_type.h"
#include "list.h"
#include "getbuf.h"

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

//static int nullstim(QSP_ARG_DECL  int,int,Staircase *);
static void mk_stair_array(SINGLE_QSP_ARG_DECL);
static int prestep(SINGLE_QSP_ARG_DECL);
#define PRESTEP			prestep(SINGLE_QSP_ARG)
static void clear_data(Trial_Class *tcp);

static List *stair_list(SINGLE_QSP_ARG_DECL);
static void tally(Staircase *stc,int rsp);
static int iftrans(Staircase *stc,int rsp);
static void adj_inc(Staircase *stc);
static void adj_val(Staircase *stc);
static int step(QSP_ARG_DECL  Staircase *stc);
static void stepall(SINGLE_QSP_ARG_DECL);
static void stair_trials(QSP_ARG_DECL  int np,int nt);

int (*stmrt)(QSP_ARG_DECL  Trial_Class *,int,Staircase *)=default_stim;		/* global var */

/* globals? */
int Abort;				/* stop the run now if set != 0 */
const char *correct_feedback_string=NULL, *incorrect_feedback_string=NULL;

static FILE *summ_file=NULL;		/* summary data */
static int recording;
static int prelim;
static int trialno;
static int _trans;		/* possible transition of current trial */
static SCRAMBLE_TYPE *order=NULL;	/* permutation buffer */

static int nstairs=0;

#define NO_STAIR_PTR	((Staircase **)NULL)
static Staircase **stair_tbl=NO_STAIR_PTR;

ITEM_INTERFACE_DECLARATIONS(Trial_Class,trial_class,0)
ITEM_INTERFACE_DECLARATIONS(Staircase,stc,0)

static List *stair_list(SINGLE_QSP_ARG_DECL)
{
	if( stc_itp==NO_ITEM_TYPE ) return(NULL);
	return(item_list(QSP_ARG  stc_itp) );
}

//static int nullstim(QSP_ARG_DECL  int c,int v,Staircase *stcp)
//{
//	sprintf(ERROR_STRING,"no stimulus routine");
//	WARN(ERROR_STRING);
//	return(0);
//}

void new_exp(SINGLE_QSP_ARG_DECL)		/** discard old stairs */
{
	List *lp;
	Node *np;
	Staircase *stcp;

	nstairs=0;

	lp = stair_list(SINGLE_QSP_ARG);
	if( lp==NULL ) return;

	/* Don't we have a routine to delete all staircases? */
	np=lp->l_head;
	while(np!=NO_NODE){
		stcp = (Staircase *)np->n_data;
		np=np->n_next;		/* must do this before freeing item! */
		del_stc(QSP_ARG  stcp);
		givbuf((void *)stcp->stc_name);
	}
}

void set_recording(int flag)
{
	recording = flag;
}

static void tally(Staircase *stc,int rsp)			/* record new data */
{
	note_trial(STAIR_CLASS(stc),STAIR_VAL(stc),rsp,STAIR_CRCT_RSP(stc));
	//note_trial(stc,rsp);
}

static void adj_inc(Staircase *stc)	/* adjust the increment */
{

	if( _trans == NO_TR ) return;
	else if( _trans == STAIR_LAST_TRIAL(stc) ) return;
	STAIR_LAST_TRIAL(stc)=_trans;

	/*
	 * For two-to-one staircases, we only adjust the increment
	 * when a reversal takes us in the slow (more difficult)
	 * direction. This is done to allow a relatively speedy
	 * exit from random excursions into never-never land
	 */

	if( STAIR_TYPE(stc) != UP_DOWN && _trans == TR_DN ) return;

	/* negative inc means WHAT??? */

	if(STAIR_INC(stc)>0){
		SET_STAIR_INC(stc,STAIR_INC(stc)+1);
		SET_STAIR_INC(stc,STAIR_INC(stc)/2);
		if( STAIR_INC(stc) < STAIR_MIN_INC(stc) )
			SET_STAIR_INC(stc,STAIR_MIN_INC(stc));
	} else {	/* negative increment */
		SET_STAIR_INC(stc,STAIR_INC(stc)-1);
		SET_STAIR_INC(stc,STAIR_INC(stc)/2);
		if( STAIR_INC(stc) > STAIR_MIN_INC(stc) )
			SET_STAIR_INC(stc,STAIR_MIN_INC(stc));
	}
}

static void adj_val(Staircase *stc)	/* set the value for the next trial */
{
	switch( _trans ){
		case TR_UP: 
			stc->stc_val += STAIR_INC(stc); break;
		case TR_DN: 
			stc->stc_val -= STAIR_INC(stc); break;
		default: break;
	}
	if( stc->stc_val < 0 ) stc->stc_val = 0;
	if( stc->stc_val >= _nvals ) stc->stc_val = (int) _nvals-1;
}

#define FC_RSP(stc,rsp)		(STAIR_CRCT_RSP(stc)==YES?rsp:(rsp^3))

static int iftrans(Staircase *stc,int rsp)	/* see if this response warrants a transition */
{
	int retval;

	if( STAIR_TYPE(stc) == UP_DOWN ){
		if( rsp == STAIR_INC_RSP(stc) ) retval=TR_UP;
		else retval=TR_DN;
	} else if( STAIR_TYPE(stc) == TWO_TO_ONE ){
		if( FC_RSP(stc,rsp) == STAIR_INC_RSP(stc) ){
			if( STAIR_LAST_RSP(stc) == STAIR_INC_RSP(stc) ){
				rsp=NO_RSP;
				SET_STAIR_LAST_RSP(stc,NO_RSP);
				retval=TR_UP;
			} else retval=NO_TR;
		} else retval=TR_DN;
	} else if( STAIR_TYPE(stc) == THREE_TO_ONE ){
		if( FC_RSP(stc,rsp) == STAIR_INC_RSP(stc) ){
			if( STAIR_LAST_RSP(stc) == STAIR_INC_RSP(stc) ){
				if( STAIR_LAST_RSP3(stc) == STAIR_INC_RSP(stc) ){
					rsp=NO_RSP;
					SET_STAIR_LAST_RSP(stc,NO_RSP);
					retval=TR_UP;
				} else retval=NO_TR;
			} else retval=NO_TR;
		} else retval=TR_DN;
	} else {
		NERROR1("bad stair type");
		retval=NO_TR;		/* NOTREACHED */
	}

	SET_STAIR_LAST_RSP3(stc, STAIR_LAST_RSP(stc) );
	SET_STAIR_LAST_RSP(stc, (int) FC_RSP(stc,rsp) );
	return(retval);
}

/* save_reponse not only saves the response, it also updates the staircase!?  a misnomer... */

#define FEEDBACK_FILENAME	"(pushed feedback_string)"

void save_response(QSP_ARG_DECL  int rsp,Staircase *stc)
{
	/* give feedback if feedback string is set */

	if( correct_feedback_string != NULL ){
		assert( incorrect_feedback_string != NULL );

		if( rsp == STAIR_CRCT_RSP(stc) ){
			chew_text(QSP_ARG correct_feedback_string,
				FEEDBACK_FILENAME);
		}
//#ifdef CAUTIOUS
//		else if( incorrect_feedback_string == NULL )
//			ERROR1("CAUTIOUS: save_response: correct_feedback_string is set, but incorrect_feedback string is not!?");
//#endif /* CAUTIOUS */
		else if( rsp != REDO && rsp != ABORT ){
			chew_text(QSP_ARG incorrect_feedback_string,
				FEEDBACK_FILENAME);
		}

	}

	/*
	if( drib_file != NULL )
		wt_dribble(drib_file,
			stc->stc_tcp,
			stc->stc_index,
			stc->stc_val,
			rsp,
			stc->stc_crctrsp);
			*/

	if( dribbling() )
		dribble(stc,rsp);

		/*
		 * BUG would be nice to also print out coin flips here,
		 * but can't know what stmrt does!
		 */

#ifdef CATCH_SIGS
	if( Redo || Abort ) rsp=REDO;
#endif /* CATCH_SIGS */

	if( rsp != REDO ){
		if( recording ) tally(stc,rsp);
		_trans= (int) iftrans(stc,rsp);
		if( STAIR_INC(stc) != STAIR_MIN_INC(stc) ) adj_inc(stc);
		adj_val(stc);
	} else {
		sprintf(ERROR_STRING,"discarding trial");
		advise(ERROR_STRING);
		if( Abort ){
			sprintf(ERROR_STRING,"aborting run");
			advise(ERROR_STRING);
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
		sprintf(ERROR_STRING,"preliminary trial number %d",trialno);
	else
		sprintf(ERROR_STRING,"trial number %d",trialno);
	advise(ERROR_STRING);

	/* stimulus routines MUST call response() for proper abort & redo */
	rsp=(*stmrt)( QSP_ARG stc->stc_tcp, stc->stc_val, stc );

	save_response(QSP_ARG rsp,stc);
	return(rsp);
}

int makestair( QSP_ARG_DECL  int st,	/* staircase type */
		Trial_Class *tcp,	/* staircase class */
		int mi,		/* mininimum increment */
		int cr,		/* correct response */
		int ir		/* increment response */
		)
{
	char str[64];
	Staircase *stcp;

//#ifdef CAUTIOUS
//	if( sc < 0 ){
//		WARN("CAUTIOUS:  negative class specification");
//		return(-1);
//	}
//#endif /* CAUTIOUS */
	assert( tcp != NULL );

	sprintf(str,"staircase.%s.%d",CLASS_NAME(tcp),CLASS_N_STAIRS(tcp) );
	stcp = new_stc(QSP_ARG  str);
	if( stcp == NO_STAIR )
		return(-1);
	stcp->stc_tcp = tcp;
	stcp->stc_index = CLASS_N_STAIRS(tcp);
	SET_CLASS_N_STAIRS(tcp,CLASS_N_STAIRS(tcp)+1);

	SET_STAIR_TYPE(stcp,st);
	SET_STAIR_CRCT_RSP(stcp,cr);
	SET_STAIR_INC_RSP(stcp,ir);

	SET_STAIR_LAST_RSP3(stcp,REDO);
	SET_STAIR_LAST_RSP(stcp,REDO);
	SET_STAIR_LAST_TRIAL(stcp,NO_TR);

	if( _nvals <= 0 ){
		WARN("X values should be read before setting up staircases");
		advise("setting initial staircase increment to 1");
		SET_STAIR_INC(stcp,1);
	} else SET_STAIR_INC(stcp, (int) _nvals/2);

	/* random initialization is ok in general, but not good
		for different types of stair on a U-shaped function! */

	/*
	stcp->stc_val = rn( _nvals-1 );
	*/

	/* normally val=0 is easiest to see (YES)
	 */

	/* n/2 is not so good for real forced choice,
		but it's probably not much worse than random */

	stcp->stc_val = (int) _nvals/2 ;

	SET_STAIR_MIN_INC(stcp,mi);
	if( mi < 0 ) SET_STAIR_INC(stcp,STAIR_INC(stcp)*(-1));

	return(nstairs++);
} /* end makestair */

COMMAND_FUNC( do_save_data )
{
	if( summ_file != NULL ){
		if( recording )
			write_exp_data(QSP_ARG  summ_file);
		else
			WARN("closing data file without writing anything");
		fclose(summ_file);
		summ_file=NULL;
	}
	/*
	if( drib_file != NULL ) {
		fclose(drib_file);
		drib_file=NULL;
	}
	*/
	if( dribbling() ) close_dribble();

	recording=0;
}

#ifdef CATCH_SIGS
void icatch()	/** when SIGINT is caught */
{
	const char *sustr="are you sure";

	if( caught ) return;
	caught=1;
	sig_set(SIGINT,icatch);	/* reset signal */
	sprintf(ERROR_STRING,"\n\007\007");
	advise(ERROR_STRING);
	if( askif("intentional interrupt") ){
		if( askif("redo last trial") ) Redo=1;
		/* is Redo initially zero? */
		else if( askif("abort run") ){
			if( askif(sustr) ){
				Abort=1;
				do_save_data(DEFAULT_SINGLE_QSP_ARG);
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
		init_stcs(SINGLE_QSP_ARG);

	if( stair_tbl != NO_STAIR_PTR )
		givbuf(stair_tbl);
	if( order != NULL )
		givbuf(order);

	lp = stair_list(SINGLE_QSP_ARG);
	n = eltcount(lp);
	stcp = stair_tbl = (Staircase **)getbuf( n * sizeof(stcp) );
	order = (SCRAMBLE_TYPE *) getbuf( n * sizeof(*order) );
	np = lp->l_head;
	while( np != NO_NODE ){
		*stcp++ = (Staircase *)np->n_data;
		np = np->n_next;
	}
}

void _run_init(SINGLE_QSP_ARG_DECL)	/* general runtime initialization */
{
advise("calling rninit");
	rninit(SINGLE_QSP_ARG);

#ifdef CATCH_SIGS
	sig_set(SIGINT,icatch);
#endif /* CATCH_SIGS */

advise("calling clrdat");
	clrdat(SINGLE_QSP_ARG);
	recording=0;
	Abort=0;

advise("calling mk_stair_array");
	mk_stair_array(SINGLE_QSP_ARG);

advise("calling initrt");
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
			if( STAIR_INC(stair_tbl[i])
				!= STAIR_MIN_INC(stair_tbl[i]) ){

				sprintf(ERROR_STRING,
		"\007no preliminary threshold for stair %d",i);
				WARN(ERROR_STRING);
				stair_tbl[i]->stc_val=(-1);
			}
	} else while( (!Abort) && np-- ) stepall(SINGLE_QSP_ARG);
	trialno=0;
	prelim=0;
	if( !Abort ) {
		sprintf(ERROR_STRING,"data logging starting");
		advise(ERROR_STRING);
		recording=1;
		while( (!Abort) && nt-- ) stepall(SINGLE_QSP_ARG);
	}
}

void _run_stairs(QSP_ARG_DECL  int np,int nt)	/** this does most everything */
{
advise("calling _run_init");
	_run_init(SINGLE_QSP_ARG);
fprintf(stderr,"calling stair_trials %d %d\n",np,nt);
	stair_trials(QSP_ARG np,nt);
advise("calling do_save_data");
	do_save_data(SINGLE_QSP_ARG);

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
		if( !Abort && STAIR_INC(stair_tbl[order[i]])
				!= STAIR_MIN_INC(stair_tbl[order[i]]) ){

			still=1;
			step( QSP_ARG  stair_tbl[order[i]] );
		}
	}
	return(still);
}

void set_summary_file(FILE *fp) { summ_file=fp; }

void add_stair(QSP_ARG_DECL  int type,Trial_Class *tcp )
{
	if( makestair(QSP_ARG  type,tcp,1,YES,YES) < 0 )
		WARN("Error creating staircase!?");
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
	if( lp == NULL ) return;

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
	Trial_Class *tcp;
	List *lp;

	lp=item_list(QSP_ARG  trial_class_itp);
//#ifdef CAUTIOUS
//	if( lp == NULL ){
//		WARN("CAUTIOUS:  index_class:  no classes defined");
//		return(NO_CLASS);
//	}
//#endif /* CAUTIOUS */
	assert( lp != NULL );

	np=lp->l_head;
	while(np!=NO_NODE){
		tcp=(Trial_Class *)np->n_data;
		if( CLASS_INDEX(tcp) == index ) return(tcp);
		np=np->n_next;
	}
	sprintf(ERROR_STRING,
		"index_class:  no class with index %d",index);
	WARN(ERROR_STRING);

	return(NO_CLASS);
}

void del_class(QSP_ARG_DECL  Trial_Class *tcp)
{
	givbuf( CLASS_DATA_TBL(tcp) );

	/* what is this data field? */
	if( CLASS_CMD(tcp) != NULL ) rls_str(CLASS_CMD(tcp));

	del_trial_class(QSP_ARG  tcp);
}

Trial_Class *new_class(SINGLE_QSP_ARG_DECL)
{
	Trial_Class *tcp;
	List *lp;
	int n;

	if( trial_class_itp == NO_ITEM_TYPE )
		init_trial_classs(SINGLE_QSP_ARG);

//#ifdef CAUTIOUS
//	if( trial_class_itp == NO_ITEM_TYPE )
//		ERROR1("CAUTIOUS:  error creating class item type");
//#endif /* CAUTIOUS */
	assert( trial_class_itp != NO_ITEM_TYPE );

	lp=item_list(QSP_ARG  trial_class_itp);
	if( lp == NULL ) n=0;
	else n=(int)eltcount(lp);

	tcp = class_for(QSP_ARG  n);
	return(tcp);
}

static void init_data_tbl(Data_Tbl *dtp)
{
	SET_DTBL_N(dtp,0);
	if( _nvals > 0 ){
		SET_DTBL_SIZE(dtp,_nvals);
		SET_DTBL_DATA(dtp,(Datum *)getbuf(_nvals*sizeof(Datum)));
	} else {
		SET_DTBL_SIZE(dtp,0);
		SET_DTBL_DATA(dtp,NULL);
		NWARN("init_data_tbl:  number of x-values is not set!?");
	}
}

// class_for creates a new class...

Trial_Class *class_for( QSP_ARG_DECL  int class_index )
{
	char newname[32];
	Trial_Class *tcp;

	sprintf(newname,"class%d",class_index);
	tcp = trial_class_of(QSP_ARG  newname);
	if( tcp != NO_CLASS )
		return(tcp);

	tcp = new_trial_class(QSP_ARG  newname);

//#ifdef CAUTIOUS
//	if( tcp == NO_CLASS ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  new_class:  error creating %s!?",newname);
//		ERROR1(ERROR_STRING);
//		IOS_RETURN_VAL(NULL)
//	}
//#endif /* CAUTIOUS */
	assert( tcp != NO_CLASS );

	/* Now do the initial setup for the class structure */

	SET_CLASS_DATA_TBL(tcp,(Data_Tbl *)getbuf(sizeof(Data_Tbl)) );
	init_data_tbl(CLASS_DATA_TBL(tcp));
	SET_CLASS_CMD(tcp, NULL);
	SET_CLASS_INDEX(tcp,class_index);
	SET_CLASS_N_STAIRS(tcp,0);

	clear_data(tcp);

	return(tcp);
}

static void clear_data(Trial_Class *tcp)	/* clear data table for this class */
{
	int val_idx;

	//SET_DTBL_N(CLASS_DATA_TBL(tcp),0);
	for(val_idx=0;val_idx<DTBL_N(CLASS_DATA_TBL(tcp));val_idx++){
		SET_DATUM_NTOTAL(DTBL_ENTRY(CLASS_DATA_TBL(tcp),val_idx),0);
		SET_DATUM_NCORR(DTBL_ENTRY(CLASS_DATA_TBL(tcp),val_idx),0);
	}
}

List *class_list(SINGLE_QSP_ARG_DECL)
{
	if( trial_class_itp == NO_ITEM_TYPE ) return(NULL);
	return( item_list(QSP_ARG  trial_class_itp) );
}

