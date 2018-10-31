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

static int iftrans(Staircase *st_p,int rsp);
static void adj_inc(Staircase *st_p);
static void adj_val(Staircase *st_p);
static void stair_trials(QSP_ARG_DECL  int np,int nt);

int (*stmrt)(QSP_ARG_DECL  Trial_Class *,int,Staircase *)=default_stim;		/* global var */

/* globals? */
int Abort;				/* stop the run now if set != 0 */
const char *correct_feedback_string=NULL, *incorrect_feedback_string=NULL;

static FILE *summ_file=NULL;		/* summary data */

static int recording;			// flag variable that controls data storage...
					// First N trials are "practice"

static int prelim;
static int trialno;
static int _trans;		/* possible transition of current trial */
static SCRAMBLE_TYPE *stair_order=NULL;	/* permutation buffer */

static int nstairs=0;

static Staircase **stair_tbl=NULL;

ITEM_INTERFACE_DECLARATIONS(Trial_Class,trial_class,0)
ITEM_INTERFACE_DECLARATIONS(Staircase,stair,0)

#define create_summ_data_obj( name, size ) _create_summ_data_obj( QSP_ARG  name, size )

static Data_Obj * _create_summ_data_obj( QSP_ARG_DECL  const char *name, int size )
{
	Data_Obj *dp;

fprintf(stderr,"creating data table %s with size %d\n",name,size);
	dp = mk_vec(name,size,N_SUMMARY_DATA_COMPS,prec_for_code(PREC_IN));	// short integer
	return dp;
}

static void set_summ_dtbl_data_obj(Summary_Data_Tbl *sdt_p, Data_Obj *dp)
{
	if( SUMM_DTBL_DATA_OBJ(sdt_p) != NULL )
		remove_reference(SUMM_DTBL_DATA_OBJ(sdt_p));

	SET_SUMM_DTBL_DATA_OBJ(sdt_p,dp);

	if( dp != NULL )
		add_reference(dp);
}

#define init_summ_dtbl_with_name(sdt_p, name, size ) _init_summ_dtbl_with_name(QSP_ARG   sdt_p, name, size )

static void _init_summ_dtbl_with_name(QSP_ARG_DECL   Summary_Data_Tbl *sdt_p, const char *name, int size )
{
	Data_Obj *dp;

	assert(size>0);
	dp = create_summ_data_obj(name,size);
	set_summ_dtbl_data_obj(sdt_p,dp);
	SET_SUMM_DTBL_DATA_PTR(sdt_p,OBJ_DATA_PTR(dp));
	SET_SUMM_DTBL_SIZE(sdt_p,OBJ_COLS(dp));
	clear_summary_data(sdt_p);
}

static void set_summ_dtbl_xval_obj(Summary_Data_Tbl *sdt_p, Data_Obj *dp)
{
	if( SUMM_DTBL_XVAL_OBJ(sdt_p) != NULL )
		remove_reference(SUMM_DTBL_XVAL_OBJ(sdt_p));

	SET_SUMM_DTBL_XVAL_OBJ(sdt_p,dp);

	if( dp != NULL )
		add_reference(dp);
}


#define init_summ_dtbl_for_stair(sdt_p, st_p) _init_summ_dtbl_for_stair(QSP_ARG  sdt_p, st_p)

static void _init_summ_dtbl_for_stair(QSP_ARG_DECL  Summary_Data_Tbl *sdt_p, Staircase *st_p)
{
	char name[LLEN];
	Data_Obj *xv_dp;

	assert( st_p != NULL );
	assert( STAIR_CLASS(st_p) != NULL );

	xv_dp = STAIR_XVAL_OBJ(st_p);	// comes from the class...
	assert(xv_dp != NULL);

	SET_STAIR_SUMM_DTBL(st_p,sdt_p);

	sprintf(name,"summary_data_class_%d_stair_%d",
		CLASS_INDEX(STAIR_CLASS(st_p)),STAIR_INDEX(st_p));

	init_summ_dtbl_with_name(sdt_p,name,OBJ_COLS(xv_dp));
	set_summ_dtbl_xval_obj(sdt_p,xv_dp);
}

void clear_summary_data( Summary_Data_Tbl *sdt_p )
{
	int i;

	// more efficient to write words?
	//	memset(SUMM_DTBL_DATA(sdt_p),0,size*sizeof(Summary_Datum));

	SET_SUMM_DTBL_N(sdt_p,0);
	for(i=0;i<SUMM_DTBL_SIZE(sdt_p);i++){
		SET_DATUM_NTOTAL(SUMM_DTBL_ENTRY(sdt_p,i),0);
		SET_DATUM_NCORR(SUMM_DTBL_ENTRY(sdt_p,i),0);
	}
}

// init_summ_dtbl - just set the values to defaults...

#define init_summ_dtbl(sdt_p) _init_summ_dtbl(QSP_ARG  sdt_p)

static void _init_summ_dtbl(QSP_ARG_DECL  Summary_Data_Tbl *sdt_p )
{
	SET_SUMM_DTBL_N(sdt_p,0);
	SET_SUMM_DTBL_CLASS(sdt_p,NULL);
	SET_SUMM_DTBL_XVAL_OBJ(sdt_p,NULL);
	SET_SUMM_DTBL_SIZE(sdt_p,0);
	SET_SUMM_DTBL_DATA_OBJ(sdt_p,NULL);
	SET_SUMM_DTBL_DATA_PTR(sdt_p,NULL);
}

void _init_summ_dtbl_for_class( QSP_ARG_DECL  Summary_Data_Tbl * sdt_p, Trial_Class *tc_p )
{
	Data_Obj *xv_dp;

	SET_CLASS_SUMM_DTBL(tc_p,sdt_p);
	SET_SUMM_DTBL_CLASS( sdt_p, tc_p );
	
	xv_dp = CLASS_XVAL_OBJ(tc_p);
	if( xv_dp != NULL ){
		char name[LLEN];
		assert(SUMM_DTBL_CLASS(sdt_p)!=NULL);
		sprintf(name,"summary_data_class%d",CLASS_INDEX( SUMM_DTBL_CLASS(sdt_p) ) );
		init_summ_dtbl_with_name(sdt_p,name,OBJ_COLS(xv_dp));
		set_summ_dtbl_xval_obj(sdt_p,xv_dp);
	} else {
		SET_SUMM_DTBL_SIZE(sdt_p,0);
		set_summ_dtbl_data_obj(sdt_p,NULL);
		SET_SUMM_DTBL_DATA_PTR(sdt_p,NULL);
		warn("init_summ_dtbl_for_class:  NULL x-value object!?");
	}
}

Summary_Data_Tbl *_new_summary_data_tbl(SINGLE_QSP_ARG_DECL)
{
	Summary_Data_Tbl *sdt_p;
	
	sdt_p = getbuf(sizeof(Summary_Data_Tbl));
	init_summ_dtbl(sdt_p);
	return sdt_p;
}

void rls_summ_dtbl(Summary_Data_Tbl *sdt_p)
{
	if( SUMM_DTBL_DATA_OBJ(sdt_p) != NULL )
		remove_reference( SUMM_DTBL_DATA_OBJ(sdt_p) );
	SET_SUMM_DTBL_DATA_PTR(sdt_p,NULL);

	if( SUMM_DTBL_XVAL_OBJ(sdt_p) != NULL )
		remove_reference( SUMM_DTBL_XVAL_OBJ(sdt_p) );

	givbuf(sdt_p);
}

static void init_seq_data_tbl(Sequential_Data_Tbl *qdt_p)
{
	SET_SEQ_DTBL_LIST(qdt_p,new_list());
}

Sequential_Data_Tbl *_new_sequential_data_tbl(SINGLE_QSP_ARG_DECL)
{
	Sequential_Data_Tbl *qdt_p;

	qdt_p = getbuf(sizeof(Sequential_Data_Tbl));
	init_seq_data_tbl(qdt_p);
	return qdt_p;
}

void new_exp(SINGLE_QSP_ARG_DECL)		/** discard old stairs */
{
	List *lp;
	Node *np;
	Staircase *st_p;

	nstairs=0;

	lp = stair_list();
	if( lp==NULL ) return;

	/* Don't we have a routine to delete all staircases? */
	np=QLIST_HEAD(lp);
	while(np!=NULL){
		st_p = (Staircase *)np->n_data;
		np=np->n_next;		/* must do this before freeing item! */
		del_stair(st_p);
		givbuf((void *)st_p->stair_name);
	}
}

void set_recording(int flag)
{
	recording = flag;
}

static void adj_inc(Staircase *st_p)	/* adjust the increment */
{

	if( _trans == NO_TRANS ) return;
	else if( _trans == STAIR_LAST_TRIAL(st_p) ) return;
	STAIR_LAST_TRIAL(st_p)=_trans;

	/*
	 * For two-to-one staircases, we only adjust the increment
	 * when a reversal takes us in the slow (more difficult)
	 * direction. This is done to allow a relatively speedy
	 * exit from random excursions into never-never land
	 */

	if( STAIR_TYPE(st_p) != UP_DOWN && _trans == TRANS_DN ) return;

	/* negative inc means WHAT??? */

	if(STAIR_INC(st_p)>0){
		SET_STAIR_INC(st_p,STAIR_INC(st_p)+1);
		SET_STAIR_INC(st_p,STAIR_INC(st_p)/2);
		if( STAIR_INC(st_p) < STAIR_MIN_INC(st_p) )
			SET_STAIR_INC(st_p,STAIR_MIN_INC(st_p));
	} else {	/* negative increment */
		SET_STAIR_INC(st_p,STAIR_INC(st_p)-1);
		SET_STAIR_INC(st_p,STAIR_INC(st_p)/2);
		if( STAIR_INC(st_p) > STAIR_MIN_INC(st_p) )
			SET_STAIR_INC(st_p,STAIR_MIN_INC(st_p));
	}
}

static void adj_val(Staircase *st_p)	/* set the value for the next trial */
{
	switch( _trans ){
		case TRANS_UP: 
			st_p->stair_val += STAIR_INC(st_p); break;
		case TRANS_DN: 
			st_p->stair_val -= STAIR_INC(st_p); break;
		default: break;
	}
	if( st_p->stair_val < 0 ) st_p->stair_val = 0;
	if( st_p->stair_val > STAIR_MAX_VAL(st_p) )
		st_p->stair_val = (int) STAIR_MAX_VAL(st_p);
}

#define FC_RSP(st_p,rsp)		(STAIR_CRCT_RSP(st_p)==YES?rsp:(rsp^3))

static int iftrans(Staircase *st_p,int rsp)	/* see if this response warrants a transition */
{
	int retval;

	if( STAIR_TYPE(st_p) == UP_DOWN ){
		if( rsp == STAIR_INC_RSP(st_p) ) retval=TRANS_UP;
		else retval=TRANS_DN;
	} else if( STAIR_TYPE(st_p) == TWO_TO_ONE ){
		if( FC_RSP(st_p,rsp) == STAIR_INC_RSP(st_p) ){
			if( STAIR_LAST_RSP(st_p) == STAIR_INC_RSP(st_p) ){
				rsp=NO_RSP;
				SET_STAIR_LAST_RSP(st_p,NO_RSP);
				retval=TRANS_UP;
			} else retval=NO_TRANS;
		} else retval=TRANS_DN;
	} else if( STAIR_TYPE(st_p) == THREE_TO_ONE ){
		if( FC_RSP(st_p,rsp) == STAIR_INC_RSP(st_p) ){
			if( STAIR_LAST_RSP(st_p) == STAIR_INC_RSP(st_p) ){
				if( STAIR_LAST_RSP3(st_p) == STAIR_INC_RSP(st_p) ){
					rsp=NO_RSP;
					SET_STAIR_LAST_RSP(st_p,NO_RSP);
					retval=TRANS_UP;
				} else retval=NO_TRANS;
			} else retval=NO_TRANS;
		} else retval=TRANS_DN;
	} else {
		NERROR1("bad stair type");
		retval=NO_TRANS;		/* NOTREACHED */
	}

	SET_STAIR_LAST_RSP3(st_p, STAIR_LAST_RSP(st_p) );
	SET_STAIR_LAST_RSP(st_p, (int) FC_RSP(st_p,rsp) );
	return(retval);
}

static void tally(Staircase *st_p,int rsp)			/* record new data */
{
	assert(st_p!=NULL);
	assert(STAIR_CLASS(st_p)!=NULL);

	assert(CLASS_SUMM_DTBL(STAIR_CLASS(st_p))!=NULL);
	update_summary(CLASS_SUMM_DTBL(STAIR_CLASS(st_p)),st_p,rsp);

	if( STAIR_SUMM_DTBL(st_p) != NULL ){
		update_summary(STAIR_SUMM_DTBL(st_p),st_p,rsp);
	}

	assert(CLASS_SEQ_DTBL(STAIR_CLASS(st_p))!=NULL);
	append_trial(CLASS_SEQ_DTBL(STAIR_CLASS(st_p)),st_p,rsp);

	if( STAIR_SEQ_DTBL(st_p) != NULL )
		append_trial(STAIR_SEQ_DTBL(st_p),st_p,rsp);
}


#define FEEDBACK_FILENAME	"(pushed feedback_string)"

#define give_trial_feedback(rsp, st_p) _give_trial_feedback(QSP_ARG  rsp, st_p)

static void _give_trial_feedback(QSP_ARG_DECL  int rsp, Staircase *st_p)
{
	if( rsp == STAIR_CRCT_RSP(st_p) ){
		assert( correct_feedback_string != NULL );
		chew_text(correct_feedback_string, FEEDBACK_FILENAME);
	} else if( rsp != REDO && rsp != ABORT ){
		assert( incorrect_feedback_string != NULL );
		chew_text(incorrect_feedback_string, FEEDBACK_FILENAME);
	}
}

/* save_reponse not only saves the response, it also updates the staircase!?  a misnomer... */

void _save_response(QSP_ARG_DECL  int rsp,Staircase *st_p)
{
	/* give feedback if feedback string is set */

	if( correct_feedback_string != NULL )
		give_trial_feedback(rsp,st_p);

	if( dribbling() )
		dribble(st_p,rsp);

		/*
		 * BUG would be nice to also print out coin flips here,
		 * but can't know what stmrt does!
		 */

#ifdef CATCH_SIGS
	if( Redo || Abort ) rsp=REDO;
#endif /* CATCH_SIGS */

	if( rsp != REDO ){
		if( recording ) tally(st_p,rsp);
		_trans= (int) iftrans(st_p,rsp);
		if( STAIR_INC(st_p) != STAIR_MIN_INC(st_p) ) adj_inc(st_p);
		adj_val(st_p);
	} else {
		sprintf(ERROR_STRING,"discarding trial");
		advise(ERROR_STRING);
		if( Abort ){
			sprintf(ERROR_STRING,"aborting run");
			advise(ERROR_STRING);
		}
	}
}

#define step(st_p) _step(QSP_ARG st_p)

static int _step(QSP_ARG_DECL Staircase *st_p)
{
	int rsp;

	if( st_p->stair_val == (-1) ) return(0);	/* discarded */
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
	rsp=(*stmrt)( QSP_ARG STAIR_CLASS(st_p), st_p->stair_val, st_p );

	save_response(rsp,st_p);
	return(rsp);
}

int _make_staircase( QSP_ARG_DECL  int st,	/* staircase type */
		Trial_Class *tc_p,	/* staircase class */
		int mi,		/* mininimum increment */
		int cr,		/* correct response */
		int ir		/* increment response */
		)
{
	char str[64];
	Staircase *st_p;
	int n_xvals;
	Data_Obj *xv_dp;
	Summary_Data_Tbl *sdt_p;

	assert( tc_p != NULL );

	sprintf(str,"staircase.%s.%d",CLASS_NAME(tc_p),CLASS_N_STAIRS(tc_p) );
	st_p = new_stair(str);
	assert(st_p!=NULL);

	SET_STAIR_CLASS(st_p,tc_p);
	st_p->stair_index = CLASS_N_STAIRS(tc_p);
	SET_CLASS_N_STAIRS(tc_p,CLASS_N_STAIRS(tc_p)+1);

	SET_STAIR_TYPE(st_p,st);
	SET_STAIR_CRCT_RSP(st_p,cr);
	SET_STAIR_INC_RSP(st_p,ir);

	SET_STAIR_LAST_RSP3(st_p,REDO);
	SET_STAIR_LAST_RSP(st_p,REDO);
	SET_STAIR_LAST_TRIAL(st_p,NO_TRANS);

	xv_dp = CLASS_XVAL_OBJ(tc_p);
	assert(xv_dp!=NULL);

	sdt_p = new_summary_data_tbl();

	init_summ_dtbl_for_stair(sdt_p,st_p);
	SET_SUMM_DTBL_CLASS( STAIR_SUMM_DTBL(st_p), tc_p );

	SET_STAIR_SEQ_DTBL(st_p,new_sequential_data_tbl());
	SET_SEQ_DTBL_CLASS( STAIR_SEQ_DTBL(st_p), tc_p );

	n_xvals = (int) OBJ_COLS(STAIR_XVAL_OBJ(st_p));
	SET_STAIR_INC(st_p, n_xvals/2);

	/* random initialization is ok in general, but not good
		for different types of stair on a U-shaped function! */

	/* normally val=0 is easiest to see (YES)
	 */

	/* n/2 is not so good for real forced choice,
		but it's probably not much worse than random */

	st_p->stair_val = n_xvals/2 ;

	SET_STAIR_MIN_INC(st_p,mi);
	if( mi < 0 ) SET_STAIR_INC(st_p,STAIR_INC(st_p)*(-1));

	return(nstairs++);
} /* end makestair */

COMMAND_FUNC( do_save_data )
{
	if( summ_file != NULL ){
		if( recording )
			write_exp_data(summ_file);
		else
			warn("closing data file without writing anything");
		fclose(summ_file);
		summ_file=NULL;
	}
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

#define mk_stair_array() _mk_stair_array(SINGLE_QSP_ARG)

static void _mk_stair_array(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;
	Staircase **st_pp;
	int n;

	if( stair_itp == NULL ) init_stairs();	// init item subsystem

	if( stair_tbl != NULL ) givbuf(stair_tbl);
	if( stair_order != NULL ) givbuf(stair_order);

	lp = stair_list();	// list of already-created staircases
	n = eltcount(lp);
	if( n < 1 ){
		warn("mk_stair_array:  no staircases specified!?");
		return;
	}

	st_pp = stair_tbl = (Staircase **)getbuf( n * sizeof(Staircase *) );
	stair_order = (SCRAMBLE_TYPE *) getbuf( n * sizeof(*stair_order) );
	np = QLIST_HEAD(lp);
	while( np != NULL ){
		*st_pp++ = (Staircase *)np->n_data;
		np = np->n_next;
	}
}

void _run_init(SINGLE_QSP_ARG_DECL)	/* general runtime initialization */
{
	rninit();		// seed random number generator

#ifdef CATCH_SIGS
	sig_set(SIGINT,icatch);
#endif /* CATCH_SIGS */

	clear_all_data_tables();
	recording=0;
	Abort=0;

	mk_stair_array();

advise("calling initrt");
	(*initrt)();
}

#define step_all_stairs() _step_all_stairs(SINGLE_QSP_ARG)

static void _step_all_stairs(SINGLE_QSP_ARG_DECL)	/** step each stair in a random order */
{
	int i;

if( nstairs==0 ) error1("step_all_stairs:  no staircases!?");

	permute(stair_order,nstairs);
	for(i=0;i<nstairs;i++){
		if( (!Abort) && step( stair_tbl[ stair_order[i] ] ) == REDO ){
			scramble(&stair_order[i], nstairs-i );
			i--;
		}
	}
}

#define prestep() _prestep(SINGLE_QSP_ARG)

static int _prestep(SINGLE_QSP_ARG_DECL)	/* step stairs below criterion in a random order */
{
	int i;
	int still=0;

	permute(stair_order,nstairs);
	for(i=0;i<nstairs;i++){
		if( !Abort && STAIR_INC(stair_tbl[stair_order[i]])
				!= STAIR_MIN_INC(stair_tbl[stair_order[i]]) ){

			still=1;
			step( stair_tbl[stair_order[i]] );
		}
	}
	return(still);
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
		while( !Abort && ndone<MAXPREL && prestep() ) ndone++;
		if( ndone>=MAXPREL ) for(i=0;i<nstairs;i++)
			if( STAIR_INC(stair_tbl[i])
				!= STAIR_MIN_INC(stair_tbl[i]) ){

				sprintf(ERROR_STRING,
		"\007no preliminary threshold for stair %d",i);
				warn(ERROR_STRING);
				stair_tbl[i]->stair_val=(-1);
			}
	} else while( (!Abort) && np-- ) step_all_stairs();
	trialno=0;
	prelim=0;
	if( !Abort ) {
		sprintf(ERROR_STRING,"data logging starting");
		advise(ERROR_STRING);
		recording=1;
		while( (!Abort) && nt-- ) step_all_stairs();
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

void set_summary_file(FILE *fp) { summ_file=fp; }

void _add_stair(QSP_ARG_DECL  int type,Trial_Class *tc_p )
{
	if( make_staircase(type,tc_p,1,YES,YES) < 0 )
		warn("Error creating staircase!?");
}

COMMAND_FUNC( del_all_stairs )
{
	List *lp;
	Node *np;
	Staircase *st_p;

	if( stair_itp == NULL ) return;	/* nothing to do */

advise("deleting all staircases");

	lp=stair_list();
	if( lp == NULL ) return;

	np=QLIST_HEAD(lp);
	while( np != NULL ){
		st_p = (Staircase *) np->n_data;
		del_stair(st_p);
		np=np->n_next;
	}
}

/* class functions */

Trial_Class *_find_class_from_index(QSP_ARG_DECL  int index)
{
	List *lp;
	Node *np;

	lp=trial_class_list();
	assert( lp != NULL );

	np=QLIST_HEAD(lp);
	while(np!=NULL){
		Trial_Class *tc_p;
		tc_p=(Trial_Class *)np->n_data;
		if( CLASS_INDEX(tc_p) == index ) return(tc_p);
		np=np->n_next;
	}
	sprintf(ERROR_STRING,
		"find_class_from_index:  no class with index %d",index);
	warn(ERROR_STRING);

	return(NULL);
}

Staircase *_find_stair_from_index(QSP_ARG_DECL  int index)
{
	List *lp;
	Node *np;

	lp=stair_list();
	assert( lp != NULL );

	np=QLIST_HEAD(lp);
	while(np!=NULL){
		Staircase *st_p;
		st_p=(Staircase *)np->n_data;
		if( STAIR_INDEX(st_p) == index ) return(st_p);
		np=np->n_next;
	}
	sprintf(ERROR_STRING,
		"find_stair_from_index:  no staircase with index %d",index);
	warn(ERROR_STRING);

	return(NULL);
}

void _del_class(QSP_ARG_DECL  Trial_Class *tc_p)
{
	rls_summ_dtbl( CLASS_SUMM_DTBL(tc_p) );

	/* what is this data field? */
	if( CLASS_CMD(tc_p) != NULL ) rls_str(CLASS_CMD(tc_p));

	del_trial_class(tc_p);
}

Trial_Class *new_class(SINGLE_QSP_ARG_DECL)
{
	Trial_Class *tc_p;
	List *lp;
	int n;

	if( trial_class_itp == NULL )
		init_trial_classs();

	assert( trial_class_itp != NULL );

	lp=item_list(trial_class_itp);
	if( lp == NULL ) n=0;
	else n=(int)eltcount(lp);

	tc_p = new_class_for_index(n);
	return(tc_p);
}

// new_class_for_index creates a new class...

Trial_Class *_new_class_for_index( QSP_ARG_DECL  int class_index )
{
	char newname[32];
	Trial_Class *tc_p;

	sprintf(newname,"class%d",class_index);

	tc_p = trial_class_of(newname);
	if( tc_p != NULL )
		return NULL;

	tc_p = create_named_class(newname);

	return(tc_p);
}

#define clear_one_class(tc_p, arg ) _clear_one_class(QSP_ARG  tc_p, arg )

static void _clear_one_class(QSP_ARG_DECL  Trial_Class *tc_p, void *arg )
{
	Summary_Data_Tbl *sdt_p;

	sdt_p = CLASS_SUMM_DTBL(tc_p);
	assert(sdt_p!=NULL);
	clear_summary_data(sdt_p);
}

void _clear_all_data_tables(SINGLE_QSP_ARG_DECL)	/* just clears data tables */
{
	iterate_over_classes(_clear_one_class,NULL);
}

void update_summary(Summary_Data_Tbl *sdt_p,Staircase *st_p,int rsp)
{
	int val;

	assert( sdt_p != NULL );

	val = STAIR_VAL(st_p);
	assert( SUMM_DTBL_SIZE(sdt_p) > 0 );
	assert( val >= 0 && val < SUMM_DTBL_SIZE(sdt_p) );

	if( rsp == STAIR_CRCT_RSP(st_p) )
		SET_DATUM_NCORR( SUMM_DTBL_ENTRY(sdt_p,val),
			1 + DATUM_NCORR( SUMM_DTBL_ENTRY(sdt_p,val) ) );
	if( rsp != REDO && rsp != ABORT )
		SET_DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdt_p,val),
			1 + DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdt_p,val) ) );
}

void append_trial( Sequential_Data_Tbl *qdt_p, Staircase *st_p , int rsp )
{
	Node *np;
	Sequence_Datum *qd_p;

	qd_p = getbuf(sizeof(Sequence_Datum));

	SET_SEQ_DATUM_CLASS_IDX(qd_p, CLASS_INDEX( STAIR_CLASS(st_p) ) );
	SET_SEQ_DATUM_STAIR_IDX(qd_p, STAIR_INDEX(st_p) );
	SET_SEQ_DATUM_XVAL_IDX(qd_p, STAIR_VAL(st_p) );
	SET_SEQ_DATUM_RESPONSE(qd_p, rsp );
	SET_SEQ_DATUM_CRCT_RSP(qd_p, STAIR_CRCT_RSP(st_p) );

	np = mk_node(qd_p);
	addTail( SEQ_DTBL_LIST(qdt_p), np );
}

