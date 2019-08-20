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

static int caught;

#endif /* CATCH_SIGS */

//void (*stim_func)(QSP_ARG_DECL  Staircase *) = _default_stim;		/* global var */
//int (*response_func)(QSP_ARG_DECL  Staircase *, Experiment *) = _default_response;		/* global var */

/* globals? */
int Abort;				/* stop the run now if set != 0 */
const char *correct_feedback_string=NULL, *incorrect_feedback_string=NULL;

ITEM_INTERFACE_DECLARATIONS(Trial_Class,trial_class,0)
ITEM_INTERFACE_DECLARATIONS(Staircase,stair,0)

#define create_summ_data_obj( name, size ) _create_summ_data_obj( QSP_ARG  name, size )

static Data_Obj * _create_summ_data_obj( QSP_ARG_DECL  const char *name, int size )
{
	Data_Obj *dp;

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


static void clear_summary_data(Summary_Data_Tbl *sdt_p )
{
	int i;

	assert(sdt_p!=NULL);

	// more efficient to write words?
	//	memset(SUMM_DTBL_DATA(sdt_p),0,size*sizeof(Summary_Datum));

	SET_SUMM_DTBL_N(sdt_p,0);
	for(i=0;i<SUMM_DTBL_SIZE(sdt_p);i++){
		SET_DATUM_NTOTAL(SUMM_DTBL_ENTRY(sdt_p,i),0);
		SET_DATUM_NCORR(SUMM_DTBL_ENTRY(sdt_p,i),0);
	}
	CLEAR_SDT_FLAG_BIT(sdt_p, SUMMARY_DATA_DIRTY);
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
advise("init_summ_dtbl_with_name calling clear_summary_data");
	clear_summary_data(sdt_p);
}


#ifdef FOOBAR

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
#endif // FOOBAR

// init_summ_dtbl - just set the values to defaults...

#define init_summ_dtbl(sdt_p) _init_summ_dtbl(QSP_ARG  sdt_p)

static void _init_summ_dtbl(QSP_ARG_DECL  Summary_Data_Tbl *sdt_p )
{
	SET_SUMM_DTBL_N(sdt_p,0);
	SET_SUMM_DTBL_CLASS(sdt_p,NULL);
//	SET_SUMM_DTBL_XVAL_OBJ(sdt_p,NULL);
	SET_SUMM_DTBL_SIZE(sdt_p,0);
	SET_SUMM_DTBL_DATA_OBJ(sdt_p,NULL);
	SET_SUMM_DTBL_DATA_PTR(sdt_p,NULL);
	SET_SUMM_DTBL_FLAGS(sdt_p,0);
}

#ifdef FOOBAR
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
#endif // FOOBAR

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

#define rls_seq_dtbl(qdt_p ) _rls_seq_dtbl(QSP_ARG  qdt_p )

static void _rls_seq_dtbl(QSP_ARG_DECL  Sequential_Data_Tbl *qdt_p )
{
	clear_sequential_data( qdt_p );
	rls_list(SEQ_DTBL_LIST(qdt_p));
	givbuf(qdt_p);
}


void _clear_sequential_data(QSP_ARG_DECL  Sequential_Data_Tbl *qdt_p)
{
	List *lp;
	Node *np;

	if( SEQ_DTBL_NEEDS_SAVING(qdt_p) ){
		warn("clear_sequential_data:  clearing unsaved data!?");
	}

	assert(qdt_p!=NULL);
	lp = SEQ_DTBL_LIST(qdt_p);
	while( (np=remHead(lp)) != NULL ){
		Sequence_Datum *qd_p;
		qd_p = NODE_DATA(np);
		givbuf(qd_p);
		rls_node(np);
	}
	CLEAR_QDT_FLAG_BIT(qdt_p,SEQUENTIAL_DATA_DIRTY);
}

static void init_seq_data_tbl(Sequential_Data_Tbl *qdt_p)
{
	SET_SEQ_DTBL_LIST(qdt_p,new_list());
	SET_SEQ_DTBL_FLAGS(qdt_p,0);
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
	if( flag ){
		SET_EXPT_FLAG_BITS(&expt1,EXPT_RECORDING);
	} else {
		CLEAR_EXPT_FLAG_BITS(&expt1,EXPT_RECORDING);
	}
}

static void adj_inc(Staircase *st_p, Transition_Code _trans)	/* adjust the increment */
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

static void adj_val(Staircase *st_p, Transition_Code _trans)	/* set the value for the next trial */
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

#define NO_RSP		(-1)
// BUG this code relies on the codes for YES and NO being 1 and 2, so that XOR with 3 flips them!

#define FC_RSP(st_p,rsp)		(STAIR_CRCT_RSP(st_p)==YES_INDEX?rsp:(rsp^3))

static Transition_Code iftrans(Staircase *st_p,int rsp)	/* see if this response warrants a transition */
{
	Transition_Code retval;

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

static void append_trial( Sequential_Data_Tbl *qdt_p, Staircase *st_p , int rsp )
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

	SET_QDT_FLAG_BIT(qdt_p,SEQUENTIAL_DATA_DIRTY);
}

static void tally(Staircase *st_p,int rsp)			/* record new data */
{
	assert(st_p!=NULL);
	assert(STAIR_CLASS(st_p)!=NULL);
	assert( CLASS_EXPT( STAIR_CLASS(st_p) ) != NULL );
	assert(STAIR_SEQ_DTBL(st_p)!=NULL);

	// We used to tally summary data here, but it's redundant with the sequential data,
	// so we don't bother.
	append_trial(STAIR_SEQ_DTBL(st_p),st_p,rsp);
}


#define FEEDBACK_FILENAME	"(pushed feedback_string)"

#define give_trial_feedback(rsp, st_p) _give_trial_feedback(QSP_ARG  rsp, st_p)

static void _give_trial_feedback(QSP_ARG_DECL  int rsp, Staircase *st_p)
{
	if( rsp == STAIR_CRCT_RSP(st_p) ){
		assert( correct_feedback_string != NULL );
		chew_text(correct_feedback_string, FEEDBACK_FILENAME);
	} else if( rsp != REDO_INDEX && rsp != ABORT_INDEX ){
		assert( incorrect_feedback_string != NULL );
		chew_text(incorrect_feedback_string, FEEDBACK_FILENAME);
	}
}

#define NORMAL_RESPONSE(rsp)	(rsp==YES_INDEX || rsp==NO_INDEX)

/* save_reponse not only saves the response, it also updates the staircase!?  a misnomer... */

void _process_response(QSP_ARG_DECL  int rsp,Staircase *st_p)
{
	/* give feedback if feedback string is set */

	if( correct_feedback_string != NULL )
		give_trial_feedback(rsp,st_p);

	//dribble(st_p,rsp);

		/*
		 * BUG would be nice to also print out coin flips here,
		 * but can't know what stim_func does!
		 */

#ifdef CATCH_SIGS
	if( Redo || Abort ) rsp=REDO_INDEX;
#endif /* CATCH_SIGS */

	// After adding undo, we tally all responses.
	// Add redo and undo to sequential data, but not summary data
	if( IS_RECORDING(&expt1) ) tally(st_p,rsp);

	if( NORMAL_RESPONSE(rsp) != UNDO_INDEX ){
		Transition_Code _trans;

		_trans= iftrans(st_p,rsp);
		if( STAIR_INC(st_p) != STAIR_MIN_INC(st_p) ) adj_inc(st_p,_trans);
		adj_val(st_p,_trans);
	}

	if( IS_ABORTING(&expt1) ){
		sprintf(ERROR_STRING,"aborting run");
		advise(ERROR_STRING);
	}
}

#define step(st_p,exp_p) _step(QSP_ARG st_p,exp_p)

static int _step(QSP_ARG_DECL Staircase *st_p, Experiment *exp_p)
{
	int rsp;

	if( st_p->stair_val == (-1) ) return(0);	/* discarded */
#ifdef CATCH_SIGS
	CLEAR_EXPT_FLAG_BITS(exp_p,EXPT_REDO);
	caught=0;
#endif /* CATCH_SIGS */

	EXPT_CURR_TRIAL_IDX(exp_p) ++;

	sprintf(ERROR_STRING,"trial number %d",EXPT_CURR_TRIAL_IDX(exp_p));
	advise(ERROR_STRING);

	/* stimulus routines MUST call response() for proper abort & redo */

	(* EXPT_STIM_FUNC(exp_p) )( QSP_ARG st_p );

	rsp=(* EXPT_RSP_FUNC(exp_p) )( QSP_ARG st_p, exp_p );

	process_response(rsp,st_p);
	return(rsp);
}

// reset_class clears the data and resets the staircases

void _reset_class(QSP_ARG_DECL  Trial_Class *tc_p)
{
	Node *np;

	if( CLASS_SUMM_DTBL(tc_p) != NULL ){
advise("reset_class calling clear_summary_data");
		clear_summary_data( CLASS_SUMM_DTBL(tc_p) );
	}

	assert(CLASS_STAIRCASES(tc_p)!=NULL);
	np = QLIST_HEAD( CLASS_STAIRCASES(tc_p) );
	if( np == NULL ){
		warn("reset_class:  no staircases!?");
		return;
	}
	while(np!=NULL){
		Staircase *stc_p;
		stc_p = NODE_DATA(np);
		assert(stc_p!=NULL);
		reset_stair(stc_p);
		np = NODE_NEXT(np);
	}
}

// reset_stair initialized the state variables of the staircase

void _reset_stair(QSP_ARG_DECL  Staircase *st_p)
{
	int n_xvals;

	SET_STAIR_LAST_RSP3(st_p,REDO_INDEX);
	SET_STAIR_LAST_RSP(st_p,REDO_INDEX);
	SET_STAIR_LAST_TRIAL(st_p,NO_TRANS);


	n_xvals = (int) OBJ_COLS(STAIR_XVAL_OBJ(st_p));

	/* random initialization is ok in general, but not good
		for different types of stair on a U-shaped function! */

	/* normally val=0 is easiest to see (YES)
	 */

	/* n/2 is not so good for real forced choice,
		but it's probably not much worse than random */

	st_p->stair_val = n_xvals/2 ;

	SET_STAIR_INC(st_p, n_xvals/2);
	if( STAIR_MIN_INC(st_p) < 0 ) SET_STAIR_INC(st_p,STAIR_INC(st_p)*(-1));

	if( STAIR_SUMM_DTBL(st_p) != NULL ){
		clear_summary_data( STAIR_SUMM_DTBL(st_p) );
	}
}

static void update_expt_stair_count( Experiment *exp_p )
{
	int n=0;
	Node *np;

	np = QLIST_HEAD( EXPT_CLASS_LIST(exp_p) );
	while( np != NULL ){
		Trial_Class *tc_p;
		tc_p = NODE_DATA(np);
		n += eltcount( CLASS_STAIRCASES(tc_p) );
		np = NODE_NEXT(np);
	}
	SET_EXPT_N_STAIRCASES(exp_p,n);
}

static void add_stair_to_class(Staircase *st_p,Trial_Class *tc_p)
{
	Node *np;

	np = mk_node(st_p);
	addTail(CLASS_STAIRCASES(tc_p),np);

	update_expt_stair_count( CLASS_EXPT(tc_p) );
}

void _make_staircase( QSP_ARG_DECL  int st,	/* staircase type */
		Trial_Class *tc_p,	/* staircase class */
		int mi,		/* mininimum increment */
		int cr,		/* correct response */
		int ir		/* increment response */
		)
{
	char str[128];
	Staircase *st_p;
	//Summary_Data_Tbl *sdt_p;
	int n;

	assert( tc_p != NULL );

	// BUG this will work as long as we don't start deleting some but not all staircases...
	n = eltcount( CLASS_STAIRCASES(tc_p) );

	// BUG possible buffer overflow
	sprintf(str,"staircase.%s.%d",CLASS_NAME(tc_p), n );
	st_p = new_stair(str);
	assert(st_p!=NULL);

	// populate with the function args
	SET_STAIR_TYPE(st_p,st);
	SET_STAIR_CLASS(st_p,tc_p);
	SET_STAIR_MIN_INC(st_p,mi);
	SET_STAIR_CRCT_RSP(st_p,cr);
	SET_STAIR_INC_RSP(st_p,ir);

	SET_STAIR_SUMM_DTBL(st_p,NULL);

	// BUG make sure not to delete some but not all staircases!?
	st_p->stair_index = eltcount( CLASS_STAIRCASES(tc_p) );

	// Must be called AFTER accessing CLASS_N_STAIRS!
	add_stair_to_class(st_p,tc_p);

	reset_stair(st_p);
} /* end makestair */

#define delete_staircase( stc_p ) _delete_staircase( QSP_ARG  stc_p )

static void _delete_staircase( QSP_ARG_DECL  Staircase *stc_p )
{
	rls_summ_dtbl( STAIR_SUMM_DTBL(stc_p) );
	rls_seq_dtbl( STAIR_SEQ_DTBL(stc_p) );

	del_stair(stc_p);
}


COMMAND_FUNC( do_save_data )
{
	close_dribble();

	CLEAR_EXPT_FLAG_BITS(&expt1,EXPT_RECORDING);
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

#ifdef FOOBAR
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
#endif // FOOBAR

#define reset_expt_classes(exp_p) _reset_expt_classes(QSP_ARG  exp_p)

void _reset_expt_classes(QSP_ARG_DECL  Experiment *exp_p)	/* just clears data tables */
{
	Node *np;
	assert( exp_p != NULL );
	assert( EXPT_CLASS_LIST(exp_p) != NULL );
	np = QLIST_HEAD( EXPT_CLASS_LIST(exp_p) );
	while(np!=NULL){
		Trial_Class *tc_p;
		tc_p = NODE_DATA(np);
		assert(tc_p!=NULL);
		reset_class(tc_p);
		np = NODE_NEXT(np);
	}
}

#define reset_experiment(exp_p) _reset_experiment(QSP_ARG  exp_p)

static void _reset_experiment(QSP_ARG_DECL  Experiment *exp_p)
{
	reset_expt_classes(exp_p);
	CLEAR_EXPT_FLAG_BITS(exp_p,(EXPT_RECORDING|EXPT_ABORT|EXPT_REDO));
	(* EXPT_INIT_FUNC(exp_p) )();
}


void _run_init(SINGLE_QSP_ARG_DECL)	/* general runtime initialization */
{
	rninit();		// seed random number generator

#ifdef CATCH_SIGS
	sig_set(SIGINT,icatch);
#endif /* CATCH_SIGS */

	reset_experiment(&expt1);
}

void _init_trial_block( QSP_ARG_DECL  Experiment *exp_p )
{
	int n_staircases;
	int n_rounds;
	int i, j, trial_idx;
	uint32_t *order;
	Staircase **trl_tbl, **stair_tbl;

	if( EXPT_TRIAL_TBL(exp_p) != NULL ){
		// Instead of releasing and reallocating, we could keep track of the size?
		givbuf( EXPT_TRIAL_TBL(exp_p) );
	}
	n_rounds = EXPT_N_RECORDED_TRIALS(exp_p);	// BUG - no preliminary trials???
	n_staircases = EXPT_N_STAIRCASES(exp_p);

	if( n_rounds <= 0 ){
		sprintf(ERROR_STRING,"init_trial_block:  number of recorded trials per staircase (%d) must be positive!?",
			n_rounds);
		warn(ERROR_STRING);
		return;
	}
	if( n_staircases <= 0 ){
		sprintf(ERROR_STRING,"init_trial_block:  number of staircases (%d) must be positive!?",
			n_staircases);
		warn(ERROR_STRING);
		return;
	}

	SET_EXPT_TRIAL_TBL(exp_p, getbuf( sizeof(Staircase *) * n_rounds * n_staircases ) );

	order = getbuf( sizeof(*order) * n_staircases );
	for(i=0;i<n_staircases;i++) order[i]=i;

	trial_idx = 0;
	trl_tbl = EXPT_TRIAL_TBL(exp_p);
	stair_tbl = EXPT_STAIR_TBL(exp_p);
	for(i=0;i<n_rounds;i++){
		permute(order,n_staircases);
		for(j=0;j<n_staircases;j++){
			trl_tbl[trial_idx++] = stair_tbl[ order[j] ];
		}
	}
}

#ifdef FOOBAR
#define step_all_stairs(exp_p) _step_all_stairs(QSP_ARG exp_p)

static void _step_all_stairs(QSP_ARG_DECL  Experiment *exp_p)	/** step each stair in a random order */
{
	int i;

if( nstairs==0 ) error1("step_all_stairs:  no staircases!?");

	permute(stair_order,nstairs);
	for(i=0;i<nstairs;i++){
		if( (!Abort) && step( stair_tbl[ stair_order[i] ], exp_p ) == REDO_INDEX ){
			scramble(&stair_order[i], nstairs-i );
			i--;
		}
	}
}

#define prestep(exp_p) _prestep(QSP_ARG  exp_p)

static int _prestep(QSP_ARG_DECL  Experiment *exp_p)	/* step stairs below criterion in a random order */
{
	int i;
	int still=0;

	permute(stair_order,nstairs);
	for(i=0;i<nstairs;i++){
		if( !Abort && STAIR_INC(stair_tbl[stair_order[i]])
				!= STAIR_MIN_INC(stair_tbl[stair_order[i]]) ){

			still=1;
			step( stair_tbl[stair_order[i]], exp_p );
		}
	}
	return(still);
}

#define run_prelim_to_criterion(exp_p) _run_prelim_to_criterion(QSP_ARG  exp_p)

static void _run_prelim_to_criterion(QSP_ARG_DECL  Experiment *exp_p)
{
	int ndone=0;

	while( !Abort && ndone<MAXPREL && prestep(exp_p) ){
		ndone++;
	}
	if( ndone>=MAXPREL ){
		int i;
		for(i=0;i<nstairs;i++){
			if( STAIR_INC(stair_tbl[i]) != STAIR_MIN_INC(stair_tbl[i]) ){
				sprintf(ERROR_STRING,
	"\007no preliminary threshold for stair %d",i);
				warn(ERROR_STRING);
				stair_tbl[i]->stair_val=(-1);
			}
		}
	}
}

static void preliminary_trials( QSP_ARG_DECL  Experiment *exp_p)
{
	if( EXPT_N_PRELIM_TRIALS(exp_p) < 0 ){
		run_prelim_to_criterion(exp_p);
	} else {
		int n_prelim_trials = EXPT_N_PRELIM_TRIALS(exp_p);

		while( (!Abort) && n_prelim_trials-- ){
			step_all_stairs(exp_p);
		}
	}
}
#endif // FOOBAR

static void stair_trials(QSP_ARG_DECL  Experiment *exp_p)
{
	int n_trials_remaining;

	SET_EXPT_FLAG_BITS(exp_p,EXPT_RECORDING);
	SET_EXPT_CURR_TRIAL_IDX(exp_p,0);

	n_trials_remaining = EXPT_N_TOTAL_TRIALS(exp_p);

	while( (!Abort) && n_trials_remaining > 0 ) {
		Staircase *st_p;

		st_p = EXPT_TRIAL_TBL(exp_p)[ EXPT_CURR_TRIAL_IDX(exp_p) ];
		assert(st_p!=NULL);
		// BUG?  need to check that redo and undo do the right thing???
		step(st_p,exp_p);
		n_trials_remaining --;
		SET_EXPT_CURR_TRIAL_IDX(exp_p, EXPT_CURR_TRIAL_IDX(exp_p)+1 );
	}
}

void _run_stairs(QSP_ARG_DECL  Experiment *exp_p )	/** this does most everything */
{
advise("calling _run_init");
	_run_init(SINGLE_QSP_ARG);
	stair_trials(QSP_ARG exp_p);
advise("calling do_save_data");
	do_save_data(SINGLE_QSP_ARG);

#ifdef CATCH_SIGS
	sig_set(SIGINT,SIG_DFL);
#endif /* CATCH_SIGS */
}

#ifdef FOOBAR
void _add_stair(QSP_ARG_DECL  int type,Trial_Class *tc_p )
{
	if( make_staircase(type,tc_p,1,YES_INDEX,YES_INDEX) < 0 )
		warn("Error creating staircase!?");
}
#endif // FOOBAR

void _delete_all_stairs(SINGLE_QSP_ARG_DECL)
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
	Node *np;

	rls_summ_dtbl( CLASS_SUMM_DTBL(tc_p) );
	rls_seq_dtbl( CLASS_SEQ_DTBL(tc_p) );
	// we would normally set these ptrs to NULL, but as we are
	// deleting the object anyway, we don't need to...

	/* what is this data field? */
	if( CLASS_STIM_CMD(tc_p) != NULL ) rls_str(CLASS_STIM_CMD(tc_p));
	if( CLASS_RESP_CMD(tc_p) != NULL ) rls_str(CLASS_RESP_CMD(tc_p));

	assert( CLASS_STAIRCASES(tc_p) != NULL );
	np = remHead( CLASS_STAIRCASES(tc_p) );
	while( np != NULL ){
		Staircase *stc_p;
		stc_p = NODE_DATA(np);
		delete_staircase(stc_p);
		np = remHead( CLASS_STAIRCASES(tc_p) );
	}
	rls_list( CLASS_STAIRCASES(tc_p) );

	del_trial_class(tc_p);	// remove object from database - also frees?
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

	// BUG?  If a class is deleted, then the available indices
	// won't match the list length...

	tc_p = new_class_for_index(n);
	return(tc_p);
}

// We used to have a global variable for this...
// This should be OK even if we allow deletion of classes...

static int next_class_index(void)
{
	static int class_index = 0;
	return class_index++;
}

#ifdef FOOBAR
static void set_class_xval_obj( Trial_Class *tc_p, Data_Obj *dp )
{
	if( CLASS_XVAL_OBJ(tc_p) != NULL )
		remove_reference(CLASS_XVAL_OBJ(tc_p));

	SET_CLASS_XVAL_OBJ(tc_p,dp);

	if( dp != NULL )
		add_reference(dp);
}
#endif // FOOBAR

Trial_Class *_create_named_class(QSP_ARG_DECL  const char *name)
{
	Trial_Class *tc_p;
	//Summary_Data_Tbl *sdt_p;

	// Make sure not in use
	tc_p = trial_class_of(name);
	if( tc_p != NULL ){
		sprintf(ERROR_STRING,"Class name \"%s\" is already in use!?",
			name);
		warn(ERROR_STRING);
		return NULL;
	}

	tc_p = new_trial_class(name );
	SET_CLASS_INDEX(tc_p,next_class_index());
	SET_CLASS_STAIRCASES(tc_p,new_list());

	SET_CLASS_STIM_CMD(tc_p, NULL);
	SET_CLASS_RESP_CMD(tc_p, NULL);

	SET_CLASS_EXPT(tc_p,&expt1);

	SET_CLASS_SUMM_DTBL(tc_p,NULL);

	sprintf(MSG_STR,"create_named_class:  created new class '%s' with index %d",CLASS_NAME(tc_p),CLASS_INDEX(tc_p));
	prt_msg(MSG_STR);

	return tc_p;
}	// create_named_class

void set_response_cmd( Trial_Class *tc_p, const char *s )
{
	if( CLASS_RESP_CMD(tc_p) != NULL ) {
		rls_str(CLASS_RESP_CMD(tc_p));
	}
	SET_CLASS_RESP_CMD( tc_p, savestr(s) );
}

// new_class_for_index creates a new class...

Trial_Class *_new_class_for_index( QSP_ARG_DECL  int class_index )
{
	char newname[32];
	Trial_Class *tc_p;

	sprintf(newname,"class%d",class_index);

	tc_p = trial_class_of(newname);
	if( tc_p != NULL ){
		sprintf(ERROR_STRING,"new_class_for_index:  class %s already exists!?",newname);
		warn(ERROR_STRING);
		return NULL;
	}

	tc_p = create_named_class(newname);

	return(tc_p);
}

void update_summary(Summary_Data_Tbl *sdt_p,Staircase *st_p,int rsp)
{
	int val;

	assert( sdt_p != NULL );

	val = STAIR_VAL(st_p);
	assert( SUMM_DTBL_SIZE(sdt_p) > 0 );
	assert( val >= 0 && val < SUMM_DTBL_SIZE(sdt_p) );

	if( NORMAL_RESPONSE(rsp) ){
		SET_DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdt_p,val),
			1 + DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdt_p,val) ) );
		if( rsp == STAIR_CRCT_RSP(st_p) ){
fprintf(stderr,"update_summary:  response was correct\n");
			SET_DATUM_NCORR( SUMM_DTBL_ENTRY(sdt_p,val),
				1 + DATUM_NCORR( SUMM_DTBL_ENTRY(sdt_p,val) ) );
		} else {
fprintf(stderr,"update_summary:  response was incorrect\n");
		}
		SET_SDT_FLAG_BIT(sdt_p, SUMMARY_DATA_DIRTY);
	}

}

static const char *name_for_stair_type(Staircase_Type stair_type)
{
	switch(stair_type){
		case NO_STAIR_TYPE:	return "no staircase type";
		case UP_DOWN:		return "up-down";
		case TWO_TO_ONE:	return "two-to-one";
		case THREE_TO_ONE:	return "three-to-one";
		default:		return "illegal value";
	}
}

static const char *name_for_last_trial(Transition_Code trans)
{
	switch(trans){
		case NO_TRANS:	return "no transition";
		case TRANS_UP:	return "up transition";
		case TRANS_DN:	return "down transition";
		default:		return "illegal value";
	}
}


void _print_stair_info( QSP_ARG_DECL  Staircase *stc_p )
{
	sprintf(MSG_STR,"Staircase %s:",STAIR_NAME(stc_p));
	prt_msg(MSG_STR);
	sprintf(MSG_STR,"\tClass: %s",CLASS_NAME(STAIR_CLASS(stc_p)));
	prt_msg(MSG_STR);
	sprintf(MSG_STR,"\tIndex: %d",STAIR_INDEX(stc_p));
	prt_msg(MSG_STR);

	prt_msg("\n\tType parameters:");
	sprintf(MSG_STR,"\t\tType: %d (%s)",STAIR_TYPE(stc_p),name_for_stair_type(STAIR_TYPE(stc_p)));
	prt_msg(MSG_STR);
	sprintf(MSG_STR,"\t\tMin inc: %d",STAIR_MIN_INC(stc_p));
	prt_msg(MSG_STR);
	sprintf(MSG_STR,"\t\tInc resp: %d",STAIR_INC_RSP(stc_p));
	prt_msg(MSG_STR);
	sprintf(MSG_STR,"\t\tCorrect resp: %d",STAIR_CRCT_RSP(stc_p));
	prt_msg(MSG_STR);

	prt_msg("\n\tState parameters:");
	sprintf(MSG_STR,"\t\tValue: %d",STAIR_VAL(stc_p));
	prt_msg(MSG_STR);
	sprintf(MSG_STR,"\t\tIncrement: %d",STAIR_INC(stc_p));
	prt_msg(MSG_STR);
	sprintf(MSG_STR,"\t\tLast rsp: %d",STAIR_LAST_RSP(stc_p));
	prt_msg(MSG_STR);
	sprintf(MSG_STR,"\t\tLast rsp2: %d",STAIR_LAST_RSP3(stc_p));
	prt_msg(MSG_STR);
	sprintf(MSG_STR,"\t\tLast trial: %d (%s)",STAIR_LAST_TRIAL(stc_p),name_for_last_trial(STAIR_LAST_TRIAL(stc_p)));
	prt_msg(MSG_STR);

	if( STAIR_SUMM_DTBL(stc_p) != NULL ){
		sprintf(MSG_STR,"\n\tSummary data:\n");
		prt_msg(MSG_STR);
		write_summary_data( STAIR_SUMM_DTBL(stc_p), tell_msgfile() );
	}

	prt_msg("");
}

void _print_class_info(QSP_ARG_DECL  Trial_Class *tc_p)
{
	Node *np;
	int n;

	sprintf(MSG_STR,"\nClass %s:\n",CLASS_NAME(tc_p));
	prt_msg(MSG_STR);

	sprintf(MSG_STR,"\tIndex: %d",CLASS_INDEX(tc_p));
	prt_msg(MSG_STR);
	sprintf(MSG_STR,"\tStimulus command: '%s'",CLASS_STIM_CMD(tc_p));
	prt_msg(MSG_STR);
	sprintf(MSG_STR,"\tResponse command: '%s'",CLASS_RESP_CMD(tc_p));
	prt_msg(MSG_STR);

	assert( CLASS_XVAL_OBJ(tc_p) != NULL );
	sprintf(MSG_STR,"\tX-value object: %s",OBJ_NAME(CLASS_XVAL_OBJ(tc_p)) );
	prt_msg(MSG_STR);

	n = eltcount( CLASS_STAIRCASES(tc_p) );
	sprintf(MSG_STR,"\n\tStaircases (%d):", n );
	prt_msg(MSG_STR);
	np = QLIST_HEAD( CLASS_STAIRCASES(tc_p) );
	while(np!=NULL){
		Staircase *stc_p;
		stc_p = NODE_DATA(np);
		sprintf(MSG_STR,"\t\t%s",STAIR_NAME(stc_p) );
		prt_msg(MSG_STR);
		np = NODE_NEXT(np);
	}

	if( CLASS_SUMM_DTBL(tc_p) != NULL ){
		sprintf(MSG_STR,"\n\tSummary data:\n");
		prt_msg(MSG_STR);
		write_summary_data( CLASS_SUMM_DTBL(tc_p), tell_msgfile() );
	}

	prt_msg("");
}

