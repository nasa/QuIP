#include "quip_config.h"

#include <stdio.h>
#include "quip_prot.h"
#include "stc.h"
#include "rn.h"
#include "item_type.h"
#include "list.h"
#include "getbuf.h"

/* globals? */
const char *correct_feedback_string=NULL, *incorrect_feedback_string=NULL;

ITEM_INTERFACE_DECLARATIONS(Trial_Class,trial_class,0)
ITEM_INTERFACE_DECLARATIONS(Staircase,stair,0)

#define create_summ_data_obj( name, size ) _create_summ_data_obj( QSP_ARG  name, size )



void clear_summary_data(Summary_Data_Tbl *sdt_p )
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
}

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
	CLEAR_QDT_FLAG_BITS(qdt_p,SEQUENTIAL_DATA_DIRTY);
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

static void append_trial( Experiment *exp_p, Staircase *st_p , int rsp )
{
	Sequence_Datum *qd_p;

	qd_p = getbuf(sizeof(Sequence_Datum));

	SET_SEQ_DATUM_TRIAL_IDX(qd_p, EXPT_CURR_TRIAL_IDX( STAIR_EXPT(st_p) ) );
	SET_SEQ_DATUM_CLASS_IDX(qd_p, CLASS_INDEX( STAIR_CLASS(st_p) ) );
	SET_SEQ_DATUM_STAIR_IDX(qd_p, STAIR_INDEX(st_p) );
	SET_SEQ_DATUM_XVAL_IDX(qd_p, STAIR_VAL(st_p) );
	SET_SEQ_DATUM_RESPONSE(qd_p, rsp );
	SET_SEQ_DATUM_CRCT_RSP(qd_p, STAIR_CRCT_RSP(st_p) );

	save_datum(exp_p,qd_p);
}

void save_datum(Experiment *exp_p, Sequence_Datum *qd_p)
{
	Node *np;

	assert(exp_p!=NULL);
	assert(EXPT_SEQ_DTBL(exp_p)!=NULL);

	np = mk_node(qd_p);
	addTail( SEQ_DTBL_LIST( EXPT_SEQ_DTBL(exp_p) ), np );

	SET_QDT_FLAG_BITS(EXPT_SEQ_DTBL(exp_p),SEQUENTIAL_DATA_DIRTY);
}

static void tally(Staircase *st_p,int rsp)			/* record new data */
{
	assert(st_p!=NULL);
	assert(STAIR_CLASS(st_p)!=NULL);
	assert( CLASS_EXPT( STAIR_CLASS(st_p) ) != NULL );
	assert(STAIR_SEQ_DTBL(st_p)!=NULL);

	// We used to tally summary data here, but it's redundant with the sequential data,
	// so we don't bother.
	append_trial(STAIR_EXPT(st_p),st_p,rsp);
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

#define update_remaining_trials( exp_p ) _update_remaining_trials( QSP_ARG  exp_p )

static void _update_remaining_trials( QSP_ARG_DECL  Experiment *exp_p )
{
	sprintf(MSG_STR,"%d",EXPT_N_TOTAL_TRIALS(exp_p)-EXPT_CURR_TRIAL_IDX(exp_p));
	assign_reserved_var("n_trials_remaining",MSG_STR);
}

#define NORMAL_RESPONSE(rsp)	(rsp==YES_INDEX || rsp==NO_INDEX)

/* save_reponse not only saves the response, it also updates the staircase!?  a misnomer... */

void _process_response(QSP_ARG_DECL  int rsp,Staircase *st_p)
{
	/* give feedback if feedback string is set */

	if( correct_feedback_string != NULL )
		give_trial_feedback(rsp,st_p);


	/*
	 * BUG would be nice to also print out coin flips here,
	 * but can't know what stim_func does!
	 */

	tally(st_p,rsp);

	if( NORMAL_RESPONSE(rsp) ){
		Transition_Code _trans;

		_trans= iftrans(st_p,rsp);
		if( STAIR_INC(st_p) != STAIR_MIN_INC(st_p) ) adj_inc(st_p,_trans);
		adj_val(st_p,_trans);

		// advance to the next trial
		SET_EXPT_CURR_TRIAL_IDX( STAIR_EXPT(st_p) , EXPT_CURR_TRIAL_IDX( STAIR_EXPT(st_p) )+1 );
	}

	if( IS_ABORTING( STAIR_EXPT(st_p) ) ){
		sprintf(ERROR_STRING,"aborting run");
		advise(ERROR_STRING);
	}

	update_remaining_trials( STAIR_EXPT(st_p) );
}

#define step(st_p,exp_p) _step(QSP_ARG st_p,exp_p)

static int _step(QSP_ARG_DECL Staircase *st_p, Experiment *exp_p)
{
	int rsp;

	if( st_p->stair_val == (-1) ) return(0);	/* discarded */

	sprintf(ERROR_STRING,"trial number %d",1+EXPT_CURR_TRIAL_IDX(exp_p));
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

static void apply_to_class_list( Experiment *exp_p, void (*func)(Trial_Class *, Experiment *) )
{
	Node *np;

	np = QLIST_HEAD( EXPT_CLASS_LIST(exp_p) );
	while( np != NULL ){
		Trial_Class *tc_p;
		tc_p = NODE_DATA(np);
		(*func)(tc_p,exp_p);
		np = NODE_NEXT(np);
	}
}

static void count_class_stairs( Trial_Class *tc_p, Experiment *exp_p )
{
	int n = EXPT_N_STAIRCASES(exp_p);
	n += eltcount( CLASS_STAIRCASES(tc_p) );
	SET_EXPT_N_STAIRCASES(exp_p,n);
}

static void update_expt_stair_count( Experiment *exp_p )
{
	SET_EXPT_N_STAIRCASES(exp_p,0);
	apply_to_class_list( exp_p, count_class_stairs );
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
	clear_sequential_data( EXPT_SEQ_DTBL(exp_p) );
	CLEAR_EXPT_FLAG_BITS(exp_p,(EXPT_ABORT|EXPT_REDO|EXPT_UNDO));
	(* EXPT_INIT_FUNC(exp_p) )();	// user-provided init func?
}


void _run_init(SINGLE_QSP_ARG_DECL)	/* general runtime initialization */
{
	rninit();		// seed random number generator

	reset_experiment(&expt1);
}

static void populate_stair_tbl( Experiment *exp_p )
{
	Staircase **stair_tbl;
	Node *class_np;
	int stair_idx = 0;

	stair_tbl = EXPT_STAIR_TBL(exp_p);
	assert( EXPT_CLASS_LIST(exp_p) != NULL );
	class_np = QLIST_HEAD( EXPT_CLASS_LIST(exp_p) );
	while( class_np != NULL ){
		Trial_Class *tc_p;
		Node *stair_np;

		tc_p = NODE_DATA(class_np);
		assert(tc_p!=NULL);
		assert(CLASS_STAIRCASES(tc_p)!=NULL);
		stair_np = QLIST_HEAD( CLASS_STAIRCASES(tc_p) );
		while( stair_np != NULL ){
			Staircase *stc_p;
			stc_p = NODE_DATA(stair_np);
			stair_tbl[stair_idx++] = stc_p;
			stair_np = NODE_NEXT(stair_np);
		}

		class_np = NODE_NEXT(class_np);
	}
	assert(stair_idx==EXPT_N_STAIRCASES(exp_p));
}

static void create_stair_table( Experiment *exp_p )
{
	int n_staircases;
	Staircase **stair_tbl;

	n_staircases = EXPT_N_STAIRCASES(exp_p);

	stair_tbl = getbuf( sizeof(Staircase *) * n_staircases );

	assert( EXPT_STAIR_TBL(exp_p) == NULL );
	SET_EXPT_STAIR_TBL(exp_p, stair_tbl );

	populate_stair_tbl(exp_p);
}

#define init_trial_tbl(exp_p) _init_trial_tbl(QSP_ARG  exp_p)

static int _init_trial_tbl(QSP_ARG_DECL  Experiment *exp_p)
{
	int n_staircases;
	int n_rounds;

	if( EXPT_TRIAL_TBL(exp_p) != NULL ){
		// Instead of releasing and reallocating, we could keep track of the size?
		givbuf( EXPT_TRIAL_TBL(exp_p) );
		SET_EXPT_TRIAL_TBL(exp_p,NULL);
	}
	n_rounds = EXPT_N_RECORDED_TRIALS(exp_p);	// BUG - no preliminary trials???
	n_staircases = EXPT_N_STAIRCASES(exp_p);

	if( n_rounds <= 0 ){
		sprintf(ERROR_STRING,"init_trial_tbl:  number of recorded trials per staircase (%d) must be positive!?",
			n_rounds);
		warn(ERROR_STRING);
		return -1;
	}
	if( n_staircases <= 0 ){
		sprintf(ERROR_STRING,"init_trial_tbl:  number of staircases (%d) must be positive!?",
			n_staircases);
		warn(ERROR_STRING);
		return -1;
	}

	// Instead of checking whether the number of trials is the same, we just allocate
	// a new block every time.
	assert( EXPT_TRIAL_TBL(exp_p) == NULL );
	SET_EXPT_TRIAL_TBL(exp_p, getbuf( sizeof(Staircase *) * n_rounds * n_staircases ) );

	return 0;
}

void _init_trial_block( QSP_ARG_DECL  Experiment *exp_p )
{
	int n_staircases;
	int n_rounds;
	int i, j, trial_idx;
	uint32_t *order;
	Staircase **trl_tbl, **stair_tbl;

	reset_experiment(exp_p);

	if( init_trial_tbl(exp_p) < 0 ){
		warn("init_trial_block:  error initializing trial table!?");
		return;
	}

	n_staircases = EXPT_N_STAIRCASES(exp_p);
	n_rounds = EXPT_N_RECORDED_TRIALS(exp_p);

	order = getbuf( sizeof(*order) * n_staircases );
	for(i=0;i<n_staircases;i++) order[i]=i;

	trial_idx = 0;
	trl_tbl = EXPT_TRIAL_TBL(exp_p);

	if( EXPT_STAIR_TBL(exp_p) == NULL ){
		create_stair_table(exp_p);
	}

	stair_tbl = EXPT_STAIR_TBL(exp_p);
	assert( stair_tbl != NULL );
	for(i=0;i<n_rounds;i++){
		permute(order,n_staircases);
		for(j=0;j<n_staircases;j++){
			trl_tbl[trial_idx++] = stair_tbl[ order[j] ];
		}
	}

	SET_EXPT_N_TOTAL_TRIALS(exp_p,n_staircases*n_rounds);
	SET_EXPT_CURR_TRIAL_IDX(exp_p,0);

	sprintf(MSG_STR,"%d",EXPT_N_TOTAL_TRIALS(exp_p));
	assign_reserved_var("n_trials_remaining",MSG_STR);
fprintf(stderr,"init_trial_block:  n_trials_remaining set to %s\n",MSG_STR);

}


static void stair_trials(QSP_ARG_DECL  Experiment *exp_p)
{
	int n_trials_remaining;

	SET_EXPT_CURR_TRIAL_IDX(exp_p,0);

	n_trials_remaining = EXPT_N_TOTAL_TRIALS(exp_p);

	while( (! IS_ABORTING(exp_p) ) && n_trials_remaining > 0 ) {
		Staircase *st_p;

		st_p = EXPT_TRIAL_TBL(exp_p)[ EXPT_CURR_TRIAL_IDX(exp_p) ];
		assert(st_p!=NULL);

		sprintf(MSG_STR,"%d",n_trials_remaining);
		assign_reserved_var("n_trials_remaining",MSG_STR);

		// BUG?  need to check that redo and undo do the right thing???
		step(st_p,exp_p);
		n_trials_remaining --;
	}
}

void _run_stairs(QSP_ARG_DECL  Experiment *exp_p )	/** this does most everything */
{
advise("calling _run_init");
	_run_init(SINGLE_QSP_ARG);
	stair_trials(QSP_ARG exp_p);
	//do_save_data(SINGLE_QSP_ARG);
}


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

	/*
	sprintf(MSG_STR,"create_named_class:  created new class '%s' with index %d",CLASS_NAME(tc_p),CLASS_INDEX(tc_p));
	prt_msg(MSG_STR);
	*/

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

static void update_summary( Summary_Data_Tbl *sdt_p, Sequence_Datum *qd_p )
{
	int val, rsp;

	assert( sdt_p != NULL );
	assert( SUMM_DTBL_SIZE(sdt_p) > 0 );

	val = SEQ_DATUM_XVAL_IDX(qd_p);
	assert( val >= 0 && val < SUMM_DTBL_SIZE(sdt_p) );

	rsp = SEQ_DATUM_RESPONSE(qd_p);

	// This just counts trials and number correct - what about RT???

	if( NORMAL_RESPONSE(rsp) ){
		if( DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdt_p,val) ) == 0 ){	// first record at this level?
			SET_SUMM_DTBL_N(sdt_p, 1 + SUMM_DTBL_N(sdt_p) );
		}

		SET_DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdt_p,val),
			1 + DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdt_p,val) ) );
		if( rsp == SEQ_DATUM_CRCT_RSP(qd_p) ){
			SET_DATUM_NCORR( SUMM_DTBL_ENTRY(sdt_p,val),
				1 + DATUM_NCORR( SUMM_DTBL_ENTRY(sdt_p,val) ) );
		}
		/* else {
fprintf(stderr,"update_class_summary:  response was incorrect\n");

		} */
	}
}

void update_class_summary(Trial_Class *tc_p,Sequence_Datum *qd_p)
{
	if( SEQ_DATUM_CLASS_IDX(qd_p) != CLASS_INDEX(tc_p) ) return;	// don't care about this one

	update_summary( CLASS_SUMM_DTBL(tc_p), qd_p );
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

