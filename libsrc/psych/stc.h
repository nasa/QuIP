
#ifndef _STC_H_
#define _STC_H_

#include <stdio.h>
#include "quip_prot.h"
#include "dobj_prot.h"
#include "item_obj.h"

typedef enum {
	YES_INDEX,	//	0
	NO_INDEX,	//	1
	REDO_INDEX,	//	2
	ABORT_INDEX,	//	3
	N_RESPONSES	//	4
} Response_Index;

#define RSP_YES		"yes"
#define RSP_NO		"no"
#define RSP_REDO	"redo"
#define RSP_ABORT	"abort"

typedef struct trial_response {
	int		tr_index;
	const char *	tr_word;
	int		tr_code;
} Trial_Response;

#define MAX_X_VALUES	1024

// forward definitions
FWD_TYPEDEF(summary_data_tbl,Summary_Data_Tbl)
FWD_TYPEDEF(sequential_data_tbl,Sequential_Data_Tbl)

typedef struct trial_class {
	Item			tc_item;
	const char *		tc_stim_cmd;
	const char *		tc_resp_cmd;
	int			tc_nstairs;
	int			tc_index;
	Data_Obj *		tc_xval_dp;
	Summary_Data_Tbl *	tc_sdt_p;
	Sequential_Data_Tbl *	tc_qdt_p;
	List *			tc_staircase_lp;
} Trial_Class;

#define CLASS_NAME(tc_p)		(tc_p)->tc_item.item_name
#define CLASS_STIM_CMD(tc_p)		(tc_p)->tc_stim_cmd
#define CLASS_RESP_CMD(tc_p)		(tc_p)->tc_resp_cmd
#define CLASS_N_STAIRS(tc_p)		(tc_p)->tc_nstairs
#define CLASS_INDEX(tc_p)		(tc_p)->tc_index
#define CLASS_SUMM_DTBL(tc_p)		(tc_p)->tc_sdt_p
#define CLASS_SEQ_DTBL(tc_p)		(tc_p)->tc_qdt_p
#define CLASS_XVAL_OBJ(tc_p)		(tc_p)->tc_xval_dp
#define CLASS_STAIRCASES(tc_p)		(tc_p)->tc_staircase_lp

#define SET_CLASS_STIM_CMD(tc_p,v)	(tc_p)->tc_stim_cmd = v
#define SET_CLASS_RESP_CMD(tc_p,v)	(tc_p)->tc_resp_cmd = v
#define SET_CLASS_N_STAIRS(tc_p,v)	(tc_p)->tc_nstairs = v
#define SET_CLASS_INDEX(tc_p,v)		(tc_p)->tc_index = v
#define SET_CLASS_SUMM_DTBL(tc_p,v)	(tc_p)->tc_sdt_p = v
#define SET_CLASS_SEQ_DTBL(tc_p,v)	(tc_p)->tc_qdt_p = v
#define SET_CLASS_XVAL_OBJ(tc_p,v)	(tc_p)->tc_xval_dp = v
#define SET_CLASS_STAIRCASES(tc_p,v)	(tc_p)->tc_staircase_lp = v

#define summ_data_t	short
#define SUMM_DATA_PREC	PREC_IN

typedef struct summary_datum {
	summ_data_t ntotal;
	summ_data_t ncorr;
} Summary_Datum;

#define N_SUMMARY_DATA_COMPS	2

#define DATUM_NTOTAL(dtm)	(dtm).ntotal
#define DATUM_NCORR(dtm)	(dtm).ncorr
#define DATUM_FRACTION(dtm)	( ((double) DATUM_NCORR(dtm)) / \
					((double) DATUM_NTOTAL(dtm)) )

#define SET_DATUM_NTOTAL(dtm,v)	(dtm).ntotal = v
#define SET_DATUM_NCORR(dtm,v)	(dtm).ncorr = v

struct summary_data_tbl {
	int			sdt_size;	// number of allocated entries (x values)
	int			sdt_npts;	// number that have non-zero n trials
	Data_Obj *		sdt_data_dp;
	Summary_Datum *		sdt_data_ptr;	// points to data in the object...
	Trial_Class *		sdt_tc_p;	// may be invalid if lumped...
	Data_Obj *		sdt_xval_dp;	// should match class
	int			sdt_flags;
};

// flag values for summary data
#define SUMMARY_DATA_DIRTY	1	// written without saved

#define SET_SDT_FLAG_BIT(sdt_p,bit)	SUMM_DTBL_FLAGS(sdt_p) |= bit
#define CLEAR_SDT_FLAG_BIT(sdt_p,bit)	SUMM_DTBL_FLAGS(sdt_p) &= ~(bit)

#define SUMM_DTBL_NEEDS_SAVING(sdt_p)	(SUMM_DTBL_FLAGS(sdt_p) & SUMMARY_DATA_DIRTY)

#define SUMM_DTBL_FLAGS(sdt_p)		(sdt_p)->sdt_flags
#define SET_SUMM_DTBL_FLAGS(sdt_p,v)	(sdt_p)->sdt_flags = v

#define SUMM_DTBL_SIZE(sdt_p)		(sdt_p)->sdt_size
#define SUMM_DTBL_N(sdt_p)		(sdt_p)->sdt_npts
#define SUMM_DTBL_ENTRY(sdt_p,idx)	(sdt_p)->sdt_data_ptr[idx]
#define SUMM_DTBL_DATA_PTR(sdt_p)	(sdt_p)->sdt_data_ptr
#define SUMM_DTBL_DATA_OBJ(sdt_p)	(sdt_p)->sdt_data_dp
#define SUMM_DTBL_CLASS(sdt_p)		(sdt_p)->sdt_tc_p
#define SUMM_DTBL_XVAL_OBJ(sdt_p)	(sdt_p)->sdt_xval_dp

#define SET_SUMM_DTBL_SIZE(sdt_p,v)	(sdt_p)->sdt_size = v
#define SET_SUMM_DTBL_N(sdt_p,v)		(sdt_p)->sdt_npts = v
#define SET_SUMM_DTBL_DATA_OBJ(sdt_p,v)	(sdt_p)->sdt_data_dp = v
#define SET_SUMM_DTBL_DATA_PTR(sdt_p,v)	(sdt_p)->sdt_data_ptr = v
#define SET_SUMM_DTBL_CLASS(sdt_p,v)	(sdt_p)->sdt_tc_p = v
#define SET_SUMM_DTBL_XVAL_OBJ(sdt_p,v)	(sdt_p)->sdt_xval_dp = v

typedef struct sequence_datum {
	int	sqd_class_idx;
	int	sqd_stair_idx;
	int	sqd_xval_idx;
	int	sqd_response;
	int	sqd_correct_response;
} Sequence_Datum;

#define SEQ_DATUM_CLASS_IDX(qd_p)	(qd_p)->sqd_class_idx
#define SEQ_DATUM_STAIR_IDX(qd_p)	(qd_p)->sqd_stair_idx
#define SEQ_DATUM_XVAL_IDX(qd_p)	(qd_p)->sqd_xval_idx
#define SEQ_DATUM_RESPONSE(qd_p)	(qd_p)->sqd_response
#define SEQ_DATUM_CRCT_RSP(qd_p)	(qd_p)->sqd_correct_response

#define SET_SEQ_DATUM_CLASS_IDX(qd_p,v)	(qd_p)->sqd_class_idx = v
#define SET_SEQ_DATUM_STAIR_IDX(qd_p,v)	(qd_p)->sqd_stair_idx = v
#define SET_SEQ_DATUM_XVAL_IDX(qd_p,v)	(qd_p)->sqd_xval_idx = v
#define SET_SEQ_DATUM_RESPONSE(qd_p,v)	(qd_p)->sqd_response = v
#define SET_SEQ_DATUM_CRCT_RSP(qd_p,v)	(qd_p)->sqd_correct_response = v

struct sequential_data_tbl {
	List *		qdt_lp;
	Data_Obj *	qdt_xval_dp;	// should match class
	Trial_Class *	qdt_tc_p;	// may be invalid if lumped...
	int		qdt_flags;
} ;

// flag bits
#define SEQUENTIAL_DATA_DIRTY	1

#define SEQ_DTBL_FLAGS(qdt_p)		(qdt_p)->qdt_flags
#define SET_SEQ_DTBL_FLAGS(qdt_p,v)	(qdt_p)->qdt_flags = v

#define SET_QDT_FLAG_BIT(qdt_p,bits)	SEQ_DTBL_FLAGS(qdt_p) |= bits
#define CLEAR_QDT_FLAG_BIT(qdt_p,bits)	SEQ_DTBL_FLAGS(qdt_p) &= ~(bits)

#define SEQ_DTBL_LIST(qdt_p)	(qdt_p)->qdt_lp
#define SEQ_DTBL_CLASS(qdt_p)	(qdt_p)->qdt_tc_p

#define SEQ_DTBL_NEEDS_SAVING(qdt_p)	(SEQ_DTBL_FLAGS(qdt_p) & SEQUENTIAL_DATA_DIRTY)

#define SET_SEQ_DTBL_LIST(qdt_p,v)	(qdt_p)->qdt_lp = v
#define SET_SEQ_DTBL_CLASS(qdt_p,v)	(qdt_p)->qdt_tc_p = v

typedef enum {
	NO_STAIR_TYPE,
	UP_DOWN,	//	1
	TWO_TO_ONE,	//	2
	THREE_TO_ONE	//	3
} Staircase_Type;


typedef struct staircase {
	Item	stair_item;
#define stair_name	stair_item.item_name

	Trial_Class *	stair_tc_p;
	int	stair_index;

	// These determine the staircase type
	Staircase_Type	stair_type;
	int	stair_min_inc;		// can be positive or negative?
	int	stair_inc_rsp;
	int	stair_crct_rsp;		// does this get diddled by coin flip???

	// These are staircase state variables
	int	stair_val;
	int	stair_inc;		// can be positive or negative?
	int	stair_last_rsp;
	int	stair_last_rsp3;
	int	stair_last_trial;

	Summary_Data_Tbl	*stair_sdt_p;
	Sequential_Data_Tbl	*stair_qdt_p;
} Staircase;

#define STAIR_NAME(st_p)	(st_p)->stair_name
#define STAIR_CLASS(st_p)	(st_p)->stair_tc_p
#define STAIR_TYPE(st_p)	(st_p)->stair_type
#define STAIR_INC(st_p)		(st_p)->stair_inc
#define STAIR_MIN_INC(st_p)	(st_p)->stair_min_inc
#define STAIR_VAL(st_p)		(st_p)->stair_val
#define STAIR_INC_RSP(st_p)	(st_p)->stair_inc_rsp
#define STAIR_CRCT_RSP(st_p)	(st_p)->stair_crct_rsp
#define STAIR_LAST_RSP(st_p)	(st_p)->stair_last_rsp
#define STAIR_LAST_RSP3(st_p)	(st_p)->stair_last_rsp3
#define STAIR_LAST_TRIAL(st_p)	(st_p)->stair_last_trial
#define STAIR_INDEX(st_p)	(st_p)->stair_index
#define STAIR_SUMM_DTBL(st_p)	(st_p)->stair_sdt_p
#define STAIR_SEQ_DTBL(st_p)	(st_p)->stair_qdt_p
#define STAIR_XVAL_OBJ(st_p)	CLASS_XVAL_OBJ( STAIR_CLASS(st_p) )
#define STAIR_MAX_VAL(st_p)	(OBJ_COLS(STAIR_XVAL_OBJ(st_p))-1)

#define SET_STAIR_CLASS(st_p,v)		(st_p)->stair_tc_p = v
#define SET_STAIR_TYPE(st_p,v)		(st_p)->stair_type = v
#define SET_STAIR_INC(st_p,v)		(st_p)->stair_inc = v
#define SET_STAIR_MIN_INC(st_p,v)	(st_p)->stair_min_inc = v
#define SET_STAIR_VAL(st_p,v)		(st_p)->stair_val = v
#define SET_STAIR_INC_RSP(st_p,v)	(st_p)->stair_inc_rsp = v
#define SET_STAIR_CRCT_RSP(st_p,v)	(st_p)->stair_crct_rsp = v
#define SET_STAIR_LAST_RSP(st_p,v)	(st_p)->stair_last_rsp = v
#define SET_STAIR_LAST_RSP3(st_p,v)	(st_p)->stair_last_rsp3 = v
#define SET_STAIR_LAST_TRIAL(st_p,v)	(st_p)->stair_last_trial = v
#define SET_STAIR_INDEX(st_p,v)		(st_p)->stair_index = v
#define SET_STAIR_SUMM_DTBL(st_p,v)	(st_p)->stair_sdt_p = v
#define SET_STAIR_SEQ_DTBL(st_p,v)	(st_p)->stair_qdt_p = v

#define NO_STC_DATA	(-1)

// BUG this should be cleaned up!
#define REDO		5	 /* historical, jkr switch box */
#define ABORT		8

#ifdef FOOBAR

#define NO_RESP		(-1)

#endif // FOOBAR

typedef struct experiment {
	int	expt_flags;

	const char *	question_string;

	int	n_preliminary_trials;
	int	n_recorded_trials;

	// BUG kind of messy?
	int	n_updn_stairs;
	int	n_dnup_stairs;
	int	n_2iup_stairs;
	int	n_2idn_stairs;
	int	n_2up_stairs;
	int	n_2dn_stairs;
	int	n_3up_stairs;
	int	n_3dn_stairs;

} Experiment;

extern Experiment expt1;	// a global singleton

extern int exp_flags;

#define DRIBBLING		1
#define KEYBOARD_RESPONSE	2

#define EXPT_N_PRELIM_TRIALS(exp_p)		((exp_p)->n_preliminary_trials)
#define EXPT_N_RECORDED_TRIALS(exp_p)		((exp_p)->n_recorded_trials)

#define EXPT_N_UPDN(exp_p)		((exp_p)->n_updn_stairs)
#define EXPT_N_DNUP(exp_p)		((exp_p)->n_dnup_stairs)
#define EXPT_N_2IUP(exp_p)		((exp_p)->n_2iup_stairs)
#define EXPT_N_2IDN(exp_p)		((exp_p)->n_2idn_stairs)
#define EXPT_N_2UP(exp_p)		((exp_p)->n_2up_stairs)
#define EXPT_N_2DN(exp_p)		((exp_p)->n_2dn_stairs)
#define EXPT_N_3UP(exp_p)		((exp_p)->n_3up_stairs)
#define EXPT_N_3DN(exp_p)		((exp_p)->n_3dn_stairs)

#define EXPT_FLAGS(exp_p)		((exp_p)->expt_flags)

#define EXPT_QUESTION(exp_p)		((exp_p)->question_string)
#define SET_EXPT_QUESTION(exp_p,v)	(exp_p)->question_string = v

#define SET_EXP_FLAG(exp_p,bit)		(exp_p)->expt_flags |= bit
#define CLEAR_EXP_FLAG(exp_p,bit)	(exp_p)->expt_flags &= ~(bit)

#define IS_DRIBBLING(exp_p)		(EXPT_FLAGS(exp_p) & DRIBBLING)
#define IS_USING_KEYBOARD(exp_p)	(EXPT_FLAGS(exp_p) & KEYBOARD_RESPONSE)

/* a global variable BAD... */
extern Data_Obj *global_xval_dp;
extern int is_fc;
extern const char *correct_feedback_string, *incorrect_feedback_string;

/* for fortran stuff... */
#define FTYPE	float

/* BUG these are not thread-safe ...  probably ok */
// But why do they need to be global???
extern void (*modrt)(QSP_ARG_DECL Trial_Class *);
extern void (*initrt)(void);
extern void (*stim_func)(QSP_ARG_DECL Staircase *);
extern int (*response_func)(QSP_ARG_DECL Staircase *, Experiment *);

/* global variables */
//extern float *xval_array;
//extern int _nvals;

/* extern int nclasses; */
extern int Abort, Redo;
extern int fc_flag;

// BUG, should be an enum for response codes...
#define YES	1
#define NO	2

typedef enum {
	NO_TRANS,	//	0
	TRANS_UP,	//	1
	TRANS_DN	//	2
} Transition_Code;

/** maximum number of preliminary trials */
#define MAXPREL		64

#define NODATA		(-7000)

/* stair.c */

ITEM_INTERFACE_PROTOTYPES(Trial_Class,trial_class)

#define init_trial_classs()	_init_trial_classs(SINGLE_QSP_ARG)
#define pick_trial_class(s)	_pick_trial_class(QSP_ARG  s)
#define trial_class_of(s)	_trial_class_of(QSP_ARG  s)
#define new_trial_class(s)	_new_trial_class(QSP_ARG  s)
#define del_trial_class(s)	_del_trial_class(QSP_ARG  s)
#define list_trial_classs(fp)	_list_trial_classs(QSP_ARG  fp)
#define trial_class_list()	_trial_class_list(SINGLE_QSP_ARG)

ITEM_INTERFACE_PROTOTYPES(Staircase,stair)

#define pick_stair(p)	_pick_stair(QSP_ARG  p)
#define del_stair(s)	_del_stair(QSP_ARG  s)
#define new_stair(s)	_new_stair(QSP_ARG  s)
#define init_stairs()	_init_stairs(SINGLE_QSP_ARG)
#define list_stairs(fp)	_list_stairs(QSP_ARG  fp)
#define stair_list()	_stair_list(SINGLE_QSP_ARG)

extern void _reset_class(QSP_ARG_DECL  Trial_Class *tc_p);
#define reset_class(tc_p) _reset_class(QSP_ARG  tc_p)

extern void _reset_stair(QSP_ARG_DECL  Staircase *st_p);
#define reset_stair(st_p) _reset_stair(QSP_ARG  st_p)

extern void _print_class_info(QSP_ARG_DECL  Trial_Class *tc_p);
#define print_class_info(tc_p) _print_class_info(QSP_ARG  tc_p)

extern void _print_stair_info(QSP_ARG_DECL  Staircase *stc_p);
#define print_stair_info(stc_p) _print_stair_info(QSP_ARG  stc_p)

#define reset_all_classes() _reset_all_classes(SINGLE_QSP_ARG)
extern void _reset_all_classes(SINGLE_QSP_ARG_DECL);

extern void set_response_cmd( Trial_Class *tc_p, const char *s );
extern Summary_Data_Tbl *_new_summary_data_tbl(SINGLE_QSP_ARG_DECL);
extern void rls_summ_dtbl(Summary_Data_Tbl *sdt_p);
extern  Sequential_Data_Tbl *_new_sequential_data_tbl(SINGLE_QSP_ARG_DECL);
#define new_summary_data_tbl() _new_summary_data_tbl(SINGLE_QSP_ARG)
#define new_sequential_data_tbl() _new_sequential_data_tbl(SINGLE_QSP_ARG)

extern void _init_summ_dtbl_for_class( QSP_ARG_DECL  Summary_Data_Tbl * sdt_p, Trial_Class *tc_p );
#define init_summ_dtbl_for_class( sdt_p, tc_p ) _init_summ_dtbl_for_class( QSP_ARG  sdt_p, tc_p )

extern void _clear_sequential_data(QSP_ARG_DECL  Sequential_Data_Tbl *qdt_p);
#define clear_sequential_data(qdt_p) _clear_sequential_data(QSP_ARG  qdt_p)

extern void _clear_summary_data(QSP_ARG_DECL  Summary_Data_Tbl *sdt_p );
#define clear_summary_data(sdt_p ) _clear_summary_data(QSP_ARG  sdt_p )

extern Trial_Class *_new_class_for_index(QSP_ARG_DECL  int index);
#define new_class_for_index(index) _new_class_for_index(QSP_ARG  index)

extern void _save_response(QSP_ARG_DECL  int rsp,Staircase *st_p);
#define save_response(rsp,st_p) _save_response(QSP_ARG  rsp,st_p)

extern void _run_init(SINGLE_QSP_ARG_DECL);
#define run_init() _run_init(SINGLE_QSP_ARG)

extern void new_exp(SINGLE_QSP_ARG_DECL);
extern void clrit(void);
extern void set_recording(int flag);
extern int _make_staircase( QSP_ARG_DECL  int st, Trial_Class *tc_p, int mi, int cr, int ir );
#define make_staircase( st, tc_p, mi, cr, ir ) _make_staircase( QSP_ARG  st, tc_p, mi, cr, ir )

extern COMMAND_FUNC( do_save_data );
#ifdef CATCH_SIGS
extern void icatch(void);
#endif /* CATCH_SIGS */
extern void _run_stairs(QSP_ARG_DECL  Experiment *exp_p);
#define run_stairs(exp_p) _run_stairs(QSP_ARG  exp_p)

extern void set_dribble_file(FILE *fp);
extern void set_summary_file(FILE *fp);
extern void _add_stair(QSP_ARG_DECL  int type,Trial_Class *tc_p);
#define add_stair(type,tc_p) _add_stair(QSP_ARG  type,tc_p)

extern void _delete_all_stairs(SINGLE_QSP_ARG_DECL);
#define delete_all_stairs() _delete_all_stairs(SINGLE_QSP_ARG)

extern void si_init(void);

extern Trial_Class *_find_class_from_index(QSP_ARG_DECL  int);
#define find_class_from_index(idx) _find_class_from_index(QSP_ARG  idx)

extern Staircase *_find_stair_from_index(QSP_ARG_DECL  int index);
#define find_stair_from_index(index) _find_stair_from_index(QSP_ARG  index)

extern Trial_Class *new_class(SINGLE_QSP_ARG_DECL);
extern void _del_class(QSP_ARG_DECL  Trial_Class *tc_p);
#define del_class(tc_p) _del_class(QSP_ARG  tc_p)


extern void update_summary(Summary_Data_Tbl *sdt_p,Staircase *st_p,int rsp);
extern void append_trial( Sequential_Data_Tbl *qdt_p, Staircase *st_p , int rsp );

/* exp.c */

extern void init_experiment( Experiment *exp_p );

extern void _init_responses(SINGLE_QSP_ARG_DECL);
#define init_responses() _init_responses(SINGLE_QSP_ARG)

#define setup_files(exp_p) _setup_files(QSP_ARG  exp_p)
extern void _setup_files(QSP_ARG_DECL  Experiment *exp_p);

#define delete_all_trial_classes() _delete_all_trial_classes(SINGLE_QSP_ARG)
extern void _delete_all_trial_classes(SINGLE_QSP_ARG_DECL);

extern Trial_Class *_create_named_class(QSP_ARG_DECL  const char *name);
#define create_named_class(name) _create_named_class(QSP_ARG  name)
extern void nullrt(void);

extern COMMAND_FUNC( do_exp_menu );
extern void _get_rsp_word(QSP_ARG_DECL const char **sptr,const char *def_rsp);
#define get_rsp_word(sptr,def_rsp) _get_rsp_word(QSP_ARG sptr,def_rsp)

extern int _get_response(QSP_ARG_DECL  Staircase *stc_p, Experiment *exp_p);
#define get_response(stc_p,exp_p) _get_response(QSP_ARG  stc_p,exp_p)

extern void init_rps(char *target,const char *s);

/* stc_edit.c */
extern COMMAND_FUNC( staircase_menu );

/* mlfit.c */
extern void _ml_fit(QSP_ARG_DECL  Summary_Data_Tbl *dp,int ntrac);
#define ml_fit(dp,ntrac) _ml_fit(QSP_ARG  dp,ntrac)
extern void _longout(QSP_ARG_DECL  Trial_Class *);
#define longout(tc_p) _longout(QSP_ARG  tc_p)
extern void _tersout(QSP_ARG_DECL  Trial_Class *);
#define tersout(tc_p) _tersout(QSP_ARG  tc_p)

extern COMMAND_FUNC( constrain_slope );
extern void set_fcflag(int flg);
extern void set_chance_rate(double chance_rate);
extern void _ogive_fit( QSP_ARG_DECL  Trial_Class *tc_p );
#define ogive_fit(tc_p) _ogive_fit(QSP_ARG  tc_p )

#ifdef QUIK
extern void pntquic(FILE *fp,Trial_Class *tc_p,int in_db);
#endif /* QUIK */
extern void _print_class_summary(QSP_ARG_DECL  Trial_Class * tc_p);
extern void _print_class_sequence(QSP_ARG_DECL  Trial_Class * tc_p);
#define print_class_summary(tc_p) _print_class_summary(QSP_ARG  tc_p)
#define print_class_sequence(tc_p) _print_class_sequence(QSP_ARG  tc_p)

extern double _regr(QSP_ARG_DECL  Summary_Data_Tbl *dp,int first);
extern void _split(QSP_ARG_DECL  Trial_Class * tc_p,int wantupper);

#define regr(dp,first) _regr(QSP_ARG  dp,first)
#define split(tc_p,wantupper) _split(QSP_ARG  tc_p,wantupper)


/* lump.c */
extern COMMAND_FUNC( do_lump );

/* asc_data.c */


extern void _iterate_over_classes( QSP_ARG_DECL  void (*func)(QSP_ARG_DECL  Trial_Class *, void *), void *arg);
#define iterate_over_classes(func, arg) _iterate_over_classes( QSP_ARG  func, arg)

extern void write_summary_data( Summary_Data_Tbl *sdt_p, FILE *fp );
extern void _write_sequential_data(QSP_ARG_DECL  Sequential_Data_Tbl *sdt_p, FILE *fp );
#define write_sequential_data(sdt_p, fp ) _write_sequential_data(QSP_ARG  sdt_p, fp )

extern int dribbling(void);
extern void dribble(Staircase *st_p, int rsp);
extern void close_dribble(void);
extern void init_dribble_file(SINGLE_QSP_ARG_DECL);
extern void mark_drib(FILE *fp);
extern void _write_exp_data(QSP_ARG_DECL  FILE *fp);
extern int  _read_exp_data(QSP_ARG_DECL  FILE *fp);
#define write_exp_data(fp) _write_exp_data(QSP_ARG  fp)
#define read_exp_data(fp) _read_exp_data(QSP_ARG  fp)


/* errbars.c */

void _print_error_bars(QSP_ARG_DECL  FILE *fp,Trial_Class *tc_p);
#define print_error_bars(fp,tc_p) _print_error_bars(QSP_ARG  fp,tc_p)


/* stc_menu.c */
extern COMMAND_FUNC( do_staircase_menu );

/* stc_util.c */

extern void general_mod(QSP_ARG_DECL Trial_Class * );
extern void _default_stim(QSP_ARG_DECL Staircase *stairp);
extern int _default_response(QSP_ARG_DECL  Staircase *stc_p, Experiment *exp_p);
#define default_stim(stc_p) _default_stim(QSP_ARG  stc_p)
#define default_response(stc_p,exp_p) _default_response(QSP_ARG  stc_p,exp_p)


/* lookmenu.c */

extern COMMAND_FUNC( lookmenu );

/* weibull.c */

extern void _w_analyse( QSP_ARG_DECL  Trial_Class * );
extern void _w_tersout(QSP_ARG_DECL  Trial_Class * );
extern void _weibull_out(QSP_ARG_DECL  Trial_Class * );
#define w_analyse(tc_p) _w_analyse(QSP_ARG  tc_p )
#define w_tersout(tc_p) _w_tersout(QSP_ARG  tc_p )
#define weibull_out(tc_p) _weibull_out(QSP_ARG  tc_p )

extern void _w_set_error_rate(QSP_ARG_DECL  double er);
#define w_set_error_rate(er) _w_set_error_rate(QSP_ARG  er)

/* xvalmenu.c */

extern int _insure_xval_array(SINGLE_QSP_ARG_DECL);
#define insure_xval_array() _insure_xval_array(SINGLE_QSP_ARG)

extern COMMAND_FUNC( xval_menu );

/* class_menu.c */
extern COMMAND_FUNC( do_class_menu );

/* chngp_menu.c */
extern COMMAND_FUNC( do_exp_param_menu );

#endif /* _STC_H_ */


