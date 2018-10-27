
#ifndef _STC_H_
#define _STC_H_

#include <stdio.h>
#include "quip_prot.h"
#include "dobj_prot.h"
#include "item_obj.h"

//#define MAXVALS		128
#define MAX_X_VALUES	1024

// forward definitions
FWD_TYPEDEF(summary_data_tbl,Summary_Data_Tbl)
FWD_TYPEDEF(sequential_data_tbl,Sequential_Data_Tbl)

typedef struct trial_class {
	Item			tc_item;
	const char *		tc_cmd;		/* general purpose string field */
	int			tc_nstairs;
	int			tc_index;
	Data_Obj *		tc_xval_dp;
	Summary_Data_Tbl *	tc_sdt_p;
	Sequential_Data_Tbl *	tc_qdt_p;
} Trial_Class;

#define CLASS_NAME(tc_p)		(tc_p)->tc_item.item_name
#define CLASS_CMD(tc_p)			(tc_p)->tc_cmd
#define CLASS_N_STAIRS(tc_p)		(tc_p)->tc_nstairs
#define CLASS_INDEX(tc_p)		(tc_p)->tc_index
#define CLASS_SUMM_DTBL(tc_p)		(tc_p)->tc_sdt_p
#define CLASS_SEQ_DTBL(tc_p)		(tc_p)->tc_qdt_p
#define CLASS_XVAL_OBJ(tc_p)		(tc_p)->tc_xval_dp

#define SET_CLASS_CMD(tc_p,v)		(tc_p)->tc_cmd = v
#define SET_CLASS_N_STAIRS(tc_p,v)	(tc_p)->tc_nstairs = v
#define SET_CLASS_INDEX(tc_p,v)		(tc_p)->tc_index = v
#define SET_CLASS_SUMM_DTBL(tc_p,v)	(tc_p)->tc_sdt_p = v
#define SET_CLASS_SEQ_DTBL(tc_p,v)	(tc_p)->tc_qdt_p = v
#define SET_CLASS_XVAL_OBJ(tc_p,v)	(tc_p)->tc_xval_dp = v



typedef struct summary_datum {
	int ntotal;
	int ncorr;
} Summary_Datum;

#define DATUM_NTOTAL(dtm)	(dtm).ntotal
#define DATUM_NCORR(dtm)	(dtm).ncorr
#define DATUM_FRACTION(dtm)	( ((double) DATUM_NCORR(dtm)) / \
					((double) DATUM_NTOTAL(dtm)) )

#define SET_DATUM_NTOTAL(dtm,v)	(dtm).ntotal = v
#define SET_DATUM_NCORR(dtm,v)	(dtm).ncorr = v

struct summary_data_tbl {
	int			sdt_size;	// number of allocated entries (x values)
	int			sdt_npts;	// number that have non-zero n trials
	Summary_Datum *		sdt_data;
	Trial_Class *		sdt_tc_p;	// may be invalid if lumped...
	Data_Obj *		sdt_xval_dp;	// should match class
};

#define SUMM_DTBL_SIZE(sdt_p)		(sdt_p)->sdt_size
#define SUMM_DTBL_N(sdt_p)		(sdt_p)->sdt_npts
#define SUMM_DTBL_DATA(sdt_p)		(sdt_p)->sdt_data
#define SUMM_DTBL_ENTRY(sdt_p,idx)	(sdt_p)->sdt_data[idx]
#define SUMM_DTBL_CLASS(sdt_p)		(sdt_p)->sdt_tc_p
#define SUMM_DTBL_XVAL_OBJ(sdt_p)	(sdt_p)->sdt_xval_dp

#define SET_SUMM_DTBL_SIZE(sdt_p,v)	(sdt_p)->sdt_size = v
#define SET_SUMM_DTBL_N(sdt_p,v)		(sdt_p)->sdt_npts = v
#define SET_SUMM_DTBL_DATA(sdt_p,v)	(sdt_p)->sdt_data = v
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
} ;

#define SEQ_DTBL_LIST(qdt_p)	(qdt_p)->qdt_lp
#define SEQ_DTBL_CLASS(qdt_p)	(qdt_p)->qdt_tc_p

#define SET_SEQ_DTBL_LIST(qdt_p,v)	(qdt_p)->qdt_lp = v
#define SET_SEQ_DTBL_CLASS(qdt_p,v)	(qdt_p)->qdt_tc_p = v


typedef struct staircase {
	Item	stair_item;
#define stair_name	stair_item.item_name

	Trial_Class *	stair_tc_p;
	int	stair_type;
	int	stair_inc;
	int	stair_min_inc;
	int	stair_val;
	int	stair_inc_rsp;
	int	stair_crct_rsp;
	int	stair_last_rsp;
	int	stair_last_rsp3;
	int	stair_last_trial;
	int	stair_index;
	Summary_Data_Tbl	*stair_sdt_p;
	Sequential_Data_Tbl	*stair_qdt_p;
} Staircase;

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

#define UP_DOWN		1
#define TWO_TO_ONE	2
#define THREE_TO_ONE	4


#define NO_RESP		(-1)

#define NO_GOOD		(-2.0)

#define REDO		5	 /* historical, jkr switch box */
#define ABORT		8

#define NO_RSP		(-1)

#define EXP_MENU_PROMPT	"experiment"

/* a global variable BAD... */
extern Data_Obj *global_xval_dp;
extern int is_fc;
extern const char *correct_feedback_string, *incorrect_feedback_string;

/* for fortran stuff... */
#define FTYPE	float

/* BUG these are not thread-safe ...  probably ok */
extern void (*modrt)(QSP_ARG_DECL Trial_Class *);
extern void (*initrt)(void);
extern int (*stmrt)(QSP_ARG_DECL Trial_Class *,int,Staircase *);

/* global variables */
//extern float *xval_array;
//extern int _nvals;

/* extern int nclasses; */
extern int Abort, Redo;
extern int fc_flag;

// BUG, should be an enum for response codes...
#define YES	1
#define NO	2

#define TRANS_UP	1
#define TRANS_DN	2
#define NO_TRANS	4

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


extern Summary_Data_Tbl *_new_summary_data_tbl(QSP_ARG_DECL Data_Obj *dp);
extern void rls_summ_dtbl(Summary_Data_Tbl *sdt_p);
extern  Sequential_Data_Tbl *_new_sequential_data_tbl(SINGLE_QSP_ARG_DECL);
#define new_summary_data_tbl(dp) _new_summary_data_tbl(QSP_ARG  dp)
#define new_sequential_data_tbl() _new_sequential_data_tbl(SINGLE_QSP_ARG)

extern void clear_summary_data( Summary_Data_Tbl *sdt_p );
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
extern void _run_stairs(QSP_ARG_DECL  int np,int nt);
#define run_stairs(np,nt) _run_stairs(QSP_ARG  np,nt)

extern void set_dribble_file(FILE *fp);
extern void set_summary_file(FILE *fp);
extern void _add_stair(QSP_ARG_DECL  int type,Trial_Class *tc_p);
#define add_stair(type,tc_p) _add_stair(QSP_ARG  type,tc_p)

//extern void list_stairs(void);
extern COMMAND_FUNC( del_all_stairs );
extern void si_init(void);

extern Trial_Class *_find_class_from_index(QSP_ARG_DECL  int);
#define find_class_from_index(idx) _find_class_from_index(QSP_ARG  idx)

extern Staircase *_find_stair_from_index(QSP_ARG_DECL  int index);
#define find_stair_from_index(index) _find_stair_from_index(QSP_ARG  index)

extern Trial_Class *new_class(SINGLE_QSP_ARG_DECL);
extern void _del_class(QSP_ARG_DECL  Trial_Class *tc_p);
#define del_class(tc_p) _del_class(QSP_ARG  tc_p)

/* exp.c */
extern Trial_Class *_create_named_class(QSP_ARG_DECL  const char *name);
#define create_named_class(name) _create_named_class(QSP_ARG  name)
extern COMMAND_FUNC( do_delete_all_classes );
extern void nullrt(void);
//extern void make_staircases(SINGLE_QSP_ARG_DECL);
extern COMMAND_FUNC( do_exp_init );

#ifdef FOOBAR
extern void exprmnt(void);
#endif /* FOOBAR */

extern COMMAND_FUNC( do_exp_menu );
extern void _get_rsp_word(QSP_ARG_DECL const char **sptr,const char *def_rsp);
#define get_rsp_word(sptr,def_rsp) _get_rsp_word(QSP_ARG sptr,def_rsp)

extern int _collect_response(QSP_ARG_DECL  const char *s);
#define collect_response(s) _collect_response(QSP_ARG  s)

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
extern void write_sequential_data( Sequential_Data_Tbl *sdt_p, FILE *fp );
extern int dribbling(void);
extern void dribble(Staircase *st_p, int rsp);
extern void close_dribble(void);
extern void init_dribble_file(SINGLE_QSP_ARG_DECL);
extern void mark_drib(FILE *fp);
extern void _write_exp_data(QSP_ARG_DECL  FILE *fp);
extern int  _read_exp_data(QSP_ARG_DECL  FILE *fp);
#define write_exp_data(fp) _write_exp_data(QSP_ARG  fp)
#define read_exp_data(fp) _read_exp_data(QSP_ARG  fp)

/* clrdat.c */

extern void clrdat(SINGLE_QSP_ARG_DECL);
extern void update_summary(Summary_Data_Tbl *sdt_p,Staircase *st_p,int rsp);
extern void append_trial( Sequential_Data_Tbl *qdt_p, Staircase *st_p , int rsp );


/* errbars.c */

void pnt_bars(QSP_ARG_DECL  FILE *fp,Trial_Class *tc_p);


/* stc_menu.c */

extern void general_mod(QSP_ARG_DECL Trial_Class * );
extern int default_stim(QSP_ARG_DECL Trial_Class * ,int val,Staircase *stairp);
extern COMMAND_FUNC( set_2afc );
extern COMMAND_FUNC( stair_menu );


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

#endif /* _STC_H_ */


