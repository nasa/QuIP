
#ifndef _STC_H_
#define _STC_H_

#include <stdio.h>
#include "quip_prot.h"
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
	Summary_Data_Tbl *	tc_sdtp;
	Sequential_Data_Tbl *	tc_qdtp;
} Trial_Class;

#define CLASS_NAME(tcp)			(tcp)->tc_item.item_name
#define CLASS_CMD(tcp)			(tcp)->tc_cmd
#define CLASS_N_STAIRS(tcp)		(tcp)->tc_nstairs
#define CLASS_INDEX(tcp)		(tcp)->tc_index
#define CLASS_SUMM_DATA_TBL(tcp)	(tcp)->tc_sdtp
#define CLASS_SEQ_DATA_TBL(tcp)		(tcp)->tc_qdtp

#define SET_CLASS_CMD(tcp,v)		(tcp)->tc_cmd = v
#define SET_CLASS_N_STAIRS(tcp,v)	(tcp)->tc_nstairs = v
#define SET_CLASS_INDEX(tcp,v)		(tcp)->tc_index = v
#define SET_CLASS_SUMM_DATA_TBL(tcp,v)	(tcp)->tc_sdtp = v
#define SET_CLASS_SEQ_DATA_TBL(tcp,v)	(tcp)->tc_qdtp = v



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

typedef struct summary_data_tbl {
	int			sdt_size;	// number of allocated entries (x values)
	int			sdt_npts;	// number that have non-zero n trials
	Summary_Datum *		sdt_data;
	Trial_Class *		sdt_tc_p;	// may be invalid if lumped...
} Summary_Data_Tbl;

#define SUMM_DTBL_SIZE(dtp)		(dtp)->sdt_size
#define SUMM_DTBL_N(dtp)		(dtp)->sdt_npts
#define SUMM_DTBL_DATA(dtp)		(dtp)->sdt_data
#define SUMM_DTBL_ENTRY(dtp,idx)	(dtp)->sdt_data[idx]
#define SUMM_DTBL_CLASS(dtp)		(dtp)->sdt_tc_p

#define SET_SUMM_DTBL_SIZE(dtp,v)	(dtp)->sdt_size = v
#define SET_SUMM_DTBL_N(dtp,v)		(dtp)->sdt_npts = v
#define SET_SUMM_DTBL_DATA(dtp,v)	(dtp)->sdt_data = v
#define SET_SUMM_DTBL_CLASS(dtp,v)	(dtp)->sdt_tc_p = v

typedef struct sequence_datum {
	int	sqd_class_idx;
	int	sqd_stair_idx;
	int	sqd_xval_idx;
	int	sqd_response;
} Sequence_Datum;

typedef struct sequential_data_tbl {
	List	*qdt_lp;
} Sequential_Data_Tbl;

#define SEQ_DTBL_LIST(qdt_p)	(qdt_p)->qdt_lp

#define SET_SEQ_DTBL_LIST(qdt_p,v)	(qdt_p)->qdt_lp = v


typedef struct staircase {
	Item	stair_item;
#define stair_name	stair_item.item_name

	Trial_Class *	stair_tcp;
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
	Summary_Data_Tbl	*stair_sdtp;
	Sequential_Data_Tbl	*stair_qdtp;
} Staircase;

#define STAIR_CLASS(stair_p)		(stair_p)->stair_tcp
#define STAIR_TYPE(stair_p)		(stair_p)->stair_type
#define STAIR_INC(stair_p)		(stair_p)->stair_inc
#define STAIR_MIN_INC(stair_p)		(stair_p)->stair_min_inc
#define STAIR_VAL(stair_p)		(stair_p)->stair_val
#define STAIR_INC_RSP(stair_p)		(stair_p)->stair_inc_rsp
#define STAIR_CRCT_RSP(stair_p)		(stair_p)->stair_crct_rsp
#define STAIR_LAST_RSP(stair_p)		(stair_p)->stair_last_rsp
#define STAIR_LAST_RSP3(stair_p)		(stair_p)->stair_last_rsp3
#define STAIR_LAST_TRIAL(stair_p)		(stair_p)->stair_last_trial
#define STAIR_INDEX(stair_p)		(stair_p)->stair_index
#define STAIR_DATA_TBL(stair_p)		(stair_p)->stair_sdtp

#define SET_STAIR_CLASS(stair_p,v)	(stair_p)->stair_tcp = v
#define SET_STAIR_TYPE(stair_p,v)		(stair_p)->stair_type = v
#define SET_STAIR_INC(stair_p,v)		(stair_p)->stair_inc = v
#define SET_STAIR_MIN_INC(stair_p,v)	(stair_p)->stair_min_inc = v
#define SET_STAIR_VAL(stair_p,v)		(stair_p)->stair_val = v
#define SET_STAIR_INC_RSP(stair_p,v)	(stair_p)->stair_inc_rsp = v
#define SET_STAIR_CRCT_RSP(stair_p,v)	(stair_p)->stair_crct_rsp = v
#define SET_STAIR_LAST_RSP(stair_p,v)	(stair_p)->stair_last_rsp = v
#define SET_STAIR_LAST_RSP3(stair_p,v)	(stair_p)->stair_last_rsp3 = v
#define SET_STAIR_LAST_TRIAL(stair_p,v)	(stair_p)->stair_last_trial = v
#define SET_STAIR_INDEX(stair_p,v)	(stair_p)->stair_index = v
#define SET_STAIR_DATA_TBL(stair_p,v)	(stair_p)->stair_sdtp = v

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
extern int is_fc;
extern const char *correct_feedback_string, *incorrect_feedback_string;

/* for fortran stuff... */
#define FTYPE	float

/* BUG these are not thread-safe ...  probably ok */
extern void (*modrt)(QSP_ARG_DECL int);
extern void (*initrt)(void);
extern int (*stmrt)(QSP_ARG_DECL Trial_Class *,int,Staircase *);

/* global variables */
extern float *xval_array;
extern int _nvals;
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

ITEM_INTERFACE_PROTOTYPES(Staircase,stair)

#define pick_stair(p)	_pick_stair(QSP_ARG  p)
#define del_stair(s)	_del_stair(QSP_ARG  s)
#define new_stair(s)	_new_stair(QSP_ARG  s)
#define init_stairs()	_init_stairs(SINGLE_QSP_ARG)
#define list_stairs(fp)	_list_stairs(QSP_ARG  fp)

extern Trial_Class *new_class_for_index(QSP_ARG_DECL  int index);
extern void save_response(QSP_ARG_DECL  int rsp,Staircase *stair);
extern void _run_init(SINGLE_QSP_ARG_DECL);
extern void new_exp(SINGLE_QSP_ARG_DECL);
extern void clrit(void);
extern void set_recording(int flag);
extern int makestair( QSP_ARG_DECL  int st, Trial_Class *tcp, int mi, int cr, int ir );
extern COMMAND_FUNC( do_save_data );
#ifdef CATCH_SIGS
extern void icatch(void);
#endif /* CATCH_SIGS */
extern void _run_stairs(QSP_ARG_DECL  int np,int nt);
extern void set_dribble_file(FILE *fp);
extern void set_summary_file(FILE *fp);
extern void add_stair(QSP_ARG_DECL  int type,Trial_Class *tcp);
//extern void list_stairs(void);
extern void delete_staircase(QSP_ARG_DECL  Staircase *stairp);
extern COMMAND_FUNC( del_all_stairs );
extern void si_init(void);

extern List *_class_list(SINGLE_QSP_ARG_DECL);
extern Trial_Class *index_class(QSP_ARG_DECL  int);
extern void del_class(QSP_ARG_DECL  Trial_Class *tcp);
extern Trial_Class *new_class(SINGLE_QSP_ARG_DECL);

#define class_list() _class_list(SINGLE_QSP_ARG)

/* exp.c */
extern COMMAND_FUNC( do_delete_all_classes );
extern void nullrt(void);
extern void run_stairs(void);
//extern void make_staircases(SINGLE_QSP_ARG_DECL);
extern COMMAND_FUNC( do_exp_init );

#ifdef FOOBAR
extern void exprmnt(void);
#endif /* FOOBAR */

extern COMMAND_FUNC( do_exp_menu );
extern void get_rsp_word(QSP_ARG_DECL const char **sptr,const char *def_rsp);
extern int response(QSP_ARG_DECL  const char *s);
extern void init_rps(char *target,const char *s);

/* stc_edit.c */
extern COMMAND_FUNC( staircase_menu );

/* mlfit.c */
extern void ml_fit(QSP_ARG_DECL  Summary_Data_Tbl *dp,int ntrac);
extern void longout(QSP_ARG_DECL  Trial_Class *);
extern void tersout(QSP_ARG_DECL  Trial_Class *);
extern COMMAND_FUNC( constrain_slope );
extern void set_fcflag(int flg);
extern void set_chance_rate(double chance_rate);
extern void ogive_fit( QSP_ARG_DECL  Trial_Class *tcp );
#ifdef QUIK
extern void pntquic(FILE *fp,Trial_Class *tcp,int in_db);
#endif /* QUIK */
extern void print_raw_data(QSP_ARG_DECL  Trial_Class * tcp);

extern double _regr(QSP_ARG_DECL  Summary_Data_Tbl *dp,int first);
extern void _split(QSP_ARG_DECL  Trial_Class * tcp,int wantupper);

#define regr(dp,first) _regr(QSP_ARG  dp,first)
#define split(tcp,wantupper) _split(QSP_ARG  tcp,wantupper)


/* lump.c */
extern COMMAND_FUNC( do_lump );

/* asc_data.c */

extern void write_summary_data( Summary_Data_Tbl *sdt_p, FILE *fp );
extern void write_sequential_data( Sequential_Data_Tbl *sdt_p, FILE *fp );
extern int dribbling(void);
extern void dribble(Staircase *stair_p, int rsp);
extern void close_dribble(void);
extern void init_dribble_file(SINGLE_QSP_ARG_DECL);
extern Summary_Data_Tbl *alloc_data_tbl( Trial_Class *tcp, int size );
extern void mark_drib(FILE *fp);
extern void write_exp_data(QSP_ARG_DECL  FILE *fp);
//extern void wtclass(Trial_Class *tcp,FILE *fp);
extern int  read_exp_data(QSP_ARG_DECL  FILE *fp);
//extern int  rd_one_summ(QSP_ARG_DECL  FILE *fp);
//extern void wt_top(QSP_ARG_DECL  FILE *fp);
extern void setup_classes(QSP_ARG_DECL  int n);
//extern void wt_dribble(FILE *fp,Trial_Class *tcp,int index,int val,int rsp,int crct);
extern void set_xval_xform(const char *s);

/* clrdat.c */

extern void clrdat(SINGLE_QSP_ARG_DECL);
extern void note_trial(Summary_Data_Tbl *sdtp,int val,int rsp,int crct);

/* rxvals.c */

extern void rdxvals(QSP_ARG_DECL  const char *fnam);


/* errbars.c */

void pnt_bars(QSP_ARG_DECL  FILE *fp,Trial_Class *tcp);


/* stc_menu.c */

extern void general_mod(QSP_ARG_DECL int );
extern int default_stim(QSP_ARG_DECL Trial_Class * ,int val,Staircase *stairp);
extern COMMAND_FUNC( set_2afc );
extern COMMAND_FUNC( stair_menu );


/* lookmenu.c */

extern COMMAND_FUNC( lookmenu );

/* weibull.c */

extern void w_analyse( QSP_ARG_DECL  Trial_Class * );
extern void w_tersout(QSP_ARG_DECL  Trial_Class * );
extern void weibull_out(QSP_ARG_DECL  Trial_Class * );

extern void _w_set_error_rate(QSP_ARG_DECL  double er);
#define w_set_error_rate(er) _w_set_error_rate(QSP_ARG  er)

/* xvalmenu.c */

extern int _insure_xval_array(SINGLE_QSP_ARG_DECL);
#define insure_xval_array() _insure_xval_array(SINGLE_QSP_ARG)

extern void set_n_xvals(int n);
extern COMMAND_FUNC( xval_menu );

#endif /* _STC_H_ */


