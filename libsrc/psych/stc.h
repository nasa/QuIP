

#ifndef NO_STAIR

#ifdef INC_VERSION
char VersionId_inc_stc[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#include <stdio.h>
#include "quip_prot.h"
#include "item_obj.h"

#ifdef SUN
#define STEPIT
#endif /* SUN */

//#define MAXVALS		128
#define MAX_X_VALUES	1024

typedef struct datum {
	int ntotal;
	int ncorr;
} Datum;

#define DATUM_NTOTAL(dtm)	(dtm).ntotal
#define DATUM_NCORR(dtm)	(dtm).ncorr
#define DATUM_FRACTION(dtm)	( ((double) DATUM_NCORR(dtm)) / \
					((double) DATUM_NTOTAL(dtm)) )

#define SET_DATUM_NTOTAL(dtm,v)	(dtm).ntotal = v
#define SET_DATUM_NCORR(dtm,v)	(dtm).ncorr = v

typedef struct data_tbl {
	int	dt_size;	// number of allocated entries (x values)
	int	dt_npts;	// number that have non-zero n trials
	Datum	*dt_data;
} Data_Tbl;

#define DTBL_SIZE(dtp)		(dtp)->dt_size
#define DTBL_N(dtp)		(dtp)->dt_npts
#define DTBL_DATA(dtp)		(dtp)->dt_data
#define DTBL_ENTRY(dtp,idx)	(dtp)->dt_data[idx]

#define SET_DTBL_SIZE(dtp,v)		(dtp)->dt_size = v
#define SET_DTBL_N(dtp,v)		(dtp)->dt_npts = v
#define SET_DTBL_DATA(dtp,v)		(dtp)->dt_data = v


typedef struct trial_class {
	Item		tc_item;
	const char *	tc_cmd;		/* general purpose string field */
	int		tc_nstairs;
	int		tc_index;
	Data_Tbl *	tc_dtp;
} Trial_Class;

#define NO_CLASS	((Trial_Class *)NULL)

#define CLASS_NAME(tcp)			(tcp)->tc_item.item_name
#define CLASS_CMD(tcp)			(tcp)->tc_cmd
#define CLASS_N_STAIRS(tcp)		(tcp)->tc_nstairs
#define CLASS_INDEX(tcp)		(tcp)->tc_index
#define CLASS_DATA_TBL(tcp)		(tcp)->tc_dtp

#define SET_CLASS_CMD(tcp,v)		(tcp)->tc_cmd = v
#define SET_CLASS_N_STAIRS(tcp,v)	(tcp)->tc_nstairs = v
#define SET_CLASS_INDEX(tcp,v)		(tcp)->tc_index = v
#define SET_CLASS_DATA_TBL(tcp,v)	(tcp)->tc_dtp = v

typedef struct staircase {
	Item	stc_item;
#define stc_name	stc_item.item_name

	Trial_Class *	stc_tcp;
	int	stc_type;
	int	stc_inc;
	int	stc_min_inc;
	int	stc_val;
	int	stc_inc_rsp;
	int	stc_crct_rsp;
	int	stc_last_rsp;
	int	stc_last_rsp3;
	int	stc_last_trial;
	int	stc_index;
} Staircase;

#define NO_STAIR	((Staircase *)NULL)

#define STAIR_CLASS(stc_p)		(stc_p)->stc_tcp
#define STAIR_TYPE(stc_p)		(stc_p)->stc_type
#define STAIR_INC(stc_p)		(stc_p)->stc_inc
#define STAIR_MIN_INC(stc_p)		(stc_p)->stc_min_inc
#define STAIR_VAL(stc_p)		(stc_p)->stc_val
#define STAIR_INC_RSP(stc_p)		(stc_p)->stc_inc_rsp
#define STAIR_CRCT_RSP(stc_p)		(stc_p)->stc_crct_rsp
#define STAIR_LAST_RSP(stc_p)		(stc_p)->stc_last_rsp
#define STAIR_LAST_RSP3(stc_p)		(stc_p)->stc_last_rsp3
#define STAIR_LAST_TRIAL(stc_p)		(stc_p)->stc_last_trial
#define STAIR_INDEX(stc_p)		(stc_p)->stc_index

#define SET_STAIR_CLASS(stc_p,v)	(stc_p)->stc_tcp = v
#define SET_STAIR_TYPE(stc_p,v)		(stc_p)->stc_type = v
#define SET_STAIR_INC(stc_p,v)		(stc_p)->stc_inc = v
#define SET_STAIR_MIN_INC(stc_p,v)	(stc_p)->stc_min_inc = v
#define SET_STAIR_VAL(stc_p,v)		(stc_p)->stc_val = v
#define SET_STAIR_INC_RSP(stc_p,v)	(stc_p)->stc_inc_rsp = v
#define SET_STAIR_CRCT_RSP(stc_p,v)	(stc_p)->stc_crct_rsp = v
#define SET_STAIR_LAST_RSP(stc_p,v)	(stc_p)->stc_last_rsp = v
#define SET_STAIR_LAST_RSP3(stc_p,v)	(stc_p)->stc_last_rsp3 = v
#define SET_STAIR_LAST_TRIAL(stc_p,v)	(stc_p)->stc_last_trial = v
#define SET_STAIR_INDEX(stc_p,v)	(stc_p)->stc_index = v

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

#define TR_UP	1
#define TR_DN	2
#define NO_TR	4

/** maximum number of preliminary trials */
#define MAXPREL		64

#define NODATA		(-7000)

/* stair.c */

ITEM_INTERFACE_PROTOTYPES(Trial_Class,trial_class)
#define PICK_TRIAL_CLASS(s)	pick_trial_class(QSP_ARG  s)

ITEM_INTERFACE_PROTOTYPES(Staircase,stc)

#define PICK_STC(p)	pick_stc(QSP_ARG  p)

extern Trial_Class *class_for(QSP_ARG_DECL  int index);
extern void save_response(QSP_ARG_DECL  int rsp,Staircase *stc);
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
extern void list_stairs(void);
extern void del_stair(QSP_ARG_DECL  Staircase *stcp);
extern COMMAND_FUNC( del_all_stairs );
extern void si_init(void);

extern List *class_list(SINGLE_QSP_ARG_DECL);
extern Trial_Class *index_class(QSP_ARG_DECL  int);
extern void del_class(QSP_ARG_DECL  Trial_Class *tcp);
extern Trial_Class *new_class(SINGLE_QSP_ARG_DECL);

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
extern void ml_fit(QSP_ARG_DECL  Data_Tbl *dp,int ntrac);
extern void longout(QSP_ARG_DECL  Trial_Class *);
extern void tersout(QSP_ARG_DECL  Trial_Class *);
extern COMMAND_FUNC( constrain_slope );
extern void set_fcflag(int flg);
extern void set_chance_rate(double chance_rate);
extern double regr(Data_Tbl *dp,int first);
extern void analyse( QSP_ARG_DECL  Trial_Class *tcp );
#ifdef QUIK
extern void pntquic(FILE *fp,Trial_Class *tcp,int in_db);
#endif /* QUIK */
extern void print_raw_data(QSP_ARG_DECL  Trial_Class * tcp);
extern void split(QSP_ARG_DECL  Trial_Class * tcp,int wantupper);



/* lump.c */
extern COMMAND_FUNC( lump );

/* ptoz.c */
// prototypes moved to include/function.h
//extern double ptoz(double prob);
//extern double ztop(double zscore);

/* asc_data.c */

extern int dribbling(void);
extern void dribble(Staircase *stc_p, int rsp);
extern void close_dribble(void);
extern void init_dribble_file(SINGLE_QSP_ARG_DECL);
extern Data_Tbl *alloc_data_tbl( Trial_Class *tcp, int size );
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
extern void note_trial(Trial_Class *tcp,int val,int rsp,int crct);

/* rxvals.c */

extern void rdxvals(QSP_ARG_DECL  const char *fnam);


/* errbars.c */

void pnt_bars(QSP_ARG_DECL  FILE *fp,Trial_Class *tcp);


/* stc_menu.c */

extern void general_mod(QSP_ARG_DECL int );
extern int default_stim(QSP_ARG_DECL Trial_Class * ,int val,Staircase *stcp);
extern COMMAND_FUNC( set_2afc );
extern COMMAND_FUNC( stair_menu );


/* lookmenu.c */

extern COMMAND_FUNC( lookmenu );

/* weibull.c */

extern void w_analyse( QSP_ARG_DECL  Trial_Class * );
extern void w_tersout(QSP_ARG_DECL  Trial_Class * );
extern void w_set_error_rate(double er);
extern void weibull_out(QSP_ARG_DECL  Trial_Class * );


/* xvalmenu.c */

extern COMMAND_FUNC( xval_menu );

#endif /* ! NO_STAIR */


