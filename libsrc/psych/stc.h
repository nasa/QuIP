

#ifndef NO_STAIR

#ifdef INC_VERSION
char VersionId_inc_stc[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#include <stdio.h>
#include "items.h"
#include "node.h"
#include "query.h"

#ifdef SUN
#define STEPIT
#endif /* SUN */

#define MAXVALS		128

typedef struct datum {
	short ntotal;
	short ncorr;
} Datum;

typedef struct data_tbl {
	short	d_npts;
	Datum	d_data[MAXVALS];
} Data_Tbl;

typedef struct trial_class {
	const char *	cl_name;
	const char *	cl_data;	/* general purpose string field */
	short		cl_nstairs;
	short		cl_index;
	Data_Tbl *	cl_dtp;
} Trial_Class;

#define NO_CLASS	((Trial_Class *)NULL)


typedef struct staircase {
	Item	stc_item;
#define stc_name	stc_item.item_name

	Trial_Class *	stc_clp;
	short	stc_type;
	short	stc_inc;
	short	stc_mininc;
	short	stc_val;
	short	stc_incrsp;
	short	stc_crctrsp;
	short	stc_lstrsp;
	short	stc_lr3;
	short	stc_lasttr;
	short	stc_index;
} Staircase;

#define NO_STAIR	((Staircase *)NULL)

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
extern int (*stmrt)(QSP_ARG_DECL int,int,Staircase *);

/* global variables */
extern float xval_array[];
extern int _nvals;
/* extern int nclasses; */
extern int Abort, Redo;
extern int fc_flag;

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
ITEM_INTERFACE_PROTOTYPES(Staircase,stc)

extern Trial_Class *class_for(QSP_ARG_DECL  int index);
extern void save_response(QSP_ARG_DECL  int rsp,Staircase *stc);
extern void _run_init(SINGLE_QSP_ARG_DECL);
extern void new_exp(SINGLE_QSP_ARG_DECL);
extern void clrit(void);
extern void set_recording(int flag);
extern int makestair( QSP_ARG_DECL  int st, int sc, int mi, int cr, int ir );
extern COMMAND_FUNC( savdat );
#ifdef CATCH_SIGS
extern void icatch(void);
#endif /* CATCH_SIGS */
extern void _run_stairs(QSP_ARG_DECL  int np,int nt);
extern void set_dribble_file(FILE *fp);
extern void set_summary_file(FILE *fp);
extern void add_stair(QSP_ARG_DECL  int type,int condition);
extern void list_stairs(void);
extern void del_stair(QSP_ARG_DECL  Staircase *stcp);
extern COMMAND_FUNC( del_all_stairs );
extern void si_init(void);

extern List *class_list(SINGLE_QSP_ARG_DECL);
extern Trial_Class *index_class(QSP_ARG_DECL  int);
extern void del_class(QSP_ARG_DECL  Trial_Class *clp);
extern Trial_Class *new_class(SINGLE_QSP_ARG_DECL);

/* exp.c */
extern COMMAND_FUNC( delcnds );
extern void nullrt(void);
extern void run_stairs(void);
extern void make_staircases(SINGLE_QSP_ARG_DECL);
extern COMMAND_FUNC( do_exp_init );

#ifdef FOOBAR
extern void exprmnt(void);
#endif /* FOOBAR */

extern COMMAND_FUNC( exp_menu );
extern void get_rsp_word(QSP_ARG_DECL const char **sptr,const char *def_rsp);
extern int response(QSP_ARG_DECL  const char *s);
extern void init_rps(char *target,const char *s);

/* stc_edit.c */
extern COMMAND_FUNC( staircase_menu );

/* mlfit.c */
extern void ml_fit(QSP_ARG_DECL  Data_Tbl *dp,int ntrac);
extern void longout(int);
extern void tersout(int);
extern COMMAND_FUNC( constrain_slope );
extern void set_fcflag(int flg);
extern void set_chance_rate(double chance_rate);
extern double regr(Data_Tbl *dp,int first);
extern void analyse( QSP_ARG_DECL  int itbl );
#ifdef QUIK
extern void pntquic(FILE *fp,int cl,int in_db);
#endif /* QUIK */
extern void pntdata(QSP_ARG_DECL  int cl);
extern void split(QSP_ARG_DECL  int cl,int wantupper);



/* lump.c */
extern COMMAND_FUNC( lump );

/* ptoz.c */
extern double ptoz(double prob);
extern double ztop(double zscore);

/* asc_data.c */

extern void markdrib(FILE *fp);
extern void wtdata(QSP_ARG_DECL  FILE *fp);
extern void wtclass(Trial_Class *clp,FILE *fp);
extern int  rddata(QSP_ARG_DECL  FILE *fp);
extern int  rd_one_summ(QSP_ARG_DECL  FILE *fp);
extern void wt_top(QSP_ARG_DECL  FILE *fp);
extern void setup_classes(QSP_ARG_DECL  int n);
extern void wt_dribble(FILE *fp,Trial_Class *clp,int index,int val,int rsp,int crct);
extern void set_xval_xform(const char *s);

/* clrdat.c */

extern void clrdat(SINGLE_QSP_ARG_DECL);
extern void note_trial(Trial_Class *clp,int val,int rsp,int crct);

/* rxvals.c */

extern void rdxvals(QSP_ARG_DECL  const char *fnam);


/* errbars.c */

void pnt_bars(QSP_ARG_DECL  FILE *fp,int cl);


/* stc_menu.c */

extern void general_mod(QSP_ARG_DECL int );
extern void interpret_text_fragment(QSP_ARG_DECL const char *s);
extern int general_stim(QSP_ARG_DECL int ,int val,Staircase *stcp);
extern COMMAND_FUNC( set_2afc );
extern COMMAND_FUNC( stair_menu );


/* lookmenu.c */

extern COMMAND_FUNC( lookmenu );

/* weibull.c */

extern void w_analyse( QSP_ARG_DECL  int itbl );
extern void w_tersout(int );
extern void w_set_error_rate(double er);
extern void weibull_out(int );


/* xvalmenu.c */

extern COMMAND_FUNC( xval_menu );

#endif /* ! NO_STAIR */


