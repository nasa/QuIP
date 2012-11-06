
#ifndef _QUERY_H_

#ifdef __cplusplus
extern "C" {
#endif

#define _QUERY_H_

#include "quip_config.h"

#include <stdio.h>

#if HAVE_STDINT_H
#include <stdint.h>
#endif

#if HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_PTHREADS
#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif /* HAVE_PTHREAD_H */
#endif /* HAVE_PTHREADS */

#include "typedefs.h"
#include "strbuf.h"
#include "items.h"
#include "hash.h"
#include "void.h"		/* defn of VOID */


/* This was 256,
 * things crapped out when I had an input line with 258 chars, there
 * is a bug which should be fixed, but for now it is expedient just
 * to spend a little memory.
 */

#ifndef LLEN
#define LLEN	1024
#endif /* ! LLEN */

typedef struct foreloop {
	const char *	f_varname;
	List *		f_lp;
	Node *		f_np;
} Foreloop;

typedef struct query {
	const char *	q_lbptr;		/* pointer into line buffer */
	FILE *		q_file;			/* input file (NULL for text) */
	int		q_count;		/* loop counter */
	char *		(*q_readfunc)(TMP_QSP_ARG_DECL  char *,int,FILE *);
	char *		q_text;
	int		q_txtsiz;
	int		q_txtfree;
	const char **	q_args;			/* macro args */
	int		q_havtext;
	int		q_saving;		/* storing loop text flag */
	struct macro *	q_macro;		/* pointer to macro (if any) */
	const char *	q_thenclause;		/* temporary buffer for IF clauses */
	int		q_flags;		/* miscellaneous */
	FILE *		q_dupfile;
	Foreloop *	q_fore;
	int		q_lineno;		/* line number of current word */
	int		q_rdlineno;		/* line number of last line read (incl lookahead) */
#ifdef MAC
	char *		q_extra;
#endif /* MAC */
} Query;

/* number of query structs */
#define MAX_Q_LVLS	64

#define SET_Q_FLAG( qsp, mask )			(qsp)->qs_query[(qsp)->qs_level].q_flags |= ( mask )
#define CLR_Q_FLAG( qsp, mask )			(qsp)->qs_query[(qsp)->qs_level].q_flags &= ~( mask )

/* flags values */
#define Q_SOCKET	1
#define Q_INTERACTIVE	2
#define Q_MPASSED	4
#define Q_FIRST_WORD	8
#define Q_LOOKAHEAD	16
#define Q_LINEDONE	32
#define Q_BUFFERED_TEXT	64
#define Q_PIPE		128

#define IS_INTERACTIVE( qp )		((( qp ) -> q_flags) & Q_INTERACTIVE)
#define NOT_PASSED(qp)			((( qp )->q_flags & Q_MPASSED)==0)
#define IS_DUPING			( THIS_QSP->qs_query[QLEVEL].q_dupfile != NULL )
#define FIRST_WORD_ON_LINE		( THIS_QSP->qs_query[QLEVEL].q_flags & Q_FIRST_WORD )
#define IS_PIPE( qp )			((( qp ) -> q_flags) & Q_PIPE)

/* size of memory chunks for loop buffer */
#define LOOPSIZE	256

/* some codes used for printing.
 * These used to be in data_obj.h, and there was a global var that held the current value.
 * To have a per-thread value, it has to be defined here...
 */

typedef enum {
	FMT_DECIMAL,
	FMT_HEX,
	FMT_OCTAL,
	FMT_UDECIMAL,
	FMT_FLOAT,
	FMT_POSTSCRIPT,
	N_PRINT_FORMATS		/* must be last */
} Number_Fmt;

/* stuff that used to be in vars.h */

typedef struct var {
	Item		v_item;
	const char *	v_value;
	void		(*v_func)(struct var *);/* used for reserved vars */
#ifdef THREAD_SAFE_QUERY
	pthread_mutex_t	v_mutex;
	int		v_flags;
#define VAR_LOCKED		1
#define VAR_IS_LOCKED(vp)	(vp->v_flags & VAR_LOCKED)
#endif /* THREAD_SAFE_QUERY */

} Var;

#define v_name	v_item.item_name
#define VARNAME_PROMPT		"variable name"
#define NO_VAR		((struct var *) NULL )


typedef struct query_stream {
	Item		qs_item;
#define qs_name	qs_item.item_name
	uint32_t	qs_flags;
	Query		qs_query[MAX_Q_LVLS];
	const char *	qs_fn_stack[MAX_Q_LVLS];	// why not make part of Query?
	int		qs_level;
	int		qs_fn_depth;		/* filename depth */
/* probably this should depend on MAX_Q_LVLS... was 512, which was
 * ok when MAX_Q_LVLS was 24, but then we made it 64...
 */
#define FNS_LEN		2048
#define N_RETSTRS	12
#define MAX_VAR_BUFS	32

	char		qs_fns_str[FNS_LEN];
	int		qs_lookahead_level;
	int		qs_former_level;
	int		qs_which_retstr;
	String_Buf	qs_retstr_arr[N_RETSTRS];
	char		qs_lbuf[MAX_Q_LVLS][LLEN];		/* holds a single line */
								/* BUG? fixed size? */
	String_Buf	qs_cv_buf[MAX_VAR_BUFS];
	int		qs_cv_which;
	String_Buf	qs_scratchbuf;
	String_Buf	qs_result;
	char		qs_prompt[LLEN];
	Item_Type *	qs_cmd_itp;				/* what is this? */
	int		qs_serial;
#define FIRST_QUERY_SERIAL 0
	int		qs_expr_level;				/* for vectree */
	int		qs_ascii_level;				/* data/ascii.c */
	Number_Fmt	qs_fmt_code;

	// Support for vector expressions (vectree.y)
	// These use to be static globals in vectree.y
	const char *	qs_yy_cp;	/* points to yy_word_buf ? */
	int		qs_semi_seen;
	int		qs_end_seen;
#define YY_LLEN 1024	/* BUG! for some reason, when we
			 * insert a bit of code in a script file,
			 * it all gets taken as one line!?
			 */
	char		qs_yy_input_line[YY_LLEN];
	char		qs_yy_last_line[YY_LLEN];
	char		qs_estr[YY_LLEN];
	const char *	qs_curr_string/*=estr */;	// NEED TO INIT!!
	double		qs_final;
	int		qs_lastlineno;
	int		qs_parser_lineno;
	/* Vec_Expr_Node * */
	void *		qs_top_node/* =NO_VEXPR_NODE */;
							// NEED TO INIT!!!
#define YY_CP		(THIS_QSP->qs_yy_cp)
#define SEMI_SEEN	(THIS_QSP->qs_semi_seen)
#define END_SEEN	(THIS_QSP->qs_end_seen)
#define YY_INPUT_LINE	(THIS_QSP->qs_yy_input_line)
#define YY_LAST_LINE	(THIS_QSP->qs_yy_last_line)
#define VEXP_STR	(THIS_QSP->qs_estr)
#define CURR_STRING	(THIS_QSP->qs_curr_string)
#define FINAL		(THIS_QSP->qs_final)
#define LASTLINENO	(THIS_QSP->qs_lastlineno)
#define PARSER_LINENO	(THIS_QSP->qs_parser_lineno)
#define TOP_NODE	(((Query_Stream *)THIS_QSP)->qs_top_node)


	// Support for scalar expressions (nexpr.y)
	// These used to be static globals in nexpr.y
	int		qs_which_expr_str;		// Needs to be initialized to 0
	int		qs_edepth;			// Needs to be intitialized to -1
#define MAXEDEPTH	20
	const char *	qs_yystrptr[MAXEDEPTH];
	void *		qs_final_expr_node_p;		// really Scalar_Expr_Node *



	const char *	qs_yy_original;
	int		qs_in_pexpr;

#define IN_PEXPR		(THIS_QSP->qs_in_pexpr)
#define WHICH_EXPR_STR		(THIS_QSP->qs_which_expr_str)
#define EDEPTH			(THIS_QSP->qs_edepth)
#define FINAL_EXPR_NODE_P	((Scalar_Expr_Node *)(THIS_QSP->qs_final_expr_node_p))
#define YYSTRPTR		THIS_QSP->qs_yystrptr
#define YY_ORIGINAL		THIS_QSP->qs_yy_original

	int		qs_chewing;
	List *		qs_chew_list;

#define CHEWING		THIS_QSP->qs_chewing
#define CHEW_LIST	THIS_QSP->qs_chew_list

	Var		qs_tmpvar;

	char		qs_ctxname[LLEN];

#ifdef HAVE_PTHREADS
	pthread_t	qs_thr;
	char		qs_error_string[LLEN];
	char		qs_msg_str[LLEN];
	int		qs_history;
#endif /* HAVE_PTHREADS */
} Query_Stream;

/* flag bits */
#define QS_INITED		1	// 0x001
#define QS_EXPAND_MACS		2	// 0x002
#define QS_HAD_INTR		4	// 0x004
#define QS_INTERACTIVE_TTYS	8	// 0x008
#define QS_FORMAT_PROMPT	16	// 0x010
#define QS_FORCE_PROMPT		32	// 0x020
#define QS_LOOKAHEAD_ENABLED	64	// 0x040
#define QS_STILL_TRYING		128	// 0x080
#define QS_STRIPPING_QUOTES	256	// 0x100
#define QS_COMPLETING		512	// 0x200
#define QS_BUILTINS_INITED	1024	// 0x400
#define QS_HALTING		2048	// 0x800


#define IS_HALTING(qsp)		((qsp)->qs_flags & QS_HALTING)

#define HAD_INTERRUPT		(QUERY_FLAGS & QS_HAD_INTR)

#define VAR_DELIM		'$'
#define IS_LEGAL_VAR_CHAR(c)	(isalnum(c) || (c)=='_' )

#define ERROR_STR_LEN	LLEN+100

//extern Query_Stream *curr_qsp;	// BUG - no current_qsp when multi-threaded!?
extern Query_Stream *default_qsp;	// the first one created

//#define THREAD_SAFE_QUERY

#ifdef THREAD_SAFE_QUERY

#define QSP_ARG_DECL			Query_Stream *qsp,
#define SINGLE_QSP_ARG_DECL		Query_Stream *qsp
#define QSP_ARG				qsp,
#define NULL_QSP_ARG			NULL,
#define DEFAULT_QSP_ARG			default_qsp,
#define SGL_DEFAULT_QSP_ARG		default_qsp
#define SINGLE_QSP_ARG			qsp
#define QSP_DECL			Query_Stream *qsp;
#define THIS_QSP			qsp
#define NULL_QSP			NULL
#define ERROR_STRING			((Query_Stream *)qsp)->qs_error_string
#define error_string			((Query_Stream *)qsp)->qs_error_string
#define DEFAULT_ERROR_STRING		default_qsp->qs_error_string
#define MSG_STR				qsp->qs_msg_str
#define HISTORY_FLAG			qsp->qs_history
#define FGETS				qpfgets

// For debugging...
#define WHENCE(func_name)	qsp->qs_name,#func_name

extern char *qpfgets(TMP_QSP_ARG_DECL  char *buf, int size, FILE *fp);

#else /* ! THREAD_SAFE_QUERY */

#define QSP_ARG_DECL
#define SINGLE_QSP_ARG_DECL		void
#define QSP_ARG
#define NULL_QSP_ARG
#define DEFAULT_QSP_ARG
#define SGL_DEFAULT_QSP_ARG
#define SINGLE_QSP_ARG
#define QSP_DECL
#define THIS_QSP			default_qsp
#define NULL_QSP
#define ERROR_STRING			error_string
#define DEFAULT_ERROR_STRING		error_string
#define error_string			THIS_QSP->qs_error_string
#define MSG_STR				msg_str
#define HISTORY_FLAG			history
#define FGETS				fgets

// For debugging...
#define WHENCE(func_name)		"",#func_name

#endif /* ! THREAD_SAFE_QUERY */

#define TMPVAR				THIS_QSP->qs_tmpvar

#define QUERY_PROMPT			THIS_QSP->qs_prompt
#define QUERY_FLAGS			THIS_QSP->qs_flags
#define QLEVEL				THIS_QSP->qs_level
#define INIT_QSP			THIS_QSP=new_query_stream(NULL_QSP_ARG  "default_query_stream");
#define BUILTINS_INITED			(QUERY_FLAGS & QS_BUILTINS_INITED)

/* libdata stuff */
#define ASCII_LEVEL			THIS_QSP->qs_ascii_level
#define THE_FMT_CODE			THIS_QSP->qs_fmt_code


/* Users make tables of these things to build menus
 */

typedef struct command {
	const char *	cmd_sel;
	void		(*cmd_func)(SINGLE_QSP_ARG_DECL);
	const char *	cmd_help;
} Command ;

#define NULL_COMMAND	(const char *)NULL,			\
			(void (*)(SINGLE_QSP_ARG_DECL))NULL,	\
			(const char *)NULL

#define NO_COMMAND	((Command *) NULL)


/* menu's used to have their own item types, now they are contexts */

/* We recode the user's spec's as an item in order to use the item
 * package's name lookup capabilities.
 */

typedef struct command_item {
	Item		ci_item;
	Command *	ci_cmdp;	/* points to an entry in the table */
	Query_Stream *	ci_qsp;		/* for deletion - see ncmds.c */
} Command_Item;

/* conflict with cimage.h ... */
/* #define ci_name		ci_item.item_name */

#define NO_COMMAND_ITEM		((Command_Item *)NULL)


/* We used to have a stack of menus (command tables);
 * Now we use the context stacking mechanisms instead, so
 * we don't need these stacks.
 */


/* We use the number of contexts to determine whether we have popped
 * the last menu and should exit.  There is the default context
 * (which is not used, but is there anyway), then the builtin menu
 * and the help menu get pushed and should always be there.
 * That accounts for 3 contexts.  We also insist that
 * there also be at least 1 application menu pushed;
 * Therefore MIN_CMD_DEPTH is 4.
 * We will exit when the command stack depth reaches this value.
 */

#define MIN_CMD_DEPTH	4



/* used to be in its own file mypipe.h */

#if HAVE_POPEN

typedef struct my_pipe {
	const char *	p_name;
	const char *	p_cmd;
	FILE *		p_fp;
	int		p_flgs;
} Pipe ;

#define NO_PIPE ((Pipe *) NULL)

/* flag values */
#define READ_PIPE	1
#define WRITE_PIPE	2

extern void close_pipe(QSP_ARG_DECL  Pipe *pp);
ITEM_INTERFACE_PROTOTYPES( Pipe, pipe )

#endif /* HAVE_POPEN */




#define COMMAND_FUNC(name)			void name(SINGLE_QSP_ARG_DECL)
#define PICK_OBJ(pmpt)				pick_obj(QSP_ARG  pmpt)
#define PICK_CUDA_VWR(pmpt)			pick_cuda_vwr(QSP_ARG  pmpt)
#define PICK_VWR(pmpt)				pick_vwr(QSP_ARG  pmpt)
#define PICK_CURSOR(pmpt)			pick_cursor(QSP_ARG  pmpt)
#define PICK_DRAGG(pmpt)			pick_dragg(QSP_ARG  pmpt)
#define PICK_DISP_OBJ(pmpt)			pick_disp_obj(QSP_ARG  pmpt)
#define PICK_FBI(pmpt)				pick_fbi(QSP_ARG  pmpt)
#define PICK_MVI(pmpt)				pick_mvi(QSP_ARG  pmpt)
#define PICK_LB(pmpt)				pick_lb(QSP_ARG  pmpt)
#define PICK_OPT_PARAM(pmpt)			pick_opt_param(QSP_ARG  pmpt)
#define PICK_OPT_PKG(pmpt)			pick_opt_pkg(QSP_ARG  pmpt)
#define PICK_STC(pmpt)				pick_stc(QSP_ARG  pmpt)
#define PICK_IMG_FILE(pmpt)			pick_img_file(QSP_ARG  pmpt)
#define PICK_DL(pmpt)				pick_dl(QSP_ARG  pmpt)
#define PICK_PIPE(pmpt)				pick_pipe(QSP_ARG  pmpt)
#define PICK_DATA_AREA(pmpt)			pick_data_area(QSP_ARG  pmpt)
#define PICK_PORT(pmpt)				pick_port(QSP_ARG  pmpt)
#define PICK_RV_INODE(pmpt)			pick_rv_inode(QSP_ARG  pmpt)
#define PICK_VF(pmpt)				pick_vf(QSP_ARG  pmpt)
#define PICK_OCVI(pmpt)				pick_ocvi(QSP_ARG  pmpt)
#define PICK_OCV_MEM(pmpt)			pick_ocv_mem(QSP_ARG  pmpt)
#define PICK_OCV_SCANNER(pmpt)			pick_ocv_scanner(QSP_ARG  pmpt)
#define PICK_OCV_SEQ(pmpt)			pick_ocv_seq(QSP_ARG  pmpt)
#define PICK_CASCADE(pmpt)			pick_ocv_ccasc(QSP_ARG  pmpt)
#define PICK_VIDEO_DEV(pmpt)			pick_video_dev(QSP_ARG pmpt)
#define PICK_VCAM(pmpt)				pick_vcam(QSP_ARG  pmpt)
#define PICK_VPORT(pmpt)			pick_vport(QSP_ARG  pmpt)
#define PICK_VISCA_CMD(pmpt)			pick_visca_cmd(QSP_ARG  pmpt)
#define PICK_CMD_SET(pmpt)			pick_cmd_set(QSP_ARG  pmpt)
#define PICK_VISCA_INQ(pmpt)			pick_visca_inq(QSP_ARG  pmpt)
#define PICK_SUBRT(pmpt)			pick_subrt(QSP_ARG  pmpt)
#define PICK_GRABBER(pmpt)			pick_grabber_(QSP_ARG  pmpt)
#define PICK_SEQ(pmpt)				pick_mviseq(QSP_ARG  pmpt)
#define PICK_OPT_ELT(pmpt)			pick_opt_elt(QSP_ARG  pmpt)
#define PICK_PROBLEM(pmpt)			pick_problem(QSP_ARG pmpt)
#define PICK_SYM(pmpt)				pick_sym(QSP_ARG pmpt)
#define PICK_PROCEDURE(pmpt)			pick_proc(QSP_ARG pmpt)
#define PICK_REQUIREMENT(pmpt)			pick_req(QSP_ARG pmpt)
#define PICK_CUDEV(pmpt)			pick_cudev(QSP_ARG pmpt)

#define PUSHCMD(ctbl,pmpt)			pushcmd(QSP_ARG  ctbl,pmpt)

#define HOW_MANY(pmpt)				how_many(QSP_ARG  pmpt)
#define HOW_MUCH(pmpt)				how_much(QSP_ARG  pmpt)
#define ASKIF(pmpt)				askif(QSP_ARG  pmpt)
#define CONFIRM(pmpt)				confirm(QSP_ARG  pmpt)
#define WHICH_ONE(pmpt,n,choices)		which_one(QSP_ARG  pmpt,n,choices)
#define WHICH_ONE2(pmpt,n,choices)		which_one2(QSP_ARG  pmpt,n,choices)

#define WARN( msg )				q_warn( QSP_ARG msg )
#define ERROR1( msg )				q_error1( QSP_ARG msg )
#define NERROR1( msg )				error1( DEFAULT_QSP_ARG msg )

#define QUIP_PROGRAM( ctbl , prompt )					\
									\
	QSP_DECL							\
	INIT_QSP							\
	rcfile(QSP_ARG  av[0]);						\
	set_args(ac,av);						\
	PUSHCMD( ctbl , prompt );					\
	while(1) do_cmd(SINGLE_QSP_ARG);



#define PROMPT_FORMAT	"Enter %s: "

/* Class definitions used to be in items.h, but now member info func
 * has a QSP in it, it has to be here...
 */

/* stuff to make super-classes */

typedef struct item_class {
	Item	icl_item;
	List *	icl_lp;
	int	icl_flags;
} Item_Class;

#define icl_name		icl_item.item_name

#define NEED_CLASS_CHOICES	1


typedef struct member_info {
	Item_Type *	mi_itp;
	void *		mi_data;
	Item *		(*mi_lookup)(QSP_ARG_DECL  const char *);
} Member_Info;

#define NO_ITEM_CLASS	((Item_Class *)NULL)
#define NO_MEMBER_INFO	((Member_Info *)NULL)

#ifdef DEBUG
#include "debug.h"
extern debug_flag_t qldebug;
extern debug_flag_t lah_debug;
#endif /* DEBUG */


/* some stuff that used to be in myerror.h */

//extern char error_string[ERROR_STR_LEN];
extern char msg_str[ERROR_STR_LEN];

extern char *	show_printable(const char *);
extern void	identify_self(int);
extern void	set_progname(char *program_name);
extern const char *	tell_progname(void);
extern void	set_max_warnings(int);
extern int	count_warnings(void);
extern void	clear_warnings(void);
extern void	set_warn_func(void (*func)(QSP_ARG_DECL  const char *));
extern void	set_error_func(void (*func)(QSP_ARG_DECL  const char *));
extern void	set_advise_func(void (*func)(const char *));
extern void	set_prt_msg_frag_func(void (*func)(const char *));
extern void	warn(QSP_ARG_DECL  const char *msg);
#define NWARN(s)	warn(DEFAULT_QSP_ARG  s)
extern void	error1(QSP_ARG_DECL  const char *msg);
extern void	advise(const char *msg);
#define ADVISE(s)	advise(s)
#define NADVISE(s)	advise(s)
extern FILE *	tell_msgfile(void);
extern FILE *	tell_errfile(void);
extern void	prt_msg(const char *msg);
extern void	prt_msg_frag(const char *msg);
extern void	error_init(void);
extern int	do_on_exit(void (*func)(void));
extern void	nice_exit(int status);
extern void	error_redir(FILE *fp);
extern void	output_redir(FILE *fp);
extern void	error2(QSP_ARG_DECL  const char *progname,const char *msg);
extern void	revert_tty(void);
extern void	tty_warn(QSP_ARG_DECL  const char *);
extern void	tty_advise(const char *);
extern void	tell_sys_error(const char *);

/* end of old myerror.h */

/* global variable(s) */
#ifdef THREAD_SAFE_QUERY
extern int n_active_threads;	// Number of qsp's
#endif /* THREAD_SAFE_QUERY */

/* prototypes */

ITEM_INTERFACE_PROTOTYPES(Query_Stream,qstream)

extern void freevar(QSP_ARG_DECL  Var *vp);
ITEM_INTERFACE_PROTOTYPES(Var,var_)
#define VAR_OF(name)		var_of(QSP_ARG name)
#define PICK_VAR(pmpt)		pick_var_(QSP_ARG pmpt)

extern Var *assign_var(QSP_ARG_DECL  const char *,const char *);
#define ASSIGN_VAR( s1 , s2 )	assign_var(QSP_ARG  s1 , s2 )

/* vars.c */
extern void var_stats(SINGLE_QSP_ARG_DECL);
// Do these really need to be in the external API?? BUG?
extern void find_vars(QSP_ARG_DECL  const char *);
extern void search_vars(QSP_ARG_DECL  const char *);
extern double dvarexists(QSP_ARG_DECL  const char *);
extern void init_reserved_vars(SINGLE_QSP_ARG_DECL);
extern void list_var_contexts(SINGLE_QSP_ARG_DECL);
extern Item_Context * new_var_context(QSP_ARG_DECL  const char *name);
extern void show_var_ctx_stk(SINGLE_QSP_ARG_DECL);
extern void push_var_ctx(QSP_ARG_DECL  const char *name);
extern void pop_var_ctx(SINGLE_QSP_ARG_DECL);
extern void restrict_var_context(QSP_ARG_DECL  int flag);

/* end of old vars.h */


/* rcfile.c */

extern void rcfile(QSP_ARG_DECL char *progname);

/* qword.c */

extern void inhibit_next_prompt_format(SINGLE_QSP_ARG_DECL);
extern char *more_macro_text(char *,int,FILE *);
extern void set_rf( QSP_ARG_DECL    char * (*func)(TMP_QSP_ARG_DECL  char *, int , FILE *) );
extern long get_rf( SINGLE_QSP_ARG_DECL );
extern void init_query_stream( Query_Stream *qsp );
extern void first_query_stream(Query_Stream *qsp);
extern void pop_input_file(SINGLE_QSP_ARG_DECL);
extern const char * current_input_file( SINGLE_QSP_ARG_DECL );
extern const char * current_input_stack( SINGLE_QSP_ARG_DECL );
extern void redir(QSP_ARG_DECL FILE *fp);
#define REDIR(fp)		redir(QSP_ARG fp)
extern void push_input_file( QSP_ARG_DECL   const char *name);
#define PUSH_INPUT_FILE(name)		push_input_file(QSP_ARG name)
extern void openloop(QSP_ARG_DECL int n);
extern void fore_loop(QSP_ARG_DECL Foreloop *frp);
extern void zap_fore(Foreloop *frp);
extern void popfile(SINGLE_QSP_ARG_DECL);
extern char *poptext(TMP_QSP_ARG_DECL  char *buf,int size,FILE *fp);
extern void pushtext(QSP_ARG_DECL const char *text);
#define PUSHTEXT(text)		pushtext(QSP_ARG  text)
extern void fullpush(QSP_ARG_DECL const char *text);
extern void closeloop(SINGLE_QSP_ARG_DECL);
extern void _whileloop(QSP_ARG_DECL int value);
extern void push_if(QSP_ARG_DECL const char *text);
extern void set_args(QSP_ARG_DECL  int ac,char **av);
extern void make_prompt(QSP_ARG_DECL char buffer[],const char *s);
extern void show_query_level(QSP_ARG_DECL int i);
extern void qdump( SINGLE_QSP_ARG_DECL );
extern void set_qflags( QSP_ARG_DECL  int);
extern FILE *tfile(SINGLE_QSP_ARG_DECL);
extern Query_Stream *new_query_stream(QSP_ARG_DECL  const char *);
extern struct var *var_of(QSP_ARG_DECL const char *);


/* BUG some (most) of these should be private functions... */

extern void dup_word(QSP_ARG_DECL const char *);
extern void end_dupline(SINGLE_QSP_ARG_DECL);

extern void tog_pmpt(SINGLE_QSP_ARG_DECL);
extern const char * qword(QSP_ARG_DECL const char *pline);
extern void qgivup(SINGLE_QSP_ARG_DECL);
extern char * rd_word(SINGLE_QSP_ARG_DECL);
extern const char * gword(QSP_ARG_DECL const char *pline);
extern void savechar(QSP_ARG_DECL Query *qp,int c);
extern void savetext(QSP_ARG_DECL Query *qp,const char *buf);
#ifdef HAVE_HISTORY
extern Query *hist_select(QSP_ARG_DECL char *buf,const char *pline);
extern void set_completion(SINGLE_QSP_ARG_DECL);
#endif /* HAVE_HISTORY */
extern const char * steal_line(QSP_ARG_DECL   const char *);
extern const char * qline(QSP_ARG_DECL const char *pline);
extern void ql_debug(SINGLE_QSP_ARG_DECL);
extern const char * nextline(QSP_ARG_DECL const char *pline);
extern int dupout(QSP_ARG_DECL   FILE *fp);
extern const char *getmarg(QSP_ARG_DECL int index);
extern void pop_it(QSP_ARG_DECL int n);
extern int intractive(SINGLE_QSP_ARG_DECL);
#define INTRACTIVE()	intractive(SINGLE_QSP_ARG)
extern char * rdmtext(SINGLE_QSP_ARG_DECL);
extern void readtty(SINGLE_QSP_ARG_DECL);
extern void closetty(void);
extern const char * nameof(QSP_ARG_DECL   const char *s);
#define NAMEOF(pmpt)				nameof(QSP_ARG  pmpt)
extern const char * nameof2(QSP_ARG_DECL   const char *s);
extern int askif(QSP_ARG_DECL   const char *s);
extern int confirm(QSP_ARG_DECL   const char *s);
extern  int which_one(QSP_ARG_DECL const char *prompt, int n, const char **choices);
extern  int which_one2(QSP_ARG_DECL const char *prompt, int n, const char **choices);
extern void lookahead_til(QSP_ARG_DECL   int);
extern void push_lookahead(SINGLE_QSP_ARG_DECL);
extern void pop_lookahead(SINGLE_QSP_ARG_DECL);
extern int enable_lookahead(QSP_ARG_DECL   int);
extern void disable_lookahead( SINGLE_QSP_ARG_DECL );
extern void enable_stripping_quotes( SINGLE_QSP_ARG_DECL );
extern void disable_stripping_quotes( SINGLE_QSP_ARG_DECL );
extern int tell_qlevel( SINGLE_QSP_ARG_DECL );
#define TELL_QLEVEL		tell_qlevel(SINGLE_QSP_ARG)
extern void lookahead( SINGLE_QSP_ARG_DECL );

extern void showm(Query *qp);

extern Item *pick_item(QSP_ARG_DECL Item_Type *,const char *);
extern Item_Type *pick_ittyp(QSP_ARG_DECL const char *prompt);
#define PICK_ITTYP(pmpt)		pick_ittyp(QSP_ARG  pmpt)

extern long how_many(QSP_ARG_DECL   const char *);
extern double how_much(QSP_ARG_DECL   const char *);
//extern Macro *pick_macro(QSP_ARG_DECL   const char *);
extern const char *var_value(QSP_ARG_DECL   const char *);

/* ncmds.c */
extern void set_bis(Command *ctbl,const char *pmpt,Command *hctbl,const char *hpmpt);
extern void mtype_init(void);
extern void pushcmd(QSP_ARG_DECL Command *ctbl,const char *pmpt);
extern void popcmd(SINGLE_QSP_ARG_DECL);
#define POPCMD() popcmd(SINGLE_QSP_ARG)
extern COMMAND_FUNC( top_menu );
extern void push_top_menu(SINGLE_QSP_ARG_DECL);
extern void reload_menu(QSP_ARG_DECL  const char *pmpt,Command *ctbl);

#ifdef SUBMENU
extern int index_menu( Command_Menu *mp, const char *key );
extern int submenu(const char *pmpt,const char *sel,const char *help,const char *subpmpt);
#endif /* SUBMENU */

extern int _add_wcmd( const char *pmpt, const char *sel, void (*func)(VOID), const char *help );
extern void add_wcmd( const char *sel, void (*func)(VOID), const char *help );
extern int wp_cmp( const void *wp1, const void *wp2 );
extern void hhelpme( QSP_ARG_DECL  const char * );
extern int wscantbl( Command *ctbl, const char *word_string );
extern void getwcmd( SINGLE_QSP_ARG_DECL );
extern int	cmd_depth(SINGLE_QSP_ARG_DECL);


/* do_cmd.c */
extern void comp_cmd(int);
extern void do_cmd(SINGLE_QSP_ARG_DECL);
extern void add_cmd_callback(void (*func)(void) );
extern void callbacks_on(SINGLE_QSP_ARG_DECL);
extern void callbacks_off(SINGLE_QSP_ARG_DECL);

/* query.c */
extern void tell_input_location( SINGLE_QSP_ARG_DECL );
extern void q_warn(QSP_ARG_DECL const char *msg );
extern void q_error1(QSP_ARG_DECL const char *msg );

/* support/permission.c */
extern void check_suid_root(void);

/* builtin.c */
extern void set_output_file(QSP_ARG_DECL  const char *);

/* trynice.c ? */
extern FILE *trynice(QSP_ARG_DECL  const char *filename, const char *mode);
#define TRYNICE(filename,mode)	trynice( QSP_ARG  filename, mode )

/* complete.c */
extern void simulate_typing(const char *);


/* item class functions - from where? */
extern Item_Class *	new_item_class(QSP_ARG_DECL  const char *);
extern void		add_items_to_class(Item_Class *,Item_Type *,
				void *,Item * (*)(QSP_ARG_DECL  const char *));
extern Member_Info *	get_member_info(QSP_ARG_DECL  Item_Class *,const char *);
extern Item *		get_member(QSP_ARG_DECL  Item_Class *,const char *);

// from node.c
extern void	report_node_data(SINGLE_QSP_ARG_DECL);

// from items.c
extern Item *	new_item(QSP_ARG_DECL  Item_Type * item_type,const char *name,size_t size);
extern void	list_item_contexts(QSP_ARG_DECL  Item_Type *);
extern void	set_del_method(QSP_ARG_DECL  Item_Type *,void (*func)(TMP_QSP_ARG_DECL  Item *) );
extern void		push_item_context(QSP_ARG_DECL  Item_Type *,Item_Context *);
#define PUSH_ITEM_CONTEXT(itp,icp)	push_item_context(QSP_ARG  itp, icp)
extern Item_Context *	pop_item_context(QSP_ARG_DECL  Item_Type *);
#define POP_ITEM_CONTEXT(itp)	pop_item_context(QSP_ARG  itp)
extern void zombie_item(QSP_ARG_DECL  Item_Type * item_type,Item *ip);

extern Item *	get_item(QSP_ARG_DECL  Item_Type * item_type,const char *name);
extern Item *	item_of(QSP_ARG_DECL  Item_Type * item_type,const char *name);
extern void	delete_item_context(QSP_ARG_DECL  Item_Context *);
extern Item_Context *	ctx_of(QSP_ARG_DECL  const char *);
extern Item_Type * new_item_type(QSP_ARG_DECL  const char * atypename);
extern void list_ittyps(SINGLE_QSP_ARG_DECL);
extern void		list_ctxs(SINGLE_QSP_ARG_DECL);
extern Item_Context *	create_item_context(QSP_ARG_DECL  Item_Type *,const char *);
extern void init_item_hist(QSP_ARG_DECL  Item_Type * itp, const char *prompt);
extern Item_Type *get_item_type(QSP_ARG_DECL const char *);
#ifdef THREAD_SAFE_QUERY
extern void report_mutex_error(QSP_ARG_DECL  int,const char *);
#endif /* THREAD_SAFE_QUERY */

/* try_hard.c */
extern FILE *try_open(QSP_ARG_DECL  const char *filename, const char *mode);
extern FILE *try_hard(QSP_ARG_DECL  const char *filename, const char *mode);
#define TRY_OPEN(fn,m)	try_open(QSP_ARG  fn, m)
#define TRY_HARD(fn,m)	try_hard(QSP_ARG  fn, m)

// debug.c
extern void set_debug(QSP_ARG_DECL  Debug_Module *);
extern void clr_debug(QSP_ARG_DECL  Debug_Module *);
extern debug_flag_t add_debug_module(QSP_ARG_DECL  const char *);

#ifdef __cplusplus
}
#endif

#endif /* _QUERY_H_ */

