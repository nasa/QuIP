#ifndef _QUERY_STACK_H_
#define _QUERY_STACK_H_

#include "quip_config.h"

#include "query.h"
#include "stack.h"
#include "query_bits.h"
#include "variable.h"
#include "quip_menu.h"
//#include "qs_basic.h"
#include "ascii_fmts.h"
#include "string_ref.h"

#ifdef BUILD_FOR_IOS
#include <dispatch/dispatch.h>
#endif /* BUILD_FOR_IOS */

//#include "my_curl.h"

// For which parser?  scalar or vector???

typedef struct parser_data {
	void *		pd_top_enp;	// really Vec_Expr_Node
	void *		pd_last_enp;	// really Vec_Expr_Node
	int		pd_end_seen;	// boolean flag?
	const char *	pd_yy_cp;
	int		pd_expr_level;
	int		pd_last_line_num;
	int		pd_parser_line_num;
	char *		pd_yy_last_line;
	char *		pd_yy_input_line;
	int		pd_semi_seen;		// boolean flag...
	char *		pd_expr_string;
	int		pd_edepth;		// same as expr_level???
	const char *	pd_curr_string;
	double		pd_final;
	//int parser_lineno;
	String_Ref *	pd_curr_infile;
	List *		pd_subroutine_context_stack /* =NO_LIST */ ;
} Parser_Data;

// This struct is used to push text frags around...

struct mouthful {
	const char *text;
	const char *filename;
} ;

#define DBL_QUOTE	'"'
#define SGL_QUOTE	'\''

// The retstr buffers are used to scan words out of input lines...
// But sometimes one of these buffers can itself become the input;
// The buffers need to be flagged as in use or not, rather than
// a circular array we should have a list of free buffers...

//#define N_RETSTRS	20
// What is this var buf stuff??
// The same argument about retstr seems to apply here as well...

#define MAX_VAR_BUFS	32

#define FIRST_QUERY_SERIAL 0

struct query_stack {
	Item		qs_item;
//	String_Buf *	qs_retstr[N_RETSTRS];
//	int		qs_which_retstr;
	List *		qs_retstr_lp;
	String_Buf *	qs_varbuf[MAX_VAR_BUFS];
	int		qs_which_var_buf;
	Menu *		qs_builtin_menu;
	Menu *		qs_help_menu;
	String_Buf *	qs_prompt_sbp;
	Stack *		qs_menu_stack;
	Stack *		qs_query_stack;

	Stack *		qs_var_fmt_stack;
	char *		qs_number_fmt_string;
	char		qs_gfmt_str[8];
	char		qs_xfmt_str[8];
	char		qs_ofmt_str[8];
	char		qs_dfmt_str[8];
	String_Buf *	qs_av_sbp;

	int		qs_max_warnings;
	int		qs_n_warnings;
	int		qs_num_warnings;
	int		qs_flags;	// where are the flag bits declared???
	int		qs_level;
	int		qs_chew_level;
#ifdef NOT_USED
	int		qs_lookahead_level;
	int		qs_former_level;
#endif /* NOT_USED */
	int		qs_ascii_level;
	int		qs_fmt_code;
	int		qs_serial;
	FILE *		qs_error_fp;
	FILE *		qs_msg_fp;
	// BUG - we should phase these out in favor of string_buf's...
	char 		qs_error_string[LLEN];
	char 		qs_msg_str[LLEN];
//	char **		qs_expr_strs;
	String_Buf *	qs_result;
	String_Buf *	qs_scratch;
	List *		qs_chew_list;
#define CHEW_LIST	QS_CHEW_LIST(THIS_QSP)
	List *		qs_callback_lp;
	List *		qs_event_lp;
	Variable *	qs_tmpvar;
#ifdef THREAD_SAFE_QUERY
#ifdef HAVE_PTHREADS
	pthread_t	qs_thr;
#endif /* HAVE_PTHREADS */
#endif /* THREAD_SAFE_QUERY */

	Parser_Data *	qs_parser_data;

	struct dobj_ascii_info *	qs_dai_p;
	Item_Type *	qs_picking_item_itp;

#ifdef BUILD_FOR_IOS
#ifdef USE_QS_QUEUE
	dispatch_queue_t qs_queue;
#endif /* USE_QS_QUEUE */
#endif /* BUILD_FOR_IOS */

	const char *	qs_prev_log_msg;
	unsigned long	qs_log_msg_count;

#ifdef HAVE_LIBCURL
	Curl_Info *	qs_curl_info;


#endif // HAVE_LIBCURL
	
};

#define QS_PICKING_ITEM_ITP(qsp)		((qsp)->qs_picking_item_itp)
#define SET_QS_PICKING_ITEM_ITP(qsp,itp)	(qsp->qs_picking_item_itp) = itp

#define	qs_top_enp		qs_parser_data->pd_top_enp
#define	qs_last_enp		qs_parser_data->pd_last_enp
#define	qs_end_seen		qs_parser_data->pd_end_seen
#define	qs_yy_cp		qs_parser_data->pd_yy_cp
#define	qs_expr_level		qs_parser_data->pd_expr_level
#define	qs_last_line_num	qs_parser_data->pd_last_line_num
#define	qs_parser_line_num	qs_parser_data->pd_parser_line_num
#define	qs_yy_last_line		qs_parser_data->pd_yy_last_line
#define	qs_yy_input_line	qs_parser_data->pd_yy_input_line
#define	qs_semi_seen		qs_parser_data->pd_semi_seen
#define	_qs_expr_string		qs_parser_data->pd_expr_string
#define	qs_edepth		qs_parser_data->pd_edepth
#define	_qs_curr_string		qs_parser_data->pd_curr_string
#define	qs_final		qs_parser_data->pd_final
#define	qs_curr_infile		qs_parser_data->pd_curr_infile
#define qs_subrt_ctx_stack	qs_parser_data->pd_subroutine_context_stack


#define ALLOC_QS_PARSER_DATA(qsp)	SET_QS_PARSER_DATA(qsp,getbuf(sizeof(Parser_Data)))

#define INSURE_QS_PARSER_DATA(qsp)	if( QS_PARSER_DATA(qsp)==NULL )		\
						ALLOC_QS_PARSER_DATA(qsp);



#define NO_QUERY_STACK ((Query_Stack *)NULL)


// this indexing of the list is probably backwards!?
#define QS_QRY_STACK(qsp)		(qsp)->qs_query_stack
#define SET_QS_QRY_STACK(qsp,stkp)	(qsp)->qs_query_stack = stkp


// Query_Stack stuff

#ifdef THREAD_SAFE_QUERY
#define QS_ERROR_STRING(qsp)		(qsp)->qs_error_string

#else /* ! THREAD_SAFE_QUERY */
#define QS_ERROR_STRING(qsp)		DEFAULT_ERROR_STRING
#endif /* ! THREAD_SAFE_QUERY */

#define QS_ERROR_FILE(qsp)		(qsp)->qs_error_fp
#define QS_MSG_FILE(qsp)		(qsp)->qs_msg_fp
#define SET_QS_ERROR_FILE(qsp,fp)	(qsp)->qs_error_fp = fp
#define SET_QS_MSG_FILE(qsp,fp)		(qsp)->qs_msg_fp = fp

#define QS_HAS_SOMETHING(qsp)		(QLEVEL>=0 && QRY_HAS_TEXT(CURR_QRY(qsp)))
#define QS_DO_CMD(qsp)			qs_do_cmd(qsp)
#define QS_RDLINENO(qsp)		QRY_RDLINENO(CURR_QRY(qsp))

#define QS_MENU_STACK(qsp)		(qsp)->qs_menu_stack
#define SET_QS_MENU_STACK(qsp,stkp)	(qsp)->qs_menu_stack = stkp

#define QS_VAR_FMT_STACK(qsp)		(qsp)->qs_var_fmt_stack
#define SET_QS_VAR_FMT_STACK(qsp,stkp)	(qsp)->qs_var_fmt_stack = stkp
#define QS_NUMBER_FMT(qsp)		(qsp)->qs_number_fmt_string
#define SET_QS_NUMBER_FMT(qsp,s)	(qsp)->qs_number_fmt_string = s
#define QS_GFORMAT(qsp)			(qsp)->qs_gfmt_str
#define QS_XFORMAT(qsp)			(qsp)->qs_xfmt_str
#define QS_DFORMAT(qsp)			(qsp)->qs_dfmt_str
#define QS_OFORMAT(qsp)			(qsp)->qs_ofmt_str
#define SET_QS_GFORMAT(qsp,s)		strcpy((qsp)->qs_gfmt_str,s)
#define SET_QS_XFORMAT(qsp,s)		strcpy((qsp)->qs_xfmt_str,s)
#define SET_QS_DFORMAT(qsp,s)		strcpy((qsp)->qs_dfmt_str,s)
#define SET_QS_OFORMAT(qsp,s)		strcpy((qsp)->qs_ofmt_str,s)
#define QS_AV_STRINGBUF(qsp)		(qsp)->qs_av_sbp
#define SET_QS_AV_STRINGBUF(qsp,sbp)	(qsp)->qs_av_sbp = sbp

#define QS_PARSER_DATA(qsp)		(qsp)->qs_parser_data
#define SET_QS_PARSER_DATA(qsp,d)	(qsp)->qs_parser_data = d

#define QS_TMPVAR(qsp)			(qsp)->qs_tmpvar
#define QS_BUILTIN_MENU(qsp)		(qsp)->qs_builtin_menu
#define SET_QS_BUILTIN_MENU(qsp,mp)	(qsp)->qs_builtin_menu=mp
#define QS_HELP_MENU(qsp)		(qsp)->qs_help_menu
#define SET_QS_HELP_MENU(qsp,mp)	(qsp)->qs_help_menu=mp
#define SET_QS_TMPVAR(qsp,vp)		(qsp)->qs_tmpvar=vp
#define QS_CHEW_LIST(qsp)		(qsp)->qs_chew_list
#define SET_QS_CHEW_LIST(qsp,lp)	(qsp)->qs_chew_list =  lp

#define QS_PROMPT_SB(qsp)		(qsp)->qs_prompt_sbp
#define SET_QS_PROMPT_SB(qsp,sbp)	(qsp)->qs_prompt_sbp = sbp
#define QS_PROMPT_STR(qsp)		SB_BUF((qsp)->qs_prompt_sbp)
#define CLEAR_QS_PROMPT(qsp)		QS_PROMPT_STR(qsp)[0] = 0

#define QS_WHICH_VAR_BUF(qsp)		(qsp)->qs_which_var_buf
#define SET_QS_WHICH_VAR_BUF(qsp,v)	(qsp)->qs_which_var_buf=v
#define QS_VAR_BUF(qsp,idx)		(qsp)->qs_varbuf[idx]
#define SET_QS_VAR_BUF(qsp,idx,sbp)	(qsp)->qs_varbuf[idx]=sbp
#define QS_RESULT(qsp)			(qsp)->qs_result
#define SET_QS_RESULT(qsp,sbp)		(qsp)->qs_result = sbp
//#define QS_SCRATCH(qsp)			(qsp)->qs_scratch
#define SET_QS_SCRATCH(qsp,sbp)		(qsp)->qs_scratch = sbp
#define QS_NAME(qsp)			(qsp)->qs_item.item_name
#define SET_QS_NAME(qsp,s)		(qsp)->qs_item.item_name = s
#define _QS_SERIAL(qsp)			(qsp)->qs_serial
#define SET_QS_SERIAL(qsp,n)		(qsp)->qs_serial=n
#define QS_LEVEL(qsp)			(qsp)->qs_level
#define SET_QS_LEVEL(qsp,l)		(qsp)->qs_level = l
#define QS_CHEW_LEVEL(qsp)		(qsp)->qs_chew_level
#define SET_QS_CHEW_LEVEL(qsp,l)	(qsp)->qs_chew_level = l
#define QS_ASCII_LEVEL(qsp)		(qsp)->qs_ascii_level
#define QS_FMT_CODE(qsp)		(qsp)->qs_fmt_code
#define SET_QS_FMT_CODE(qsp,c)		(qsp)->qs_fmt_code=c
#define QS_LOOKAHEAD_LEVEL(qsp)		(qsp)->qs_lookahead_level
#define SET_QS_LOOKAHEAD_LEVEL(qsp,l)	(qsp)->qs_lookahead_level = l
#ifdef NOT_USED
#define QS_FORMER_LEVEL(qsp)		(qsp)->qs_former_level
#define SET_QS_FORMER_LEVEL(qsp,l)	(qsp)->qs_former_level = l
#endif /* NOT_USED */

#define QS_FLAGS(qsp)			(qsp)->qs_flags
#define SET_QS_FLAGS(qsp,f)		(qsp)->qs_flags=f
#define SET_QS_FLAG_BITS(qsp,bits)	(qsp)->qs_flags |= (bits)
#define CLEAR_QS_FLAG_BITS(qsp,bits)	(qsp)->qs_flags &= ~(bits)

#define CURR_QRY(qsp)			((Query *)TOP_OF_STACK((qsp)->qs_query_stack))
#define PREV_QRY(qsp)			((Query *)NODE_DATA(nth_elt((qsp)->qs_query_stack,1)))
#define FIRST_MENU(qsp)			((Menu *)BOTTOM_OF_STACK((qsp)->qs_menu_stack))
#define FIRST_QRY(qsp)			((Query *)BOTTOM_OF_STACK((qsp)->qs_query_stack))
#define QS_RETSTR_IDX(qsp)		(qsp)->qs_which_retstr
#define SET_QS_RETSTR_IDX(qsp,n)	(qsp)->qs_which_retstr=n
#define QS_RETSTR_AT_IDX(qsp,idx)	(qsp)->qs_retstr[idx]
#define SET_QS_RETSTR_AT_IDX(qsp,idx,sbp)	(qsp)->qs_retstr[idx] = sbp
//#define QS_WHICH_ESTR(qsp)		(qsp)->qs_which_estr
//#define SET_QS_WHICH_ESTR(qsp,idx)	(qsp)->qs_which_estr= idx

#define SET_QS_EDEPTH(qsp,d)		(qsp)->qs_edepth=d
#define QS_EDEPTH(qsp)			(qsp)->qs_edepth
//#define QS_ESTRING(qsp)			((qsp)->qs_estr)[QS_WHICH_ESTR(qsp)]
//#define SET_QS_ESTR_ARRAY(qsp,str_p)	(qsp)->qs_estr = str_p
//#define QS_ESTRING(qsp)			(qsp)->qs_expr_string
//#define SET_QS_ESTRING(qsp,s)		(qsp)->qs_expr_string = s
#define _QS_CURR_STRING(qsp)		(qsp)->_qs_curr_string
#define SET_QS_CURR_STRING(qsp,s)	(qsp)->_qs_curr_string=s
//#define CURR_STRING			QS_CURR_STRING(THIS_QSP)
//#define SET_CURR_STRING(s)		SET_QS_CURR_STRING(THIS_QSP , s)

#define QS_MAX_WARNINGS(qsp)		(qsp)->qs_max_warnings
#define QS_N_WARNINGS(qsp)		(qsp)->qs_n_warnings
#define SET_QS_MAX_WARNINGS(qsp,n)	(qsp)->qs_max_warnings=n
#define SET_QS_N_WARNINGS(qsp,n)	(qsp)->qs_n_warnings = n
#define INC_QS_N_WARNINGS(qsp)	SET_QS_N_WARNINGS(qsp,1+QS_N_WARNINGS(qsp))

#define QS_CALLBACK_LIST(qsp)		(qsp)->qs_callback_lp
#define SET_QS_CALLBACK_LIST(qsp,lp)	(qsp)->qs_callback_lp = lp
#define QS_EVENT_LIST(qsp)		(qsp)->qs_event_lp
#define SET_QS_EVENT_LIST(qsp,lp)	(qsp)->qs_event_lp = lp

#define QS_DOBJ_ASCII_INFO(qsp)			(qsp)->qs_dai_p
#define SET_QS_DOBJ_ASCII_INFO(qsp,dai_p)	(qsp)->qs_dai_p = dai_p

#define INSURE_QS_DOBJ_ASCII_INFO(qsp)		\
						\
if( QS_DOBJ_ASCII_INFO(qsp) == NULL ){		\
	SET_QS_DOBJ_ASCII_INFO(qsp,getbuf(sizeof(struct dobj_ascii_info)));	\
}

#ifdef BUILD_FOR_IOS
#define QS_QUEUE(qsp)			(qsp)->qs_queue
#define SET_QS_QUEUE(qsp,q)		(qsp)->qs_queue = q
#endif /* BUILD_FOR_IOS */

// Are these for the scalar parser, the vector parser, or both?

//#define TOP_NODE		((Query_Stack *)THIS_QSP)->qs_top_enp
#define LAST_NODE		THIS_QSP->qs_last_enp
#define END_SEEN		THIS_QSP->qs_end_seen
#define YY_CP			THIS_QSP->qs_yy_cp
#define SET_YY_CP(s)		THIS_QSP->qs_yy_cp = s
#define EXPR_LEVEL		THIS_QSP->qs_expr_level
#define SET_EXPR_LEVEL(l)	THIS_QSP->qs_expr_level = l
#define LASTLINENO		THIS_QSP->qs_last_line_num
#define SET_LASTLINENO(n)	THIS_QSP->qs_last_line_num =  n
#define PARSER_LINENO		THIS_QSP->qs_parser_line_num
#define SET_PARSER_LINENO(n)	THIS_QSP->qs_parser_line_num = n
#define YY_LAST_LINE 		THIS_QSP->qs_yy_last_line
#define YY_INPUT_LINE 		THIS_QSP->qs_yy_input_line
#define SEMI_SEEN 		THIS_QSP->qs_semi_seen
#define SET_SEMI_SEEN(v) 	THIS_QSP->qs_semi_seen = v
//#define VEXP_STR		((THIS_QSP->qs_estr)[THIS_QSP->qs_which_estr])
//#define VEXP_STR		QS_EXPR_STRING(THIS_QSP)
#define FINAL			THIS_QSP->qs_final
#define CURR_INFILE		THIS_QSP->qs_curr_infile
#define SUBRT_CTX_STACK		THIS_QSP->qs_subrt_ctx_stack

#define SET_QS_YY_INPUT_LINE(qsp,l)	(qsp)->qs_yy_input_line=l
#define SET_QS_YY_LAST_LINE(qsp,l)	(qsp)->qs_yy_last_line=l
//#define QS_EXPR_STRING(qsp)		(qsp)->qs_expr_string
#define SET_QS_EXPR_STRING(qsp,l)	(qsp)->_qs_expr_string=l


#define NEW_QUERY_STACK		((Query_Stack *)getbuf(sizeof(Query_Stack)))

#define PREV_LOG_MSG(qsp)		(qsp)->qs_prev_log_msg
#define SET_PREV_LOG_MSG(qsp,s)		(qsp)->qs_prev_log_msg = s

#define LOG_MSG_COUNT(qsp)		(qsp)->qs_log_msg_count
#define SET_LOG_MSG_COUNT(qsp,c)	(qsp)->qs_log_msg_count = c
#define INCREMENT_LOG_MSG_COUNT(qsp)	SET_LOG_MSG_COUNT(qsp,1+LOG_MSG_COUNT(qsp))

//#define CURRENT_FILENAME		QRY_FILENAME(CURR_QRY(THIS_QSP))

/*#define CURRENT_INPUT_FILENAME	"(CURRENT_INPUT_FILENAME not implemented)" */
//#define CURRENT_INPUT_FILENAME	QRY_FILENAME(CURR_QRY(THIS_QSP))


// some prototypes

extern void rls_mouthful(Mouthful *mfp);


#endif /* !  _QUERY_STACK_H_ */
