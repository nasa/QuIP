#ifndef _QUERY_STACK_H_
#define _QUERY_STACK_H_

#include "quip_config.h"

#include "stack.h"
#include "llen.h"
#include "query_bits.h"
#include "variable.h"
#include "quip_menu.h"
#include "ascii_fmts.h"
#include "string_ref.h"
#include "typed_scalar.h"

#ifdef BUILD_FOR_IOS
#include <dispatch/dispatch.h>
#endif /* BUILD_FOR_IOS */


#define MAXEDEPTH	20	// for variables inside expressions
#define MAX_E_STRINGS	64

// BUG should make this an opaque struct, and move this
// to a local header in libsrc/interpreter...

typedef struct scalar_parser_data {
	const char *		spd_yystrstk[MAXEDEPTH];	// stack of input
	const char *		spd_original_string;
	int			spd_edepth;	// init to -1
	int			spd_which_str;	// init to 0
	int			spd_in_pexpr;	// init to 0
	int			spd_estrings_inited;	// init to 0
	String_Buf *		spd_expr_string[MAX_E_STRINGS];
	Typed_Scalar		spd_string_scalar[MAX_E_STRINGS];
	Scalar_Expr_Node *	spd_final_expr_node_p;
	List *			spd_free_enp_lp;
} Scalar_Parser_Data;

#define SPD_YYSTRSTK(spd_p)			(spd_p)->spd_yystrstk
#define SPD_ORIGINAL_STRING(spd_p)		(spd_p)->spd_original_string
#define SPD_EDEPTH(spd_p)			(spd_p)->spd_edepth
#define SPD_WHICH_STR(spd_p)			(spd_p)->spd_which_str
#define SPD_IN_PEXPR(spd_p)			(spd_p)->spd_in_pexpr
#define SPD_ESTRINGS_INITED(spd_p)		(spd_p)->spd_estrings_inited
#define SPD_EXPR_STRING(spd_p)			(spd_p)->spd_expr_string
#define SPD_STRING_SCALAR(spd_p)		(spd_p)->spd_string_scalar
#define SPD_FINAL_EXPR_NODE_P(spd_p)		(spd_p)->spd_final_expr_node_p
#define SPD_FREE_EXPR_NODE_LIST(spd_p)		(spd_p)->spd_free_enp_lp

#define SET_SPD_YYSTRSTK(spd_p,val)		(spd_p)->spd_yystrstk = val
#define SET_SPD_ORIGINAL_STRING(spd_p,val)	(spd_p)->spd_original_string = val
#define SET_SPD_EDEPTH(spd_p,val)		(spd_p)->spd_edepth = val
#define SET_SPD_WHICH_STR(spd_p,val)		(spd_p)->spd_which_str = val
#define SET_SPD_IN_PEXPR(spd_p,val)		(spd_p)->spd_in_pexpr = val
#define SET_SPD_ESTRINGS_INITED(spd_p,val)	(spd_p)->spd_estrings_inited = val
#define SET_SPD_FREE_EXPR_NODE_LIST(spd_p,val)	(spd_p)->spd_free_enp_lp = val


// This struct is used to push text frags around...

struct mouthful {
	const char *text;
	const char *filename;
} ;

#define DBL_QUOTE	'"'
#define SGL_QUOTE	'\''

#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif

// when is this ever undefined???  BUG
#ifndef PATH_MAX
#ifndef _POSIX_PATH_MAX
#define PATH_MAX	256
#else
#define PATH_MAX	_POSIX_PATH_MAX
#endif /* POSIX_PATH_MAX */
#endif /* ! PATH_MAX */

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
	Item		qs_item;	// name of this query_stack
	int		qs_serial;

	// interpreter vars
	int		qs_flags;	// where are the flag bits declared???
//	List *		qs_retstr_lp;	// how does this relate to qs_result?
	String_Buf *	qs_varbuf[MAX_VAR_BUFS];	// why fixed size array?
	String_Buf *	qs_result;	// used in var expansion
	int		qs_which_var_buf;
	Stack *		qs_query_stack;
	int		_qs_level;	// used for lookahead
	int		_qs_stop_level;	// used for lookahead
	Stack *		qs_stop_level_stack;
	int		qs_ascii_level;
	int		qs_chew_level;
	String_Buf *	qs_av_sbp;			// what is this?
	String_Buf *	qs_scratch;
	List *		qs_chew_list;		// text for deferred execution
#define CHEW_LIST	QS_CHEW_LIST(THIS_QSP)
	List *		qs_callback_lp;		// per-cmd callback functions
	List *		qs_event_lp;
	Variable *	qs_tmpvar;		// what is this for?

	// stuff for word scanning
	int		qs_word_scan_flags;
	int		qs_start_quote;		/* holds the value
						 * of the starting quote char,
						 * if in a quote,
						 * otherwise 0
						 */
	int		qs_n_quotations;	/* to handle things like:
						 * "a b c"X"x y z"
						 * where we don't want to
						 * strip the outer pair...
						 */
	String_Buf *	qs_ret_sbp;
	char *		qs_ret_str;			// avoids passing a lot of stuff...
	char *		qs_ret_ptr;

	// menu stuff
	Menu *		qs_builtin_menu;
	Menu *		qs_help_menu;
	String_Buf *	qs_cmd_prompt_sbp;
	String_Buf *	qs_qry_prompt_sbp;	// question prompt
	Stack *		qs_menu_stack;

	String_Buf *	qs_output_filename;

	char		qs_pathname[PATH_MAX];

	Stack *		qs_int_var_fmt_stack;

	Integer_Output_Fmt *	qs_int_var_fmt_p;	// for variables
	char 			qs_flt_var_fmt[16];

	// for formatted ascii input to data objects
	struct dobj_ascii_info *	qs_dai_p;

	int		qs_max_warnings;
	int		qs_n_warnings;
	int		qs_num_warnings;
	FILE *		qs_error_fp;
	FILE *		qs_msg_fp;
	// BUG - we should phase these out in favor of string_buf's...
	char		qs_error_string[LLEN];
	char		qs_msg_str[LLEN];
	List *		qs_expected_warning_lp;

#ifdef THREAD_SAFE_QUERY
#ifdef HAVE_PTHREADS
	pthread_t	qs_thr;
#endif /* HAVE_PTHREADS */
#endif /* THREAD_SAFE_QUERY */

	/* These are not part of vector_parser_data because we can have a stack of parsers... */
	List *			qs_vector_parser_data_stack;
	List *			qs_vector_parser_data_freelist;
	char			qs_vector_parser_error_string[LLEN];
	Vector_Parser_Data *	qs_vector_parser_data;		// current parser

	// if we allow reentrant parsing, then we have to have more of these...
#define MAX_SCALAR_PARSER_CALL_DEPTH	2
	Scalar_Parser_Data *	qs_scalar_parser_data_tbl[MAX_SCALAR_PARSER_CALL_DEPTH];
	int			qs_scalar_parser_call_depth;

	int		qs_max_vectorizable;

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
	
#ifdef THREAD_SAFE_QUERY
	// Because item context stacks are now per-thread,
	// we might like to import contexts from the invoking thread...
	int		qs_parent_serial;
#endif // THREAD_SAFE_QUERY
};

#define MAX_VECTORIZABLE		QS_MAX_VECTORIZABLE(THIS_QSP)
#define SET_MAX_VECTORIZABLE(v)		QS_MAX_VECTORIZABLE(THIS_QSP,v)

#define QS_MAX_VECTORIZABLE(qsp)		(qsp)->qs_max_vectorizable
#define SET_QS_MAX_VECTORIZABLE(qsp,v)		(qsp)->qs_max_vectorizable = v
#define QS_PICKING_ITEM_ITP(qsp)		((qsp)->qs_picking_item_itp)
#define SET_QS_PICKING_ITEM_ITP(qsp,itp)	(qsp->qs_picking_item_itp) = itp

// this indexing of the list is probably backwards!?
#define QS_QRY_STACK(qsp)		(qsp)->qs_query_stack
#define SET_QS_QRY_STACK(qsp,stkp)	(qsp)->qs_query_stack = stkp

#define QS_OUTPUT_FILENAME(qsp)		((qsp)->qs_output_filename==NULL?NULL:sb_buffer((qsp)->qs_output_filename))

#define SET_QS_OUTPUT_FILENAME(qsp,str)					\
									\
	{								\
	if( (qsp)->qs_output_filename==NULL )				\
		(qsp)->qs_output_filename = create_stringbuf(str);	\
	else copy_string((qsp)->qs_output_filename,str);		\
	}

#define QS_PATHNAME(qsp)		(qsp)->qs_pathname

/* query_stack flags - flag bits */
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
#define QS_HISTORY		4096	// 0x1000
#define QS_CHEWING		8192	// 0x2000
#define QS_PROCESSING_CALLBACKS	0x004000
#define QS_SILENT		0x008000
#define QS_SILENCE_CHECKED	0x010000
#define QS_TIME_FMT_UTC		0x020000
#define QS_HAS_PREV_LOG_MSG	0x040000
#define QS_SUSPENDED		0x080000
#define QS_APPENDING		0x100000

#define HAS_PREV_LOG_MSG(qsp)	(QS_FLAGS(qsp) & QS_HAS_PREV_LOG_MSG)

#define DISPLAYING_UTC(qsp)	(QS_FLAGS(qsp) & QS_TIME_FMT_UTC)

#define IS_SILENT(qsp)		(QS_FLAGS(qsp) & QS_SILENT)
#define SILENCE_CHECKED(qsp)	(QS_FLAGS(qsp) & QS_SILENCE_CHECKED)
#define IS_COMPLETING(qsp)	(QS_FLAGS(qsp) & QS_COMPLETING)

#define IS_CHEWING(qsp)		(QS_FLAGS(qsp) & QS_CHEWING)
#define IS_HALTING(qsp)		(QS_FLAGS(qsp) & QS_HALTING)
#define IS_PROCESSING_CALLBACKS(qsp)		(QS_FLAGS(qsp) & QS_PROCESSING_CALLBACKS)
#define IS_TRACKING_HISTORY(qsp)	(QS_FLAGS(qsp) & QS_HISTORY)
#define IS_EXITING(qsp)		(QS_FLAGS(qsp) & QS_EXITING)
#define IS_STILL_TRYING(qsp)	(QS_FLAGS(qsp) & QS_STILL_TRYING)

#define HAD_INTERRUPT(qsp)	(QS_FLAGS(qsp) & QS_HAD_INTR)

//#define NEED_TO_SAVE(qp) ((qp) != (&THIS_QSP->qs_query[0]) && ((qp)-1)->q_saving )
#define NEED_TO_SAVE(qp) ( (qp) != FIRST_QRY(THIS_QSP) && QRY_IS_SAVING(UNDER_QRY(qp)) )

#define APPEND_FLAG		((QS_FLAGS(THIS_QSP)&QS_APPENDING)?1:0)
#define SET_APPEND_FLAG(v)	{if(v) SET_QS_FLAG_BITS(THIS_QSP,QS_APPENDING); \
				else CLEAR_QS_FLAG_BITS(THIS_QSP,QS_APPENDING);}

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

#define QS_HAS_SOMETHING(qsp)		(QLEVEL>=Q_STOP_LEVEL && QRY_HAS_TEXT(CURR_QRY(qsp)))
#define QS_LINES_READ(qsp)		QRY_LINES_READ(CURR_QRY(qsp))

#define QS_MENU_STACK(qsp)		(qsp)->qs_menu_stack
#define SET_QS_MENU_STACK(qsp,stkp)	(qsp)->qs_menu_stack = stkp

#define QS_INT_VAR_FMT_STACK(qsp)		(qsp)->qs_int_var_fmt_stack
#define SET_QS_INT_VAR_FMT_STACK(qsp,stkp)	(qsp)->qs_int_var_fmt_stack = stkp
#define QS_EXPECTED_WARNING_LIST(qsp)	(qsp)->qs_expected_warning_lp
#define SET_QS_EXPECTED_WARNING_LIST(qsp,lp)	(qsp)->qs_expected_warning_lp = lp
#define QS_INT_VAR_FMT_P(qsp)		(qsp)->qs_int_var_fmt_p
#define SET_QS_INT_VAR_FMT_P(qsp,iof_p)	(qsp)->qs_int_var_fmt_p = iof_p
#define QS_FLT_VAR_FMT(qsp)		(qsp)->qs_flt_var_fmt
#define SET_QS_FLT_VAR_FMT(qsp,s)	strcpy((qsp)->qs_flt_var_fmt,s)
#define QS_GFORMAT(qsp)			(qsp)->qs_gfmt_str
#define QS_XFORMAT(qsp)			(qsp)->qs_xfmt_str
#define QS_DFORMAT(qsp)			(qsp)->qs_dfmt_str
#define QS_OFORMAT(qsp)			(qsp)->qs_ofmt_str
#define QS_PFORMAT(qsp)			(qsp)->qs_pfmt_str
#define SET_QS_GFORMAT(qsp,s)		strcpy((qsp)->qs_gfmt_str,s)
#define SET_QS_XFORMAT(qsp,s)		strcpy((qsp)->qs_xfmt_str,s)
#define SET_QS_DFORMAT(qsp,s)		strcpy((qsp)->qs_dfmt_str,s)
#define SET_QS_OFORMAT(qsp,s)		strcpy((qsp)->qs_ofmt_str,s)
#define SET_QS_PFORMAT(qsp,s)		strcpy((qsp)->qs_pfmt_str,s)
#define QS_AV_STRINGBUF(qsp)		(qsp)->qs_av_sbp
#define SET_QS_AV_STRINGBUF(qsp,sbp)	(qsp)->qs_av_sbp = sbp

#define QS_VECTOR_PARSER_DATA(qsp)		(qsp)->qs_vector_parser_data
#define QS_VECTOR_PARSER_DATA_STACK(qsp)	(qsp)->qs_vector_parser_data_stack
#define QS_VECTOR_PARSER_DATA_FREELIST(qsp)	(qsp)->qs_vector_parser_data_freelist
#define QS_VECTOR_PARSER_ERROR_STRING(qsp,d)	(qsp)->qs_vector_parser_error_string
#define SET_QS_VECTOR_PARSER_DATA(qsp,d)	(qsp)->qs_vector_parser_data = d
#define SET_QS_VECTOR_PARSER_DATA_STACK(qsp,v)	(qsp)->qs_vector_parser_data_stack = v
#define SET_QS_VECTOR_PARSER_DATA_FREELIST(qsp,v)	(qsp)->qs_vector_parser_data_freelist = v

#ifdef FOOBAR
#define THIS_VPD		(QS_VECTOR_PARSER_DATA(THIS_QSP))
#endif // FOOBAR



#define QS_SCALAR_PARSER_DATA_AT_IDX(qsp,idx)		(qsp)->qs_scalar_parser_data_tbl[idx]
#define SET_QS_SCALAR_PARSER_DATA_AT_IDX(qsp,idx,d)	(qsp)->qs_scalar_parser_data_tbl[idx] = d
#define QS_SCALAR_PARSER_CALL_DEPTH(qsp)		(qsp)->qs_scalar_parser_call_depth
#define SET_QS_SCALAR_PARSER_CALL_DEPTH(qsp,v)		(qsp)->qs_scalar_parser_call_depth = v

#define QS_CURR_SCALAR_PARSER_DATA(qsp)	QS_SCALAR_PARSER_DATA_AT_IDX(qsp,QS_SCALAR_PARSER_CALL_DEPTH(qsp))
#define SET_QS_CURR_SCALAR_PARSER_DATA(qsp,v)	QS_SCALAR_PARSER_DATA_AT_IDX(qsp,QS_SCALAR_PARSER_CALL_DEPTH(qsp)) = v

#define QS_TMPVAR(qsp)			(qsp)->qs_tmpvar
#define QS_BUILTIN_MENU(qsp)		(qsp)->qs_builtin_menu
#define SET_QS_BUILTIN_MENU(qsp,mp)	(qsp)->qs_builtin_menu=mp
#define QS_HELP_MENU(qsp)		(qsp)->qs_help_menu
#define SET_QS_HELP_MENU(qsp,mp)	(qsp)->qs_help_menu=mp
#define SET_QS_TMPVAR(qsp,vp)		(qsp)->qs_tmpvar=vp
#define QS_CHEW_LIST(qsp)		(qsp)->qs_chew_list
#define SET_QS_CHEW_LIST(qsp,lp)	(qsp)->qs_chew_list =  lp

#define QS_QRY_PROMPT_SB(qsp)		(qsp)->qs_qry_prompt_sbp
#define SET_QS_QRY_PROMPT_SB(qsp,sbp)	(qsp)->qs_qry_prompt_sbp = sbp
#define QS_QRY_PROMPT_STR(qsp)		sb_buffer((qsp)->qs_qry_prompt_sbp)

#define QS_CMD_PROMPT_SB(qsp)		(qsp)->qs_cmd_prompt_sbp
#define SET_QS_CMD_PROMPT_SB(qsp,sbp)	(qsp)->qs_cmd_prompt_sbp = sbp
#define QS_CMD_PROMPT_STR(qsp)		sb_buffer((qsp)->qs_cmd_prompt_sbp)
#define CLEAR_QS_CMD_PROMPT(qsp)	QS_CMD_PROMPT_STR(qsp)[0] = 0

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
#define QS_LEVEL(qsp)			(qsp)->_qs_level
#define SET_QS_LEVEL(qsp,l)		(qsp)->_qs_level = l
#define QS_STOP_LEVEL(qsp)		(qsp)->_qs_stop_level
#define SET_QS_STOP_LEVEL(qsp,l)	(qsp)->_qs_stop_level = l
#define QS_STOP_LEVEL_STACK(qsp)	(qsp)->qs_stop_level_stack
#define SET_QS_STOP_LEVEL_STACK(qsp,v)	(qsp)->qs_stop_level_stack = v
#define QS_CHEW_LEVEL(qsp)		(qsp)->qs_chew_level
#define SET_QS_CHEW_LEVEL(qsp,l)	(qsp)->qs_chew_level = l
#define QS_ASCII_LEVEL(qsp)		(qsp)->qs_ascii_level
//#define QS_FMT_CODE(qsp)		(qsp)->qs_fmt_code
//#define SET_QS_FMT_CODE(qsp,c)		(qsp)->qs_fmt_code=c
#define QS_LOOKAHEAD_LEVEL(qsp)		(qsp)->qs_lookahead_level
#define SET_QS_LOOKAHEAD_LEVEL(qsp,l)	(qsp)->qs_lookahead_level = l

#ifdef THREAD_SAFE_QUERY
#define QS_PARENT_SERIAL(qsp)		(qsp)->qs_parent_serial
#define SET_QS_PARENT_SERIAL(qsp,n)	(qsp)->qs_parent_serial=n
#endif //  THREAD_SAFE_QUERY

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

#ifdef FOOBAR
//#define QS_RETSTR_IDX(qsp)		(qsp)->qs_which_retstr
//#define SET_QS_RETSTR_IDX(qsp,n)	(qsp)->qs_which_retstr=n
//#define QS_RETSTR_AT_IDX(qsp,idx)	(qsp)->qs_retstr[idx]
//#define SET_QS_RETSTR_AT_IDX(qsp,idx,sbp)	(qsp)->qs_retstr[idx] = sbp
#endif // FOOBAR

#define QS_RET_STR(qsp)			(qsp)->qs_ret_str
#define QS_RET_STRBUF(qsp)		(qsp)->qs_ret_sbp
#define QS_RET_PTR(qsp)			(qsp)->qs_ret_ptr
#define SET_QS_RET_STR(qsp,v)		(qsp)->qs_ret_str = v
#define SET_QS_RET_PTR(qsp,v)		(qsp)->qs_ret_ptr = v
#define SET_QS_RET_STRBUF(qsp,v)	(qsp)->qs_ret_sbp = v

#define QS_LINE_PTR(qsp)		QRY_LINE_PTR(CURR_QRY(qsp))
#define SET_QS_LINE_PTR(qsp,v)		QRY_LINE_PTR(CURR_QRY(qsp)) = v

#define ADD_TO_RESULT(c)		*(QS_RET_PTR(THIS_QSP)++) = (c);

//#define QS_WHICH_ESTR(qsp)		(qsp)->qs_which_estr
//#define SET_QS_WHICH_ESTR(qsp,idx)	(qsp)->qs_which_estr= idx

#define SET_QS_EDEPTH(qsp,d)		(qsp)->qs_edepth=d
#define QS_EDEPTH(qsp)			(qsp)->qs_edepth

#define QS_WORD_SCAN_FLAGS(qsp)		(qsp)->qs_word_scan_flags
#define QS_START_QUOTE(qsp)		(qsp)->qs_start_quote
#define QS_N_QUOTATIONS(qsp)		(qsp)->qs_n_quotations

#define QS_MAX_WARNINGS(qsp)		(qsp)->qs_max_warnings
#define QS_N_WARNINGS(qsp)		(qsp)->qs_n_warnings
#define SET_QS_MAX_WARNINGS(qsp,n)	(qsp)->qs_max_warnings=n
#define SET_QS_N_WARNINGS(qsp,n)	(qsp)->qs_n_warnings = n
#define INC_QS_N_WARNINGS(qsp)	SET_QS_N_WARNINGS(qsp,QS_N_WARNINGS(qsp)+1)
#define DEC_QS_N_WARNINGS(qsp)	SET_QS_N_WARNINGS(qsp,QS_N_WARNINGS(qsp)-1)

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

#ifdef FOOBAR

#define LAST_NODE		VPD_LAST_ENP( THIS_VPD )
#define END_SEEN		VPD_END_SEEN( THIS_VPD )
#define YY_CP			VPD_YY_CP( THIS_VPD )
#define EXPR_LEVEL		VPD_EXPR_LEVEL( THIS_VPD )
#define LAST_LINE_NUM		VPD_LAST_LINE_NUM( THIS_VPD )
#define PARSER_LINE_NUM		VPD_PARSER_LINE_NUM( THIS_VPD )
#define YY_LAST_LINE		VPD_YY_LAST_LINE( THIS_VPD )
#define YY_INPUT_LINE		VPD_YY_INPUT_LINE( THIS_VPD )
#define YY_WORD_BUF		VPD_YY_WORD_BUF( THIS_VPD )
#define SEMI_SEEN		VPD_SEMI_SEEN( THIS_VPD )
#define FINAL			VPD_FINAL( THIS_VPD )
#define SUBRT_CTX_STACK		VPD_SUBRT_CTX_STACK( THIS_VPD )

#define SET_LAST_NODE(v)		SET_VPD_LAST_ENP( THIS_VPD, v )
#define SET_END_SEEN(v)			SET_VPD_END_SEEN( THIS_VPD, v )
#define SET_YY_CP(v)			SET_VPD_YY_CP( THIS_VPD, v )
#define SET_EXPR_LEVEL(v)		SET_VPD_EXPR_LEVEL( THIS_VPD, v )
#define SET_LAST_LINE_NUM(v)		SET_VPD_LAST_LINE_NUM( THIS_VPD, v )
#define SET_PARSER_LINE_NUM(v)		SET_VPD_PARSER_LINE_NUM( THIS_VPD, v )
#define SET_YY_LAST_LINE(v)		SET_VPD_YY_LAST_LINE( THIS_VPD, v )
#define SET_YY_INPUT_LINE(v)		SET_VPD_YY_INPUT_LINE( THIS_VPD, v )
#define SET_SEMI_SEEN(v)		SET_VPD_SEMI_SEEN( THIS_VPD, v )
#define SET_FINAL(v)			SET_VPD_FINAL( THIS_VPD, v )
#define SET_SUBRT_CTX_STACK(v)		SET_VPD_SUBRT_CTX_STACK( THIS_VPD, v )

//#define CURR_INFILE			VPD_CURR_INFILE( THIS_VPD )
//#define SET_CURR_INFILE(v)		SET_VPD_CURR_INFILE( THIS_VPD, v )
#endif // FOOBAR

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

#define set_stop_level(l) SET_QS_STOP_LEVEL(THIS_QSP,l)
extern void _push_stop_level(QSP_ARG_DECL  int l);
extern int _pop_stop_level(SINGLE_QSP_ARG_DECL);
#define push_stop_level(l) _push_stop_level(QSP_ARG  l)
#define pop_stop_level() _pop_stop_level(SINGLE_QSP_ARG)


extern void _init_scalar_parser_data_at_idx(QSP_ARG_DECL   int idx);
#define init_scalar_parser_data_at_idx(idx) _init_scalar_parser_data_at_idx(QSP_ARG   idx)

#endif /* !  _QUERY_STACK_H_ */
