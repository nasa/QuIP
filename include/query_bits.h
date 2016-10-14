
#ifndef _QUERY_BITS_H_

#ifdef __cplusplus
extern "C" {
#endif

#define _QUERY_BITS_H_

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

/* This was 256,
 * things crapped out when I had an input line with 258 chars, there
 * is a bug which should be fixed, but for now it is expedient just
 * to spend a little memory.
 */

#ifndef LLEN
//#define LLEN	1024
#define LLEN	512
#endif /* ! LLEN */

/* query struct flags values */
#define Q_SOCKET		1
#define Q_INTERACTIVE		2
#define Q_MPASSED		4
#define Q_FIRST_WORD		8
#define Q_LOOKAHEAD		16	// 0x10
#define Q_LINEDONE		32	// 0x20
#define Q_BUFFERED_TEXT		64	// 0x40
#define Q_PIPE			128	// 0x80
#define Q_IN_MACRO		256	// 0x100
#define Q_HAS_SOMETHING		512	// 0x200
#define Q_EXPANDING_MACROS	1024	// 0x400
#define Q_STRIPPING_QUOTES	2048	// 0x800
#define Q_SAVING		4096	// 0x1000
#define Q_FILE_INPUT		8192	// 0x2000
#define Q_MACRO_INPUT		0x4000

#define Q_PRIMARY_INPUT		(Q_SOCKET|Q_INTERACTIVE|Q_MACRO_INPUT|Q_FILE_INPUT)
#define Q_NON_INPUT_MASK	(~Q_PRIMARY_INPUT)

#define IS_PRIMARY_INPUT( qp )		(QRY_FLAGS(qp) & Q_PRIMARY_INPUT)

#define IS_INTERACTIVE( qp )		(QRY_FLAGS(qp) & Q_INTERACTIVE)
#define NOT_PASSED(qp)			((QRY_FLAGS(qp) & Q_MPASSED)==0)
#define IS_PIPE( qp )			(QRY_FLAGS(qp) & Q_PIPE)

#define IS_DUPING			( QRY_DUPFILE(CURR_QRY(THIS_QSP)) != NULL )
#define FIRST_WORD_ON_LINE		( QRY_FLAGS(CURR_QRY(THIS_QSP)) & Q_FIRST_WORD )

/* size of memory chunks for loop buffer */
#define LOOPSIZE	256

/* some codes used for printing.
 * These used to be in data_obj.h, and there was a global var that held the current value.
 * To have a per-thread value, it has to be defined here...
 */



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
#define QS_PROCESSING_CALLBACKS	0x4000
#define QS_SILENT		0x8000
#define QS_SILENCE_CHECKED	0x10000
#define QS_TIME_FMT_UTC		0x20000
#define QS_HAS_PREV_LOG_MSG	0x40000
#define QS_SUSPENDED		0x80000

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

#define VAR_DELIM		'$'
#define IS_LEGAL_VAR_CHAR(c)	(isalnum(c) || (c)=='_' )

#define ERROR_STR_LEN	LLEN+100

#define TMPVAR				THIS_QSP->qs_tmpvar

#define QUERY_PROMPT			QS_PROMPT(THIS_QSP)
#define QLEVEL				QS_LEVEL(THIS_QSP)
#define INIT_QSP			THIS_QSP=new_query_stream(NULL_QSP_ARG  "default_query_stream");

//#define BUILTINS_INITED			(QUERY_FLAGS & QS_BUILTINS_INITED)

/* libdata stuff */
#define ASCII_LEVEL			QS_ASCII_LEVEL(THIS_QSP)
#define THE_FMT_CODE			QS_FMT_CODE(THIS_QSP)


//
typedef struct foreach_loop {
	const char *	f_varname;
	List *		f_word_lp;
	Node *		f_word_np;
} Foreach_Loop;


/* Foreach_Loop */
#define FL_VARNAME(flp)		flp->f_varname
#define FL_LIST(flp)		flp->f_word_lp
#define FL_NODE(flp)		flp->f_word_np
#define FL_WORD(flp)		((const char *)NODE_DATA(FL_NODE(flp)))

#define SET_FL_VARNAME(flp,s)	flp->f_varname = s
#define SET_FL_LIST(flp,lp)	flp->f_word_lp = lp
#define SET_FL_NODE(flp,np)	flp->f_word_np = np
#define NEW_FOREACH_LOOP	((Foreach_Loop*) getbuf( sizeof(*frp) ))


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


#endif /* HAVE_POPEN */


#define PROMPT_FORMAT	"Enter %s: "

#ifdef QUIP_DEBUG
#include "debug.h"
extern debug_flag_t qldebug;
extern debug_flag_t lah_debug;
#endif /* QUIP_DEBUG */


#ifdef MOVED
/* global variable(s) */
#ifdef THREAD_SAFE_QUERY
extern int n_active_threads;	// Number of qsp's
#endif /* THREAD_SAFE_QUERY */
#endif // MOVED - to list.h

#ifdef __cplusplus
}
#endif

#endif /* _QUERY_BITS_H_ */

