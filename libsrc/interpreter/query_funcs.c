#include "quip_config.h"

/*
General approach:

*	next_query_word
		next_raw_input_word
			next_word_from_level
				(qline) (only if necessary)
					nextline
				sync line numbers
				next_word_from_input_line
					var_expand
						get_varval
					handles quoting
					perform variable expansion
					counts line endings
		perform macro expansion, loop if necessary

*	lookahead
*	lookahead_til
		skip spaces and reads lines as necessary,
		stops when a non-space character is found.
		used to determine when file is exhausted
		Can advance lines_read...
		nextline

When should we count lines?
We increment lines_read when we see a newline...  Because of lookahead,
This can get ahead of the current line.
When should we sync line numbers?

*/

//#define QUIP_DEBUG_LINENO

#define SYNC_LINENO									\
	{										\
	SET_QRY_LINENO(CURR_QRY(THIS_QSP), QRY_LINES_READ(CURR_QRY(THIS_QSP)) );	\
	DEBUG_LINENO(sync_lineno) }

#define INC_QRY_LINES_READ							\
										\
	SET_QRY_LINES_READ(CURR_QRY(THIS_QSP), QRY_LINES_READ(CURR_QRY(THIS_QSP)) + 1 );


#ifdef QUIP_DEBUG_LINENO

#define INCREMENT_LINES_READ(whence)						\
										\
	{									\
	INC_QRY_LINES_READ							\
fprintf(stderr,"increment_lines_read: %s\n",#whence);				\
	DEBUG_LINENO(increment_lines_read)					\
	}

#define DEBUG_LINENO(whence)					\
assert(THIS_QSP!=NULL);\
if( QLEVEL >=0 ){								\
assert(CURR_QRY(THIS_QSP)!=NULL);\
	fprintf(stderr,"%s:  Line %d (%d lines read)\n",	\
		#whence,QRY_LINENO(CURR_QRY(THIS_QSP)),QRY_LINES_READ(CURR_QRY(THIS_QSP)));	\
}

#else // ! QUIP_DEBUG_LINENO

#define INCREMENT_LINES_READ(whence)		INC_QRY_LINES_READ
#define DEBUG_LINENO(whence)

#endif // ! QUIP_DEBUG_LINENO

/* used to be query.c ... */

/**/
/**		input and output stuff		**/
/**/

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* abort(3) */
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	// isatty (at least on macOS)
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_SYS_FILIO_H
#include <sys/filio.h>
#endif


#include "quip_prot.h"
#include "query_stack.h"
#include "query_prot.h"
#include "query_private.h"
//#include "macros.h"
#include "debug.h"
//#include "items.h"
//#include "savestr.h"
#include "strbuf.h"
#include "variable.h"
#include "node.h"
#include "warn.h"
#include "nexpr.h"
#include "item_prot.h"
#include "ascii_fmts.h"
#include "getbuf.h"

#ifdef HAVE_HISTORY
#include "history.h"
#endif /* HAVE_HISTORY */

#ifdef USE_QS_QUEUE
// This is just for debugging!
#define QUEUE_CHECK(whence)						\
if( QS_QUEUE(THIS_QSP) != dispatch_get_current_queue() ){		\
sprintf(DEFAULT_ERROR_STRING,"%s:  dispatch queue mismatch!?",#whence);	\
advise(DEFAULT_ERROR_STRING);						\
abort();								\
}
#else /* !USE_QS_QUEUE */
#define QUEUE_CHECK(whence)
#endif /* !USE_QS_QUEUE */


/* The lookahead strategy is very complex...  It is desirable
 * to try to read ahead, in order to do things like popping
 * the input at EOF:  if we continue reading past EOF in the
 * normal case we will just get the next word from the next
 * file on the input stack.
 *
 * In order to keep the line numbering correct, we keep two counters,
 * QRY_LINENO and QRY_LINES_READ.  Lookahead may advance the latter,
 * and they aren't sync'd until the lookahead word is used.
 *
 * ../vectree/vectree.y uses another variable, LASTLINENO, which doesn't
 * seem to be used anywhere else, and is part of the query stack...  Probably
 * not very well thought through!
 */


/* global vars */

#ifdef QUIP_DEBUG
debug_flag_t qldebug=0;
debug_flag_t lah_debug=0;
#endif /* QUIP_DEBUG */

// moved default_qsp to error.c...
//Query_Stack *default_qsp=NULL;

// can be shared by all threads
static int n_cmd_args=0;
static int has_stdin=0;		// where should we set this?

#define word_scan_flags		QS_WORD_SCAN_FLAGS(THIS_QSP)
#define start_quote		QS_START_QUOTE(THIS_QSP)
#define n_quotations		QS_N_QUOTATIONS(THIS_QSP)

#define SET_WORD_SCAN_FLAG_BITS(bits)	QS_WORD_SCAN_FLAGS(THIS_QSP) |= bits
#define CLEAR_WORD_SCAN_FLAG_BITS(bits)	QS_WORD_SCAN_FLAGS(THIS_QSP) &= ~(bits)

#define DBL_QUOTE	'"'
#define SGL_QUOTE	'\''

#define MACRO_LOCATION_PREFIX	"Macro "

static inline void clear_query_text(Query *qp)
{
	copy_string(QRY_TEXT_BUF(qp),"");
}

// input_on_stdin()
// call this from a unix program.
// mobile apps have only their startup file

void input_on_stdin(void)
{
	has_stdin=1;
}


// for debugging

#define TEST_FLAG( f, s )					\
								\
	if( QRY_FLAGS(qp) & f ) {				\
		prt_msg_frag("\t");				\
		prt_msg(s);					\
	}

static const char * next_raw_input_word(QSP_ARG_DECL  const char* pline);

static void show_query_flags(QSP_ARG_DECL  Query *qp)
{
	sprintf(ERROR_STRING,"Query at 0x%lx has flags = 0x%x",
		(long)qp,QRY_FLAGS(qp));
	advise(ERROR_STRING);

	/* Now go through all the flags... */
	TEST_FLAG( Q_SOCKET, "socket" )
	TEST_FLAG( Q_INTERACTIVE, "interactive" )
	TEST_FLAG( Q_MPASSED, "macro args passed" )
	TEST_FLAG( Q_FIRST_WORD, "first word" )
	TEST_FLAG( Q_LOOKAHEAD, "lookahead" )
	TEST_FLAG( Q_LINEDONE, "line done" )
	TEST_FLAG( Q_BUFFERED_TEXT, "buffered text" )
	TEST_FLAG( Q_PIPE, "pipe" )
	TEST_FLAG( Q_IN_MACRO, "in macro" )
	TEST_FLAG( Q_HAS_SOMETHING, "has something" )
	TEST_FLAG( Q_EXPANDING_MACROS, "expanding macros" )
	TEST_FLAG( Q_STRIPPING_QUOTES, "stripping quotes" )
	TEST_FLAG( Q_SAVING, "saving text" )
	TEST_FLAG( Q_FILE_INPUT, "primary file input" )
	TEST_FLAG( Q_MACRO_INPUT, "primary macro input" )
}

/*
 * Toggle forcing of prompts.
 */

COMMAND_FUNC( tog_pmpt )
{
	if( QS_FLAGS(THIS_QSP) & QS_FORCE_PROMPT ){
		CLEAR_QS_FLAG_BITS(THIS_QSP,QS_FORCE_PROMPT);
		advise("suppressing prompts for non-tty input");
	} else {
		SET_QS_FLAG_BITS(THIS_QSP,QS_FORCE_PROMPT);
		advise("printing prompts for non-tty input");
	}
}

#define ESCAPED_SPACE	( *input_ptr == '\\' && isspace( *(input_ptr+1) ) )

static void skip_white_space(QSP_ARG_DECL  const char **input_pp)
{
	const char *input_ptr;

	input_ptr = *input_pp;

	/* skip over spaces */
	while( *input_ptr && ( isspace( *input_ptr ) || ESCAPED_SPACE ) ){
		// If file has both CR and NL, just count as one line
		if( *input_ptr == '\n' ){
			INCREMENT_LINES_READ(skip_white_space)
			if( *(input_ptr+1) == '\r' ) input_ptr++;
		} else if( *input_ptr == '\r' ){
			INCREMENT_LINES_READ(skip_white_space)
			if( *(input_ptr+1) == '\n' ) input_ptr++;
		}
		input_ptr++;
	}

	*input_pp = input_ptr;
}

static void discard_line_content(QSP_ARG_DECL  const char **input_pp)
{
	const char *input_ptr;
	input_ptr = *input_pp;
	while( *input_ptr && *input_ptr!='\n' && *input_ptr!='\r' ) input_ptr++;
	if( *input_ptr == '\n' || *input_ptr == '\r' ){
		INCREMENT_LINES_READ(discard_line_content)
		input_ptr++;
	}
	*input_pp = input_ptr;
}

// eatup_space_for_lookahead should never change the level!?

static void eatup_space_for_lookahead(SINGLE_QSP_ARG_DECL)
{
	const char *str;
	int lno;
	Query *qp=CURR_QRY(THIS_QSP);
#ifdef CAUTIOUS
	int orig_level;
#endif // CAUTIOUS

	if( !QRY_HAS_TEXT(qp) ) return;

	str=QRY_LINE_PTR(qp) ;
	assert( str != NULL );

	lno = QRY_LINES_READ(qp);
#ifdef CAUTIOUS
	orig_level = QLEVEL;
#endif // CAUTIOUS

	skip_white_space(QSP_ARG  &str);
	/* comments can be embedded in lines */

	/* the buffer may contain multiple lines */
	while( *str == '#' ){
		discard_line_content(QSP_ARG  &str);
		skip_white_space(QSP_ARG  &str);
	}

	if( *str == 0 )
		CLEAR_QRY_FLAG_BITS(qp,Q_HAS_SOMETHING);

	if( QRY_HAS_TEXT(qp) ){
		SET_QRY_LINE_PTR(qp,str);
	}

#ifdef CAUTIOUS
	assert(orig_level == QLEVEL);
#endif // CAUTIOUS

	if( lno != QRY_LINES_READ(qp) )
		SET_QRY_FLAG_BITS(qp,Q_LOOKAHEAD_ADVANCED_LINE);
} // eatup_space_for_lookahead


/*
 * Try to read the next word.  Called from next_query_word() and read_macro_body().
 *
 * Try to read the next word, after we have already got one
 * that is ready to interpret.  The reason for this is so that
 * empty levels will be popped before the next word is interpreted.
 * That way, the code following the do_cmd() call can tell when
 * a file has been exhausted.
 *
 * If there is no more text at this level, then a level will
 * be popped and qlevel will no longer be equal to former_level.
 * This gives external routines a way to detect when
 * the level has changed, i.e. when a file has been
 * completely read or a macro finished.
 *
 * This strategy isn't quite enough when the goal is to determine
 * whether or not the current input is an interactive terminal...
 *
 * there is a problem with the lookahead word
 * due to loops:  the next word after the "Close"
 * gets eaten up at the wrong level;
 * should close_loop do the pop_file???
 *
 * This is by fixed (?) by having close_loop call lookahead after
 * q_count reaches zero.
 *
 * we don't lookahead if we're in a loop, since the next
 * word might be "Close"
 *
 * we don't lookahead if we're in an if clause, since
 * if the if clause is at the end of a macro, the file
 * will get popped, releasing the if text!
 *
 * there seems to be a bug when the if clause is the last line
 * of a macro, the macro arg's get popped or otherwise screwed...
 *
 * Flash:  this seems to be fixed (?) by changing when we disable lookahead
 * (see do_if in builtin.c).
 *
 * There is also a problem with lookahead when the current
 * word is "PopFile"
 *
 * the main function of this lookahead is to detect
 * when redir files are closed or macros finished
 *
 *
 *	In order to do lookahead:
 *		NOT interactive or networked
 *		NOT reading stdin
 *		NOT going to go back and loop
 */

// Why QLEVEL and not 0???

void lookahead(SINGLE_QSP_ARG_DECL)
{
	lookahead_til(0);
}

// lookahead_til won't try to read at stop_level...

int _lookahead_til(QSP_ARG_DECL  int stop_level)
{
	int initial_level = QLEVEL;

#ifdef BUILD_FOR_OBJC
	if( QLEVEL < 0 ){
		return 0;	// nothing to interpret
	}
#endif /* BUILD_FOR_OBJC */

	CLEAR_QRY_FLAG_BITS(CURR_QRY(THIS_QSP),Q_LOOKAHEAD_ADVANCED_LINE);

	if( IS_HALTING(THIS_QSP) ){
		return 0;
	}

	// Not used???
	//QS_FORMER_LEVEL( THIS_QSP ) = QS_LEVEL( THIS_QSP );
	while(
		QLEVEL >= stop_level
	        && (QS_FLAGS(THIS_QSP) & QS_LOOKAHEAD_ENABLED)
		&& (!IS_INTERACTIVE( CURR_QRY(THIS_QSP) ) )
		// the socket flag was getting set in the query stack,
		// not the query item, so this test always succeeded???
		&& ( ( QRY_FLAGS( CURR_QRY(THIS_QSP) ) & Q_SOCKET ) == 0 )
		/* inhibit lookahead if we are saving (don't eatup spaces) */
		/* BUT the saving flag is set one level down... */

		/* Better to check for saving in eatup_space? */
		&& ( ! ( QLEVEL>0 && QRY_IS_SAVING( PREV_QRY(THIS_QSP) ) ) )

	){
		Query *qp;
		int _level;

		/* do look-ahead */

		qp= CURR_QRY(THIS_QSP);
		_level=QLEVEL;

		/* Eating space (and comments) here causes a problem,
		 * because the comment lines don't get saved (say when
		 * we are reading the body of a loop, and so line number
		 * reporting on subsequent iterations of the loop is messed
		 * up.  Two possible solutions:  1)  save the comment lines;
		 * or 2) save the individual lines, along with their numbers.
		 * Solution #1 would probably be simpler to implement, but slightly
		 * slower to run...
		 */

		if( QRY_HAS_TEXT(qp) ) {
DEBUG_LINENO(lookahead_til before eatup_space_for_lookahead #1)
			eatup_space_for_lookahead(SINGLE_QSP_ARG);
		}
		if( QRY_HAS_TEXT(CURR_QRY(THIS_QSP)) ){
			return 1;
		}
		while( (QLEVEL == _level) && (QRY_HAS_TEXT(qp) == 0) ){
			/* nextline() never pops more than one level */
DEBUG_LINENO(lookahead_til before nextline)
			nextline("");	// lookahead_til
DEBUG_LINENO(lookahead_til after nextline)
			// But because it can pop a level, we should not any
			// the space here...

			// halting bit can be set if we run out of input on a secondary thread
			// (primary thread will exit)
			if( IS_HALTING(THIS_QSP) ) {
				return 0;
			}

			if( QLEVEL == _level && QRY_HAS_TEXT(CURR_QRY(THIS_QSP)) ){
DEBUG_LINENO(lookahead_til before eatup_space_for_lookahead #2)
				eatup_space_for_lookahead(SINGLE_QSP_ARG);
			}
		}
	}

#ifdef BUILD_FOR_OBJC
	if( QLEVEL < 0 ){
		return 0;	// done with startup file?
	}
#endif /* BUILD_FOR_OBJC */
	
	assert(QLEVEL>=0);

	if( QLEVEL != initial_level ){
		assert(CURR_QRY(THIS_QSP) != NULL);
		SET_QRY_FLAG_BITS(CURR_QRY(THIS_QSP),Q_LOOKAHEAD_ADVANCED_LINE);
	}

	return 0;

} // end lookahead_til

// read_ith_macro_arg - read an argument after a macro name has been recognized.
// 

static const char *read_ith_macro_arg(QSP_ARG_DECL  Macro *mp, int i)
{
	Macro_Arg *map;
	const char *s;

	map = MACRO_ARG(mp,i);

	if( MA_ITP(map) != NULL ){
		Item *ip;
		ip = pick_item(MA_ITP(map), MA_PROMPT(map) );
		if( ip != NULL )
			s=ITEM_NAME(ip);
		else
			s="invalid_item_name"; /* BUG? put what the user actually entered? */
	} else {
		s=nameof(MA_PROMPT(map) );

		// This can happen if we are out of input
		// at the lowest level, as in a macro call
		// with missing args at the end of a macro
		// called from the base level (e.g. an event)
		if( s == NULL ) return NULL;
	}

	// if we don't save, we can't know whether or not to free?
	s = save_possibly_empty_str(s);

	return s;
}

#ifdef QUIP_DEBUG

static void debug_macro_arg(QSP_ARG_DECL  Macro *mp, const char **args, int i)
{
	if( strlen(args[i]) < LLEN-80 ){
		sprintf(ERROR_STRING,
			"debug_macro_arg:  macro arg %d saved at 0x%lx (%s)",
			i,(long)args[i],args[i]);
	} else {
		sprintf(ERROR_STRING,
			"debug_macro_arg:  macro arg %d saved at 0x%lx (%lu chars)",
			i,(long)args[i],(long)strlen(args[i]));
	}
	advise(ERROR_STRING);
}

#define DEBUG_MACRO_ARG(idx)					\
	if( debug & qldebug ) debug_macro_arg(QSP_ARG  mp,args,idx);

#else /* ! QUIP_DEBUG */

#define DEBUG_MACRO_ARG(idx)

#endif /* ! QUIP_DEBUG */


// read the macro args for a macro expansion

static const char **read_macro_args(QSP_ARG_DECL  Macro *mp)
{
	const char **args;
	int i;

	args = (const char **)getbuf(MACRO_N_ARGS(mp) * sizeof(char *));

	/* first read and store the macro arguments */
	for(i=0;i<MACRO_N_ARGS(mp);i++){
		args[i] = read_ith_macro_arg(QSP_ARG  mp,i);
		if( args[i] == NULL ){
			int j;

			sprintf(ERROR_STRING,
		"Missing arguments for macro %s!?", MACRO_NAME(mp));
			warn(ERROR_STRING);

			for(j=0;j<i;j++) rls_str(args[j]);
			givbuf(args);

			return NULL;
		}
		DEBUG_MACRO_ARG(i)
	}
	return args;
}

/* Now see if this macro has been expanded already.
 *
 * Originally (and still), we only did this test if
 * the macro didn't allow recursion -
 * but a BUG was discovered in exit_macro,
 * which would pop multiple instances of a recursively called
 * macro.  A kludgy work-around is to put a dummy macro call
 * in-between...
 */

static int check_macro_recursion(QSP_ARG_DECL  Macro *mp)
{
	int i;

	if( ! RECURSION_FORBIDDEN(mp) )
		return 0;

	i=QLEVEL;
	while( i>=0 )
		if( QRY_MACRO(QRY_AT_LEVEL(THIS_QSP,i--)) == mp ){
			sprintf(ERROR_STRING,
				"Macro recursion, macro \"%s\":  ",MACRO_NAME(mp));
		warn(ERROR_STRING);
		if( verbose )
			qdump(SINGLE_QSP_ARG);
		return -1;
	}
	return 0;
}

static void push_macro(QSP_ARG_DECL  Macro *mp, const char **args)
{
	Query *qp;

#ifdef QUIP_DEBUG
if( debug&qldebug ){
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  Pushing text for macro %s, addr 0x%lx",
WHENCE_L(push_macro),
MACRO_NAME(mp),(u_long)MACRO_TEXT(mp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	// we use MSG_STR as a scratch buffer...
	sprintf(MSG_STR,"%s%s",MACRO_LOCATION_PREFIX,MACRO_NAME(mp));
	push_text(MACRO_TEXT(mp), MSG_STR);

	qp = CURR_QRY(THIS_QSP) ;

	assert( ( QRY_FLAGS(qp) & Q_MPASSED ) == 0 );	// why?

	SET_QRY_MACRO(qp,mp);
	SET_QRY_ARGS(qp,args);
	SET_QRY_COUNT(qp, 0);
	SET_QRY_FLAGS( qp, QRY_FLAGS(qp) | Q_MACRO_INPUT );
	SET_QRY_LINES_READ(qp, 1 );
}

/*
 * read the macro arguments, if appropriate,
 * and then push the text of the macro onto the input.
 */

static inline int expand_macro(QSP_ARG_DECL  Macro *mp)
{
	const char **args;

	if( MACRO_N_ARGS(mp) > 0 ){
		args = read_macro_args(QSP_ARG  mp);
		if( args == NULL ) return 0;
	} else {
		args = NULL;
	}

	SET_MACRO_FLAG_BITS(mp,MACRO_INVOKED);

	/* does the macro have an empty body? */
	if( MACRO_TEXT(mp) != NULL )
		push_macro(QSP_ARG  mp,args);

	return(1);
}

/*
 * Check word in buf for macro.
 *
 * If the buffer contains a macro name, then expand it.
 */

static inline int expand_macro_if(QSP_ARG_DECL  const char *buf)
{
	Macro *mp;

	/* Return if we've disabled macro expansion */
	if( !(QS_FLAGS(THIS_QSP) & QS_EXPAND_MACS) ) return(0);

	/* Does the buffer contain a macro name?  If not, return */
	mp=macro_of(buf);
	if( mp==NULL ) return(0);

	if( check_macro_recursion(QSP_ARG  mp) < 0 )
		return 0;

	return expand_macro(QSP_ARG  mp);
}

/*
 * Get next word from the top of the query file stack.
 *
 * Get next word from the top of the query file stack.
 * Calls next_raw_input_word() to get the next raw word.
 * Macro and variable expansion is performed.
 * If new input is needed and the input is an interactive
 * tty, then the prompt in the argument pline will be printed.
 * Calls lookahead() before returning.
 *
 * Returns the buffer provided by next_raw_input_word()...
 */

// was qword

const char * _next_query_word(QSP_ARG_DECL const char *pline)
		/* prompt */
{
	const char *buf;

	do {
		do {
			if( QLEVEL < Q_STOP_LEVEL ){
				return NULL;
			}

			buf=next_raw_input_word(QSP_ARG  pline);		/* read a raw word */
			if( IS_HALTING(THIS_QSP) ){
				// returning NULL here can cause
				// problems when the command will complete
				// and doesn't check for a null value...
				// e.g.  echo $undefined_var_name
				// craps out...
				return "";
			}
		} while( buf == NULL );

#ifdef QUIP_DEBUG
if( debug & qldebug ){
if( strlen(buf) < LLEN-80 ){
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  next_raw_input_word returned 0x%lx \"%s\"",
WHENCE_L(next_query_word),(u_long)buf,buf);
} else {
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  next_raw_input_word returned 0x%lx (%lu chars)",
WHENCE_L(next_query_word),(u_long)buf,(long)strlen(buf));
}
advise(ERROR_STRING);
}
#endif	/* QUIP_DEBUG */

		/* at this point, the word is complete (in buf) */

#ifdef HAVE_HISTORY
		if( IS_INTERACTIVE(CURR_QRY(THIS_QSP)) && *buf && IS_TRACKING_HISTORY(THIS_QSP) ){
			add_def(pline,buf);
		}
#endif /* HAVE_HISTORY */

		// Advance the line number to match the last line read
		// before we do lookahead.  We do this before
		// macro expansion.
		//
		// But where do we do lookahead???

		/* now see if the word is a macro */
	} while( expand_macro_if(QSP_ARG  buf) );		/* returns 1 if a macro */

	// This can happen if we run out of input while trying to
	// read the macro args...
	if( QLEVEL < Q_STOP_LEVEL ) return NULL;

	return(buf);
} /* end next_query_word() */


// Get more input, and enlarge the return buffer if necessary.

static inline void replenish_buffer(SINGLE_QSP_ARG_DECL)
{
	const char *input_ptr;
	unsigned long n_need;
	int n_have;

	n_have = (int) ((QS_RET_PTR(THIS_QSP))-QS_RET_STR(THIS_QSP));	/* cast for pc */

	input_ptr=qline("");
	assert(input_ptr!=NULL);
	n_need = strlen(QS_RET_STR(THIS_QSP))+strlen(input_ptr)+16;

	if( sb_size( QS_RET_STRBUF(THIS_QSP) ) < n_need ){
		ADD_TO_RESULT(0)	// make sure a well-formed C string
		enlarge_buffer(QS_RET_STRBUF(THIS_QSP),n_need);
		SET_QS_RET_STR(THIS_QSP,sb_buffer(QS_RET_STRBUF(THIS_QSP)));
		SET_QS_RET_PTR(THIS_QSP,QS_RET_STR(THIS_QSP)+n_have);
	}
	// Does qline update the input line ptr?  it should!
}	// end replenish_buffer

static void escape_newline(SINGLE_QSP_ARG_DECL)
{
	INCREMENT_LINES_READ(escape_newline)
	if( * QS_LINE_PTR(THIS_QSP) == 0 ){	/* end of line */
		replenish_buffer(SINGLE_QSP_ARG);
	}
}

static inline int scan_another_char(SINGLE_QSP_ARG_DECL)
{
	int c;

	if( QS_LINE_PTR(THIS_QSP) == NULL )
		return -1;

	c = * QS_LINE_PTR(THIS_QSP)++;

	if( c == 0 ) QS_LINE_PTR(THIS_QSP)--;

	return c;
}

static inline int get_octal_escape( QSP_ARG_DECL  int c)
{
	int val=0;
	int i;
	for(i=0;i<3;i++){
		val <<= 3;
		val += c -'0';
		if( i < 2 ){
			c = scan_another_char(SINGLE_QSP_ARG);
			if( c < 0 ){
			} else if( ! isdigit(c) ){
			}
		}
	}
	return val;
}

/********* Supporting routines for next_word_from_input_line *********************/
/*
 * These used to all be part of next_word_from_input_line, but they have been broken out
 * to improve readability...  Efficiency may be impacted?
 */

// When should we translate backslash sequences?
// Shall we preserve backslashes within quoted strings?  yes...
// But we need to note a backslash, because we sometimes need to escape
// quotes into a quoted string.  But we also may need to escape a backslash!

static void after_backslash(QSP_ARG_DECL  int c)
{
	if( c == 't' ){
		ADD_TO_RESULT('\t')
	} else if( c == 'r' ){
		ADD_TO_RESULT('\r')
	} else if( c == 'n' ){
		ADD_TO_RESULT('\n')
	} else if( c == 'b' ){
		ADD_TO_RESULT('\b')
	} else if( c == '$' ){
		ADD_TO_RESULT('\\')
		ADD_TO_RESULT('$')
	} else if( c == '\n' || c=='\r' ){
		escape_newline(SINGLE_QSP_ARG);
		// BUG should eat another char if \n\r or \r\n?
		// This doesn't really come up though...
	} else if( isdigit(c) ){
		int val;

		val = get_octal_escape(QSP_ARG  c);
		if( val > 0 )
			ADD_TO_RESULT(val)
		else
			ADD_TO_RESULT('?')
	} else {
		ADD_TO_RESULT(c)	/* backslash before normal char */
	}
}	// end after_backslash


// next_word_from_input_line word_scan flags
#define RW_HAVBACK	1	// backslash seen
#define RW_HAVSOME	2	// what is the difference between this and NWSEEN?
#define RW_INQUOTE	4	// opening quote seen
#define RW_NOVAREXP	8	// inhibit variable expansion (single quotes)
#define RW_INCOMMENT	16	// comment delimiter seen
#define RW_SAVING	32	// buffering text for loop
#define RW_NWSEEN	64	// non-whitespace seen
#define RW_ALLDONE	128	// newline seen and not escaped


static inline void add_quote_string_char(QSP_ARG_DECL  int c)
{
	ADD_TO_RESULT(c)
	if( IS_PRIMARY_INPUT(CURR_QRY(THIS_QSP)) ){
		if( c == '\n' && !(word_scan_flags & RW_HAVBACK) ){
			SET_WORD_SCAN_FLAG_BITS(RW_ALLDONE);
		}
	}
}

static inline void handle_white_space(QSP_ARG_DECL  int c)
{
	if( c == '\n' ){
		INCREMENT_LINES_READ(save_normal_char)
	}
	if( word_scan_flags & RW_NWSEEN ){
		// We've already seen non-whitespace, so we're done!
		SET_WORD_SCAN_FLAG_BITS(RW_ALLDONE);
	}
}

static inline void save_normal_char(QSP_ARG_DECL  int c)
{
	if( word_scan_flags & RW_INQUOTE ){
		add_quote_string_char(QSP_ARG  c);
		return;
	}
	// not in quote
	if( c == '#' ){		// comment delimiter
		SET_WORD_SCAN_FLAG_BITS(RW_INCOMMENT);
		return;
	}
	if( isspace(c) ){
		handle_white_space(QSP_ARG  c);
	} else {		/* a good character */
		ADD_TO_RESULT(c)
		SET_WORD_SCAN_FLAG_BITS(RW_NWSEEN);
	}
} // save_normal_char

/* process_normal - process a char not preceded by a backslash
 *
 * First skip leading whitespace (RW_NWSEEN indicates non-white seen)
 */

#ifdef FOOBAR
static void process_normal(QSP_ARG_DECL  int c, char **result_pp, const char **sp )
{
	char *result_ptr;
	const char *s;

	result_ptr = *result_pp;
	s = *sp;

	/* Nothing special, most characters processed here.
	 * We know that the previous character was not
	 * a backslash.
	 */

	/* If this char is a backslash, don't save, but
	 * remember.
	 */
	if( c == '\\' ){
		// To be here, we know the preceding char wasn't a backslash
		SET_WORD_SCAN_FLAG_BITS(RW_HAVBACK);
	} else {
		save_normal_char(QSP_ARG  c);
		// Should we set this if we are in a comment?
//		SET_WORD_SCAN_FLAG_BITS(RW_HAVSOME);
	} // end not backslash

	*result_pp = result_ptr;
	*sp = s;
}	// end process_normal
#endif // FOOBAR

static void left_shift_result(SINGLE_QSP_ARG_DECL)
{
	char *buf;

	buf = QS_RET_STR(THIS_QSP);
	assert( *buf != 0 );

	buf++;
	while(*buf){
		*(buf-1) = *buf;
		buf++;
	}
	*(buf-1) = *buf;	// copy the null
}

static inline void strip_quotes(QSP_ARG_DECL  int quote_char)
{
	/* quote_char should hold the right value,
	 * because we've only seen 1 quotation
	 */

	if( quote_char == SGL_QUOTE ){
		SET_WORD_SCAN_FLAG_BITS(RW_NOVAREXP);
	}

	/* it used to be a bug if the quote didn't come at the end */

	/* This test is flawed because it would incorrectly
	 * strip quotes from something like 'a'b'c' ...
	 * But that's kind of pathological, isn't it?
	 *
	 * Now we keep track of that with n_quotations!
	 */
	if( *(QS_RET_PTR(THIS_QSP)-1) == quote_char ){
		*(QS_RET_PTR(THIS_QSP)-1)=0;	/* erase the closing quote */

		/* We used to strip the leading quote by simple incrementing
		 * the start pointer:
		 * result_buf++;
		 * but now that we are using the
		 * String_Buf structure we have to move the data...
		 * bummer.
		 */
		left_shift_result(SINGLE_QSP_ARG);
	}
} // end strip_quotes

// The qp arg seems ignored here???

static inline void sync_lbptrs(SINGLE_QSP_ARG_DECL)
{
	int ql;

	ql=QLEVEL;

	while( ql != 0 && (QRY_FLAGS(QRY_AT_LEVEL(THIS_QSP,ql-1)) & Q_SAVING) ){
		ql--;
		SET_QRY_LINE_PTR(QRY_AT_LEVEL(THIS_QSP,ql),
			QRY_LINE_PTR(QRY_AT_LEVEL(THIS_QSP,ql+1)));
		if( QRY_HAS_TEXT(QRY_AT_LEVEL(THIS_QSP,ql+1)) )
			SET_QRY_FLAG_BITS(QRY_AT_LEVEL(THIS_QSP,ql),Q_HAS_SOMETHING);
		else
			CLEAR_QRY_FLAG_BITS(QRY_AT_LEVEL(THIS_QSP,ql),Q_HAS_SOMETHING);
	}
}



#define LEFT_CURLY	'{'
#define RIGHT_CURLY	'}'

// get_varval is called when a var delimiter '$' is encountered

static char * get_varval(QSP_ARG_DECL  char **spp)			/** see if buf containts a variable */
{
	const char *val_str;
	char *sp;
	char *vname;
	char tmp_vnam[LLEN];
	int had_curly=0;

	sp = *spp;

	assert( *sp == VAR_DELIM );

	sp++;		/* skip over $ sign */

	if( *sp == LEFT_CURLY ){
		sp++;
		had_curly=1;
	}

	if( *sp == VAR_DELIM ){		/* variable recursion? */
		vname = get_varval(QSP_ARG  &sp);
		if( vname==NULL ) return(NULL);
	} else {			/* read in a varaible name */
		int i;

		i=0;
		while( IS_LEGAL_VAR_CHAR(*sp) && i < LLEN )
			tmp_vnam[i++] = *sp++;
		if( i == LLEN ){
			warn("Variable name too long");
			i--;
		}
		tmp_vnam[i]=0;
		vname = tmp_vnam;
	}

	if( had_curly ){
		if( *sp != RIGHT_CURLY ){
			sprintf(ERROR_STRING,"Right curly brace expected:  \"%s\"",
				*spp);
			warn(ERROR_STRING);
		} else {
			sp++;
		}
	}

	*spp = sp;

	if( strlen(vname) <= 0 ){
		sprintf(ERROR_STRING,"null invalid variable name \"%s\"",sp);
		advise(ERROR_STRING);
		return(NULL);
	}
	val_str = var_value(vname);

	/* if not a user defined variable, check environment */
	if( val_str == NULL ){
		val_str=getenv(vname);
	}

	// Why do we copy variable values to a buffer
	// in order to correctly deal with values that
	// change before everything is evaluated.

	if( val_str != NULL ) {
		char *s;

#ifdef QUIP_DEBUG
if( debug&qldebug ){
if( strlen(val_str) < LLEN-80 ){
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  copying value string from 0x%lx ( \"%s\")",
WHENCE_L(get_varval),
(long)val_str,val_str);
} else {
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  copying value string from 0x%lx ( %lu chars)",
WHENCE_L(get_varval),
(long)val_str,(long)strlen(val_str));
}
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		copy_string(QS_VAR_BUF(THIS_QSP,QS_WHICH_VAR_BUF(THIS_QSP)),val_str);

		s=sb_buffer(QS_VAR_BUF(THIS_QSP,QS_WHICH_VAR_BUF(THIS_QSP)));

#ifdef QUIP_DEBUG
if( debug&qldebug ){
if( strlen(s) < LLEN-80 ){
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  copied value in var buf #%d, at 0x%lx ( \"%s\")",
WHENCE_L(get_varval),
QS_WHICH_VAR_BUF(THIS_QSP),
(long)s,s);
} else {
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  copied value in var buf #%d, at 0x%lx (%lu chars)",
WHENCE_L(get_varval),
QS_WHICH_VAR_BUF(THIS_QSP),
(long)s,(long)strlen(s));
}
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		SET_QS_WHICH_VAR_BUF(THIS_QSP,
			(QS_WHICH_VAR_BUF(THIS_QSP)+1) % MAX_VAR_BUFS );

		return(s);
	}

	sprintf(ERROR_STRING,"Undefined variable \"%s\"!?",vname);
	warn(ERROR_STRING);

	return(NULL);		/* no match, no expansion */
} // end get_varval

/* var_expand - expand variables in a buffer
 * This buffer may contain multiple words, spaces, etc.
 *
 * The strategy is this:  we simply copy characters until we encounter
 * a variable delimiter ('$').  Then we try to read a variable name.
 */

#define RESULT		(QS_RESULT(THIS_QSP))
#define SCRATCHBUF	(QS_SCRATCH)

static void var_expand(QSP_ARG_DECL  String_Buf *sbp)
{
	char *sp;
	u_int n_to_copy;
	char *start;
	int backslash_previous;

	if( sb_buffer(RESULT) == NULL ){
		assert( SB_SIZE(RESULT) == 0 );

		enlarge_buffer(RESULT,LLEN);
	}
	if( sb_buffer(SCRATCHBUF) == NULL ){
		assert( SB_SIZE(SCRATCHBUF) == 0 );
		enlarge_buffer(SCRATCHBUF,LLEN);
	}

	*(sb_buffer(RESULT)) = 0;
	sp=sb_buffer(sbp);

	start=sp;
	n_to_copy=0;
	backslash_previous=0;
	while(*sp){
		/* We might like to be able to make this test fail
		 * if a backslash precedes the delimiter?
		 */
		if( *sp == '\\' ){
			if( backslash_previous ){
				/* double backslashes should normally be taken out
				 * by next_word_from_input_line
				 */
				backslash_previous=0;
				sp++;
				n_to_copy++;
			} else {
				backslash_previous=1;
				/* We may not copy the backslash??? */
				//n_to_copy++;

				/* copy the chars up to now */
				if( n_to_copy > 0 ){
					/* make sure destination space is large enough */
					if( n_to_copy > SB_SIZE(SCRATCHBUF) )
						enlarge_buffer(SCRATCHBUF,n_to_copy);
					strncpy(sb_buffer(SCRATCHBUF),start,n_to_copy);
					sb_buffer(SCRATCHBUF)[n_to_copy]=0;
					cat_string(RESULT,sb_buffer(SCRATCHBUF));
				}
				sp++;
				start=sp;
				n_to_copy=0;
			}
		} else if( *sp == VAR_DELIM && !backslash_previous ){
			char *vv;

			/* copy the chars up to now */
			if( n_to_copy > 0 ){
				/* make sure destination space is large enough */
				if( n_to_copy > SB_SIZE(SCRATCHBUF) )
					enlarge_buffer(SCRATCHBUF,n_to_copy);
				strncpy(sb_buffer(SCRATCHBUF),start,n_to_copy);
				sb_buffer(SCRATCHBUF)[n_to_copy]=0;
				cat_string(RESULT,sb_buffer(SCRATCHBUF));
			}

			vv = get_varval(QSP_ARG  &sp);

#ifdef QUIP_DEBUG
if( debug&qldebug ){
if( strlen(vv) < LLEN-80 ){
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  get_varval returned 0x%lx ( \"%s\")",
WHENCE_L(var_expand),
(long)vv,vv);
} else {
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  get_varval returned 0x%lx (%lu chars)",
WHENCE_L(var_expand),
(long)vv,(long)strlen(vv));
}
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			if( vv != NULL ){
				cat_string(RESULT,vv);
			}

			/* sp is updated by get_varval() */
			start=sp;
			n_to_copy=0;
		} else {
			sp++;
			n_to_copy++;
			backslash_previous=0;
		}
	}
	if( n_to_copy > 0 ){
		/*
		strncpy(buf,start,n_to_copy);
		buf[n_to_copy]=0;
		*/
		cat_string(RESULT,start);
	}
	/* now overwrite the argument string with the result */
	/* This is a bit inefficient... */
	copy_strbuf(sbp,RESULT);
}

#define QRY_RETSTR	QRY_RETSTR_AT_IDX(CURR_QRY(THIS_QSP),		\
				QRY_RETSTR_IDX(CURR_QRY(THIS_QSP)))
#define SET_QRY_RETSTR(sbp)						\
			SET_QRY_RETSTR_AT_IDX(CURR_QRY(THIS_QSP),	\
				QRY_RETSTR_IDX(CURR_QRY(THIS_QSP)),sbp)

// This version wraps around, but there's no check that N_QRY_RETSTRS is large enough.
// But otherwise we have to put reset_return_strings everwhere...
#define NEXT_QRY_RETSTR							\
									\
	SET_QRY_RETSTR_IDX(CURR_QRY(THIS_QSP),				\
		( QRY_RETSTR_IDX(CURR_QRY(THIS_QSP)) >= (N_QRY_RETSTRS-1) ? \
		0 : (1+QRY_RETSTR_IDX(CURR_QRY(THIS_QSP))) ) );

static String_Buf *query_return_string(SINGLE_QSP_ARG_DECL)
{
	String_Buf *sbp;

	assert( QRY_RETSTR_IDX(CURR_QRY(THIS_QSP)) < N_QRY_RETSTRS );

	// Better to do this at struct init?
	if( (sbp = QRY_RETSTR) == NULL ){
		SET_QRY_RETSTR(new_stringbuf());
		sbp = QRY_RETSTR;
	}

	// This might be a good place to increment?
	// We advance the index for next time
	NEXT_QRY_RETSTR
	return sbp;
}

static void clear_return_string_contents(String_Buf *sbp)
{
	(*(sb_buffer(sbp))) = 0;	/* default is "" */
}

static void insure_adequate_size(QSP_ARG_DECL  String_Buf *sbp)
{
	u_int need_size;
	Query *qp;

	qp = CURR_QRY(THIS_QSP);

	// The word should really be much less than the whole mess,
	// although it could be a quoted string?

	need_size = (int)strlen(QRY_LINE_PTR(qp) )+16;	/* conservative estimate */

	if( SB_SIZE(sbp) < need_size ){
		enlarge_buffer(sbp, need_size);
	}
}

/* We don't let backslash's escape
 * newlines within comments...
 * For multi-line comments, use a new delimiter.
 */

static inline void check_for_end_of_comment(QSP_ARG_DECL  int c)
{
	if( c == '\n' ){
		INCREMENT_LINES_READ(check_for_end_of_comment)
		CLEAR_WORD_SCAN_FLAG_BITS(RW_INCOMMENT);
	}
}

static inline void save_text_to_query(QSP_ARG_DECL  Query *qp, const char *buf)
{
	assert( QRY_TEXT_BUF(qp) != NULL );
	cat_string(QRY_TEXT_BUF(qp),buf);
}


/*
 *	Save as string for later interpretation.
 *
 *	We save text for reinterpreting in loops;
 *	we save parsed words, and quotes (which are
 *	ignored when parsing words).
 *
 *	We save at the next level down, so that we are at the same
 *	levels the first time and subsequent times...
 */

static inline void save_text_for_loop(QSP_ARG_DECL  const char* buf)
	/* query structure pointer */
	/* text to save */
{
	int ql;

	assert(QLEVEL>0);
	assert( QRY_FLAGS(QRY_AT_LEVEL(THIS_QSP,QLEVEL-1)) & Q_SAVING );

	// Do we save at multiple levels???
	// nested loops?
	ql=QLEVEL;
	while( ql > 0 && (QRY_FLAGS(QRY_AT_LEVEL(THIS_QSP,ql-1)) & Q_SAVING) ){
		Query *qp;
		ql--;
		qp=QRY_AT_LEVEL(THIS_QSP,ql);
		save_text_to_query(QSP_ARG  qp,buf);
	}
} // end save_text_for_loop

static inline void save_char_for_loop(QSP_ARG_DECL  int c)
{
	char buf[2];

	buf[0]=(char)c;
	buf[1]=0;
	save_text_for_loop(QSP_ARG  buf);
}

#define IS_QUOTE_CHAR(c)	( c == DBL_QUOTE || c == SGL_QUOTE )

// Check for opening or closing quote

static inline void check_quote_stuff(QSP_ARG_DECL  int c)
{
	//assert( IS_QUOTE_CHAR(c) );	// not needed as long as we only call below after "if"

	if( word_scan_flags & RW_INQUOTE ){	// already in a quote?
		/* check if the character is the closing quote mark
		 * AND it does not follow a backslash
		 */
		if( c == start_quote && !(word_scan_flags & RW_HAVBACK) ){
			CLEAR_WORD_SCAN_FLAG_BITS(RW_INQUOTE);
			n_quotations++;
		}
		return;
	} else {
		/* this is the opening quote */
		start_quote = c;
		SET_WORD_SCAN_FLAG_BITS(RW_INQUOTE);
	}
}

static inline void process_this_character(QSP_ARG_DECL  int c )
{
	if( word_scan_flags & RW_SAVING ) {
		save_char_for_loop(QSP_ARG  c);
	}

	/* now do something with this character */

	if( word_scan_flags & RW_INCOMMENT ){
		check_for_end_of_comment(QSP_ARG  c);
		return;
	}

	if( IS_QUOTE_CHAR(c) )
		check_quote_stuff(QSP_ARG  c);

	if( word_scan_flags & RW_HAVBACK ){		/* char follows backslash */
		after_backslash(QSP_ARG  c);
		CLEAR_WORD_SCAN_FLAG_BITS(RW_HAVBACK);
		return;
	}
	
	/* If this char is a backslash, don't save, but
	 * remember.
	 */
	if( c == '\\' ){
		// To be here, we know the preceding char wasn't a backslash
		SET_WORD_SCAN_FLAG_BITS(RW_HAVBACK);
	} else {
		save_normal_char(QSP_ARG  c);
		// Should we set this if we are in a comment?
//		SET_WORD_SCAN_FLAG_BITS(RW_HAVSOME);
	}
} // process_this_character

// return 1 if more to do, 0 otherwise
// called only from next_word_from_input_line

static int process_next_input_character(SINGLE_QSP_ARG_DECL)
{
	Query *qp;
	int c;

	qp = CURR_QRY(THIS_QSP);

	c = scan_another_char(SINGLE_QSP_ARG);
	if( c == 0 ) return 0;

	process_this_character(QSP_ARG  c);

	if( word_scan_flags & RW_ALLDONE )
		return 0;
	else
		return 1;
}	// end process_next_input_character

static void transfer_input_characters(SINGLE_QSP_ARG_DECL)
{
	while( process_next_input_character(SINGLE_QSP_ARG) )
		;
}

/*
 * Copy the next query word from the query stack's
 * line buffer (q_lbptr) into a dynamically growable
 * buffer.  Originally we used a circular list of buffers,
 * but that was flawed, because we had no way of marking
 * the buffers as unavailable for reuse.  The tricky part is knowing
 * when we can release a buffer, because the word we read might
 * get pushed as input to be scanned.  If we associate each
 * buffer with the query level that it is at, then we should be OK?
 * because if we push it, we will be at a higher level?
 *
 * Will copy text from the current query buffer until the end of text
 * is reached, or a white space character is encountered,
 * or a quote is closed.
 * Text is quoted if the next char is a quote char.  Space
 * characters will be included if enclosed in single or double quotes.
 * Standard
 * C backslash sequences like \n, \b are replaced by the characters
 * they signify.  Similarly, a backslash followed by (any number of, but
 * hopefully 3 or fewer) octal digits
 * will be replaced by the appropriate character.
 * Otherwise single backslashes are skipped, two backslashes produce
 * a single backslash.  A backslash preceding a quote mark inhibits
 * the quoting action of the quote.  This is probably a bug, in that
 * a backslash preceding an opening quote should have no effect.
 * Quotes are stripped when they are the first and last characters
 * of the extracted word; this is probably buggy given the allowability
 * of multiple quotations alluded to above.
 *
 *
 * If a single backslash precedes a dollar sign (VAR_DELIM), we might want
 * to preserve it so it can have an action in variable expansion?
 *
 * Returns ptr to text if some text is copied, NULL otherwise
 *
 * Originally, variable expansion was only done here on strings enclosed in
 * double quotes...  To avoid the proliferation of double quotes in scripts
 * (and the errors resulting from forgetting them), we relax this, and scan
 * for variables in all cases EXCEPT single quoted strings.
 * (This makes redundant the check (elsewhere) for the initial character
 * being a dollar sign.)
 *
 * BUG?  because next_word_from_input_line uses a dynamically growable string buffer,
 * it can return a string which is longer than LLEN...  This can happen
 * for instance when a closing quote is missing...
 *
 * variable expansion is performed at the end of next_word_from_input_line, by calling var_expand
 */

// called only by next_raw_input_word

static char * next_word_from_input_line(SINGLE_QSP_ARG_DECL)
{
	String_Buf *sbp;

	assert(QS_LINE_PTR(THIS_QSP)!=NULL);
	if( * QS_LINE_PTR(THIS_QSP) == 0 )
		return NULL;	// input exhausted

	word_scan_flags=0;
	n_quotations=0;

	// BUG shouldn't need two separate flags???
	if( NEED_TO_SAVE( CURR_QRY(THIS_QSP) ) ){
		SET_WORD_SCAN_FLAG_BITS(RW_SAVING);
	}

	sbp = query_return_string(SINGLE_QSP_ARG);

	insure_adequate_size(QSP_ARG  sbp);
	clear_return_string_contents(sbp);

	SET_QS_RET_STRBUF(THIS_QSP,sbp);
	SET_QS_RET_STR(THIS_QSP,sb_buffer(sbp));
	SET_QS_RET_PTR(THIS_QSP,sb_buffer(sbp));


// should we skip initial spaces?

	transfer_input_characters(SINGLE_QSP_ARG);


	/* If we are here, our input pointer should be pointing at a null
	 * byte OR a whitespace character...
	 *
	 * We get here when we are done reading the word, either because
	 * we have run out of input or because we encountered a white
	 * space character.
	 *
	 * We don't want to return an empty string,
	 * so we check the string length.
	 */

	if( word_scan_flags & RW_HAVBACK )
		advise("still have backslash at end of buffer!?");

	* QS_RET_PTR(THIS_QSP)=0;	// terminate C string

	assert( strlen(QS_RET_STR(THIS_QSP)) < (SB_SIZE(sbp)-1) );

	if( start_quote && (word_scan_flags & RW_INQUOTE) ){
		sprintf(ERROR_STRING,"next_word_from_input_line:  no closing quote (start_quote = %c)",start_quote);
		warn(ERROR_STRING);
		/* If the buffer has overflowed, we can't print into error_string! */
#define BC_STR	"buffer contained "
		if( strlen(QS_RET_STR(THIS_QSP))+strlen(BC_STR) < (LLEN-4) ){
			sprintf(ERROR_STRING,"%s\"%s\"",BC_STR,QS_RET_STR(THIS_QSP));
			advise(ERROR_STRING);
		} else {
			sprintf(ERROR_STRING,"buffer contains %ld chars",(long)strlen(QS_RET_STR(THIS_QSP)));
			advise(ERROR_STRING);
		}
	}

	if( ! (word_scan_flags & RW_NWSEEN) )
		return(NULL);

	// The level should not be popped???
	if( * QS_LINE_PTR(THIS_QSP) == 0 ){
		// hope this doesn't mess up line numbering...
		SET_QS_LINE_PTR(THIS_QSP,NULL);
		CLEAR_QRY_FLAG_BITS(CURR_QRY(THIS_QSP),Q_HAS_SOMETHING);
		// Should we clear the HAVE_SOMETHING flag???
	}

	// sync up the lbptr's at the saving levels...
	if( word_scan_flags & RW_SAVING ) {
		sync_lbptrs(SINGLE_QSP_ARG);
	}

	/* strip quotes if they enclose the entire string */
	/* This is useful in vt script, but bad if we are using these routines to pass input
	 * to the vt expression parser!?
	 */

	if( (QS_FLAGS(THIS_QSP) & QS_STRIPPING_QUOTES)
			&& IS_QUOTE_CHAR( * QS_RET_STR(THIS_QSP) )
			&& n_quotations==1 ){
		strip_quotes(QSP_ARG  start_quote);
	}

	/* BUG this will prevent variable expansion of lines
	 * which contain single quoted strings and vars...
	 */

	if( ! (word_scan_flags & RW_NOVAREXP) )
		var_expand(QSP_ARG  sbp);

	return(sb_buffer(sbp));
} // end next_word_from_input_line

static const char *next_word_from_level(QSP_ARG_DECL  const char *pline)
{
	Query *qp;
	const char *buf;

	if( QLEVEL < 0 ) return NULL;

	assert( ! IS_HALTING(THIS_QSP) );

	qp=(CURR_QRY(THIS_QSP));
	if( !QRY_HAS_TEXT(qp) )	/* need to read more input */
	{
		buf=qline(pline);
	}

	if( QLEVEL < 0 ){
		return NULL;
	}

	SYNC_LINENO

	qp=(CURR_QRY(THIS_QSP));	/* qline may pop the level!!! */
	//eatup_space(SINGLE_QSP_ARG);

	if( QRY_HAS_TEXT(qp) ){
		/* next_word_from_input_line() returns non-NULL if successful */
		SYNC_LINENO
		if( (buf=next_word_from_input_line(SINGLE_QSP_ARG)) == NULL ){
			CLEAR_QRY_FLAG_BITS(qp,Q_HAS_SOMETHING);
			return NULL;
		}
    } else return NULL;

	return(buf);
} // end next_word_from_level

/*
 *	Get a raw word.
 *
 *	Get a raw word from the top of the query file stack.  If there is
 *	no current text, will get more by calling qline().  Strips leading
 *	white space, returns the next space delimited word by calling next_word_from_input_line().
 *	No macro or variable expansion is performed. (sic)
 *
 *	Variable expansion performed in next_word_from_input_line, unless single-quoted...
 *
 *	returns buffer returned by next_word_from_input_line()
 */

// was gword

static const char * next_raw_input_word(QSP_ARG_DECL  const char* pline)
		/* prompt string */
{
	SET_QS_FLAG_BITS(THIS_QSP, QS_STILL_TRYING);
	if( IS_HALTING(THIS_QSP) ){
		// clear the has_something flag
		// so that we won't try to read more
		CLEAR_QRY_FLAG_BITS(CURR_QRY(THIS_QSP),Q_HAS_SOMETHING);
		return NULL;
	}

	return next_word_from_level(QSP_ARG  pline);
} // next_raw_input_word

/*
 *	Save a single character.
 */

#ifdef NOT_YET
/*
 * read a line from the input, and pass it back.
 * This is used by the matlab interpreter (and others???) to read
 * input, but to avoid most of the other input processing...
 */

const char * steal_line(QSP_ARG_DECL  const char* pline)
{
	/* int n; */
	const char *buf;

	buf=qline(pline);
	/*
	n=strlen(buf);
	if( n>1 && (buf[n-1] == '\n' || buf[n-1] == '\r') )
		buf[n-1]=0;
	*/
	//SET_QRY_HAS_TEXT(CURR_QRY(THIS_QSP),0);
	CLEAR_QRY_FLAG_BITS(CURR_QRY(THIS_QSP),Q_HAS_SOMETHING);
	SET_QRY_LINENO(CURR_QRY(THIS_QSP), QRY_LINES_READ(CURR_QRY(THIS_QSP)) );
	return(buf);
} // end steal_line
#endif /* NOT_YET */

/*
 * Read a line from the current query file.
 *
 * Read a line from the query stack by repeatedly calling nextline()
 * until some text is obtained.
 * If transcripting is on,
 * saves the line to the transcript file.
 */

const char * _qline(QSP_ARG_DECL  const char *pline)
		/* prompt string */
{
	Query *qp;
	const char *buf;

	while(1) {
		if( QLEVEL < 0 ){
			return NULL;
		}

		/* if the current level is out, nextline will pop 1 level */
DEBUG_LINENO(qline before nextline)
		buf=nextline(pline);	// qline
DEBUG_LINENO(qline after nextline)

		if( IS_HALTING(THIS_QSP) ) return NULL;

		if( QLEVEL < 0 ){
			return NULL;
		}

		qp=(CURR_QRY(THIS_QSP));

		if( QRY_HAS_TEXT(qp) ){
			if( IS_DUPING ){
				dup_word(QRY_LINE_PTR(qp) );
				dup_word("\n");
			}
			return(buf);
		}
	}
}

#ifdef HAVE_HISTORY
#ifdef TTY_CTL

COMMAND_FUNC( set_completion )
{
	if( askif("complete commands") ){
		advise("enabling automatic command completion");
		SET_QS_FLAG_BITS(THIS_QSP,QS_COMPLETING);
	} else {
		advise("disabling automatic command completion");
		CLEAR_QS_FLAG_BITS(THIS_QSP,QS_COMPLETING);
		sane_tty(SINGLE_QSP_ARG);
	}
}

#endif /* TTY_CTL */
#endif /* HAVE_HISTORY */

static void halt_stack(SINGLE_QSP_ARG_DECL)
{
	if( QS_SERIAL == FIRST_QUERY_SERIAL ){
		nice_exit(0);
	} else {
		SET_QS_FLAG_BITS(THIS_QSP,QS_HALTING);
		CLEAR_QS_FLAG_BITS(THIS_QSP,QS_STILL_TRYING);
	}
}

#ifdef HAVE_HISTORY
#ifdef TTY_CTL
static const char *hist_select(QSP_ARG_DECL const char* pline)
{
	const char *s;
	Query *qp;

	qp = CURR_QRY(THIS_QSP);

	s=get_response_from_user(QSP_ARG  pline,QRY_FILE_PTR(qp),stderr);
	if( s==NULL ){			/* ^D */
		if( QLEVEL > 0 ){
			pop_file();
			return NULL;
		} else {
			advise("EOF");
			nice_exit(0);
		}
	}

	SET_QRY_FLAG_BITS(qp,Q_HAS_SOMETHING);
	SET_QRY_LINE_PTR(qp,s);

	return(s);
} // end hist_select
#endif /* TTY_CTL */
#endif /* HAVE_HISTORY */

static void query_stream_finished(SINGLE_QSP_ARG_DECL)
{
	if( QLEVEL > 0 ){	/* EOF redir file */
		pop_file();
	} else if( has_stdin ){
sprintf(ERROR_STRING,"EOF on %s",QS_NAME(THIS_QSP));
advise(ERROR_STRING);
		halt_stack(SINGLE_QSP_ARG);
		// halting master stack will exit program,
		// but other threads have to get out gracefully...
	} else {
		// Normally if we encounter EOF on the root
		// query stack, we want to quit.
		// But under iOS, there is no stdin,
		// and the first file is the startup file.
		//
		// This caused an infinite loop of prompt printing...
		pop_file();
	}
}

#ifdef HAVE_HISTORY
#ifdef TTY_CTL

// How do we tell an empty line from a null string ('')?
// In the latter case, we want to re
static const char * get_line_interactive(QSP_ARG_DECL  const char *pline)
{
	const char *s;
	int start_level=QLEVEL;

	while( start_level == QLEVEL ) {
		fputs(pline,stderr);
		s=hist_select(QSP_ARG  pline);
		if( s == NULL ) return NULL;
		if( strlen(s) > 0 ) return s;
	};
	return NULL;
}

#endif // TTY_CTL
#endif // HAVE_HISTORY

// Cautious helper function makes sure a newly-read line has a newline char at the end

static int check_for_complete_line(QSP_ARG_DECL  const char *buf)
{
	int n;

	n=(int)strlen(buf);
	assert( n < LLEN );

	if( n == 0 ){
		// This case occurred when interpreting a
		// stored encrypted file on the ipod simulator...
		// Not sure why it is happening, because
		// the file seems to contain the right data...
		//
		// This message also prints when we read the
		// startup file, but that's probably not using
		// fgets, as we decrypt to a buffer...
//advise("query read function returned an empty string!?");
		return 0;
	}
				
	n--;

	if( QRY_READFUNC(CURR_QRY(THIS_QSP)) == ((READFUNC_CAST) FGETS) && buf[n] != '\n' &&
		buf[n] != '\r' ){
		warn("check_for_complete_line:  input line not terminated by \\n or \\r");
		sprintf(ERROR_STRING,"line:  \"%s\"",buf);
		advise(ERROR_STRING);
		return -1;
	}
	return 0;
}

/*
 * Get the next line from the top of the query stack.
 *
 * Gets the next line.  If the input is an interactive tty, prints
 * the prompt and calls hist_select() (if HAVE_HISTORY is defined).
 * Otherwise calls the query structures read function (fgets() for
 * normal files).  If EOF is encountered, pops the file and returns.
 *
 * advances q_lines_read, which may not be the current lineno if
 * called from lookahead
 *
 * We return the buffer, AND set QRY_LINE_PTR(qp)  - redundant?
 */

const char * _nextline(QSP_ARG_DECL  const char *pline)
		/* prompt */
{
	Query *qp;
	String_Buf *sbp;
#ifdef MAC
	extern void set_mac_pmpt(char *);
#endif /* MAC */

	if( IS_HALTING(THIS_QSP) ){
		if( QLEVEL > 0 )		// or should >= ???  BUG?
			pop_file();
		return(NULL);
	}

	qp=(CURR_QRY(THIS_QSP));

	// buf might be NULL if we are at the end of a macro?

#ifdef MAC
	// what is this???
	set_mac_pmpt(pline);
#endif /* MAC */

#ifdef HAVE_HISTORY
#ifdef TTY_CTL
	// shouldn't this be any interactive shell?
	while( IS_INTERACTIVE(qp) && IS_TRACKING_HISTORY(THIS_QSP) && IS_COMPLETING(THIS_QSP) )
	{
		const char *s;
		s = get_line_interactive(QSP_ARG  pline);
		if( s != NULL ){
			return s;
		}
		qp = CURR_QRY(THIS_QSP);
	}
#endif /* TTY_CTL */
#endif /* HAVE_HISTORY */

	// BUG check force prompt flag and print prompt here if set
	// force_pmpt introduced for socket connection?

	/* We used to advance the line number here, but when reading from a buffer
	 * we advance the line numbers in next_word_from_input_line, and that renders this unnecessary.
	 *
	 * The problem is, lines get read from various places, and
	 * we need to count them all...
	 *
	 * It would certainly be simplest to count them when we
	 * read them...  But the "line buffer" can contain multiple
	 * lines in the case of a macro or a loop, so we have to
	 * count as we scan.
	 */

//	INCREMENT_LINES_READ(nextline)

	/* Call the read function - fgets if it is a regular file */

	// Make sure we have at least LLEN chars in the buffer...
	sbp = QRY_BUFFER(qp);
	assert(sbp!=NULL);
	if( SB_SIZE(sbp) < LLEN )
		enlarge_buffer(sbp,LLEN);

	if( (*(QRY_READFUNC(qp)))(QSP_ARG  (void *)sb_buffer(sbp),(int)sb_size(sbp),(void *)QRY_FILE_PTR(qp)) == NULL ){
		/* this means EOF if reading with fgets()
		 * or end of a macro...
		 */
		query_stream_finished(SINGLE_QSP_ARG);
		return("");
	} else {		/* have something */
		if( check_for_complete_line(QSP_ARG  sb_buffer(sbp)) < 0 ){
fprintf(stderr,"check_for_complete_line returning NULL\n");
			return NULL;
		}

		//SET_QRY_HAS_TEXT(qp,1);	// why is this commented out???  BUG?
		SET_QRY_FLAG_BITS(qp,Q_HAS_SOMETHING);
		SET_QRY_LINE_PTR(qp,sb_buffer(sbp));
		return(sb_buffer(sbp));
	}
	/* NOTREACHED */

	/* But just to make in ANSI C++ complaint return something */

	/* OK, but this causes warnings from other compilers...
	 * Is the C++ compiler stupid or what?
	 */
	return NULL;
} // end nextline

static const char *getmarg(QSP_ARG_DECL  int index)
{
	if( index < 0 || index >= MACRO_N_ARGS(QRY_MACRO(CURR_QRY(THIS_QSP))) ){
		sprintf(ERROR_STRING,
			"getmarg:  arg index %d out of range for macro %s (%d args)",
			1+index,MACRO_NAME(QRY_MACRO(CURR_QRY(THIS_QSP))),
			MACRO_N_ARGS(QRY_MACRO(CURR_QRY(THIS_QSP))));
		warn(ERROR_STRING);
		return(NULL);
	}
#ifdef QUIP_DEBUG
if( debug & qldebug ){
if( strlen(QRY_ARG_AT_IDX(CURR_QRY(THIS_QSP),index))<(LLEN-80) ){
sprintf(ERROR_STRING,
"%s - %s (qlevel = %d):  returning macro arg %d at 0x%lx (%s)",
WHENCE_L(getmarg),
index,
(long)QRY_ARG_AT_IDX(CURR_QRY(THIS_QSP),index),
QRY_ARG_AT_IDX(CURR_QRY(THIS_QSP),index)
);
} else {
sprintf(ERROR_STRING,
"%s - %s (qlevel = %d):  returning macro arg %d at 0x%lx (%lu chars)",
WHENCE_L(getmarg),
index,
(long)QRY_ARG_AT_IDX(CURR_QRY(THIS_QSP),index),
(long)strlen(QRY_ARG_AT_IDX(CURR_QRY(THIS_QSP),index))
);
}
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	return( QRY_ARG_AT_IDX(CURR_QRY(THIS_QSP),index) );
}

/* return the value of the INTERACTIVE flag - input is not a file or macro */

int intractive(SINGLE_QSP_ARG_DECL)
{
	// We need to call lookahead to make sure
	// that we really know what the current input file is.

	lookahead(SINGLE_QSP_ARG);

	if( QLEVEL < 0 ) return 0;

//	if(!(QS_FLAGS(THIS_QSP) & QS_INITED)) init_query_stack(THIS_QSP);
	return IS_INTERACTIVE( CURR_QRY(THIS_QSP) );
}


/* scan_line_remainder
 *
 * scan from qp_lbptr to the end of the line.
 * We expect only white space before a comment delimiter.
 */

static inline int scan_line_remainder(QSP_ARG_DECL  const char *location )
{
	int c;
	int comment_seen=0;
	int status=0;
	const char *s = QS_LINE_PTR(THIS_QSP);

	while( (c=(*s++)) && c != '\n' ){
		if( ! isspace(c) ){
			if( c == '#' )
				comment_seen=1;
			else if( ! comment_seen )
				status= -1;
		}
	}
	QS_LINE_PTR(THIS_QSP) = s;
	return status;
}

// No recursive calls.  Therefore it
// should be safe to use a single static buffer...
// Assumes no parallel defintion of macros!?  (BUG?)
// BUG should convert to stringbuf!

static const char *extract_line_for_macro(SINGLE_QSP_ARG_DECL)
{
	static char linebuf[LLEN];
	const char *from;
	char *to;
	int level;
	int n;

	level = QLEVEL;
	while( ! QRY_HAS_TEXT(CURR_QRY(THIS_QSP)) ){
		/*from=*/qline("");
		// BUG - we need to handle premature EOF?
		if( QLEVEL != level ){
			sprintf(ERROR_STRING,"extract_line_for_macro:  premature EOF!?");
			warn(ERROR_STRING);
			return NULL;
		}
	}

	// Copy up until the next newline

	from = QRY_LINE_PTR(CURR_QRY(THIS_QSP)) ;
	to = linebuf;
	n=0;
	if( *from == '.' ){
		if( *(from+1) != '\n' ){
	warn("Macro definition should be terminated by a line containing a single '.'!?");
		}
	}
	while( *from && *from != '\n' ){
		// check for buffer overrun
		if( n >= LLEN-1 ){
			sprintf(ERROR_STRING,"extract_line_for_macro:  buffer too small!?");
			warn(ERROR_STRING);
			to--;
			n--;
		}
		*to++ = *from++;
		n++;
	}
	if( *from == '\n' )
		from++;		// advance, but don't copy...

	SET_QRY_LINE_PTR(CURR_QRY(THIS_QSP),from);
	if( *from == 0 ){	// out of text?
		CLEAR_QRY_FLAG_BITS(CURR_QRY(THIS_QSP),Q_HAS_SOMETHING);
	}

	*to = 0;	// terminate line

	INCREMENT_LINES_READ(extract_line_for_macro)

	return linebuf;
} // extract_line_for_macro

/* read in macro text
 *
 * Read text until a line with a single period '.' is encountered.
 *
 * The original implementation broke when we added the ability to
 * read encrypted files.  If we decrypt a file into a buffer, and then
 * push the entire buffer onto the input stack, it appears as one
 * great big line...
 */

static int get_next_macro_line(QSP_ARG_DECL  String_Buf *mac_sbp)
{
	const char *s;

	s=extract_line_for_macro(SINGLE_QSP_ARG);		// read_macro_body
	if( s == NULL ) return -1;

	if( *s == '.' )
		return 0;

	cat_string(mac_sbp,s);
	cat_string(mac_sbp,"\n");
	return 1;
}

String_Buf * read_macro_body(SINGLE_QSP_ARG_DECL)
{
	const char *instructions="Enter text of macro; terminate with line beginning with '.'";
	Query *qp;
	int status;
	/* we make a new one for each and every macro... */

	/* The old one had a memory leak, because the string buffers were
	 * never deallocated...
	 */
	String_Buf *mac_sbp;

	// How could we ever get here without initializing the query stack???
	//if(!(QS_FLAGS(THIS_QSP) & QS_INITED)) init_query_stack(THIS_QSP);
	assert( QS_FLAGS(THIS_QSP) & QS_INITED );

	SYNC_LINENO

	qp = CURR_QRY(THIS_QSP);

	if( IS_INTERACTIVE(qp) || (QS_FLAGS(THIS_QSP) & QS_FORCE_PROMPT) )
		advise(instructions);

	mac_sbp = new_stringbuf();
	copy_string(mac_sbp,"");	// initialize - redundant?

	while( (status=get_next_macro_line(QSP_ARG  mac_sbp)) == 1 )
		;

	if( status != 0 )
		warn("bad macro definition");

	// BUG?  We might return NULL so that we can print an error message with
	// the macro name...
	return(mac_sbp);
} /* end read_macro_body */

static const char *check_macro_arg_item_spec(QSP_ARG_DECL  Macro_Arg *map, const char *s)
{
	int n;
	char item_type_name[LLEN];
	char pmpt[LLEN];

	map->ma_itp=NULL;		// default

	if( *s != '<' ) return s;

	n=(int)strlen(s);
	if( s[n-1] != '>' ){
		warn("Unterminated macro argument item type specification.");
		return s;
	}

	strcpy(item_type_name,s+1);
	item_type_name[n-2]=0;	/* kill closing bracket */

	map->ma_itp = get_item_type(item_type_name);
	if( map->ma_itp == NULL ){
		warn("Unable to process macro argument item type specification.");
		return s;
	}
	// Now read the normal macro arg description/prompt
	sprintf(pmpt,"prompt for %s",item_type_name);
	return nameof(pmpt);
}

// Read the macro args for a new macro definition

static Macro_Arg * read_macro_arg_spec(QSP_ARG_DECL int i)
{
	char pstr[LLEN];
	char pstr2[LLEN];
	const char *s;
	Macro_Arg *map;
	static const char *nsuff[]={"st","nd","rd","th"};

	map = getbuf(sizeof(Macro_Arg));

	if( i<3 )
		sprintf(pstr,"prompt for %d%s argument",
			i+1,nsuff[i]);
	else
		sprintf(pstr,"prompt for %d%s argument",
			i+1,nsuff[3]);

	sprintf(pstr2,"%s (or optional item type spec)",pstr);

	/* this won't be freed until the macro is released... */
	s = nameof(pstr2);

	/* We can specify the item type of the prompted-for object
	 * by preceding the prompt with an item type name in brackets,
	 * e.g. <Data_Obj> image
	 */
	s = check_macro_arg_item_spec(QSP_ARG  map, s);
	map->ma_prompt = savestr(s);
	return map;
}

void rls_macro_arg( Macro_Arg * map )
{
	rls_str(map->ma_prompt);
	givbuf(map);
}

static inline Macro_Arg ** read_macro_arg_table(QSP_ARG_DECL  int n)
{
	Macro_Arg **ma_tbl;
	int i;

	assert(n>0);
	ma_tbl = getbuf(n*sizeof(Macro_Arg));
	for(i=0;i<n;i++)
		ma_tbl[i] = read_macro_arg_spec(QSP_ARG  i);

	// At this point, there should be no more text on the line,
	// except possibly a comment...


	return ma_tbl;
}

static inline void check_macro_def_line(SINGLE_QSP_ARG_DECL)
{
	if( QRY_LINENO(CURR_QRY(THIS_QSP)) == QRY_LINES_READ(CURR_QRY(THIS_QSP)) ){
		if( scan_line_remainder(QSP_ARG  "macro declaration") < 0 )
			warn("extra text after macro args!?");
	}
}

Macro_Arg **setup_macro_args(QSP_ARG_DECL  int n)
{
	Macro_Arg **ma_tbl;

	if( n > 0 ){
		ma_tbl = read_macro_arg_table(QSP_ARG  n);
	} else {
		ma_tbl = NULL;
	}
	check_macro_def_line(SINGLE_QSP_ARG);

	return ma_tbl;
}

#ifdef NOT_USED
/*
 * Redirect input to tty
 */

void readtty(SINGLE_QSP_ARG_DECL)
{ redir(tfile(SINGLE_QSP_ARG), "/dev/tty" ); }
#endif /* NOT_USED */

void disable_stripping_quotes(SINGLE_QSP_ARG_DECL)
{
	CLEAR_QS_FLAG_BITS(THIS_QSP,QS_STRIPPING_QUOTES);
}

void enable_stripping_quotes(SINGLE_QSP_ARG_DECL)
{
	SET_QS_FLAG_BITS(THIS_QSP,QS_STRIPPING_QUOTES);
}

static void first_query_stack(Query_Stack *qsp)
{
	// In iOS, redirecting to stdin doesn't seem to
	// cause any harm, because reading it must return
	// EOF...  But on MacOS, it is a problem because
	// the program hangs waiting for input
	//
	// The above comment was written for the cocoa app,
	// but for the native mac command line version
	// we need to do this too...

#ifdef BUILD_FOR_CMD_LINE
	redir(stdin, "-" );
#endif // BUILD_FOR_CMD_LINE

#ifdef QUIP_DEBUG
	qldebug = add_debug_module("query");;
	lah_debug = add_debug_module("lookahead");;
#endif /* QUIP_DEBUG */
}

Query_Stack *new_qstk(QSP_ARG_DECL  const char *name)
{
	Query_Stack *new_qsp;
	Query_Stack *qsp_to_free=NULL;
#ifdef THREAD_SAFE_QUERY
	// Why would this be null?  First time?
	//
	// We need to have a non-null arg to pass to new_qstack...

	// qsp will be null the first time that this is called...
	if( qsp == NULL ){
		//qsp=&dummy_qs;	// may be passed to new_qstack
		// if NEW_QUERY_STACK calls new_qsp, this
		// could lead to infinite recursion???
		qsp = NEW_QUERY_STACK;	// getbuf
		SET_QS_SERIAL(qsp, n_active_threads);
		qsp_to_free = qsp;
	}
#endif /* THREAD_SAFE_QUERY */

	// We used to use a custom routine here - are there problems using the template??
	new_qsp = new_query_stack(name);

	if( qsp_to_free != NULL ){
		//DEFAULT_QSP = qsp_to_free;	// why?  appears to do nothing?
		givbuf(qsp_to_free);
	}

#ifdef THREAD_SAFE_QUERY
	SET_QS_SERIAL(new_qsp, n_active_threads);
#else /* ! THREAD_SAFE_QUERY */
	SET_QS_SERIAL(new_qsp, 0);
#endif /* ! THREAD_SAFE_QUERY */


	init_query_stack(new_qsp);


#ifdef THREAD_SAFE_QUERY
	if( n_active_threads == 0 ){
		default_qsp = new_qsp;
		first_query_stack(new_qsp);	/* point this at stdin */
	}
#else /* ! THREAD_SAFE_QUERY */
	default_qsp = new_qsp;
	first_query_stack(new_qsp);	/* point this at stdin */
#endif /* ! THREAD_SAFE_QUERY */

	// We increment n_active threads here, although the thread
	// isn't created until a teeny bit later...

#ifdef THREAD_SAFE_QUERY
	n_active_threads++;

	if( _QS_SERIAL(new_qsp) == FIRST_QUERY_SERIAL )
		SET_QS_FLAG_BITS(new_qsp,QS_HISTORY);
	else {
		CLEAR_QS_FLAG_BITS(new_qsp,QS_HISTORY);
		// Why do we have to do this - maybe so a new thread will be in a clean context?
		// but the context stacks are for the item types, not the query stacks???
		//setup_all_item_type_contexts(QSP_ARG  new_qsp);
	}
#else /* ! THREAD_SAFE_QUERY */
	SET_QS_FLAG_BITS(new_qsp,QS_HISTORY);
#endif /* ! THREAD_SAFE_QUERY */

	return(new_qsp);
} // new_qsp

#ifdef NOT_USED
/* dup stuff */

void end_dupline(SINGLE_QSP_ARG_DECL)
{
	FILE *fp;

	fp = QRY_DUPFILE(CURR_QRY(THIS_QSP));

	assert( IS_DUPING );

	fputs("\r",fp);
	fflush(fp);
	SET_QRY_FLAG_BITS(CURR_QRY(THIS_QSP), Q_FIRST_WORD );
}
#endif /* NOT_USED */

// For reasons that are no longer clear, this code quotes strings that contain
// a space...  This would only make sense if enclosing quotes had already
// been stripped.

static int need_to_chunk(const char *s)		/* does s contain non-quoted spaces? */
{
#ifdef FOOBAR
	int insgl=0, indbl=0;

	while( *s ){
		if( !(insgl|indbl) ){
			if( *s == ' ' || *s == '\t' )
				return(1);
			else if( *s == '\'' ) insgl=1;
			else if( *s == '"' ) indbl=1;
		} else if( insgl ){
			if( *s == '\'' ) insgl=0;
		} else if( indbl ){
			if( *s == '"' ) indbl=0;
		}
		s++;
	}
	return(0);
#else /* ! FOOBAR */
	return(0);	// never chunk
#endif /* ! FOOBAR */
}

// dup_word - write this word to the transcript file
//
// It is somewhat unfortunate that the indentation doesn't
// come out the way we would like it...

void _dup_word(QSP_ARG_DECL  const char *s)
{
	int chunkme;
	FILE *fp;

#ifdef THREAD_SAFE_QUERY
	// This null test wasn't needed until we tried to exit a thread...
	if( s == NULL ) return;
#endif /* THREAD_SAFE_QUERY */

	fp = QRY_DUPFILE(CURR_QRY(THIS_QSP));

	assert( IS_DUPING );

	if( ! FIRST_WORD_ON_LINE )
		fputs(" ",fp);

	// Why do we ever need to chunk???
	chunkme = need_to_chunk(s);

	if( chunkme ) fputs("'",fp);
	fputs(s,fp);
	if( chunkme ) fputs("'",fp);
	//CLR_Q_FLAG( THIS_QSP, Q_FIRST_WORD );
	CLEAR_QRY_FLAG_BITS( CURR_QRY(THIS_QSP), Q_FIRST_WORD );
}

#ifdef NOT_USED
void ql_debug(SINGLE_QSP_ARG_DECL)
{
	sprintf(ERROR_STRING,"qlevel = %d",QLEVEL);
	advise(ERROR_STRING);
		sprintf(ERROR_STRING,"q_file = 0x%lx",(u_long)QRY_FILE_PTR(CURR_QRY(THIS_QSP)));
	advise(ERROR_STRING);
}

#endif /* NOT_USED */

int _dupout(QSP_ARG_DECL  FILE *fp)			/** save input text to file fp */
{
	if( IS_DUPING ){
		warn("already dup'ing");
		return(-1);
	} else {
		SET_QRY_DUPFILE(CURR_QRY(THIS_QSP),fp);
		return(0);
	}
}


#include "quip_config.h"

/* local prototypes */



/*
 * Set input reading function for the current level of the query stack.
 * Set input reading function for the current level of the query stack.
 * User supplied functions should have the calling syntax of fgets(3),
 * which is the default.
 */

void set_query_readfunc( QSP_ARG_DECL  char * (*rfunc)(QSP_ARG_DECL  void *buf, int size, void *fp ) )
{
	assert( QS_FLAGS(THIS_QSP) & QS_INITED );
	assert( QLEVEL >= 0 );

	SET_QRY_READFUNC(CURR_QRY(THIS_QSP), rfunc);
}

// We have a stack of parser environments, and a free list to keep them around
// when they are popped...

static void init_vector_parser_data_stack(Query_Stack *qsp)
{
	SET_QS_VECTOR_PARSER_DATA_STACK(qsp,new_list());
	SET_QS_VECTOR_PARSER_DATA_FREELIST(qsp,new_list());
	SET_QS_VECTOR_PARSER_DATA(qsp,NULL);
}

#define init_vector_parser_data(vpd_p) _init_vector_parser_data(QSP_ARG  vpd_p)

static void _init_vector_parser_data(QSP_ARG_DECL  Vector_Parser_Data *vpd_p)
{
	bzero(vpd_p,sizeof(*vpd_p));

	// // Now allocate the strings
	SET_VPD_YY_INPUT_LINE(vpd_p,new_stringbuf());
	SET_VPD_YY_LAST_LINE(vpd_p,new_stringbuf());
	SET_VPD_EXPR_STRING(vpd_p,new_stringbuf());
	SET_VPD_YY_WORD_BUF(vpd_p,new_stringbuf());
	SET_VPD_EDEPTH(vpd_p, -1);
	SET_VPD_CURR_STRING(vpd_p, sb_buffer(VPD_EXPR_STRING(vpd_p)) );
	SET_VPD_SUBRT_CTX_STACK(vpd_p,new_list());
}

#define rls_vector_parser_data(vpd_p) _rls_vector_parser_data(QSP_ARG  vpd_p)

static void _rls_vector_parser_data(QSP_ARG_DECL  Vector_Parser_Data *vpd_p)
{
	rls_stringbuf(VPD_YY_INPUT_LINE(vpd_p));
	rls_stringbuf(VPD_YY_LAST_LINE(vpd_p));
	rls_stringbuf(VPD_EXPR_STRING(vpd_p));
	rls_stringbuf(VPD_YY_WORD_BUF(vpd_p));
	rls_list(VPD_SUBRT_CTX_STACK(vpd_p));
}

static Vector_Parser_Data *find_free_vector_parser_data(SINGLE_QSP_ARG_DECL)
{
	Vector_Parser_Data *vpd_p;

	if( QLIST_HEAD( QS_VECTOR_PARSER_DATA_FREELIST(THIS_QSP) ) != NULL ){
		Node *np;
		np = remHead( QS_VECTOR_PARSER_DATA_FREELIST(THIS_QSP) );
		vpd_p = NODE_DATA(np);
		rls_node(np);
	} else {
		vpd_p = getbuf( sizeof(*vpd_p) );
	}
	return vpd_p;
}

void push_vector_parser_data(SINGLE_QSP_ARG_DECL)
{
	Vector_Parser_Data *vpd_p;
	Node *np;

	vpd_p = find_free_vector_parser_data(SINGLE_QSP_ARG);
	assert(vpd_p!=NULL);

	init_vector_parser_data(vpd_p);

	np = mk_node(vpd_p);
	addHead( QS_VECTOR_PARSER_DATA_STACK(THIS_QSP), np );
	SET_QS_VECTOR_PARSER_DATA(THIS_QSP,vpd_p);
}

void pop_vector_parser_data(SINGLE_QSP_ARG_DECL)
{
	Vector_Parser_Data *vpd_p;
	Node *np;

	np = remHead( QS_VECTOR_PARSER_DATA_STACK(THIS_QSP) );
	assert(np!=NULL);

	vpd_p = NODE_DATA(np);
	rls_vector_parser_data(vpd_p);	// prevent leaks!

	addHead( QS_VECTOR_PARSER_DATA_FREELIST(THIS_QSP), np );

	np = QLIST_HEAD( QS_VECTOR_PARSER_DATA_STACK(THIS_QSP) );
	if( np != NULL ){
		vpd_p = NODE_DATA(np);
	} else {
		vpd_p = NULL;
	}
	SET_QS_VECTOR_PARSER_DATA(THIS_QSP,vpd_p);
}

void _init_scalar_parser_data_at_idx(QSP_ARG_DECL  int idx)
{
	Scalar_Parser_Data *spd_p;

	spd_p = getbuf(sizeof(Scalar_Parser_Data));
	SET_QS_SCALAR_PARSER_DATA_AT_IDX(THIS_QSP,idx,spd_p);

	SET_SPD_EDEPTH(spd_p,-1);
	SET_SPD_WHICH_STR(spd_p,0);
	SET_SPD_IN_PEXPR(spd_p,0);
	SET_SPD_ORIGINAL_STRING(spd_p,NULL);
	SET_SPD_ESTRINGS_INITED(spd_p,0);
	SET_SPD_FREE_EXPR_NODE_LIST(spd_p,NULL);
	// set spd_expr_string's to NULL ???
}

// Initialize a Query_Stack

void init_query_stack(Query_Stack *qsp)
{
	int i;
	const char *save_name;
	int save_serial;

	assert( qsp != NULL );

	// THIS_QSP is referenced in init_scalar_parser_data, which is
	// called below, and that evaluates to default_qsp in single-thread
	// compilation
	if( default_qsp == NULL )
		default_qsp = qsp;

	// This test is bad - the new memory may not have
	// been zeroed, so this may fail when there is
	// garbage in the structure...
	//if( QS_FLAGS(qsp) & QS_INITED ) return;

	// Should set everything to zero first...
	// But we have to restore the name pointer and serial number
	save_name = QS_NAME(qsp);
	save_serial = _QS_SERIAL(qsp);
	memset(qsp,0,sizeof(*qsp));
	SET_QS_NAME(qsp,save_name);
	SET_QS_SERIAL(qsp,save_serial);


	SET_QS_FLAGS(qsp,
			  QS_INITED
			| QS_EXPAND_MACS
			| QS_INTERACTIVE_TTYS
			| QS_FORMAT_PROMPT
			| QS_LOOKAHEAD_ENABLED
			| QS_STRIPPING_QUOTES
			| QS_COMPLETING
			/* | QS_PROCESSING_CALLBACKS */	// set only when a function is queued
			);

	SET_QS_QRY_STACK(qsp,new_stack());
	SET_QS_MENU_STACK(qsp,new_stack());
	SET_QS_LEVEL(qsp,(-1));
	SET_QS_STOP_LEVEL(qsp,0);
	SET_QS_STOP_LEVEL_STACK(qsp,NULL);

	for(i=0;i<MAX_VAR_BUFS;i++){
		SET_QS_VAR_BUF(qsp,i,new_stringbuf());
	}

	SET_QS_SCRATCH(qsp,new_stringbuf());
	SET_QS_RESULT(qsp,new_stringbuf());

	SET_QS_CMD_PROMPT_SB(qsp,new_stringbuf());
	SET_QS_QRY_PROMPT_SB(qsp,new_stringbuf());

	//SET_QS_FMT_CODE(qsp, FMT_DECIMAL);

	init_vector_parser_data_stack(qsp);

	for(i=0;i<MAX_SCALAR_PARSER_CALL_DEPTH;i++){
		SET_QS_SCALAR_PARSER_DATA_AT_IDX(qsp,i,NULL);
	}
	init_scalar_parser_data_at_idx(0);

	SET_QS_SCALAR_PARSER_CALL_DEPTH(qsp,-1);

	SET_QS_CHEW_LIST(qsp, NULL);
	SET_QS_CALLBACK_LIST(qsp, NULL);

	SET_QS_VAR_FMT_STACK(qsp,NULL);
	SET_QS_NUMBER_FMT(qsp,NULL);

	SET_QS_EXPECTED_WARNING_LIST(qsp,NULL);

	SET_QS_AV_STRINGBUF(qsp,NULL);

	SET_QS_DOBJ_ASCII_INFO(qsp,NULL);
	SET_QS_PICKING_ITEM_ITP(qsp,NULL);

	/* used to initialize query structs here, now do as needed */

#ifdef USE_QS_QUEUE
	dispatch_queue_t queue;

	if( QS_SERIAL(qsp) == 0 ){	// first one
		SET_QS_QUEUE(qsp,dispatch_get_main_queue());
	} else {
		sprintf(ERROR_STRING,"query_queue.%d",QS_SERIAL(qsp));
		queue = dispatch_queue_create(ERROR_STRING, NULL);
		// Check for error
		SET_QS_QUEUE(qsp,queue);
	}
#endif /* USE_QS_QUEUE */
}

void _push_stop_level(QSP_ARG_DECL  int l)
{
	if( QS_STOP_LEVEL_STACK(THIS_QSP) == NULL )
		SET_QS_STOP_LEVEL_STACK(THIS_QSP,new_stack());

	push_item( QS_STOP_LEVEL_STACK(THIS_QSP), (void *) ((long)Q_STOP_LEVEL) );

	SET_QS_STOP_LEVEL(THIS_QSP,l);
}

int _pop_stop_level(SINGLE_QSP_ARG_DECL)
{
	void *vp;

	assert( QS_STOP_LEVEL_STACK(THIS_QSP) != NULL );
	assert( stack_size(QS_STOP_LEVEL_STACK(THIS_QSP)) > 0 );

	vp = pop_item(QS_STOP_LEVEL_STACK(THIS_QSP));
	return (int) ((long)vp);
}


// The filename is now part of the query struct, so we don't have a separate stack...

#ifdef THREAD_SAFE_QUERY

//char *qpfgets( TMP_QSP_ARG_DECL char *buf, int size, FILE *fp )
char *qpfgets( QSP_ARG_DECL void *buf, int size, void *fp )
{
	return( fgets(buf,size,(FILE *)fp) );
}
#endif /* THREAD_SAFE_QUERY */



// used to initialize line number with 0, but now we start at 1
// and increment when we see a newline char

#define SET_QUERY_DEFAULTS(qp)						\
									\
	SET_QRY_LINENO(qp,1);						\
	SET_QRY_LINES_READ(qp,1);					\
	SET_QRY_COUNT(qp,0);						\
	SET_QRY_FLAGS(qp,0);						\
	SET_QRY_DUPFILE(qp,NULL);					\
	SET_QRY_PIPE(qp,NULL);						\
	/*SET_QRY_TEXT_BUF(qp,NULL);*/					\
	SET_QRY_MACRO(qp,NULL);					\
	if( QRY_BUFFER(qp) == NULL ){					\
		SET_QRY_BUFFER(qp,new_stringbuf());			\
	}

/*
 * Read input from a file.
 * Create a new query structure for this file and push onto the top
 * of the query stack.
 */

void redir_with_flags(QSP_ARG_DECL FILE *fp, const char *filename, uint32_t flag_bits )
		/* file from which to redirect input */
{
	Query *qp;

	if( !(QS_FLAGS(THIS_QSP) & QS_INITED) ){
		init_query_stack(THIS_QSP);
	}

	// We used to check QLEVEL here against MAX_Q_LVLS,
	// because we used to have a fixed array of these things...

	qp = new_query();

	/* the stack is initialized in init_query_stack */
	SET_QRY_IDX(qp,eltcount(QS_QRY_STACK(THIS_QSP)));
	push_item( QS_QRY_STACK(THIS_QSP), qp );

	SET_QS_LEVEL(THIS_QSP,QS_LEVEL(THIS_QSP)+1);
	//qp=(CURR_QRY(THIS_QSP));

	SET_QUERY_DEFAULTS(qp)
	SET_QRY_FLAGS(qp,flag_bits);

	SET_QRY_FILE_PTR(qp,fp);
	SET_QRY_FILENAME(qp,filename);

	// fp used to be null for network port redirect, but now
	// we need it to hold a pointer to a port struct...
#ifdef HAVE_ISATTY
	if( ! QRY_IS_SOCKET(qp) ){
		/* We want to set the interactive flag if we have redirected
		 * to /dev/tty, but not if it is to a serial port slaved
		 * to another machine...  but in general we have no way
		 * of knowing whether it is a terminal or a machine at
		 * the other end of a serial line...  so we have a global
		 * flag in the query stack,
		 * which can be cleared from the serial menu.
		 */
		if( fp != NULL ){
			if( isatty( fileno(QRY_FILE_PTR(qp)) ) && isatty( fileno(stderr) ) &&
				(QS_FLAGS(THIS_QSP)&QS_INTERACTIVE_TTYS) ){
	
				SET_QRY_FLAG_BITS(qp, Q_INTERACTIVE);
			}
		} else {
			CLEAR_QRY_FLAG_BITS(qp, Q_INTERACTIVE);
		}
	}

#else /* ! HAVE_ISATTY */

	// what should the default assumption be???
	if( QLEVEL == 0 && has_stdin ) {
		SET_QRY_FLAG_BITS(qp, Q_INTERACTIVE);
	}

#endif /* ! HAVE_ISATTY */

	if( fp != NULL && ! QRY_IS_SOCKET(qp) )
		SET_QRY_READFUNC(qp, (READFUNC_CAST) FGETS );
	/* else set to a default?  BUG? */

} // redir_with_flags

void _redir(QSP_ARG_DECL FILE *fp, const char *filename)
{
	redir_with_flags(QSP_ARG  fp, filename, 0 );
}

/*
 *	Copy macro arguments up a level.
 *	Used in loops.
 *	This allows macro arguments to be referenced inside
 *	loops which occur in the macro.
 *
 *	They should not be free'd when the loop redirect finishes.
 */

static void share_macro_args(QSP_ARG_DECL Query *qpto,Query *qpfr)
{
	if( QRY_MACRO(qpfr) == NULL ){
		return;
	}

	SET_QRY_MACRO(qpto, QRY_MACRO(qpfr) );
	SET_QRY_ARGS(qpto, QRY_ARGS(qpfr) );
	SET_QRY_FLAG_BITS(qpto, Q_MPASSED);
}


/*
 * This function is useful when we want to specify "-" as the input file
 */

/* dup_input
 *
 * used for loops - we push a copy of the input on the input stack, and
 * continue processing...  when the loop is done, we pop, and decide whether
 * or not to re-push.  The first time through, all text read gets saved
 * at the next level down (?)
 *
 * We stop duping if the level is popped...
 */

static void dup_input(SINGLE_QSP_ARG_DECL)
{
	const char *s;
#ifdef QUIP_DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"dup_input:  current qlevel = %d, duping at %d",QLEVEL,QLEVEL+1);
advise(ERROR_STRING);
sprintf(ERROR_STRING,"dup_input:  current input file is %s",CURRENT_FILENAME);
advise(ERROR_STRING);
sprintf(ERROR_STRING,"q_file = 0x%lx\nq_readfunc = 0x%lx",
(u_long)QRY_FILE_PTR(CURR_QRY(THIS_QSP)),(u_long)QRY_READFUNC(CURR_QRY(THIS_QSP)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	//push_input_file( QSP_ARG CURRENT_FILENAME );
	s = CURRENT_FILENAME;
	redir( QRY_FILE_PTR(CURR_QRY(THIS_QSP)), s );

	/* these two lines are so we can have within-line loops */
	// Clear the direct-input flags!
	// why???
	// When we are interactively entering stuff, it is not displayed!?

	//SET_QRY_FLAGS( CURR_QRY(THIS_QSP),
	//	(QRY_FLAGS(QRY_AT_LEVEL(THIS_QSP,QLEVEL-1))) & Q_NON_INPUT_MASK );
	SET_QRY_FLAGS( CURR_QRY(THIS_QSP),
		(QRY_FLAGS(QRY_AT_LEVEL(THIS_QSP,QLEVEL-1)))
			/* & Q_NON_INPUT_MASK */ );

	SET_QRY_LINE_PTR( CURR_QRY(THIS_QSP),QRY_LINE_PTR(QRY_AT_LEVEL(THIS_QSP,QLEVEL-1)));
	SET_QRY_LINES_READ( CURR_QRY(THIS_QSP),QRY_LINES_READ(QRY_AT_LEVEL(THIS_QSP,QLEVEL-1)));

	/* the absence of the next line caused a subtle bug
	 * for loops within macros that were preceded by a doubly
	 * redirected file... */

	SET_QRY_READFUNC( CURR_QRY(THIS_QSP),QRY_READFUNC(QRY_AT_LEVEL(THIS_QSP,QLEVEL-1)));

	/* loops within macros */
	share_macro_args(QSP_ARG CURR_QRY(THIS_QSP),QRY_AT_LEVEL(THIS_QSP,QLEVEL-1));
} // end of dup_input

/* stuff for loops on input */

static void insure_query_text_buf(Query *qp)
{
	if( QRY_TEXT_BUF(qp) == NULL )
		SET_QRY_TEXT_BUF(qp,new_stringbuf());
	copy_string( QRY_TEXT_BUF(qp),"");	// clear any old contents
}

/*
 * Open input loop with count of n.
 *
 * Here is the strategy for loops:
 * we allocate a storage buffer when we open a loop.  We then save
 * all text AT THE CURRENT QLEVEL into that buffer,
 * and then redirect when the loop is closed.
 *
 * The flag to test for when saving is whether the count is > 0
 * and the q_file is not NULL
 */

void _open_loop(QSP_ARG_DECL int n)
			/* loop count */
{
	Query *qp;

	qp=(CURR_QRY(THIS_QSP));

	dup_input(SINGLE_QSP_ARG);

	SET_QRY_COUNT(qp,n);
	insure_query_text_buf(qp);
	SET_QRY_FLAG_BITS(qp,Q_SAVING);
}

void _foreach_loop(QSP_ARG_DECL Foreach_Loop *frp)
{
	Query *qp;

	qp=(CURR_QRY(THIS_QSP));

	dup_input(SINGLE_QSP_ARG);

#define FORELOOP	(-2)

	assign_var(FL_VARNAME(frp),(const char *)NODE_DATA(QLIST_HEAD(FL_LIST(frp))));

	SET_QRY_COUNT(qp, FORELOOP);		/* BUG should be some unique code */
	SET_QRY_FORLOOP(qp, frp);
	insure_query_text_buf(qp);
	SET_QRY_FLAG_BITS(qp,Q_SAVING);
}

void _zap_fore(QSP_ARG_DECL  Foreach_Loop *frp)
{
	Node *np;

	while( (np=remTail(FL_LIST(frp))) != NULL ){
		rls_str((const char *)NODE_DATA(np));
		rls_node(np);
	}
	rls_list(FL_LIST(frp));
	rls_str(FL_VARNAME(frp));
	givbuf(frp);
}

//#ifdef BUILD_FOR_OBJC
//#define READING_FROM_TERMINAL(qp)	1
//#else // ! BUILD_FOR_OBJC
//#define READING_FROM_TERMINAL(qp)	( QRY_FILE_PTR(qp) == tfile() )
//#endif // ! BUILD_FOR_OBJC

#ifdef BUILD_FOR_CMD_LINE
#define READING_FROM_TERMINAL(qp)	( QRY_FILE_PTR(qp) == tfile() )
#else // ! BUILD_FOR_CMD_LINE
#define READING_FROM_TERMINAL(qp)	1
#endif // ! BUILD_FOR_CMD_LINE

static inline void close_query_file(QSP_ARG_DECL  Query *qp)
{
	if( ! QRY_HAS_FILE_PTR(qp) ) return;

#ifdef HAVE_POPEN
	if( IS_PIPE(qp) ){
		assert(QRY_PIPE(qp)!=NULL);
		close_pipe(QSP_ARG  QRY_PIPE(qp));
		return;
	}
#endif /* HAVE_POPEN */

	/* before we close the file, make sure it
	 * isn't a dup of a lower file for looping input
	 */
	if( QLEVEL == 0 || QRY_FILE_PTR(qp) != QRY_FILE_PTR(PREV_QRY(THIS_QSP)) ){
		if( ! READING_FROM_TERMINAL(qp) )
			fclose(QRY_FILE_PTR(qp));
	}
}

static inline void release_macro_args(Query *qp)
{
	int i;

	if( MACRO_N_ARGS(QRY_MACRO(qp)) > 0 ){
		for(i=0;i<MACRO_N_ARGS(QRY_MACRO(qp));i++){
			rls_str(QRY_ARG_AT_IDX(qp,i));
		}
		givbuf((char *)QRY_ARGS(qp));
		SET_QRY_ARGS(qp,NULL);
	}
}

/*
 * Close and pop redir input file.
 * Cleans up appropriately.
 */

Query * _pop_file(SINGLE_QSP_ARG_DECL)
{
	Query *qp;

	if( QLEVEL<0 ){
		warn("pop_file:  no file to pop");
		return NULL;
	}

	qp = CURR_QRY(THIS_QSP);

	close_query_file(QSP_ARG  qp);
	SET_QRY_FILE_PTR(qp,NULL);

	pop_item( QS_QRY_STACK(THIS_QSP) );

	/* IF macro open && not a loop in a macro */
	if( (QRY_MACRO(qp) != NULL) && NOT_PASSED(qp) )
		release_macro_args(qp);		/* free macro args if any */

	rls_query(qp);	// add to query free list

	QLEVEL--;

	return(qp);	// we may want to access the flags of the old thing...
			// But where do we release resources?

} /* end pop_file() */

//char *poptext(TMP_QSP_ARG_DECL  char *buf,int size,FILE* fp)
static char *poptext(QSP_ARG_DECL  void *buf,int size,void* fp)
{
	return(NULL);
}

#ifdef QUIP_DEBUG
static inline void debug_pushtext(QSP_ARG_DECL  const char *text)
{
	if( debug & qldebug ){
		if( strlen(text) < LLEN-80 ){
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  pushing at level %d, text 0x%lx  \"%s\"",
				WHENCE_L(push_text),
				QLEVEL,(u_long)text,text);
		} else {
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  pushing at level %d, text 0x%lx  (%lu chars)",
				WHENCE_L(push_text),
				QLEVEL,(u_long)text,(long)strlen(text));
		}
		advise(ERROR_STRING);
	}
}

#define DEBUG_PUSHTEXT		debug_pushtext(QSP_ARG  text);

#else /* ! QUIP_DEBUG */

#define DEBUG_PUSHTEXT

#endif /* ! QUIP_DEBUG */

/*
 * Push text onto the input.
 * Scan text as if it were keyboard input.
 *
 * We don't copy the text - should we?
 *
 * We don't have to as long as all we are going to do is read it...
 *
 * We push the text onto the line buffer, even though it
 * might be many lines.  That didn't cause any trouble
 * until we tried to read a macro definition from an
 * encrypted file...
 */

void _push_text(QSP_ARG_DECL const char *text, const char *filename)
{
	Query *qp, *old_qp;

//QUEUE_CHECK(push_text)
	if( QLEVEL >= 0 )
		old_qp=(CURR_QRY(THIS_QSP));
	else
		old_qp = NULL;

	redir((FILE *)NULL, filename );
	qp=(CURR_QRY(THIS_QSP));
	SET_QRY_LINE_PTR(qp,text);
	SET_QRY_FLAG_BITS(qp,(Q_HAS_SOMETHING | Q_BUFFERED_TEXT));

	if( old_qp == NULL ){
		SET_QRY_LINENO(qp, 1 );
		SET_QRY_LINES_READ(qp, 1 );
	} else {
		SET_QRY_LINES_READ(qp, QRY_LINES_READ(old_qp) );
		SET_QRY_LINENO(qp, QRY_LINENO(old_qp) );
		// Not exactly right, but close?
	}

	DEBUG_PUSHTEXT

	SET_QRY_READFUNC(qp,poptext);

}

/* fullpush(text)		just like push_text, but also passes up macro args */

static void fullpush(QSP_ARG_DECL const char *text, const char *filename)
{
	//Query *qp;

	/* push text & carry over macro args. */

	//qp=(CURR_QRY(THIS_QSP));
#ifdef QUIP_DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  level %d, text \"%s\"",
WHENCE_L(fullpush),QLEVEL + 1,text);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	push_text(text, filename);
	share_macro_args(QSP_ARG CURR_QRY(THIS_QSP),QRY_AT_LEVEL(THIS_QSP,QLEVEL-1));	// fullpush
}

void _close_loop(SINGLE_QSP_ARG_DECL)
{
	Query *qp;
	Query *loop_qp;
	const char *errmsg="Can't close loop, no loop open";
	const char *s;

	if( QLEVEL <= 0 || QRY_COUNT(QRY_AT_LEVEL(THIS_QSP,QLEVEL-1)) == 0 ){
		warn(errmsg);
		return;
	}

	/* the lookahead word may have popped the level already...
	 * How would we know???
	 */

	loop_qp=pop_file();	// are we sure we should do this?

	qp=(CURR_QRY(THIS_QSP));

	CLEAR_QRY_FLAG_BITS(qp,Q_SAVING);

	if( QRY_COUNT(qp) == FORELOOP ){
		FL_NODE(QRY_FORLOOP(qp)) = NODE_NEXT(FL_NODE(QRY_FORLOOP(qp)));
		if( FL_NODE(QRY_FORLOOP(qp)) == NULL ){
			zap_fore(QRY_FORLOOP(qp));	// this releases the foreach struct also
			SET_QRY_FORLOOP(qp,NULL);
			goto lup_dun;
		}
		assign_var(FL_VARNAME(QRY_FORLOOP(qp)),
			(const char *)FL_WORD(QRY_FORLOOP(qp)) );

	} else if( QRY_COUNT(qp) < 0 ){		/* do/while loop */

		// don't have to do anything ...

	} else {		// regular repeat loop

		// decrement the count
		SET_QRY_COUNT( qp, QRY_COUNT(qp)-1 );
		if( QRY_COUNT(qp) <= 0 ) goto lup_dun;
	}

	//push_input_file( QSP_ARG CURRENT_FILENAME );
	// BUG need to pass the filename up!?
	s=CURRENT_FILENAME;

	assert(QRY_TEXT_BUF(qp) != NULL );
	// does fullpush push the macro pointer?
	fullpush(QSP_ARG  sb_buffer(QRY_TEXT_BUF(qp)), s );

	/* This is right if we haven't finished the current line yet... */
	SET_QRY_LINES_READ(CURR_QRY(THIS_QSP),QRY_LINES_READ(qp));
	if( QRY_FLAGS(qp) & Q_LINEDONE ){
		INCREMENT_LINES_READ(close_loop)
	}

	return;

lup_dun:

//advise("close_loop:  loop done...");
//qdump(qsp);

	// Now we are done with the loop
	// - why is the line number wrong??
	// Is it?

	// set text to null string...
	clear_query_text(qp);

	SET_QRY_LINES_READ(qp, QRY_LINES_READ(loop_qp));

	/* lookahead may have been inhibited by q_count==1 */
	lookahead(SINGLE_QSP_ARG);

//advise("close_loop:  loop DONE...");
//qdump(qsp);

}



/* "while" can be used instead of "end",
 * but first we need to make sure that a loop is open!
 */

void _whileloop(QSP_ARG_DECL  int value)
{

	Query *qp;
	const char *errmsg="Can't close with While, no loop open";

	assert( QLEVEL >= 0 );

	if( QLEVEL == 0 || QRY_COUNT(QRY_AT_LEVEL(THIS_QSP,QLEVEL-1)) == 0 ){
		warn(errmsg);
		return;
	}

	pop_file();
	qp=(CURR_QRY(THIS_QSP));
	CLEAR_QRY_FLAG_BITS(qp,Q_SAVING);

	if( ! value ){
		clear_query_text(qp);
		SET_QRY_COUNT(qp, 0);	// do loop sets count to -1
	} else {
		assert( QRY_TEXT_BUF(qp) != NULL );
		fullpush(QSP_ARG  sb_buffer(QRY_TEXT_BUF(qp)), CURRENT_FILENAME );
	}
}


/*
 * We remember the if-clause string so we can free
 * it at the right time
 */

void _push_if(QSP_ARG_DECL const char *text)
{
	//Query *qp;

	fullpush(QSP_ARG  text, CURRENT_FILENAME );
	//qp=(CURR_QRY(THIS_QSP));
	// BUG?? should we set the line number too?
}

/*
 * Tell the interpreter about the command line arguments.
 * Usually the first call from main().
 */

void set_args(QSP_ARG_DECL  int ac,char** av)
		/* ac = number of arguments */
		/* av = pointer to arg strings */
{
	int i;
	char acstr[32];

	sprintf(acstr,"%d",ac-1);  /* don't count command name */
	create_reserved_var("argc",acstr);

	for(i=0;i<ac;i++){
		/* allow the cmd args to be referenced $argv1 etc, because $1 $2 don't work inside macros.
		 * BUG - we really ought to copy the shell and allow variable subscripting:  $argv[1]
		 */
		sprintf(acstr,"argv%d",i);
		create_reserved_var(acstr,av[i]);

		n_cmd_args++;
	}

	set_progname(av[0]);
}

void inhibit_next_prompt_format(SINGLE_QSP_ARG_DECL)
{
	CLEAR_QS_FLAG_BITS(THIS_QSP, QS_FORMAT_PROMPT);
}

void enable_prompt_format(SINGLE_QSP_ARG_DECL)
{
	SET_QS_FLAG_BITS(THIS_QSP, QS_FORMAT_PROMPT);
}

static void show_query_level(QSP_ARG_DECL int i)
{
	Query *qp;

	if( i< 0 || i > QLEVEL ){
		sprintf(ERROR_STRING,"Requested level %d out of range 0-%d",
			i,QLEVEL);
		warn(ERROR_STRING);
		return;
	}

	qp = QRY_AT_LEVEL(THIS_QSP,i);

	show_query_flags(QSP_ARG  qp);

	if( QRY_HAS_TEXT(qp) ){
		sprintf(ERROR_STRING,
			"\tLevel %d line buffer at 0x%lx:\n\"%s\"",i,(u_long)QRY_LINE_PTR(qp),QRY_LINE_PTR(qp));
		advise(ERROR_STRING);
	}

	if( QRY_TEXT_BUF(qp) != NULL ){
		sprintf(ERROR_STRING,
			"\tstored text:\n\"%s\"",sb_buffer(QRY_TEXT_BUF(qp)));
		advise(ERROR_STRING);
	}

	if( QRY_IS_SAVING(qp) )
		advise("\tSaving...");
	sprintf(ERROR_STRING,
		"\tCount = %d",QRY_COUNT(qp) );
	advise(ERROR_STRING);

	if( QRY_MACRO(qp) != NULL ){
		sprintf(ERROR_STRING,
			"\tIn macro \"%s\"",MACRO_NAME(QRY_MACRO(qp)));
		advise(ERROR_STRING);
	}
	if( QRY_HAS_FILE_PTR(qp) ){
		sprintf(ERROR_STRING,
			"\tFile pointer = 0x%lx",(long)QRY_FILE_PTR(qp) );
		advise(ERROR_STRING);
	}
	if( QRY_FILENAME(qp) != NULL ){
		sprintf(ERROR_STRING,
			"\tFile name = %s",QRY_FILENAME(qp) );
		advise(ERROR_STRING);
	}

}

/* Show state for debugging */
void qdump(SINGLE_QSP_ARG_DECL)
{
	int i;

	advise("");
	sprintf(ERROR_STRING,"Query stack %s:",QS_NAME(THIS_QSP));
	advise(ERROR_STRING);
	for( i=QLEVEL; i>= 0 ; i-- ){
		sprintf(ERROR_STRING,"Input level %d (%s):",i,QRY_FILENAME(QRY_AT_LEVEL(THIS_QSP,i)));
		advise(ERROR_STRING);
		show_query_level(QSP_ARG  i);
		advise("\n");
	}
#ifdef MALLOC_DEBUG
	sprintf(ERROR_STRING,"checking heap...");
	advise(ERROR_STRING);

	if( malloc_verify() ){
		sprintf(ERROR_STRING,"  OK");
		advise(ERROR_STRING);
	} else {
		sprintf(ERROR_STRING,"\nCORRUPTED HEAP DETECTED!!!");
		warn(ERROR_STRING);
	}
#endif /* MALLOC_DEBUG */
}

#ifdef NOT_USED
void set_qflags(QSP_ARG_DECL int flag)
{
	SET_QRY_FLAG_BITS( CURR_QRY(THIS_QSP), flag );
}
#endif /* NOT_USED */

//#ifndef BUILD_FOR_OBJC
#ifdef BUILD_FOR_CMD_LINE

/**/
/**		stuff to do with the control terminal	**/
/**/

#define NO_TTY ((FILE *)(-44))	// why not NULL???

/*
 * Return a descriptor for the control tty
 */

static FILE *ttyfile=NO_TTY;

FILE *_tfile(SINGLE_QSP_ARG_DECL)
{
	char ttn[24];

	if( ttyfile != NO_TTY ){
		return(ttyfile);
	}

	if( IS_INTERACTIVE(CURR_QRY(THIS_QSP)) ){
		return(QRY_FILE_PTR(CURR_QRY(THIS_QSP)));
	}

	/*
	if( !isatty( fileno(stderr) ) ) ERROR1("stderr not a tty");
	strcpy(ttn,ttyname(fileno(stderr)));
	*/

	strcpy(ttn,"/dev/tty");
	ttyfile=fopen(ttn,"r");
	if( (!ttyfile) && verbose ){
		sprintf(ERROR_STRING,"tfile:  can't open control tty: %s",ttn);
		warn(ERROR_STRING);
	}

	return(ttyfile);
}
//#endif // ! BUILD_FOR_OBJC
#endif // BUILD_FOR_CMD_LINE

/*
 * Return a pointer to the named variable or NULL.
 * Works for macro args, i.e. $1 $2, by using a query_stack variable
 * tmpvar.
 *
 * This also works for the cmd line args if we are not in a macro.
 *
 * which function used to be called simple_var_of?  var__of ?
 */

Variable *_var_of(QSP_ARG_DECL const char *name)
		/* variable name */
{
	int i;
	Variable *vp;
	const char *s;

	vp = var__of(name);
	if( vp != NULL ) return(vp);

	/* if not set, try to import from env */
	s = getenv(name);
	if( s != NULL ){
		vp = new_var_(name);
		SET_VAR_VALUE(vp,savestr(s));
		SET_VAR_FLAGS(vp,VAR_RESERVED);
		return(vp);
	}

	/* numbered variable? (macro arg. or cmd line arg.) */

	i=0;
	s=name;
	while( *s ){
		if( !isdigit(*s) ) return(NULL);
		i*=10;
		i+=(*s)-'0';
		s++;
	}

	i--;	/* variables start at 1, indices at 0 */

	/* first see if we're in a macro! */
	if( QRY_MACRO(CURR_QRY(THIS_QSP)) != NULL ){
		/*
		 * range checking is done in getmarg(),
		 * which returns NULL if out of range.
		 * It used to return the null string, but
		 * this was changed so that the null string
		 * could be a valid argument.
		 */
		const char *v;
		//Variable *vp;
		v = getmarg(QSP_ARG  i);
		if( v == NULL ) return(NULL);

#ifdef QUIP_DEBUG
if( debug & qldebug ){
if( strlen(v) < LLEN-80 ){
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  found macro arg with value \"%s\" at 0x%lx",
WHENCE_L(var_of),
v,
(long)v
);
} else {
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  found macro arg with %lu chars at 0x%lx",
WHENCE_L(var_of),
(long)strlen(v),
(long)v
);
}
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

		// Now we need a temporary variable;
		// We don't want to put this in the database,
		// because we could have a $1 at every level...
		if( (vp=QS_TMPVAR(THIS_QSP)) == NULL ){
			NEW_VARIABLE(vp)
			SET_QS_TMPVAR(THIS_QSP, vp );
		}
		SET_VAR_NAME(vp,name);
		SET_VAR_VALUE(vp,v);
		SET_VAR_FLAGS(vp,VAR_RESERVED);
		//TMPVAR.v_func = NULL;
		return(vp);
	} else {	/* command line arg? */
		/*
		 * A subtle BUG?...  Should macro args
		 * be passed to tty redirects from
		 * within a macro?
		 * Here they are not...
		 */
		if( i >= n_cmd_args ){
			sprintf(ERROR_STRING,
"variable index %d too high; only %d command line arguments\n",
			i+1,n_cmd_args);
			warn(ERROR_STRING);
			return(NULL);
		} else {
			char varname[32];
			sprintf(varname,"argv%d",i+1);
			vp = var__of(varname);
			return(vp);
		}
	}
	/* this suppresses a lint error message about no return val... */
	/* NOTREACHED */
	/* return(NULL); */
} /* end var_of */

#ifdef HAVE_ROUND
#define round_func	round
#elif HAVE_RINT
#define round_func	rint

#elif HAVE_FLOOR

#define round_func	my_round

double my_round( double n )
{
	// BUG?  the behavior of this may not match the real round(),
	// especially for negative numbers???
	return floor( n + 0.5 );
}
#else // ! HAVE_FLOOR

#error 'Neither round nor rint nor floor is present, no rounding function.'

#endif // ! HAVE_RINT && ! HAVE_ROUND && ! HAVE_FLOOR

void push_top_menu(SINGLE_QSP_ARG_DECL)
{
	push_menu( BOTTOM_OF_STACK( QS_MENU_STACK(THIS_QSP) ) );
}


const char *save_possibly_empty_str(const char *s)
{
	char *new_s;
	
	assert(s!=NULL);
	new_s = getbuf(strlen(s)+1);
	strcpy(new_s,s);
	return(new_s);
}

const char *savestr(const char *s)
{
	char *new_s;
	
	assert(s!=NULL);
	assert(*s!=0);
	new_s = getbuf(strlen(s)+1);
	strcpy(new_s,s);
	return(new_s);
}

void rls_str(const char *s)
{
	givbuf((void *)s);
}

#ifdef FOOBAR

/* Do we not need these? */

void callbacks_on(SINGLE_QSP_ARG_DECL)
{
	if( IS_PROCESSING_CALLBACKS(THIS_QSP) )
		warn("callbacks_on:  callbacks are already being processed");
	SET_QS_FLAG_BITS(THIS_QSP,QS_PROCESSING_CALLBACKS);
}

void callbacks_off(SINGLE_QSP_ARG_DECL)
{
	if( ! IS_PROCESSING_CALLBACKS(THIS_QSP) )
		warn("callbacks_off:  callbacks are already being inhibited");
	CLEAR_QS_FLAG_BITS(THIS_QSP,QS_PROCESSING_CALLBACKS);
}
#endif /* FOOBAR */

static void add_func_to_list(List *lp,void (*func)(SINGLE_QSP_ARG_DECL) )
{
	Node *np;

	/* BUG? without the cast, gcc reports that assigning a function pointer
	 * to void * is a violation of ANSI - we put the cast in to
	 * eliminate the warning, but will this cause a problem on
	 * some systems?  Why the ANSI prohibition??
	 */

	/* We don't check to see whether or not this function is on the list
	 * already - we therefore need to make sure that we only call this
	 * once for each function!
	 */
#ifdef CAUTIOUS
	np = QLIST_HEAD(lp);
	while(np!=NULL){
		assert( func != ((void (*)(SINGLE_QSP_ARG_DECL))NODE_DATA(np)) );
		np = NODE_NEXT(np);
	}
#endif // CAUTIOUS

	np = mk_node((void *) func);
	addTail(lp,np);
}


void _add_event_func(QSP_ARG_DECL  void (*func)(SINGLE_QSP_ARG_DECL) )	/** set event processing function */
{
	if( QS_EVENT_LIST(THIS_QSP) == NULL ){
		SET_QS_EVENT_LIST(THIS_QSP, new_list() );
	}
	add_func_to_list(QS_EVENT_LIST(THIS_QSP),func);
}

int _rem_event_func(QSP_ARG_DECL  void (*func)(SINGLE_QSP_ARG_DECL) )
{
	Node *np;
	np = remData(QS_EVENT_LIST(THIS_QSP),(void *)func);
	if( np != NULL ){
		rls_node(np);
		return(0);
	} else {
		return(-1);
	}
}

void add_cmd_callback(QSP_ARG_DECL  void (*func)(SINGLE_QSP_ARG_DECL) )
{
	if( QS_CALLBACK_LIST(THIS_QSP) == NULL ){
		SET_QS_CALLBACK_LIST(THIS_QSP, new_list() );
	}

	add_func_to_list(QS_CALLBACK_LIST(THIS_QSP), func);

	// Make sure the flag is set
	SET_QS_FLAG_BITS(THIS_QSP,QS_PROCESSING_CALLBACKS);
}

static void show_margs(QSP_ARG_DECL  Macro *mp)
{
	int i;

	for(i=0;i<MACRO_N_ARGS(mp);i++) {
		sprintf(msg_str,"\t$%d:  %s",i+1,MACRO_PROMPT(mp,i));
		prt_msg(msg_str);
	}
}

static void show_mflags(QSP_ARG_DECL  Macro *mp)
{
	if( RECURSION_FORBIDDEN(mp) )
		sprintf(msg_str,"\trecursion forbidden");
	else
		sprintf(msg_str,"\trecursion allowed");
	prt_msg(msg_str);
}

void macro_info(QSP_ARG_DECL  Macro *mp)
{
	sprintf(msg_str,"Macro \"%s\" (file \"%s\", line %d)",
		MACRO_NAME(mp),MACRO_FILENAME(mp),MACRO_LINENO(mp));
	prt_msg(msg_str);

	show_margs(QSP_ARG  mp);
	show_mflags(QSP_ARG  mp);
	prt_msg("\n");
}

static void macro_prolog(QSP_ARG_DECL  Macro *mp)
{
	int i;

	sprintf(msg_str,"Define %s %d", MACRO_NAME(mp),MACRO_N_ARGS(mp));
	prt_msg(msg_str);

	for(i=0;i<MACRO_N_ARGS(mp);i++) {
		sprintf(msg_str,"\t'%s'",MACRO_PROMPT(mp,i));
		prt_msg(msg_str);
	}
}

void show_macro(QSP_ARG_DECL  Macro *mp)		/** show macro text */
{
	macro_info(QSP_ARG  mp);
	prt_msg(MACRO_TEXT(mp));
}

void dump_macro(QSP_ARG_DECL  Macro *mp)		/** show macro text */
{
	macro_prolog(QSP_ARG  mp);
	prt_msg_frag(MACRO_TEXT(mp));
	prt_msg(".");
}

Query *query_at_level(QSP_ARG_DECL  int l)
{
	Node *np;
	// We add levels at the head of the list,
	// So what we need here is nth_elt_from_end...

#ifdef FOOBAR
	int n;
	
	n = eltcount( QS_QRY_STACK(THIS_QSP) );
	if( n == 0 ) return NULL;
	np = nth_elt( QS_QRY_STACK(THIS_QSP), n-(1+l) );
#endif // FOOBAR
	np = nth_elt_from_tail( QS_QRY_STACK(THIS_QSP), l );


	if( np == NULL ) return NULL;
	return (Query *)NODE_DATA(np);
}

COMMAND_FUNC( do_pop_menu )
{
	pop_menu();
}

// This message doesn't match the function name, but it is only called from macro creation...
// BUG - we need a better way of checking on a per-command basis...

inline int _check_adequate_return_strings(QSP_ARG_DECL  int n)
{
	if( n > N_QRY_RETSTRS ){
		sprintf(ERROR_STRING,"%d return strings needed, system limit is %d",
			n,N_QRY_RETSTRS);
		advise(ERROR_STRING);
		return -1;
	}
	return 0;
}

inline int current_line_number(SINGLE_QSP_ARG_DECL)
{
	return QRY_LINENO(CURR_QRY(THIS_QSP));
}

void exit_current_file(SINGLE_QSP_ARG_DECL)
{
	int i,done_level;
	i=QLEVEL;
	done_level=(-1);	// pointless initialization to quiet compiler
	while( i >= 0 ){
		if( QRY_READFUNC(QRY_AT_LEVEL(THIS_QSP,i)) == ((READFUNC_CAST) FGETS) ){
			done_level=i;
			i = -1;
		}
		i--;
	}
	if( done_level < 0 ){
		warn("exit_file:  no file to exit!?");
		return;
	}
	i=QLEVEL;
	while(i>=done_level){
		pop_file();
		i--;
	}
}

static int macro_exit_level(SINGLE_QSP_ARG_DECL)
{
	int i,done_level;
	Macro *mp;

	done_level=(-1);	// pointless initialization to quiet compiler
	i=QLEVEL;
	mp = NULL;
	while( i >= 0 ){
		if( mp != NULL ){
			if( QRY_MACRO(QRY_AT_LEVEL(THIS_QSP,i)) != mp ){
				done_level=i;
				i = -1;
			}
			// We need another test here to see if we are
			// in a different invocation of a recursive macro!?
		} else if( QRY_MACRO(QRY_AT_LEVEL(THIS_QSP,i)) != NULL ){
			/* There is a macro to pop... */
			mp = QRY_MACRO(QRY_AT_LEVEL(THIS_QSP,i));
		}
		i--;
	}
	if( mp == NULL ){
		warn("exit_macro:  no macro to exit!?");
		return -1;
	}
	assert( done_level != (-1) );
	return done_level;
}

void exit_current_macro(SINGLE_QSP_ARG_DECL)
{
	int i,done_level;

	done_level = macro_exit_level(SINGLE_QSP_ARG);
	if( done_level < 0 ) return;

	i=QLEVEL;
	while(i>done_level){
		pop_file();
		i--;
	}
}

inline const char *query_filename(SINGLE_QSP_ARG_DECL)
{
	return QRY_FILENAME( CURR_QRY(THIS_QSP) );
}

// Make a table of the unique levels.
// n_ptr is a return argument.
// The table has to be freed by the caller!!!

int *get_levels_to_print(QSP_ARG_DECL  int *n_ptr)
{
	int max_levels_to_print;
	int *level_tbl;
	int ql;
	int n_levels_to_print;
	int i;
	const char *filename;

	ql = QLEVEL;
	max_levels_to_print = ql + 1;
	level_tbl = getbuf( max_levels_to_print * sizeof(*level_tbl) );

	// We would like to print the macro names with the deepest one
	// last, but for cases where the macro is repeated (e.g. loops)
	// we only want to print the deepest case.
	// That makes things tricky, because we need to scan
	// from deepest to shallowest, but we want to print
	// in the reverse order...
	n_levels_to_print=1;
	level_tbl[0]=ql;
	ql--;	// it looks like this line could be deleted...
	//i = THIS_QSP->qs_fn_depth;
	i=QLEVEL;
	i--;
	// When we have a loop, the same input gets duplicated;
	// We don't want to print this twice, so we make an array of which
	// things to print.
	filename=query_filename(SINGLE_QSP_ARG);
	while( i >= 0 ){
		Query *qp;

		qp=QRY_AT_LEVEL(THIS_QSP,i);
		if( strcmp( QRY_FILENAME(qp),filename) ){
			level_tbl[n_levels_to_print] = i;
			filename=QRY_FILENAME(qp);
			n_levels_to_print++;
		}
		i--;
	}
	*n_ptr = n_levels_to_print;
	return level_tbl;
}

#ifdef BUILD_FOR_IOS

/* On iOS, the filenames start with the full bundle path,
 * which is kind of long and clutters the screen...
 */

#define ADJUST_PATH(pathname)						\
									\
	{								\
		int _j=(int)strlen(pathname)-1;				\
		while(_j>0 && pathname[_j]!='/')			\
			_j--;						\
		if( pathname[_j] == '/' ) _j++;				\
		pathname += _j;						\
	}

#else /* ! BUILD_FOR_IOS */

#define ADJUST_PATH(pathname)

#endif /* ! BUILD_FOR_IOS */

static void tell_macro_location(QSP_ARG_DECL  const char *location_string, int n)
{
	const char *mname;
	Macro *mp;
	const char *filename;	// macro file name
	mname = location_string+strlen(MACRO_LOCATION_PREFIX);
	// don't use get_macro, because it prints a warning,
	// causing infinite regress!?
	mp = macro_of(mname);
	assert( mp != NULL );

	filename = MACRO_FILENAME(mp);
	ADJUST_PATH(filename);
	sprintf(MSG_STR,"%s line %d (File %s, line %d):",
		mname, n, filename, MACRO_LINENO(mp)+n);
	advise(MSG_STR);
}

void print_qs_levels(QSP_ARG_DECL  int *level_to_print, int n_levels_to_print)
{
	int i;
	int ql,n;
	const char *filename;
	char msg[LLEN];

	i=n_levels_to_print-1;
	while(i>=0){
		ql=level_to_print[i];	// assume ql matches fn_level?
		//assert( ql >= 0 && ql < MAX_Q_LVLS );
		//filename=THIS_QSP->qs_fn_stack[ql];
		filename=QRY_FILENAME(QRY_AT_LEVEL(THIS_QSP,ql));
		n = QRY_LINENO(QRY_AT_LEVEL(THIS_QSP,ql) );
		if( !strncmp(filename,MACRO_LOCATION_PREFIX,6) ){
			tell_macro_location(QSP_ARG  filename, n);
		} else {
			ADJUST_PATH(filename);
			sprintf(msg,"%s (input level %d), line %d:",
				filename,ql,n);
			advise(msg);
		}
		i--;
	}
}

inline const char *current_filename(SINGLE_QSP_ARG_DECL)
{
	return QRY_FILENAME( CURR_QRY(THIS_QSP) );
}

inline void _reset_return_strings(SINGLE_QSP_ARG_DECL)
{
	SET_QRY_RETSTR_IDX(CURR_QRY(THIS_QSP),0);
}

#ifdef HAVE_POPEN
void redir_from_pipe(QSP_ARG_DECL  Pipe *pp, const char *cmd)
{
	redir(pp->p_fp, cmd);
	SET_QRY_PIPE( CURR_QRY(THIS_QSP) , pp );
	SET_QRY_FLAG_BITS(CURR_QRY(THIS_QSP), Q_PIPE);
}
#endif // HAVE_POPEN

inline void set_query_filename(Query *qp, const char *filename)
{
	SET_QRY_FILENAME(qp,filename);
}

inline void set_query_arg_at_index(Query *qp, int idx, const char *s)
{
	SET_QRY_ARG_AT_IDX(qp,idx,s);
}

inline void set_query_args(Query *qp, const char **args)
{
	SET_QRY_ARGS(qp,args);
}

inline void set_query_macro(Query *qp, Macro *mp)
{
	SET_QRY_MACRO(qp,mp);
}

