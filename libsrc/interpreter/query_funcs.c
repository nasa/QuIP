#include "quip_config.h"

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
 * file on the input stack.  Having lookahead on, however,
 * screws up line numbering, because the line number gets advanced
 * before the last word of the previous line has been processed.
 * Perhaps this can be fixed with different timing in the checking
 * of the line number (see ../warmenu/vectree.y).
 *
 * The above comment was written a while ago - perhaps the line
 * numbering issue has been addressed by the two lineno variables,
 * lineno & rdlineno?
 *
 */


/* global vars */

// BUG?  what is the sense of a "current" query stack?
// If we are single-threaded, then there is only one...
// If we are multi-threaded, then each thread has its own, and the threads
// are logically concurrent - so there is not a "current" one???
//Query_Stack *curr_qsp=NULL;

#ifdef QUIP_DEBUG
debug_flag_t qldebug=0;
debug_flag_t lah_debug=0;

#endif /* QUIP_DEBUG */

// moved default_qsp to error.c...
//Query_Stack *default_qsp=NULL;

static int n_cmd_args=0;
static int has_stdin=0;		// where should we set this?

#define DBL_QUOTE	'"'
#define SGL_QUOTE	'\''


void input_on_stdin(void)	// call this from a unix program
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

static void eatup_space(SINGLE_QSP_ARG_DECL)
{
	const char *str;

	if( !QRY_HAS_TEXT(CURR_QRY(THIS_QSP)) ) return;

	str=QRY_LINE_PTR(CURR_QRY(THIS_QSP)) ;

//#ifdef CAUTIOUS
//	if( str == (char *)NULL ) {
//		NWARN("CAUTIOUS:  eatup_space:  null line buf ptr");
//		CLEAR_QRY_FLAG_BITS(CURR_QRY(THIS_QSP),Q_HAS_SOMETHING);
//		return;
//	}
//#endif /* CAUTIOUS */

	assert( str != NULL );

skipit:

	/* skip over spaces */
	while( *str && isspace( *str ) ){
		// What if file has both??
		if( *str == '\n' || *str == '\r' ){
			SET_QRY_RDLINENO(CURR_QRY(THIS_QSP),
				QRY_RDLINENO(CURR_QRY(THIS_QSP)) + 1 );
#ifdef QUIP_DEBUG_LINENO
sprintf(DEFAULT_ERROR_STRING,"eatup_space:  advanced line number to %d after seeing newline char",
QRY_RDLINENO(CURR_QRY(THIS_QSP)));
advise(DEFAULT_ERROR_STRING);
#endif /* QUIP_DEBUG_LINENO */
		}
		str++;
	}

	/* comments can be embedded in lines */

	/* the buffer may contain multiple lines */
	if( *str == '#' ){
		while( *str && *str!='\n' && *str!='\r' ) str++;

		/* We used to count the line here, but because
		 * we weren't advancing the char ptr, the line
		 * got counted again above after the goto skipit.
		 */

		goto skipit;
	}

	if( *str == '\\' && isspace( *(str+1) ) ){
		str++;
		goto skipit;
	}

	if( *str == 0 )
		CLEAR_QRY_FLAG_BITS(CURR_QRY(THIS_QSP),Q_HAS_SOMETHING);

	if( QRY_HAS_TEXT(CURR_QRY(THIS_QSP)) ){
		SET_QRY_LINE_PTR(CURR_QRY(THIS_QSP),str);
	}
} // eatup_space


/*
 * Try to read the next word.  Called from qword() and rdmtext().
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
	lookahead_til(QSP_ARG  /* QLEVEL */ 0 );	// in quip arg is 0???
}

// lookahead_til won't try to read at stop_level...

int lookahead_til(QSP_ARG_DECL  int stop_level)
{
#ifdef BUILD_FOR_OBJC
	if( QLEVEL < 0 ){
		return 0;	// nothing to interpret
	}
#endif /* BUILD_FOR_OBJC */

	if( IS_HALTING(THIS_QSP) ) return 0;

	// Not used???
	//QS_FORMER_LEVEL( THIS_QSP ) = QS_LEVEL( THIS_QSP );

#ifdef QUIP_DEBUG
if( debug & lah_debug ){
sprintf(ERROR_STRING,"lookahead at level %d",QS_LEVEL( THIS_QSP ));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	while(
		QLEVEL >= stop_level
	        && (QS_FLAGS(THIS_QSP) & QS_LOOKAHEAD_ENABLED)
		&& (!IS_INTERACTIVE( CURR_QRY(THIS_QSP) ) )
		// the socket flag was getting set in the query stack,
		// not the query item, so this test always succeeded???
		&& ( ( QRY_FLAGS( CURR_QRY(THIS_QSP) ) & Q_SOCKET ) == 0 )
		/* inhibit lookahead if we are saving (don't eatup spaces) */
		/* BUT the saving flag is set one level down... */
		&& ( ! ( QLEVEL>0 && QRY_IS_SAVING( PREV_QRY(THIS_QSP) ) ) )

	){
		Query *qp;
		int _level;

		/* do look-ahead */
#ifdef QUIP_DEBUG
if( debug & lah_debug ){
sprintf(ERROR_STRING,"looking ahead, qlevel = %d, stop_level = %d",QLEVEL,stop_level);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

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
			eatup_space(SINGLE_QSP_ARG);
		}
		if( QRY_HAS_TEXT(CURR_QRY(THIS_QSP)) ){
#ifdef QUIP_DEBUG
if( debug & lah_debug ){
sprintf(ERROR_STRING,"looking ahead returning with text, qlevel = %d, text = \"%s\"",QLEVEL,
QRY_LINE_PTR(CURR_QRY(THIS_QSP)) );
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			return 1;
		}
		while( (QLEVEL == _level) && (QRY_HAS_TEXT(qp) == 0) ){
#ifdef QUIP_DEBUG
if( debug & lah_debug ){
sprintf(ERROR_STRING,"lookahead() calling nextline, qlevel = %d, original level = %d",QLEVEL,_level);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			/* nextline() never pops more than one level */
			nextline(QSP_ARG "" );	// lookahead_til
			// But because it can pop a level, we should not any
			// the space here...

			// halting bit can be set if we run out of input on a secondary thread
			// (primary thread will exit)
			if( IS_HALTING(THIS_QSP) ) return 0;

			if( QLEVEL == _level && QRY_HAS_TEXT(CURR_QRY(THIS_QSP)) ){
				eatup_space(SINGLE_QSP_ARG);
			}
		}
	}

	return 0;

} // end lookahead_til

/*
 * Check word in buf for macro.
 *
 * If the buffer contains a macro name, read the macro arguments,
 * and then push the text of the macro onto the input.
 */

static int exp_mac(QSP_ARG_DECL  const char *buf)
{
	int i;
	Macro *mp;
	Query *qp;
	const char **args;

	/* Return if we've disabled macro expansion */
	if( !(QS_FLAGS(THIS_QSP) & QS_EXPAND_MACS) ) return(0);

//if( verbose ){
//sprintf(ERROR_STRING,"exp_mac %s checking \"%s\"",THIS_QSP->qs_name,buf);
//advise(ERROR_STRING);
//}
	/* Does the buffer contain a macro name?  If not, return */
	mp=macro_of(QSP_ARG  buf);
	if( mp==NO_MACRO ) return(0);

#ifdef QUIP_DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"exp_mac expanding %s",buf);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	/* Now see if this macro has been expanded already.
	 *
	 * Originally (and still), we only did this test if
	 * the macro didn't allow recursion -
	 * but a BUG was discovered in exit_macro,
	 * which would pop multiple instances of a recursively called
	 * macro.  A kludgy work-around is to put a dummy macro call
	 * in-between...
	 */

	if( RECURSION_FORBIDDEN(mp) ){
		i=QLEVEL;
		while( i>=0 )
			if( QRY_MACRO(QRY_AT_LEVEL(THIS_QSP,i--)) == mp ){
				sprintf(ERROR_STRING,
					"Macro recursion, macro \"%s\":  ",MACRO_NAME(mp));
				WARN(ERROR_STRING);
				if( verbose )
					qdump(SINGLE_QSP_ARG);
				return(0);
			}
	}

	/* All systems go - now read the arguments */
	if( MACRO_N_ARGS(mp) > 0 ){
		args = (const char **)getbuf(MACRO_N_ARGS(mp) * sizeof(char *));

		/* first read and store the macro arguments */
		for(i=0;i<MACRO_N_ARGS(mp);i++){
			Macro_Arg *map;
			const char *s;

			map = MACRO_ARG(mp,i);

			if( MA_ITP(map) != NULL ){
				Item *ip;
				ip = pick_item(QSP_ARG  MA_ITP(map), MA_PROMPT(map) );
				if( ip != NO_ITEM )
					s=ITEM_NAME(ip);
				else
					s="xyzzy"; /* BUG? put what the user actually entered? */
			} else {
				s=nameof(QSP_ARG  MA_PROMPT(map) );

				// This can happen if we are out of input
				// at the lowest level, as in a macro call
				// with missing args at the end of a macro
				// called from the base level (e.g. an event)
				if( s == NULL ){
	sprintf(ERROR_STRING,"Missing arguments for macro %s!?",
						MACRO_NAME(mp));
					WARN(ERROR_STRING);
					return 0;
				}
			}
			if( MACRO_TEXT(mp) != NULL ){	/* don't save if no work to do */
				args[i] = savestr(s);
#ifdef QUIP_DEBUG
if( debug & qldebug ){
if( strlen(args[i]) < LLEN-80 ){
sprintf(ERROR_STRING,"exp_mac:  macro arg %d saved at 0x%lx (%s)",i,(long)args[i],args[i]);
} else {
sprintf(ERROR_STRING,"exp_mac:  macro arg %d saved at 0x%lx (%lu chars)",i,(long)args[i],(long)strlen(args[i]));
}
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			}
		}
	} else {
		args = NULL;
	}

	SET_MACRO_FLAG_BITS(mp,MACRO_INVOKED);

	/* does the macro have an empty body? */
	if( MACRO_TEXT(mp) == NULL )
		return(1);

#ifdef QUIP_DEBUG
if( debug&qldebug ){
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  Pushing text for macro %s, addr 0x%lx",
WHENCE_L(exp_mac),
MACRO_NAME(mp),(u_long)MACRO_TEXT(mp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

/*sprintf(ERROR_STRING,"pushing macro %s, qlevel = %d",MACRO_NAME(mp),QLEVEL);*/
/*advise(ERROR_STRING);*/

	// we use MSG_STR as a scratch buffer...
	sprintf(MSG_STR,"Macro %s",MACRO_NAME(mp));

	PUSH_TEXT(MACRO_TEXT(mp), MSG_STR);
	qp = CURR_QRY(THIS_QSP) ;
//#ifdef CAUTIOUS
//if( QRY_FLAGS(qp) & Q_MPASSED ){
//WARN("args passed flag set!?");
//abort();
//}
//#endif	/* CAUTIOUS */
	assert( ( QRY_FLAGS(qp) & Q_MPASSED ) == 0 );

	SET_QRY_MACRO(qp,mp);
	SET_QRY_ARGS(qp,args);
	SET_QRY_COUNT(qp, 0);
	SET_QRY_FLAGS( qp, QRY_FLAGS(qp) | Q_MACRO_INPUT );

	/* Maybe this is where we should reset the line number? */
	SET_QRY_RDLINENO(qp, 1 );

	return(1);
}

/*
 * Get next word from the top of the query file stack.
 *
 * Get next word from the top of the query file stack.
 * Calls gword() to get the next raw word.
 * Macro and variable expansion is performed.
 * If new input is needed and the input is an interactive
 * tty, then the prompt in the argument pline will be printed.
 * Calls lookahead() before returning.
 *
 * Returns the buffer provided by gword()...
 */

/*static*/ const char * qword(QSP_ARG_DECL const char *pline)
		/* prompt */
{
	const char *buf;

	do {
		do {
			if( QLEVEL < 0 ){
				return NULL;
			}

			buf=gword(QSP_ARG  pline);		/* read a raw word */
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
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  gword returned 0x%lx \"%s\"",
WHENCE_L(qword),(u_long)buf,buf);
} else {
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  gword returned 0x%lx (%lu chars)",
WHENCE_L(qword),(u_long)buf,(long)strlen(buf));
}
advise(ERROR_STRING);
}
#endif	/* QUIP_DEBUG */

		/* at this point, the word is complete (in buf) */

#ifdef HAVE_HISTORY
		if( IS_INTERACTIVE(CURR_QRY(THIS_QSP)) && *buf && IS_TRACKING_HISTORY(THIS_QSP) ){
//fprintf(stderr,"adding response '%s' to history list for prompt '%s'\n",buf,pline);
			add_def(QSP_ARG  pline,buf);
		}
#endif /* HAVE_HISTORY */

		// Advance the line number to match the last line read
		// before we do lookahead.  We do this before
		// macro expansion.
		//
		// But where do we do lookahead???


//sprintf(ERROR_STRING,"qword:  setting lineno to rdlineno (%d), qlevel = %d; buf = '%s'",
//QRY_RDLINENO(CURR_QRY(THIS_QSP)),QLEVEL,buf);
//advise(ERROR_STRING);
		SET_QRY_LINENO(CURR_QRY(THIS_QSP), QRY_RDLINENO(CURR_QRY(THIS_QSP)) );

		/* now see if the word is a macro */
	} while( exp_mac(QSP_ARG  buf) );		/* returns 1 if a macro */

	// This can happen if we run out of input while trying to
	// read the macro args...
	if( QLEVEL < 0 ) return NULL;
	
	// Not necessary?
	SET_QRY_LINENO( CURR_QRY(THIS_QSP) , QRY_RDLINENO(CURR_QRY(THIS_QSP)) );

//if( verbose ){
//sprintf(ERROR_STRING,"qword %s returning \"%s\"",
//QS_NAME(THIS_QSP),buf);
//advise(ERROR_STRING);
//}
	return(buf);
} /* end qword() */

#ifdef NOT_YET
/*
 * Force a breakout from gword().
 * Called after interrupt.
 */

void qgivup(SINGLE_QSP_ARG_DECL)
{
	CLEAR_QS_FLAG_BITS(THIS_QSP,QS_STILL_TRYING);
}
#endif /* NOT_YET */

/********* Supporting routines for rd_word *********************/
/*
 * These used to all be part of rd_word, but they have been broken out
 * to improve readability...  Efficiency may be impacted?
 */

static void after_backslash(QSP_ARG_DECL  int c,char **bufp, const char **sp,
	char **startp, Query *qp, u_int *need_p, String_Buf *sbp)
{
	char *buf;
	const char *s;
	char *start;

	buf = *bufp;
	s = *sp;
	start = *startp;

	if( isdigit(c) ){
		*buf=0;
		while( isdigit(c) ){
			*buf <<= 3;
			*buf += c -'0';
			c=(*s++);
		}
		buf++;
		s--;
	} else if( c == 't' ){
		*buf++ = '\t';
	} else if( c == 'r' ){
		*buf++ = '\r';
	} else if( c == 'n' ){
		*buf++ = '\n';
	} else if( c == 'b' ){
		*buf++ = '\b';
	} else if( c == '$' ){
		*buf++ = '\\';
		*buf++ = '$';
	} else if( c == '\n' || c=='\r' ){
		/* an escaped newline */
		/* read the next line */
		int nhave;

		// advance line counter after an escaped newline
		qp->q_rdlineno++;
#ifdef QUIP_DEBUG_LINENO
sprintf(ERROR_STRING,"after_backslash (qlevel = %d):  EOL char seen, rdlineno set to %d",
QLEVEL,qp->q_rdlineno);
advise(ERROR_STRING);
#endif /* QUIP_DEBUG_LINENO */

		// We need to continue after reading an escaped newline
		// if we are in a quote!?

		nhave = (int) (buf-start);	/* cast for pc */
		start[nhave]=0;

		if( *s == 0 ){	/* end of line */
#ifdef QUIP_DEBUG
if( debug & qldebug ){
advise("reading additional line after escaped newline");
}
#endif /* QUIP_DEBUG */

			s=qline(QSP_ARG  "");		// after_backslash
			// BUG?  what if we are out of input???  TEST

			// We used to reset qp here, but the function exits
			// so there are no more refs...
			//qp=CURR_QRY(THIS_QSP);

			*need_p += strlen(s)+16;

			if( sbp->sb_size < *need_p ){
				enlarge_buffer(sbp,*need_p);

				start=sbp->sb_buf;
				buf=start+nhave;
//if( verbose ){
//sprintf(ERROR_STRING,"after_backslash %s:  enlarged buffer, new start = 0x%lx",
//THIS_QSP->qs_name,(u_long)start);
//advise(ERROR_STRING);
//}
			}

		}
#ifdef QUIP_DEBUG
else if( debug & qldebug ){
advise("continuing to read word after escaped newline");
if( strlen(s)<80 ){
sprintf(ERROR_STRING,"remaining text:  \"%s\"",s);
} else {
sprintf(ERROR_STRING,"%lu characters remaining",(long)strlen(s));
}
advise(ERROR_STRING);
}

#endif /* QUIP_DEBUG */

	} else {
		*buf++ = (char)c; /* backslash before normal char */
	}

	*sp = s;
	*bufp = buf;
	*startp = start;

}	// end after_backslash

static void unsavechar(QSP_ARG_DECL  Query *qp, int c)
{
	int n;
	int ql;

	ql = QLEVEL;

	while( ql != 0 && (QRY_FLAGS(QRY_AT_LEVEL(THIS_QSP,ql-1)) & Q_SAVING) ){
		ql--;
		qp=QRY_AT_LEVEL(THIS_QSP,ql);
//#ifdef CAUTIOUS
//		if( QRY_TEXT(qp) == NULL ){
//			ERROR1("CAUTIOUS:  unsavechar:  no saved text!?");
//			IOS_RETURN
//		}
//#endif /* CAUTIOUS */
		assert( QRY_TEXT(qp) != NULL );

		n=(int)strlen(QRY_TEXT(qp));

//#ifdef CAUTIOUS
//		if( n <= 0 ) {
//			ERROR1("CAUTIOUS:  unsavechar:  no saved text!?");
//			IOS_RETURN
//		}
//
//#endif /* CAUTIOUS */

		assert( n > 0 );

		n--;

//#ifdef CAUTIOUS
//		if( (QRY_TEXT(qp))[n] != c ){
//			ERROR1("CAUTIOUS:  unsavechar:  last char not what expected!?");
//			IOS_RETURN
//		}
//#endif /* CAUTIOUS */

		assert( QRY_TEXT(qp)[n] == c );

		(QRY_TEXT(qp))[n] = 0;
		SET_QRY_TXTFREE(qp,QRY_TXTFREE(qp)+1);
	}
}

// rd_word flags
#define RW_HAVBACK	1
#define RW_HAVSOME	2	// what is the difference between this and NWSEEN?
#define RW_INQUOTE	4
#define RW_NOVAREXP	8
#define RW_INCOMMENT	16
#define RW_SAVING	32
#define RW_NWSEEN	64
#define RW_NEWLINE	128
#define RW_ALLDONE	256

/* process_normal - process a char not preceded by a backslash
 *
 * First skip leading whitespace (RW_NWSEEN indicates non-white seen)
 */

static void process_normal(QSP_ARG_DECL  Query *qp, int c, char **bufp, const char **sp, int *flagp )
{
	char *buf;
	const char *s;
	int flags;

	buf = *bufp;
	s = *sp;
	flags = *flagp;

	/* Nothing special, most characters processed here.
	 * We know that the previous character was not
	 * a backslash.
	 */

	/* If this char is a backslash, don't save, but
	 * remember.
	 */
	if( c == '\\' ){
		// To be here, we know the preceding char wasn't a backslash
		flags |= RW_HAVBACK;
	} else {
		if( flags & RW_INQUOTE ){
			*buf++ = (char)c;
			// Newlines have to be escaped into quotes
			//
			// The old quip didn't do this; this must have
			// been introduced to fix line counting?
			// Perhaps we should have an additional condition,
			// which enforces this the first time it is scanned,
			// but not afterwards...  How can we know?
			// We should do this check if the input is a
			// macro or a file...  MACRO
			if( IS_PRIMARY_INPUT(CURR_QRY(THIS_QSP)) ){
				if( c == '\n' && !(flags & RW_HAVBACK) ){
					flags |= RW_ALLDONE;
					goto pn_done;
				}
			}
		} else {		// not in quote
			if( c == '#' ){		// comment delimiter
				flags |= RW_INCOMMENT;
			} else if( isspace(c) ){
				if( flags & RW_NWSEEN ){
					if( c == '\n' ){
						// Set the flag, but don't increment the line
						// counter until the word is interpreted...
						// BUT instead of doing that,
						// what about just leaving the newline
						// there to be seen next time???
						// PROBLEM:  we've already saved
						// the newline!?  Should we un-save?
						s--;	// leave the newline
						if( flags & RW_SAVING ){
							unsavechar(QSP_ARG  qp,'\n');
						}
					}
					flags |= RW_ALLDONE;
					goto pn_done;
				} else {
					/* This is a space, but we haven't
					 * seen any non-spaces yet.
					 * Don't copy to output buffer.
					 * This is like eatup_space,
					 * but without side effects...
					 */
					if( c == '\n' ){
//advise("process_normal:  leading newline white space");
						flags |= RW_NEWLINE;
						goto pn_done;
					}
				}
			} else {		/* a good character */
				*buf++ = (char)c;
				flags |= RW_NWSEEN;
//advise("process_normal:  non-white seen");
			}
		} // end not in quote
		// Should we set this if we are in a comment?
//fprintf(stderr,"process_normal '%c':  HAVSOME\n",c);
//		flags |= RW_HAVSOME;
	} // end not backslash

pn_done:
	*bufp = buf;
	*sp = s;
	*flagp = flags;

	//return retval;

}	// end process_normal

static void strip_quotes(char *start,char *buf, int start_quote, int *flagp)
{
	/* BUG if the first character is a backslash escaped
	 * quote, then this may do the wrong thing...
	 */

	/* start_quote should hold the right value,
	 * because we've only seen 1 quotation
	 */

	if( start_quote == SGL_QUOTE ){
		*flagp |= RW_NOVAREXP;
	}

	/* it used to be a bug if the quote didn't come at the end */

	/* This test is flawed because it would incorrectly
	 * strip quotes from something like 'a'b'c' ...
	 * But that's kind of pathological, isn't it?
	 */
	if( *(buf-1) == start_quote ){
		*(buf-1)=0;	/* erase the closing quote */

		/* We used to strip the leading quote by simple incrementing
		 * the start pointer:
		 * start++;
		 * but now that we are using the
		 * String_Buf structure we have to move the data...
		 * bummer.
		 */
		start++;
		while(*start){
			*(start-1) = *start;
			start++;
		}
		*(start-1) = *start;	// copy the null
	}
} // end strip_quotes

// The qp arg seems ignored here???

static void sync_lbptrs(SINGLE_QSP_ARG_DECL)
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

static char * get_varval(QSP_ARG_DECL  char **spp)			/** see if buf containts a variable */
{
	const char *val_str;
	char *sp;
	char *vname;
	char tmp_vnam[LLEN];
	int had_curly=0;

	sp = *spp;

//#ifdef CAUTIOUS
//	if( *sp != VAR_DELIM ){
//		WARN("CAUTIOUS:  get_varval:  1st char should be var_delim");
//		return(NULL);
//	}
//#endif /* CAUTIOUS */

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
			WARN("Variable name too long");
			i--;
		}
		tmp_vnam[i]=0;
		vname = tmp_vnam;
	}

	if( had_curly ){
		if( *sp != RIGHT_CURLY ){
			sprintf(ERROR_STRING,"Right curly brace expected:  \"%s\"",
				*spp);
			WARN(ERROR_STRING);
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
	val_str = var_value(QSP_ARG  vname);

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

		s=SB_BUF(QS_VAR_BUF(THIS_QSP,QS_WHICH_VAR_BUF(THIS_QSP)));

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
	WARN(ERROR_STRING);

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
	/* char buf[LLEN]; */
	char *sp;
	u_int n_to_copy;
	char *start;
	int backslash_previous;

#ifdef QUIP_DEBUG
if( debug&qldebug ){
if( strlen(SB_BUF(sbp)) < LLEN-80 ){
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  var_expand \"%s\" BEGIN",
WHENCE_L(var_expand),
SB_BUF(sbp));
} else {
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  var_expand (%ld chars) BEGIN",
WHENCE_L(var_expand),
(long)strlen(SB_BUF(sbp)));
}
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( SB_BUF(RESULT) == NULL ){
//#ifdef CAUTIOUS
//		if( SB_SIZE(RESULT) != 0 ){
//			ERROR1("CAUTIOUS:  result = NULL but size != 0 !?!?");
//			IOS_RETURN
//		}
//#endif /* CAUTIOUS */
		assert( SB_SIZE(RESULT) == 0 );

		enlarge_buffer(RESULT,LLEN);
	}
	if( SB_BUF(SCRATCHBUF) == NULL ){
//#ifdef CAUTIOUS
//		if( SB_SIZE(SCRATCHBUF) != 0 ){
//			ERROR1("CAUTIOUS:  result = NULL but size != 0 !?!?");
//			IOS_RETURN
//		}
//#endif /* CAUTIOUS */
		assert( SB_SIZE(SCRATCHBUF) == 0 );
		enlarge_buffer(SCRATCHBUF,LLEN);
	}

	*(SB_BUF(RESULT)) = 0;
	sp=SB_BUF(sbp);

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
				 * by rd_word
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
					strncpy(SB_BUF(SCRATCHBUF),start,n_to_copy);
					SB_BUF(SCRATCHBUF)[n_to_copy]=0;
					cat_string(RESULT,SB_BUF(SCRATCHBUF));
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
				strncpy(SB_BUF(SCRATCHBUF),start,n_to_copy);
				SB_BUF(SCRATCHBUF)[n_to_copy]=0;
				cat_string(RESULT,SB_BUF(SCRATCHBUF));
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
 * If a single backslash precedes a dollar sign (VAR_DELIM), we might want
 * to preserve it so it can have an action in variable expansion?
 *
 * Returns 0 if some text is copied, -1 otherwise
 *
 * Originally, variable expansion was only done here on strings enclosed in
 * double quotes...  To avoid the proliferation of double quotes in scripts
 * (and the errors resulting from forgetting them), we relax this, and scan
 * for variables in all cases EXCEPT single quoted strings.
 * (This makes redundant the check (elsewhere) for the initial character
 * being a dollar sign.)
 *
 * BUG?  because rd_word uses a dynamically growable string buffer,
 * it can return a string which is longer than LLEN...  This can happen
 * for instance when a closing quote is missing...
 *
 * variable expansion is performed at the end of rd_word, by calling var_expand
 */

static char * rd_word(SINGLE_QSP_ARG_DECL)
{
	Query *qp;

	int start_quote=0;	/* holds the value of the starting quote char,
				 * if in a quote, otherwise 0
				 */
	int n_quotations=0;	/* to handle things like "a b c"X"x y z" */

	char *start,*buf;
	int flags=0;
	u_int need_size;
	const char *s;
	String_Buf *sbp;

//QUEUE_CHECK(rd_word)
	qp=(CURR_QRY(THIS_QSP));

	//if( qp != FIRST_QRY(THIS_QSP) && QRY_IS_SAVING(PREV_QRY(THIS_QSP)) )
	if( NEED_TO_SAVE(qp) ){
		flags |= RW_SAVING;
	}

	/* actually, the size needed is probably LESS than
	 * the input buffer, because backslash sequences
	 * will be reduced in length, while everything else
	 * is copied literally.
	 */


#define QRY_RETSTR	QRY_RETSTR_AT_IDX(CURR_QRY(THIS_QSP),		\
				QRY_RETSTR_IDX(CURR_QRY(THIS_QSP)))
#define SET_QRY_RETSTR(sbp)						\
			SET_QRY_RETSTR_AT_IDX(CURR_QRY(THIS_QSP),	\
				QRY_RETSTR_IDX(CURR_QRY(THIS_QSP)),sbp)

#define NEXT_QRY_RETSTR							\
									\
	SET_QRY_RETSTR_IDX(CURR_QRY(THIS_QSP),				\
		( QRY_RETSTR_IDX(CURR_QRY(THIS_QSP)) >= (N_QRY_RETSTRS-1) ? \
		0 : (1+QRY_RETSTR_IDX(CURR_QRY(THIS_QSP))) ) );


/* Ideally, we should reset this before a command is read - but qline
 * could pop the level when trying to read a word, which will happen
 * when one macro calls a lot of other macros...  a qsp-wide flag
 * might solve this...
 */

	if( (sbp = QRY_RETSTR) == NO_STRINGBUF ){
		SET_QRY_RETSTR(new_stringbuf());
		sbp = QRY_RETSTR;
	}

	// This might be a good place to increment?
	// We advance the index for next time
	NEXT_QRY_RETSTR

	// The word should really be much less than the whole mess,
	// although it could be a quoted string?

	need_size = (int)strlen(QRY_LINE_PTR(qp) )+16;	/* conservative estimate */

	if( SB_SIZE(sbp) < need_size ){
		enlarge_buffer(sbp, need_size);
	}

	start=SB_BUF(sbp);
	*start=0;			/* default is "" */

	buf=start;

	/* Old comment:
	 * no check for overflow in the following loop
	 * because both buffers have size LLEN
	 *
	 * New comment:
	 * the destination buffer is dynamically growable...
	 * but how big should it be?
	 */

	s=QRY_LINE_PTR(qp) ;		/* this is the read scan pointer */
	assert(s!=NULL);

	/* Eventually we will want to strip quote marks,
	 * but we don't do it right away, because if we are
	 * saving the text (as in a loop), we want to save
	 * the quote's too.
	 */

	if( *s == DBL_QUOTE || *s == SGL_QUOTE )
		start_quote = *s;
	else start_quote=0;

	/* If the input buffer is a recently read line, then we know it
	 * while have less than LLEN chars, but if we are reading from a macro
	 * then it could be longer...
	 */

	while( *s ){		/* scan the input buffer */
		int c;

		c=(*s++);

//#ifdef CAUTIOUS
//		// This test detects garbage in the line buffer...
//		// caused by memory corruption!?
//// This is also triggered by UTF-8 multi-char sequences...
//// We need to fix this, but if they only occur in comments, then we
//// can safely ignore.
//#ifdef FOO
//		if( ! isascii(c) ){
//			// Could be UTF8???
//fprintf(stderr,
//"CAUTIOUS:  rd_word encountered a non-ascii character (0x%x)!? (UTF-8?)\n",c&0xff);
//			abort();
//		}
//#endif // FOO
//#endif // CAUTIOUS
		// We can get UTF8 chars when using other editors!?
		//assert( isascii(c) );

		if( flags & RW_SAVING ) {
			savechar(QSP_ARG  qp,c);
		}

		/* now do something with this character */

		if( flags & RW_INCOMMENT ){
			/* skip all characters until a newline */
			if( c == '\n' ){
				// increment the line counter
				qp->q_rdlineno ++;
				flags &= ~RW_INCOMMENT;
				/* We don't let backslash's escape
				 * newlines within comments...
				 * For multi-line comments, use a new delimiter.
				 */
			}
		} else {
			// Check for opening or closing quote
			// BUG?  should we do this before backslash checking???
			if( flags & RW_INQUOTE ){
				/* check if the character is the closing quote mark */
				if( c == start_quote && !(flags & RW_HAVBACK) ){
					flags &= ~RW_INQUOTE;
					n_quotations++;
				}
			} else {
				if( c == DBL_QUOTE || c == SGL_QUOTE ){
					/* this is the opening quote */
					start_quote = c;
					flags |= RW_INQUOTE;
				}
			}

			if( flags & RW_HAVBACK ){		/* char follows backslash */
				after_backslash(QSP_ARG  c,&buf,&s,&start,qp,&need_size,sbp);
				/* BUG maybe we shouldn't have any after escaped nl ? */
				//flags |= RW_HAVSOME;
				flags &= ~RW_HAVBACK;
			} else {
				process_normal(QSP_ARG  qp,c,&buf,&s,&flags);
				if( flags & RW_NEWLINE ){
					// advance line counter if we process a newline
					qp->q_rdlineno ++;
#ifdef QUIP_DEBUG_LINENO
sprintf(ERROR_STRING,"rd_word advanced line number to %d",qp->q_rdlineno);
advise(ERROR_STRING);
#endif /* QUIP_DEBUG_LINENO */
					flags &= ~RW_NEWLINE;
				}
				if( flags & RW_ALLDONE ){
					goto alldone;
				}
			}
		} // end not in comment
	} // end while buffer has something

	/* If we are here, our input pointer should be pointing at a null
	 * byte...  There was a nasty problem with the original implementation
	 * of rls_mouthful - it released the text string (which had
	 * been the input line buffer), and the action of free (at least
	 * with mallocScribble enabled) caused it to be non-null.
	 * MallocScribble was turned on, because prior to that the freed
	 * memory could be reallocated and rewritten at any time,
	 * and problems occurred later...
	 *
	 * Why not just set the line ptr to NULL here??
	 */

alldone:

	/* We get here when we are done reading the word, either because
	 * we have run out of input or because we encountered a white
	 * space character.
	 *
	 * We don't want to return an empty string,
	 * so we check the string length.
	 */

	if( flags & RW_HAVBACK )
		advise("still have backslash at end of buffer!?");

	*buf=0;

//#ifdef CAUTIOUS
//if( strlen(start) >= (SB_SIZE(sbp)-1) ){
//sprintf(ERROR_STRING,"start len = %ld, retstr size = %zd",
//(long)strlen(start), SB_SIZE(sbp) );
//advise(ERROR_STRING);
//
//ERROR1("CAUTIOUS too much stuff!!!");
//IOS_RETURN_VAL(NULL)
//}
//#endif	/* CAUTIOUS */
	assert( strlen(start) < (SB_SIZE(sbp)-1) );

	if( start_quote && (flags & RW_INQUOTE) ){
		sprintf(ERROR_STRING,"rd_word:  no closing quote (start_quote = %c)",start_quote);
		WARN(ERROR_STRING);
		/* If the buffer has overflowed, we can't print into error_string! */
#define BC_STR	"buffer contained "
		if( strlen(start)+strlen(BC_STR) < (LLEN-4) ){
			sprintf(ERROR_STRING,"%s\"%s\"",BC_STR,start);
			WARN(ERROR_STRING);
		} else {
			sprintf(ERROR_STRING,"buffer contains %ld chars",(long)strlen(start));
			WARN(ERROR_STRING);
		}
	}

	if( ! (flags & RW_NWSEEN) )
		return(NULL);

	if( *s ){
		SET_QRY_LINE_PTR(qp,s);	/* current text scan ptr */
	} else {
		// hope this doesn't mess up line numbering...
		SET_QRY_LINE_PTR(qp,NULL);
		CLEAR_QRY_FLAG_BITS(qp,Q_HAS_SOMETHING);
		// Should we clear the HAVE_SOMETHING flag???
	}

	// sync up the lbptr's at the saving levels...
	if( flags & RW_SAVING ) {
		sync_lbptrs(SINGLE_QSP_ARG);
	}

	/* strip quotes if they enclose the entire string */
	/* This is useful in vt script, but bad if we are using these routines to pass input
	 * to the vt expression parser!?
	 */

	if( (QS_FLAGS(THIS_QSP) & QS_STRIPPING_QUOTES)
			&& (*start==SGL_QUOTE || *start==DBL_QUOTE)
			&& n_quotations==1 ){
		strip_quotes(start,buf,start_quote,&flags);
	}

	/* BUG this will prevent variable expansion of lines
	 * which contain single quoted strings and vars...
	 */

	if( ! (flags & RW_NOVAREXP) )
		var_expand(QSP_ARG  sbp);

	return(SB_BUF(sbp));
} // end rd_word

/*
 *	Get a raw word.
 *
 *	Get a raw word from the top of the query file stack.  If there is
 *	no current text, will get more by calling qline().  Strips leading
 *	white space, returns the next space delimited word by calling rd_word().
 *	No macro or variable expansion is performed. (sic)
 *
 *	Variable expansion performed in rd_word, unless single-quoted...
 *
 *	returns buffer returned by rd_word()
 */

const char * gword(QSP_ARG_DECL  const char* pline)
		/* prompt string */
{
	Query *qp;
	int need;
	const char *buf=NULL;	/* initialize to elim warning */

	//Let's initialize when we create the thing!?
	//if( !(QS_FLAGS(THIS_QSP) & QS_INITED) ) init_query_stack(THIS_QSP);

	if( IS_HALTING(THIS_QSP) ){
		CLEAR_QRY_FLAG_BITS(CURR_QRY(THIS_QSP),Q_HAS_SOMETHING);
	}

gwtop:
	if( QLEVEL < 0 ) return NULL;

	need=1;

	SET_QS_FLAG_BITS(THIS_QSP, QS_STILL_TRYING);

	qp=(CURR_QRY(THIS_QSP));
	if( !QRY_HAS_TEXT(qp) )	/* need to read more input */
	{
		if( IS_HALTING(THIS_QSP) ) return(NULL);
		buf=qline(QSP_ARG  pline );
	}

	if( QLEVEL < 0 ){
		return NULL;
	}


	qp=(CURR_QRY(THIS_QSP));	/* qline may pop the level!!! */
	//eatup_space(SINGLE_QSP_ARG);

	if( QRY_HAS_TEXT(qp) ){
		/* rd_word() returns non-NULL if successful */
		if( (buf=rd_word(SINGLE_QSP_ARG)) != NULL ){
			need=0;
		} else {
			//SET_QRY_HAS_TEXT(qp,0);
			CLEAR_QRY_FLAG_BITS(qp,Q_HAS_SOMETHING);
		}
	}
	if( need ){
		if( IS_STILL_TRYING(THIS_QSP) ) goto gwtop;	/* try again */
		else return(NULL);
	}
	return(buf);
} // end gword

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

	buf=qline(QSP_ARG  pline);
	/*
	n=strlen(buf);
	if( n>1 && (buf[n-1] == '\n' || buf[n-1] == '\r') )
		buf[n-1]=0;
	*/
	//SET_QRY_HAS_TEXT(CURR_QRY(THIS_QSP),0);
	CLEAR_QRY_FLAG_BITS(CURR_QRY(THIS_QSP),Q_HAS_SOMETHING);
	SET_QRY_LINENO(CURR_QRY(THIS_QSP), QRY_RDLINENO(CURR_QRY(THIS_QSP)) );
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

const char * qline(QSP_ARG_DECL  const char *pline)
		/* prompt string */
{
	Query *qp;
	const char *buf;

#ifdef QUIP_DEBUG
//if( debug & qldebug ){
//sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  qlevel = %d",
//WHENCE_L(qline),QLEVEL);
//advise(ERROR_STRING);
//}
#endif /* QUIP_DEBUG */
	while(1) {
		if( QLEVEL < 0 ){
			return NULL;
		}

		/* if the current level is out, nextline will pop 1 level */
		buf=nextline(QSP_ARG  pline);	// qline

		if( IS_HALTING(THIS_QSP) ) return NULL;

		if( QLEVEL < 0 ){
			return NULL;
		}

		qp=(CURR_QRY(THIS_QSP));

		if( QRY_HAS_TEXT(qp) ){
			if( IS_DUPING ){
				dup_word(QSP_ARG  QRY_LINE_PTR(qp) );
				dup_word(QSP_ARG  "\n");
			}
			return(buf);
		}
	}
}

#ifdef HAVE_HISTORY
#ifdef TTY_CTL

COMMAND_FUNC( set_completion )
{
	if( askif(QSP_ARG  "complete commands") ){
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
		nice_exit(QSP_ARG  0);
	} else {
		SET_QS_FLAG_BITS(THIS_QSP,QS_HALTING);
		CLEAR_QS_FLAG_BITS(THIS_QSP,QS_STILL_TRYING);
	}
}

#ifdef HAVE_HISTORY
#ifdef TTY_CTL
static Query *hist_select(QSP_ARG_DECL char *buf,int buf_size,const char* pline)
{
	const char *s;
	Query *qp;

	qp = CURR_QRY(THIS_QSP);

	s=get_sel(QSP_ARG  pline,QRY_FILE_PTR(qp),stderr);
	if( s==NULL ){			/* ^D */
		if( QLEVEL > 0 ){
			pop_file(SINGLE_QSP_ARG);
			qp = CURR_QRY(THIS_QSP);
			return(qp);
		} else {
			advise("EOF");
			nice_exit(QSP_ARG  0);
		}
	}
	// check for buffer overrun?
	if( strlen(s) > buf_size-2 ){
		sprintf(ERROR_STRING,"hist_select:  buffer too small!?");
		WARN(ERROR_STRING);
		// BUG - well, do something about it!
	}

	strcpy(buf,s);
	strcat(buf,"\n");
	SET_QRY_FLAG_BITS(qp,Q_HAS_SOMETHING);
	SET_QRY_LINE_PTR(qp,buf);

	return(qp);
} // end hist_select
#endif /* TTY_CTL */
#endif /* HAVE_HISTORY */

/*
 * Get the next line from the top of the query stack.
 *
 * Gets the next line.  If the input is an interactive tty, prints
 * the prompt and calls hist_select() (if HAVE_HISTORY is defined).
 * Otherwise calls the query structures read function (fgets() for
 * normal files).  If EOF is encountered, pops the file and returns.
 *
 * advances q_rdlineno, which may not be the current lineno if lookahead
 * is enabled...
 *
 * We return the buffer, AND set QRY_LINE_PTR(qp)  - redundant?
 */

const char * nextline(QSP_ARG_DECL  const char *pline)
		/* prompt */
{
	Query *qp;
	char *buf;
	int _is_i;
#ifdef MAC
	extern void set_mac_pmpt(char *);
#endif /* MAC */

	if( IS_HALTING(THIS_QSP) ){
		if( QLEVEL > 0 )		// or should >= ???  BUG?
			pop_file(SINGLE_QSP_ARG);
		return(NULL);
	//	return("");
	}

	qp=(CURR_QRY(THIS_QSP));
	buf=QRY_BUFFER(CURR_QRY((THIS_QSP)));

	// buf might be NULL if we are at the end of a macro?

#ifdef HAVE_HISTORY
#ifdef TTY_CTL
nltop:
#endif // TTY_CTL
#endif // HAVE_HISTORY

#ifdef MAC
	set_mac_pmpt(pline);
#else	/* ! MAC */

// for debugging

	if( (_is_i=IS_INTERACTIVE(qp)) || (QS_FLAGS(THIS_QSP) & QS_FORCE_PROMPT) ){

		/* only force prompts at level 0 */
		/* but what about redir to /dev/tty? */

		if( _is_i || QLEVEL==0 ){

			fputs(pline,stderr);
#ifdef HAVE_HISTORY
#ifdef TTY_CTL
			if( IS_TRACKING_HISTORY(THIS_QSP) &&
					/*(QS_FLAGS(THIS_QSP) & QS_COMPLETING)*/
					IS_COMPLETING(THIS_QSP) ){
				qp=hist_select(QSP_ARG  buf,LLEN,pline);
				if( QRY_HAS_TEXT(qp) ){
					return(QRY_LINE_PTR(qp) );
				}
				else goto nltop;
			}
#endif /* TTY_CTL */
#endif /* HAVE_HISTORY */
		}
	}
#endif /* ! MAC */


	/* We used to advance the line number here, but when reading from a buffer
	 * we advance the line numbers in rd_word, and that renders this unnecessary.
	 *
	 * The problem is, lines get read from various places, and
	 * we need to count them all...
	 *
	 * It would certainly be simplest to count them when we
	 * read them...  But the "line buffer" can contain multiple
	 * lines in the case of a macro or a loop, so we have to
	 * count as we scan.
	 */

	if( QRY_RDLINENO(qp) == 0 ){
		SET_QRY_RDLINENO(qp,1);		// count the first line
#ifdef QUIP_DEBUG_LINENO
sprintf(ERROR_STRING,"nextline:  initialized first line number to %d",QRY_RDLINENO(qp));
advise(ERROR_STRING);
#endif /* QUIP_DEBUG_LINENO */
	}


	/* Call the read function - fgets if it is a regular file */

	if( (*(QRY_READFUNC(qp)))(QSP_ARG  (void *)buf,LLEN,(void *)QRY_FILE_PTR(qp)) == NULL ){
		/* this means EOF if reading with fgets()
		 * or end of a macro...
		 */
		if( QLEVEL > 0 ){	/* EOF redir file */
			pop_file(SINGLE_QSP_ARG);
			return("");
		} else if( has_stdin ){
sprintf(ERROR_STRING,"EOF on %s",QS_NAME(THIS_QSP));
advise(ERROR_STRING);
			halt_stack(SINGLE_QSP_ARG);		// nextline
			// halting master stack will exit program,
			// but other threads have to get out gracefully...
			return("");		// nextline
		} else {
			// Normally if we encounter EOF on the root
			// query stack, we want to quit.
			// But under iOS, there is no stdin,
			// and the first file is the startup file.
			//
			// This caused an infinite loop of prompt printing...
			pop_file(SINGLE_QSP_ARG);
			return("");
		}
	} else {		/* have something */
		/* make sure that we have a complete line */
		int n;

		n=(int)strlen(buf);
//#ifdef CAUTIOUS
//		/* fgets() should not read more than LLEN-1 chars... */
//		if( n > LLEN-1 ) {
//			ERROR1("CAUTIOUS:  line too long");
//			IOS_RETURN_VAL(NULL)
//		}
//#endif	/* CAUTIOUS */
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
			return("");
		}
				
		n--;

		if( QRY_READFUNC(qp) == ((READFUNC_CAST) FGETS) && buf[n] != '\n' &&
			buf[n] != '\r' ){
			WARN("nextline:  input line not terminated by \\n or \\r");
			sprintf(ERROR_STRING,"line:  \"%s\"",buf);
			advise(ERROR_STRING);
		}
		//SET_QRY_HAS_TEXT(qp,1);	// why is this commented out???
		SET_QRY_FLAG_BITS(qp,Q_HAS_SOMETHING);
		SET_QRY_LINE_PTR(qp,buf);
		return(buf);
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
		WARN(ERROR_STRING);
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

	//lookahead_til(QSP_ARG  0);	// used to be lookahead?
	lookahead(SINGLE_QSP_ARG);

	if( QLEVEL < 0 ) return 0;

//	if(!(QS_FLAGS(THIS_QSP) & QS_INITED)) init_query_stack(THIS_QSP);
	return IS_INTERACTIVE( CURR_QRY(THIS_QSP) );
}


/* scan_remainder
 *
 * scan from qp_lbptr to the end of the buffer.
 * We expect only one newline (at the end), and nothing
 * except white space before a comment delimiter.
 *
 * The above comment applies to the old case where
 * we never read a macro definition from a buffered file.
 * Now, we expect only white space or a comment
 * before the FIRST newline.
 */

static void scan_remainder(QSP_ARG_DECL  const char *s, const char *location )
{
	int c;

	while( (c=(*s++)) ){

		if( isspace(c) ){
			if( c == '\n' ) return;
		} else {
			if( c == '#' ) return;
			else {
				sprintf(ERROR_STRING,"Extra text encountered after %s",location);
				WARN(ERROR_STRING);
				return;
			}
		}
		/* else not a space and comment seen */
	}
}

// extract_line - a helper function for macro reading.
//
// For the time being, this function is only used when reading macro
// definitions, so there will be no recursive calls.  Therefore it
// should be safe to use a single static buffer...

static const char *extract_line(SINGLE_QSP_ARG_DECL)
{
	static char linebuf[LLEN];
	const char *from;
	char *to;
	int level;
	int n;

	level = QLEVEL;
	while( ! QRY_HAS_TEXT(CURR_QRY(THIS_QSP)) ){
		/*from=*/qline(QSP_ARG  "");
		// BUG - we need to handle premature EOF?
		if( QLEVEL != level ){
			sprintf(ERROR_STRING,"extract_line:  premature EOF!?");
			WARN(ERROR_STRING);
			return NULL;
		}
	}

	// Copy up until the next newline

	from = QRY_LINE_PTR(CURR_QRY(THIS_QSP)) ;
	to = linebuf;
	n=0;
	while( *from && *from != '\n' ){
		// check for buffer overrun
		if( n >= LLEN-1 ){
			sprintf(ERROR_STRING,"extract_line:  buffer too small!?");
			WARN(ERROR_STRING);
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

	SET_QRY_RDLINENO(CURR_QRY(THIS_QSP),
		1+QRY_RDLINENO(CURR_QRY(THIS_QSP)) );	// count this line

	return linebuf;
}

/* read in macro text
 *
 * Read text until a line with a single period '.' is encountered.
 *
 * The original implementation broke when we added the ability to
 * read encrypted files.  If we decrypt a file into a buffer, and then
 * push the entire buffer onto the input stack, it appears as one
 * great big line...
 */

String_Buf * rdmtext(SINGLE_QSP_ARG_DECL)
{
	const char *ms="Enter text of macro; terminate with line beginning with '.'";
	Query *qp;
	const char *s;
	/* we make a new one for each and every macro... */

	/* The old one had a memory leak, because the string buffers were
	 * never deallocated...
	 */
	String_Buf *mac_sbp;

	mac_sbp = new_stringbuf();

	if(!(QS_FLAGS(THIS_QSP) & QS_INITED)) init_query_stack(THIS_QSP);

	qp = CURR_QRY(THIS_QSP);

	if( IS_INTERACTIVE(qp) || (QS_FLAGS(THIS_QSP) & QS_FORCE_PROMPT) ) advise(ms);

	/* lookahead line may have been read already... */

	if( QRY_HAS_TEXT(qp) ){
		/* this case only arises for scripted macros,
		 * so don't worry about strcat'ing \n
		 */

		/* check for a null-body macro */

		// this is just so we can warn if there's garbage on the line
		scan_remainder(QSP_ARG  QRY_LINE_PTR(qp), "macro declaration");
	}

	copy_string(mac_sbp,"");	// initialize

	//s=qline(QSP_ARG  "");
	//qp->q_rdlineno++;	// count this line

	s=extract_line(SINGLE_QSP_ARG);
	if( s == NULL ) goto bad_def;

	/* The first line is always the null string -
	 * I suppose that is because the terminating newline gets eaten.
	 *
	 * There can be stuff here if there is a comment
	 * on the same line as the last argument prompt.
	 */

	while( *s != '.' ){
		s=extract_line(SINGLE_QSP_ARG);
		if( s == NULL ) goto bad_def;
		if( *s != '.' ){
			cat_string(mac_sbp,s);
			// Newlines are stripped by extract_line
			cat_string(mac_sbp,"\n");
		}
	}
//dun:
	//SET_QRY_HAS_TEXT(qp,0);	/* don't read '.' */

	// We don't want to clear this flag if we are reading from
	// a buffer containing many lines...
	//CLEAR_QRY_FLAG_BITS(qp,Q_HAS_SOMETHING);

	//qp->q_lbptr++;		// skip the dot?

	//qp->q_lineno = qp->q_rdlineno;	// in case a warning is printed by scan_remainder
	//scan_remainder(QSP_ARG  qp,"end of macro definition");
	//qp->q_rdlineno++;

	s++;	// skip '.'
	scan_remainder(QSP_ARG  s,"end of macro definition");

	return(mac_sbp);
bad_def:
	WARN("incomplete macro definition");
	// BUG?  We might return NULL so that we can print an error message with
	// the macro name...
	return(mac_sbp);
} /* end rdmtext */

Macro_Arg * read_macro_arg(QSP_ARG_DECL int i)
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
	s = NAMEOF(pstr2);
	/* We can specify the item type of the prompted-for object
	 * by preceding the prompt with an item type name in brackets,
	 * e.g. <Data_Obj>
	 */
	map->ma_itp=NO_ITEM_TYPE;		// default
	if( *s == '<' ){
		int n;
		n=(int)strlen(s);
		if( s[n-1] == '>' ){
			strcpy(pstr2,s+1);
			pstr2[n-2]=0;	/* kill closing bracket */
			map->ma_itp = get_item_type(QSP_ARG  pstr2);
			if( map->ma_itp == NO_ITEM_TYPE ){
WARN("Unable to process macro argument item type specification.");
			}
		} else {
			WARN("Unterminated macro argument item type specification.");
		}
		s=NAMEOF(pstr);
	}
	map->ma_prompt = savestr(s);
	return map;
}

void rls_macro_arg( Macro_Arg * map )
{
	rls_str(map->ma_prompt);
	givbuf(map);
}

#ifdef NOT_USED
/*
 * Redirect input to tty
 */

void readtty(SINGLE_QSP_ARG_DECL)
{ redir(QSP_ARG  tfile(SINGLE_QSP_ARG), "/dev/tty" ); }
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
	redir(QSP_ARG  stdin, "-" );
#endif // BUILD_FOR_CMD_LINE

#ifdef QUIP_DEBUG
	qldebug = add_debug_module(QSP_ARG  "query");;
	lah_debug = add_debug_module(QSP_ARG  "lookahead");;
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
	new_qsp = new_query_stack(QSP_ARG  name);

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

//#ifdef CAUTIOUS
//	if( ! IS_DUPING ){
//		WARN("CAUTIOUS:  NOT duping!?");
//		return;
//	}
//#endif
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

void dup_word(QSP_ARG_DECL  const char *s)
{
	int chunkme;
	FILE *fp;

#ifdef THREAD_SAFE_QUERY
	// This null test wasn't needed until we tried to exit a thread...
	if( s == NULL ) return;
#endif /* THREAD_SAFE_QUERY */

	fp = QRY_DUPFILE(CURR_QRY(THIS_QSP));

//#ifdef CAUTIOUS
//	if( ! IS_DUPING ){
//		WARN("CAUTIOUS:  NOT duping!?");
//		return;
//	}
//#endif
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

int dupout(QSP_ARG_DECL  FILE *fp)			/** save input text to file fp */
{
	if( IS_DUPING ){
		WARN("already dup'ing");
		return(-1);
	} else {
		SET_QRY_DUPFILE(CURR_QRY(THIS_QSP),fp);
		return(0);
	}
}

void savechar(QSP_ARG_DECL  Query *qp,int c)
{
	char buf[2];

	buf[0]=(char)c;
	buf[1]=0;
	savetext(QSP_ARG  qp,buf);
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

void savetext(QSP_ARG_DECL  Query *qp,const char* buf)
	/* query structure pointer */
	/* text to save */
{
	int n_more;
	char *str;
	int ql;
	//int first_ql;
	Query *save_qp;

	/*first_ql=*/ql=QLEVEL;

	save_qp = QRY_AT_LEVEL(THIS_QSP,ql-1);
	if( (QRY_FLAGS(save_qp) & Q_SAVING) == 0 ){
		NERROR1("savetext:  saving flag not set!?");
		IOS_RETURN
	}

	n_more=(int)strlen(buf)+1;

	// Do we save at multiple levels???
	while( ql != 0 && (QRY_FLAGS(QRY_AT_LEVEL(THIS_QSP,ql-1)) & Q_SAVING) ){
		ql--;
        qp=QRY_AT_LEVEL(THIS_QSP,ql);
//#ifdef CAUTIOUS
//		if( QRY_TEXT(qp) == NULL ){
//			//int index;
//			NERROR1("CAUTIOUS:  whoa, null text buffer!!!");
//			IOS_RETURN
//		}
//#endif /* CAUTIOUS */
        
		assert( QRY_TEXT(qp) != NULL );
        
		while( n_more > QRY_TXTFREE(qp) ){
			SET_QRY_TXTSIZE(qp,
				QRY_TXTSIZE(qp) + LOOPSIZE) ;
			SET_QRY_TXTFREE(qp,
				QRY_TXTFREE(qp) + LOOPSIZE);
			str=(char*) getbuf(QRY_TXTSIZE(qp));
			if( str==NULL ) {
				NERROR1("save_text");
				IOS_RETURN
			}
			strcpy(str,QRY_TEXT(qp));

			givbuf(QRY_TEXT(qp));
			SET_QRY_TEXT(qp,str);
		}

		strcat(QRY_TEXT(qp),buf);
		SET_QRY_TXTFREE(qp,
			QRY_TXTFREE(qp) - n_more);

	}
//advise("savetext DONE");
//qdump(SINGLE_QSP_ARG);
} // end savetext

// end of old qdup.c

// start of old qword.c

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
	// These checks probably are CAUTIOUS...  BUG?

	if( !(QS_FLAGS(THIS_QSP) & QS_INITED) ) init_query_stack(THIS_QSP);
	if( QLEVEL < 0 ) {
		ERROR1("no query file");
		IOS_RETURN
	}

	/* can be used to cause input to be read from a socket */
	/* why was this void * here???
	query[qlevel].q_readfunc= (void *)func;
	*/
	SET_QRY_READFUNC(CURR_QRY(THIS_QSP), rfunc);
}

static void init_parser_data(Query_Stack *qsp)
{
	ALLOC_QS_VECTOR_PARSER_DATA(qsp);
	// Now allocate the strings
	SET_QS_YY_INPUT_LINE(qsp,getbuf(LLEN));
	SET_QS_YY_LAST_LINE(qsp,getbuf(LLEN));
	SET_QS_EXPR_STRING(qsp,getbuf(LLEN));
	SET_QS_EDEPTH(qsp, -1);
	SET_QS_CURR_STRING(qsp, qsp->_qs_expr_string );

	ALLOC_QS_SCALAR_PARSER_DATA(qsp);
	// initialize the fields
	SET_QS_SPD_EDEPTH(qsp,-1);
	SET_QS_SPD_WHICH_STR(qsp,0);
	SET_QS_SPD_IN_PEXPR(qsp,0);
	SET_QS_SPD_ORIGINAL_STRING(qsp,NULL);
	SET_QS_SPD_ESTRINGS_INITED(qsp,0);
	// set spd_expr_string's to NULL ???
	//SET_QS_SPD_YYSTRPTR(qsp,NULL);
}

// Initialize a Query_Stack

void init_query_stack(Query_Stack *qsp)
{
	int i;
	const char *save_name;
	int save_serial;

//#ifdef CAUTIOUS
//	if( qsp == NULL ) {
//		ERROR1("CAUTIOUS:  init_query_stack passed NULL query stack pointer");
//		IOS_RETURN
//	}
//#endif /* CAUTIOUS */
	assert( qsp != NULL );

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

	// not used?
	//SET_QS_FORMER_LEVEL(qsp,0);

	//qsp->qs_fn_depth=(-1);
	//SET_QS_LOOKAHEAD_LEVEL(qsp, 0);	/* BUG don't need this var, because never written!? */
	SET_QS_WHICH_VAR_BUF(qsp, 0);

	for(i=0;i<MAX_VAR_BUFS;i++){
		SET_QS_VAR_BUF(qsp,i,new_stringbuf());
	}

	SET_QS_SCRATCH(qsp,new_stringbuf());
	SET_QS_RESULT(qsp,new_stringbuf());

	// We used to initialize return strings here,
	// but those are now part of the query structure,
	// not the query stack.

	//CLEAR_QS_PROMPT(qsp);
	SET_QS_PROMPT_SB(qsp,new_stringbuf());

	//qsp->qs_cmd_itp = NO_ITEM_TYPE;
	SET_QS_FMT_CODE(qsp, FMT_DECIMAL);

	// nexpr.y initializations
	//SET_QS_WHICH_ESTR(qsp, 0);
	//SET_QS_ESTR_ARRAY = (char **) getbuf(sizeof(char *)*

	init_parser_data(qsp);

	// This is really a query flag, and no query has been pushed yet!?
	//CLEAR_QS_FLAG_BITS(qsp,QS_CHEWING);
	SET_QS_CHEW_LIST(qsp, NO_LIST);
	SET_QS_CALLBACK_LIST(qsp, NO_LIST);

	SET_QS_VAR_FMT_STACK(qsp,NO_STACK);
	SET_QS_NUMBER_FMT(qsp,NULL);

	SET_QS_AV_STRINGBUF(qsp,NULL);

	// BUG?  perhaps we should not allocate this until we need it?
	/*
	SET_QS_DOBJ_ASCII_INFO(qsp,getbuf(sizeof(Dobj_Ascii_Info)));
	init_dobj_ascii_info(QSP_ARG  QS_DOBJ_ASCII_INFO(qsp) );
	*/
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

// The filename is now part of the query struct, so we don't have a separate stack...

#ifdef THREAD_SAFE_QUERY

//char *qpfgets( TMP_QSP_ARG_DECL char *buf, int size, FILE *fp )
char *qpfgets( QSP_ARG_DECL void *buf, int size, void *fp )
{
	return( fgets(buf,size,(FILE *)fp) );
}
#endif /* THREAD_SAFE_QUERY */

// BUG - should use string_buf for buffers instead of fixed size buffer...

#define SET_QUERY_DEFAULTS(qp)						\
									\
	SET_QRY_LINENO(qp,0);						\
	SET_QRY_RDLINENO(qp,0);						\
	SET_QRY_COUNT(qp,0);						\
	SET_QRY_FLAGS(qp,0);						\
	SET_QRY_DUPFILE(qp,NULL);					\
	SET_QRY_TEXT(qp,NULL);						\
	SET_QRY_MACRO(qp,NO_MACRO);					\
	if( QRY_BUFFER(qp) == NULL ){					\
		SET_QRY_BUFFER(qp, getbuf(LLEN) );			\
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
}

void redir(QSP_ARG_DECL FILE *fp, const char *filename)
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

static void mpass(QSP_ARG_DECL Query *qpto,Query *qpfr)
{
	if( QRY_MACRO(qpfr) == NO_MACRO ){
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
	redir( QSP_ARG QRY_FILE_PTR(CURR_QRY(THIS_QSP)), s );

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
	SET_QRY_RDLINENO( CURR_QRY(THIS_QSP),QRY_RDLINENO(QRY_AT_LEVEL(THIS_QSP,QLEVEL-1)));

	/* the absence of the next line caused a subtle bug
	 * for loops within macros that were preceded by a doubly
	 * redirected file... */

	SET_QRY_READFUNC( CURR_QRY(THIS_QSP),QRY_READFUNC(QRY_AT_LEVEL(THIS_QSP,QLEVEL-1)));

	/* loops within macros */
	mpass(QSP_ARG CURR_QRY(THIS_QSP),QRY_AT_LEVEL(THIS_QSP,QLEVEL-1));
} // end of dup_input

/* stuff for loops on input */

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

void open_loop(QSP_ARG_DECL int n)
			/* loop count */
{
	Query *qp;

	qp=(CURR_QRY(THIS_QSP));
/*
if( n < 0 ) advise("opening do loop");
sprintf(ERROR_STRING,"saving to q_text buffer at level %d",qlevel);
advise(ERROR_STRING);
*/

	dup_input(SINGLE_QSP_ARG);

	SET_QRY_COUNT(qp,n);
	SET_QRY_TEXT(qp,(char*) getbuf( LOOPSIZE ));
	//if( QRY_TEXT(qp) == NULL ) mem_err("open_loop");
	if( QRY_TEXT(qp) == NULL ) {
		ERROR1("open_loop");
		IOS_RETURN
	}
	SET_QRY_TXTSIZE(qp,LOOPSIZE);
	SET_QRY_TXTFREE(qp,LOOPSIZE);
	CLEAR_QRY_TEXT(qp);
	SET_QRY_FLAG_BITS(qp,Q_SAVING);
}

void fore_loop(QSP_ARG_DECL Foreach_Loop *frp)
{
	Query *qp;

	qp=(CURR_QRY(THIS_QSP));

//advise("fore_loop:  BEGIN");
//qdump(qsp);
	dup_input(SINGLE_QSP_ARG);
//advise("fore_loop:  after dup_input");
//qdump(qsp);

#define FORELOOP	(-2)

	ASSIGN_VAR(FL_VARNAME(frp),(const char *)NODE_DATA(QLIST_HEAD(FL_LIST(frp))));

	SET_QRY_COUNT(qp, FORELOOP);		/* BUG should be some unique code */
	SET_QRY_FORLOOP(qp, frp);
	SET_QRY_TEXT(qp,(char*) getbuf( LOOPSIZE ) );
//#ifdef CAUTIOUS
//	if( QRY_TEXT(qp) == NULL ) {
//		/*mem_err*/
//		ERROR1("CAUTIOUS:  fore_loop");
//		IOS_RETURN
//	}
//#endif /* CAUTIOUS */
	assert( QRY_TEXT(qp) != NULL );

	SET_QRY_TXTSIZE(qp,LOOPSIZE);
	SET_QRY_TXTFREE(qp,LOOPSIZE);
	CLEAR_QRY_TEXT(qp);
	SET_QRY_FLAG_BITS(qp,Q_SAVING);
//advise("fore_loop:  DONE");
//qdump(qsp);
}

void zap_fore(Foreach_Loop *frp)
{
	Node *np;

	while( (np=remTail(FL_LIST(frp))) != NO_NODE ){
		rls_str((const char *)NODE_DATA(np));
		rls_node(np);
	}
	rls_list(FL_LIST(frp));
	rls_str(FL_VARNAME(frp));
	givbuf(frp);
}

/*
 * Close and pop redir input file.
 * Cleans up appropriately.
 */

Query * pop_file(SINGLE_QSP_ARG_DECL)
{
	int i;
	Query *qp;

	//if( QLEVEL<=0 ){
	/* We used to not allow the popping of the zero level input.
	 * But to support iOS apps with no stdin, we want to do it...
	 */
	if( QLEVEL<0 ){
		WARN("pop_file:  no file to pop");
		return NULL;
	}

	qp = CURR_QRY(THIS_QSP);
	if( QRY_HAS_FILE_PTR(qp) ){
#ifdef HAVE_POPEN
		if( IS_PIPE(qp) ){
			Pipe *pp;
			// We used to call pclose here, but that doesn't
			// destroy the pipe struct that we created in pipes.c
			// We overload q_dupfile with a pointer to the pipe struct
			//pclose(QRY_FILE_PTR(qp));
			pp = (Pipe *)QRY_DUPFILE(qp);
			close_pipe(QSP_ARG  pp);
		} else {
#endif /* HAVE_POPEN */
			/* before we close the file, make sure it
			 * isn't a dup of a lower file for looping
			 * input
			 */
			if( QLEVEL == 0 ||
		QRY_FILE_PTR(qp) != QRY_FILE_PTR(PREV_QRY(THIS_QSP)) ){
#ifdef BUILD_FOR_OBJC
				fclose(QRY_FILE_PTR(qp));
#else // ! BUILD_FOR_OBJC
				if( QRY_FILE_PTR(qp) != tfile(SINGLE_QSP_ARG) )
					fclose(QRY_FILE_PTR(qp));
#endif // ! BUILD_FOR_OBJC
				
			}
#ifdef HAVE_POPEN
		}
#endif /* HAVE_POPEN */
	}
	SET_QRY_FILE_PTR(qp,NULL);
	//pop_input_file(SINGLE_QSP_ARG);

	/* When qlevel used to be the index of our table of query structs,
	 * this line accomplished the pop - but now we have to explicitly pop
	 * from the stack...
	 */
	pop_item( QS_QRY_STACK(THIS_QSP) );
	QLEVEL--;

	// We used to free the thenclause here...

	/* free macro args if any */

	/* macro open && not a loop in a macro */
	if( QRY_MACRO(qp) != NO_MACRO && NOT_PASSED(qp) ){
		/* exiting macro, free args */
		if( MACRO_N_ARGS(QRY_MACRO(qp)) > 0 ){
			for(i=0;i<MACRO_N_ARGS(QRY_MACRO(qp));i++){
				rls_str(QRY_ARG_AT_IDX(qp,i));
			}
#ifdef QUIP_DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s (qlevel = %d):  freeing macro args",
WHENCE_L(pop_file));
advise(ERROR_STRING);
}
#endif	/* QUIP_DEBUG */
			givbuf((char *)QRY_ARGS(qp));
			SET_QRY_ARGS(qp,NULL);
		}
	}
	/* We release if non-null in SET_QRY_FILENAME...
	 * If we want to do this here, we would need to add
	 * a check in SET_QRY_FILENAME for NULL, so we don't
	 * try to save a string at the null ptr...
	 */

	/*
	if( QRY_FILENAME(qp) != NULL ){
		rls_str(QRY_FILENAME(qp));
		SET_QRY_FILENAME(qp,NULL);
	}
	*/

	rls_query(qp);	// add to query free list

	return(qp);	// we may want to access the flags of the old thing...
			// But where do we release resources?

} /* end pop_file() */

//char *poptext(TMP_QSP_ARG_DECL  char *buf,int size,FILE* fp)
static char *poptext(QSP_ARG_DECL  void *buf,int size,void* fp)
{
	return(NULL);
}

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

void push_text(QSP_ARG_DECL const char *text, const char *filename)
{
	Query *qp, *old_qp;

//QUEUE_CHECK(push_text)
	if( QLEVEL >= 0 )
		old_qp=(CURR_QRY(THIS_QSP));
	else
		old_qp = NULL;
	redir(QSP_ARG  (FILE *)NULL, filename );
	qp=(CURR_QRY(THIS_QSP));
	SET_QRY_LINE_PTR(qp,text);
	SET_QRY_FLAG_BITS(qp,(Q_HAS_SOMETHING | Q_BUFFERED_TEXT));

	if( old_qp == NULL ){
		SET_QRY_LINENO(qp, 1 );
		SET_QRY_RDLINENO(qp, 1 );
	} else {
		SET_QRY_RDLINENO(qp, QRY_RDLINENO(old_qp) );
		SET_QRY_LINENO(qp, QRY_LINENO(old_qp) );
		// Not exactly right, but close?
	}
#ifdef QUIP_DEBUG
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
#endif /* QUIP_DEBUG */
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
	push_text(QSP_ARG  text, filename);
	mpass(QSP_ARG CURR_QRY(THIS_QSP),QRY_AT_LEVEL(THIS_QSP,QLEVEL-1));
}

COMMAND_FUNC( close_loop )
{
	Query *qp;
	Query *loop_qp;
	const char *errmsg="Can't Close, no loop open";
	const char *s;

//advise("close_loop:  BEGIN");
//qdump(qsp);

	if( QLEVEL <= 0 || QRY_COUNT(QRY_AT_LEVEL(THIS_QSP,QLEVEL-1)) == 0 ){
		WARN(errmsg);
		return;
	}

	/* the lookahead word may have popped the level already...
	 * How would we know???
	 */

	loop_qp=pop_file(SINGLE_QSP_ARG);	// are we sure we should do this?

//advise("close_loop:  after pop_file");
//qdump(qsp);

	qp=(CURR_QRY(THIS_QSP));

	CLEAR_QRY_FLAG_BITS(qp,Q_SAVING);

	if( QRY_COUNT(qp) == FORELOOP ){
		FL_NODE(QRY_FORLOOP(qp)) = NODE_NEXT(FL_NODE(QRY_FORLOOP(qp)));
		if( FL_NODE(QRY_FORLOOP(qp)) == NO_NODE ){
			zap_fore(QRY_FORLOOP(qp));	// this releases the foreach struct also
			SET_QRY_FORLOOP(qp,NULL);
			goto lup_dun;
		}
		ASSIGN_VAR(FL_VARNAME(QRY_FORLOOP(qp)),
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

	// does fullpush push the macro pointer?
	fullpush(QSP_ARG  QRY_TEXT(qp), s );

	/* This is right if we haven't finished the current line yet... */
	if( QRY_FLAGS(qp) & Q_LINEDONE )
		SET_QRY_RDLINENO(CURR_QRY(THIS_QSP),1+QRY_RDLINENO(qp));
	else
		SET_QRY_RDLINENO(CURR_QRY(THIS_QSP),QRY_RDLINENO(qp));

//advise("close_loop:  after pushing repeat");
//qdump(qsp);

	return;

lup_dun:

//advise("close_loop:  loop done...");
//qdump(qsp);

	// Now we are done with the loop - why is the line number wrong??
	givbuf(QRY_TEXT(qp));
	SET_QRY_TEXT(qp, NULL);
	SET_QRY_RDLINENO(qp, QRY_RDLINENO(loop_qp));

	/* lookahead may have been inhibited by q_count==1 */
	lookahead(SINGLE_QSP_ARG);

//advise("close_loop:  loop DONE...");
//qdump(qsp);

}



/* in an older version, the non-ansii arg decl was enclosed in #ifndef __C2MAN__ ...
 * not sure why, perhaps c2man wasn't interpreting typedefs.h properly...
 */

void _whileloop(QSP_ARG_DECL  int value)
{

	Query *qp;
	const char *errmsg="Can't close with While, no loop open";



	/* While can be used instead of Close,
	 * but first we need to make sure that a loop is open!
	 */

//#ifdef CAUTIOUS
//	if( QLEVEL < 0 ){
//		ERROR1("CAUTIOUS:  negative qlevel in While");
//		IOS_RETURN
//	}
//#endif /* CAUTIOUS */
	assert( QLEVEL >= 0 );

	if( QLEVEL == 0 || QRY_COUNT(QRY_AT_LEVEL(THIS_QSP,QLEVEL-1)) == 0 ){
		WARN(errmsg);
		return;
	}

/*
sprintf(ERROR_STRING,"while clearing save flag at level %d", qlevel-1);
advise(ERROR_STRING);
*/


	pop_file(SINGLE_QSP_ARG);
	qp=(CURR_QRY(THIS_QSP));
	CLEAR_QRY_FLAG_BITS(qp,Q_SAVING);

	if( ! value ){
/*
sprintf(ERROR_STRING,"releasing while loop text buffer at level %d",
qp-&query[0]);
advise(ERROR_STRING);
*/
		givbuf(QRY_TEXT(qp));
		SET_QRY_TEXT(qp, NULL);
		/*
		 * Return the count to the default state;
		 * Otherwise lookahead will be inhibited after this...
		 *
		 * This revealed a bug when a repeat loop followed a do loop
		 */
		SET_QRY_COUNT(qp, 0);
	} else {
		const char *s;
		//push_input_file( QSP_ARG CURRENT_FILENAME );
		s = CURRENT_FILENAME ;
		fullpush(QSP_ARG  QRY_TEXT(qp), s );
	}
}


/*
 * We remember the if-clause string so we can free
 * it at the right time
 */

void push_if(QSP_ARG_DECL const char *text)
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
	create_reserved_var(QSP_ARG  "argc",acstr);

	for(i=0;i<ac;i++){
		/* allow the cmd args to be referenced $argv1 etc, because $1 $2 don't work inside macros.
		 * BUG - we really ought to copy the shell and allow variable subscripting:  $argv[1]
		 */
		sprintf(acstr,"argv%d",i);
		create_reserved_var(QSP_ARG  acstr,av[i]);

		n_cmd_args++;
	}

	set_progname(av[0]);
}

void inhibit_next_prompt_format(SINGLE_QSP_ARG_DECL)
{
	CLEAR_QS_FLAG_BITS(THIS_QSP, QS_FORMAT_PROMPT);
}

static void show_query_level(QSP_ARG_DECL int i)
{
	Query *qp;

	if( i< 0 || i > QLEVEL ){
		sprintf(ERROR_STRING,"Requested level %d out of range 0-%d",
			i,QLEVEL);
		WARN(ERROR_STRING);
		return;
	}

	qp = QRY_AT_LEVEL(THIS_QSP,i);

	show_query_flags(QSP_ARG  qp);

	if( QRY_HAS_TEXT(qp) ){
		sprintf(ERROR_STRING,
			"\tLevel %d line buffer at 0x%lx:\n\"%s\"",i,(u_long)QRY_LINE_PTR(qp),QRY_LINE_PTR(qp));
		advise(ERROR_STRING);
	}

	if( QRY_TEXT(qp) != NULL ){
		sprintf(ERROR_STRING,
			"\tstored text:\n%s",QRY_TEXT(qp));
		advise(ERROR_STRING);
	}

	if( QRY_IS_SAVING(qp) )
		advise("\tSaving...");
	sprintf(ERROR_STRING,
		"\tCount = %d",QRY_COUNT(qp) );
	advise(ERROR_STRING);

	if( QRY_MACRO(qp) != NO_MACRO ){
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
		WARN(ERROR_STRING);
	}
#endif /* MALLOC_DEBUG */
}

#ifdef NOT_USED
void set_qflags(QSP_ARG_DECL int flag)
{
	SET_QRY_FLAG_BITS( CURR_QRY(THIS_QSP), flag );
}
#endif /* NOT_USED */

#ifndef BUILD_FOR_OBJC

/**/
/**		stuff to do with the control terminal	**/
/**/

#define NO_TTY ((FILE *)(-44))

/*
 * Return a descriptor for the control tty
 */

static FILE *ttyfile=NO_TTY;

FILE *tfile(SINGLE_QSP_ARG_DECL)
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
		WARN(ERROR_STRING);
	}

	return(ttyfile);
}
#endif // ! BUILD_FOR_OBJC

/*
 * Return a pointer to the named variable or NO_VARIABLE.
 * Works for macro args, i.e. $1 $2, by using a query_stack variable
 * tmpvar.
 *
 * This also works for the cmd line args if we are not in a macro.
 *
 * which function used to be called simple_var_of?  var__of ?
 */

Variable *var_of(QSP_ARG_DECL const char *name)
		/* variable name */
{
	int i;
	Variable *vp;
	const char *s;

	vp = var__of(QSP_ARG  name);
	if( vp != NO_VARIABLE ) return(vp);

	/* if not set, try to import from env */
	s = getenv(name);
	if( s != NULL ){
		vp = new_var_(QSP_ARG  name);
		SET_VAR_VALUE(vp,savestr(s));
		SET_VAR_FLAGS(vp,VAR_RESERVED);
		return(vp);
	}

	/* numbered variable? (macro arg. or cmd line arg.) */

	i=0;
	s=name;
	while( *s ){
		if( !isdigit(*s) ) return(NO_VARIABLE);
		i*=10;
		i+=(*s)-'0';
		s++;
	}

	i--;	/* variables start at 1, indices at 0 */

	/* first see if we're in a macro! */
	if( QRY_MACRO(CURR_QRY(THIS_QSP)) != NO_MACRO ){
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
		if( v == NULL ) return(NO_VARIABLE);

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
		if( (vp=QS_TMPVAR(THIS_QSP)) == NO_VARIABLE ){
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
			WARN(ERROR_STRING);
			return(NO_VARIABLE);
		} else {
			char varname[32];
			sprintf(varname,"argv%d",i+1);
			vp = var__of(QSP_ARG  varname);
			return(vp);
		}
	}
	/* this suppresses a lint error message about no return val... */
	/* NOTREACHED */
	/* return(NO_VARIABLE); */
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
	PUSH_MENU_PTR( BOTTOM_OF_STACK( QS_MENU_STACK(THIS_QSP) ) );
}


const char *savestr(const char *s)
{
	char *new_s;
	
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
		WARN("callbacks_on:  callbacks are already being processed");
	SET_QS_FLAG_BITS(THIS_QSP,QS_PROCESSING_CALLBACKS);
}

void callbacks_off(SINGLE_QSP_ARG_DECL)
{
	if( ! IS_PROCESSING_CALLBACKS(THIS_QSP) )
		WARN("callbacks_off:  callbacks are already being inhibited");
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
	while(np!=NO_NODE){
//		if( func == ((void (*)())NODE_DATA(np)) ){
//			sprintf(DEFAULT_ERROR_STRING,
//				"CAUTIOUS:  add_func_to_list:  function already on list!?");
//			NWARN(DEFAULT_ERROR_STRING);
//			return;
//		}
		assert( func != ((void (*)())NODE_DATA(np)) );
		np = NODE_NEXT(np);
	}
#endif // CAUTIOUS

	np = mk_node((void *) func);
	addTail(lp,np);
}


void add_event_func(QSP_ARG_DECL  void (*func)(SINGLE_QSP_ARG_DECL) )	/** set event processing function */
{
	if( QS_EVENT_LIST(THIS_QSP) == NO_LIST ){
		SET_QS_EVENT_LIST(THIS_QSP, new_list() );
	}
	add_func_to_list(QS_EVENT_LIST(THIS_QSP),func);
}

int rem_event_func(QSP_ARG_DECL  void (*func)(SINGLE_QSP_ARG_DECL) )
{
	Node *np;
	np = remData(QS_EVENT_LIST(THIS_QSP),(void *)func);
	if( np != NO_NODE ){
		rls_node(np);
		return(0);
	} else {
		return(-1);
	}
}

void add_cmd_callback(QSP_ARG_DECL  void (*func)(SINGLE_QSP_ARG_DECL) )
{
	if( QS_CALLBACK_LIST(THIS_QSP) == NO_LIST )
		SET_QS_CALLBACK_LIST(THIS_QSP, new_list() );
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
	POP_MENU;
}


