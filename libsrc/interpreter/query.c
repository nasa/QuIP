#include "quip_config.h"

char VersionId_interpreter_query[] = QUIP_VERSION_STRING;

//#define DEBUG_LINENO

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

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_SYS_FILIO_H
#include <sys/filio.h>
#endif

#include "query.h"
#include "macros.h"
#include "debug.h"
#include "items.h"
#include "savestr.h"
#include "strbuf.h"
#include "node.h"
#include "getbuf.h"


#ifdef HAVE_HISTORY
#include "history.h"
#endif /* HAVE_HISTORY */


/* The lookahead strategy is very complex...  It is desirable
 * to try to read ahead, in order to do things like popping
 * the input at EOF:  if we continue reading past EOF in the
 * normal case we will just get the next word from the next
 * file on the input stack.  Having lookahead on, however,
 * screws up line numbering, because the line number gets advanced
 * before the last word of the previous line has been processed.
 * Perhaps this can be fixed with different timing in the checking
 * of the line number (see ../warmenu/vectree.y).
 */


/* global vars */

// BUG?  what is the sense of a "current" query stream?
// If we are single-threaded, then there is only one...
// If we are multi-threaded, then each thread has its own, and the threads
// are logically concurrent - so there is not a "current" one???
//Query_Stream *curr_qsp=NULL;

int n_active_threads=0;
Query_Stream *default_qsp=NULL;

#define DBL_QUOTE	'"'
#define SGL_QUOTE	'\''

/* local prototypes */
static void eatup_space(Query *qp);
static int exp_mac(QSP_ARG_DECL  const char *buf);
static void var_expand(QSP_ARG_DECL  String_Buf *sbp);
static char * get_varval(QSP_ARG_DECL  char **spp);

ITEM_INTERFACE_DECLARATIONS(Query_Stream,qstream)



/*
 * Toggle forcing of prompts.
 */

void tog_pmpt(SINGLE_QSP_ARG_DECL)
{
	if( QUERY_FLAGS & QS_FORCE_PROMPT ){
		QUERY_FLAGS &= ~QS_FORCE_PROMPT;
		advise("suppressing prompts for non-tty input");
	} else {
		QUERY_FLAGS |= QS_FORCE_PROMPT;
		advise("printing prompts for non-tty input");
	}
}


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
 * there is a problem with the lookahead word
 * due to loops:  the next word after the "Close"
 * gets eaten up at the wrong level;
 * should closeloop do the popfile???
 *
 * This is by fixed (?) by having closeloop call lookahead after
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

void lookahead(SINGLE_QSP_ARG_DECL)
{
	lookahead_til(QSP_ARG  THIS_QSP->qs_lookahead_level);
}

void lookahead_til(QSP_ARG_DECL  int level)
{
/*sprintf(ERROR_STRING,"lookahead_til %d:  qlevel = %d",level,THIS_QSP->qs_level); */
/*advise(ERROR_STRING); */
	THIS_QSP->qs_former_level = THIS_QSP->qs_level;

#ifdef DEBUG
if( debug & lah_debug ){
sprintf(ERROR_STRING,"lookahead at level %d",THIS_QSP->qs_level);
advise(ERROR_STRING);
}
#endif /* DEBUG */

	while(	(QUERY_FLAGS & QS_LOOKAHEAD_ENABLED)
		&& (!IS_INTERACTIVE(&THIS_QSP->qs_query[QLEVEL]) )
		&& ( (THIS_QSP->qs_query[QLEVEL].q_flags & Q_SOCKET ) == 0 )
		&& QLEVEL > level
		&& (QLEVEL<=0 || !THIS_QSP->qs_query[QLEVEL-1].q_count)

	){
		Query *qp;
		int level;

		/* do look-ahead */
#ifdef DEBUG
if( debug & lah_debug ){
sprintf(ERROR_STRING,"looking ahead, qlevel =  %d",QLEVEL);
advise(ERROR_STRING);
}
#endif /* DEBUG */

		qp=(&THIS_QSP->qs_query[QLEVEL]);
		level=QLEVEL;

		/* Eating space (and comments) here causes a problem,
		 * because the comment lines don't get saved (say when
		 * we are reading the body of a loop, and so line number
		 * reporting on subsequent iterations of the loop is messed
		 * up.  Two possible solutions:  1)  save the comment lines;
		 * or 2) save the individual lines, along with their numbers.
		 * Solution #1 would probably be simpler to implement, but slightly
		 * slower to run...
		 */

		if( qp->q_havtext ) {
//advise("lookahead_til calling eatup_space (havtext)");
			eatup_space(qp);	/* lookahead_til */
		}
		while( (QLEVEL == level) && (qp->q_havtext == 0) ){
#ifdef DEBUG
if( debug & lah_debug ){
sprintf(ERROR_STRING,"lookahead() calling nextline, qlevel =  %d",QLEVEL);
advise(ERROR_STRING);
}
#endif /* DEBUG */
			/* nextline() never pops more than one level */
//advise("lookahead_til calling nextline");
			nextline(QSP_ARG "" );	// lookahead_til
			if( qp->q_havtext ) {
//advise("lookahead_til calling eatup_space #2");
				eatup_space(qp);	/* lookahead_til */
			}
		}
		if( qp->q_havtext ){
#ifdef DEBUG
if( debug & lah_debug ){
sprintf(ERROR_STRING,"looking ahead returning with text, qlevel =  %d",QLEVEL);
advise(ERROR_STRING);
}
#endif /* DEBUG */
			return;
		}
	}

#ifdef DEBUG
if( debug & lah_debug ){
if( !(QUERY_FLAGS & QS_LOOKAHEAD_ENABLED) ){
sprintf(ERROR_STRING,"lookahead not enabled, qlevel =  %d",QLEVEL);
advise(ERROR_STRING);
} else if(IS_INTERACTIVE(&THIS_QSP->qs_query[QLEVEL]) ){
sprintf(ERROR_STRING,"no lookahead (interactive), qlevel =  %d",QLEVEL);
advise(ERROR_STRING);
} else if( (THIS_QSP->qs_query[QLEVEL].q_flags & Q_SOCKET ) != 0 ) {
sprintf(ERROR_STRING,"no lookahead (socket), qlevel =  %d",QLEVEL);
advise(ERROR_STRING);
} else if( QLEVEL <= THIS_QSP->qs_lookahead_level ) {
sprintf(ERROR_STRING,"no lookahead (lvl =  %d), qlevel =  %d",
THIS_QSP->qs_lookahead_level,QLEVEL);
advise(ERROR_STRING);
} else if ( THIS_QSP->qs_query[QLEVEL-1].q_count ) {
sprintf(ERROR_STRING,"no lookahead (count =  %d), qlevel =  %d",
THIS_QSP->qs_query[QLEVEL-1].q_count,QLEVEL-1);
advise(ERROR_STRING);
} }
#endif /* DEBUG */
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

const char * qword(QSP_ARG_DECL const char *pline)
		/* prompt */
{
	const char *buf;

	do {
		/*lookahead();*/

		buf=gword(QSP_ARG  pline);		/* read a raw word */

//if( verbose && buf != NULL ){
//sprintf(ERROR_STRING,"gword %s returned \"%s\" to qword",
//THIS_QSP->qs_name,buf);
//advise(ERROR_STRING);
//}
#ifdef THREAD_SAFE_QUERY
		if( IS_HALTING(qsp) ) return(NULL);
#endif /* THREAD_SAFE_QUERY */

#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  gword returned 0x%lx \"%s\"",
WHENCE(qword),(int_for_addr)buf,buf);
advise(ERROR_STRING);
}
#endif	/* DEBUG */

		/* at this point, the word is complete (in buf) */

#ifdef HAVE_HISTORY
		if( IS_INTERACTIVE(&THIS_QSP->qs_query[QLEVEL]) && *buf && HISTORY_FLAG ){
			add_def(QSP_ARG  pline,buf);
		}
#endif /* HAVE_HISTORY */

		// Advance the line number to match the last line read
		// before we do lookahead.  We do this before
		// macro expansion.

#ifdef DEBUG_LINENO
sprintf(ERROR_STRING,"qword:  setting lineno to rdlineno (%d), qlevel = %d; buf = '%s'",
THIS_QSP->qs_query[QLEVEL].q_rdlineno,QLEVEL,buf);
advise(ERROR_STRING);
#endif /* DEBUG_LINENO */
		THIS_QSP->qs_query[QLEVEL].q_lineno = THIS_QSP->qs_query[QLEVEL].q_rdlineno;

		/* now see if the word is a macro */

	} while( exp_mac(QSP_ARG  buf) );		/* returns 1 if a macro */

	// Not necessary?
	THIS_QSP->qs_query[QLEVEL].q_lineno = THIS_QSP->qs_query[QLEVEL].q_rdlineno;
#ifdef OLD_LOOKAHEAD
	lookahead();
#endif /* OLD_LOOKAHEAD */

//if( verbose ){
//sprintf(ERROR_STRING,"qword %s returning \"%s\"",
//THIS_QSP->qs_name,buf);
//advise(ERROR_STRING);
//}
	return(buf);
} /* end qword() */

/*
 * Force a breakout from gword().
 * Called after interrupt.
 */

void qgivup(SINGLE_QSP_ARG_DECL)
{
	QUERY_FLAGS &= ~QS_STILL_TRYING;
}

static void after_backslash(QSP_ARG_DECL  int c,char **bufp, const char **sp, char **startp, Query *qp)
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
		//int nhave;

		qp->q_rdlineno++;
#ifdef DEBUG_LINENO
sprintf(ERROR_STRING,"after_backslash (qlevel = %d):  EOL char seen, rdlineno set to %d",
QLEVEL,qp->q_rdlineno);
advise(ERROR_STRING);
#endif /* DEBUG_LINENO */

#ifdef DONT_DO_THIS_HERE
		nhave = (int) (buf-start);	/* cast for pc */
		start[nhave]=0;

		if( *s == 0 ){	/* end of line */
#ifdef DEBUG
if( debug & qldebug ){
advise("reading additional line after escaped newline");
}
#endif /* DEBUG */

			s=qline(QSP_ARG  "");		// after_backslash
			qp=(&THIS_QSP->qs_query[QLEVEL]);

			need_size += strlen(s)+16;

			if( sbp->sb_size < need_size ){
				enlarge_buffer(sbp,need_size);

				start=sbp->sb_buf;
				buf=start+nhave;
//if( verbose ){
//sprintf(ERROR_STRING,"after_backslash %s:  enlarged buffer, new start = 0x%lx",
//THIS_QSP->qs_name,(int_for_addr)start);
//advise(ERROR_STRING);
//}
			}

		}
#ifdef DEBUG
else if( debug & qldebug ){
advise("continuing to read word after escaped newline");
sprintf(ERROR_STRING,"remaining text:  \"%s\"",s);
advise(ERROR_STRING);
}
#endif /* DEBUG */

#endif /* DONT_DO_THIS_HERE */

	} else {
		*buf++ = c; /* backslash before normal char */
	}

	*sp = s;
	*bufp = buf;
	*startp = start;

}	// end after_backslash

#ifdef FOOBAR
/* return values for process_normal */

#define PN_BACKSLASH	1
#define PN_COMMENT	2
#define PN_NONWHITE	4
#define PN_HAVSOME	8
#define PN_ALLDONE	16
#define PN_NEWLINE	32
#endif /* FOOBAR */

// rd_word flags
#define RW_HAVBACK	1
#define RW_HAVSOME	2
#define RW_INQUOTE	4
#define RW_NOVAREXP	8
#define RW_INCOMMENT	16
#define RW_SAVING	32
#define RW_NWSEEN	64
#define RW_NEWLINE	128
#define RW_ALLDONE	256

static void process_normal(QSP_ARG_DECL  Query *qp, int c, char **bufp, const char **sp, int *flagp )
{
	char *buf;
	const char *s;
	int flags;

//sprintf(ERROR_STRING,"process_normal qp = 0x%lx,  0x%x ('%c')",(long)qp,c,c);
//advise(ERROR_STRING);

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
		//havback=1;
		//return PN_BACKSLASH;
		flags |= RW_HAVBACK;
	} else {
		if( /*in_quote*/ flags & RW_INQUOTE ){
			*buf++ = c;
		} else {
			if( c == '#' ){		// comment delimiter
				//in_comment=1;
				//return PN_COMMENT;
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
#ifdef Q_LINEDONE
						qp->q_flags |= Q_LINEDONE;
//sprintf(ERROR_STRING,"process_normal:  setting LINEDONE flag, qlevel = %d",
//QLEVEL);
//advise(ERROR_STRING);
#endif /* Q_LINEDONE */
					}
					//retval = PN_ALLDONE;
					flags |= RW_ALLDONE;
					goto pn_done;
				} else {
					/* This is a space, but we haven't
					 * seen any non-spaces yet.
					 * Don't copy to output buffer.
					 */
					if( c == '\n' ){
//advise("process_normal:  leading newline white space");
						//retval=PN_NEWLINE;
						flags |= RW_NEWLINE;
						goto pn_done;
					}
				}
			} else {		/* a good character */
				*buf++ = c;
				//nonwhite_seen=1;
				//retval |= PN_NONWHITE;
				flags |= RW_NWSEEN;
//advise("process_normal:  non-white seen");
			}
		} // end not in quote
		//havsome=1;
		//retval |= PN_HAVSOME;
		// Should we set this if we are in a comment?
		flags |= RW_HAVSOME;
	}

pn_done:
	*bufp = buf;
	*sp = s;
	*flagp = flags;

	//return retval;

}	// end process_normal


/*
 * Copy the next query word from the query stack's
 * line buffer (q_lbptr) into a dynamically growable
 * buffer.
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

char * rd_word(SINGLE_QSP_ARG_DECL)
{
	Query *qp;
	//int havback=0, havsome=0;

	int start_quote=0;	/* holds the value of the starting quote char,
				 * if in a quote, otherwise 0
				 */
	//int in_quote=0;		/* possibly redundant w/ start_quote... */
	int n_quotations=0;	/* to handle things like "a b c"X"x y z" */

	//int inhib_varexp=0;	/* set to 1 to inhibit variable expansion */
	//int in_comment=0;

	char *start, *buf;
	//int am_saving=0;
	//int nonwhite_seen=0;
	int flags=0;
	u_int need_size;
	const char *s;
	String_Buf *sbp;

//if( verbose ){
//sprintf(ERROR_STRING,"rd_word (%s):  scanning...",THIS_QSP->qs_name);
//advise(ERROR_STRING);
//qdump(THIS_QSP);
//}
	qp=(&THIS_QSP->qs_query[QLEVEL]);
	if( qp != (&THIS_QSP->qs_query[0]) && (qp-1)->q_saving ){
/*
sprintf(ERROR_STRING,"%s - %s:  setting am_saving at level %d",
WHENCE(rd_word),QLEVEL);
advise(ERROR_STRING);
*/
		//am_saving=1;
		flags |= RW_SAVING;
	}

	/* actually, the size needed is probably LESS than
	 * the input buffer, because backslash sequences
	 * will be reduced in length, while everything else
	 * is copied literally.
	 */


	/* BUG? - do we need to keep track of which string buffers are in use??? */
	THIS_QSP->qs_which_retstr++;
	THIS_QSP->qs_which_retstr %= N_RETSTRS;
	sbp = &THIS_QSP->qs_retstr_arr[THIS_QSP->qs_which_retstr];

	need_size = strlen(qp->q_lbptr)+16;	/* conservative estimate */
	if( sbp->sb_size < need_size ){
		enlarge_buffer(sbp, need_size);
//if( verbose ){
//sprintf(ERROR_STRING,"rd_word %s:  enlarged buffer (0x%lx), new start is 0x%lx",
//THIS_QSP->qs_name,(int_for_addr)sbp,(int_for_addr)sbp->sb_buf);
//advise(ERROR_STRING);
//}
	}

	start=sbp->sb_buf;
#ifdef DEBUG
if( debug&qldebug )
//if( verbose )
{
sprintf(ERROR_STRING,"%s - %s: reading into buffer %d at 0x%lx",
WHENCE(rd_word),
THIS_QSP->qs_which_retstr,(u_long)start);
advise(ERROR_STRING);
}
#endif /* DEBUG */

	*start=0;			/* default is "" */

	buf=start;

#ifdef Q_LINEDONE
	/* increment the line counter if the LINEDONE flag is set
	 * We do it this way so that we don't report the line number
	 * of the following line -
	 * after we see the end of a line,
	 * we need to defer incrementing the count until
	 * we've processed the commands.
 	 */
//sprintf(ERROR_STRING,"rd_word:  checking LINEDONE flag");
//advise(ERROR_STRING);
	if( qp->q_flags & Q_LINEDONE ){
		// Why only increment when buffering text???
		if( qp->q_flags & Q_BUFFERED_TEXT ){
			/* should be not just macros, but loops too? */
			qp->q_rdlineno ++;
#ifdef DEBUG_LINENO
sprintf(ERROR_STRING,"rd_word:  LINEDONE flag set, rdlineno set to %d",
qp->q_rdlineno);
advise(ERROR_STRING);
#endif /* DEBUG_LINENO */
		}
//else {
//sprintf(ERROR_STRING,"rd_word:  LINEDONE flag set, but BUFFERED_TEXT flag not set" );
//advise(ERROR_STRING);
//}
		qp->q_flags &= ~Q_LINEDONE;
	}
#endif /* Q_LINEDONE */

	/* no check for overflow in the following loop
	 * because both buffers have size LLEN
	 */

	s=qp->q_lbptr;		/* this is the read scan pointer */

#ifdef DEBUG
if( debug&qldebug )
//if( verbose )
{
sprintf(ERROR_STRING,"%s - %s:  buffer 0x%lx contains \"%s\"",
WHENCE(rd_word),(int_for_addr)s,s);
advise(ERROR_STRING);
}
#endif /* DEBUG */

	/* Eventually we will want to strip quote marks,
	 * but we don't do it right away, because if we are
	 * saving the text (as in a loop), we want to save
	 * the quote's too.
	 */

	if( *s == DBL_QUOTE || *s == SGL_QUOTE )
		start_quote = *s;
	else start_quote=0;

#ifdef DEBUG
if( debug&qldebug ){
	if( start_quote ){
sprintf(ERROR_STRING,"%s - %s:  start_quote = \"%c\"",
WHENCE(rd_word),start_quote);
advise(ERROR_STRING);
}
}
#endif /* DEBUG */

	/* If the input buffer is a recently read line, then we know it
	 * while have less than LLEN chars, but if we are reading from a macro
	 * then it could be longer...
	 */

//advise("rd_word:  scanning");
	while( *s ){		/* scan the input buffer */
		int c;

		c=(*s++);	// get the next char

#ifdef DEBUG
//if( debug&qldebug ){
//sprintf(ERROR_STRING,"%s - %s:  char = \"%c\"",
//WHENCE(rd_word),c);
//advise(ERROR_STRING);
//}
#endif /* DEBUG */

		if( /* in_quote */ flags & RW_INQUOTE ){
			/* check if the character is the closing quote mark */
			if( c == start_quote && !/*havback*/ (flags & RW_HAVBACK) ){
#ifdef DEBUG
if( debug&qldebug ){
sprintf(ERROR_STRING,"%s - %s:  matching quote seen",
WHENCE(rd_word));
advise(ERROR_STRING);
}
#endif /* DEBUG */
				//in_quote=0;
				flags &= ~RW_INQUOTE;
				n_quotations++;
			}
		} else {
			if( c == DBL_QUOTE || c == SGL_QUOTE ){
				/* this is the opening quote */
				start_quote = c;
				//in_quote=1;
				flags |= RW_INQUOTE;
#ifdef DEBUG
if( debug&qldebug ){
	if( start_quote ){
sprintf(ERROR_STRING,"%s - %s:  setting in_quote, start_quote = \"%c\"",
WHENCE(rd_word),start_quote);
advise(ERROR_STRING);
}
}
#endif /* DEBUG */
			}
		}

		/* now do something with this character */

		if( /*am_saving*/ flags & RW_SAVING ) {
//sprintf(ERROR_STRING,"rd_word:  qp = 0x%lx, saving char 0x%x ('%c')",(long)qp,c,c);
//advise(ERROR_STRING);
			savechar(QSP_ARG  qp,c);
		}

		if( /*in_comment*/ flags & RW_INCOMMENT ){
			/* skip all characters until a newline */
			if( c == '\n' ){
				// should we increment the line counter??
				//in_comment=0;
				flags &= ~RW_INCOMMENT;
			}
		} else {
			if( /*havback*/ flags & RW_HAVBACK ){		/* char follows backslash */
				after_backslash(QSP_ARG  c,&buf,&s,&start,qp);
				/* BUG maybe we shouldn't have any after escaped nl ? */
				//havsome=1;
				//havback=0;
				flags |= RW_HAVSOME;
				flags &= ~RW_HAVBACK;
			} else {
#ifdef FOOBAR
				switch( process_normal(QSP_ARG  qp,c,&buf,&s,&flags) ){
					case PN_BACKSLASH: havback=1; break;
					case PN_COMMENT: in_comment=1; break;
					case PN_NONWHITE: nonwhite_seen=1; break;
					case PN_HAVSOME: havsome=1; break;
					case PN_NONWHITE|PN_HAVSOME: nonwhite_seen=1; havsome=1; break;
					case PN_ALLDONE: goto alldone; break;
					case PN_NEWLINE:
//advise("PN_NEWLINE");
						qp->q_rdlineno ++;
						break;
					case 0: break;
					default:
						sprintf(ERROR_STRING,"Bad return value from process_normal");
						ERROR1(ERROR_STRING);
						break;
				}
#endif /* FOOBAR */
				process_normal(QSP_ARG  qp,c,&buf,&s,&flags);
				if( flags & RW_NEWLINE ){
					qp->q_rdlineno ++;
					flags &= ~RW_NEWLINE;
				}
				if( flags & RW_ALLDONE ) goto alldone;
			}
		}
	}
//advise("rd_word:  DONE scanning");

	/* We get here when we are done reading the word, either because
	 * we have run out of input or because we encountered a white
	 * space character.
	 */

alldone:

#ifdef DEBUG
/* This printout often displays garbage chars at the end of the current string ? */
//if( debug & qldebug )
//if( verbose )
//{
//sprintf(ERROR_STRING,"rd_word %s:  buffer contains \"%s\"",
//THIS_QSP->qs_name,start);
//advise(ERROR_STRING);
//sprintf(ERROR_STRING,"rd_word %s:  start = 0x%lx, buf = 0x%lx",
//THIS_QSP->qs_name,
//(u_long)start, (u_long)buf);
//advise(ERROR_STRING);
//}
#endif /* DEBUG */

	if( ! (flags & RW_NWSEEN) ){
//advise("rd_word:  no non-white chars seen, returning NULL");
		return NULL;
	}
//else advise("rd_word:  must have seen something!?");

	if( flags & RW_HAVBACK )
		advise("still have backslash at end of buffer!?");

	*buf=0;
#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  buffer at 0x%lx contains \"%s\"",
WHENCE(rd_word),(int_for_addr)start,start);
advise(ERROR_STRING);
}
#endif /* DEBUG */

#ifdef CAUTIOUS
if( strlen(start) >= (sbp->sb_size-1) ){
sprintf(ERROR_STRING,"start len = %ld, retstr size = %d",
(long)strlen(start), sbp->sb_size );
advise(ERROR_STRING);

ERROR1("CAUTIOUS too much stuff!!!");
}
#endif	/* CAUTIOUS */

	if( start_quote && (flags & RW_INQUOTE) ){
//sprintf(ERROR_STRING,"start len = %ld, retstr size = %d",
//(long)strlen(start), sbp->sb_size );
//advise(ERROR_STRING);
		WARN("rd_word:  no closing quote");
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

	/* it's important to do this before calling savetext */
	/* WHY? */
	if( flags & RW_HAVSOME ){
//sprintf(ERROR_STRING,"rd_word:  setting lbptr to \"%s\", qp = 0x%lx",s,(long)qp);
//advise(ERROR_STRING);
		qp->q_lbptr=s;	/* current text scan ptr */
	} else return(NULL);

	// We don't need to save a space, but we need to sync
	// up the lbptr's...
	if( flags & RW_SAVING ) sync_lbptrs(QSP_ARG  qp);

//	if( flags & RW_SAVING ) {
//advise("rd_word:  saving a space");
//		savetext(QSP_ARG  qp," ");
//	}

	/* strip quotes if they enclose the entire string */
	/* This is useful in vt script, but bad if we are using these routines to pass input
	 * to the vt expression parser!?
	 */
	if( (QUERY_FLAGS & QS_STRIPPING_QUOTES) && (*start==SGL_QUOTE || *start==DBL_QUOTE) && n_quotations==1 ){
		/* BUG if the first character is a backslash escaped
		 * quote, then this may do the wrong thing...
		 */

		/* start_quote should hold the right value,
		 * because we've only seen 1 quotation
		 */
#ifdef DEBUG
//if( debug & qldebug )
//if( verbose )
//{
//sprintf(ERROR_STRING,"rd_word %s:  stripping quotes from quoted string???",
//THIS_QSP->qs_name);
//advise(ERROR_STRING);
//}
#endif /* DEBUG */
		
		if( start_quote == SGL_QUOTE )
			//inhib_varexp=1;
			flags |= RW_NOVAREXP;

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
			*(start-1) = *start;
		}
	}

	/* BUG this will prevent variable expansion of lines
	 * which contain single quoted strings and vars...
	 */

	if( ! /*inhib_varexp*/ (flags & RW_NOVAREXP) ) 
		var_expand(QSP_ARG  sbp);

//if( verbose ){
//sprintf(ERROR_STRING,"rd_word %s:  returning buffer at 0x%lx:  '%s'",
//THIS_QSP->qs_name,(int_for_addr)sbp->sb_buf,sbp->sb_buf);
//advise(ERROR_STRING);
//}

	return(sbp->sb_buf);
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

	if( !(QUERY_FLAGS & QS_INITED) ) init_query_stream(THIS_QSP);

gwtop:
	need=1;
	QUERY_FLAGS |= QS_STILL_TRYING;
	qp=(&THIS_QSP->qs_query[QLEVEL]);
//sprintf(ERROR_STRING,"gword:  qlevel = %d, havtext = %d",QLEVEL,qp->q_havtext);
//advise(ERROR_STRING);
	if( !qp->q_havtext )	/* need to read more input */
	{
//if( verbose ){
//sprintf(DEFAULT_ERROR_STRING,"gword %s calling qline (qlevel = %d)",THIS_QSP->qs_name,THIS_QSP->qs_level);
//advise(DEFAULT_ERROR_STRING);
//}
		buf=qline(QSP_ARG   pline );
#ifdef THREAD_SAFE_QUERY
		if( IS_HALTING(qsp) ) return(NULL);
#endif /* THREAD_SAFE_QUERY */
	}

	qp=(&THIS_QSP->qs_query[QLEVEL]);	/* qline may pop the level!!! */

	/* why eatup_space here?
	 * If we are saving, we might like to save all the white space - especially
	 * newlines, for correct line counting.
	 */
	//eatup_space(qp);		/* gword */
	if( qp->q_havtext ){
		/* rd_word() returns non-NULL if successful */
//if(verbose){
//sprintf(ERROR_STRING,"gword %s calling rd_word (level = %d)",THIS_QSP->qs_name,THIS_QSP->qs_level);
//advise(ERROR_STRING);
//if( THIS_QSP->qs_serial > FIRST_QUERY_SERIAL ) qdump(THIS_QSP);
//}
		if( (buf=rd_word(SINGLE_QSP_ARG)) != NULL ){
#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  rd_word returned 0x%lx \"%s\"",
WHENCE(gword),
(u_long)buf,buf);
advise(ERROR_STRING);
}
#endif	/* DEBUG */
			need=0;
		} else {
#ifdef DEBUG
//if( debug & qldebug ){
//advise("gword() got nuttin from rd_word");
//}
#endif	/* DEBUG */
			qp->q_havtext=0;
		}
	}
	if( need ){
		if( QUERY_FLAGS & QS_STILL_TRYING ) goto gwtop;	/* try again */
		else return(NULL);
	}
	return(buf);
}

/*
 *	Save a single character.
 */


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
	THIS_QSP->qs_query[QLEVEL].q_havtext=0;
	THIS_QSP->qs_query[QLEVEL].q_lineno = THIS_QSP->qs_query[QLEVEL].q_rdlineno;
	return(buf);
} // end steal_line

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

#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  qlevel =  %d",
WHENCE(qline),QLEVEL);
advise(ERROR_STRING);
}
#endif /* DEBUG */
	while(1) {
		/* if the current level is out, nextline will pop 1 level */
		buf=nextline(QSP_ARG  pline);	// qline
		qp=(&THIS_QSP->qs_query[QLEVEL]);

		if( qp->q_havtext ){
			if( IS_DUPING ){
				dup_word(QSP_ARG  qp->q_lbptr);
				dup_word(QSP_ARG  "\n");
			}
			return(buf);
		}
	}
}

#ifdef HAVE_HISTORY
#ifdef TTY_CTL

void set_completion(SINGLE_QSP_ARG_DECL)
{
	if( askif(QSP_ARG  "complete commands") ){
		advise("enabling automatic command completion");
		QUERY_FLAGS |= QS_COMPLETING;
	} else {
		advise("disabling automatic command completion");
		QUERY_FLAGS &= ~QS_COMPLETING;
		sane_tty();
	}
}

#endif /* TTY_CTL */
#endif /* HAVE_HISTORY */

static void halt_stream(Query_Stream *qsp)
{
#ifdef THREAD_SAFE_QUERY

	if( qsp->qs_serial == FIRST_QUERY_SERIAL )
		nice_exit(0);
	else {
		QUERY_FLAGS |= QS_HALTING;
		QUERY_FLAGS &= ~QS_STILL_TRYING;
	}

#else /* ! THREAD_SAFE_QUERY */

	nice_exit(0);

#endif /* ! THREAD_SAFE_QUERY */
}

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
 * We return the buffer, AND set qp->q_lbptr - redundant?
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

#ifdef THREAD_SAFE_QUERY
	if( IS_HALTING(THIS_QSP) ) return(NULL);
#endif /* THREAD_SAFE_QUERY */

#ifdef DEBUG
//if( debug & qldebug ){
//sprintf(ERROR_STRING,"in nextline qlevel =  %d",QLEVEL);
//advise(ERROR_STRING);
//}
#endif /* DEBUG */
	qp=(&THIS_QSP->qs_query[QLEVEL]);

	buf=(THIS_QSP->qs_lbuf[QLEVEL]);

nltop:
#ifdef MAC
	set_mac_pmpt(pline);
#else	/* ! MAC */
	if( (_is_i=IS_INTERACTIVE(qp)) || (QUERY_FLAGS & QS_FORCE_PROMPT) ){
		/* only force prompts at level 0 */
		/* but what about redir to /dev/tty? */

		if( _is_i || QLEVEL==0 ){
			
			fputs(pline,stderr);
#ifdef HAVE_HISTORY
#ifdef TTY_CTL
			if( HISTORY_FLAG && (QUERY_FLAGS & QS_COMPLETING) ){
#ifdef DEBUG
/*
if( debug & qldebug ){
sprintf(ERROR_STRING,"calling hist_select qlevel =  %d",QLEVEL);
advise(ERROR_STRING);
}
*/
#endif /* DEBUG */
				qp=hist_select(QSP_ARG  buf,pline);
				if( qp->q_havtext ) return(qp->q_lbptr);
				else goto nltop;
			}
#endif /* TTY_CTL */
#endif /* HAVE_HISTORY */
		}
	}
#endif /* ! MAC */

#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,
"%s - %s:  calling readfunc (0x%lx), qlevel =  %d",
WHENCE(nextline),(u_long)qp->q_readfunc,QLEVEL);
advise(ERROR_STRING);
}
#endif /* DEBUG */

	/* advance the line number right before reading */
	qp->q_rdlineno++;
#ifdef DEBUG_LINENO
sprintf(ERROR_STRING,"nextline:  advanced line number to %d",qp->q_rdlineno);
advise(ERROR_STRING);
#endif /* DEBUG_LINENO */

	if( (*(qp->q_readfunc))(QSP_ARG  buf,LLEN,qp->q_file) == NULL ){
		/* this means EOF if reading with fgets()
		 * or end of a macro...
		 */
		if( QLEVEL > 0 ){	/* EOF redir file */
			popfile(SINGLE_QSP_ARG);
			return("");
		} else {
			sprintf(ERROR_STRING,"EOF on %s",THIS_QSP->qs_name);
			advise(ERROR_STRING);
			halt_stream(THIS_QSP);
			// halting master stream will exit program,
			// but other threads have to get out gracefully...
			return("");
		}
	} else if( buf[0] == '\n' || buf[0]=='\r' ) {
		/* blank line */
		goto nltop;
	} else {		/* have something */
		/* make sure that we have a complete line */
		int n;

		n=strlen(buf);
#ifdef CAUTIOUS
		/* fgets() should not read more than LLEN-1 chars... */
		if( n > LLEN-1 ) ERROR1("CAUTIOUS:  line too long");
#endif	/* CAUTIOUS */
		n--;

		if( qp->q_readfunc == FGETS && buf[n] != '\n' &&
			buf[n] != '\r' ){
			WARN("nextline:  input line not terminated by \\n or \\r");
			sprintf(ERROR_STRING,"line:  \"%s\"",buf);
			advise(ERROR_STRING);
		}
		qp->q_havtext=1;
		qp->q_lbptr=buf;
#ifdef DEBUG_LINENO
sprintf(ERROR_STRING,"nextline:  returning \"%s\" (qlevel = %d, rdlineno = %d)",
buf,QLEVEL,qp->q_rdlineno);
advise(ERROR_STRING);
#endif /* DEBUG_LINENO */
		return(buf);
	}
	/* NOTREACHED */

	/* But just to make in ANSI C++ complaint return something */

	/* OK, but this causes warnings from other compilers...
	 * Is the C++ compiler stupid or what?
	 */
	return NULL;
} // end nextline

const char *getmarg(QSP_ARG_DECL  int index)
{
	if( index < 0 || index >= THIS_QSP->qs_query[QLEVEL].q_macro->m_nargs ){
		sprintf(ERROR_STRING,
		"getmarg:  arg index %d out of range for macro %s (%d args)",
		1+index,THIS_QSP->qs_query[QLEVEL].q_macro->m_name,
		THIS_QSP->qs_query[QLEVEL].q_macro->m_nargs);
		WARN(ERROR_STRING);
		return(NULL);
	}
	return( THIS_QSP->qs_query[QLEVEL].q_args[index] );
}

void pop_it(QSP_ARG_DECL  int n)
{
#ifdef OLD_LOOKAHEAD
	/* a problem introduced by the lookahead word */
	/* this kludge assumes that pop_it only called
		from a script */

	if( QLEVEL != THIS_QSP->qs_former_level ) n--;
	/* why decrement only once, and not by (qsp->qs_former_level-QLEVEL) ?? */
#endif /* OLD_LOOKAHEAD */

	while( n-- > 0 && QLEVEL > 0 ) popfile(SINGLE_QSP_ARG);
	if( n>0 ) WARN("couldn't pop requested number of times");
}

/* return the value of the INTERACTIVE flag - input is not a file or macro */

int intractive(SINGLE_QSP_ARG_DECL)
{
#ifdef MAC
	if( QLEVEL==0 ) return(1);
	else return(0);
#else
	Query *qp;

	lookahead(SINGLE_QSP_ARG);

	if(!(QUERY_FLAGS & QS_INITED)) init_query_stream(THIS_QSP);

	qp=(&THIS_QSP->qs_query[QLEVEL]);
	return( qp->q_flags & Q_INTERACTIVE );
#endif /* ! MAC */
}

/* read in macro text */


char * rdmtext(SINGLE_QSP_ARG_DECL)
{
	const char *ms="Enter text of macro; terminate with line beginning with '.'";
	Query *qp;
	const char *s;
	/* we make a new one for each and every macro... */
	String_Buf mac_text;
	
	mac_text.sb_buf=NULL;
	mac_text.sb_size=0;

	if(!(QUERY_FLAGS & QS_INITED)) init_query_stream(THIS_QSP);

	qp = &THIS_QSP->qs_query[QLEVEL];

	if( IS_INTERACTIVE(qp) || (QUERY_FLAGS & QS_FORCE_PROMPT) ) advise(ms);

	/* lookahead line may have been read already... */

	if( qp->q_havtext ){
#ifndef Q_LINEDONE
		/* Now we leave the newline in the buffer when we read a word,
		 * to make line-counting simpler.  But we don't want it part of the
		 * macro body.
		 */
		if( *qp->q_lbptr == '\n' )
			qp->q_lbptr++;
#endif /* Q_LINEDONE */
		/* this case only arises for scripted macros,
			so don't worry about strcat'ing \n */

		/* check for a null-body macro */	

		qp->q_havtext = 0;

		if( *qp->q_lbptr != '.' )
			copy_string(&mac_text,qp->q_lbptr);
		else goto dun;
	}

	s=qline(QSP_ARG  "");
	while( *s != '.' ){
		cat_string(&mac_text,s);

		if( IS_INTERACTIVE(qp) )
			cat_string(&mac_text,"\n");	/* why? */

		s=qline(QSP_ARG  "");
	}
dun:
	qp->q_havtext=0;	/* don't read '.' */
#ifdef OLD_LOOKAHEAD
	lookahead();
#endif /* OLD_LOOKAHEAD */
	return(mac_text.sb_buf);
} /* end rdmtext */

/*
 * Redirect input to tty
 */

void readtty(SINGLE_QSP_ARG_DECL)
{ redir(QSP_ARG  tfile(SINGLE_QSP_ARG)); }


/* no macro expansion in nameof ... jbm 3/15/89 */

/*
 * Get a string from the query file.
 *
 * Get a string from the query file by calling qword().
 * Macro expansion is disabled during this call.
 * The prompt string is prefixed by "Enter " and postfixed by a colon.
 * Used to get user command arguments.
 */

const char * nameof(QSP_ARG_DECL  const char *prompt)
		/* user prompt */
{
	char pline[LLEN];
	int v;
	const char *buf;

	make_prompt(QSP_ARG  pline,prompt);

	/* turn macros off so we can enter macro names!? */

	v=QUERY_FLAGS&QS_EXPAND_MACS;		/* save current value */
	QUERY_FLAGS &= ~QS_EXPAND_MACS;
	buf=qword(QSP_ARG  pline);
	QUERY_FLAGS |= v;		/* restore macro state */
	return(buf);
}

/*
 * Get a string from the query file with macro expansion.
 *
 * Like nameof(), but macro expansion is enabled and the prompts
 * are not modified.  Used to get command words.
 */

const char * nameof2(QSP_ARG_DECL  const char *prompt)
{
	char pline[LLEN];
	const char *buf;

	strcpy(pline,prompt);
	buf=qword(QSP_ARG  pline);

	return(buf);
}

int tell_qlevel(SINGLE_QSP_ARG_DECL)
{
	return(QLEVEL);
}

void disable_stripping_quotes(SINGLE_QSP_ARG_DECL)
{
	QUERY_FLAGS &= ~QS_STRIPPING_QUOTES;
}

void enable_stripping_quotes(SINGLE_QSP_ARG_DECL)
{
	QUERY_FLAGS |= QS_STRIPPING_QUOTES;
}

static void eatup_space(Query *qp)
{
	const char *str;

	if( !qp->q_havtext ) return;

	str=qp->q_lbptr;
	if( str == (char *)NULL ) {
		NWARN("Huh???:  null line buf ptr");
		/* BUG need a good trace display !? */
		/*
		qdump();
		*/
		qp->q_havtext=0;
		return;
	}

#ifdef DEBUG_LINENO
sprintf(DEFAULT_ERROR_STRING,"eatup_space:  buffer contains \"%s\"",str);
advise(DEFAULT_ERROR_STRING);
#endif /* DEBUG_LINENO */

skipit:

	/* skip over spaces */
	while( *str && isspace( *str ) ){
		// What if file has both??
		if( *str == '\n' || *str == '\r' ){
			/* We only need to do this when we are reading from a buffer,
			 * because when we are reading directly from a file the
			 * line numbers are advanced by nextline...
			 */
			if( READING_BUFFERED_TEXT(qp) ){
				qp->q_rdlineno++;
#ifdef DEBUG_LINENO
sprintf(DEFAULT_ERROR_STRING,"eatup_space:  advanced line number to %d after seeing newline char",qp->q_rdlineno);
advise(DEFAULT_ERROR_STRING);
#endif /* DEBUG_LINENO */
			}
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

#ifdef FOOBAR
		/* We need to cound this line if we're in a macro;
		 * if it's a regular file,
		 * it's already been counted...
		 */

		//if( (qp->q_flags & Q_BUFFERED_TEXT) && 

		if( *str == '\n' || *str == '\r' ){
			if( READING_BUFFERED_TEXT(qp) ){
				qp->q_rdlineno++;
#ifdef DEBUG_LINENO
sprintf(DEFAULT_ERROR_STRING,"eatup_space:  advanced line number to %d after seeing newline char in comment line",qp->q_rdlineno);
advise(DEFAULT_ERROR_STRING);
#endif /* DEBUG_LINENO */
			}
		}
#endif /* FOOBAR */

		goto skipit;
	}

	if( *str == '\\' && isspace( *(str+1) ) ){
		str++;
		goto skipit;
	}

	if( *str == 0 ) qp->q_havtext=0;
	if( qp->q_havtext )
		qp->q_lbptr=str;
} // eatup_space

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
	if( !(QUERY_FLAGS & QS_EXPAND_MACS) ) return(0);

//if( verbose ){
//sprintf(ERROR_STRING,"exp_mac %s checking \"%s\"",THIS_QSP->qs_name,buf);
//advise(ERROR_STRING);
//}
	/* Does the buffer contain a macro name?  If not, return */
	mp=macro_of(QSP_ARG  buf);
	if( mp==NO_MACRO ) return(0);

//if( verbose ){
//sprintf(ERROR_STRING,"exp_mac expanding %s for %s",buf,THIS_QSP->qs_name);
//advise(ERROR_STRING);
//}
	/* now see if this macro has been expanded already */

	if( RECURSION_FORBIDDEN(mp) ){
		i=QLEVEL;
		while( i>=0 )
			if( THIS_QSP->qs_query[i--].q_macro == mp ){
				sprintf(ERROR_STRING,
					"Macro recursion, macro \"%s\":  ",mp->m_name);
				WARN(ERROR_STRING);
				if( verbose )
					qdump(SINGLE_QSP_ARG);
				return(0);
			}
	}

	/* All systems go - now read the arguments */
	if( mp->m_nargs > 0 ){
		args = (const char **)getbuf(mp->m_nargs * sizeof(char *));

		/* first read and store the macro arguments */
		for(i=0;i<mp->m_nargs;i++){
			const char *s;

			if( mp->m_itps != NULL ){
				Item_Type *itp;
				itp = mp->m_itps[i];
				if( itp != NO_ITEM_TYPE ){
					Item *ip;
					ip = pick_item(QSP_ARG  itp, mp->m_prompt[i]);
					if( ip != NO_ITEM )
						s=ip->item_name;
					else
						s="xyzzy"; /* BUG? put what the user actually entered? */
				} else {
					s=nameof(QSP_ARG  mp->m_prompt[i]);
				}
			} else {
				s=nameof(QSP_ARG  mp->m_prompt[i]);
			}
			if( mp->m_text != NULL )	/* don't save if no work to do */
				args[i] = savestr(s);
		}
	} else {
		args = NULL;
	}

	/* does the macro have an empty body? */
	if( mp->m_text == NULL )
		return(1);

	/* push_input_file just remembers the name... */
	//sprintf(msg_str,"Macro \"%s\"",mp->m_name);
	sprintf(msg_str,"Macro %s",mp->m_name);
	push_input_file(QSP_ARG  msg_str);

#ifdef DEBUG
if( debug&qldebug ){
sprintf(ERROR_STRING,"%s - %s:  Pushing text for macro %s, addr 0x%lx",
WHENCE(exp_mac),
mp->m_name,(int_for_addr)mp->m_text);
advise(ERROR_STRING);
}
#endif /* DEBUG */

/*sprintf(ERROR_STRING,"pushing macro %s, qlevel = %d",mp->m_name,QLEVEL);*/
/*advise(ERROR_STRING);*/

	PUSHTEXT(mp->m_text);
	qp=(&THIS_QSP->qs_query[QLEVEL]);
#ifdef CAUTIOUS
if( qp->q_flags & Q_MPASSED ){
WARN("args passed flag set!?");
abort();
}
#endif	/* CAUTIOUS */
	qp->q_macro=mp;
	qp->q_args=args;
	qp->q_count = 0;

	/* Maybe this is where we should reset the line number? */
	qp->q_rdlineno = 1;

	return(1);
}

/* var_expand - expand variables in a buffer
 * This buffer may contain multiple words, spaces, etc.
 *
 * The strategy is this:  we simply copy characters until we encounter
 * a variable delimiter ('$').  Then we try to read a variable name.
 */

#define RESULT		(THIS_QSP->qs_result)
#define SCRATCHBUF	(THIS_QSP->qs_scratchbuf)

static void var_expand(QSP_ARG_DECL  String_Buf *sbp)
{
	/* char buf[LLEN]; */
	char *sp;
	u_int n_to_copy;
	char *start;
	int backslash_previous;

	if( RESULT.sb_buf == NULL ){
#ifdef CAUTIOUS
		if( RESULT.sb_size != 0 )
			ERROR1("result = NULL but size != 0 !?!?");
#endif /* CAUTIOUS */
		enlarge_buffer(&RESULT,LLEN);
	}
	if( SCRATCHBUF.sb_buf == NULL ){
#ifdef CAUTIOUS
		if( SCRATCHBUF.sb_size != 0 )
			ERROR1("result = NULL but size != 0 !?!?");
#endif /* CAUTIOUS */
		enlarge_buffer(&SCRATCHBUF,LLEN);
	}

	*RESULT.sb_buf = 0;
	sp=sbp->sb_buf;

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
					if( n_to_copy > SCRATCHBUF.sb_size )
						enlarge_buffer(&SCRATCHBUF,n_to_copy);
					strncpy(SCRATCHBUF.sb_buf,start,n_to_copy);
					SCRATCHBUF.sb_buf[n_to_copy]=0;
					cat_string(&RESULT,SCRATCHBUF.sb_buf);
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
				if( n_to_copy > SCRATCHBUF.sb_size )
					enlarge_buffer(&SCRATCHBUF,n_to_copy);
				strncpy(SCRATCHBUF.sb_buf,start,n_to_copy);
				SCRATCHBUF.sb_buf[n_to_copy]=0;
				cat_string(&RESULT,SCRATCHBUF.sb_buf);
			}

			vv = get_varval(QSP_ARG  &sp);

			if( vv != NULL )
				cat_string(&RESULT,vv);

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
		cat_string(&RESULT,start);
	}
	/* now overwrite the argument string with the result */
	/* This is a bit inefficient... */
	copy_strbuf(sbp,&RESULT);
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

#ifdef CAUTIOUS
	if( *sp != VAR_DELIM ){
		WARN("CAUTIOUS:  get_varval:  1st char should be var_delim");
		return(NULL);
	}
#endif /* CAUTIOUS */

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
	if( val_str == NULL )
		val_str=getenv(vname);

	if( val_str != NULL ) {
		char *s;

		copy_string(&THIS_QSP->qs_cv_buf[THIS_QSP->qs_cv_which],val_str);

		s=THIS_QSP->qs_cv_buf[THIS_QSP->qs_cv_which].sb_buf;

		THIS_QSP->qs_cv_which++;
		THIS_QSP->qs_cv_which %= MAX_VAR_BUFS;

		return(s);
	}
	return(NULL);		/* no match, no expansion */
}

Query_Stream *new_query_stream(QSP_ARG_DECL  const char *name)
{
	Query_Stream *new_qsp, dummy_qs;

#ifdef THREAD_SAFE_QUERY
	if( qsp == NULL ){
		qsp=&dummy_qs;	// may be passed to new_qstream
		qsp->qs_serial = n_active_threads;
	}
#endif /* THREAD_SAFE_QUERY */

	new_qsp = new_qstream(QSP_ARG  name);

	init_query_stream(new_qsp);

	if( n_active_threads == 0 ){
		default_qsp = new_qsp;
		first_query_stream(new_qsp);	/* point this at stdin */
	}

	// We increment n_active threads here, although the thread
	// isn't created until a teeny bit later...
	new_qsp->qs_serial = n_active_threads++;

#ifdef THREAD_SAFE_QUERY
	if( new_qsp->qs_serial == FIRST_QUERY_SERIAL )
		new_qsp->qs_history=1;
	else {
		new_qsp->qs_history=0;
		setup_all_item_type_contexts(QSP_ARG  new_qsp);
	}
#endif /* THREAD_SAFE_QUERY */

	return(new_qsp);
}

void tell_input_location( SINGLE_QSP_ARG_DECL )
{
	const char *filename;
	int ql,n;
	char msg[LLEN];
	int i;
	int n_levels_to_print;
	int level_to_print[MAX_Q_LVLS];

	if( THIS_QSP == NULL ) return;

	filename=current_input_file(SINGLE_QSP_ARG);

	/* If it's really a file (not a macro) then
	 * it's probably OK not to show the whole input
	 * stack...
	 */

	/* Only print the filename if it's not the console input */
	if( !strcmp(filename,"-") ) return;

	ql = tell_qlevel(SINGLE_QSP_ARG);
	// We would like to print the macro names with the deepest one
	// last, but for cases where the macro is repeated (e.g. loops)
	// we only want to print the deepest case.
	// That makes things tricky, because we need to scan
	// from deepest to shallowest, but we want to print
	// in the reverse order...
	n_levels_to_print=1;
	level_to_print[0]=ql;
	ql--;
	i = THIS_QSP->qs_fn_depth;
	i--;
	while( i >= 0 ){
		if( strcmp(THIS_QSP->qs_fn_stack[i],filename) ){
			level_to_print[n_levels_to_print] = i;
			filename=THIS_QSP->qs_fn_stack[i];
			n_levels_to_print++;
		}
		i--;
	}
	i=n_levels_to_print-1;
	while(i>=0){
		ql=level_to_print[i];	// assume ql matches fn_level?
		filename=THIS_QSP->qs_fn_stack[ql];
		n = THIS_QSP->qs_query[ql].q_lineno;
		if( !strncmp(filename,"Macro ",6) ){
			const char *mname;
			Macro *mp;
			mname = filename+6;
			// don't use get_macro, because it prints a warning,
			// causing infinite regress!?
			mp = macro_of(QSP_ARG  mname);
#ifdef CAUTIOUS
			if( mp == NO_MACRO ){
				sprintf(ERROR_STRING,
	"CAUTIOUS:  tell_input_loc:  macro '%s' not found!?",mname);
				ERROR1(ERROR_STRING);
			}
#endif /* CAUTIOUS */
			sprintf(msg,"%s line %d (File %s, line %d):",
				filename, n, mp->m_filename, mp->m_lineno+n);
			advise(msg);
		} else {
			sprintf(msg,"%s (input level %d), line %d:",
				filename,ql,n);
			advise(msg);
		}
		i--;
	}
}

void q_warn( QSP_ARG_DECL const char *msg )
{
	tell_input_location(SINGLE_QSP_ARG);
	warn(QSP_ARG  msg);
}

void q_error1( QSP_ARG_DECL  const char *msg )
{
	tell_input_location(SINGLE_QSP_ARG);
	error1(QSP_ARG  msg);
}



