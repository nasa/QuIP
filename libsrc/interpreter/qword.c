#include "quip_config.h"

char VersionId_interpreter_qword[] = QUIP_VERSION_STRING;

/**/
/**		input and output stuff		**/
/**/


#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_SYS_FILIO_H
#include <sys/filio.h>		/* what for?  needed this on SUN... */
#endif

#include "query.h"
#include "macros.h"
#include "debug.h"
#include "items.h"
#include "savestr.h"
#include "node.h"
#include "getbuf.h"
#include "history.h"
#include "substr.h"

#ifdef DEBUG
debug_flag_t qldebug=0;
debug_flag_t lah_debug=0;
#endif /* DEBUG */

#define QUERY_LEVEL	((long)(qp-&THIS_QSP->qs_query[0]))

static int n_cmd_args=0;

/* local prototypes */


static void mpass(QSP_ARG_DECL Query *qpto,Query *qpfr);
static void dup_input(SINGLE_QSP_ARG_DECL);
#ifdef DEBUG
static void show_file_stack(SINGLE_QSP_ARG_DECL);
#endif /* DEBUG */
static void reset_fns(SINGLE_QSP_ARG_DECL);


/*
 * Set input reading function for the current level of the query stack.
 * Set input reading function for the current level of the query stack.
 * User supplied functions should have the calling syntax of fgets(3),
 * which is the default.
 */

void set_rf( QSP_ARG_DECL  char * (*func)(TMP_QSP_ARG_DECL  char *buf, int size, FILE *fp ) )
{
	if( !(QUERY_FLAGS & QS_INITED) ) init_query_stream(THIS_QSP);
	if( QLEVEL < 0 ) ERROR1("no query file");

	/* can be used to cause input to be read from a socket */
	/* why was this void * here???
	query[qlevel].q_readfunc= (void *)func;
	*/
	THIS_QSP->qs_query[QLEVEL].q_readfunc= func;
}

long get_rf(SINGLE_QSP_ARG_DECL)
{
	return((long)THIS_QSP->qs_query[QLEVEL].q_readfunc);
}

/* This function was added because a CAUTIOUS warning about a non-null thenclause
 * started appearing...
 */

static void init_query(Query *qp)
{
	qp->q_thenclause = NULL;
}

static void init_string_buf(String_Buf *sbp)
{
	sbp->sb_buf = NULL;
	sbp->sb_size = 0;
}

// Initialize a Query_Stream

void init_query_stream(Query_Stream *qsp)
{
	int i;

#ifdef CAUTIOUS
	if( qsp == NULL ) ERROR1("CAUTIOUS:  init_query_stream passed NULL query stream pointer");
#endif /* CAUTIOUS */
		
	if( qsp->qs_flags & QS_INITED ) return;
	qsp->qs_flags =	  QS_INITED
			| QS_EXPAND_MACS
			| QS_INTERACTIVE_TTYS
			| QS_FORMAT_PROMPT
			| QS_LOOKAHEAD_ENABLED
			| QS_STRIPPING_QUOTES
			| QS_COMPLETING
			;

	qsp->qs_level=(-1);
	qsp->qs_former_level=0;
	qsp->qs_fn_depth=(-1);
	qsp->qs_lookahead_level = 0;	/* BUG don't need this var, because never written!? */
	qsp->qs_which_retstr = 0;
	qsp->qs_cv_which = 0;

	for(i=0;i<MAX_VAR_BUFS;i++)
		init_string_buf(&qsp->qs_cv_buf[i]);

	init_string_buf(&qsp->qs_scratchbuf);
	init_string_buf(&qsp->qs_result);

	for(i=0;i<N_RETSTRS;i++)
		init_string_buf(&qsp->qs_retstr_arr[i]);

	qsp->qs_prompt[0]=0;
	qsp->qs_cmd_itp = NO_ITEM_TYPE;
	qsp->qs_fmt_code = FMT_DECIMAL; 

	// nexpr.y initializations
	qsp->qs_which_expr_str = 0;
	qsp->qs_edepth = -1;
	// vectree.y initialization
	qsp->qs_curr_string = qsp->qs_estr;

	qsp->qs_chewing = 0;
	qsp->qs_chew_list = NO_LIST;

	for(i=0;i<MAX_Q_LVLS;i++){
		init_query(&qsp->qs_query[i]);
	}
}

/* first_query_stream is what we call to push
 * the standard input onto the intput stack
 */

void first_query_stream(Query_Stream *qsp)
{
	push_input_file(QSP_ARG  "-");
	redir(QSP_ARG  stdin);

	ASSIGN_VAR("verbose","0");	/* have to do this somewhere... */
#ifdef DEBUG
	qldebug = add_debug_module(QSP_ARG  "query");;
	lah_debug = add_debug_module(QSP_ARG  "lookahead");;
#endif /* DEBUG */
}

#ifdef DEBUG
static void show_file_stack(SINGLE_QSP_ARG_DECL)
{
	int i;

	for(i=0;i<=THIS_QSP->qs_fn_depth;i++){
		sprintf(msg_str,"File %d:  %s",i,THIS_QSP->qs_fn_stack[i]);
		advise(msg_str);
	}
}
#endif /* DEBUG */

/* push this filename onto the stack of filenames.
 * Call this BEFORE pushing the new file.
 */

void push_input_file(QSP_ARG_DECL const char *name)
{
#ifdef CAUTIOUS
	if( THIS_QSP->qs_fn_depth != QLEVEL ){
		sprintf(ERROR_STRING,
	"CAUTIOUS:  Pushing filename \"%s\", qlevel = %d, fn_depth = %d",
			name,QLEVEL,THIS_QSP->qs_fn_depth);
		WARN(ERROR_STRING);
		show_file_stack(SINGLE_QSP_ARG);
	}
#endif /* CAUTIOUS */

/*
sprintf(msg_str,"fn_depth = %d, max = %d",fn_depth+1,MAX_Q_LVLS);
*/
	if( ++THIS_QSP->qs_fn_depth >= MAX_Q_LVLS ){
int i;
for(i=0;i<MAX_Q_LVLS;i++){
sprintf(ERROR_STRING,"\t%d\t%s",i,THIS_QSP->qs_fn_stack[i]);
advise(ERROR_STRING);
}
		ERROR1("too many filenames pushed");
	}
#ifdef CAUTIOUS
	if( THIS_QSP->qs_fn_depth < 0 )
		ERROR1("CAUTIOUS:  fn_depth < 0 !?");
#endif /* CAUTIOUS */

	THIS_QSP->qs_fn_stack[THIS_QSP->qs_fn_depth]=savestr(name);

	reset_fns(SINGLE_QSP_ARG);
}

void pop_input_file(SINGLE_QSP_ARG_DECL)
{
#ifdef CAUTIOUS
	if( THIS_QSP->qs_fn_depth < 0 ){
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,
		"CAUTIOUS:  no filename to pop (qlevel = %d  fn_depth = %d)",
			QLEVEL,THIS_QSP->qs_fn_depth);
		return;
	}

	if( THIS_QSP->qs_fn_depth != QLEVEL ){
		sprintf(ERROR_STRING,
	"CAUTIOUS:  Popping filename %s, qlevel = %d, fn_depth = %d",
			THIS_QSP->qs_fn_stack[THIS_QSP->qs_fn_depth],QLEVEL,THIS_QSP->qs_fn_depth);
		WARN(ERROR_STRING);
		show_file_stack(SINGLE_QSP_ARG);
	}
#endif /* CAUTIOUS */

	rls_str( THIS_QSP->qs_fn_stack[THIS_QSP->qs_fn_depth] );
	THIS_QSP->qs_fn_stack[THIS_QSP->qs_fn_depth] = NULL;
	THIS_QSP->qs_fn_depth--;

	reset_fns(SINGLE_QSP_ARG);
}

/* File stack string:  input filenames, or macro names separated
 * by colons.
 *
 * There is a BUG in this implementation due to lookahead popping of
 * files...  the state of the file stack when a command is executed
 * may conceal a macro or filename when it calls another
 * macro right at the end...  who cares for now...
 */


static void reset_fns(SINGLE_QSP_ARG_DECL)
{
	int i;
	int len=0;

	/* reset the filestack string */

#define INIT_STR	"-;"
#define SEMI_STR	";"


	strcpy(THIS_QSP->qs_fns_str,INIT_STR);
	len+=strlen(INIT_STR);
	for(i=0;i<=THIS_QSP->qs_fn_depth;i++){
		if( len+strlen(THIS_QSP->qs_fn_stack[i])+strlen(SEMI_STR) >= FNS_LEN ){
			ERROR1("filename stack overflow");
			return;
		}
		strcat(THIS_QSP->qs_fns_str,THIS_QSP->qs_fn_stack[i]);
		len+=strlen(THIS_QSP->qs_fn_stack[i]);
		strcat(THIS_QSP->qs_fns_str,SEMI_STR);
		len+=strlen(SEMI_STR);
	}
}

const char * current_input_stack(SINGLE_QSP_ARG_DECL)
{
	/* might create a data object without using the iterpreter? */
	if( THIS_QSP == NULL )
		return("");

	return(THIS_QSP->qs_fns_str);
}

const char * current_input_file(SINGLE_QSP_ARG_DECL)
{
	if( THIS_QSP->qs_fn_depth >= 0 ){
#ifdef CAUTIOUS
		if( THIS_QSP->qs_fn_stack[THIS_QSP->qs_fn_depth] == NULL ){
			sprintf(ERROR_STRING,
				"CAUTIOUS:  filename stack at depth %d is NULL!?",
				THIS_QSP->qs_fn_depth);
			WARN(ERROR_STRING);
		}
#endif /* CAUTIOUS */
		return(THIS_QSP->qs_fn_stack[THIS_QSP->qs_fn_depth]);
	}


	if( THIS_QSP->qs_fn_depth == -1 ){	/* is this the default state w/ stdin? */
		return("-");
	}
	return(NULL);
}

#ifdef THREAD_SAFE_QUERY

char *qpfgets( TMP_QSP_ARG_DECL char *buf, int size, FILE *fp )
{
	return( fgets(buf,size,fp) );
}
#endif /* THREAD_SAFE_QUERY */

/*
 * Read input from a file.
 * Create a new query structure for this file and push onto the top
 * of the query stack.
 */

void redir(QSP_ARG_DECL FILE *fp)
		/* file from which to redirect input */
{
	Query *qp;

	if( !(THIS_QSP->qs_flags & QS_INITED) ) init_query_stream(THIS_QSP);

	if( ++QLEVEL >= MAX_Q_LVLS ){
		QLEVEL--;
		qdump(SINGLE_QSP_ARG);
		ERROR1("too many nested files");
	}

	qp=(&THIS_QSP->qs_query[QLEVEL]);
	qp->q_file=fp;
	qp->q_lineno=0;		// redir
	qp->q_rdlineno=0;	// first words are on line 1
				// we used to pre-advance in nextline, but not now.
				// but we now advance when we call qline/nextline...
				// When we read a file at level 0, this causes a problem?
	qp->q_havtext=0;
	qp->q_saving=0;
	qp->q_count=0;
	qp->q_flags=0;
	qp->q_dupfile=NULL;

#ifdef NO_STDIO
	if( QLEVEL == 0 ) {
		qp->q_flags |= Q_INTERACTIVE;
	}
#else
#ifdef MAC
	if( QLEVEL == 0 ) {
		qp->q_flags |= Q_INTERACTIVE;
	}
#else
	if( fp != NULL ){
		/* We want to set the interactive flag if we have redirected
		 * to /dev/tty, but not if it is to a serial port slaved
		 * to another machine...  but in general we have no way
		 * of knowing whether it is a terminal or a machine at
		 * the other end of a serial line...  so we have a global
		 * flag, which can be cleared from the serial menu.
		 */
		if( isatty( fileno(qp->q_file) ) && isatty( fileno(stderr) ) &&
			(THIS_QSP->qs_flags&QS_INTERACTIVE_TTYS) )

			qp->q_flags |= Q_INTERACTIVE;
	}
#endif /* ! MAC */
#endif /* ! NO_STDIO */

#ifdef CAUTIOUS
	/* why do this check???
	 * The thenclause field is supposed to be set to null when
	 * we release the old string, if it is not null here it might mean
	 * that we forgot to release it...  but what about the first time?
	 */
	if( qp->q_thenclause != NULL ){
		sprintf(ERROR_STRING,"CAUTIOUS:  thenclause (0x%lx) was not null!? (qlevel = %d)",
			(int_for_addr)qp->q_thenclause,QLEVEL);
		WARN(ERROR_STRING);
	}
#endif
	qp->q_thenclause=NULL;
/*
sprintf(ERROR_STRING,"initializing q_text to NULL at level %d",qlevel);
advise(ERROR_STRING);
*/

	qp->q_text=NULL;
	qp->q_macro=NO_MACRO;

#ifdef THREAD_SAFE_QUERY
	if( fp != NULL ) qp->q_readfunc = qpfgets;
#else /* ! THREAD_SAFE_QUERY */
	if( fp != NULL ) qp->q_readfunc = fgets;
#endif /* ! THREAD_SAFE_QUERY */

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
#ifdef CAUTIOUS
if( qpto <= qpfr ){
WARN("CAUTIOUS:  mpass:  passing macro args down?!");
abort();
}
#endif
	if( qpfr->q_macro == NO_MACRO ) return;

	qpto->q_macro = qpfr->q_macro;
	qpto->q_args = qpfr->q_args;
	qpto->q_flags |= Q_MPASSED;
}


/*
 * This function is useful when we want to specify "-" as the input file
 */

static void dup_input(SINGLE_QSP_ARG_DECL)
{
	Query *qp;

	qp=(&THIS_QSP->qs_query[QLEVEL]);
#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"dup_input:  current qlevel = %d, duping at %d",QLEVEL,QLEVEL+1);
advise(ERROR_STRING);
sprintf(ERROR_STRING,"dup_input:  current input file is %s",current_input_file(SINGLE_QSP_ARG));
advise(ERROR_STRING);
sprintf(ERROR_STRING,"q_file = 0x%lx\nq_readfunc = 0x%lx",
(int_for_addr)qp->q_file,(int_for_addr)qp->q_readfunc);
advise(ERROR_STRING);
}
#endif /* DEBUG */

	push_input_file( QSP_ARG current_input_file(SINGLE_QSP_ARG) );
	redir( QSP_ARG qp->q_file );

	/* these two lines are so we can have within-line loops */
	(qp+1)->q_havtext=qp->q_havtext;
//sprintf(ERROR_STRING,"dup_input:  setting lbptr = 0x%lx",(long)qp->q_lbptr);
//advise(ERROR_STRING);
	(qp+1)->q_lbptr =qp->q_lbptr;
	(qp+1)->q_rdlineno =qp->q_rdlineno;
//sprintf(ERROR_STRING,"dup_input:  rdlineno set to %d",qp->q_rdlineno);
//advise(ERROR_STRING);

	/* used to copy LINEDONE bit here... */
	(qp+1)->q_flags = qp->q_flags;	// any reason not to do this?
					// BUG q_saving and q_havtxt are
					// flags that have their own
					// words!?

	/* the absence of the next line caused a subtle bug
	 * for loops within macros that were preceded by a doubly
	 * redirected file... */

	(qp+1)->q_readfunc = qp->q_readfunc;

	/* loops within macros */
	mpass(QSP_ARG qp+1,qp);
}

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

void openloop(QSP_ARG_DECL int n)
			/* loop count */
{
	Query *qp;

	qp=(&THIS_QSP->qs_query[QLEVEL]);
/*
if( n < 0 ) advise("opening do loop");
sprintf(ERROR_STRING,"saving to q_text buffer at level %d",qlevel);
advise(ERROR_STRING);
*/

//sprintf(ERROR_STRING,"openloop:  buf \"%s\"",qp->q_lbptr);
//advise(ERROR_STRING);
	dup_input(SINGLE_QSP_ARG);

	qp->q_count=n;
	qp->q_text=(char*) getbuf( LOOPSIZE );
	if( qp->q_text == NULL ) mem_err("openloop");
	qp->q_txtsiz=LOOPSIZE;
	qp->q_txtfree=LOOPSIZE;
	*qp->q_text=0;
	qp->q_saving=1;
}

void fore_loop(QSP_ARG_DECL Foreloop *frp)
{
	Query *qp;

	qp=(&THIS_QSP->qs_query[QLEVEL]);

	dup_input(SINGLE_QSP_ARG);

#define FORELOOP	(-2)

	ASSIGN_VAR(frp->f_varname,(const char *)frp->f_lp->l_head->n_data);

	qp->q_count= FORELOOP;		/* BUG should be some unique code */
	qp->q_fore = frp;
	qp->q_text=(char*) getbuf( LOOPSIZE );
#ifdef CAUTIOUS
	if( qp->q_text == NULL ) mem_err("CAUTIOUS:  fore_loop");
#endif /* CAUTIOUS */
	qp->q_txtsiz=LOOPSIZE;
	qp->q_txtfree=LOOPSIZE;
	*qp->q_text=0;
	qp->q_saving=1;
}

void zap_fore(Foreloop *frp)
{
	Node *np;

	while( (np=remTail(frp->f_lp)) != NO_NODE ){
		rls_str(np->n_data);
		rls_node(np);
	}
	rls_list(frp->f_lp);
	rls_str(frp->f_varname);
	givbuf(frp);
}

void end_fore()
{
	
}

/*
 * Close and pop redir input file.
 * Cleans up appropriately.
 */

void popfile(SINGLE_QSP_ARG_DECL)
{
	int i;
	Query *qp;

	if( QLEVEL<=0 ){
		WARN("popfile:  no file to pop");
		return;
	}

	qp = &THIS_QSP->qs_query[QLEVEL];
	if( qp->q_file != NULL ){
		if( IS_PIPE(qp) ){
			Pipe *pp;
			// We used to call pclose here, but that doesn't
			// destroy the pipe struct that we created in pipes.c
			// We overload q_dupfile with a pointer to the pipe struct
			//pclose(qp->q_file);
			pp = (Pipe *)qp->q_dupfile;
advise("closing pipe");
			close_pipe(QSP_ARG  pp);
		} else {
			/* before we close the file, make sure it
			 * isn't a dup of a lower file for looping
			 * input
			 */
			if( QLEVEL == 0 ||
		( qp->q_file != (qp-1)->q_file && qp->q_file != tfile(SINGLE_QSP_ARG) ) )
				fclose(qp->q_file);
		}
	}
	qp->q_file=NULL;
	pop_input_file(SINGLE_QSP_ARG);
	QLEVEL--;

	/* free extra text if any */
	if( qp->q_thenclause != NULL ){
#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"popfile:  freeing string \"%s\"",qp->q_thenclause);
advise(ERROR_STRING);
}
#endif /* DEBUG */
		rls_str(qp->q_thenclause);
		qp->q_thenclause = NULL;
	}

	/* free macro args if any */

	/* macro open && not a loop in a macro */
	if( qp->q_macro != NO_MACRO && NOT_PASSED(qp) ){
		/* exiting macro, free args */
		if( qp->q_macro->m_nargs > 0 ){
			for(i=0;i<qp->q_macro->m_nargs;i++)
				rls_str(qp->q_args[i]);
#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  freeing macro args",
WHENCE(popfile));
advise(ERROR_STRING);
}
#endif	/* DEBUG */
			givbuf(qp->q_args);
			qp->q_args=NULL;
		}
	}

} /* end popfile() */

char *poptext(TMP_QSP_ARG_DECL  char *buf,int size,FILE* fp)
{
	return(NULL);
}

/*
 * Push text onto the input.
 * Scan text as if it were keyboard input.
 *
 * We don't copy the text - should we?
 */

void pushtext(QSP_ARG_DECL const char *text)
{
	Query *qp, *old_qp;

	old_qp=(&THIS_QSP->qs_query[QLEVEL]);
	redir(QSP_ARG  (FILE *)NULL);
	qp=(&THIS_QSP->qs_query[QLEVEL]);
	qp->q_lbptr=text;
	qp->q_havtext=1;
	qp->q_flags |= Q_BUFFERED_TEXT;
	qp->q_lineno = old_qp->q_lineno;	// Not exactly right, but close?
	qp->q_rdlineno = old_qp->q_rdlineno;	// Not exactly right, but close?
#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  pushing at level %ld, text 0x%lx  \"%s\"",
WHENCE(pushtext),
QUERY_LEVEL,(int_for_addr)text,text);
advise(ERROR_STRING);
}
#endif /* DEBUG */
	qp->q_readfunc = poptext;
}

/* fullpush(text)		just like pushtext, but also passes up macro args */

void fullpush(QSP_ARG_DECL const char *text)
{
	Query *qp;

	/* push text & carry over macro args. */

	qp=(&THIS_QSP->qs_query[QLEVEL]);
#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  level %ld, text \"%s\"",
WHENCE(fullpush),QUERY_LEVEL + 1,text);
advise(ERROR_STRING);
}
#endif /* DEBUG */
	pushtext(QSP_ARG  text);
	mpass(QSP_ARG qp+1,qp);
}

COMMAND_FUNC( closeloop )
{
	Query *qp;
	const char *errmsg="Can't Close, no loop open";

	if( QLEVEL <= 0 || THIS_QSP->qs_query[QLEVEL-1].q_count == 0 ){
		WARN(errmsg);
		return;
	}

	qp=(&THIS_QSP->qs_query[QLEVEL]);

#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  clearing save flag at level %d",
WHENCE(closeloop),QLEVEL-1);
advise(ERROR_STRING);
}
#endif /* DEBUG */



	/* the lookahead word may have popped the level already...
	 * How would we know???
	 */

	popfile(SINGLE_QSP_ARG);
	qp--;
	qp->q_saving=0;

	if( qp->q_count < 0		/* do/while loop */
		|| --qp->q_count > 0 ){

		if( qp->q_count == FORELOOP ){
			qp->q_fore->f_np = qp->q_fore->f_np->n_next;
			if( qp->q_fore->f_np == NO_NODE ){
				zap_fore(qp->q_fore);
				qp->q_fore=NULL;
				goto lup_dun;
			}
			ASSIGN_VAR(qp->q_fore->f_varname,
				(const char *)qp->q_fore->f_np->n_data );
		}
		push_input_file( QSP_ARG current_input_file(SINGLE_QSP_ARG) );
		fullpush(QSP_ARG  qp->q_text);
		/* This is right if we haven't finished the current line yet... */
		(qp+1)->q_rdlineno = qp->q_rdlineno;
	} else {

lup_dun:

		givbuf(qp->q_text);
		qp->q_text = NULL;
		qp->q_rdlineno = (qp+1)->q_rdlineno;

		/* lookahead may have been inhibited by q_count==1 */
		lookahead(SINGLE_QSP_ARG);
	}
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

#ifdef CAUTIOUS
	if( QLEVEL < 0 )
		ERROR1("CAUTIOUS:  negative qlevel in While");
#endif /* CAUTIOUS */

	if( QLEVEL == 0 || THIS_QSP->qs_query[QLEVEL-1].q_count == 0 ){
		WARN(errmsg);
		return;
	}

/*
sprintf(ERROR_STRING,"while clearing save flag at level %d", qlevel-1);
advise(ERROR_STRING);
*/


	qp=(&THIS_QSP->qs_query[QLEVEL]);
	popfile(SINGLE_QSP_ARG);
	qp--;
	qp->q_saving=0;

	if( ! value ){
/*
sprintf(ERROR_STRING,"releasing while loop text buffer at level %d",
qp-&query[0]);
advise(ERROR_STRING);
*/
		givbuf(qp->q_text);
		qp->q_text = NULL;
		/*
		 * Return the count to the default state;
		 * Otherwise lookahead will be inhibited after this...
		 *
		 * This revealed a bug when a repeat loop followed a do loop
		 */
		qp->q_count = 0;
	} else {
		push_input_file( QSP_ARG current_input_file(SINGLE_QSP_ARG) );
		fullpush(QSP_ARG  qp->q_text);
	}
}



/*
 * We remember the if-clause string so we can free
 * it at the right time
 */

void push_if(QSP_ARG_DECL const char *text)
{
	Query *qp;
	const char *fname;

	fname=current_input_file(SINGLE_QSP_ARG) ;
	push_input_file(QSP_ARG  fname);

	fullpush(QSP_ARG  text);
	qp=(&THIS_QSP->qs_query[QLEVEL]);
	qp->q_thenclause = text;
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
	ASSIGN_VAR("argc",acstr);

	for(i=0;i<ac;i++){
		/* allow the cmd args to be referenced $argv1 etc, because $1 $2 don't work inside macros.
		 * BUG - we really ought to copy the shell and allow variable subscripting:  $argv[1]
		 */
		sprintf(acstr,"argv%d",i);
		ASSIGN_VAR(acstr,av[i]);

		n_cmd_args++;
	}

	set_progname(av[0]);
}

void inhibit_next_prompt_format(SINGLE_QSP_ARG_DECL)
{
	THIS_QSP->qs_flags &= ~QS_FORMAT_PROMPT;
}

/* Make prompt takes a query string (like "number of elements") and prepends "Enter " and appends ":  ".
 * We can inhibit this with a one-shot flag...
 */

void make_prompt(QSP_ARG_DECL char buffer[LLEN],const char* s)
{
	if( THIS_QSP->qs_flags & QS_FORMAT_PROMPT ){
		if(  s[0]  != 0 ) sprintf(buffer,PROMPT_FORMAT,s);
		else  buffer[0]=0;
	} else {
		strcpy(buffer,s);	/* BUG possible overrun error */
		THIS_QSP->qs_flags |= QS_FORMAT_PROMPT; /* this is a one-shot deal. */
	}
}

void show_query_level(QSP_ARG_DECL int i)
{
	Query *qp;

	if( i< 0 || i > QLEVEL ){
		sprintf(ERROR_STRING,"Requested level %d out of range 0-%d",
			i,QLEVEL);
		WARN(ERROR_STRING);
		return;
	}

	qp = &THIS_QSP->qs_query[i];
sprintf(ERROR_STRING,"show_query_level %s %d:  qp = 0x%lx",
THIS_QSP->qs_name,i,(int_for_addr)qp);
advise(ERROR_STRING);

	if( ! qp->q_havtext ){
		sprintf(ERROR_STRING,"%d:\t<No buffered input text>",i);
		advise(ERROR_STRING);
	} else  {
		sprintf(ERROR_STRING,"Level %d at 0x%lx:\n%s",i,(int_for_addr)qp->q_lbptr,qp->q_lbptr);
		advise(ERROR_STRING);
	}

	if( qp->q_saving ) advise("\tsaving");

	if( qp->q_text != NULL ){
		sprintf(ERROR_STRING,"stored text:\n%s",qp->q_text);
		advise(ERROR_STRING);
	}

	if( qp->q_thenclause != NULL ){
		sprintf(ERROR_STRING,
			"Then clause:\n%s",qp->q_thenclause);
		advise(ERROR_STRING);
	}

	if( qp->q_macro != NO_MACRO ){
		sprintf(ERROR_STRING,
			"In macro \"%s\"",qp->q_macro->m_name);
		advise(ERROR_STRING);
	}
}

/* Show state for debugging */
void qdump(SINGLE_QSP_ARG_DECL)
{
	int i;

	advise("");
	sprintf(ERROR_STRING,"Query stream %s:",THIS_QSP->qs_name);
	advise(ERROR_STRING);
	for( i=QLEVEL; i>= 0 ; i-- ){
		sprintf(ERROR_STRING,"Input level %d (%s):",i,THIS_QSP->qs_fn_stack[i]);
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

void set_qflags(QSP_ARG_DECL int flag)
{
	SET_Q_FLAG( THIS_QSP, flag );
}


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

	if( IS_INTERACTIVE(&THIS_QSP->qs_query[QLEVEL]) ){
		return(THIS_QSP->qs_query[QLEVEL].q_file);
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

/*
 * Return a pointer to the named variable or NO_VAR.
 * Works for macro args, i.e. $1 $2, by using a query_stream variable
 * tmpvar.
 *
 * This also works for the cmd line args if we are not in a macro. 
 *
 * which function used to be called simple_var_of?  var__of ?
 */

Var *var_of(QSP_ARG_DECL const char *name)
		/* variable name */
{
	int i;
	Var *vp;
	const char *s;

	vp = var__of(QSP_ARG  name);
	if( vp != NO_VAR ) return(vp);

	/* if not set, try to import from env */
	s = getenv(name);
	if( s != NULL ){
		vp=ASSIGN_VAR(name,s);
		return(vp);
	}

	/* numbered variable? (macro arg. or cmd line arg.) */

	i=0;
	s=name;
	while( *s ){
		if( !isdigit(*s) ) return(NO_VAR);
		i*=10;
		i+=(*s)-'0';
		s++;
	}

	i--;	/* variables start at 1, indices at 0 */

	/* first see if we're in a macro! */
	if( THIS_QSP->qs_query[THIS_QSP->qs_level].q_macro != NO_MACRO ){
		/*
		 * range checking is done in getmarg(),
		 * which returns NULL if out of range.
		 * It used to return the null string, but
		 * this was changed so that the null string
		 * could be a valid argument.
		 */
		TMPVAR.v_name = name;
		TMPVAR.v_value = getmarg(QSP_ARG  i);
		TMPVAR.v_func = NULL;
		if( TMPVAR.v_value == NULL ) return(NO_VAR);
		else return(&TMPVAR);
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
			return(NO_VAR);
		} else {
			char varname[32];
			sprintf(varname,"argv%d",i+1);
			vp = var__of(QSP_ARG  varname);
			return(vp);
		}
	}
	/* this suppresses a lint error message about no return val... */
	/* NOTREACHED */
	/* return(NO_VAR); */
} /* end var_of */

