/* The report_bug function if ifdef'd MAIL_BUGS, but has not been
 * implemented.  The idea originally was that if a user generated a lot
 * of warnings, then the implementor might like to know so that the
 * program could be designed so that this might not happen so often.
 * But in practice, lots of warnings can occur, these occurences are not
 * so much bugs as user input errors.  On the other hand, when a CAUTIOUS
 * error occurs, then that really is a program bug...
 */

#include "quip_config.h"

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>	/* strcat() */
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_TIME_H
#include <time.h>
#endif

#include "query_prot.h"
#include "query_stack.h"	// BUG?
#include "warn.h"
#include "ttyctl.h"
#ifdef NO_STDIO
#endif /* NO_STDIO */

#define MAX_EXIT_FUNCS	4

Query_Stack *default_qsp=NULL;
static const char *_progname=NULL;

/* THis is a hack - sometimes words get embedded in error msgs,
 * the words can be as long as LLEN, so the error string needs to be a bit
 * longer.  There should be a hard check against overrunning the strings BUG.
 */

//char msg_str[ERROR_STR_LEN];

static int n_exit_funcs=0;

// local prototypes needed for auto-initialization
static void tty_error1(QSP_ARG_DECL  const char *);
static void tty_prt_msg_frag(QSP_ARG_DECL  const char *);

// function ptr variables
static void (*exit_func_tbl[MAX_EXIT_FUNCS])(SINGLE_QSP_ARG_DECL);
static void (*warn_vec)(QSP_ARG_DECL  const char *)=tty_warn;
static void (*error_vec)(QSP_ARG_DECL  const char *)=tty_error1;
static void (*advise_vec)(QSP_ARG_DECL  const char *)=tty_advise;
static void (*prt_msg_frag_vec)(QSP_ARG_DECL  const char *)=tty_prt_msg_frag;

static void check_silent(SINGLE_QSP_ARG_DECL)
{
	char *s;

	if(  SILENCE_CHECKED(THIS_QSP) ) return;

	SET_QS_FLAG_BITS(THIS_QSP,QS_SILENCE_CHECKED);
	s=getenv("SILENT");
	if( s == NULL )
		CLEAR_QS_FLAG_BITS(THIS_QSP,QS_SILENT);
	else {
		if( *s == '0' )
			CLEAR_QS_FLAG_BITS(THIS_QSP,QS_SILENT);
		else
			SET_QS_FLAG_BITS(THIS_QSP,QS_SILENT);
	}
}

static int silent(SINGLE_QSP_ARG_DECL)
{
	check_silent(SINGLE_QSP_ARG);
	return( IS_SILENT(THIS_QSP) );
}

/*
 * Tell the error module the name of the program, to be printed
 * when and if an error occurs
 */

void set_progname(const char *program_name)
	/* pointer to the (static) program name */
{
	_progname=program_name;
	// can't do this here, qsp is still NULL...
	//assign_var(DEFAULT_QSP_ARG  "program_name", program_name);
}

/*
 * Return a pointer to the name of the program
 */

const char *_tell_progname(SINGLE_QSP_ARG_DECL)
{
	if( _progname == NULL ){
		warn("tell_progname():  progname not set!?");
		return("");
	}
	return(_progname);
}

/*
 * Call func(char *) in place of default warn(s)
 */


void set_warn_func(void (*func)(QSP_ARG_DECL  const char *))
		/* user supplied error function */
{
	warn_vec=func;
}

/*
 * Call func(char *) in place of default error1(s)
 */

void set_error_func(void (*func)(QSP_ARG_DECL  const char *))		/* user supplied error function */
{
	error_vec=func;
}

/*
 * Call func(char *) in place of default advise(s)
 */


void set_advise_func(void (*func)(QSP_ARG_DECL  const char *))
		/* user supplied error function */
{
	advise_vec=func;
}

/*
 * Call func(char *) in place of default prt_msg_frag(s)
 */

void set_prt_msg_frag_func(void (*func)(QSP_ARG_DECL  const char *))
		/* user supplied error function */
{
	prt_msg_frag_vec=func;
}

void set_max_warnings(QSP_ARG_DECL  int n)
{
	SET_QS_MAX_WARNINGS( THIS_QSP, n );
}

static void check_max_warnings(SINGLE_QSP_ARG_DECL)
{
	INC_QS_N_WARNINGS(THIS_QSP);

	if( QS_MAX_WARNINGS(THIS_QSP) > 0 &&
		QS_N_WARNINGS(THIS_QSP) >= QS_MAX_WARNINGS(THIS_QSP) ){

		sprintf(ERROR_STRING,"Too many warnings (%d max)",
			QS_MAX_WARNINGS(THIS_QSP));
		error1(ERROR_STRING);
	}
}

/*
 * Print warning message msg
 *
 * We'd like to print the input line number where this occurred,
 * but to do that we need a qsp?
 * To do that, we introduced another function script_warn, w/ macro WARN
 * but changed script_warn to _warn
 */

#define deliver_warning(msg)	_deliver_warning(QSP_ARG  msg)

static void _deliver_warning(QSP_ARG_DECL  const char* msg)
	/* warning message */
{
	if( ! silent(SINGLE_QSP_ARG) ){
		(*warn_vec)(QSP_ARG  msg);
	}
	check_max_warnings(SINGLE_QSP_ARG);
}

#ifdef NOT_NEEDED
int count_warnings()
{
	return(QS_N_WARNINGS(THIS_QSP));
}

void clear_warnings()
{
	QS_N_WARNINGS(THIS_QSP)=0;
}
#endif /* NOT_NEEDED */

/*
 * Print error message and exit
 */

void _error1(QSP_ARG_DECL  const char* msg)
	/* error message */
{
	(*error_vec)(QSP_ARG  msg);
    // We put the while loop here to indicate to the static analyzer
    // that error1 does not return...
    // but in fact, on iOS it can return, and in fact
    // it has to in order for a fatal alert to display - system stuff
    // doesn't happen until control is given back to the system...
    //
#ifndef BUILD_FOR_IOS
	while(1) ;  // silence compiler
#endif // ! BUILD_FOR_IOS
}

/*
 * Print advisory message to errfile.  File pointer errfile defaults
 * to stderr, but may be reset with error_redir().
 */

void _advise(QSP_ARG_DECL  const char* msg)
	/* Advisory message */
{
	if( ! silent(SINGLE_QSP_ARG) )
		(*advise_vec)(QSP_ARG  msg);
}

// BUG not thread safe, because uses a single static string

#define DATE_STRING_LEN	32

const char *get_date_string(SINGLE_QSP_ARG_DECL)
{
	time_t timeval;
	char *s;
	static char buf[DATE_STRING_LEN];	// must be at least 26

	time(&timeval);

#ifdef HAVE_GMTIME_R
	if( DISPLAYING_UTC( THIS_QSP ) ) {
		struct tm tm1, *tm_p;
		tm_p = gmtime_r(&timeval,&tm1);
		s = asctime_r(tm_p,buf);
	} else {
		s=ctime_r(&timeval,buf);
		// BUG?  check return value?
	}
#else // ! HAVE_GMTIME_R
#ifdef HAVE_CTIME_R
	s=ctime_r(&timeval,buf);
#else // ! HAVE_CTIME_R
#ifdef HAVE_CTIME
	s=ctime(&timeval);
#endif // HAVE_CTIME
#endif // ! HAVE_CTIME_R
#endif // ! HAVE_GMTIME_R

	assert( strlen(s) < DATE_STRING_LEN );

	/* erase trailing newline... */
	s[ strlen(s)-1 ] = '\0';
	return(s);
}

void _log_message(QSP_ARG_DECL  const char *msg)
{
	const char *log_time;

	log_time = get_date_string(SINGLE_QSP_ARG);

	if( HAS_PREV_LOG_MSG(THIS_QSP) ){
		if( !strcmp(msg,PREV_LOG_MSG(THIS_QSP)) ){
			INCREMENT_LOG_MSG_COUNT(THIS_QSP);
			return;
		} else {
			unsigned long c;
			if( (c=LOG_MSG_COUNT(THIS_QSP)) > 1 ){
				sprintf(ERROR_STRING,
"%s:  previous message was repeated %ld time%s", log_time,c-1,c==2?"":"s");
				advise(ERROR_STRING);
			}
		}
		rls_str(PREV_LOG_MSG(THIS_QSP));
	} else {
		SET_QS_FLAG_BITS(THIS_QSP,QS_HAS_PREV_LOG_MSG);
	}
	sprintf(ERROR_STRING,"%s:  %s", log_time,msg);
	advise(ERROR_STRING);
	SET_PREV_LOG_MSG(THIS_QSP,savestr(msg));
	SET_LOG_MSG_COUNT(THIS_QSP,1);
}




// BUG not thread-safe
// BUG - doesn't work if the input string is too long...
// Should convert to use resizable string_buf's...
#define PRINTABLE_LEN	(8*LLEN)
static char printable_str[PRINTABLE_LEN];

/* We call show_unprintable to make sure that we can see everything in
 * a string with unprintable characters.  These are represented as \xxx
 * octal escapes...  We do not escape things like line feeds, because if we
 * put them in a message string it is because we want them to be printed literally.
 * Similarly, what about backslashes?  We don't want to have all backslashes doubled,
 * although this would be necessary if we were going to invert this transformation...
 */

static const char *show_unprintable(QSP_ARG_DECL  const char* s)
{
	char *to;
	const char *fr;

	fr=s;
	to=printable_str;

	if( strlen(s) >= PRINTABLE_LEN ){
		sprintf(ERROR_STRING,
	"show_unprintable:  input string length (%ld) is greater than buffer size (%d)!?",
			(long) strlen(s), PRINTABLE_LEN );
		warn(ERROR_STRING);
		//return(s);		/* print a warning here? */
		return("<string too long>");
	}

	/* BUG we aren't making sure that we don't overrun printable_str!? */
	while(*fr){
		if( isascii(*fr) ){
			if( isprint(*fr) || isspace(*fr) ){
				/* Don't escape backslashes */
				/*
				if( *fr == '\\' )
					*to++ = '\\';
				*/
				*to++ = *fr;
			} else {
//advise("show_unprintable expanding a non-printing char...");
				*to++ = '\\';
				*to++ = '0' + (((*fr)>>6)&0x3);
				*to++ = '0' + (((*fr)>>3)&0x7);
				*to++ = '0' + (((*fr)>>0)&0x7);
			}
		} else {
			// assume UTF8???
			// are there any unprintable UTF8 characters???
			*to++ = *fr;
		}
		fr++;
	}
	*to = 0;

	return(printable_str);
}

int string_is_printable(const char *s)
{
	while( *s ){
		if( ! isprint(*s) ) return(0);
		s++;
	}
	return(1);
}
	
char *show_printable(QSP_ARG_DECL  const char* s)
{
	char *to;
	const char *fr;

	fr=s;
	to=printable_str;

	while(*fr){
		if( isprint(*fr) ){
			*to++ = *fr;
		} else if( isspace(*fr) ){
			if( *fr == '\n' ){
				*to++ = '\\';
				*to++ = 'n';
			} else if( *fr == '\r' ){
				*to++ = '\\';
				*to++ = 'r';
			} else if( *fr == '\t' ){
				*to++ = '\\';
				*to++ = 't';
			} else if( *fr == '\b' ){
				*to++ = '\\';
				*to++ = 'b';
			} else {
				*to++ = *fr;
			}
		} else {
advise("show_printable expanding a control char...");
			if( *fr == '\\' ){
				*to++ = '\\';
				*to++ = *fr;
			} else {
				*to++ = '\\';
				*to++ = '0' + (((*fr)>>6)&0x3);
				*to++ = '0' + (((*fr)>>3)&0x7);
				*to++ = '0' + (((*fr)>>0)&0x7);
			}
		}
		fr++;
	}
	*to = 0;

	return(printable_str);
}

// _prt_msg is define'd to be c_prt_msg ???

void _prt_msg(QSP_ARG_DECL  const char* msg)
{
	const char *printable;

	printable = show_unprintable(QSP_ARG  msg);

	prt_msg_frag(printable);
	prt_msg_frag("\n");
}

void _prt_msg_frag(QSP_ARG_DECL  const char* msg)
{
	if( ! silent(SINGLE_QSP_ARG) ){
		// BUG the vector should be part of qsp!
		(*prt_msg_frag_vec)(QSP_ARG  msg);
	}
}

/*
 * Cause void func(void) to be called on exit
 * The functions are called in the order which the calls to do_on_exit
 * are made.  The number is limited to MAX_EXIT_FUNCS (currently 4).
 * Return value 0 if successful, -1 if too many exit functions.
 */

int _do_on_exit(QSP_ARG_DECL  void (*func)(SINGLE_QSP_ARG_DECL))
{
	if( n_exit_funcs >= MAX_EXIT_FUNCS ){
		warn("too many exit functions requested");
		return(-1);
	}
	exit_func_tbl[n_exit_funcs++] = func;
	return(0);
}

#define call_exit_funcs() _call_exit_funcs(SINGLE_QSP_ARG)

static void _call_exit_funcs(SINGLE_QSP_ARG_DECL)
{
	int i;

	for(i=0;i<n_exit_funcs;i++){
		(*exit_func_tbl[i])(SINGLE_QSP_ARG);
	}
}

/*
 * Call user exit functions, then exit
 *
 * We use the already_exiting flag to avoid infinite recursion if
 * an error occurs in an exit func
 */

void _nice_exit(QSP_ARG_DECL  int status)
		/* exit status */
{
	static int already_exiting=0;

	if( ! already_exiting ){
		already_exiting=1;
		call_exit_funcs();
	}

	exit(status);
}

/*
 * Cause error messages, warnings, and advisories to be printed to fp
 * instead of stderr or previous default.  If previous value is not
 * stderr, the file will be closed.
 */

void error_redir(QSP_ARG_DECL  FILE *fp)
     /* file pointer for messages */
{
#ifndef NO_STDIO
	if( QS_ERROR_FILE(THIS_QSP) != NULL && QS_ERROR_FILE(THIS_QSP) != stderr )
#else
	if( QS_ERROR_FILE(THIS_QSP) != NULL )
#endif
		fclose(QS_ERROR_FILE(THIS_QSP));

	SET_QS_ERROR_FILE(THIS_QSP,fp);
}

FILE *_tell_msgfile(SINGLE_QSP_ARG_DECL)
{
#ifndef NO_STDIO
	if( QS_MSG_FILE(THIS_QSP) == NULL )
		SET_QS_MSG_FILE(THIS_QSP,stdout);
#else
	if( QS_MSG_FILE(THIS_QSP) == NULL ) warn("null msgfile - no stdio!??");
#endif
	return(QS_MSG_FILE(THIS_QSP));
}

FILE *_tell_errfile(SINGLE_QSP_ARG_DECL)
{
#ifndef NO_STDIO
	if( QS_ERROR_FILE(THIS_QSP) == NULL )
		SET_QS_ERROR_FILE(THIS_QSP,stderr);
#else
	if( QS_ERROR_FILE(THIS_QSP) == NULL ) warn("null errfile - no stdio!??");
#endif
	return(QS_ERROR_FILE(THIS_QSP));
}

void output_redir(QSP_ARG_DECL  FILE *fp)
     /* file pointer for messages */
{
#ifndef NO_STDIO
	if( QS_MSG_FILE(THIS_QSP) != NULL && QS_MSG_FILE(THIS_QSP) != stdout )
#else
	if( QS_MSG_FILE(THIS_QSP) != NULL )
#endif
	{
		fclose(QS_MSG_FILE(THIS_QSP));
	}
	SET_QS_MSG_FILE(THIS_QSP,fp);
}

#ifdef MAC
/* used on mac because console window disappears when program exits */

void error_wait()
{
	char s[256];
	
	fprintf(stderr,"hit return to exit\n");
	scanf("%s\n",s);
}
#endif /* MAC */

static void tty_error1(QSP_ARG_DECL  const char *s1)
{
	const char *pn;
	char msg[LLEN];

	if( (pn=tell_progname()) != NULL )
		sprintf(msg,"\n%sprogram %s:  %s",ERROR_PREFIX,pn,s1);
	else
		sprintf(msg,"\n%s%s",ERROR_PREFIX,s1);

#ifndef NO_STDIO
	if( QS_ERROR_FILE(THIS_QSP)==NULL )
		SET_QS_ERROR_FILE(THIS_QSP,stderr);
#else
	if( QS_ERROR_FILE(THIS_QSP)==NULL )
		SET_QS_ERROR_FILE(THIS_QSP),tryhard("ERRORS.TXT","w"));
#endif
	fprintf(QS_ERROR_FILE(THIS_QSP),"%s\n",msg);
	fflush(QS_ERROR_FILE(THIS_QSP));

#ifdef MAIL_BUGS
	report_bug("error",msg);
#endif /* MAIL_BUGS */

#ifdef TTY_CTL
	if( QS_ERROR_FILE(THIS_QSP) == stderr ) ttynorm(fileno(stderr));
#endif /* TTY_CTL */

	// We call fatal_alert from ios_error, why needed here???
//#ifdef BUILD_FOR_IOS
//	fatal_alert(QSP_ARG  msg);
//#endif // BUILD_FOR_IOS

#ifdef MAC
	error_wait();
#endif /* MAC */

	nice_exit(1);
}

void expect_warning(QSP_ARG_DECL  const char *msg)
{
	const char *e=QS_EXPECTED_WARNING(THIS_QSP);
	if( e != NULL ){
		advise("OOPS - expect_warning called more than once!?");
		rls_str(e);
	}
	SET_QS_EXPECTED_WARNING(THIS_QSP,savestr(msg));
}

// Call this to confirm that a warning has been issued as expected

void check_expected_warning(SINGLE_QSP_ARG_DECL)
{
	if( QS_EXPECTED_WARNING(THIS_QSP) != NULL ){
		sprintf(ERROR_STRING,"Expected warning beginning with \"%s\" never issued!?",
			QS_EXPECTED_WARNING(THIS_QSP));
		warn(ERROR_STRING);
	}
}

static int is_expected(QSP_ARG_DECL  const char *warning_msg)
{
	int retval=0;
	const char *e=QS_EXPECTED_WARNING(THIS_QSP);

	if( e != NULL ){
		if( !strncmp(e,warning_msg,strlen(e)) ){
			retval=1;
			DEC_QS_N_WARNINGS(THIS_QSP);
		}
		// This is a one-shot!
		rls_str(e);
		SET_QS_EXPECTED_WARNING(THIS_QSP,NULL);
	}
	return retval;
}

static const char *get_warning_prefix(QSP_ARG_DECL  const char *warning_msg)
{
	if( is_expected(QSP_ARG  warning_msg) )
		return EXPECTED_PREFIX;
	else
		return WARNING_PREFIX;
}

static void format_warning(QSP_ARG_DECL  char *dest, const char *msg)
{
	const char *prefix;
	prefix = get_warning_prefix(QSP_ARG  msg);
	// BUG - possible buffer overrun
	sprintf(dest,"%s%s",prefix,msg);
	assert(strlen(dest)<LLEN);	// at this point, it's too late!?
}

void tty_warn(QSP_ARG_DECL  const char *warning_message)
{
	char msg_to_print[LLEN];	// BUG use String_Buf?
	format_warning(QSP_ARG  msg_to_print,warning_message);
	tty_advise(QSP_ARG  msg_to_print);

#ifdef MAIL_BUGS
	report_bug("warning",msg_to_print);
#endif /* MAIL_BUGS */
}

static void tty_prt_msg_frag(QSP_ARG_DECL  const char *s)
{
#ifndef NO_STDIO
	if( QS_MSG_FILE(THIS_QSP)==NULL ){
		SET_QS_MSG_FILE(THIS_QSP,stdout);
	}
	fprintf(QS_MSG_FILE(THIS_QSP),"%s",s);
	fflush(QS_MSG_FILE(THIS_QSP));
#endif
}

void tty_advise(QSP_ARG_DECL  const char *s)
{
#ifndef NO_STDIO
	const char *pn;

	// is this test CAUTIOUS?  why would this ever be null?
	if( QS_ERROR_FILE(THIS_QSP)==NULL )
		SET_QS_ERROR_FILE(THIS_QSP,stderr);

#ifdef BUILD_FOR_OBJC
	pn = NULL;
#else
	//pn = tell_progname();
	pn = NULL;	// Don't print the program name now,
			// because we are comparing outputs to old quip
			// which doesn't do this.
#endif /* ! BUILD_FOR_OBJC */

	if( pn != NULL )
        	fprintf(QS_ERROR_FILE(THIS_QSP),"%s:  %s\n",pn,s);
	else
        	fprintf(QS_ERROR_FILE(THIS_QSP),"%s\n",s);

        fflush(QS_ERROR_FILE(THIS_QSP));
#endif
}

#ifdef NOT_USED
/*
 * Print an error message along with identifying the program
 */

void error2(QSP_ARG_DECL  const char *progname,const char* msg)
			/* progname =user-supplied program name */
			/* msg = error message */
{
	char msgstr[LLEN];

	sprintf(msgstr,"program %s,  %s",progname,msg);
	error1(msgstr);
}

void revert_tty()
{
 	set_error_func(tty_error1);
 	set_warn_func(tty_warn);
 	set_advise_func(tty_advise);
 	set_prt_msg_frag_func(tty_prt_msg_frag);
}
#endif /* NOT_USED */

/*
 * This routine is to be used in place of perror,
 * but instead of writing to stderr, writes to error_file
 * (which is often stderr, but may be redirected within
 * a script).
 */

void _tell_sys_error(QSP_ARG_DECL  const char* s)
{
#ifdef SUN
	extern char *sys_errlist[];
#endif /* SUN */
    
	if( s != NULL && *s )
		sprintf(ERROR_STRING,"%s: ",s);
	else ERROR_STRING[0]=0;

#ifdef HAVE_ERRNO_H
#ifdef SUN
	strcat(ERROR_STRING,sys_errlist[errno]);
#else /* ! SUN */
	strcat(ERROR_STRING,strerror(errno));
#endif /* ! SUN */
#else /* ! HAVE_ERRNO_H */

	strcat(ERROR_STRING,"(don't know how to determine system error code!?)");
#endif /* ! HAVE_ERRNO_H */

	advise(ERROR_STRING);
}

static void tell_input_location( SINGLE_QSP_ARG_DECL )
{
	int n_levels_to_print;
	int *level_tbl;

	if( THIS_QSP == NULL ) return;

	if( QLEVEL < 0 ) return;	// a callback in IOS?


	/* If it's really a file (not a macro) then
	 * it's probably OK not to show the whole input
	 * stack...
	 */

	/* OLD:  Only print the filename if it's not the console input */
	// BUT it could be a command line redirect, so print it anyway
	//filename=query_filename(SINGLE_QSP_ARG);
	//if( !strcmp(filename,"-") ){
	//	return;
	//}

	level_tbl = get_levels_to_print(QSP_ARG  &n_levels_to_print);
	print_qs_levels(QSP_ARG  level_tbl, n_levels_to_print);
	givbuf(level_tbl);
}

void q_error1( QSP_ARG_DECL  const char *msg )
{
	tell_input_location(SINGLE_QSP_ARG);
	_error1(QSP_ARG  msg);
}

// _warn - print a warning, preceded by a script input location

void _warn( QSP_ARG_DECL  const char *msg )
{
	tell_input_location(SINGLE_QSP_ARG);
	deliver_warning(msg);
}

