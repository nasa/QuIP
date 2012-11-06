/* The report_bug function if ifdef'd MAIL_BUGS, but has not been
 * implemented.  The idea originally was that if a user generated a lot
 * of warnings, then the implementor might like to know so that the
 * program could be designed so that this might not happen so often.
 * But in practice, lots of warnings can occur, these occurences are not
 * so much bugs as user input errors.  On the other hand, when a CAUTIOUS
 * error occurs, then that really is a program bug...
 */

#include "quip_config.h"

char VersionId_qutil_error[] = QUIP_VERSION_STRING;

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

#include "query.h"
#include "ttyctl.h"
#ifdef NO_STDIO
#endif /* NO_STDIO */

#define MAX_EXIT_FUNCS	4

#define DEFAULT_MAX_WARNINGS	-1
static int max_warnings=DEFAULT_MAX_WARNINGS;
static int n_warnings=0;

static FILE *errfile=NULL;
static FILE *msgfile=NULL;

static char *_progname=NULL;
static int _identify_self=0;

/* THis is a hack - sometimes words get embedded in error msgs,
 * the words can be as long as LLEN, so the error string needs to be a bit
 * longer.  There should be a hard check against overrunning the strings BUG.
 */

char msg_str[ERROR_STR_LEN];

static int n_exit_funcs=0;

/* local prototypes */
static void (*exit_func_tbl[MAX_EXIT_FUNCS])(void);
static void tty_error1(QSP_ARG_DECL  const char *);
static void tty_prt_msg_frag(const char *);
static void (*warn_vec)(QSP_ARG_DECL  const char *)=tty_warn;
static void (*error_vec)(QSP_ARG_DECL  const char *)=tty_error1;
static void (*advise_vec)(const char *)=tty_advise;
static void (*prt_msg_frag_vec)(const char *)=tty_prt_msg_frag;
static const char *show_unprintable(const char *);

static int silent(void)
{
	static int silence_checked=0;
	static int am_silent;
	char *s;

	if( ! silence_checked ){
		silence_checked=1;
		s=getenv("SILENT");
		if( s == NULL )
			am_silent=0;
		else {
			if( *s == '0' )
				am_silent=0;
			else
				am_silent=1;
		}
	}
	return(am_silent);
}

/*
 * Tell the error module the name of the program, to be printed
 * when and if an error occurs
 */

void set_progname(char *program_name)
	/* pointer to the (static) program name */
{
	_progname=program_name;
}

/*
 * Return a pointer to the name of the program
 */

const char *tell_progname(void)
{
	if( _progname == NULL ){
		NWARN("tell_progname():  progname not set!?");
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


void set_advise_func(void (*func)(const char *))
		/* user supplied error function */
{
	advise_vec=func;
}

/*
 * Call func(char *) in place of default prt_msg_frag(s)
 */

void set_prt_msg_frag_func(void (*func)(const char *))
		/* user supplied error function */
{
	prt_msg_frag_vec=func;
}

void set_max_warnings(int n)
{
	max_warnings = n;
}

/*
 * Print warning message msg
 *
 * We'd like to print the input line number where this occurred,
 * but to do that we need a qsp?
 * To do that, we introduced another function q_warn, w/ macro WARN
 */

void warn(QSP_ARG_DECL  const char* msg)
	/* warning message */
{
	if( ! silent() )
		(*warn_vec)(QSP_ARG  msg);

	n_warnings++;
	if( max_warnings > 0 && n_warnings >= max_warnings ){
		sprintf(ERROR_STRING,"Too many warnings (%d max)",
			max_warnings);
		error1(QSP_ARG  ERROR_STRING);
	}
}

int count_warnings()
{
	return(n_warnings);
}

void clear_warnings()
{
	n_warnings=0;
}

/*
 * Print error message and exit
 */

void error1(QSP_ARG_DECL  const char* msg)
	/* error message */
{
	(*error_vec)(QSP_ARG  msg);
}

/*
 * Print advisory message to errfile.  File pointer errfile defaults
 * to stderr, but may be reset with error_redir().
 */

void advise(const char* msg)
	/* Advisory message */
{
	if( ! silent() )
		(*advise_vec)(msg);
}

static char printable_str[LLEN];

/* We call show_unprintable to make sure that we can see everything in
 * a string with unprintable characters.  These are represented as \xxx
 * octal escapes...  We do not escape things like line feeds, because if we
 * put them in a message string it is because we want them to be printed literally.
 * Similarly, what about backslashes?  We don't want to have all backslashes doubled,
 * although this would be necessary if we were going to invert this transformation...
 */

static const char *show_unprintable(const char* s)
{
	char *to;
	const char *fr;

	fr=s;
	to=printable_str;

	if( strlen(s) >= LLEN ){
#ifdef LONG_64_BIT
		sprintf(DEFAULT_ERROR_STRING,"show_unprintable:  input string length (%ld) is greater than buffer size (%d)!?",
#else
		sprintf(DEFAULT_ERROR_STRING,"show_unprintable:  input string length (%d) is greater than buffer size (%d)!?",
#endif
			strlen(s), LLEN );
		NWARN(DEFAULT_ERROR_STRING);
		//return(s);		/* print a warning here? */
		return("<string too long>");
	}

	/* BUG we aren't making sure that we don't overrun printable_str!? */
	while(*fr){
		if( isprint(*fr) || isspace(*fr) ){
			/* Don't escape backslashes */
			/*
			if( *fr == '\\' )
				*to++ = '\\';
			*/
			*to++ = *fr;
		} else {
			*to++ = '\\';
			*to++ = '0' + (((*fr)>>6)&0x3);
			*to++ = '0' + (((*fr)>>3)&0x7);
			*to++ = '0' + (((*fr)>>0)&0x7);
		}
		fr++;
	}
	*to = 0;

	return(printable_str);
}

	
char *show_printable(const char* s)
{
	char *to;
	const char *fr;

	fr=s;
	to=printable_str;

	while(*fr){
		if( *fr == '\\' ){
			*to++ = '\\';
			*to++ = *fr;
		} else if( *fr == '\n' ){
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
		} else if( isprint(*fr) || isspace(*fr) ){
			*to++ = *fr;
		} else {
			*to++ = '\\';
			*to++ = '0' + (((*fr)>>6)&0x3);
			*to++ = '0' + (((*fr)>>3)&0x7);
			*to++ = '0' + (((*fr)>>0)&0x7);
		}
		fr++;
	}
	*to = 0;

	return(printable_str);
}

	
void prt_msg(const char* msg)
{
	const char *printable;

	printable = show_unprintable(msg);

	prt_msg_frag(printable);
	prt_msg_frag("\n");
}

void prt_msg_frag(const char* msg)
{
	if( ! silent() )
		(*prt_msg_frag_vec)(msg);
}

void error_init()
{
#ifndef NO_STDIO
	if( errfile == NULL ) errfile=stderr;
#endif /* NO_STDIO */
}

/*
 * Cause void func(void) to be called on exit
 * The functions are called in the order which the calls to do_on_exit
 * are made.  The number is limited to MAX_EXIT_FUNCS (currently 4).
 * Return value 0 if successful, -1 if too many exit functions.
 */

int do_on_exit(void (*func)(void))
{
	if( n_exit_funcs >= MAX_EXIT_FUNCS ){
		NWARN("too many exit functions requested");
		return(-1);
	}
	exit_func_tbl[n_exit_funcs++] = func;
	return(0);
}

/*
 * Call user exit functions, then exit
 */

void nice_exit(int status)
		/* exit status */
{
	int i;

	for(i=0;i<n_exit_funcs;i++){
		(*exit_func_tbl[i])();
	}

	exit(status);
}

FILE *tell_errfile()
{
#ifndef NO_STDIO
	if( errfile == NULL ) errfile=stderr;
#else
	if( errfile == NULL ) NWARN("null errfile - no stdio!??");
#endif
	return(errfile);
}

/*
 * Cause error messages, warnings, and advisories to be printed to fp
 * instead of stderr or previous default.  If previous value is not
 * stderr, the file will be closed.
 */

void error_redir(FILE *fp)
     /* file pointer for messages */
{
#ifndef NO_STDIO
	if( errfile != NULL && errfile != stderr )
#else
	if( errfile != NULL )
#endif
		fclose(errfile);
	errfile=fp;
}

FILE *tell_msgfile()
{
#ifndef NO_STDIO
	if( msgfile == NULL ) msgfile=stdout;
#else
	if( msgfile == NULL ) NWARN("null msgfile - no stdio!??");
#endif
	return(msgfile);
}

void output_redir(FILE *fp)
     /* file pointer for messages */
{
#ifndef NO_STDIO
	if( msgfile != NULL && msgfile != stdout )
#else
	if( msgfile != NULL )
#endif
		fclose(msgfile);
	msgfile=fp;
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
		sprintf(msg,"ERROR in program %s:  %s",pn,s1);
	else
		sprintf(msg,"ERROR:  %s",s1);

#ifndef NO_STDIO
	if( errfile==NULL ) errfile=stderr;
#else
	if( errfile==NULL ) errfile=tryhard("ERRORS.TXT","w");
#endif
	fprintf(errfile,"%s\n",msg);
	fflush(errfile);

#ifdef MAIL_BUGS
	report_bug("error",msg);
#endif /* MAIL_BUGS */

#ifdef TTY_CTL
	if( errfile == stderr ) ttynorm(fileno(stderr));
#endif /* TTY_CTL */

#ifdef MAC
	error_wait();
#endif /* MAC */

	nice_exit(1);
}

void tty_warn(QSP_ARG_DECL  const char *s)
{
	char msg[LLEN];

	/* NO - use q_warn or WARN ... */
	/* tell_input_location(); */		/* print line number */

	sprintf(msg,"WARNING:  %s",s);
	tty_advise(msg);

#ifdef MAIL_BUGS
	report_bug("warning",msg);
#endif /* MAIL_BUGS */
}

static void tty_prt_msg_frag(const char *s)
{
#ifndef NO_STDIO
	if( msgfile==NULL ) msgfile=stdout;
	fprintf(msgfile,"%s",s);
	fflush(msgfile);
#endif
}

void identify_self(int flag)
{
	_identify_self = flag;
}

void tty_advise(const char *s)
{
#ifndef NO_STDIO
	if( errfile==NULL ) errfile=stderr;

	if( _identify_self )
        	fprintf(errfile,"%s:  %s\n",tell_progname(),s);
	else
        	fprintf(errfile,"%s\n",s);

        fflush(errfile);
#endif
}

/*
 * Print an error message along with identifying the program
 */

void error2(QSP_ARG_DECL  const char *progname,const char* msg)
			/* progname =user-supplied program name */
			/* msg = error message */
{
	char msgstr[LLEN];

	sprintf(msgstr,"program %s,  %s",progname,msg);
	ERROR1(msgstr);
}

void revert_tty()
{
 	set_error_func(tty_error1);
 	set_warn_func(tty_warn);
 	set_advise_func(tty_advise);
 	set_prt_msg_frag_func(tty_prt_msg_frag);
}

/*
 * This routine is to be used in place of perror,
 * but instead of writing to stderr, writes to error_file
 * (which is often stderr, but may be redirected within
 * a script).
 */

void tell_sys_error(const char* s)
{
#ifdef SUN
	extern char *sys_errlist[];
#endif /* SUN */

	if( s != NULL && *s )
		sprintf(DEFAULT_ERROR_STRING,"%s: ",s);
	else DEFAULT_ERROR_STRING[0]=0;

#ifdef SUN
	strcat(DEFAULT_ERROR_STRING,sys_errlist[errno]);
#else /* ! SUN */
	strcat(DEFAULT_ERROR_STRING,strerror(errno));
#endif /* ! SUN */

	NADVISE(DEFAULT_ERROR_STRING);
}

