
#include "quip_config.h"

#ifdef HAVE_HISTORY
#ifdef TTY_CTL


#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

//#ifdef HAVE_ERRNO_H
#include <errno.h>
//#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_LIBIO_H
#include <libio.h>
#endif

#ifdef _STDIO_USES_IOSTREAM
#define FRcnt(f) (((f)->_IO_read_end - (f)->_IO_read_ptr) > 0 ? (f)->_IO_read_end - (f)->_IO_read_ptr : 0 )
#else
/* not sure what system uses this... */
//#define FRcnt(f) ((f)->_cnt)
#define FRcnt(f) ((f)->_r)
#endif

#ifdef HAVE_TERMCAP
// On the mac, termcap stuff uses curses?
#ifdef HAVE_CURSES_H
#include <curses.h>
#endif /* HAVE_CURSES_H */
#ifdef HAVE_TERM_H
#include <term.h>
#endif /* HAVE_TERM_H */
#endif /* HAVE_TERMCAP */


#include "quip_prot.h"
#include "ttyctl.h"
#include "history.h"
#include "sigpush.h"

#ifdef QUIP_DEBUG
static u_long comp_debug=0;
#endif /* QUIP_DEBUG */

// BUG another global...

/* global BUG not thread-safe */
// However, it shouldn't be possible to have multiple output terminals...
static const char *h_bpmpt="";

static int exit_func_set=0;

static int ers_char=0,kill_char=0;
static const char *so, *se, *ce;
#define TYPING_BUFFER_SIZE	1024
static char typing_buffer[TYPING_BUFFER_SIZE];

static int hint_pushed=0;

#define UP_ARROW	0xa00
#define DN_ARROW	0xb00
#define RT_ARROW	0xc00
#define LF_ARROW	0xd00
#define IS_ARROW(c)	( (c==UP_ARROW) || (c==DN_ARROW) || (c==RT_ARROW) || (c==LF_ARROW) )

#define ESC		033
#define BELL		7		/* ^G		*/
#define FWD_CHAR	6		/* ^F forward	*/
#define CYC_CHAR	14		/* ^N next	*/


#define NO_TERMCAP	((char *)0)

typedef struct completion_data {
	const char *	selection_string;
	u_int		n_so_far;
	FILE *		tty_out;
	FILE *		tty_in;
	int 		edit_mode/* =0 */;
	const char *	prompt;
	char		chars_typed[LLEN];
	int		had_intr;
} Completion_Data;

#define THIS_SELECTION			((cdp)->selection_string)
#define N_SO_FAR			((cdp)->n_so_far)
#define TTY_OUT				((cdp)->tty_out)
#define TTY_IN				((cdp)->tty_in)
#define EDIT_MODE			((cdp)->edit_mode)
#define SET_EDIT_MODE(cdp,val)		((cdp)->edit_mode) = val
#define PROMPT				((cdp)->prompt)
#define CHARS_TYPED			((cdp)->chars_typed)


#define ERASE_ONE_CHAR							\
									\
									\
	erase_completion(cdp);							\
	UNDISPLAY_CHAR							\
	N_SO_FAR--;							\
	/*								\
	 * control chars are printed w/ 2 chars				\
	 * ^A ^B etc							\
	 */								\
	if( iscntrl(CHARS_TYPED[N_SO_FAR]) ){				\
		fputc('\b',TTY_OUT);					\
		fputc(' ',TTY_OUT);					\
		fputc('\b',TTY_OUT);					\
	}								\
	CHARS_TYPED[N_SO_FAR]=0;


#define BEGIN_EDIT					\
							\
			strcpy(edit_string,THIS_SELECTION);	\
			SET_EDIT_MODE(cdp,1);

/* local prototypes */

static void	init_tty_chars(SINGLE_QSP_ARG_DECL);
static int	pnt_strlen(const char *);

/* BUG - should be per query stream?? */
static char *simulated_input=NULL;

static int pnt_strlen(const char *s)
{
	int n=0;

	while(*s){
		if( iscntrl(*s) ) n++;
		n++;
		s++;
	}
	return(n);
}

void tty_reset(FILE *tty)
{
	ttynorm(fileno(tty));
	echoon(fileno(tty));
}

void sane_tty(SINGLE_QSP_ARG_DECL)	/** call this before exiting */
{
	FILE *fp;

	if( verbose )
		advise("Resetting tty to sane state");

	fp=tfile(SINGLE_QSP_ARG);
	tty_reset(fp);
}

static void erase_completion( Completion_Data *cdp )
{
	int i,n;

	if( *THIS_SELECTION ){
		n=pnt_strlen(THIS_SELECTION);
		i=n;
		while(i--) fputc(' ',TTY_OUT);
		i=n;
		while(i--) fputc('\b',TTY_OUT); /* backspace */
	}
}

static void show_char(FILE *fp, int c)
{
	if( iscntrl(c) ){
		fputc('^',fp);
		if( ( c ) ==0177 ) fputc('?',fp);
		else fputc(c+'A'-1,fp);
	} else fputc(c,fp);	/* do the echo */
}

/* show_completion
 *
 * Print the string THIS_SELECTION from the current point.
 * If curflag is set, we print the first char, and then
 * go into standout mode.  Otherwise, we go into standout
 * mode for the first character.
 *
 * After we've printed the string, we backspace over what has been printed.
 */

static void show_completion( Completion_Data *cdp, int curflag )
{
	int i;
	const char *s;

	s=THIS_SELECTION;

	// skip the characters already typed...
	for(i=0;i<N_SO_FAR;i++)
		s++;

	if( *s ){
		if( curflag ){
			show_char(TTY_OUT,*s++);
			if( *so ) fputs(so,TTY_OUT);
		} else {
			fputs(so,TTY_OUT);
			show_char(TTY_OUT,*s++);
		}
		while( *s )
			show_char(TTY_OUT,*s++);

		if( *se ) fputs(se,TTY_OUT);
		if( *ce ) fputs(ce,TTY_OUT);
		i=pnt_strlen(THIS_SELECTION);
		i -= pnt_strlen(CHARS_TYPED);
		assert(i>=0);
		while(i--) fputc('\b',TTY_OUT); /* backspace */
	}
}

#ifdef SGI
#define MAX_SAVE_CHARS	256
static int saved_chars[MAX_SAVE_CHARS];
static int n_saved=0;
static int next_read=0;
static int next_save=0;

void save_keystroke(int c)
{
	saved_chars[next_save]=c;
	next_save++;
	if( next_save >= MAX_SAVE_CHARS ) next_save=0;
	n_saved++;
}

int get_keystroke()
{
	int c;

	/* c=getc(tty_in); */

	while( n_saved == 0 )
		;

	c = saved_chars[next_read++];
	if( next_read >= MAX_SAVE_CHARS ) next_read=0;
	n_saved--;
	return(c);
}
#endif /* SGI */

/* this is like push_text(), but we use it when we're interactive,
 * to allow keyboard OR mouse generated text input.
 */

void simulate_typing(const char *str)
{

/* A persistent bug in the Mac implementation has led to this code.
 * Sometimes, for reasons unknown, it is as if a key is stuck.  This
 * can be fixed by doing a hard reset from the menu bar of the terminal
 * app.  But the errant keystrokes are coming in via X keypress events,
 * eventually entering this routine.  Why is the event not cleared?
 * And why is an X keypress event generated for keystrokes outside
 * of the X11 window?
 */

//fprintf(stderr,"simulate_typeing \"%s\"\n",str);
	if( simulated_input != NULL ){
		/* concatenate this string onto the unprocessed simulated input */

		/* first make sure there's enough space */
		if( (long)(strlen(simulated_input)+strlen(str)+1) > (TYPING_BUFFER_SIZE-(simulated_input-typing_buffer)) ){
			NWARN("typing buffer overflow");
			return;
		}

		strcat(simulated_input,str);
	} else {
		/* BUG make sure str is not too long! */
		strcpy(typing_buffer,str);
		simulated_input = typing_buffer;
	}
}

int keyboard_hit(QSP_ARG_DECL  FILE *tty_in)
{
	int ready=0;

	/* if there are buffered chars, we're ready */

	if( FRcnt(tty_in) > 0 ){
		ready=1;
	}

	else {
#ifdef FIONREAD
		/* check for chars typed but not in stdio buf */

	/* This used to be long, but on IA64, long is 64 bits.
	 * So we changed it to int... there should be an include
	 * file that tells us what to do here?
	 */
		int n=1;

		static int try_fionread=1;

		if( try_fionread &&
			ioctl(fileno(tty_in),FIONREAD,&n) < 0 ){

			if( errno == EINVAL ){  /* not supported? */
				/* when does this happen? */
NWARN("ioctl FIONREAD - EINVAL");
				try_fionread=0;
				ready=1;
			} else {
				tell_sys_error("ioctl (FIONREAD check_events)");
				ERROR1("FIX ME");
			}
		}
		if( n>0 ){
			ready=1;
		}
#else /* ! FIONREAD */
		ready=1;
#endif /* ! FIONREAD */
	}
	return(ready);
} /* end keyboard_hit */

/* wait for something to be typed, calling user event handlers
 * while nothing has been typed...
 *
 * This used to be a static function, but if we are going to
 * call it externally, we probably ought to give it a more sensible name.
 */

void check_events(QSP_ARG_DECL  FILE *tty_in)
{
	int ready=0;
	/* simulated_input=NULL; */

	while( !ready ){
		call_event_funcs(SINGLE_QSP_ARG);

		if( simulated_input != NULL && *simulated_input != 0 ){
			return;
		}

		ready = keyboard_hit(QSP_ARG  tty_in);


		/* When we compile for profiling, we find that
		 * we hang deep inside Xlib checking for events...
		 * This stops when we put a printf near the
		 * event checking, which prints out a lot!?
		 * Maybe this is due to calling the thing too fast
		 * (without the printing)...  There is really no
		 * reason to call this thing more than 10 or 20 times
		 * per second, all it does is make the process hog
		 * a cpu even when nothing is happening.
		 * Therefore, we will sleep here for 100 msec
		 */
		usleep(50000);
	}
} /* end check_events */

void hist_bis(const char *pmpt)
{
	h_bpmpt=pmpt;
}

#ifdef HAVE_TERMCAP
static int fetch_termcap_entry(SINGLE_QSP_ARG_DECL)
{
	char *s;
	static char tc_ent[1024];
	int stat;

	s=getenv("TERM");
	if( s==NULL ){
		WARN("init_tty_chars:  no TERM in environment");
		so="";
		ce="";
		se="";
		return -1;
	}

	stat=tgetent(tc_ent,s);
	if( stat!=1 ){
		if( stat==(-1) ){
			WARN("init_tty_chars:  can't open termcap file");
		} else if( stat==0 ){
			sprintf(ERROR_STRING,
			"no termcap entry for terminal \"%s\"",s);
			WARN(ERROR_STRING);
		} else {
			WARN("unrecognized error status from tgetent()");
		}
		return -1;
	}
	return 0;
}
#endif /* HAVE_TERMCAP */


static void init_tty_chars(SINGLE_QSP_ARG_DECL)
{
#ifdef HAVE_TERMCAP
	char *tptr;
	extern char *tgetstr();
	static char tbuf[32];

	if( fetch_termcap_entry(SINGLE_QSP_ARG) < 0 )
		return;

	tptr=tbuf;
	so=tgetstr("so",&tptr);
	if( so==NO_TERMCAP ){
		WARN("no standout string in termcap");
		so="";
	}
	se=tgetstr("se",&tptr);
	if( se==NO_TERMCAP ){
		WARN("no standend string in termcap");
		se="";
	}
	ce=tgetstr("ce",&tptr);
	if( ce==NO_TERMCAP ){
		WARN("no clear-to-eol string in termcap");
		ce="";
	}
#else /* ! HAVE_TERMCAP */
	so="";
	se="";
	ce="";
#endif /* ! HAVE_TERMCAP */
}

static void insure_tty_chars(SINGLE_QSP_ARG_DECL)
{
	static int have_tty_chars=0;

	if( !have_tty_chars ){
		init_tty_chars(SINGLE_QSP_ARG);
		have_tty_chars=1;
	}
}


static int next_character(FILE *tty_in)
{
	int c;

	if( simulated_input != NULL ){
		/* keystrokes in a viewer window */
		c= (*simulated_input);
		simulated_input++;
		if( *simulated_input == 0 ) simulated_input=NULL;
	} else {
		/* keystrokes in the console window */

		/* had get_keystroke() ifdef SGI??? */
		c=getc(tty_in);
	}
	return(c);
}



	/* the pc does its own echo */

#define UNDISPLAY_CHAR							\
									\
	fputc('\b',TTY_OUT);						\
	fputc(' ',TTY_OUT);						\
	fputc('\b',TTY_OUT);


/* BUG globals are not thread-safe */
// BUT we probably will never have two threads interactive at the same time???
static char edit_string[LLEN];

#define IS_PICKING_ITEM		QS_PICKING_ITEM_ITP(THIS_QSP) != NULL

static void insure_special_chars(FILE *fp)
{
	if( ers_char == 0 ){
		ers_char = get_erase_chr(fileno(fp));
		if( ers_char == -1 ){
			NWARN("Erase character defaulting to ^H");
			ers_char='\b';
		}
	}

	if( kill_char == 0 ){
		kill_char = get_kill_chr(fileno(fp));
		if( kill_char == -1 ){
			NWARN("Kill character defaulting to ^U");
			kill_char='';
		}
	}
}

static const char * handle_completed_line(QSP_ARG_DECL  int c,Completion_Data *cdp)
{
	/* we do show_completion a second time
		since the cursor will move! */
	if( THIS_SELECTION != NULL ){
		if( strlen(THIS_SELECTION) > N_SO_FAR )
			show_completion(cdp,0);
	} else {
		THIS_SELECTION = "";
	}

	fputc(c,TTY_OUT);	/* do the echo */

	/* Add response to history lists...
	 * We do this in next_query_word() also,
	 * to save values for suppressed prompts
	 */

	if( EDIT_MODE ){
		if( *PROMPT && *edit_string )
			add_def(QSP_ARG  PROMPT,edit_string);
		return(edit_string);
	} else {
		if( *PROMPT && THIS_SELECTION != NULL && *THIS_SELECTION ){
			add_def(QSP_ARG  PROMPT,THIS_SELECTION);
		}

		// Store the newline too...
		if( N_SO_FAR >= (LLEN-1) ){
			WARN("too many input chars!?");
		} else {
			//CHARS_TYPED[N_SO_FAR++] = c;
			// We do this after returning from get_response_from_user.
			// that way we get THIS_SELECTION also!
		}
		CHARS_TYPED[N_SO_FAR] = 0;
		// should we add a newline to THIS_SELECTION?
		return(THIS_SELECTION);
	}
} // handle_completed_line

static void check_for_completion(QSP_ARG_DECL  Completion_Data *cdp)
{
	u_int l;
	if( IS_PICKING_ITEM ){
		// we are picking an item...
		THIS_SELECTION=find_partial_match(QSP_ARG  QS_PICKING_ITEM_ITP(THIS_QSP),CHARS_TYPED);
		l=strlen(THIS_SELECTION);
		if( l == 0 ) THIS_SELECTION=CHARS_TYPED;
		if( l > N_SO_FAR ){
			show_completion(cdp,1);
		}
	} else {	// not picking an item
		THIS_SELECTION=get_match(QSP_ARG  PROMPT,CHARS_TYPED);
		l=strlen(THIS_SELECTION);

		/* We only want to check against builtins
		 * if we're parsing a command, not if we're
		 * fetching an argument;
		 * so we assume that the convention is that
		 * command prompts always end in "> "
		 */

#define IS_COMMAND_PROMPT(s)	(!strcmp((s)+strlen((s))-2,"> "))

		/* If we don't have a match yet, and we are
		 * looking for a command, then try to match
		 * against the builtin menu.
		 */

		if( l == 0 && IS_COMMAND_PROMPT(PROMPT) ) {
			THIS_SELECTION=get_match(QSP_ARG  h_bpmpt,CHARS_TYPED);
			l=strlen(THIS_SELECTION);
		}

		if( l == 0 ) THIS_SELECTION=CHARS_TYPED;

		if( l > N_SO_FAR ){
			// We have found a match from the history list...
			show_completion(cdp,1);
		} else {	/* nothing typed yet... */
			/* get the match to reset the current history list,
			 * but don't show it if nothing has been typed.
			 * The user can see the defaults with ^N.
			 */
			THIS_SELECTION=get_match(QSP_ARG  PROMPT,CHARS_TYPED);
			THIS_SELECTION=CHARS_TYPED;
		}
	}
} // someting typed

static int handle_escape_sequence(QSP_ARG_DECL  Completion_Data *cdp)
{
	int c;

	/* what if the user simply typed ESC??? */
	/* here we should see if we have more characters... */
	check_events(QSP_ARG  TTY_IN);
	c = next_character(TTY_IN);
	if( c == '[' ){		/* what we expect for arrow keys */
		check_events(QSP_ARG  TTY_IN);
		c = next_character(TTY_IN);
		if( c == 'A' )
			c = UP_ARROW;
		else if( c == 'B' )
			c = DN_ARROW;
		else if( c == 'C' )
			c = RT_ARROW;
		else if( c == 'D' )
			c = LF_ARROW;
		else {
	sprintf(ERROR_STRING,"Unexpected arrow key char seen:  0%o !?",c);
	WARN(ERROR_STRING);
		}
	} else {
		sprintf(ERROR_STRING,"Unexpected char 0%o seenm after escape",c);
		WARN(ERROR_STRING);
		return -1;
	}
	return c;
}

static int handle_arrow_key(QSP_ARG_DECL  int c, Completion_Data *cdp)
{
	if( c == UP_ARROW ){
		while( N_SO_FAR ){
			ERASE_ONE_CHAR
		}
		erase_completion(cdp);
		THIS_SELECTION=cyc_match(QSP_ARG  CHARS_TYPED,CYC_FORWARD);
		if( strlen(THIS_SELECTION) > N_SO_FAR )
			show_completion(cdp,1);
		BEGIN_EDIT
		return -1;
	} else if( c == DN_ARROW ){
		while( N_SO_FAR ){
			ERASE_ONE_CHAR
		}
		erase_completion(cdp);
		/* BUG need to cycle the opposite way */
		THIS_SELECTION=cyc_match(QSP_ARG  CHARS_TYPED,CYC_BACKWARD);
		if( strlen(THIS_SELECTION) > N_SO_FAR )
			show_completion(cdp,1);
		BEGIN_EDIT
		return -1;
	} else if( c == RT_ARROW ){
		/* check and see if there are any right chars to move over! */
		/* THIS_SELECTION holds the edit string */
		if( !EDIT_MODE ){
			BEGIN_EDIT
		}
		if( N_SO_FAR < strlen(edit_string) ){
			show_char(TTY_OUT,edit_string[N_SO_FAR]);
			N_SO_FAR ++;
		}
		/* else maybe should beep or something */

		/* BUG need to move the cursor */
		return -1;
	} else if( c == LF_ARROW ){
		if( !EDIT_MODE ){
			BEGIN_EDIT
		}
		/* make sure that there are some chars to move over */
		if( N_SO_FAR > 0 ){
			N_SO_FAR --;
			fputc('\b',TTY_OUT);
		}
		/* maybe should beep or something */
		return -1;
	}
	return 0;
}

static void handle_eof(int c, Completion_Data *cdp)

{		/* ^D */
	/* echo it first */
	if( c== EOF ){
		// This never seems to be executed???
		fprintf(TTY_OUT,"kbd EOF\n");
	} else
		fprintf(TTY_OUT,"^D\n");
}

static void handle_interrupt(QSP_ARG_DECL  Completion_Data *cdp)
{
	/*
	 * interrupts block
	 * until the last command finishes
	 */
#ifdef QUIP_DEBUG
if( debug & comp_debug ) advise("resetting tty after interrupt");
#endif /* QUIP_DEBUG */
	tty_reset(TTY_OUT);
	putc('\n',TTY_OUT);

	if( hint_pushed ){
#ifdef QUIP_DEBUG
if( debug & comp_debug ) advise("popping history interrupt handler");
#endif /* QUIP_DEBUG */
		sigpop(SIGINT);
		hint_pushed=0;
	}
#ifdef QUIP_DEBUG
if( debug & comp_debug ) advise("sending SIGINT to process id");
#endif /* QUIP_DEBUG */

	kill( getpid(), SIGINT );
}

static void accept_this_word(Completion_Data *cdp)
{
	if( strlen(THIS_SELECTION) > N_SO_FAR ){
		fputs(so,TTY_OUT);
		while( THIS_SELECTION[N_SO_FAR] &&
			!isspace(THIS_SELECTION[N_SO_FAR]) ){
			CHARS_TYPED[N_SO_FAR]=THIS_SELECTION[N_SO_FAR];
			show_char(TTY_OUT,CHARS_TYPED[N_SO_FAR]);
			N_SO_FAR++;
		}
		/*
		 * Adding a single space here may cause
		 * a match to be missed if the match
		 * text has more than one space,
		 * but since this choice
		 * should be available with ^N,
		 * we don't consider this a bug.
		 */
		fputs(se,TTY_OUT);
		CHARS_TYPED[N_SO_FAR++]=' ';	/* space */
		fputc(' ',TTY_OUT);
		CHARS_TYPED[N_SO_FAR]=0;
	} else fputc(BELL,TTY_OUT);
}

static void my_insert_character(int c, Completion_Data *cdp)
{
	char tmp_tail[LLEN];

	erase_completion(cdp);

	strcpy(tmp_tail,&edit_string[N_SO_FAR]);
	edit_string[N_SO_FAR]=c;
	N_SO_FAR++;
	strcpy(&edit_string[N_SO_FAR],tmp_tail);
	show_char(TTY_OUT,c);
	show_completion(cdp,1);
}

static void append_character(QSP_ARG_DECL  int c, Completion_Data *cdp)
{
	show_char(TTY_OUT,c);

	// what if there is no THIS_SELECTION???
	if( THIS_SELECTION != NULL ){
		if( c != THIS_SELECTION[N_SO_FAR] )
			erase_completion(cdp);
	}
	if( N_SO_FAR >= (LLEN-1) ){
		WARN("too many input chars!?");
	} else {
		CHARS_TYPED[N_SO_FAR++] = c;
	}
	CHARS_TYPED[N_SO_FAR] = 0;
}

static void handle_normal_character(QSP_ARG_DECL  int c, Completion_Data *cdp)
{
	/* just a regular character.
	 * If we are in edit mode, we insert the character at the cursor location.
	 * Otherwise, we add this character, and check for a new completion...
	 *
	 * BUG:  edit mode is badly broken!?
	 */
	if( EDIT_MODE ){
		my_insert_character(c,cdp);
	} else {
		append_character(QSP_ARG  c, cdp);
	}
}

static int check_special_char(QSP_ARG_DECL  int c, Completion_Data *cdp)
{
	if( c== ers_char ){
		if( N_SO_FAR > 0 ){
			ERASE_ONE_CHAR
		} else fputc(BELL,TTY_OUT);
		return 1;
	} else if( c == kill_char ){
		while( N_SO_FAR ){
			ERASE_ONE_CHAR
		}
		return 1;
	} else if( c==FWD_CHAR || c=='\t' ){ /* ^F or TAB accept word but don't end line */
		accept_this_word(cdp);
		return 1;
	} else if( c == CYC_CHAR ){	/* ^N cycles default */
		/*
		 * Since we don't pass the prompt to cyc_match,
		 * this will not cycle between
		 * regular menu commands and builtins.
		 * Might be considered a BUG?
		 */
		erase_completion(cdp);
		THIS_SELECTION=cyc_match(QSP_ARG  CHARS_TYPED,CYC_FORWARD);
		if( strlen(THIS_SELECTION) > N_SO_FAR )
			show_completion(cdp,1);
		return 1;
	}
	return 0;
}

static void init_completion_data(Completion_Data *cdp,const char *prompt,FILE *tty_out,FILE *tty_in)
{
	cdp->selection_string = "";
	cdp->prompt = prompt;
	cdp->tty_out = tty_out;
	cdp->tty_in = tty_in;
	cdp->n_so_far = 0;
	cdp->edit_mode = 0;
	cdp->had_intr = 0;
	cdp->chars_typed[0] = 0;
}

const char *get_response_from_user( QSP_ARG_DECL  const char *prompt, FILE *tty_in, FILE *tty_out )
{
	int c;
	// This has to be static because it is returned...
	// We assume we don't have to worry about thread safety for interactive...
	static struct completion_data _this_completion;

#ifdef QUIP_DEBUG
if( comp_debug <= 0 ) comp_debug=add_debug_module(QSP_ARG  "completion");
#endif /* QUIP_DEBUG */

	/* BUG need to check here that tty_in, tty_out
	 * are really tty's!!
	 */
	assert( tty_in != NULL );
	assert( tty_out != NULL );

	insure_tty_chars(SINGLE_QSP_ARG);

	init_completion_data(&_this_completion,prompt,tty_out,tty_in);

	if( !exit_func_set ){
		do_on_exit(sane_tty);
		exit_func_set=1;
	}

	ttycbrk(fileno(tty_in));
	echooff(fileno(tty_in));

	insure_special_chars(tty_in);

	while(1){
		if( strlen(_this_completion.chars_typed) > 0 ){	/* if something typed */
			check_for_completion(QSP_ARG  &_this_completion);
		}

nextchar:
		/*
		 *  Process events in case we will block here
		 *
		 *  check_events won't return until there is either some simulated input,
		 *  or a key has been typed in the console window.
		 */
		check_events(QSP_ARG  tty_in);

		/* An event might have generated some simulated typing.
		 * if so, we might want to forget any typing that has happened.
		 * Fortunately, this is what happens!
		 */

		/* by the time we get to here, something should have been typed */

		c = next_character(tty_in);

/* too bad SIGINT doesn't cause getc to return! */

		/* Check for arrow keys, function keys... */
		if( c == ESC )
			c=handle_escape_sequence(QSP_ARG  &_this_completion);

		if( IS_ARROW(c) ){
			if( handle_arrow_key(QSP_ARG  c,&_this_completion) < 0 )
				goto nextchar;
		}

		/* normally ^D will be returned (raw mode),
		 * but if a ^c has reset the mode, then no
		 */
		if( c==EOF || c == 04 ){
			handle_eof(c,&_this_completion);
			return( NULL );
		}

		// Should this code be ifdef MY_INTR???
		// I can't find anywhere where had_intr is set?
		if( _this_completion.had_intr ){
			handle_interrupt(QSP_ARG  &_this_completion);
			return("");
		}
		if( c=='\r' || c=='\n' ){
			const char *s;
			// only return if there is something
			s = handle_completed_line(QSP_ARG  c,&_this_completion);
			//if( s != NULL && *s != 0 )
			return s;
		} else if( check_special_char(QSP_ARG  c, &_this_completion) ){
			goto nextchar;
		} else {
			handle_normal_character(QSP_ARG  c, &_this_completion);
		}

	} /* while(1) */
} /* end get_response_from_user() */

#endif /* TTY_CTL */

#endif /* HAVE_HISTORY */

