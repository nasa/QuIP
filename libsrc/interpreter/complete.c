
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
int had_intr=0;

/* static */ FILE *_tty_out;
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


#ifdef HAVE_TERMCAP
static char tc_ent[1024];
static char tbuf[32];
#endif /* HAVE_TERMCAP */

#define NO_TERMCAP	((char *)0)


/* local prototypes */

static void	init_tty_chars(SINGLE_QSP_ARG_DECL);
static int	pnt_strlen(const char *);
static void	ers_def(const char *);
static void	show_def(const char *,int), show_char(int);

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

/* ers_def
 * 
 * Erase any completed characters on the display.
 */

static void ers_def( const char *def_str )
{
	int i,n;

	if( *def_str ){
		n=pnt_strlen(def_str);
		i=n;
		while(i--) fputc(' ',_tty_out);
		i=n;
		while(i--) fputc('\b',_tty_out); /* backspace */
	}
}

/* show_def
 *
 * Print the string def_str from the current point.
 * If curflag is set, we print the first char, and then
 * go into standout mode.  Otherwise, we go into standout
 * mode for the first character.
 *
 * After we've printed the string, we backspace over what has been printed.
 */

static void show_def( const char *def_str, int curflag )
{
	int i;
	const char *s;

	s=def_str;
	if( *s ){
		if( curflag ){
			show_char(*s++);
			if( *so ) fputs(so,_tty_out);
		} else {
			fputs(so,_tty_out);
			show_char(*s++);
		}
		while( *s )
			show_char(*s++);

		if( *se ) fputs(se,_tty_out);
		if( *ce ) fputs(ce,_tty_out);
		i=pnt_strlen(def_str);
		while(i--) fputc('\b',_tty_out); /* backspace */
	}
}

static void show_char(int c)
{
	if( iscntrl(c) ){
		fputc('^',_tty_out);
		if( ( c ) ==0177 ) fputc('?',_tty_out);
		else fputc(c+'A'-1,_tty_out);
	} else fputc(c,_tty_out);	/* do the echo */
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

static void init_tty_chars(SINGLE_QSP_ARG_DECL)
{
#ifdef HAVE_TERMCAP
	char *s;
	char *tptr;
	int stat;
	extern char *tgetstr();

	s=getenv("TERM");
	if( s==NULL ){
		WARN("init_tty_chars:  no TERM in environment");
		so="";
		ce="";
		se="";
	}

	stat=tgetent(tc_ent,s);
	if( stat!=1 ){
		if( stat==(-1) )
			WARN("init_tty_chars:  can't open termcap file");
		else if( stat==0 ){
			sprintf(ERROR_STRING,
			"no termcap entry for terminal \"%s\"",s);
			WARN(ERROR_STRING);
		} else WARN("unrecognized error status from tgetent()");
	}
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
	fputc('\b',tty_out);						\
	fputc(' ',tty_out);						\
	fputc('\b',tty_out);


#define ERASE_ONE_CHAR							\
									\
									\
	ers_def(&def_str[n_so_far]);					\
	UNDISPLAY_CHAR							\
	n_so_far--;							\
	/*								\
	 * control chars are printed w/ 2 chars				\
	 * ^A ^B etc							\
	 */								\
	if( iscntrl(sel_str[n_so_far]) ){				\
		fputc('\b',tty_out);					\
		fputc(' ',tty_out);					\
		fputc('\b',tty_out);					\
	}								\
	sel_str[n_so_far]=0;

/* BUG globals are not thread-safe */
static char sel_str[LLEN];
static char edit_string[LLEN];

const char *get_sel( QSP_ARG_DECL  const char *prompt, FILE *tty_in, FILE *tty_out )
{
	int c;
	u_int n_so_far;
	const char *def_str;
	static int have_tty_chars=0;
	int edit_mode=0;		/* complete until we see an arrow key */

#ifdef QUIP_DEBUG
if( comp_debug <= 0 ) comp_debug=add_debug_module(QSP_ARG  "completion");
#endif /* QUIP_DEBUG */

	/* BUG need to check here that tty_in, tty_out
	 * are really tty's!!
	 */


	if( !have_tty_chars ){
		init_tty_chars(SINGLE_QSP_ARG);
		have_tty_chars=1;
	}

	_tty_out=tty_out;	/* set global BUG not thread-safe */

	sel_str[0]=0;	/* clear it out */
	n_so_far=0;

	had_intr=0;

	if( !exit_func_set ){
		do_on_exit(sane_tty);
		exit_func_set=1;
	}

	ttycbrk(fileno(tty_in));
	echooff(fileno(tty_in));

	if( ers_char == 0 ){
		ers_char = get_erase_chr(fileno(tty_in));
		if( ers_char == -1 ){
			NWARN("Erase character defaulting to ^H");
			ers_char='\b';
		}
	}

	if( kill_char == 0 ){
		kill_char = get_kill_chr(fileno(tty_in));
		if( kill_char == -1 ){
			NWARN("Kill character defaulting to ^U");
			kill_char='';
		}
	}

/* On wheatstone, we don't seem to respond to typing in the plot window, but if we put an
 * advise() here then we do!?!?
 */
//advise("w");
	while(1){
		/* Try to complete a response from the history list first... */

		if( strlen(sel_str) > 0 ){	/* if something typed */
			u_int l;

			def_str=get_match(QSP_ARG  prompt,sel_str);
			l=strlen(def_str);

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

			if( l == 0 && IS_COMMAND_PROMPT(prompt) ) {
				def_str=get_match(QSP_ARG  h_bpmpt,sel_str);
				l=strlen(def_str);
			}

			if( l > n_so_far )
				show_def(&def_str[n_so_far],1);

			if( l == 0 ) def_str=sel_str;
		} else {	/* nothing typed yet... */
			/* get the match to reset the current history list,
			 * but don't show it if nothing has been typed.
			 * The user can see the defaults with ^N.
			 */
			def_str=get_match(QSP_ARG  prompt,sel_str);
			def_str=sel_str;
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
		if( c == ESC ){
			/* what if the user simply typed ESC??? */
			/* here we should see if we have more characters... */
			check_events(QSP_ARG  tty_in);
			c = next_character(tty_in);
			if( c == '[' ){		/* what we expect for arrow keys */
				check_events(QSP_ARG  tty_in);
				c = next_character(tty_in);
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
			}
		}

#define BEGIN_EDIT					\
							\
			strcpy(edit_string,def_str);	\
			edit_mode=1;

		if( IS_ARROW(c) ){

			if( c == UP_ARROW ){
				while( n_so_far ){
					ERASE_ONE_CHAR
				}
				ers_def(&def_str[n_so_far]);
				def_str=cyc_match(QSP_ARG  sel_str,CYC_FORWARD);
				if( strlen(def_str) > n_so_far )
					show_def(&def_str[n_so_far],1);
				BEGIN_EDIT
				goto nextchar;
			} else if( c == DN_ARROW ){
				while( n_so_far ){
					ERASE_ONE_CHAR
				}
				ers_def(&def_str[n_so_far]);
				/* BUG need to cycle the opposite way */
				def_str=cyc_match(QSP_ARG  sel_str,CYC_BACKWARD);
				if( strlen(def_str) > n_so_far )
					show_def(&def_str[n_so_far],1);
				BEGIN_EDIT
				goto nextchar;
			} else if( c == RT_ARROW ){
				/* check and see if there are any right chars to move over! */
				/* def_str holds the edit string */
				if( !edit_mode ){
					BEGIN_EDIT
				}
				if( n_so_far < strlen(edit_string) ){
					show_char(edit_string[n_so_far]);
					n_so_far ++;
				}
				/* else maybe should beep or something */

				/* BUG need to move the cursor */
				goto nextchar;
			} else if( c == LF_ARROW ){
				if( !edit_mode ){
					BEGIN_EDIT
				}
				/* make sure that there are some chars to move over */
				if( n_so_far > 0 ){
					n_so_far --;
					fputc('\b',tty_out);
				}
				/* maybe should beep or something */
				goto nextchar;
			}
		}

		/* normally ^D will be returned (raw mode),
		 * but if a ^c has reset the mode, then no
		 */
		if( c==EOF || c == 04 ){		/* ^D */
			/* echo it first */
			if( c== EOF ){
				// This never seems to be executed???
				fprintf(tty_out,"kbd EOF\n");
			} else
				fprintf(tty_out,"^D\n");
			return( NULL );
		}
		// Should this code be ifdef MY_INTR???
		// I can't find anywhere where had_intr is set?
		if( had_intr ){
			/*
			 * interrupts block
			 * until the last command finishes
			 */
#ifdef QUIP_DEBUG
if( debug & comp_debug ) advise("resetting tty after interrupt");
#endif /* QUIP_DEBUG */
			tty_reset(tty_out);
			putc('\n',tty_out);

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
			return("");
		}
		if( c=='\r' || c=='\n' ){

			/* we do show_def a second time
				since the cursor will move! */
			if( strlen(def_str) > n_so_far )
				show_def(&def_str[n_so_far],0);
			fputc(c,tty_out);	/* do the echo */

			/* Add response to history lists...
			 * We do this in qword() also,
			 * to save values for suppressed prompts
			 */

			if( edit_mode ){
				if( *prompt && *edit_string )
					add_def(QSP_ARG  prompt,edit_string);
				return(edit_string);
			} else {
				if( *prompt && *def_str )
					add_def(QSP_ARG  prompt,def_str);

				// Store the newline too...
				if( n_so_far >= (LLEN-1) ){
					WARN("too many input chars!?");
				} else {
					//sel_str[n_so_far++] = c;
					// We do this after returning from get_sel.
					// that way we get def_str also!
				}
				sel_str[n_so_far] = 0;
//fprintf(stderr,"get_sel:  sel_str = \"%s\", def_str = \"%s\"\n",
//sel_str,def_str);
				// should we add a newline to def_str?
				return(def_str);
			}
		}

		if( c== ers_char ){
			if( n_so_far > 0 ){
				ERASE_ONE_CHAR
			} else fputc(BELL,tty_out);
		} else if( c == kill_char ){
			while( n_so_far ){
				ERASE_ONE_CHAR
			}
		} else if( c==FWD_CHAR || c=='\t' ){ /* ^F or TAB accept word but don't end line */
			if( strlen(def_str) > n_so_far ){
				fputs(so,tty_out);
				while( def_str[n_so_far] &&
					!isspace(def_str[n_so_far]) ){
					sel_str[n_so_far]=def_str[n_so_far];
					show_char(sel_str[n_so_far]);
					n_so_far++;
				}
				/*
				 * Adding a single space here may cause
				 * a match to be missed if the match
				 * text has more than one space,
				 * but since this choice
				 * should be available with ^N,
				 * we don't consider this a bug.
				 */
				fputs(se,tty_out);
				sel_str[n_so_far++]=' ';	/* space */
				fputc(' ',tty_out);
				sel_str[n_so_far]=0;
			} else fputc(BELL,tty_out);
		} else if( c == CYC_CHAR ){	/* ^N cycles default */
			/*
			 * Since we don't pass the prompt to cyc_match,
			 * this will not cycle between
			 * regular menu commands and builtins.
			 * Might be considered a BUG?
			 */
			ers_def(&def_str[n_so_far]);
			def_str=cyc_match(QSP_ARG  sel_str,CYC_FORWARD);
			if( strlen(def_str) > n_so_far )
				show_def(&def_str[n_so_far],1);
			goto nextchar;
		} else {
			/* just a regular character.
			 * If we are in edit mode, we insert the character at the cursor location.
			 * Otherwise, we add this character, and check for a new completion...
			 *
			 * BUG:  edit mode is badly broken!?
			 */
			if( edit_mode ){
				char tmp_tail[LLEN];

				ers_def(&def_str[n_so_far]);

				strcpy(tmp_tail,&edit_string[n_so_far]);
				edit_string[n_so_far]=c;
				n_so_far++;
				strcpy(&edit_string[n_so_far],tmp_tail);
				show_char(c);
				show_def(&edit_string[n_so_far],1);
			} else {
				show_char(c);

				if( c != def_str[n_so_far] )
					ers_def(&def_str[n_so_far]);
				if( n_so_far >= (LLEN-1) ){
					WARN("too many input chars!?");
				} else {
					sel_str[n_so_far++] = c;
				}
				sel_str[n_so_far] = 0;
			}
		} /* end regular character */
	} /* while(1) */
} /* end get_sel() */

#endif /* TTY_CTL */

#endif /* HAVE_HISTORY */

