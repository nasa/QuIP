/* support for talking to an auxiliary mouse/trackball on one of the serial lines */

/* works well with new TrackMan marble,
 * but flaky with older Trackman Vista...
 */

#include "quip_config.h"

char VersionId_interpreter_mouse[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* write() */
#endif

#ifdef HAVE_STRING_H
#include <string.h>	/* strcat() */
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#ifdef HAVE_SYS_FILIO_H
#include <sys/filio.h>	/* FIONREAD */
#endif

#include "chewtext.h"
#include "debug.h"
#include "query.h"		/* ttys_are_interactive */
#include "filerd.h"		/* interp_file() */
#include "serial.h"
#include "my_stty.h"
#include "callback.h"		/* add_event_func */
#include "history.h"		/* simulate_typing */
#include "submenus.h"

#ifdef TTY_CTL
#include "ttyctl.h"
#endif /* TTY_CTL */

typedef enum {
	LEFT_UP,
	LEFT_DN,
	MIDDLE_UP,
	MIDDLE_DN,
	RIGHT_UP,
	RIGHT_DN,
	MIDDLE2_UP,
	MIDDLE2_DN,
	MOTION,
	N_MOUSE_EVENTS		/* must be last */
} Mouse_Event;

static const char *mouse_action[N_MOUSE_EVENTS]={
	NULL,NULL,
	NULL,NULL,
	NULL,NULL,
	NULL,NULL,
	NULL
};

static const char *mouse_event_name[N_MOUSE_EVENTS+1]={
	"left_up",
	"left_dn",
	"middle_up",
	"middle_dn",
	"right_up",
	"right_dn",
	"middle2_up",
	"middle2_dn",
	"motion",
	"null_event"
};


/* local prototypes */
static int read_mouse_input(void);
static Mouse_Event interpret_packet(void);
static void flush_mouse(void);


static int next_mouse_char(void);
static int mouse_fd=(-1);

static COMMAND_FUNC( do_mouse_open )
{
	const char *s;
	int c;

	s=NAMEOF("device file");
	mouse_fd=open_serial_device(QSP_ARG  s);
	/* We expect to see M3, then some junk... */

	usleep(14000+63000);
	c=next_mouse_char();
	if( c != 'M' )
		WARN("expected to see M when mouse was opened!?");
	c=next_mouse_char();
	if( c != '3' )
		WARN("expected to see M3 when mouse was opened!?");
}

static void send_mouse(char *chardata,int n)
{
	if( mouse_fd < 0 ) return;

	if( write(mouse_fd,chardata,n) != n )
		NWARN("error writing string");
}

static COMMAND_FUNC(do_send_mouse)
{
	const char *s;
	char str[LLEN];

	s=NAMEOF("text to send");
	strcpy(str,s);
	strcat(str,"\n");

	if( mouse_fd < 0 ) {
		WARN("mouse device not open");
		return;
	}

	/*
	send_serial(mouse_fd,str,strlen(str));
	*/
	send_mouse(str,strlen(str));
}

static COMMAND_FUNC( do_stty )
{
	set_stty_fd(mouse_fd);
	stty_menu(SINGLE_QSP_ARG);
}

#define MAX_BUFFERED_CHARS	256

static char mouse_buffer[MAX_BUFFERED_CHARS];
static int n_buffered_chars=0;
static char *next_buffered_char=NULL,*next_buffer_loc=mouse_buffer;
static int n_buffer_locs=MAX_BUFFERED_CHARS;
#define SLEEPTIME	1000		/* 1000 usec = 1 msec */
static char packet[4]={0,0,0,0};

static int read_mouse_input()
{
	int n,n2;

	if( ioctl(mouse_fd,FIONREAD,&n) < 0 ){
		perror("ioctl FIONREAD");
		return(-1);
	}
	if( n <= 0 ) return(0);

#ifdef CAUTIOUS
	if( n_buffer_locs <= 0 ){
		NWARN("CAUTIOUS:  no available buffer locations!?");
		return(-1);
	}
#endif /* CAUTIOUS */

	if( n > n_buffer_locs ){
		sprintf(DEFAULT_ERROR_STRING,"%d mouse chars available, but only %d free buffer locs",
			n,n_buffer_locs);
		NWARN(DEFAULT_ERROR_STRING);
		n = n_buffer_locs;
	}

	if( (n2=read(mouse_fd,next_buffer_loc,n)) != n ){
		if( n2 < 0 )
			perror("read");
		sprintf(DEFAULT_ERROR_STRING,"expected %d chars from mouse, read %d!?",n,n2);
		NWARN(DEFAULT_ERROR_STRING);
		if( n2 >= 0 ) n=n2;
		else return(-1);
	}

	if( n_buffered_chars == 0 )
		next_buffered_char = next_buffer_loc;

	next_buffer_loc += n;
	n_buffered_chars += n;
	n_buffer_locs--;
	return(n);
}

static int next_mouse_char()
{
	int c;

	while( n_buffered_chars == 0 ){
		if( read_mouse_input() <= 0 )
			usleep(1000);	/* about 1 char time at 9600 baud */
	}

	c = *next_buffered_char++;
	n_buffered_chars --;

	if( n_buffered_chars == 0 ){
		next_buffer_loc = mouse_buffer;
		n_buffer_locs = MAX_BUFFERED_CHARS;
	}
	return( c );
}

static void next_mouse_packet(void)
{
	if( mouse_fd < 0 ){
		NWARN("mouse device not open");
		return;
	}

	/* see if we already have a buffered packet */

#define IS_HEADER_BYTE(b)	( ((b)&0100) == 0100 )

next_packet:
	packet[0] = next_mouse_char();
	if( ! IS_HEADER_BYTE(packet[0]) ){
		sprintf(DEFAULT_ERROR_STRING,
	"next_mouse_packet:  synchronization error (char = 0x%x)",packet[0]);
		NWARN(DEFAULT_ERROR_STRING);
		advise("expected bit 0100 to be set");
		goto next_packet;
	}
again1:
	packet[1]=next_mouse_char();
	if( packet[1] & 0100 ){
		NWARN("corrupt packet");
		packet[0] = packet[1];
		goto again1;
	}
	packet[2]=next_mouse_char();
	if( packet[2] & 0100 ){
		NWARN("corrupt packet");
		packet[0] = packet[2];
		goto again1;
	}

	/* now we might have a 4th byte... */

	if( n_buffered_chars == 0 ){
		/* wait at least 1 char time at 1200 baud 120 chars/sec approx 8 ms */
		usleep(10000);
		read_mouse_input();
	}

	if( n_buffered_chars > 0 ){
		/* peek at the next character in the buffer.
		 * If it is not a header character, add it to the packet.
		 */
		if( ! IS_HEADER_BYTE(*next_buffered_char) ){
			packet[3] = next_mouse_char();
		}
	}
	/* else packet[3] is unchanged, representing old state of middle button */
}

#define LEFT_BUTTON	1
#define MIDDLE_BUTTON	2
#define RIGHT_BUTTON	4
#define MIDDLE_BUTTON2	8

static int button_state=0;	/* all buttons up */

static Mouse_Event interpret_packet()
{
	int left_button,right_button,middle_button,middle_button2;
	char dx,dy;
	int new_state,changed;
	Mouse_Event event;

	/* this is for the TrackMan Marble */

	if( verbose ){
		sprintf(msg_str,"mouse packet:  0%o  0%o  0%o  0%o",
			packet[0], packet[1], packet[2], packet[3] );
		prt_msg(msg_str);
	}

	if( (packet[1] & 0100) || (packet[2] & 0100) ){
		if( verbose ) NWARN("corrupt mouse packet");
		return(N_MOUSE_EVENTS);
	}

	left_button = packet[0] & 0040;
	right_button = packet[0] & 0020;
	middle_button = packet[3] & 0040;
	middle_button2 = packet[3] & 0020;

	new_state = 0;
	new_state |= left_button ? LEFT_BUTTON : 0 ;
	new_state |= right_button ? RIGHT_BUTTON : 0 ;
	new_state |= middle_button ? MIDDLE_BUTTON : 0 ;
	new_state |= middle_button2 ? MIDDLE_BUTTON2 : 0 ;

	changed = button_state ^ new_state;
	button_state = new_state;

	dx=packet[1] & 077;
	dx |= (packet[0] & 0003)<<6;
	dy=packet[2] & 077;
	dy |= (packet[0] & 0014)<<4;

	/* we may have a real event that the software does not
	 * register as a change, if we have flushed some intervening
	 * stuff...
	 * Therefore we assign a default value to event...
	 */
	event=N_MOUSE_EVENTS;
	if( changed & LEFT_BUTTON )
		event = left_button ? LEFT_DN : LEFT_UP ;
	else if( changed & MIDDLE_BUTTON )
		event = middle_button ? MIDDLE_DN : MIDDLE_UP ;
	else if( changed & MIDDLE_BUTTON2 )
		event = middle_button2 ? MIDDLE2_DN : MIDDLE2_UP ;
	else if( changed & RIGHT_BUTTON )
		event = right_button ? RIGHT_DN : RIGHT_UP ;
	else if( dx != 0 || dy != 0 )
		event = MOTION;

	if( verbose ) {
		prt_msg_frag(mouse_event_name[event]);
		if( event==MOTION ){
			sprintf(msg_str,"  dx = %d   dy = %d",dx,dy);
			prt_msg(msg_str);
		} else prt_msg("");
	}
	return(event);
}

static COMMAND_FUNC( do_wtfor )
{
	next_mouse_packet();
	interpret_packet();
}

static COMMAND_FUNC( do_check_mouse )
{
	if( n_buffered_chars >= 3 ) {
		advise("mouse data available");
		ASSIGN_VAR("mouse_data","1");
	} else {
		advise("no mouse data available");
		ASSIGN_VAR("mouse_data","0");
	}
}

static void flush_mouse()
{
	/* do multiple passes in case the buffer is full... */

	/* The code doesn't seem to do what the above comment indicates!? */

	read_mouse_input();

	if( n_buffered_chars ){
		sprintf(msg_str,"flush_mouse:  flushing %d extra characters from mouse buffer",n_buffered_chars);
		prt_msg(msg_str);

		n_buffered_chars=0;
		next_buffered_char=NULL;
		next_buffer_loc=mouse_buffer;
		n_buffer_locs=MAX_BUFFERED_CHARS;
		read_mouse_input();
	} else {
		if( verbose )
			prt_msg("flush_mouse:  no extra chars to flush");
	}
}

/* this is the callback function - how do we get the query stream ??? */

static void check_mouse(SINGLE_QSP_ARG_DECL)
{
	Mouse_Event event;

	read_mouse_input();

	if( n_buffered_chars >= 3 ) {
		next_mouse_packet();
		event=interpret_packet();
		if( event == N_MOUSE_EVENTS )	/* bad packet */
			return;

		/* here we might figure out what kind of event this is */
		if( mouse_action[event] != NULL ){
			if( verbose ){
				sprintf(ERROR_STRING,
		"Mouse event %s producing input \"%s\"",mouse_event_name[event],
					mouse_action[event]);
				advise(ERROR_STRING);
			}
			if( intractive(SINGLE_QSP_ARG) )
				simulate_typing(mouse_action[event]);
			else
				chew_text(QSP_ARG  mouse_action[event]);
		}
	}
}

static COMMAND_FUNC( poll_mouse )
{
	static int polling=0;

	if( ASKIF("check for mouse events") ){
		if( polling ){
			WARN("already polling mouse");
			return;
		}
		add_event_func(check_mouse);
		polling=1;
	} else {
		if( ! polling ) {
			WARN("not already polling mouse");
			return;
		}
		if( rem_event_func(check_mouse) < 0 )
			WARN("error removing mouse handler");
		polling=0;
	}
}

static COMMAND_FUNC( set_action )
{
	int i;
	const char *s;

	i=WHICH_ONE("mouse event",N_MOUSE_EVENTS,mouse_event_name);
	s=NAMEOF("action text");

	if( i < 0 ) return;
	if( mouse_action[i] != NULL ) rls_str(mouse_action[i]);

	mouse_action[i] = savestr(s);
}

static COMMAND_FUNC( do_flush_mouse )
{
	flush_mouse();
}

Command mouse_ctbl[]={
{ "mouse",	do_mouse_open,	"open mouse device"			},
{ "send",	do_send_mouse,	"send text to mouse"			},
{ "wait",	do_wtfor,	"wait for a mouse event"		},
{ "stty",	do_stty,	"hardware control submenu"		},
{ "check",	do_check_mouse,	"set $mouse_data to reflect packet availability"	},
{ "flush",	do_flush_mouse,	"discard buffered chars"		},
{ "action",	set_action,	"specify action for mouse event"	},
{ "poll",	poll_mouse,	"check for mouse events in the interpreter"		},
{ "quit",	popcmd,		"exit program"				},
{ NULL,		NULL,		NULL					}
};

COMMAND_FUNC( mouse_menu )
{
	PUSHCMD(mouse_ctbl,"mouse");
}


