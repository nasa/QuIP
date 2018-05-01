/* support for talking to an auxiliary mouse/trackball
 * on one of the serial lines.
 *
 * The point of this is to use the trackball as an input
 * device that has nothing to do with the window system,
 * which is why we don't just use the system drivers.
 *
 * JBM:  This was hacked up based on unknown documentation.
 * Now, in 2013 I'm trying to resuscitate it, but the 
 * TrackMan marble seems to have disappeared, and the old
 * code doesn't seem to work with the other trackballs.
 * I found a lot of information in xf86-input-mouse-1.7.2,
 * it would be nice to import the whole thing (to be able
 * to support any mouse), but that is more than I am up for
 * in the short term.
 */

/* works well with new TrackMan marble,
 * but flaky with older Trackman Vista...
 *
 * Where is the trackMan marble???
 */

#include "quip_config.h"

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

#include "quip_prot.h"
#include "serial.h"
#include "my_stty.h"
#include "quip_menu.h"
#include "history.h"		/* simulate_typing */

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


#define HIST_SIZE	2048
static char mouse_hist[HIST_SIZE];
static int i_hist_free=0;
static int n_hist_free=HIST_SIZE;

// BUG? these global vars should go into a struct, and be part of qsp? not thread-safe...

static int mouse_fd=(-1);

#define MAX_BUFFERED_CHARS	256

static char mouse_buffer[MAX_BUFFERED_CHARS];
static int n_buffered_chars=0;
static char *next_buffered_char=NULL,*next_buffer_loc=mouse_buffer;
static int n_buffer_locs=MAX_BUFFERED_CHARS;
#define SLEEPTIME	1000		/* 1000 usec = 1 msec */
static unsigned char packet[4]={0,0,0,0};

static int read_mouse_input(SINGLE_QSP_ARG_DECL)
{
	int n;
	ssize_t n2;

#ifdef FIONREAD
	if( ioctl(mouse_fd,FIONREAD,&n) < 0 ){
		perror("ioctl FIONREAD");
		return(-1);
	}
	if( n <= 0 ) return(0);
#else // ! FIONREAD
	n=1;
#endif // ! FIONREAD
    
//fprintf(stderr,"read_mouse_input:  %d char%s available\n",n,n>1?"s":"");

#ifdef CAUTIOUS
	if( n_buffer_locs <= 0 ){
		warn("CAUTIOUS:  no available buffer locations!?");
		return(-1);
	}
#endif /* CAUTIOUS */

	if( n > n_buffer_locs ){
		sprintf(ERROR_STRING,"%d mouse chars available, but only %d free buffer locs",
			n,n_buffer_locs);
		warn(ERROR_STRING);
		n = n_buffer_locs;
	}

	if( (n2=read(mouse_fd,next_buffer_loc,n)) != n ){
		if( n2 < 0 )
			perror("read");
		sprintf(ERROR_STRING,"expected %d chars from mouse, read %zd!?",n,n2);
		warn(ERROR_STRING);
		if( n2 >= 0 ) n=(int)n2;
		else return(-1);
	}
	// We've read n more characters

	// For testing, save them to the history buffer also
	if( n < n_hist_free ){
		strncpy(mouse_hist+i_hist_free,next_buffer_loc,n);
		i_hist_free += n;
		n_hist_free -= n;
	}


	if( n_buffered_chars == 0 )
		next_buffered_char = next_buffer_loc;

	next_buffer_loc += n;
	n_buffered_chars += n;
	n_buffer_locs--;
	return(n);
}

#ifdef NOT_USED
static void show_mouse_buffer(const char *s)
{
	int i;

	fprintf(stderr,"%s\n",s);
	if( n_buffered_chars == 0 ){
		fprintf(stderr,"No buffered mouse chars\n");
	}
	fprintf(stderr,"%d buffered mouse char%s:\n",n_buffered_chars,n_buffered_chars==1?"":"s");
	for(i=0;i<n_buffered_chars;i++){
		fprintf(stderr,"\t0%o\t0x%x\n",
			0xff & next_buffered_char[i],
			0xff & next_buffered_char[i] );
	}
}
#endif // NOT_USED

static int next_mouse_char(SINGLE_QSP_ARG_DECL)
{
	int c;

	while( n_buffered_chars == 0 ){
		if( read_mouse_input(SINGLE_QSP_ARG) <= 0 )
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

static int peek_next_char(SINGLE_QSP_ARG_DECL)
{
	if( n_buffered_chars == 0 ) return -1;
//fprintf(stderr,"n_buffered_chars = %d, next_buffered_char = 0x%lx, i = %ld\n",
//n_buffered_chars,(long)next_buffered_char,(long)(next_buffered_char-mouse_buffer));
	return (int) ((unsigned char) *next_buffered_char);
}

static COMMAND_FUNC( do_mouse_open )
{
	const char *s;
	int c;

	s=NAMEOF("device file");
	mouse_fd=open_serial_device(s);
	// BUG we should set the speed, etc.

	/* We expect to see M3, then some junk... */

	usleep(14000+63000);
advise("do_mouse_open:  back from sleep");
	c=next_mouse_char(SINGLE_QSP_ARG);
	if( c != 'M' )
		warn("expected to see M when mouse was opened!?");
	c=next_mouse_char(SINGLE_QSP_ARG);
	if( c != '3' )
		warn("expected to see M3 when mouse was opened!?");
}

static void send_mouse(QSP_ARG_DECL  char *chardata,int n)
{
	if( mouse_fd < 0 ) return;

	if( write(mouse_fd,chardata,n) != n )
		warn("error writing string");
}

static COMMAND_FUNC(do_send_mouse)
{
	const char *s;
	char str[LLEN];

	s=NAMEOF("text to send");
	strcpy(str,s);
	strcat(str,"\n");

	if( mouse_fd < 0 ) {
		warn("mouse device not open");
		return;
	}

	/*
	send_serial(mouse_fd,str,strlen(str));
	*/
	send_mouse(QSP_ARG  str,(int)strlen(str));
}

static COMMAND_FUNC( do_stty )
{
	set_stty_fd(mouse_fd);
	do_stty_menu(SINGLE_QSP_ARG);
}

#define LEFT_BUTTON_DOWN	4
#define MIDDLE_BUTTON_DOWN	2
#define RIGHT_BUTTON_DOWN	1

#ifdef NOT_USED

#define SET_CHAR(c,mask)			\
	if( b & mask )				\
		s[i] = c;			\
	else	s[i] = '_';			\
	i++;

static void display_button_state(QSP_ARG_DECL  int b)
{
	char s[4];
	int i=0;

	SET_CHAR('L',LEFT_BUTTON_DOWN)
	SET_CHAR('M',MIDDLE_BUTTON_DOWN)
	SET_CHAR('R',RIGHT_BUTTON_DOWN)
	s[i]=0;

	prt_msg_frag(s);
}

#endif // NOT_USED

// This motion seemed to be correct for the missing trackman marble,
// but what about the trackman vista, and Al's trackball???
// The motion packets appear to be 5 bytes...

static void decode_motion_packet(int *px, int *py, unsigned char pkt[])
{
	*px  =  pkt[1] & 077;	// 0x3f
	*px |= (pkt[0] & 0003)<<6;	// 0x03
	*py  =  pkt[2] & 077;	// 0x3f
	*py |= (pkt[0] & 0014)<<4;	// 0x0c

	*px = (int)( (char) *px );	// extend the sign bit
	*py = (int)( (char) *py );	// extend the sign bit
}

static void show_history()
{
	int i;

	fprintf(stderr,"Mouse history:\n");
	for(i=0;i<i_hist_free;i++){
		fprintf(stderr,"%d:\t0%o\t0x%x\n",i,mouse_hist[i],mouse_hist[i]);
	}
}

static void next_mouse_packet(SINGLE_QSP_ARG_DECL)
{
	int i;
	int button_state=0;
	int dx,dy;
	int c;

	for(i=0;i<4;i++)
		packet[i]=0;	// reset to known state

	if( mouse_fd < 0 ){
		warn("mouse device not open");
		return;
	}

	/* see if we already have a buffered packet */

#define IS_HEADER_BYTE(b)	( ((b)&0100) == 0100 )
#define SHOW_PACKET_CHAR(i)	\
fprintf(stderr,"packet[%d] = 0x%x (0%o)\n",i,0xff&packet[i],0xff&packet[i]);

next_packet:
	packet[0] = (unsigned char) next_mouse_char(SINGLE_QSP_ARG);
//SHOW_PACKET_CHAR(0);
	if( ! IS_HEADER_BYTE(packet[0]) ){
		sprintf(ERROR_STRING,
	"next_mouse_packet:  synchronization error (char = 0x%x)",packet[0]);
		warn(ERROR_STRING);
		advise("expected bit 0100 to be set");
		if( verbose )
			show_history();
		goto next_packet;
	}

//again1:
	packet[1]= (unsigned char)next_mouse_char(SINGLE_QSP_ARG);
//SHOW_PACKET_CHAR(1);
	//if( packet[1] & 0100 ){
		//warn("corrupt packet");
		//packet[0] = packet[1];
		//goto again1;
	//}
	packet[2]= (unsigned char)next_mouse_char(SINGLE_QSP_ARG);
//SHOW_PACKET_CHAR(2);
	//if( packet[2] & 0100 ){
		//warn("corrupt packet");
		//packet[0] = packet[2];
		//goto again1;
	//}

	/* now we might have a 4th byte... */

	if( n_buffered_chars == 0 ){
		/* wait at least 1 char time at 1200 baud
		 * 120 chars/sec approx 8 ms
		 */
		usleep(10000);
		read_mouse_input(SINGLE_QSP_ARG);
	}

	c = peek_next_char(SINGLE_QSP_ARG);
	if( c >= 0 ){
		// It's a real character
		if( (c & 0100) == 0 ){
//fprintf(stderr,"reading 4th byte, c = 0x%x\n",c);
			packet[3] = (unsigned char)next_mouse_char(SINGLE_QSP_ARG);
//SHOW_PACKET_CHAR(3);
		}
		else {
//fprintf(stderr,"next_buffered_char = 0x%x (0%o)\n",c,c);
		}
	}

	/* else packet[3] is unchanged, representing old state of middle button */

	/* New logic for decoding packets from Al's trackman
	 * T-CD2-8F
	 *
	 * What we get:
	 *
	 * left down:	0340	0100	0340
	 * left up:	0300	0100	0340
	 * right down:	0320	0100	0340
	 * right up:	0300	0100	0340
	 * middle down:	0300	0100	0040	0364
	 * middle up:	0300	0100	0040	0360
	 *
	 * OOPS those were with the port set to 8 data bits...  it should be 7!
	 */

	// How do we determine when there is a 4th byte???
	//if( (packet[2] & 0100 ) == 0 ){	// middle button?
	//	packet[3] = next_mouse_char(SINGLE_QSP_ARG);
	//	if( packet[3] & 4 ) button_state |= MIDDLE_BUTTON_DOWN;
	//}
	// It seems to be that  the first byte of the packet has bit 6 set (0100)
	// And that that bit is never set in other bytes...
	// So after we have read 3 bytes, we peek ahead, and if that byte
	// is not a header byte we add it to the packet.

	if( packet[0] & 040 ) button_state |= LEFT_BUTTON_DOWN;
	if( packet[0] & 020 ) button_state |= RIGHT_BUTTON_DOWN;
	if( packet[3] & 040 ) button_state |= MIDDLE_BUTTON_DOWN;

	sprintf(MSG_STR,"%d",button_state);
	assign_reserved_var("mouse_buttons", MSG_STR);

	decode_motion_packet(&dx,&dy,packet);

	// This code prints the mouse state
	//display_button_state(QSP_ARG  button_state);
	//sprintf(MSG_STR,"\t%d %d",dx,dy);
	//prt_msg(MSG_STR);

	sprintf(MSG_STR,"%d",dx);
	assign_reserved_var("mouse_dx", MSG_STR);
	sprintf(MSG_STR,"%d",dy);
	assign_reserved_var("mouse_dy", MSG_STR);

}

#define LEFT_BUTTON	1
#define MIDDLE_BUTTON	2
#define RIGHT_BUTTON	4
#define MIDDLE_BUTTON2	8

static int button_state=0;	/* all buttons up */

static Mouse_Event interpret_packet(SINGLE_QSP_ARG_DECL)
{
	int left_button,right_button,middle_button,middle_button2;
	int dx,dy;
	int new_state,changed;
	Mouse_Event event;

	/* this is for the TrackMan Marble */

#define PKT(i)	(0xff&packet[i])
	if( verbose ){
		sprintf(msg_str,"mouse packet:  0%o  0%o  0%o  0%o",
			PKT(0), PKT(1), PKT(2), PKT(3) );
		prt_msg(msg_str);
	}

	//if( (packet[1] & 0100) || (packet[2] & 0100) ){
	//	if( verbose ) warn("corrupt mouse packet");
	//	return(N_MOUSE_EVENTS);
	//}

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

	decode_motion_packet(&dx,&dy,packet);
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
	next_mouse_packet(SINGLE_QSP_ARG);
	interpret_packet(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_check_mouse )
{
	if( n_buffered_chars >= 3 ) {
		advise("mouse data available");
		assign_reserved_var("mouse_data","1");
	} else {
		advise("no mouse data available");
		assign_reserved_var("mouse_data","0");
	}
}

static void flush_mouse(SINGLE_QSP_ARG_DECL)
{
	/* do multiple passes in case the buffer is full... */

	/* The code doesn't seem to do what the above comment indicates!? */

	read_mouse_input(SINGLE_QSP_ARG);

	if( n_buffered_chars ){
		while( n_buffered_chars ){
			sprintf(msg_str,"flush_mouse:  flushing %d extra characters from mouse buffer",n_buffered_chars);
			prt_msg(msg_str);

			n_buffered_chars=0;
			next_buffered_char=NULL;
			next_buffer_loc=mouse_buffer;
			n_buffer_locs=MAX_BUFFERED_CHARS;
			usleep(10000);	// 10 ms, about 1 char at 1200 baud
			read_mouse_input(SINGLE_QSP_ARG);
		}
	} else {
		if( verbose )
			prt_msg("flush_mouse:  no extra chars to flush");
	}
}

/* this is the callback function - how do we get the query stream ??? */

static void check_mouse(SINGLE_QSP_ARG_DECL)
{
	Mouse_Event event;

	read_mouse_input(SINGLE_QSP_ARG);

	if( n_buffered_chars >= 3 ) {
		next_mouse_packet(SINGLE_QSP_ARG);
		event=interpret_packet(SINGLE_QSP_ARG);
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
			// Why simulate typing instead of chew_text???
			//if( intractive(SINGLE_QSP_ARG) )
			//	simulate_typing(mouse_action[event]);
			//else
				chew_text(mouse_action[event],"(mouse event)");
		}
	}
}

static COMMAND_FUNC( poll_mouse )
{
	static int polling=0;

	if( ASKIF("check for mouse events") ){
		if( polling ){
			warn("already polling mouse");
			return;
		}
		add_event_func(check_mouse);
		polling=1;
	} else {
		if( ! polling ) {
			warn("not already polling mouse");
			return;
		}
		if( rem_event_func(check_mouse) < 0 )
			warn("error removing mouse handler");
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
	flush_mouse(SINGLE_QSP_ARG);
}

// \044 is escape for dollar sign (to elim Xcode warning)

#define ADD_CMD(s,f,h)	ADD_COMMAND(mouse_menu,s,f,h)

MENU_BEGIN(mouse)
ADD_CMD( mouse,		do_mouse_open,	open mouse device )
ADD_CMD( send,		do_send_mouse,	send text to mouse )
ADD_CMD( wait,		do_wtfor,	wait for a mouse event )
ADD_CMD( stty,		do_stty,	hardware control submenu )
ADD_CMD( check,		do_check_mouse,	set \044mouse_data to reflect packet availability )
ADD_CMD( flush,		do_flush_mouse,	discard buffered chars )
ADD_CMD( action,	set_action,	specify action for mouse event )
ADD_CMD( poll,		poll_mouse,	check for mouse events in the interpreter )
MENU_END(mouse)

COMMAND_FUNC( do_mouse_menu )
{
	CHECK_AND_PUSH_MENU(mouse);
}


