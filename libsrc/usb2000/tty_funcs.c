/* tty_funcs.c */

#include "quip_config.h"

#include "quip_prot.h"
#include "usb2000.h"

#ifdef USE_SERIAL_LINE

#ifdef HAVE_TERMIOS_H
#include <termios.h>	/* */
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* usleep */
#endif

#ifdef HAVE_STRING_H
#include <string.h>	/* strlen */
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* strtol */
#endif

#ifdef TTY_CTL
#include "ttyctl.h"	/* set_baud */
#endif /* TTY_CTL */

#include "serial.h"
#include "debug.h"

//#define DEBUG

static int usb2000_fd=(-1);

static u_short baud_clk_timer = _16_BIT_BAUD_TIMER;	/* default integration clock timer is 16 bits (pg.14) */

#define RECV_BUF_SIZE	4

static short verify_data_mode(SINGLE_QSP_ARG_DECL)
{
	char pkt[] = "\n";
	int reply ;

	//send_serial(QSP_ARG usb2000_fd, pkt, strlen(pkt) );
	send_usb2000_packet(QSP_ARG  pkt, strlen(pkt) );

#ifdef DEBUG
if( debug & usb2000_debug ){
advise("Checking data mode...");
}
#endif /* DEBUG */

	if( ( reply = recv_a_byte(SINGLE_QSP_ARG) ) == LF ) {

		if( recv_a_byte(SINGLE_QSP_ARG) == NAK ) {

			if( get_tail(SINGLE_QSP_ARG) < 0 ) {
				WARN("ERROR: mode might be ascii");
				WARN("The device needs a reboot");

			} else {
				ascii_mode = TRUE;
#ifdef DEBUG
if( debug & usb2000_debug ){
advise("Data mode is ascii.");
}
#endif /* DEBUG */
				return 0;
			}
		}

		return -1;

	} else if( reply == NAK ) {
		ascii_mode = FALSE;
#ifdef DEBUG
if( debug & usb2000_debug ){
advise("Data mode is binary.");
}
#endif /* DEBUG */
		return 0;
	}

	/* this means that the baud rate might be different */
	return -1;
}
/*
 * This function simply receives data and doesn't use it. This way the
 * buffer gets cleared!
 * BUG: need to improve this function.
 */
void clear_input_buf(SINGLE_QSP_ARG_DECL)
{
	u_char buf[MAX_PKT_SIZE];

	if( !(recv_somex(usb2000_fd, buf, MAX_PKT_SIZE,MAX_PKT_SIZE-1)) )
		usleep(1000);

	usleep(1000);

	if( !(recv_somex(usb2000_fd, buf, MAX_PKT_SIZE, MAX_PKT_SIZE-1)) )
		usleep(1000);
}

int recv_a_byte(SINGLE_QSP_ARG_DECL)
{
	unsigned char recv_buf[RECV_BUF_SIZE] = {0};
	u_short n_received = 0;
	int n_waits = 0;
	int value_recvd;

	while( n_received < 1 && n_waits < MAX_WAITS ) {
		n_received = recv_somex(usb2000_fd, recv_buf, RECV_BUF_SIZE, 1 );
		if( !n_received ) {
			n_waits++;
			usleep(1); /* .001 ms */
		}
	}

	if( n_waits >= MAX_WAITS && n_received == 0 ) {
		sprintf(ERROR_STRING,"No value received after waiting %d n_waits?", n_waits);
		WARN(ERROR_STRING);
		return -1;
	}

	value_recvd = (int)recv_buf[0];

#ifdef DEBUG
if( debug & usb2000_debug ){
sprintf(ERROR_STRING,"byte recvd: 0x%.2x", value_recvd);
advise(ERROR_STRING);
}
#endif /* DEBUG */

	return value_recvd;
}


/* A value is 'defined' to be either
 * data teminated by SPACE for ascii mode
 * or two bytes for binary mode.
 */

int recv_a_value(SINGLE_QSP_ARG_DECL)
{
	unsigned char recv_buf[RECV_BUF_SIZE] = {0,0,0,0};
#define MAX_SIZEOF_VALUE	10
	char value[MAX_SIZEOF_VALUE];
	u_short n_received;
	int n_waits;
	u_short i;
	int n_value;
	u_short n_words_received;

	n_waits = 0;
	n_received = 0;
	i = 0;
	value[0] = '\0';

	if( !ascii_mode ) {
advise("recv_a_value:  not ascii mode");
		/* all binary data is in the form of a word (pg.12) */
		for( n_words_received=0;  n_words_received<2; n_words_received++ ) {

			while( n_received < 1 && n_waits < MAX_WAITS ) {
				n_received = recv_somex(usb2000_fd, recv_buf, RECV_BUF_SIZE, 1 );
				if( !n_received ) {
					n_waits++;
					usleep(1); /* .001 ms */
				}
			}

			if( n_waits >= MAX_WAITS && n_received == 0 ) {
				sprintf(ERROR_STRING,"No value received after waiting %d n_waits?", n_waits);
				WARN(ERROR_STRING);
				return -1;
			}

			n_waits = 0;
			n_received = 0;

			value[n_words_received] = recv_buf[0];
		}

		sprintf(value, "%.2x%.2x", value[0], value[1]);

		n_value = strtol(value, NULL, 16);

	} else {

		while( recv_buf[0] != SPACE ) {

			while( n_received < 1 && n_waits < MAX_WAITS ) {
				n_received = recv_somex(usb2000_fd, recv_buf, RECV_BUF_SIZE, 1 );
				if( !n_received ) {
					n_waits++;
					usleep(1); /* .001 ms */
				}
			}

			if( n_waits >= MAX_WAITS && n_received == 0 ) {
				sprintf(ERROR_STRING,"No value received after waiting %d n_waits?", n_waits);
				WARN(ERROR_STRING);
				sprintf(ERROR_STRING,"%d characters already received",i);
				advise(ERROR_STRING);

				if( i > 0 ){
					value[i]=0;	/* terminate string */
					sprintf(ERROR_STRING,"string \"%s\" received so far",value);
					advise(ERROR_STRING);
				}
				return -1;
			}

			n_waits = 0;
			n_received = 0;

			#ifdef CAUTIOUS
			if( i > MAX_SIZEOF_VALUE-1 ) {
				value[MAX_SIZEOF_VALUE-1]=0;	/* terminate string */
				sprintf(ERROR_STRING, "CAUTIOUS:  recv_a_value:  Impossiblely large value (%s) is being received, Reception ABORTED.", value);
				WARN(ERROR_STRING);
			}
			#endif /* CAUTIOUS */

			value[i++] = recv_buf[0];
		}

		value[--i] = '\0';	/* replace space with null char */
		n_value = (int)strtol(value, NULL, 10);

	} /* else */

#ifdef DEBUG
if( debug & usb2000_debug ){
sprintf(ERROR_STRING, "n_value: %d", n_value );
advise(ERROR_STRING);
}
#endif /* DEBUG */

	return n_value;
}


int send_pkt( QSP_ARG_DECL  const char *pkt )
{
	if( ascii_mode )
		send_serial(usb2000_fd, (u_char *)pkt, strlen(pkt) );

	else	/* if binary mode */
		send_hex(usb2000_fd, (u_char *)pkt );

	return 0;

}

int xmit_pxl_mode_pkt(QSP_ARG_DECL  const char *pkt, int pxl_mode, int n )
{
	u_short i;

	if( ascii_mode ) {

		int n_expect=(-1);	/* init to quiet compiler */
		int echo_buf[7];	/* we will not be receiving more than
					 * 7 bytes at a time
					 */

		while( *pkt ){

			int n_waits;

			send_serial(usb2000_fd, (u_char *)pkt, 1 );

			if( *pkt == '\n' || *pkt == '\r' ) {
				if( *(pkt+1) ) {	/* not the last char */
					n_expect = 2;

				} else {
					if( pxl_mode != 1 )
						n_expect = 7;	/* lf cr ack lf cr > space */
					else
						n_expect = 6;	/* LF ACK LF CR > SPACE */
				}

			} else {
				n_expect = 1;
			}

			n_waits=0;

			for( i=0; i<n_expect; i++ ) {
				if ( *(pkt+1) && *pkt != '\n' ) {
					if( ( echo_buf[i] = recv_a_byte(SINGLE_QSP_ARG) ) != (int)*pkt )
						return -1;

				} else {
					if( ( echo_buf[i] = recv_a_byte(SINGLE_QSP_ARG) ) < 0 )
						return -1;
				}
			}

			pkt++;
		}

		/* n_expect will be either 6 or 7
		 * and the position of ACK will be
		 * echo_buf[1] or echo_buf[2] respectively
		 */

		if( echo_buf[n_expect-5] != ACK ) {
			WARN("ACK not received");
			return -1;
		}

	} else {	/* if binary mode */

		u_short n_of_CRs=(-1);	/* init to silence compiler warning */

		/* I have observed that a delay is not required in binary mode.
		 */

		send_hex(usb2000_fd, (u_char *)pkt );


		/* The expected number of CRs is not documented in
		 * the manual. I found these out by testing all
		 * the pixel modes. Once again, there is no mention
		 * of CRs getting sent back in this mode!
		 *
		 * What happens is that the device 'acknowledges
		 * the reception of a pixel mode value by sending a
		 * CR.
		 */

		switch(pxl_mode) {
			case 0:		n_of_CRs = 1;		break;
			case 1:		n_of_CRs = 1;		break;
			case 3:		n_of_CRs = 4;		break;
			case 4:		n_of_CRs = 2+n;		break;
		}

		for( i=0; i<n_of_CRs; i++ ) {
			if( recv_a_byte(SINGLE_QSP_ARG) != CR ) {
				WARN("CR not received");
				return -1;
			}
		}

		if ( recv_a_byte(SINGLE_QSP_ARG) != ACK ) {
			WARN("ERROR: no ACK received");
			clear_input_buf(SINGLE_QSP_ARG);
			return -1;
		}
	}

	return 0;

}


static int get_echo(QSP_ARG_DECL  char *pkt)
{
	u_short i;
	u_short len;
	int *echo_bufp;

	len = strlen(pkt);

	echo_bufp = (int *)malloc(len);

	if( ascii_mode ) {
		for( i=0; i<len; i++ ) {
			*(echo_bufp+i) = recv_a_byte(SINGLE_QSP_ARG);

			if( *(echo_bufp+i) < 0 )
				return -1;

			if( *(echo_bufp+i) != *(pkt+i) ) {
				sprintf(ERROR_STRING, "Unexpected 0x%x instead of 0x%x received .... please restart the usb2000",
					*(echo_bufp+i), *(pkt+i) );
				WARN(ERROR_STRING);

				clear_input_buf(SINGLE_QSP_ARG);
				return -1;
			}
		}
	}
	return 0;
}

#define baud_rate(data_word) _baud_rate(QSP_ARG  data_word)

static void _baud_rate(QSP_ARG_DECL  int data_word)
{
	switch(data_word){
		case 0:	set_baud(usb2000_fd, B2400);  advise("Communicating at 2400 baud");  break;
		case 1:	set_baud(usb2000_fd, B4800);  advise("Communicating at 4800 baud");  break;
		case 2:	set_baud(usb2000_fd, B9600);  advise("Communicating at 9600 baud");  break;
		case 3:	set_baud(usb2000_fd, B19200); advise("Communicating at 19200 baud"); break;
		case 4:	set_baud(usb2000_fd, B38400); advise("Communicating at 38400 baud"); break;
		case 5:	set_baud(usb2000_fd, B57600); advise("Communicating at 57600 baud"); break;
	}
}

int set_baud_rate(QSP_ARG_DECL  int data_word)
{
	char pkt[20];
	char cmd[] = "K";
	int reply;

	if(data_word == 5 && baud_clk_timer != _16_BIT_BAUD_TIMER) {
		WARN("ERROR: 57600 baud rate requires use of 16 bit baud rate generator");
		return -1;
	}

	make_pkt( pkt, cmd, data_word );

	/*
	 * refer to manual pg.10 for description
	 * of steps.
	 */

	/* step 1: */
	send_pkt(QSP_ARG  pkt);

	if( get_echo(QSP_ARG  (char *)pkt) < 0 )
		return -1;

	/* step 2: */

	/*
	 * BUG: the following should work fine, however, the device
	 * doesn't send an ACK and yet it still changes the baud rate!?
	 */

	if( ( reply = recv_a_byte(SINGLE_QSP_ARG) ) == NAK ) {
		WARN("ERROR: (1st) ACK not received");
		clear_input_buf(SINGLE_QSP_ARG);
		return -1;
	}

	if( reply != ACK ){
		sprintf(ERROR_STRING,"set_baud_rate: reply (0x%x) != ACK (0x%x)", reply,ACK);
		advise(ERROR_STRING);
	}

	/* step 3: */
	usleep(5000);

	/*
	 * Since an ACK has been received we can go ahead
	 * and change the baud rate.
	 */

	/* step 4: */
	baud_rate(data_word);


	/* step 5: */
	send_pkt(QSP_ARG  pkt);

	if( ascii_mode ) {

		u_short i;

		for( i=0; i<2; i++ ) {
			if( recv_a_byte(SINGLE_QSP_ARG) < 0 ) {
				return -1;
			}
		}
	}

	if ( recv_a_byte(SINGLE_QSP_ARG) != ACK ) {
		WARN("(2nd) ACK not received");
		clear_input_buf(SINGLE_QSP_ARG);
		return -1;
	}

	if( get_tail(SINGLE_QSP_ARG) < 0 )
		return -1;

	return 0;

}

static short verify_baud_rate(SINGLE_QSP_ARG_DECL)
{
	int rate=0;
	int reply;

	reply = verify_data_mode(SINGLE_QSP_ARG);

	while( reply < 0 && rate < 6 ) {

		baud_rate(rate++);
		reply = verify_data_mode(SINGLE_QSP_ARG);

		if( rate > 5 && reply < 0 ) {
			WARN("ERROR: Unable to determine baud rate.....Device cannot be used (is device off?)");
			return -1;
		}

		clear_input_buf(SINGLE_QSP_ARG);
	}

	return 0;
}

void init_usb2000(SINGLE_QSP_ARG_DECL)
{
	usb2000_fd = open_serial_device("/dev/usb2000");

	if( usb2000_fd < 0 )
		error1("unable to open usb2000 serial port device");

	/* we first verify the baud rate and data mode */

	if( verify_baud_rate(SINGLE_QSP_ARG) < 0 )
		return;
#ifdef DEBUG
else if( debug & usb2000_debug ){
advise("Data mode verified");
}
#endif /* DEBUG */

	/* Some default settings are now done */

	if( set_timer(QSP_ARG  _16_BIT_BAUD_TIMER) < 0 )
		return;
#ifdef DEBUG
else if( debug & usb2000_debug ){
advise("16 bits assigned to baud rate timer");
}
#endif /* DEBUG */

	if( data_strg(QSP_ARG  DISABLE) < 0 )
		return;
#ifdef DEBUG
else if( debug & usb2000_debug ){
advise("Data storage mode disabled.....spectral scans will be transmitted out serial port");
}
#endif /* DEBUG */

	if( do_cmd_1_arg(QSP_ARG  CHKSUM, DISABLE) < 0 )
		return;
#ifdef DEBUG
else if( debug & usb2000_debug ){
advise("Checksum mode disabled");
}
#endif /* DEBUG */

	if( set_integ_time(QSP_ARG  100) < 0 )
		return;
#ifdef DEBUG
else if( debug & usb2000_debug ){
advise("Integration time set to 100 milliseconds");
}
#endif /* DEBUG */

}


int set_timer(QSP_ARG_DECL  int data_word)
{
	int timer_mode;

	/* Changing the timer mode re-initializes the
	 * baud rate to 9600(manual, pg.14). Instead
	 * of relying on the device (which has proved to
	 * be unreliable!) we change the baud rate
	 * ourselves.
	 */

	if( set_baud_rate(QSP_ARG  2) < 0 )
		return -1;

	timer_mode = do_cmd_1_arg(QSP_ARG  TIMER, data_word);

	if( timer_mode < 0 )
		return timer_mode;

	if( data_word == _8_BIT_TIMER ) {
		intg_clk_timer = _8_BIT_TIMER;
		baud_clk_timer = _16_BIT_BAUD_TIMER;

	} else {
		intg_clk_timer = _16_BIT_TIMER;
		baud_clk_timer = _8_BIT_BAUD_TIMER;
	}

	return timer_mode;
}

void send_usb2000_packet(QSP_ARG_DECL  const char *pkt, int len)
{
	send_serial(usb2000_fd,(u_char *)pkt,len);
}

#endif /* USE_SERIAL_LINE */

