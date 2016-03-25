/* usb_funcs.c
 *
 * This functions replace the ones in tty_funcs.c, but they don't work yet
 * because we don't know the usb protocol.
 */

#include "quip_config.h"


#include "usb2000.h"

#ifndef USE_SERIAL_LINE

#ifdef HAVE_USB_H
#include <usb.h>	/* */
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* usleep */
#endif

#ifdef HAVE_STRING_H
#include <string.h>	/* strlen */
#endif


#include "debug.h"

#ifdef HAVE_ASM_ERRNO_H
#include <asm/errno.h>
#endif


#define OCEAN_OPTICS_VENDOR_ID		0x2457
#define OCEAN_OPTICS_USB2000_PID	0x1002

static usb_dev_handle *spectrometer_dev_h;
static struct usb_device const* spectrometer_dev;
static int spectrometer_ifNum;

//#define DEBUG


static u_short baud_clk_timer = _16_BIT_BAUD_TIMER;	/* default integration clock timer is 16 bits (pg.14) */

static short verify_data_mode()
{
	char pkt[] = "\n";
	int reply ;

	//send_serial(usb2000_fd, pkt, strlen(pkt) );
	send_usb2000_packet(pkt, strlen(pkt) );

	advise("Checking data mode");

	if( ( reply = recv_a_byte() ) == LF ) {

		if( recv_a_byte() == NAK ) {

			if( get_tail() < 0 ) {
				warn("ERROR: mode might be ascii");
				warn("The device needs a reboot");
			
			} else {
				ascii_mode = TRUE;
				advise("....data mode is ascii");
				return 0;
			}
		}
		
		return -1;
		
	} else if( reply == NAK ) {
		ascii_mode = FALSE;
		advise("....data mode is binary");
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
void clear_input_buf()
{
	//char buf[MAX_PKT_SIZE];
	
	//if( !(recv_somex(usb2000_fd, buf, MAX_PKT_SIZE-1,MAX_PKT_SIZE-1)) )
	//	usleep(1000);

	usleep(1000);

	//if( !(recv_somex(usb2000_fd, buf, MAX_PKT_SIZE-1, MAX_PKT_SIZE-1)) )
		//usleep(1000);
}

static void make_pxl_mode_pkt( char *pkt, int pxl_mode, int x, int y, int n, int *pxl_locations_p)
{
	u_short i;

	if( ascii_mode ) {
		switch(pxl_mode) {
			case 0:
				sprintf(pkt, "P0\n");
				break;
			case 1:
				sprintf(pkt, "P1\n%d\n", n);
				break;
			case 3:
				sprintf(pkt, "P3\n%d\n%d\n%d\n", x, y, n);
				break;
			case 4:
				sprintf(pkt, "P4\n%d\n", n);
				
				for(i=0; i<n; i++) {
					pkt += strlen(pkt);
					sprintf(pkt, "%d\n", pxl_locations_p[i]);
				}
				break;
		}

	} else {	/* binary mode */
		switch(pxl_mode) {

				/* P = 0x50 */
			case 0:
				sprintf(pkt, "500000");
				break;
			case 1:
				sprintf(pkt, "500001%.4x", n);
				break;
			case 3:
				sprintf(pkt, "500003%.4x%.4x%.4x", x, y, n);
				break;
			case 4:
				sprintf(pkt, "500004%.4x", n);

				for(i=0; i<n; i++) {
					pkt += strlen(pkt);
					sprintf(pkt, "%.4x", pxl_locations_p[i]);
				}
				break;
		}
	}
}

int recv_a_byte()
{
	unsigned char recv_buf[3] = {0};
	u_short n_received = 0;
	int n_waits = 0;
	int value_recvd;
	int ret;

	while( n_received < 1 && n_waits < MAX_WAITS ) {
		ret=usb_get_string(spectrometer_dev_h,0/* index */,0/* langid */,recv_buf,3);
		if( ret < 0 ){
			warn("usb_get_string: error!?");
		}

		if( ret == 0 ) {
			n_waits++;
			usleep(1); /* .001 ms */
		}
	}

	if( n_waits >= MAX_WAITS && n_received == 0 ) {
		sprintf(error_string,"No value received after waiting %d n_waits?", n_waits);
		warn(error_string);
		return -1;
	}
	
	value_recvd = (int)recv_buf[0];

	#ifdef DEBUG
	sprintf(error_string,"byte recvd: 0x%.2x", value_recvd);
	advise(error_string);
	#endif /* DEBUG */	

	return value_recvd; 
}


/* A value is 'defined' to be either
 * data teminated by SPACE for ascii mode
 * or two bytes for binary mode.
 */
 
int recv_a_value()
{
	unsigned char recv_buf[1] = {0};
#define MAX_SIZEOF_VALUE	10
	unsigned char value[MAX_SIZEOF_VALUE];
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
		
		/* all binary data is in the form of a word (pg.12) */ 
		for( n_words_received=0;  n_words_received<2; n_words_received++ ) {
	
			while( n_received < 1 && n_waits < MAX_WAITS ) {
				n_received=usb_get_string(spectrometer_dev_h,0/* index */,0/* langid */,recv_buf,1);
				//n_received = recv_somex( usb2000_fd, recv_buf, 1, 1 );
				if( !n_received ) {
					n_waits++;
					usleep(1); /* .001 ms */
				}
			}
			
			if( n_waits >= MAX_WAITS && n_received == 0 ) {
				sprintf(error_string,"No value received after waiting %d n_waits?", n_waits);
				warn(error_string);
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
				n_received=usb_get_string(spectrometer_dev_h,0/* index */,0/* langid */,recv_buf,1);
				//n_received = recv_somex( usb2000_fd, recv_buf, 1, 1 );
				if( !n_received ) {
					n_waits++;
					usleep(1); /* .001 ms */
				}
			}
		
			if( n_waits >= MAX_WAITS && n_received == 0 ) {
				sprintf(error_string,"No value received after waiting %d n_waits?", n_waits);
				warn(error_string);
				sprintf(error_string,"%d characters already received",i);
				advise(error_string);
				
				if( i > 0 ){
					value[i]=0;	/* terminate string */
					sprintf(error_string,"string \"%s\" received so far",value);
					advise(error_string);
				}
				return -1;		
			}
			
			n_waits = 0;
			n_received = 0;
	
			#ifdef CAUTIOUS
			if( i > MAX_SIZEOF_VALUE-1 ) {
				value[i]=0;	/* terminate string */
				sprintf(error_string, "Impossiblely large value (%s) is being received, Reception ABORTED.", value);
				warn(error_string);
			}
			#endif /* CAUTIOUS */
	
			value[i++] = recv_buf[0];
		}
	
		value[--i] = '\0';	/* replace space with null char */
		n_value = (int)strtol(value, NULL, 10);
	
	} /* else */

	#ifdef DEBUG
	sprintf(error_string, "n_value: %d", n_value ); 
	advise(error_string);
	#endif /* DEBUG */
	
	return n_value;
}


int send_pkt( u_char *pkt )
{
	int ret=0;

advise("send_pkt BEGIN");
	if( ascii_mode ) 
		//send_serial( usb2000_fd, pkt, strlen(pkt) );
		ret=usb_control_msg(spectrometer_dev_h,0/*request_type*/,0/*request*/,0/* value */,
			0/*index*/,pkt,strlen(pkt),1/* timeout */ );
	
	else 	/* if binary mode */	
		//send_hex( usb2000_fd, pkt ); 
		warn("binary mode not implemented yet for usb");
	
	if( ret < 0 )
		warn("error returned by usb_control_msg");

	return 0;

}

int xmit_pxl_mode_pkt( char *pkt, int pxl_mode, int n )
{
	u_short i;

advise("xmit_pxl_mode_pkt BEGIN");
	if( ascii_mode ) {
		
		int n_expect;
		int echo_buf[7];	/* we will not be receiving more than 
					 * 7 bytes at a time 
					 */ 
	
		while( *pkt ){
			
			int n_waits;
			int ret;
	
			//send_serial( usb2000_fd, pkt, 1 );
			ret=usb_control_msg(spectrometer_dev_h,0/*request_type*/,0/*request*/,0/* value */,
				0/*index*/,pkt,1,1/* timeout */ );

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
					if( ( echo_buf[i] = recv_a_byte() ) != (int)*pkt ) 
						return -1;
	
				} else {		
					if( ( echo_buf[i] = recv_a_byte() ) < 0 ) 
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
			warn("ACK not received");
			return -1;
		}	
	
	} else {	/* if binary mode */
		
		u_short n_of_CRs=(-1);	/* init to silence compiler warning */
		
		/* I have observed that a delay is not required in binary mode.
		 */ 
		
		//send_hex( usb2000_fd, pkt );
		warn("missing call to send_hex");

	
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
			if( recv_a_byte() != CR ) {
				warn("CR not received");
				return -1;
			}	
		}
		
		if ( recv_a_byte() != ACK ) {
			warn("ERROR: no ACK received");
			clear_input_buf();
			return -1;
		}
	}
	
	return 0;

}


static int get_echo(char *pkt)
{
	u_short i;
	u_short len;
	int *echo_bufp;

	len = strlen(pkt);
	
	echo_bufp = (int *)malloc(len);

	if( ascii_mode ) {
		for( i=0; i<len; i++ ) {
			*(echo_bufp+i) = recv_a_byte();
			
			if( *(echo_bufp+i) < 0 )	
				return -1;
			
			if( *(echo_bufp+i) != *(pkt+i) ) { 
				sprintf(error_string, "Unexpected 0x%x instead of 0x%x received .... please restart the usb2000", 
					*(echo_bufp+i), *(pkt+i) );
				warn(error_string);
				
				clear_input_buf();
				return -1;
			}
		}
	}
	return 0;
}

void init_usb2000()
{
	/* find the interface */

	/* WARNING: not thread-safe. usb_busses in libusb/usb.c is a global
	 * variable.
	 */
	struct usb_bus *usbbus;
	struct usb_device *usbdev;
	usb_dev_handle *usbdev_h;
	struct usb_device const* dev;
	int n;
	int err;

	usb_init();
	n=usb_find_busses();
	if( n <= 0 ){
		warn("No USB busses found");
		return;
	}

	n=usb_find_devices();
	if( n <= 0 ){
		warn("No USB devices found");
		return;
	}

	usbbus = usb_get_busses();
	for (; usbbus; usbbus = usbbus->next) {
		for (usbdev = usbbus->devices; usbdev; usbdev=usbdev->next) {
			usbdev_h = usb_open(usbdev);
			if (usbdev_h) {
				dev = usb_device(usbdev_h);

				if( dev->descriptor.idVendor == OCEAN_OPTICS_VENDOR_ID &&
					dev->descriptor.idProduct == OCEAN_OPTICS_USB2000_PID ){

					sprintf(error_string,"Ocean Optics USB2000 found at %s/%s",
						usbbus->dirname,usbdev->filename);
					advise(error_string);
					spectrometer_dev_h = usbdev_h;
					spectrometer_dev = dev;
//sprintf(error_string,"num altsettings = %d, first interface number = %d",
//usbdev->config->interface->num_altsetting,
//usbdev->config->interface->altsetting->bInterfaceNumber);
//advise(error_string);
					spectrometer_ifNum = usbdev->config->interface->altsetting->bInterfaceNumber;
					if( (err=usb_claim_interface(usbdev_h,spectrometer_ifNum)) < 0 ){
						tell_sys_error("usb_claim_interface");
						if( err == -EBUSY )
							advise("interface not available");
						else if( err == -ENOMEM )
							advise("not enough memory");
						return;
					}
					return;
				}

				//usb_release_interface(usbdev_h, hidif->interface);
				usb_close(usbdev_h);
			} else {
				sprintf(error_string,"Failed to open USB device %s/%s",
					usbbus->dirname,usbdev->filename);
	  			warn(error_string);
			}
		}
	}
	warn("No Ocean Optics USB200 found.");
	return;

	//usb_release_interface(spectrometer_dev_h, spectrometer_dev->config->interface->altsetting->bInterfaceNumber);

	#ifdef DEBUG
	usb2000_debug = add_debug_module("usb2000");
	#endif /* DEBUG */

	/* we first verify the baud rate and data mode */

	//if( verify_baud_rate() < 0 )
		//return;
	//else
		//advise("Data mode verified");

	/* Some default settings are now done */
	
	if( set_timer(_16_BIT_BAUD_TIMER) < 0 )
		return;
	else 
		advise("16 bits assigned to baud rate timer");

	if( data_strg(DISABLE) < 0 )
		return;
	else
		advise("Data storage mode disabled.....spectral scans will be transmitted out serial port");
	
	if( do_cmd_1_arg(CHKSUM, DISABLE) < 0 )
		return;
	else 
		advise("Checksum mode disabled");

	if( set_integ_time(100) < 0 )
		return;
	else 
		advise("Integration time set to 100 milliseconds");
}


int set_timer(int data_word) 	
{ 
	int timer_mode; 
	
	/* Changing the timer mode re-initializes the 
	 * baud rate to 9600(manual, pg.14). Instead
	 * of relying on the device (which has proved to
	 * be unreliable!) we change the baud rate
	 * ourselves.
	 */
	 
	//if( set_baud_rate(2) < 0 )
		//return -1;
	
	timer_mode = do_cmd_1_arg(TIMER, data_word);	
	
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

void send_usb2000_packet(char *pkt, int len)
{
	int ret;
	//send_serial(usb2000_fd,pkt,len);
advise("send_usb2000_packet BEGIN");
// 0xc8 requesttype is IN transfer
// 0x48 requesttype is OUT transfer
	ret=usb_control_msg(spectrometer_dev_h,0xc8/*request_type*/,0/*request*/,0/* value */,
				0/*index*/,pkt,len,1/* timeout */ );
	if( ret < 0 )
		warn("error returned by usb_control_msg");
}


#endif /* ! USE_SERIAL_LINE */
