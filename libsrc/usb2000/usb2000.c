/* usb2000.c */

#include "quip_config.h"

#ifdef HAVE_STDLIB_H
#include <stdlib.h>     /* malloc(), strtol() */
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>     /* usleep() */
#endif

#ifdef HAVE_MATH_H
#include <math.h>	/* fabs() */
#endif

#include "quip_prot.h"
#include "usb2000.h"


#ifdef DEBUG
#include "debug.h"
u_long usb2000_debug=0;
#endif /* DEBUG */

/* globals */
u_short ascii_mode = TRUE;			/* default mode is binary (pg.12) */
u_short data_strg_mode = DISABLE;		/* default mode is to xmit the spectra thru serial port */

u_short intg_clk_timer = _8_BIT_TIMER;	/* default integration clock timer is 8 bits (pg.14) */
u_short chk_sum_mode = DISABLE;		/* default */
u_short data_comp_mode = DISABLE;

USB2000_Cmd_Def usb2000_cmd_tbl[]={

{ ADD_SCANS,		"A",		ONE_ARG		},
{ PB_WIDTH,		"B",		ONE_ARG		},
{ COMP,			"G",		ONE_ARG		},
{ INTG_TIME,		"I",		ONE_ARG		},
{ LAMP,			"J",		ONE_ARG		},
{ BAUD,			"K",		ONE_ARG		},
{ CLR_MEM,		"L",		ONE_ARG		},
{ STORGE,		"M",		ONE_ARG		},
{ PXL_MODE,		"P",		N_ARGS		},
{ SPEC_ACQ,		"S",		NO_ARG		},
{ TRIG_MODE,		"T",		ONE_ARG		},
{ N_MEM_SCANS,		"W",		ONE_ARG		},
{ RD_MEM_SCAN,		"Z",		ONE_ARG		},
{ ASCII,		"aA",		NO_ARG		},
{ BINARY,		"bB",		NO_ARG		},
{ CHKSUM,		"k",		ONE_ARG		},
{ VER,			"v",		NO_ARG		},
{ CALIB_CONSTS,		"x",		N_ARGS		},
{ TIMER,		"y",		ONE_ARG		},
{ INQ,			"?",		ONE_ARG		},
{ ACCESSORIES,		"+",		NO_ARG		},
{ USB_ID,		"=",		NO_ARG		},

{ LS_450_ANALOG_OP,	"oA",		ONE_ARG		},
{ LS_450_LED_MODE,	"oj",		ONE_ARG		},
{ LS_450_TEMPERATURE,	"ot",		NO_ARG		},
{ LS_450_CALIB_CONST,	"ou",		N_ARGS		},
{ LS_450_INQ,		"o?",		ONE_ARG		}

};


#ifdef DEBUG
/* a debugging function */
static void dump_buf(char *buf)
{
	while( *buf != '\0') {
		sprintf(ERROR_STRING, "0x%x	%c", *buf, *buf );
		advise(ERROR_STRING);
		buf++;
	}

}
#endif // DEBUG

void _make_pkt(QSP_ARG_DECL  char *pkt, const char *cmd, u_int arg )
{
	if( ascii_mode ) {
			sprintf((char *)pkt,"%s%d\n", cmd, arg);
	} else {
		u_short cmd_len;

		cmd_len = strlen(cmd);

		if( cmd_len == 1 )
			sprintf(pkt,"%x%.4x", *cmd, arg);

		else if( cmd_len == 2 )
			sprintf(pkt,"%.2x%.2x%.4x", *cmd, *(cmd+1), arg);

#ifdef CAUTIOUS
		else {
			sprintf(ERROR_STRING,"ERROR: cmd_len: %d not possible", cmd_len);
			warn(ERROR_STRING);
		}
#endif /* CAUTIOUS */
	}
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

static int get_echo(QSP_ARG_DECL  const char *pkt)
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


int get_tail(SINGLE_QSP_ARG_DECL)
{
	u_short i;
	int recv_buf[4];

	if( ascii_mode ) {
		/* We need to recv LF CR > SPACE */

		for(i=0; i<4; i++) {
			if( ( recv_buf[i] = recv_a_byte(SINGLE_QSP_ARG) ) < 0 )
				return -1;
		}

		if( recv_buf[0] != LF || recv_buf[1] != CR || recv_buf[2] != 0x3e || recv_buf[3] != SPACE ) {
			sprintf(ERROR_STRING, "ERROR: Unexpected trailing chars (0x%x 0x%x 0x%x 0x%x) received",
				recv_buf[0], recv_buf[1] ,recv_buf[2] ,recv_buf[3]);
			WARN(ERROR_STRING);
			clear_input_buf(SINGLE_QSP_ARG);
			return -1;
		}
	}

	return 0;
}

/*
 * do_cmd_1_arg/do_inq_1_arg means that the cmd has ONE_ARG in the table.
 */

int do_cmd_1_arg(QSP_ARG_DECL  Cmd_Index cmd_index, int data_word)
{
	u_short len;
	char pkt[20];
	USB2000_Cmd_Def *ucdp;

	ucdp = &usb2000_cmd_tbl[cmd_index];

	make_pkt( pkt, ucdp->ucd_cmd, data_word );

	send_pkt(QSP_ARG  pkt);

	len = strlen(pkt);

	if( get_echo(QSP_ARG  pkt) < 0 )
		return -1;

	if ( recv_a_byte(SINGLE_QSP_ARG) != ACK ) {
		WARN("ERROR: no ACK received");
		return -1;
	}

	if( get_tail(SINGLE_QSP_ARG) < 0 )
		return -1;

	return 0;
}

static int do_inq_1_arg(QSP_ARG_DECL  Cmd_Index cmd_index)
{
	char pkt[MAX_PKT_SIZE];
	int data_value;
	u_short len;

	USB2000_Cmd_Def *ucdp;

	ucdp = &usb2000_cmd_tbl[cmd_index];

	if( ucdp->ucd_index == LS_450_ANALOG_OP	|| ucdp->ucd_index == LS_450_LED_MODE )
		sprintf(pkt, "o?%c", *(ucdp->ucd_cmd+1) );
	else
		sprintf(pkt, "?%s", ucdp->ucd_cmd);

	len = strlen(pkt);

	//send_serial( usb2000_fd, pkt, len);
	send_usb2000_packet(QSP_ARG  pkt,len);

	/* For inquiries (ASCII mode only) in which the  pkt is of the format:
	 * (o) ? inq
	 * the response is of the form:
	 * (o) ? inq ACK/NAK data_word SPACE \n \r > SPACE.
	 * For binary mode the response is:
	 * ACK/NAK data_word
	 */

	if( get_echo(QSP_ARG  pkt) < 0 )
		return -1;

	if ( recv_a_byte(SINGLE_QSP_ARG) != ACK ) {
		WARN("ERROR: no ACK received");
		return -1;
	}


	if( ( data_value = recv_a_value(SINGLE_QSP_ARG) ) < 0 )
		return -1;

	if( get_tail(SINGLE_QSP_ARG) < 0 )
		return -1;

	return data_value;

}


int add_scans(QSP_ARG_DECL  int data_word)	 	{ 	return do_cmd_1_arg(QSP_ARG  ADD_SCANS, data_word);		}
int set_pb_width(QSP_ARG_DECL  int data_word)		{ 	return do_cmd_1_arg(QSP_ARG  PB_WIDTH, data_word);		}
int set_lamp(QSP_ARG_DECL  int data_word) 		{ 	return do_cmd_1_arg(QSP_ARG  LAMP, data_word);			}
int clr_spectra(QSP_ARG_DECL  int data_word) 		{ 	return do_cmd_1_arg(QSP_ARG  CLR_MEM, data_word);		}
int set_trig_mode(QSP_ARG_DECL  int data_word) 	{ 	return do_cmd_1_arg(QSP_ARG  TRIG_MODE, data_word);		}

int get_n_scans(SINGLE_QSP_ARG_DECL)			{ 	return do_inq_1_arg(QSP_ARG  ADD_SCANS);				}
int get_pb_width(SINGLE_QSP_ARG_DECL)			{ 	return do_inq_1_arg(QSP_ARG  PB_WIDTH);				}
int get_integ_time(SINGLE_QSP_ARG_DECL)		{ 	return do_inq_1_arg(QSP_ARG  INTG_TIME);				}
int get_lamp_status(SINGLE_QSP_ARG_DECL)		{ 	return do_inq_1_arg(QSP_ARG  LAMP);				}
int get_baud_rate(SINGLE_QSP_ARG_DECL)			{ 	return do_inq_1_arg(QSP_ARG  BAUD);				}
int get_trig_mode(SINGLE_QSP_ARG_DECL)			{ 	return do_inq_1_arg(QSP_ARG  TRIG_MODE);				}
int get_timer_type(SINGLE_QSP_ARG_DECL)		{ 	return do_inq_1_arg(QSP_ARG  TIMER);				}

int set_analog_op(QSP_ARG_DECL  int analog_op_val) 	{ 	return do_cmd_1_arg(QSP_ARG  LS_450_ANALOG_OP, analog_op_val); 	}
int set_led_mode(QSP_ARG_DECL  int led_mode)		{ 	return do_cmd_1_arg(QSP_ARG  LS_450_LED_MODE, led_mode); 	}
int get_analog_op(SINGLE_QSP_ARG_DECL) 		{ 	return do_inq_1_arg(QSP_ARG  LS_450_ANALOG_OP);			}
int get_led_mode(SINGLE_QSP_ARG_DECL)			{ 	return do_inq_1_arg(QSP_ARG  LS_450_LED_MODE); 			}


int set_integ_time(QSP_ARG_DECL  int data_word)
{

	if( intg_clk_timer == _8_BIT_TIMER && data_word > 255 ) {
		WARN("ERROR:8-bit integration clock timer is in use. Cannot use value greater than 255");
		return -1;
	}

	return do_cmd_1_arg(QSP_ARG  INTG_TIME, data_word);
}


int data_strg(QSP_ARG_DECL  int data_word)
{
	if( do_cmd_1_arg(QSP_ARG  STORGE, data_word) < 0 )
		return -1;

	if( data_word == ENABLE_STRG )
		data_strg_mode = ENABLE;
	else
		data_strg_mode = DISABLE;

	return 0;
}

int set_checksum(QSP_ARG_DECL  int data_word)
{
	if( do_cmd_1_arg(QSP_ARG  CHKSUM, data_word) < 0 )
		return -1;

	if( data_word == ENABLE )
		chk_sum_mode = ENABLE;
	else
		chk_sum_mode = DISABLE;

	return 0;
}

int data_comp(QSP_ARG_DECL  int data_word)
{
	if( do_cmd_1_arg(QSP_ARG  COMP, data_word) < 0 )
		return -1;

	if( data_word == ENABLE )
		data_comp_mode = ENABLE;
	else
		data_comp_mode = DISABLE;

	return 0;
}

int get_n_of_scans(SINGLE_QSP_ARG_DECL)
{
	char pkt[6];	/* will contain either "W1\n" or "W0001" */
	u_short len;
	int is_ack;
	int data_value;
	char cmd[] = "W";

	make_pkt( pkt, cmd, 1 );

	len = strlen(pkt);

	send_pkt(QSP_ARG  pkt);

	if( get_echo(QSP_ARG  pkt) < 0 )
		return -1;

	if ( ( is_ack = recv_a_byte(SINGLE_QSP_ARG) ) != ACK ) {
		if( is_ack == NAK ) {
			if( get_tail(SINGLE_QSP_ARG) < 0 )
				return -1;

		} else {

			sprintf(ERROR_STRING,"ERROR: Unexpected char 0x%x received instead of ACK/NAK", is_ack);
			WARN(ERROR_STRING);

		}
		return -1;
	}

	if( ( data_value = recv_a_value(SINGLE_QSP_ARG) ) < 0 )
		return -1;

	if( get_tail(SINGLE_QSP_ARG) < 0 )
		return -1;

	return data_value;
}

/* This function is used by spec_acq() to get STX in the spectra header */
static int get_STX(SINGLE_QSP_ARG_DECL)
{
	u_short i;
	u_short n_to_get;
	int response=(-1);	/* initialize to silence compiler warnings */

	/* get STX/ETX (manual pg.11) */

	if( ascii_mode )
		n_to_get = 2;		/* considering echo */
	else
		n_to_get = 1;

	for( i=0; i<n_to_get; i++ ) {
		if( ( response = recv_a_byte(SINGLE_QSP_ARG) ) < 0 ) {
			return -1;
		}
	}

	if( response != STX ) {
	/* STX is sent back for successful spectra acquisition */

		if ( response!=ETX ) {

			sprintf(ERROR_STRING, "unexpected 0x%x instead of ETX", response);
			WARN(ERROR_STRING);

			/* we now clear up any 'trash' that may be present
			 * in the input buffer
			 */

			clear_input_buf(SINGLE_QSP_ARG);
			return -1;
		}

		#ifdef	CAUTIOUS
		/* ETX is a 'legal' reply. We don't expect any 'trash'
		 * to be present in the input buffer.
		 * This is only to be on the safe side.
		 */

		clear_input_buf(SINGLE_QSP_ARG);
		#endif /* CAUTIOUS */

		return -1;
	}

	return 0;
}


/* refer to pg.11 of manual for header details */
static int recv_spec_headers(QSP_ARG_DECL  Pxl_Mode_Info *pxl_mode_info_p)
{
	int pxl_case;
	u_short i;

	/* We recv the first six spectra headers
	 * mentioned on pg. 11 of manual.
	 */

	for(i=0; i<6; i++) {
		if( recv_a_value(SINGLE_QSP_ARG) < 0 )
			return -1;
	}

	if( ( pxl_case = recv_a_value(SINGLE_QSP_ARG) ) < 0 )
		return -1;

	/* refer to pg.11 of manual and look at the
	 * pixel mode cases to understand the calculations
	 * below.
	 */

	/* If the case is not 0 then the spectral header
	 * will contain the parameters passed to the
	 * Pixel Mode command
	 */

	switch(pxl_case) {
		case 0:
			pxl_mode_info_p->pxl_case = 0;
			break;

		case 1:
			pxl_mode_info_p->pxl_case = 1;

			if( ( pxl_mode_info_p->n = recv_a_value(SINGLE_QSP_ARG) ) < 0 )
				return -1;

			break;

		case 2:
			pxl_mode_info_p->pxl_case = 2;

			WARN("ERROR: recv_spec_headers: Impossible pixel mode");
			return -1;

		case 3:
		{
			/*
			 * For ascii mode we expect x SPACE y SPACE n SPACE.
			 * For binary mode we expect x y n, with each value
			 * comprising 16 bits.
			 * Briefly put: we expect 3 values.
			 */

			pxl_mode_info_p->pxl_case = 3;

			if( ( pxl_mode_info_p->x = recv_a_value(SINGLE_QSP_ARG) ) < 0 )
				return -1;

			if( ( pxl_mode_info_p->y = recv_a_value(SINGLE_QSP_ARG) ) < 0 )
				return -1;

			if( ( pxl_mode_info_p->n = recv_a_value(SINGLE_QSP_ARG) ) < 0 )
				return -1;

			/*
			for(i=0; i<3; i++) {
				if( recv_a_value(SINGLE_QSP_ARG) < 0 )
					return -1;
			}
			*/

			break;
		}

		case 4:
		{
			short n_pixels;

			pxl_mode_info_p->pxl_case = 4;

			if( ( n_pixels = recv_a_value(SINGLE_QSP_ARG) ) < 0 )		/* i.e. the number of pixels */
				return -1;

			pxl_mode_info_p->n = n_pixels;

			/*
			 * The spectral header has pixel locations.
			 */

			for( i=0; i<n_pixels; i++ ) {

				if( (pxl_mode_info_p->pixels[i]=recv_a_value(SINGLE_QSP_ARG))<0 )
					return -1;
			}

			/*
			while(n_pixels--) {
				if( recv_a_value(SINGLE_QSP_ARG) < 0 )
					return -1;
			}
			*/

			break;
		}

		default:
			sprintf(ERROR_STRING, "recv_spec_headers: Impossible pixel mode (%d)", pxl_case);
			WARN(ERROR_STRING);
			return -1;
	}

	return 0;
}

/*
 * These are the coefficients for the 3rd order equation describing
 * the relationship between the pixel number and wavlength.
 */
#define	W0	(347.5329171)
#define	W1	(0.360240559)
#define	W2	(-1.38931e-05)
#define	W3	(-2.83118e-09)


/*
 * These are the coefficients for the 7rd order equation describing
 * the non-linearity correction for each pixel number.
 */
#define N0 	(9.934660e-001)
#define N1 	(3.368403e-005)
#define N2 	(1.010907e-007)
#define N3 	(-1.057911e-010)
#define N4 	(5.690444e-014)
#define N5 	(-1.704312e-017)
#define N6 	(2.684913e-021)
#define N7 	(-1.739109e-025)


static int get_spectrum(QSP_ARG_DECL  Spectral_Data *sdp, u_short *n_spec_recvd, Pxl_Mode_Info *pxl_mode_info_p)
{
	int datum_received;
	u_short check_sum=0;
	u_short i=0;

	*n_spec_recvd=0;

	if( recv_spec_headers(QSP_ARG  pxl_mode_info_p) < 0 ) {
		WARN("ERROR: error receiving spectra header");
		return -1;
	}


	/*
	 * BUG: the microcode version 2.30.0 is buggy! Enabling data compression
	 * causes the pixel mode header to change to weird value.
	 * I've been told by tech support at Ocean Optics that this has
	 * been fixed for microcode version 2.31.0.
	 */

	if(data_comp_mode == ENABLE && !ascii_mode) {


		/* First value is sent uncompressed (pg.18) by the device. */
		if( ( datum_received = recv_a_value(SINGLE_QSP_ARG) ) < 0 )
			return -1;

		sdp->sd_spec_data[i++] = (u_short)datum_received;

		while( ( datum_received = recv_a_byte(SINGLE_QSP_ARG) ) != 0xfffd ) {
			if( datum_received < 0 )
				return -1;

			if( datum_received >= 0x80 ) {
				if( ( datum_received = recv_a_value(SINGLE_QSP_ARG) ) < 0 )
					return -1;

				sdp->sd_spec_data[i++] = (u_short)datum_received;

			} else {

				sdp->sd_spec_data[i] = sdp->sd_spec_data[i-1] + (char)datum_received;
				i++;
			}
		}

	} else {

		while( ( ( datum_received = recv_a_value(SINGLE_QSP_ARG) ) != 0xfffd ) && datum_received >= 0 ) {
			sdp->sd_spec_data[i++] = (u_short)datum_received;
			(*n_spec_recvd)++;

			if(!ascii_mode && chk_sum_mode == ENABLE)
				check_sum += (u_short)datum_received;
		}
	}

	//sprintf(ERROR_STRING, "n_of_spectra received: %d", *n_spec_recvd);
	//advise(ERROR_STRING);

#ifdef DEBUG
if( debug & usb2000_debug ){
if(chk_sum_mode==ENABLE) {
sprintf(ERROR_STRING, "check_sum(calculated): 0x%x", check_sum );
advise(ERROR_STRING);
}
}
#endif /* DEBUG */

	return check_sum;
}

static void do_wavlen_crktion(Spectral_Data *sdp, int n_spec_recvd, Pxl_Mode_Info *pxl_mode_info_p )
{
	u_short i;
	u_short j;

	switch(pxl_mode_info_p->pxl_case) {
		case 0:
			for( i=0; i<n_spec_recvd; i++ )
				sdp->sd_wavlen[i] = W0 + W1*i + W2*i*i + W3*i*i*i;

			break;

		case 1:
			for( i=0; i<n_spec_recvd; i++ ) {
				j = i*pxl_mode_info_p->n;
				sdp->sd_wavlen[i] = W0 + W1*j + W2*j*j + W3*j*j*j;
			}

			break;

		case 2:
			NWARN("Impossible pixel mode");
			break;

		case 3:
			for( i=0; i<n_spec_recvd; i++ ) {
				j = pxl_mode_info_p->x+i*pxl_mode_info_p->n;
				sdp->sd_wavlen[i] = W0 + W1*j + W2*j*j + W3*j*j*j;
			}

			break;

		case 4:
			for( i=0; i<n_spec_recvd; i++ ) {
				j = pxl_mode_info_p->pixels[i];
				sdp->sd_wavlen[i] = W0 + W1*j + W2*j*j + W3*j*j*j;
			}

			break;

		default:
			NWARN("do_wavelen_crktion:  unknown case");
	}

	/*
	for( i=0; i<n_spec_recvd; i++ )
		sdp->sd_wavlen[i] = W0 + W1*i + W2*i*i + W3*i*i*i;
	*/


}

static void do_non_linear_crktion(Spectral_Data *sdp, int n_spec_recvd)
{
	u_short i;
	float rd;

	for( i=0; i<n_spec_recvd; i++ ) {
		rd = sdp->sd_spec_data[i];

		sdp->sd_spec_data[i] = N0 + N1*rd + N2*rd*rd + N3*rd*rd*rd + N4*rd*rd*rd*rd + N5*rd*rd*rd*rd*rd + N6*rd*rd*rd*rd*rd*rd + N7*rd*rd*rd*rd*rd*rd*rd;
	/*
	sprintf(ERROR_STRING, "xform7[%d]: %f", i, sdp->sd_spec_data[i] );
	advise(ERROR_STRING);
	*/

	}
}

int get_scan(QSP_ARG_DECL  Spectral_Data *sdp)
{
	char pkt[6];		/* will contain either "Z1\n" or "Z0001" */
	int data_word = 1;	/* pg.12, 1 or 0 doesn't matter */
	char cmd[] = "Z";
	u_short len;
	u_short n_spec_recvd[1];
	Pxl_Mode_Info pxl_mode_info;

	/* BUG: we should have a routine here that checks
	 * if we have scans in memory or not.
	 * This will be better than sending a pkt
	 * and getting a NAK due to absence of spectra
	 * in data memory.
	 */

	make_pkt( pkt, cmd, data_word );

	send_pkt(QSP_ARG  pkt);

	len = strlen(pkt);

	if( get_echo(QSP_ARG  pkt) < 0 )
		return -1;

	if ( recv_a_byte(SINGLE_QSP_ARG) != ACK ) {
		WARN("ERROR: no ACK received");
		return -1;
	}

	if( get_spectrum(QSP_ARG  sdp, n_spec_recvd, &pxl_mode_info) < 0 )
		return -1;

	if( get_tail(SINGLE_QSP_ARG) < 0 )
		return -1;

	do_wavlen_crktion(sdp, n_spec_recvd[0], &pxl_mode_info);
	do_non_linear_crktion(sdp, n_spec_recvd[0]);

	return n_spec_recvd[0];

}

int spec_acq(QSP_ARG_DECL  Spectral_Data *sdp)
{
	const char *cmdp="S\0";
	int check_sum;
	int device_check_sum;
	char tmp[5];
	u_short i;
	int data_recvd;
	int n;
	u_short n_spec_recvd[1];
	Pxl_Mode_Info pxl_mode_info;

	//send_serial( usb2000_fd, cmdp, strlen(cmdp) );
	send_usb2000_packet(QSP_ARG  cmdp,strlen(cmdp));

	if( get_STX(SINGLE_QSP_ARG) < 0 )
		return -1;


	if( data_strg_mode == DISABLE ) {
		if( ( n = get_spectrum(QSP_ARG  sdp, n_spec_recvd, &pxl_mode_info) ) < 0 )
			return -1;

		/* In check sum mode the check sum is transmitted after the
		 * spectrum terminating header.
		 */

		if(chk_sum_mode == ENABLE) {
			check_sum = n;

			if(!ascii_mode) {

				for( i=0; i<2; i++) {
					if( ( data_recvd = recv_a_byte(SINGLE_QSP_ARG) ) < 0 )
						return -1;

					tmp[i] = (u_char)data_recvd;
				}

				sprintf(tmp, "%.2x%.2x", tmp[0], tmp[1]);

				tmp[4] = '\0';

				device_check_sum = strtol( tmp, NULL, 16 );

#ifdef DEBUG
sprintf(ERROR_STRING, "check_sum(received): 0x%x ", check_sum );
advise(ERROR_STRING);
#endif /* DEBUG */
				if( device_check_sum != check_sum ) {
					sprintf(ERROR_STRING, "ERROR: Unexpected check sum(0x%x) instead of 0x%x",
						check_sum, device_check_sum );
					WARN(ERROR_STRING);
					return -1;
				}

			} else {
				if( recv_a_value(SINGLE_QSP_ARG) != 0 )
					return -1;
			}
		}
	}

	if( get_tail(SINGLE_QSP_ARG) < 0 )
		return -1;

#ifdef DEBUG
if( debug & usb2000_debug ){
sprintf(ERROR_STRING,"case:%d n:%d x:%d y:%d",pxl_mode_info.pxl_case,pxl_mode_info.n ,pxl_mode_info.x, pxl_mode_info.y);
advise(ERROR_STRING);

for(i=0;i<10;i++) {
sprintf(ERROR_STRING,"pixel(%d):%d",i,pxl_mode_info.pixels[i]);
advise(ERROR_STRING);
}
}
#endif /* DEBUG */

	do_wavlen_crktion(sdp, n_spec_recvd[0], &pxl_mode_info);
	do_non_linear_crktion(sdp, n_spec_recvd[0]);

	return n_spec_recvd[0];
}

int get_ver(QSP_ARG_DECL  char *ver)
{
	const char *pkt = "v";
	int data_value;
	u_short len;

	len = 1;
	//send_serial( usb2000_fd, pkt, len );
	send_usb2000_packet(QSP_ARG  pkt,len);

	if( get_echo(QSP_ARG  pkt) < 0 )
		return -1;

	if ( recv_a_byte(SINGLE_QSP_ARG) != ACK ) {
		WARN("ERROR: no ACK received");
		return -1;
	}

	if( ( data_value = recv_a_value(SINGLE_QSP_ARG) ) < 0 )
		return -1;

	if( get_tail(SINGLE_QSP_ARG) < 0 )
		return -1;

	sprintf(ver,"%d",data_value);
	sprintf( ver, "%c.%c%c.%c",ver[0], ver[1], ver[2], ver[3]);
	return 0;
}

int get_temp(QSP_ARG_DECL  char *temp)
{
	char pkt[] = "ot";
	u_short i;
	u_short len;
	int char_recvd;
	u_short index = 0;

	len = strlen(pkt);

	//send_serial( usb2000_fd, pkt, len );
	send_usb2000_packet(QSP_ARG  pkt,len);

	/* The response is of the form
	 * o t ACK/NAK temperature LF CR > SPACE
	 */

	if( get_echo(QSP_ARG  pkt) < 0 )
		return -1;


	if ( recv_a_byte(SINGLE_QSP_ARG) != ACK ) {
		WARN("ERROR: no ACK received");
		return -1;
	}

	/* I've observed that the size of the string is always 4 */

	for( i=0; i<4; i++) {
		if( ( char_recvd = recv_a_byte(SINGLE_QSP_ARG) ) < 0 )
			return -1;
		sprintf(&temp[index++], "%c", (char)char_recvd);
	}

	temp[index] = '\0';

	if( get_tail(SINGLE_QSP_ARG) < 0 )
		return -1;

	return 0;
}

int set_data_mode(QSP_ARG_DECL  Cmd_Index data_mode)
{
	u_short len;
	char pkt[3];
	USB2000_Cmd_Def *ucdp;
	u_short n_to_get;
	u_short i;

	ucdp = &usb2000_cmd_tbl[data_mode];
	sprintf(pkt, "%s", ucdp->ucd_cmd);
	len = strlen(pkt);

	//send_serial( usb2000_fd, pkt, len );
	send_usb2000_packet(QSP_ARG  pkt,len);

	/****************************************************************
		bin->ascii	ascii->bin	ascii->ascii	bin->bin
	cmd:	aA		bB		aA		bB

	reply:	A		b		a		B
		ACK		B		A		ACK
		LF		ACK		ACK		LF
		CR		LF		LF		CR
		>		CR		CR		>
		SPACE		>		>		SPACE
				SPACE		SPACE

	*****************************************************************/

	if( ascii_mode )
		n_to_get = len;
	else
		n_to_get = len-1;

	for( i=0; i<n_to_get; i++ ) {
		if( recv_a_byte(SINGLE_QSP_ARG) < 0 )
			return -1;
	}

	if ( recv_a_byte(SINGLE_QSP_ARG) != ACK ) {
		WARN("ERROR: no ACK received");
		clear_input_buf(SINGLE_QSP_ARG);
		return -1;
	}

       	if(data_mode == ASCII)
               	ascii_mode=TRUE;
	else
       	        ascii_mode=FALSE;

	if( get_tail(SINGLE_QSP_ARG) < 0 )
		return -1;

	return 0;
}


static short xmit_calib_const_pkt(QSP_ARG_DECL  char *pkt)
{
	u_short i;
	int n_expect;
	int echo_buf[7];	/* we will not be receiving more than
				 * 7 bytes at a time
				 */

	while( *pkt ){

		u_short n_waits;

		//send_serial( usb2000_fd, pkt, 1 );
		send_usb2000_packet(QSP_ARG   pkt, 1 );

		if( *pkt == '\n' || *pkt == '\r' ) {
			if( *(pkt+1) ) {	/* not the last char */
				n_expect = 2;

			} else {
				n_expect = 7;	/* LF CR ACK LF CR > SPACE */
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

	if( echo_buf[2] != ACK ) {
		WARN("ACK not received");
		clear_input_buf(SINGLE_QSP_ARG);
		return -1;
	}

	return 0;

}


int set_calib_const(QSP_ARG_DECL  Cmd_Index cmd_index, int calib_index, const char *coeff_value)
{
	USB2000_Cmd_Def *ucdp;
	char pkt[100];		/* BUG need to use a definite value */

	ucdp = &usb2000_cmd_tbl[cmd_index];

	sprintf(pkt, "%s%d\n%s\n", ucdp->ucd_cmd, calib_index, coeff_value);

#ifdef DEBUG
	dump_buf(pkt);
#endif /* DEBUG */

	if( xmit_calib_const_pkt(QSP_ARG  pkt) < 0 )
		return -1;

	return 0;

}

#define wavlen_to_pxl(wavlen) _wavlen_to_pxl(QSP_ARG  wavlen)

static u_short _wavlen_to_pxl(QSP_ARG_DECL  float wavlen)
{
	u_short i;
	float diff;
	float best_diff=1;
	float tmp_wavlen,best_wav=0;
	u_short pixel=0;

	for( i=0; i<2048; i++ ) {
		tmp_wavlen = W0 + W1*i + W2*i*i + W3*i*i*i;

		diff = fabs(tmp_wavlen-wavlen);

		if( diff<best_diff ) {
			best_diff = diff;
			best_wav=tmp_wavlen;
			pixel = i;
		}
	}

	sprintf(ERROR_STRING,"approximating %fnm to %fnm",
		wavlen, best_wav);
	advise(ERROR_STRING);

	return pixel;
}

#ifdef FOOBAR
int round(float f)
{
	int i;

	i=(int)f;

	if( (f-i)>=0.5 )
		f++;
	else
		f = (float)i;

	#ifdef DEBUG
	sprintf(ERROR_STRING,"my_int_round: %f to %d",f, (int)f);
	advise(ERROR_STRING);
	#endif /* DEBUG */

	return (int)f;
}
#endif /* FOOBAR */


int set_pxl_mode(QSP_ARG_DECL  int pxl_mode, float x, float y, float n, float *wavelengths_p)
{
	char pkt[50];
	USB2000_Cmd_Def *ucdp;
	u_short i;

	ucdp = &usb2000_cmd_tbl[PXL_MODE];

	switch(pxl_mode) {
		case 0:
			make_pxl_mode_pkt( pkt, pxl_mode, 0, 0, 0, NULL );
			break;

		case 1:

			n = n*BIN_WIDTH;
			n = round(n);

			make_pxl_mode_pkt( pkt, pxl_mode, 0, 0, (int)n, NULL );
			break;

		case 2:
			pxl_mode = 3;

			x = wavlen_to_pxl(x);
			y = wavlen_to_pxl(y);

			n = n*BIN_WIDTH;
			n = round(n);

			make_pxl_mode_pkt( pkt, pxl_mode, (int)x, (int)y, (int)n, NULL );
			break;

		case 3:
		{
			int pxl_locations[10];
			pxl_mode = 4;

			for( i=0; i<n; i++)
				pxl_locations[i] = wavlen_to_pxl( *(wavelengths_p+i) );

			make_pxl_mode_pkt( pkt, pxl_mode, 0, 0, n, pxl_locations);
			break;
		}

	} /* switch */

	/*
	 * Setting pixel mode requires the use of a delay for ascii
	 * mode. This is an undocumented feature which should've been
	 * mentioned in the manual!
	 */

	if( xmit_pxl_mode_pkt(QSP_ARG  pkt, pxl_mode, n) < 0 ) {
		return -1;
	}

	return 0;
}

int do_calib_inq(QSP_ARG_DECL  Cmd_Index cmd_index, int calib_index, char *calib_const)
{
	char pkt[7];		/* (o) + ? + x + index + \n + \0 added by sprintf */
	USB2000_Cmd_Def *ucdp;
	int byte_recvd = 0;
	u_short i = 0;
	char recv_buf[MAX_SIZEOF_CALIB_CONST+1];
	u_short buf_index;

	ucdp = &usb2000_cmd_tbl[cmd_index];

	if( ucdp->ucd_index == LS_450_CALIB_CONST )
		sprintf(pkt, "o?u%d\n", calib_index);
	else
		sprintf(pkt, "?x%d\n", calib_index);

	//send_serial( usb2000_fd, pkt, strlen(pkt) );
	send_usb2000_packet(QSP_ARG   pkt, strlen(pkt) );

	while(byte_recvd != 0x3e) {	/* i.e > */

		if( ( byte_recvd = recv_a_byte(SINGLE_QSP_ARG) ) < 0 ) {
			clear_input_buf(SINGLE_QSP_ARG);
			return -1;
		}

		recv_buf[i++] = (char)byte_recvd;
	}

	if( ( recv_buf[i++] = recv_a_byte(SINGLE_QSP_ARG) ) != 0x20 ) {
		clear_input_buf(SINGLE_QSP_ARG);
		return -1;
	}

	recv_buf[i] = '\0';

#ifdef DEBUG
	dump_buf(recv_buf);
#endif /* DEBUG */

	buf_index = strlen(pkt)+1;

	if( recv_buf[buf_index] != ACK ) {
		WARN("ACK not received");
		return -1;
	}

	buf_index++;
	i=0;

	while( recv_buf[buf_index+i] != '\n' ) {
		*(calib_const+i) = recv_buf[buf_index+i];
		i++;

	}

	*(calib_const+i) = '\0';

	return 0;
}


