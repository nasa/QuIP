#include "quip_config.h"

char VersionId_polhemus_polh_dev[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* qsort() */
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifndef INSIDE_TRACK
#ifdef HAVE_TERMIOS_H
#include <termios.h>
#endif /* HAVE_TERMIOS_H */
#endif /* !INSIDE_TRACK */

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif /* HAVE_PTHREAD_H */

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#include "myerror.h"
#include "getbuf.h"
#include "savestr.h"
#include "sigpush.h"

#include "ioctl_polhemus.h"
#include "polh_dev.h"
#include "debug.h"

#include "query.h"
#define POLHEMUS_SYNC_VAR_NAME	"polhemus_sync_mode"

#ifdef INSIDE_TRACK
#define USE_DATA_DEV			/* We'd like to avoid having two devices,
					 * but for now that's how it is...
					 */
#endif

/* globals */

/* polhemus file descriptors */
static int active_mask=0;

int polh_fd = (-1);
#ifdef USE_DATA_DEV
static int polh_data_fd = (-1);
#endif /* USE_DATA_DEV */

int which_receiver = POLH_RECV;		/* polhemus receiver number */
int polh_continuous = 0;		/* continuous output flag */
int polh_units = PH_CM_FMT;		/* units format (default is cm.) */
int n_active_stations=0;
Station_Data station_info[2];

static int polh_software_hz = DEFAULT_SYNC_HZ;	/* software synchronization hz rate */
static char last_command[64];

#define RESP_BUF_SIZE	64
#define POLHEMUS_LINE_SIZE	(RESP_BUF_SIZE*sizeof(short))

short resp_buf[RESP_BUF_SIZE];
#define polh_line_buffer	((char *)resp_buf)

int n_response_chars;

/* Implement data buffering for polhemus device */
#define POLHEMUS_BUFFER_SIZE	4096
static char polh_buffer[POLHEMUS_BUFFER_SIZE];
static int polh_next_wr=0;
static int polh_next_rd=0;
static int n_polh_free=POLHEMUS_BUFFER_SIZE;
static int n_polh_avail=0;

#define GET_CHUNK(n_want)						\
									\
	if( (m=read(polh_fd,&polh_buffer[polh_next_wr],n_want)) !=	\
							n_want ){	\
		if( m < 0 ){						\
			tell_sys_error("read");				\
			warn("error reading polhemus data");		\
		} else {						\
			sprintf(error_string,				\
		"Expected %d polhemus bytes, got %d",n_want,m);		\
			warn(error_string);				\
		}							\
	} else {							\
		if( debug & debug_polhemus ){				\
			sprintf(error_string,				\
				"fill_polh_buffer read %d bytes",m);	\
			advise(error_string);				\
		}							\
	}								\
	polh_next_wr += n_want;						\
	if( polh_next_wr >= POLHEMUS_BUFFER_SIZE )			\
		polh_next_wr = 0;					\
	n_polh_free -= n_want;						\
	n_polh_avail += n_want;


int n_printable(short *buf,int n)
{
	char *s;
	int i;

	for(i=0;i<n;i++) {
		s = (char *) &buf[i];
		if( ! isprint(*s) ) return(2*i);
		if( !isprint(*(s+1)) ) return(1+2*i);
	}

	return(2*n);
}

void print_string(short *buf,int n)
{
	char cbuf[LLEN];

	if( n > LLEN-1 ){
		warn("print_string:  buffer size too small for data");
		return;
	}

	memcpy(cbuf,buf,n);
	cbuf[n]=0;
	prt_msg(cbuf);
}

void display_buffer(short *buf,int n)
{
	char *cp;
	char str[4],*s;

	/*
	if( (m=n_printable(buf,n)) > 0 ) print_string(buf,m);
	*/

	while(n--) {
		cp = (char *)buf;

		str[0]=(*cp);
		str[1]=0;
		s=show_printable(str);
		sprintf(msg_str,"\t0x%x\t\t0x%x\t%s", *buf, *cp, s);
		prt_msg_frag(msg_str);

		cp++;
		str[0]=(*cp);
		str[1]=0;
		s=show_printable(str);
		sprintf(msg_str,"\t0x%x\t%s", *cp, s);
		prt_msg(msg_str);

		buf++;
 	}
}

#ifdef INSIDE_TRACK
static int open_polh_dev(void)
{
	if( (polh_fd = open(POLH_DEV, O_RDWR)) < 0 ) {
		sprintf(error_string,"open_polh_dev: opening polhemus device %s",POLH_DEV);
		tell_sys_error(error_string);
		return(-1);
	}

#ifdef USE_DATA_DEV
	if( (polh_data_fd = open(POLH_DATA_DEV, O_RDONLY) ) < 0) {
		sprintf(error_string,"open_polh_dev: opening polhemus data device %s",POLH_DATA_DEV);
		tell_sys_error(error_string);
		return(-1);
	}
#endif /* USE_DATA_DEV */

	return(0);
}

#else

void flush_input_data()
{
	int n;

	usleep(100000);

	if( ioctl(polh_fd,FIONREAD,&n) < 0 ){
		perror("ioctl (FIONREAD)");
		warn("error getting polhemus word count");
		n=1;	/* just in case... */
	}
	if( n > 0 ){
		sprintf(error_string,"Flushing %d pending input characters...",n);
		advise(error_string);

		if( tcflush(polh_fd,TCIFLUSH) < 0 ){
			tell_sys_error("tcflush");
		}
	}
}

void flush_polh_buffer(void)
{
	flush_input_data();

	polh_next_wr=0;
	polh_next_rd=0;
	n_polh_free=POLHEMUS_BUFFER_SIZE;
	n_polh_avail=0;
}

void fill_polh_buffer(void)
{
	int n,m,n1;

	if( ioctl(polh_fd,FIONREAD,&n) < 0 ){
		perror("ioctl (FIONREAD)");
		warn("error getting number of readable polhemus chars");
		return;
	}
	if( n == 0 ) return;
//sprintf(error_string,"fill_polh_buffer:  %d chars available",n);
//advise(error_string);
	if( n <= n_polh_free ){		/* we have room for the data */
		if( n <= (POLHEMUS_BUFFER_SIZE-polh_next_wr) ){		/* all fits without wraparound */
//sprintf(error_string,"fill_polh_buffer:  trying to get chunk of %d chars",n);
//advise(error_string);

			GET_CHUNK(n)
		} else {						/* need to wrap-around */
			n1=POLHEMUS_BUFFER_SIZE-polh_next_wr;
//sprintf(error_string,"fill_polh_buffer:  trying to get chunk of %d chars",n1);
//advise(error_string);
			GET_CHUNK(n1)
			n -= n1;
//sprintf(error_string,"fill_polh_buffer:  trying to get chunk of %d chars",n);
//advise(error_string);
			GET_CHUNK(n)
		}
	} else {
		sprintf(error_string,"%d chars readable polhemus characters, but only %d free buffer locs",
			n,n_polh_free);
		warn(error_string);
	}
}

/* Pretty inefficient - need a string xfer func, but for now try this and see... */

int polh_getc()
{
	int c;

	if( n_polh_avail < 1 )
		return(-1);
	c = polh_buffer[polh_next_rd];
	polh_next_rd++;
	if( polh_next_rd >= POLHEMUS_BUFFER_SIZE )
		polh_next_rd=0;
	n_polh_avail--;
	n_polh_free++;
	return(c);
}

int polh_ungetc(int c)
{
	polh_next_rd--;
	if( polh_next_rd < 0 ){
		polh_next_rd = POLHEMUS_BUFFER_SIZE-1;
	}
	n_polh_avail++;
	n_polh_free--;
	/* Don't need to put it back, should already be there! */
	//polh_buffer[polh_next_rd]=c;
	return 0;		// BUG need to add error checks
}

static int open_polh_dev(void)
{
	if( (polh_fd = open(POLH_DEV, O_RDWR)) < 0 ) {
		sprintf(error_string,"open_polh_dev: opening polhemus device %s",POLH_DEV);
		tell_sys_error(error_string);
		return(-1);
	}

	/* Now set baud rate etc. */

	/* after fastrak power-up, the device doesn't respond to the reset (W)
	 * command until a carriage return has been sent, which triggers
	 * an error message!?
	 */

	flush_input_data();

	/* Should we flush any old data, reset device to known state, etc?
	 * What if the device is currently in continuous mode?
  	 */
	if(send_polh_cmd(PH_REINIT_SYS, NULL) < 0 ) {
		warn("open_polh_dev:  unable to reinitialize system");
		return(-1);
	}
	/* The reinit command will halt continuous output, but we should now wait a few
	 * clicks and then flush any pending output from the device.
	 */
	flush_input_data();

	return(0);
} /* end open_polh_dev */

	
#endif

void read_response(int display_flag)
{
	int n;

	n = polhemus_word_count();

	if( n <= 0 ) return;

	if( n > RESP_BUF_SIZE ){
		sprintf(error_string,"%d response words available, buffer size is only %d",
			n,RESP_BUF_SIZE);
		warn(error_string);
		n = RESP_BUF_SIZE;
	}

	n *= 2;		/* byte count */

	if( (n_response_chars=read(polh_fd,resp_buf,n)) != n ){
		if( n_response_chars < 0 ) {
			tell_sys_error("read_response");
			warn("error reading polhemus data");
		} else {
			sprintf(error_string,"read_response:  %d bytes requested, %d actually read",n,n_response_chars);
			warn(error_string);
		}
	}
	if( display_flag )
		display_buffer(resp_buf,n_response_chars/2);
}

void clear_polh_dev(void)
{
	short clear_buf[RESP_BUF_SIZE];
	int n;
	int n_want;
	int n_read;
	int total=0;

	if( polhemus_word_count()<0 ){
		advise("no bytes to clear from polhemus device");
		return;
	}

	while ( (n = polhemus_word_count()) != 0 ) {
		if( n < RESP_BUF_SIZE ) n_want = n;
		else n_want = RESP_BUF_SIZE;
		if( n_want % 2 ) n_want++;

		if( (n_read=read(polh_fd, clear_buf, n_want*2)) != (n_want*2) ) {
			if( n_read < 0 ) {
				tell_sys_error("clear_polh_dev");
				warn("error clearing polhemus device");
				return;
			} else {
				sprintf(error_string,"clear_polh_dev: %d bytes requested, %d actually read",n_want*2, n_read);
				warn(error_string);
			}	
		}	
		total += n_read;
	}
	return;
}

static void init_output_data_format(int station)
{
	Polh_Record_Format *prfp;

	prfp = &station_info[station].sd_multi_prf;

	prfp->rf_n_data = 4;
	prfp->rf_output[0] = SECONDS;
	prfp->rf_output[1] = MSECS;
	prfp->rf_output[2] = XYZ_INT;
	prfp->rf_output[3] = EULER_INT;
	prfp->rf_station = station;
	prfp->rf_n_words = 12;
	polhemus_output_data_format( prfp );

	/*
	show_output_data_format(station);
	*/
}

static int set_default_polh(void)
{
#ifdef INSIDE_TRACK
	/* initialize to floating output */
	/* station number, xyz_int, euler_int, crlf */
	if(send_string("O1,2,4,1\r") < 0 ) { 
		warn("set_default_polh: Unable to initialize to floating output");
		return(-1);
	}

	usleep(100000);	/* sleep 1/10 sec */
#endif

	clear_polh_dev();

	/* set to centimeter output */
	if(send_polh_cmd(PH_CM_FMT, NULL) < 0 ) {
		warn("set_default_polh: Unable to set to centimeter output");
		return(-1);
	}

	/* set active station to 1 */
#ifdef USE_DATA_DEV
	if(send_string("l1,1\r") < 0 ){
		warn("set_default_polh: Unable to set active station number to 1");
		return(-1);
	}
#endif /* USE_DATA_DEV */

	/* Set the default record formats */
	init_output_data_format(0);
	init_output_data_format(1);

	get_active_stations();	/* make sure the flags reflect reality */

	if( ! STATION_IS_ACTIVE(0) )
		activate_station(0,1);	/* make station 1 active */
	if( STATION_IS_ACTIVE(1) )
		activate_station(1,0);	/* make station 2 inactive */
	curr_station=0;
	
	usleep(100000);
	clear_polh_dev();

	if( send_string("l2,0\r") < 0 ) { 
		warn("set_default_polh: Unable to set non-active station number to 2");
		return(-1);
	}
	
	usleep(100000);
	clear_polh_dev();

	return(0);
}

int init_polh_dev(void)
{
#ifdef CAUTIOUS
	if( polh_fd >= 0 
#ifdef USE_DATA_DEV
			|| polh_data_fd >= 0
#endif /* USE_DATA_DEV */
			) {
		warn("CAUTIOUS: init_polh_dev: polhemus device already intialized!?");
		return(0);
	}
#endif /* CAUTIOUS */

	if(open_polh_dev() < 0 || set_default_polh() < 0) return(-1);

	return(0);
}

/* What is the point of this?  Are we trying to reset the driver?
 */

int reopen_polh_dev(void)
{
#ifdef CAUTIOUS
	if( (!(polh_fd >= 0))
#ifdef USE_DATA_DEV
			&& !(polh_data_fd >= 0)
#endif /* USE_DATA_DEV */
			) {
		warn("CAUTIOUS: reopen_polh_dev: polhemus device not opened to close!?"); 
		return(-1);
	}
#endif /* CAUTIOUS */

	if( close(polh_fd) < 0 ) {
		tell_sys_error("unable to close polhemus device");
		return(-1);
	}

#ifdef USE_DATA_DEV
	if( close(polh_data_fd) < 0 ) {
		tell_sys_error("unable to close polhemus data device");
		return(-1);
	}
	polh_data_fd=-1;
#endif /* USE_DATA_DEV */
	polh_fd=-1;
	
	return( init_polh_dev() );
}

#ifdef INSIDE_TRACK
#define MAX_POLHEMUS_READS	100

static int read_until(char *databuf, int n_want, int fd )
{
	int n_read;
	int n_orig;
	int num_reads=0;

	n_orig = n_want;

	/* Keep reading data until we get enough requested bytes */
	while( (n_read = read(fd, &databuf[n_orig-n_want], n_want)) != n_want ){
		if( n_read < 0  ){
			tell_sys_error("read_until");
			return(-1);
		}
		sprintf(msg_str,"read_until (fd=%d):  %d bytes requested, %d actually read",
			fd,n_want,n_read);
		advise(msg_str);
		n_want -= n_read;
		if( ++num_reads >= MAX_POLHEMUS_READS ) {
			warn("read_polh_dev: timed out from reading data");
			return(-1);
		}
	}
{
int i;
for(i=0;i<n_orig;i++){
sprintf(error_string,"%d\t0x%x\t%c",i,databuf[i],databuf[i]);
advise(error_string);
}
}

	return(0);
}

int read_polh_dev( short* databuf, int n_want )
{
	int n;

	if( (n=read_until((char *)databuf,n_want,polh_fd)) < 0 )
		warn("error reading polhemus");
	return(n);
}

#ifdef USE_DATA_DEV
int read_polh_data( void *raw_pdp, int n_want )
{
	int n;

	if( (n=read_until((char *)raw_pdp,n_want,polh_data_fd)) < 0 )
		warn("error reading polhemus data");
	return(n);
}
#else /* ! USE_DATA_DEV */

int read_polh_data( void *raw_pdp, int n_want )
{
	int n;

	if( (n=read_until((char *)raw_pdp,n_want,polh_fd)) < 0 )
		warn("error reading polhemus data");
	return(n);
}

#endif /* ! USE_DATA_DEV */

#define CHECK_STR( str )								\
											\
	if( strncmp( resp_chars+posn, str, strlen(str) ) ){				\
		/* display buffer */							\
		sprintf(error_string,"Expected to find string \"%s\" at position %d",	\
				str,posn);						\
		warn(error_string);							\
		display_buffer(&resp_buf[0],n_response_chars/2);			\
		return;									\
	}										\
	posn += strlen(str);

static void parse_error()
{
	int posn=0;
	char *resp_chars;
	int error_code;

	read_response(0);
	resp_chars = (char *) resp_buf;

	/* We assume that we have already read "2 E*" ... */

	CHECK_STR("2 E*ERROR*")
	CHECK_STR(last_command)
	CHECK_STR("*ERROR*")
	/* error code? */
	CHECK_STR(" EC ");
	/* now read in a number... */
	while( isspace(resp_chars[posn]) )
		posn++;
	if( resp_chars[posn] != '-' ){
		sprintf(error_string,"parse_error:  expected '-' at position %d",posn);
		warn(error_string);
		display_buffer(resp_buf,n_response_chars/2);
		return;
	}
	posn++;
	error_code=0;
	while( isdigit(resp_chars[posn]) ){
		error_code *= 10;
		error_code += resp_chars[posn] - '0';
		posn++;
	}
	error_code *= -1;
	switch( error_code ){
		case -1:	warn("required field missing"); break;
		case -2:	warn("required numeric field is non-numeric"); break;
		case -3:	warn("value is outside required range"); break;
		case -4:	warn("specified frequency not hardware configured"); break;
		case -99:	warn("undefined input - cannot identify command"); break;
		default:
				sprintf(error_string,"Unrecognized error code %d",error_code);
				warn(error_string);
				display_buffer(resp_buf,n_response_chars/2);
				return;
	}


advise("parse_error: calling display_buffer");
	display_buffer(&resp_buf[0],n_response_chars/2);			\
}

#endif /* INSIDE_TRACK */

/* read_polh_line()	reads data until a newline or carriage return is seen */

char *read_polh_line(void)
{
	int c;
	int i=0;

	while(1) {
		fill_polh_buffer();
		while( i < (POLHEMUS_LINE_SIZE-1) && (c=polh_getc()) >= 0 ){
/*
sprintf(error_string,"read_polh_line has character 0x%x",c);
advise(error_string);
*/
			/* the polhemus terminates lines with \r\n combos, but we
			 * end the line when we see the \r...  so when we go to read
			 * the next line, the first char we see is an \n - skip it!
			 */
			if( i!=0 || c!='\n')
				polh_line_buffer[i++] = c;
			if( c == '\r' ){
				polh_line_buffer[i] = 0;
				return(polh_line_buffer);
			}
		}
	}
	/* NOTREACHED */
	return(NULL);
}

static int recv_polh_data(QSP_ARG_DECL  Ph_Cmd_Code cmd)
{
	char *s;

	s=read_polh_line();

	switch(polh_cmds[cmd].pc_code){
		case PH_SYNC_MODE:
			/* "2 y0\r" */
			if( strncmp(s,"2 y",3) ){
				sprintf(error_string,"recv_polh_data PH_SYNC_MODE:  read \"%s\", expected \"2 y...\"",
					show_printable(s));
				warn(error_string);
			} else {
				switch(s[3]){
					case '0':
						prt_msg("Internal sync");
						ASSIGN_VAR(POLHEMUS_SYNC_VAR_NAME,"internal");
						break;
					case '1':
						prt_msg("External sync");
						ASSIGN_VAR(POLHEMUS_SYNC_VAR_NAME,"external");
						break;
					case '2':
						prt_msg("External sync");
						ASSIGN_VAR(POLHEMUS_SYNC_VAR_NAME,"software");
						break;
#ifdef CAUTIOUS
					default:
						sprintf(error_string,
					"CAUTIOUS:  recv_polh_data:  read \"%s\", unhandled sync mode",
							show_printable(s));
						warn(error_string);
						break;
#endif /* CAUTIOUS */
				}
			}
			break;

		case PH_STATUS:
sprintf(error_string,"PH_STATUS recv_polh_data received \"%s\", NOT parsing...",show_printable(s));
advise(error_string);
			break;
		case PH_STATION:
			/* 21l1000\r */
			if( strncmp(s,"21l",3) ){
				sprintf(error_string,"recv_polh_data PH_STATION:  read \"%s\", expected \"21l...\"",
					show_printable(s));
				warn(error_string);
			} else {
				int i=3;
				char *str;

				while(i<7){
					str=NULL;
					switch(s[i]){
						case '0':
							str="disabled";
							break;
						case '1':
							str="enabled";
							break;
						default:
							sprintf(error_string,
			"CAUTIOUS:  recv_polh_data PH_STATION:  read \"%s\", unexpected status char in position %d",
								show_printable(s),i);
							warn(error_string);
							break;
					}
					if( str != NULL ){
						sprintf(msg_str,"Station %d is %s",i-2,str);
						prt_msg(msg_str);
					}
					i++;
				}
			}
			break;
		case PH_ALIGNMENT:
sprintf(error_string,"PH_ALIGNMENT recv_polh_data received \"%s\", NOT parsing...",show_printable(s));
advise(error_string);
			break;
		case PH_XMTR_ANGLES:
		case PH_RECV_ANGLES:
		case PH_REF_BORESIGHT:
		case PH_ATT_FILTER:
		case PH_POS_FILTER:
		case PH_ANGULAR_ENV:
		case PH_POSITIONAL_ENV:
		case PH_HEMISPHERE:
if( debug & debug_polhemus ){
sprintf(error_string,"recv_polh_data received \"%s\", NOT parsing...",show_printable(s));
advise(error_string);
}
sprintf(error_string,"recv_polh_data received \"%s\", NOT parsing...",show_printable(s));
advise(error_string);
			/* BUG - need to parse the string */
			break;
		default:
			sprintf(error_string,"Unhandled case in recv_polh_data for %s command",
				polh_cmds[cmd].pc_name);
			warn(error_string);
sprintf(error_string,"recv_polh_data received \"%s\"",show_printable(s));
advise(error_string);
			return(-1);
			break;
	}
	return(0);
}

#define MAX_POLHEMUS_COUNT_WAITS	10 	/* one second */	

#ifdef FOOBAR
{
	int n,nwant,count_waits=0;
#ifdef CAUTIOUS
	if(polh_cmds[cmd].pc_rec_size <= 0) {
		warn("recv_polh_data: bad output record size!?");
		return(-1);
	}
#endif

	nwant = polh_cmds[cmd].pc_rec_size;
	n = polhemus_byte_count();

sprintf(error_string,"expect %d bytes in polhemus response, byte count is %d",
nwant,n);
advise(error_string);

	if( n < 0 ){
		warn("recv_polh_data:  problem determining byte count");
		return(-1);
	}


	while( n < nwant ){
advise("waiting for polhemus data");
		usleep(100000);	/* sleep 1/10 sec. */
		n = polhemus_byte_count();
		if( n < 0 ){
			warn("recv_polh_data:  problem determining byte count");
			return(-1);
		}
		if( ++count_waits >= MAX_POLHEMUS_COUNT_WAITS ) {
			warn("recv_polh_data: timed out waiting for data");
			return(-1);
		}
	}

sprintf(error_string,"expect %d bytes in polhemus response, byte count is %d",
nwant,n);
advise(error_string);

	if( n != nwant ){
		sprintf(error_string,"recv_polh_data:  expected %d bytes, but %d are available!?",
				nwant,n);
		warn(error_string);
		if( n > 2 ){	/* start to read anyway, check for error code... */
			parse_error();
			return(-1);
		} else {
			read_response(1);
			return(-1);
		}
	}

	if( read_polh_dev(resp_buf, nwant ) < 0 ) {
		sprintf(error_string, "recv_polh_data: error reading output record type %s of size %d bytes",
			polh_cmds[cmd].pc_cmdstr, nwant);
		warn(error_string);
		return(-1);
	}

	/* check for errors in the output record */
	if( check_polh_output((char *)resp_buf, curr_station, cmd ) < 0 ) {
		sprintf(error_string,"recv_polh_data: Unable to get %s", polh_cmds[cmd].pc_name);
		warn(error_string);
		return(-1);
	}	

#ifdef DEBUG
if( debug & debug_polhemus ){
sprintf(error_string,"received data:  %s",(char *)resp_buf);
advise(error_string);
}
#endif /* DEBUG */

	/* Now the buffer should be empty... */
	if( (n = polhemus_byte_count()) > 0 ){
		sprintf(error_string,"After reading response there are %d bytes available!?",n);
		warn(error_string);
	}
	return(0);
}
#endif /* FOOBAR */

/* Send a command to the polhemus */

int send_polh_cmd(Ph_Cmd_Code cmd, const char * cmdargs) 
{
	char code_id[LLEN];
	char command[LLEN];

#ifdef CAUTIOUS
	if( cmd < 0 || cmd >= N_PH_CMD_CODES ) {
		warn("send_polh_cmd: Unhandled command code!?");
		return(-1);
	}
#endif
	
	/* check if we can peform the requested operation with the command code */
	/* why does reinit command trigger this message??? */
	if( !CAN_SET(cmd) ) {
		sprintf(error_string,
			"send_polh_cmd: command %s cannot be used to SET", polh_cmds[cmd].pc_name);
		warn(error_string);
		return(-1);	
	}
		
	/* Print the command code itself, and possibly a station specifier.
	 * Check if the command needs transmitter, receiver, station number
	 * or if it doesn't need anything.
	 */
	switch( polh_cmds[cmd].pc_trs ) {
		case PH_NEED_XMTR : sprintf(code_id, "%s%d", polh_cmds[cmd].pc_cmdstr, POLH_XMTR_NUM); break;
		case PH_NEED_RECV : sprintf(code_id, "%s%d", polh_cmds[cmd].pc_cmdstr, which_receiver); break;
		case PH_NEED_STAT : sprintf(code_id, "%s%d", polh_cmds[cmd].pc_cmdstr, curr_station+1); break;
		case PH_NEED_NONE : sprintf(code_id, "%s", polh_cmds[cmd].pc_cmdstr); break;
#ifdef CAUTIOUS
		default : warn("send_polh_cmd: Unknown transmitter/receiver/station flag value!?"); return(-1); break;
#endif
	}

	/* Add additional arguments, if any.
	 * The Polhemus manual says that the 
	 * compensation and station commands
	 * are not followed by commas although 
	 * they have arguments.
	 */ 
	if(cmdargs) {
//sprintf(error_string,"Appending command args \"%s\" to %s command",
//show_printable(cmdargs),polh_cmds[cmd].pc_name);
//advise(error_string);
		switch(cmd){
			case PH_STATION:
			case PH_POS_FILTER:
			case PH_ATT_FILTER:
			case PH_SYNC_MODE:
				sprintf(command, "%s%s", code_id, cmdargs);
				break;
			case PH_REF_BORESIGHT:
			case PH_XMTR_ANGLES:
			case PH_RECV_ANGLES:
			case PH_ALIGNMENT:
			case PH_ANGULAR_ENV:
			case PH_POSITIONAL_ENV:
			case PH_HEMISPHERE:
				sprintf(command, "%s,%s", code_id, cmdargs);
				break;
			default:
				sprintf(error_string,"Unhandled Case in send_polh_cmd:  %s, cmd_args = \"%s\"",
					polh_cmds[cmd].pc_name,show_printable(cmdargs));
				warn(error_string);
				sprintf(command, "%s,%s", code_id, cmdargs);
				break;
		}
	} else {
		/* If there are no args, any needed carriage return should be part of the command string... */
		sprintf(command, "%s", code_id);
	}

	/* add a return if necessary */
	switch(cmd){
		case PH_INCHES_FMT:
		case PH_CM_FMT:
		case PH_REINIT_SYS:
		case PH_CONTINUOUS:
		case PH_NONCONTINUOUS:
		case PH_SINGLE_RECORD:
			/* do nothing */
			break;
		case PH_RESET_ALIGNMENT:
		case PH_BORESIGHT:
		case PH_REF_BORESIGHT:
		case PH_XMTR_ANGLES:
		case PH_RECV_ANGLES:
		case PH_RESET_BORESIGHT:
		case PH_ALIGNMENT:
		case PH_ATT_FILTER:
		case PH_POS_FILTER:
		case PH_ANGULAR_ENV:
		case PH_POSITIONAL_ENV:
		case PH_HEMISPHERE:
		case PH_SYNC_MODE:
			strcat(command,"\r");
			break;
		default:
			sprintf(error_string,"Unhandled case in send_polh_cmd:  %s",
				polh_cmds[cmd].pc_name);
			warn(error_string);
			break;
	}


#ifdef DEBUG
if( debug & debug_polhemus ){
sprintf(error_string,"Ready to send command string \"%s\"",show_printable(command));
advise(error_string);
}
#endif /* DEBUG */

	strcpy(last_command,command);

	/* Walt Johnson's Fastrak doesn't seem to expect a carriage return
	 *  after single char commands...
	 *  What other commands are like this?
	 */

	/*
	if( strlen(command) > 1 )
		strcat(command,"\r");
		*/

	/* write the command to the device */
	if( send_string(command) < 0 ) {
		sprintf(error_string,"send_polh_cmd: Unable to send command string %s",command);
		warn(error_string);
		return(-1);
	}
	/* FIXME - here we should put some error-checking to ensure that the command is set */

	return(0);
}

int get_polh_info(Ph_Cmd_Code cmd, const char * cmdargs) 
{
	char code_id[LLEN];
	char command[LLEN];

#ifdef CAUTIOUS
	if( cmd < 0 || cmd >= N_PH_CMD_CODES ) {
		warn("get_polh_info: Unhandled command code!?");
		return(-1);
	}
#endif
	
	/* check if we can peform the requested operation with the command code */
	if( !CAN_GET(cmd) ){
		sprintf(error_string, "get_polh_info: command %s is not a GET command",
			polh_cmds[cmd].pc_name);
		warn(error_string);
		return(-1);	
	}
		
	/* Print the command code itself, and possibly a station specifier.
	 * Check if the command needs transmitter, receiver, station number
	 * or if it doesn't need anything.
	 */
	switch( polh_cmds[cmd].pc_trs ) {
		case PH_NEED_XMTR : sprintf(code_id, "%s%d", polh_cmds[cmd].pc_cmdstr, POLH_XMTR_NUM); break;
		case PH_NEED_RECV : sprintf(code_id, "%s%d", polh_cmds[cmd].pc_cmdstr, which_receiver); break;
		case PH_NEED_STAT : sprintf(code_id, "%s%d", polh_cmds[cmd].pc_cmdstr, curr_station+1); break;
		case PH_NEED_NONE : sprintf(code_id, "%s", polh_cmds[cmd].pc_cmdstr); break;
#ifdef CAUTIOUS
		default : warn("send_polh_cmd: Unknown transmitter/receiver/station flag value!?"); return(-1); break;
#endif
	}

#ifdef FOOBAR
	/* Add additional arguments, if any.
	 * The Polhemus manual says that the 
	 * compensation and station commands
	 * are not followed by commas although 
	 * they have arguments.
	 */ 
	if(cmdargs) {
		switch(cmd){
			case PH_STATION:
advise("Oops");
				sprintf(command, "%s%s\r", code_id, cmdargs);
				break;
			case PH_POS_FILTER:
			case PH_ATT_FILTER:
		case PH_ANGULAR_ENV:
		case PH_POSITIONAL_ENV:
			case PH_HEMISPHERE:
advise("oops");
				sprintf(command, "%s%s", code_id, cmdargs);
				break;
			default:
				sprintf(error_string,"get_polh_info:  missing case for command %s",
					polh_cmds[cmd].pc_name);
				warn(error_string);
				sprintf(command, "%s,%s", code_id, cmdargs);
				break;
		}
	} else {
		sprintf(command, "%s", code_id);
	}
#endif

	sprintf(command, "%s", code_id);
	switch(cmd){
		case PH_RESET_ALIGNMENT:
		case PH_ALIGNMENT:
		case PH_STATION:
		case PH_REF_BORESIGHT:
		case PH_XMTR_ANGLES:
		case PH_RECV_ANGLES:
		case PH_ATT_FILTER:
		case PH_POS_FILTER:
		case PH_ANGULAR_ENV:
		case PH_POSITIONAL_ENV:
		case PH_HEMISPHERE:
		case PH_SYNC_MODE:
			strcat(command,"\r");
			break;
		case PH_STATUS:
			break;
		default:
sprintf(error_string,"get_polh_info:  unhandled case for %s command",polh_cmds[cmd].pc_name);
warn(error_string);
			break;
	}


#ifdef DEBUG
if( debug & debug_polhemus ){
sprintf(error_string,"ready to send command string \"%s\"",show_printable(command));
advise(error_string);
}
#endif /* DEBUG */

	strcpy(last_command,command);

	/* not all fastrak commands need the carriage return??? */

	/* write the command to the device */
	if( send_string(command) < 0 ) {
		sprintf(error_string,"send_polh_cmd: Unable to send command string %s",command);
		warn(error_string);
		return(-1);
	}
	
	return recv_polh_data(cmd);	
} /* end get_polh_info */

int send_string(const char *cmd)
{
	int numwritten;
	int len;
	
	if(cmd == NULL) {
		warn("send_string: null command string!?");
		return(-1);
	}

#ifdef DEBUG
if( debug & debug_polhemus ){
	sprintf(msg_str, "send_string: %s", show_printable(cmd) );
	advise(msg_str);
}
#endif
	
	len = strlen(cmd);
	numwritten = write(polh_fd, (const void*)cmd, len);

	if( numwritten != len ) {
		if(numwritten < 0 ) 
			tell_sys_error("send_string");
		else {
			sprintf(error_string, "Requested %d bytes to be written to polhemus device, %d actually written",
				len, numwritten);
			warn(error_string);
		}
		return(-1);
	}

	return(0);
}

		
#define MAX_PH_READS	100000	/* max. number of continuous reads */


int set_polh_sync_mode(int mode_code)
{
	int stat = 0;
	char arg_str[4];
	
	switch( mode_code ) {
		case PH_EXTERNAL_SYNC :
		case PH_INTERNAL_SYNC :
			sprintf(arg_str,"%d",mode_code);
			stat = send_polh_cmd(PH_SYNC_MODE,arg_str);
			break;
		case PH_SOFTWARE_SYNC :
			warn("Software sync not implemented!");
			return(-1);
#ifdef CAUTIOUS
		default :
			warn("set_polh_sync_mode:  Unexpected sync mode requsted!?");
			return(-1);
#endif
	}

	if(stat < 0) return(-1);
	else return(0);
}

void set_polh_sync_rate(long rate)
{
	polh_software_hz = rate;
	return;
}

/* The insidetrak device only has 2 stations, but fastrak has 4.
 */

void activate_station(int station,int flag)
{
	int bit;
#ifndef INSIDE_TRACK
	char cmd_str[32];
#endif

#ifdef CAUTIOUS
	if( station < 0 || station >= MAX_POLHEMUS_STATIONS ) {
		sprintf(error_string,
	"CAUTIOUS:  activate_station: bad station number %d, must be in the range from 0 to %d",
			station,MAX_POLHEMUS_STATIONS-1);
		warn(error_string);
		return;
	}
	if( flag < 0 || flag > 1 ){
		sprintf(error_string,
			"CAUTIOUS:  activate_station:  bad flag %d, should be 0 or 1",flag);
		warn(error_string);
		return;
	}
#endif /* CAUTIOUS */

	if( STATION_IS_ACTIVE(station) == flag ){
		sprintf(error_string,"activate_station:  station %d is already %s",
				station+1,flag?"activated":"deactivated");
		warn(error_string);
		return;
	}

#ifdef FOOBAR
	if( station == 0 )
		bit = STATION_1_ACTIVE;
	else
		bit = STATION_2_ACTIVE;
#else
	bit = 1 << station;
#endif

	if( flag )
		active_mask |= bit;
	else
		active_mask &= ~bit;

#ifdef INSIDE_TRACK
	if( ioctl( polh_fd, POLHEMUS_SET_ACTIVE, &active_mask ) < 0 ){
		perror("ioctl POLHEMUS_SET_ACTIVE");
		warn("error setting active stations");
		return;
	}
#else
	/* just send the command - this ought to work for insidetrak too? */
	sprintf(cmd_str,"l%d,%d\r",station+1,flag);
	if( send_string(cmd_str) < 0 ) {
		sprintf(error_string,"activate_station: Unable to send command string \"%s\"",
			cmd_str);
		warn(error_string);
		return;
	}
#endif

	/* remember state */
	if( flag ){
		station_info[station].sd_flags |= STATION_ACTIVE;
		which_receiver=station;
		curr_station=station;
	} else
		station_info[station].sd_flags &= ~STATION_ACTIVE;

	n_active_stations = (STATION_IS_ACTIVE(0)?1:0) + (STATION_IS_ACTIVE(1)?1:0);

	if( flag )
		sprintf(msg_str,"Activating station %d",station+1);
	else
		sprintf(msg_str,"Deactivating station %d",station+1);

	prt_msg(msg_str);

	/* Assume that if we change the station, we are working
	 * with the corresponding receiver for that station.
	 */

	return;
}

void set_polh_units(Ph_Cmd_Code cmd)
{

#ifdef CAUTIOUS
	if( cmd != PH_INCHES_FMT && cmd != PH_CM_FMT ) {
		warn("set_polh_units: unhandled unit format requested!?");
		return;
	}
#endif /* CAUTIOUS */
	
	if( send_polh_cmd(cmd,NULL) < 0 ) {
		warn("set_polh_units: unable to set requested units!?");
		return;
	}

	polh_units = cmd;
	return;
}


#ifdef CAUTIOUS
#define CHECK_POL_FD( who )							\
										\
	if( polh_fd < 0 ){							\
		sprintf(error_string,"%s:  polhemus device not open",who);	\
		warn(error_string);						\
		return(-1);							\
	}
#else

#define CHECK_POL_FD( who )

#endif /* ! CAUTIOUS */

int polhemus_byte_count(void)
{
	int n;

	CHECK_POL_FD("polhemus_byte_count");

#ifdef INSIDE_TRACK
	if( ioctl(polh_fd,POLHEMUS_GET_COUNT,&n) < 0 ){
		perror("ioctl");
		warn("error getting polhemus word count");
		return(-1);
	}
	n *= 2;		/* change word count to byte count */
#else
	/* need to use FIONREAD here??? */
	if( ioctl(polh_fd,FIONREAD,&n) < 0 ){
		perror("ioctl (FIONREAD)");
		warn("error getting polhemus word count");
		return(-1);
	}
#endif

	return(n);
}

int polhemus_word_count(void)
{
	int n;

	CHECK_POL_FD("polhemus_word_count");

#ifdef INSIDE_TRACK
	if( ioctl(polh_fd,POLHEMUS_GET_COUNT,&n) < 0 ){
		perror("ioctl");
		warn("error getting polhemus word count");
		return(-1);
	}
#else
	/* need to use FIONREAD here??? */
	if( ioctl(polh_fd,FIONREAD,&n) < 0 ){
		perror("ioctl (FIONREAD)");
		warn("error getting polhemus word count");
		return(-1);
	}
	n /= 2;		/* change byte count to word count */
#endif

	return(n);
}

int polhemus_output_data_format( Polh_Record_Format *prfp )
{
	int i;
	char cmdstr[64], codestr[16];
	Polh_Record_Format sprf;	/* the single version, no timestamps */
	int station;

	station = prfp->rf_station;

#ifdef CAUTIOUS
	if( station != 0 && station != 1 ){
		sprintf(error_string,
			"CAUTIOUS:  polhemus_output_data_format:  station (%d) should be 0 or 1",
			station);
		warn(error_string);
		return(-1);
	}
#endif /* CAUTIOUS */

	sprf.rf_station = station;

	CHECK_POL_FD("polhemus_output_data");

	/* start counting at 2, there is a leading word and a cr-lf */
	prfp->rf_n_words = sprf.rf_n_words = 2;
	sprf.rf_n_data=0;
	for(i=0;i<prfp->rf_n_data;i++){
		Polh_Output_Type type;

		type = prfp->rf_output[i];
		prfp->rf_n_words += od_tbl[type].od_words;
		if( type != SECONDS && type != MSECS && type != DATE ){
			sprf.rf_n_words += od_tbl[type].od_words;
			sprf.rf_output[ sprf.rf_n_data ] = type;
			sprf.rf_n_data++;
		}
	}

	/* we do this after we have computed the number of words per record */
#ifdef INSIDE_TRACK
	if( ioctl(polh_fd,POLHEMUS_SET_RECORD_FORMAT,prfp) < 0 ){
		perror("ioctl POLHEMUS_SET_RECORD_FORMAT");
		warn("error setting polhemus output data structure");
		return(-1);
	}
#else
	//warn("no ioctl for set record format???");
	/* We don't have a driver for the fastrack, so there is nothing to do!? */
#endif

	/* Now we send the command to change the operation.
	 * It would be cleaner to have the driver take care of this in
	 * the ioctl routine, but the driver doesn't do any commands itself yet...
	 */

	sprintf(cmdstr,"O%d",prfp->rf_station+1);
	for(i=0;i<prfp->rf_n_data;i++){
		Polh_Output_Type type;

		type = prfp->rf_output[i];
		if( od_tbl[type].od_code > 0 ){
			sprintf(codestr,",%d",od_tbl[ type ].od_code);
			strcat(cmdstr,codestr);
		}
	}
	strcat(cmdstr,",1\r");	/* cr lf at end of record */

	if( send_string(cmdstr) < 0 ) { 
		warn("polhemus_output_data_format:  Unable to set output data format");
		return(-1);
	}

	station_info[station].sd_multi_prf = *prfp;
	station_info[station].sd_single_prf = sprf;

	return(0);
}

static void decode_activation_state(int station,int code_char)
{
	switch( code_char ){
		case '1':
			if( ! STATION_IS_ACTIVE(station) ){
				if( verbose ) {
					sprintf(error_string,
				"setting active flag for station %d",station);
					advise(error_string);
				}
				station_info[station].sd_flags |= STATION_ACTIVE;
			}
			break;
		case '0':
			if( STATION_IS_ACTIVE(station) ){
				if( verbose ) {
					sprintf(error_string,
				"clearing active flag for station %d",station);
					advise(error_string);
				}
				station_info[station].sd_flags &= ~STATION_ACTIVE;
			}
			break;
		default:
			sprintf(error_string,"decode_activation_state:  bad code char");
			warn(error_string);
			break;
	}
}

void get_active_stations()
{
	char *s;

	if( get_polh_info(PH_STATION,"") < 0 ){
		warn("Unable to get current active station!");
		return;
	}

	s=(char *)resp_buf;

	if( *s != '2' ){
		sprintf(error_string,"get_active_stations:  expected first char to be 2 (\"%s\")",
			show_printable(s));
		warn(error_string);
	}

	if( s[1] != '1' && s[1] != '2' ){
		sprintf(error_string,"get_active_stations:  expected second char to be 1 or 2 (\"%s\")",
			show_printable(s));
		warn(error_string);
	}

	if( s[2] != 'l' ){
		sprintf(error_string,"get_active_stations:  expected third char to be l (\"%s\")",
			show_printable(s));
		warn(error_string);
	}

	decode_activation_state(0,s[3]);
	decode_activation_state(1,s[4]);

	n_active_stations = 0;
	if( STATION_IS_ACTIVE(0) ){
		n_active_stations++;
		active_mask |= STATION_1_ACTIVE;
	}
	if( STATION_IS_ACTIVE(1) ){
		n_active_stations++;
		active_mask |= STATION_2_ACTIVE;
	}
}

void show_output_data_format(int station)
{
	Polh_Record_Format *prfp;
	int i;

	prfp = &station_info[station].sd_multi_prf;

	sprintf(msg_str,"Station %d:",prfp->rf_station);
	prt_msg(msg_str);

	sprintf(msg_str,"\t%d output words.",prfp->rf_n_words);
	prt_msg(msg_str);

	sprintf(msg_str,"\t%d output fields:",prfp->rf_n_data);
	prt_msg(msg_str);

	for(i=0;i<prfp->rf_n_data;i++){
		Polh_Output_Type type;

		type = prfp->rf_output[i];
		sprintf(msg_str,"\t\t%s",od_tbl[type].od_name);
		prt_msg(msg_str);
	}
}


