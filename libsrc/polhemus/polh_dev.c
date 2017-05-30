#include "quip_config.h"

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
#include <sys/ioctl.h>      // FIONREAD
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

//#include "myerror.h"
//#include "getbuf.h"
//#include "savestr.h"
//#include "sigpush.h"
#include "quip_prot.h"

//#include "ioctl_polhemus.h"
#include "polh_dev.h"
#include "debug.h"

#define POLHEMUS_SYNC_VAR_NAME	"polhemus_sync_mode"

#ifdef INSIDE_TRACK
#define USE_DATA_DEV			/* We'd like to avoid having two devices,
					 * but for now that's how it is...
					 */
#endif

/* globals */

/* polhemus file descriptors */
static int active_mask=0;
int curr_station_idx=0;

int polh_fd = (-1);
#ifdef USE_DATA_DEV
static int polh_data_fd = (-1);
#endif /* USE_DATA_DEV */

int which_receiver = POLH_RECV;		/* polhemus receiver number, defaults to 1 */
int polh_continuous = 0;		/* continuous output flag */
int polh_units = PH_CM_FMT;		/* units format (default is cm.) */
int n_active_stations=0;
Station_Data station_info[2];

static long polh_software_hz = DEFAULT_SYNC_HZ;	/* software synchronization hz rate */
static char last_command[64];

#define RESP_BUF_SIZE	64
#define POLHEMUS_LINE_SIZE	(RESP_BUF_SIZE*sizeof(short))

short resp_buf[RESP_BUF_SIZE];
#define polh_line_buffer	((char *)resp_buf)

ssize_t n_response_chars;

/* Implement data buffering for polhemus device */
//#define POLHEMUS_BUFFER_SIZE	4096
#define POLHEMUS_BUFFER_SIZE	8192
static char polh_buffer[POLHEMUS_BUFFER_SIZE];
static int polh_next_wr=0;
static int polh_next_rd=0;
static int n_polh_free=POLHEMUS_BUFFER_SIZE;
static int n_polh_avail=0;

#ifdef QUIP_DEBUG
#define POLH_BUF_DEBUG							\
		if( debug & debug_polhemus ){				\
			sprintf(DEFAULT_ERROR_STRING,			\
				"fill_polh_buffer read %ld bytes",(long)m);	\
			NADVISE(DEFAULT_ERROR_STRING);			\
		}
#else	// ! QUIP_DEBUG
#define POLH_BUF_DEBUG
#endif	// ! QUIP_DEBUG

#define GET_CHUNK(n_want)						\
									\
	if( (m=read(polh_fd,&polh_buffer[polh_next_wr],n_want)) !=	\
							n_want ){	\
		if( m < 0 ){						\
			_tell_sys_error(DEFAULT_QSP_ARG  "read");	\
			NWARN("error reading polhemus data");		\
		} else {						\
			sprintf(DEFAULT_ERROR_STRING,			\
		"Expected %ld polhemus bytes, got %ld",(long)n_want,(long)m);		\
			NWARN(DEFAULT_ERROR_STRING);			\
		}							\
	} else {							\
		POLH_BUF_DEBUG						\
	}								\
	polh_next_wr += n_want;						\
	if( polh_next_wr >= POLHEMUS_BUFFER_SIZE )			\
		polh_next_wr = 0;					\
	n_polh_free -= n_want;						\
	n_polh_avail += n_want;


#ifdef NOT_USED

static int n_printable(short *buf,int n)
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

static void print_string(short *buf,int n)
{
	char cbuf[LLEN];

	if( n > LLEN-1 ){
		NWARN("print_string:  buffer size too small for data");
		return;
	}

	memcpy(cbuf,buf,n);
	cbuf[n]=0;
	_prt_msg(DEFAULT_QSP_ARG  cbuf);
}

#endif // NOT_USED

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
		s=show_printable(DEFAULT_QSP_ARG  str);
		sprintf(DEFAULT_MSG_STR,"\t0x%x\t\t0x%x\t%s", *buf, *cp, s);
		_prt_msg_frag(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);

		cp++;
		str[0]=(*cp);
		str[1]=0;
		s=show_printable(DEFAULT_QSP_ARG  str);
		sprintf(DEFAULT_MSG_STR,"\t0x%x\t%s", *cp, s);
		_prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);

		buf++;
 	}
}

#ifdef INSIDE_TRACK
static int open_polh_dev(void)
{
	if( (polh_fd = open(POLH_DEV, O_RDWR)) < 0 ) {
		sprintf(DEFAULT_ERROR_STRING,"open_polh_dev: opening polhemus device %s",POLH_DEV);
		_tell_sys_error(DEFAULT_QSP_ARG  DEFAULT_ERROR_STRING);
		return(-1);
	}

#ifdef USE_DATA_DEV
	if( (polh_data_fd = open(POLH_DATA_DEV, O_RDONLY) ) < 0) {
		sprintf(DEFAULT_ERROR_STRING,"open_polh_dev: opening polhemus data device %s",POLH_DATA_DEV);
		_tell_sys_error(DEFAULT_QSP_ARG  DEFAULT_ERROR_STRING);
		return(-1);
	}
#endif /* USE_DATA_DEV */

	return(0);
}

#else	// ! INSIDE_TRACK

void flush_input_data()
{
	int n;

	usleep(100000);

	if( ioctl(polh_fd,FIONREAD,&n) < 0 ){
		perror("ioctl (FIONREAD)");
		NWARN("error getting polhemus word count");
		n=1;	/* just in case... */
	}
	if( n > 0 ){
		sprintf(DEFAULT_ERROR_STRING,"Flushing %d pending input characters...",n);
		NADVISE(DEFAULT_ERROR_STRING);

		if( tcflush(polh_fd,TCIFLUSH) < 0 ){
			_tell_sys_error(DEFAULT_QSP_ARG  "tcflush");
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
	int n, n1;
	ssize_t m;

	if( ioctl(polh_fd,FIONREAD,&n) < 0 ){
		perror("ioctl (FIONREAD)");
		NWARN("error getting number of readable polhemus chars");
		return;
	}
	if( n == 0 ) return;
//sprintf(DEFAULT_ERROR_STRING,"fill_polh_buffer:  %d chars available",n);
//NADVISE(DEFAULT_ERROR_STRING);
	if( n <= n_polh_free ){		/* we have room for the data */
		if( n <= (POLHEMUS_BUFFER_SIZE-polh_next_wr) ){		/* all fits without wraparound */
//sprintf(DEFAULT_ERROR_STRING,"fill_polh_buffer:  trying to get chunk of %d chars",n);
//NADVISE(DEFAULT_ERROR_STRING);

			GET_CHUNK(n)
		} else {						/* need to wrap-around */
			n1=POLHEMUS_BUFFER_SIZE-polh_next_wr;
//sprintf(DEFAULT_ERROR_STRING,"fill_polh_buffer:  trying to get chunk of %d chars",n1);
//NADVISE(DEFAULT_ERROR_STRING);
			GET_CHUNK(n1)
			n -= n1;
//sprintf(DEFAULT_ERROR_STRING,"fill_polh_buffer:  trying to get chunk of %d chars",n);
//NADVISE(DEFAULT_ERROR_STRING);
			GET_CHUNK(n)
		}
	} else {
		sprintf(DEFAULT_ERROR_STRING,"%d chars readable polhemus characters, but only %d free buffer locs",
			n,n_polh_free);
		NWARN(DEFAULT_ERROR_STRING);
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
		sprintf(DEFAULT_ERROR_STRING,"open_polh_dev: opening polhemus device %s",POLH_DEV);
		_tell_sys_error(DEFAULT_QSP_ARG  DEFAULT_ERROR_STRING);
		return(-1);
	}

	/* Now set baud rate etc. */

	/* The old Walt Johnson unit in the vision lab is set to 115200 baud...
	 * Is this selected by the DIP switches?
	 */

	/* after fastrak power-up, the device doesn't respond to the reset (W)
	 * command until a carriage return has been sent, which triggers
	 * an error message!?
	 */

fprintf(stderr,"open_polh_dev:  flushing input data...\n");
	flush_input_data();

	/* Should we flush any old data, reset device to known state, etc?
	 * What if the device is currently in continuous mode?
  	 */
fprintf(stderr,"open_polh_dev:  sending REINIT_SYS command...\n");
	if(send_polh_cmd(PH_REINIT_SYS, NULL) < 0 ) {
		NWARN("open_polh_dev:  unable to reinitialize system");
		return(-1);
	}
	/* The reinit command will halt continuous output, but we should now wait a few
	 * clicks and then flush any pending output from the device.
	 */
fprintf(stderr,"open_polh_dev:  flushing input data again...\n");
	flush_input_data();

fprintf(stderr,"open_polh_dev:  done.\n");
	return(0);
} /* end open_polh_dev */

	
#endif	// ! INSIDE_TRACK


void read_response(int display_flag)
{
	int n;

	n = polhemus_word_count();

	if( n <= 0 ) return;

	if( n > RESP_BUF_SIZE ){
		sprintf(DEFAULT_ERROR_STRING,"%d response words available, buffer size is only %d",
			n,RESP_BUF_SIZE);
		NWARN(DEFAULT_ERROR_STRING);
		n = RESP_BUF_SIZE;
	}

	n *= 2;		/* byte count */

	if( (n_response_chars=read(polh_fd,resp_buf,n)) != n ){
		if( n_response_chars < 0 ) {
			_tell_sys_error(DEFAULT_QSP_ARG  "read_response");
			NWARN("error reading polhemus data");
		} else {
			sprintf(DEFAULT_ERROR_STRING,"read_response:  %d bytes requested, %zd actually read",n,n_response_chars);
			NWARN(DEFAULT_ERROR_STRING);
		}
	}
	if( display_flag )
		display_buffer(resp_buf,(int)n_response_chars/2);
}

void clear_polh_dev(void)
{
	short clear_buf[RESP_BUF_SIZE];
	int n;
	int n_want;
	ssize_t n_read;
	int total=0;

	if( polhemus_word_count()<0 ){
		NADVISE("no bytes to clear from polhemus device");
		return;
	}

	while ( (n = polhemus_word_count()) != 0 ) {
		if( n < RESP_BUF_SIZE ) n_want = n;
		else n_want = RESP_BUF_SIZE;
		if( n_want % 2 ) n_want++;

		if( (n_read=read(polh_fd, clear_buf, n_want*2)) != (n_want*2) ) {
			if( n_read < 0 ) {
				_tell_sys_error(DEFAULT_QSP_ARG  "clear_polh_dev");
				NWARN("error clearing polhemus device");
				return;
			} else {
				sprintf(DEFAULT_ERROR_STRING,"clear_polh_dev: %d bytes requested, %zd actually read",n_want*2, n_read);
				NWARN(DEFAULT_ERROR_STRING);
			}	
		}	
		total += n_read;
	}
	return;
}

static void init_output_data_format(int station_idx)
{
	Polh_Record_Format *prfp;

	prfp = &station_info[station_idx].sd_multi_prf;

	prfp->rf_n_data = 4;
	prfp->rf_output[0] = SECONDS;
	prfp->rf_output[1] = MSECS;
	prfp->rf_output[2] = XYZ_INT;
	prfp->rf_output[3] = EULER_INT;
	prfp->rf_station = (short)station_idx;
	prfp->rf_n_words = 12;
	polhemus_output_data_format( prfp );

	/*
	show_output_data_format(station_idx);
	*/
}

static int set_default_polh(SINGLE_QSP_ARG_DECL)
{
#ifdef INSIDE_TRACK
	/* initialize to floating output */
	/* station number, xyz_int, euler_int, crlf */
	if(send_string("O1,2,4,1\r") < 0 ) { 
		NWARN("set_default_polh: Unable to initialize to floating output");
		return(-1);
	}

	usleep(100000);	/* sleep 1/10 sec */
#endif // INSIDE_TRACK

fprintf(stderr,"set_default_polh:  clearing device...\n");
	clear_polh_dev();

	/* set to centimeter output */
fprintf(stderr,"set_default_polh:  sending centimeter command...\n");
	if(send_polh_cmd(PH_CM_FMT, NULL) < 0 ) {
		NWARN("set_default_polh: Unable to set to centimeter output");
		return(-1);
	}

	/* set active station to 1 */
#ifdef USE_DATA_DEV
	if(send_string("l1,1\r") < 0 ){
		NWARN("set_default_polh: Unable to set active station number to 1");
		return(-1);
	}
#endif /* USE_DATA_DEV */

	/* Set the default record formats */
	init_output_data_format(0);
	init_output_data_format(1);

fprintf(stderr,"set_default_polh:  getting active stations...\n");
	get_active_stations(SINGLE_QSP_ARG);
		/* make sure the flags reflect reality */

	if( ! STATION_IS_ACTIVE(0) )
		activate_station(0,1);	/* make station 1 active */
	if( STATION_IS_ACTIVE(1) )
		activate_station(1,0);	/* make station 2 inactive */
	curr_station_idx=0;
	
fprintf(stderr,"set_default_polh:  sleeping 100 ms...\n");
	usleep(100000);
fprintf(stderr,"set_default_polh:  clearing...\n");
	clear_polh_dev();

fprintf(stderr,"set_default_polh:  setting non-active station number to 2 (?)...\n");
	if( send_string("l2,0\r") < 0 ) { 
		NWARN("set_default_polh: Unable to set non-active station number to 2");
		return(-1);
	}
	
fprintf(stderr,"set_default_polh:  sleeping 100 ms...\n");
	usleep(100000);
fprintf(stderr,"set_default_polh:  clearing...\n");
	clear_polh_dev();
fprintf(stderr,"set_default_polh:  done.\n");

	return(0);
}

int init_polh_dev(SINGLE_QSP_ARG_DECL)
{
#ifdef CAUTIOUS
	if( polh_fd >= 0 
#ifdef USE_DATA_DEV
			|| polh_data_fd >= 0
#endif /* USE_DATA_DEV */
			) {
		NWARN("CAUTIOUS: init_polh_dev: polhemus device already intialized!?");
		return(0);
	}
#endif /* CAUTIOUS */

	if(open_polh_dev() < 0 || set_default_polh(SINGLE_QSP_ARG) < 0) return(-1);

fprintf(stderr,"init_polh_dev:  done.\n");
	return(0);
} // init_polh_dev

/* What is the point of this?  Are we trying to reset the driver?
 */

int reopen_polh_dev(SINGLE_QSP_ARG_DECL)
{
#ifdef CAUTIOUS
	if( (!(polh_fd >= 0))
#ifdef USE_DATA_DEV
			&& !(polh_data_fd >= 0)
#endif /* USE_DATA_DEV */
			) {
		NWARN("CAUTIOUS: reopen_polh_dev: polhemus device not opened to close!?"); 
		return(-1);
	}
#endif /* CAUTIOUS */

	if( close(polh_fd) < 0 ) {
		_tell_sys_error(DEFAULT_QSP_ARG  "unable to close polhemus device");
		return(-1);
	}

#ifdef USE_DATA_DEV
	if( close(polh_data_fd) < 0 ) {
		_tell_sys_error(DEFAULT_QSP_ARG  "unable to close polhemus data device");
		return(-1);
	}
	polh_data_fd=-1;
#endif /* USE_DATA_DEV */
	polh_fd=-1;
	
	return( init_polh_dev(SINGLE_QSP_ARG) );
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
			_tell_sys_error(DEFAULT_QSP_ARG  "read_until");
			return(-1);
		}
		sprintf(DEFAULT_MSG_STR,"read_until (fd=%d):  %d bytes requested, %d actually read",
			fd,n_want,n_read);
		NADVISE(DEFAULT_MSG_STR);
		n_want -= n_read;
		if( ++num_reads >= MAX_POLHEMUS_READS ) {
			NWARN("read_polh_dev: timed out from reading data");
			return(-1);
		}
	}
{
int i;
for(i=0;i<n_orig;i++){
sprintf(DEFAULT_ERROR_STRING,"%d\t0x%x\t%c",i,databuf[i],databuf[i]);
NADVISE(DEFAULT_ERROR_STRING);
}
}

	return(0);
}

int read_polh_dev( short* databuf, int n_want )
{
	int n;

	if( (n=read_until((char *)databuf,n_want,polh_fd)) < 0 )
		NWARN("error reading polhemus");
	return(n);
}

#ifdef USE_DATA_DEV
int read_polh_data( void *raw_pdp, int n_want )
{
	int n;

	if( (n=read_until((char *)raw_pdp,n_want,polh_data_fd)) < 0 )
		NWARN("error reading polhemus data");
	return(n);
}
#else /* ! USE_DATA_DEV */

int read_polh_data( void *raw_pdp, int n_want )
{
	int n;

	if( (n=read_until((char *)raw_pdp,n_want,polh_fd)) < 0 )
		NWARN("error reading polhemus data");
	return(n);
}

#endif /* ! USE_DATA_DEV */

#define CHECK_STR( str )								\
											\
	if( strncmp( resp_chars+posn, str, strlen(str) ) ){				\
		/* display buffer */							\
		sprintf(DEFAULT_ERROR_STRING,"Expected to find string \"%s\" at position %d",	\
				str,posn);						\
		NWARN(DEFAULT_ERROR_STRING);							\
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
		sprintf(DEFAULT_ERROR_STRING,"parse_error:  expected '-' at position %d",posn);
		NWARN(DEFAULT_ERROR_STRING);
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
		case -1:	NWARN("required field missing"); break;
		case -2:	NWARN("required numeric field is non-numeric"); break;
		case -3:	NWARN("value is outside required range"); break;
		case -4:	NWARN("specified frequency not hardware configured"); break;
		case -99:	NWARN("undefined input - cannot identify command"); break;
		default:
				sprintf(DEFAULT_ERROR_STRING,"Unrecognized error code %d",error_code);
				NWARN(DEFAULT_ERROR_STRING);
				display_buffer(resp_buf,n_response_chars/2);
				return;
	}


NADVISE("parse_error: calling display_buffer");
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
sprintf(DEFAULT_ERROR_STRING,"read_polh_line has character 0x%x",c);
NADVISE(DEFAULT_ERROR_STRING);
*/
			/* the polhemus terminates lines with \r\n combos, but we
			 * end the line when we see the \r...  so when we go to read
			 * the next line, the first char we see is an \n - skip it!
			 */
			if( i!=0 || c!='\n')
				polh_line_buffer[i++] = (char)c;
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
	char expected[8];

fprintf(stderr,"recv_polh_data:  reading line...\n");
	s=read_polh_line();
fprintf(stderr,"recv_polh_data:  line read:  \"%s\".\n",s);

	switch(polh_cmds[cmd].pc_code){
		case PH_SYNC_MODE:
			/* "2 y0\r" */
			if( strncmp(s,"2 y",3) ){
				sprintf(DEFAULT_ERROR_STRING,"recv_polh_data PH_SYNC_MODE:  read \"%s\", expected \"2 y...\"",
					show_printable(DEFAULT_QSP_ARG  s));
				NWARN(DEFAULT_ERROR_STRING);
			} else {
				switch(s[3]){
					case '0':
						_prt_msg(DEFAULT_QSP_ARG  "Internal sync");
						ASSIGN_VAR(POLHEMUS_SYNC_VAR_NAME,"internal");
						break;
					case '1':
						_prt_msg(DEFAULT_QSP_ARG  "External sync");
						ASSIGN_VAR(POLHEMUS_SYNC_VAR_NAME,"external");
						break;
					case '2':
						_prt_msg(DEFAULT_QSP_ARG  "External sync");
						ASSIGN_VAR(POLHEMUS_SYNC_VAR_NAME,"software");
						break;
#ifdef CAUTIOUS
					default:
						sprintf(DEFAULT_ERROR_STRING,
					"CAUTIOUS:  recv_polh_data:  read \"%s\", unhandled sync mode",
							show_printable(DEFAULT_QSP_ARG  s));
						NWARN(DEFAULT_ERROR_STRING);
						break;
#endif /* CAUTIOUS */
				}
			}
			break;

		case PH_STATUS:
sprintf(DEFAULT_ERROR_STRING,"PH_STATUS recv_polh_data received \"%s\", NOT parsing...",show_printable(DEFAULT_QSP_ARG  s));
NADVISE(DEFAULT_ERROR_STRING);
			break;
		case PH_STATION:
			/* 21l1000\r */
			sprintf(expected,"2%cl",'1'+curr_station_idx);
			if( strncmp(s,expected,3) ){
				sprintf(DEFAULT_ERROR_STRING,"recv_polh_data PH_STATION:  read \"%s\", expected \"%s...\"",
					show_printable(DEFAULT_QSP_ARG  s),
					expected);
				NWARN(DEFAULT_ERROR_STRING);
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
							sprintf(DEFAULT_ERROR_STRING,
			"CAUTIOUS:  recv_polh_data PH_STATION:  read \"%s\", unexpected status char in position %d",
								show_printable(DEFAULT_QSP_ARG  s),i);
							NWARN(DEFAULT_ERROR_STRING);
							break;
					}
					if( str != NULL ){
						sprintf(DEFAULT_MSG_STR,"Station %d is %s",i-2,str);
						_prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);
					}
					i++;
				}
			}
			break;
		case PH_ALIGNMENT:
sprintf(DEFAULT_ERROR_STRING,"PH_ALIGNMENT recv_polh_data received \"%s\", NOT parsing...",show_printable(DEFAULT_QSP_ARG  s));
NADVISE(DEFAULT_ERROR_STRING);
			break;
		case PH_XMTR_ANGLES:
		case PH_RECV_ANGLES:
		case PH_REF_BORESIGHT:
		case PH_ATT_FILTER:
		case PH_POS_FILTER:
		case PH_ANGULAR_ENV:
		case PH_POSITIONAL_ENV:
		case PH_HEMISPHERE:
#ifdef QUIP_DEBUG
if( debug & debug_polhemus ){
sprintf(DEFAULT_ERROR_STRING,"recv_polh_data received \"%s\", NOT parsing...",show_printable(DEFAULT_QSP_ARG  s));
NADVISE(DEFAULT_ERROR_STRING);
}
#endif // QUIP_DEBUG
sprintf(DEFAULT_ERROR_STRING,"recv_polh_data received \"%s\", NOT parsing...",show_printable(DEFAULT_QSP_ARG  s));
NADVISE(DEFAULT_ERROR_STRING);
			/* BUG - need to parse the string */
			break;
		default:
			sprintf(DEFAULT_ERROR_STRING,"Unhandled case in recv_polh_data for %s command",
				polh_cmds[cmd].pc_name);
			NWARN(DEFAULT_ERROR_STRING);
sprintf(DEFAULT_ERROR_STRING,"recv_polh_data received \"%s\"",show_printable(DEFAULT_QSP_ARG  s));
NADVISE(DEFAULT_ERROR_STRING);
			return(-1);
			break;
	}
	return(0);
}

#define MAX_POLHEMUS_COUNT_WAITS	10 	/* one second */	

/* Send a command to the polhemus */

int send_polh_cmd(Ph_Cmd_Code cmd, const char * cmdargs) 
{
	char code_id[LLEN];
	char command[LLEN];

#ifdef CAUTIOUS
	// cmd is an unsigned type!
	if( /* cmd < 0 || */ cmd >= N_PH_CMD_CODES ) {
		NWARN("send_polh_cmd: Unhandled command code!?");
		return(-1);
	}
#endif
	
	/* check if we can peform the requested operation with the command code */
	/* why does reinit command trigger this message??? */
	if( !CAN_SET(cmd) ) {
		sprintf(DEFAULT_ERROR_STRING,
			"send_polh_cmd: command %s cannot be used to SET", polh_cmds[cmd].pc_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);	
	}
		
	/* Print the command code itself, and possibly a station specifier.
	 * Check if the command needs transmitter, receiver, station number
	 * or if it doesn't need anything.
	 */
	switch( polh_cmds[cmd].pc_trs ) {
		case PH_NEED_XMTR : sprintf(code_id, "%s%d", polh_cmds[cmd].pc_cmdstr, POLH_XMTR_NUM); break;
		case PH_NEED_RECV : sprintf(code_id, "%s%d", polh_cmds[cmd].pc_cmdstr, which_receiver); break;
		case PH_NEED_STAT : sprintf(code_id, "%s%d", polh_cmds[cmd].pc_cmdstr, curr_station_idx+1); break;
		case PH_NEED_NONE : sprintf(code_id, "%s", polh_cmds[cmd].pc_cmdstr); break;
#ifdef CAUTIOUS
		default : NWARN("send_polh_cmd: Unknown transmitter/receiver/station flag value!?"); return(-1); break;
#endif
	}

	/* Add additional arguments, if any.
	 * The Polhemus manual says that the 
	 * compensation and station commands
	 * are not followed by commas although 
	 * they have arguments.
	 */ 
	if(cmdargs) {
//sprintf(DEFAULT_ERROR_STRING,"Appending command args \"%s\" to %s command",
//show_printable(DEFAULT_QSP_ARG  cmdargs),polh_cmds[cmd].pc_name);
//NADVISE(DEFAULT_ERROR_STRING);
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
				sprintf(DEFAULT_ERROR_STRING,"Unhandled Case in send_polh_cmd:  %s, cmd_args = \"%s\"",
					polh_cmds[cmd].pc_name,show_printable(DEFAULT_QSP_ARG  cmdargs));
				NWARN(DEFAULT_ERROR_STRING);
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
			sprintf(DEFAULT_ERROR_STRING,"Unhandled case in send_polh_cmd:  %s",
				polh_cmds[cmd].pc_name);
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}


#ifdef QUIP_DEBUG
if( debug & debug_polhemus ){
sprintf(DEFAULT_ERROR_STRING,"Ready to send command string \"%s\"",show_printable(DEFAULT_QSP_ARG  command));
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

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
		sprintf(DEFAULT_ERROR_STRING,"send_polh_cmd: Unable to send command string %s",command);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	/* FIXME - here we should put some error-checking to ensure that the command is set */

	return(0);
}

int get_polh_info(QSP_ARG_DECL  Ph_Cmd_Code cmd, const char * cmdargs) 
{
	char code_id[LLEN];
	char command[LLEN];

#ifdef CAUTIOUS
	// cmd is an unsigned type!
	if( /* cmd < 0 || */ cmd >= N_PH_CMD_CODES ) {
		NWARN("get_polh_info: Unhandled command code!?");
		return(-1);
	}
#endif
	
	/* check if we can peform the requested operation with the command code */
	if( !CAN_GET(cmd) ){
		sprintf(DEFAULT_ERROR_STRING, "get_polh_info: command %s is not a GET command",
			polh_cmds[cmd].pc_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);	
	}
		
	/* Print the command code itself, and possibly a station specifier.
	 * Check if the command needs transmitter, receiver, station number
	 * or if it doesn't need anything.
	 */
	switch( polh_cmds[cmd].pc_trs ) {
		case PH_NEED_XMTR : sprintf(code_id, "%s%d", polh_cmds[cmd].pc_cmdstr, POLH_XMTR_NUM); break;
		case PH_NEED_RECV : sprintf(code_id, "%s%d", polh_cmds[cmd].pc_cmdstr, which_receiver); break;
		case PH_NEED_STAT : sprintf(code_id, "%s%d", polh_cmds[cmd].pc_cmdstr, curr_station_idx+1); break;
		case PH_NEED_NONE : sprintf(code_id, "%s", polh_cmds[cmd].pc_cmdstr); break;
#ifdef CAUTIOUS
		default : NWARN("send_polh_cmd: Unknown transmitter/receiver/station flag value!?"); return(-1); break;
#endif
	}

fprintf(stderr,"get_polh_info:  code_id = \"%s\"\n",code_id);

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
NADVISE("Oops");
				sprintf(command, "%s%s\r", code_id, cmdargs);
				break;
			case PH_POS_FILTER:
			case PH_ATT_FILTER:
		case PH_ANGULAR_ENV:
		case PH_POSITIONAL_ENV:
			case PH_HEMISPHERE:
NADVISE("oops");
				sprintf(command, "%s%s", code_id, cmdargs);
				break;
			default:
				sprintf(DEFAULT_ERROR_STRING,"get_polh_info:  missing case for command %s",
					polh_cmds[cmd].pc_name);
				NWARN(DEFAULT_ERROR_STRING);
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
sprintf(DEFAULT_ERROR_STRING,"get_polh_info:  unhandled case for %s command",polh_cmds[cmd].pc_name);
NWARN(DEFAULT_ERROR_STRING);
			break;
	}


#ifdef QUIP_DEBUG
//if( debug & debug_polhemus ){
sprintf(DEFAULT_ERROR_STRING,"ready to send command string \"%s\"",show_printable(DEFAULT_QSP_ARG  command));
NADVISE(DEFAULT_ERROR_STRING);
//}
#endif /* QUIP_DEBUG */

	strcpy(last_command,command);

	/* not all fastrak commands need the carriage return??? */

	/* write the command to the device */
	if( send_string(command) < 0 ) {
		sprintf(DEFAULT_ERROR_STRING,"send_polh_cmd: Unable to send command string %s",command);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

fprintf(stderr,"get_polh_info:  command sent, calling recv_polh_data\n");
	return recv_polh_data(QSP_ARG  cmd);	
} /* end get_polh_info */

int send_string(const char *cmd)
{
	ssize_t numwritten;
	int len;
	
	if(cmd == NULL) {
		NWARN("send_string: null command string!?");
		return(-1);
	}

#ifdef QUIP_DEBUG
if( debug & debug_polhemus ){
	sprintf(DEFAULT_MSG_STR, "send_string: %s", show_printable(DEFAULT_QSP_ARG  cmd) );
	NADVISE(DEFAULT_MSG_STR);
}
#endif
	
	len = (int)strlen(cmd);
	numwritten = write(polh_fd, (const void*)cmd, len);

	if( numwritten != len ) {
		if(numwritten < 0 ) 
			_tell_sys_error(DEFAULT_QSP_ARG  "send_string");
		else {
			sprintf(DEFAULT_ERROR_STRING, "Requested %d bytes to be written to polhemus device, %zd actually written",
				len, numwritten);
			NWARN(DEFAULT_ERROR_STRING);
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
			NWARN("Software sync not implemented!");
			return(-1);
#ifdef CAUTIOUS
		default :
			NWARN("set_polh_sync_mode:  Unexpected sync mode requsted!?");
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

void activate_station(int station_idx,int flag)
{
	int bit;
#ifndef INSIDE_TRACK
	char cmd_str[32];
#endif

fprintf(stderr,"activate_station:  station %d, flag = %d\n",
station_idx,flag);

#ifdef CAUTIOUS
	if( station_idx < 0 || station_idx >= MAX_POLHEMUS_STATIONS ) {
		sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  activate_station: bad station number %d, must be in the range from 0 to %d",
			station_idx,MAX_POLHEMUS_STATIONS-1);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	// what is flag???
	if( flag < 0 || flag > 1 ){
		sprintf(DEFAULT_ERROR_STRING,
			"CAUTIOUS:  activate_station:  bad flag %d, should be 0 or 1",flag);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */

	if( STATION_IS_ACTIVE(station_idx) == flag ){
		sprintf(DEFAULT_ERROR_STRING,"activate_station:  station %d is already %s",
				station_idx+1,flag?"activated":"deactivated");
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

#ifdef FOOBAR
	if( station_idx == 0 )
		bit = STATION_1_ACTIVE;
	else
		bit = STATION_2_ACTIVE;
#else
	bit = 1 << station_idx;
#endif

	if( flag )
		active_mask |= bit;
	else
		active_mask &= ~bit;

#ifdef INSIDE_TRACK
	if( ioctl( polh_fd, POLHEMUS_SET_ACTIVE, &active_mask ) < 0 ){
		perror("ioctl POLHEMUS_SET_ACTIVE");
		NWARN("error setting active stations");
		return;
	}
#else
	/* just send the command - this ought to work for insidetrak too? */
	sprintf(cmd_str,"l%d,%d\r",station_idx+1,flag);
	if( send_string(cmd_str) < 0 ) {
		sprintf(DEFAULT_ERROR_STRING,"activate_station: Unable to send command string \"%s\"",
			cmd_str);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
#endif

	/* remember state */
	if( flag ){
		station_info[station_idx].sd_flags |= STATION_ACTIVE;
		which_receiver=station_idx;
		curr_station_idx=station_idx;
	} else
		station_info[station_idx].sd_flags &= ~STATION_ACTIVE;

	n_active_stations = (STATION_IS_ACTIVE(0)?1:0) + (STATION_IS_ACTIVE(1)?1:0);

	if( flag )
		sprintf(DEFAULT_MSG_STR,"Activating station %d",station_idx+1);
	else
		sprintf(DEFAULT_MSG_STR,"Deactivating station %d",station_idx+1);

	_prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);

	/* Assume that if we change the station, we are working
	 * with the corresponding receiver for that station.
	 */

	return;
}

void set_polh_units(Ph_Cmd_Code cmd)
{

#ifdef CAUTIOUS
	if( cmd != PH_INCHES_FMT && cmd != PH_CM_FMT ) {
		NWARN("set_polh_units: unhandled unit format requested!?");
		return;
	}
#endif /* CAUTIOUS */
	
	if( send_polh_cmd(cmd,NULL) < 0 ) {
		NWARN("set_polh_units: unable to set requested units!?");
		return;
	}

	polh_units = cmd;
	return;
}


#ifdef CAUTIOUS
#define CHECK_POL_FD( who )							\
										\
	if( polh_fd < 0 ){							\
		sprintf(DEFAULT_ERROR_STRING,"%s:  polhemus device not open",who);	\
		NWARN(DEFAULT_ERROR_STRING);						\
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
		NWARN("error getting polhemus word count");
		return(-1);
	}
	n *= 2;		/* change word count to byte count */
#else
	/* need to use FIONREAD here??? */
	if( ioctl(polh_fd,FIONREAD,&n) < 0 ){
		perror("ioctl (FIONREAD)");
		NWARN("error getting polhemus word count");
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
		NWARN("error getting polhemus word count");
		return(-1);
	}
#else
	/* need to use FIONREAD here??? */
	if( ioctl(polh_fd,FIONREAD,&n) < 0 ){
		perror("ioctl (FIONREAD)");
		NWARN("error getting polhemus word count");
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
	int station_idx;

	station_idx = prfp->rf_station;

#ifdef CAUTIOUS
	if( station_idx != 0 && station_idx != 1 ){
		sprintf(DEFAULT_ERROR_STRING,
			"CAUTIOUS:  polhemus_output_data_format:  station (%d) should be 0 or 1",
			station_idx);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
#endif /* CAUTIOUS */

	sprf.rf_station = (short)station_idx;

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
		NWARN("error setting polhemus output data structure");
		return(-1);
	}
#else
	//NWARN("no ioctl for set record format???");
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
		NWARN("polhemus_output_data_format:  Unable to set output data format");
		return(-1);
	}

	station_info[station_idx].sd_multi_prf = *prfp;
	station_info[station_idx].sd_single_prf = sprf;

	return(0);
}

static void decode_activation_state(int station_idx,int code_char)
{
	switch( code_char ){
		case '1':
			if( ! STATION_IS_ACTIVE(station_idx) ){
				if( verbose ) {
					sprintf(DEFAULT_ERROR_STRING,
				"setting active flag for station %d",station_idx);
					NADVISE(DEFAULT_ERROR_STRING);
				}
				station_info[station_idx].sd_flags |= STATION_ACTIVE;
			}
			break;
		case '0':
			if( STATION_IS_ACTIVE(station_idx) ){
				if( verbose ) {
					sprintf(DEFAULT_ERROR_STRING,
				"clearing active flag for station %d",station_idx);
					NADVISE(DEFAULT_ERROR_STRING);
				}
				station_info[station_idx].sd_flags &= ~STATION_ACTIVE;
			}
			break;
		default:
			sprintf(DEFAULT_ERROR_STRING,"decode_activation_state:  bad code char");
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}
}

void get_active_stations(SINGLE_QSP_ARG_DECL)
{
	char *s;

fprintf(stderr,"get_active_stations:  getting active station...\n");
	if( get_polh_info(QSP_ARG  PH_STATION,"") < 0 ){
		NWARN("Unable to get current active station!");
		return;
	}
fprintf(stderr,"get_active_stations:  response buffer:\n");
display_buffer(&resp_buf[0],(int)n_response_chars/2);

	s=(char *)resp_buf;

	if( *s != '2' ){
		sprintf(DEFAULT_ERROR_STRING,"get_active_stations:  expected first char to be 2 (\"%s\")",
			show_printable(DEFAULT_QSP_ARG  s));
		NWARN(DEFAULT_ERROR_STRING);
	}

	if( s[1] != '1' && s[1] != '2' ){
		sprintf(DEFAULT_ERROR_STRING,"get_active_stations:  expected second char to be 1 or 2 (\"%s\")",
			show_printable(DEFAULT_QSP_ARG  s));
		NWARN(DEFAULT_ERROR_STRING);
	}

	if( s[2] != 'l' ){
		sprintf(DEFAULT_ERROR_STRING,"get_active_stations:  expected third char to be l (\"%s\")",
			show_printable(DEFAULT_QSP_ARG  s));
		NWARN(DEFAULT_ERROR_STRING);
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

void show_output_data_format(int station_idx)
{
	Polh_Record_Format *prfp;
	int i;

	prfp = &station_info[station_idx].sd_multi_prf;

	sprintf(DEFAULT_MSG_STR,"Station %d:",prfp->rf_station);
	_prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);

	sprintf(DEFAULT_MSG_STR,"\t%d output words.",prfp->rf_n_words);
	_prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);

	sprintf(DEFAULT_MSG_STR,"\t%d output fields:",prfp->rf_n_data);
	_prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);

	for(i=0;i<prfp->rf_n_data;i++){
		Polh_Output_Type type;

		type = prfp->rf_output[i];
		sprintf(DEFAULT_MSG_STR,"\t\t%s",od_tbl[type].od_name);
		_prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);
	}
}


