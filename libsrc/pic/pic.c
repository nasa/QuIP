#include "quip_config.h"

char VersionId_pic_pic[] = QUIP_VERSION_STRING;

/* This is not a general purpose library for PIC microcontrollers,
 * it is tailored to work with JBM's microcode on the LED controller
 * boards designed by Denver Hinds.
 */

#ifdef HAVE_PIC


//#define MARK_TIME		// enable debug output

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif /* HAVE_UNISTD_H */

#ifdef HAVE_STRING_H
#include <string.h>
#endif /* HAVE_STRING_H */

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif /* HAVE_CTYPE_H */

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif /* HAVE_SYS_TIME_H */

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif /* HAVE_PTHREAD_H */

#include "debug.h"
#include "data_obj.h"
#include "pic.h"
#include "serial.h"
#include "version.h"
#include "rt_sched.h"
#include "query.h"
#include "serbuf.h"		// expect()
#include "ttyctl.h"		// keyhit()

/* static */ u_long pic_debug=0;
static int channels_per_pixel;

#define TIMER_STOPPED	1
#define TIMER_RUNNING	2

static int pic_nlcr_echo=1;	/* newer firmware echos cr-nl for cr, and nl-cr for nl... */

static int sync_to_video=1;	/* if 0 use non-waiting load func */
static const char *pic_prompt=NULL;
#define DEFAULT_PIC_PROMPT	"LED> "

#define INSURE_PROMPT								\
										\
		if( pic_prompt == NULL ){					\
			pic_prompt=DEFAULT_PIC_PROMPT;				\
			if( verbose ){						\
				sprintf(ERROR_STRING,				\
		"No prompt defined, defaulting to \"%s\"",pic_prompt);		\
				advise(ERROR_STRING);				\
			}							\
		}


/* We need to do a bit more than vanilla tty control, could do it in a script
 * but better to do it here.  This code is linux-specific, see ../support/termio.c
 * for portable defn's of these macros...
 */

#define GETTERM			tcgetattr(fd,&tiobuf)
#define SETTERM			tcsetattr(fd,TCSADRAIN,&tiobuf)
static struct termios tiobuf;

/* We don't want to reset the buffer if we are going to be scanning the response...
 */

#define WAIT_FOR_PROMPT(pdp)					\
								\
	if( (pdp)->pd_prompt_seen ){					\
		sprintf(ERROR_STRING,"WAIT_FOR_PROMPT %s:  prompt already seen!?",(pdp)->pd_name);	\
		WARN(ERROR_STRING);				\
	} else {						\
		INSURE_PROMPT					\
		expected_response(QSP_ARG  (pdp)->pd_sbp,pic_prompt);	\
		reset_buffer((pdp)->pd_sbp);			\
		(pdp)->pd_prompt_seen=1;					\
	}



#ifdef MARK_TIME
static int zero_set=0;
static struct timeval tv_now, tv_zero;
#define MARK(s)									\
		{ int delta_ms;							\
		gettimeofday(&tv_now,NULL);					\
		if( !zero_set ){						\
			tv_zero=tv_now;						\
			zero_set = 1;						\
		}								\
		delta_ms = 1000*(tv_now.tv_sec - tv_zero.tv_sec) + (tv_now.tv_usec - tv_zero.tv_usec)/1000; \
		sprintf(ERROR_STRING,"%s:  %2d.%03d",s,delta_ms/1000,delta_ms%1000);	\
		advise(ERROR_STRING); }
#else
#define MARK(s)
#endif
			

#define LED_PROMPT	"> "

enum {
	CHECK_SYNTAX,
	WRITE_DATA,
	VERIFY_DATA
};

static PIC_State pic_state={
	0,			/* flags */
	{ 0, 0 }		/* led buffers */
};
				// BUG need one per device!

/* pic state flags */
#define PIC_ECHO_ON	1
#define PIC_ECHOES	(pic_state.ps_flags & PIC_ECHO_ON)

typedef struct pic_device {
	char *		pd_name;
	int		pd_fd;
	Serial_Buffer *	pd_sbp;
	int		pd_prompt_seen;
} PIC_Device;

#define MAX_PIC_DEVICES	6
static int n_pics_active=0;

/* These are the data to be passed to each serial port writer thread */

typedef struct per_proc_info {
	const char *	ppi_name;
	int		ppi_index;		/* thread index */
	pid_t		ppi_pid;
	int		ppi_flags;

	PIC_Device *	ppi_pdp;
	void *		ppi_ptr;		/* data ptr */
	int		ppi_inc;		/* ptr increment in bytes */
	int		ppi_count;
	int		ppi_cmd;
} Proc_Info;

/* ppi flags */
#define PIC_READY_TO_GO		1
#define PIC_WAITING		2
#define PIC_STREAMING		4
#define PIC_EXITING		8

PIC_Device *pdp_tbl[MAX_PIC_DEVICES];
PIC_Device *curr_pdp=NULL;

static pthread_t pic_thr[MAX_PIC_DEVICES];
static Proc_Info ppi[MAX_PIC_DEVICES];


static int ask_addr(QSP_ARG_DECL  const char *prompt,int max);
#define ASK_ADDR(pmpt,max)		ask_addr(QSP_ARG  pmpt,max)
static void stream_data(QSP_ARG_DECL  int cmd_char);
#define STREAM_DATA(cmd_char)		stream_data(QSP_ARG  cmd_char)
static void stream_bytes(QSP_ARG_DECL  int cmd_char);
#define STREAM_BYTES(cmd_char)		stream_bytes(QSP_ARG  cmd_char)

static char addr_str[8];
static char byte_str[8];
static const char *firmware_version=NULL;

#define CHECK_PIC						\
								\
	if( curr_pdp == NULL ){					\
		WARN("No PIC device selected!?");		\
		return;						\
	}

ITEM_INTERFACE_DECLARATIONS(PIC_Device,pic_dev)

#define CHECK_RANGE( name, number, min, max )			\
{								\
	range_ok=1;						\
	if( number < min || number > max ) {			\
		sprintf(ERROR_STRING,				\
"%s (%d) must be between %d and %d", name, number, min, max);	\
		WARN(ERROR_STRING);				\
		range_ok=0;					\
	}							\
}


#define GET_WITH_LIMITS( var, string, minval, maxval )				\
										\
	var = HOW_MANY(string);							\
	CHECK_RANGE(string,var, minval, maxval)					\
	if( !range_ok ) ret_stat=(-1);



#define GET_PATTERN( var, string )						\
										\
	GET_WITH_LIMITS( var, string, MIN_CROSSPOINT_PATTERN, MAX_CROSSPOINT_PATTERN)


#define GET_SIGNAL( var, string )						\
										\
	GET_WITH_LIMITS( var, string, MIN_SIGNAL_NUMBER, MAX_SIGNAL_NUMBER )


static int ask_addr(QSP_ARG_DECL  const char *prompt,int max)
{
	int a;

	a=HOW_MANY(prompt);
	if( a < 0 || a > max ){
		sprintf(ERROR_STRING,"Address (0x%x) must be between 0 and 0x%d",a,max);
		WARN(ERROR_STRING);
		return(-1);
	}
	return(a);
}

#define SEND_PIC_CMD(pdp,buf)	send_pic_cmd(QSP_ARG  pdp,buf)


static void send_pic_cmd(QSP_ARG_DECL  PIC_Device *pdp, const char* buf )
{
	char cmd[LLEN];

	if( pdp == NULL ){
		sprintf(ERROR_STRING,"send_pic_cmd:  no PIC device specified");
		WARN(ERROR_STRING);
		return;
	}

	/* append carriage return */
	sprintf(cmd, "%s\n", buf);		/* in cbreak mode, \r is echoed as \n?? */

if( debug & pic_debug ){
sprintf(ERROR_STRING,"send_pic_cmd:  sending \"%s\"",buf);
advise(ERROR_STRING);
}
	send_serial(QSP_ARG  pdp->pd_fd, (u_char *)cmd, strlen(cmd));

	if( pic_nlcr_echo )
		strcat(cmd,"\r");	/* echo will add \r? */
	if( PIC_ECHOES ){
	/* listen for the echo */
if( debug & pic_debug ) advise("send_pic_cmd:  listening for cmd echo");
		expected_response(QSP_ARG  pdp->pd_sbp,cmd);
if( debug & pic_debug ) advise("send_pic_cmd:  DONE listening for cmd echo");
	}

	pdp->pd_prompt_seen=0;
}

static void get_firmware_version(SINGLE_QSP_ARG_DECL)
{
	char buf[16];
	int i;

	sprintf(buf, "%s", pic_tbl[PIC_RPT_VER].pc_str);
	reset_buffer(curr_pdp->pd_sbp);
if( debug & pic_debug ){
sprintf(ERROR_STRING,"Sending \"%s\" to PIC",printable_string(buf));
advise(ERROR_STRING);
}
	send_pic_cmd(QSP_ARG  curr_pdp,buf);

if( debug & pic_debug ) advise("get_firmware_version:  listening for output string");
	expected_response(QSP_ARG  curr_pdp->pd_sbp,"Vision Lab LED Controller version ");
if( debug & pic_debug ) advise("get_firmware_version:  calling read_until_string");
	read_until_string(QSP_ARG  buf,curr_pdp->pd_sbp,"\r",CONSUME_MARKER);

	/* Why is this here?
	 * Maybe we don't know if the pic will send \n\r or \r\n...
	 * Should we test strlen(buf) to make sure it is sensible?
	 */
	if( buf[ (i=strlen(buf)-1) ] == '\n' )
		buf[i] = 0;

if( debug & pic_debug ){
sprintf(ERROR_STRING,"get_firmware_version:  version string is \"%s\"",buf);
advise(ERROR_STRING);
}
	if( firmware_version != NULL )
		givbuf((char *)firmware_version);
	firmware_version = savestr(buf);

	/* Do we need to sleep here to wait for the prompt? */
if( debug & pic_debug ) advise("get_firmware_version:  listening for prompt");

	WAIT_FOR_PROMPT(curr_pdp)
} /* end get_firmware_version */

static COMMAND_FUNC( do_report_version )
{
	CHECK_PIC

	get_firmware_version(SINGLE_QSP_ARG);

	sprintf(msg_str,"Firmware version:  %s",firmware_version);
	prt_msg(msg_str);

	ASSIGN_VAR("firmware_version",firmware_version);
}

static char *fmt_addr(int addr)
{
	sprintf(addr_str,"%04x",addr);
	return(addr_str);
}

static char *fmt_byte(int val)
{
	sprintf(byte_str,"%02x",val);
	return(byte_str);
}

#define DATA_ADDR_PMPT	"data memory address"
#define MAX_DATA_ADDR	0x0400				/* BUG -double check this value! */

#define PGM_ADDR_PMPT	"program memory address"
// 0x400 is 1k, device has 8k or 0x2000
#define MAX_PGM_ADDR	0x1fff

static int read_pgm_mem(QSP_ARG_DECL  int addr)
{
	const char *s;
	char buf[LLEN];
	int ra,rd;

	sprintf(buf, "%s", pic_tbl[PIC_RD_PGM].pc_str);
	reset_buffer(curr_pdp->pd_sbp);
if( debug & pic_debug ){
sprintf(ERROR_STRING,"Sending \"%s\" to PIC",printable_string(buf));
advise(ERROR_STRING);
}
	SEND_PIC_CMD(curr_pdp,buf);
if( debug & pic_debug ) advise("listening for address prompt");
	expected_response(QSP_ARG  curr_pdp->pd_sbp,"Enter program memory address: ");
	s = fmt_addr(addr);
	SEND_PIC_CMD(curr_pdp,s);

if( debug & pic_debug ) advise("read_pgm_mem:  listening for reply");
	//read_until_string(QSP_ARG  buf,curr_pdp->pd_sbp,"\n",CONSUME_MARKER);
	read_until_string(QSP_ARG  buf,curr_pdp->pd_sbp,"\r",CONSUME_MARKER);

	// now parse the string
	sscanf(buf,"%4x:  %4x",&ra,&rd);
//sprintf(ERROR_STRING,"string \"%s\", ra = 0x%04x, rd = 0x%04x",buf,ra,rd);
//advise(ERROR_STRING);

	WAIT_FOR_PROMPT(curr_pdp)

	return(rd);
} /* end read_pgm_mem() */

static int read_mem(QSP_ARG_DECL  int addr)
{
	const char *s;
	char buf[LLEN];
	int ra,rd;

	sprintf(buf, "%s", pic_tbl[PIC_RD_DATA].pc_str);
	reset_buffer(curr_pdp->pd_sbp);
if( debug & pic_debug ){
sprintf(ERROR_STRING,"Sending \"%s\" to PIC",printable_string(buf));
advise(ERROR_STRING);
}
	SEND_PIC_CMD(curr_pdp,buf);
if( debug & pic_debug ) advise("listening for address prompt");
	expected_response(QSP_ARG  curr_pdp->pd_sbp,"Enter data address: ");
	s = fmt_addr(addr);
	SEND_PIC_CMD(curr_pdp,s);

if( debug & pic_debug ) advise("read_mem:  listening for reply");
	//read_until_string(QSP_ARG  buf,curr_pdp->pd_sbp,"\n");
	read_until_string(QSP_ARG  buf,curr_pdp->pd_sbp,"\r",CONSUME_MARKER);

	// now parse the string
	sscanf(buf,"%4x:  %4x",&ra,&rd);
//sprintf(ERROR_STRING,"string \"%s\", ra = 0x%04x, rd = 0x%04x",buf,ra,rd);
//advise(ERROR_STRING);

	WAIT_FOR_PROMPT(curr_pdp)

	return(rd);
} /* end read_mem() */

static void pic_echo(QSP_ARG_DECL  int onoff)
{
	const char *s;
	char buf[4];

	sprintf(buf, "%s", pic_tbl[PIC_ECHO].pc_str);
	reset_buffer(curr_pdp->pd_sbp);
if( debug & pic_debug ){
sprintf(ERROR_STRING,"Sending \"%s\" to PIC",printable_string(buf));
advise(ERROR_STRING);
}
	SEND_PIC_CMD(curr_pdp,buf);

if( debug & pic_debug ) advise("listening for yes/no prompt");
	expected_response(QSP_ARG  curr_pdp->pd_sbp,"Enter 1 to enable command echo, 0 to disable: ");
//advise("prompt seen, sending byte");
//show_buffer(curr_pdp->pd_sbp);

	if( onoff ) s = fmt_byte(1);
	else s = fmt_byte(0);

	SEND_PIC_CMD(curr_pdp,s);

	if( onoff )
		pic_state.ps_flags |= PIC_ECHO_ON;
	else
		pic_state.ps_flags &= ~PIC_ECHO_ON;

	WAIT_FOR_PROMPT(curr_pdp)
} /* end pic_echo */

static void goto_pgm_mem(QSP_ARG_DECL  int addr)
{
	const char *s;
	char buf[LLEN];

	sprintf(buf, "%s", pic_tbl[PIC_GOTO].pc_str);
	reset_buffer(curr_pdp->pd_sbp);
if( debug & pic_debug ){
sprintf(ERROR_STRING,"Sending \"%s\" to PIC",printable_string(buf));
advise(ERROR_STRING);
}
	SEND_PIC_CMD(curr_pdp,buf);
if( debug & pic_debug ) advise("listening for address prompt");
	expected_response(QSP_ARG  curr_pdp->pd_sbp,"Enter program memory address: ");
	s = fmt_addr(addr);
	SEND_PIC_CMD(curr_pdp,s);

if( debug & pic_debug ) advise("goto_pgm_mem:  listening for reply");

	/* We might expect some printout here, such as a boot message...
	 * Therefore, we don't expect the prompt immediately,
	 * but read until we see it.
	 */
	INSURE_PROMPT
if( debug & pic_debug ){
sprintf(ERROR_STRING,"Reading until prompt \"%s\" encountered...",
pic_prompt);
advise(ERROR_STRING);
}
	read_until_string(QSP_ARG  buf,curr_pdp->pd_sbp,pic_prompt,PRESERVE_MARKER);
if( debug & pic_debug ){
sprintf(ERROR_STRING,"Text read before prompt: \"%s\"",buf);
advise(ERROR_STRING);
}
	reset_buffer(curr_pdp->pd_sbp);
	curr_pdp->pd_prompt_seen=1;
}

static COMMAND_FUNC( do_read_pgm_mem )
{
	int addr,d;
	int n;

	if( (addr=ASK_ADDR(PGM_ADDR_PMPT,MAX_PGM_ADDR)) < 0 ) return;

	n = HOW_MANY("number of locations to read");
	if( n <= 0 ){
		WARN("number of locations to read must be positive");
		return;
	}

	CHECK_PIC

	while(n--){
		d=read_pgm_mem(QSP_ARG  addr);
		sprintf(msg_str,"P 0x%04x:  0x%04x",addr,d);
		prt_msg(msg_str);
		addr++;
	}
}


static COMMAND_FUNC( do_read_mem )
{
	int addr,d;
	int n;

	addr=ASK_ADDR(DATA_ADDR_PMPT,MAX_DATA_ADDR);

	n = HOW_MANY("number of locations to read");
	if( n <= 0 ){
		WARN("number of locations to read must be positive");
		return;
	}
	if( (addr) < 0 ) return;


	CHECK_PIC

	while(n--){
		d=read_mem(QSP_ARG  addr);
		sprintf(msg_str,"D 0x%04x:  0x%04x",addr,d);
		prt_msg(msg_str);
		addr++;
	}
}


static void write_pgm_mem(QSP_ARG_DECL  int addr, int data)
{
	const char *s;
	char buf[LLEN];

	sprintf(buf, "%s", pic_tbl[PIC_WR_PGM].pc_str);
	reset_buffer(curr_pdp->pd_sbp);
if( debug & pic_debug ){
sprintf(ERROR_STRING,"Sending \"%s\" to PIC",printable_string(buf));
advise(ERROR_STRING);
}
	SEND_PIC_CMD(curr_pdp,buf);
if( debug & pic_debug ) advise("listening for address prompt");
	expected_response(QSP_ARG  curr_pdp->pd_sbp,"Enter program memory address: ");
	s = fmt_addr(addr);
	SEND_PIC_CMD(curr_pdp,s);
	expected_response(QSP_ARG  curr_pdp->pd_sbp,"Enter program memory value: ");
	s = fmt_addr(data);
	SEND_PIC_CMD(curr_pdp,s);

	WAIT_FOR_PROMPT(curr_pdp)
} /* end write_pgm_mem() */

static void write_mem(QSP_ARG_DECL  int addr, int data)
{
	const char *s;
	char buf[LLEN];

	sprintf(buf, "%s", pic_tbl[PIC_WR_DATA].pc_str);
	reset_buffer(curr_pdp->pd_sbp);
if( debug & pic_debug ){
sprintf(ERROR_STRING,"Sending \"%s\" to PIC",printable_string(buf));
advise(ERROR_STRING);
}
	SEND_PIC_CMD(curr_pdp,buf);
if( debug & pic_debug ) advise("listening for address prompt");
	expected_response(QSP_ARG  curr_pdp->pd_sbp,"Enter data address: ");
	s = fmt_addr(addr);
	SEND_PIC_CMD(curr_pdp,s);
	expected_response(QSP_ARG  curr_pdp->pd_sbp,"Enter program memory value: ");
	s = fmt_addr(data);
	SEND_PIC_CMD(curr_pdp,s);

	WAIT_FOR_PROMPT(curr_pdp)
} /* write_mem() */

static COMMAND_FUNC( do_write_pgm_mem )
{
	int addr;
	int data;

	addr=ASK_ADDR(PGM_ADDR_PMPT,MAX_PGM_ADDR);
	data = HOW_MANY("14 bit data value");		/* no range checking... */

	if( addr < 0 ) return;


	CHECK_PIC

	write_pgm_mem(QSP_ARG  addr,data);
}

static COMMAND_FUNC( do_write_mem )
{
	int addr;
	int data;

	addr=ASK_ADDR(DATA_ADDR_PMPT,MAX_DATA_ADDR);
	data = HOW_MANY("8 bit data value");		/* no range checking... */

	if( addr < 0 ) return;


	CHECK_PIC

	write_mem(QSP_ARG  addr,data);
}

static COMMAND_FUNC( do_goto )
{
	int addr;
	addr=ASK_ADDR(PGM_ADDR_PMPT,MAX_PGM_ADDR);
	if( addr < 0 ) return;

	CHECK_PIC

	goto_pgm_mem(QSP_ARG  addr);
}

static void open_pic_device(QSP_ARG_DECL  const char *s)
{
	int fd;
	PIC_Device *pdp;

	pdp = pic_dev_of(QSP_ARG  s);
	if( pdp != NULL ){
		sprintf(ERROR_STRING,
	"open_pic_device:  pic device %s is already open",s);
		WARN(ERROR_STRING);
		return;
	}
	if( n_pics_active >= MAX_PIC_DEVICES ){
		sprintf(ERROR_STRING,
			"Max number (%d) of PIC devices already open, won't attempt to open %s.",
			MAX_PIC_DEVICES,s);
		WARN(ERROR_STRING);
		return;
	}

	if( (fd = open_serial_device(QSP_ARG  s)) < 0 ){ 
		sprintf(ERROR_STRING,"Unable to open pic device %s",s);
		WARN(ERROR_STRING);
		return;
	}

	pdp = new_pic_dev(QSP_ARG  s);
	pdp->pd_fd=fd;
	pdp->pd_prompt_seen=0;

	ttyraw(fd);
	echooff(fd);			/* BUG should set all options here... */
	// This appears to be the older firmware???
	//set_baud(fd,B19200);
	// This is for new firmware???
	set_baud(fd,B38400);

	/* set various other flags - not sure which are truly needed... */

	GETTERM;

	/* We should probably set ALL flags just so we know??? */

	/* input options */
	tiobuf.c_iflag |= IGNPAR;	/* ignore parity */

	/* control flag options */
	tiobuf.c_cflag &= ~PARENB;	/* no parity on output */
	tiobuf.c_cflag &= ~CRTSCTS;	/* disable hardware flow control */

	/* output options */
	tiobuf.c_oflag |= CSTOPB;
	tiobuf.c_oflag |= HUPCL;

	/* local flags */
	tiobuf.c_lflag &= ~ECHOE;
	tiobuf.c_lflag &= ~ECHOK;
	tiobuf.c_lflag &= ~ECHOKE;
	tiobuf.c_lflag &= ~ECHOCTL;

	SETTERM;

	pdp->pd_sbp = (Serial_Buffer *)getbuf(sizeof(*pdp->pd_sbp));
	pdp->pd_sbp->sb_fd = fd;
	reset_buffer(pdp->pd_sbp);

	curr_pdp = pdp;
	pdp_tbl[n_pics_active] = pdp;
	n_pics_active++;
}

static COMMAND_FUNC( do_select_device )
{
	const char *s;
	PIC_Device *pdp;

	s=NAMEOF("pic device");

	pdp = pic_dev_of(QSP_ARG  s);
	if( pdp != NULL ){
		curr_pdp = pdp;
	} else {
		open_pic_device(QSP_ARG  s);
	}
}

static int get_hex_digit(QSP_ARG_DECL  const char **sp)
{
	const char *s;
	int d;

	s = *sp;
	if( *s == 0 ){
		WARN("get_hex_digit:  no more buffered chars!?");
		return(-1);
	}

	if( isdigit(*s) ){
		d = *s - '0';
	} else if( isalpha(*s) ){
		if( isupper(*s) )
			d = 10 + (*s) - 'A';
		else
			d = 10 + (*s) - 'a';

		if( d < 10 || d > 15 ){
			WARN("bad alpha hex digit");
			d=(-1);
		}
	}
#ifdef CAUTIOUS
	  else {
		/* silence compiler */
		d=0;
ERROR1("CAUTIOUS:  get_hex_digit:  character is not a hex digit!?");
	}
#endif /* CAUTIOUS */
	s++;
	*sp = s;
	return(d);
}
		
static int get_byte(QSP_ARG_DECL  const char **sp)
{
	int n,c;

	n = get_hex_digit(QSP_ARG  sp);
	if( n < 0 ) {
		WARN("get_byte:  problem with first digit");
		return(n);
	}
	c = n << 4;
	n = get_hex_digit(QSP_ARG  sp);
	if( n < 0 ) {
		WARN("get_byte:  problem with second digit");
		return(n);
	}
	c += n;
	return(c);
}

static int get_word(QSP_ARG_DECL  const char **sp)
{
	int lsb,msb;

	lsb=get_byte(QSP_ARG  sp);
	if( lsb < 0 ) {
		WARN("get_word: problem with first byte");
		return(lsb);
	}
	msb=get_byte(QSP_ARG  sp);
	if( msb < 0 ) {
		WARN("get_word: problem with second byte");
		return(msb);
	}
	return( (msb<<8) + lsb );
}

static int get_addr(QSP_ARG_DECL  const char **sp)
{
	int lsb,msb;

	msb=get_byte(QSP_ARG  sp);
	if( msb < 0 ) {
		WARN("get_addr: problem with first byte");
		return(msb);
	}
	lsb=get_byte(QSP_ARG  sp);
	if( lsb < 0 ) {
		WARN("get_addr: problem with second byte");
		return(lsb);
	}
	return( (msb<<8) + lsb );
}

/* we can call this with write_data = 0 for a syntax check...
 */

static int scan_line(QSP_ARG_DECL   const char **sp, int what_to_do )
{
	const char *s;
	int a,b,c,count;
	int checksum1,checksum2;
	int w;

	s = *sp;
	if( *s != ':' ){
		sprintf(ERROR_STRING,"hex file line \"%s\" does not begin with ':' !?",s);
		WARN(ERROR_STRING);
		return(-1);
	}
	s++;
	*sp = s;
	count = get_byte(QSP_ARG  sp);
	if( count < 0 ){
		WARN("scan_line:  problem with count");
		return(-1);
	}
	if( count & 1 ){
		sprintf(ERROR_STRING,"count (0x%02x) should be even for PIC",count);
		WARN(ERROR_STRING);
		return(-1);
	}
	a = get_addr(QSP_ARG  sp);
	if( a < 0 ){
		WARN("scan_line:  problem with address");
		return(-1);
	}
	if( a & 1 ){
		sprintf(ERROR_STRING,"Address (0x%04x) is odd, should be even for PIC",a);
		WARN(ERROR_STRING);
		return(-1);
	}
	a /= 2;
	checksum1 = get_byte(QSP_ARG  sp);
	if( checksum1 < 0 ){
		WARN("scan_line:  problem with first checksum byte");
		return(-1);
	}
	/* not sure what to do with checksum1... */
	c = count / 2;		/* load by words, not bytes */
	while(c--){
		w = get_word(QSP_ARG  sp);
		if( w < 0 ){
			WARN("scan_line:  problem with data word");
			return(-1);
		}
		if( what_to_do == WRITE_DATA ){
			write_pgm_mem(QSP_ARG  a,w);
			if( verbose ){
				sprintf(ERROR_STRING,"P %04x %04x",a,w);
				advise(ERROR_STRING);
			}
			a++;
		} else if( what_to_do == VERIFY_DATA ){
			int d;
			d = read_pgm_mem(QSP_ARG  a);
			if( d != w ){
				sprintf(ERROR_STRING,"0x%04x:  0x%04x  --  0x%04x expected",a,d,w);
				WARN(ERROR_STRING);
			}
			a++;
		}
	}
	checksum2 = get_byte(QSP_ARG  sp);	/* checksum */
	if( checksum2 < 0 ){
		WARN("missing final checksum!?");
		return(-1);
	}

	s = *sp;
	b = strlen(s);
	if( b != 0 ){
		sprintf(ERROR_STRING,"scan_line:  %d extra characters at end of record!?",b);
		WARN(ERROR_STRING);
		return(-1);
	}

	/* If we got here, then it is a legal hexfile line.
	 * But we may not want to send it, for example if it is the configuration bits
	 */
	if( a >= 0x2000 ){
		if( a == 0x2007 && count == 2 ){		/* configuration word, don't sweat it */
			advise("\nignoring configuration word");
			return(1);
		}
		sprintf(ERROR_STRING,"Address 0x%x out of range",a);
		WARN(ERROR_STRING);
		return(-1);
	}
	
	return(0);
}

#ifdef FOOBAR
static int old_load_hex_line(QSP_ARG_DECL  char *s)
{
	char *s2;

	s2=s;
	if( scan_line(QSP_ARG  &s2,CHECK_SYNTAX) < 0 ){
		WARN("aborting hex load");
		return(-1);
	}
	s2=s;
	if( scan_line(QSP_ARG  &s2,WRITE_DATA) < 0 ){
		WARN("error writing line from hex file");
		return(-1);
	}
	return(0);
}
#endif /* FOOBAR */

static int load_hex_line(QSP_ARG_DECL  const char *s)
{
	const char *s2;
	int status;

	s2=s;
	if( (status=scan_line(QSP_ARG  &s2,CHECK_SYNTAX)) != 0 ){
		if( status < 0 ){
			WARN("aborting hex load");
			return(-1);
		}
		return(0);	/* a line we choose not to send */
	}
	SEND_PIC_CMD(curr_pdp,s);
	WAIT_FOR_PROMPT(curr_pdp)
	return(0);
} /* end load_hex_line */

static void verify_hex_line(QSP_ARG_DECL  const char *s)
{
	int status;
	const char *s2;

	s2=s;
	if( (status=scan_line(QSP_ARG  &s2,VERIFY_DATA)) != 0 ){
		if( status < 0 )
			WARN("error verifying hex data");
		else
			WARN("unexpected status returned from scan_line!?");
	}
}

/* BUG we should only allow the N command if the firmware supports it! */

#define VALIDATE_CMD_CHAR(c)							\
										\
	if( c != 'L' && c != 'T' && c != 'N' ){						\
		sprintf(ERROR_STRING,"CAUTIOUS:  cmd_char 0x%x should be 'L' (0x%x), 'T' (0x%x) or 'N' (0x%x)",	\
			c,'L','T','N');						\
		WARN(ERROR_STRING);						\
		return;								\
	}


static void send_short_data(QSP_ARG_DECL  int cmd_char, Data_Obj *dp)	/* send a single squirt */
{
	dimension_t n;
	u_short s1,s2,s3,s4,*sp;
	char buf[32];

#ifdef CAUTIOUS
	VALIDATE_CMD_CHAR(cmd_char)
#endif

	if( dp->dt_comps != 4 ){
		sprintf(ERROR_STRING,"send_short_data:  object %s (%d) should have 4 components",
			dp->dt_name,dp->dt_comps);
		WARN(ERROR_STRING);
		return;
	}
	/* we don't need to do this, but for now we will because we are lazy */
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"Sorry, object %s should be contiguous for send_short_data",dp->dt_name);
		WARN(ERROR_STRING);
		return;
	}

	n = dp->dt_n_mach_elts / 4;
	sp = (u_short *) dp->dt_data;
	while(n--){
		s1 = *sp++;
		s2 = *sp++;
		s3 = *sp++;
		s4 = *sp++;
		sprintf(buf,"%c%04x%04x%04x%04x",cmd_char,s1,s2,s3,s4);
		SEND_PIC_CMD(curr_pdp,buf);
	}
}

static void stream_long_data(QSP_ARG_DECL  Proc_Info *pip)
{
	char cmd_char;
	PIC_Device *pdp;
	void *ptr;
	int inc, count;
	char buf[32];
	long *lp, l1,l2;

	pdp = pip->ppi_pdp;
	ptr = pip->ppi_ptr;
	inc = pip->ppi_inc;
	count = pip->ppi_count;
	cmd_char = pip->ppi_cmd;

	sprintf(buf,"z");		/* wait_even command */
	SEND_PIC_CMD(pdp,buf);

	WAIT_FOR_PROMPT(pdp)

	lp = (long *) ptr;
	while(count--){
		l1 = *lp;
		l2 = *(lp+1);
		sprintf(buf,"%c%08lx%08lx",cmd_char,l1,l2);
		SEND_PIC_CMD(pdp,buf);
		WAIT_FOR_PROMPT(pdp)
		lp += inc/sizeof(long);
	}
}

static void * pic_streamer(void *argp)
{
	Proc_Info *pip;

	pip = (Proc_Info*) argp;

	pip->ppi_pid = getpid();

	/* tell the parent that we're ready, and wait for siblings */
	pip->ppi_flags |= PIC_READY_TO_GO;

	/* wait for host to signal that all threads are ready before we start */

	pip->ppi_flags |= PIC_STREAMING;

	stream_long_data(DEFAULT_QSP_ARG  pip);

	pip->ppi_flags = PIC_EXITING;

	return(NULL);
}

static void start_pic_threads(QSP_ARG_DECL  Data_Obj *dp, int cmd_char)
{
	int i;
	pthread_attr_t attr1;

	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);

	for(i=0;i<channels_per_pixel;i++){
		sprintf(ERROR_STRING,"pic_thread%d",i);
		ppi[i].ppi_name = savestr(ERROR_STRING);

		ppi[i].ppi_index=i;
		ppi[i].ppi_flags = 0;

		if( channels_per_pixel == 1 )
			ppi[i].ppi_pdp = curr_pdp;
		else
			ppi[i].ppi_pdp = pdp_tbl[i];

		ppi[i].ppi_inc = dp->dt_comps * ELEMENT_SIZE(dp);
		ppi[i].ppi_count = dp->dt_n_mach_elts / dp->dt_comps ;
		ppi[i].ppi_ptr = ((char *)dp->dt_data) + i * 8;	/* each channel is 64 bits */
		ppi[i].ppi_cmd = cmd_char;

		pthread_create(&pic_thr[i],&attr1,pic_streamer,&ppi[i]);
	}
}

static void wait_pic_threads(SINGLE_QSP_ARG_DECL)
{
	int i;

	for(i=0;i<channels_per_pixel;i++){
		if( pthread_join(pic_thr[i],NULL) != 0 ){
			WARN("wait_pic_threads:  error return from pthread_join()");
		}
	}
}

/* send data and return immediately - used with capture flow */

static COMMAND_FUNC( do_led64 )
{
	Data_Obj *dp;
	int cmd_char;

	dp = PICK_OBJ("64 bit data object");
	if( dp == NO_OBJ ) return;

	CHECK_PIC

	/* BUG - the N command is not necessarily present.
	 * Need to have a database of firmware versions and cmds.
	 */

	if( sync_to_video )
		cmd_char = 'L';
	else
		cmd_char = 'N';

	send_short_data(QSP_ARG  cmd_char,dp);

	/* need to issue a command to sync up the prompt! */
	WAIT_FOR_PROMPT(curr_pdp)		/* BUG?  need to pass pdp? */
}

static COMMAND_FUNC( do_set_vsync )
{
	sync_to_video = ASKIF("Synchronize LED transitions with video signal");
}

static COMMAND_FUNC( do_timed64 )
{
	Data_Obj *dp;

	dp = PICK_OBJ("64 bit data object");
	if( dp == NO_OBJ ) return;

	CHECK_PIC

	send_short_data(QSP_ARG  'T',dp);		/* assume someone else has sync'd */
}

static COMMAND_FUNC( do_wait_pmpt )
{
	CHECK_PIC

	WAIT_FOR_PROMPT(curr_pdp)
}

static COMMAND_FUNC( do_led8 )
{
	int chan, data;
	char buf[4];

	chan = HOW_MANY("channel index (0-7)");
	data = HOW_MANY("data (0-255)");

	if( chan < 0 || chan > 7 ){
		WARN("bad channel index");
		return;
	}
	if( data < 0 || data > 255 ){
		WARN("bad data value");
		return;
	}

	CHECK_PIC

	SEND_PIC_CMD(curr_pdp,"l");

	expected_response(QSP_ARG  curr_pdp->pd_sbp,"Enter channel number: ");
	sprintf(buf,"%02x",chan);
	SEND_PIC_CMD(curr_pdp,buf);

	expected_response(QSP_ARG  curr_pdp->pd_sbp,"Enter led byte: ");
	sprintf(buf,"%02x",data);
	SEND_PIC_CMD(curr_pdp,buf);

	WAIT_FOR_PROMPT(curr_pdp)
}

static COMMAND_FUNC( do_set_timer )
{
	int counth;
	char buf[4];

	//countl=HOW_MANY("counter_l");
	counth=HOW_MANY("timer count");
	if( counth < 0 || counth > 255 ){
		WARN("count must be in the range 0-255");
		return;
	}

	CHECK_PIC

	SEND_PIC_CMD(curr_pdp,"t");

	//expected_response(QSP_ARG  curr_pdp->pd_sbp,"Enter LED counterl: ");
	//sprintf(buf,"%02x",countl);
	//SEND_PIC_CMD(curr_pdp,buf);

	expected_response(QSP_ARG  curr_pdp->pd_sbp,"Enter LED counterh: ");
	sprintf(buf,"%02x",counth);
	SEND_PIC_CMD(curr_pdp,buf);

	WAIT_FOR_PROMPT(curr_pdp)
}

static COMMAND_FUNC( do_wait_even )
{
	CHECK_PIC
	SEND_PIC_CMD(curr_pdp,"z");
	WAIT_FOR_PROMPT(curr_pdp)
}

static COMMAND_FUNC( do_wait_odd )
{
	CHECK_PIC
	SEND_PIC_CMD(curr_pdp,"y");
	WAIT_FOR_PROMPT(curr_pdp)
}

static void stream_byte_data(QSP_ARG_DECL  int cmd_char, Data_Obj *dp)
{
	dimension_t n;
	u_char c1;
	u_char *cp;
	char buf[32];

	if( dp->dt_comps != 1 ){
		sprintf(ERROR_STRING,"stream_byte_data:  object %s (%d) should have 1 component",
			dp->dt_name,dp->dt_comps);
		WARN(ERROR_STRING);
		return;
	}
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"Sorry, object %s should be contiguous for stream_byte_data",dp->dt_name);
		WARN(ERROR_STRING);
		return;
	}
	n = dp->dt_n_mach_elts ;
	cp = (u_char *) dp->dt_data;
	while(n--){
		c1 = *cp++;
		sprintf(buf,"%c%02x",cmd_char,c1);
		SEND_PIC_CMD(curr_pdp,buf);
		WAIT_FOR_PROMPT(curr_pdp)
	}
}


static void stream_data(QSP_ARG_DECL  int cmd_char)
{
	Data_Obj *dp;
	int bytes_per_pixel;

	dp = PICK_OBJ("LED data vector");

#ifdef CAUTIOUS
	VALIDATE_CMD_CHAR(cmd_char)
#endif

	if( dp == NO_OBJ ){
		WARN("stream_data:  invalid data vector");
		return;
	}
	/* make sure that each vector element has a multiple of 64 bits.
	 * The multiplier needs to be less than or equal to the number of active pic's.
	 */

	bytes_per_pixel = ELEMENT_SIZE(dp) * dp->dt_comps;
	if( bytes_per_pixel%8 != 0 ){
		sprintf(ERROR_STRING,"stream_data:  data vector %s has %d bytes per pixel.",
			dp->dt_name,bytes_per_pixel);
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,"which is not an integral number of 64 bit channels.");
		advise(ERROR_STRING);
		return;
	}

	channels_per_pixel = bytes_per_pixel/8;

	if( channels_per_pixel > n_pics_active ){
		sprintf(ERROR_STRING,"stream_data:  vector %s has data for %d channels, but only %d devices are active",
			dp->dt_name,channels_per_pixel,n_pics_active);
		WARN(ERROR_STRING);
		return;
	}

	/* If 1 channel, use current device.
	 * If n_channels == number of active devices, use all devices in order.
	 */
	if( channels_per_pixel != n_pics_active ){
		if( channels_per_pixel > 1 ){
			sprintf(ERROR_STRING,"stream_data:  vector %s has data for %d channels, will use the first %d of %d active devices",
				dp->dt_name,channels_per_pixel,channels_per_pixel,n_pics_active);
			advise(ERROR_STRING);
		} else {
			sprintf(ERROR_STRING,"stream_data:  using active device %s",curr_pdp->pd_name);
			advise(ERROR_STRING);
		}
	}

	/* we don't need to do this, but for now we will because we are lazy */
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"Sorry, object %s should be contiguous for stream_long_data",dp->dt_name);
		WARN(ERROR_STRING);
		return;
	}

	/* create 1 thread per channel */
	start_pic_threads(QSP_ARG  dp,cmd_char);

	/* here we should wait for the threads to finish... */
	wait_pic_threads(SINGLE_QSP_ARG);
}

static void stream_bytes(QSP_ARG_DECL  int cmd_char)
{
	Data_Obj *dp;

	dp=PICK_OBJ("byte data vector");
	if( dp == NO_OBJ ){
		WARN("stream_bytes:  invalid data vector");
		return;
	}
	if( MACHINE_PREC(dp) != PREC_UBY ){
		sprintf(ERROR_STRING,"stream_bytes:  data vector %s (%s) should have %s precision",
			dp->dt_name,name_for_prec(MACHINE_PREC(dp)),name_for_prec(PREC_UBY));
		WARN(ERROR_STRING);
		return;
	}
	stream_byte_data(QSP_ARG  cmd_char,dp);
}

static COMMAND_FUNC( do_stream_led )
{
	CHECK_PIC

	STREAM_DATA('L');
}

static COMMAND_FUNC( do_stream_timed )
{
	CHECK_PIC

	STREAM_DATA('T');
}

static COMMAND_FUNC( do_stream_pwm )
{
	CHECK_PIC

	STREAM_BYTES('P');
}

static COMMAND_FUNC( do_load )
{
	const char *s;
	FILE *fp;
	int l;
	int problems=0;
	int count=0;

	s=NAMEOF("hex file");
	fp=TRY_OPEN(s,"r");
	if( !fp ) return;

	CHECK_PIC

	PUSH_INPUT_FILE(s);
	REDIR(fp);
	l = TELL_QLEVEL;
	while( (TELL_QLEVEL == l) && (problems==0) ){
		prt_msg_frag(".");
		if( ((++count) & 0x1f ) == 0 )
			prt_msg("");
		s=NAMEOF("hex file line");
		problems=load_hex_line(QSP_ARG  s);
		lookahead_til(QSP_ARG  l-1);
	}
	prt_msg("");
	/* fp is closed automatically when it is popped */
	if( TELL_QLEVEL == l ){	/* problems */
		popfile(SINGLE_QSP_ARG);
	}
}

static COMMAND_FUNC( do_verify )
{
	const char *s;
	FILE *fp;
	int l;
	int problems=0;
	int count=0;

	s=NAMEOF("hex file");
	fp=TRY_OPEN(s,"r");
	if( !fp ) return;

	CHECK_PIC

	PUSH_INPUT_FILE(s);
	REDIR(fp);
	l = TELL_QLEVEL;
	while( (TELL_QLEVEL == l) && (problems==0) ){
		prt_msg_frag(".");
		if( ((++count) & 0x1f ) == 0 )
			prt_msg("");
		s=NAMEOF("hex file line");
		verify_hex_line(QSP_ARG  s);
		lookahead_til(QSP_ARG  l-1);
	}
	prt_msg("");
	/* fp is closed automatically when it is popped */
	if( TELL_QLEVEL == l ){	/* problems */
		popfile(SINGLE_QSP_ARG);
	}
}

static COMMAND_FUNC( do_need )
{
	const char *s;

	s=NAMEOF("expected firmware version");
	if( firmware_version == NULL )
		get_firmware_version(SINGLE_QSP_ARG);
	if( strcmp(s,firmware_version) ){
		sprintf(ERROR_STRING,"Expected firmware version %s, but version %s is running!?",
			s,firmware_version);
		ERROR1(ERROR_STRING);
	}
}

static COMMAND_FUNC( do_pic_echo )
{
	int yn;

	yn = ASKIF("Should PIC echo commands");

	pic_echo(QSP_ARG  yn);
}

static COMMAND_FUNC( do_send_char )
{
	const char *s;

	s=NAMEOF("single character command");
	/* should we check that it really is a single char? */

	reset_buffer(curr_pdp->pd_sbp);
	SEND_PIC_CMD(curr_pdp,s);
	WAIT_FOR_PROMPT(curr_pdp)
}

COMMAND_FUNC( do_pic_set_pmpt )
{
	const char *s;

	s=NAMEOF("prompt used by PIC interpreter");

	if( pic_prompt != NULL ) rls_str(pic_prompt);
	pic_prompt = savestr(s);
}

static COMMAND_FUNC( do_flush )
{
	int n;

	CHECK_PIC

	/* read in any chars that may be readable */
	n=n_serial_chars(QSP_ARG  curr_pdp->pd_sbp->sb_fd);
	if( n > 0 )
		replenish_buffer(QSP_ARG  curr_pdp->pd_sbp,n);

	/* now flush 'em */
	reset_buffer(curr_pdp->pd_sbp);

	//read_until_string(QSP_ARG  buf,curr_pdp->pd_sbp,"\r",CONSUME_MARKER);
}

static COMMAND_FUNC( do_sync )
{
	CHECK_PIC

	do_flush(SINGLE_QSP_ARG);
	SEND_PIC_CMD(curr_pdp,"");	/* null command */

	WAIT_FOR_PROMPT(curr_pdp)
}

static COMMAND_FUNC( do_pic_crnl )
{
	pic_nlcr_echo = ASKIF("pic echoes two newline chars");
}

static Command pic_main_ctbl[] = {
{ "device",	do_select_device,	"select pic device"			},
{ "version",	do_report_version,	"report firmware version number"	},
{ "read_pgm",	do_read_pgm_mem,	"read program memory"			},
{ "write_pgm",	do_write_pgm_mem,	"write program memory"			},
{ "read_mem",	do_read_mem,		"read memory"				},
{ "write_mem",	do_write_mem,		"write memory"				},
{ "load",	do_load,		"load hex file"				},
{ "verify",	do_verify,		"verify correct loading of hex file"	},
{ "goto",	do_goto,		"begin execution at specified location"	},
{ "led8",	do_led8,		"specify a byte of LED data"	},
{ "stream_led",	do_stream_led,		"stream data to LED controllers"	},
{ "stream_timer",do_stream_timed,	"stream 64 bit data, fixed timed pulses"	},
{ "stream_pwm",	do_stream_pwm,		"stream variable pulse widths, fixed LED pattern"	},
{ "led64",	do_led64,		"specify 64 bits of LED data"		},
{ "vsync",	do_set_vsync,		"specify synch. w/ video signal"	},
{ "timed64",	do_timed64,		"specify 64 bit data, timed pulses"	},
{ "timer_counts",	do_set_timer,	"set LED timer counts"			},
{ "wait_even",	do_wait_even,		"wait for the end of an even field"	},
{ "wait_odd",	do_wait_odd,		"wait for the end of an odd field"	},
{ "wait_prompt",	do_wait_pmpt,	"wait for PIC prompt"			},
{ "need_version",	do_need,	"specify desired firmware version"	},
{ "flush",	do_flush,		"flush pending input"			},
{ "sync",	do_sync,		"sync program with pic prompt"		},
{ "pic_echo",	do_pic_echo,		"enable/disable command echo"		},
{ "set_prompt",	do_pic_set_pmpt,	"specify prompt used by PIC interpreter"	},
{ "pic_crnl_echo",	do_pic_crnl,	"specify whether firmware echoes nl-cr for cr"	},
{ "send_char_cmd",	do_send_char,	"send a single-character command"	},
{ "quit",	popcmd,			"exit submenu"				},
{ NULL_COMMAND									}
};

/* This was /dev/ttyS0 on fourier, but we insist on using a symlink
 * /dev/pic to make this portable to other systems which might use
 * a different port.
 */

#define PIC_TTY_DEV	"/dev/pic"

static void check_pic_tbl(SINGLE_QSP_ARG_DECL)		// make sure it's in the right order
{
	int i;

	for(i=0;i<N_PIC_CMDS;i++){
		if( pic_tbl[i].pc_code != i ){
			sprintf(ERROR_STRING,"Command code at tbl location %d is %d, expected %d",i,
				pic_tbl[i].pc_code,i);
			ERROR1(ERROR_STRING);
		}
	}
}

COMMAND_FUNC( picmenu )
{
	if( pic_debug == 0 ) {
		check_pic_tbl(SINGLE_QSP_ARG);
		auto_version(QSP_ARG  "PIC","VersionId_pic");
		pic_debug = add_debug_module(QSP_ARG  "pic");

		/* Should we automatically open /dev/pic? */

		/* BUG we need some way of insuring that the pic is in a known state */
		pic_state.ps_flags |= PIC_ECHO_ON;
		/* BUG we should only do this when we want to stream data... */
		//rt_sched(1);
	}

	PUSHCMD(pic_main_ctbl, "pic");
}

#endif /* HAVE_PIC */

