#include "quip_config.h"

//#define MARK_TIME		// enable debug output

#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include <sys/time.h>
#include <pthread.h>
#include <termios.h>

#include "quip_prot.h"
#include "pic.h"
#include "serbuf.h"
#include "item_type.h"
#include "debug.h"
#include "serial.h"
#include "getbuf.h"
#include "ttyctl.h"
#include "data_obj.h"
#include "query_stack.h"	// like to eliminate this dependency...

/* static */ u_long pic_debug=0;
static int channels_per_pixel;

#define TIMER_STOPPED	1
#define TIMER_RUNNING	2

static int pic_nlcr_echo=1;	/* newer firmware echos cr-nl for cr, and nl-cr for nl... */

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
		warn(ERROR_STRING);				\
	} else {						\
		INSURE_PROMPT					\
		expected_response((pdp)->pd_sbp,pic_prompt);	\
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
#ifdef THREAD_SAFE_QUERY
	Query_Stack *	ppi_qsp;
#endif // THREAD_SAFE_QUERY
} Proc_Info;

/* ppi flags */
#define PIC_READY_TO_GO		1
#define PIC_WAITING		2
#define PIC_STREAMING		4
#define PIC_EXITING		8

static PIC_Device *pdp_tbl[MAX_PIC_DEVICES];
static PIC_Device *curr_pdp=NULL;

static pthread_t pic_thr[MAX_PIC_DEVICES];
static Proc_Info ppi[MAX_PIC_DEVICES];


static int ask_addr(QSP_ARG_DECL  char *prompt,int max);
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
		warn("No PIC device selected!?");		\
		return;						\
	}

//ITEM_INTERFACE_DECLARATIONS_STATIC(PIC_Device,pic_dev,0)
static Item_Type *pic_dev_itp=NULL;
static ITEM_INIT_FUNC(PIC_Device,pic_dev,0)
static ITEM_NEW_FUNC(PIC_Device,pic_dev)
static ITEM_CHECK_FUNC(PIC_Device,pic_dev)

#define pic_dev_of(s)	_pic_dev_of(QSP_ARG  s)
#define new_pic_dev(s)	_new_pic_dev(QSP_ARG  s)

#define CHECK_RANGE( name, number, min, max )			\
{								\
	range_ok=1;						\
	if( number < min || number > max ) {			\
		sprintf(ERROR_STRING,				\
"%s (%d) must be between %d and %d", name, number, min, max);	\
		warn(ERROR_STRING);				\
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


static int ask_addr(QSP_ARG_DECL  char *prompt,int max)
{
	int a;

	a=HOW_MANY(prompt);
	if( a < 0 || a > max ){
		sprintf(ERROR_STRING,"Address (0x%x) must be between 0 and 0x%d",a,max);
		warn(ERROR_STRING);
		return(-1);
	}
	return(a);
}


#define send_pic_cmd(pdp,buf) _send_pic_cmd(QSP_ARG  pdp, buf )

static void _send_pic_cmd(QSP_ARG_DECL  PIC_Device *pdp, const char* buf )
{
	char cmd[LLEN];

	if( pdp == NULL ){
		sprintf(ERROR_STRING,"send_pic_cmd:  no PIC device specified");
		warn(ERROR_STRING);
		return;
	}

	/* append carriage return */
	sprintf(cmd, "%s\n", buf);		/* in cbreak mode, \r is echoed as \n?? */

if( debug & pic_debug ){
sprintf(ERROR_STRING,"send_pic_cmd:  sending \"%s\"",buf);
advise(ERROR_STRING);
}
	send_serial(pdp->pd_fd, (u_char *)cmd, strlen(cmd));

	if( pic_nlcr_echo )
		strcat(cmd,"\r");	/* echo will add \r? */
	if( PIC_ECHOES ){
	/* listen for the echo */
if( debug & pic_debug ) advise("send_pic_cmd:  listening for cmd echo");
		expected_response(pdp->pd_sbp,cmd);
if( debug & pic_debug ) advise("send_pic_cmd:  DONE listening for cmd echo");
	}

	pdp->pd_prompt_seen=0;
}

#define get_firmware_version() _get_firmware_version(SINGLE_QSP_ARG)

static void _get_firmware_version(SINGLE_QSP_ARG_DECL)
{
	char buf[16];
	int i;

	sprintf(buf, "%s", pic_tbl[PIC_RPT_VER].pc_str);
	reset_buffer(curr_pdp->pd_sbp);
if( debug & pic_debug ){
sprintf(ERROR_STRING,"Sending \"%s\" to PIC",printable_string(buf));
advise(ERROR_STRING);
}
	send_pic_cmd(curr_pdp,buf);

if( debug & pic_debug ) advise("get_firmware_version:  listening for output string");
	expected_response(curr_pdp->pd_sbp,"Vision Lab LED Controller version ");
if( debug & pic_debug ) advise("get_firmware_version:  calling read_until_string");
	read_until_string(buf,curr_pdp->pd_sbp,"\r",CONSUME_MARKER);

	/* Why is this here? */
	if( buf[ (i=strlen(buf)-1) ] == '\n' )
		buf[i] = 0;

	if( firmware_version != NULL )
		givbuf((char *)firmware_version);
	firmware_version = savestr(buf);

if( debug & pic_debug ) advise("get_firmware_version:  listening for prompt");
	WAIT_FOR_PROMPT(curr_pdp)
} /* end get_firmware_version */

static COMMAND_FUNC( do_report_version )
{
	CHECK_PIC

	get_firmware_version();

	sprintf(msg_str,"Firmware version:  %s",firmware_version);
	prt_msg(msg_str);

	assign_var("firmware_version",firmware_version);
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

#define read_pgm_mem(addr) _read_pgm_mem(QSP_ARG  addr)

static int _read_pgm_mem(QSP_ARG_DECL  int addr)
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
	send_pic_cmd(curr_pdp,buf);
if( debug & pic_debug ) advise("listening for address prompt");
	expected_response(curr_pdp->pd_sbp,"Enter program memory address: ");
	s = fmt_addr(addr);
	send_pic_cmd(curr_pdp,s);

if( debug & pic_debug ) advise("listening for reply");
	//read_until_string(buf,curr_pdp->pd_sbp,"\n",CONSUME_MARKER);
	read_until_string(buf,curr_pdp->pd_sbp,"\r",CONSUME_MARKER);

	// now parse the string
	sscanf(buf,"%4x:  %4x",&ra,&rd);
//sprintf(ERROR_STRING,"string \"%s\", ra = 0x%04x, rd = 0x%04x",buf,ra,rd);
//advise(ERROR_STRING);

	WAIT_FOR_PROMPT(curr_pdp)

	return(rd);
} /* end read_pgm_mem() */

#define read_mem(addr) _read_mem(QSP_ARG  addr)

static int _read_mem(QSP_ARG_DECL  int addr)
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
	send_pic_cmd(curr_pdp,buf);
if( debug & pic_debug ) advise("listening for address prompt");
	expected_response(curr_pdp->pd_sbp,"Enter data address: ");
	s = fmt_addr(addr);
	send_pic_cmd(curr_pdp,s);

if( debug & pic_debug ) advise("listening for reply");
	//read_until_string(buf,curr_pdp->pd_sbp,"\n",CONSUME_MARKER);
	read_until_string(buf,curr_pdp->pd_sbp,"\r",CONSUME_MARKER);

	// now parse the string
	sscanf(buf,"%4x:  %4x",&ra,&rd);
//sprintf(ERROR_STRING,"string \"%s\", ra = 0x%04x, rd = 0x%04x",buf,ra,rd);
//advise(ERROR_STRING);

	WAIT_FOR_PROMPT(curr_pdp)

	return(rd);
} /* end read_mem() */

#define pic_echo(onoff) _pic_echo(QSP_ARG  onoff)

static void _pic_echo(QSP_ARG_DECL  int onoff)
{
	const char *s;
	char buf[4];

	sprintf(buf, "%s", pic_tbl[PIC_ECHO].pc_str);
	reset_buffer(curr_pdp->pd_sbp);
if( debug & pic_debug ){
sprintf(ERROR_STRING,"Sending \"%s\" to PIC",printable_string(buf));
advise(ERROR_STRING);
}
	send_pic_cmd(curr_pdp,buf);

if( debug & pic_debug ) advise("listening for yes/no prompt");
	expected_response(curr_pdp->pd_sbp,"Enter 1 to enable command echo, 0 to disable: ");
//advise("prompt seen, sending byte");
//show_buffer(curr_pdp->pd_sbp);

	if( onoff ) s = fmt_byte(1);
	else s = fmt_byte(0);

	send_pic_cmd(curr_pdp,s);

	if( onoff )
		pic_state.ps_flags |= PIC_ECHO_ON;
	else
		pic_state.ps_flags &= ~PIC_ECHO_ON;

	WAIT_FOR_PROMPT(curr_pdp)
} /* end pic_echo */

#define goto_pgm_mem(addr) _goto_pgm_mem(QSP_ARG  addr)

static void _goto_pgm_mem(QSP_ARG_DECL  int addr)
{
	const char *s;
	char buf[LLEN];

	sprintf(buf, "%s", pic_tbl[PIC_GOTO].pc_str);
	reset_buffer(curr_pdp->pd_sbp);
if( debug & pic_debug ){
sprintf(ERROR_STRING,"Sending \"%s\" to PIC",printable_string(buf));
advise(ERROR_STRING);
}
	send_pic_cmd(curr_pdp,buf);
if( debug & pic_debug ) advise("listening for address prompt");
	expected_response(curr_pdp->pd_sbp,"Enter program memory address: ");
	s = fmt_addr(addr);
	send_pic_cmd(curr_pdp,s);

if( debug & pic_debug ) advise("listening for reply");

	/* We might expect some printout here, such as a boot message...
	 * Therefore, we don't expect the prompt immediately, but read until we see it.
	 */
	INSURE_PROMPT
	read_until_string(buf,curr_pdp->pd_sbp,pic_prompt,CONSUME_MARKER);
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
		warn("number of locations to read must be positive");
		return;
	}

	CHECK_PIC

	while(n--){
		d=read_pgm_mem(addr);
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
		warn("number of locations to read must be positive");
		return;
	}
	if( (addr) < 0 ) return;


	CHECK_PIC

	while(n--){
		d=read_mem(addr);
		sprintf(msg_str,"D 0x%04x:  0x%04x",addr,d);
		prt_msg(msg_str);
		addr++;
	}
}

#define write_pgm_mem(addr, data) _write_pgm_mem(QSP_ARG  addr, data)

static void _write_pgm_mem(QSP_ARG_DECL  int addr, int data)
{
	const char *s;
	char buf[LLEN];

	sprintf(buf, "%s", pic_tbl[PIC_WR_PGM].pc_str);
	reset_buffer(curr_pdp->pd_sbp);
if( debug & pic_debug ){
sprintf(ERROR_STRING,"Sending \"%s\" to PIC",printable_string(buf));
advise(ERROR_STRING);
}
	send_pic_cmd(curr_pdp,buf);
if( debug & pic_debug ) advise("listening for address prompt");
	expected_response(curr_pdp->pd_sbp,"Enter program memory address: ");
	s = fmt_addr(addr);
	send_pic_cmd(curr_pdp,s);
	expected_response(curr_pdp->pd_sbp,"Enter program memory value: ");
	s = fmt_addr(data);
	send_pic_cmd(curr_pdp,s);

	WAIT_FOR_PROMPT(curr_pdp)
} /* end write_pgm_mem() */

#define write_mem(addr, data) _write_mem(QSP_ARG  addr, data)

static void _write_mem(QSP_ARG_DECL  int addr, int data)
{
	const char *s;
	char buf[LLEN];

	sprintf(buf, "%s", pic_tbl[PIC_WR_DATA].pc_str);
	reset_buffer(curr_pdp->pd_sbp);
if( debug & pic_debug ){
sprintf(ERROR_STRING,"Sending \"%s\" to PIC",printable_string(buf));
advise(ERROR_STRING);
}
	send_pic_cmd(curr_pdp,buf);
if( debug & pic_debug ) advise("listening for address prompt");
	expected_response(curr_pdp->pd_sbp,"Enter data address: ");
	s = fmt_addr(addr);
	send_pic_cmd(curr_pdp,s);
	expected_response(curr_pdp->pd_sbp,"Enter program memory value: ");
	s = fmt_addr(data);
	send_pic_cmd(curr_pdp,s);

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

	write_pgm_mem(addr,data);
}

static COMMAND_FUNC( do_write_mem )
{
	int addr;
	int data;

	addr=ASK_ADDR(DATA_ADDR_PMPT,MAX_DATA_ADDR);
	data = HOW_MANY("8 bit data value");		/* no range checking... */

	if( addr < 0 ) return;


	CHECK_PIC

	write_mem(addr,data);
}

static COMMAND_FUNC( do_goto )
{
	int addr;
	addr=ASK_ADDR(PGM_ADDR_PMPT,MAX_PGM_ADDR);
	if( addr < 0 ) return;

	CHECK_PIC

	goto_pgm_mem(addr);
}

#define open_pic_device(s) _open_pic_device(QSP_ARG  s)

static void _open_pic_device(QSP_ARG_DECL  const char *s)
{
	int fd;
	PIC_Device *pdp;

	pdp = pic_dev_of(s);
	if( pdp != NULL ){
		sprintf(ERROR_STRING,
	"open_pic_device:  pic device %s is already open",s);
		warn(ERROR_STRING);
		return;
	}
	if( n_pics_active >= MAX_PIC_DEVICES ){
		sprintf(ERROR_STRING,
			"Max number (%d) of PIC devices already open, won't attempt to open %s.",
			MAX_PIC_DEVICES,s);
		warn(ERROR_STRING);
		return;
	}

	if( (fd = open_serial_device(s)) < 0 ){ 
		sprintf(ERROR_STRING,"Unable to open pic device %s",s);
		warn(ERROR_STRING);
		return;
	}

	pdp = new_pic_dev(s);
	pdp->pd_fd=fd;
	pdp->pd_prompt_seen=0;

	ttyraw(fd);
	echooff(fd);			/* BUG should set all options here... */
	//set_baud(fd,B19200);
	set_baud(fd,B38400);

	/* set various other flags - not sure which are truly needed... */

	GETTERM;

	/* input options */
	tiobuf.c_iflag |= IGNPAR;	/* ignore parity */

	tiobuf.c_cflag &= ~PARENB;	/* no parity on output */

	tiobuf.c_oflag &= ~CRTSCTS;	/* disable hardware flow control */
	tiobuf.c_oflag |= CSTOPB;
	tiobuf.c_oflag |= HUPCL;

	tiobuf.c_lflag &= ~ECHOE;
	tiobuf.c_lflag &= ~ECHOK;
	tiobuf.c_lflag &= ~ECHOKE;
	tiobuf.c_lflag &= ~ECHOCTL;

	SETTERM;

	pdp->pd_sbp = getbuf(sizeof(*pdp->pd_sbp));
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

	pdp = pic_dev_of(s);
	if( pdp != NULL ){
		curr_pdp = pdp;
	} else {
		open_pic_device(s);
	}
}

#define get_hex_digit(sp) _get_hex_digit(QSP_ARG  sp)

static int _get_hex_digit(QSP_ARG_DECL  const char **sp)
{
	const char *s;
	int d;

	s = *sp;
	if( *s == 0 ){
		warn("get_hex_digit:  no more buffered chars!?");
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
			warn("bad alpha hex digit");
			d=(-1);
		}
	} else {
		warn("bad hex digit");
		d=(-1);
	}
	s++;
	*sp = s;
	return(d);
}
		
#define get_byte(sp) _get_byte(QSP_ARG  sp)

static int _get_byte(QSP_ARG_DECL  const char **sp)
{
	int n,c;

	n = get_hex_digit(sp);
	if( n < 0 ) {
		warn("get_byte:  problem with first digit");
		return(n);
	}
	c = n << 4;
	n = get_hex_digit(sp);
	if( n < 0 ) {
		warn("get_byte:  problem with second digit");
		return(n);
	}
	c += n;
	return(c);
}

#define get_word(sp) _get_word(QSP_ARG  sp)

static int _get_word(QSP_ARG_DECL  const char **sp)
{
	int lsb,msb;

	lsb=get_byte(sp);
	if( lsb < 0 ) {
		warn("get_word: problem with first byte");
		return(lsb);
	}
	msb=get_byte(sp);
	if( msb < 0 ) {
		warn("get_word: problem with second byte");
		return(msb);
	}
	return( (msb<<8) + lsb );
}

#define get_addr(sp) _get_addr(QSP_ARG  sp)

static int _get_addr(QSP_ARG_DECL  const char **sp)
{
	int lsb,msb;

	msb=get_byte(sp);
	if( msb < 0 ) {
		warn("get_addr: problem with first byte");
		return(msb);
	}
	lsb=get_byte(sp);
	if( lsb < 0 ) {
		warn("get_addr: problem with second byte");
		return(lsb);
	}
	return( (msb<<8) + lsb );
}

/* we can call this with write_data = 0 for a syntax check...
 */

#define scan_line(sp,what_to_do) _scan_line(QSP_ARG  sp,what_to_do)

static int _scan_line(QSP_ARG_DECL  const char **sp, int what_to_do )
{
	const char *s;
	int a,b,c,count;
	int checksum1,checksum2;
	int w;

	s = *sp;
	if( *s != ':' ){
		sprintf(ERROR_STRING,"hex file line \"%s\" does not begin with ':' !?",s);
		warn(ERROR_STRING);
		return(-1);
	}
	s++;
	*sp = s;
	count = get_byte(sp);
	if( count < 0 ){
		warn("scan_line:  problem with count");
		return(-1);
	}
	if( count & 1 ){
		sprintf(ERROR_STRING,"count (0x%02x) should be even for PIC",count);
		warn(ERROR_STRING);
		return(-1);
	}
	a = get_addr(sp);
	if( a < 0 ){
		warn("scan_line:  problem with address");
		return(-1);
	}
	if( a & 1 ){
		sprintf(ERROR_STRING,"Address (0x%04x) is odd, should be even for PIC",a);
		warn(ERROR_STRING);
		return(-1);
	}
	a /= 2;
	checksum1 = get_byte(sp);
	if( checksum1 < 0 ){
		warn("scan_line:  problem with first checksum byte");
		return(-1);
	}
	/* not sure what to do with checksum1... */
	c = count / 2;		/* load by words, not bytes */
	while(c--){
		w = get_word(sp);
		if( w < 0 ){
			warn("scan_line:  problem with data word");
			return(-1);
		}
		if( what_to_do == WRITE_DATA ){
			write_pgm_mem(a,w);
			if( verbose ){
				sprintf(ERROR_STRING,"P %04x %04x",a,w);
				advise(ERROR_STRING);
			}
			a++;
		} else if( what_to_do == VERIFY_DATA ){
			int d;
			d = read_pgm_mem(a);
			if( d != w ){
				sprintf(ERROR_STRING,"0x%04x:  0x%04x  --  0x%04x expected",a,d,w);
				warn(ERROR_STRING);
			}
			a++;
		}
	}
	checksum2 = get_byte(sp);	/* checksum */
	if( checksum2 < 0 ){
		warn("missing final checksum!?");
		return(-1);
	}

	s = *sp;
	b = strlen(s);
	if( b != 0 ){
		sprintf(ERROR_STRING,"scan_line:  %d extra characters at end of record!?",b);
		warn(ERROR_STRING);
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
		warn(ERROR_STRING);
		return(-1);
	}
	
	return(0);
}

#ifdef FOOBAR
static int old_load_hex_line(char *s)
{
	char *s2;

	s2=s;
	if( scan_line(&s2,CHECK_SYNTAX) < 0 ){
		warn("aborting hex load");
		return(-1);
	}
	s2=s;
	if( scan_line(&s2,WRITE_DATA) < 0 ){
		warn("error writing line from hex file");
		return(-1);
	}
	return(0);
}
#endif /* FOOBAR */

#define load_hex_line(s) _load_hex_line(QSP_ARG  s)

static int _load_hex_line(QSP_ARG_DECL  const char *s)
{
	const char *s2;
	int status;

	s2=s;
	if( (status=scan_line(&s2,CHECK_SYNTAX)) != 0 ){
		if( status < 0 ){
			warn("aborting hex load");
			return(-1);
		}
		return(0);	/* a line we choose not to send */
	}
	send_pic_cmd(curr_pdp,s);
	WAIT_FOR_PROMPT(curr_pdp)
	return(0);
} /* end load_hex_line */

#define verify_hex_line(s) _verify_hex_line(QSP_ARG  s)

static void _verify_hex_line(QSP_ARG_DECL  const char *s)
{
	int status;
	const char *s2;

	s2=s;
	if( (status=scan_line(&s2,VERIFY_DATA)) != 0 ){
		if( status < 0 )
			warn("error verifying hex data");
		else
			warn("unexpected status returned from scan_line!?");
	}
}

#define VALIDATE_CMD_CHAR(c)							\
										\
	if( c != 'L' && c != 'T' ){						\
		sprintf(ERROR_STRING,"CAUTIOUS:  cmd_char 0x%x should be 'L' (0x%x) or 'T' (0x%x)",	\
			c,'L','T');						\
		warn(ERROR_STRING);						\
		return;								\
	}


#define send_short_data(cmd_char,dp) _send_short_data(QSP_ARG  cmd_char,dp)

static void _send_short_data(QSP_ARG_DECL  int cmd_char, Data_Obj *dp)	/* send a single squirt */
{
	dimension_t n;
	u_short s1,s2,s3,s4,*sp;
	char buf[32];

#ifdef CAUTIOUS
	VALIDATE_CMD_CHAR(cmd_char)
#endif

	if( OBJ_COMPS(dp) != 4 ){
		sprintf(ERROR_STRING,"send_short_data:  object %s (%d) should have 4 components",
			OBJ_NAME(dp),OBJ_COMPS(dp));
		warn(ERROR_STRING);
		return;
	}
	/* we don't need to do this, but for now we will because we are lazy */
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"Sorry, object %s should be contiguous for send_short_data",OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}

	n = OBJ_N_MACH_ELTS(dp) / 4;
	sp = OBJ_DATA_PTR(dp);
	while(n--){
		s1 = *sp++;
		s2 = *sp++;
		s3 = *sp++;
		s4 = *sp++;
		sprintf(buf,"%c%04x%04x%04x%04x",cmd_char,s1,s2,s3,s4);
		send_pic_cmd(curr_pdp,buf);
	}
}

#define stream_long_data(pip) _stream_long_data(QSP_ARG  pip)

static void _stream_long_data(QSP_ARG_DECL  Proc_Info *pip)
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
	send_pic_cmd(pdp,buf);

	WAIT_FOR_PROMPT(pdp)

	lp = ptr;
	while(count--){
		l1 = *lp;
		l2 = *(lp+1);
		sprintf(buf,"%c%08lx%08lx",cmd_char,l1,l2);
		send_pic_cmd(pdp,buf);
		WAIT_FOR_PROMPT(pdp)
		lp += inc/sizeof(long);
	}
}

static void * pic_streamer(void *argp)
{
	Proc_Info *pip;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *qsp;
#endif // THREAD_SAFE_QUERY

	pip = (Proc_Info*) argp;

	pip->ppi_pid = getpid();

	/* tell the parent that we're ready, and wait for siblings */
	pip->ppi_flags |= PIC_READY_TO_GO;

	/* wait for host to signal that all threads are ready before we start */

	pip->ppi_flags |= PIC_STREAMING;

#ifdef THREAD_SAFE_QUERY
	qsp = pip->ppi_qsp;
#endif // THREAD_SAFE_QUERY
	stream_long_data(pip);

	pip->ppi_flags = PIC_EXITING;

	return(NULL);
}

#define start_pic_threads(dp, cmd_char) _start_pic_threads(QSP_ARG  dp, cmd_char)

static void _start_pic_threads(QSP_ARG_DECL  Data_Obj *dp, int cmd_char)
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

		ppi[i].ppi_inc = OBJ_COMPS(dp) * ELEMENT_SIZE(dp);
		ppi[i].ppi_count = OBJ_N_MACH_ELTS(dp) / OBJ_COMPS(dp) ;
		ppi[i].ppi_ptr = ((char *)OBJ_DATA_PTR(dp)) + i * 8;	/* each channel is 64 bits */
		ppi[i].ppi_cmd = cmd_char;
#ifdef THREAD_SAFE_QUERY
		ppi[i].ppi_qsp = THIS_QSP;
#endif // THREAD_SAFE_QUERY

		pthread_create(&pic_thr[i],&attr1,pic_streamer,&ppi[i]);
	}
}

#define wait_pic_threads() _wait_pic_threads(SINGLE_QSP_ARG)

static void _wait_pic_threads(SINGLE_QSP_ARG_DECL)
{
	int i;

	for(i=0;i<channels_per_pixel;i++){
		if( pthread_join(pic_thr[i],NULL) != 0 ){
			warn("wait_pic_threads:  error return from pthread_join()");
		}
	}
}

/* send data and return immediately - used with capture flow */

static COMMAND_FUNC( do_led64 )
{
	Data_Obj *dp;

	dp = pick_obj("64 bit data object");
	if( dp == NULL ) return;

	CHECK_PIC

	send_short_data('L',dp);		/* assume someone else has sync'd */

	/* need to issue a command to sync up the prompt! */
	WAIT_FOR_PROMPT(curr_pdp)		/* BUG?  need to pass pdp? */
}

static COMMAND_FUNC( do_timed64 )
{
	Data_Obj *dp;

	dp = pick_obj("64 bit data object");
	if( dp == NULL ) return;

	CHECK_PIC

	send_short_data('T',dp);		/* assume someone else has sync'd */
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
		warn("bad channel index");
		return;
	}
	if( data < 0 || data > 255 ){
		warn("bad data value");
		return;
	}

	CHECK_PIC

	send_pic_cmd(curr_pdp,"l");

	expected_response(curr_pdp->pd_sbp,"Enter channel number: ");
	sprintf(buf,"%02x",chan);
	send_pic_cmd(curr_pdp,buf);

	expected_response(curr_pdp->pd_sbp,"Enter led byte: ");
	sprintf(buf,"%02x",data);
	send_pic_cmd(curr_pdp,buf);

	WAIT_FOR_PROMPT(curr_pdp)
}

static COMMAND_FUNC( do_set_timer )
{
	int counth;
	char buf[4];

	//countl=HOW_MANY("counter_l");
	counth=HOW_MANY("timer count");
	if( counth < 0 || counth > 255 ){
		warn("count must be in the range 0-255");
		return;
	}

	CHECK_PIC

	send_pic_cmd(curr_pdp,"t");

	//expected_response(curr_pdp->pd_sbp,"Enter LED counterl: ");
	//sprintf(buf,"%02x",countl);
	//send_pic_cmd(curr_pdp,buf);

	expected_response(curr_pdp->pd_sbp,"Enter LED counterh: ");
	sprintf(buf,"%02x",counth);
	send_pic_cmd(curr_pdp,buf);

	WAIT_FOR_PROMPT(curr_pdp)
}

static COMMAND_FUNC( do_wait_even )
{
	CHECK_PIC
	send_pic_cmd(curr_pdp,"z");
	WAIT_FOR_PROMPT(curr_pdp)
}

static COMMAND_FUNC( do_wait_odd )
{
	CHECK_PIC
	send_pic_cmd(curr_pdp,"y");
	WAIT_FOR_PROMPT(curr_pdp)
}

#define stream_byte_data(cmd_char,dp) _stream_byte_data(QSP_ARG  cmd_char,dp)

static void _stream_byte_data(QSP_ARG_DECL  int cmd_char, Data_Obj *dp)
{
	dimension_t n;
	u_char c1;
	u_char *cp;
	char buf[32];

	if( OBJ_COMPS(dp) != 1 ){
		sprintf(ERROR_STRING,"stream_byte_data:  object %s (%d) should have 1 component",
			OBJ_NAME(dp),OBJ_COMPS(dp));
		warn(ERROR_STRING);
		return;
	}
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"Sorry, object %s should be contiguous for stream_byte_data",OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}
	n = OBJ_N_MACH_ELTS(dp) ;
	cp = OBJ_DATA_PTR(dp);
	while(n--){
		c1 = *cp++;
		sprintf(buf,"%c%02x",cmd_char,c1);
		send_pic_cmd(curr_pdp,buf);
		WAIT_FOR_PROMPT(curr_pdp)
	}
}


static void stream_data(QSP_ARG_DECL  int cmd_char)
{
	Data_Obj *dp;
	int bytes_per_pixel;

	dp = pick_obj("LED data vector");

#ifdef CAUTIOUS
	VALIDATE_CMD_CHAR(cmd_char)
#endif

	if( dp == NULL ){
		warn("stream_data:  invalid data vector");
		return;
	}
	/* make sure that each vector element has a multiple of 64 bits.
	 * The multiplier needs to be less than or equal to the number of active pic's.
	 */

	bytes_per_pixel = ELEMENT_SIZE(dp) * OBJ_COMPS(dp);
	if( bytes_per_pixel%8 != 0 ){
		sprintf(ERROR_STRING,"stream_data:  data vector %s has %d bytes per pixel.",
			OBJ_NAME(dp),bytes_per_pixel);
		warn(ERROR_STRING);
		sprintf(ERROR_STRING,"which is not an integral number of 64 bit channels.");
		advise(ERROR_STRING);
		return;
	}

	channels_per_pixel = bytes_per_pixel/8;

	if( channels_per_pixel > n_pics_active ){
		sprintf(ERROR_STRING,"stream_data:  vector %s has data for %d channels, but only %d devices are active",
			OBJ_NAME(dp),channels_per_pixel,n_pics_active);
		warn(ERROR_STRING);
		return;
	}

	/* If 1 channel, use current device.
	 * If n_channels == number of active devices, use all devices in order.
	 */
	if( channels_per_pixel != n_pics_active ){
		if( channels_per_pixel > 1 ){
			sprintf(ERROR_STRING,"stream_data:  vector %s has data for %d channels, will use the first %d of %d active devices",
				OBJ_NAME(dp),channels_per_pixel,channels_per_pixel,n_pics_active);
			advise(ERROR_STRING);
		} else {
			sprintf(ERROR_STRING,"stream_data:  using active device %s",curr_pdp->pd_name);
			advise(ERROR_STRING);
		}
	}

	/* we don't need to do this, but for now we will because we are lazy */
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"Sorry, object %s should be contiguous for stream_long_data",OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}

	/* create 1 thread per channel */
	start_pic_threads(dp,cmd_char);

	/* here we should wait for the threads to finish... */
	wait_pic_threads();
}

static void stream_bytes(QSP_ARG_DECL  int cmd_char)
{
	Data_Obj *dp;

	dp=pick_obj("byte data vector");
	if( dp == NULL ){
		warn("stream_bytes:  invalid data vector");
		return;
	}
	if( OBJ_MACH_PREC(dp) != PREC_UBY ){
		sprintf(ERROR_STRING,"stream_bytes:  data vector %s (%s) should have %s precision",
			OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)),NAME_FOR_PREC_CODE(PREC_UBY));
		warn(ERROR_STRING);
		return;
	}
	stream_byte_data(cmd_char,dp);
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
	fp=try_open(s,"r");
	if( !fp ) return;

	CHECK_PIC

	redir(fp,s);
	l = QLEVEL;
	while( (QLEVEL == l) && (problems==0) ){
		prt_msg_frag(".");
		if( ((++count) & 0x1f ) == 0 )
			prt_msg("");
		s=NAMEOF("hex file line");
		problems=load_hex_line(s);
		lookahead_til(l-1);
	}
	prt_msg("");
	/* fp is closed automatically when it is popped */
	if( QLEVEL == l ){	/* problems */
		pop_file();
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
	fp=try_open(s,"r");
	if( !fp ) return;

	CHECK_PIC

	redir(fp,s);
	l = QLEVEL;
	while( (QLEVEL == l) && (problems==0) ){
		prt_msg_frag(".");
		if( ((++count) & 0x1f ) == 0 )
			prt_msg("");
		s=NAMEOF("hex file line");
		verify_hex_line(s);
		lookahead_til(l-1);
	}
	prt_msg("");
	/* fp is closed automatically when it is popped */
	if( QLEVEL == l ){	/* problems */
		pop_file();
	}
}

static COMMAND_FUNC( do_need )
{
	const char *s;

	s=NAMEOF("expected firmware version");
	if( firmware_version == NULL )
		get_firmware_version();
	if( strcmp(s,firmware_version) ){
		sprintf(ERROR_STRING,"Expected firmware version %s, but version %s is running!?",
			s,firmware_version);
		error1(ERROR_STRING);
	}
}

static COMMAND_FUNC( do_pic_echo )
{
	int yn;

	yn = ASKIF("Should PIC echo commands");

	pic_echo(yn);
}

static COMMAND_FUNC( do_send_char )
{
	const char *s;

	s=NAMEOF("single character command");
	/* should we check that it really is a single char? */

	reset_buffer(curr_pdp->pd_sbp);
	send_pic_cmd(curr_pdp,s);
	WAIT_FOR_PROMPT(curr_pdp)
}

static COMMAND_FUNC( do_pic_set_pmpt )
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
	n=n_serial_chars(curr_pdp->pd_sbp->sb_fd);
	if( n > 0 )
		replenish_buffer(curr_pdp->pd_sbp,n);

	/* now flush 'em */
	reset_buffer(curr_pdp->pd_sbp);

	//read_until_string(buf,curr_pdp->pd_sbp,"\r",CONSUME_MARKER);
}

static COMMAND_FUNC( do_sync )
{
	CHECK_PIC

	do_flush(SINGLE_QSP_ARG);
	send_pic_cmd(curr_pdp,"");	/* null command */

	WAIT_FOR_PROMPT(curr_pdp)
}

static COMMAND_FUNC( do_pic_crnl )
{
	pic_nlcr_echo = ASKIF("pic echoes two newline chars");
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(pic_main_menu,s,f,h)

MENU_BEGIN(pic_main)

ADD_CMD( device,	do_select_device,	select pic device			)
ADD_CMD( version,	do_report_version,	report firmware version number		)
ADD_CMD( read_pgm,	do_read_pgm_mem,	read program memory			)
ADD_CMD( write_pgm,	do_write_pgm_mem,	write program memory			)
ADD_CMD( read_mem,	do_read_mem,		read memory				)
ADD_CMD( write_mem,	do_write_mem,		write memory				)
ADD_CMD( load,		do_load,		load hex file				)
ADD_CMD( verify,	do_verify,		verify correct loading of hex file	)
ADD_CMD( goto,		do_goto,		begin execution at specified location	)
ADD_CMD( led8,		do_led8,		specify a byte of LED data		)
ADD_CMD( stream_led,	do_stream_led,		stream data to LED controllers		)
ADD_CMD( stream_timer,	do_stream_timed,	stream 64 bit data with fixed timed pulses	)
ADD_CMD( stream_pwm,	do_stream_pwm,		stream variable pulse widths with fixed LED pattern	)
ADD_CMD( led64,		do_led64,		specify 64 bits of LED data		)
ADD_CMD( timed64,	do_timed64,		specify 64 bit data with timed pulses	)
ADD_CMD( timer_counts,	do_set_timer,		set LED timer counts			)
ADD_CMD( wait_even,	do_wait_even,		wait for the end of an even field	)
ADD_CMD( wait_odd,	do_wait_odd,		wait for the end of an odd field	)
ADD_CMD( wait_prompt,	do_wait_pmpt,		wait for PIC prompt			)
ADD_CMD( need_version,	do_need,		specify desired firmware version	)
ADD_CMD( flush,		do_flush,		flush pending input			)
ADD_CMD( sync,		do_sync,		sync program with pic prompt		)
ADD_CMD( pic_echo,	do_pic_echo,		enable/disable command echo		)
ADD_CMD( set_prompt,	do_pic_set_pmpt,	specify prompt used by PIC interpreter	)
ADD_CMD( pic_crnl_echo,	do_pic_crnl,		specify whether firmware echoes nl-cr for cr	)
ADD_CMD( send_char_cmd,	do_send_char,		send a single-character command		)

MENU_END(pic_main)

/* This was /dev/ttyS0 on fourier, but we insist on using a symlink
 * /dev/pic to make this portable to other systems which might use
 * a different port.
 */

#define PIC_TTY_DEV	"/dev/pic"

#define check_pic_tbl() _check_pic_tbl(SINGLE_QSP_ARG)

static void _check_pic_tbl(SINGLE_QSP_ARG_DECL)		// make sure it's in the right order
{
	int i;

	for(i=0;i<N_PIC_CMDS;i++){
		if( pic_tbl[i].pc_code != i ){
			sprintf(ERROR_STRING,"Command code at tbl location %d is %d, expected %d",i,
				pic_tbl[i].pc_code,i);
			error1(ERROR_STRING);
		}
	}
}

COMMAND_FUNC( do_pic_menu )
{
	if( pic_debug == 0 ) {
		check_pic_tbl();
		pic_debug = add_debug_module("pic");

		/* Should we automatically open /dev/pic? */

		/* BUG we need some way of insuring that the pic is in a known state */
		pic_state.ps_flags |= PIC_ECHO_ON;
		/* BUG we should only do this when we want to stream data... */
		//rt_sched(1);
	}

	CHECK_AND_PUSH_MENU(pic_main);
}

