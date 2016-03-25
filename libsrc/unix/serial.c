#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* system() */
#endif

#ifdef HAVE_SYS_FILE_H
#include <sys/file.h>	/* flock() */
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_SYS_TERMIOS_H
#include <sys/termios.h>	/* to compile on airy??? */
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#ifdef HAVE_SYS_FILIO_H
#include <sys/filio.h>
#endif

#include "quip_prot.h"
#include "serial.h"
#include "history.h"

#ifdef NOT_SURE
#include "stamps.h"
#include "debug.h"
#include "query.h"		/* ttys_are_interactive */
#include "submenus.h"
#endif /* NOT_SURE */

#ifdef TTY_CTL
#include "ttyctl.h"
#include "my_stty.h"
#endif /* TTY_CTL */

static Serial_Port *default_spp=NO_SERIAL_PORT;

#define CHECK_DEFAULT_SERIAL_PORT(rname,return_flag)					\
											\
	if( default_spp == NO_SERIAL_PORT ){						\
		sprintf(ERROR_STRING,"%s:  no default serial port selected",rname);	\
		WARN(ERROR_STRING);							\
		if( return_flag ) return;						\
	}


//ITEM_INTERFACE_DECLARATIONS_STATIC(Serial_Port,serial_port)
static Item_Type *serial_port_itp=NO_ITEM_TYPE;
static ITEM_INIT_FUNC(Serial_Port,serial_port)
static ITEM_NEW_FUNC(Serial_Port,serial_port)
static ITEM_CHECK_FUNC(Serial_Port,serial_port)
//static ITEM_GET_FUNC(Serial_Port,serial_port)
static ITEM_LIST_FUNC(Serial_Port,serial_port)
//static ITEM_ENUM_FUNC(Serial_Port,serial_port)
static ITEM_DEL_FUNC(Serial_Port,serial_port)
static ITEM_PICK_FUNC(Serial_Port,serial_port)

static COMMAND_FUNC( do_list_serial_ports );

#define PICK_SERIAL_PORT(pmpt)		pick_serial_port(QSP_ARG  pmpt)

// BUG - not thread safe!?
static ssize_t n_raw_chars=0;
static int n_lin_chars=0;

static char linbuf[LLEN];

int open_serial_device(QSP_ARG_DECL  const char * s)
{
	int fd;
	Serial_Port *spp;

	spp=serial_port_of(QSP_ARG  s);
	if( spp != NO_SERIAL_PORT ){
		sprintf(ERROR_STRING,"Serial port %s is already open",s);
		WARN(ERROR_STRING);
		return(spp->sp_fd);
	}

fprintf(stderr,"opening %s...\n",s);
	fd=open(s,O_RDWR);
	if( fd < 0 ){
		sprintf(ERROR_STRING,"error opening tty file \"%s\"",s);
		WARN(ERROR_STRING);
		return(fd);
	}
fprintf(stderr,"%s opened...\n",s);


	/* BUG - this block used to be ifdef LINUX, need test for flock in configure.ac */
	if( flock(fd,LOCK_EX|LOCK_NB) < 0 ){
		if( errno == EWOULDBLOCK ){
			sprintf(ERROR_STRING,"Unable to obtain exclusive lock on tty file %s",s);
			WARN(ERROR_STRING);
			advise("Make sure the port is not in use by another process");
		} else {
			perror("flock");
			sprintf(ERROR_STRING,"unable to get exclusive lock on serial device %s",s);
			WARN(ERROR_STRING);
		}
		return(-1);
	}



	spp = new_serial_port(QSP_ARG  s);
	if( spp == NO_SERIAL_PORT ){
		sprintf(ERROR_STRING,"Unable to create serial port structure for %s",s);
		ERROR1(ERROR_STRING);
		return -1;	// NOTREACHED - silence static analyzer
	}

	spp->sp_fd = fd;

#ifdef TTY_CTL
fprintf(stderr,"Putting %s into raw mode...\n",s);
	ttyraw(fd);
#endif /* TTY_CTL */


	default_spp = spp;

	return(fd);
}

static COMMAND_FUNC( do_open )
{
	const char *s;

	s=NAMEOF("device file");

	if( open_serial_device(QSP_ARG  s) < 0 )
		WARN("Error opening serial device");
}

void send_serial(QSP_ARG_DECL  int fd,const u_char *chardata,int n)
{
	ssize_t n_written;

#ifdef CAUTIOUS
	if( fd < 0 ){
		sprintf(ERROR_STRING,
			"CAUTIOUS:  send_serial passed invalid file descriptor (%d)",fd);
		WARN(ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */

try_again:
	n_written = write(fd,chardata,n);
	if( n_written < 0 ){
		perror("write");
		WARN("send_serial:  error writing string");
	} else if( n_written != n ){
		sprintf(ERROR_STRING,
			"send_serial:  %d chars requested, %zd actually written",
			n,n_written);
		WARN(ERROR_STRING);
		n -= n_written;
		chardata += n_written;
		goto try_again;
	}
}

#define COPY_TAIL(n)							\
				sto=str;				\
				sfr=str+(n);				\
				do {					\
					*sto ++ = *sfr++;		\
				} while( *(sfr-1) );


static void expand_escape_sequences(u_char *str,int buflen)
{
	u_char *sto,*sfr;

	while( *str ){
		if( *str == '\\' ){
			str++;
			if( *str == 0 ){
				NWARN("expand_escape_sequences:  single backslash at end of string");
				return;
			} else if( *str == 'r' ){
				*(str-1) = '\r';
				COPY_TAIL(1)
			} else if( *str == 'n' ){
				*(str-1) = '\n';
				COPY_TAIL(1)
			} else if( *str == 'b' ){
				*(str-1) = '\b';
				COPY_TAIL(1)
			} else if( *str == 't' ){
				*(str-1) = '\t';
				COPY_TAIL(1)
			} else if( *str == 'f' ){
				*(str-1) = '\f';
				COPY_TAIL(1)
			} else if( *str == '\\' ){
				*(str-1) = '\\';
				COPY_TAIL(1)
			} else if( isdigit(*str) ){
				int v;
				v=*str-'0';
				str++;
				if( *str == 0 ){
					NWARN("expand_excape_sequences:  missing digit chars");
					return;
				} else if( !isdigit(*str) ){
					NWARN("expand_excape_sequences:  missing second digit");
					return;
				}
				v *= 8;
				v += *str - '0';
				str++;
				if( *str == 0 ){
					NWARN("expand_excape_sequences:  missing digit char");
					return;
				} else if( !isdigit(*str) ){
					NWARN("expand_excape_sequences:  missing third digit");
					return;
				}
				v *= 8;
				v += *str - '0';
				str -= 2;
				*(str-1)= (u_char) v;
				COPY_TAIL(3)
			}
		}
		str++;
	}
}
					
static COMMAND_FUNC( do_send )
{
	const char *s;
	u_char str[LLEN];

	s=NAMEOF("text to send");
	strcpy((char *)str,s);
	/* Do we want to append a newline??? */
	/* strcat(str,"\n"); */

	CHECK_DEFAULT_SERIAL_PORT("do_send",1);

	expand_escape_sequences(str,LLEN);

	send_serial(QSP_ARG  default_spp->sp_fd,str,(int)strlen((char *)str));
}

static int get_hex_digit(QSP_ARG_DECL  int c)
{
	if( isdigit(c) ) return(c-'0');
	if( c>='a' && c<= 'f' ) return(10+c-'a');
	if( c>='A' && c<= 'F' ) return(10+c-'A');

	sprintf(ERROR_STRING,"get_hex_digit:  Illegal hex digit '%c' (0x%x)",c,c);
	NWARN(ERROR_STRING);
	return(-1);
}

static COMMAND_FUNC( do_hex )
{
	const char *s;

	s=NAMEOF("hex digits");

	CHECK_DEFAULT_SERIAL_PORT("do_hex",1);

	send_hex(QSP_ARG  default_spp->sp_fd,(const u_char *)s);
}

/* assume s points to a string of two or more hex digits...
 * return the value of the byte corresponding to the first two digits.
 */

int hex_byte(QSP_ARG_DECL  const u_char *s)
{
	int d1,d2;

	/* get first digit */

	d1=get_hex_digit(QSP_ARG  *s++);
	if( d1 < 0 ) return(-1);

	if( *s==0 ){
		NWARN("missing second hex digit");
		return(-1);
	}

	d2=get_hex_digit(QSP_ARG  *s++);
	if( d2 < 0 ) return(-1);

	return( (d1<<4) + d2 );
}

void send_hex(QSP_ARG_DECL  int fd,const u_char *s)
{
	u_char *to,str[LLEN];
	int nc=0;

	to=str;
	while( *s ){
		*to++ = (u_char) hex_byte(QSP_ARG  s);
		s += 2;
		nc++;
	}
	*to=0;

	send_serial(QSP_ARG  fd,str,nc);
}

int n_serial_chars(QSP_ARG_DECL  int fd)
{
#ifdef FIONREAD
	int n;

	if( ioctl(fd,FIONREAD,&n) < 0 ){
		perror("n_serial_chars:  ioctl FIONREAD");
		n=0;
		ERROR1("unable to monitor serial input");
	}
	return(n);
#else // ! FIONREAD
    WARN("n_serial_chars:  No FIONREAD ioctl!?");
    return 1;
#endif // ! FIONREAD
}

static int int_nreadable(QSP_ARG_DECL  Serial_Port *spp)
{
	int n;

	n = n_serial_chars(QSP_ARG  spp->sp_fd);
	return(n);
}

static int get_nreadable(QSP_ARG_DECL  Serial_Port *spp)
{
	int n;
	char s[32];

	/* find out how many chars are there */
	n = n_serial_chars(QSP_ARG  spp->sp_fd);
	if( verbose ){
		sprintf(ERROR_STRING,"%d readable chars on serial port %s",
							n,spp->sp_name);
		advise(ERROR_STRING);
	}
	sprintf(s,"%d",n);
	assign_reserved_var(QSP_ARG  "n_readable",s);
	return(0);
}

static COMMAND_FUNC( do_get_nreadable )
{
	CHECK_DEFAULT_SERIAL_PORT("do_get_nreadable",1);

	get_nreadable(QSP_ARG  default_spp);
}

/* It's the user's responsibility to make sure that max_want is not
 * larger than the size of the buffer.
 * If max_want <= 0, then we just read whatever is available.
 */

ssize_t recv_somex(QSP_ARG_DECL  int fd,u_char *buf,int bufsize, int max_want)
{
	int n;

	if( bufsize <= max_want ){
		sprintf(ERROR_STRING,
"recv_somex:  bufsize (%d) must be at least 1 greater than max_want (%d)",
			bufsize,max_want);
		WARN(ERROR_STRING);
		abort();
	}

	/* find out how many chars are there */
	n = n_serial_chars(QSP_ARG  fd);

	if( verbose ){
		sprintf(ERROR_STRING,"%d readable chars on serial port",n);
		advise(ERROR_STRING);
	}

	if( n <= 0 ) return(n);
	if( max_want > 0 && max_want<n ){
		n=max_want;
	}

	if( n > (bufsize-1) ) {
		/*
		sprintf(ERROR_STRING,"Available chars (%d) exceed serial buffer size (%d)",n,bufsize);
		advise(ERROR_STRING);
		*/
		n=(bufsize-1);
	}

	if( (n_raw_chars=read(fd,buf,n)) != n ) WARN("error reading");

	buf[n_raw_chars]=0;
/*
{
int i;
for(i=0;i<n_raw_chars;i++)
buf[i] &= 0x7f;
}
*/

	/*
	if( verbose ){
		sprintf(ERROR_STRING,"String read:  \"%s\"",buf);
		advise(ERROR_STRING);
	}
	*/
	return(n_raw_chars);
}


static char * recv_line(QSP_ARG_DECL  Serial_Port *spp)
{
	char *lin_ptr;
	int eol_seen=0;


	lin_ptr = linbuf;
	n_lin_chars=0;

	/* if there's no input already, wait for some */

	while(1){
		int i,j;

		while(n_raw_chars<=0){
			if( recv_somex(QSP_ARG  spp->sp_fd,spp->sp_rawbuf,
							RAWBUF_SIZE,0) == 0 )
				sleep(1);
		}

		i=0;
		while( (i<n_raw_chars) &&
			spp->sp_rawbuf[i]!='\n' &&
			spp->sp_rawbuf[i]!='\r' ){

			if( n_lin_chars >= LLEN ){
				WARN("line buffer overflow");

				sprintf(ERROR_STRING,
			"i = %d, n_raw_chars = %zd, n_lin_chars = %d",
				i,n_raw_chars,n_lin_chars);

				advise(ERROR_STRING);
			} else {
				*lin_ptr++ = spp->sp_rawbuf[i];
				n_lin_chars ++;
			}
			i++;
		}
		if( i==n_raw_chars ) i--;

		if( spp->sp_rawbuf[i]=='\n' || spp->sp_rawbuf[i]=='\r' )
			eol_seen=1;

		/* if this is a prompt,
		 * there won't be a newline at the end,
		 * so we look for the common prompt terminator "> "
		 */

		if( spp->sp_rawbuf[i]==' ' ){
			if( i>=1 && spp->sp_rawbuf[i-1]=='>' )
				eol_seen=1;
		}

		/* We are either at a line break,
		 * or the end of the raw buffer
		 */

		/* shift rawbuf */
		i++;
		j=0;
		while( spp->sp_rawbuf[i] )
			spp->sp_rawbuf[j++] = spp->sp_rawbuf[i++];
		n_raw_chars = j;

		/* copy the NULL */
		spp->sp_rawbuf[n_raw_chars] = 0;

		*lin_ptr=0;

		if( eol_seen ){
			assign_reserved_var(QSP_ARG  "last_line",linbuf);
			return(linbuf);
		}
	}
	/* NOTREACHED */
}

static COMMAND_FUNC( do_insist )
{
	char *s;

	CHECK_DEFAULT_SERIAL_PORT("do_insist",0);

	if( default_spp != NULL )
		s=recv_line(QSP_ARG  default_spp);
	else	s=NULL;

	if( s != NULL )
		assign_reserved_var(QSP_ARG  "serial_response",s);
	else
		assign_reserved_var(QSP_ARG  "serial_response","(null)");
}

static COMMAND_FUNC( do_raw_recv )
{
	CHECK_DEFAULT_SERIAL_PORT("do_raw_recv",1);

	if( recv_somex(QSP_ARG  default_spp->sp_fd,default_spp->sp_rawbuf,RAWBUF_SIZE,0) == 0 && verbose ){
		sprintf(ERROR_STRING,"no characters to receive on serial port %s",default_spp->sp_name);
		advise(ERROR_STRING);
	}
}

static COMMAND_FUNC( do_recv )
{
	ssize_t n;

	CHECK_DEFAULT_SERIAL_PORT("do_recv",0);

	if( default_spp != NULL )
		n=recv_somex(QSP_ARG  default_spp->sp_fd,default_spp->sp_rawbuf,
								RAWBUF_SIZE,0);
	else 	n=0;

	if( n > 0 )
		assign_reserved_var(QSP_ARG  "serial_response",
					(char *)default_spp->sp_rawbuf);
	else
		assign_reserved_var(QSP_ARG  "serial_response","(null)");
}

static COMMAND_FUNC( do_stty )
{
	CHECK_DEFAULT_SERIAL_PORT("do_stty",0);

#ifdef TTY_CTL
	if( default_spp != NULL )
		set_stty_fd(default_spp->sp_fd);
#endif // TTY_CTL
	do_stty_menu(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_wtfor )
{
	const char *wait_str;
	char *s;

	wait_str=NAMEOF("text string to wait for");
	s=NULL;

	CHECK_DEFAULT_SERIAL_PORT("do_wtfor",1);

	while(s==NULL){
		char *l;

		l=recv_line(QSP_ARG  default_spp);

		/* now look for a match */
		s = strstr(l,wait_str);	/* does wait_str occur in s? */
	}
}

static COMMAND_FUNC( dump_serial_chars )
{
	CHECK_DEFAULT_SERIAL_PORT("dump_serial_chars",1);

	dump_char_buf(QSP_ARG  default_spp->sp_rawbuf);
}

void set_raw_len(unsigned char *s)
{
	n_raw_chars = strlen((char *)s);
}

void dump_char_buf(QSP_ARG_DECL  unsigned char *buf)
{
	int i;

	for(i=0;i<n_raw_chars;i++){
		if( isprint(buf[i]) ){
			if( isspace(buf[i]) ){
				if( buf[i] == '\n' )
					sprintf(msg_str,
				"\t0x%x\t0%o\t\\n",buf[i],buf[i]);
				else if( buf[i] == '\r' )
					sprintf(msg_str,
				"\t0x%x\t0%o\t\\r",buf[i],buf[i]);
				else if( buf[i] == ' ' /* space */ )
					sprintf(msg_str, "\t0x%x\t0%o",buf[i],buf[i]);
				else
					sprintf(msg_str,
				"\t0x%x\t0%o\t???",buf[i],buf[i]);
			} else {
				sprintf(msg_str,"\t0x%x\t0%o\t%c",buf[i],buf[i],buf[i]);
			}
		} else
			sprintf(msg_str,"\t0x%x\t0%o",buf[i],buf[i]);
		prt_msg(msg_str);
	}
}

static void interp_file(QSP_ARG_DECL  const char *filename)
{
	FILE *fp;

	if( !strcmp(filename,"-") ){
		redir(QSP_ARG stdin,"-");
	} else {
		fp=TRY_OPEN( filename,"r");
		if( fp ){
			redir(QSP_ARG fp,filename);
		}
	}
}

/* interp_file() will open the port using stdio, so we make sure that it is not open here...
 */

static COMMAND_FUNC( do_tty_redir )
{
	const char *s;
	char cmd_str[LLEN];
	Serial_Port *spp;
	int status;

	s = NAMEOF("serial port for input redirection");

	spp = serial_port_of(QSP_ARG  s);
	if( spp != NO_SERIAL_PORT ) {
		sprintf(ERROR_STRING,"Serial port %s is already open, close before calling redir",spp->sp_name);
		WARN(ERROR_STRING);
		return;
	}

	/* BUG should confirm that file exists and is a tty */

	// Why do we put in cooked mode???
	// Why do we use system instead of our own stty facility?
	// Maybe this was written before the internal stty utilities existed?
	sprintf(cmd_str,"stty cooked < %s",s);
	status=system(cmd_str);
	if( status < 0 )
		WARN("Failed to reset serial line!?");

	/* ttys_are_interactive=0; */		/* assume a machine is connected */
	SET_QS_FLAG_BITS(THIS_QSP,QS_INTERACTIVE_TTYS);

	interp_file(QSP_ARG s);
	push_top_menu(SINGLE_QSP_ARG);
}

static void close_serial_device(SINGLE_QSP_ARG_DECL)
{
	CHECK_DEFAULT_SERIAL_PORT("close_serial_device",1);

	if( default_spp->sp_fd < 0 ){
		WARN("no serial port open, can't close");
		return;
	}

	if( close(default_spp->sp_fd) < 0 ){
		tell_sys_error("close");
		sprintf(ERROR_STRING,"error closing serial device %s",default_spp->sp_name);
		WARN(ERROR_STRING);
	}
	del_serial_port(QSP_ARG  default_spp);
	rls_str(default_spp->sp_name);
	default_spp = NO_SERIAL_PORT;
}

static COMMAND_FUNC( do_serial_info )
{
	CHECK_DEFAULT_SERIAL_PORT("do_serial_info",0);

	if( default_spp != NULL )
		sprintf(msg_str,"Current serial device is %s",default_spp->sp_name);
	else	sprintf(msg_str,"No current serial device.");

	prt_msg(msg_str);
}

static COMMAND_FUNC( do_select_serial )
{
	Serial_Port *spp;

	spp = PICK_SERIAL_PORT("default port for serial operations"); 
	if( spp == NO_SERIAL_PORT ) return;

	default_spp = spp;
}

Serial_Port *default_serial_port()
{
	return(default_spp);
}

static COMMAND_FUNC( do_connect )
{
	Serial_Port *spp1,*spp2;
	int n1,n2;
	ssize_t nr;

	spp1 = PICK_SERIAL_PORT("first port"); 
	spp2 = PICK_SERIAL_PORT("second port"); 
	if( spp1 == NO_SERIAL_PORT || spp2 == NO_SERIAL_PORT ){
		return;
	}
	if( spp1 == spp2 ){
		WARN("the two serial ports must be different");
		return;
	}

	while(1){
		n1 = int_nreadable(QSP_ARG  spp1);
		if( n1 > 0 ){
			if( (nr=recv_somex(QSP_ARG  spp1->sp_fd,spp1->sp_rawbuf,RAWBUF_SIZE,0)) == 0 ){
				WARN("Oops - no chars received on port 1, expected some!?");
			} else {
				if( nr!=n1 ){
					sprintf(ERROR_STRING,"Expected %d chars on %s, received %zd!?",
							n1,spp1->sp_name,nr);
					advise(ERROR_STRING);
				}
				send_serial(QSP_ARG  spp2->sp_fd,spp1->sp_rawbuf,(int)nr);
sprintf(ERROR_STRING,"%zd chars send from %s to %s",nr,spp1->sp_name,spp2->sp_name);
advise(ERROR_STRING);
dump_char_buf(QSP_ARG  spp1->sp_rawbuf);
			}
		}
		n2 = int_nreadable(QSP_ARG  spp2);
		if( n2 > 0 ){
			if( (nr=recv_somex(QSP_ARG  spp2->sp_fd,spp2->sp_rawbuf,RAWBUF_SIZE,0)) == 0 ){
				WARN("Oops - no chars received on port 2, expected some!?");
			} else {
				if( nr!=n2 ){
					sprintf(ERROR_STRING,"Expected %d chars on %s, received %zd!?",
							n2,spp2->sp_name,nr);
					advise(ERROR_STRING);
				}
				send_serial(QSP_ARG  spp1->sp_fd,spp2->sp_rawbuf,(int)nr);
sprintf(ERROR_STRING,"%zd chars send from %s to %s",nr,spp2->sp_name,spp1->sp_name);
advise(ERROR_STRING);
dump_char_buf(QSP_ARG  spp2->sp_rawbuf);
			}
		}
	}
}

/* do_tty_term - connect program console to a different serial port?
 */

static COMMAND_FUNC( do_tty_term )
{
	Serial_Port *spp;
#ifdef TTY_CTL
	FILE *console_input_fp;		/* for the user's typing */
	FILE *console_output_fp;	/* print here to display to user */
#endif // TTY_CTL

	spp = PICK_SERIAL_PORT("");
	if( spp == NO_SERIAL_PORT ) return;

#ifdef TTY_CTL

#ifdef HISTORY
	if( _tty_out == NULL ){
		console_output_fp = TRY_HARD("/dev/tty","w");
	} else
		console_output_fp = _tty_out;
#else // ! HISTORY
	console_output_fp = stdout;
#endif // ! HISTORY

	console_input_fp = tfile(SINGLE_QSP_ARG);

	/* we might like raw, but want to catch ^C etc. */
	/* BUT we don't want to map CR to NL... */
	ttyraw(fileno(console_input_fp));
	/*
	ttycbrk(fileno(console_input_fp));
	echooff(fileno(console_input_fp));
	set_tty_flag("icrnl",fileno(console_input_fp),0);
	*/
	/* don't map cr to nl */

	ttyraw(fileno(console_output_fp));
	/*
	ttycbrk(fileno(console_output_fp));
	echooff(fileno(console_output_fp));
	*/

	ttyraw(spp->sp_fd);

	/* Now loop, checking for chars either on the input or on the serial line */
	while(1){
		int n_to_print;

		n_to_print = n_serial_chars(QSP_ARG  spp->sp_fd);

		if( n_to_print == 0 && !keyboard_hit(QSP_ARG  console_input_fp) )
			usleep(1000);

		while( n_to_print > 0 ){
			u_char *buf;
			int i;

			if( recv_somex(QSP_ARG  spp->sp_fd,spp->sp_rawbuf,RAWBUF_SIZE,0) == 0 ){
				ERROR1("expected chars but none!?");
			}
			buf=spp->sp_rawbuf;
			for(i=0;i<n_raw_chars;i++){
if( verbose ) {
if( isalnum(buf[i]) || ispunct(buf[i]) )
sprintf(ERROR_STRING,"output: 0x%02x\t\t%c",buf[i],buf[i]);
else
sprintf(ERROR_STRING,"output: 0x%02x",buf[i]);
advise(ERROR_STRING);
}
				fputc(buf[i],console_output_fp);
			}
			fflush(console_output_fp);
			n_to_print = n_serial_chars(QSP_ARG  spp->sp_fd);
		}
		/* now check for the users typing... */
		while( keyboard_hit(QSP_ARG  console_input_fp) ){
			u_char buf[4];
			int c;

			c = getc(console_input_fp);

			/* BUG should test against the user's personal interrupt char... */
			if( c == 3 ){		/* ^C */
				nice_exit(QSP_ARG  0);
			}

			/* BUG make sure not EOF */
			buf[0]=c;
			/* send to the serial port */
			send_serial(QSP_ARG  spp->sp_fd,buf,1);
		}
	}
#else // ! TTY_CTL
	WARN("Sorry, build with TTY_CTL to connect console to a serial port.");
#endif // ! TTY_CTL
}

static COMMAND_FUNC( do_list_serial_ports )
{
	list_serial_ports(SINGLE_QSP_ARG);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(serial_menu,s,f,h)

// \044 is ascii octal escape for dollar sign '$'
// putting the dollar sign there causes warnings in Xcode
//  "Lexical or Preprocessor issue:  '$' in identifier"

MENU_BEGIN( serial )
ADD_CMD( open,		do_open,		open device )
ADD_CMD( close,		close_serial_device,	close device )
ADD_CMD( list,		do_list_serial_ports,	list all open serial ports )
ADD_CMD( select,	do_select_serial,	select default serial port )
ADD_CMD( info,		do_serial_info,
					report name of current default serial device )
ADD_CMD( send,		do_send,		send text )
ADD_CMD( hex,		do_hex,			send hex codes )
ADD_CMD( recv,		do_recv,
					check for input and store in \044serial_response )
ADD_CMD( rawrecv,	do_raw_recv,		read raw input and store internally )
ADD_CMD( wait_for,	do_wtfor,		read input until string received )
ADD_CMD( insist,	do_insist,		wait for complete line of input text )
ADD_CMD( stty,		do_stty,		hardware control submenu )
ADD_CMD( check,		do_get_nreadable,	set \044n_readable to # readable chars )
ADD_CMD( dump,		dump_serial_chars,	dump unprintable chars to screen )
ADD_CMD( redir,		do_tty_redir,		redirect input to tty file )
ADD_CMD( term,		do_tty_term,		connect console to serial port )
ADD_CMD( stamps,	do_stamp_menu,		submenu for reading char w/ timestamps )
ADD_CMD( connect,	do_connect,
			establish bidirectional connection between two open ports )
MENU_END( serial )

COMMAND_FUNC( do_ser_menu )
{
	PUSH_MENU(serial);
}


#ifdef FOOBAR

#include "quip_config.h"

#undef _WINDLL /* Need stdin to be defined */

#include <string.h>

#include "query.h"
#include "filerd.h"
#include "debug.h"


void filerd(SINGLE_QSP_ARG_DECL)
{
	const char *s;

	s=nameof(QSP_ARG "input file");
	interp_file(QSP_ARG s);
}

void copycmd(SINGLE_QSP_ARG_DECL)
{
	FILE *fp;

	fp=TRYNICE( nameof(QSP_ARG  "transcript file"), "w" );
	if( fp ) {
		if(dupout(QSP_ARG fp)==(-1))
			fclose(fp);
	}
}

#endif /* FOOBAR */
