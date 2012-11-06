
#include "quip_config.h"

char VersionId_interpreter_tty_flags[] = QUIP_VERSION_STRING;

#ifdef TTY_CTL

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* isatty() */
#endif

#ifdef HAVE_SYS_FILIO_H
#include <sys/filio.h>
#endif

#ifdef HAVE_SYS_TERMIOS_H			/* SUN */
#include <sys/termios.h>
#define GETTERM			ioctl(fd,TCGETS,&tiobuf)
#define SETTERM			ioctl(fd,TCSETS,&tiobuf)
#endif

#ifdef HAVE_TERMIOS_H
#include <termios.h>
#define GETTERM			tcgetattr(fd,&tiobuf)
#define SETTERM			tcsetattr(fd,TCSADRAIN,&tiobuf)
#endif

/* sys/ioctl.h used w/ slackware, but causes error w/ red hat!? */
#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>		/* FIONREAD */
#endif 

#include "ttyctl.h"
#include "serial.h"
#include "query.h"

static struct termios tiobuf;

static void show_baud(void);

static Termio_Option if_opts[]={
{ "ignbrk",	IGNBRK,	"ignore input break",		""			},
{ "brkint",	BRKINT,	"generate SIGINT on BREAK",	"read BREAK as \0"	},
{ "ignpar",	IGNPAR,	"ignore framing/parity errors",	""			},
{ "parmrk",	PARMRK,	"mark framing/parity errors",	"error chars read \0"	},
{ "inpck",	INPCK,	"enable input parity",		"disable input parity"	},
{ "istrip",	ISTRIP,	"strip 8th bit",		""			},
{ "inlcr",	INLCR,	"translate NL to CR on input",	""			},
{ "igncr",	IGNCR,	"ignore CR on input",		""			},
{ "icrnl",	ICRNL,	"translate CR to NL on input",	""			},
{ "ixon",	IXON,	"enable XON/XOFF flow control on output",	""	},
{ "ixoff",	IXOFF,	"enable XON/XOFF flow control on input",	""	},
#ifdef IUCLC
{ "iuclc",	IUCLC,	"map upper to lower",		""			},
#endif /* IUCLC */
{ "ixany",	IXANY,	"enable any character to restart output",	""	},
{ "imaxbel",	IMAXBEL,"ring bell when input queue is full",		""	},
{ NULL,	0,	NULL,				NULL			}
};

static Termio_Option of_opts[]={
{ "opost",	OPOST,	"enable implementation-defined output processing",	""},
#ifdef OLCUC
{ "olcuc",	OLCUC,	"map lowercase chars to uppercase on output",	""},
{ "ocrnl",	OCRNL,	"map CR to NL on output",	""			},
{ "onocr",	ONOCR,	"don't output CR at column zero",	""		},
{ "onlret",	ONLRET,	"don't output CR",		""			},
{ "ofill",	OFILL,	"send fill chars for delay",	""			},
{ "ofdel",	OFDEL,	"fill char is DEL", 		"fill char is NUL"	},
#endif /* OLCUC */
{ "onlcr",	ONLCR,	"map NL to CR-NL on output",	""			},
{ NULL,	0,	NULL, NULL						}
};

static Termio_Option cf_opts[]={
{ "parenb",	PARENB,	"enable parity",		""			},
{ "parodd",	PARODD,	"odd parity",			"even parity"		},
{ "hupcl",	HUPCL,	"lower modem ctl lines after close",	""		},
{ "cstopb",	CSTOPB,	"two stop bits",		"one stop bit"		},
{ "cread",	CREAD,	"enable receiver",		"disable receiver"	},
{ "clocal",	CLOCAL,	"ignore modem control lines",	""			},
#ifdef CRTSCTS	// was #ifndef SGI
{ "crtscts",	CRTSCTS,"use rts/cts flow control",	""			},
#endif /* CRTSCTS */
{ NULL,	0,	NULL, NULL						}
};

static Termio_Option lf_opts[]={
{ "isig",	ISIG,	"generate signals from special chars",	""	},
{ "icanon",	ICANON,	"canonical input mode",	""			},
{ "iexten",	IEXTEN,	"enable implementation-defined input processing",	""},
{ "echo",	ECHO,	"echo input chars", "don't echo input chars"		},
{ "echoe",	ECHOE,	"erase previous char/word",	""			},
{ "echok",	ECHOK,	"erase line on KILL char",	""			},
{ "echonl",	ECHONL,	"echo NL",	""					},
{ "noflsh",	NOFLSH,	"disable flushinq queues during signals",	""	},
#ifdef XCASE
{ "xcase",	XCASE,	"uppercase-only terminal",	""			},
#endif /* XCASE */
{ "tostop",	TOSTOP,	"send SIGTTOU to background process writing to ctl tty",	""},
{ "echoprt",	ECHOPRT,"print chars as they are erased",	""		},
{ "echoctl",	ECHOCTL,"echo ctl chars printable",	""			},
{ "echoke",	ECHOKE, "echo KILL by erasing chars",	""			},
{ "flusho",	FLUSHO,	"output is being flushed",	""			},
{ "pendin",	PENDIN,	"all chars reprinted when next char read",	""	},
{ NULL,	0,	NULL, NULL						}
};

/* local prototypes */
static Termio_Option *search_to_tbl(Termio_Option *tbl,const char *name);

static Termio_Option *search_to_tbl(Termio_Option *tbl,const char* name)
{
	while( tbl->to_name != NULL ){
		if( !strcmp(tbl->to_name,name) )
			return(tbl);
		tbl++;
	}
	return(NO_TERM_OPT);
}

static Termio_Option *find_tty_flag( const char *flagname )
{
	Termio_Option *top;
	top = search_to_tbl(lf_opts,flagname);
	if( top != NO_TERM_OPT ) return top;
	top = search_to_tbl(cf_opts,flagname);
	if( top != NO_TERM_OPT ) return top;
	top = search_to_tbl(if_opts,flagname);
	if( top != NO_TERM_OPT ) return top;
	top = search_to_tbl(of_opts,flagname);
	if( top != NO_TERM_OPT ) return top;

	return top;
}

int get_flag_value( QSP_ARG_DECL   const char *flagname )
{
	Termio_Option *top;

	top = find_tty_flag(flagname);
	if( top == NO_TERM_OPT ){
		return ASKIF("dummy flag value");
	} else {
		return ASKIF(top->to_enastr);
	}
}

void set_tty_flag(const char *flagname,int fd,int value)
{
	Termio_Option *top;
	tcflag_t *flag_ptr;

	top = search_to_tbl(lf_opts,flagname);
	if( top != NO_TERM_OPT ) flag_ptr = (&tiobuf.c_lflag);

	if( top == NO_TERM_OPT ){
		top = search_to_tbl(cf_opts,flagname);
		if( top != NO_TERM_OPT ) flag_ptr = (&tiobuf.c_cflag);
	}

	if( top == NO_TERM_OPT ){
		top = search_to_tbl(if_opts,flagname);
		if( top != NO_TERM_OPT ) flag_ptr = (&tiobuf.c_iflag);
	}

	if( top == NO_TERM_OPT ){
		top = search_to_tbl(of_opts,flagname);
		if( top != NO_TERM_OPT ) flag_ptr = (&tiobuf.c_oflag);
	}

	if( top == NO_TERM_OPT ){
		sprintf(DEFAULT_ERROR_STRING,"Unrecognized flag %s",flagname);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	if( fd < 0 ) {
		NWARN("no serial device open");
		return;
	}

	GETTERM;

	if( value )
		*flag_ptr |=   top->to_bit;
	else
		*flag_ptr &= ~ top->to_bit;

	SETTERM;
}

void show_all(int fd)
{
	GETTERM;

	/* print baud rate also! */
	show_baud();

	switch( tiobuf.c_cflag & CSIZE ){
		case CS5: prt_msg("5 data bits"); break;
		case CS6: prt_msg("6 data bits"); break;
		case CS7: prt_msg("7 data bits"); break;
		case CS8: prt_msg("8 data bits"); break;
		default: prt_msg("wacky data bits"); break;
	}

	show_term_flags(tiobuf.c_cflag,cf_opts);
	show_term_flags(tiobuf.c_iflag,if_opts);
	show_term_flags(tiobuf.c_oflag,of_opts);
	show_term_flags(tiobuf.c_lflag,lf_opts);
}

void dump_all(int fd)
{
	GETTERM;

	/*
	switch( tiobuf.c_cflag & CSIZE ){
		case CS5: prt_msg("5 data bits"); break;
		case CS6: prt_msg("6 data bits"); break;
		case CS7: prt_msg("7 data bits"); break;
		case CS8: prt_msg("8 data bits"); break;
		default: prt_msg("wacky data bits"); break;
	}
	*/

	dump_term_flags(tiobuf.c_cflag,cf_opts);
	dump_term_flags(tiobuf.c_iflag,if_opts);
	dump_term_flags(tiobuf.c_oflag,of_opts);
	dump_term_flags(tiobuf.c_lflag,lf_opts);
}

static void show_baud(void)
{
	int b;

#ifdef CBAUD
	switch( tiobuf.c_cflag & CBAUD ){
		case B50: b=50; break;
		case B75: b=75; break;
		case B110: b=110; break;
		case B134: b=134; break;
		case B150: b=150; break;
		case B200: b=200; break;
		case B300: b=300; break;
		case B600: b=600; break;
		case B1200: b=1200; break;
		case B2400: b=2400; break;
		case B4800: b=4800; break;
		case B9600: b=9600; break;
		case B19200: b=19200; break;
		case B38400: b=38400; break;
#ifdef B57600
		case B57600: b=57600; break;
#endif
#ifdef B115200
		case B115200: b=115200; break;
#endif
#ifdef B230400
		case B230400: b=230400; break;
#endif
#ifdef B460800
		case B460800: b=460800; break;
#endif
#ifdef B500000
		case B500000: b=500000; break;
#endif
#ifdef B576000
		case B576000: b=576000; break;
#endif
#ifdef B921600
		case B921600: b=921600; break;
#endif
#ifdef B1000000
		case B1000000: b=1000000; break;
#endif
#ifdef B1500000
		case B1500000: b=1500000; break;
#endif
#ifdef B2000000
		case B2000000: b=2000000; break;
#endif
#ifdef B2500000
		case B2500000: b=2500000; break;
#endif
#ifdef B3000000
		case B3000000: b=3000000; break;
#endif
#ifdef B3500000
		case B3500000: b=3500000; break;
#endif
#ifdef B4000000
		case B4000000: b=4000000; break;
#endif

#ifdef CAUTIOUS
		default:
			b=0;	/* elim warning */
			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  show_baud:  undefined baud rate constant 0x%x!?",tiobuf.c_cflag & CBAUD );
			NWARN(DEFAULT_ERROR_STRING);
			sprintf(DEFAULT_ERROR_STRING,"\tB9600 = 0x%x",B9600);
			advise(DEFAULT_ERROR_STRING);
			break;
#endif /* CAUTIOUS */

	}

#else /* ! CBAUD (OSX and what else???) */
	b = tiobuf.c_ispeed ;
#endif /* ! CBAUD */

	sprintf(msg_str,"%d baud",b);
	prt_msg(msg_str);
}

#endif /* TTY_CTL */

