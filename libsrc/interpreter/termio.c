#include "quip_config.h"

#ifdef TTY_CTL

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* exit(), added for OSX */
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* isatty() */
#endif

#ifdef HAVE_SYS_FILIO_H
#include <sys/filio.h>
#endif

#ifdef HAVE_TERMIOS_H
#include <termios.h>
#define GETTERM			tcgetattr(fd,&tiobuf)
#define SETTERM			tcsetattr(fd,TCSADRAIN,&tiobuf)
#else /* ! HAVE_TERMIOS_H */
#ifdef HAVE_SYS_TERMIOS_H
#include <sys/termios.h>
#define GETTERM			ioctl(fd,TCGETS,&tiobuf)
#define SETTERM			ioctl(fd,TCSETS,&tiobuf)
#endif /* HAVE_SYS_TERMIOS_H */
#endif /* ! HAVE_TERMIOS_H */



#ifdef FOOBAR
/* SGI? */
#define GETTERM			ioctl(fd,TCGETA,&tiobuf)
#define SETTERM			ioctl(fd,TCSETA,&tiobuf)
#endif /* FOOBAR */

#ifdef HAVE_SYS_IOCTL_H
/* sys/ioctl.h used w/ slackware, but causes error w/ red hat!? */
#include <sys/ioctl.h>		/* FIONREAD */
#endif

#include "quip_prot.h"
#include "ttyctl.h"

static struct termios tiobuf;


void _show_term_flags(QSP_ARG_DECL  u_long flag,Termio_Option *tbl)
{
	while(tbl->to_name != NULL ){
		if( flag & tbl->to_bit ){
			sprintf(msg_str,"  %s",tbl->to_name);
		} else {
			sprintf(msg_str," -%s",tbl->to_name);
		}
		prt_msg_frag(msg_str);

		tbl++;
	}
	prt_msg("");
}

#define SETFLAG_CMD_WORD	"setflag"

void _dump_term_flags(QSP_ARG_DECL  u_long flag, Termio_Option *tbl)
{
	while(tbl->to_name != NULL ){
		if( flag & tbl->to_bit ){
			sprintf(msg_str,"%s %s yes",SETFLAG_CMD_WORD,tbl->to_name);
		} else {
			sprintf(msg_str,"%s %s no",SETFLAG_CMD_WORD,tbl->to_name);
		}
		prt_msg(msg_str);

		tbl++;
	}
}

void set_ndata(int fd,int n)
{
	GETTERM;

	switch(n){
		case 8:
			tiobuf.c_cflag &= ~CSIZE;
			tiobuf.c_cflag |= CS8;
			break;
		case 7:
			tiobuf.c_cflag &= ~CSIZE;
			tiobuf.c_cflag |= CS7;
			break;
		default:
			sprintf(DEFAULT_ERROR_STRING,
				"bad number of bits requested:  %d",n);
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}

	SETTERM;
}

void set_parity(int fd,int flag,int odd)
{
	GETTERM;

	if( flag ){
		tiobuf.c_cflag |= PARENB;
		if( odd )
			tiobuf.c_cflag |= PARODD;
		else
			tiobuf.c_cflag &= ~PARODD;
	} else {
		tiobuf.c_cflag &= ~PARENB;
		tiobuf.c_iflag |= IGNPAR;
		tiobuf.c_iflag &= ~PARMRK;
	}

	SETTERM;
}

/* We used to have 017 here in place of CBAUD...
 * Also note functions cfgetispeed etc ??
 * MacOSX doesn't have CBAUD...
 */

void set_baud(int fd,int rate)
{
	GETTERM;

#ifdef CFSETSPEED

	if( cfsetspeed(&tiobuf,rate) < 0 ){
		tell_sys_error("cfsetspeed");
		NWARN("Error setting baud rate!?");
	}

#else // ! CFSETSPEED

#ifdef CBAUD
	tiobuf.c_cflag &= ~CBAUD;
	tiobuf.c_cflag |= (rate & CBAUD);
#else
	/* OSX and what else? */
	tiobuf.c_ispeed = rate;
	tiobuf.c_ospeed = rate;
#endif
#endif // ! CFSETSPEED

	SETTERM;
}

/* no newline */

void tty_nonl(int fd)
{
	GETTERM;

	tiobuf.c_lflag &= ~(ECHONL);
	tiobuf.c_oflag &= ~ONLCR;

	SETTERM;
}

void ttyraw(int fd)		/** RAW mode */
{
	GETTERM;

	tiobuf.c_lflag &= ~(ICANON|ECHO|ISIG);

	tiobuf.c_iflag &= ~ICRNL;

	tiobuf.c_oflag &= ~(ONLCR | OCRNL);

	tiobuf.c_cc[VMIN] = 1;

	/* time shouldn't matter when min=1,
	 * but fails on onyx with time=0
	 */
	tiobuf.c_cc[VTIME] = 1;

	SETTERM;
}

void ttycbrk(int fd)		/** CBREAK mode */
{
	GETTERM;

	tiobuf.c_lflag &= ~(ICANON|ECHO);
	tiobuf.c_lflag |= ISIG;

	tiobuf.c_iflag |= ICRNL;

	tiobuf.c_cc[VMIN] = 1;		/* min # chars to return from read */

	/* time shouldn't matter when min=1,
	 * but fails on onyx with time=0
	 */
	tiobuf.c_cc[VTIME] = 1;		/* shouldn't matter when min=1 */

	SETTERM;
}

void ttycook(int fd)		/** COOKED mode */
{
	GETTERM;

	tiobuf.c_lflag |= ICANON|ECHO|ISIG;

	tiobuf.c_oflag |= ONLCR;
	tiobuf.c_iflag |= ICRNL;


	tiobuf.c_cc[VMIN] = 1;
	tiobuf.c_cc[VTIME] = 0;

	SETTERM;
}

void echoon(int fd)		/** enable echo */
{
	GETTERM;

	tiobuf.c_lflag |= ECHO|ECHOE;

	SETTERM;
}

void echooff(int fd)		/** disable echo */
{
	GETTERM;

	tiobuf.c_lflag &= ~(ECHO);

	SETTERM;

}

void ttynorm(int fd)		/** stty -cbreak -raw echo */
{
	GETTERM;

	tiobuf.c_lflag |= ISIG|ICANON|ECHO|ECHOE;
	tiobuf.c_oflag |= ONLCR;
	tiobuf.c_iflag |= ICRNL;


	tiobuf.c_cc[VMIN] = 1;
	tiobuf.c_cc[VTIME] = 0;


	SETTERM;
}



void waitq(int fd)
{
#ifdef	TIOCOUTQ
	int n=1;
	if( ! isatty( fd ) ) return;
	while(n!=0){
		ioctl( fd, TIOCOUTQ, &n );
	}
#endif
}

int keyhit(int fd)
{
	long l;

	if( ioctl(fd,FIONREAD,&l) == -1 ){
		_tell_sys_error(DEFAULT_QSP_ARG  "keyhit");
		exit(1);
		return -1;
	}
	return( (int) l );
}

int get_erase_chr(int fd)
{
	int c;

	if( GETTERM == -1 )
		return(-1);

#ifdef SGI
	c=tiobuf.c_cc[VERASE+1];
#else
	c=tiobuf.c_cc[VERASE];
#endif

	/*
	show_ctl_chars();
	*/

	return(c);
}

int get_kill_chr(int fd)
{
	int c;

	if( GETTERM == -1 )
		return(-1);

#ifdef SGI
	c=tiobuf.c_cc[VKILL+1];
#else
	c=tiobuf.c_cc[VKILL];
#endif

	/*
	show_ctl_chars();
	*/

	return(c);
}
#endif /* TTY_CTL */

