#include "quip_config.h"

char VersionId_ports_xmitrecv[] = QUIP_VERSION_STRING;

#include "quip_config.h"

#include <stdio.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* write() */
#endif
#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
/* these are needed for select(2): */
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#include "nports.h"
#include "getbuf.h"
#include "debug.h"
#include "submenus.h"	/* call_event_funcs */

/* local prototypes */
static long read_expected_bytes(QSP_ARG_DECL  Port *mpp,void *buf,u_long n_want_bytes);
static int wait_for_data(QSP_ARG_DECL  Port *mpp);


/*
 * Read port *mpp repeatedly as needed until n_want_bytes
 * chars have been placed into buf.
 * BUG!!! read_expected_bytes is supposed to return the number actually
 * read (or -1 for error), but it doesn't really return the number
 * read; in fact, if the requested number all come in at the first
 * attempt, 0 is returned!?  This bug is probably harmless since
 * the return value is only sign checked...
 */

static long read_expected_bytes(QSP_ARG_DECL  Port *mpp,void *buf,u_long n_want_bytes)
{
	int n;
	int have=0;

	/* BUG questionable pc cast */
	while( (n=read_port(QSP_ARG  mpp,&(((char *)buf)[have]),n_want_bytes)) != (int)n_want_bytes ){
		if( n==(-1) ){		/* why would this happen? */
			/* BUG - Is it possible that an error inside
			 * of read_port would cause the port to be
			 * deallocated?  In that case, we might be
			 * in trouble referencing the former port's name.
			 */
			sprintf(error_string,
				"read_expected_bytes: problem reading port \"%s\"",
				mpp->mp_name);
			WARN(error_string);
			return(-1);
		} else if( n==0 ){
			return(-1); /* EOF */
		}
		have+=n;
		n_want_bytes-=n;
	}
	return(have);
}

/*
 * Get an long integer from the port.
 * Returns the value of the integer, or -1.
 *
 * This seems like a BUG, what if -1 is the value to be transmitted!?
 */

int32_t get_port_int32(QSP_ARG_DECL  Port *mpp)
{
	int32_t word;
	int32_t net_data;
	int32_t n;

	n=read_expected_bytes(QSP_ARG  mpp,&net_data,sizeof(int32_t));

	if( n < 0 ) return(BAD_PORT_LONG);

	/* now decode the length */
	/* what shall the byte order be?? */
	/* how about msb first? */

	word = ntohl(net_data);

	return(word);
}

/*
 * Receive a text packet.  This function is called from recv_data(),
 * after that function has seen a P_TEXT code.
 * The text packets consist of a word giving the length of the string
 * in bytes, followed by the text.
 *
 * A new buffer is allocated for the text.  recv_text() returns
 * the address of this buffer, or a null pointer if some error occurs,
 * usually EOF.
 */

const char *recv_text(QSP_ARG_DECL  Port *mpp)
{
	char *buf;
	long len;		/* number requested */
	long nread;		/* number read on last gulp */

	len=get_port_int32(QSP_ARG  mpp);
	if( len<=0 ) return(NULL);

	buf=(char *)getbuf(len);
	if( buf==NULL ) mem_err("recv_text");

	nread = read_expected_bytes(QSP_ARG  mpp,buf,len);
	if( nread < 0 ) return(NULL);
	return(buf);
}

/*
 * Transmit a word.  Companion routine to get_port_word().
 */

int put_port_int32(Port *mpp,int32_t  wrd)
{
	int32_t net_data;

	net_data = htonl(wrd);

	if( write_port(mpp,&net_data,sizeof(net_data)) != sizeof(net_data) ){
		/* BUG
		 * One reason that this fails is that the listening
		 * program has been killed (^C).  (on pc anyway...)
		 * We should probably do something more active here
		 * to straighten things out.
		 */
		return(-1);
	}
	else return(0);
}

/*
 * Transmit a string.  Send the code word P_TEXT, followed by a word
 * giving the length of the string, followed by the string itself.
 */
/* flag ignored, but there to be compatible w/ xmit_data() */

void xmit_text(QSP_ARG_DECL  Port *mpp,const void *text,int flag)
{
	uint32_t code;

	code=P_TEXT;
	if( put_port_int32(mpp,code) == (-1) ){
		WARN("xmit_text:  error sending code");
		return;
	}

	code=strlen((char *)text)+1;
	if( put_port_int32(mpp,code) == (-1) ){
		WARN("xmit_text:  error sending string length");
		return;
	}

	if( (u_char)write_port(mpp,text,code) != code )
		WARN("xmit_text:  error sending text data");
}

int write_port(Port *mpp,const void *buf,u_long  n)
{
	int n2;

	n2=write(mpp->mp_sock,buf,(int)n);
	if( n2 < 0 )
		tell_sys_error("(write_port) write:");
	return(n2);
}


/*
 * Try to read n characters from a port.
 * Returns number of characters read, 0 for EOF, -1 for error.
 */

int read_port(QSP_ARG_DECL  Port *mpp,void *buf,u_long  n)
{
	int n2;

#ifdef HAVE_SELECT
	if( wait_for_data(QSP_ARG  mpp) < 0 ){
		return(-1);
	}
#endif /* HAVE_SELECT */

	if( (n2=read(mpp->mp_sock,buf,(int)n)) < 0 ){
		if( errno != EINTR ){
			tell_sys_error("(read_port) read");
			WARN("error reading stream packet");
			return(-1);
		} else if( errno == 0 ){		/* EOF */
advise("should this happen?????");
			goto eof_encountered;
		} else {
			WARN("read_port:  unexpected read error");
		}
	} else if( n2 == 0 ){		/* EOF */
eof_encountered:
		if( verbose ){
			sprintf(error_string,"EOF encountered on port \"%s\"",
				mpp->mp_name);
			advise(error_string);
		}
		/* we used to close the linked port here,
		 * but now we take care of that in reset_port()
		 */
		return(0);
	}
#ifdef CAUTIOUS
if( n2 == 0 ) WARN("CAUTIOUS:  read_port() returning 0 normally!?");
#endif /* CAUTIOUS */
	return(n2);
}

/* Shutdown a port and then release it's resources */

void close_port(QSP_ARG_DECL  Port *mpp)
{
	close(mpp->mp_sock);

	if( mpp->mp_o_sock != (-1) )
		close(mpp->mp_o_sock);

	givbuf((char *)mpp->mp_addrp);
	mpp->mp_addrp = NULL;
	delport(QSP_ARG  mpp);
}



#ifdef HAVE_SELECT

/*
 * Use select() to wait for readable data on this port.
 * Returns 0 when there is readable data, -1 for some other error.
 */

static int wait_for_data(QSP_ARG_DECL  Port *mpp)
{
	fd_set rd_fds, null_fds;
	struct timeval time_out;
	int v;
	int width;

	FD_ZERO(&rd_fds);
	FD_ZERO(&null_fds);
	FD_SET( mpp->mp_sock, &rd_fds );
	width=getdtablesize();

	time_out.tv_sec=0;
	time_out.tv_usec=100000;	/* 10 Hz poll */

	while( (v=select( width, &rd_fds, &null_fds, &null_fds,
		&time_out )) <= 0 ){
		/* Nothing to do now */

		if( v < 0 ){		/* select error */
			if( errno != EINTR ){
				tell_sys_error("(wait_for_data) select");
				return(-1);
			}
			/* else EINTR, which is normal */

			/* we used to call a "recovery" function here!? */
			/* but it had been commented out for a long time... */
		}

		call_event_funcs(SINGLE_QSP_ARG);
		FD_SET( mpp->mp_sock, &rd_fds );

		/* We want to sleep here, but maybe not for too long.
		 * If we don't sleep, then the daemon burns up cpu when it's
		 * inactive, but if we always sleep here, then response is slow
		 * when we have a lot of transactions.
		 *
		 * Let's try this approach:  when we have data,
		 * we reset sleeptime to something small.  Then,
		 * every time we have nothing, we double it, up until
		 * an upper limit of half a second.
		 */
		if( mpp->mp_sleeptime < 0 )
			mpp->mp_sleeptime=MIN_MP_SLEEPTIME;


		if( mpp->mp_sleeptime > 0 ){
			usleep(mpp->mp_sleeptime);
			mpp->mp_sleeptime *= 1.5;
			if( mpp->mp_sleeptime > MAX_MP_SLEEPTIME )
				mpp->mp_sleeptime = MAX_MP_SLEEPTIME;
		}
	}
	if( !FD_ISSET(mpp->mp_sock,&rd_fds) ){
		WARN("wait_for_data:  shouldn't happen");
		return(-1);
	}
	/* When we have data, reset the sleeptime to something small */
	if( mpp->mp_sleeptime > 0 )
		mpp->mp_sleeptime=MIN_MP_SLEEPTIME;
	return(0);
}

#endif /* HAVE_SELECT */

