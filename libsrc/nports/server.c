#include "quip_config.h"

char VersionId_ports_server[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#ifdef HAVE_NETDB_H
#include <netdb.h>
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
#include "typedefs.h"
#include "uio.h"
#include "submenus.h"	/* call_event_funcs */

static int msg_sock;
static struct sockaddr from;

static socklen_t from_len = sizeof(struct sockaddr);


static void null_reset(void);
static void (*reset_func)(void)=null_reset;

void set_reset_func( void (*func)(void) )
{
	reset_func = func;
}

/* Return the port number actually used, or -1 in case of error */

int open_server_port(QSP_ARG_DECL  const char *name,int  port_no)
{
	Port *mpp;
	socklen_t length;
	int on=1;

	if ( (port_no < 2001) || (port_no > 6999) ) {
		WARN("Illegal port number");
		advise("Use 2001-6999");
		advise("Check /etc/services for other conflicts");
		return(-1);
	}

	mpp=new_port(QSP_ARG  name);
	if( mpp==NO_PORT ){
		if( verbose ){
			sprintf(error_string,"open_server_port %s %d failed to create port struct",
				name,port_no);
			WARN(error_string);
		}
		return(-1);
	}

	mpp->mp_o_sock=socket(AF_INET,SOCK_STREAM,0);
#ifdef FOOBAR
	// pc-nfs???
	mpp->mp_o_sock=socket(PF_INET,SOCK_STREAM,0);
#endif /* FOOBAR */

	mpp->mp_sock=(-1);
	mpp->mp_sleeptime=(-1);
		
	if( mpp->mp_o_sock<0 ){
		tell_sys_error("open_server_port (socket)");
		WARN("error opening stream socket");
		delport(QSP_ARG  mpp);
		return(-1);
	}

	setsockopt(mpp->mp_o_sock,SOL_SOCKET,SO_REUSEADDR,
		(char *)&on,sizeof(on));

	mpp->mp_addrp =
		(struct sockaddr_in *) getbuf( sizeof(*(mpp->mp_addrp) ) );
	if( mpp->mp_addrp == NULL ) mem_err("open_server_port");

	mpp->mp_flags = 0;
	mpp->mp_flags |= PORT_SERVER;
	mpp->mp_pp = NO_PORT;
	mpp->mp_portnum = port_no;

	mpp->mp_addrp->sin_family = AF_INET;
	mpp->mp_addrp->sin_addr.s_addr = INADDR_ANY;
	length=sizeof(*(mpp->mp_addrp));
	mpp->mp_addrp->sin_port = htons( port_no );
	if( bind(mpp->mp_o_sock,(struct sockaddr *)(mpp->mp_addrp), length) ){
		tell_sys_error("bind");
		goto cleanup;
	}

	if( getsockname(mpp->mp_o_sock,
		(struct sockaddr *)mpp->mp_addrp,&length) ){
		WARN("open_server_port:  error getting socket name");
		goto cleanup;
	}
	/* We used to make sure that the port number
	 * was the one we requested, but we no longer
	 * insist upon that...
	 */

	if( port_listen(mpp) < 0 )
		goto cleanup;

	return(port_no);

cleanup:
	givbuf((char *)mpp->mp_addrp);
	mpp->mp_addrp = NULL;
	delport(QSP_ARG  mpp);
	return(-1);
}

/* Put the server port into the listening state */

int port_listen(Port *mpp)
{
#ifdef DEBUG
	if( debug & debug_ports ) advise("port_listen:  listening for a request");
#endif /* DEBUG */

	if( ! IS_SERVER(mpp) ){
		NWARN(DEFAULT_ERROR_STRING);
		sprintf(DEFAULT_ERROR_STRING,"port_listen:  %s is not a server port",mpp->mp_name);
		return(-1);
	}

	/* first close old connection */
	if( mpp->mp_sock != (-1) ){
		sprintf(DEFAULT_ERROR_STRING,"closing old connection on port %s (sock = 0x%x)",
			mpp->mp_name,mpp->mp_sock);
		advise(DEFAULT_ERROR_STRING);
		close(mpp->mp_sock);
	}

	if( listen(mpp->mp_o_sock,PORT_BACKLOG) != 0 ){
		tell_sys_error("listen");
		return(-1);
	}
	return(0);
}

int get_connection(QSP_ARG_DECL  Port *mpp)
{
#ifdef HAVE_SELECT
	fd_set rd_fds, null_fds;
	struct timeval time_out;
	int v;

	int width;

	/* hang until someone requests a hookup */
sprintf(error_string,"\nWaiting for a new connection request on port %d...\n",
mpp->mp_portnum);
advise(error_string);
	if( mpp->mp_sleeptime <= 0 )
		mpp->mp_sleeptime = MIN_MP_SLEEPTIME;

	/*
	 * we use select to hang instead of accept,
	 * since the SunView notifier has a special
	 * select() which allows window op's to happen
	 */

	FD_ZERO(&rd_fds);
	FD_ZERO(&null_fds);
	FD_SET( mpp->mp_o_sock, &rd_fds );
	width=getdtablesize();
	time_out.tv_sec=0;
	time_out.tv_usec=100000;	/* 10 Hz poll */

	/* Wait for a connection */

	while( (v=select( width, &rd_fds, &null_fds, &null_fds,
		&time_out )) <= 0 )
#ifdef FOOBAR
	while( (v=select( FD_SETSIZE, &rd_fds, &null_fds, &null_fds,
		&time_out )) <= 0 )
#endif /* FOOBAR */
	{
		/* Nothing yet... */

		if( v < 0 ){		/* select error */
			if( errno != EINTR ){
				tell_sys_error("(port_listen) select");
				return(-1);
			}
		}
		FD_SET( mpp->mp_o_sock, &rd_fds );
		call_event_funcs(SINGLE_QSP_ARG);
		if( mpp->mp_sleeptime < 0 )
			usleep(250000);		/* Check again in 250 msec */
		else if( mpp->mp_sleeptime > 0 )
			usleep( mpp->mp_sleeptime );
	}

	if( !FD_ISSET(mpp->mp_o_sock,&rd_fds) ){
		WARN("port_listen:  shouldn't happen");
		return(-1);
	}
#endif /* HAVE_SELECT */

//advise("port_listen:  select detected something...");

	while( (msg_sock=accept(mpp->mp_o_sock,&from,&from_len)) == -1 ){
		if ( errno == EWOULDBLOCK) {
			sleep (1); /* Try accept again in 1 second */
			continue;
		}
		if( errno != EINTR ){
			tell_sys_error("(port_listen) accept");
			return(-1);
		}
		call_event_funcs(SINGLE_QSP_ARG);

		if( mpp->mp_sleeptime < 0 )
			usleep(250000);		/* Check again in 250 msec */
		else if( mpp->mp_sleeptime > 0 )
			usleep( mpp->mp_sleeptime );
	}

	mpp->mp_sock=msg_sock;

	/* Now we set a flag to show that this socket is connected */
	mpp->mp_flags |= PORT_CONNECTED;
	return(0);
}

static void null_reset(void)
{ }

/* Do we need this?  the original socket should persist in the listening state... */
#ifdef FOOBAR

int relisten(Port *mpp)
{
	/* call the user's reset function */
	/* this was introduced to allow the OMDR to be reset */

	(*reset_func)();

	if( verbose ){
		sprintf(error_string,"%s (relisten):  listening for a new connection on port %s %d",
			tell_progname(), mpp->mp_name, ntohs(mpp->mp_addrp->sin_port) );
		advise(error_string);
	}

	/* close this message socket, but listen on the port */
	close(mpp->mp_sock);
	/* should we set mp_sock to an invalid value?? */
	if( port_listen(mpp) == (-1) ){
		return(-1);
	}
	return(0);
}

#endif /* FOOBAR */


