#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
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

#include "ports.h"
#include "getbuf.h"
#include "debug.h"
#include "typedefs.h"
//#include "uio.h"
//#include "submenus.h"	/* call_event_funcs */

static int msg_sock;
#ifdef HAVE_SOCKET

#ifdef BUILD_FOR_WINDOWS
// BUG should get this from an include file!?
typedef int32_t socklen_t;
#endif // BUILD_FOR_WINDOWS

static struct sockaddr from;
static socklen_t from_len = sizeof(struct sockaddr);


#endif // HAVE_SOCKET


static void null_reset(void);
static void (*reset_func)(void)=null_reset;

void set_reset_func( void (*func)(void) )
{
	reset_func = func;
}


/* Put the server port into the listening state */

static int port_listen(QSP_ARG_DECL  Port *mpp)
{
#ifdef QUIP_DEBUG
	if( debug & debug_ports ) advise("port_listen:  listening for a request");
#endif /* QUIP_DEBUG */

	if( ! IS_SERVER(mpp) ){
		warn(ERROR_STRING);
		sprintf(ERROR_STRING,"port_listen:  %s is not a server port",mpp->mp_name);
		return(-1);
	}

	/* first close old connection */
	if( mpp->mp_sock != (-1) ){
		sprintf(ERROR_STRING,"closing old connection on port %s (sock = 0x%x)",
			mpp->mp_name,mpp->mp_sock);
		advise(ERROR_STRING);
#ifdef BUILD_FOR_WINDOWS
		closesocket(mpp->mp_sock);
#else
		close(mpp->mp_sock);
#endif
	}

	/* The backlog argument allows us to stack connection requests
	 * (currently set to 5); however, because we don't fork servers
	 * when we get a new connection, a pending request doesn't actually
	 * get serviced until the present one is finished.  For one-way
	 * communication from the client to the server, the client can
	 * send its data and exit, and the server will (hopefully) get
	 * to it later.  But what happens if the server crashes after
	 * the client finishes?  Presumably if bidirectional communication
	 * is required, the process would hang while waiting...
	 */

fprintf(stderr,"calling listen...\n");
	if( listen(mpp->mp_o_sock,PORT_BACKLOG) != 0 ){
		tell_sys_error("listen");
		return(-1);
	}
fprintf(stderr,"listen returned...\n");
	return(0);
} // port_listen

/* Return the port number actually used, or -1 in case of error */

int open_server_port(QSP_ARG_DECL  const char *name,int  port_no)
{
	Port *mpp;
#ifdef HAVE_SOCKET
	socklen_t length;
#endif // HAVE_SOCKET
	int on=1;

	if ( (port_no < 2001) || (port_no > 6999) ) {
		warn("Illegal port number");
		advise("Use 2001-6999");
		advise("Check /etc/services for other conflicts");
		return(-1);
	}

	mpp=new_port(name);
	if( mpp==NULL ){
		if( verbose ){
			sprintf(ERROR_STRING,"open_server_port %s %d failed to create port struct",
				name,port_no);
			warn(ERROR_STRING);
		}
		return(-1);
	}

#ifdef HAVE_SOCKET
	mpp->mp_o_sock=socket(AF_INET,SOCK_STREAM,0);
//fprintf(stderr,"socket returned %d (0x%x)\n",
//mpp->mp_o_sock,mpp->mp_o_sock);
#ifdef FOOBAR
	// pc-nfs???
	mpp->mp_o_sock=socket(PF_INET,SOCK_STREAM,0);
#endif /* FOOBAR */
#else // ! HAVE_SOCKET
	warn("open_server_port:  Sorry, no socket implementation available!?");
	mpp->mp_o_sock=(-1);
#endif // HAVE_SOCKET

#ifdef BUILD_FOR_WINDOWS
	if( mpp->mp_o_sock == INVALID_SOCKET ){
		int e;
		e=WSAGetLastError();
		fprintf(stderr,"Invalid socket, error code = %d\n",e);
		switch(e){
			case WSANOTINITIALISED:
				warn("Missing call to WSAStartup!?");
				break;
			default:
				warn("Unclassified error.");
				break;
		}
	}
#endif // BUILD_FOR_WINDOWS

	// sleeptime used to be signed, but not now.
	//mpp->mp_sleeptime=(-1);
	mpp->mp_sleeptime=0;
	mpp->mp_sock=(-1);
		
	if( mpp->mp_o_sock<0 ){
		tell_sys_error("open_server_port (socket)");
		warn("error opening stream socket");
		delport(QSP_ARG  mpp);
		return(-1);
	}

#ifdef HAVE_SOCKET
fprintf(stderr,"Setting socket options...\n");
	setsockopt(mpp->mp_o_sock,SOL_SOCKET,SO_REUSEADDR,
		(char *)&on,sizeof(on));

	mpp->mp_addrp =
		(struct sockaddr_in *) getbuf( sizeof(*(mpp->mp_addrp) ) );
	if( mpp->mp_addrp == NULL ) mem_err("open_server_port");

	mpp->mp_flags = 0;
	mpp->mp_flags |= PORT_SERVER;	// makes it keep trying!
	mpp->mp_pp = NULL;
	mpp->mp_text_var_name=NULL;
	mpp->mp_output_filename=NULL;
	mpp->mp_auth_string=NULL;
	mpp->mp_portnum = port_no;

	mpp->mp_addrp->sin_family = AF_INET;
	mpp->mp_addrp->sin_addr.s_addr = INADDR_ANY;
	length=sizeof(*(mpp->mp_addrp));
	mpp->mp_addrp->sin_port = htons( port_no );
fprintf(stderr,"Binding socket...\n");
	if( bind(mpp->mp_o_sock,(struct sockaddr *)(mpp->mp_addrp), length) ){
		tell_sys_error("bind");
		warn("open_server_port:  couldn't bind to port");
		goto cleanup;
	}

fprintf(stderr,"Getting socket name...\n");
	if( getsockname(mpp->mp_o_sock,
		(struct sockaddr *)mpp->mp_addrp,&length) ){
		warn("open_server_port:  error getting socket name");
		goto cleanup;
	}
	/* We used to make sure that the port number
	 * was the one we requested, but we no longer
	 * insist upon that...
	 */

	if( port_listen(QSP_ARG  mpp) < 0 )
		goto cleanup;

fprintf(stderr,"Back from port_listen...\n");
#endif // HAVE_SOCKET

	return(port_no);

cleanup:
	givbuf((char *)mpp->mp_addrp);
	mpp->mp_addrp = NULL;
	delport(QSP_ARG  mpp);
	return(-1);
} // open_server_port

#ifdef SERVER_SLEEPS_WHILE_WAITING
static void sleep_if(Port *mpp,sleep_time_t default_t)
{
	// BUG sleeptime is now unsigned?
	if( mpp->mp_sleeptime < 0 )
#ifdef BUILD_FOR_WINDOWS
		Sleep(default_t/1000);
#else
		usleep(default_t);
#endif
	else if( mpp->mp_sleeptime > 0 )
#ifdef BUILD_FOR_WINDOWS
		Sleep(1+mpp->mp_sleeptime/1000);
#else
		usleep( mpp->mp_sleeptime );
#endif
}
#endif // SERVER_SLEEPS_WHILE_WAITING

int get_connection(QSP_ARG_DECL  Port *mpp)
{
#ifndef BUILD_FOR_WINDOWS
#ifdef HAVE_SELECT
	fd_set rd_fds, null_fds;
	struct timeval time_out;
	int v;

	int width;

	/* hang until someone requests a hookup */
	if( verbose ){
sprintf(ERROR_STRING,"\nWaiting for a new connection request on port %d...\n",
mpp->mp_portnum);
advise(ERROR_STRING);
	}

//fprintf(stderr,"port flags = 0x%x (%d)\n",
//mpp->mp_flags,mpp->mp_flags);

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
#ifdef BUILD_FOR_WINDOWS
	width=256;	// what is the right value??? BUG?
#else
	width=getdtablesize();
#endif
	time_out.tv_sec=0;
	time_out.tv_usec=100000;	/* 10 Hz poll */

	/* Wait for a connection */
//fprintf(stderr,"get_connection:  calling select\n");

	while( (v=select( width, &rd_fds, &null_fds, &null_fds,
		&time_out )) <= 0 )
	{
		/* Nothing yet... */

		if( v < 0 ){		/* select error */
			if( errno != EINTR ){
				tell_sys_error("(get_connection) select");
				return(-1);
			}
		}
		FD_SET( mpp->mp_o_sock, &rd_fds );
		call_event_funcs(SINGLE_QSP_ARG);
#ifdef SERVER_SLEEPS_WHILE_WAITING
		sleep_if(mpp,250000);	/* Check again in 250 msec */
#endif // SERVER_SLEEPS_WHILE_WAITING
	}

	if( !FD_ISSET(mpp->mp_o_sock,&rd_fds) ){
		warn("get_connection:  shouldn't happen");
		return(-1);
	}
#else /* ! HAVE_SELECT */
	warn("No select function!?");
#endif /* ! HAVE_SELECT */
#endif // ! BUILD_FOR_WINDOWS

//advise("get_connection:  select detected something, calling accept...");

#ifndef BUILD_FOR_WINDOWS
#define INVALID_SOCKET	(-1)	// BUG make sure this is not already defined!?
#endif // ! BUILD_FOR_WINDOWS

#ifdef HAVE_SOCKET
	// BUG from should not be global
	// BUG we should do something with the information
	// returned in from
	while( (msg_sock=accept(mpp->mp_o_sock,&from,&from_len)) == INVALID_SOCKET ){
#ifdef BUILD_FOR_WINDOWS
		switch( WSAGetLastError() ){
			default:
				warn("Unknown Windows error!?");
				break;
		}
#endif // BUILD_FOR_WINDOWS

		if ( errno == EWOULDBLOCK) {
#ifdef BUILD_FOR_WINDOWS
			Sleep(1000);
#else
			sleep (1); /* Try accept again in 1 second */
#endif
			continue;
		}
		if( errno != EINTR ){
			tell_sys_error("(get_connection) accept");
			return(-1);
		}
		call_event_funcs(SINGLE_QSP_ARG);

#ifdef SERVER_SLEEPS_WHILE_WAITING
		sleep_if(mpp,250000);	/* Check again in 250 msec */
#endif // SERVER_SLEEPS_WHILE_WAITING
		
		
	}
#else // ! HAVE_SOCKET
	warn("Sorry, no accept implementation!?");
#endif // ! HAVE_SOCKET

//fprintf(stderr,"get_connection:  setting mp_sock to %d\n",msg_sock);
	mpp->mp_sock=msg_sock;

	/* Now we set a flag to show that this socket is connected */
	mpp->mp_flags |= PORT_CONNECTED;
	return(0);

}	// end get_connection

static void null_reset(void)
{ }


