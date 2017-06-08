#include "quip_config.h"

#define TRUE	1
#define FALSE	0

#include <stdio.h>

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif /* HAVE_ERRNO_H */

#ifdef HAVE_STRING_H
#include <string.h>		/* memmove */
#endif /* HAVE_STRING_H */

#ifdef HAVE_STRINGS_H
#include <strings.h>		/* bcopy - deprecated */
#endif /* HAVE_STRINGS_H */

#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* close */
#endif /* HAVE_UNISTD_H */

#include "typedefs.h"

#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif /* HAVE_NETDB_H */

/* these are needed for select(2): */
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif /* HAVE_SYS_TYPES_H */

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif /* HAVE_SYS_TIME_H */

#include "ports.h"
#include "getbuf.h"
#include "debug.h"


#ifdef BUILD_FOR_IOS
#define DEFAULT_MAX_RETRIES	0
#else // ! BUILD_FOR_IOS
#define DEFAULT_MAX_RETRIES	-1	// try forever
#endif // ! BUILD_FOR_IOS

// BUG using these static variables is not thread-safe,
// consider moving to query_stack struct!

#ifdef HAVE_SLEEP
#define MIN_WAIT_SECONDS	2
#define DEFAULT_WAIT_SECONDS	MIN_WAIT_SECONDS
static int wait_seconds=DEFAULT_WAIT_SECONDS;
#endif /* HAVE_SLEEP */

static int max_retries=DEFAULT_MAX_RETRIES;
static int n_retries;

#define CLEANUP_PORT				\
						\
	givbuf((char *)mpp->mp_addrp);		\
	mpp->mp_addrp = NULL;			\
	delport(QSP_ARG  mpp); /* delport calls rls_str for the name */


void set_max_client_retries(QSP_ARG_DECL  int new_max)
{
	max_retries = new_max;
}

int open_client_port(QSP_ARG_DECL  const char *name,const char *hostname,int port_no)
{
	Port *mpp;
	int first_attempt = TRUE;
	struct hostent FAR *hp;

	if ( (port_no < 2001) || (port_no > 6999) ) {
		WARN("Illegal port number");
		advise("Use 2001-6999");
		advise("Check /etc/services for other conflicts");
		return(-1);
	}

fresh_start:

	mpp=new_port(QSP_ARG  name);
	if( mpp==NULL ){
		if( verbose ){
			sprintf(ERROR_STRING,"open_client_port %s %s %d failed to create port struct",
				name,hostname,port_no);
			WARN(ERROR_STRING);
		}
		return(-1);
	}

advise("port struct created...");

#ifdef HAVE_SOCKET
	mpp->mp_sock=socket(AF_INET,SOCK_STREAM,0);
#else // ! HAVE_SOCKET
	WARN("open_client_port:  Sorry, no socket implementation!?");
	mpp->mp_sock = (-1);
#endif // ! HAVE_SOCKET

	if( mpp->mp_sock<0 ){
		tell_sys_error("open_client_port (socket)");
		WARN("error opening stream socket");
		delport(QSP_ARG  mpp);
		return(-1);
	}

#ifdef HAVE_SOCKET
	/* don't need REUSEADDR for client sockets??? */

	/*
	setsockopt(mpp->mp_sock,SOL_SOCKET,SO_REUSEADDR,
		(char FAR *)&on,sizeof(on));
		*/

	mpp->mp_addrp =
		(struct sockaddr_in *) getbuf( sizeof(struct sockaddr_in) );
	if( mpp->mp_addrp == NULL ) mem_err("open_client_port");

	mpp->mp_flags = 0;
	mpp->mp_o_sock = (-1);
	mpp->mp_flags |= PORT_CLIENT;
	mpp->mp_pp = NULL;
	// sleeptime used to be signed, but not now.
	//mpp->mp_sleeptime=(-1);
	mpp->mp_sleeptime=0;

	mpp->mp_text_var_name=NULL;
	mpp->mp_output_filename=NULL;
fprintf(stderr,"calling gethostbyname with hostname %s\n",hostname);
	hp=gethostbyname((const char FAR *)hostname); 
fprintf(stderr,"back from gethostbyname hp = 0x%lx\n",(long)hp);

	if( hp == NULL ){
		switch( h_errno ){
			case HOST_NOT_FOUND:
				sprintf(ERROR_STRING,
			"open_client_port:  host \"%s\" not found",
					hostname);
				WARN(ERROR_STRING);
				break;
			default:
				sprintf(ERROR_STRING,
			"open_client_port:  unknown host:  \"%s\"",
					hostname);
				WARN(ERROR_STRING);
				break;
		}
		goto cleanup;
	}

	mpp->mp_addrp->sin_family=AF_INET;
#ifdef HAVE_MEMCPY
	// src and dst are reversed relative to bcopy...
	memcpy( &(mpp->mp_addrp->sin_addr), hp->h_addr, hp->h_length );
#else // ! HAVE_MEMCPY

#ifdef HAVE_BCOPY
	bcopy(hp->h_addr,&(mpp->mp_addrp->sin_addr),hp->h_length);
#else /* ! HAVE_BCOPY */
	ERROR1("open_client_port:  need to implement without bcopy!?");
#endif /* ! HAVE_BCOPY */
#endif // ! HAVE_MEMCPY

advise("calling htons...");
	mpp->mp_addrp->sin_port = htons(port_no);
fprintf(stderr,"calling connect (port = %d)...\n",port_no);

// BUG how many times should we try???
// On iOS, there is no ctl-C to interrupt!

	while( connect(mpp->mp_sock,(struct sockaddr *)mpp->mp_addrp,
		sizeof(*(mpp->mp_addrp)) ) < 0 ){
fprintf(stderr,"connect failed...\n");
		if( first_attempt ){
			first_attempt = FALSE;
			if( verbose )
				tell_sys_error("open_client_port (connect)");

			if( errno == ECONNREFUSED ){
				sprintf(ERROR_STRING,
	"Host %s refused or ignored connection request on port %d;",
					hostname,port_no);
			} else {
				tell_sys_error("connect");
				sprintf(ERROR_STRING,
	"No response from host %s on port %d;",
					hostname,port_no);
			}
			advise(ERROR_STRING);

			if( max_retries > 1 ){
				sprintf(ERROR_STRING,
			"Will retry up to %d times.",max_retries);
				advise(ERROR_STRING);
			} else if( max_retries==1 ){
				advise("Will retry once.");
			} else if( max_retries==0 ){
				advise("Will not retry.");
			} else {
				advise("Will retry forever.");
			}
			n_retries=0;
#ifdef HAVE_SLEEP
			if( max_retries != 0 ){
				sprintf(ERROR_STRING,
			"Waiting %d seconds between attempts...",
					wait_seconds);
				advise(ERROR_STRING);
			}
#endif /* HAVE_SLEEP */
		}
#ifdef HAVE_SLEEP
		sleep(wait_seconds);
#endif /* HAVE_SLEEP */

		goto retry;
	}
fprintf(stderr,"connect succeeded...\n");
#endif // HAVE_SOCKET
	mpp->mp_flags |= PORT_CONNECTED;
//advise("connected!");

	// This seems to succeed even when the server is connected to another
	// client, although all packet processing is deferred...

	return(port_no);
retry:
#ifdef BUILD_FOR_WINDOWS
	closesocket(mpp->mp_sock);
#else
	close(mpp->mp_sock);
#endif

	CLEANUP_PORT

	if( max_retries >= 0 && n_retries >= max_retries ){
		if( max_retries >= 1 ){
			sprintf(ERROR_STRING,
				"Giving up after %d unsuccessful %s.",
				n_retries,n_retries>1?"retries":"retry");
			advise(ERROR_STRING);
		}
		return(-1);
	}
	n_retries++;	// count the one we will do next

#ifdef HAVE_SLEEP
	sleep(wait_seconds-1);
#endif /* HAVE_SLEEP */

fprintf(stderr,"retrying connect...\n");

	goto fresh_start;

cleanup:
	CLEANUP_PORT
	return(-1);
} /* end open_client_port */



