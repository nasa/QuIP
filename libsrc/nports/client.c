#include "quip_config.h"

char VersionId_ports_client[] = QUIP_VERSION_STRING;

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
#include <strings.h>		/* bcopy */
#endif /* HAVE_STRINGS_H */

#include "typedefs.h"
#include "uio.h"

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

#include "nports.h"
#include "getbuf.h"
#include "debug.h"

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
	if( mpp==NO_PORT ){
		if( verbose ){
			sprintf(error_string,"open_client_port %s %s %d failed to create port struct",
				name,hostname,port_no);
			WARN(error_string);
		}
		return(-1);
	}

	mpp->mp_sock=socket(AF_INET,SOCK_STREAM,0);

#ifdef FOOBAR
	// is this a relic from pc-nfs?
	mpp->mp_sock=socket(PF_INET,SOCK_STREAM,0);
#endif /* FOOBAR */
		
	if( mpp->mp_sock<0 ){
		tell_sys_error("open_client_port (socket)");
		WARN("error opening stream socket");
		delport(QSP_ARG  mpp);
		return(-1);
	}

	/* don't need REUSEADDR for client sockets??? */

	/*
	setsockopt(mpp->mp_sock,SOL_SOCKET,SO_REUSEADDR,
		(char FAR *)&on,sizeof(on));
		*/

	mpp->mp_addrp =
		(struct sockaddr_in *) getbuf( sizeof(*(mpp->mp_addrp) ) );
	if( mpp->mp_addrp == NULL ) mem_err("open_client_port");

	mpp->mp_flags = 0;
	mpp->mp_o_sock = (-1);
	mpp->mp_flags |= PORT_CLIENT;
	mpp->mp_pp = NO_PORT;
	mpp->mp_sleeptime=(-1);

	hp=gethostbyname((const char FAR *)hostname); 

	if( hp == NULL ){
		switch( h_errno ){
			case HOST_NOT_FOUND:
				sprintf(error_string,
			"open_client_port:  host \"%s\" not found",
					hostname);
				WARN(error_string);
				break;
			default:
				sprintf(error_string,
			"open_client_port:  unknown host:  \"%s\"",
					hostname);
				WARN(error_string);
				break;
		}
		goto cleanup;
	}

	mpp->mp_addrp->sin_family=AF_INET;
#ifdef HAVE_BCOPY
	bcopy(hp->h_addr,&(mpp->mp_addrp->sin_addr),hp->h_length);
#else /* ! HAVE_BCOPY */
	error1("open_client_port:  need to implement without bcopy!?");
#endif /* ! HAVE_BCOPY */

	mpp->mp_addrp->sin_port = htons(port_no);

	while( connect(mpp->mp_sock,(struct sockaddr *)mpp->mp_addrp,
		sizeof(*(mpp->mp_addrp)) ) < 0 ){
		if( first_attempt ){
			first_attempt = FALSE;
			if( verbose )
				tell_sys_error("open_client_port (connect)");
			sprintf(error_string,
		"No response from host %s on port %d, waiting...",
				hostname,port_no);
			advise(error_string);
		}
#ifdef HAVE_SLEEP
		sleep(1);
#endif /* HAVE_SLEEP */

		goto retry;
	}

	mpp->mp_flags |= PORT_CONNECTED;

	return(port_no);
retry:

	close(mpp->mp_sock);

	givbuf((char *)mpp->mp_addrp);
	mpp->mp_addrp = NULL;
	delport(QSP_ARG  mpp);		/* delport calls rls_str for the name */

	sleep(3);
	goto fresh_start;

cleanup:
	givbuf((char *)mpp->mp_addrp);
	mpp->mp_addrp = NULL;
	delport(QSP_ARG  mpp);

	return(-1);
} /* end open_client_port */



