#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif

#ifdef BUILD_FOR_WINDOWS
#include <winsock2.h>
#include <ws2tcpip.h>
#endif // BUILD_FOR_WINDOWS

#include "ports.h"

#define get_it( opt )	if( getsockopt(mpp->mp_sock, SOL_SOCKET, opt, \
			valbuf, &vbsiz) != 0 ) \
			tell_sys_error("getsockopt"); \
			if( vbsiz!=sizeof(int) ){ \
				sprintf(msg_str,"return value has size %d!!!",vbsiz); \
				prt_msg(msg_str);}

#define VBUFSIZ	128

#ifdef HAVE_SOCKET
void show_sockopts(QSP_ARG_DECL  Port *mpp)
{
	char valbuf[VBUFSIZ];
	socklen_t vbsiz=VBUFSIZ;
	int *ival=(int *)valbuf;

	sprintf(msg_str,"Options for socket \"%s\":",mpp->mp_name);
	prt_msg(msg_str);

	get_it( SO_DEBUG )
	sprintf(msg_str,"\tDEBUG\t%d",*ival);
	prt_msg(msg_str);
	get_it( SO_REUSEADDR )
	sprintf(msg_str,"\tREUSEADDR\t%d",*ival);
	prt_msg(msg_str);
	get_it( SO_KEEPALIVE )
	sprintf(msg_str,"\tKEEPALIVE\t%d",*ival);
	prt_msg(msg_str);
	get_it( SO_DONTROUTE )
	sprintf(msg_str,"\tDONTROUTE\t%d",*ival);
	prt_msg(msg_str);
	get_it( SO_LINGER )
	sprintf(msg_str,"\tLINGER\t%d",*ival);
	prt_msg(msg_str);
#ifdef SO_BROADCAST
	get_it( SO_BROADCAST )
	sprintf(msg_str,"\tBROADCAST\t%d",*ival);
	prt_msg(msg_str);
#endif
#ifdef SO_OOBINLINE
	get_it( SO_OOBINLINE )
	sprintf(msg_str,"\tOOBINLINE\t%d",*ival);
	prt_msg(msg_str);
#endif
#ifdef SO_SNDBUF
	get_it( SO_SNDBUF )
	sprintf(msg_str,"\tSNDBUF\t%d",*ival);
	prt_msg(msg_str);
#endif
#ifdef SO_RCVBUF
	get_it( SO_RCVBUF )
	sprintf(msg_str,"\tRCVBUF\t%d",*ival);
	prt_msg(msg_str);
#endif
#ifdef SO_TYPE
	get_it( SO_TYPE )
	sprintf(msg_str,"\tTYPE\t%d",*ival);
	prt_msg(msg_str);
#endif
#ifdef SO_ERROR
	get_it( SO_ERROR )
	sprintf(msg_str,"\tERROR\t%d",*ival);
	prt_msg(msg_str);
#endif
}
#endif // HAVE_SOCKET

