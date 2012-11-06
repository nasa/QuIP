#include "quip_config.h"

char VersionId_ports_portmenu[] = QUIP_VERSION_STRING;

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif /* HAVE_STRING_H */

#include "nports.h"

#include "uio.h"
#include "data_obj.h"
#include "query.h"
#include "debug.h"

/* this function gets called instead of fgets ... */

static Port *the_port=NO_PORT;
static int dont_proc_text=0;

/* local prototypes */
static void text_proc(QSP_ARG_DECL  const char *s);



static COMMAND_FUNC( start_client )
{
	const char *s;
	char handle[64];
	char str[16];
#define HNLEN	128
	char hostname[HNLEN];
	int port_no,actual;

	s= NAMEOF("handle for this port");
	strcpy(handle,s);
	s=NAMEOF("hostname");
	strcpy(hostname,s);
	port_no = (int)HOW_MANY("port number");

	actual = open_client_port(QSP_ARG  handle, hostname, port_no );
	if( actual >= 0 ){	/* success */
		if( actual != port_no ){
			sprintf(ERROR_STRING,"start_client:  Service port number %d requested, %d actually used",port_no,actual);
			advise(ERROR_STRING);
		}
	}
	sprintf(str,"%d",actual);
	ASSIGN_VAR("actual_port_number",str);
}

static COMMAND_FUNC( start_server )
{
	const char *s;
	char handle[64];
	char str[16];
	int port_no,actual;

	s= NAMEOF("handle for this port");
	strcpy(handle,s);
	/* we don't need the hostname to read !! */
	port_no = (int)HOW_MANY("port number");

	actual = open_server_port(QSP_ARG  handle, port_no );
	if( actual >= 0 ){	/* success */
		if( actual != port_no ){
			sprintf(ERROR_STRING,"start_server:  Port number %d requested, %d actually used",port_no,actual);
			advise(ERROR_STRING);
		}
	}
	sprintf(str,"%d",actual);
	ASSIGN_VAR("actual_port_number",str);
}

COMMAND_FUNC( clssock )
{
	Port *mpp;

	mpp = PICK_PORT("");
	if( mpp==NO_PORT ) return;
	reset_port(QSP_ARG  mpp );
}

#ifdef FOOBAR
COMMAND_FUNC( lstnport )
{
	Port *mpp;

	mpp = PICK_PORT("");
	if( mpp==NO_PORT ) return;

	if( (mpp->mp_flags & PORT_SERVER) == 0 ){
		sprintf(ERROR_STRING,"lstnport:  port %s is not a server port!?",
			mpp->mp_name);
		WARN(ERROR_STRING);
		return;
	}
	port_listen(mpp);
}
#endif

COMMAND_FUNC( connect_port )
{
	Port *mpp;

	mpp = PICK_PORT("");
	if( mpp==NO_PORT ) return;

	if( (mpp->mp_flags & PORT_SERVER) == 0 ){
		sprintf(ERROR_STRING,"lstnport:  port %s is not a server port!?",
			mpp->mp_name);
		WARN(ERROR_STRING);
		return;
	}

	get_connection(QSP_ARG  mpp);
}

COMMAND_FUNC( do_portinfo )
{
	Port *mpp;

	mpp = PICK_PORT("");
	if( mpp==NO_PORT ) return;

	portinfo(mpp);
}

COMMAND_FUNC( do_getopts )
{
	Port *mpp;

	mpp = PICK_PORT("");
	if( mpp==NO_PORT ) return;

	show_sockopts(mpp);
}

char *pushfunc( TMP_QSP_ARG_DECL   char *buf, int size, FILE *fp )
{
	Packet *pkp;
	int l;

top:
	dont_proc_text=1;
	pkp=recv_data(QSP_ARG  the_port);
	if( pkp == NO_PACKET ) return(NULL);

	dont_proc_text=0;

	switch( pkp->pk_code ){
		case P_ERROR:
			WARN("pushfunc:  data reception error (P_ERROR)");
			return(NULL);
		case P_TEXT:
			l=strlen(pkp->pk_data);
			if( l==0 ) return(NULL);
			else if( (l+1) > size )
				WARN("pushfunc:  too much text for buffer");
			strncpy(buf,pkp->pk_data,size);
			buf[size-1]=0;

			/* free the packet data */
			givbuf(pkp->pk_data);

			break;	/* goes to normal return */
		default:
			/* do nothing except try again */
			/* actually, in a viewer, we would want to redisplay */

			/* BUG can this code ever be reached,
			 * or is this condition detected further down?
			 */

			if( verbose ){
				sprintf(ERROR_STRING,
				"Ignoring unrecognized packet code %d",
					pkp->pk_code);
				advise(ERROR_STRING);
			}

			goto top;

	}
	return(buf);
}

COMMAND_FUNC( port_redir )
{
	Port *mpp;
	char str[128];

	mpp = PICK_PORT("");
	if( mpp==NO_PORT ) return;

	if( ! IS_CONNECTED(mpp) ){
		sprintf(ERROR_STRING,
"port_redir:  Port %s is not connected to a client!?",mpp->mp_name);
		WARN(ERROR_STRING);
		return;
	}

	the_port=mpp;

	sprintf(str,"Port %s",mpp->mp_name);
	push_input_file(QSP_ARG  str);

	redir(QSP_ARG  NULL);
	set_qflags(QSP_ARG  Q_SOCKET);
	set_rf(QSP_ARG  pushfunc);
}

static COMMAND_FUNC(do_list_ports) {list_ports(SINGLE_QSP_ARG);}

static COMMAND_FUNC(do_set_sleeptime)
{
	Port *mpp;
	int n;

	mpp = PICK_PORT("");
	n=HOW_MANY("microseconds to sleep when waiting");

	if( mpp==NO_PORT ) return;

	mpp->mp_sleeptime = n;
}

Command port_ctbl[]={
{ "server",	start_server,		"open a server port"			},
{ "client",	start_client,		"open a port to a server"		},
{ "connect",	connect_port,		"accept a connection from a client"	},
{ "xmit",	do_port_xmit,		"transmit data over an open port"	},
{ "receive",	do_port_recv,		"receive data on open port"		},
{ "close",	clssock,		"close a currently open port"		},
{ "list",	do_list_ports,		"list all ports"			},
{ "info",	do_portinfo,		"give info about a port"		},
{ "redir",	port_redir,		"redirect input to port"		},
{ "sleeptime",	do_set_sleeptime,	"set time to sleep when waiting"	},
{ "getopts",	do_getopts,		"get port (socket) options"		},
{ "quit",	popcmd,			"quit"					},
{ NULL_COMMAND									}
};

static void text_proc(QSP_ARG_DECL   const char *s)	/* simple text processing */
{
	if( !dont_proc_text ){
		const char *vn;

		vn=NAMEOF("name of variable for text storage");
		if( *s ) {
			ASSIGN_VAR(vn,s);
			/* We use givbuf here (instead of rls_str)
			 * because the buffer is obtained with getbuf
			 * in recv_text() (xmitrecv.c).
			 *
			 * BUG?  shouldn't we release the buffer in the same
			 * piece of code where we allocate it???
			 */
			givbuf(s);
		}
	}
}

COMMAND_FUNC( portmenu )
{
	init_ports(SINGLE_QSP_ARG);

	if( !have_port_data_type(QSP_ARG  P_TEXT) ){
		if( define_port_data_type(QSP_ARG  (int)P_TEXT, "text", "text to send",
			recv_text,
			text_proc,
			nameof,
			xmit_text) == -1 )

			WARN("error adding text data type to port tables");
	}

	PUSHCMD(port_ctbl,"ports");
}

