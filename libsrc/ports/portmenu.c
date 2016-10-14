#include "quip_config.h"

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif /* HAVE_STRING_H */

#include "ports.h"

//#include "uio.h"
#include "data_obj.h"
#include "query.h"
#include "function.h"
#include "debug.h"

/* this function gets called instead of fgets ... */

// BUG should be per qsp, not a global!  Not thread safe!
//static Port *the_port=NO_PORT;	// why global???
// This variable is global so that port_read can have the same args
// as fread, but the fp arg is not used, so we could put a port ptr there???

static int dont_proc_text=0;

static double port_exists(QSP_ARG_DECL  const char *name)
{
	Port *mpp;

	mpp = port_of(QSP_ARG  name);
	if( mpp==NO_PORT ) return(0.0);
	return(1.0);
}

void null_proc(QSP_ARG_DECL  const char *s)
{}

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
//fprintf(stderr,"server started on port %d\n",actual);
}

static COMMAND_FUNC( do_reset_socket )
{
	Port *mpp;

	mpp = PICK_PORT("");
	if( mpp==NO_PORT ) return;
	reset_port(QSP_ARG  mpp );
}

static COMMAND_FUNC( do_close_socket )
{
	Port *mpp;

	mpp = PICK_PORT("");
	if( mpp==NO_PORT ) return;
	mpp->mp_flags &= ~PORT_SERVER;	// force closure
	reset_port(QSP_ARG  mpp );
}

static COMMAND_FUNC( do_connect_port )
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

static COMMAND_FUNC( do_portinfo )
{
	Port *mpp;

	mpp = PICK_PORT("");
	if( mpp==NO_PORT ) return;

	portinfo(QSP_ARG  mpp);
}

static COMMAND_FUNC( do_getopts )
{
	Port *mpp;

	mpp = PICK_PORT("");
	if( mpp==NO_PORT ) return;

#ifdef HAVE_SOCKET
	show_sockopts(QSP_ARG  mpp);
#else // ! HAVE_SOCKET
	WARN("do_getopts:  Sorry, no socket implementation!?");
#endif // ! HAVE_SOCKET
}

// port_read acts as readfunc when we are redirecting.
// Text packets return, others are processed...

static char *port_read( QSP_ARG_DECL   /*char */ void *vbuf, int size, /*FILE*/ void *fp )
{
	Packet *pkp;
	long l;
	char *buf;
	Port *the_port;

	the_port = (Port *)fp;

	if( the_port->mp_flags & PORT_ZOMBIE ){
		delport(QSP_ARG  the_port);
		the_port=NULL;
		return NULL;
	}

	buf=(char *)vbuf;
top:
//fprintf(stderr,"port_read:  waiting for data...\n");
	dont_proc_text=1;
	pkp=recv_data(QSP_ARG  the_port);
	if( pkp == NO_PACKET ) {
//fprintf(stderr,"port_read:  no more data...\n");
		// when we run out of data, clear the redir flag
		// so a client can close the port and not create
		// a zombie.
		the_port->mp_flags &= ~( PORT_REDIR |
					PORT_AUTHENTICATED |
					PORT_CONNECTED );
		return(NULL);
	}

	dont_proc_text=0;

//fprintf(stderr,"port_read:  packet_code = %d\n",pkp->pk_pdt->pdt_code);

	switch( pkp->pk_pdt->pdt_code ){

		case P_TEXT:
			// BUG should allow a flag to be set
			// that allows us to insist
			// on ONLY encrypted text, for security purposes...
			// 
			// Then we would check that flag here,
			// and complain if it's set, otherwise
			// fall-through...
			if( IS_SECURE(the_port) ) {
advise("plain text received on secure port!?");
				goto shutdown_port;
			}

			// fall-thru

		case P_ENCRYPTED_TEXT:
			if( NEEDS_AUTH(the_port) ) {
advise("encrypted text received without authorization!?");
				goto shutdown_port;
			}

			l=strlen(pkp->pk_user_data);
			if( l==0 ){
advise("empty encrypted text packet, returning null...");
				return(NULL);
			}
			else if( (l+1) > size )
				WARN("port_read:  too much text for buffer");
//fprintf(stderr,"received %ld bytes of encrypted text\n",l);
			strncpy(buf,pkp->pk_user_data,size);
			buf[size-1]=0;
//fprintf(stderr,"\n%s\n\n",buf);

			// We used to free the packet data here,
			// but now we manage that in packets.c ...

			break;	/* goes to normal return */
		case P_PLAIN_FILE:
			if( IS_SECURE(the_port) ){
advise("plain file received on secure port!?");
				goto shutdown_port;
			}

			// fall-thru
		case P_ENCRYPTED_FILE:
			if( NEEDS_AUTH(the_port) ) {
advise("encrypted file received without authorization!?");
				goto shutdown_port;
			}

			receive_port_file(QSP_ARG  the_port,pkp);
			*buf = 0;	// return empty string
			break;
		case P_AUTHENTICATION:
			// the decrypted string should match what we
			// are waiting for...
			if( ! IS_SECURE(the_port) ) {
advise("Authentication packet received on non-secure port!?");
				goto shutdown_port;
			}

			if( strcmp(pkp->pk_user_data,the_port->mp_auth_string) ){
				log_message("Authentication string mismatch!?");
advise("Authentication string mismatch!?");
//fprintf(stderr,"Expected:  \"%s\"\n",the_port->mp_auth_string);
//fprintf(stderr,"Received:  \"%s\"\n",pkp->pk_user_data);
				goto shutdown_port;
			}
			the_port->mp_flags |= PORT_AUTHENTICATED;
			// skip this packet...

			// BUG - need to free the packet data...
			goto top;
			break;
		default:
			if( NEEDS_AUTH(the_port) ) {
advise("default packet code case, port needs auth...");
				goto shutdown_port;
			}

			/* do nothing except try again */
			/* actually, in a viewer, we would want to redisplay */

			/* BUG can this code ever be reached,
			 * or is this condition detected further down?
			 */

			if( verbose ){
				sprintf(ERROR_STRING,
	"port_read:  ignoring %s packet (code %d)",
					PORT_DATATYPE_NAME(pkp->pk_pdt),
					pkp->pk_pdt->pdt_code);
				advise(ERROR_STRING);
			}

			goto top;

	}
//fprintf(stderr,"port_read:  returning data buffer\n");
	return(buf);

shutdown_port:
//fprintf(stderr,"port_read:  shutting down port\n");
	log_message("Unexpected packet type, shutting down port.");
	*buf=0;
	// We don't expect to be authenticated here...
	the_port->mp_flags &= ~(PORT_REDIR|PORT_AUTHENTICATED|PORT_CONNECTED);
	return NULL;
} // end port_read

static COMMAND_FUNC( do_set_secure )
{
	Port *mpp;
	const char *s;

	mpp = PICK_PORT("");
	s = NAMEOF("authentication string");

	if( mpp==NO_PORT ) return;

	mpp->mp_flags |= PORT_SECURE;
	mpp->mp_flags &= ~PORT_AUTHENTICATED;

	if( mpp->mp_auth_string != NULL ){
		rls_str(mpp->mp_auth_string);
	}

	mpp->mp_auth_string = savestr(s);
}

static COMMAND_FUNC( do_port_redir )
{
	Port *mpp;
	// BUG this string holds the "filename" of the redirect stream.
	// not thread-safe, should be per-qsp!
	static char str[128];

	mpp = PICK_PORT("");
	if( mpp==NO_PORT ) return;

	if( ! IS_CONNECTED(mpp) ){
		sprintf(ERROR_STRING,
"do_port_redir:  Port %s is not connected to a client!?",mpp->mp_name);
		WARN(ERROR_STRING);
		return;
	}

	sprintf(str,"Port %s",mpp->mp_name);

	redir_with_flags(QSP_ARG  (FILE *)mpp, str, Q_SOCKET );
	//SET_QS_FLAG_BITS(THIS_QSP, Q_SOCKET);
	set_query_readfunc(QSP_ARG  port_read);

	mpp->mp_flags |= PORT_REDIR;
}

static COMMAND_FUNC(do_list_ports) {list_ports(SINGLE_QSP_ARG);}

// BUG?  should we have separate menus for servers and clients?

static COMMAND_FUNC(do_set_sleeptime)
{
	Port *mpp;
	sleep_time_t n;

	mpp = PICK_PORT("");
	n=(sleep_time_t)HOW_MANY("microseconds to sleep when waiting (server)");

	if( mpp==NO_PORT ) return;

	mpp->mp_sleeptime = n;
}

static COMMAND_FUNC(do_set_max_retries)
{
	int n;

	n=(int)HOW_MANY("maximum number of connection retries");

	set_max_client_retries(QSP_ARG  n);
}

static COMMAND_FUNC(do_test_reachability)
{
	const char *s;

	s=NAMEOF("hostname or IP address");
#ifdef BUILD_FOR_IOS
	test_reachability(QSP_ARG  s);
#else /* ! BUILD_FOR_IOS */
	// suppress compiler warning
	sprintf(ERROR_STRING,"NOT testing reachability of %s",s);
	advise(ERROR_STRING);
#endif /* ! BUILD_FOR_IOS */
}

static COMMAND_FUNC(do_send_obj)
{
	Data_Obj *dp;
	Port *mpp;
	size_t s;

	mpp = PICK_PORT("");
	dp = PICK_OBJ("data vector");

	if( mpp == NO_PORT || dp == NO_OBJ ) return;

	s = OBJ_N_MACH_ELTS(dp) * PREC_SIZE( OBJ_PREC_PTR(dp) );
	if( write_port(QSP_ARG  mpp,OBJ_DATA_PTR(dp), s ) != s ){
		WARN("Error writing data vector!?");
	}
	// WHY did this test code close the port here???
	//  // close the port here?  
	//  close_port(QSP_ARG  mpp);
}


static void send_constant_data(QSP_ARG_DECL  Port *mpp,
						int n, int val )
{
	char *buf;
	int i;

	buf = getbuf(n);

	for(i=0;i<n;i++)
		buf[i]=(char)val;	// make sure this not magic number!

	if( write_port(QSP_ARG  mpp,buf, n ) != n ){
		WARN("Error writing constant data vector!?");
	}
	//close_port(QSP_ARG  mpp);

	givbuf(buf);
}

static COMMAND_FUNC(do_bad_magic)
{
	Port *mpp;

	mpp = PICK_PORT("");
	send_constant_data(QSP_ARG  mpp, 64, 0xaa );
}

static COMMAND_FUNC(do_partial_magic)
{
	Port *mpp;

	mpp = PICK_PORT("");
	send_constant_data(QSP_ARG  mpp, 3, 0xaa );
}

static COMMAND_FUNC(do_partial_type)
{
	Port *mpp;

	mpp = PICK_PORT("");
	if( put_port_int32(QSP_ARG  mpp,PORT_MAGIC_NUMBER) == (-1) ){
		WARN("do_partial_type:  error sending magic number");
		return;
	}

	send_constant_data(QSP_ARG  mpp, 3, 0xaa );
}

static COMMAND_FUNC(do_bad_type)
{
	Port *mpp;

	mpp = PICK_PORT("");
	if( put_port_int32(QSP_ARG  mpp,PORT_MAGIC_NUMBER) == (-1) ){
		WARN("do_bad_type:  error sending magic number");
		return;
	}
	if( put_port_int32(QSP_ARG  mpp,MAX_PORT_CODES+1) == (-1) ){
		WARN("do_bad_type:  error sending packet code");
		return;
	}

	send_constant_data(QSP_ARG  mpp, 32, 0xaa );
}

static COMMAND_FUNC(do_empty_packet)
{
	Port *mpp;

	mpp = PICK_PORT("");
	if( put_port_int32(QSP_ARG  mpp,PORT_MAGIC_NUMBER) == (-1) ){
		WARN("do_empty_packet:  error sending magic number");
		return;
	}
	if( put_port_int32(QSP_ARG  mpp,P_TEXT) == (-1) ){
		WARN("do_empty_packet:  error sending packet code");
		return;
	}
	close_port(QSP_ARG  mpp);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(port_test_menu,s,f,h)

MENU_BEGIN(port_test)
ADD_CMD( send_obj,	do_send_obj,	send raw packet from specified data )
ADD_CMD( partial_magic,	do_partial_magic, send packet with incomplete magic number )
ADD_CMD( bad_magic,	do_bad_magic,	send packet with bad magic number )
ADD_CMD( partial_type,	do_partial_type,	send packet with incomplete packet type )
ADD_CMD( bad_type,	do_bad_type,	send packet with bad packet type )
ADD_CMD( empty_packet,	do_empty_packet,	send packet with missing data )
MENU_END(port_test)

static COMMAND_FUNC(do_port_tests)
{
	PUSH_MENU(port_test);
}

static COMMAND_FUNC( do_port_check )
{
	int v;
	Port *mpp;

	mpp = PICK_PORT("");

	if( mpp == NO_PORT ) return;

	v=check_port_data(QSP_ARG  mpp, 0);
	if( v == 1 )
		ASSIGN_VAR("port_ready","1");
	else
		ASSIGN_VAR("port_ready","0");
}

#undef ADD_CMD

#define ADD_CMD(s,f,h)	ADD_COMMAND(ports_menu,s,f,h)

MENU_BEGIN(ports)
ADD_CMD( reachability,	do_test_reachability,	test network reachability )
ADD_CMD( server,	start_server,		open a server port			)
ADD_CMD( secure,	do_set_secure,		insist on secure communications on a port )
ADD_CMD( client,	start_client,		open a port to a server			)
ADD_CMD( connect,	do_connect_port,	accept a connection from a client	)
ADD_CMD( check,		do_port_check,		check for readable data			)
ADD_CMD( xmit,		do_port_xmit,		transmit data over an open port		)
ADD_CMD( receive,	do_port_recv,		receive data on open port		)
ADD_CMD( close,		do_close_socket,	close a currently open port		)
ADD_CMD( reset,		do_reset_socket,	reset a currently open port		)
ADD_CMD( port_output_file,		do_set_port_output_file,	specify filename for next file received)
ADD_CMD( text_variable,		do_set_text_var,	specify variable for next text received)
ADD_CMD( list,		do_list_ports,		list all ports				)
ADD_CMD( info,		do_portinfo,		give info about a port			)
ADD_CMD( redir,		do_port_redir,		redirect input to port			)
ADD_CMD( sleeptime,	do_set_sleeptime,	set time to sleep when server is waiting		)
ADD_CMD( max_retries,	do_set_max_retries,	set max number of client retries )
ADD_CMD( getopts,	do_getopts,		get port (socket) options		)
ADD_CMD( tests,		do_port_tests,		test functions to send bad packets )
MENU_END(ports)

#ifdef BUILD_FOR_WINDOWS
static WSADATA wsa_data;
#endif // BUILD_FOR_WINDOWS

static void init_default_port_data_types(SINGLE_QSP_ARG_DECL)
{
#ifdef BUILD_FOR_WINDOWS
	// Do this here, hopefully it will be called in time
	int status;
	status=WSAStartup(1,&wsa_data);
	if( status != 0 ){
		fprintf(stderr,"Error returned by WSAStartup:  %d\n",status);
	}
#endif // BUILD_FOR_WINDOWS

	if( define_port_data_type(QSP_ARG  (int)P_TEXT, "text",
			"text to send", recv_text,
			nameof, xmit_text) == -1 )
		WARN("error adding text data type to port tables");

	if( define_port_data_type(QSP_ARG  (int)P_FILE_AS_TEXT, "file_as_text",
			"local name", recv_text,
			nameof, xmit_file_as_text) == -1 )
		WARN("error adding text data type to port tables");

	if( define_port_data_type(QSP_ARG  (int)P_PLAIN_FILE, "plain_file",
			"local filename", recv_plain_file,
			nameof, xmit_plain_file) == -1 )
		WARN("error adding plain_file data type to port tables");

	if( define_port_data_type(QSP_ARG  (int)P_FILENAME, "filename",
			"local filename", recv_filename,
			nameof, xmit_filename) == -1 )
		WARN("error adding plain_file data type to port tables");

#ifdef HAVE_ENCRYPTION
	if( define_port_data_type(QSP_ARG  (int)P_AUTHENTICATION,
			"authentication", "data to send", recv_enc_text,
			nameof, xmit_auth) == -1 )
		WARN("error adding authentication data type to port tables");

	if( define_port_data_type(QSP_ARG  (int)P_ENCRYPTED_TEXT,
			"encrypted_text", "text to send", recv_enc_text,
			nameof, xmit_enc_text) == -1 )
		WARN("error adding encrypted text data type to port tables");

	if( define_port_data_type(QSP_ARG  (int)P_ENCRYPTED_FILE,
			"encrypted_file", "local filename", recv_enc_file,
			nameof, xmit_enc_file) == -1 )
		WARN("error adding encrypted file data type to port tables");

	if( define_port_data_type(QSP_ARG  (int)P_ENC_FILE_AS_TEXT,
			"encrypted_file_as_text", "local filename", recv_enc_text,
			nameof, xmit_enc_file_as_text) == -1 )
		WARN("error adding encrypted file data type to port tables");
#endif /* HAVE_ENCRYPTION */

	// do this now too...
	DECLARE_STR1_FUNCTION( port_exists,	port_exists	)
}

COMMAND_FUNC( do_port_menu )
{
	Port_Data_Type *pdtp;

	pdtp = pdt_of(QSP_ARG  "text");	// BUG expensive flag lookup!
	if( pdtp == NULL ) init_default_port_data_types(SINGLE_QSP_ARG);

	PUSH_MENU(ports);
}

