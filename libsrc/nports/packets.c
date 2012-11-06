#include "quip_config.h"

char VersionId_ports_packets[] = QUIP_VERSION_STRING;

#include <stdio.h>
#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif /* HAVE_SIGNAL_H */

#include "nports.h"
#include "debug.h"
#include "data_obj.h"

static Port_Data_Type pdt_tbl[ MAX_PORT_CODES ];
static Port *recv_data_port;

/* local prototypes */
static const char *null_recv(QSP_ARG_DECL  Port *);
static int get_port_data_type(SINGLE_QSP_ARG_DECL);
/* static void rd_pipe(SIGNAL_ARG_TYPE); */	/* not needed? */



static const char *null_recv(QSP_ARG_DECL  Port *mpp)
{ return(NULL); }

void null_xmit(QSP_ARG_DECL  Port *mpp, const void *s,int flag)
{}

void null_proc(QSP_ARG_DECL  const char *s)
{}

#ifdef PC
static void ports_cleanup(void);

/* As a DOS application, this routine does nothing */
/* _WINDOWS needs to call this routine */

static void ports_cleanup(void)
{
#ifdef WINDOWS
	tkdll_cleanup();
#endif /* WINDOWS */
}
#endif /* PC */

static int get_port_data_type(SINGLE_QSP_ARG_DECL)
{
	int i;
	int n;
	const char *typelist[MAX_PORT_CODES];
	Port_Data_Type *pdtp;

	init_ports(SINGLE_QSP_ARG);

	/* build list of valid type names */

	n=0;
	pdtp = pdt_tbl;
	for(i=0;i<MAX_PORT_CODES;i++){
		if( *pdtp->pdt_name ){
			typelist[n++] = pdtp->pdt_name;
		}
		pdtp++;
	}
	if( n <= 0 ){
		WARN("no data types defined");
		return(-1);
	}
	i=WHICH_ONE("data type",n,typelist);
	return(i);
}

COMMAND_FUNC( do_port_xmit )
{
	int type;
	Port_Data_Type *pdtp;
	const void *vp;
	Port *mpp;


	type=get_port_data_type(SINGLE_QSP_ARG);
	mpp = PICK_PORT("");

	if( type >= 0 ) {
		pdtp = &pdt_tbl[type];
		vp=(*(pdtp->data_func))(QSP_ARG  pdtp->pdt_prompt);
	} else {	/* invalid data type */
		WARN("invalid port data type requested");
		pdtp = NULL;
		vp=NULL;
	}

	if( mpp==NO_PORT ) goto oops;

	if( ! IS_CONNECTED(mpp) ){
		sprintf(error_string,"do_port_xmit:  Port %s is not connected",
			mpp->mp_name);
		WARN(error_string);
		goto oops;
	}

	if( vp == NULL || mpp == NO_PORT ){
oops:
		/* eat a word to avoid a syntax error */
		vp = (void *)NAMEOF("something");
		return;
	}

	(*(pdtp->xmit_func))(QSP_ARG  mpp,(char *)vp,1);
}

/* we can't dynamically add types, since codes have to be consistent
 * between separate processes
 */

int define_port_data_type(QSP_ARG_DECL  int code,const char *my_typename,const char *prompt,
	const char * (*recvfunc)(QSP_ARG_DECL  Port *),
	void (*procfunc)(QSP_ARG_DECL const char *),
	const char * (*datafunc)(QSP_ARG_DECL const char *),
	void (*xmitfunc)(QSP_ARG_DECL  Port *,const void *,int) )
{
	init_ports(SINGLE_QSP_ARG);

	if( code < 0 || code >= MAX_PORT_CODES ){
		WARN("invalid port data type code number");
		return(-1);
	}
	pdt_tbl[ code ].pdt_name = my_typename;
	pdt_tbl[ code ].pdt_prompt = prompt;
	pdt_tbl[ code ].recv_func = recvfunc;
	pdt_tbl[ code ].proc_func = procfunc;
	pdt_tbl[ code ].data_func = datafunc;
	pdt_tbl[ code ].xmit_func = xmitfunc;
	return( 0 );
}

#ifdef FOOBAR
/* the signal() call with this arg is commented out - why? */
static void rd_pipe(SIGNAL_ARG_DECL)
{
	portinfo(recv_data_port);
	NERROR1("SIGPIPE on read");
}
#endif /* FOOBAR */

/*
 * This function is called when some error occurs on a port.
 * If it is a server port we will wait for a new connection if the flag is
 * so set.  Otherwise we close.
 */

int reset_port(QSP_ARG_DECL  Port *mpp)
{
	if( IS_SERVER(mpp) ){
		while( get_connection(QSP_ARG  mpp) ==(-1) )
			;
		mpp->mp_sleeptime=500000;	/* default is 0.5 seconds */
		return(1);
	} else {				/* write port */
		close_port(QSP_ARG  mpp);
	}
	return(0);
}

/* 
 * Receive a packet of data of any type.
 */

Packet *recv_data(QSP_ARG_DECL  Port *mpp)
{
	long code;
	static Packet dpk1;
	Packet *pkp=(&dpk1);

#ifdef SIGPIPE
recv_data_port=mpp;
	/* why is this commented out??? */
	/*
	signal(SIGPIPE,rd_pipe);
	*/
#endif

top:
	code=get_port_int32(QSP_ARG  mpp);
	if( code == -1L ){
		if( reset_port(QSP_ARG  mpp) ) goto top;
		else return(NO_PACKET);
	}

#ifdef CAUTIOUS
	if( code < 0 || code >= MAX_PORT_CODES ){
		sprintf(msg_str,"CAUTIOUS:  code = %ld",code);
		prt_msg(msg_str);
		ERROR1("CAUTIOUS:  recv_data:  type code out of range");
	}
#endif /* CAUTIOUS */

	if( *pdt_tbl[code].pdt_name == 0 ){
		sprintf(msg_str,"recv_data:  code is %ld",code);
		prt_msg(msg_str);
		ERROR1("recv_data:  not initialized for this type code");
	}

	pkp->pk_code= (int) code;
	pkp->pk_data = pdt_tbl[code].recv_func(QSP_ARG  mpp);

	if( pkp->pk_data == NULL ){		/* connection broken ? */
		if( reset_port(QSP_ARG  mpp) ) goto top;
		else return(NO_PACKET);
	}

	/* do any additional processing on the data */
	/* but don't process null strings */

	/* what is this typically used for??? */

	if( pkp->pk_code != P_TEXT || *pkp->pk_data )
		(*pdt_tbl[pkp->pk_code].proc_func)(QSP_ARG  pkp->pk_data);

	return(pkp);
}

COMMAND_FUNC( do_port_recv )
{
	Packet *pkp;
	Port *mpp;
	int type;

	mpp = PICK_PORT("");
	if( mpp==NO_PORT ) return;

	type=get_port_data_type(SINGLE_QSP_ARG);

	if( mpp==NO_PORT ) return;
	if( type < 0 ) return;

	if( ! IS_CONNECTED(mpp) ){
		sprintf(error_string,"do_port_recv:  Port %s is not connected",
			mpp->mp_name);
		WARN(error_string);
		return;
	}

	/* Now we should make sure that the port is actually connected... */

	do {
		pkp=recv_data(QSP_ARG  mpp);
		if( pkp == NO_PACKET ) return;

		if( pkp->pk_code == P_ERROR ){
			WARN("read_sock:  data reception error (P_ERROR)");
			return;
		}

		/*
		 * When we would call this to get a handshaking word from a daemon
		 * after the daemon has sent us an image, the image used to get
		 * eaten up instead of the text, and the interpreter gagged on
		 * the name of the variable for text storage...  this should
		 * fix that!?
		 *
		 * A BUG remains, in that if we request to receive data, and
		 * get text instead, the proc_func will prompt for the variable
		 * name, which will not be there...  The variable name should
		 * be gotten here when we know that we want text.
		 */
	} while ( pkp->pk_code != type );

	return;
}

int have_port_data_type(QSP_ARG_DECL  int code)
{
	init_ports(SINGLE_QSP_ARG);
	if( *pdt_tbl[code].pdt_name ) return(1);
	else return(0);
}


void init_pdt_tbl(VOID)
{
	Port_Data_Type *pdtp;
	int i;

	pdtp = pdt_tbl;
	for(i=0;i<MAX_PORT_CODES;i++){
		pdtp->pdt_name="";
		pdtp->pdt_prompt="";
		pdtp->recv_func=null_recv;
		pdtp->proc_func=null_proc;
		pdtp->data_func=nameof;
		pdtp->xmit_func=null_xmit;
		pdtp++;
	}
}

