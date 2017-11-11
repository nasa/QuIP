#include "quip_config.h"

#include <stdio.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif /* HAVE_UNISTD_H */

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif /* HAVE_SIGNAL_H */

#include "ports.h"
#include "debug.h"
#include "data_obj.h"

static Item_Type *pdt_itp=NULL;
static ITEM_INIT_FUNC(Port_Data_Type,pdt,0)
ITEM_CHECK_FUNC(Port_Data_Type,pdt)
static ITEM_NEW_FUNC(Port_Data_Type,pdt)
static ITEM_PICK_FUNC(Port_Data_Type,pdt)
static ITEM_ENUM_FUNC(Port_Data_Type,pdt)

#define new_pdt(s)	_new_pdt(QSP_ARG  s)
#define pick_pdt(p)	_pick_pdt(QSP_ARG  p)
#define pdt_list()	_pdt_list(SINGLE_QSP_ARG)

Packet *last_packet=NULL;

#ifdef FOOBAR
static const char *null_recv(QSP_ARG_DECL  Port *mpp)
{ return(NULL); }
#endif // FOOBAR

static Packet *the_packet=NULL;	// BUG not thread-safe!

static Packet *available_packet(void)
{
	if( the_packet == NULL ){
		the_packet = getbuf(sizeof(*the_packet));
		the_packet->pk_data = NULL;
	}

	// Freeing this here lets us be sloppy elsewhere...
	if( the_packet->pk_data != NULL ){
		givbuf((char *)(the_packet->pk_data));
	}

	// Initialize the packet fields
	the_packet->pk_pdt = NULL;
	the_packet->pk_size = 0;
	the_packet->pk_data = NULL;
	the_packet->pk_user_data = NULL;
	the_packet->pk_extra = NULL;

	return the_packet;
}

void null_xmit(QSP_ARG_DECL  Port *mpp, const void *s,int flag)
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



static void set_port_text_var(QSP_ARG_DECL  Port *mpp, const char *s )
{
	if( mpp->mp_text_var_name != NULL ){
		sprintf(ERROR_STRING,
	"set_port_text_var %s:  previous variable name '%s' never used!?",
			mpp->mp_name, mpp->mp_text_var_name);
		warn(ERROR_STRING);
		rls_str(mpp->mp_text_var_name);
	}
	mpp->mp_text_var_name=savestr(s);
}

COMMAND_FUNC( do_set_text_var )
{
	Port *mpp;
	const char *s;

	mpp = pick_port("");
	s=NAMEOF("variable name for storage of next text");

	if( mpp == NULL ) return;

	set_port_text_var(QSP_ARG  mpp,s);
}

static void set_port_output_file(QSP_ARG_DECL  Port *mpp, const char *s )
{
	if( mpp->mp_output_filename != NULL ){
		sprintf(ERROR_STRING,
	"set_port_output_file %s:  previous output filename '%s' never used!?",
			mpp->mp_name, mpp->mp_output_filename);
		warn(ERROR_STRING);
		rls_str(mpp->mp_output_filename);
	}
	mpp->mp_output_filename=savestr(s);
}

COMMAND_FUNC( do_set_port_output_file )
{
	Port *mpp;
	const char *s;

	mpp = pick_port("");
	s=NAMEOF("local filename for next received file");

	if( mpp == NULL ) return;

	set_port_output_file(QSP_ARG  mpp,s);
}


COMMAND_FUNC( do_port_xmit )
{
	Port_Data_Type *pdtp;
	const void *vp;
	Port *mpp;

	pdtp=pick_pdt("");
	mpp = pick_port("");

	if( pdtp==NULL || mpp==NULL ) goto oops;

	vp=(*(pdtp->data_func))(QSP_ARG  pdtp->pdt_prompt);

	if( ! IS_CONNECTED(mpp) ){
		sprintf(ERROR_STRING,"do_port_xmit:  Port %s is not connected",
			mpp->mp_name);
		warn(ERROR_STRING);
		goto oops;
	}

	if( vp == NULL || mpp == NULL ){
oops:
		/* eat a word to avoid a syntax error */
		/*vp = (void *)*/ NAMEOF("something");
		return;
	}

	(*(pdtp->xmit_func))(QSP_ARG  mpp,(char *)vp,1);

	// If there is a write error, the port may need to be reset.
	// If it is a server port, then it will listen for a new connection,
	// otherwise it will close itself...
	if( NEEDS_RESET(mpp) ){
		sprintf(ERROR_STRING,"Resetting port %s after write error...",
			mpp->mp_name);
		advise(ERROR_STRING);
		reset_port(QSP_ARG  mpp);
	}
}

/* we can't dynamically add types, since codes have to be consistent
 * between separate processes
 */

int _define_port_data_type(QSP_ARG_DECL  int code,const char *my_typename,const char *prompt,
	//const char * (*recvfunc)(QSP_ARG_DECL  Port *),
	long (*recvfunc)(QSP_ARG_DECL  Port *, Packet *pkp),
	const char * (*datafunc)(QSP_ARG_DECL const char *),
	void (*xmitfunc)(QSP_ARG_DECL  Port *,const void *,int) )
{
	Port_Data_Type *pdtp;

#ifdef CAUTIOUS
	// If this happens, it's a programming error...
	if( code < 0 || code >= MAX_PORT_CODES ){
		sprintf(ERROR_STRING,"CAUTIOUS:  define_port_data_type:  invalid port data type code (%d), should be in the range 0-%d",
			code,MAX_PORT_CODES-1);
		warn(ERROR_STRING);
		return(-1);
	}
#endif /* CAUTIOUS */

#ifdef CAUTIOUS
	pdtp = pdt_of(my_typename);
	if( pdtp != NULL ){
		sprintf(ERROR_STRING,
	"CAUTIOUS:  define_port_data_type:  %s already defined!?",
			PORT_DATATYPE_NAME(pdtp));
		warn(ERROR_STRING);
		return -1;
	}
#endif /* CAUTIOUS */

	pdtp = new_pdt(my_typename);
#ifdef CAUTIOUS
	if( pdtp == NULL ){
		sprintf(ERROR_STRING,
"CAUTIOUS:  define_port_data_type:  couldn't create new data type for %s!?",
			my_typename);
		warn(ERROR_STRING);
		return -1;
	}
#endif /* CAUTIOUS */

	pdtp->pdt_code = code;
	pdtp->pdt_prompt = prompt;	// do we need to save it?
					// we assume not, assume
					// passed as static string...
	pdtp->recv_func = recvfunc;
	//pdtp->proc_func = procfunc;
	pdtp->data_func = datafunc;
	pdtp->xmit_func = xmitfunc;

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
		// If we call this too many times, we get
		// a system error "too many open files" !?
		// BUG need to do something special for windows?
		if( close(mpp->mp_sock) < 0 ){
			tell_sys_error("close");
		}

		mpp->mp_flags &= ~(PORT_AUTHENTICATED|PORT_CONNECTED);

sprintf(ERROR_STRING,"reset_port:  listening for a new connection on port %s",
mpp->mp_name);
advise(ERROR_STRING);

		while( get_connection(QSP_ARG  mpp) ==(-1) )
			;
		mpp->mp_sleeptime=500000;	/* default is 0.5 seconds */
		return(1);
	} else {				/* write port */
		close_port(QSP_ARG  mpp);
	}
	return(0);
}

static Port_Data_Type *port_data_type_for_code(QSP_ARG_DECL  Packet_Type code)
{
	List *lp;
	Node *np;
	Port_Data_Type *pdtp;

	lp = pdt_list();
	if( lp == NULL ) return NULL;
	np=QLIST_HEAD(lp);
	while(np!=NULL){
		pdtp=(Port_Data_Type *)NODE_DATA(np);
		if( pdtp->pdt_code == code )
			return pdtp;
		np = NODE_NEXT(np);
	}
	return NULL;
}

/* 
 * Receive a packet of data of any type.
 */

Packet *recv_data(QSP_ARG_DECL  Port *mpp)
{
	int32_t code;
	Packet *pkp;
	Port_Data_Type *pdtp;

	pkp = available_packet();

#ifdef SIGPIPE
	/* why is this commented out??? */
	/*
	signal(SIGPIPE,rd_pipe);
	*/
#endif /* SIGPIPE */

top:
	// receive a magic word before the port data code
	code=get_port_int32(mpp);
	if( code == -1L ){
		// We don't log this, this is what happens
		// when an upload client closes.
		//log_message("Missing magic number, resetting port");
		if( reset_port(QSP_ARG  mpp) ) goto top;
		else return(NULL);
	}

	if( code != PORT_MAGIC_NUMBER ){
		sprintf(MSG_STR,"Bad magic number (0x%x), resetting port",
			code);
		log_message(MSG_STR);
		if( reset_port(QSP_ARG  mpp) ) goto top;
		else return(NULL);
	}

	SET_PORT_FLAG_BITS(mpp,PORT_RECEIVING_PACKET);

	code=get_port_int32(mpp);
	if( code == -1L ){
		log_message("Missing packet code, resetting port");
		if( reset_port(QSP_ARG  mpp) ) goto top;
		else return(NULL);
	}

	pdtp = port_data_type_for_code(QSP_ARG  code);

	// This could happen either because of an out-of-range
	// code value, or a valid code for which the receiver
	// has not been initialized...

	if( pdtp == NULL ) {
		// shouldn't happen
		sprintf(MSG_STR,
			"Unrecognized packet code (0x%x), resetting port",code);
		log_message(MSG_STR);
		if( reset_port(QSP_ARG  mpp) ) goto top;
		else return(NULL);
	}

	// error msg printed from port_data_type_for_code...

	pkp->pk_pdt= pdtp;
	pkp->pk_size = pdtp->recv_func(QSP_ARG  mpp, pkp);
	//pkp->pk_data = buf;

	CLEAR_PORT_FLAG_BITS(mpp,PORT_RECEIVING_PACKET);

	if( pkp->pk_size <= 0 ){	/* connection broken ? */
		log_message("Bad packet data, resetting port");
		if( reset_port(QSP_ARG  mpp) ) goto top;
		else {
//advise("recv_data:  reset failed, returning NULL");
			return(NULL);
		}
	}

	/* do any additional processing on the data */
	/* but don't process null strings */

	/* what is this typically used for??? */

	/* We use proc_func to prompt the user for a variable name
	 * for text storage (in the case of P_TEXT).  There is a problem
	 * with this, in that it gets called base on what type of
	 * packet is received, not what is requested in the command,
	 * and this makes the command syntax depend on packet type.
	 * It would be better to set the text var with a separate command.
	 * The same applies to output filename when a plain file is received.
	 * What other types of processing do we need to do?
	 */

	return(pkp);
}

static void save_data_to_file(QSP_ARG_DECL  const char *filename, const char *buf, size_t size)
{
	FILE *fp;

	fp = try_open(filename,"w");
	if( !fp ) return;

	if( fwrite(buf,1,size,fp) != size ){
		sprintf(ERROR_STRING,
	"save_data_to_file:  fwrite error!?");
		warn(ERROR_STRING);
	}
	fclose(fp);
}

void receive_port_file( QSP_ARG_DECL  Port *mpp, Packet *pkp )
{
	if( mpp->mp_output_filename != NULL ){
		save_data_to_file(QSP_ARG  mpp->mp_output_filename,
				pkp->pk_user_data, pkp->pk_size);
		rls_str(mpp->mp_output_filename);
		mpp->mp_output_filename=NULL;	// single use!
	} else {
warn("receive_port_file:  No output filename specified!?\n   --> Use port_output_file command");
	}
}

COMMAND_FUNC( do_port_recv )
{
	Packet *pkp;
	Port *mpp;
	//int type;
	Port_Data_Type *pdtp;

	mpp = pick_port("");

	// We specify what type of packet we want - why not just
	// take what we get?
	pdtp = pick_pdt("");

	if( mpp==NULL || pdtp==NULL ) return;

	if( ! IS_CONNECTED(mpp) ){
		sprintf(ERROR_STRING,"do_port_recv:  Port %s is not connected",
			mpp->mp_name);
		warn(ERROR_STRING);
		return;
	}

	/* Now we should make sure that the port is actually connected... */

	/* memory leak? - where do we free the packets?
	 * Answer:  there is only one, statically allocated packet
	 * that is declared inside recv_data.
	 * BUG should be per-qsp for thread safety!
	 */

	do {
		pkp=recv_data(QSP_ARG  mpp);
		if( pkp == NULL ){
			return;
		}
		if( pkp->pk_pdt != pdtp ){
			sprintf(ERROR_STRING,
		"Received %s packet after request for %s!?",
				PORT_DATATYPE_NAME(pkp->pk_pdt),PORT_DATATYPE_NAME(pdtp));
			warn(ERROR_STRING);
		}

	} while ( pkp->pk_pdt != pdtp );

	/* Now we have the packet - how do we decide what to do with it? */
	if( pkp->pk_pdt->pdt_code == P_TEXT
			|| pkp->pk_pdt->pdt_code == P_ENCRYPTED_TEXT ){
		if( mpp->mp_text_var_name != NULL ){
			// Save the text to a variable
			assign_var(mpp->mp_text_var_name,(const char *)pkp->pk_user_data);
			rls_str(mpp->mp_text_var_name);
			mpp->mp_text_var_name=NULL;	// single use!
		} else {
warn("do_port_recv text:  No variable name specified!?\n   --> Use text_variable command");
		}
	} else if( pkp->pk_pdt->pdt_code == P_PLAIN_FILE ||
			pkp->pk_pdt->pdt_code == P_ENCRYPTED_FILE ){
		receive_port_file(QSP_ARG  mpp,pkp);
	} else if( pkp->pk_pdt->pdt_code == P_DATA ){
		// data objects are added to the local database
		// as they are received, nothing to do.
	} else {
		sprintf(ERROR_STRING,
"do_port_recv:  no action implemented for %s packet!?",
				PORT_DATATYPE_NAME(pkp->pk_pdt));
		warn(ERROR_STRING);
	}
}

