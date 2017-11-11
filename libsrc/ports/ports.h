
#ifndef _NPORTS_H_
#define _NPORTS_H_

/* This file contains the private stuff shared between files in the module */

#include "quip_config.h"

#ifndef FAR
#ifdef NEED_FAR_POINTERS
#define FAR  far
#else
#define FAR
#endif
#endif // ! FAR

#include "quip_prot.h"

#include "data_obj.h"

#if HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#if HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif

#if HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif

#if HAVE_NETDB_H
#include <netdb.h>
#endif

#ifdef BUILD_FOR_WINDOWS
#include <winsock2.h>
#endif // BUILD_FOR_WINDOWS

#include "nports_api.h"

////////////////////////////////////////////////////////////////////

/* sleep_time_t is the type of the arg to usleep.
 * useconds_t if _XOPEN_SOURCE defined to be 500...
 */
//#if _XOPEN_SOURCE == 500
#ifdef _USECONDS_T
#define sleep_time_t		useconds_t
#else // ! _USECONDS_T
#define sleep_time_t		long
#endif // ! _USECONDS_T

struct my_port {
	Item			mp_item;
#define mp_name			mp_item.item_name

	int			mp_sock;	/* current socket */
	int			mp_o_sock;	/* listening socket for server */
	int			mp_portnum;	/* in sockaddr also, but here for convenience... */
	struct sockaddr_in *	mp_addrp;
	short			mp_flags;
	struct my_port *	mp_pp;

	/* number of microseconds to sleep when waiting for data */
	sleep_time_t		mp_sleeptime;
	const char *		mp_text_var_name;
	const char *		mp_output_filename;
	const char *		mp_auth_string;
};

/* trial and error showd that 140*64 bytes was largest X by 64 image */
#define MAX_PORT_BYTES	9000

#define PORT_MAGIC_NUMBER 0x1a2b3c4d

#define PORT_BACKLOG	5		/* see listen(2) */

typedef struct port_data_type {
	Item	pdt_item;
#define pdt_name	pdt_item.item_name

	Packet_Type	pdt_code;

	const char *pdt_prompt;
	const char *(*data_func)(QSP_ARG_DECL  const char *);	/* prompt user for data */
	long (*recv_func)(QSP_ARG_DECL  Port *, Packet *pkp);		/* receive data from other process */
	void (*xmit_func)(QSP_ARG_DECL  Port *,const void *,int);	/* transmit data to other process */

} Port_Data_Type ;

struct d_packet {
	Port_Data_Type	*pk_pdt;	/* text, object, file, etc. */
	long		pk_size;
	char *		pk_data;	// pointer to free
	char *		pk_user_data;	// skip possible prefix
	void *		pk_extra;	// point to a dp?
};

#define MIN_MP_SLEEPTIME	10		/* 10 microseconds */
#define MAX_MP_SLEEPTIME	1000000		/* 1 second */

////////////////////////////////////////////////////////////////////

extern ITEM_CHECK_PROT(Port_Data_Type,pdt)
#define pdt_of(s)	_pdt_of(QSP_ARG  s)

#ifdef HAVE_ENCRYPTION
#include "my_encryption.h"
#endif /* HAVE_ENCRYPTION */

#ifdef QUIP_DEBUG
extern debug_flag_t debug_ports;
#endif /* QUIP_DEBUG */


/* flag bits */
#define PORT_SERVER	1
#define PORT_CLIENT	2
#define PORT_REDIR	4	// we have redirected to this port
#define PORT_ZOMBIE	8	// sockets are close, waiting to free struct
#define PORT_CONNECTED	16
#define PORT_SECURE	32	// only accept encrypted packets
				// and insist on authentication
#define PORT_AUTHENTICATED	64	// valid authentication packet received
#define PORT_RECEIVING_PACKET	128
#define PORT_NEEDS_RESET	256	// reset port at opportune time

#define IS_SERVER(mpp)		((mpp)->mp_flags & PORT_SERVER)
#define IS_CLIENT(mpp)		((mpp)->mp_flags & PORT_CLIENT)
#define IS_CONNECTED(mpp)	((mpp)->mp_flags & PORT_CONNECTED)
#define IS_SECURE(mpp)		((mpp)->mp_flags & PORT_SECURE)
#define IS_AUTHENTICATED(mpp)	((mpp)->mp_flags & PORT_AUTHENTICATED)
#define IS_RECEIVING_PACKET(mpp)	((mpp)->mp_flags & PORT_RECEIVING_PACKET)
#define NEEDS_AUTH(mpp)		(IS_SECURE(mpp) && !IS_AUTHENTICATED(mpp))
#define NEEDS_RESET(mpp)	((mpp)->mp_flags & PORT_NEEDS_RESET)

#define SET_PORT_FLAG_BITS(mpp,bits)	(mpp)->mp_flags |= bits
#define CLEAR_PORT_FLAG_BITS(mpp,bits)	(mpp)->mp_flags &= ~(bits)

/* trial and error showd that 140*64 bytes was largest X by 64 image */
#define MAX_PORT_BYTES	9000

/* codes identifying packets */

#define PORT_BACKLOG	5		/* see listen(2) */

#define BAD_PORT_LONG	(0xffffffff)

#define MIN_MP_SLEEPTIME	10		/* 10 microseconds */
#define MAX_MP_SLEEPTIME	1000000		/* 1 second */


/* dataport.c */

#ifdef CRAY
#define CONV_LEN	2048
extern void cray2ieee(void *cbuf, float *p, int n);
extern void ieee2cray(float *p, void *cbuf, int n);
#endif /* CRAY */


/* ports.c */
ITEM_INTERFACE_PROTOTYPES(Port,port)

#define new_port(s)	_new_port(QSP_ARG  s)
#define del_port(s)	_del_port(QSP_ARG  s)
#define port_of(s)	_port_of(QSP_ARG  s)
#define list_ports(fp)	_list_ports(QSP_ARG  fp)

#define pick_port(pmpt)	_pick_port(QSP_ARG  pmpt)

extern Port *get_channel(Port *mpp,int mode);
extern void delport(QSP_ARG_DECL  Port *mpp);
extern void portinfo(QSP_ARG_DECL  Port *mpp);
//extern void close_all_ports();

/* xmitrecv.c */
extern int check_port_data(QSP_ARG_DECL  Port *mpp, uint32_t usecs);
extern long recv_text(QSP_ARG_DECL  Port *mpp, Packet *pkp);
extern void xmit_text(QSP_ARG_DECL  Port *mpp,const void *text,int);
extern long recv_plain_file(QSP_ARG_DECL  Port *mpp, Packet *pkp);
extern long recv_filename(QSP_ARG_DECL  Port *mpp, Packet *pkp);
extern void xmit_plain_file(QSP_ARG_DECL  Port *mpp,const void *local_filename,int);
extern void xmit_filename(QSP_ARG_DECL  Port *mpp,const void *local_filename,int);
extern void xmit_file_as_text(QSP_ARG_DECL  Port *mpp,const void *local_filename,int);
#ifdef HAVE_ENCRYPTION
extern long recv_enc_file(QSP_ARG_DECL  Port *mpp, Packet *pkp);
extern void xmit_enc_file(QSP_ARG_DECL  Port *mpp,const void *local_filename,int);
extern void xmit_enc_file_as_text(QSP_ARG_DECL  Port *mpp,const void *local_filename,int);
extern long recv_enc_text(QSP_ARG_DECL  Port *mpp, Packet *pkp);
extern void xmit_enc_text(QSP_ARG_DECL  Port *mpp,const void *text,int);
extern void xmit_auth(QSP_ARG_DECL  Port *mpp,const void *text,int);
#endif /* HAVE_ENCRYPTION */

extern void if_pipe(int);

/* server.c & client.c */
extern void set_max_client_retries(QSP_ARG_DECL  int n);
extern int get_connection(QSP_ARG_DECL  Port *mpp);
extern void nofunc(void);
extern void set_port_event_func( void (*func)(void) );
extern void port_disable_events(void);
extern void port_enable_events(void);
extern void set_reset_func(void (*func)(void) );
extern void set_recovery(void (*func)(void) );
extern int open_server_port(QSP_ARG_DECL  const char *name,int port_no);
extern int open_client_port(QSP_ARG_DECL  const char *name,const char *hostname,int port_no);
/* extern void null_reset(void); */
extern int relisten(Port *mpp);
extern void close_port(QSP_ARG_DECL  Port *mpp);

/* sockopts.c */
void show_sockopts(QSP_ARG_DECL  Port *mpp);

/* verports.c */
extern void verports(SINGLE_QSP_ARG_DECL);

/* packets.c */
extern void null_xmit(QSP_ARG_DECL  Port *,const void *,int);
extern COMMAND_FUNC( do_set_text_var );
extern COMMAND_FUNC( do_set_port_output_file );
extern COMMAND_FUNC( do_port_xmit );
extern int reset_port(QSP_ARG_DECL  Port *mpp);
extern Packet *recv_data(QSP_ARG_DECL  Port *mpp);
extern COMMAND_FUNC( do_port_recv );
extern int have_port_data_type(QSP_ARG_DECL  int code);
extern void receive_port_file( QSP_ARG_DECL  Port *mpp, Packet *pkp );

#define PORT_DATATYPE_NAME(pdtp)	((pdtp)->pdt_name)


#endif /* _NPORTS_H_ */

