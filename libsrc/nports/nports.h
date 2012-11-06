
#ifndef _NPORTS_H_
#define _NPORTS_H_

/* This file contains the private stuff shared between files in the module */

#include "quip_config.h"

#ifdef NEED_FAR_POINTERS
#define FAR  far
#else
#define FAR
#endif

#include "items.h"
#include "query.h"

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

#include "nports_api.h"

#define NO_PORT ((struct my_port *)NULL)

#ifdef DEBUG
extern debug_flag_t debug_ports;
#endif /* DEBUG */


/* flag bits */
#define PORT_SERVER	1
#define PORT_CLIENT	2
#define PORT_CONNECTED	16

#define IS_SERVER(mpp)		((mpp)->mp_flags & PORT_SERVER)
#define IS_CLIENT(mpp)		((mpp)->mp_flags & PORT_CLIENT)
#define IS_CONNECTED(mpp)	((mpp)->mp_flags & PORT_CONNECTED)

/* trial and error showd that 140*64 bytes was largest X by 64 image */
#define MAX_PORT_BYTES	9000

/* codes identifying packets */

#define NO_PACKET	((struct d_packet *) NULL)


/* packet code values */
#define P_ERROR		(-1)
#define P_TEXT	0
#define P_DATA	1
#define P_FILE	2
#define MAX_PORT_CODES	3

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

extern Port *get_channel(Port *mpp,int mode);
extern void delport(QSP_ARG_DECL  Port *mpp);
extern void portinfo(Port *mpp);
extern void close_all_ports();

/* xmitrecv.c */
extern const char *recv_text(QSP_ARG_DECL  Port *mpp);
extern int put_port_int32(Port *mpp,int32_t wrd);
extern void xmit_text(QSP_ARG_DECL  Port *mpp,const void *text,int);
extern void if_pipe(int);

/* server.c & client.c */
extern int get_connection(QSP_ARG_DECL  Port *mpp);
extern void nofunc();
extern void set_port_event_func( void (*func)(void) );
extern void port_disable_events(void);
extern void port_enable_events(void);
extern void set_reset_func(void (*func)(void) );
extern void set_recovery(void (*func)(void) );
extern int open_server_port(QSP_ARG_DECL  const char *name,int port_no);
extern int open_client_port(QSP_ARG_DECL  const char *name,const char *hostname,int port_no);
extern int port_listen(Port *mpp);
extern int write_port(Port *mpp,const void *buf,u_long n);
/* extern void null_reset(void); */
extern int relisten(Port *mpp);
extern void close_port(QSP_ARG_DECL  Port *mpp);

/* sockopts.c */
void show_sockopts(Port *mpp);

/* verports.c */
extern void verports(SINGLE_QSP_ARG_DECL);

/* packets.c */
extern void init_pdt_tbl(void);
extern void null_proc(QSP_ARG_DECL  const char *);
extern void null_xmit(QSP_ARG_DECL  Port *,const void *,int);
extern void init_ports(SINGLE_QSP_ARG_DECL);
extern COMMAND_FUNC( do_port_xmit );
// moved to nports_api.h
/* extern int define_port_data_type(QSP_ARG_DECL  int code,const char *my_typename,const char *prompt,
	const char *(*recvfunc)(Port *),
	void (*procfunc)(QSP_ARG_DECL  const char *),
	const char *(*datafunc)(QSP_ARG_DECL  const char *),
	void (*xmitfunc)(Port *,const void *,int) );
	*/
extern int reset_port(QSP_ARG_DECL  Port *mpp);
extern Packet *recv_data(QSP_ARG_DECL  Port *mpp);
extern COMMAND_FUNC( do_port_recv );
extern int have_port_data_type(QSP_ARG_DECL  int code);

#endif /* _NPORTS_H_ */

