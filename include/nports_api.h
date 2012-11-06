
/* This file contains the public prototypes and definitions for the nports lib */



#ifndef _NPORTS_API_H_
#define _NPORTS_API_H_

#include "quip_config.h"

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

#include "items.h"
#include "query.h"
#include "data_obj.h"

typedef struct my_port {
	Item			mp_item;
#define mp_name			mp_item.item_name

	int			mp_sock;	/* current socket */
	int			mp_o_sock;	/* listening socket for server */
	int			mp_portnum;	/* in sockaddr also, but here for convenience... */
	struct sockaddr_in *	mp_addrp;
	short			mp_flags;
	struct my_port *	mp_pp;
	long			mp_sleeptime;	/* number of microseconds to sleep when waiting for data */
} Port;

#define NO_PORT ((struct my_port *)NULL)


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
typedef struct d_packet {
	int		pk_code;		/* text or object */
	const char *	pk_data;
} Packet;

#define NO_PACKET	((struct d_packet *) NULL)


/* packet code values */
#define P_ERROR		(-1)
#define P_TEXT	0
#define P_DATA	1
#define P_FILE	2
#define MAX_PORT_CODES	3

#define PORT_BACKLOG	5		/* see listen(2) */

#define BAD_PORT_LONG	(0xffffffff)

typedef struct port_data_type {
	Item	pdt_item;
#define pdt_name	pdt_item.item_name

	const char *pdt_prompt;
	const char *(*data_func)(QSP_ARG_DECL  const char *);	/* prompt user for data */
	void (*proc_func)(QSP_ARG_DECL  const char *);	/* post-processing of received data */
	const char *(*recv_func)(QSP_ARG_DECL  Port *);		/* receive data from other process */
	void (*xmit_func)(QSP_ARG_DECL  Port *,const void *,int);	/* transmit data to other process */

} Port_Data_Type ;

#define MIN_MP_SLEEPTIME	10		/* 10 microseconds */
#define MAX_MP_SLEEPTIME	1000000		/* 1 second */



/* Public prototypes */

extern int write_port(Port *mpp,const void *buf,u_long n);
extern int put_port_int32(Port *mpp,int32_t wrd);
extern int read_port(QSP_ARG_DECL  Port *mpp,void *buf,u_long n);
extern int32_t get_port_int32(QSP_ARG_DECL  Port *mpp);
extern int define_port_data_type(QSP_ARG_DECL  int code,const char *my_typename,const char *prompt,
	const char *(*recvfunc)(QSP_ARG_DECL  Port *),
	void (*procfunc)(QSP_ARG_DECL  const char *),
	const char *(*datafunc)(QSP_ARG_DECL  const char *),
	void (*xmitfunc)(QSP_ARG_DECL  Port *,const void *,int) );
extern Data_Obj * recv_obj(QSP_ARG_DECL  Port *mpp);
extern void xmit_obj(QSP_ARG_DECL  Port *mpp,Data_Obj *dp,int dataflag);
extern void null_proc(QSP_ARG_DECL  const char *);

/* xmitrecv.c */
extern void if_pipe(int);


#endif /* _NPORTS_API_H_ */

