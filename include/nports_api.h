
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

#include "item_type.h"
#include "query.h"
#include "data_obj.h"

struct my_port;
typedef struct my_port Port;

struct d_packet;
typedef struct d_packet Packet;

// used in fileio module...

/* packet code values */
typedef enum {
	P_TEXT,			// 0
	P_DATA,			// 1
	P_IMG_FILE,		// 2
	P_PLAIN_FILE,		// 3
	P_FILE_AS_TEXT,		// 4
	P_ENCRYPTED_TEXT,	// 5
	P_ENCRYPTED_FILE,	// 6
	P_ENC_FILE_AS_TEXT,	// 7
	P_AUTHENTICATION,	// 8
	P_FILENAME,		// 9
	MAX_PORT_CODES		// must be last
} Packet_Type;


#define BAD_PORT_LONG	(0xffffffff)

/* Public prototypes */

extern ssize_t write_port(QSP_ARG_DECL  Port *mpp,const void *buf,u_long n);
extern int put_port_int32(QSP_ARG_DECL  Port *mpp,int32_t wrd);
extern ssize_t read_port(QSP_ARG_DECL  Port *mpp,void *buf,u_long n);
extern int32_t get_port_int32(QSP_ARG_DECL  Port *mpp);
extern int define_port_data_type(QSP_ARG_DECL  int code,const char *my_typename,const char *prompt,
	//const char *(*recvfunc)(QSP_ARG_DECL  Port *),
	long (*recvfunc)(QSP_ARG_DECL  Port *, Packet *pkp),
	const char *(*datafunc)(QSP_ARG_DECL  const char *),
	void (*xmitfunc)(QSP_ARG_DECL  Port *,const void *,int) );
extern long recv_obj(QSP_ARG_DECL  Port *mpp, Packet *pkp);
extern void xmit_obj(QSP_ARG_DECL  Port *mpp,Data_Obj *dp,int dataflag);
extern void null_proc(QSP_ARG_DECL  const char *);

/* xmitrecv.c */
#ifdef FOOBAR
extern void if_pipe(int);
#endif // FOOBAR

#ifdef BUILD_FOR_IOS
extern void test_reachability(QSP_ARG_DECL  const char *s);
#endif /* BUILD_FOR_IOS */

#endif /* _NPORTS_API_H_ */

