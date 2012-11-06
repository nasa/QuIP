
#ifndef SERIAL_H

#define SERIAL_H

#include "query.h"

#define RAWBUF_SIZE	2048

typedef struct serial_port {
	char *		sp_name;
	int		sp_fd;
	u_char		sp_rawbuf[RAWBUF_SIZE];
} Serial_Port;

#define NO_SERIAL_PORT ((Serial_Port *)NULL)


/* serial.c */
extern int open_serial_device(QSP_ARG_DECL  const char *);
extern void send_serial(QSP_ARG_DECL  int fd,const u_char *buf,int n);
extern void send_hex(QSP_ARG_DECL  int fd,const u_char *s);
extern int n_serial_chars(QSP_ARG_DECL  int fd);
extern int recv_somex(QSP_ARG_DECL  int fd,u_char *buf,int bufsize,int max_want);
extern int hex_byte(QSP_ARG_DECL  const u_char *);
extern void set_raw_len(u_char *);
extern void dump_char_buf(u_char *);
extern Serial_Port *default_serial_port(void);


#endif /* ! SERIAL_H */

