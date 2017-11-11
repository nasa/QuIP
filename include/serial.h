
#ifndef SERIAL_H

#define SERIAL_H

#define RAWBUF_SIZE	2048

typedef struct serial_port {
	char *		sp_name;
	int		sp_fd;
	u_char		sp_rawbuf[RAWBUF_SIZE];
} Serial_Port;

/* serial.c */
extern void set_raw_len(u_char *);
extern Serial_Port *default_serial_port(void);

extern int _open_serial_device(QSP_ARG_DECL  const char *);
extern void _send_serial(QSP_ARG_DECL  int fd,const u_char *buf,int n);
extern void _send_hex(QSP_ARG_DECL  int fd,const u_char *s);
extern int _n_serial_chars(QSP_ARG_DECL  int fd);
extern ssize_t _recv_somex(QSP_ARG_DECL  int fd,u_char *buf,int bufsize,int max_want);
extern int _hex_byte(QSP_ARG_DECL  const u_char *);
extern void _dump_char_buf(QSP_ARG_DECL  u_char *);

#define open_serial_device(s) _open_serial_device(QSP_ARG  s)
#define send_serial(fd,buf,n) _send_serial(QSP_ARG  fd,buf,n)
#define send_hex(fd,s) _send_hex(QSP_ARG  fd,s)
#define n_serial_chars(fd) _n_serial_chars(QSP_ARG  fd)
#define recv_somex(fd,buf,bufsize,max_want) _recv_somex(QSP_ARG  fd,buf,bufsize,max_want)
#define hex_byte(s) _hex_byte(QSP_ARG  s)
#define dump_char_buf(s) _dump_char_buf(QSP_ARG  s)

#endif /* ! SERIAL_H */

