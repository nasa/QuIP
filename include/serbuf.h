
#include "typedefs.h"

#define BUFSIZE	1024

typedef struct serial_buffer {
	int	sb_fd;
	int	sb_n_recvd;
	int	sb_n_scanned;
	u_char	sb_buf[BUFSIZE];
} Serial_Buffer;

/* expect.c */

/* consume flag values for read_until_string */
typedef enum {
	PRESERVE_MARKER,
	CONSUME_MARKER
} Consume_Flag;

extern char * printable_string(const char *);
extern char * printable_version(int);
extern int buffered_char(QSP_ARG_DECL  Serial_Buffer *);
extern char * expect_string(QSP_ARG_DECL  Serial_Buffer *,const char *);
extern void _expected_response(QSP_ARG_DECL  Serial_Buffer *,const char *);
#define expected_response(sbp,s) _expected_response(QSP_ARG  sbp,s)

extern void _read_until_string(QSP_ARG_DECL  char *,Serial_Buffer *,const char *,Consume_Flag);
#define read_until_string(dst,sbp,m,f) _read_until_string(QSP_ARG  dst,sbp,m,f)

extern void reset_buffer(Serial_Buffer *);
extern void init_response_buf(int fd);
extern int get_number(QSP_ARG_DECL  Serial_Buffer *);
extern void show_buffer(Serial_Buffer *);
extern int _replenish_buffer(QSP_ARG_DECL  Serial_Buffer *,int max);
#define replenish_buffer(sbp,max) _replenish_buffer(QSP_ARG  sbp,max)


