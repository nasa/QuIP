
#ifdef TTY_CTL

#ifndef TTYCTL_D
#define TTYCTL_D

#if HAVE_TERMIOS_H
#include <termios.h>
#endif

#include "typedefs.h"
#include "query.h"

typedef struct termio_option {
	const char *	to_name;
	long		to_bit;
	const char *	to_enastr;
	const char *	to_disstr;
} Termio_Option;

#define NO_TERM_OPT	((Termio_Option *) NULL )

/* prototypes from termio.c */

extern void set_ndata(int fd,int n);
extern void set_parity(int fd,int flag,int odd);
extern void set_baud(int fd,int rate);
extern void tty_nonl(int fd);
extern void ttyraw(int fd);
extern void ttycbrk(int fd);
extern void ttycook(int fd);
extern void echoon(int fd);
extern void echooff(int fd);
extern void ttynorm(int fd);
extern void waitq(int fd);
extern int keyhit(int fd);
extern int get_erase_chr(int fd);
extern int get_kill_chr(int fd);
extern void show_term_flags(u_long flag,Termio_Option *tbl);
extern void dump_term_flags(u_long flag,Termio_Option *tbl);

/* tty_flags.c */
extern void show_all(int fd);
extern void dump_all(int fd);
extern void set_tty_flag(const char *flagname,int fd,int value);
extern int get_flag_value(QSP_ARG_DECL   const char *flagname);



#endif /* ! TTYCTL_D */
#endif /* TTY_CTL */
