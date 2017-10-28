#ifndef _TTY_CTL_H_
#define _TTY_CTL_H_


#ifdef TTY_CTL

#if HAVE_TERMIOS_H
#include <termios.h>
#endif

#include "typedefs.h"
#include "quip_fwd.h"

typedef struct termio_option {
	const char *	to_name;
	long		to_bit;
	const char *	to_enastr;
	const char *	to_disstr;
} Termio_Option;

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
extern void _show_term_flags(QSP_ARG_DECL  u_long flag,Termio_Option *tbl);
extern void _dump_term_flags(QSP_ARG_DECL  u_long flag,Termio_Option *tbl);

#define show_term_flags(flag,tbl) _show_term_flags(QSP_ARG  flag,tbl)
#define dump_term_flags(flag,tbl) _dump_term_flags(QSP_ARG  flag,tbl)

// my_stty.c
extern void set_tty_flag(const char *flagname,int fd,int value);

#endif /* TTY_CTL */

#endif /* ! _TTY_CTL_H_ */
