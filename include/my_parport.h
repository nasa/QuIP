#ifndef _MY_PARPORT_H_
#define _MY_PARPORT_H_

#include "item_type.h"

typedef struct parport {
	char *	pp_name;
	int	pp_fd;
} ParPort;

ITEM_INTERFACE_PROTOTYPES(ParPort,parport)

#define new_parport(s)		_new_parport(QSP_ARG  s)
#define del_parport(s)		_del_parport(QSP_ARG  s)
#define parport_of(s)		_parport_of(QSP_ARG  s)

extern ParPort *	open_parport(QSP_ARG_DECL  const char *name);
extern int		read_parport_status(ParPort *ppp);
extern int		read_til_transition(ParPort *ppp, int mask);

#endif // _MY_PARPORT_H_

