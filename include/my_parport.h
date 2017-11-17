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

extern ParPort *	_open_parport(QSP_ARG_DECL  const char *name);
#define open_parport(name) _open_parport(QSP_ARG  name)

extern int		_read_til_transition(QSP_ARG_DECL  ParPort *ppp, int mask);
#define read_til_transition(ppp, mask) _read_til_transition(QSP_ARG  ppp, mask)

extern int		_read_parport_status(QSP_ARG_DECL  ParPort *ppp);
#define read_parport_status(ppp) _read_parport_status(QSP_ARG  ppp)

#endif // _MY_PARPORT_H_

