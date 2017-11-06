#ifndef _SIGPUSH_H_
#define _SIGPUSH_H_

#include "quip_prot.h"

void inhibit_sigs(void);

void _sigpush(QSP_ARG_DECL  int sig,void (*action)(int));
void _sigpop(QSP_ARG_DECL  int sig);

#define sigpush(sig,action) _sigpush(QSP_ARG  sig,action)
#define sigpop(sig) _sigpop(QSP_ARG  sig)

#endif // _SIGPUSH_H_

