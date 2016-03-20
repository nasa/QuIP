
/* stuff from sched.c in jbm lib */

#include "query_stack.h"

#ifdef ALLOW_RT_SCHED
extern int try_rt_sched;
extern int rt_is_on;
#endif

extern void rt_sched(QSP_ARG_DECL  int flag);

