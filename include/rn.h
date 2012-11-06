
#ifndef RN_H_READ
#define RN_H_READ

#include "typedefs.h"

u_long rn( u_long max );
void set_random_seed(void);
void set_seed(QSP_ARG_DECL  u_long seed);
void rninit(SINGLE_QSP_ARG_DECL);
void scramble(QSP_ARG_DECL  u_long *buf,u_long n);
void permute(QSP_ARG_DECL  u_long *buf,u_long n);


#endif /* under RN_H_READ */
