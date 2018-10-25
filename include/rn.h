#ifndef _RN_H_
#define _RN_H_

#include "quip_prot.h"

#define SCRAMBLE_PREC PREC_UDI
#define SCRAMBLE_TYPE uint32_t

extern u_long rn(u_long);
extern void rninit(SINGLE_QSP_ARG_DECL);
extern void set_seed(QSP_ARG_DECL  u_long seed);

extern void set_random_seed(SINGLE_QSP_ARG_DECL);

#define SCRAMBLE_PREC PREC_UDI
#define SCRAMBLE_TYPE uint32_t

extern void _permute(QSP_ARG_DECL  SCRAMBLE_TYPE *buf,int n);
extern void _scramble(QSP_ARG_DECL  SCRAMBLE_TYPE* buf,u_long n);
#define permute(buf,n) _permute(QSP_ARG  buf,n)
#define scramble(buf,n) _scramble(QSP_ARG  buf,n)

#endif /* _RN_H_ */

