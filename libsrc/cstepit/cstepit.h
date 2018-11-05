

#ifndef VARS

#include "optimize.h"

#define VARS		MAX_OPT_PARAMS
#define VVARS		(VARS+5)
#define DATA 100
#define GRAIN 1000
#define STRING 80



/* stepit.c */

extern int _stepit(QSP_ARG_DECL  void (*func)(void));
#define stepit(func) _stepit(QSP_ARG  func)

extern void halt_cstepit(void);
extern void getvals(double *,int);
extern void setfobj(double);
extern void settrace(int);
extern void setmaxcalls(int);

extern void _set_opt_param_vals(QSP_ARG_DECL  double *,int);
extern void _set_opt_param_minmax(QSP_ARG_DECL  double *,double *,int);
extern void _set_opt_param_delta(QSP_ARG_DECL  double *,double *,int);
extern int _reset_n_opt_params(QSP_ARG_DECL  int);

#define set_opt_param_vals(d,n) _set_opt_param_vals(QSP_ARG  d,n)
#define set_opt_param_minmax(d1,d2,n) _set_opt_param_minmax(QSP_ARG  d1,d2,n)
#define set_opt_param_delta(d1,d2,n) _set_opt_param_delta(QSP_ARG  d1,d2,n)
#define reset_n_opt_params(n) _reset_n_opt_params(QSP_ARG  n)

/* quick.c */

extern void fn(void);


/* cs_supp.c */
COMMAND_FUNC( run_cstepit_scr );
void run_cstepit_c(QSP_ARG_DECL  float (*func)(SINGLE_QSP_ARG_DECL));

#endif /* undef VARS */

