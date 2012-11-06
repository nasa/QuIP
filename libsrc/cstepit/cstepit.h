

#ifdef INC_VERSION
char VersionId_inc_cstepit[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#ifndef VARS

#include "optimize.h"

#define VARS		MAX_OPT_PARAMS
#define VVARS		(VARS+5)
#define DATA 100
#define GRAIN 1000
#define STRING 80



/* stepit.c */

extern int stepit(void (*func)(void));
extern void halt_cstepit(void);
extern void getvals(double *,int);
extern void setvals(double *,int);
extern void setfobj(double);
extern void setminmax(double *,double *,int);
extern void setdelta(double *,double *,int);
extern void settrace(int);
extern void setmaxcalls(int);
extern int reset_n_params(int);

/* quick.c */

extern void fn(void);


/* cs_supp.c */
COMMAND_FUNC( run_cstepit_scr );
void run_cstepit_c(float (*func)(void));

#endif /* undef VARS */

