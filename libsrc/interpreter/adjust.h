#ifndef NO_ADJUST

#include "typedefs.h"
#include "query.h"

extern int adjusting;

extern float adj_val(void);
extern void setaps(float loval,float hival,
	float start, float incr,
	float maxincr,float minincr,float startincr);
extern void setup_adjuster(float *flist,int n);
extern void set_adj_func(void (*func)(QSP_ARG_DECL float));
extern void set_mouse_func(void (*func)(void));
extern COMMAND_FUNC( do_adjust );

#define NO_ADJUST

#endif /* NO_ADJUST */

