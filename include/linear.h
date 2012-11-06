
#ifndef DACMAX

#include "data_obj.h"		/* use data_obj's for lintbls */
#include "cmaps.h"		/* pick up defn for N_COMPS */

#define DACMAX			255
#define N_LIN_LVLS		256
#define MAX_LIN_LVLS		2048
#define PHOSMAX			(N_LIN_LVLS-1)

#define DEF_GAM			2.5
#define DEF_VZ			0.0

/* a global */
extern int phosmax;

extern Data_Obj *default_lt_dp;

#define LT_DATA(lt_dp,comp,index)	*(((u_short *)lt_dp->dt_data)+comp*lt_dp->dt_cinc+index*lt_dp->dt_pinc)

extern Data_Obj *	new_lintbl(QSP_ARG_DECL  const char *name);
extern u_short		lintbl_data(Data_Obj *lt_dp, int component, int index);
extern void		set_lintbl(QSP_ARG_DECL  Data_Obj *lt_dp);
#ifdef HAVE_X11
extern void		install_default_lintbl(QSP_ARG_DECL  Dpyable *);
#endif /* HAVE_X11 */
extern void		set_n_linear(QSP_ARG_DECL  int n);

#endif /* ! DACMAX */

