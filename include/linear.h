
#ifndef _LINEAR_H_
#define _LINEAR_H_

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

#define LT_DATA(lt_dp,comp,index)	*(((u_short *)OBJ_DATA_PTR(lt_dp))+comp*OBJ_COMP_INC(lt_dp)+index*OBJ_PXL_INC(lt_dp))

extern Data_Obj *	new_lintbl(QSP_ARG_DECL  const char *name);
extern u_short		lintbl_data(Data_Obj *lt_dp, int component, int index);
extern void		set_lintbl(QSP_ARG_DECL  Data_Obj *lt_dp);
extern void		install_default_lintbl(QSP_ARG_DECL  Dpyable *);
extern void		set_n_linear(QSP_ARG_DECL  int n);

#endif /* ! _LINEAR_H_ */

