
#include <gsl/gsl_matrix.h>
#include "data_obj.h"

/* gslmenu.c */

extern gsl_matrix init_gsl_matrix_from_data_obj(Data_Obj *dp);
extern void gsl_svd(QSP_ARG_DECL  Data_Obj *a_dp, Data_Obj *w_dp, Data_Obj *v_dp);
extern void gsl_solve(QSP_ARG_DECL  Data_Obj *x_dp, Data_Obj *u_dp, Data_Obj *w_dp,
						Data_Obj *v_dp, Data_Obj *b_dp);

