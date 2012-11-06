
#include <gsl/gsl_matrix.h>
#include "data_obj.h"
#include "query.h"

/* gslmenu.c */

extern gsl_matrix init_gsl_matrix_from_data_obj(Data_Obj *dp);
extern void gsl_svd(Data_Obj *a_dp, Data_Obj *w_dp, Data_Obj *v_dp);

