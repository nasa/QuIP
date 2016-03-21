
#include "data_obj.h"
#include "query.h"

/* numrec.c */

extern void float_init_rowlist(float **list,Data_Obj *dp);
extern void double_init_rowlist(double **list,Data_Obj *dp);

extern void dp_choldc(Data_Obj *a_dp, Data_Obj *p_dp);
extern void dp_svd(Data_Obj *a_dp,Data_Obj *w_dp,Data_Obj *v_dp);
extern void dp_zroots(Data_Obj *r_dp,Data_Obj *a_dp,int polish);
extern void dp_svbksb(Data_Obj *x_dp,Data_Obj *u_dp,Data_Obj *w_dp,Data_Obj *v_dp,Data_Obj *b_dp);
extern void dp_jacobi(QSP_ARG_DECL  Data_Obj *v_dp,Data_Obj *d_dp,Data_Obj *a_dp,int *);
extern void dp_eigsrt(QSP_ARG_DECL  Data_Obj *v_dp,Data_Obj *d_dp);
extern void dp_moment(QSP_ARG_DECL  Data_Obj *d_dp);

/* nrmenu.c */

extern COMMAND_FUNC( nrmenu );


