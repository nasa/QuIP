
#include "data_obj.h"

/* numrec.c */

extern void float_init_rowlist(float **list,Data_Obj *dp);
extern void double_init_rowlist(double **list,Data_Obj *dp);

extern void dp_choldc(Data_Obj *a_dp, Data_Obj *p_dp);
extern void dp_svd(Data_Obj *a_dp,Data_Obj *w_dp,Data_Obj *v_dp);
extern void dp_zroots(Data_Obj *r_dp,Data_Obj *a_dp,int polish);
extern void dp_svbksb(Data_Obj *x_dp,Data_Obj *u_dp,Data_Obj *w_dp,Data_Obj *v_dp,Data_Obj *b_dp);

extern void _dp_jacobi(QSP_ARG_DECL  Data_Obj *v_dp,Data_Obj *d_dp,Data_Obj *a_dp,int *);
extern void _dp_eigsrt(QSP_ARG_DECL  Data_Obj *v_dp,Data_Obj *d_dp);
extern void _dp_moment(QSP_ARG_DECL  Data_Obj *d_dp);

#define dp_jacobi(v_dp,d_dp,a_dp,n) _dp_jacobi(QSP_ARG  v_dp,d_dp,a_dp,n)
#define dp_eigsrt(v_dp,d_dp) _dp_eigsrt(QSP_ARG  v_dp,d_dp)
#define dp_moment(d_dp) _dp_moment(QSP_ARG  d_dp)

/* nrmenu.c */

extern COMMAND_FUNC( nrmenu );


