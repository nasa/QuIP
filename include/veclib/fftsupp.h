extern dimension_t bitrev_size;
extern dimension_t *bitrev_data;

extern void bitrev_init(dimension_t len);
/*
extern int fft_row_size_ok(QSP_ARG_DECL  Data_Obj *dp);
extern int fft_size_ok(QSP_ARG_DECL  Data_Obj *dp);
extern int fft_col_size_ok(QSP_ARG_DECL  Data_Obj *dp);
*/
extern  int row_fft_ok(QSP_ARG_DECL  Data_Obj *dp, const char *funcname );
extern  int cpx_fft_ok(QSP_ARG_DECL  Data_Obj *dp, const char *funcname );
extern int real_row_fft_ok(QSP_ARG_DECL  Data_Obj *real_dp,Data_Obj *cpx_dp,const char *funcname);
extern int real_fft_type(QSP_ARG_DECL  Data_Obj *real_dp,Data_Obj *cpx_dp,const char *funcname);

