#ifndef _DOBJ_PRIVATE_H_
#define _DOBJ_PRIVATE_H_

// ascii.c
extern int64_t _next_input_int_with_format(QSP_ARG_DECL   const char *pmpt);
extern double _next_input_flt_with_format(QSP_ARG_DECL  const char *pmpt);
#define next_input_int_with_format(pmpt) _next_input_int_with_format(QSP_ARG   pmpt)
#define next_input_flt_with_format(pmpt) _next_input_flt_with_format(QSP_ARG  pmpt)

extern Data_Obj * _temp_scalar(QSP_ARG_DECL  const char *s, Precision *prec_p);
#define temp_scalar(s,prec_p) _temp_scalar(QSP_ARG  s,prec_p)

#endif // ! _DOBJ_PRIVATE_H_

