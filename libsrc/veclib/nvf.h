
/* nvf.h	New Vector Functions */

#ifndef NO_NEW_VEC_FUNC

#ifdef __cplusplus
extern "C" {
#endif

// Should this be a configurable option?
// No quat support in cuda for the moment.
#define QUATERNION_SUPPORT

//#define OLD_STYLE_CONVERSIONS

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	// abs
#endif /* HAVE_STDLIB_H */

#include "quip_config.h"
#include "quip_prot.h"
#include "warn.h"
#include "veclib_api.h"
//#include "new_chains.h"
#include "vec_chain.h"

#ifndef N_PROCESSORS
#define N_PROCESSORS 1
#endif /* undef N_PROCESSORS */


/* globals */
extern const char *argset_type_name[N_ARGSET_TYPES];
extern const Vector_Function *this_vfp;		/* a global */
extern int for_real;
extern const char *type_strings[];		/* see obj_args.c */
extern dimension_t bitrev_size;
extern dimension_t *bitrev_data;
extern int n_processors;


/* vectbl.c */

/* dispatch.c */

extern void _launch_threads(QSP_ARG_DECL void (*func)(HOST_CALL_ARG_DECLS), int vf_code, Vec_Obj_Args oa[]);
#define launch_threads(func, vf_code, oa) _launch_threads(QSP_ARG func, vf_code, oa)

extern COMMAND_FUNC( set_n_processors );

/* vec_args.c */
extern int is_ram(Data_Obj *);
extern void zero_oargs(Vec_Obj_Args *oap);

extern void _do_vfunc( QSP_ARG_DECL   Vector_Function *vfp );
extern void _mixed_location_error(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap);
#define do_vfunc( vfp ) _do_vfunc( QSP_ARG   vfp )
#define mixed_location_error(vfp, oap) _mixed_location_error(QSP_ARG  vfp, oap)

extern int are_ram_args(Vec_Obj_Args *oap);
#ifdef HAVE_CUDA
extern int are_gpu_args(Vec_Obj_Args *oap);
#endif /* HAVE_CUDA */

extern int _check_obj_devices( QSP_ARG_DECL  Vec_Obj_Args *oap );
#define check_obj_devices(oap) _check_obj_devices(QSP_ARG  oap)

/* vec_call.c */
extern int cktype(Data_Obj *dp1,Data_Obj *dp2);

/* vf_menu.c */
extern void _do_vcode(QSP_ARG_DECL   Vec_Func_Code code);
#define do_vcode(code) _do_vcode(QSP_ARG   code)

/* convert.c */
extern void convert(QSP_ARG_DECL  Data_Obj *,Data_Obj *);

/* entries.c */
extern void fft2d( Data_Obj *, Data_Obj * );
extern void fftrows( Data_Obj *, Data_Obj * );
extern void ift2d( Data_Obj *, Data_Obj * );
extern void iftrows( Data_Obj *, Data_Obj * );

/* warmenu.c */
extern void set_perf(int);

/* lin_util.c */
extern int _prodimg(QSP_ARG_DECL  Data_Obj *,Data_Obj *,Data_Obj *);
extern void _transpose(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr);
extern void _vec_xform(QSP_ARG_DECL  Data_Obj *,Data_Obj *,Data_Obj *);
extern void _homog_xform(QSP_ARG_DECL  Data_Obj *,Data_Obj *,Data_Obj *);
extern double _determinant(QSP_ARG_DECL  Data_Obj *dp);
#define prodimg(dpto,dpfr1,dpfr2) _prodimg(QSP_ARG  dpto,dpfr1,dpfr2)
#define transpose(dpto,dpfr) _transpose(QSP_ARG  dpto,dpfr)
#define vec_xform(dpto,dpfr,dpxf) _vec_xform(QSP_ARG  dpto,dpfr,dpxf)
#define homog_xform(dpto,dpfr,dpxf) _homog_xform(QSP_ARG  dpto,dpfr,dpxf)
#define determinant(dp) _determinant(QSP_ARG  dp)

/* wrap.c */
//extern void dp_scroll(QSP_ARG_DECL  Data_Obj *,Data_Obj *,incr_t,incr_t);
//extern void wrap(QSP_ARG_DECL  Data_Obj *,Data_Obj *);

/* obj_args.c */
extern void _private_show_obj_args(QSP_ARG_DECL  char *,const Vec_Obj_Args *,void (*f)(QSP_ARG_DECL  const char *));
#define private_show_obj_args(s,va_p,f) _private_show_obj_args(QSP_ARG  s,va_p,f)

// moved to master include...
//extern void set_obj_arg_flags(Vec_Obj_Args *);

/* cksiz.c */
extern int _cksiz(QSP_ARG_DECL  int,Data_Obj *,Data_Obj *);
extern int _check_bitmap(QSP_ARG_DECL  Data_Obj *,Data_Obj *);
#define cksiz(n,dp1,dp2) _cksiz(QSP_ARG  n,dp1,dp2)
#define check_bitmap(dp1,dp2) _check_bitmap(QSP_ARG  dp1,dp2)

/* sampmenu.c */
extern COMMAND_FUNC( do_samp_menu );

/* typtbl.c */
extern int _check_vfa_tbl_size(QSP_ARG_DECL  Vec_Func_Array vfa_tbl[], int size);
#define check_vfa_tbl_size(vfa_tbl, size) _check_vfa_tbl_size(QSP_ARG  vfa_tbl, size)

#include "veclib_prot.h"

#ifdef __cplusplus
}
#endif


#endif /* ! NO_NEW_VEC_FUNC */

