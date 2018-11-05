
#ifndef _VECLIB_API_H_
#define _VECLIB_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "veclib/vec_func.h"
#include "veclib/obj_args.h"
#include "veclib/slow_len.h"

extern COMMAND_FUNC(do_comp_menu);
extern COMMAND_FUNC( do_vl_menu );

extern void setvarg1(Vec_Obj_Args *oap, Data_Obj *dp);
extern void setvarg2(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *srcv);
extern void setvarg3(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *src1,Data_Obj *src2);
extern void setvarg4(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *src1,Data_Obj *src2,Data_Obj *src3);
extern void setvarg5(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *src1,Data_Obj *src2,Data_Obj *src3, Data_Obj *src4);

/* Public data structures */

/* Public prototypes */

// which file did this come from?
extern int _insure_static(QSP_ARG_DECL  const Vec_Obj_Args *oap);
extern void add_link(void (*func)(LINK_FUNC_ARG_DECLS),LINK_FUNC_ARG_DECLS);

#define insure_static(oap) _insure_static(QSP_ARG  oap)

/* vectbl.c */
extern void vl_init(SINGLE_QSP_ARG_DECL);
extern void vl2_pf_init(SINGLE_QSP_ARG_DECL);

/* vec_args.c */
extern int _perf_vfunc(QSP_ARG_DECL  Vec_Func_Code code, Vec_Obj_Args *oap);
#define perf_vfunc(code, oap) _perf_vfunc(QSP_ARG  code, oap)

/* vec_call.c */
#ifdef HAVE_ANY_GPU
extern void set_gpu_dispatch_func( int (*)(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap) );
#endif /* HAVE_ANY_GPU */

extern int _call_vfunc( QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap );
#define call_vfunc( vfp, oap ) _call_vfunc( QSP_ARG  vfp, oap )

/* warmenu.c */
extern void set_perf(int);
extern COMMAND_FUNC( do_yuv2rgb );

/* obj_args.c */
extern void clear_obj_args(Vec_Obj_Args *);
extern void _show_obj_args(QSP_ARG_DECL  const Vec_Obj_Args *);
extern void set_obj_arg_flags(Vec_Obj_Args *);

#define show_obj_args(oap) _show_obj_args(QSP_ARG  oap)

/* cksiz.c */
extern int _old_cksiz(QSP_ARG_DECL  int,Data_Obj *,Data_Obj *);
#define old_cksiz(idx,dp1,dp2) _old_cksiz(QSP_ARG  idx,dp1,dp2)

extern debug_flag_t veclib_debug;

/* lin_util.c */
extern int _xform_chk(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform);
extern void _inner(QSP_ARG_DECL  Data_Obj *, Data_Obj *, Data_Obj *);

#define xform_chk(dpto,dpfr,xform) _xform_chk(QSP_ARG  dpto,dpfr,xform)
#define inner(dpto,dp1,dp2) _inner(QSP_ARG  dpto,dp1,dp2)

#ifdef __cplusplus
}
#endif


#endif /* ! _VECLIB_API_H_ */

