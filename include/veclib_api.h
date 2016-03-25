
#ifndef _VECLIB_API_H_
#define _VECLIB_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "veclib/vec_func.h"
#include "veclib/obj_args.h"
//#include "veclib/vl2_veclib_prot.h"

extern COMMAND_FUNC(do_comp_menu);
extern COMMAND_FUNC( do_vl_menu );

extern int use_sse_extensions;

extern void show_obj_args(QSP_ARG_DECL  const Vec_Obj_Args *);

extern void setvarg1(Vec_Obj_Args *oap, Data_Obj *dp);
extern void setvarg2(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *srcv);
extern void setvarg3(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *src1,Data_Obj *src2);
extern void setvarg4(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *src1,Data_Obj *src2,Data_Obj *src3);
extern void setvarg5(Vec_Obj_Args *oap,Data_Obj *dstv,Data_Obj *src1,Data_Obj *src2,Data_Obj *src3, Data_Obj *src4);

/* Public data structures */

/* Public prototypes */

// which file did this come from?
extern int insure_static(const Vec_Obj_Args *oap);
extern void add_link(void (*func)(LINK_FUNC_ARG_DECLS),LINK_FUNC_ARG_DECLS);

/* vectbl.c */
extern void vl_init(SINGLE_QSP_ARG_DECL);
extern void vl2_pf_init(SINGLE_QSP_ARG_DECL);

/* vec_args.c */
extern int perf_vfunc(QSP_ARG_DECL  Vec_Func_Code code, Vec_Obj_Args *oap);

/* vec_call.c */
#ifdef HAVE_ANY_GPU
extern void set_gpu_dispatch_func( int (*)(Vector_Function *vfp, Vec_Obj_Args *oap) );
#endif /* HAVE_ANY_GPU */

extern int call_vfunc( QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap );

/* warmenu.c */
extern void set_perf(int);
extern COMMAND_FUNC( do_yuv2rgb );


/* convert.c */
extern void convert(QSP_ARG_DECL  Data_Obj *,Data_Obj *);

/* entries.c */
//extern void fft2d( Data_Obj *, Data_Obj * );
extern void ift2d( Data_Obj *, Data_Obj * );
extern void vsum(HOST_CALL_ARG_DECLS);
extern void vminv(HOST_CALL_ARG_DECLS);
extern void vmaxv(HOST_CALL_ARG_DECLS);
extern void vramp2d(HOST_CALL_ARG_DECLS);


/* lin_util.c */
extern void inner(QSP_ARG_DECL  Data_Obj *, Data_Obj *, Data_Obj *);
extern void xform_list(QSP_ARG_DECL  Data_Obj *,Data_Obj *,Data_Obj *);

/* obj_args.c */
extern void clear_obj_args(Vec_Obj_Args *);

/* these were moved here from nvf.h when wrap() and scale() were moved to vec_util */
extern void vmov(HOST_CALL_ARG_DECLS);
extern void vsmul(HOST_CALL_ARG_DECLS);
extern void vsadd(HOST_CALL_ARG_DECLS);
extern int old_cksiz(QSP_ARG_DECL  int,Data_Obj *,Data_Obj *);

extern debug_flag_t veclib_debug;

#ifdef __cplusplus
}
#endif


#endif /* ! _VECLIB_API_H_ */

