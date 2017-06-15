
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
//extern Vec_Func_Array quip_vfa_tbl[N_VEC_FUNCS];
//extern Vector_Function vec_func_tbl[N_VEC_FUNCS];
extern dimension_t bitrev_size;
extern dimension_t *bitrev_data;
extern int insist_real, insist_cpx, insist_quat;
extern int n_processors;


#ifdef FOOBAR
// These need to go in vecgen.h?
typedef uint32_t	index_type;
typedef uint32_t	count_type;
#endif // FOOBAR

//#include "nvproto.h"

/* vectbl.c */
//ITEM_INTERFACE_PROTOTYPES(Vector_Function,vf)

/* dispatch.c */
//extern void vec_dispatch(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap);
extern void launch_threads(QSP_ARG_DECL
	void (*func)(HOST_CALL_ARG_DECLS),
	int vf_code, Vec_Obj_Args oa[]);
extern COMMAND_FUNC( set_n_processors );

/* fftsupp.c */
//#include "veclib/fftsupp.h"

//extern void bitrev_init(dimension_t);
//extern int real_fft_check(QSP_ARG_DECL  Data_Obj *, Data_Obj *, const char *);
//extern int real_row_fft_check(QSP_ARG_DECL  Data_Obj *, Data_Obj *, const char *);
//extern int fft_size_ok(QSP_ARG_DECL  Data_Obj *);
//extern int fft_row_size_ok(QSP_ARG_DECL  Data_Obj *);
//extern int fft_col_size_ok(QSP_ARG_DECL  Data_Obj *);

/* vectbl.c */

//extern void vl_init(SINGLE_QSP_ARG_DECL);

/* vec_args.c */
extern int check_obj_devices( Vec_Obj_Args *oap );
extern int is_ram(Data_Obj *);
extern void zero_oargs(Vec_Obj_Args *oap);
extern void do_vfunc( QSP_ARG_DECL   Vector_Function *vfp );
extern void mixed_location_error(QSP_ARG_DECL  Vector_Function *vfp, Vec_Obj_Args *oap);
extern int are_ram_args(Vec_Obj_Args *oap);
#ifdef HAVE_CUDA
extern int are_gpu_args(Vec_Obj_Args *oap);
#endif /* HAVE_CUDA */

/* vec_call.c */
#ifdef HAVE_CUDA
extern void set_gpu_dispatch_func( int (*)(Vector_Function *vfp, Vec_Obj_Args *oap) );
#endif /* HAVE_CUDA */
extern int perf_vfunc(QSP_ARG_DECL  Vec_Func_Code code, Vec_Obj_Args *oap);
extern int cktype(Data_Obj *dp1,Data_Obj *dp2);

/* vf_menu.c */
extern void do_vcode(QSP_ARG_DECL   Vec_Func_Code code);

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
extern int prodimg(QSP_ARG_DECL  Data_Obj *,Data_Obj *,Data_Obj *);
extern void inner(QSP_ARG_DECL  Data_Obj *, Data_Obj *, Data_Obj *);
extern void transpose(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr);
extern int xform_chk(Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *xform);
extern double determinant(Data_Obj *);
extern void vec_xform(QSP_ARG_DECL  Data_Obj *,Data_Obj *,Data_Obj *);
extern void homog_xform(QSP_ARG_DECL  Data_Obj *,Data_Obj *,Data_Obj *);

/* wrap.c */
//extern void dp_scroll(QSP_ARG_DECL  Data_Obj *,Data_Obj *,incr_t,incr_t);
//extern void wrap(QSP_ARG_DECL  Data_Obj *,Data_Obj *);

/* obj_args.c */
extern void private_show_obj_args(QSP_ARG_DECL  char *,const Vec_Obj_Args *,void (*f)(QSP_ARG_DECL  const char *));
// moved to master include...
//extern void set_obj_arg_flags(Vec_Obj_Args *);

/* cksiz.c */
extern int cksiz(QSP_ARG_DECL  int,Data_Obj *,Data_Obj *);
extern int check_bitmap(QSP_ARG_DECL  Data_Obj *,Data_Obj *);

/* sampmenu.c */
extern COMMAND_FUNC( do_samp_menu );

/* typtbl.c */
extern int check_vfa_tbl_size(QSP_ARG_DECL  Vec_Func_Array vfa_tbl[], int size);

#include "veclib_prot.h"

#ifdef __cplusplus
}
#endif


#endif /* ! NO_NEW_VEC_FUNC */

