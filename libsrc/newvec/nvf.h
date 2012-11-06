
/* nvf.h	New Vector Functions */

#ifndef NO_NEW_VEC_FUNC

#ifdef __cplusplus
extern "C" {
#endif

// Should this be a configurable option?
// No quat support in cuda for the moment.
#define QUATERNION_SUPPORT

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	// abs
#endif /* HAVE_STDLIB_H */

#include "quip_config.h"
#include "nvf_api.h"
#include "new_chains.h"

#ifdef USE_SSE
extern int use_sse_extensions;
#endif /* USE_SSE */

#ifndef N_PROCESSORS
#define N_PROCESSORS 1
#endif /* undef N_PROCESSORS */

#define vf_name	vf_name

#define NO_NEW_VEC_FUNC	((Vec_Func *)NULL)

/* flag bits */
#define DST_VEC			1
#define SRC1_VEC		2
#define SRC2_VEC		4
#define SRC3_VEC		8
#define BITMAP_SRC		16			/* 0x010 */
#define BITMAP_DST		32			/* 0x020 */
#define SRC_SCALAR1		64			/* 0x040 */
#define SRC_SCALAR2		128			/* 0x080 */
#define NEIGHBOR_COMPARISONS	256		/* 0x100 */
#define TWO_SCALAR_RESULTS	512		/* 0x200 */
						/* 0x400 unused */
#define CPX_2_REAL		2048		/* 0x800 */
#define INDEX_RESULT		4096		/* 0x1000 */
#define PROJECTION_OK		8192		/* 0x2000 */
#define REAL_2_CPX		0x4000
#define FWD_FT			0x8000
#define INV_FT			0x10000
#define SRC_SCALAR3		0x20000
#define SRC4_VEC		0x40000
#define SRC_SCALAR4		0x80000

#define INDEX_RESULT_VEC	(DST_VEC|INDEX_RESULT)

#define V_NO_ARGS	DST_VEC
/*#define	V_INPLACE	(SRC1_VEC) */
#define	V_UNARY		(DST_VEC|SRC1_VEC)
#define	V_UNARY_C	(V_UNARY|CPX_2_REAL)
#define	V_FWD_FFT	(V_UNARY|REAL_2_CPX|FWD_FT)	/* vfft, source can be real or cpx */
#define	V_INV_FFT	(V_UNARY|CPX_2_REAL|INV_FT)	/* vift, dest can be real or cpx */
#define	S_UNARY		(DST_VEC|SRC_SCALAR1)
#define V_PROJECTION	(V_UNARY|PROJECTION_OK)
#define V_PROJECTION2	(VV_BINARY|PROJECTION_OK)

#define V_SCALRET2	(TWO_SCALAR_RESULTS|SRC1_VEC|INDEX_RESULT_VEC)

#define VV_SOURCES	(SRC1_VEC|SRC2_VEC)
#define	VS_SOURCES	(SRC1_VEC|SRC_SCALAR1)
#define	SS_SOURCES	(SRC_SCALAR1|SRC_SCALAR2)

#define	VV_BINARY	(DST_VEC|VV_SOURCES)
#define	VS_BINARY	(DST_VEC|VS_SOURCES)
#define	SS_BINARY	(DST_VEC|SS_SOURCES)
#define	VS_SELECT	(VS_BINARY|BITMAP_SRC)
#define	VV_SELECT	(VV_BINARY|BITMAP_SRC)
#define	SS_SELECT	(SS_BINARY|BITMAP_SRC)
#define	SS_RAMP		(SS_BINARY)
#define	SSS_RAMP	(SS_BINARY|SRC_SCALAR3)
#define	VV_TEST		(VV_SOURCES|BITMAP_DST)
#define	VS_TEST		(VS_SOURCES|BITMAP_DST)

#define VVVVCA		(VV_BINARY|SRC3_VEC|SRC4_VEC)
#define VVVSCA		(VV_BINARY|SRC3_VEC|SRC_SCALAR1)
#define VSVVCA		(VS_BINARY|SRC2_VEC|SRC3_VEC)
#define VSVSCA		(VS_BINARY|SRC2_VEC|SRC_SCALAR2)
#define SSVVCA		(SS_BINARY|SRC1_VEC|SRC2_VEC)
#define SSVSCA		(SS_BINARY|SRC1_VEC|SRC_SCALAR3)

#define V_INT_UNARY	(INDEX_RESULT_VEC|SRC1_VEC)
#define V_INT_PROJECTION	(V_INT_UNARY|PROJECTION_OK)

/* VV_SCALRET is used for vdot... */

#define CAN_PROJECT(flag)	((flag)&PROJECTION_OK)

typedef struct vec_func_array {
	Vec_Func_Code	vfa_code;
	void		(*vfa_func[N_FUNCTION_TYPES])(Vec_Obj_Args *);
} Vec_Func_Array;


/* globals */
extern const char *argset_type_name[N_ARGSET_TYPES];
extern Vec_Func *this_vfp;		/* a global */
extern int for_real;
extern const char *type_strings[];		/* see obj_args.c */
extern Vec_Func_Array vfa_tbl[N_VEC_FUNCS];
extern Vec_Func vec_func_tbl[N_VEC_FUNCS];
extern dimension_t bitrev_size;
extern dimension_t *bitrev_data;
extern int insist_real, insist_cpx, insist_quat;
extern int n_processors;


typedef uint32_t	index_type;
typedef uint32_t	count_type;

#include "nvproto.h"

/* vectbl.c */
ITEM_INTERFACE_PROTOTYPES(Vec_Func,vf)

/* dispatch.c */
extern void vec_dispatch(QSP_ARG_DECL  Vec_Func *vfp, Vec_Obj_Args *oap);
extern COMMAND_FUNC( set_n_processors );
extern COMMAND_FUNC( set_use_sse );

/* fftsupp.c */

extern void bitrev_init(dimension_t);
extern int real_fft_check(QSP_ARG_DECL  Data_Obj *, Data_Obj *, const char *);
extern int real_row_fft_check(QSP_ARG_DECL  Data_Obj *, Data_Obj *, const char *);
extern int fft_size_ok(QSP_ARG_DECL  Data_Obj *);
extern int fft_row_size_ok(QSP_ARG_DECL  Data_Obj *);
extern int fft_col_size_ok(QSP_ARG_DECL  Data_Obj *);

/* vectbl.c */

//extern void vl_init(SINGLE_QSP_ARG_DECL);

/* vec_args.c */
extern int is_ram(Data_Obj *);
extern void zero_oargs(Vec_Obj_Args *oap);
extern void do_vfunc( QSP_ARG_DECL   Vec_Func *vfp );
#ifdef HAVE_CUDA
extern int are_gpu_args(Vec_Obj_Args *oap);
extern int are_ram_args(Vec_Obj_Args *oap);
extern void mixed_location_error(QSP_ARG_DECL  Vec_Func *vfp, Vec_Obj_Args *oap);
#endif /* HAVE_CUDA */

/* vec_call.c */
#ifdef HAVE_CUDA
extern void set_gpu_dispatch_func( int (*)(Vec_Func *vfp, Vec_Obj_Args *oap) );
#endif /* HAVE_CUDA */
extern int call_vfunc( QSP_ARG_DECL  Vec_Func *vfp, Vec_Obj_Args *oap );
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
extern void private_show_obj_args(char *,Vec_Obj_Args *,void (*f)(const char *));
extern void show_obj_args(Vec_Obj_Args *);
extern void clear_obj_args(Vec_Obj_Args *);
extern void set_obj_arg_flags(Vec_Obj_Args *);

/* cksiz.c */
extern int old_cksiz(QSP_ARG_DECL  int,Data_Obj *,Data_Obj *);
extern int cksiz(QSP_ARG_DECL  int,Data_Obj *,Data_Obj *);
extern int check_bitmap(QSP_ARG_DECL  Data_Obj *,Data_Obj *);

/* sampmenu.c */
extern COMMAND_FUNC( sampmenu );

/* scale.c */
extern void scale(QSP_ARG_DECL  Data_Obj *,double,double);

#ifdef USE_SSE
/* fast_mov.c */
extern void simd_vec_rvmov(float *,float *,u_long);
extern void simd_obj_rvmov(Vec_Obj_Args *oap);

/* fast_sp.c */
extern void simd_obj_rvadd(Vec_Obj_Args *oap);
extern void simd_obj_rvsub(Vec_Obj_Args *oap);
extern void simd_obj_rvmul(Vec_Obj_Args *oap);
extern void simd_obj_rvdiv(Vec_Obj_Args *oap);
extern void simd_obj_rvsadd(Vec_Obj_Args *oap);
extern void simd_obj_rvssub(Vec_Obj_Args *oap);
extern void simd_obj_rvsmul(Vec_Obj_Args *oap);
extern void simd_obj_rvsdiv(Vec_Obj_Args *oap);
#endif /* USE_SSE */

/* bm_funcs.c */


#ifdef __cplusplus
}
#endif


#endif /* ! NO_NEW_VEC_FUNC */

