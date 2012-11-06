
/* nvf_api.h	New Vector Functions */

#ifndef _NVF_API_H_
#define _NVF_API_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Public data structures */

#include "vecgen.h"
#include "data_obj.h"

typedef struct vec_func {
	const char *	vf_name;			/* name */
	int		vf_code;	/* an index (Vec_Func_Code) */
	uint32_t	vf_flags;		/* argument calling sequence */
	void		(*vf_func)(Vec_Obj_Args *);	/* function vector */
	uint32_t	vf_precmask;			/* prec field */
	int		vf_typemask;			/* type field */
} Vec_Func ;

extern Vec_Func vec_func_tbl[N_VEC_FUNCS];

/* Public prototypes */

/* vectbl.c */
extern void vl_init(SINGLE_QSP_ARG_DECL);

/* vec_args.c */
extern int perf_vfunc(QSP_ARG_DECL  Vec_Func_Code code, Vec_Obj_Args *oap);

/* vec_call.c */
#ifdef HAVE_CUDA
extern void set_gpu_dispatch_func( int (*)(Vec_Func *vfp, Vec_Obj_Args *oap) );
#endif /* HAVE_CUDA */

/* warmenu.c */
extern void set_perf(int);

/* convert.c */
extern void convert(QSP_ARG_DECL  Data_Obj *,Data_Obj *);

/* size.c */
extern int reduce(QSP_ARG_DECL  Data_Obj *, Data_Obj *);
extern int enlarge(QSP_ARG_DECL  Data_Obj *, Data_Obj *);

/* entries.c */
extern void fft2d( Data_Obj *, Data_Obj * );
extern void ift2d( Data_Obj *, Data_Obj * );
extern void vsum(Vec_Obj_Args *);
extern void vminv(Vec_Obj_Args *);
extern void vmaxv(Vec_Obj_Args *);
extern void vramp2d(Vec_Obj_Args *);


/* lin_util.c */
extern void inner(QSP_ARG_DECL  Data_Obj *, Data_Obj *, Data_Obj *);
extern void xform_list(QSP_ARG_DECL  Data_Obj *,Data_Obj *,Data_Obj *);

/* wrap.c */
extern void dp_scroll(QSP_ARG_DECL  Data_Obj *,Data_Obj *,incr_t,incr_t);
extern void wrap(QSP_ARG_DECL  Data_Obj *,Data_Obj *);


#ifdef __cplusplus
}
#endif


#endif /* ! _NVF_API_H_ */

