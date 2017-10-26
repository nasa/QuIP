#ifndef _VEC_FUNC_H_
#define _VEC_FUNC_H_

/* Public data structures */

#include "vecgen.h"
#include "data_obj.h"
#include "obj_args.h"

typedef struct vector_function {
	Item		vf_item;
	int		vf_code;
	uint32_t	vf_flags;
	uint32_t	vf_precmask;
	int		vf_typemask;

#ifdef FOOBAR
	// We will make these tables part of the platform struct...
	void		(*vl2_func)(HOST_CALL_ARG_DECLS);
#ifdef HAVE_OPENCL
	void		(*ocl_func)(HOST_CALL_ARG_DECLS);
#endif // HAVE_OPENCL
#ifdef HAVE_CUDA
	void		(*cu2_func)(HOST_CALL_ARG_DECLS);
#endif // HAVE_CUDA
#endif // FOOBAR
} Vector_Function;

// Flag bits

#define NO_VEC_FUNC	((Vector_Function *)NULL)


/* Vector_Function stuff - moved to obj_args.h */

#define VF_NAME(vfp)			(vfp)->vf_item.item_name
#define VF_FLAGS(vfp)			(vfp)->vf_flags
#define VF_CODE(vfp)			(vfp)->vf_code
#define VF_TYPEMASK(vfp)		(vfp)->vf_typemask
#define VF_PRECMASK(vfp)		(vfp)->vf_precmask

#define SET_VF_NAME(vfp,s)		(vfp)->vf_item.item_name = s
#define SET_VF_FLAGS(vfp,f)		(vfp)->vf_flags = f
#define SET_VF_CODE(vfp,c)		(vfp)->vf_code = c
#define SET_VF_TYPEMASK(vfp,m)		(vfp)->vf_typemask = m
#define SET_VF_PRECMASK(vfp,m)		(vfp)->vf_precmask = m

#define FIND_VEC_FUNC(code)		(&vec_func_tbl[code])

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

//extern Vector_Function vl2_func_tbl[];

/*
typedef struct platform_func {
	int	pf_code;
	void	(*pf_func)(Vec_Obj_Args *);
} Platform_Func;

#define PF_FUNC(code,name)				\
	{	code,	HOST_CALL_NAME(name)	},

#define NULL_PF_FUNC	((void (*)(Vec_Obj_Args *))NULL)

*/


#ifdef BUILD_FOR_OBJC

// BUG no gpu support even though this code might support mac native...

#define CREAT_VEC_FUNC(func,code,arg_code,precmask,typemask)		\
{									\
	.vf_item	=	{					\
		.item_name	=	#func,				\
		.item_magic	=	QUIP_ITEM_MAGIC			\
	},								\
	.vf_code	=	code,					\
	.vf_flags	=	arg_code,				\
	/*.vl2_func	=	h_vl2_##func,*/				\
	.vf_precmask	=	precmask,				\
	.vf_typemask	=	typemask,				\
},

#define CREAT_CPU_VEC_FUNC(func,code,arg_code,precmask,typemask)	\
	CREAT_VEC_FUNC(func,code,arg_code,precmask,typemask)


#else // ! BUILD_FOR_OBJC

#define CREAT_VEC_FUNC(func,code,arg_code,precmask,typemask)		\
{									\
	.vf_item	=	{					\
		.item_name	=	#func,				\
	},								\
	.vf_code	=	code,					\
	.vf_flags	=	arg_code,				\
	/*VEC_FUNCS(func)*/						\
	.vf_precmask	=	precmask,				\
	.vf_typemask	=	typemask,				\
},

#define CREAT_CPU_VEC_FUNC(func,code,arg_code,precmask,typemask)		\
{									\
	.vf_item	=	{					\
		.item_name	=	#func,				\
	},								\
	.vf_code	=	code,					\
	.vf_flags	=	arg_code,				\
	/*CPU_FUNC(func)*/						\
	.vf_precmask	=	precmask,				\
	.vf_typemask	=	typemask,				\
},

#ifdef FOOBAR
#ifdef HAVE_CUDA

#ifdef HAVE_OPENCL

#define VEC_FUNCS(func)							\
	.vl2_func	=	h_vl2_##func,				\
	.ocl_func	=	h_ocl_##func,				\
	.cu2_func	=	h_cu2_##func,

#define CPU_FUNC(func)							\
	.vl2_func	=	h_vl2_##func,				\
	.ocl_func	=	NULL,					\
	.cu2_func	=	NULL,

#else // ! HAVE_OPENCL

#define VEC_FUNCS(func)							\
	.vl2_func	=	h_vl2_##func,				\
	.cu2_func	=	h_cu2_##func,

#define CPU_FUNC(func)							\
	.vl2_func	=	h_vl2_##func,				\
	.cu2_func	=	NULL,

#endif // ! HAVE_OPENCL

#else // ! HAVE_CUDA

#ifdef HAVE_OPENCL

#define VEC_FUNCS(func)							\
	.vl2_func	=	h_vl2_##func,				\
	.ocl_func	=	h_ocl_##func,

#define CPU_FUNC(func)							\
	.vl2_func	=	h_vl2_##func,				\
	.ocl_func	=	NULL,

#else // ! HAVE_OPENCL

#define VEC_FUNCS(func)							\
	.vl2_func	=	h_vl2_##func,

#define CPU_FUNC(func)							\
	.vl2_func	=	h_vl2_##func,

#endif  // ! HAVE_OPENCL
#endif // ! HAVE_CUDA
#endif // FOOBAR

#endif // ! BUILD_FOR_OBJC


extern Vector_Function vec_func_tbl[];

extern ITEM_PICK_PROT(Vector_Function, vec_func)
extern ITEM_LIST_PROT(Vector_Function, vec_func)

#define pick_vec_func(p)	_pick_vec_func(QSP_ARG p)
#define list_vec_funcs(fp)	_list_vec_funcs(QSP_ARG fp)

#define BEGIN_VFUNC_DECLS	Vector_Function vec_func_tbl[]={

#define END_VFUNC_DECLS		};

#endif /* ! _VEC_FUNC_H_ */

