#include "quip_config.h"

#ifdef USE_SSE

//#include "nvf.h"
#include "quip_prot.h"
//#include "veclib/vl2_veclib_prot.h"
#include "simd_prot.h"
//#include "sp_prot.h"

extern int use_sse_extensions;	// in ../veclib/nvf.h, dispatch.c

/* Tips on intrinsic simd using gcc, from http://ds9a.nl/gcc-simd/example.html */

typedef int v4sf __attribute__ ((vector_size(sizeof(float)*4))); // vector of four single floats

typedef union {
	v4sf	v4;
	float f[4];
} fv4 ;

#define NOT_ALIGNED(a)		((((int_for_addr) (a))&0xf)!=0)


#ifdef CAUTIOUS

#define CHECK_ALIGNMENT_3(name)					\
								\
	if( NOT_ALIGNED(v1) || NOT_ALIGNED(v2)			\
				|| NOT_ALIGNED(v3) ){		\
		sprintf(DEFAULT_ERROR_STRING,				\
"CAUTIOUS:  simd_%s:  argument not aligned.",#name);		\
		NWARN(DEFAULT_ERROR_STRING);			\
		return;						\
	}


#define CHECK_ALIGNMENT_2(name)					\
								\
	if( NOT_ALIGNED(v1) || NOT_ALIGNED(v2) ){		\
		sprintf(DEFAULT_ERROR_STRING,				\
"CAUTIOUS:  simd_%s:  argument not aligned.",#name);		\
		NWARN(DEFAULT_ERROR_STRING);			\
		return;						\
	}

#else /* !CAUTIOUS */
#define CHECK_ALIGNMENT_3(name) /* nop */
#define CHECK_ALIGNMENT_2(name) /* nop */
#endif /* !CAUTIOUS */
		

/* Addresses must aligned on 4-word boundaries.
 * Because a float is 4 bytes, this means that a 16 byte
 * boundary.
 */

/* Assume first elements are aligned.
 * BUG we should be able to proceed if they are all equally
 * mis-aligned, by doing the odd elements at the beginning.
 * If the misalignment is different for one of the elements
 * then we have to go back to the old serial method.
 */

#define SIMD_VEC_FUNC_3(name,op)				\
								\
void simd_vec_##name(float *v1,float *v2,float *v3,unsigned long n)	\
{								\
	int n_fast,n_extra;					\
	v4sf *fv1, *fv2, *fv3;					\
								\
	CHECK_ALIGNMENT_3(name)					\
								\
	n_extra = n % 4;					\
	n_fast = n/4;						\
	fv1 = (v4sf *)v1;					\
	fv2 = (v4sf *)v2;					\
	fv3 = (v4sf *)v3;					\
								\
	while( n_fast -- )					\
		*fv1++ = *fv2++ op *fv3++;			\
								\
	if( n_extra > 0 ){					\
		v1=(float *)fv1;				\
		v2=(float *)fv2;				\
		v3=(float *)fv3;				\
		while(n_extra--)				\
			*v1++ = *v2++ op *v3++;			\
	}							\
}


#define SIMD_VEC_FUNC_1S_2(name,op)				\
								\
void simd_vec_##name(float *v1,float *v2,float scalar_val,unsigned long n)	\
{								\
	int n_fast,n_extra;					\
	v4sf *fv1, *fv2;					\
	fv4 fvscal;						\
								\
	CHECK_ALIGNMENT_2(name)					\
								\
	n_extra = n % 4;					\
	n_fast = n/4;						\
	fvscal.f[0]=scalar_val;					\
	fvscal.f[1]=scalar_val;					\
	fvscal.f[2]=scalar_val;					\
	fvscal.f[3]=scalar_val;					\
	fv1 = (v4sf *)v1;					\
	fv2 = (v4sf *)v2;					\
								\
	while( n_fast -- )					\
		*fv1++ = *fv2++ op fvscal.v4;			\
								\
	if( n_extra > 0 ){					\
		v1=(float *)fv1;				\
		v2=(float *)fv2;				\
		while(n_extra--)				\
			*v1++ = *v2++ op scalar_val;		\
	}							\
}


/* just for vmov? */

#define SIMD_VEC_FUNC_2(name)					\
								\
void simd_vec_##name(float *v1,float *v2,unsigned long n)	\
{								\
	int n_fast,n_extra;					\
	v4sf *fv1, *fv2;					\
								\
	CHECK_ALIGNMENT_2(name)					\
								\
	n_extra = n % 4;					\
	n_fast = n/4;						\
	fv1 = (v4sf *)v1;					\
	fv2 = (v4sf *)v2;					\
								\
	while( n_fast -- )					\
		*fv1++ = *fv2++ ;				\
								\
	if( n_extra > 0 ){					\
		v1=(float *)fv1;				\
		v2=(float *)fv2;				\
		while(n_extra--)				\
			*v1++ = *v2++ ;				\
	}							\
}

#include "simd_funcs.c"

//SIMD_VEC_FUNC_3(rvadd,+)
//SIMD_VEC_FUNC_3(rvsub,-)
//SIMD_VEC_FUNC_3(rvmul,*)
//SIMD_VEC_FUNC_3(rvdiv,/)
//
//SIMD_VEC_FUNC_1S_2(rvsadd,+)
//SIMD_VEC_FUNC_1S_2(rvssub,-)
//SIMD_VEC_FUNC_1S_2(rvsmul,*)
//SIMD_VEC_FUNC_1S_2(rvsdiv,/)
//
//SIMD_VEC_FUNC_2(rvmov)

#define SIMD_OBJ_FUNC_3( name )						\
									\
void simd_obj_##name(HOST_CALL_ARG_DECLS)					\
{									\
	if( (!N_IS_CONTIGUOUS(OA_DEST(oap))) ||				\
		(!N_IS_CONTIGUOUS(OA_SRC1(oap))) ||				\
		(!N_IS_CONTIGUOUS(OA_SRC2(oap))) || 				\
		( ! use_sse_extensions ) ){				\
		/*h_vl2_sp_##name(oap);*/					\
		NERROR1("bad SIMD call #1");\
	} else {							\
		/* make sure that addresses are aligned */		\
		if( NOT_ALIGNED(OBJ_DATA_PTR(OA_DEST(oap))) ||		\
		    NOT_ALIGNED(OBJ_DATA_PTR(OA_SRC1(oap))) ||			\
		    NOT_ALIGNED(OBJ_DATA_PTR(OA_SRC2(oap))) ){			\
NWARN("data vectors must be aligned on 16 byte boundary for SSE acceleration");\
			/*h_vl2_sp_##name(oap);*/				\
			NERROR1("Bad SIMD call #2");\
		} else {						\
			simd_vec_##name((float *)OBJ_DATA_PTR(OA_DEST(oap)),	\
				(float *)OBJ_DATA_PTR(OA_SRC1(oap)),		\
				(float *)OBJ_DATA_PTR(OA_SRC2(oap)),		\
				OBJ_N_MACH_ELTS(OA_DEST(oap)));		\
		}							\
	}								\
}

#define SIMD_OBJ_FUNC_1S_2( name )					\
									\
void simd_obj_##name(HOST_CALL_ARG_DECLS)					\
{									\
	if( (!N_IS_CONTIGUOUS(OA_DEST(oap))) ||				\
		(!N_IS_CONTIGUOUS(OA_SRC1(oap))) ||				\
		( ! use_sse_extensions ) ){				\
		sp_obj_##name(oap);					\
	} else {							\
		/* make sure that addresses are aligned */		\
		if( NOT_ALIGNED(OBJ_DATA_PTR(OA_DEST(oap))) ||		\
		    NOT_ALIGNED(OBJ_DATA_PTR(OA_SRC1(oap))) ){			\
NWARN("data vectors must be aligned on 16 byte boundary for SSE acceleration");\
			sp_obj_##name(oap);				\
		} else {						\
			simd_vec_##name((float *)OBJ_DATA_PTR(OA_DEST(oap)),	\
				(float *)OBJ_DATA_PTR(OA_SRC1(oap)),		\
				SVAL_FLOAT(OA_SVAL1(oap)),			\
				OBJ_N_MACH_ELTS(OA_DEST(oap)));		\
		}							\
	}								\
}


#define SIMD_OBJ_FUNC_2( name )						\
									\
void simd_obj_##name(HOST_CALL_ARG_DECLS)					\
{									\
	if( (!N_IS_CONTIGUOUS(OA_DEST(oap))) ||				\
		(!N_IS_CONTIGUOUS(OA_SRC1(oap))) ||				\
		( ! use_sse_extensions ) ){				\
		sp_obj_##name(oap);					\
	} else {							\
		/* make sure that addresses are aligned */		\
		if( NOT_ALIGNED(OBJ_DATA_PTR(OA_DEST(oap))) ||		\
		    NOT_ALIGNED(OBJ_DATA_PTR(OA_SRC1(oap))) ){			\
NWARN("data vectors must be aligned on 16 byte boundary for SSE acceleration");\
			sp_obj_##name(oap);				\
		} else {						\
			simd_vec_##name((float *)OBJ_DATA_PTR(OA_DEST(oap)),	\
				(float *)OBJ_DATA_PTR(OA_SRC1(oap)),		\
				OBJ_N_MACH_ELTS(OA_DEST(oap)));		\
		}							\
	}								\
}

SIMD_OBJ_FUNC_3(rvadd)
SIMD_OBJ_FUNC_3(rvsub)
SIMD_OBJ_FUNC_3(rvmul)
SIMD_OBJ_FUNC_3(rvdiv)

SIMD_OBJ_FUNC_1S_2(rvsadd)
SIMD_OBJ_FUNC_1S_2(rvssub)
SIMD_OBJ_FUNC_1S_2(rvsmul)
SIMD_OBJ_FUNC_1S_2(rvsdiv)

SIMD_OBJ_FUNC_2(rvmov)

#ifdef FOOBAR

/* Old assembler-based implementation doesn't work w/ -O2 */

#define SIMD_VVMAX	SIMD_VV_FUNC( "maxps" )
#define SIMD_VSMAX	SIMD_VS_FUNC( "maxps" )
#define SIMD_VVMIN	SIMD_VV_FUNC( "minps" )
#define SIMD_VSMIN	SIMD_VS_FUNC( "minps" )

#define SIMD_VVNAND	SIMD_VV_FUNC( "andnps" )
#define SIMD_VSNAND	SIMD_VS_FUNC( "andnps" )
#define SIMD_VVOR	SIMD_VV_FUNC( "orps" )
#define SIMD_VSOR	SIMD_VS_FUNC( "orps" )
#define SIMD_VVXOR	SIMD_VV_FUNC( "xorps" )
#define SIMD_VSXOR	SIMD_VS_FUNC( "xorps" )


SIMD_FUNC( _simd_vvmax, SIMD_VVMAX )
SIMD_FUNC( _simd_vvmin, SIMD_VVMIN )

SIMD_FUNC( _simd_vsmax, SIMD_VSMAX )
SIMD_FUNC( _simd_vsmin, SIMD_VSMIN )

SIMD_FUNC( _simd_vvnand, SIMD_VVNAND )
SIMD_FUNC( _simd_vvor,   SIMD_VVOR )
SIMD_FUNC( _simd_vvxor,  SIMD_VVXOR )
SIMD_FUNC( _simd_vsnand, SIMD_VSNAND )
SIMD_FUNC( _simd_vsor,   SIMD_VSOR )
SIMD_FUNC( _simd_vsxor,  SIMD_VSXOR )


#define SIMD_2_VEC_SCALAR( normfunc, sse_func )				\
									\
	if( (!N_IS_CONTIGUOUS(OA_DEST(oap))) ||				\
				(!N_IS_CONTIGUOUS(OA_SRC1(oap))) )		\
		normfunc(oap);						\
	else {								\
		float sv[4];						\
									\
		sv[0]=sv[1]=sv[2]=sv[3]= * ( (float *) OA_SVAL1(oap) );\
									\
		sse_func((float *)OBJ_DATA_PTR(OA_DEST(oap)),		\
			(float *)OBJ_DATA_PTR(OA_SRC1(oap)),sv,			\
			OBJ_N_MACH_ELTS(OA_DEST(oap)));			\
	}

void simd_vsadd(HOST_CALL_ARG_DECLS) { SIMD_2_VEC_SCALAR( sp_obj_rvsadd, _simd_vsadd ) }
void simd_vssub(HOST_CALL_ARG_DECLS) { SIMD_2_VEC_SCALAR( sp_obj_rvssub, _simd_vssub ) }
void simd_vsmul(HOST_CALL_ARG_DECLS) { SIMD_2_VEC_SCALAR( sp_obj_rvsmul, _simd_vsmul ) }
void simd_vsdiv(HOST_CALL_ARG_DECLS) { SIMD_2_VEC_SCALAR( sp_obj_rvsdiv, _simd_vsdiv ) }

#endif /* FOOBAR */
#endif /* USE_SSE */

