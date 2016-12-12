
#include "veclib/speed_undefs.h"

#define dst	fast_dst
#define src1	fast_src1
#define src2	fast_src2
#define src3	fast_src3
#define src4	fast_src4

#ifdef BUILD_FOR_CUDA

#define GENERIC_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)			\
										\
	GENERIC_FAST_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)			\
	GENERIC_FLEN_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)

#define SLOW_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)

#define GENERIC_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)		\
								\
	GENERIC_FAST_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)	\
	GENERIC_FLEN_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)

#define _VEC_FUNC_2V_CONV(n,type,statement)		\
							\
	_GENERIC_FAST_CONV_FUNC(n,std_type,type)	\
	_GENERIC_FLEN_CONV_FUNC(n,std_type,type)

#else // ! BUILD_FOR_CUDA

// Why is it that only CUDA needs the len versions???

#define GENERIC_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)			\
										\
	GENERIC_FAST_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)

#define SLOW_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)

#define GENERIC_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)		\
								\
	GENERIC_FAST_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)


#define _VEC_FUNC_2V_CONV(n,type,statement)		\
							\
	_GENERIC_FAST_CONV_FUNC(n,std_type,type)


#endif // ! BUILD_FOR_CUDA

#include "veclib/fast_eqsp_defs.h"

