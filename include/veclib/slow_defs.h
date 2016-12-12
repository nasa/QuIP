
#include "veclib/speed_undefs.h"

#define dst	slow_dst
#define src1	slow_src1
#define src2	slow_src2
#define src3	slow_src3
#define src4	slow_src4

#define _VEC_FUNC_MM_NOCC( func_name, c1, c2, s1, gpu_c1, gpu_c2 )	\
	__VEC_FUNC_MM_NOCC( func_name, gpu_c1, gpu_c2 )

#define _VEC_FUNC_2V_PROJ( func_name, s1, s2, gpu_expr )		\
	__VEC_FUNC_2V_PROJ( func_name, gpu_expr )

#define _VEC_FUNC_2V_PROJ_IDX( func_name, s1, s2, gpu_s1, gpu_s2 )	\
	__VEC_FUNC_2V_PROJ_IDX( func_name, gpu_s1, gpu_s2 )

#define _VEC_FUNC_3V_PROJ( func_name, s1, s2 )				\
	__VEC_FUNC_3V_PROJ( func_name)

#define _VEC_FUNC_CPX_2V_PROJ( func_name, s1, s2, gpu_re_expr, gpu_im_expr )		\
	__VEC_FUNC_CPX_2V_PROJ( func_name, gpu_re_expr, gpu_im_expr )

#define _VEC_FUNC_CPX_3V_PROJ( func_name, s1, s2 )				\
	__VEC_FUNC_CPX_3V_PROJ( func_name )

#define _VEC_FUNC_QUAT_2V_PROJ( func_name, s1, s2, gpu_re_expr, gpu_im_expr1, gpu_im_expr2, gpu_im_expr3 )		\
	__VEC_FUNC_QUAT_2V_PROJ( func_name, gpu_re_expr, gpu_im_expr1, gpu_im_expr2, gpu_im_expr3 )		\

#ifdef BUILD_FOR_CUDA

#define GENERIC_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)		\
									\
	GENERIC_SLOW_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)		\
	GENERIC_SLEN_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)

#define SLOW_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)	\
								\
	GENERIC_SLOW_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)	\
	GENERIC_SLEN_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)


#define GENERIC_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)		\
								\
	GENERIC_SLOW_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)	\
	GENERIC_SLEN_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)

#define _VEC_FUNC_2V_CONV(n,type,statement)		\
							\
	_GENERIC_SLOW_CONV_FUNC(n,std_type,type)	\
	_GENERIC_SLEN_CONV_FUNC(n,std_type,type)

#else // ! BUILD_FOR_CUDA

// Why is it that only CUDA needs the len versions???

#define GENERIC_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)		\
									\
	GENERIC_SLOW_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)


#define SLOW_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)		\
									\
	GENERIC_SLOW_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)


#define GENERIC_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)			\
									\
	GENERIC_SLOW_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)


#define _VEC_FUNC_2V_CONV(n,type,statement)		\
							\
	_GENERIC_SLOW_CONV_FUNC(n,std_type,type)


#endif // ! BUILD_FOR_CUDA

#define GPU_INDEX_TYPE	SLOW_GPU_INDEX_TYPE


#ifdef BUILD_FOR_OPENCL

#define SET_INDEX( this_index )					\
									\
	this_index.d5_dim[0] = get_global_id(0);			\
	this_index.d5_dim[1] = this_index.d5_dim[2] = this_index.d5_dim[3] = this_index.d5_dim[4] = 0;

#endif // BUILD_FOR_OPENCL


#ifdef BUILD_FOR_CUDA

	// What if we have to have blocks in 2 or more dims?

#define SET_INDEX( this_index )					\
									\
	this_index.d5_dim[0] = blockIdx.x * blockDim.x + threadIdx.x;	\
	this_index.d5_dim[1] = this_index.d5_dim[2] = this_index.d5_dim[3] = this_index.d5_dim[4] = 0;

#endif // BUILD_FOR_CUDA

#define IDX1_0	index1.d5_dim[0]		// used to be index1.x
#define IDX1_1	index1.d5_dim[1]		// used to be index1.y
#define IDX1_2	index1.d5_dim[2]		// used to be index1.y
#define INC1_0	inc1.d5_dim[0]			// used to be inc1.x
#define INC1_1	inc1.d5_dim[1]			// used to be inc1.x
#define INC1_2	inc1.d5_dim[2]			// used to be inc1.x

#define SCALE_INDEX(idx,inc)	idx.d5_dim[0] *= inc;

