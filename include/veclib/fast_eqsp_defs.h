
// These definitions are shared between fast and eqsp functions

#define GPU_INDEX_TYPE	int

#ifdef BUILD_FOR_OPENCL

#define SET_INDEX( this_index )			\
						\
	this_index = get_global_id(0);		\

#endif // BUILD_FOR_OPENCL

#ifdef BUILD_FOR_CUDA

#define SET_INDEX( this_index )					\
								\
	this_index = blockIdx.x * blockDim.x + threadIdx.x;

#endif // BUILD_FOR_CUDA

#define IDX1_0	index1		// used to be index1.x
#define IDX1_1	index1		// used to be index1.y
#define IDX1_2	index1		// used to be index1.y
#define INC1_0	inc1		// used to be inc1.x
#define INC1_1	inc1		// used to be inc1.x
#define INC1_2	inc1		// used to be inc1.x

#define SCALE_INDEX(idx,inc)	idx *= inc;

#define _VEC_FUNC_MM_NOCC( func_name, c1, c2, s1, gpu_c1, gpu_c2 )
#define _VEC_FUNC_2V_PROJ( func_name, s1, s2, gpu_expr )
#define _VEC_FUNC_2V_PROJ_IDX( func_name, s1, s2, gpu_s1, gpu_s2 )
#define _VEC_FUNC_3V_PROJ( func_name, s1, s2 )
#define _VEC_FUNC_CPX_2V_PROJ( func_name, s1, s2, gpu_re_expr, gpu_im_expr )
#define _VEC_FUNC_CPX_3V_PROJ( func_name, s1, s2 )
#define _VEC_FUNC_QUAT_2V_PROJ( func_name, s1, s2, gpu_re_expr, gpu_im_expr1, gpu_im_expr2, gpu_im_expr3 )

