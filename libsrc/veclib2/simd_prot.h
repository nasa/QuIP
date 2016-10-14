#include "veclib/vecgen.h"

extern void simd_vec_rvmov(float *,float*,unsigned long);

extern void simd_vec_rvadd(float *,float*,float*,unsigned long);
extern void simd_vec_rvmul(float *,float*,float*,unsigned long);
extern void simd_vec_rvsub(float *,float*,float*,unsigned long);
extern void simd_vec_rvdiv(float *,float*,float*,unsigned long);

extern void simd_vec_rvsadd(float *,float*,float,unsigned long);
extern void simd_vec_rvsmul(float *,float*,float,unsigned long);
extern void simd_vec_rvssub(float *,float*,float,unsigned long);
extern void simd_vec_rvsdiv(float *,float*,float,unsigned long);

extern void simd_obj_rvmov(HOST_CALL_ARG_DECLS);
extern void simd_obj_rvadd(HOST_CALL_ARG_DECLS);
extern void simd_obj_rvsub(HOST_CALL_ARG_DECLS);
extern void simd_obj_rvmul(HOST_CALL_ARG_DECLS);
extern void simd_obj_rvdiv(HOST_CALL_ARG_DECLS);
extern void simd_obj_rvsadd(HOST_CALL_ARG_DECLS);
extern void simd_obj_rvssub(HOST_CALL_ARG_DECLS);
extern void simd_obj_rvsmul(HOST_CALL_ARG_DECLS);
extern void simd_obj_rvsdiv(HOST_CALL_ARG_DECLS);

