/* cu2_port.m4 BEGIN */
include(`../../include/veclib/gen_port.m4')

#define BUILD_FOR_CUDA
#define BUILD_FOR_GPU

suppress_if

define(`QUATERNION_SUPPORT')
define(`BUILD_FOR_GPU')
define(`BUILD_FOR_CUDA')


define(`pf_str',`cu2')
define(`DIM3',`dim3')



define(`OFFSET_A',`')
define(`OFFSET_B',`')
define(`OFFSET_C',`')
define(`OFFSET_D',`')
define(`OFFSET_E',`')

define(`OS_ARG',`')
define(`THREAD_INDEX_X',`blockIdx.x * blockDim.x + threadIdx.x')

// Do we still need these???

dnl extern void *TMPVEC_NAME(Platform_Device *pdp, size_t size, size_t len, const char *whence);
dnl extern void FREETMP_NAME(void *a, const char *whence);

dnl	CLEAR_GPU_ERROR(whence)
define(`CLEAR_GPU_ERROR',`/* clear_gpu_error */CLEAR_CUDA_ERROR($1)/* clear_gpu_error */')
define(`CHECK_GPU_ERROR',`CHECK_CUDA_ERROR($1)')

dnl	OBJ_MAX_THREADS_PER_BLOCK(dp)
define(`OBJ_MAX_THREADS_PER_BLOCK',`PFDEV_CUDA_MAX_THREADS_PER_BLOCK(OBJ_PFDEV($1))')

suppress_no

extern void *TMPVEC_NAME`(Platform_Device *pdp, size_t size, size_t len, const char *whence);'
extern void FREETMP_NAME`(void *a, const char *whence);'

/* cu2_port.m4 DONE */

