#ifndef _CU2_PORT_H_
#define _CU2_PORT_H_

#define QUATERNION_SUPPORT
#undef USE_SSE
#define BUILD_FOR_GPU
#define BUILD_FOR_CUDA

#ifdef pf_str
#undef pf_str
#endif // pf_str
#define pf_str			cu2


#include "veclib/gen_port.h"

// Do we still need these???

extern void *TMPVEC_NAME(size_t size, size_t len, const char *whence);
extern void FREETMP_NAME(void *a, const char *whence);

#define CLEAR_GPU_ERROR(whence)		CLEAR_CUDA_ERROR(whence)
#define CHECK_GPU_ERROR(whence)		CHECK_CUDA_ERROR(whence)
#define OBJ_MAX_THREADS_PER_BLOCK(dp)	PFDEV_CUDA_MAX_THREADS_PER_BLOCK(OBJ_PFDEV(dp))

#endif // ! _CU2_PORT_H_

