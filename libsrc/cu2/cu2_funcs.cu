#include "quip_config.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
//#include <helper_cuda.h>
#include <curand.h>

extern "C" {

#ifdef _PLATFORM_H_
foobar1
#endif // _PLATFORM_H_
#include "my_cu2.h"
#ifdef HAVE_CUDA
//foobar
#endif // HAVE_CUDA
#ifdef BUILD_FOR_CUDA
//foobar2
#endif // BUILD_FOR_CUDA
#include "platform.h"
#include "m4_cu2_veclib.c"

}

