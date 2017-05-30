#ifndef _CUDA_SUPP_H_
#define _CUDA_SUPP_H_

// The cuda includes were inside the extern C block
// until cuda version 7...

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
//#include <nppi.h>
//#include <npps.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "data_obj.h"
#include "veclib/obj_args.h"
//#include "gl_info.h"

// cuda_error.c

#ifdef HAVE_CUDA
extern void describe_cuda_error2(const char *whence,const char *func_name,
	/*cudaError_t*/ CUresult e);
extern void describe_cuda_error(const char *whence,
	/*cudaError_t*/ CUresult e);
extern void describe_cuda_driver_error2(const char *whence,
	const char *func_name, cudaError_t e);
extern void describe_cuda_driver_error(const char *whence,
	cudaError_t e);
#endif /* HAVE_CUDA */

// cuda_viewer.cpp

extern void xfer_cuda_flag(Data_Obj *dpto, Data_Obj *dpfr, uint32_t flagbit);

// cuda_streams.c

extern COMMAND_FUNC(do_new_stream);
extern COMMAND_FUNC(do_list_cuda_streams);
extern COMMAND_FUNC(do_cuda_stream_info);
extern COMMAND_FUNC(do_sync_stream);

#ifdef __cplusplus
}
#endif

#endif /* _CUDA_SUPP_H_ */

