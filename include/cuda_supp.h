#ifndef _CUDA_SUPP_H_
#define _CUDA_SUPP_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "query.h"
#include "data_obj.h"

// cuda_error.c

#ifdef HAVE_CUDA
extern void describe_cuda_error2(const char *whence,const char *func_name, cudaError_t e);
extern void describe_cuda_error(const char *whence, cudaError_t e);
#endif /* HAVE_CUDA */

// cuda_viewer.cpp

extern void xfer_cuda_flag(Data_Obj *dpto, Data_Obj *dpfr, uint32_t flagbit);

// cuda_streams.c

extern void init_cuda_streams(SINGLE_QSP_ARG_DECL);
extern COMMAND_FUNC(do_new_stream);
extern COMMAND_FUNC(do_sync_stream);


#ifdef __cplusplus
}
#endif

#endif /* _CUDA_SUPP_H_ */

