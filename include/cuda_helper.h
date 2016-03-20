
#ifndef _CUDA_HELPER_H_
#define _CUDA_HELPER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "data_obj.h"

extern void xfer_cuda_flag(Data_Obj *dpto, Data_Obj *dpfr, uint32_t flagbit);

#ifdef __cplusplus
}
#endif

#endif /* _CUDA_HELPER_H_ */

