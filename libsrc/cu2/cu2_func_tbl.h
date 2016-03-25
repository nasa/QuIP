#ifndef _CU2_FUNC_TBL_H_
#define _CU2_FUNC_TBL_H_

#include "quip_config.h"

#include "platform.h"

#ifdef HAVE_CUDA
extern Dispatch_Function cu2_func_tbl[];
extern Vec_Func_Array cu2_vfa_tbl[];
#endif // HAVE_CUDA


#endif // ! _CU2_FUNC_TBL_H_

