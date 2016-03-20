#ifndef _OCL_PORT_H_
#define _OCL_PORT_H_

#define QUATERNION_SUPPORT
#undef USE_SSE
#define BUILD_FOR_GPU

#ifdef pf_str
#undef pf_str
#endif // pf_str
#define pf_str			ocl


#include "veclib/gen_port.h"

extern void *TMPVEC_NAME(size_t size, size_t len, const char *whence);
extern void FREETMP_NAME(void *a, const char *whence);
extern int get_max_threads_per_block(Data_Obj *odp);
extern int max_threads_per_block;

#endif // ! _OCL_PORT_H_

