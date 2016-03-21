
//#ifndef BUILD_FOR_IOS
//
//// Why is this stuff here?
//#ifdef __APPLE__ //Mac OSX has a different name for the header file
//#include <OpenCL/opencl.h>
//#else
//#include <CL/cl.h>
//#endif
//
//#endif // BUILD_FOR_IOS


#include "item_type.h"
#include "veclib/vec_func.h"
#include "veclib/obj_args.h"

////#define BUILD_FOR_OPENCL
//#undef BUILD_FOR_OPENCL

#define index_type	int32_t	// for vmaxi etc
#define INDEX_PREC	PREC_DI	// for vmaxi etc

