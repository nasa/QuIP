// No include guard here, meant to be included multiple
// times with different platform defns...

// We only declare prototypes for the host untyped
// functions, because the typed functions are static
// and local to the file where the untyped functions
// are declared...

#include "veclib/vec_func.h"
#include "veclib/obj_args.h"

// obsolete file... 
//#include "veclib/method_defs.h"// why does this need to be included multiple times? NO
				// include guard added...

#ifdef NOT_USED
// declare the prototypes for the main functions, g_vadd etc
// that are not typed
#include "veclib/math_funcs.c"
#include "veclib/cpx_vec.c"
#include "veclib/all_vec.c"
#include "veclib/all_same_prec_vec.c"
#include "veclib/signed_vec.c"
#include "veclib/intvec.c"
#include "veclib/conv_vec.c"	// conversions
#include "veclib/new_conv.c"	// conversions
#endif // NOT_USED

#include "veclib/type_undefs.h"
#include "veclib/method_undefs.h"	// undefines the platform-specific and overloaded
					// defns, e.g. _VEC_FUNC_5V etc

#ifdef NOT_USED
// Here we have all the funcs w/ real & complex versions
// In include/veclib/all_vec.c rvadd is declared (cvadd is in a different file)
// So there is no regular vadd...  we add them here as special cases

HOST_PROTOTYPE(vmov)		// special case because of real/cpx versions
HOST_PROTOTYPE(vmul)		// special case because of real/cpx versions
HOST_PROTOTYPE(vsmul)		// special case because of real/cpx versions
HOST_PROTOTYPE(vadd)		// special case because of real/cpx versions
HOST_PROTOTYPE(vsadd)		// special case because of real/cpx versions
HOST_PROTOTYPE(vsub)		// special case because of real/cpx versions
HOST_PROTOTYPE(vssub)		// special case because of real/cpx versions
HOST_PROTOTYPE(vdiv)		// special case because of real/cpx versions
HOST_PROTOTYPE(vsdiv)		// special case because of real/cpx versions
HOST_PROTOTYPE(vsdiv2)		// special case because of real/cpx versions
HOST_PROTOTYPE(vset)		// special case because of real/cpx versions
HOST_PROTOTYPE(vsum)		// special case because of real/cpx versions
#ifndef BUILD_FOR_GPU
HOST_PROTOTYPE(vrand)		// special case because of real/cpx versions
#endif // BUILD_FOR_GPU
HOST_PROTOTYPE(vdot)		// special case because of real/cpx versions
HOST_PROTOTYPE(vsqr)		// special case because of real/cpx versions
HOST_PROTOTYPE(vneg)		// special case because of real/cpx versions
HOST_PROTOTYPE(vexp)		// special case because of real/cpx versions
HOST_PROTOTYPE(vpow)		// special case because of real/cpx versions
HOST_PROTOTYPE(vfft)		// special case because of real/cpx versions
HOST_PROTOTYPE(vift)		// special case because of real/cpx versions
HOST_PROTOTYPE(vvv_slct)	// special case because of real/cpx versions
HOST_PROTOTYPE(vvs_slct)	// special case because of real/cpx versions
HOST_PROTOTYPE(vss_slct)	// special case because of real/cpx versions
#ifndef BUILD_FOR_GPU
//HOST_PROTOTYPE(fft2d)		// special case because of real/cpx versions
//HOST_PROTOTYPE(ift2d)		// special case because of real/cpx versions
//HOST_PROTOTYPE(fftrows)		// special case because of real/cpx versions
//HOST_PROTOTYPE(iftrows)		// special case because of real/cpx versions
#endif // BUILD_FOR_GPU

// master convert entry point
//extern void HOST_CALL_NAME(convert)(QSP_ARG_DECL
//		VFCODE_ARG_DECL  Data_Obj *dpto, Data_Obj *dpfr);
extern void HOST_CALL_NAME(convert)(HOST_CALL_ARG_DECLS);

#endif // NOT_USED

