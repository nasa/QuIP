
// First the typed functions...

#undef BUILD_FOR_CPU
#define BUILD_FOR_HOST

#include "vl2_host_call_defs.h"
#include "veclib/host_typed_call_defs.h"

#include "veclib/gen_host_calls.c"


//static void h_vl2_bit_rvset(Vec_Obj_Args *oap)
//{
	//NWARN("Need to implement h_vl2_bit_rvset!");
//}

//static void h_vl2_bit_rvmov(Vec_Obj_Args *oap)
//{
	//NWARN("Need to implement h_vl2_bit_rvmov!");
//}

#include "vl2_host_untyped_call_defs.h"
#include "veclib/host_fft_funcs.c"

#ifdef FOOBAR
#include "veclib/gen_entries.c"
#endif // FOOBAR

