// used for host calls NOT kernels 

//#include "veclib/both_call_utils.h"

// SP stuff

#include "veclib/sp_defs.h"
#include "veclib/gen_float_calls.h"
#include "veclib/type_undefs.h"

// DP stuff

#include "veclib/dp_defs.h"
#include "veclib/gen_float_calls.h"
#include "veclib/type_undefs.h"

// BY stuff

#include "veclib/by_defs.h"
#include "veclib/gen_int_calls.h"
#include "veclib/type_undefs.h"


// IN stuff

#include "veclib/in_defs.h"
#include "veclib/gen_int_calls.h"
#include "veclib/type_undefs.h"


// DI stuff

#include "veclib/di_defs.h"
#include "veclib/gen_int_calls.h"
#include "veclib/type_undefs.h"


// LI stuff

#include "veclib/li_defs.h"
#include "veclib/gen_int_calls.h"
#include "veclib/type_undefs.h"


// UBY stuff

#include "veclib/uby_defs.h"
#include "veclib/gen_uint_calls.h"
#include "veclib/type_undefs.h"


// UIN stuff

#include "veclib/uin_defs.h"
#include "veclib/gen_uint_calls.h"
#include "veclib/type_undefs.h"


// UDI stuff

#include "veclib/udi_defs.h"
#include "veclib/gen_uint_calls.h"
#include "veclib/type_undefs.h"


// ULI stuff

#include "veclib/uli_defs.h"
#include "veclib/gen_uint_calls.h"
// gpu_int.cl contains special case for left-shift!?  cuda bug?
#include "veclib/type_undefs.h"

// bit stuff
//
// This may not work on a GPU, because different threads will need to read
// and write the same word!?

#include "veclib/bit_defs.h"
#include "veclib/gen_bit_calls.h"
#include "veclib/type_undefs.h"


// Why is this only for not building kernels?

#ifndef BUILDING_KERNELS

// Now mixed precision functions...
// We currently implement 4:
// inby, spdp, uindi, and ubyin
// The second code indicates the destination precision,
// Most common use is vsum

#include "veclib/inby_defs.h"
#include "veclib/gen_mixed_int_calls.h"
#include "veclib/type_undefs.h"

#include "veclib/uindi_defs.h"
#include "veclib/gen_mixed_uint_calls.h"
#include "veclib/type_undefs.h"

#include "veclib/ubyin_defs.h"
#include "veclib/gen_mixed_uint_calls.h"
#include "veclib/type_undefs.h"

#include "veclib/spdp_defs.h"
#include "veclib/gen_mixed_float_calls.h"
#include "veclib/type_undefs.h"

#endif // ! BUILDING_KERNELS

#include "veclib/method_undefs.h"



