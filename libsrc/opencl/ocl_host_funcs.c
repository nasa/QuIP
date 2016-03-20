
#include "ocl_host_call_defs.h"

// First the typed functions...

#include "veclib/host_typed_call_defs.h"
#include "veclib/gen_host_calls.c"

#include "veclib/bit_defs.h"
#include "veclib/bitmap_ops.c"
#include "veclib/type_undefs.h"


static void h_ocl_sp_rvfft(HOST_CALL_ARG_DECLS)
{ NWARN("h_ocl_sp_rvfft not implemented!?"); }

static void h_ocl_dp_rvfft(HOST_CALL_ARG_DECLS)
{ NWARN("h_ocl_dp_rvfft not implemented!?"); }

static void h_ocl_sp_cvfft(HOST_CALL_ARG_DECLS)
{ NWARN("h_ocl_sp_cvfft not implemented!?"); }

static void h_ocl_dp_cvfft(HOST_CALL_ARG_DECLS)
{ NWARN("h_ocl_dp_cvfft not implemented!?"); }

static void h_ocl_sp_rvift(HOST_CALL_ARG_DECLS)
{ NWARN("h_ocl_sp_rvift not implemented!?"); }

static void h_ocl_dp_rvift(HOST_CALL_ARG_DECLS)
{ NWARN("h_ocl_dp_rvift not implemented!?"); }

static void h_ocl_sp_cvift(HOST_CALL_ARG_DECLS)
{ NWARN("h_ocl_sp_cvift not implemented!?"); }

static void h_ocl_dp_cvift(HOST_CALL_ARG_DECLS)
{ NWARN("h_ocl_dp_cvift not implemented!?"); }


#ifdef FOOBAR
// Now we make the untyped calls
// We use entries.c instead of the *vec.c files from the include directory...

#include "ocl_host_untyped_call_defs.h"
#include "veclib/gen_entries.c"
#endif // FOOBAR

