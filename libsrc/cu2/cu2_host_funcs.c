
#include "cu2_host_call_defs.h"

// First the typed functions...

#include "veclib/host_typed_call_defs.h"
#include "veclib/gen_host_calls.c"

#ifdef FOOBAR
// Now we make the untyped calls
// We use entries.c instead of the *vec.c files from the include directory...

#include "cu2_host_untyped_call_defs.h"
#endif // FOOBAR


#define UNIMP_MSG(f)	sprintf(DEFAULT_ERROR_STRING,"%s is unimplemented!?",#f);\
	NWARN(DEFAULT_ERROR_STRING);

void h_cu2_sp_rvfft(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_sp_rvfft)
}

void h_cu2_sp_rvift(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_sp_rvift)
}

void h_cu2_dp_rvfft(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_dp_rvfft)
}

void h_cu2_dp_rvift(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_dp_rvift)
}

void h_cu2_sp_cvfft(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_sp_cvfft)
}

void h_cu2_sp_cvift(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_sp_cvift)
}

void h_cu2_dp_cvfft(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_dp_cvfft)
}

void h_cu2_dp_cvift(HOST_CALL_ARG_DECLS)
{
	UNIMP_MSG(h_cu2_dp_cvift)
}


