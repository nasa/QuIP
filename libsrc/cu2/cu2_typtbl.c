// Now that the typed functions are static, we include this file at the end
// of ocl_veclib.c...

/* We have a table of functions that take the following arg types:
 *
 *	byte, short, long, float, double, u_byte,
 *	u_short, u_long, u_byte/short, short/byte, u_short/long, float/double, bitmap
 *
 * This is one row of the table - 
 *  then the table has separate rows for:
 *		real
 *		complex
 *		mixed (r/c)
 *		quaternion
 *		mixed (r/q)
 *
 */

//#include "cu2_typtbl.h"
#include "cu2_func_tbl.h"
#include "veclib/gen_typtbl.h"

static void nullobjf(HOST_CALL_ARG_DECLS)
{
	NWARN("CAUTIOUS:  attempt to call a function for an unimplemented precision!?");
	/* no more global this_vfp... */
	/*
	NADVISE("nullobjf:");
	sprintf(DEFAULT_ERROR_STRING,
		"Oops, function %s has not been implemented for %s %s precision (functype = %d)",
		VF_NAME(this_vfp), type_strings[OA_FUNCTYPE(oap)%N_ARGSET_PRECISIONS],
		argset_type_name[(OA_FUNCTYPE(oap)/N_ARGSET_PRECISIONS)+1],OA_FUNCTYPE(oap));
	NWARN(DEFAULT_ERROR_STRING);
	*/
	NADVISE("Need to add better error checking!");
	abort();
}

// Do these have to be in order, or do they get sorted???
// They get sorted, but we don't have an easy test
// to make sure that everything is here that should be!?

Vec_Func_Array PLATFORM_SYMBOL_NAME(vfa_tbl)[N_VEC_FUNCS]={

#include "veclib/gen_func_array.c"

};

