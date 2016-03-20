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

//#include "ocl_typtbl.h"
#include "veclib/gen_typtbl.h"

static void nullobjf(HOST_CALL_ARG_DECLS)
{
	NWARN("CAUTIOUS:  attempt to call an OpenCL function for an unimplemented precision!?");
	show_obj_args(DEFAULT_QSP_ARG  oap);
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

Vec_Func_Array ocl_vfa_tbl[N_VEC_FUNCS]={

#include "veclib/gen_func_array.c"

};

#define N_OCL_ARRAYED_VEC_FUNCS (sizeof(ocl_vfa_tbl)/sizeof(Vec_Func_Array))

void check_ocl_vfa_tbl(SINGLE_QSP_ARG_DECL)
{
//	if( N_OCL_ARRAYED_VEC_FUNCS != N_VEC_FUNCS ){
//		sprintf(ERROR_STRING,
//	"ocl_vfa_tbl has %ld entries, expected %d!?\n",
//			N_OCL_ARRAYED_VEC_FUNCS, N_VEC_FUNCS );
//		WARN(ERROR_STRING);
////		return -1;
//	}
	assert( N_OCL_ARRAYED_VEC_FUNCS == N_VEC_FUNCS );
	check_vfa_tbl(QSP_ARG  ocl_vfa_tbl, N_OCL_ARRAYED_VEC_FUNCS);
}

