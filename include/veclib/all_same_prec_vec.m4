
/* This is a file which gets included in other files...
 * To implement each precision, we first include a file
 * defining all the macros, then we include this file.
 *
 * For gpu implementation, some functions need different definitions...
 */


// How do we handle bit precision?
_VEC_FUNC_1V_SCAL( rvset, dst = scalar1_val )
/* don't need this for all types */
dnl /* SCALAR_BIT_METHOD( bvset, SETBIT( scalar1_val ) ) */


// vmov used to be defined differently, presumably to do
// fast moves of contiguous objects...
// We should bring that back, if possible

_VEC_FUNC_2V( rvmov, dst = src1 )


