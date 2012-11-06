#include "quip_config.h"

char VersionId_vec_util_scramble[] = QUIP_VERSION_STRING;


#include "data_obj.h"
#include "rn.h"
#include "vec_util.h"

#define SCRAMBLE_REQUIRED_PRECISION PREC_UDI
#define SCRAMBLE_REQUIRED_TYPE u_long

/* The old scramble function is in the support library */

void dp_scramble(QSP_ARG_DECL  Data_Obj *dp)
{
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(error_string,"dp_scramble:  object %s must be contiguous",dp->dt_name);
		NWARN(error_string);
		return;
	}
	if( MACHINE_PREC(dp) != SCRAMBLE_REQUIRED_PRECISION ){
		sprintf(error_string,"dp_scramble:  object %s must have %s precision",dp->dt_name,prec_name[SCRAMBLE_REQUIRED_PRECISION]);
		NWARN(error_string);
		return;
	}
	scramble(QSP_ARG  ((SCRAMBLE_REQUIRED_TYPE *)dp->dt_data), dp->dt_n_type_elts );
}

/* permute_elements fills the destination object by using the corresponding elements
 * of perm_dp and indices to lookup src_dp.  For the simple 1D case, they are all vectors.
 *
 * This supports the vector indexing operation:
 * dst = src[ perm ];
 *
 * Compare to LUT mapping:
 *
 * dst = map[ bytes ];
 *
 * In LUT mapping we assume the index is a byte (?) and that map is a small table.
 * In permutation, we think of the src and dst as having the same size, with the perm
 * running over that size.  Both cases are really examples of "sampling":
 *
 * samples = data[ sample_coordinates ];	// sample_coordinates has same size as samples, range of data coords
 *
 * note we also support "rendering":
 *
 * image[ dest_coords ] = samples;	// dest_coords has same size as samples, in range of image coords
 *
 * The above was written in preparation for writing the routine permute_elements, but perhaps
 * it is not needed!?
 */

/*
permute_elements( Data_Obj *dst_dp, Data_Obj *src_dp, Data_Obj *perm_dp )
{
}
*/

