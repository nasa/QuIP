#include "quip_config.h"

#include "quip_prot.h"
#include "data_obj.h"
#include "rn.h"
#include "vec_util.h"
#include "veclib_api.h"
#include "veclib/vl2_veclib_prot.h"

/* The old scramble function is in the support library */

// rn(N) returns a random integer in the range 0-N...
// So if j is initialized to N, we need to call rn(j-1)...
//
// BUG rn is implemented with modulo operator, is not
// perfectly distributed...

#define SCRAMBLE_DATA(type)				\
							\
	{						\
		type tmp;				\
		type *data;				\
							\
		data = (type *) OBJ_DATA_PTR(dp);	\
							\
		while(j>0){				\
			i=rn(j);			\
			tmp=data[i];			\
			data[i]=data[j];		\
			data[j]=tmp;			\
			j--;				\
		}					\
	}

void _dp_scramble(QSP_ARG_DECL  Data_Obj *dp)
{
	unsigned long i,j;

	INSIST_RAM_OBJ(dp,scramble)

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"dp_scramble:  object %s must be contiguous",OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}
	if( ! IS_REAL(dp) ){
		sprintf(ERROR_STRING,"dp_scrable:  Sorry, scrambling non-real data is not yet supported");
		WARN(ERROR_STRING);
		return;
	}

	//scramble(QSP_ARG  ((SCRAMBLE_TYPE *)OBJ_DATA_PTR(dp)), OBJ_N_TYPE_ELTS(dp) );

	j=OBJ_N_TYPE_ELTS(dp)-1;
	switch( OBJ_MACH_PREC(dp) ){
		case PREC_SP: SCRAMBLE_DATA(float) break;
		case PREC_DP: SCRAMBLE_DATA(double) break;
		case PREC_BY: SCRAMBLE_DATA(char) break;
		case PREC_IN: SCRAMBLE_DATA(short) break;
		case PREC_DI: SCRAMBLE_DATA(int32_t) break;
		case PREC_LI: SCRAMBLE_DATA(int64_t) break;
		case PREC_UBY: SCRAMBLE_DATA(u_char) break;
		case PREC_UIN: SCRAMBLE_DATA(u_short) break;
		case PREC_UDI: SCRAMBLE_DATA(uint32_t) break;
		case PREC_ULI: SCRAMBLE_DATA(uint64_t) break;
		default:
			sprintf(ERROR_STRING,"dp_permute:  object %s has unsupported precision (%s)!?",
				OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)));
			WARN(ERROR_STRING);
			break;
	}
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

