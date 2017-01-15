
#include "quip_config.h"
#include "quip_prot.h"

#include "veclib/slow_len.h"

// This file is a helper function for gpu routines - it is here
// because it is shared by all gpu platforms...

// OLD:

/* The setup_slow_len functions initialize the len variable (dim3),
 * and return the number of dimensions that are set.  We also need to
 * return which dimensions are used, so that we know which increments
 * to use (important for outer ops).
 *
 * We have a problem with complex numbers - tdim is 2 (in order that
 * we be able to index real and complex parts), but it is really one element.
 *
 * In general, we'd like to collapse contiguous dimensions.  For example,
 * in the case of a subimage of a 3-component image, we might collapse
 * the component and row dimensions into a single mega-row.  Otherwise
 * we need to support 3 lengths etc.
 * [But do we do this?  It seems not...  We have a problem converting a
 * single-component float to a multi-component byte image, because three
 * dimensions are involved, and the laptop only supports 2 (low compute
 * capability).  But this script used to work on the laptop before the rewrite!?]
 *
 * When scanning multiple vectors, the length at each level needs to be the
 * maximum of lengths needed by the each of the vectors.  We use max_d to
 * hold this value.
 */

// NEW:

/* We don't use the multi-dimensional kernel array feature of CUDA etc,
 * instead we pass the number of dimensions and increments as a kernel parameter,
 * from which we can compute the indices...  This frees us to work with objects
 * with more than 3 dimensions.
 */

// BUG name conflict!?
#define MAXD(m,n)	(m>n?m:n)
//#define MAX2(szi_p)	MAXD(szi_p->szi_dst_dim[i_dim],szi_p->szi_src_dim[1][i_dim])
//#define MAX3(szi_p)	MAXD(MAX2(szi_p),szi_p->szi_src_dim[1][i_dim])


int setup_slow_len(	Vector_Args *vap,
			dimension_t start_dim,	// 0, or 1 for complex?
			int i_first,		// index of first source vector
			int n_vec,		// number of source vectors
			Platform_Device *pdp )
{
	int i_dim;
	dimension_t max_d;
#ifdef FOOBAR
	int n_set=0;
#endif // FOOBAR

	SET_VA_ITERATION_TOTAL(vap,1);
	for(i_dim=0;i_dim<start_dim;i_dim++)
		SET_VA_SLOW_SIZE_DIM(vap,i_dim,1);

	for(i_dim=start_dim;i_dim<N_DIMENSIONS;i_dim++){
		int i_src;

		/* Find the max len of all the objects at this level */
		max_d=DIMENSION(VA_DEST_DIMSET(vap),i_dim);
		for(i_src=i_first;i_src<(i_first+n_vec-1);i_src++){
			max_d=MAXD(max_d,
				DIMENSION(VARG_DIMSET(VA_SRC(vap,i_src)),i_dim));
//fprintf(stderr,"setup_slow_len:  i_dim = %d, i_src = %d, max_d = %d\n",i_dim,i_src,max_d);
		}
		SET_VA_SLOW_SIZE_DIM(vap,i_dim,max_d);

#ifdef FOOBAR
		if( max_d > 1 ){
			if( n_set == 0 ){
				SET_VA_LEN_X(vap,max_d);
				SET_VA_DIM_INDEX(vap,n_set,i_dim);
				n_set ++;
			} else if ( n_set == 1 ){
				SET_VA_LEN_Y(vap,max_d);
				SET_VA_DIM_INDEX(vap,n_set,i_dim);
				n_set ++;
			} else if( n_set == 2 ){
	/* CUDA compute capability 1.3 and below can't use 3-D grids; So here
	 * we need to conditionally complain if the number set is 3.
	 */
				if( PFDEV_MAX_DIMS(pdp) == 2 ){
NWARN("Sorry, CUDA compute capability >= 2.0 required for 3-D array operations");
					return(-1);
				}
				SET_VA_LEN_Z(vap,max_d);
				SET_VA_DIM_INDEX(vap,n_set,i_dim);
				n_set ++;
			} else {
				NWARN("setup_slow_len:  Too many dimensions requested.");
				return(-1);
			}
		}
#endif // FOOBAR
		//SET_VA_ITERATION_COUNT(vap,i_dim,max_d);
		SET_VA_ITERATION_TOTAL(vap,VA_ITERATION_TOTAL(vap)*max_d);
	}
#ifdef FOOBAR
//fprintf(stderr,"setup_slow_len:  n_set = %d\n",n_set);
	if( n_set == 0 ){
		SET_VA_LEN_X(vap,1);
		SET_VA_LEN_Y(vap,1);
		SET_VA_LEN_Z(vap,1);
		SET_VA_DIM_INDEX(vap,0,(-1));
		SET_VA_DIM_INDEX(vap,1,(-1));
		SET_VA_DIM_INDEX(vap,2,(-1));
	} else if( n_set == 1 ){
		SET_VA_LEN_Y(vap,1);
		SET_VA_LEN_Z(vap,1);
		SET_VA_DIM_INDEX(vap,1,(-1));
		SET_VA_DIM_INDEX(vap,2,(-1));
	} else if( n_set == 2 ){
		SET_VA_LEN_Z(vap,1);
		SET_VA_DIM_INDEX(vap,2,(-1));
	}
	return(n_set);
#endif // FOOBAR
	return 1;
}

