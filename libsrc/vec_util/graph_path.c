#include "quip_config.h"

char VersionId_vec_util_graph_path[] = QUIP_VERSION_STRING;

/* extend_shorted_path subroutine - from Rivest et al, p. 624
 *
 * We will use this to implement Faster-All-Pairs-Shortest-Paths (p. 627)
 * by repeated calls to this function.
 */

#include "data_obj.h"
#include "vec_util.h"

void extend_shortest_paths(QSP_ARG_DECL   Data_Obj *dst_dp, Data_Obj *src_dp )
{
	float *dstp, *dst_base, *srcp;
	int n;
	int i,j,k;

	/* BUG need to check for contiguity */
	/* BUG need to check that sizes match */
	/* BUG need to check that type is float */
	/* BUG need to check that shape is square */
	/* BUG need to check that depth is 1 */

	n = dst_dp->dt_cols;

	srcp = (float *)src_dp->dt_data;
	dst_base = (float *)dst_dp->dt_data;

	for(i=0;i<n-1;i++){
		for(j=i+1;j<n;j++){
			/* foreach pair(i,j), i<j */
			/* see if the path i->k->j is shorter than the current i->j */

			/* We treat the matrix as symmetric because we assume
			 * the graph is undirected.
			 */

			dstp = dst_base + i*n + j;
			for(k=0;k<n;k++){
				float d;
				d = *(srcp+i*n+k) + *(srcp+k*n+j);
				if( *dstp > d ){
					*dstp = d;
					*(dst_base+j*n+i) = d;
				}
			}
		}
	}
}

