
#include "quip_config.h"


char VersionId_newvec_size[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include <string.h>
#include <math.h>		/* floor() */

#include "nvf.h"

static int change_size(QSP_ARG_DECL  Data_Obj *,Data_Obj *,Dimension_Set *,Dimension_Set *);

/* Out new strategy for changing size is to make a subsampled image for destination and
 * source, based on the dimension values for each of the 5 dimensions.
 * This is more general than enlarge and reduce, because we can be enlarging in one
 * dimension while reducing in another, all with a single op.
 *
 * BUT this logic appears to cause multiple overwrites on reduce!?
 */

static int change_size(QSP_ARG_DECL  Data_Obj *dst_dp,Data_Obj *src_dp,
			Dimension_Set *enlargement_factor, Dimension_Set *reduction_factor )
{
	Vec_Obj_Args oargs;
	Dimension_Set ss_size, ss_n;
	Data_Obj *src_ss_dp, *dst_ss_dp;
	dimension_t i,j,k,l,m;
	index_t offsets[N_DIMENSIONS]={0,0,0,0,0};
	incr_t dst_incrs[N_DIMENSIONS], src_incrs[N_DIMENSIONS];
	index_t dst_indices[N_DIMENSIONS]={0,0,0,0,0}, src_indices[N_DIMENSIONS]={0,0,0,0,0};

	/* For simplicity, we don't allow size changes to be combined with conversions */

	if( !dp_same_prec(QSP_ARG  dst_dp,src_dp,"change_size") )
		return(-1);

	for(i=0;i<N_DIMENSIONS;i++){
		if( dst_dp->dt_type_dim[i] > src_dp->dt_type_dim[i] ){
			/* enlargement - subsample the destination */
			enlargement_factor->ds_dimension[i] =
				floor( dst_dp->dt_type_dim[i] / src_dp->dt_type_dim[i] );
			reduction_factor->ds_dimension[i] = 0;

			ss_size.ds_dimension[i] = src_dp->dt_type_dim[i];
			ss_n.ds_dimension[i] = enlargement_factor->ds_dimension[i];
			dst_incrs[i] = ss_n.ds_dimension[i];
			src_incrs[i] = 1;
		} else {
			/* reduction - subsample the source */
			reduction_factor->ds_dimension[i] =
				ceil( src_dp->dt_type_dim[i] / dst_dp->dt_type_dim[i] );
			enlargement_factor->ds_dimension[i] = 0;

			ss_size.ds_dimension[i] = floor( src_dp->dt_type_dim[i] /
				reduction_factor->ds_dimension[i] );
			/* We don't need to do this multiple times, just pick one and do it */
			/*ss_n.ds_dimension[i] = reduction_factor->ds_dimension[i]; */
			ss_n.ds_dimension[i] = 1;
			src_incrs[i] = reduction_factor->ds_dimension[i];
			dst_incrs[i] = 1;
		}
	}
	/* make the subsamples.
	 * the column increment is expressed in columns, etc.
	 */
	dst_ss_dp=make_subsamp(QSP_ARG  "chngsize_dst_obj",dst_dp,&ss_size,offsets,dst_incrs);
	src_ss_dp=make_subsamp(QSP_ARG  "chngsize_src_obj",src_dp,&ss_size,offsets,src_incrs);

	oargs.oa_dest = dst_ss_dp;
	oargs.oa_dp[0] = src_ss_dp;
	oargs.oa_argstype = REAL_ARGS;

	for(i=0;i<ss_n.ds_dimension[4];i++){		/* foreach sequence to copy */
		if( dst_incrs[4] > 1 )
			dst_indices[4]=i;
		else	src_indices[4]=i;
		for(j=0;j<ss_n.ds_dimension[3];j++){	/* foreach frame to copy */
			if( dst_incrs[3] > 1 )
				dst_indices[3]=j;
			else	src_indices[3]=j;
			for(k=0;k<ss_n.ds_dimension[2];k++){	/* foreach row */
				if( dst_incrs[2] > 1 )
					dst_indices[2]=k;
				else	src_indices[2]=k;
				for(l=0;l<ss_n.ds_dimension[1];l++){	/* foreach col */
					if( dst_incrs[1] > 1 )
						dst_indices[1]=l;
					else	src_indices[1]=l;
					for(m=0;m<ss_n.ds_dimension[0];m++){ /* foreach comp */
						if( dst_incrs[0] > 1 )
							dst_indices[0]=m;
						else	src_indices[0]=m;
						/* relocate the appropriate subsample */
						dst_ss_dp->dt_data = multiply_indexed_data(dst_dp,dst_indices);
						src_ss_dp->dt_data = multiply_indexed_data(src_dp,src_indices);

						vmov(&oargs);
					}
				}
			}
		}
	}
	delvec(QSP_ARG  dst_ss_dp);
	delvec(QSP_ARG  src_ss_dp);

	dst_dp->dt_flags |= DT_ASSIGNED;

	return(0);
}

int reduce(QSP_ARG_DECL  Data_Obj *lil_dp,Data_Obj *big_dp)		/* reduce into lil_dp */
{
	Dimension_Set ds1,ds2;

	if( change_size(QSP_ARG  lil_dp,big_dp,&ds1,&ds2) < 0 )
		return(-1);

	lil_dp->dt_flags |= DT_ASSIGNED;

	return(0);
}

int enlarge(QSP_ARG_DECL  Data_Obj *big_dp,Data_Obj *lil_dp)		/* reduce into lil_dp */
{
	Dimension_Set ds1,ds2;

	if( change_size(QSP_ARG  big_dp,lil_dp,&ds1,&ds2) < 0 )
		return(-1);

	big_dp->dt_flags |= DT_ASSIGNED;

	return(0);
}

