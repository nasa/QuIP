
#include "quip_config.h"

#include <stdio.h>
#include <string.h>
#include <math.h>		/* floor() */

#include "quip_prot.h"
#include "vec_util.h"
#include "veclib_api.h"
#include "getbuf.h"
#include "platform.h"

/* Out new strategy for changing size is to make a subsampled image for destination and
 * source, based on the dimension values for each of the 5 dimensions.
 * This is more general than enlarge and reduce, because we can be enlarging in one
 * dimension while reducing in another, all with a single op.
 *
 * BUT this logic appears to cause multiple overwrites on reduce!?
 *
 * enlargement_factor and reduction_factor were return variables
 * that were not used...  Maybe they can be eliminated altogether?
 *
 * This also seem like an inefficient strategy for the GPU...
 * but at least it should work!?
 */

static int change_size(QSP_ARG_DECL  Data_Obj *dst_dp,Data_Obj *src_dp )
{
	Dimension_Set ef, *enlargement_factor=&ef;
	Dimension_Set rf, *reduction_factor=&rf;
	Vec_Obj_Args oa1, *oap=&oa1;
	Dimension_Set size_ds, n_ds;
	Dimension_Set *size_dsp=(&size_ds), *n_dsp=(&n_ds);
	Data_Obj *src_ss_dp, *dst_ss_dp;
	dimension_t i,j,k,l,m;
	index_t offsets[N_DIMENSIONS]={0,0,0,0,0};
	incr_t dst_incrs[N_DIMENSIONS], src_incrs[N_DIMENSIONS];
	index_t dst_indices[N_DIMENSIONS]={0,0,0,0,0}, src_indices[N_DIMENSIONS]={0,0,0,0,0};
	index_t dst_offset, src_offset;

	/* For simplicity, we don't allow size changes to be combined with conversions */

	if( !dp_same_prec(dst_dp,src_dp,"change_size") )
		return(-1);

	for(i=0;i<N_DIMENSIONS;i++){
		if( OBJ_TYPE_DIM(dst_dp,i) > OBJ_TYPE_DIM(src_dp,i) ){
			/* enlargement - subsample the destination */
			set_dimension(enlargement_factor,i,
				floor( OBJ_TYPE_DIM(dst_dp,i) / OBJ_TYPE_DIM(src_dp,i) ) );
			set_dimension(reduction_factor,i, 0);

			set_dimension(size_dsp,i, OBJ_TYPE_DIM(src_dp,i) );
			set_dimension(n_dsp,i, DIMENSION(enlargement_factor,i) );
			dst_incrs[i] = DIMENSION(n_dsp,i);
			src_incrs[i] = 1;
		} else {
			/* reduction - subsample the source */
			set_dimension(reduction_factor,i,
				ceil( OBJ_TYPE_DIM(src_dp,i) / OBJ_TYPE_DIM(dst_dp,i) ) );
			set_dimension(enlargement_factor,i, 0 );

			set_dimension(size_dsp,i, floor( OBJ_TYPE_DIM(src_dp,i) /
				DIMENSION(reduction_factor,i) ) );
			/* We don't need to do this multiple times, just pick one and do it */
			/*set_dimension(n_dsp,i, DIMENSION(reduction_factor,i) ); */
			set_dimension(n_dsp,i, 1);
			src_incrs[i] = DIMENSION(reduction_factor,i);
			dst_incrs[i] = 1;
		}
	}
	/* make the subsamples.
	 * the column increment is expressed in columns, etc.
	 */
	dst_ss_dp=make_subsamp("chngsize_dst_obj",dst_dp,size_dsp,offsets,dst_incrs);
	src_ss_dp=make_subsamp("chngsize_src_obj",src_dp,size_dsp,offsets,src_incrs);

	clear_obj_args(oap);
	SET_OA_DEST(oap,dst_ss_dp);
	SET_OA_SRC_OBJ(oap,0, src_ss_dp);
	SET_OA_ARGSTYPE(oap, REAL_ARGS);
	SET_OA_PFDEV(oap,OBJ_PFDEV(dst_dp));

	for(i=0;i<DIMENSION(n_dsp,4);i++){		/* foreach sequence to copy */
		if( dst_incrs[4] > 1 )
			dst_indices[4]=i;
		else	src_indices[4]=i;
		for(j=0;j<DIMENSION(n_dsp,3);j++){	/* foreach frame to copy */
			if( dst_incrs[3] > 1 )
				dst_indices[3]=j;
			else	src_indices[3]=j;
			for(k=0;k<DIMENSION(n_dsp,2);k++){	/* foreach row */
				if( dst_incrs[2] > 1 )
					dst_indices[2]=k;
				else	src_indices[2]=k;
				for(l=0;l<DIMENSION(n_dsp,1);l++){	/* foreach col */
					if( dst_incrs[1] > 1 )
						dst_indices[1]=l;
					else	src_indices[1]=l;
					for(m=0;m<DIMENSION(n_dsp,0);m++){ /* foreach comp */
						if( dst_incrs[0] > 1 )
							dst_indices[0]=m;
						else	src_indices[0]=m;
						/* relocate the appropriate subsample */

						dst_offset = dst_indices[0]*OBJ_COMP_INC(dst_dp) +
								dst_indices[1]*OBJ_PXL_INC(dst_dp) +
								dst_indices[2]*OBJ_ROW_INC(dst_dp) +
								dst_indices[3]*OBJ_FRM_INC(dst_dp) +
								dst_indices[4]*OBJ_SEQ_INC(dst_dp) ;
	( * PF_OFFSET_DATA_FN(OBJ_PLATFORM(dst_ss_dp)) ) (QSP_ARG  dst_ss_dp, dst_offset );

						src_offset = src_indices[0]*OBJ_COMP_INC(src_dp) +
								src_indices[1]*OBJ_PXL_INC(src_dp) +
								src_indices[2]*OBJ_ROW_INC(src_dp) +
								src_indices[3]*OBJ_FRM_INC(src_dp) +
								src_indices[4]*OBJ_SEQ_INC(src_dp) ;
	( * PF_OFFSET_DATA_FN(OBJ_PLATFORM(src_ss_dp)) ) (QSP_ARG  src_ss_dp, src_offset );

						// This doesn't check for cuda obj...
						//vmov(oap);
						perf_vfunc(FVMOV, oap );
					}
				}
			}
		}
	}
	delvec(dst_ss_dp);
	delvec(src_ss_dp);

	SET_OBJ_FLAG_BITS(dst_dp, DT_ASSIGNED);

	return(0);
}

int reduce(QSP_ARG_DECL  Data_Obj *lil_dp,Data_Obj *big_dp)		/* reduce into lil_dp */
{
	if( change_size(QSP_ARG  lil_dp,big_dp) < 0 )
		return(-1);

	SET_OBJ_FLAG_BITS(lil_dp, DT_ASSIGNED);

	return(0);
}

int enlarge(QSP_ARG_DECL  Data_Obj *big_dp,Data_Obj *lil_dp)		/* reduce into lil_dp */
{
	if( change_size(QSP_ARG  big_dp,lil_dp) < 0 )
		return(-1);

	SET_OBJ_FLAG_BITS(big_dp, DT_ASSIGNED);

	return(0);
}

