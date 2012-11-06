/* fast down-sampling - should we use integer or SSE float? */

#include "quip_config.h"

char VersionId_v4l2_fastdown[] = QUIP_VERSION_STRING;

#include "data_obj.h"
#include "my_video_dev.h"


#define _fast_downsample_gray( dst_type, src_type )			\
	{								\
	dst_type *dst_ptr;						\
	src_type *src_ptr,*src_ptr2;					\
									\
	dst_ptr = (dst_type *) dst_dp->dt_data;				\
	src_ptr = (src_type *) src_dp->dt_data;				\
	i=dst_dp->dt_rows;						\
	while(i--){							\
		j=dst_dp->dt_cols;					\
		src_ptr2 = src_ptr + src_dp->dt_rowinc;			\
		while(j--){						\
			*dst_ptr = *src_ptr + *(src_ptr+1) +		\
		  		*src_ptr2 + *(src_ptr2+1);		\
			dst_ptr++;					\
			src_ptr+=2;					\
			src_ptr2+=2;					\
		}							\
		src_ptr = src_ptr2;					\
	}								\
	}

#define _fast_downsample_rgb( dst_type, src_type )			\
	{								\
	dst_type *dst_ptr;						\
	src_type *src_ptr,*src_ptr2;					\
									\
	dst_ptr = (dst_type *) dst_dp->dt_data;				\
	src_ptr = (src_type *) src_dp->dt_data;				\
	i=dst_dp->dt_rows;						\
	while(i--){							\
		j=dst_dp->dt_cols;					\
		src_ptr2 = src_ptr + src_dp->dt_rowinc;			\
		while(j--){						\
			*dst_ptr = *src_ptr + *(src_ptr+3) +		\
		  		*src_ptr2 + *(src_ptr2+3);		\
			dst_ptr++; src_ptr++; src_ptr2++;		\
			*dst_ptr = *src_ptr + *(src_ptr+3) +		\
		  		*src_ptr2 + *(src_ptr2+3);		\
			dst_ptr++; src_ptr++; src_ptr2++;		\
			*dst_ptr = *src_ptr + *(src_ptr+3) +		\
		  		*src_ptr2 + *(src_ptr2+3);		\
			dst_ptr++; src_ptr+=4; src_ptr2+=4;		\
		}							\
		src_ptr = src_ptr2;					\
	}								\
	}

void fast_downsample(Data_Obj *dst_dp, Data_Obj *src_dp)
{
	u_long i,j;

	if( dst_dp->dt_prec != PREC_IN ){
		sprintf(DEFAULT_ERROR_STRING,
	"fast_downsample:  destination object %s has type %s, should be %s",
			dst_dp->dt_name,name_for_prec(dst_dp->dt_prec),
			name_for_prec(PREC_IN));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( src_dp->dt_prec != PREC_IN && src_dp->dt_prec != PREC_UBY ){
		sprintf(DEFAULT_ERROR_STRING,
	"fast_downsample:  source object %s has type %s, should be %s or %s",
			src_dp->dt_name,name_for_prec(src_dp->dt_prec),
			name_for_prec(PREC_IN), name_for_prec(PREC_UBY));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	if( dst_dp->dt_comps != src_dp->dt_comps ){
		sprintf(DEFAULT_ERROR_STRING,
"fast_subsample:  component count mismatch between destination %s (%d) and %s (%d)",
			dst_dp->dt_name,dst_dp->dt_comps,src_dp->dt_name,
			src_dp->dt_comps);

		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( 2*dst_dp->dt_cols != src_dp->dt_cols ||
		2*dst_dp->dt_rows != src_dp->dt_rows ){
		sprintf(DEFAULT_ERROR_STRING,
"fast_downsample:  source object %s (%d x %d) should be twice the size of destination object %s (%d x %d)",
			src_dp->dt_name,src_dp->dt_rows,src_dp->dt_cols,
			dst_dp->dt_name,dst_dp->dt_rows,dst_dp->dt_cols);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( src_dp->dt_prec == PREC_UBY ){
		if( src_dp->dt_comps == 3 )
			_fast_downsample_rgb( short, u_char )
		else
			_fast_downsample_gray( short, u_char )
	} else {
		if( src_dp->dt_comps == 3 )
			_fast_downsample_rgb( short, short )
		else
			_fast_downsample_gray( short, short )
	}
}

