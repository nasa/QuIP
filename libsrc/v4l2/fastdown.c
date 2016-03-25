/* fast down-sampling - should we use integer or SSE float? */

#include "quip_config.h"

#include "quip_prot.h"
#include "data_obj.h"
#include "my_video_dev.h"
#include "my_v4l2.h"


#define _fast_downsample_gray( dst_type, src_type )			\
	{								\
	dst_type *dst_ptr;						\
	src_type *src_ptr,*src_ptr2;					\
									\
	dst_ptr = (dst_type *) OBJ_DATA_PTR(dst_dp);			\
	src_ptr = (src_type *) OBJ_DATA_PTR(src_dp);			\
	i=OBJ_ROWS(dst_dp);						\
	while(i--){							\
		j=OBJ_COLS(dst_dp);					\
		src_ptr2 = src_ptr + OBJ_ROW_INC(src_dp);		\
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
	dst_ptr = (dst_type *) OBJ_DATA_PTR(dst_dp);			\
	src_ptr = (src_type *) OBJ_DATA_PTR(src_dp);			\
	i=OBJ_ROWS(dst_dp);						\
	while(i--){							\
		j=OBJ_COLS(dst_dp);					\
		src_ptr2 = src_ptr + OBJ_ROW_INC(src_dp);		\
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

	if( OBJ_PREC(dst_dp) != PREC_IN ){
		sprintf(DEFAULT_ERROR_STRING,
	"fast_downsample:  destination object %s has type %s, should be %s",
			OBJ_NAME(dst_dp),PREC_NAME(OBJ_PREC_PTR(dst_dp)),
			NAME_FOR_PREC_CODE(PREC_IN));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( OBJ_PREC(src_dp) != PREC_IN && OBJ_PREC(src_dp) != PREC_UBY ){
		sprintf(DEFAULT_ERROR_STRING,
	"fast_downsample:  source object %s has type %s, should be %s or %s",
			OBJ_NAME(src_dp),PREC_NAME(OBJ_PREC_PTR(src_dp)),
			NAME_FOR_PREC_CODE(PREC_IN), NAME_FOR_PREC_CODE(PREC_UBY));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	if( OBJ_COMPS(dst_dp) != OBJ_COMPS(src_dp) ){
		sprintf(DEFAULT_ERROR_STRING,
"fast_subsample:  component count mismatch between destination %s (%d) and %s (%d)",
			OBJ_NAME(dst_dp),OBJ_COMPS(dst_dp),OBJ_NAME(src_dp),
			OBJ_COMPS(src_dp));

		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( 2*OBJ_COLS(dst_dp) != OBJ_COLS(src_dp) ||
		2*OBJ_ROWS(dst_dp) != OBJ_ROWS(src_dp) ){
		sprintf(DEFAULT_ERROR_STRING,
"fast_downsample:  source object %s (%d x %d) should be twice the size of destination object %s (%d x %d)",
			OBJ_NAME(src_dp),OBJ_ROWS(src_dp),OBJ_COLS(src_dp),
			OBJ_NAME(dst_dp),OBJ_ROWS(dst_dp),OBJ_COLS(dst_dp));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( OBJ_PREC(src_dp) == PREC_UBY ){
		if( OBJ_COMPS(src_dp) == 3 )
			_fast_downsample_rgb( short, u_char )
		else
			_fast_downsample_gray( short, u_char )
	} else {
		if( OBJ_COMPS(src_dp) == 3 )
			_fast_downsample_rgb( short, short )
		else
			_fast_downsample_gray( short, short )
	}
}

