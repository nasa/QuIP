#include "quip_config.h"

char VersionId_vec_util_integral[] = QUIP_VERSION_STRING;

/* integral (cumulative sum) image, as used in Viola-Jones */
#include "vec_util.h"
#include "data_obj.h"

/*
 * What do we do about the first row (column)?  To avoid special cases, we
 * need to pad with a row and column of zeros...
 */

void cum_sum( QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp )
{
	long *lp;
	u_char *cp;
	dimension_t r,c;
	int d,e;

	if( dst_dp->dt_prec != PREC_DI ){
		sprintf(ERROR_STRING,"cum_sum:  destination image %s (%s) must have %s precision",
			dst_dp->dt_name,name_for_prec(MACHINE_PREC(dst_dp)),name_for_prec(PREC_DI));
		WARN(ERROR_STRING);
		return;
	}
	if( src_dp->dt_prec != PREC_UBY ){
		sprintf(ERROR_STRING,"cum_sum:  source image %s (%s) must have %s precision",
			src_dp->dt_name,name_for_prec(MACHINE_PREC(src_dp)), name_for_prec(PREC_UBY));
		WARN(ERROR_STRING);
		return;
	}

	if( ! IS_CONTIGUOUS(dst_dp) ){
		sprintf(ERROR_STRING,"cum_sum:  image %s must be contiguous",dst_dp->dt_name);
		WARN(ERROR_STRING);
		return;
	}
	if( ! IS_CONTIGUOUS(src_dp) ){
		sprintf(ERROR_STRING,"cum_sum:  image %s must be contiguous",src_dp->dt_name);
		WARN(ERROR_STRING);
		return;
	}
	if( dst_dp->dt_rows != src_dp->dt_rows+1 ){
		sprintf(ERROR_STRING,"cum_sum:  height of destination image %s (%d) should be 1 greater than width of source image %s (%d)",
			dst_dp->dt_name,dst_dp->dt_rows,src_dp->dt_name,src_dp->dt_rows);
		WARN(ERROR_STRING);
		return;
	}
	if( dst_dp->dt_cols != src_dp->dt_cols+1 ){
		sprintf(ERROR_STRING,"cum_sum:  width of destination image %s (%d) should be 1 greater than width of source image %s (%d)",
			dst_dp->dt_name,dst_dp->dt_cols,src_dp->dt_name,src_dp->dt_cols);
		WARN(ERROR_STRING);
		return;
	}
	if( dst_dp->dt_comps != 1 ){
		sprintf(ERROR_STRING,"cum_sum:  destination image %s (%d) should have a depth of 1",
			dst_dp->dt_name,dst_dp->dt_comps);
		WARN(ERROR_STRING);
		return;
	}
	if( src_dp->dt_comps != 1 ){
		sprintf(ERROR_STRING,"cum_sum:  destination image %s (%d) should have a depth of 1",
			src_dp->dt_name,src_dp->dt_comps);
		WARN(ERROR_STRING);
		return;
	}

	lp = (long *)dst_dp->dt_data;
	cp = (u_char *)src_dp->dt_data;
	d = dst_dp->dt_cols;	/* offset to go up one row */
	e = d+1;	/* offset to go up one row and one to the left */

	*lp++ = 0;	/* init upper left corner */
	c = src_dp->dt_cols;
	while(c--){
		*lp++=0;	/* init top row */
	}
	r = src_dp->dt_rows;
//sprintf(ERROR_STRING,"d = %d,   e = %d",d,e);
//advise(ERROR_STRING);
//sprintf(ERROR_STRING,"dst_dp->dt_data = 0x%lx",(u_long)dst_dp->dt_data);
//advise(ERROR_STRING);
	while( r -- ){
		*lp++ = 0;	/* init leftmost column */
		c = src_dp->dt_cols;
		while(c--){
			*lp = *cp++ + *(lp-1) + *(lp-d) - *(lp-e);
//sprintf(ERROR_STRING,"sat[%ld][%ld] = %d = %d + %d +%d - %d      lp = 0x%lx, 0x%lx 0x%lx",
//src_dp->dt_rows-r,src_dp->dt_cols-c,
//*lp,*(cp-1),*(lp-1),*(lp-d),*(lp-e), (u_long)lp,(u_long)(lp-d),(u_long)(lp-e));
//advise(ERROR_STRING);
			lp++;
		}
	}
}

