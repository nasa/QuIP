#include "quip_config.h"

/* integral (cumulative sum) image, as used in Viola-Jones */
#include "vec_util.h"
#include "quip_prot.h"

/*
 * What do we do about the first row (column)?  To avoid special cases, we
 * need to pad with a row and column of zeros...
 *
 * What is the difference between this and cumsum.c???
 * This is two dimensional, but wants to add up bytes...
 * Should implement for different precisions!?
 */

void _cum_sum( QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp )
{
	int32_t *lp;
	u_char *cp;
	dimension_t r,c;
	int d,e;

	INSIST_RAM_OBJ(dst_dp,cum_sum)
	INSIST_RAM_OBJ(src_dp,cum_sum)

	if( OBJ_PREC(dst_dp) != PREC_DI ){
		sprintf(ERROR_STRING,
			"cum_sum:  destination image %s (%s) must have %s precision",
			OBJ_NAME(dst_dp),OBJ_MACH_PREC_NAME(dst_dp),PREC_DI_NAME);
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_PREC(src_dp) != PREC_UBY ){
		sprintf(ERROR_STRING,"cum_sum:  source image %s (%s) must have %s precision",
			OBJ_NAME(src_dp),OBJ_MACH_PREC_NAME(src_dp), PREC_UBY_NAME);
		WARN(ERROR_STRING);
		return;
	}

	if( ! IS_CONTIGUOUS(dst_dp) ){
		sprintf(ERROR_STRING,"cum_sum:  image %s must be contiguous",OBJ_NAME(dst_dp));
		WARN(ERROR_STRING);
		return;
	}
	if( ! IS_CONTIGUOUS(src_dp) ){
		sprintf(ERROR_STRING,"cum_sum:  image %s must be contiguous",OBJ_NAME(src_dp));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_ROWS(dst_dp) != OBJ_ROWS(src_dp)+1 ){
		sprintf(ERROR_STRING,"cum_sum:  height of destination image %s (%d) should be 1 greater than height of source image %s (%d)",
			OBJ_NAME(dst_dp),OBJ_ROWS(dst_dp),OBJ_NAME(src_dp),OBJ_ROWS(src_dp));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_COLS(dst_dp) != OBJ_COLS(src_dp)+1 ){
		sprintf(ERROR_STRING,
			"cum_sum:  width of destination image %s (%d) should %d",
			OBJ_NAME(dst_dp),OBJ_COLS(dst_dp),1+OBJ_COLS(src_dp));
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,
			"Should be 1 greater than width of source image %s (%d)",
			OBJ_NAME(src_dp),OBJ_COLS(src_dp));
		advise(ERROR_STRING);
		return;
	}
	if( OBJ_COMPS(dst_dp) != 1 ){
		sprintf(ERROR_STRING,
	"cum_sum:  destination image %s (%d) should have a depth of 1",
			OBJ_NAME(dst_dp),OBJ_COMPS(dst_dp));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_COMPS(src_dp) != 1 ){
		sprintf(ERROR_STRING,
	"cum_sum:  destination image %s (%d) should have a depth of 1",
			OBJ_NAME(src_dp),OBJ_COMPS(src_dp));
		WARN(ERROR_STRING);
		return;
	}

	lp = (int32_t *)OBJ_DATA_PTR(dst_dp);
	cp = (u_char *)OBJ_DATA_PTR(src_dp);
	d = OBJ_COLS(dst_dp);	/* offset to go up one row */
	e = d+1;	/* offset to go up one row and one to the left */

	*lp++ = 0;	/* init upper left corner */
	c = OBJ_COLS(src_dp);
	while(c--){
		*lp++=0;	/* init top row */
	}
	r = OBJ_ROWS(src_dp);
//sprintf(ERROR_STRING,"d = %d,   e = %d",d,e);
//advise(ERROR_STRING);
//sprintf(ERROR_STRING,"OBJ_DATA_PTR(dst_dp) = 0x%lx",(u_long)OBJ_DATA_PTR(dst_dp));
//advise(ERROR_STRING);
	while( r -- ){
		*lp++ = 0;	/* init leftmost column */
		c = OBJ_COLS(src_dp);
		while(c--){
			*lp = *cp++ + *(lp-1) + *(lp-d) - *(lp-e);
//sprintf(ERROR_STRING,"sat[%ld][%ld] = %d = %d + %d +%d - %d      lp = 0x%lx, 0x%lx 0x%lx",
//OBJ_ROWS(src_dp)-r,OBJ_COLS(src_dp)-c,
//*lp,*(cp-1),*(lp-1),*(lp-d),*(lp-e), (u_long)lp,(u_long)(lp-d),(u_long)(lp-e));
//advise(ERROR_STRING);
			lp++;
		}
	}
}

