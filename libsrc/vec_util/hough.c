#include "quip_config.h"

#include <math.h>
#include "vec_util.h"
#include "quip_prot.h"
#include "getbuf.h"

/* We parameterize lines by the distance from the origin, and the angle of the normal
 * vector, but where should we place the origin?  For now we just use memory coordinates...
 *
 * When we cast a vote for the pixel at x,y, we support r,theta pairs that satisfy:
 *
 * x cos theta + y sin theta - r = 0
 *
 * So, for each theta, we compute an r, round, and increment the bin.
 */

static int tbl_size=0;
static float *sin_tbl=NULL, *cos_tbl=NULL;

void _hough( QSP_ARG_DECL  Data_Obj *xform_dp, Data_Obj *src_dp, float threshold, float x0, float y0 )
{
	int n_angles;
	int n_distances;
	float dx,dy,max_distance;
	float dist_inc, ang_inc;
	float *src_ptr, *dst_ptr, *dst_base;
	float f_r,w1,w2;
	dimension_t x,y;
	int i_ang, i_r;
	float r,ang;
	float f_x,f_y;

	INSIST_RAM_OBJ(xform_dp,hough);
	INSIST_RAM_OBJ(src_dp,hough);

	if( OBJ_PREC(xform_dp) != PREC_SP ){
		sprintf(ERROR_STRING,"hough:  transform %s precision (%s) should be %s",
			OBJ_NAME(xform_dp),OBJ_PREC_NAME(xform_dp),
			NAME_FOR_PREC_CODE(PREC_SP));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_PREC(src_dp) != PREC_SP ){
		sprintf(ERROR_STRING,"hough:  source %s precision (%s) should be %s",
			OBJ_NAME(src_dp),OBJ_PREC_NAME(src_dp),
			NAME_FOR_PREC_CODE(PREC_SP));
		WARN(ERROR_STRING);
		return;
	}
	if( !IS_CONTIGUOUS(xform_dp) ){
		sprintf(ERROR_STRING,"hough:  transform image %s should be contiguous",
			OBJ_NAME(xform_dp));
		WARN(ERROR_STRING);
		return;
	}
	if( !IS_CONTIGUOUS(src_dp) ){
		sprintf(ERROR_STRING,"hough:  source image %s should be contiguous",
			OBJ_NAME(src_dp));
		WARN(ERROR_STRING);
		return;
	}

	n_angles = OBJ_COLS(xform_dp);
	n_distances = OBJ_ROWS(xform_dp);


	/* Let angles go from 0 to pi, the distances may have to be negative.
	 * If the origin is in the corner, then the max distance is the length of the diagonal.
	 */
	dx = OBJ_COLS(src_dp) + 1;
	dy = OBJ_ROWS(src_dp) + 1;
	max_distance = (float)sqrt(dx*dx+dy*dy);	/* open cv uses dx+dy to save time? */

	dist_inc = 2 * max_distance / (n_distances-1);
	ang_inc = (float)(4*atan(1)/n_angles);

	if( tbl_size == 0 ){
		tbl_size = n_angles;
		cos_tbl=(float *)getbuf(sizeof(float)*tbl_size);
		sin_tbl=(float *)getbuf(sizeof(float)*tbl_size);
	} else if( tbl_size != n_angles ){
		givbuf(cos_tbl);
		givbuf(sin_tbl);
		tbl_size = n_angles;
		cos_tbl=(float *)getbuf(sizeof(float)*tbl_size);
		sin_tbl=(float *)getbuf(sizeof(float)*tbl_size);
	}
	ang=0;
	for(i_ang=0;i_ang<n_angles;i_ang++){
		cos_tbl[i_ang] = (float) cos(ang);
		sin_tbl[i_ang] = (float) sin(ang);
		ang += ang_inc;
	}

	src_ptr = (float *)OBJ_DATA_PTR(src_dp);
	dst_base = (float *)OBJ_DATA_PTR(xform_dp);

	for(y=0;y<OBJ_ROWS(src_dp);y++){
		for(x=0;x<OBJ_COLS(src_dp);x++){
			if( *src_ptr >= threshold ){
//sprintf(ERROR_STRING,"adding votes for pixel at %d %d",x,y);
//advise(ERROR_STRING);
				/* add votes for x,y */
				//ang=0;
				for(i_ang=0;i_ang<n_angles;i_ang++){
					f_x = x - x0;
					f_y = y - y0;
					r = f_x*cos_tbl[i_ang]+f_y*sin_tbl[i_ang];
					f_r = r / dist_inc;	/* dist_inc is the bin width */
					i_r = (int) floor(f_r);
					w2 = f_r - i_r;
					w1 = 1 - w2;
					i_r += n_distances/2;
					dst_ptr = dst_base + i_ang + i_r * n_angles;

					/* distribute the votes using linear interpolation */
					*dst_ptr = *dst_ptr + w1;
					dst_ptr += n_angles;
					*dst_ptr = *dst_ptr + w2;
				}
			}
			src_ptr++;
		}
	}
}

