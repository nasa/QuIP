#include "quip_config.h"

char VersionId_vec_util_hough[] = QUIP_VERSION_STRING;

#include "vec_util.h"
#include "data_obj.h"
#include <math.h>

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

void hough( QSP_ARG_DECL  Data_Obj *xform_dp, Data_Obj *src_dp, float threshold, float x0, float y0 )
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

	if( xform_dp->dt_prec != PREC_SP ){
		sprintf(error_string,"hough:  transform %s precision (%s) should be %s",
			xform_dp->dt_name,name_for_prec(xform_dp->dt_prec),
			name_for_prec(PREC_SP));
		WARN(error_string);
		return;
	}
	if( src_dp->dt_prec != PREC_SP ){
		sprintf(error_string,"hough:  source %s precision (%s) should be %s",
			src_dp->dt_name,name_for_prec(src_dp->dt_prec),
			name_for_prec(PREC_SP));
		WARN(error_string);
		return;
	}
	if( !IS_CONTIGUOUS(xform_dp) ){
		sprintf(error_string,"hough:  transform image %s should be contiguous",
			xform_dp->dt_name);
		WARN(error_string);
		return;
	}
	if( !IS_CONTIGUOUS(src_dp) ){
		sprintf(error_string,"hough:  source image %s should be contiguous",
			src_dp->dt_name);
		WARN(error_string);
		return;
	}

	n_angles = xform_dp->dt_cols;
	n_distances = xform_dp->dt_rows;


	/* Let angles go from 0 to pi, the distances may have to be negative.
	 * If the origin is in the corner, then the max distance is the length of the diagonal.
	 */
	dx = src_dp->dt_cols + 1;
	dy = src_dp->dt_rows + 1;
	max_distance = sqrt(dx*dx+dy*dy);	/* open cv uses dx+dy to save time? */

	dist_inc = 2 * max_distance / (n_distances-1);
	ang_inc = 4*atan(1)/n_angles;

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
		cos_tbl[i_ang] = cos(ang);
		sin_tbl[i_ang] = sin(ang);
		ang += ang_inc;
	}

	src_ptr = (float *)src_dp->dt_data;
	dst_base = (float *)xform_dp->dt_data;

	for(y=0;y<src_dp->dt_rows;y++){
		for(x=0;x<src_dp->dt_cols;x++){
			if( *src_ptr >= threshold ){
//sprintf(error_string,"adding votes for pixel at %d %d",x,y);
//advise(error_string);
				/* add votes for x,y */
				ang=0;
				for(i_ang=0;i_ang<n_angles;i_ang++){
					f_x = x - x0;
					f_y = y - y0;
					r = f_x*cos_tbl[i_ang]+f_y*sin_tbl[i_ang];
					f_r = r / dist_inc;	/* dist_inc is the bin width */
					i_r = floor(f_r);
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

