#include "quip_config.h"

char VersionId_vec_util_histo[] = QUIP_VERSION_STRING;

/* histogram function for data objects */

#include "vec_util.h"
#include "data_obj.h"

static void zero_dimension(Data_Obj *dp,float *base,int dim,long index);


#define HISTOGRAM(type)									\
											\
	{										\
	type *frm_base, *row_base, *p_ptr;						\
	frm_base = (type *) data_dp->dt_data;						\
	n_bins = histo_dp->dt_cols;							\
	histbuf = (float *) histo_dp->dt_data;						\
											\
	for(i=0;i<n_bins;i++)								\
		*( histbuf + i* histo_dp->dt_pinc) =0;					\
											\
	for(i=0;i<data_dp->dt_frames;i++){						\
		row_base = frm_base;							\
		for(j=0;j<data_dp->dt_rows;j++){					\
			p_ptr=row_base;							\
			for(k=0;k<data_dp->dt_cols;k++){				\
				num = *p_ptr;						\
				num -= min_limit;					\
				num /= bin_width;					\
				num += 0.5;						\
				index = (incr_t)num;	/* convert to integer */	\
				if( index < 0 ){					\
					index=0;					\
					n_under++;					\
				} else if( index >= (incr_t) n_bins ){			\
					index = (incr_t)n_bins-1;			\
					n_over++;					\
				}							\
				* ( histbuf + index*histo_dp->dt_pinc ) += 1.0;		\
											\
				p_ptr += data_dp->dt_pinc;				\
			}								\
			row_base += data_dp->dt_rowinc;					\
		}									\
		frm_base += data_dp->dt_finc;						\
	}										\
	}

void compute_histo(QSP_ARG_DECL  Data_Obj *histo_dp,Data_Obj *data_dp,double bin_width,double min_limit)
{
	dimension_t i,j,k;
	float num;
	float *histbuf;
	incr_t index;
	dimension_t n_bins;
	int n_under=0, n_over=0;

	if( histo_dp->dt_prec != PREC_SP ){
		WARN("histogram precision must be float");
		return;
	}
	if( histo_dp->dt_comps != 1 ){
		WARN("histogram data must be real");
		return;
	}
	if( histo_dp->dt_rows > 1 || histo_dp->dt_frames > 1 ){
		WARN("only using first row of histogram image");
	}
	if( data_dp->dt_comps != 1 ){
		WARN("input data must be real");
		return;
	}
	switch( data_dp->dt_prec ){
		case PREC_SP: HISTOGRAM(float) break;
		case PREC_DP: HISTOGRAM(double) break;
		case PREC_UBY: HISTOGRAM(u_char) break;
		case PREC_BY: HISTOGRAM(char) break;
		case PREC_UIN: HISTOGRAM(u_short) break;
		case PREC_IN: HISTOGRAM(short) break;
		case PREC_UDI: HISTOGRAM(u_long) break;
		case PREC_DI: HISTOGRAM(long) break;
		default:
			NWARN("unhandled source precision in histogram");
			return;
	}

	if( (n_under > 0) || (n_over > 0) ){
		sprintf(error_string,
			"Histogram for %s had %d underflows and %d overflows",
			data_dp->dt_name,n_under,n_over);
		advise(error_string);
	}
}

#define MAX_DIMENSIONS	(N_DIMENSIONS-1)


static void zero_dimension(Data_Obj *dp,float *base,int dim,long index)
{
	dimension_t i;

	if( dim > 1 ){
		for(i=0;i<dp->dt_type_dim[dim];i++)
			zero_dimension(dp,base+i*dp->dt_type_inc[dim],dim-1,i);
	} else {
		for(i=0;i<dp->dt_cols;i++)
			base[i] = 0.0;
	}
}

void multivariate_histo(QSP_ARG_DECL  Data_Obj *histo_dp,Data_Obj *data_dp,float *width_array,float *min_array)
{
	dimension_t n_dimensions;
	dimension_t i,j,k;
	unsigned int l;
	float *fbase, *fptr, *f;
	float *histbuf;
	incr_t index[MAX_DIMENSIONS];
	int n_bins[MAX_DIMENSIONS];
	int n_under[MAX_DIMENSIONS], n_over[MAX_DIMENSIONS];

	if( histo_dp->dt_prec != PREC_SP ){
		NWARN("2D histogram precision must be float");
		return;
	}
	if( histo_dp->dt_comps != 1 ){
		NWARN("2D histogram data must be real");
		return;
	}
	if( histo_dp->dt_pinc != 1 ){
		NWARN("2D histogram data must be contiguous");
		return;
	}

	n_dimensions = data_dp->dt_comps;

	if( n_dimensions > MAX_DIMENSIONS ){
		NWARN("Too many 2D histogram dimensions");
		return;
	}

	if( data_dp->dt_prec != PREC_SP ){
		NWARN("2D data precision must be float");
		return;
	}

	fbase = (float *) data_dp->dt_data;

	for(l=0;l<n_dimensions;l++){
		n_over[l]=0;
		n_under[l]=0;
		n_bins[l] = histo_dp->dt_type_dim[l+1];
	}

	histbuf = (float *) histo_dp->dt_data;

	zero_dimension(histo_dp,(float *)histo_dp->dt_data,n_dimensions,0L);
	for(l=0;l<MAX_DIMENSIONS;l++)
		index[l]=0;

	for(i=0;i<data_dp->dt_frames;i++){
		fptr = fbase;
		for(j=0;j<data_dp->dt_rows;j++){
			f=fptr;
			for(k=0;k<data_dp->dt_cols;k++){
				float num[MAX_DIMENSIONS];

				for(l=0;l<n_dimensions;l++){
					num[l] = f[l];	/* assume cinc=1 */
					num[l] -= min_array[l];
					num[l] /= width_array[l];
					num[l] += 0.5;
					index[l] = num[l];  /* cast to int */
					if( index[l] < 0 ){
						index[l]=0;
						n_under[l]++;
					} else if( index[l] >= n_bins[l] ){
						index[l] = n_bins[l]-1;
						n_over[l]++;
					}
				}

				histbuf[
					index[0] +
					index[1] * histo_dp->dt_rowinc +
					index[2] * histo_dp->dt_finc
					  ] += 1.0;

				f += data_dp->dt_pinc;
			}
			fptr += data_dp->dt_rowinc;
		}
		fbase += data_dp->dt_finc;
	}
	for(l=0;l<n_dimensions;l++){
		if( (n_under[l] > 0) || (n_over[l] > 0) ){
			sprintf(error_string,
	"Histogram for %s had %d underflows and %d overflows in dimension %d",
			data_dp->dt_name,n_under[l],n_over[l],l);
			advise(error_string);
		}
	}

}

