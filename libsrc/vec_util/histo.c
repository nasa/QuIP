#include "quip_config.h"

/* histogram function for data objects */

#include "vec_util.h"
#include "quip_prot.h"

static void zero_dimension(Data_Obj *dp,float *base,int dim,long index);

// it would be convenient to support double destination also?

#define HISTOGRAM(src_type)						\
									\
	{								\
	src_type *frm_base, *row_base, *p_ptr;				\
	frm_base = (src_type *) OBJ_DATA_PTR(data_dp);			\
	n_bins = OBJ_COLS(histo_dp);					\
	histbuf = (float *) OBJ_DATA_PTR(histo_dp);			\
									\
	for(i=0;i<n_bins;i++)						\
		*( histbuf + i* OBJ_PXL_INC(histo_dp)) =0;		\
									\
	for(i=0;i<OBJ_FRAMES(data_dp);i++){				\
		row_base = frm_base;					\
		for(j=0;j<OBJ_ROWS(data_dp);j++){			\
			p_ptr=row_base;					\
			for(k=0;k<OBJ_COLS(data_dp);k++){		\
				num = (float) *p_ptr;			\
				num -= min_limit;			\
				num /= bin_width;			\
				num += 0.5;				\
				index = (incr_t)num;	/* conv. to int. */ \
				if( index < 0 ){			\
					index=0;			\
					n_under++;			\
				} else if( index >= (incr_t) n_bins ){	\
					index = (incr_t)n_bins-1;	\
					n_over++;			\
				}					\
				* ( histbuf + index*OBJ_PXL_INC(histo_dp) ) += 1.0;		\
									\
				p_ptr += OBJ_PXL_INC(data_dp);		\
			}						\
			row_base += OBJ_ROW_INC(data_dp);		\
		}							\
		frm_base += OBJ_FRM_INC(data_dp);			\
	}								\
	}

void _compute_histo(QSP_ARG_DECL  Data_Obj *histo_dp,Data_Obj *data_dp,double bin_width,double min_limit)
{
	dimension_t i,j,k;
	float num;
	float *histbuf;
	incr_t index;
	dimension_t n_bins;
	int n_under=0, n_over=0;

	INSIST_RAM_OBJ(histo_dp,compute_histo);
	INSIST_RAM_OBJ(data_dp,compute_histo);

	if( OBJ_PREC(histo_dp) != PREC_SP ){
		WARN("histogram precision must be float");
		return;
	}
	if( OBJ_COMPS(histo_dp) != 1 ){
		WARN("histogram data must be real");
		return;
	}
	if( OBJ_ROWS(histo_dp) > 1 || OBJ_FRAMES(histo_dp) > 1 ){
		WARN("only using first row of histogram image");
	}
	if( OBJ_COMPS(data_dp) != 1 ){
		WARN("input data must be real");
		return;
	}
	switch( OBJ_PREC(data_dp) ){
		case PREC_SP: HISTOGRAM(float) break;
		case PREC_DP: HISTOGRAM(double) break;
		case PREC_UBY: HISTOGRAM(u_char) break;
		case PREC_BY: HISTOGRAM(char) break;
		case PREC_UIN: HISTOGRAM(u_short) break;
		case PREC_IN: HISTOGRAM(short) break;
		case PREC_UDI: HISTOGRAM(u_long) break;
		case PREC_DI: HISTOGRAM(long) break;
		default:
			sprintf(ERROR_STRING,"Sorry, precision %s not allowed for histogram source",PREC_NAME(OBJ_PREC_PTR(data_dp)));
			WARN(ERROR_STRING);
			return;
	}

	if( (n_under > 0) || (n_over > 0) ){
		sprintf(ERROR_STRING,
			"Histogram for %s had %d underflows and %d overflows",
			OBJ_NAME(data_dp),n_under,n_over);
		advise(ERROR_STRING);
	}
}

#define MAX_DIMENSIONS	(N_DIMENSIONS-1)


static void zero_dimension(Data_Obj *dp,float *base,int dim,long index)
{
	dimension_t i;

	if( dim > 1 ){
		for(i=0;i<OBJ_TYPE_DIM(dp,dim);i++)
			zero_dimension(dp,base+i*OBJ_TYPE_INC(dp,dim),dim-1,i);
	} else {
		for(i=0;i<OBJ_COLS(dp);i++)
			base[i] = 0.0;
	}
}

void _multivariate_histo(QSP_ARG_DECL  Data_Obj *histo_dp,Data_Obj *data_dp,float *width_array,float *min_array)
{
	dimension_t n_dimensions;
	dimension_t i,j,k;
	unsigned int l;
	float *fbase, *fptr, *f;
	float *histbuf;
	incr_t index[MAX_DIMENSIONS];
	int n_bins[MAX_DIMENSIONS];
	int n_under[MAX_DIMENSIONS], n_over[MAX_DIMENSIONS];

	INSIST_RAM_OBJ(histo_dp,compute_histo);
	INSIST_RAM_OBJ(data_dp,compute_histo);

	if( OBJ_PREC(histo_dp) != PREC_SP ){
		warn("2D histogram precision must be float");
		return;
	}
	if( OBJ_COMPS(histo_dp) != 1 ){
		warn("2D histogram data must be real");
		return;
	}
	if( OBJ_PXL_INC(histo_dp) != 1 ){
		warn("2D histogram data must be contiguous");
		return;
	}

	n_dimensions = OBJ_COMPS(data_dp);

	if( n_dimensions > MAX_DIMENSIONS ){
		warn("Too many 2D histogram dimensions");
		return;
	}

	if( OBJ_PREC(data_dp) != PREC_SP ){
		warn("2D data precision must be float");
		return;
	}

	fbase = (float *) OBJ_DATA_PTR(data_dp);

	for(l=0;l<n_dimensions;l++){
		n_over[l]=0;
		n_under[l]=0;
		n_bins[l] = OBJ_TYPE_DIM(histo_dp,l+1);
	}

	histbuf = (float *) OBJ_DATA_PTR(histo_dp);

	zero_dimension(histo_dp,(float *)OBJ_DATA_PTR(histo_dp),n_dimensions,0L);
	for(l=0;l<MAX_DIMENSIONS;l++)
		index[l]=0;

	for(i=0;i<OBJ_FRAMES(data_dp);i++){
		fptr = fbase;
		for(j=0;j<OBJ_ROWS(data_dp);j++){
			f=fptr;
			for(k=0;k<OBJ_COLS(data_dp);k++){
				float num[MAX_DIMENSIONS];

				for(l=0;l<n_dimensions;l++){
					num[l] = f[l];	/* assume cinc=1 */
					num[l] -= min_array[l];
					num[l] /= width_array[l];
					num[l] += 0.5;
					index[l] = (incr_t)num[l];  /* cast to int */
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
					index[1] * OBJ_ROW_INC(histo_dp) +
					index[2] * OBJ_FRM_INC(histo_dp)
					  ] += 1.0;

				f += OBJ_PXL_INC(data_dp);
			}
			fptr += OBJ_ROW_INC(data_dp);
		}
		fbase += OBJ_FRM_INC(data_dp);
	}
	for(l=0;l<n_dimensions;l++){
		if( (n_under[l] > 0) || (n_over[l] > 0) ){
			sprintf(ERROR_STRING,
	"Histogram for %s had %d underflows and %d overflows in dimension %d",
			OBJ_NAME(data_dp),n_under[l],n_over[l],l);
			advise(ERROR_STRING);
		}
	}

}

