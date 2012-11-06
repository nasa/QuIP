#include "quip_config.h"

char VersionId_vec_util_quads[] = QUIP_VERSION_STRING;

/* specialized routine to make up 4-tuples for color transparency simulations */

#include "vec_util.h"
#include "data_obj.h"

#define COPY_COMPONENTS( dst_col , src_row , src_col )			\
									\
			to = to_base + index * target->dt_rowinc	\
				+ dst_col * target->dt_pinc;		\
			from = from_base				\
				+ src_row * ri				\
				+ src_col * pi;				\
			for(i=0;i<td;i++){				\
				*to = *from;				\
				from += source->dt_cinc;		\
				to += target->dt_cinc;			\
			}

void make_all_quads(QSP_ARG_DECL  Data_Obj *target,Data_Obj *source)
{
	dimension_t n_col_pairs,n_row_pairs,n_quads;
	dimension_t n_cols, n_rows;
	dimension_t ri, pi, td;
	int index;
	float *from, *to, *from_base, *to_base;
	dimension_t first_col, second_col, first_row, second_row;
	dimension_t i;
	
	if( target->dt_comps != source->dt_comps ){
		WARN("target and source must have same number of components");
		return;
	}
	if( (target->dt_prec != PREC_SP) || (source->dt_prec != PREC_SP) ){
		WARN("target and source must be float precision");
		return;
	}
	if( target->dt_cols != 4 ){
		WARN("target should have four columns");
		return;
	}

	n_cols=source->dt_cols;
	n_rows=source->dt_rows;
	n_col_pairs = n_cols*(n_cols-1)/2;
	n_row_pairs = n_rows*(n_rows-1)/2;
	n_quads = n_col_pairs * n_row_pairs;

	if( target->dt_rows != n_quads ){
		sprintf(error_string,"target should have %d rows",n_quads);
		WARN(error_string);
		return;
	}

	index=0;
	ri = source->dt_rowinc;
	pi = source->dt_pinc;
	td = source->dt_comps;
	from_base = (float *)source->dt_data;
	to_base = (float *)target->dt_data;

	for(first_col=0;first_col<(n_cols-1);first_col++){
	    for(second_col=first_col+1;second_col<n_cols;second_col++){
		for(first_row=0;first_row<(n_rows-1);first_row++){
		    for(second_row=first_row+1;second_row<n_rows;second_row++){
			COPY_COMPONENTS(0,first_row,first_col);
			COPY_COMPONENTS(1,first_row,second_col);
			COPY_COMPONENTS(2,second_row,first_col);
			COPY_COMPONENTS(3,second_row,second_col);
			index++;
		    }
		}
	    }
	}
}

