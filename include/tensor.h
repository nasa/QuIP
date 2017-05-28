#ifndef _TENSOR_H_
#define _TENSOR_H_

#include "data_obj.h"

struct tensor {
	Item			tns_item;
	int			tns_n_dimensions;
	dimension_t *		tns_dim_tbl;
	increment_t *		tns_inc_tbl;
	struct data_info	tns_info;
};


#endif // ! _TENSOR_H_

