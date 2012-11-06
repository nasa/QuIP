#include "quip_config.h"

char VersionId_dataf_dfuncs[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include "data_obj.h"
#include "function.h"

/* support for data functions in expr.y */

double obj_exists(QSP_ARG_DECL  const char *name)
{
	Data_Obj *dp;

	dp = dobj_of(QSP_ARG  name);
	if( dp==NO_OBJ ) return(0.0);
	return(1.0);
}

#define FETCH_BIT								\
										\
				unsigned int bitnum;				\
				bitmap_word bit,*lp;				\
				bitnum = dp->dt_bit0;				\
				lp = (bitmap_word *)dp->dt_data;		\
				lp += index/BITS_PER_BITMAP_WORD;		\
				bitnum += index % BITS_PER_BITMAP_WORD;		\
				if( bitnum >= BITS_PER_BITMAP_WORD ){		\
					bitnum -= BITS_PER_BITMAP_WORD;		\
					lp++;					\
				}						\
				bit = 1 << bitnum;				\
				if( *lp & bit )					\
					d = 1.0;				\
				else						\
					d = 0.0;

/* return a given component of the pointed-to pixel */

double comp_func( Data_Obj *dp, index_t index )
{
	double d;
	mach_prec mp;

	if( dp==NO_OBJ ) return(0.0);

#ifdef HAVE_CUDA
	if( ! IS_RAM(dp) ){
		sprintf(DEFAULT_ERROR_STRING,
			"Can't use value functions on CUDA device object %s",
			dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(0.0);
	}
#endif /* HAVE_CUDA */
			
	if( !IS_SCALAR(dp) ){
		sprintf(DEFAULT_ERROR_STRING,"comp_func:  %s is not a scalar",
			dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(0.0);
	}
	if( dp->dt_mach_dim[0] <= (dimension_t)index ){
		sprintf(DEFAULT_ERROR_STRING,
		"Component index %d out of range for object %s",
			index,dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
	}
	mp = MACHINE_PREC(dp);
	switch( mp ){
		case PREC_SP:
			d = (* (((float *)dp->dt_data)+index) );
			break;
		case PREC_DP:
			d = (* (((double *)dp->dt_data)+index) );
			break;
		case PREC_IN:
			d = (* (((short *)dp->dt_data)+index) );
			break;
		case PREC_DI:
			d = (* (((int32_t *)dp->dt_data)+index) );
			break;
		case PREC_LI:
			d = (* (((int64_t *)dp->dt_data)+index) );
			break;
		case PREC_BY:
			d = (* (((char *)dp->dt_data)+index) );
			break;
		case PREC_UIN:
			d = (* (((u_short *)dp->dt_data)+index) );
			break;
		case PREC_UDI:
			if( IS_BITMAP(dp) ){
				FETCH_BIT
			} else {
				d = (* (((uint32_t *)dp->dt_data)+index) );
			}
			break;
		case PREC_ULI:
			if( IS_BITMAP(dp) ){
				FETCH_BIT
			} else {
				d = (* (((uint64_t *)dp->dt_data)+index) );
			}
			break;
		case PREC_UBY:
			d = (* (((u_char *)dp->dt_data)+index) );
			break;
#ifdef CAUTIOUS
		case PREC_NONE:
		case N_MACHINE_PRECS:
		default:
			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  comp_func:  object %s has invalid machine precision %d",
				dp->dt_name,mp);
			NERROR1(DEFAULT_ERROR_STRING);
			d = 0.0;	// quiet compiler
			break;
#endif /* CAUTIOUS */
	}
	return(d);
} // end comp_func

double val_func( Data_Obj *dp )
{
	if( dp==NO_OBJ ) return(0.0);
	if( !IS_SCALAR(dp) ){
		sprintf(DEFAULT_ERROR_STRING,"val_func:  %s is not a scalar",
			dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(0.0);
	}
	if( dp->dt_mach_dim[0] > 1 ){
		sprintf(DEFAULT_ERROR_STRING,
			"value:  %s has %d components; returning comp. #0",
			dp->dt_name,dp->dt_mach_dim[0]);
		NWARN(DEFAULT_ERROR_STRING);
	}
	return( comp_func(dp,0) );
}

static double re_func( Data_Obj *dp )
{
	if( dp==NO_OBJ ) return(0.0);
	if( !IS_SCALAR(dp) ){
		sprintf(DEFAULT_ERROR_STRING,
			"re_func:  %s is not a scalar",dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(0.0);
	}
	if( dp->dt_mach_dim[0] == 1 ){
		sprintf(DEFAULT_ERROR_STRING,
			"%s is real, not complex!?",dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
	} else if( dp->dt_mach_dim[0] != 2 ){
		sprintf(DEFAULT_ERROR_STRING,
			"%s is multidimensional, not complex!?",dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
	}
	return( comp_func(dp,0) );
}

static double im_func( Data_Obj *dp )
{
	if( dp==NO_OBJ ) return(0.0);
	if( !IS_SCALAR(dp) ){
		sprintf(DEFAULT_ERROR_STRING,
			"im_func:  %s is not a scalar",dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(0.0);
	}
	if( dp->dt_mach_dim[0] != 2 ){
		sprintf(DEFAULT_ERROR_STRING,
			"%s is not complex; returning 0.0", dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
	}
	return( comp_func(dp,1) );
}

static double contig_func( Data_Obj *dp )
{
	if( IS_CONTIGUOUS(dp) ){
		return(1);
	} else {
		return(0);
	}
}

void init_dfuncs(void)
{
	setdatafunc("value",val_func);
	setdatafunc("Re",re_func);
	setdatafunc("Im",im_func);
	setdatafunc("is_contiguous",contig_func);
	setstrfunc("obj_exists",obj_exists);
}

