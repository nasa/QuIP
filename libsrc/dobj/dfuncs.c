#include "quip_config.h"

#include <stdio.h>
#include <string.h>
#include "quip_prot.h"
#include "data_obj.h"
#include "function.h"

/* support for data functions in expr.y */

// could be integer return?

double obj_exists(QSP_ARG_DECL  const char *name)
{
	Data_Obj *dp;
	dp = dobj_of(QSP_ARG  name);
	if( dp==NO_OBJ ) return(0.0);
	return(1.0);
}

#define FETCH_BIT								\
										\
				bitnum_t bitnum;				\
				bitmap_word bit,*lp;				\
				bitnum = OBJ_BIT0(dp);				\
				lp = (bitmap_word *)OBJ_DATA_PTR(dp);		\
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

#ifdef FOOBAR
#ifdef HAVE_CUDA
	if( ! object_is_in_ram(DEFAULT_QSP_ARG  dp,
		"use value functions on CUDA object") ){
		return(0.0);
	}
#endif /* HAVE_CUDA */
#endif // FOOBAR
			
	if( !IS_SCALAR(dp) ){
		sprintf(DEFAULT_ERROR_STRING,"comp_func:  %s is not a scalar",
			OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(0.0);
	}
	if( OBJ_MACH_DIM(dp,0) <= (dimension_t)index ){
		sprintf(DEFAULT_ERROR_STRING,
		"Component index %d out of range for object %s",
			index,OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
	}
	mp = OBJ_MACH_PREC(dp);
	switch( mp ){
		case PREC_SP:
			d = (* (((float *)OBJ_DATA_PTR(dp))+index) );
			break;
		case PREC_DP:
			d = (* (((double *)OBJ_DATA_PTR(dp))+index) );
			break;
		case PREC_IN:
			d = (* (((short *)OBJ_DATA_PTR(dp))+index) );
			break;
		case PREC_DI:
			d = (* (((int32_t *)OBJ_DATA_PTR(dp))+index) );
			break;
		case PREC_LI:
			d = (* (((int64_t *)OBJ_DATA_PTR(dp))+index) );
			break;
		case PREC_BY:
			d = (* (((char *)OBJ_DATA_PTR(dp))+index) );
			break;
		case PREC_UIN:
			d = (* (((u_short *)OBJ_DATA_PTR(dp))+index) );
			break;
		case PREC_UDI:
			if( IS_BITMAP(dp) ){
				FETCH_BIT
			} else {
				d = (* (((uint32_t *)OBJ_DATA_PTR(dp))+index) );
			}
			break;
		case PREC_ULI:
			if( IS_BITMAP(dp) ){
				FETCH_BIT
			} else {
				d = (* (((uint64_t *)OBJ_DATA_PTR(dp))+index) );
			}
			break;
		case PREC_UBY:
			d = (* (((u_char *)OBJ_DATA_PTR(dp))+index) );
			break;
#ifdef CAUTIOUS
		case PREC_NONE:
		case N_MACHINE_PRECS:
		default:
			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  comp_func:  object %s has invalid machine precision %d",
				OBJ_NAME(dp),mp);
			NERROR1(DEFAULT_ERROR_STRING);
			d = 0.0;	// quiet compiler
			break;
#endif /* CAUTIOUS */
	}
	return(d);
} // end comp_func

double val_func(QSP_ARG_DECL  Data_Obj *dp )
{
	if( dp==NO_OBJ ) return(0.0);
	if( !IS_SCALAR(dp) ){
		sprintf(DEFAULT_ERROR_STRING,"val_func:  %s is not a scalar",
			OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(0.0);
	}
	if( OBJ_MACH_DIM(dp,0) > 1 ){
		sprintf(DEFAULT_ERROR_STRING,
			"value:  %s has %d components; returning comp. #0",
			OBJ_NAME(dp),OBJ_MACH_DIM(dp,0));
		NWARN(DEFAULT_ERROR_STRING);
	}
	return( comp_func(dp,0) );
}

static double re_func(QSP_ARG_DECL  Data_Obj *dp )
{
	if( dp==NO_OBJ ) return(0.0);
	if( !IS_SCALAR(dp) ){
		sprintf(DEFAULT_ERROR_STRING,
			"re_func:  %s is not a scalar",OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(0.0);
	}
	if( OBJ_MACH_DIM(dp,0) == 1 ){
		sprintf(DEFAULT_ERROR_STRING,
			"%s is real, not complex!?",OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
	} else if( OBJ_MACH_DIM(dp,0) != 2 ){
		sprintf(DEFAULT_ERROR_STRING,
			"%s is multidimensional, not complex!?",OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
	}
	return( comp_func(dp,0) );
}

static double im_func(QSP_ARG_DECL  Data_Obj *dp )
{
	if( dp==NO_OBJ ) return(0.0);
	if( !IS_SCALAR(dp) ){
		sprintf(DEFAULT_ERROR_STRING,
			"im_func:  %s is not a scalar",OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(0.0);
	}
	if( OBJ_MACH_DIM(dp,0) != 2 ){
		sprintf(DEFAULT_ERROR_STRING,
			"%s is not complex; returning 0.0", OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
	}
	return( comp_func(dp,1) );
}

static double contig_func(QSP_ARG_DECL  Data_Obj *dp )
{
	if( IS_CONTIGUOUS(dp) ){
		return(1);
	} else {
		return(0);
	}
}

static Data_Obj *obj_for_string(QSP_ARG_DECL  const char *string)
{
	Dimension_Set *dsp;
	Data_Obj *dp;

	INIT_DIMSET_PTR(dsp)

	/* this is just a string that we treat as a row vector
	 * of character data...
	 * We haven't actually created the data yet.
	 */
	SET_DIMENSION(dsp,0,1);
	SET_DIMENSION(dsp,1,(dimension_t)strlen(string)+1);
	SET_DIMENSION(dsp,2,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,4,1);
	dp=make_dobj(QSP_ARG  localname(),dsp,prec_for_code(PREC_STR));
	if( dp != NULL ){
		strcpy((char *)OBJ_DATA_PTR(dp),string);
	}
	return(dp);
}

void init_dfuncs(SINGLE_QSP_ARG_DECL)
{
	DECLARE_DOBJ_FUNCTION( value,		val_func	)
	DECLARE_DOBJ_FUNCTION( Re,		re_func		)
	DECLARE_DOBJ_FUNCTION( Im,		im_func		)
	DECLARE_DOBJ_FUNCTION( is_contiguous,	contig_func	)
	DECLARE_DOBJV_STR_ARG_FUNCTION( string_obj,	obj_for_string	)
	DECLARE_STR1_FUNCTION( obj_exists,	obj_exists	)
}

