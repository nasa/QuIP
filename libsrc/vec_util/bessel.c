
#include "quip_config.h"

char VersionId_vec_util_bessel[] = QUIP_VERSION_STRING;

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "data_obj.h"
#include "vec_util.h"

/* local prototypes */
static int checkem(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,const char *whence);

static int checkem(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,const char *whence)
{
	if( whence == NULL ) whence="checkem";

	if( !dp_same(QSP_ARG  dpto,dpfr,whence) ) return(-1);

	if( dpto->dt_prec != PREC_SP ){
		WARN("bessel precision must be float");
		return(-1);
	}
	if( !IS_CONTIGUOUS(dpto) || !IS_CONTIGUOUS(dpfr) ){
		NWARN("Sorry, bessel images must be contiguous");
		return(-1);
	}
	if( dpto->dt_seqs > 1 ){
		NWARN("Sorry, bessel only works for single sequences");
		return(-1);
	}
	if( dpto->dt_comps > 1 ){
		NWARN("Sorry, bessel only works for single component");
		return(-1);
	}
	return(0);
}

int bessel_of(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,int order)
{
	register float *toptr, *frptr;
	register dimension_t k,j,i;

	if( checkem(QSP_ARG  dpto,dpfr,"bessel_of") < 0 ) return(-1);

	toptr = (float *)dpto->dt_data;
	frptr = (float *)dpfr->dt_data;

	if( order == 0 ){
		for(i=0;i<dpto->dt_frames;i++)
			for(j=0;j<dpto->dt_rows;j++)
				for(k=0;k<dpto->dt_cols;k++)
					*toptr++ = j0( *frptr++ );
	} else if ( order == 1 ) {
		for(i=0;i<dpto->dt_frames;i++)
			for(j=0;j<dpto->dt_rows;j++)
				for(k=0;k<dpto->dt_cols;k++)
					*toptr++ = j1( *frptr++ );
	} else {
		for(i=0;i<dpto->dt_frames;i++)
			for(j=0;j<dpto->dt_rows;j++)
				for(k=0;k<dpto->dt_cols;k++)
					*toptr++ = jn( order, *frptr++ );
	}
	return(0);
}

int acos_of(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr)
{
	register float *toptr, *frptr;
	register dimension_t k,j,i;

	if( checkem(QSP_ARG  dpto,dpfr,"acos_of") < 0 ) return(-1);

	toptr = (float *)dpto->dt_data;
	frptr = (float *)dpfr->dt_data;

	for(i=0;i<dpto->dt_frames;i++)
		for(j=0;j<dpto->dt_rows;j++)
			for(k=0;k<dpto->dt_cols;k++)
				*toptr++ = acos( *frptr++ );

	return(0);
}

int asin_of(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr)
{
	register float *toptr, *frptr;
	register dimension_t k,j,i;

	if( checkem(QSP_ARG  dpto,dpfr,"asin_of") < 0 ) return(-1);

	toptr = (float *)dpto->dt_data;
	frptr = (float *)dpfr->dt_data;

	for(i=0;i<dpto->dt_frames;i++)
		for(j=0;j<dpto->dt_rows;j++)
			for(k=0;k<dpto->dt_cols;k++)
				*toptr++ = asin( *frptr++ );

	return(0);
}



