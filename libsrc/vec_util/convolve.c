
#include "quip_config.h"

char VersionId_vec_util_convolve[] = QUIP_VERSION_STRING;

#include "vec_util.h"

void img_clear(Data_Obj *dp)
{
	dimension_t i,j;
	float *ptr;
	incr_t xos, yos;

	ptr=(float *)dp->dt_data;
	for(j=0;j<dp->dt_rows;j++){
		yos = j * dp->dt_rowinc;
		for(i=0;i<dp->dt_cols;i++){
			xos = i * dp->dt_pinc;
			*(ptr + yos + xos ) = 0.0;
		}
	}
}

void add_impulse(double amp,Data_Obj *image_dp,Data_Obj *ir_dp,dimension_t x,dimension_t y)
{
	float *image_ptr, *irptr;
	incr_t i,j;
	incr_t yos,xos,offset;		/* offsets into image */
	incr_t iryos,iros;	/* offsets into impulse response */
	incr_t pinc, ir_pinc;

	pinc = image_dp->dt_pinc;
	ir_pinc = ir_dp->dt_pinc;

	image_ptr = (float *) image_dp->dt_data;
	irptr = (float *) ir_dp->dt_data;
	
	j=ir_dp->dt_rows;
	while( j-- ){			/* foreach impulse row */
		yos = ((y+j)-ir_dp->dt_rows/2);
#ifdef NOWRAP
		if( yos < 0 || yos >= (incr_t) image_dp->dt_rows ) continue;
#else
		if( yos < 0 ) yos+=image_dp->dt_rows;
		else if( yos >= image_dp->dt_rows ) yos-=image_dp->dt_rows;
#endif /* NOWRAP */
		yos *= image_dp->dt_rowinc;
		iryos = j * ir_dp->dt_rowinc;
		i=ir_dp->dt_cols;
		while(i--){
			xos = ((x+i)-ir_dp->dt_cols/2);
#ifdef NOWRAP
			if( xos < 0 || xos >= (incr_t) image_dp->dt_cols ) continue;
#else
			if( xos < 0 ) xos+=image_dp->dt_cols;
			else if( xos >= image_dp->dt_cols ) xos-=image_dp->dt_cols;
#endif /* NOWRAP */
			offset = (yos + xos*pinc);
			iros = (iryos + i*ir_pinc);

			*(image_ptr+offset) += *(irptr+iros) * amp;
		}
	}
}

void convolve(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpfilt)
{
	dimension_t i,j;
	float val, *frptr;
	dimension_t yos, offset;

	img_clear(dpto);

	frptr = (float *) dpfr->dt_data;
	j=dpto->dt_rows;
	while(j--){
		yos = j * dpfr->dt_rowinc;
		i=dpfr->dt_cols;
		while(i--){
			offset = yos+i*dpfr->dt_pinc;
			val = *(frptr+offset);
			add_impulse(val,dpto,dpfilt,i,j);
		}
	}
}

