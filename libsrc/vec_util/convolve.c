
#include "quip_config.h"
#include "quip_prot.h"

#include "vec_util.h"

void img_clear(Data_Obj *dp)
{
	dimension_t i,j;
	float *ptr;
	incr_t xos, yos;

	ptr=(float *)OBJ_DATA_PTR(dp);
	for(j=0;j<OBJ_ROWS(dp);j++){
		yos = j * OBJ_ROW_INC(dp);
		for(i=0;i<OBJ_COLS(dp);i++){
			xos = i * OBJ_PXL_INC(dp);
			*(ptr + yos + xos ) = 0.0;
		}
	}
}

void add_impulse(double amp,Data_Obj *image_dp,Data_Obj *ir_dp,posn_t x,posn_t y)
{
	float *image_ptr, *irptr;
	incr_t i,j;
	incr_t yos,xos,offset;		/* offsets into image */
	incr_t iryos,iros;	/* offsets into impulse response */
	incr_t pinc, ir_pinc;

	pinc = OBJ_PXL_INC(image_dp);
	ir_pinc = OBJ_PXL_INC(ir_dp);

	image_ptr = (float *) OBJ_DATA_PTR(image_dp);
	irptr = (float *) OBJ_DATA_PTR(ir_dp);
	
	j=OBJ_ROWS(ir_dp);
	while( j-- ){			/* foreach impulse row */
		yos = ((y+j)-OBJ_ROWS(ir_dp)/2);
#ifdef NOWRAP
		if( yos < 0 || yos >= (incr_t) OBJ_ROWS(image_dp) ) continue;
#else
		if( yos < 0 ) yos+=OBJ_ROWS(image_dp);
		else if( yos >= OBJ_ROWS(image_dp) ) yos-=OBJ_ROWS(image_dp);
#endif /* NOWRAP */
		yos *= OBJ_ROW_INC(image_dp);
		iryos = j * OBJ_ROW_INC(ir_dp);
		i=OBJ_COLS(ir_dp);
		while(i--){
			xos = ((x+i)-OBJ_COLS(ir_dp)/2);
#ifdef NOWRAP
			if( xos < 0 || xos >= (incr_t) OBJ_COLS(image_dp) ) continue;
#else
			if( xos < 0 ) xos+=OBJ_COLS(image_dp);
			else if( xos >= OBJ_COLS(image_dp) ) xos-=OBJ_COLS(image_dp);
#endif /* NOWRAP */
			offset = (yos + xos*pinc);
			iros = (iryos + i*ir_pinc);

			*(image_ptr+offset) += *(irptr+iros) * amp;
		}
	}
}

void _convolve(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpfilt)
{
	dimension_t i,j;
	float val, *frptr;
	dimension_t yos, offset;

	// where is the other error checking done???
	INSIST_RAM_OBJ(dpto,convolve)
	INSIST_RAM_OBJ(dpfr,convolve)
	INSIST_RAM_OBJ(dpfilt,convolve)

	img_clear(dpto);

	frptr = (float *) OBJ_DATA_PTR(dpfr);
	j=OBJ_ROWS(dpto);
	while(j--){
		yos = j * OBJ_ROW_INC(dpfr);
		i=OBJ_COLS(dpfr);
		while(i--){
			offset = yos+i*OBJ_PXL_INC(dpfr);
			val = *(frptr+offset);
			add_impulse(val,dpto,dpfilt,i,j);
		}
	}
}

