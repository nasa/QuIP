#include "quip_config.h"

#include "quip_prot.h"
#include "data_obj.h"
#include "pupfind.h"

/*
 * We want to see how much we can accelerate the computation by
 * coding it in C...  we don't really like doing this until we are
 * sure the algorithm is good!
 *
 * We assume all the initialization has been done in the script.
 * We assume there are 16 fields in the ring buffer.
 */

#define NF		16
#define THRESH		20

#define ZER_RAD		3
#define SEARCH_RADIUS	16

static void * frame_base[NF];	/* pointers to the frames in the ring buffer */
static int frames_known=0;

#define N 76800

static short *pdiff;
static float w10,w00,w11;

#define COMPUTE						\
							\
	s = + (*p1 + *p3) - ( *p2 + *p4 );		\
	*dst = s > THRESH ? s : THRESH;			\
	dst ++;						\
	p1 += 8;					\
	p2 += 8;					\
	p3 += 8;					\
	p4 += 8;

#define COMPUTE10					\
							\
	COMPUTE						\
	COMPUTE						\
	COMPUTE						\
	COMPUTE						\
	COMPUTE						\
	COMPUTE						\
	COMPUTE						\
	COMPUTE						\
	COMPUTE						\
	COMPUTE

#define COMPUTE100					\
							\
	COMPUTE10					\
	COMPUTE10					\
	COMPUTE10					\
	COMPUTE10					\
	COMPUTE10					\
	COMPUTE10					\
	COMPUTE10					\
	COMPUTE10					\
	COMPUTE10					\
	COMPUTE10

void compute_diff_image(int newest, int previous,int component_offset)
{
	register short *dst;
	register u_char *p1,*p2,*p3,*p4;
	short s;
	/* our target image (pdiff) is 240x320 = 76800 */
	int n=768;

	p1 = (u_char *)frame_base[ newest ];
	p2 = (u_char *)frame_base[ previous ];
	p1 += component_offset;
	p2 += component_offset;
	p3 = p1 + 4;			/* pixel offset */
	p4 = p2 + 4;
	dst = pdiff;
	
	while(n--){
		COMPUTE100
	}
}

static void setup_frame_ptrs(SINGLE_QSP_ARG_DECL)
{
	int i;
	char name[64];
	Data_Obj *dp;

	for(i=0;i<NF;i++){
		sprintf(name,"f%d",i);
		dp = GET_OBJ(name);
		if( dp == NULL ) error1("missing frame object");
		frame_base[i] = OBJ_DATA_PTR(dp);
	}
	frames_known=1;
}

COMMAND_FUNC( setup_diff_computation )
{
	Data_Obj *dp;

	dp=GET_OBJ("pdiff");
	if( dp == NULL ) error1("missing destination object 'pdiff'");

	if( OBJ_COLS(dp) != 320 ){
		sprintf(ERROR_STRING,"Object pdiff (%d) should have 320 columns",OBJ_COLS(dp));
		error1(ERROR_STRING);
	}

	if( OBJ_ROWS(dp) != 240 ){
		sprintf(ERROR_STRING,"Object pdiff (%d) should have 240 rows",OBJ_ROWS(dp));
		error1(ERROR_STRING);
	}

	pdiff = (short *)OBJ_DATA_PTR(dp);

	if( !frames_known ) setup_frame_ptrs(SINGLE_QSP_ARG);
}

void _setup_blur(QSP_ARG_DECL  Data_Obj *dp)
{
	/* BUG need to do checks here */

	float *f;

	f=(float *)OBJ_DATA_PTR(dp);
	w11 = *f;
	f++;
	w10 = *f;
	f+=3;
	w00 = *f;
sprintf(DEFAULT_ERROR_STRING,"w11 = %g, w10 = %g, w00 = %g",w11,w10,w00);
advise(DEFAULT_ERROR_STRING);
}

/*
 * Compute gaussian curvature:
 *
 * gxx*gyy - gxy^2
 *
 * input image patch:
 *
 * 	a	b	c	d	e
 * 	f	g	h	i	j
 * 	k	l	m	n	o
 * 	p	q	r	s	t
 * 	u	v	w	x	y
 *
 * gx:
 *
 * 	.	c-a	d-b	e-c	.
 * 	.	h-f	i-g	j-h	.
 * 	.	m-k	n-l	o-m	.
 * 	.	r-p	s-q	t-r	.
 * 	.	w-u	x-v	y-w	.
 *
 * gy:
 *
 * 	.	.	.	.	.
 * 	k-a	l-b	m-c	n-d	o-e
 * 	p-f	q-g	r-h	s-i	t-j
 * 	u-k	v-l	w-m	x-n	y-o
 * 	.	.	.	.	.
 *
 * gxy:
 *
 * 	.	.	.	.	.
 * 	.	m+a-c-k	n+b-d-l	o+c-e-m	.
 * 	.	r+f-h-p	s+g-i-q	t+h-j-r	.
 * 	.	w+k-m-u	x+l-n-v	y+m-o-w	.
 * 	.	.	.	.	.
 * gxx:
 *
 * 			e+a-2c
 * 			j+f-2h
 * 			o+k-2m
 * 			t+p-2r
 * 			y+u-2w
 *
 * gyy:
 *
 * 	.	.	.	.	.
 * 	.	.	.	.	.
 * 	u+a-2k	v+b-2l	w+c-2m	x+d-2n	y+e-2o
 * 	.	.	.	.	.
 * 	.	.	.	.	.
 *
 * input image patch:
 *
 * 	a	b	c	d	e
 * 	f	g	h	i	j
 * 	k	l	m	n	o
 * 	p	q	r	s	t
 * 	u	v	w	x	y
 *
 * gxx*gyy - gxy^2
 *
 * gxx:		o+k-2m		(or	n+l-2m	)
 * gyy:		w+c-2m		(or	r+h-2m	)
 * gxy:		s+g-i-q			s+g-i-q
 *
 * gxx*gyy	(o+k-2m)*(w+c-2m)	(n+l-2m)*(r+h-2m)
 * 		ow+oc+kw+kc		nr+nh+lr+lh
 * 		  -2m(o+k+w+c)		  -2m(n+l+r+h)
 * 		  +4mm			  +4mm
 *
 * gxy^2	(s+g-i-q)(s+g-i-q)
 * 		ss+gg+ii+qq
 * 		  +2(sg+iq-si-sq-gi-gq)
 *
 * on a 3x3 patch:
 *
 * 		a	b	c
 * 		d	e	f
 * 		g	h	i
 *
 * gxx:		d+f-2e
 * gyy:		b+h-2e
 * gxy:		a+i-g-c
 *
 */


static float *curv_dst;
static float *fcurv_dst;

COMMAND_FUNC( setup_curv_computation )
{
	Data_Obj *dp;

	dp=GET_OBJ("curv");
	if( dp == NULL ) error1("setup_curv_computation:  missing curvature object");

	curv_dst = (float *)OBJ_DATA_PTR(dp);

	dp=GET_OBJ("fcurv");
	if( dp == NULL ) error1("setup_curv_computation:  missing filtered curvature object");

	fcurv_dst = (float *)OBJ_DATA_PTR(dp);

	/* BUG verify type, size, contiguity here */

	if( !frames_known ) setup_frame_ptrs(SINGLE_QSP_ARG);
}

#define N_CURV_ROWS	128
#define N_CURV_COLS	128
#define FIRST_SOURCE_ROW	((240-(2+N_CURV_ROWS))/2)
#define FIRST_SOURCE_COL	((640-(2+N_CURV_COLS))/2)
#define BYTES_PER_ROW		2560

#define COMPUTE_CURVATURE					\
								\
			gxx = a + f - 2 * e;			\
			gyy = b + h - 2 * e;			\
			gxy = a + i - g - c;			\
			gc = gxx * gyy - gxy * gxy;		\
			*dst++ = gc>0.0 ? gc : 0.0 ;

#define DEBUG_DUMP2									\
sprintf(ERROR_STRING,"k=%d\t dst = 0x%lx\tframe_base[%d] = 0x%lx\tp_2 = 0x%lx",				\
		k,dst,frame_index,frame_base[frame_index],p_2);				\
advise(ERROR_STRING);

#define DEBUG_DUMP									\
sprintf(ERROR_STRING,"frame_base[%d] = 0x%lx\tp_2 = 0x%lx\te = %g\tdst = 0x%x",		\
		frame_index,frame_base[frame_index],p_2,e,dst);				\
advise(ERROR_STRING);

#define EYE_CAM_INDEX	2
#define N_TRIPLETS	(N_CURV_ROWS/3)

void compute_curvature(int frame_index)
{
	unsigned char *p_1;
	unsigned char *p_2;
	unsigned char *p_3;
	unsigned char *row1_base,*row2_base,*row3_base;
	float	v11,	v12,	v13;
	float	v21,	v22,	v23;
	float	v31,	v32,	v33;
	float gxx,gyy,gxy,gc;
	float *dst;
	int k,j;

	dst = curv_dst;

	row1_base = (unsigned char *)frame_base[frame_index];
	row1_base += EYE_CAM_INDEX;
	row1_base += FIRST_SOURCE_ROW * BYTES_PER_ROW;
	row1_base += 4 * FIRST_SOURCE_COL;

	row2_base = row1_base + BYTES_PER_ROW;
	row3_base = row2_base + BYTES_PER_ROW;

	/* For now we assume the source region is centered...
	 * Ultimately we will want to pass coordinates.
	 */

	for(k=0;k<N_CURV_ROWS;k++){
		p_1 = row1_base;	p_2 = row2_base;	p_3 = row3_base;

		v11 = *p_1;	v21 = *p_2;	v31 = *p_3;
		p_1 +=4;	p_2 +=4;	p_3 +=4;
		v12 = *p_1;	v22 = *p_2;	v32 = *p_3;

		for(j=0;j<N_TRIPLETS;j++){
#include "curv_setup1.h"
			p_1 +=4;	p_2 +=4;	p_3 +=4;
			c=(*p_1);	f=(*p_2);	i=(*p_3);

			COMPUTE_CURVATURE


#include "curv_setup2.h"
			p_1 +=4;	p_2 +=4;	p_3 +=4;
			c=(*p_1);	f=(*p_2);	i=(*p_3);

			COMPUTE_CURVATURE

#include "curv_setup3.h"
			p_1 +=4;	p_2 +=4;	p_3 +=4;
			c=(*p_1);	f=(*p_2);	i=(*p_3);

			COMPUTE_CURVATURE
		}
		/* Assuming there are 128 columns in the curvature image,
		 * we have executed the column loop floor(128/3) = 126/3 = 42 times
		 * So we have to do two more interations
		 */
#include "curv_setup1.h"
		p_1 +=4;	p_2 +=4;	p_3 +=4;
		c=(*p_1);	f=(*p_2);	i=(*p_3);

		COMPUTE_CURVATURE

#include "curv_setup2.h"
		p_1 +=4;	p_2 +=4;	p_3 +=4;
		c=(*p_1);	f=(*p_2);	i=(*p_3);

		COMPUTE_CURVATURE

		row1_base = row2_base;
		row2_base = row3_base;
		row3_base += BYTES_PER_ROW;		/* rowinc = 640*4 */
	}
}

/* We normalize the 3x3 symmetric kernel so that the corners have a value
 * of 1.  This is OK since we are only looking for the maximum...
 * Then because of symmetry, we only need to remember two weights.
 *
 * 	a	b	c
 * 	d	e	f
 * 	g	h	i
 */

#define APPLY_FILTER						\
								\
	v = w11 * ( a + c + g + i );				\
	v += w10 * ( b + d + f + h );				\
	*dst++ = v + w00 * e;


/* assume a 3x3 kernel, let's see how fast or slow this is */

COMMAND_FUNC( blur_curvature )
{
	float *p_1;
	float *p_2;
	float *p_3;
	float *row1_base,*row2_base,*row3_base;
	float	v11,	v12,	v13;
	float	v21,	v22,	v23;
	float	v31,	v32,	v33;
	float *dst;
	float v;
	int k,j;

	dst = fcurv_dst;

#define FLOATS_PER_CURV_ROW	128	/* 128 * sizeof(float) */
	row1_base = curv_dst;
	row2_base = row1_base + FLOATS_PER_CURV_ROW;
	row3_base = row2_base + FLOATS_PER_CURV_ROW;

	/* For now we assume the source region is centered...
	 * Ultimately we will want to pass coordinates.
	 */

	/* for a 3x3 convolution on 128, we have 42 loops exactly... */

	k=128;
	while(k--) *dst++=0.0;

	for(k=0;k<N_CURV_ROWS-2;k++){
		p_1 = row1_base;	p_2 = row2_base;	p_3 = row3_base;

		v11 = *p_1;  v21 = *p_2;  v31 = *p_3;
		p_1++;       p_2++;       p_3++;
		v12 = *p_1;  v22 = *p_2;  v32 = *p_3;

		*dst++ = 0.0;

		for(j=0;j<N_TRIPLETS;j++){
#include "curv_setup1.h"
			p_1++;       p_2++;       p_3++;
			c=(*p_1);    f=(*p_2);    i=(*p_3);
			APPLY_FILTER
#include "curv_setup2.h"
			p_1++;       p_2++;       p_3++;
			c=(*p_1);    f=(*p_2);    i=(*p_3);
			APPLY_FILTER
#include "curv_setup3.h"
			p_1++;       p_2++;       p_3++;
			c=(*p_1);    f=(*p_2);    i=(*p_3);
			APPLY_FILTER
		}

		*dst++ = 0.0;

		row1_base=row2_base;
		row2_base=row3_base;
		row3_base+=128;
	}
	k=128;
	while(k--) *dst++=0.0;
}

