#include "quip_config.h"

/* based on DCT code written by John McGowan */

/* #include <stdio.h> */
/* #include <stdlib.h> */
/* #include <string.h> */
#include <math.h>
/* #include <assert.h> */
/* #include <memory.h> */

#include "quip_prot.h"
#include "vec_util.h"
#include "data_obj.h"

#include "dct8.h"

static dct_type r11, r12, r13; 	/* 2nd stage, lines 5 & 6, a, b-a, -b-a */
static dct_type r21, r22, r23;            	/* lines 4 & 7 */
static dct_type r31, r32, r33; 	/* 3rd stage, lines 2 & 6 */
static dct_type sqrt2, sqrt8;  	/* (see Figures 1 & 2, Equation 2) */

#define PI (3.14159267)

static void setup(void);
static void mat8_1D(QSP_ARG_DECL  dct_type *f);
static void imat8_1D(QSP_ARG_DECL  dct_type *f);
static void dct8_1D(dct_type *f);
static void dct8_2D(QSP_ARG_DECL  Data_Obj *dp,int);
static void init_mat(SINGLE_QSP_ARG_DECL);


/* Take the dct of an 8x8 block.
 * Typically, this will be a subimage of a larger image.
 */

static void dct8_2D(QSP_ARG_DECL  Data_Obj *dp,int direction)
{
	int row, col;
	float *ptr;
	dct_type g[DCT_SIZE];

#ifdef CAUTIOUS
	if( OBJ_COLS(dp) != DCT_SIZE || OBJ_ROWS(dp) != DCT_SIZE ){
		sprintf(ERROR_STRING,"Object %s should be 8x8 for DCT",OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_PREC(dp) != PREC_SP ){
		sprintf(ERROR_STRING,"Object %s has prec %s, should be float for DCT",
			OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_COMPS(dp) != 1 ){
		sprintf(ERROR_STRING,"Object %s has %d components, should be 1 for DCT",
			OBJ_NAME(dp),OBJ_COMPS(dp));
		WARN(ERROR_STRING);
		return;
	}
#endif	/* CAUTIOUS */

	setup();		/* initialize constants if necessary */

	/* apply 1D dct to each row */

	ptr = (float *)OBJ_DATA_PTR(dp);

	for (row=0 ; row < DCT_SIZE ; row++) {
		/* copy row */
		for (col=0 ; col < DCT_SIZE ; col++)
			g[col] = *(ptr+col*OBJ_PXL_INC(dp));

		if( direction == OLD_DCT )
			dct8_1D(g);

		else if( direction == FWD_DCT )
			mat8_1D(QSP_ARG  g);
		else
			imat8_1D(QSP_ARG  g);

		/* copy back */
		for (col=0 ; col < DCT_SIZE ; col++)
			*(ptr+col*OBJ_PXL_INC(dp)) = (float)g[col];

		/* next row */
		ptr += OBJ_ROW_INC(dp);
	}

	/* apply 1D dct to each col */

	ptr = (float *)OBJ_DATA_PTR(dp);

	for (col=0 ; col < DCT_SIZE ; col++) {
		for (row=0; row < DCT_SIZE ; row++)
			g[row] = *(ptr+row*OBJ_ROW_INC(dp));

		/* dct8_1D(g); */

		if( direction == OLD_DCT )
			dct8_1D(g);

		else if( direction == FWD_DCT )
			mat8_1D(QSP_ARG  g);
		else
			imat8_1D(QSP_ARG  g);


		for (row=0; row < DCT_SIZE ; row++)
			*(ptr+row*OBJ_ROW_INC(dp)) = (float) g[row];

		/* next col */
		ptr += OBJ_PXL_INC(dp);
	}
}

static void dct8_1D(dct_type *f)
{
	dct_type s10, s11, s12, s13, s14, s15, s16, s17,	/* sij = signal on */
		s20, s21, s22, s23, s24, s25, s26, s27,		/* line j after */
		s34, s35, s36, s37, t;                      	/* stage i */
								/* (see Figure 1) */
	setup();

	/* stage 1 (see Figures 1 & 2 in Loeffler et al) */

	s10 = f[0] + f[7];
	s11 = f[1] + f[6];
	s12 = f[2] + f[5];
	s13 = f[3] + f[4];
	s14 = f[3] - f[4];
	s15 = f[2] - f[5];
	s16 = f[1] - f[6];
	s17 = f[0] - f[7];

	/* stage 2 */

	s20 = s10 + s13;
	s21 = s11 + s12;
	s22 = s11 - s12;
	s23 = s10 - s13;

	t = (s14 + s17) * r21;
	s24 = s17 * r22 + t;
	s27 = s14 * r23 + t;

	t = (s15 + s16) * r11;
	s25 = s16 * r12 + t;
	s26 = s15 * r13 + t;

	/* stage 3 */
	/* s30 */ f[0] = s20 + s21;
	/* s31 */ f[4] = s20 - s21;

	t = (s22 + s23) * r31;
	/* s32 */ f[2] = s23 * r32 + t;	/* s23 * (v-r31) + (s22+s23)*r31 */
	                               	/* s23 * v + s22*r31 + */
	                               	/* s23 * sin(u) + s22*cos(u) + */
	/* s33 */ f[6] = s22 * r33 + t;

	s34 = s24 + s26;
	s35 = s27 - s25;
	s36 = s24 - s26;
	s37 = s27 + s25;

	/* stage 4 */
	f[7] = s37 - s34;
	f[3] = s35 * sqrt2;
	f[5] = s36 * sqrt2;
	f[1] = s37 + s34;

	/* scale */
	f[0] /= sqrt8;
	f[1] /= sqrt8;
	f[2] /= sqrt8;
	f[3] /= sqrt8;
	f[4] /= sqrt8;
	f[5] /= sqrt8;
	f[6] /= sqrt8;
	f[7] /= sqrt8;
}


/* factors for Loeffler et al algorithm (see Figures 1 & 2, Equation 2) */

static void setup(void)
{
	dct_type pi, u, v;
	static int done_once=0;

	if( done_once ) return;

	sqrt2 = sqrt(2.0);
	sqrt8 = sqrt2 * 2.0;
	pi = 4.0 * atan(1.0);

	u = pi / 16.0;   	/* phase for stage 2 rotation 'c1' */
	v = sin(u);
	r11 = cos(u);         	/*  'a' in Equation 2 */
	r12 =  v - r11;       	/*  b-a */
	r13 = -v - r11;       	/* -b-a */

	u *= 3.0;        	/* phase for stage 2 rotation 'c3' */
	v = sin(u);
	r21 = cos(u);         	/*   a  */
	r22 =  v - r21;       	/*  b-a */
	r23 = -v - r21;       	/* -b-a */

	u *= 2.0;        	/* phase for stage 3 rotation "sqrt(2)c6" (note typo) */
	v = sin(u) * sqrt2;
	r31 = cos(u) * sqrt2; 	/*   a  */
	r32 =  v - r31;       	/*  b-a */
	r33 = -v - r31;       	/* -b-a */

	done_once = 1;
}

void compute_dct(QSP_ARG_DECL  Data_Obj *dp,int direction)
{
	dimension_t nx,ny;		/* number of blocks in x and y */
	dimension_t i,j;
	Data_Obj *block_dp;


	/* some fatal errors */

	if( OBJ_PREC(dp) != PREC_SP ){
		sprintf(ERROR_STRING,"Object %s has prec %s, should be float for DCT",
			OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_COMPS(dp) != 1 ){
		sprintf(ERROR_STRING,"Object %s has %d components, should be 1 for DCT",
			OBJ_NAME(dp),OBJ_COMPS(dp));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_FRAMES(dp) != 1 ){
		sprintf(ERROR_STRING,"Object %s has %d frames, should be 1 for DCT",
			OBJ_NAME(dp),OBJ_FRAMES(dp));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_SEQS(dp) != 1 ){
		sprintf(ERROR_STRING,"Object %s has %d seqs, should be 1 for DCT",
			OBJ_NAME(dp),OBJ_SEQS(dp));
		WARN(ERROR_STRING);
		return;
	}

	/* some non-fatal errors */

	if( (OBJ_COLS(dp)%8) != 0 || (OBJ_ROWS(dp)%8) != 0 ){
		sprintf(ERROR_STRING,"Image %s dimension(s) not a multiple of 8 for DCT",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
	}

	block_dp=mk_subimg(dp,0,0,"dct_block",DCT_SIZE,DCT_SIZE);
	if( block_dp == NULL )
		WARN("couldn't create subimage for DCT block");

	nx = OBJ_COLS(dp)/DCT_SIZE;
	ny = OBJ_ROWS(dp)/DCT_SIZE;

	for(i=0;i<nx;i++){
		for(j=0;j<ny;j++){
			relocate(block_dp,i*DCT_SIZE,j*DCT_SIZE,0);
			dct8_2D(QSP_ARG  block_dp,direction);
		}
	}

	delvec(block_dp);
}


/* Here is a dct algorithm based on a matrix multiplication */

/* static dct_type dct_mat[DCT_SIZE][DCT_SIZE]; */
static Data_Obj *mat_dp;

static int mat_inited=0;

static void init_mat(SINGLE_QSP_ARG_DECL)
{
	int i,j;
	dct_type pi,v;
	float *ptr;

	if( mat_inited ) return;

	mat_dp = mk_img("dct_mat",DCT_SIZE,DCT_SIZE,1,PREC_FOR_CODE(PREC_SP));
	if( mat_dp == NULL ) return;

	pi = 4 * atan(1.0);

	v=sqrt(1.0/DCT_SIZE);

	ptr = (float *)OBJ_DATA_PTR(mat_dp);

	for(j=0;j<DCT_SIZE;j++)
		/* dct_mat[0][j] = v; */
		ptr[j] = (float) v;

	v=sqrt(2.0/DCT_SIZE);

	for(i=1;i<DCT_SIZE;i++)
		for(j=0;j<DCT_SIZE;j++)
			/* dct_mat[i][j] = */
			ptr[ i * DCT_SIZE + j ] = (float)
			(v * cos( pi*(2*j+1)*i/(2.0*DCT_SIZE) ));

	mat_inited++;
}

static void mat8_1D(QSP_ARG_DECL  dct_type *f)
{
	dct_type tmpvec[DCT_SIZE];
	int i,j;
	float *ptr;

	if( ! mat_inited ) init_mat(SINGLE_QSP_ARG);

	/* multiply by the matrix */

	ptr = (float *)OBJ_DATA_PTR(mat_dp);

	for(i=0;i<DCT_SIZE;i++){
		tmpvec[i] = 0.0;
		for(j=0;j<DCT_SIZE;j++){
			tmpvec[i] += f[j] * /* dct_mat[i][j] */
					ptr[i*DCT_SIZE+j] ;
		}
	}
	for(i=0;i<DCT_SIZE;i++)
		f[i] = tmpvec[i];	/* do it in-place */
}

/* inverse transform */
static void imat8_1D(QSP_ARG_DECL  dct_type *f)
{
	dct_type tmpvec[DCT_SIZE];
	int i,j;
	float *ptr;

	if( ! mat_inited ) init_mat(SINGLE_QSP_ARG);

	/* multiply by the matrix */


	ptr = (float *)OBJ_DATA_PTR(mat_dp);

	for(i=0;i<DCT_SIZE;i++){
		tmpvec[i] = 0.0;
		for(j=0;j<DCT_SIZE;j++){
			/* note reversal of i and j */
			tmpvec[i] += f[j] * /* dct_mat[j][i] */
				ptr[j*DCT_SIZE+i] ;
/* printf("f[%d] %g  mat %g   sum %g\n", i,f[i], ptr[j*DCT_SIZE+i], tmpvec[i] ); */
		}
	}
	for(i=0;i<DCT_SIZE;i++)
		f[i] = tmpvec[i];	/* do it in-place */
}

