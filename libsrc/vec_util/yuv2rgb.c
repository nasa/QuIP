#include "quip_config.h"
#include "quip_prot.h"

/* Derived from libng, xawtv source code...
 */

#include "data_obj.h"
#include "vec_util.h"

/* ------------------------------------------------------------------- */

#define CLIP		320

#define RED_NULL	128
#define BLUE_NULL	128
#define LUN_MUL		256
#define RED_MUL		512
#define BLUE_MUL	512

#define GREEN1_MUL	(-RED_MUL/2)
#define GREEN2_MUL	(-BLUE_MUL/6)
#define RED_ADD		(-RED_NULL * RED_MUL)
#define BLUE_ADD	(-BLUE_NULL * BLUE_MUL)
#define GREEN1_ADD	(-RED_ADD/2)
#define GREEN2_ADD	(-BLUE_ADD/6)

/* lookup tables */
/* If these are fixed, then this is thread-safe... */
static unsigned int  ng_yuv_gray[256];
static unsigned int  ng_yuv_red[256];
static unsigned int  ng_yuv_blue[256];
static unsigned int  ng_yuv_g1[256];
static unsigned int  ng_yuv_g2[256];
static unsigned int  ng_clip[256 + 2 * CLIP];

#define GRAY(val)		ng_yuv_gray[val]
#define RED(gray,red)		ng_clip[ CLIP + gray + ng_yuv_red[red] ]
#define GREEN(gray,red,blue)	ng_clip[ CLIP + gray + ng_yuv_g1[red] +	\
							ng_yuv_g2[blue] ]
#define BLUE(gray,blue)		ng_clip[ CLIP + gray + ng_yuv_blue[blue] ]


static int tbls_inited=0;
#define INSURE_TABLES		if( !tbls_inited ) init_tables();

static void init_tables(void)
{
	int i;

	/* init Lookup tables */
	for (i = 0; i < 256; i++) {
		ng_yuv_gray[i] =           (i * LUN_MUL) >> 8;	/* = i ??? */
		ng_yuv_red[i]  = (RED_ADD + i * RED_MUL) >> 8;	/* RED_MUL = 512
								 * RED_ADD =
								 * (-RED_NULL  * RED_MUL)
								 * RED_NULL = 128
								 */
	//		( RED_MUL*( i - RED_NULL ) )>>8
	//		( i - RED_NULL ) * 2			?? +- 256?
		ng_yuv_blue[i] = (BLUE_ADD   + i * BLUE_MUL)   >> 8;
		ng_yuv_g1[i]   = (GREEN1_ADD + i * GREEN1_MUL) >> 8;
		ng_yuv_g2[i]   = (GREEN2_ADD + i * GREEN2_MUL) >> 8;
	}
	for (i = 0; i < CLIP; i++)
		ng_clip[i] = 0;
	for (; i < CLIP + 256; i++)
		ng_clip[i] = i - CLIP;
	for (; i < 2 * CLIP + 256; i++)
		ng_clip[i] = 255;

	/* register stuff */
	//ng_conv_register(NG_PLUGIN_MAGIC,"built-in",conv_list,nconv);

	tbls_inited=1;
}

#ifdef NOT_USED

/* ------------------------------------------------------------------- */
/* packed pixel yuv to gray / rgb									*/

static void _yuv422_to_rgb24(unsigned char* dest, unsigned char* s, int p)
{
	unsigned char* d = dest;
	int gray;

	INSURE_TABLES

	while (p) {
		gray = GRAY(s[0]);
		d[0] = RED(gray,s[3]);
		d[1] = GREEN(gray,s[3],s[1]);
		d[2] = BLUE(gray,s[1]);
		gray = GRAY(s[2]);
		d[3] = RED(gray,s[3]);
		d[4] = GREEN(gray,s[3],s[1]);
		d[5] = BLUE(gray,s[1]);
		d += 6;
		s += 4;
		p -= 2;
	}
}
#endif /* NOT_USED */

#define INSIST_BYTE(dp,whence)						\
	if( OBJ_PREC(dp) != PREC_BY && OBJ_PREC(dp) != PREC_UBY ){	\
		sprintf(ERROR_STRING,					\
	"%s:  object %s (%s) must have %s or %s precision!?",		\
			#whence,OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)), \
			PREC_BY_NAME, PREC_UBY_NAME);			\
		WARN(ERROR_STRING);					\
		return;							\
	}

#define INSIST_NCOMPS(dp,n,whence)					\
	if( OBJ_COMPS(dp) != n ){					\
		sprintf(ERROR_STRING,					\
	"%s:  object %s (%ld) must have %d components!?",		\
			#whence,OBJ_NAME(dp),(long)OBJ_COMPS(dp),n);	\
		WARN(ERROR_STRING);					\
		return;							\
	}

#define INSIST_SAME_AREA(dp1,dp2,whence)					\
	if( OBJ_ROWS(dp1) != OBJ_ROWS(dp2) || OBJ_COLS(dp1) != OBJ_COLS(dp2) ){	\
		sprintf(ERROR_STRING,					\
	"%s:  object %s (%ld x %ld) must be same size as object %s (%ld x %ld).", \
			#whence,					\
			OBJ_NAME(dp1),(long)OBJ_ROWS(dp1),(long)OBJ_COLS(dp1),	\
			OBJ_NAME(dp2),(long)OBJ_ROWS(dp2),(long)OBJ_COLS(dp2) ); \
		WARN(ERROR_STRING);					\
		return;							\
	}

#define CHECK_EXTRA_DIMS(dp,whence)					\
	CHECK_EXTRA_DIM(dp,4,whence)	/* n sequences */		\
	CHECK_EXTRA_DIM(dp,3,whence)	/* n frames */

#define CHECK_EXTRA_DIM(dp,idx,whence)					\
	if( OBJ_TYPE_DIM(dp,idx) != 1 ){				\
		sprintf(ERROR_STRING,					\
	"%s:  ignoring extra %ss, object %s (%ld)",#whence,		\
			DIMENSION_NAME(idx),OBJ_NAME(dp),		\
			(long)OBJ_TYPE_DIM(dp,idx) );			\
		WARN(ERROR_STRING);					\
	}

/* This function assumes that src_dp points to an image w/ YUYV samples... */

void yuv422_to_rgb24(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj * src_dp )
{
	unsigned char *y_p, *u_p, *v_p, *d_p;
	unsigned char *dst_p;
	unsigned int i,j;
	int gray;

	INSIST_RAM_OBJ(dst_dp,yuv422_to_rgb24)
	INSIST_RAM_OBJ(src_dp,yuv422_to_rgb24)

	// Make sure all objects are correct type and compatible sizes.
	INSIST_BYTE(dst_dp,yuv422_to_rgb24)
	INSIST_BYTE(src_dp,yuv422_to_rgb24)
	INSIST_NCOMPS(src_dp,2,yuv422_to_rgb24)
	INSIST_NCOMPS(dst_dp,3,yuv422_to_rgb24)
	INSIST_SAME_AREA(dst_dp,src_dp,yuv422_to_rgb24)

	CHECK_EXTRA_DIMS(dst_dp,yuv422_to_rgb24)
	CHECK_EXTRA_DIMS(src_dp,yuv422_to_rgb24)

//advise("yuv420p_to_rgb24");
	INSURE_TABLES

	dst_p = (unsigned char *)OBJ_DATA_PTR(dst_dp);
	y_p  = (unsigned char *)OBJ_DATA_PTR(src_dp);
	u_p  = y_p + 1;
	v_p  = y_p + 3;

//sprintf(error_string,"y_p = 0x%lx, u_p = 0x%lx, v_p = 0x%lx",y_p,u_p,v_p);
//advise(error_string);
	for (i = 0; i < OBJ_ROWS(dst_dp); i++) {
		d_p = dst_p;
		for (j = 0; j < OBJ_COLS(dst_dp); j+= 2) {
			gray = GRAY(*y_p);
			// for display sometimes we want BGR!?
			//*(d_p++) = RED(gray,*v_p);
			//*(d_p++) = BLUE(gray,*u_p);
			*(d_p++) = (u_char) BLUE(gray,*u_p);
			*(d_p++) = (u_char) GREEN(gray,*v_p,*u_p);
			*(d_p++) = (u_char) RED(gray,*v_p);
			y_p+=2;
			gray = GRAY(*y_p);
			//*(d_p++) = RED(gray,*v_p);
			//*(d_p++) = BLUE(gray,*u_p);
			*(d_p++) = (u_char) BLUE(gray,*u_p);
			*(d_p++) = (u_char) GREEN(gray,*v_p,*u_p);
			*(d_p++) = (u_char) RED(gray,*v_p);

			y_p+=2;
			u_p+=4;
			v_p+=4;
		}
		/* BUG assumes source is contiguous... */
		dst_p += OBJ_ROW_INC(dst_dp);
	}
}

/* This function assumes that src_dp points to an image w/ YUYV samples... */

void yuv422_to_gray(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj * src_dp )
{
	unsigned char *y_p, *d_p;
	unsigned char *dst_p;
	unsigned int i,j;

	INSIST_RAM_OBJ(dst_dp,yuv422_to_gray)
	INSIST_RAM_OBJ(src_dp,yuv422_to_gray)

	// Make sure all objects are correct type and compatible sizes.
	INSIST_BYTE(dst_dp,yuv422_to_gray)
	INSIST_BYTE(src_dp,yuv422_to_gray)
	INSIST_NCOMPS(src_dp,2,yuv422_to_gray)
	INSIST_NCOMPS(dst_dp,1,yuv422_to_gray)
	INSIST_SAME_AREA(dst_dp,src_dp,yuv422_to_gray)

	CHECK_EXTRA_DIMS(dst_dp,yuv422_to_gray)
	CHECK_EXTRA_DIMS(src_dp,yuv422_to_gray)

//advise("yuv420p_to_rgb24");
	INSURE_TABLES

	dst_p = (unsigned char *)OBJ_DATA_PTR(dst_dp);
	y_p  = (unsigned char *)OBJ_DATA_PTR(src_dp);

	for (i = 0; i < OBJ_ROWS(dst_dp); i++) {
		d_p = dst_p;
		for (j = 0; j < OBJ_COLS(dst_dp); j+= 2) {
			*(d_p++) = (u_char) GRAY(*y_p);
			y_p+=2;
			*(d_p++) = (u_char) GRAY(*y_p);
			y_p+=2;
		}
		/* BUG assumes source is contiguous... */
		dst_p += OBJ_ROW_INC(dst_dp);
	}
}

#ifdef NOT_USED

/* This function seems to assume that y, u and v are stored sequentially, not interleaved */

void yuv420p_to_rgb24(Data_Obj *dst_dp, unsigned char *src )
{
	unsigned char *y_p, *u_p, *v_p, *d_p;
	unsigned char *u0_p, *v0_p;
	unsigned char *dst_p;
	unsigned int i,j;
	int gray;

//advise("yuv420p_to_rgb24");
	INSURE_TABLES

	dst_p = (unsigned char *)OBJ_DATA_PTR(dst_dp);
	y_p  = src;
	u0_p  = y_p + OBJ_COLS(dst_dp) * OBJ_ROWS(dst_dp);
	v0_p  = u0_p + OBJ_COLS(dst_dp) * OBJ_ROWS(dst_dp) / 2;	/* was 4 instead of 2? */

//sprintf(error_string,"y_p = 0x%lx, u_p = 0x%lx, v_p = 0x%lx",y_p,u_p,v_p);
//advise(error_string);
	for (i = 0; i < OBJ_ROWS(dst_dp); i++) {
	d_p = dst_p;
	v_p=v0_p; u_p=u0_p;
	for (j = 0; j < OBJ_COLS(dst_dp); j+= 2) {
//sprintf(error_string,"i = %d, j = %d,	y = %d	u = %d	v = %d",i,j,*y_p,*u_p,*v_p);
//prt_msg(error_string);
		gray = GRAY(*y_p);
		// for display sometimes we want BGR!?
		//*(d_p++) = RED(gray,*v_p);
		//*(d_p++) = BLUE(gray,*u_p);
		*(d_p++) = BLUE(gray,*u_p);
		*(d_p++) = GREEN(gray,*v_p,*u_p);
		*(d_p++) = RED(gray,*v_p);
		y_p++;
		gray = GRAY(*y_p);
		//*(d_p++) = RED(gray,*v_p);
		//*(d_p++) = BLUE(gray,*u_p);
		*(d_p++) = BLUE(gray,*u_p);
		*(d_p++) = GREEN(gray,*v_p,*u_p);
		*(d_p++) = RED(gray,*v_p);
		y_p++; u_p++; v_p++;
	}
	dst_p += OBJ_ROW_INC(dst_dp);
//	if( i&1 ){
		u0_p += OBJ_COLS(dst_dp)/2;
		v0_p += OBJ_COLS(dst_dp)/2;
//	}
	}
} /* end yuv420p_to_rgb24 */
#endif /* NOT_USED */


/* ------------------------------------------------------------------- */
