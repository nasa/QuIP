#include "quip_config.h"

char VersionId_vec_util_yuv2rgb[] = QUIP_VERSION_STRING;

/* Derived from libng, xawtv source code...
 */

//#include "config.h"
//
//#include <stdio.h>
//#include <stdlib.h>
//#include <unistd.h>
//#include <pthread.h>
//#include <sys/time.h>
//#include <sys/types.h>

//#include "grab-ng.h"
#include "data_obj.h"
#include "vec_util.h"
//#include "my_grabber.h"

/* ------------------------------------------------------------------- */

#define CLIP         320

# define RED_NULL    128
# define BLUE_NULL   128
# define LUN_MUL     256
# define RED_MUL     512
# define BLUE_MUL    512

#define GREEN1_MUL  (-RED_MUL/2)
#define GREEN2_MUL  (-BLUE_MUL/6)
#define RED_ADD     (-RED_NULL  * RED_MUL)
#define BLUE_ADD    (-BLUE_NULL * BLUE_MUL)
#define GREEN1_ADD  (-RED_ADD/2)
#define GREEN2_ADD  (-BLUE_ADD/6)

/* lookup tables */
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

static void init_tables(void);

/* ------------------------------------------------------------------- */
/* packed pixel yuv to gray / rgb                                      */

void _yuv422_to_rgb24(unsigned char* dest, unsigned char* s, int p)
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

/* This function assumes that src_dp points to an image w/ YUYV samples... */

void yuv422_to_rgb24(Data_Obj *dst_dp, Data_Obj * src_dp )
{
    unsigned char *y_p, *u_p, *v_p, *d_p;
    unsigned char *dst_p;
    unsigned int i,j;
    int gray;

//advise("yuv420p_to_rgb24");
    INSURE_TABLES

    dst_p = (unsigned char *)dst_dp->dt_data;
    y_p  = (unsigned char *)src_dp->dt_data;
    u_p  = y_p + 1;
    v_p  = y_p + 3;

//sprintf(error_string,"y_p = 0x%lx, u_p = 0x%lx, v_p = 0x%lx",y_p,u_p,v_p);
//advise(error_string);
    for (i = 0; i < dst_dp->dt_rows; i++) {
	d_p = dst_p;
	for (j = 0; j < dst_dp->dt_cols; j+= 2) {
	    gray   = GRAY(*y_p);
	    // for display sometimes we want BGR!?
	    //*(d_p++) = RED(gray,*v_p);
	    //*(d_p++) = BLUE(gray,*u_p);
	    *(d_p++) = BLUE(gray,*u_p);
	    *(d_p++) = GREEN(gray,*v_p,*u_p);
	    *(d_p++) = RED(gray,*v_p);
	    y_p+=2;
	    gray   = GRAY(*y_p);
	    //*(d_p++) = RED(gray,*v_p);
	    //*(d_p++) = BLUE(gray,*u_p);
	    *(d_p++) = BLUE(gray,*u_p);
	    *(d_p++) = GREEN(gray,*v_p,*u_p);
	    *(d_p++) = RED(gray,*v_p);

	    y_p+=2;
	    u_p+=4;
	    v_p+=4;
	}
	/* BUG assumes source is contiguous... */
	dst_p += dst_dp->dt_rowinc;
    }
}

/* This function assumes that src_dp points to an image w/ YUYV samples... */

void yuv422_to_gray(Data_Obj *dst_dp, Data_Obj * src_dp )
{
    unsigned char *y_p, *d_p;
    unsigned char *dst_p;
    unsigned int i,j;

//advise("yuv420p_to_rgb24");
    INSURE_TABLES

    dst_p = (unsigned char *)dst_dp->dt_data;
    y_p  = (unsigned char *)src_dp->dt_data;

    for (i = 0; i < dst_dp->dt_rows; i++) {
	d_p = dst_p;
	for (j = 0; j < dst_dp->dt_cols; j+= 2) {
	    *(d_p++)   = GRAY(*y_p);
	    y_p+=2;
	    *(d_p++)   = GRAY(*y_p);
	    y_p+=2;
	}
	/* BUG assumes source is contiguous... */
	dst_p += dst_dp->dt_rowinc;
    }
}


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

    dst_p = (unsigned char *)dst_dp->dt_data;
    y_p  = src;
    u0_p  = y_p + dst_dp->dt_cols * dst_dp->dt_rows;
    v0_p  = u0_p + dst_dp->dt_cols * dst_dp->dt_rows / 2;	/* was 4 instead of 2? */

//sprintf(error_string,"y_p = 0x%lx, u_p = 0x%lx, v_p = 0x%lx",y_p,u_p,v_p);
//advise(error_string);
    for (i = 0; i < dst_dp->dt_rows; i++) {
	d_p = dst_p;
	v_p=v0_p; u_p=u0_p;
	for (j = 0; j < dst_dp->dt_cols; j+= 2) {
//sprintf(error_string,"i = %d, j = %d,    y = %d     u = %d    v = %d",i,j,*y_p,*u_p,*v_p);
//prt_msg(error_string);
	    gray   = GRAY(*y_p);
	    // for display sometimes we want BGR!?
	    //*(d_p++) = RED(gray,*v_p);
	    //*(d_p++) = BLUE(gray,*u_p);
	    *(d_p++) = BLUE(gray,*u_p);
	    *(d_p++) = GREEN(gray,*v_p,*u_p);
	    *(d_p++) = RED(gray,*v_p);
	    y_p++;
	    gray   = GRAY(*y_p);
	    //*(d_p++) = RED(gray,*v_p);
	    //*(d_p++) = BLUE(gray,*u_p);
	    *(d_p++) = BLUE(gray,*u_p);
	    *(d_p++) = GREEN(gray,*v_p,*u_p);
	    *(d_p++) = RED(gray,*v_p);
	    y_p++; u_p++; v_p++;
	}
	dst_p += dst_dp->dt_rowinc;
//	if( i&1 ){
		u0_p += dst_dp->dt_cols/2;
		v0_p += dst_dp->dt_cols/2;
//	}
    }
} /* end yuv420p_to_rgb24 */


/* ------------------------------------------------------------------- */

static void init_tables(void)
{
    int i;
    
    /* init Lookup tables */
    for (i = 0; i < 256; i++) {
        ng_yuv_gray[i] = i * LUN_MUL >> 8;			/* = i ??? */
        ng_yuv_red[i]  = (RED_ADD    + i * RED_MUL)    >> 8;	/* RED_MUL = 512
								 * RED_ADD = (-RED_NULL  * RED_MUL)
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
