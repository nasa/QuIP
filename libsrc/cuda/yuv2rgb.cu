#include "quip_config.h"

#ifdef HAVE_CUDA

char VersionId_cuda_yuv2rgb[] = QUIP_VERSION_STRING;

/* Derived from libng, xawtv source code...  */

#include "my_cuda.h"
#include "data_obj.h"
#include "host_call_utils.h"

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
/* These should be in constant memory? */
/* Or shared? */
static __shared__ unsigned int  ng_yuv_gray[256];
static __shared__ unsigned int  ng_yuv_red[256];
static __shared__ unsigned int  ng_yuv_blue[256];
static __shared__ unsigned int  ng_yuv_g1[256];
static __shared__ unsigned int  ng_yuv_g2[256];
static __shared__ unsigned int  ng_clip[256 + 2 * CLIP];

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
/* assumes interleaved yuyv */

__global__ void decode_two_pixels_yuv2rgb(unsigned char *dst_p,unsigned char *yuyv_p)
{
	int index,gray;
	unsigned char *d_p, *y_p, *v_p, *u_p;

	index = blockIdx.x * blockDim.x + threadIdx.x;

	d_p = dst_p  + 6*index;		// two RGB pixels
	y_p = yuyv_p + 4*index;
	u_p = y_p + 1;
	v_p = y_p + 3;

	gray   = GRAY(*y_p);
	d_p[0] = BLUE(gray,*u_p);
	d_p[1] = GREEN(gray,*v_p,*u_p);
	d_p[2] = RED(gray,*v_p);

	y_p += 2;

	gray   = GRAY(*y_p);
	d_p[3] = BLUE(gray,*u_p);
	d_p[4] = GREEN(gray,*v_p,*u_p);
	d_p[5] = RED(gray,*v_p);
}

#define SETUP_THREADS(dp)						\
									\
	n_thr_need = dp->dt_rows * dp->dt_cols/2;			\
	n_blocks = n_thr_need / max_threads_per_block;			\
	n_extra = n_thr_need % n_blocks;				\
	if( n_extra > 0 ) NERROR1("OOPS:  Need to handle case of extra threads");



/* This function assumes that src_dp points to an image w/ YUYV samples... */

void cuda_yuv422_to_rgb24(Data_Obj *dst_dp, Data_Obj * src_dp )
{
	unsigned char *y_p, *dst_p;
	int max_threads_per_block, n_blocks, n_thr_need, n_extra;

	GET_MAX_THREADS(dst_dp)
	INSURE_TABLES
	SETUP_THREADS(dst_dp)

	dst_p = (unsigned char *)dst_dp->dt_data;
	y_p  = (unsigned char *)src_dp->dt_data;

	decode_two_pixels_yuv2rgb<<< n_blocks , max_threads_per_block >>>
		(dst_p,y_p);
}

#ifdef FOOBAR

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


/* This function seems to assume that y, u and v are stored sequentially,
 * not interleaved
 */

void yuv420p_to_rgb24(Data_Obj *dst_dp, unsigned char *src )
{
	unsigned char *y_p;
	unsigned char *u0_p, *v0_p;
	unsigned char *dst_p;
	unsigned int i,j;
	int gray;

	int n_thr_need, n_blocks, n_extra;

	INSURE_TABLES
	SETUP_THREADS(dst_dp)

	dst_p = (unsigned char *)dst_dp->dt_data;

	/* It looks like the components are not interleaved? */
	y_p  = src;
	u0_p  = y_p + dst_dp->dt_cols * dst_dp->dt_rows;
	v0_p  = u0_p + dst_dp->dt_cols * dst_dp->dt_rows / 2;
							/* was 4 instead of 2? */

	decode_two_pixels_yuv2rgb<<< n_blocks , max_threads_per_block >>>
		(dst_p,y_p,u0_p,v0_p);
}
#endif /* FOOBAR */


/* ------------------------------------------------------------------- */

__global__ void init_tbl_entries(void)
{
	int i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

        ng_yuv_gray[i] = i * LUN_MUL >> 8;
        ng_yuv_red[i]  = (RED_ADD    + i * RED_MUL)    >> 8;
        ng_yuv_blue[i] = (BLUE_ADD   + i * BLUE_MUL)   >> 8;
        ng_yuv_g1[i]   = (GREEN1_ADD + i * GREEN1_MUL) >> 8;
        ng_yuv_g2[i]   = (GREEN2_ADD + i * GREEN2_MUL) >> 8;
        ng_clip[i+CLIP] = i ;
}

__global__ void const_tbl_entries(unsigned int *ptr, unsigned int value)
{
	int i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

        ptr[i] = value;
}

static void init_tables(void)
{
	// BUG make sure we can have this many threads in a block.

	init_tbl_entries<<< 1 , 256  >>>();
	const_tbl_entries<<< 1 , CLIP  >>>( &ng_clip[0], 0 );
	const_tbl_entries<<< 1 , CLIP  >>>( &ng_clip[CLIP+256], 255 );
	tbls_inited=1;
}

#endif /* HAVE_CUDA */

