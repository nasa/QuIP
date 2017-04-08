include(`../../include/veclib/cu2_port.m4')

my_include(`../../include/veclib/gpu_call_utils.m4')
my_include(`../cu2/cu2_host_call_defs.m4')

define(`CLIP',`320')

define(`RED_NULL',`128')
define(`BLUE_NULL',`128')
define(`LUN_MUL',`256')
define(`RED_MUL',`512')
define(`BLUE_MUL',`512')

define(`GREEN1_MUL',`(-RED_MUL/2)')
define(`GREEN2_MUL',`(-BLUE_MUL/6)')
define(`RED_ADD',`(-RED_NULL  * RED_MUL)')
define(`BLUE_ADD',`(-BLUE_NULL * BLUE_MUL)')
define(`GREEN1_ADD',`(-RED_ADD/2)')
define(`GREEN2_ADD',`(-BLUE_ADD/6)')

dnl	/* lookup tables */
dnl	/* These should be in constant memory? */
dnl	/* Or shared? */
static __shared__ unsigned int  ng_yuv_gray[256];
static __shared__ unsigned int  ng_yuv_red[256];
static __shared__ unsigned int  ng_yuv_blue[256];
static __shared__ unsigned int  ng_yuv_g1[256];
static __shared__ unsigned int  ng_yuv_g2[256];
static __shared__ unsigned int  ng_clip[256 + 2 * CLIP];

dnl	GRAY(val)
define(`GRAY',`ng_yuv_gray[$1]')
dnl	RED(gray,red)
define(`RED',`ng_clip[ CLIP + $1 + ng_yuv_red[$2] ]')
dnl	GREEN(gray,red,blue)
define(`GREEN',`ng_clip[ CLIP + $1 + ng_yuv_g1[$2] + ng_yuv_g2[$2] ]')
dnl	BLUE(gray,blue)
define(`BLUE',`ng_clip[ CLIP + $1 + ng_yuv_blue[$2] ]')


static int tbls_inited=0;
define(`INSURE_TABLES',`if( !tbls_inited ) init_tables();')

static void init_tables(void);

dnl	/* ------------------------------------------------------------------- */
dnl	/* packed pixel yuv to gray / rgb                                      */
dnl	/* assumes interleaved yuyv */

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

dnl	SETUP_THREADS(dp)
define(`SETUP_THREADS',`

	n_thr_need = OBJ_ROWS($1) * OBJ_COLS($1)/2;
	n_blocks = n_thr_need / max_threads_per_block;
	n_extra = n_thr_need % n_blocks;
	if( n_extra > 0 ) NERROR1("OOPS:  Need to handle case of extra threads");
')



/* This function assumes that src_dp points to an image w/ YUYV samples... */

void cuda_yuv422_to_rgb24(Data_Obj *dst_dp, Data_Obj * src_dp )
{
	unsigned char *y_p, *dst_p;
	//int max_threads_per_block, n_blocks, n_thr_need, n_extra;
	Vector_Args va1, *vap=(&va1);

	BLOCK_VARS_DECLS
dnl	//GET_MAX_THREADS(dst_dp)
	INSURE_TABLES
dnl	//SETUP_THREADS(dst_dp)
	SETUP_BLOCKS_XYZ(OBJ_PFDEV(dst_dp))

	dst_p = (unsigned char *)OBJ_DATA_PTR(dst_dp);
	y_p  = (unsigned char *)OBJ_DATA_PTR(src_dp);

	decode_two_pixels_yuv2rgb<<< /*n_blocks , max_threads_per_block*/ NN_GPU >>>
		(dst_p,y_p);
}

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

__global__ void const_tbl_entries(unsigned int index, unsigned int value)
{
	int i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

        ng_clip[index+i] = value;
}

static void init_tables(void)
{
	// BUG make sure we can have this many threads in a block.

	init_tbl_entries<<< 1 , 256  >>>();
	// can't take the address of a shared variable, cuda 5 warning???
	const_tbl_entries<<< 1 , CLIP  >>>( 0, 0 );
	const_tbl_entries<<< 1 , CLIP  >>>( CLIP+256, 255 );
	tbls_inited=1;
}


