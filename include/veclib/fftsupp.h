#include "shape_bits.h"

typedef enum {
	ELLIOT_AND_RAO,
	TEXAS_INSTRUMENTS
} Real_FFT_Algorithm_Code;

extern Real_FFT_Algorithm_Code real_fft_algorithm;

typedef struct fft_args {
	void *		fft_src_addr;
	void *		fft_dst_addr;
	incr_t		fft_src_inc;
	incr_t		fft_dst_inc;
	dimension_t	fft_len;
	int		fft_isi;	// inverse flag
	struct platform_device *	fft_pdp;
#ifdef HAVE_OPENCL
	dimension_t	fft_src_offset;
	dimension_t	fft_dst_offset;
#endif // HAVE_OPENCL
} FFT_Args;


/* FFT args */

#define FFT_LEN(fap)			(fap)->fft_len
#define FFT_ISI(fap)			(fap)->fft_isi
#define FFT_SRC(fap)			(fap)->fft_src_addr
#define FFT_DST(fap)			(fap)->fft_dst_addr
#define FFT_DINC(fap)			(fap)->fft_dst_inc
#define FFT_SINC(fap)			(fap)->fft_src_inc
#define FFT_PFDEV(fap)			(fap)->fft_pdp

#define SET_FFT_LEN(fap,v)		(fap)->fft_len = v
#define SET_FFT_ISI(fap,v)		(fap)->fft_isi = v
#define SET_FFT_SRC(fap,v)		(fap)->fft_src_addr = v
#define SET_FFT_DST(fap,v)		(fap)->fft_dst_addr = v
#define SET_FFT_DINC(fap,v)		(fap)->fft_dst_inc = v
#define SET_FFT_SINC(fap,v)		(fap)->fft_src_inc = v
#define SET_FFT_PFDEV(fap,v)		(fap)->fft_pdp = v

#ifdef HAVE_OPENCL
#define FFT_SRC_OFFSET(fap)		(fap)->fft_src_offset
#define FFT_DST_OFFSET(fap)		(fap)->fft_dst_offset
#define SET_FFT_SRC_OFFSET(fap,v)	(fap)->fft_src_offset = v
#define SET_FFT_DST_OFFSET(fap,v)	(fap)->fft_dst_offset = v
#endif // HAVE_OPENCL

#define FFT2D_REAL_XFORM_ROWS	1
#define FFT2D_REAL_XFORM_COLS	2


extern dimension_t bitrev_size;
extern dimension_t *bitrev_data;

extern void bitrev_init(dimension_t len);
/*
extern int fft_row_size_ok(QSP_ARG_DECL  Data_Obj *dp);
extern int fft_size_ok(QSP_ARG_DECL  Data_Obj *dp);
extern int fft_col_size_ok(QSP_ARG_DECL  Data_Obj *dp);
*/
extern  int row_fft_ok(QSP_ARG_DECL  Data_Obj *dp, const char *funcname );
extern  int cpx_fft_ok(QSP_ARG_DECL  Data_Obj *dp, const char *funcname );
extern int real_row_fft_ok(QSP_ARG_DECL  Data_Obj *real_dp,Data_Obj *cpx_dp,const char *funcname);
extern int real_fft_type(QSP_ARG_DECL  Data_Obj *real_dp,Data_Obj *cpx_dp,const char *funcname);

extern void show_fft_args(FFT_Args *fap);

