/* glue to link opencv subroutines with vt... */

#include "quip_config.h"

#ifdef HAVE_OPENCV

#include "quip_prot.h"
#include "data_obj.h"
#include "opencv_glue.h"

/* OpenCV_Image */
ITEM_INTERFACE_DECLARATIONS(OpenCV_Image,ocvi,0)
/* OpenCV_MemStorage */
ITEM_INTERFACE_DECLARATIONS(OpenCV_MemStorage,ocv_mem,0)
/* OpenCV_Scanner */
ITEM_INTERFACE_DECLARATIONS(OpenCV_Scanner,ocv_scanner,0)
/* OpenCV_Seq */
ITEM_INTERFACE_DECLARATIONS(OpenCV_Seq,ocv_seq,0)
/* OpenCV_ classifier cascade */
ITEM_INTERFACE_DECLARATIONS(OpenCV_Cascade,ocv_ccasc,0)

static OpenCV_Image *make_new_ocvi(QSP_ARG_DECL  const char * obj_name);
static OpenCV_MemStorage *make_new_ocv_mem(QSP_ARG_DECL  const char * obj_name);
static OpenCV_Scanner *make_new_ocv_scanner(QSP_ARG_DECL  const char * obj_name);
static OpenCV_Seq *make_new_ocv_seq(QSP_ARG_DECL  const char * obj_name);
static Data_Obj * creat_dp_for_ocvi(QSP_ARG_DECL OpenCV_Image *ocvi_p );

static OpenCV_Image *make_new_ocvi(QSP_ARG_DECL  const char * obj_name)
{
	OpenCV_Image *ocvi_p;

	ocvi_p = ocvi_of(QSP_ARG  obj_name);
	if( ocvi_p != NO_OPENCV_IMAGE ){
		sprintf(ERROR_STRING,"OpenCV image %s already exists!?",obj_name);
		WARN(ERROR_STRING);
		return(NO_OPENCV_IMAGE);
	}

	ocvi_p = new_ocvi(QSP_ARG  obj_name);
	if( ocvi_p == NO_OPENCV_IMAGE ){
		sprintf(ERROR_STRING,"Error creating OpenCV image %s",obj_name);
		WARN(ERROR_STRING);
		return(NO_OPENCV_IMAGE);
	}
	ocvi_p->ocv_dp = NO_OBJ;

	return(ocvi_p);
}

static OpenCV_MemStorage *make_new_ocv_mem(QSP_ARG_DECL  const char * obj_name)
{
	OpenCV_MemStorage *ocv_mem_p;

	ocv_mem_p = ocv_mem_of(QSP_ARG  obj_name);
	if( ocv_mem_p != NO_OPENCV_MEM ){
		sprintf(ERROR_STRING,"OpenCV mem storage %s already exists!?",obj_name);
		WARN(ERROR_STRING);
		return(NO_OPENCV_MEM);
	}

	ocv_mem_p = new_ocv_mem(QSP_ARG  obj_name);
	if( ocv_mem_p == NO_OPENCV_MEM ){
		sprintf(ERROR_STRING,"Error creating OpenCV mem storage %s",obj_name);
		WARN(ERROR_STRING);
		return(NO_OPENCV_MEM);
	}
	ocv_mem_p->ocv_dp = NO_OBJ;

	return(ocv_mem_p);
}

static OpenCV_Scanner *make_new_ocv_scanner(QSP_ARG_DECL  const char * obj_name)
{
	OpenCV_Scanner *ocv_scanner_p;

	ocv_scanner_p = ocv_scanner_of(QSP_ARG  obj_name);
	if( ocv_scanner_p != NO_OPENCV_SCANNER ){
		sprintf(ERROR_STRING,"OpenCV scanner %s already exists!?",obj_name);
		WARN(ERROR_STRING);
		return(NO_OPENCV_SCANNER);
	}

	ocv_scanner_p = new_ocv_scanner(QSP_ARG  obj_name);
	if( ocv_scanner_p == NO_OPENCV_SCANNER ){
		sprintf(ERROR_STRING,"Error creating OpenCV scanner %s",obj_name);
		WARN(ERROR_STRING);
		return(NO_OPENCV_SCANNER);
	}
	ocv_scanner_p->ocv_dp = NO_OBJ;
	ocv_scanner_p->ocv_mem = cvCreateMemStorage(0);

	return(ocv_scanner_p);
}

static OpenCV_Seq *make_new_ocv_seq(QSP_ARG_DECL  const char * obj_name)
{
	OpenCV_Seq *ocv_seq_p;

	ocv_seq_p = ocv_seq_of(QSP_ARG  obj_name);
	if( ocv_seq_p != NO_OPENCV_SEQ ){
		sprintf(ERROR_STRING,"OpenCV seq %s already exists!?",obj_name);
		WARN(ERROR_STRING);
		return(NO_OPENCV_SEQ);
	}

	ocv_seq_p = new_ocv_seq(QSP_ARG  obj_name);
	if( ocv_seq_p == NO_OPENCV_SEQ ){
		sprintf(ERROR_STRING,"Error creating OpenCV seq %s",obj_name);
		WARN(ERROR_STRING);
		return(NO_OPENCV_SEQ);
	}
	ocv_seq_p->ocv_dp = NO_OBJ;
	ocv_seq_p->ocv_seq = NULL;

	return(ocv_seq_p);
}

//#ifdef FOOBAR
OpenCV_Image * load_ocv_image(QSP_ARG_DECL   const char * obj_name, const char * filename )
{
	OpenCV_Image *ocvi_p;

	ocvi_p = make_new_ocvi(QSP_ARG  obj_name);
	if( ocvi_p == NO_OPENCV_IMAGE ) return(ocvi_p);

	if( (ocvi_p->ocv_image = cvLoadImage( filename, CV_LOAD_IMAGE_COLOR)) == 0 ){
		sprintf(ERROR_STRING,"Error opening file %s!?",filename);
		WARN(ERROR_STRING);
		/* delete new struct here */
		del_ocvi(QSP_ARG  ocvi_p);
		// free name here?
		return(NO_OPENCV_IMAGE);
	}

	return(ocvi_p);
}
//#endif /* FOOBAR */

void save_ocv_image( OpenCV_Image *ocvi_p , const char* filename)
{
	// Save the image to the given filename.
	/* check for success??? */
	/* Why the third arg?  difference versions of lib? */
#if CV_MAJOR_VERSION >= 2
	cvSaveImage(filename, ocvi_p->ocv_image , 0 );
#else
	cvSaveImage(filename, ocvi_p->ocv_image /* , 0 */ );
#endif
}

static Data_Obj * creat_dp_for_ocvi(QSP_ARG_DECL OpenCV_Image *ocvi_p )
{

	IplImage* img;
	Dimension_Set dimset;
	Data_Obj *dp;
	prec_t p;

	img = ocvi_p->ocv_image;

	switch(img->depth) {
		case IPL_DEPTH_8U: p = PREC_UBY; break;
		case IPL_DEPTH_8S: p = PREC_BY; break;
		case IPL_DEPTH_16S: p = PREC_IN; break;
		case IPL_DEPTH_16U: p = PREC_UIN; break;
		case IPL_DEPTH_32S: p = PREC_DI; break;
		/* case IPL_DEPTH_32U: p = PREC_UDI; break; */
		case IPL_DEPTH_32F: p = PREC_SP; break;
		case IPL_DEPTH_64F: p = PREC_DP; break;
		default:
			sprintf(ERROR_STRING,"creat_dp_for_ocvi:  unrecognized depth code %d!?",img->depth);
			WARN(ERROR_STRING);
			return(NO_OBJ);
			break;
	}
	if( img->dataOrder == 0 ){
		dimset.ds_dimension[0] = img->nChannels;
		dimset.ds_dimension[1] = img->width;	// Image width.
		dimset.ds_dimension[2] = img->height;	// Image height.
		dimset.ds_dimension[3] = 1;
	} else {
		dimset.ds_dimension[0] = 1;
		dimset.ds_dimension[1] = img->width;	// Image width.
		dimset.ds_dimension[2] = img->height;	// Image height.
		dimset.ds_dimension[3] = img->nChannels;
		sprintf(ERROR_STRING,"OpenCV image components are not interleaved!? (dataOrder = %d)",img->dataOrder);
		advise(ERROR_STRING);
	}
	dimset.ds_dimension[4] = 1;

	dp = _make_dp(QSP_ARG  ocvi_p->ocv_name,&dimset,PREC_FOR_CODE(p));
	if( dp == NO_OBJ ) return(dp);

	SET_OBJ_DATA_PTR(dp, img->imageData);
	SET_OBJ_UNALIGNED_PTR(dp, img->imageDataOrigin);

        return(dp);
}

OpenCV_Image * create_ocv_image(QSP_ARG_DECL  const char *obj_name,
			long width,long height,int bit_depth_code,int n_channels)
{
	OpenCV_Image *ocvi_p;

	ocvi_p = make_new_ocvi(QSP_ARG  obj_name);
	if( ocvi_p == NO_OPENCV_IMAGE ) return(ocvi_p);

	if( (ocvi_p->ocv_image = cvCreateImage(cvSize(width,height), bit_depth_code, n_channels)) == 0 ){
		sprintf(ERROR_STRING,"create_ocv_image:  error creating ocv image for %s...",obj_name);
		WARN(ERROR_STRING);
		del_ocvi(QSP_ARG  ocvi_p);
		// release name here??
		return(NO_OPENCV_IMAGE);
	}

	/* 
	 * This is where we allocate a data_obj struct, and then fill in
	 * all the fields, and set the data ptr accordingly...
	 * (as done in make_frame_obj in ../newmeteor/mcapt.c)
	 */
	if( (ocvi_p->ocv_dp = creat_dp_for_ocvi(QSP_ARG  ocvi_p)) == NO_OBJ ){
		sprintf(ERROR_STRING,"Error creating QuIP image for OpenCV image %s",ocvi_p->ocv_name);
		WARN(ERROR_STRING);
	}

	return(ocvi_p);
}

OpenCV_MemStorage * create_ocv_mem(QSP_ARG_DECL  const char *obj_name)
{
	OpenCV_MemStorage *ocv_mem_p;
	ocv_mem_p = make_new_ocv_mem(QSP_ARG  obj_name);
	if( ocv_mem_p == NO_OPENCV_MEM ) return(ocv_mem_p);

	if( (ocv_mem_p->ocv_mem = cvCreateMemStorage(0)) == 0 ){
		sprintf(ERROR_STRING,"create_ocv_mem:  error creating ocv mem storage for %s...",obj_name);
		WARN(ERROR_STRING);
		del_ocv_mem(QSP_ARG  ocv_mem_p);
		// release name here???
		return(NO_OPENCV_MEM);
	}
	return(ocv_mem_p);
}

OpenCV_Scanner * create_ocv_scanner(QSP_ARG_DECL  const char *obj_name)
{
	OpenCV_Scanner *ocv_scanner_p;
	ocv_scanner_p = make_new_ocv_scanner(QSP_ARG  obj_name);
	if( ocv_scanner_p == NO_OPENCV_SCANNER ) return(ocv_scanner_p);
	return(ocv_scanner_p);
}

OpenCV_Seq * create_ocv_seq(QSP_ARG_DECL  const char *obj_name)
{
	OpenCV_Seq *ocv_seq_p;
	ocv_seq_p = make_new_ocv_seq(QSP_ARG  obj_name);
	if( ocv_seq_p == NO_OPENCV_SEQ ) return(ocv_seq_p);
	ocv_seq_p->ocv_seq = NULL;
	return(ocv_seq_p);
}

OpenCV_Image *creat_ocvi_from_dp(QSP_ARG_DECL  Data_Obj *dp)
{
	OpenCV_Image* ocvi_p = make_new_ocvi(QSP_ARG  OBJ_NAME(dp));
	if( ocvi_p == NO_OPENCV_IMAGE ) {
		return(ocvi_p);
	}
	ocvi_p->ocv_image = (IplImage *)getbuf(sizeof(*ocvi_p->ocv_image));
	/* don't need to test here because getbuf kill whole program if failure... */

	/* Now copy over the fields... */
	IplImage* img = ocvi_p->ocv_image;
	/*img->nSize =
		dp->dt_nelts * ELEMENT_SIZE(dp);*/	/* in bytes... */
	img->nSize = sizeof(*img);
	img->ID = 0;

	img->nChannels = OBJ_COMPS(dp);
	int precision;
	int bpp; // Bytes per pixel.
	switch (OBJ_PREC(dp)) {
		case PREC_UBY: precision = IPL_DEPTH_8U; bpp = 8; break;
		case PREC_BY: precision = IPL_DEPTH_8S;	bpp = 8; break;
		case PREC_IN: precision = IPL_DEPTH_16S; bpp = 16; break;	
		case PREC_UIN: precision = IPL_DEPTH_16U; bpp = 16; break;
		case PREC_DI: precision = IPL_DEPTH_32S; bpp = 32; break;
		case PREC_SP: precision = IPL_DEPTH_32F; bpp = 32; break;
		case PREC_DP: precision = IPL_DEPTH_64F; bpp = 64; break;
		default:
			sprintf(ERROR_STRING,"creat_dp_for_ocvi:  unrecognized depth code %d!?",img->depth);
			WARN(ERROR_STRING);
			return(NO_OPENCV_IMAGE);
			break;

	}
	img->depth = precision;
	img->widthStep = OBJ_COLS(dp)*OBJ_COMPS(dp);
	if (OBJ_FRAMES(dp) != 1) {
		img->dataOrder = 1; // Separate color channels.
		/* img->widthStep = bpp*img->width; */
		img->imageSize = img->height * img->widthStep * OBJ_COMPS(dp);
	} else {
		img->dataOrder = 0; // Interleaved color channels.
		/* img->widthStep = bpp*img->width*OBJ_COMPS(dp); */
		img->imageSize = OBJ_ROWS(dp) * img->widthStep; // Image size in bytes.
	}
	img->origin = 0;		// 0 for upper-left origin.
	img->width = OBJ_COLS(dp);	// Image width.
	img->height = OBJ_ROWS(dp);	// Image height.
	img->roi = NULL;
	img->maskROI = NULL;
	img->imageId = NULL;
	img->tileInfo = NULL;
	img->imageData = (char *)OBJ_DATA_PTR(dp);
	img->imageDataOrigin = (char *)OBJ_UNALIGNED_PTR(dp);
	return(ocvi_p);
}

#endif /* HAVE_OPENCV */

