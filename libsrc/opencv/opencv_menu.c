#include "quip_config.h"

#ifdef HAVE_OPENCV

#include "quip_prot.h"
#include "opencv_glue.h"
#include "data_obj.h"
#include <math.h>	// fabs

// local prototypes
static COMMAND_FUNC( do_ocv_smooth );
static COMMAND_FUNC( do_convert_color );
static COMMAND_FUNC( do_ocv_not );
static COMMAND_FUNC( do_ocv_erode );
static COMMAND_FUNC( do_ocv_dilate );
static COMMAND_FUNC( do_ocv_binary_threshold );
static COMMAND_FUNC( do_ocv_canny );
static COMMAND_FUNC( do_ocv_zero );
static COMMAND_FUNC( do_creat_img );
static COMMAND_FUNC( do_create_mem );
static COMMAND_FUNC( do_create_scanner );
static COMMAND_FUNC( do_create_seq );
static COMMAND_FUNC( do_seq_is_null );
static COMMAND_FUNC( do_load_img );
static COMMAND_FUNC( do_save_img );
static COMMAND_FUNC( do_find_contours );
static COMMAND_FUNC( do_find_next_contour );
static COMMAND_FUNC( do_aspect_ratio );
static COMMAND_FUNC( do_ocv_area );
static COMMAND_FUNC( do_centroid );
static COMMAND_FUNC( do_image_info );
static COMMAND_FUNC( do_import_img );
static COMMAND_FUNC( do_new_cascade );
static CvPoint2D32f* FindFace(QSP_ARG_DECL  IplImage* img,
					CvMemStorage* storage,
					CvHaarClassifierCascade* cascade,
					int frame_number);
static COMMAND_FUNC( do_find_face );
static COMMAND_FUNC( do_face_finder );

#ifdef NOT_USED
void report_opencv_version(SINGLE_QSP_ARG_DECL)
{
	sprintf(ERROR_STRING,"OpenCV version %d.%d.%d",
		CV_MAJOR_VERSION,CV_MINOR_VERSION,CV_SUBMINOR_VERSION);
	advise(ERROR_STRING);
}
#endif /* NOT_USED */

static COMMAND_FUNC( do_ocv_smooth )
{
	OpenCV_Image *src, *dst;
	int blur_size;

	dst = pick_ocvi("destination image");
	src = pick_ocvi("source image");
	blur_size = HOW_MANY("blur size in pixels");

	if( dst == NULL || src == NULL ) return;

	/* BUG?  need to verify order of args... */
	cvSmooth( src->ocv_image, dst->ocv_image, CV_BLUR, blur_size, blur_size, 0, 0 );
}

static COMMAND_FUNC( do_convert_color )
{
	OpenCV_Image *src, *dst;
	const char * s;

	dst = pick_ocvi("destination image");
	src = pick_ocvi("source image");
	s = NAMEOF("OpenCV conversion code");

	if( dst == NULL || src == NULL ) return;

	int code;
	if (strcmp(s, "CV_RGB2GRAY") == 0) {
		code = CV_RGB2GRAY;
	} else if (strcmp(s, "CV_GRAY2RGB") == 0) {
		code = CV_GRAY2RGB;
	} else if (strcmp(s, "CV_BGR2GRAY") == 0) {
		code = CV_BGR2GRAY;
	} else if (strcmp(s, "CV_GRAY2BGR") == 0) {
		code = CV_GRAY2BGR;
	} else {
		code = -1;
		sprintf(ERROR_STRING,"Error (do_convert_color): No such OpenCV conversion code: %s\n",s);
		//WARN(ERROR_STRING);
		return;
	}
	cvCvtColor(src->ocv_image, dst->ocv_image, code);
}

static COMMAND_FUNC( do_ocv_not )
{
	OpenCV_Image *src, *dst;

	dst = pick_ocvi("destination image");
	src = pick_ocvi("source image");

	if( dst == NULL || src == NULL ) return;

	cvNot( src->ocv_image, dst->ocv_image );
}

// Erosion operator.
static COMMAND_FUNC( do_ocv_erode )
{
	OpenCV_Image *src, *dst;
	int iterations;

	dst = pick_ocvi("destination image");
	src = pick_ocvi("source image");
	iterations = HOW_MANY("number of iterations");

	if( dst == NULL || src == NULL ) return;

	// Perform an erosion for the given number of iterations.
	cvErode(src->ocv_image, dst->ocv_image, NULL, iterations);
}

// Dilation operator.
static COMMAND_FUNC( do_ocv_dilate )
{
	OpenCV_Image *src, *dst;
	int iterations;

	dst = pick_ocvi("destination image");
	src = pick_ocvi("source image");
	iterations = HOW_MANY("number of iterations");

	if( dst == NULL || src == NULL ) return;

	// Perform an erosion for the given number of iterations.
	cvDilate(src->ocv_image, dst->ocv_image, NULL, iterations);
}

// Binary threshold.
static COMMAND_FUNC( do_ocv_binary_threshold )
{
	OpenCV_Image *src, *dst;
	double threshold;
	double max_value;

	dst = pick_ocvi("destination image");
	src = pick_ocvi("source image");
	threshold = HOW_MANY("threshold above which will be on");
	max_value = HOW_MANY("on value");

	if( dst == NULL || src == NULL ) return;
	cvThreshold(src->ocv_image, dst->ocv_image, threshold, max_value, CV_THRESH_BINARY);
}

static COMMAND_FUNC( do_ocv_canny )
{
	OpenCV_Image *src, *dst;
	int edge_thresh;
	int edge_thresh2;

	dst = pick_ocvi("destination image");
	src = pick_ocvi("source image");
	edge_thresh = HOW_MANY("edge threshold");
	edge_thresh2 = HOW_MANY("edge threshold 2");

	if( dst == NULL || src == NULL ) return;

	// Run the edge detector on grayscale.
	// The optional aperture_size parameter has been omitted.
	cvCanny(src->ocv_image, dst->ocv_image, (float)edge_thresh, (float)edge_thresh2, 3);
}

static COMMAND_FUNC( do_ocv_zero )
{
	OpenCV_Image *dst;

	dst = pick_ocvi("destination image");

	if( dst == NULL ) return;

	cvZero( dst->ocv_image );
}

static COMMAND_FUNC( do_creat_img )
{
	const char *s;
	long w,h,n_channels;
	OpenCV_Image *ocvi_p;

	s=NAMEOF("name for image");
	w=HOW_MANY("width");
	h=HOW_MANY("height");
	n_channels = HOW_MANY("number of components");
	/* BUG should have a precision switch here, using which_one... */

	ocvi_p = create_ocv_image(QSP_ARG  s,w,h,IPL_DEPTH_8U,n_channels);

	if( ocvi_p == NULL ) WARN("Error creating openCV image!?");
}

/* Create a MemStorage object with the given string name. */
static COMMAND_FUNC( do_create_mem )
{
	const char *s;
	OpenCV_MemStorage *ocv_mem_p;

	s=NAMEOF("name for MemStorage");
	ocv_mem_p = create_ocv_mem(QSP_ARG  s);

	if( ocv_mem_p == NULL ) WARN("Error creating openCV memory area!?");
}

/* Create a Scanner object with the given string name. */
static COMMAND_FUNC( do_create_scanner )
{
	const char *s;
	OpenCV_Scanner *ocv_scanner_p;

	s=NAMEOF("name for Scanner");
	ocv_scanner_p = create_ocv_scanner(QSP_ARG  s);
	/*sprintf(ERROR_STRING, "do_create_scanner: Address of new scanner: %p", &(ocv_scanner_p->ocv_scanner));
	WARN(ERROR_STRING);*/

	if( ocv_scanner_p == NULL ) WARN("Error creating openCV scanner!?");
}

/* Create a Seq object with the given string name. */
static COMMAND_FUNC( do_create_seq )
{
	const char *s;
	OpenCV_Seq *ocv_seq_p;

	s=NAMEOF("name for Seq");
	ocv_seq_p = create_ocv_seq(QSP_ARG  s);
	/*sprintf(ERROR_STRING, "do_create_seq: Address of new seq: %p", ocv_seq_p->ocv_seq);
	WARN(ERROR_STRING);*/

	if( ocv_seq_p == NULL ) WARN("Error creating openCV sequence!?");
}

/* Create a Seq object with the given string name. */
static COMMAND_FUNC( do_seq_is_null )
{
	OpenCV_Seq *ocv_seq_p;
	ocv_seq_p=pick_ocv_seq("sequence");
	if( ocv_seq_p == NULL ) return;
	if( ocv_seq_p->ocv_seq == NULL ) {
		assign_var("seq_is_null", "1");
	} else {
		assign_var("seq_is_null", "0");
	}
}

static COMMAND_FUNC( do_load_img )
{
	const char *object_name;
	const char *filename;
	long is_color;
	OpenCV_Image *ocvi_p;

	object_name=NAMEOF("object name");
	filename=NAMEOF("image filename");
	is_color=HOW_MANY("is color? (boolean)");
	ocvi_p = load_ocv_image(QSP_ARG  object_name, filename);

	if( ocvi_p == NULL ) WARN("Error loading openCV image!?");
}

static COMMAND_FUNC( do_save_img )
{
	const char *filename;
	OpenCV_Image *ocvi_p;

	ocvi_p=pick_ocvi("openCV image");
	filename=NAMEOF("file name");
	if( ocvi_p == NULL ) return;

	save_ocv_image(ocvi_p, filename);
}

static COMMAND_FUNC( do_find_contours )
{
	OpenCV_Scanner *ocv_scanner_p;
	OpenCV_Image *ocvi_p;
	/* OpenCV_MemStorage *ocv_mem_p; */

	ocv_scanner_p=pick_ocv_scanner("scanner");
	ocvi_p=pick_ocvi("binary image");
	/* ocv_mem_p=PICK_OCV_MEM("memory storage"); */
	if( ocv_scanner_p == NULL ) return;
	if( ocvi_p == NULL ) return;
	/* if( ocv_mem_p == NULL ) return; */
	ocv_scanner_p->ocv_scanner = cvStartFindContours(ocvi_p->ocv_image,
			ocv_scanner_p->ocv_mem, sizeof(CvContour), CV_RETR_EXTERNAL,
			CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
}


static COMMAND_FUNC( do_find_next_contour )
{
	OpenCV_Scanner *ocv_scanner_p;
	OpenCV_Seq *ocv_seq_p;
	/*int success = 0;*/

	ocv_scanner_p=pick_ocv_scanner("scanner");
	ocv_seq_p=pick_ocv_seq("sequence");
	/*success=pick_obj("success flag");*/

	if( ocv_scanner_p == NULL ) return;
	if( ocv_seq_p == NULL ) return;

	if ((ocv_seq_p->ocv_seq = cvFindNextContour(ocv_scanner_p->ocv_scanner)) != NULL) {
		assign_var("contour_success", "1");
	} else {
		assign_var("contour_success", "0");
	}
}

static COMMAND_FUNC( do_aspect_ratio )
{
	OpenCV_Seq *ocv_seq_p;
	ocv_seq_p=pick_ocv_seq("sequence");
	if( ocv_seq_p == NULL ) return;
	if( ocv_seq_p->ocv_seq == NULL ) {
		sprintf(ERROR_STRING, "do_aspect_ratio: sequence is NULL.\n");
		//WARN(ERROR_STRING);
		return;
	}
	CvRect r = cvBoundingRect(ocv_seq_p->ocv_seq, 1);
	float aspectRatio = (float)(r.width)/(float)(r.height);
	char aspect[30];
	sprintf(aspect, "%f", aspectRatio);
	assign_var("contour_aspect", aspect);
	/*sprintf(ERROR_STRING,"Aspect ratio = %f", aspectRatio);
	WARN(ERROR_STRING);*/
}

static COMMAND_FUNC( do_ocv_area )
{
	OpenCV_Seq *ocv_seq_p;
	ocv_seq_p=pick_ocv_seq("sequence");
	if( ocv_seq_p == NULL ) return;
	if( ocv_seq_p->ocv_seq == NULL ) {
		sprintf(ERROR_STRING, "do_ocv_area: sequence is NULL.\n");
		WARN(ERROR_STRING);
		return;
	}

/* Not sure when this change took effect... */
//#if CV_VERSION_INT > MAKE_VERSION_INT(1,1,0)
#if CV_VERSION_INT > MAKE_VERSION_INT(2,0,0)
	float area = fabs(cvContourArea(ocv_seq_p->ocv_seq, CV_WHOLE_SEQ, 0 ));
#else
	float area = fabs(cvContourArea(ocv_seq_p->ocv_seq, CV_WHOLE_SEQ ));
#endif

	char area_string[30];
	sprintf(area_string, "%f", area);
	assign_var("contour_area", area_string);
}

static COMMAND_FUNC( do_centroid )
{
	OpenCV_Seq *ocv_seq_p;
	ocv_seq_p=pick_ocv_seq("sequence");
	if( ocv_seq_p == NULL ) return;
	if( ocv_seq_p->ocv_seq == NULL ) {
		sprintf(ERROR_STRING, "do_centroid: sequence is NULL.\n");
		//WARN(ERROR_STRING);
		return;
	}
	CvMoments moments;
	double M00, M01, M10;
	float x,y;
	cvMoments(ocv_seq_p->ocv_seq, &moments, 1);
	M00 = cvGetSpatialMoment(&moments, 0, 0);
	M10 = cvGetSpatialMoment(&moments, 1, 0);
	M01 = cvGetSpatialMoment(&moments, 0, 1);
	x = (int)(M10/M00);
	y = (int)(M01/M00);
	/* char msg[30];
	sprintf(msg, "M00 = %f, M10 = %f, M01 = %f", M00, M10, M01);
	WARN(msg); */

	char number[30];
	sprintf(number, "%f", x);
	assign_var("centroid_x", number);
	sprintf(number, "%f", y);
	assign_var("centroid_y", number);


}

static COMMAND_FUNC( do_image_info )
{
	OpenCV_Image *ocvi_p;
	ocvi_p=pick_ocvi("openCV image");
	if( ocvi_p == NULL ) return;

	sprintf(ERROR_STRING,"Image Info for \"%s\":",ocvi_p->ocv_name);
	prt_msg(ERROR_STRING);

	IplImage* img = ocvi_p->ocv_image;

	sprintf(ERROR_STRING, "\tSize: (%d, %d)",img->width, img->height);
	prt_msg(ERROR_STRING);
	sprintf(ERROR_STRING, "\tnSize: %d",img->nSize);
	prt_msg(ERROR_STRING);
	sprintf(ERROR_STRING, "\tID: %d",img->ID);
	prt_msg(ERROR_STRING);
	sprintf(ERROR_STRING, "\tnChannels: %d",img->nChannels);
	prt_msg(ERROR_STRING);
	sprintf(ERROR_STRING, "\tdepth: %d",img->depth);
	prt_msg(ERROR_STRING);
	sprintf(ERROR_STRING, "\tdataOrder: %d",img->dataOrder);
	prt_msg(ERROR_STRING);
	sprintf(ERROR_STRING, "\torigin: %d",img->origin);
	prt_msg(ERROR_STRING);
	sprintf(ERROR_STRING, "\timageSize: %d",img->imageSize);
	prt_msg(ERROR_STRING);
	sprintf(ERROR_STRING, "\timageData: 0x%lx",(u_long)img->imageData);
	prt_msg(ERROR_STRING);
	sprintf(ERROR_STRING, "\twidthStep: %d",img->widthStep);
	prt_msg(ERROR_STRING);
	sprintf(ERROR_STRING, "\timageDataOrigin: 0x%lx",(u_long)img->imageDataOrigin);
	prt_msg(ERROR_STRING);
}

static COMMAND_FUNC( do_import_img )
{
	Data_Obj *dp;
	OpenCV_Image *ocvi_p;

	dp = pick_obj("image");

	/* Possibly do some tests here */

	if( (ocvi_p = creat_ocvi_from_dp(QSP_ARG  dp) ) == NULL ){
		sprintf(ERROR_STRING,"Error creating OpenCV image from QuIP image %s",OBJ_NAME(dp));
		WARN(ERROR_STRING);
	}
}

static COMMAND_FUNC( do_new_cascade )
{
	const char * s;
	const char *cascade_name;
	OpenCV_Cascade *casc_p;

	s=NAMEOF("classifier cascade");
	cascade_name = NAMEOF("classifier specification file");

	casc_p = ocv_ccasc_of(s);
	if( casc_p != NULL ){
		sprintf(ERROR_STRING,"Classifier cascade %s already exists!?",s);
		WARN(ERROR_STRING);
		return;
	}

	casc_p = new_ocv_ccasc(s);
	if( casc_p == NULL ){
		sprintf(ERROR_STRING,"Error creating classifier cascade %s",s);
		WARN(ERROR_STRING);
		return;
	}

	casc_p->ocv_cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0,0,0);
	if(casc_p->ocv_cascade == NULL) {
		sprintf(ERROR_STRING,"Error loading cascade from file %s",cascade_name);
		WARN(ERROR_STRING);
		return;
		/* BUG release struct here */
	}
}

#define false 0
#define true 1

#ifndef bool
// When do we need this definition?
#define bool int
#endif // undef bool

static CvPoint2D32f* FindFace(QSP_ARG_DECL  IplImage* img, CvMemStorage* storage, CvHaarClassifierCascade* cascade, int frame_number)
{
	bool face_found = false;
	CvPoint2D32f* features = NULL;
	int max_faces = 4;
	//int scale = 1;
	//CvPoint pt1, pt2;		// two points to represent the face locations.
	int i;
	int k;

	if( cascade == NULL ){
		WARN("FindFace:  cascade is NULL !?");
		return(NULL);
	}

	/*
	IplImage* img_copy = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, img->nChannels);
	if(img->origin == IPL_ORIGIN_TL) {
		cvCopy(img, img_copy, 0);
	} else {
		cvFlip(img, img_copy,0);
	}
	*/

	// Clear the memory storage which was used before.
	cvClearMemStorage(storage);

	if (cascade) {
		// Detect the objects and store them in the sequence.
/*
CVAPI(CvSeq*) cvHaarDetectObjects( const CvArr* image,
                     CvHaarClassifierCascade* cascade, CvMemStorage* storage, 
                     double scale_factor CV_DEFAULT(1.1),
                     int min_neighbors CV_DEFAULT(3), int flags CV_DEFAULT(0),
                     CvSize min_size CV_DEFAULT(cvSize(0,0)), CvSize max_size CV_DEFAULT(cvSize(0,0)));
		     */

//#if CV_VERSION_INT > MAKE_VERSION_INT(1,1,0)
#if CV_VERSION_INT > MAKE_VERSION_INT(2,0,0)
		CvSeq* faces = cvHaarDetectObjects(
					img, cascade, storage,
					1.1, 4, CV_HAAR_DO_CANNY_PRUNING,
					cvSize(30,30),cvSize(60,60) );
#else
		CvSeq* faces = cvHaarDetectObjects(
					img, cascade, storage,
					1.1, 4, CV_HAAR_DO_CANNY_PRUNING,
					cvSize(30,30) );
#endif
		if (faces != NULL) {
			if( verbose ){
				sprintf(ERROR_STRING,"%d faces found in frame %d.",faces->total, frame_number);
				advise(ERROR_STRING);
			}
		} else {
			/* is this a CAUTIOUS check? */
			sprintf(ERROR_STRING,"faces is NULL.");
			advise(ERROR_STRING);
		}
		char msg[60];

		sprintf(msg,"%d",faces->total);
		assign_var("n_faces",msg);

		sprintf(ERROR_STRING, "faces %d %d ", frame_number, faces->total);

		for (i=0; i<(faces ? faces->total:0); ++i) {
			char vname[64], vval[64];

			CvRect* r = (CvRect*)cvGetSeqElem(faces, i);
			sprintf(msg, "%d %d %d %d ", r->x, r->y, r->width, r->height);
			strcat(ERROR_STRING, msg);

			sprintf(vname,"face%d_x",i+1);
			sprintf(vval,"%d",r->x);
			assign_var(vname,vval);

			sprintf(vname,"face%d_y",i+1);
			sprintf(vval,"%d",r->y);
			assign_var(vname,vval);

			sprintf(vname,"face%d_w",i+1);
			sprintf(vval,"%d",r->width);
			assign_var(vname,vval);

			sprintf(vname,"face%d_h",i+1);
			sprintf(vval,"%d",r->height);
			assign_var(vname,vval);

#ifdef FOOBAR
			/* don't draw here - we might not be viewing... */
			// Draw rectangle.
			CvPoint pt1, pt2;
			pt1.x = r->x;
			pt2.x = r->x+r->width;
			pt1.y = r->y;
			pt2.y = r->y+r->height;
			cvRectangle(img, pt1,pt2, CV_RGB(255,0,0), 3, 8, 0);
#endif /* FOOBAR */
		}
		if (faces->total < max_faces) {
			// Append zeros to make the output always have a fixed
			// number of columns.
			for (k=0; k< (max_faces - faces->total); ++k) {
				sprintf(msg, "0 0 0 0 ");
				strcat(ERROR_STRING, msg);
			}
		}
		prt_msg(ERROR_STRING);
		//cvSaveImage("output.jpg", img);
	}
	if (!face_found) {
		features = NULL;
	//	cvCircle(img, cvPoint(img->width/2, img->height/2), 30, CV_RGB(255,0,0), 2, 8);
	}

	//cvReleaseImage(&img_copy);
	return features;
} // end FindFace

static CvMemStorage* storage = NULL;

static COMMAND_FUNC( do_find_face )
{
	OpenCV_Image *src;
	OpenCV_Cascade *casc_p;
	CvPoint2D32f* features;
	int frame_number;

	src = pick_ocvi("input OpenCV image");
	casc_p = pick_cascade("classifier cascade");
	frame_number = HOW_MANY("frame number");

	if( src == NULL || casc_p == NULL ) return;

	if( storage == NULL )
		storage = cvCreateMemStorage(0); // what is the arg here?

	features = FindFace(QSP_ARG  src->ocv_image, storage, casc_p->ocv_cascade, frame_number);
	//cvReleaseImage(&src->ocv_image);
	/* BUG need to do something with the feature array */
	if( features == NULL )
		WARN("find_face:  Null feature array!?");

	// Deallocate memory.
	cvReleaseMemStorage(&storage);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(face_finder_menu,s,f,h)

MENU_BEGIN(face_finder)
ADD_CMD( cascade,	do_new_cascade,	create a new classifier cascade )
ADD_CMD( find_face,	do_find_face,	find face/eyes )
MENU_END(face_finder)

static COMMAND_FUNC( do_face_finder )
{
	PUSH_MENU(face_finder);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(open_cv_menu,s,f,h)

MENU_BEGIN(open_cv)
ADD_CMD( face_finder,		do_face_finder,		face-finder submenu )
ADD_CMD( new_image,		do_creat_img,		Create new OpenCV image )
ADD_CMD( import,		do_import_img,		import an image from QuIP to OpenCV )
ADD_CMD( load_image,		do_load_img,		load an image file )
ADD_CMD( save_image,		do_save_img,		save an image file )
ADD_CMD( zero,			do_ocv_zero,		zero an OpenCV image )
ADD_CMD( area,			do_ocv_area,		Area of a sequence )
ADD_CMD( aspect_ratio,		do_aspect_ratio,	Aspect ratio of a sequence )
ADD_CMD( canny,			do_ocv_canny,		find edges w/ Canny detector )
ADD_CMD( centroid,		do_centroid,		Centroid of a sequence )
ADD_CMD( convert_color,		do_convert_color,	convert to another color space )
ADD_CMD( dilate,		do_ocv_dilate,		dilate binary image )
ADD_CMD( erode,			do_ocv_erode,		erode binary image )
ADD_CMD( find_contours,		do_find_contours,	Find contours )
ADD_CMD( find_next_contour,	do_find_next_contour,	Find next contour )

ADD_CMD( binary_threshold,	do_ocv_binary_threshold,	binary threshold image )
ADD_CMD( smooth,		do_ocv_smooth,		smooth image )
ADD_CMD( not,			do_ocv_not,		complement image )
/* MemStorage */
ADD_CMD( new_mem,		do_create_mem,		Create a new MemStorage object )
ADD_CMD( new_scanner,		do_create_scanner,	Create a new MemStorage object )
ADD_CMD( new_seq,		do_create_seq,		Create a new MemStorage object )
ADD_CMD( is_seq_null,		do_seq_is_null,		Determine if a sequence is NULL )
ADD_CMD( image_info,		do_image_info,		Display OpenCV image info )
MENU_END(open_cv)

COMMAND_FUNC( do_ocv_menu )
{
	PUSH_MENU( open_cv );
}


#endif /* HAVE_OPENCV */

