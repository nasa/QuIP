#include "quip_config.h"

char VersionId_opencv_opencv_menu[] = QUIP_VERSION_STRING;

#ifdef HAVE_OPENCV

#include "data_obj.h"
#include "opencv_glue.h"
#include "query.h"
#include "string.h"
#include "debug.h"		/* verbose */
#include "version.h"		/* auto_version() */
#include "submenus.h"		/* ocv_menu() */


void report_opencv_version(SINGLE_QSP_ARG_DECL)
{
	sprintf(ERROR_STRING,"OpenCV version %d.%d.%d",
		CV_MAJOR_VERSION,CV_MINOR_VERSION,CV_SUBMINOR_VERSION);
	advise(ERROR_STRING);
}

COMMAND_FUNC( do_ocv_smooth )
{
	OpenCV_Image *src, *dst;
	int blur_size;

	dst = PICK_OCVI("destination image");
	src = PICK_OCVI("source image");
	blur_size = HOW_MANY("blur size in pixels");

	if( dst == NO_OPENCV_IMAGE || src == NO_OPENCV_IMAGE ) return;

	/* BUG?  need to verify order of args... */
	cvSmooth( src->ocv_image, dst->ocv_image, CV_BLUR, blur_size, blur_size, 0, 0 );
}

COMMAND_FUNC( do_convert_color )
{
	OpenCV_Image *src, *dst;
	const char * s;

	dst = PICK_OCVI("destination image");
	src = PICK_OCVI("source image");
	s = NAMEOF("OpenCV conversion code");

	if( dst == NO_OPENCV_IMAGE || src == NO_OPENCV_IMAGE ) return;

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

COMMAND_FUNC( do_ocv_not )
{
	OpenCV_Image *src, *dst;

	dst = PICK_OCVI("destination image");
	src = PICK_OCVI("source image");

	if( dst == NO_OPENCV_IMAGE || src == NO_OPENCV_IMAGE ) return;

	cvNot( src->ocv_image, dst->ocv_image );
}

// Erosion operator.
COMMAND_FUNC( do_ocv_erode )
{
	OpenCV_Image *src, *dst;
	int iterations;

	dst = PICK_OCVI("destination image");
	src = PICK_OCVI("source image");
	iterations = HOW_MANY("number of iterations");

	if( dst == NO_OPENCV_IMAGE || src == NO_OPENCV_IMAGE ) return;

	// Perform an erosion for the given number of iterations.
	cvErode(src->ocv_image, dst->ocv_image, NULL, iterations);
}

// Dilation operator.
COMMAND_FUNC( do_ocv_dilate )
{
	OpenCV_Image *src, *dst;
	int iterations;

	dst = PICK_OCVI("destination image");
	src = PICK_OCVI("source image");
	iterations = HOW_MANY("number of iterations");

	if( dst == NO_OPENCV_IMAGE || src == NO_OPENCV_IMAGE ) return;

	// Perform an erosion for the given number of iterations.
	cvDilate(src->ocv_image, dst->ocv_image, NULL, iterations);
}

// Binary threshold.
COMMAND_FUNC( do_ocv_binary_threshold )
{
	OpenCV_Image *src, *dst;
	double threshold;
	double max_value;

	dst = PICK_OCVI("destination image");
	src = PICK_OCVI("source image");
	threshold = HOW_MANY("threshold above which will be on");
	max_value = HOW_MANY("on value");

	if( dst == NO_OPENCV_IMAGE || src == NO_OPENCV_IMAGE ) return;
	cvThreshold(src->ocv_image, dst->ocv_image, threshold, max_value, CV_THRESH_BINARY);
}

COMMAND_FUNC( do_ocv_canny )
{
	OpenCV_Image *src, *dst;
	int edge_thresh;
	int edge_thresh2;

	dst = PICK_OCVI("destination image");
	src = PICK_OCVI("source image");
	edge_thresh = HOW_MANY("edge threshold");
	edge_thresh2 = HOW_MANY("edge threshold 2");

	if( dst == NO_OPENCV_IMAGE || src == NO_OPENCV_IMAGE ) return;

	// Run the edge detector on grayscale.
	// The optional aperture_size parameter has been omitted.
	cvCanny(src->ocv_image, dst->ocv_image, (float)edge_thresh, (float)edge_thresh2, 3);
}

COMMAND_FUNC( do_ocv_zero )
{
	OpenCV_Image *dst;

	dst = PICK_OCVI("destination image");

	if( dst == NO_OPENCV_IMAGE ) return;

	cvZero( dst->ocv_image );
}

COMMAND_FUNC( do_creat_img )
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
}

/* Create a MemStorage object with the given string name. */
COMMAND_FUNC( do_create_mem )
{
	const char *s;
	OpenCV_MemStorage *ocv_mem_p;

	s=NAMEOF("name for MemStorage");
	ocv_mem_p = create_ocv_mem(QSP_ARG  s);
}

/* Create a Scanner object with the given string name. */
COMMAND_FUNC( do_create_scanner )
{
	const char *s;
	OpenCV_Scanner *ocv_scanner_p;

	s=NAMEOF("name for Scanner");
	ocv_scanner_p = create_ocv_scanner(QSP_ARG  s);
	/*sprintf(ERROR_STRING, "do_create_scanner: Address of new scanner: %p", &(ocv_scanner_p->ocv_scanner));
	WARN(ERROR_STRING);*/
}

/* Create a Seq object with the given string name. */
COMMAND_FUNC( do_create_seq )
{
	const char *s;
	OpenCV_Seq *ocv_seq_p;

	s=NAMEOF("name for Seq");
	ocv_seq_p = create_ocv_seq(QSP_ARG  s);
	/*sprintf(ERROR_STRING, "do_create_seq: Address of new seq: %p", ocv_seq_p->ocv_seq);
	WARN(ERROR_STRING);*/
}

/* Create a Seq object with the given string name. */
COMMAND_FUNC( do_seq_is_null )
{
	OpenCV_Seq *ocv_seq_p;
	ocv_seq_p=PICK_OCV_SEQ("sequence");
	if( ocv_seq_p == NO_OPENCV_SEQ ) return;
	if( ocv_seq_p->ocv_seq == NULL ) {
		ASSIGN_VAR("seq_is_null", "1");
	} else {
		ASSIGN_VAR("seq_is_null", "0");
	}
}

COMMAND_FUNC( do_load_img )
{
	const char *object_name;
	const char *filename;
	long is_color;
	OpenCV_Image *ocvi_p;

	object_name=NAMEOF("object name");
	filename=NAMEOF("image filename");
	is_color=HOW_MANY("is color? (boolean)");
	ocvi_p = load_ocv_image(QSP_ARG  object_name, filename);
}

COMMAND_FUNC( do_save_img )
{
	const char *filename;
	OpenCV_Image *ocvi_p;

	ocvi_p=PICK_OCVI("openCV image");
	filename=NAMEOF("file name");
	if( ocvi_p == NO_OPENCV_IMAGE ) return;

	save_ocv_image(ocvi_p, filename);
}

COMMAND_FUNC( do_find_contours )
{
	OpenCV_Scanner *ocv_scanner_p;
	OpenCV_Image *ocvi_p;
	/* OpenCV_MemStorage *ocv_mem_p; */

	ocv_scanner_p=PICK_OCV_SCANNER("scanner");
	ocvi_p=PICK_OCVI("binary image");
	/* ocv_mem_p=PICK_OCV_MEM("memory storage"); */
	if( ocv_scanner_p == NO_OPENCV_SCANNER ) return;
	if( ocvi_p == NO_OPENCV_IMAGE ) return;
	/* if( ocv_mem_p == NO_OPENCV_MEM ) return; */
	ocv_scanner_p->ocv_scanner = cvStartFindContours(ocvi_p->ocv_image,
			ocv_scanner_p->ocv_mem, sizeof(CvContour), CV_RETR_EXTERNAL,
			CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
}


COMMAND_FUNC( do_find_next_contour )
{
	OpenCV_Scanner *ocv_scanner_p;
	OpenCV_Seq *ocv_seq_p;
	/*int success = 0;*/

	ocv_scanner_p=PICK_OCV_SCANNER("scanner");
	ocv_seq_p=PICK_OCV_SEQ("sequence");
	/*success=PICK_OBJ("success flag");*/

	if( ocv_scanner_p == NO_OPENCV_SCANNER ) return;
	if( ocv_seq_p == NO_OPENCV_SEQ ) return;

	if ((ocv_seq_p->ocv_seq = cvFindNextContour(ocv_scanner_p->ocv_scanner)) != NULL) {
		ASSIGN_VAR("contour_success", "1");
	} else {
		ASSIGN_VAR("contour_success", "0");
	}
}

COMMAND_FUNC( do_aspect_ratio )
{
	OpenCV_Seq *ocv_seq_p;
	ocv_seq_p=PICK_OCV_SEQ("sequence");
	if( ocv_seq_p == NO_OPENCV_SEQ ) return;
	if( ocv_seq_p->ocv_seq == NULL ) {
		sprintf(ERROR_STRING, "do_aspect_ratio: sequence is NULL.\n");
		//WARN(ERROR_STRING);
		return;
	}
	CvRect r = cvBoundingRect(ocv_seq_p->ocv_seq, 1);
	float aspectRatio = (float)(r.width)/(float)(r.height);
	char aspect[30];
	sprintf(aspect, "%f", aspectRatio);
	ASSIGN_VAR("contour_aspect", aspect);
	/*sprintf(ERROR_STRING,"Aspect ratio = %f", aspectRatio);
	WARN(ERROR_STRING);*/
}

COMMAND_FUNC( do_ocv_area )
{
	OpenCV_Seq *ocv_seq_p;
	ocv_seq_p=PICK_OCV_SEQ("sequence");
	if( ocv_seq_p == NO_OPENCV_SEQ ) return;
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
	ASSIGN_VAR("contour_area", area_string);
}

COMMAND_FUNC( do_centroid )
{
	OpenCV_Seq *ocv_seq_p;
	ocv_seq_p=PICK_OCV_SEQ("sequence");
	if( ocv_seq_p == NO_OPENCV_SEQ ) return;
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
	ASSIGN_VAR("centroid_x", number);
	sprintf(number, "%f", y);
	ASSIGN_VAR("centroid_y", number);


}

COMMAND_FUNC( do_image_info )
{
	OpenCV_Image *ocvi_p;
	ocvi_p=PICK_OCVI("openCV image");
	if( ocvi_p == NO_OPENCV_IMAGE ) return;

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

COMMAND_FUNC( do_import_img )
{
	Data_Obj *dp;
	OpenCV_Image *ocvi_p;

	dp = PICK_OBJ("image");

	/* Possibly do some tests here */

	if( (ocvi_p = creat_ocvi_from_dp(QSP_ARG  dp) ) == NO_OPENCV_IMAGE ){
		sprintf(ERROR_STRING,"Error creating OpenCV image from QuIP image %s",dp->dt_name);
		WARN(ERROR_STRING);
	}
}

COMMAND_FUNC( do_new_cascade )
{
	const char * s;
	const char *cascade_name;
	OpenCV_Cascade *casc_p;

	s=NAMEOF("classifier cascade");
	cascade_name = NAMEOF("classifier specification file");

	casc_p = ocv_ccasc_of(QSP_ARG  s);
	if( casc_p != NO_CASCADE ){
		sprintf(ERROR_STRING,"Classifier cascade %s already exists!?",s);
		WARN(ERROR_STRING);
		return;
	}

	casc_p = new_ocv_ccasc(QSP_ARG  s);
	if( casc_p == NO_CASCADE ){
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
#define bool int

CvPoint2D32f* FindFace(QSP_ARG_DECL  IplImage* img, CvMemStorage* storage, CvHaarClassifierCascade* cascade, int frame_number)
{
	bool face_found = false;
	int faces_found = 0;
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
			faces_found = faces->total;
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
		ASSIGN_VAR("n_faces",msg);

		sprintf(ERROR_STRING, "faces %d %d ", frame_number, faces->total);

		for (i=0; i<(faces ? faces->total:0); ++i) {
			char vname[64], vval[64];

			CvRect* r = (CvRect*)cvGetSeqElem(faces, i);
			sprintf(msg, "%d %d %d %d ", r->x, r->y, r->width, r->height);
			strcat(ERROR_STRING, msg);

			sprintf(vname,"face%d_x",i+1);
			sprintf(vval,"%d",r->x);
			ASSIGN_VAR(vname,vval);

			sprintf(vname,"face%d_y",i+1);
			sprintf(vval,"%d",r->y);
			ASSIGN_VAR(vname,vval);

			sprintf(vname,"face%d_w",i+1);
			sprintf(vval,"%d",r->width);
			ASSIGN_VAR(vname,vval);

			sprintf(vname,"face%d_h",i+1);
			sprintf(vval,"%d",r->height);
			ASSIGN_VAR(vname,vval);

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
}

static CvMemStorage* storage = NULL;

COMMAND_FUNC( do_find_face )
{
	OpenCV_Image *src;
	OpenCV_Cascade *casc_p;
	CvPoint2D32f* features;
	int frame_number;

	src = PICK_OCVI("input OpenCV image");
	casc_p = PICK_CASCADE("classifier cascade");
	frame_number = HOW_MANY("frame number");

	if( src == NO_OPENCV_IMAGE || casc_p == NO_CASCADE ) return;

	if( storage == NULL )
		storage = cvCreateMemStorage(0); // what is the arg here?

	features = FindFace(QSP_ARG  src->ocv_image, storage, casc_p->ocv_cascade, frame_number);
	//cvReleaseImage(&src->ocv_image);
	/* BUG need to do something with the feature array */

	// Deallocate memory.
	cvReleaseMemStorage(&storage);
}

static Command face_finder_ctbl[]={
{ "cascade",		do_new_cascade,		"create a new classifier cascade"	},
{ "find_face",		do_find_face,		"find face/eyes"			},
{ "quit",		popcmd,			"exit submenu"				},
{ NULL_COMMAND										}
};

COMMAND_FUNC( do_face_finder )
{
	PUSHCMD(face_finder_ctbl,"face_finder");
}

static Command ocv_ctbl[]={
{ "face_finder",	do_face_finder,		"face-finder submenu"			},
{ "new_image",		do_creat_img,		"Create new OpenCV image"		},
{ "import",		do_import_img,		"import an image from QuIP to OpenCV"	},
{ "load_image",		do_load_img,		"load an image file"			},
{ "save_image",		do_save_img,		"save an image file"			},
{ "zero",		do_ocv_zero,		"zero an OpenCV image"			},
{ "area",		do_ocv_area,		"Area of a sequence."			},
{ "aspect_ratio",	do_aspect_ratio,	"Aspect ratio of a sequence."		},
{ "canny",		do_ocv_canny,		"find edges w/ Canny detector"		},
{ "centroid",		do_centroid,		"Centroid of a sequence."		},
{ "convert_color",	do_convert_color,	"convert to another color space"	},
{ "dilate",		do_ocv_dilate,		"dilate binary image"			},
{ "erode",		do_ocv_erode,		"erode binary image"			},
{ "find_contours",	do_find_contours,	"Find contours."			},
{ "find_next_contour",	do_find_next_contour,	"Find next contour."			},

{ "binary_threshold",	do_ocv_binary_threshold,"binary threshold image"		},
{ "smooth",		do_ocv_smooth,		"smooth image"				},
{ "not",		do_ocv_not,		"complement image"			},
/* MemStorage */
{ "new_mem",		do_create_mem,		"Create a new MemStorage object."	},
{ "new_scanner",	do_create_scanner,	"Create a new MemStorage object."	},
{ "new_seq",		do_create_seq,		"Create a new MemStorage object."	},
{ "is_seq_null",	do_seq_is_null,		"Determine if a sequence is NULL."	},
{ "image_info",		do_image_info,		"Display OpenCV image info."		},
{ "quit",		popcmd,			"exit submenu"				},
{ NULL_COMMAND										}
};

static int ocv_inited=0;

COMMAND_FUNC( ocv_menu )
{
	if( !ocv_inited ){
		auto_version(QSP_ARG  "OPENCV","VersionId_opencv");
		ocv_inited=1;
	}
	PUSHCMD( ocv_ctbl, "OpenCV" );
}


#endif /* HAVE_OPENCV */

