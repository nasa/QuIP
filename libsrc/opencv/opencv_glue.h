
#ifndef _OPENCV_GLUE_H_
#define _OPENCV_GLUE_H_

#include "quip_config.h"

#ifdef _CH_
#pragma package <opencv>
#endif

//#ifndef _EiC

//#include "cv.h"
//#include "highgui.h"
// on fc12, we install opencv w/ yum, and includes are in /usr/include/opencv
#ifdef HAVE_OPENCV_CVVER_H
#include "opencv/cvver.h"
#endif

#ifndef CV_MAJOR_VERSION
#ifdef HAVE_OPENCV2_CORE_VERSION_HPP
#include <opencv2/core/version.hpp>
#endif
#ifdef HAVE_OPENCV2_CORE_CORE_C_H
#include <opencv2/core/core_c.h>
#endif
#ifdef HAVE_OPENCV2_IMGPROC_IMGPROC_C_H
#include <opencv2/imgproc/imgproc_c.h>
#endif
#ifdef HAVE_OPENCV2_OBJDETECT_OBJDETECT_HPP
#include <opencv2/objdetect/objdetect.hpp>
#endif
#ifdef HAVE_OPENCV2_HIGHGUI_HIGHGUI_C_H
#include <opencv2/highgui/highgui_c.h>
#endif
#endif

#ifndef CV_MAJOR_VERSION
#error "Cannot determine version of OpenCV!?"
#endif

#define MAKE_VERSION_INT(major,minor,micro)	( (major) << 16 | (minor) << 8 | micro )
#define CV_VERSION_INT				MAKE_VERSION_INT(CV_MAJOR_VERSION,CV_MINOR_VERSION,CV_SUBMINOR_VERSION)

#ifdef HAVE_OPENCV_CV_H
#include "opencv/cv.h"
#endif

#ifdef HAVE_OPENCV_HIGHGUI_H
#include "opencv/highgui.h"
#endif

//#endif /*  _EiC */

#include "data_obj.h"

typedef struct opencv_image {
	Item	 ocv_item;
	Data_Obj *ocv_dp;
	IplImage *ocv_image;
} OpenCV_Image;

typedef struct opencv_mem {
	Item	 ocv_item;
	Data_Obj *ocv_dp;
	CvMemStorage *ocv_mem;
} OpenCV_MemStorage;

typedef struct opencv_scanner {
	Item	 ocv_item;
	Data_Obj *ocv_dp;
	CvContourScanner ocv_scanner;
	CvMemStorage *ocv_mem;
} OpenCV_Scanner;

typedef struct opencv_seq {
	Item	 ocv_item;
	Data_Obj *ocv_dp;
	CvSeq* ocv_seq;
} OpenCV_Seq;

typedef struct opencv_cascade {
	Item				ocv_item;
	CvHaarClassifierCascade *	ocv_cascade;
} OpenCV_Cascade;



#define ocv_name	ocv_item.item_name

/* opencv_glue.c */
ITEM_INTERFACE_PROTOTYPES(OpenCV_Image,ocvi)
#define PICK_OCVI(p)	pick_ocvi(QSP_ARG  p)

/* extern OpenCV_Image *make_new_ocvi(const char * obj_name); */
extern OpenCV_Image * load_ocv_image(QSP_ARG_DECL  const char * obj_name, const char * filename );
extern OpenCV_Image * create_ocv_image(QSP_ARG_DECL  const char *obj_name,
			long width,long height,int bit_depth_code,int n_channels);
extern OpenCV_MemStorage * create_ocv_mem(QSP_ARG_DECL  const char *obj_name);
extern OpenCV_Scanner * create_ocv_scanner(QSP_ARG_DECL  const char *obj_name);
extern OpenCV_Seq * create_ocv_seq(QSP_ARG_DECL  const char *obj_name);
extern void save_ocv_image( OpenCV_Image *ocvi_p , const char* filename);
extern OpenCV_Image *creat_ocvi_from_dp(QSP_ARG_DECL  Data_Obj *dp);
/* extern OpenCV_MemStorage *make_new_ocv_mem(char *); */

/* MemStorage */
ITEM_INTERFACE_PROTOTYPES(OpenCV_MemStorage,ocv_mem)

/* Scanner */
ITEM_INTERFACE_PROTOTYPES(OpenCV_Scanner,ocv_scanner)
#define PICK_OCV_SCANNER(p)	pick_ocv_scanner(QSP_ARG  p)

/* Seq */
ITEM_INTERFACE_PROTOTYPES(OpenCV_Seq,ocv_seq)
#define PICK_OCV_SEQ(p)	pick_ocv_seq(QSP_ARG  p)

/* Cascade */
ITEM_INTERFACE_PROTOTYPES(OpenCV_Cascade,ocv_ccasc)
#define PICK_CASCADE(p)	pick_ocv_ccasc(QSP_ARG  p)


#endif // ! _OPENCV_GLUE_H_


