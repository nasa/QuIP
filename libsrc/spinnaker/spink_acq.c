#include "quip_config.h"
#include "spink.h"
#include "quip_prot.h"

#ifdef HAVE_LIBSPINNAKER

#define alloc_cam_buffers(skc_p, n) _alloc_cam_buffers(QSP_ARG  skc_p, n)

static void _alloc_cam_buffers(QSP_ARG_DECL  Spink_Cam *skc_p, int n)
{
	spinImage hImage=NULL;
	int i;

	for(i=0;i<n;i++){
		if( next_spink_image(&hImage,skc_p) < 0 ){
			sprintf(ERROR_STRING,"alloc_cam_buffers:  Error getting image %d",i);
			warn(ERROR_STRING);
			skc_p->skc_img_tbl[i] = NULL;
		} else {
			skc_p->skc_img_tbl[i] = hImage;
		}
	}
	skc_p->skc_n_buffers = n;
}

#define release_cam_buffers(skc_p) _release_cam_buffers(QSP_ARG  skc_p)

static void _release_cam_buffers(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	int i;

	for(i=0;i<skc_p->skc_n_buffers;i++){
		if( release_spink_image(skc_p->skc_img_tbl[i]) < 0 ){
			sprintf(ERROR_STRING,"release_cam_buffers:  Error releasing image %d",i);
			warn(ERROR_STRING);
		}
		skc_p->skc_img_tbl[i] = NULL;
	}
	skc_p->skc_n_buffers = 0;
}

void _set_n_spink_buffers(QSP_ARG_DECL  Spink_Cam *skc_p, int n)
{
	assert(skc_p!=NULL);
	assert(n>=MIN_N_BUFFERS&&n<=MAX_N_BUFFERS);

	if( skc_p->skc_n_buffers > 0 )
		release_cam_buffers(skc_p);

	alloc_cam_buffers(skc_p,n);
}

//
// Retrieve next received image
//
// *** NOTES ***
// Capturing an image houses images on the camera buffer. Trying to
// capture an image that does not exist will hang the camera.
//
// *** LATER ***
// Once an image from the buffer is saved and/or no longer needed, the
// image must be released in orer to keep the buffer from filling up.
//


int _next_spink_image(QSP_ARG_DECL  spinImage *img_p, Spink_Cam *skc_p)
{
	spinCamera hCam;
	bool8_t isIncomplete = False;

	insure_current_camera(skc_p);
	hCam = skc_p->skc_current_handle;

	if( get_next_image(hCam,img_p) < 0 ) return -1;

	//
	// Ensure image completion
	//
	// *** NOTES ***
	// Images can easily be checked for completion. This should be done
	// whenever a complete image is expected or required. Further, check
	// image status for a little more insight into why an image is
	// incomplete.
	//
	if( image_is_incomplete(*img_p,&isIncomplete) < 0 ){
		// non-fatal error
		if( release_spink_image(*img_p) < 0 ) return -1;
		return 0;
	}

	// Check image for completion
	if (isIncomplete) {
		spinImageStatus imageStatus = IMAGE_NO_ERROR;
		if( get_image_status(*img_p,&imageStatus) < 0 ){
			// non-fatal error
			if( release_spink_image(*img_p) < 0 ) return -1;
			return 0;
		}
		printf("Image incomplete with image status %d...\n", imageStatus);
		if( release_spink_image(*img_p) < 0 ) return -1;
		return 0;
	}

	return 0;
}


//
// Print image information; height and width recorded in pixels
//
// *** NOTES ***
// Images have quite a bit of available metadata including things such
// as CRC, image status, and offset values, to name a few.
//

#define print_image_info(hImg) _print_image_info(QSP_ARG  hImg)

static int _print_image_info(QSP_ARG_DECL  spinImage hImg)
{
	size_t width = 0;
	size_t height = 0;

	// Retrieve image width
	if( get_image_width(hImg,&width) < 0 ) return -1;
	if( get_image_height(hImg,&height) < 0 ) return -1;
	printf("width = %u, ", (unsigned int)width);
	printf("height = %u\n", (unsigned int)height);
	return 0;
}


//
// Begin acquiring images
//
// *** NOTES ***
// What happens when the camera begins acquiring images depends on the
// acquisition mode. Single frame captures only a single image, multi
// frame catures a set number of images, and continuous captures a
// continuous stream of images. Because the example calls for the retrieval
// of 10 images, continuous mode has been set.
//
// *** LATER ***
// Image acquisition must be ended when no more images are needed.
//

int _spink_start_capture(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	spinCamera hCam;
	
	insure_current_camera(skc_p);
	hCam = skc_p->skc_current_handle;

	if( begin_acquisition(hCam) < 0 ) return -1;

	return 0;
}

//
// End acquisition
//
// *** NOTES ***
// Ending acquisition appropriately helps ensure that devices clean up
// properly and do not need to be power-cycled to maintain integrity.
//

int _spink_stop_capture(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	spinCamera hCam;
	
	insure_current_camera(skc_p);
	hCam = skc_p->skc_current_handle;

	if( end_acquisition(hCam) < 0 ) return -1;

	return 0;
}

//
// Set acquisition mode to continuous
//
// *** NOTES ***
// Because the example acquires and saves 10 images, setting acquisition
// mode to continuous lets the example finish. If set to single frame
// or multiframe (at a lower number of images), the example would just
// hang. This would happen because the example has been written to acquire
// 10 images while the camera would have been programmed to retrieve
// less than that.
//
// Setting the value of an enumeration node is slightly more complicated
// than other node types, and especially so in C. It can roughly be broken
// down into four steps: first, the enumeration node is retrieved from the
// nodemap; second, the entry node is retrieved from the enumeration node;
// third, an integer is retrieved from the entry node; and finally, the
// integer is set as the new value of the enumeration node.
//
// It is important to note that there are two sets of functions that might
// produce erroneous results if they were to be mixed up. The first two
// functions, spinEnumerationSetIntValue() and
// spinEnumerationEntryGetIntValue(), use the integer values stored on each
// individual cameras. The second two, spinEnumerationSetEnumValue() and
// spinEnumerationEntryGetEnumValue(), use enum values defined in the
// Spinnaker library. The int and enum values will most likely be
// different from another.
//

#define set_acquisition_continuous(skc_p) _set_acquisition_continuous(QSP_ARG  skc_p)

static int _set_acquisition_continuous(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	spinNodeHandle hAcquisitionMode = NULL;
	spinNodeHandle hAcquisitionModeContinuous = NULL;
	int64_t acquisitionModeContinuous = 0;
	spinNodeMapHandle hMap;

	if( get_node_map_handle(&hMap,skc_p->skc_cam_map,"set_acquisition_cont") < 0 )
		return -1;

	if( fetch_spink_node(hMap, "AcquisitionMode", &hAcquisitionMode) < 0 ){
		warn("set_acquisition_continuous:  error getting AcquisitionMode node!?");
		return -1;
	}

	if( ! spink_node_is_available(hAcquisitionMode) ){
		warn("set_acquisition_continuous:  AcquisitionMode node is not available!?");
		return -1;
	}
	if( ! spink_node_is_readable(hAcquisitionMode) ){
		warn("set_acquisition_continuous:  AcquisitionMode node is not readable!?");
		return -1;
	}

	if( get_enum_entry_by_name(hAcquisitionMode,"Continuous", &hAcquisitionModeContinuous) < 0 ){
		warn("set_acquisition_continuous:  error getting enumeration entry by name!?");
		return -1;
	}

	if( ! spink_node_is_available(hAcquisitionModeContinuous) ){
		warn("set_acquisition_continuous:  AcquisitionModeContinuous node is not available!?");
		return -1;
	}
	if( ! spink_node_is_readable(hAcquisitionModeContinuous) ){
		warn("set_acquisition_continuous:  AcquisitionModeContinuous node is not readable!?");
		return -1;
	}

	if( get_enum_int_val(hAcquisitionModeContinuous,&acquisitionModeContinuous) < 0 ){
		warn("set_acquisition_continuous:  error getting enumeration int val!?");
		return -1;
	}

	if( ! spink_node_is_writable(hAcquisitionMode) ){
		warn("set_acquisition_continuous:  AcquisitionMode node is not writable!?");
		return -1;
	}

	// Set integer as new value of enumeration node
	if( set_enum_int_val(hAcquisitionMode,acquisitionModeContinuous) < 0 ) {
		warn("set_acquisition_continuous:  error setting enumeration int val!?");
		return -1;
	}

	printf("Acquisition mode set to continuous...\n");
	return 0;
}

#define init_one_frame(index, data ) _init_one_frame(QSP_ARG  index, data )

static Data_Obj * _init_one_frame(QSP_ARG_DECL  int index, void *data )
{
	Data_Obj *dp;
	char fname[32];
	Dimension_Set ds1;

	sprintf(fname,"frame%d",index);
	//assign_var("newest",fname+5);

	dp = dobj_of(fname);
	if( dp == NULL ){
fprintf(stderr,"init_one_frame:  creating %s\n",fname);
		SET_DS_SEQS(&ds1,1);
		SET_DS_FRAMES(&ds1,1);
		SET_DS_ROWS(&ds1,1024);	// BUG - get real values!!!
		SET_DS_COLS(&ds1,1280);
		SET_DS_COMPS(&ds1,1);
		dp = _make_dp(QSP_ARG  fname,&ds1,PREC_FOR_CODE(PREC_UBY));
		assert( dp != NULL );

		SET_OBJ_DATA_PTR( dp, data);
		//fcp->fc_frm_dp_tbl[index] = dp;
	} else {
		sprintf(ERROR_STRING,"init_one_frame:  object %s already exists!?",
			fname);
		warn(ERROR_STRING);
	}
	return dp;
} // end init_one_frame

int _spink_test_acq(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	spinImage hResultImage = NULL;
	const unsigned int k_numImages = 10;
	unsigned int imageCnt = 0;
	spinImage hConvertedImage = NULL;

printf("spink_test_acq BEGIN\n");
printf("spink_test_acq will call set_acquisition_continuous\n");
	if( set_acquisition_continuous(skc_p) < 0 ){
		warn("spink_test_acq:  unable to set continuous acquisition!?");
		return -1;
	}
printf("spink_test_acq will call spink_start_capture\n");
	if( spink_start_capture(skc_p) < 0 ){
		warn("spink_test_acq:  unable to start capture!?");
		return -1;
	}

	for (imageCnt = 0; imageCnt < k_numImages; imageCnt++) {
		void *data_ptr;
		Data_Obj *dp;

		if( next_spink_image(&hResultImage,skc_p) < 0 ){
			warn("spink_test_acq:  unable to get next image!?");
			return -1;	// cleanup???
		}
		printf("Grabbed image %d, ", imageCnt);
		if( print_image_info(hResultImage) < 0 ) return -1;
		if( create_empty_image(&hConvertedImage) < 0 ) return -1;
		if( convert_spink_image(hResultImage,PixelFormat_Mono8,hConvertedImage) < 0 ) return -1;
// get_image_data		ImageGetData
// SPINNAKERC_API spinImageGetData(spinImage hImage, void** ppData);
		get_image_data(hConvertedImage,&data_ptr);
		dp = init_one_frame(imageCnt,data_ptr);
fprintf(stderr,"Created %s\n",OBJ_NAME(dp));

		//if( destroy_spink_image(hConvertedImage) < 0 ) return -1;
		if( release_spink_image(hResultImage) < 0 ) return -1;
	}

	if( spink_stop_capture(skc_p) < 0 ) return -1;
printf("spink_test_acq DONE\n");
	return 0;
}

void onImageEvent( spinImage hImage, void *user_data_p )
{
fprintf(stderr,"image event!\n");
}

static spinImageEvent ev1;
typedef struct my_event_info {
	int64_t a_value;
} Image_Event_Info;

static Image_Event_Info inf1;

void setup_events(Spink_Cam *skc_p)
{

	if( create_image_event(&ev1,onImageEvent,(void *)(&inf1) ) < 0 ) return;
	if( register_cam_image_event(hCam, ev1) < 0 ) return;
}

#endif // HAVE_LIBSPINNAKER

