#include "quip_config.h"
#include "spink.h"
#include "quip_prot.h"

#ifdef HAVE_LIBSPINNAKER


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

	if( get_spink_cam_from_list(&hCam,hCameraList,skc_p->skc_sys_idx) < 0 )
		return -1;

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
	if( release_spink_cam(hCam) < 0 )
		return -1;

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
	
	//hCam = skc_p->skc_handle;
	if( get_spink_cam_from_list(&hCam,hCameraList,skc_p->skc_sys_idx) < 0 )
		return -1;

	if( begin_acquisition(hCam) < 0 ) return -1;

	if( release_spink_cam(hCam) < 0 )
		return -1;

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
	
	//hCam = skc_p->skc_handle;
	if( get_spink_cam_from_list(&hCam,hCameraList,skc_p->skc_sys_idx) < 0 )
		return -1;

	if( end_acquisition(hCam) < 0 ) return -1;

	if( release_spink_cam(hCam) < 0 )
		return -1;

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

	if( get_enumeration_entry_by_name(hAcquisitionMode,"Continuous", &hAcquisitionModeContinuous) < 0 ){
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

	if( get_enumeration_int_val(hAcquisitionModeContinuous,&acquisitionModeContinuous) < 0 ){
		warn("set_acquisition_continuous:  error getting enumeration int val!?");
		return -1;
	}

	if( ! spink_node_is_writeable(hAcquisitionMode) ){
		warn("set_acquisition_continuous:  AcquisitionMode node is not writeable!?");
		return -1;
	}

	// Set integer as new value of enumeration node
	if( set_enumeration_int_val(hAcquisitionMode,acquisitionModeContinuous) < 0 ) {
		warn("set_acquisition_continuous:  error setting enumeration int val!?");
		return -1;
	}

	printf("Acquisition mode set to continuous...\n");
	return 0;
}

int _spink_test_acq(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	spinImage hResultImage = NULL;
	const unsigned int k_numImages = 10;
	unsigned int imageCnt = 0;
	spinImage hConvertedImage = NULL;

printf("spink_test_acq BEGIN\n");
	if( set_acquisition_continuous(skc_p) < 0 ){
		warn("spink_test_acq:  unable to set continuous acquisition!?");
		return -1;
	}
	if( spink_start_capture(skc_p) < 0 ){
		warn("spink_test_acq:  unable to start capture!?");
		return -1;
	}

	for (imageCnt = 0; imageCnt < k_numImages; imageCnt++) {
		if( next_spink_image(&hResultImage,skc_p) < 0 ){
			warn("spink_test_acq:  unable to get next image!?");
			return -1;	// cleanup???
		}
		printf("Grabbed image %d, ", imageCnt);
		if( print_image_info(hResultImage) < 0 ) return -1;
		if( create_empty_image(&hConvertedImage) < 0 ) return -1;
		if( convert_spink_image(hResultImage,PixelFormat_Mono8,hConvertedImage) < 0 ) return -1;
		if( destroy_spink_image(hConvertedImage) < 0 ) return -1;
		if( release_spink_image(hResultImage) < 0 ) return -1;
	}

	if( spink_stop_capture(skc_p) < 0 ) return -1;
printf("spink_test_acq DONE\n");
	return 0;
}

#endif // HAVE_LIBSPINNAKER

