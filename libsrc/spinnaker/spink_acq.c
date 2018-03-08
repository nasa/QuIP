#include "quip_config.h"
#include "spink.h"

int _get_enumeration_entry_by_name(QSP_ARG_DECL  spinNodeHandle hEnum, const char *tag, spinNodeHandle *hdl_p)
{
	spinError err;

	err = spinEnumerationGetEntryByName(hEnum, tag, hdl_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinEnumerationGetEntryByName");
		return -1;
	}
	return 0;
}

int _get_enumeration_int_val(QSP_ARG_DECL  spinNodeHandle hNode, int64_t *int_ptr)
{
	spinError err;

	err = spinEnumerationEntryGetIntValue(hNode, int_ptr);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinEnumerationEntryGetIntValue");
		return -1;
	}
}

int _set_enumeration_int_val(QSP_ARG_DECL  spinNodeHandle hNode, int64_t v)
{
	spinError err;

	err = spinEnumerationSetIntValue(hNode, v);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinEnumerationSetIntValue");
		return -1;
	}
	return 0;
}

//
// Release image from camera
//
// *** NOTES ***
// Images retrieved directly from the camera (i.e. non-converted
// images) need to be released in order to keep from filling the
// buffer.
//

int _release_spink_image(QSP_ARG_DECL  spinImage hImage)
{
	spinError err;

	err = spinImageRelease(hImage);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinImageRelease");
		return -1;
	}
	return 0;
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
	spinError err;

	err = spinCameraGetNextImage(skc_p->skc_handle, img_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraGetNextImage");
		return -1;
	}

	//
	// Ensure image completion
	//
	// *** NOTES ***
	// Images can easily be checked for completion. This should be done
	// whenever a complete image is expected or required. Further, check
	// image status for a little more insight into why an image is
	// incomplete.
	//
	bool8_t isIncomplete = False;

	err = spinImageIsIncomplete(*img_p, &isIncomplete);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinImageIsIncomplete");
		// non-fatal error
		if( release_spink_image(*img_p) < 0 ) return -1;
		return 0;
	}
	// Check image for completion
	if (isIncomplete) {
		spinImageStatus imageStatus = IMAGE_NO_ERROR;

		err = spinImageGetStatus(*img_p, &imageStatus);
		if (err != SPINNAKER_ERR_SUCCESS) {
			report_spink_error(err,"spinImageGetStatus");
			// non-fatal error
			if( release_spink_image(*img_p) < 0 ) return -1;
			return 0;
		}
		printf("Image incomplete with image status %d...\n", imageStatus);
		if( release_spink_image(*img_p) < 0 ) return -1;
		return 0;
	}
}

//
// Print image information; height and width recorded in pixels
//
// *** NOTES ***
// Images have quite a bit of available metadata including things such
// as CRC, image status, and offset values, to name a few.
//

static int _print_image_info(QSP_ARG_DECL  spinImage hImg)
{
	size_t width = 0;
	size_t height = 0;
	spinError err;


	// Retrieve image width
	err = spinImageGetWidth(hImg, &width);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinImageGetWidth");
		return -1;
	} else {
		printf("width = %u, ", (unsigned int)width);
	}

	// Retrieve image height
	err = spinImageGetHeight(hImg, &height);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinImageGetHeight");
		return -1;
	} else {
		printf("height = %u\n", (unsigned int)height);
	}
	return 0;
}


int _create_empty_image(QSP_ARG_DECL  spinImage *hImg_p)
{
	spinError err;

	err = spinImageCreateEmpty(hImg_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinImageCreateEmpty");
		return -1;
	}
	return 0;
}

//
// Convert image to mono 8
//
// *** NOTES ***
// Images not gotten from a camera directly must be created and
// destroyed. This includes any image copies, conversions, or
// otherwise. Basically, if the image was gotten, it should be
// released, if it was created, it needs to be destroyed.
//
// Images can be converted between pixel formats by using the
// appropriate enumeration value. Unlike the original image, the
// converted one does not need to be released as it does not affect the
// camera buffer.
//
// Optionally, the color processing algorithm can also be set using
// the alternate spinImageConvertEx() function.
//
// *** LATER ***
// The converted image was created, so it must be destroyed to avoid
// memory leaks.
//

// BUG mode should be an arg, but not sure what type???

int _convert_spink_image(QSP_ARG_DECL  spinImage hDestImg, spinImage hSrcImg )
{
	spinError err;

	err = spinImageConvert(hSrcImg, PixelFormat_Mono8, hDestImg);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinImageConvert");
		return -1;
	}
	return 0;
}

//
// Destroy converted image
//
// *** NOTES ***
// Images that are created must be destroyed in order to avoid memory
// leaks.
//

int _destroy_spink_image(QSP_ARG_DECL  spinImage hImg)
{
	spinError err;

	err = spinImageDestroy(hImg);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinImageDestroy");
		return -1;
	}
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
	spinError err;
	
	hCam = skc_p->skc_handle;
	err = spinCameraBeginAcquisition(hCam);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraBeginAcquisition");
		return -1;
	}
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
	spinError err;
	
	hCam = skc_p->skc_handle;
	err = spinCameraEndAcquisition(hCam);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraEndAcquisition");
		return -1;
	}
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

	if( get_spink_node(skc_p->skc_genicam_node_map, "AcquisitionMode", &hAcquisitionMode) < 0 )
		return -1;

	if( ! spink_node_is_available(hAcquisitionMode) ) return -1;
	if( ! spink_node_is_readable(hAcquisitionMode) ) return -1;

	if( get_enumeration_entry_by_name(hAcquisitionMode,"Continuous", &hAcquisitionModeContinuous) < 0 )
		return -1;

	if( ! spink_node_is_available(hAcquisitionModeContinuous) ) return -1;
	if( ! spink_node_is_readable(hAcquisitionModeContinuous) ) return -1;

	if( get_enumeration_int_val(hAcquisitionModeContinuous,&acquisitionModeContinuous) < 0 )
		return -1;

	if( ! spink_node_is_writable(hAcquisitionMode) ) return -1;

	// Set integer as new value of enumeration node
	if( set_enumeration_int_val(hAcquisitionMode,acquisitionModeContinuous) < 0 ) return -1;

	printf("Acquisition mode set to continuous...\n");
}

int _spink_test_acq(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	spinImage hResultImage = NULL;
	const unsigned int k_numImages = 10;
	unsigned int imageCnt = 0;
	spinImage hConvertedImage = NULL;

	if( set_acquisition_continuous(skc_p) < 0 ) return -1;
	if( spink_start_capture(skc_p) < 0 ) return -1;

	for (imageCnt = 0; imageCnt < k_numImages; imageCnt++) {
		if( next_spink_image(&hResultImage,skc_p) < 0 )
			return -1;	// cleanup???
		printf("Grabbed image %d, ", imageCnt);
		if( print_image_info(hResultImage) < 0 ) return -1;
		if( create_empty_image(&hConvertedImage) < 0 ) return -1;
		if( convert_spink_image(hConvertedImage,hResultImage) < 0 ) return -1;
		if( destroy_spink_image(hConvertedImage) < 0 ) return -1;
		if( release_spink_image(hResultImage) < 0 ) return -1;
	}

	if( spink_stop_capture(skc_p) < 0 ) return -1;
}

