#include "quip_config.h"
#include "spink.h"
#include "quip_prot.h"

#ifdef HAVE_LIBSPINNAKER

// The use of image events seems to be incompatible with grab/release!?
// We don't need to grab, because the event handler already gives us the image.
// Presumably it is released when the handler exits...  This may be incompatible with
// how we have done recording, where the disk writer releases the image when writing
// is complete...

static spinImageEvent ev1;

void _enable_image_events(QSP_ARG_DECL  Spink_Cam *skc_p, void (*func)(spinImage,void *))
{
	Image_Event_Info *inf_p;

	if( IS_EVENTFUL(skc_p) ) return;

	insure_current_camera(skc_p);
	assert( skc_p->skc_current_handle != NULL );
	inf_p = getbuf(sizeof(*inf_p));		// when to release???  MEMORY LEAK BUG!
#ifdef THREAD_SAFE_QUERY
	inf_p->ei_qsp = THIS_QSP;
#endif // THREAD_SAFE_QUERY
	inf_p->ei_skc_p = skc_p;
	if( create_image_event(&ev1,func,(void *)(inf_p) ) < 0 ) return;
	if( register_cam_image_event(skc_p->skc_current_handle, ev1) < 0 ) return;

	skc_p->skc_flags |= SPINK_CAM_EVENTS_READY;
	//assign_var("image_ready","0");
}

#ifdef NOT_USED
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
#endif // NOT_USED

#define report_image_status(status) _report_image_status(QSP_ARG  status)

static int _report_image_status(QSP_ARG_DECL  spinImageStatus status)
{
	size_t len;
	char buf[LLEN];

	if( get_image_status_description(status,NULL,&len) < 0 ){
		sprintf(ERROR_STRING,"report_image_status:  Error getting image status description length!?");
		warn(ERROR_STRING);
		return -1;
	}
	if( len >= LLEN ){
		sprintf(ERROR_STRING,"report_image_status:  status description needs %ld chars, but only %d provided!?",
			len,LLEN);
		warn(ERROR_STRING);
		return -1;
	}

	len=LLEN;
	if( get_image_status_description(status,buf,&len) < 0 ){
		sprintf(ERROR_STRING,"report_image_status:  Error getting image status description!?");
		warn(ERROR_STRING);
		return -1;
	}
//fprintf(stderr,"image_status: %s\n",buf);
	return 0;
}

#define set_script_var(varname, value) _set_script_var(QSP_ARG  varname, value)

static void _set_script_var(QSP_ARG_DECL  const char *varname, int value)
{
	char buf[32];
	sprintf(buf,"%d",value);
	assign_reserved_var(varname,buf);
}

Data_Obj * _grab_spink_cam_frame(QSP_ARG_DECL  Spink_Cam * skc_p )
{
	spinImage hImage;
	spinImageStatus status;
	void *data_ptr;
	int index;
	Data_Obj *dp;

	if( ! IS_CAPTURING(skc_p) ){
		sprintf(ERROR_STRING,"grab_spink_cam_frame:  %s is not capturing!?",skc_p->skc_name);
		warn(ERROR_STRING);
		return NULL;
	}

	if( next_spink_image(&hImage,skc_p) < 0 ){
		sprintf(ERROR_STRING,"grab_spink_cam_frame:  Error getting image!?");
		warn(ERROR_STRING);
		return NULL;
	}

	if( get_image_status(hImage,&status) < 0 ){
		sprintf(ERROR_STRING,"grab_spink_cam_frame:  Error getting image status!?");
		warn(ERROR_STRING);
		return NULL;
	}
	if( status != IMAGE_NO_ERROR ){
		if( report_image_status(status) < 0 ) return NULL;
	}

	get_image_data(hImage,&data_ptr);

	index = skc_p->skc_newest;	// temporary - needs to be changed
	index ++;
	if( index >= skc_p->skc_n_buffers ) index=0;
	assert(index>=0&&index<skc_p->skc_n_buffers);
	dp = skc_p->skc_frm_dp_tbl[index];
	point_obj_to_ext_data(dp,data_ptr);
	SET_OBJ_EXTRA(dp,hImage);
	skc_p->skc_newest = index;
	if( skc_p->skc_oldest < 0 ) skc_p->skc_oldest = index;
	set_script_var("newest",skc_p->skc_newest);	// BUG need cam-specific var?
	set_script_var("oldest",skc_p->skc_oldest);	// BUG need cam-specific var?

	return( skc_p->skc_frm_dp_tbl[index] );
}

void _release_oldest_spink_frame(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	Data_Obj *dp;
	int index;
	spinImage hImage;

	if( ! IS_CAPTURING(skc_p) ){
		sprintf(ERROR_STRING,"release_oldest_spink_frame:  %s is not capturing!?",
			skc_p->skc_name);
		warn(ERROR_STRING);
		return;
	}

	if( skc_p->skc_oldest < 0 ){
		sprintf(ERROR_STRING,"No frames have been grabbed by %s, can't release!?",
			skc_p->skc_name);
		warn(ERROR_STRING);
		return;
	}
	index = skc_p->skc_oldest;
	assert(index>=0&&index<skc_p->skc_n_buffers);
	dp = skc_p->skc_frm_dp_tbl[index];
	assert(dp!=NULL);
	hImage = OBJ_EXTRA(dp);
	if( release_spink_image(hImage) < 0 ){
		sprintf(ERROR_STRING,"release_oldest_spink_frame %s:  Error releasing image %d",skc_p->skc_name,index);
		warn(ERROR_STRING);
	}
	point_obj_to_ext_data(dp,NULL);
	SET_OBJ_EXTRA(dp,NULL);
	if( skc_p->skc_newest == index ){
//fprintf(stderr,"release_oldest_spink_frame:  last frame was released\n");
		skc_p->skc_newest = -1;
		skc_p->skc_oldest = -1;
	} else {
		index++;
		if( index >= skc_p->skc_n_buffers ) index = 0;
		skc_p->skc_oldest = index;
//fprintf(stderr,"release_oldest_spink_frame:  oldest frame is now %d\n",index);
	}
}

#define init_frame_by_index(skc_p, idx) _init_frame_by_index(QSP_ARG  skc_p, idx)

static Data_Obj * _init_frame_by_index(QSP_ARG_DECL  Spink_Cam *skc_p, int idx)
{
	Data_Obj *dp;
	char fname[64];
	Dimension_Set ds1;

	sprintf(fname,"cam%d.frame%d",skc_p->skc_sys_idx,idx);
	//assign_var("newest",fname+5);

	dp = dobj_of(fname);
	assert(dp==NULL);
	SET_DS_SEQS(&ds1,1);
	SET_DS_FRAMES(&ds1,1);
	SET_DS_ROWS(&ds1,skc_p->skc_rows);
	SET_DS_COLS(&ds1,skc_p->skc_cols);
	SET_DS_COMPS(&ds1,1);
	dp = _make_dp(QSP_ARG  fname,&ds1,PREC_FOR_CODE(PREC_UBY));
	assert( dp != NULL );
	return dp;
}

void _set_n_spink_buffers(QSP_ARG_DECL  Spink_Cam *skc_p, int n)
{
	int i;
	size_t sz;

	assert(skc_p!=NULL);
	assert(n>=MIN_N_BUFFERS&&n<=MAX_N_BUFFERS);


	if( skc_p->skc_n_buffers > 0 ){
		assert(skc_p->skc_frm_dp_tbl!=NULL);
		// BUG -  We need to release resources from the dobj structs!!!
		// MEMORY LEAK!
		givbuf(skc_p->skc_frm_dp_tbl);
	}

	skc_p->skc_n_buffers = n;

	sz = n * sizeof(Data_Obj *);
	skc_p->skc_frm_dp_tbl = getbuf(sz);

	for(i=0;i<n;i++){
		Data_Obj *dp;
		dp = init_frame_by_index(skc_p,i);
		skc_p->skc_frm_dp_tbl[i] = dp;
	}
	//alloc_cam_buffers(skc_p,n);
}

void _show_n_buffers(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	sprintf(MSG_STR,"%s:  %d buffers",skc_p->skc_name,skc_p->skc_n_buffers);
	prt_msg(MSG_STR);
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

// triggering events seems to break next/release...  it might be a good way to go
// for streaming, however.

int _spink_start_capture(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	spinCamera hCam;
	
	if( IS_CAPTURING(skc_p) ){
		sprintf(ERROR_STRING,"spink_stop_capture:  %s is already capturing!?",
			skc_p->skc_name);
		warn(ERROR_STRING);
		return 0;
	}

	insure_current_camera(skc_p);
	skc_p->skc_newest = -1;
	skc_p->skc_oldest = -1;
	hCam = skc_p->skc_current_handle;

//	setup_events(skc_p);	// make this optional???
				// should cleanup events when stopping capture?

	if( begin_acquisition(hCam) < 0 ) return -1;

	skc_p->skc_flags |= SPINK_CAM_CAPTURING;

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
	
	if( ! IS_CAPTURING(skc_p) ){
		sprintf(ERROR_STRING,"spink_stop_capture:  %s is not capturing!?",
			skc_p->skc_name);
		warn(ERROR_STRING);
		return 0;
	}

	insure_current_camera(skc_p);
	hCam = skc_p->skc_current_handle;

	if( end_acquisition(hCam) < 0 ) return -1;

	skc_p->skc_flags &= ~SPINK_CAM_CAPTURING;

	return 0;
}

void _set_camera_node(QSP_ARG_DECL  Spink_Cam *skc_p, const char *node_name, const char *entry_name)
{
	spinNodeMapHandle hMap;
	spinNodeHandle hNode = NULL;
	spinNodeHandle hEntry = NULL;
	int64_t int_val;

	if( get_node_map_handle(&hMap,skc_p->skc_cam_map,"set_acquisition_continuous") < 0 )
		return;

	if( fetch_spink_node(hMap, node_name, &hNode) < 0 ){
		sprintf(ERROR_STRING,"set_camera_node:  error getting %s node!?",node_name);
		warn(ERROR_STRING);
		return;
	}

	if( ! spink_node_is_available(hNode) ){
		sprintf(ERROR_STRING,"set_camera_node:  %s node is not available!?",node_name);
		warn(ERROR_STRING);
		return;
	}
	if( ! spink_node_is_readable(hNode) ){
		sprintf(ERROR_STRING,"set_camera_node:  %s node is not readable!?",node_name);
		warn(ERROR_STRING);
		return;
	}

	if( get_enum_entry_by_name(hNode,entry_name, &hEntry) < 0 ){
		warn("set_acquisition_continuous:  error getting enumeration entry by name!?");
		return;
	}

	if( ! spink_node_is_available(hEntry) ){
		sprintf(ERROR_STRING,"set_camera_node:  %s node is not available!?",entry_name);
		warn(ERROR_STRING);
		return;
	}
	if( ! spink_node_is_readable(hEntry) ){
		sprintf(ERROR_STRING,"set_camera_node:  %s node is not readable!?",entry_name);
		warn(ERROR_STRING);
		return;
	}

	if( get_enum_int_val(hEntry,&int_val) < 0 ){
		sprintf(ERROR_STRING,"set_camera_node:  error getting enumeration int val for %s!?",entry_name);
		warn(ERROR_STRING);
		return;
	}

	if( ! spink_node_is_writable(hNode) ){
		sprintf(ERROR_STRING,"set_camera_node:  %s node is not readable!?",node_name);
		warn(ERROR_STRING);
		return;
	}

	// Set integer as new value of enumeration node
	if( set_enum_int_val(hNode,int_val) < 0 ) {
		sprintf(ERROR_STRING,"set_camera_node:  error setting %s node to %s!?",node_name,entry_name);
		warn(ERROR_STRING);
		return;
	}
}

int _set_acquisition_continuous(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	set_camera_node(skc_p,"AcquisitionMode","Continuous");
//	printf("Acquisition mode set to continuous...\n");
	return 0;
}

#endif // HAVE_LIBSPINNAKER

