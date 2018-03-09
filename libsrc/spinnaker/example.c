// To build:
//
//	gcc -I/usr/include/spinnaker/spinc -c example.c
//	gcc -o ./test example.o -lSpinnaker_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "SpinnakerC.h"
#define MAX_BUFF_LEN 256
#define MAX_NODE_CHARS 35

// globals
static spinSystem hSystem = NULL;
static spinCameraList hCameraList = NULL;
static size_t numCameras = 0;

static void report_spink_error(spinError error, const char *whence )
{
	const char *msg;

	switch(error){
		case SPINNAKER_ERR_SUCCESS:
			msg = "Success"; break;
		case SPINNAKER_ERR_ERROR:
			msg = "Error"; break;
		case SPINNAKER_ERR_NOT_INITIALIZED:
			msg = "Not initialized"; break;
		case SPINNAKER_ERR_NOT_IMPLEMENTED:
			msg = "Not implemented"; break;
		case SPINNAKER_ERR_RESOURCE_IN_USE:
			msg = "Resource in use"; break;
		case SPINNAKER_ERR_ACCESS_DENIED:
			msg = "Access denied"; break;
		case SPINNAKER_ERR_INVALID_HANDLE:
			msg = "Invalid handle"; break;
		case SPINNAKER_ERR_INVALID_ID:
			msg = "Invalid ID"; break;
		case SPINNAKER_ERR_NO_DATA:
			msg = "No data"; break;
		case SPINNAKER_ERR_INVALID_PARAMETER:
			msg = "Invalid parameter"; break;
		case SPINNAKER_ERR_IO:
			msg = "I/O error"; break;
		case SPINNAKER_ERR_TIMEOUT:
			msg = "Timeout"; break;
		case SPINNAKER_ERR_ABORT:
			msg = "Abort"; break;
		case SPINNAKER_ERR_INVALID_BUFFER:
			msg = "Invalid buffer"; break;
		case SPINNAKER_ERR_NOT_AVAILABLE:
			msg = "Not available"; break;
		case SPINNAKER_ERR_INVALID_ADDRESS:
			msg = "Invalid address"; break;
		case SPINNAKER_ERR_BUFFER_TOO_SMALL:
			msg = "Buffer too small"; break;
		case SPINNAKER_ERR_INVALID_INDEX:
			msg = "Invalid index"; break;
		case SPINNAKER_ERR_PARSING_CHUNK_DATA:
			msg = "Chunk data parsing error"; break;
		case SPINNAKER_ERR_INVALID_VALUE:
			msg = "Invalid value"; break;
		case SPINNAKER_ERR_RESOURCE_EXHAUSTED:
			msg = "Resource exhausted"; break;
		case SPINNAKER_ERR_OUT_OF_MEMORY:
			msg = "Out of memory"; break;
		case SPINNAKER_ERR_BUSY:
			msg = "Busy"; break;

		case GENICAM_ERR_INVALID_ARGUMENT:
			msg = "genicam invalid argument"; break;
		case GENICAM_ERR_OUT_OF_RANGE:
			msg = "genicam range error"; break;
		case GENICAM_ERR_PROPERTY:
			msg = "genicam property error"; break;
		case GENICAM_ERR_RUN_TIME:
			msg = "genicam run time error"; break;
		case GENICAM_ERR_LOGICAL:
			msg = "genicam logical error"; break;
		case GENICAM_ERR_ACCESS:
			msg = "genicam access error"; break;
		case GENICAM_ERR_TIMEOUT:
			msg = "genicam timeout error"; break;
		case GENICAM_ERR_DYNAMIC_CAST:
			msg = "genicam dynamic cast error"; break;
		case GENICAM_ERR_GENERIC:
			msg = "genicam generic error"; break;
		case GENICAM_ERR_BAD_ALLOCATION:
			msg = "genicam bad allocation"; break;

		case SPINNAKER_ERR_IM_CONVERT:
			msg = "image conversion error"; break;
		case SPINNAKER_ERR_IM_COPY:
			msg = "image copy error"; break;
		case SPINNAKER_ERR_IM_MALLOC:
			msg = "image malloc error"; break;
		case SPINNAKER_ERR_IM_NOT_SUPPORTED:
			msg = "image operation not supported"; break;
		case SPINNAKER_ERR_IM_HISTOGRAM_RANGE:
			msg = "image histogram range error"; break;
		case SPINNAKER_ERR_IM_HISTOGRAM_MEAN:
			msg = "image histogram mean error"; break;
		case SPINNAKER_ERR_IM_MIN_MAX:
			msg = "image min/max error"; break;
		case SPINNAKER_ERR_IM_COLOR_CONVERSION:
			msg = "image color conversion error"; break;

//		case SPINNAKER_ERR_CUSTOM_ID = -10000

		default:
			fprintf(stderr,
		"report_spink_error (%s):  unhandled error code %d!?\n",
				whence,error);
			msg = "unhandled error code";
			break;
	}
	fprintf(stderr,"%s:  %s\n",whence,msg);
}

static int get_spink_system(spinSystem *hSystem_p)
{
	spinError err;

	err = spinSystemGetInstance(hSystem_p);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		report_spink_error(err,"spinSystemGetInstance");
		return -1;
	}
	return 0;
}

static int get_spink_cameras(spinSystem hSystem, spinCameraList *hCameraList_p, size_t *num_p )
{
	spinError err;


	// Create empty camera list
	err = spinCameraListCreateEmpty(hCameraList_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraListCreateEmpty");
		return -1;
	}

	// Retrieve cameras from system
	err = spinSystemGetCameras(hSystem, *hCameraList_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinSystemGetCameras");
		return -1;
	}

	// Retrieve number of cameras
	err = spinCameraListGetSize(*hCameraList_p, num_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraListGetSize");
		return -1;
	}
	return 0;
}

static int get_spink_cam_from_list(spinCamera *hCam_p, spinCameraList hCameraList, int idx )
{
	spinError err;

	err = spinCameraListGet(hCameraList, idx, hCam_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraListGet");
		return -1;
	}
	return 0;
}

static int get_spink_transport_level_map(spinNodeMapHandle *mapHdl_p, spinCamera hCam )
{
	spinError err;

	err = spinCameraGetTLDeviceNodeMap(hCam, mapHdl_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraGetTLDeviceNodeMap");
		return -1;
	}
	return 0;
}

static int connect_spink_cam(spinCamera hCam)
{
	spinError err;

	err = spinCameraInit(hCam);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraInit");
		return -1;
	}
	return 0;
}

static int disconnect_spink_cam(spinCamera hCam)
{
	spinError err;

	err = spinCameraDeInit(hCam);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraDeInit");
		return -1;
	}
	return 0;
}

static int get_camera_node_map(spinNodeMapHandle *map_p, spinCamera hCam )
{
	spinError err;

	// Retrieve nodemap from camera
	err = spinCameraGetNodeMap(hCam, map_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraGetNodeMap");
		return -1;
	}
	return 0;
}

static int init_one_spink_cam(int idx)
{
	spinCamera hCam = NULL;
	spinNodeMapHandle hNodeMapTLDevice = NULL;
	spinNodeMapHandle hNodeMap = NULL;

	if( get_spink_cam_from_list(&hCam,hCameraList,idx) < 0 )
		return -1;

//	if( get_spink_transport_level_map(&hNodeMapTLDevice,hCam) < 0 )
//		return -1;

	if( connect_spink_cam(hCam) < 0 ) return -1;
fprintf(stderr,"Camera %d is connected...\n",idx+1);

	// camera must be connected before fetching these...
//	if( get_camera_node_map(&hNodeMap,hCam) < 0 )
//		return -1;

	return 0;
}

static int init_spink_cameras()
{
	int i;

	for(i=0;i<numCameras;i++){
		if( init_one_spink_cam(i) < 0 )
			return -1;
	}
	return 0;
}

static int release_one_spink_cam(int idx)
{
	spinCamera hCam = NULL;
	spinError err;

	if( get_spink_cam_from_list(&hCam,hCameraList,idx) < 0 )
		return -1;

fprintf(stderr,"Releasing camera %d\n",idx+1);
	err = spinCameraRelease(hCam);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraRelease");
		return -1;
	}
	return 0;
}

static int disconnect_one_spink_cam(int idx)
{
	spinCamera hCam = NULL;

	if( get_spink_cam_from_list(&hCam,hCameraList,idx) < 0 )
		return -1;

	if( disconnect_spink_cam(hCam) < 0 ) return -1;
	return 0;
}

static int release_spink_cameras(spinCameraList hCameraList)
{
	int i;

	for(i=0;i<numCameras;i++){
		if( disconnect_one_spink_cam(i) < 0 ) return -1;
		if( release_one_spink_cam(i) < 0 ) return -1;
	}
	return 0;
}

static int release_spink_cam_list(spinCameraList *hCamList_p )
{
	spinError err;

	if( *hCamList_p == NULL ){
		fprintf(stderr,"release_spink_cam_list:  null list!?\n");
		return -1;
	}

	err = spinCameraListClear(*hCamList_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraListClear");
		return -1;
	}

fprintf(stderr,"Destroying camera list\n");
	err = spinCameraListDestroy(*hCamList_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraListDestroy");
		return -1;
	}

	*hCamList_p = NULL;
	return 0;
}

static int release_spink_system(spinSystem hSystem)
{
	spinError err;

	err = spinSystemReleaseInstance(hSystem);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinSystemReleaseInstance");
		return -1;
	}
	return 0;
}

static void release_spink_cam_system(void)
{
	assert( hSystem != NULL );
	if( release_spink_cameras(hCameraList) < 0 ) return;

	if( release_spink_cam_list(&hCameraList) < 0 ) return;
	if( release_spink_system(hSystem) < 0 ) return;
}

int main(int ac, char **av)
{
	if( get_spink_system(&hSystem) < 0 ) return -1;
	if( get_spink_cameras(hSystem,&hCameraList,&numCameras) < 0 ) return -1;
	if( init_spink_cameras() < 0 ) return -1;

	// Do stuff here!!!

	release_spink_cam_system();

	return 0;
}

