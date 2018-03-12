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

static int get_spink_camera_list(spinSystem hSystem, spinCameraList *hCameraList_p, size_t *num_p )
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

static int release_one_spink_cam(spinCamera hCam)
{
	spinError err;

	err = spinCameraRelease(hCam);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraRelease");
		return -1;
	}
	return 0;
}

static int init_one_spink_cam(int idx)
{
	spinCamera hCam = NULL;
	spinNodeMapHandle hNodeMapTLDevice = NULL;
	//spinNodeMapHandle hNodeMap = NULL;

	if( get_spink_cam_from_list(&hCam,hCameraList,idx) < 0 )
		return -1;

//	if( get_spink_transport_level_map(&hNodeMapTLDevice,hCam) < 0 )
//		return -1;

	if( connect_spink_cam(hCam) < 0 ) return -1;
fprintf(stderr,"Camera %d is connected...\n",idx+1);

	release_one_spink_cam(hCam);	// release does not disconnect???

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

static int release_spink_cameras(spinCameraList hCamList)
{
	int idx;

	for(idx=0;idx<numCameras;idx++){
		spinCamera hCam;

		if( get_spink_cam_from_list(&hCam,hCamList,idx) < 0 )
			return -1;
		if( disconnect_spink_cam(hCam) < 0 ) return -1;
		if( release_one_spink_cam(hCam) < 0 ) return -1;
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

static int fetch_spink_node(spinNodeMapHandle hMap, const char *tag, spinNodeHandle *hdl_p)
{
	spinError err;

	err = spinNodeMapGetNode(hMap, tag, hdl_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinNodeMapGetNode");
		return -1;
	}
	return 0;
}

static int spink_node_is_available(spinNodeHandle hdl)
{
	spinError err;
	bool8_t isAvailable = False;

	err = spinNodeIsAvailable(hdl, &isAvailable);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinNodeIsAvailable");
		return 0;
	}
	if( isAvailable )
		return 1;
	return 0;
}

static int spink_node_is_readable(spinNodeHandle hdl)
{
	spinError err;
	bool8_t isReadable = False;

	err = spinNodeIsReadable(hdl, &isReadable);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinNodeIsReadable");
		return 0;
	}
	if( isReadable )
		return 1;
	return 0;
}

//		err = spinEnumerationGetEntryByName(hAcquisitionMode, "Continuous", &hAcquisitionModeContinuous);
//		if (err != SPINNAKER_ERR_SUCCESS) {
//			printf("Unable to set acquisition mode to continuous (enum entry retrieval). Aborting with error %d...\n\n", err);
//			return err;
//		}

static int get_node_type(spinNodeType *type_p, spinNodeHandle hNode)
{
	spinError err;

	err = spinNodeGetType(hNode, type_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinNodeGetType");
		return -1;
	}
	return 0;
}

/*
static int display_spink_node(spinNodeHandle hNode, int level)
{
	spinNodeType type;

fprintf(stderr,"display_spink_node:  chosenRead = %d\n",chosenRead);
	if (chosenRead == VALUE) {
		if( print_value_node(hNode,level) < 0 ) return -1;
	} else if (chosenRead == INDIVIDUAL) {
		if( get_node_type(&type,hNode) < 0 ) return -1;
		switch (type) {
			case RegisterNode:
			case EnumEntryNode:
			case CategoryNode:
			case PortNode:
			case BaseNode:
			case UnknownNode:
				warn("OOPS - unahndled node type!?");
				break;
			case ValueNode:
				if( print_value_node(hNode,level) < 0 ) return -1;
				break;
			case StringNode:
				if( print_string_node(hNode, level + 1) < 0 ) return -1;
				break;
			case IntegerNode:
				if( print_int_node(hNode, level + 1) < 0 ) return -1;
				break;
			case FloatNode:
				if( print_float_node(hNode, level + 1) < 0 ) return -1;
				break;
			case BooleanNode:
				if( print_bool_node(hNode, level + 1) < 0 ) return -1;
				break;
			case CommandNode:
				if( print_cmd_node(hNode, level + 1) < 0 ) return -1;
				break;
			case EnumerationNode:
				if( print_enum_node(hNode, level + 1) < 0 ) return -1;
				break;
		}
	} else {
		// assert
		error1("Unexpected value for chosenRead!?");
	}
	return 0;
}
*/

static int get_node_value_string(char *buf, size_t *buflen_p, spinNodeHandle hNode )
{
	spinError err;
	//const unsigned int k_maxChars = MAX_NODE_CHARS;
	size_t n_need;

	// Ensure allocated buffer is large enough for storing the string
	err = spinNodeToString(hNode, NULL, &n_need);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinNodeToString (get_node_value_string, getting required size)");
		return -1;
	}
	if( n_need <= *buflen_p ) {	// does this cound the terminating null???
		err = spinNodeToString(hNode, buf, buflen_p);
		if (err != SPINNAKER_ERR_SUCCESS) {
			report_spink_error(err,"spinNodeToString (get_node_value_string, getting string value)");
			return -1;
		}
	} else {
		strcpy(buf,"(too many chars)");
	}

	return 0;
}

static int get_display_name(char *buf, size_t *len_p, spinNodeHandle hdl)
{
	spinError err;

	err = spinNodeGetDisplayName(hdl, buf, len_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinNodeGetDisplayName");
		return -1;
	}
	return 0;
}

static int print_node_value(spinNodeHandle hNode )
{
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;
	char value[MAX_BUFF_LEN];
	size_t valueLength = MAX_BUFF_LEN;
	spinNodeType type;
	char out_string[256];

	if( get_node_type(&type,hNode) < 0 ) return -1;

	if( get_display_name(displayName,&displayNameLength,hNode) < 0 ) return -1;

	if( type == CategoryNode ){
		sprintf(out_string,"%s", displayName);
	} else {
		if( get_node_value_string(value,&valueLength,hNode) < 0 ) return -1;
		sprintf(out_string,"%s:  %s", displayName,value);
	}
	printf("%s\n",out_string);

	return 0;
}

static int display_one_node(spinNodeMapHandle hMap, const char *tag)
{
	spinNodeHandle hNode = NULL;

	if( fetch_spink_node(hMap,tag,&hNode) < 0 ) return -1;

	if( ! spink_node_is_available(hNode) )
		fprintf(stderr,"Node not available!?\n");
	if( ! spink_node_is_readable(hNode) )
		fprintf(stderr,"Node not readable!?\n");

	if( print_node_value(hNode) < 0 ) return -1;
	return 0;
}

static int show_some_features(spinCamera hCam)
{
	spinNodeMapHandle hMap;
	spinNodeHandle hAcquisitionMode = NULL;
	spinNodeHandle hAcquisitionModeContinuous = NULL;
	int64_t acquisitionModeContinuous = 0;

	if( get_camera_node_map(&hMap, hCam ) < 0 ) return -1;

	if( display_one_node(hMap,"AcquisitionMode") < 0 ) return -1;
	if( display_one_node(hMap,"AcquisitionMode") < 0 ) return -1;
	if( display_one_node(hMap,"AcquisitionMode") < 0 ) return -1;
}

static int examine_one_camera(int idx)
{
	spinCamera hCam;

	if( get_spink_cam_from_list(&hCam,hCameraList,idx) < 0 ) return -1;
	if( show_some_features(hCam) < 0 ) return -1;
	if( release_one_spink_cam(hCam) < 0 ) return -1;
	return 0;
}

static int examine_spink_cameras(void)
{
	int idx;

	for(idx=0;idx<numCameras;idx++)
		if(examine_one_camera(idx) < 0 )
			return -1;
	return 0;
}

int main(int ac, char **av)
{
	if( get_spink_system(&hSystem) < 0 ) return -1;
	if( get_spink_camera_list(hSystem,&hCameraList,&numCameras) < 0 ) return -1;
	if( init_spink_cameras() < 0 ) return -1;

	// Do stuff here!!!
	if( examine_spink_cameras() < 0 ) return -1;


	release_spink_cam_system();

	return 0;
}

