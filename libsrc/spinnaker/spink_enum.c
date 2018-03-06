#include "SpinnakerC.h"
#include <stdio.h>
#include <string.h>

#include "spink.h"

//
// Retrieve TL nodemap from interface
//
// *** NOTES ***
// Each interface has a nodemap that can be retrieved in order to access
// information about the interface itself, any devices connected, or
// addressing information if applicable.
//

int get_spink_map( spinInterface hInterface, spinNodeMapHandle *hMap_p)
{
	spinError err = SPINNAKER_ERR_SUCCESS;

	err = spinInterfaceGetTLNodeMap(hInterface, hMap_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to retrieve interface nodemap. Aborting with error %d...\n\n", err);
		return -1;
	}
	return 0;
}

int get_spink_node( spinNodeMapHandle hMap, const char *tag, spinNodeHandle *hdl_p)
{
	spinError err;

	err = spinNodeMapGetNode(hMap, tag, hdl_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to retrieve node (%s). Aborting with error %d...\n\n",tag,err);
		return -1;
	}
	return 0;
}

int spink_node_available(spinNodeHandle hdl)
{
	// Check availability
	bool8_t isAvailable = False;

	err = spinNodeIsAvailable(hdl, &isAvailable);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to check node availability (interface display name). Aborting with error %d...\n\n", err);
		return 0;
	}
	if( isAvailable )
		return 1;
	return 0;
}

int spink_node_is_readable(spinNodeHandle hdl)
{
	// Check readability
	bool8_t isReadable = False;

	err = spinNodeIsReadable(hdl, &isReadable);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to check node readability (interface display name). Aborting with error %d...\n\n", err);
		return 0;
	}
	if( isReadable )
		return 1;
	return 0;
}

int spink_get_string(spinNodeHandle hdl, char *buf, size_t *len_p)
{
	err = spinStringGetValue(hdl, buf, len_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to retrieve node string value. Aborting with error %d...\n\n", err);
		return -1;
	}
	return 0;
}

//
// Print interface display name
//

void print_interface_name(spinNodeHandle hInterfaceDisplayName)
{
	// Print
	char buf[MAX_BUFF_LEN];
	size_t len = MAX_BUFF_LEN;

	if( ! spink_node_is_available(hInterfaceDisplayName) ) return;
	if( ! spink_node_is_readable(hInterfaceDisplayName) ) return;

	if( spink_get_string(hInterfaceDisplayName,buf,&len) < 0 ) return;

	printf("Interface Display Name:  %s\n", buf);
}

//
// Retrieve list of cameras from the interface
//
// *** NOTES ***
// Camera lists can be retrieved from an interface or the system object.
// Camera lists retrieved from an interface, such as this one, only return
// cameras attached on that specific interface whereas camera lists
// retrieved from the system will return all cameras on all interfaces.
//
// *** LATER ***
// Camera lists must be cleared manually. This must be done prior to
// releasing the system and while the camera list is still in scope.
//

int get_spink_cam_list(spinInterface hInterface, spinCameraList *hCamList_p, size_t num_p)
{
	// Create empty camera list
	err = spinCameraListCreateEmpty(hCamList_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to create camera list. Aborting with error %d...\n\n", err);
		return -1;
	}

	// Retrieve cameras
	err = spinInterfaceGetCameras(hInterface, *hCamList_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to retrieve camera list. Aborting with error %d...\n\n", err);
		return -1;
	}

	// Retrieve number of cameras
	err = spinCameraListGetSize(*hCamList_p, num_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to retrieve number of cameras. Aborting with error %d...\n\n", err);
		return -1;
	}

	// Return if no cameras detected
	if( *num_p == 0 ){
		printf("\tNo devices detected.\n\n");
		return release_spink_cam_list(*hCamList_p);
	}
	return 0;
}

//
// Clear and destroy camera list before losing scope
//
// *** NOTES ***
// Camera lists do not automatically clean themselves up. This must be done
// manually. The same is true of interface lists.
//

int release_spink_cam_list( spinCameraList hCamList )
{
	err = spinCameraListClear(hCameraList);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to clear camera list. Aborting with error %d...\n\n", err);
		return -1;
	}

	err = spinCameraListDestroy(hCameraList);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to destroy camera list. Aborting with error %d...\n\n", err);
		return -1;
	}

	return 0;
}

int release_spink_interface_list( spinInterfaceList hInterfaceList )
{
	// Clear and destroy interface list before releasing system
	err = spinInterfaceListClear(hInterfaceList);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to clear interface list. Aborting with error %d...\n\n", err);
		return -1;
	}

	err = spinInterfaceListDestroy(hInterfaceList);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to destroy interface list. Aborting with error %d...\n\n", err);
		return -1;
	}
	return 0;
}

int release_spink_interface(spinInterface hInterface)
{
	// Release interface
	err = spinInterfaceRelease(hInterface);
	if (err != SPINNAKER_ERR_SUCCESS){
		fprintf(stderr,"Error releasing spink interface!?\n");
		return -1;
	}
	return 0;
}

//
// Select camera
//
// *** NOTES ***
// Each camera is retrieved from a camera list with an index. If the
// index is out of range, an exception is thrown.
//
// *** LATER ***
// Each camera handle needs to be released before losing scope or the
// system is released.
//

int get_spink_cam_from_list(spinCamera *hCam_p, spinCameraList hCameraList, int idx )
{
	err = spinCameraListGet(hCameraList, idx, hCam_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to retrieve camera. Aborting with error %d...\n\n", err);
		return -1;
	}
	return 0;
}

int get_spink_interface_from_list(spinInterface *hInterface_p, spinInterfaceList hInterfaceList, int idx )
{
	err = spinInterfaceListGet(hInterfaceList, idx, hInterface_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to retrieve camera. Aborting with error %d...\n\n", err);
		return -1;
	}
	return 0;
}

// Retrieve TL device nodemap; please see NodeMapInfo_C example for
// additional comments on transport layer nodemaps.

int get_spink_transport_level_map( spinNodeMapHandle *mapHdl_p, spinCamera hCam )
{
	err = spinCameraGetTLDeviceNodeMap(hCam, &mapHdl_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to retrieve TL device nodemap. Aborting with error %d...\n\n", err);
		return -1;
	}
	return 0;
}

//
// Retrieve device vendor name
//
// *** NOTES ***
// Grabbing node information requires first retrieving the node and
// then retrieving its information. There are two things to keep in
// mind. First, a node is distinguished by type, which is related
// to its value's data type.  Second, nodes should be checked for
// availability and readability/writability prior to making an
// attempt to read from or write to the node.
//

int get_spink_vendor_name( spinNodeMapHandle hNodeMapTLDevice, char *buf, size_t *len_p )
{
	spinNodeHandle hDeviceVendorName = NULL;

	if( get_spink_node(hNodeMapTLDevice,"DeviceVendorName",&hDeviceVendorName) < 0 ) return -1;
	if( ! spink_node_is_available(hDeviceVendorName) ) return -1;
	if( ! spink_node_is_readable(hDeviceVendorName) ) return -1;

	if( spink_get_string(hDeviceVendorName,buf,len_p) < 0 ) return -1;
	return 0;
}


//
// Retrieve device model name
//
// *** NOTES ***
// Because C has no try-catch blocks, each function returns an error
// code to suggest whether an error has occurred. Errors can be
// sufficiently handled with these return codes. Checking availability
// and readability/writability makes for safer and more complete code;
// however, keeping in mind example conciseness and legibility, only
// this example and NodeMapInfo_C demonstrate checking node
// availability and readability/writability while other examples
// handle errors with error codes alone.
//

int get_spink_model_name( spinNodeMapHandle hNodeMapTLDevice, char *buf, size_t *len_p )
{
	spinNodeHandle hDeviceModelName = NULL;

	if( get_spink_node(hNodeMapTLDevice,"DeviceModelName",&hDeviceModelName) < 0 ) return -1;
	if( ! spink_node_is_available(hDeviceModelName) ) return -1;
	if( ! spink_node_is_readable(hDeviceModelName) ) return -1;
	if( spink_get_string(hDeviceModelName,buf,len_p) < 0 ) return -1;
	return 0;
}

int print_spink_cam_info( spinCameraList hCameraList, int idx )
{
	spinCamera hCam = NULL;
	spinNodeMapHandle hNodeMapTLDevice = NULL;
	char deviceVendorName[MAX_BUFF_LEN];
	char deviceModelName[MAX_BUFF_LEN];
	size_t vendor_len = MAX_BUFF_LEN;
	size_t model_len = MAX_BUFF_LEN;

	if( get_spink_cam_from_list(&hCam,hCameraList,idx) < 0 ) return -1;

	if( get_spink_transport_level_map(&hNodeMapTLDevice,hCam) < 0 ) return -1;

	if( get_spink_vendor_name(hNodeMapTLDevice,deviceVendorName,&vendor_len) < 0 ) return -1;
	if( get_spink_model_name(hNodeMapTLDevice,deviceModelName,&model_len) < 0 ) return -1;

	printf("\tDevice %d %s %s\n\n", idx, deviceVendorName, deviceModelName);

	// release the camera?
	if( release_spink_cam(hCam) < 0 ) return -1;

	return 0;
}

//
// Release camera before losing scope
//
// *** NOTES ***
// Every handle that is created for a camera must be released before
// the system is released or an exception will be thrown.
//

int release_spink_cam(hCam)
{
	err = spinCameraRelease(hCam);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to release camera. Aborting with error %d...\n\n", err);
		return -1;
	}
	return 0;
}

int release_spink_system(spinSystem hSystem)
{

	err = spinSystemReleaseInstance(hSystem);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to release system instance. Aborting with  error %d...\n\n", err);
		return -1;
	}
	return 0;
}

// This function queries an interface for its cameras and then prints out
// device information.

int query_spink_interface(spinInterface hInterface)
{
	spinNodeMapHandle hNodeMapInterface = NULL;
	spinNodeHandle hInterfaceDisplayName = NULL;
	spinCameraList hCameraList = NULL;
	spinError err = SPINNAKER_ERR_SUCCESS;
	unsigned int i = 0;

	if( get_spink_map(hInterface,&hNodeMapInterface) < 0 ) return -1;

	if( get_spink_node(hNodeMapInterface,"InterfaceDisplayName",&hInterfaceDisplayName) < 0 ) return -1;

	print_interface_name(hInterfaceDisplayName);

	if( get_spink_cam_list(hInterface, &hCameraList, &numCameras) < 0 ) return -1;

	// Print device vendor and model name for each camera on the interface
	for (i = 0; i < numCameras; i++) {
		if( print_spink_cam_info(hCameraList,i) < 0 )
			return -1;

	if( release_spink_cam_list(hCameraList) < 0 ) return -1;
	
	return 0;
}

//
// Retrieve singleton reference to system object
//
// *** NOTES ***
// Everything originates with the system object. It is important to notice
// that it has a singleton implementation, so it is impossible to have
// multiple system objects at the same time.
//
// *** LATER ***
// The system object should be cleared prior to program completion.  If not
// released explicitly, it will be released automatically.
//

int get_spink_system(spinSystem *hSystem_p)
{
	spinSystem hSystem = NULL;

	err = spinSystemGetInstance(hSystem_p);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to retrieve system instance. Aborting with error %d...\n\n", err);
		return -1;
	}
	return 0;
}

//
// Retrieve list of interfaces from the system
//
// *** NOTES ***
// Interface lists are retrieved from the system object.
//
// *** LATER ***
// Interface lists must be cleared and destroyed manually. This must be
// done prior to releasing the system and while the interface list is still
// in scope.
//

int get_spink_interfaces(spinSystem hSystem, spinInterfaceList *hInterfaceList_p, size_t *numInterfaces_p)
{
	//spinInterfaceList hInterfaceList = NULL;
	//size_t numInterfaces = 0;

	// Create empty interface list
	err = spinInterfaceListCreateEmpty(hInterfaceList_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to create empty interface list. Aborting with error %d...\n\n", err);
		return -1;
	}

	// Retrieve interfaces from system
	err = spinSystemGetInterfaces(hSystem, *hInterfaceList_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to retrieve interface list. Aborting with error %d...\n\n", err);
		return -1;
	}

	// Retrieve number of interfaces
	err = spinInterfaceListGetSize(*hInterfaceList_p, numInterfaces_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to retrieve number of interfaces. Aborting with err %d...\n\n", err);
		return -1;
	}

	printf("Number of interfaces detected: %u\n\n", (unsigned int)numInterfaces);

	return 0;
}

//
// Retrieve list of cameras from the system
//
// *** NOTES ***
// Camera lists can be retrieved from an interface or the system object.
// Camera lists retrieved from the system, such as this one, return all
// cameras available on the system.
//
// *** LATER ***
// Camera lists must be cleared and destroyed manually. This must be done
// prior to releasing the system and while the camera list is still in
// scope.
//

int spinCameraList get_spink_cameras(spinSystem hSystem, spinCameraList *hCameraList_p, size_t num_p )
{

	// Create empty camera list
	err = spinCameraListCreateEmpty(hCameraList_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to create camera list. Aborting with error %d...\n\n", err);
		return -1;
	}

	// Retrieve cameras from system
	err = spinSystemGetCameras(hSystem, *hCameraList_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		printf("Unable to retrieve camera list. Aborting with error %d...\n\n", err);
		return -1;
	}

	// Retrieve number of cameras
	err = spinCameraListGetSize(*hCameraList_p, num_p);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to retrieve number of cameras. Aborting with  error %d...\n\n", err);
		return -1;
	}
	return 0;
}

// Example entry point; this function sets up the system and retrieves
// interfaces for the example.
int main(/*int argc, char** argv*/)
{
	spinSystem hSystem = NULL;
	spinInterfaceList hInterfaceList;
	size_t *numInterfaces;
	spinCameraList hCameraList = NULL;
	size_t numCameras = 0;

	spinError errReturn = SPINNAKER_ERR_SUCCESS;
	spinError err = SPINNAKER_ERR_SUCCESS;
	unsigned int i = 0;
	size_t numInterfaces = 0;

	// Print application build information
	printf("Application build date: %s %s \n\n", __DATE__, __TIME__);

	if( get_spink_system(&hSystem) < 0 )
		exit(1);

	if( get_spink_interfaces(hSystem,hInterfaceList,&numInterfaces) < 0 ) exit(1);
	if( get_spink_cameras(hSystem,hCameraList,&numCameras) < 0 ) exit(1);

	printf("Number of cameras detected: %u\n\n", (unsigned int)numCameras);

	// Finish if there are no cameras
	if (numCameras == 0 || numInterfaces == 0)
	{
		// Clear and destroy camera list before releasing system
		if( release_spink_cam_list(hCameraList) < 0 ) exit(1);
		if( release_spink_interface_list(hInterfaceList) < 0 ) exit(1);
		if( release_spink_system(hSystem) < 0 ) exit(1);

		printf("\nNot enough cameras/interfaces!\n");
		printf("Done! Press Enter to exit...\n");
		getchar();

		return -1;
	}

	printf("\n*** QUERYING INTERFACES ***\n\n");

	//
	// Run example on each interface
	//
	// *** NOTES ***
	// In order to run all interfaces in a loop, each interface needs to
	// retrieved using its index.
	//
	for (i = 0; i < numInterfaces; i++)
	{
		// Select interface
		spinInterface hInterface = NULL;

		if( get_spink_interface_from_list(&hInterface,hInterfaceList,i) < 0 )
			exit(1);

		// Run example
		if( query_spink_interface(hInterface) < 0 )
			exit(1);

		if( release_spink_interface(hInterface) < 0 )
			exit(1);
	}

	if( release_spink_cam_list(hCameraList) < 0 ) exit(1)
	if( release_spink_interface_list(hInterfaceList) < 0 ) exit(1)
	if( release_spink_system(hSystem) < 0 ) exit(1);

	exit(0);
}
