//#include <unistd.h>
#include <stdlib.h>
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

#ifdef HAVE_LIBSPINNAKER

int _spink_node_is_implemented(QSP_ARG_DECL  spinNodeHandle hdl)
{
	bool8_t isImplemented = False;

	if( node_is_implemented(hdl,&isImplemented) < 0 ) return 0;
	if( isImplemented )
		return 1;
	return 0;
}

int _spink_node_is_available(QSP_ARG_DECL  spinNodeHandle hdl)
{
	bool8_t isAvailable = False;

	if( node_is_available(hdl, &isAvailable) < 0 )
		return 0;
	if( isAvailable )
		return 1;
	return 0;
}

int _spink_node_is_readable(QSP_ARG_DECL spinNodeHandle hdl)
{
	bool8_t isReadable = False;

	if( node_is_readable(hdl, &isReadable) < 0 ) return 0;
	if( isReadable )
		return 1;
	return 0;
}

int _spink_node_is_writeable(QSP_ARG_DECL spinNodeHandle hdl)
{
	bool8_t isWriteable = False;

	if( node_is_writeable(hdl, &isWriteable) < 0 ) return 0;
	if( isWriteable )
		return 1;
	return 0;
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


int _get_spink_cam_list(QSP_ARG_DECL spinInterface hInterface, spinCameraList *hCamList_p, size_t *num_p)
{
	if( create_empty_cam_list(hCamList_p) < 0 ) return -1;
	if( get_iface_cameras(hInterface,hCamList_p) < 0 ) return -1;
	if( get_n_cameras(*hCamList_p,num_p) < 0 ) return -1;

	// Return if no cameras detected
	if( *num_p == 0 ){
		printf("\tNo devices detected.\n\n");
		return release_spink_cam_list(hCamList_p);
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


int _release_spink_cam_list(QSP_ARG_DECL  spinCameraList *hCamList_p )
{
DEBUG_MSG(release_spin_cam_list BEGIN)
	if( *hCamList_p == NULL ){
		fprintf(stderr,"release_spink_cam_list:  null list!?\n");
		return -1;
	}

	if( clear_cam_list(*hCamList_p) < 0 ) return -1;
	if( destroy_cam_list(*hCamList_p) < 0 ) return -1;

	*hCamList_p = NULL;
DEBUG_MSG(release_spin_cam_list DONE)
	return 0;
}

int _release_spink_interface_list(QSP_ARG_DECL  spinInterfaceList *hInterfaceList_p )
{
DEBUG_MSG(release_spink_interfacelist BEGIN)
	if( *hInterfaceList_p == NULL ){
		fprintf(stderr,"release_spink_interface_list:  null list!?\n");
		return -1;
	}

	// Clear and destroy interface list before releasing system
	if( clear_iface_list(*hInterfaceList_p) < 0 ) return -1;
	if( destroy_iface_list(*hInterfaceList_p) < 0 ) return -1;

	*hInterfaceList_p = NULL;
DEBUG_MSG(release_spink_interfacelist DONE)
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

//int _get_spink_cam_from_list(QSP_ARG_DECL  spinCamera *hCam_p, spinCameraList hCameraList, int idx )
//{
	//return get_cam_from_list(hCameraList,idx,hCam_p);
//}

int _get_spink_interface_from_list(QSP_ARG_DECL spinInterface *hInterface_p, spinInterfaceList hInterfaceList, int idx )
{
	return get_iface_from_list(hInterfaceList,idx,hInterface_p);
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


int _get_spink_interfaces(QSP_ARG_DECL spinSystem hSystem, spinInterfaceList *hInterfaceList_p, size_t *numInterfaces_p)
{
	if( create_empty_iface_list(hInterfaceList_p) < 0 ) return -1;
	if( get_iface_list(hSystem,*hInterfaceList_p) < 0 ) return -1;
	if( get_n_interfaces(*hInterfaceList_p, numInterfaces_p) < 0 ) return -1;
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

int _get_spink_cameras(QSP_ARG_DECL spinSystem hSystem, spinCameraList *hCameraList_p, size_t *num_p )
{
	if( create_empty_cam_list(hCameraList_p) < 0 ) return -1;
	if( get_cameras_from_system(hSystem,*hCameraList_p) < 0 ) return -1;
	if( get_n_cameras(*hCameraList_p,num_p) < 0 ) return -1;
	return 0;
}

/////////////////////////////////////////////////////////////

int _lookup_spink_node(QSP_ARG_DECL  Spink_Node *skn_p, spinNodeHandle *hdl_p)
{
	spinNodeMapHandle hMap;

//fprintf(stderr,"lookup_spink_node node = %s calling get_node_map_handle\n",skn_p->skn_name);
//fprintf(stderr,"lookup_spink_node:  node %s belongs to map %s\n",skn_p->skn_name,skn_p->skn_skm_p->skm_name);
	if( get_node_map_handle(&hMap,skn_p->skn_skm_p,"lookup_spink_node") < 0 )
		return -1;

//fprintf(stderr,"lookup_spink_node %s calling fetch_spink_node\n",skn_p->skn_name);
	if( fetch_spink_node(hMap,skn_p->skn_name,hdl_p) < 0 )
		return -1;

	return 0;
}

//
// Print interface display name
//

int _get_interface_name(QSP_ARG_DECL  char *buf, size_t buflen, spinInterface hInterface)
{
	spinNodeMapHandle hNodeMapInterface = NULL;
	spinNodeHandle hInterfaceDisplayName = NULL;

	if( get_iface_map(hInterface,&hNodeMapInterface) < 0 ) return -1;

	if( fetch_spink_node(hNodeMapInterface,"InterfaceDisplayName",&hInterfaceDisplayName) < 0 ) return -1;

	if( ! spink_node_is_available(hInterfaceDisplayName) ) return -1;
	if( ! spink_node_is_readable(hInterfaceDisplayName) ) return -1;

	if( spink_get_string(hInterfaceDisplayName,buf,&buflen) < 0 ) return -1;
	return 0;
}

void _print_interface_name(QSP_ARG_DECL  spinNodeHandle hInterfaceDisplayName)
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

int _get_camera_vendor_name(QSP_ARG_DECL  char *buf, size_t buflen, spinCamera hCam)
{
	spinNodeMapHandle hNodeMapTLDevice = NULL;
	spinNodeHandle hDeviceVendorName = NULL;

//fprintf(stderr,"get_camera_vendor_name calling get_device_node_map\n");
	if( get_device_node_map(hCam,&hNodeMapTLDevice) < 0 ) return -1;
//fprintf(stderr,"get_camera_vendor_name:  map = 0x%lx  *map = 0x%lx\n", (u_long)hNodeMapTLDevice,(u_long)*((void **)hNodeMapTLDevice));

	if( fetch_spink_node(hNodeMapTLDevice,"DeviceVendorName",&hDeviceVendorName) < 0 ) return -1;
	if( ! spink_node_is_available(hDeviceVendorName) ) return -1;
	if( ! spink_node_is_readable(hDeviceVendorName) ) return -1;
	if( spink_get_string(hDeviceVendorName,buf,&buflen) < 0 ) return -1;
//fprintf(stderr,"get_camera_vendor_name: vendor name is \"%s\"\n",buf);

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

int _get_spink_model_name( QSP_ARG_DECL  spinNodeMapHandle hNodeMapTLDevice, char *buf, size_t *len_p )
{
	spinNodeHandle hDeviceModelName = NULL;

	if( fetch_spink_node(hNodeMapTLDevice,"DeviceModelName",&hDeviceModelName) < 0 ) return -1;
	if( ! spink_node_is_available(hDeviceModelName) ) return -1;
	if( ! spink_node_is_readable(hDeviceModelName) ) return -1;
	if( spink_get_string(hDeviceModelName,buf,len_p) < 0 ) return -1;
	return 0;
}

int _get_camera_model_name(QSP_ARG_DECL  char *buf, size_t buflen, spinCamera hCam)
{
	spinNodeHandle hDeviceModelName = NULL;
	spinNodeMapHandle hNodeMapTLDevice;

	if( get_device_node_map(hCam,&hNodeMapTLDevice) < 0 )
		return -1;
//fprintf(stderr,"get_camera_model_name: device node map = 0x%lx, *map = 0x%lx\n", (u_long)hNodeMapTLDevice,(u_long)*((void **)hNodeMapTLDevice));

//fprintf(stderr,"get_camera_model_name calling fetch_spink_node, map = 0x%lx (*map = 0x%lx)\n", (u_long)hNodeMapTLDevice, (u_long) *((void **)hNodeMapTLDevice));
	if( fetch_spink_node(hNodeMapTLDevice,"DeviceModelName",&hDeviceModelName) < 0 ) return -1;
	if( ! spink_node_is_available(hDeviceModelName) ) return -1;
	if( ! spink_node_is_readable(hDeviceModelName) ) return -1;
	if( spink_get_string(hDeviceModelName,buf,&buflen) < 0 ) return -1;

//fprintf(stderr,"get_camera_model_name obtained \"%s\"\n",buf);
	return 0;
}

int _print_indexed_spink_cam_info( QSP_ARG_DECL  spinCameraList hCameraList, int idx )
{
	spinCamera hCam = NULL;
	//spinNodeMapHandle hNodeMapTLDevice = NULL;
	char deviceVendorName[MAX_BUFF_LEN];
	char deviceModelName[MAX_BUFF_LEN];
	size_t vendor_len = MAX_BUFF_LEN;
	size_t model_len = MAX_BUFF_LEN;

	if( get_cam_from_list(hCameraList,idx,&hCam) < 0 ) return -1;

	if( get_camera_vendor_name(deviceVendorName,vendor_len,hCam) < 0 ) return -1;
	if( get_camera_model_name(deviceModelName,model_len,hCam) < 0 ) return -1;

	printf("\tDevice %d %s %s\n\n", idx, deviceVendorName, deviceModelName);

	// release the camera
	if( release_spink_cam(hCam) < 0 ) return -1;

	return 0;
}

int _get_spink_interface_cameras(QSP_ARG_DECL  spinInterface hInterface)
{
	unsigned int i = 0;

	spinCameraList hCameraList = NULL;
	size_t numCameras = 0;

	if( get_spink_cam_list(hInterface, &hCameraList, &numCameras) < 0 )
		return -1;

	if( numCameras == 0 ) return 0;

	// Print device vendor and model name for each camera on the interface
	for (i = 0; i < numCameras; i++) {
		if( print_indexed_spink_cam_info(hCameraList,i) < 0 )
			return -1;
	}

	if( release_spink_cam_list(&hCameraList) < 0 ) return -1;
	
	return 0;
}


#endif // HAVE_LIBSPINNAKER
