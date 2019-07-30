#include "quip_config.h"
#include <stdio.h>
#include <string.h>
#include "quip_prot.h"
#include "spink.h"

Spink_Cam *current_skc_p = NULL;
int max_display_name_len=0;

void _insure_current_camera(QSP_ARG_DECL  Spink_Cam *skc_p)
{
//fprintf(stderr,"insure_current_camera %s BEGIN\n",skc_p->skc_name);
	if( skc_p == current_skc_p ) {
//fprintf(stderr,"insure_current_camera:  %s is already current\n",skc_p->skc_name);
		return;
	}

	if( current_skc_p != NULL ){
//fprintf(stderr,"insure_current_camera %s will release old camera %s\n", skc_p->skc_name,current_skc_p->skc_name);
		if( release_current_camera(1) < 0 )
			error1("insure_current_camera:  failed to release previous camera!?");
	}

#ifdef HAVE_LIBSPINNAKER
	if( skc_p->skc_current_handle == NULL ){
		spinCamera hCam;
//fprintf(stderr,"insure_current_camera %s needs to refresh the camera handle\n", skc_p->skc_name);
		if( get_cam_from_list(hCameraList,skc_p->skc_sys_idx,&hCam) < 0 )
			error1("insure_current_camera:  error getting camera from list!?");
		skc_p->skc_current_handle = hCam;
//fprintf(stderr,"insure_current_camera %s:  new handle = 0x%lx\n", skc_p->skc_name,(u_long)hCam);
	}
#endif // HAVE_LIBSPINNAKER
	current_skc_p = skc_p;
}

#ifdef HAVE_LIBSPINNAKER

// We may release the camera while it is running...

int _release_current_camera(QSP_ARG_DECL  int strict)
{
	if( current_skc_p==NULL ){
		if( strict ){
			sprintf(ERROR_STRING,"Unnecessary call to release_current_camera!?");
			warn(ERROR_STRING);
		}
		return 0;
	}

	if( spink_release_cam(current_skc_p) < 0 )
		return -1;
	current_skc_p = NULL;
	return 0;
}

void _report_node_access_error(QSP_ARG_DECL  spinNodeHandle hNode, const char *w)
{
	char dname[256];
	size_t len=256;
	if( get_display_name(dname,&len,hNode) < 0 ){
		strcpy(dname,"(unable to get display name)");
	}
	sprintf(ERROR_STRING,"feature node '%s' not %s!?",dname,w);
	advise(ERROR_STRING);
}

//
// Retrieve value of any node type as string
//
// *** NOTES ***
// Because value nodes return any node type as a string, it can be much
// easier to deal with nodes as value nodes rather than their actual
// individual types.
//

int _get_node_value_string(QSP_ARG_DECL  char *buf, size_t *buflen_p, spinNodeHandle hNode )
{
	size_t n_need;

	// Ensure allocated buffer is large enough for storing the string
	if( node_to_string(hNode, NULL, &n_need) < 0 ) return -1;
	if( n_need <= *buflen_p ) {	// does this cound the terminating null???
		if( node_to_string(hNode, buf, buflen_p) < 0 ) return -1;
	} else {
		strcpy(buf,"(too many chars)");
	}

	return 0;
}

//
// Retrieve string node value
//
// *** NOTES ***
// The Spinnaker SDK requires a character array to hold the string and
// an integer for the number of characters. Ensure that the size of the
// character array is large enough to hold the entire string.
//
// Throughout the examples in C, 256 is typically used as the size of a
// character array. This will typically be sufficient, but not always.
// For instance, a lookup table register node (which is not explored in
// this example) may be much larger.
//

int _get_string_node_string(QSP_ARG_DECL  char *buf, size_t *buflen_p, spinNodeHandle hNode )
{
	size_t n_need;

	// Ensure allocated buffer is large enough for storing the string
	if( get_string_value(hNode, NULL, &n_need) < 0 ) return -1;
	if(n_need <= *buflen_p) {
		if( get_string_value(hNode, buf, buflen_p) < 0 ) return -1;
	} else {
		strcpy(buf,"(...)");
	}
	return 0;
}

// This function retrieves and prints the display names of an enumeration node
// and its current entry (which is actually housed in another node unto itself).
//
// Retrieve current entry node
//
// *** NOTES ***
// Returning the current entry of an enumeration node delivers the entry
// node rather than the integer value or symbolic. The current entry's
// integer and symbolic need to be retrieved from the entry node because
// they cannot be directly accessed through the enumeration node in C.
//
//
// Retrieve current symbolic
//
// *** NOTES ***
// Rather than retrieving the current entry node and then retrieving its
// symbolic, this could have been taken care of in one step by using the
// enumeration node's ToString() method.
//

int _get_display_name_len(QSP_ARG_DECL  spinNodeHandle hdl)
{
	size_t len;

	if( get_node_display_name(hdl,NULL,&len) < 0 ) return 0;
	return (int) len;
}

//
// Retrieve display name
//
// *** NOTES ***
// A node's 'display name' is generally more appropriate for output and
// user interaction whereas its 'name' is what the camera understands.
// Generally, its name is the same as its display namebut without
// spaces - for instance, the name of the node that houses a camera's
// serial number is 'DeviceSerialNumber' while its display name is
// 'Device Serial Number'.
//

int _get_display_name(QSP_ARG_DECL  char *buf, size_t *len_p, spinNodeHandle hdl)
{
	return get_node_display_name(hdl,buf,len_p);
}

int _get_node_name(QSP_ARG_DECL  char *buf, size_t *len_p, spinNodeHandle hdl)
{
	return get_node_short_name(hdl, buf, len_p);
}

int _traverse_by_node_handle(QSP_ARG_DECL  spinNodeHandle hNode, int level, int (*func)(QSP_ARG_DECL spinNodeHandle hNode, int level) )
{
	unsigned int i = 0;
	spinNodeType type;

	if( ! spink_node_is_implemented(hNode) ){
		report_node_access_error(hNode,"implemented");
		return 0;
	}

	if( ! spink_node_is_available(hNode) ){
		if( verbose )
			report_node_access_error(hNode,"available");
		return 0;
	}

	if( level == 0 ){	// Root node
		// set global index?
	}

	if( (*func)(QSP_ARG  hNode,level) < 0 )
		return -1;

	if( get_node_type(hNode,&type) < 0 ) return -1;

	if( type == CategoryNode ){
		size_t numberOfFeatures = 0;

		if( get_n_features(hNode,&numberOfFeatures) < 0 ) return -1;

		for (i = 0; i < numberOfFeatures; i++) {
			spinNodeHandle hFeatureNode = NULL;

			if( get_feature_by_index(hNode, i, &hFeatureNode) < 0 ) return -1;

			if( ! spink_node_is_implemented(hNode) ){
				report_node_access_error(hNode,"implemented");
				continue;
			}

			if( ! spink_node_is_available(hFeatureNode) ){
				if( verbose )
					report_node_access_error(hFeatureNode,"available");
				continue;
			}
			// Set global var to communicate node index.
			// This works because func is called before recursion to sub-nodes
			current_node_idx = i;
			if( traverse_by_node_handle(hFeatureNode,level+1,func) < 0 ) return -1;
		}
	} else if( type == EnumerationNode ){
		size_t num;
		if( get_n_enum_entries(hNode,&num) < 0 ) return -1;

		for (i = 0; i < num; i++) {
			spinNodeHandle hEnumEntryNode = NULL;

			if( get_enum_entry_by_index(hNode, i, &hEnumEntryNode) < 0 ) return -1;

			if( ! spink_node_is_implemented(hNode) ){
				report_node_access_error(hNode,"implemented");
				continue;
			}

			if( ! spink_node_is_available(hEnumEntryNode) ){
				if( verbose )
					report_node_access_error(hEnumEntryNode,"available");
				continue;
			}
			// Set global var to communicate node index.
			// This works because func is called before recursion to sub-nodes
			current_node_idx = i;
			if( traverse_by_node_handle(hEnumEntryNode,level+1,func) < 0 ) return -1;
		}
	}
	return 0;
}
//////////////////////////////////////////////



int _get_node_map_handle(QSP_ARG_DECL  spinNodeMapHandle *hMap_p, Spink_Map *skm_p, const char *whence)
{
	spinCamera hCam;
	Spink_Cam *skc_p;

//fprintf(stderr,"get_node_map_handle map_name = %s BEGIN\n",skm_p->skm_name);
	skc_p = skm_p->skm_skc_p;

	assert(skc_p!=NULL);
	insure_current_camera(skc_p);

	assert(skc_p->skc_current_handle!=NULL);
//fprintf(stderr,"get_node_map_handle:  %s has current handle 0x%lx\n",skc_p->skc_name, (u_long)skc_p->skc_current_handle);
	hCam = skc_p->skc_current_handle;

//fprintf(stderr,"get_node_map_handle switching on map type %d\n",skm_p->skm_type);
	switch( skm_p->skm_type ){
		case INVALID_NODE_MAP:
		case N_NODE_MAP_TYPES:
			sprintf(ERROR_STRING,"get_node_map_handle (%s):  invalid type code!?",whence);
			warn(ERROR_STRING);
			break;
		case CAM_NODE_MAP:
			if( ! IS_CONNECTED(skm_p->skm_skc_p) ){
//fprintf(stderr,"init_one_spink_cam:  connecting %s\n",skm_p->skm_skc_p->skc_name);
				if( connect_spink_cam(hCam) < 0 ) return -1;
				skm_p->skm_skc_p->skc_flags |= SPINK_CAM_CONNECTED;
			} else {
//fprintf(stderr,"init_one_spink_cam:  %s is already connected\n",skm_p->skm_skc_p->skc_name);
			}

			if( get_camera_node_map(hCam,hMap_p) < 0 ){
				sprintf(ERROR_STRING,
			"get_node_map_handle (%s):  error getting camera node map!?",whence);
				error1(ERROR_STRING);
			}
			break;
		case DEV_NODE_MAP:
//fprintf(stderr,"get_node_map_handle calling get_device_node_map\n");
			if( get_device_node_map(hCam,hMap_p) < 0 ){
				sprintf(ERROR_STRING,
			"get_node_map_handle (%s):  error getting device node map!?",whence);
				error1(ERROR_STRING);
			}
			break;
		case STREAM_NODE_MAP:
			if( get_stream_node_map(hCam,hMap_p) < 0 ){
				sprintf(ERROR_STRING,
			"get_node_map_handle (%s):  error getting stream node map!?",whence);
				error1(ERROR_STRING);
			}
			break;
	}
	return 0;
}

void _list_nodes_from_map(QSP_ARG_DECL  Spink_Map *skm_p)
{
	// lists nodes in the current context stack?
	list_spink_nodes( tell_msgfile() );
}

#endif // HAVE_LIBSPINNAKER


