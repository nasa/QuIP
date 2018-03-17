#include "quip_config.h"
#include <stdio.h>
#include <string.h>
#include "quip_prot.h"
#include "spink.h"

#ifdef HAVE_LIBSPINNAKER

Spink_Cam *current_skc_p = NULL;
int max_display_name_len=0;

#define MAX_NODE_VALUE_CHARS_TO_PRINT	24	// must be less than LLEN !!

// BUG - if the constant above is changed, these definitions should be changed
// to match...   We could generate them programmatically, but that's extra work
// that we will skip for now.

#define INT_NODE_FMT_STR	"%-24ld"
#define FLT_NODE_FMT_STR	"%-24f"
#define STRING_NODE_FMT_STR	"%-24s"

int _release_current_camera(SINGLE_QSP_ARG_DECL)
{
	assert(current_skc_p!=NULL);
//fprintf(stderr,"release_current_camera:  releasing %s\n",current_skc_p->skc_name);
	if( spink_release_cam(current_skc_p) < 0 )
		return -1;
	current_skc_p = NULL;
	return 0;
}

void _insure_current_camera(QSP_ARG_DECL  Spink_Cam *skc_p)
{
//fprintf(stderr,"insure_current_camera %s BEGIN\n",skc_p->skc_name);
	if( skc_p == current_skc_p ) {
//fprintf(stderr,"insure_current_camera:  %s is already current\n",skc_p->skc_name);
		return;
	}

	if( current_skc_p != NULL ){
//fprintf(stderr,"insure_current_camera %s will release old camera %s\n", skc_p->skc_name,current_skc_p->skc_name);
		if( release_current_camera() < 0 )
			error1("insure_current_camera:  failed to release previous camera!?");
	}

	if( skc_p->skc_current_handle == NULL ){
		spinCamera hCam;
//fprintf(stderr,"insure_current_camera %s needs to refresh the camera handle\n", skc_p->skc_name);
		if( get_cam_from_list(hCameraList,skc_p->skc_sys_idx,&hCam) < 0 )
			error1("insure_current_camera:  error getting camera from list!?");
		skc_p->skc_current_handle = hCam;
//fprintf(stderr,"insure_current_camera %s:  new handle = 0x%lx\n", skc_p->skc_name,(u_long)hCam);
	} else {
//fprintf(stderr,"insure_current_camera %s:   camera already has a non-NULL handle 0x%lx\n", skc_p->skc_name,(u_long)skc_p->skc_current_handle);
	}
	current_skc_p = skc_p;
}

// This helper function deals with output indentation, of which there is a lot.

#define indent(level) _indent(QSP_ARG  level)

static void _indent(QSP_ARG_DECL  unsigned int level)
{
	unsigned int i = 0;

	for (i = 0; i < level; i++) {
		prt_msg_frag("   ");
	}
}

#define report_node_access_error(hNode, w) _report_node_access_error(QSP_ARG  hNode, w)

static void _report_node_access_error(QSP_ARG_DECL  spinNodeHandle hNode, const char *w)
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

#define print_node_value(hNode, type) _print_node_value(QSP_ARG  hNode, type)

static void _print_node_value(QSP_ARG_DECL  spinNodeHandle hNode, spinNodeType type)
{
	char val_buf[MAX_BUFF_LEN];
	size_t buf_len = MAX_BUFF_LEN;
	int64_t integerValue = 0;
	double floatValue = 0.0;
	bool8_t booleanValue = False;
	spinNodeHandle hCurrentEntryNode = NULL;

	switch (type) {
		case RegisterNode:
		case EnumEntryNode:
		case PortNode:
		case BaseNode:
		case UnknownNode:
			sprintf(MSG_STR,STRING_NODE_FMT_STR,"OOPS!?");
			break;
		case CategoryNode:
			sprintf(MSG_STR,STRING_NODE_FMT_STR,"");
			break;
		case ValueNode:
			if( get_node_value_string(val_buf,&buf_len,hNode) < 0 ) return;
			sprintf(MSG_STR,STRING_NODE_FMT_STR,val_buf);
			break;
		case StringNode:
			if( get_string_node_string(val_buf,&buf_len,hNode) < 0 ) return;
			sprintf(MSG_STR,STRING_NODE_FMT_STR,val_buf);
			break;
		case IntegerNode:
			if( get_int_value(hNode, &integerValue) < 0 ) return;
			sprintf(MSG_STR,INT_NODE_FMT_STR, integerValue);
			break;
		case FloatNode:
			if( get_float_value(hNode,&floatValue) < 0 ) return;
			sprintf(MSG_STR,FLT_NODE_FMT_STR, floatValue);
			break;
		case BooleanNode:
			if( get_bool_value(hNode,&booleanValue) < 0 ) return;
			sprintf(MSG_STR,STRING_NODE_FMT_STR, (booleanValue ? "true" : "false"));
			break;
		case CommandNode:
			if( get_tip_value(hNode,val_buf,&buf_len) < 0 ) return;
			if( buf_len > MAX_NODE_VALUE_CHARS_TO_PRINT) {
				int i;
				for (i = 0; i < MAX_NODE_VALUE_CHARS_TO_PRINT-3; i++) {
					MSG_STR[i] = val_buf[i];
				}
				MSG_STR[i++]='.';
				MSG_STR[i++]='.';
				MSG_STR[i++]='.';
				MSG_STR[i++]=0;
			} else {
				sprintf(MSG_STR,STRING_NODE_FMT_STR, val_buf);
			}
			break;
		case EnumerationNode:
			if( get_current_entry(hNode,&hCurrentEntryNode) < 0 ) return;
			if( get_entry_symbolic(hCurrentEntryNode, val_buf, &buf_len) < 0 ) return;
			sprintf(MSG_STR,STRING_NODE_FMT_STR,val_buf);
			break;
		default:
			sprintf(MSG_STR,STRING_NODE_FMT_STR,"Whoa!?");
			break;
	}
	prt_msg_frag(MSG_STR);
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

// Helper functions for displaying node info

#define print_display_name(hNode) _print_display_name(QSP_ARG  hNode)

static void _print_display_name(QSP_ARG_DECL  spinNodeHandle hNode)
{
	char fmt_str[16];
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;

	assert(max_display_name_len>0);

	if( get_display_name(displayName,&displayNameLength,hNode) < 0 ) return;

	sprintf(fmt_str,"%%-%ds",max_display_name_len+3);
	sprintf(MSG_STR,fmt_str,displayName);
	prt_msg_frag(MSG_STR);
}

#define print_node_type(type) _print_node_type(QSP_ARG  type)

static void _print_node_type(QSP_ARG_DECL  spinNodeType type)
{
	const char *s;

	switch (type) {
		case RegisterNode:	s="register"; break;
		case EnumEntryNode:	s="enum_entry"; break;
		case PortNode:		s="port"; break;
		case BaseNode:		s="base"; break;
		case UnknownNode:	s="unknown"; break;
		case CategoryNode:	s="category"; break;
		case ValueNode:		s="value"; break;
		case StringNode:	s="string"; break;
		case IntegerNode:	s="integer"; break;
		case FloatNode:		s="float"; break;
		case BooleanNode:	s="boolean"; break;
		case CommandNode:	s="command"; break;
		case EnumerationNode:	s="enumeration"; break;
		default:		s="unrecognized!?"; break;
	}
	
	sprintf(MSG_STR,"%-16s",s);
	prt_msg_frag(MSG_STR);
}

#define show_rw_status(hNode) _show_rw_status(QSP_ARG  hNode)

static void _show_rw_status(QSP_ARG_DECL  spinNodeHandle hNode)
{
	if( spink_node_is_readable(hNode) ){
		if( spink_node_is_writable(hNode) ){
			prt_msg("   (read/write)");
		} else {
			prt_msg("   (read-only)");
		}
	} else if( spink_node_is_writable(hNode) ){
			prt_msg("   (write-only)");
	} else {
		prt_msg("   (no read or write access!?)");
	}
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

int _traverse_spink_node_tree(QSP_ARG_DECL  spinNodeHandle hNode, int level, int (*func)(QSP_ARG_DECL spinNodeHandle hNode, int level) )
{
	size_t numberOfFeatures = 0;
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

	if( (*func)(QSP_ARG  hNode,level) < 0 )
		return -1;

	if( get_node_type(hNode,&type) < 0 ) return -1;

	if( type != CategoryNode ) return 0;

	// recurse - assumes a category node
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
		if( traverse_spink_node_tree(hFeatureNode,level+1,func) < 0 ) return -1;
	}
	return 0;
}
//////////////////////////////////////////////

#define display_spink_node(hNode, level) _display_spink_node(QSP_ARG  hNode, level)

static int _display_spink_node(QSP_ARG_DECL  spinNodeHandle hNode, int level)
{
	spinNodeType type;

	indent(level);
	print_display_name(hNode);
	if( get_node_type(hNode,&type) < 0 ) return -1;
	print_node_type(type);
	print_node_value(hNode,type);
	show_rw_status(hNode);
	return 0;
}


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
//	if( release_spink_cam(hCam) < 0 )
//		return -1;
	return 0;
}

#define announce_map(type) _announce_map(QSP_ARG  type)

static void _announce_map(QSP_ARG_DECL  Node_Map_Type type)
{
	switch(type){
		case DEV_NODE_MAP:
			prt_msg("\n*** PRINTING TL DEVICE NODEMAP ***\n");
			break;
		case STREAM_NODE_MAP:
			prt_msg("\n*** PRINTING TL STREAM NODEMAP ***\n");
			break;
		case CAM_NODE_MAP:
			prt_msg("\n*** PRINTING GENICAM NODEMAP ***\n");
			break;

		default:
			error1("announce_map:  invalide map type!?");
			break;
	}
}

#define print_node_map(hCam, skm_p ) _print_node_map(QSP_ARG  hCam, skm_p )

static int _print_node_map(QSP_ARG_DECL  spinCamera hCam, Spink_Map *skm_p )
{
	spinNodeMapHandle hMap = NULL;
	spinNodeHandle hNode = NULL;

	announce_map(skm_p->skm_type);

	// get the map
	if( get_node_map_handle(&hMap,skm_p,"print_node_map") < 0 )
		return -1;

	// Retrieve root node from nodemap
	if( fetch_spink_node(hMap, "Root", &hNode) < 0 ) return -1;

	// Print values recursively
	if( traverse_spink_node_tree(hNode,0,_display_spink_node) < 0 ) return -1;

	return 0;
}

// This function acts as the body of the example. First the TL device and
// TL stream nodemaps are retrieved and their nodes printed. Following this,
// the camera is initialized and then the GenICam node is retrieved
// and its nodes printed.

int _print_camera_nodes(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	spinCamera hCam=NULL;

	assert(skc_p!=NULL);
	//hCam = skc_p->skc_handle;
	if( get_cam_from_list(hCameraList,skc_p->skc_sys_idx,&hCam) < 0 )
		return -1;

	//
	// Retrieve TL device nodemap
	//
	// *** NOTES ***
	// The TL device nodemap is available on the transport layer. As such,
	// camera initialization is unnecessary. It provides mostly immutable
	// information fundamental to the camera such as the serial number,
	// vendor, and model.
	//
	if( print_node_map(hCam, skc_p->skc_dev_map ) < 0 )
		return -1;

	//hNodeMapTLDevice = skc_p->skc_TL_dev_node_map;
	//hNodeMap = skc_p->skc_genicam_node_map;

	//
	// Retrieve TL stream nodemap
	//
	// *** NOTES ***
	// The TL stream nodemap is also available on the transport layer. Camera
	// initialization is again unnecessary. As you can probably guess, it
	// provides information on the camera's streaming performance at any
	// given moment. Having this information available on the transport
	// layer allows the information to be retrieved without affecting camera
	// performance.
	//
	if( print_node_map(hCam, skc_p->skc_stream_map ) < 0 )
		return -1;

	//
	// Retrieve GenICam nodemap
	//
	// *** NOTES ***
	// The GenICam nodemap is the primary gateway to customizing and
	// configuring the camera to suit your needs. Configuration options such
	// as image height and width, trigger mode enabling and disabling, and the
	// sequencer are found on this nodemap.
	//
	if( print_node_map(hCam, skc_p->skc_cam_map ) < 0 )
		return -1;

	//
	// Deinitialize camera
	//
	// *** NOTES ***
	// Camera deinitialization helps ensure that devices clean up properly
	// and do not need to be power-cycled to maintain integrity.
	//
	if( camera_deinit(hCam) < 0 ) return -1;

	return 0;
}

void _list_nodes_from_map(QSP_ARG_DECL  Spink_Map *skm_p)
{
	list_spink_nodes( tell_msgfile() );
}

void _print_spink_node_info(QSP_ARG_DECL  spinNodeHandle hNode)
{
	assert(hNode != NULL);

	// The saved handles seem to go stale!?
	if( ! spink_node_is_available(hNode) ){
		report_node_access_error(hNode,"available");
		return;
	}

	display_spink_node(hNode, 1);
}

#endif // HAVE_LIBSPINNAKER


