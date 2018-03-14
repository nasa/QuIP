#include "quip_config.h"
#include <stdio.h>
#include <string.h>
#include "quip_prot.h"
#include "spink.h"

#ifdef HAVE_LIBSPINNAKER

// Use the following enum and global constant to select whether nodes are read
// as 'value' nodes or their individual types.
typedef enum _readType {
	VALUE,
	INDIVIDUAL
} readType;

const readType chosenRead = VALUE;

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
	//const unsigned int k_maxChars = MAX_NODE_CHARS;
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

// This function retrieves and prints the display name and value of an integer
// node.
//
// Retrieve integer node value
//
// *** NOTES ***
// Keep in mind that the data type of an integer node value is an
// int64_t as opposed to a standard int. While it is true that the two
// are often interchangeable, it is recommended to use the int64_t
// to avoid the introduction of bugs into software built with the
// Spinnaker SDK.
//

#define print_int_node(hNode, level) _print_int_node(QSP_ARG  hNode, level)

static int _print_int_node(QSP_ARG_DECL  spinNodeHandle hNode, unsigned int level)
{
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;
	int64_t integerValue = 0;

	if( get_display_name(displayName,&displayNameLength,hNode) < 0 ) return -1;
	if( get_int_value(hNode, &integerValue) < 0 ) return -1;

	// Print value
	indent(level);
	sprintf(MSG_STR,"%s: %ld", displayName, integerValue);
	prt_msg(MSG_STR);

	return 0;
}

// This function retrieves and prints the display name and value of a float node.
//
// Retrieve float node value
//
// *** NOTES ***
// Please take note that floating point numbers in the Spinnaker SDK are
// almost always represented by the larger data type double rather than
// float.
//

#define print_float_node(hNode, level) _print_float_node(QSP_ARG  hNode, level)

static int _print_float_node(QSP_ARG_DECL  spinNodeHandle hNode, unsigned int level)
{
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;
	double floatValue = 0.0;

	if( get_display_name(displayName,&displayNameLength,hNode) < 0 ) return -1;
	if( get_float_value(hNode,&floatValue) < 0 ) return -1;

	// Print value
	indent(level);
	sprintf(MSG_STR,"%s:  %f\n", displayName, floatValue);
	prt_msg(MSG_STR);

	return 0;
}

// This function retrieves and prints the display name and value of a boolean,
// printing "true" for true and "false" for false rather than the corresponding
// integer value ('1' and '0', respectively).
//
// Retrieve value as a string representation
//
// *** NOTES ***
// Boolean node type values are represented by the standard bool data
// type. The boolean ToString() method returns either a '1' or '0' as a
// a string rather than a more descriptive word like 'true' or 'false'.
//

#define print_bool_node(hNode, level) _print_bool_node(QSP_ARG  hNode, level)

static int _print_bool_node(QSP_ARG_DECL  spinNodeHandle hNode, unsigned int level)
{
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;
	bool8_t booleanValue = False;

	if( get_display_name(displayName,&displayNameLength,hNode) < 0 ) return -1;
	if( get_bool_value(hNode,&booleanValue) < 0 ) return -1;

	indent(level);
	sprintf(MSG_STR,"%s: %s\n", displayName, (booleanValue ? "true" : "false"));
	prt_msg(MSG_STR);

	return 0;
}

// This function retrieves and prints the display name and tooltip of a command
// node, limiting the number of printed characters to a macro-defined maximum.
// The tooltip is printed below as command nodes do not have an intelligible
// value.
//
// Retrieve tooltip
//
// *** NOTES ***
// All node types have a tooltip available. Tooltips provide useful
// information about nodes. Command nodes do not have a method to
// retrieve values as their is no intelligible value to retrieve.
//

#define print_cmd_node(hNode, level) _print_cmd_node(QSP_ARG  hNode, level)

static int _print_cmd_node(QSP_ARG_DECL  spinNodeHandle hNode, unsigned int level)
{
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;
	unsigned int i = 0;
	char toolTip[MAX_BUFF_LEN];
	size_t toolTipLength = MAX_BUFF_LEN;
	const unsigned int k_maxChars = MAX_NODE_CHARS;

	if( get_display_name(displayName,&displayNameLength,hNode) < 0 ) return -1;
	if( get_tip_value(hNode,toolTip,&toolTipLength) < 0 ) return -1;

	// Print tooltip
	indent(level);
	sprintf(MSG_STR,"%s: ", displayName);
	prt_msg_frag(MSG_STR);

	// Ensure that the value length is not excessive for printing

	if (toolTipLength > k_maxChars) {
		for (i = 0; i < k_maxChars; i++) {
			printf("%c", toolTip[i]);
		}
		prt_msg("...");
	} else {
		sprintf(MSG_STR,"%s", toolTip);
		prt_msg(MSG_STR);
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

#define print_enum_node(hEnumerationNode, level) _print_enum_node(QSP_ARG  hEnumerationNode, level)

static int _print_enum_node(QSP_ARG_DECL  spinNodeHandle hEnumerationNode, unsigned int level)
{
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;
	spinNodeHandle hCurrentEntryNode = NULL;
	char currentEntrySymbolic[MAX_BUFF_LEN];
	size_t currentEntrySymbolicLength = MAX_BUFF_LEN;

	if( get_display_name(displayName,&displayNameLength,hEnumerationNode) < 0 ) return -1;
	if( get_current_entry(hEnumerationNode,&hCurrentEntryNode) < 0 ) return -1;
	if( get_entry_symbolic(hCurrentEntryNode, currentEntrySymbolic, &currentEntrySymbolicLength) < 0 ) return -1;

	// Print current entry symbolic
	indent(level);
	sprintf(MSG_STR,"%s: %s", displayName, currentEntrySymbolic);
	prt_msg(MSG_STR);

	return 0;
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

int _traverse_spink_node_tree(QSP_ARG_DECL  spinNodeHandle hCategoryNode, int level, int (*func)(QSP_ARG_DECL spinNodeHandle hNode, int level) )
{
	size_t numberOfFeatures = 0;
	unsigned int i = 0;

	if( ! spink_node_is_implemented(hCategoryNode) ){
		report_node_access_error(hCategoryNode,"implemented");
		return 0;
	}

	if( ! spink_node_is_available(hCategoryNode) ){
		report_node_access_error(hCategoryNode,"available");
		return 0;
	}

	if( (*func)(QSP_ARG  hCategoryNode,level) < 0 )
		return -1;

	// recurse
	if( get_n_features(hCategoryNode,&numberOfFeatures) < 0 ) return -1;

	for (i = 0; i < numberOfFeatures; i++) {
		spinNodeHandle hFeatureNode = NULL;
		spinNodeType type = UnknownNode;

		if( get_feature_by_index(hCategoryNode, i, &hFeatureNode) < 0 ) return -1;

		if( ! spink_node_is_implemented(hCategoryNode) ){
			report_node_access_error(hCategoryNode,"implemented");
			continue;
		}

		if( ! spink_node_is_available(hFeatureNode) ){
			report_node_access_error(hFeatureNode,"available");
			continue;
		}

		if( ! spink_node_is_readable(hFeatureNode) ){
			report_node_access_error(hFeatureNode,"readable");
			continue;
		}

		if( get_node_type(&type,hFeatureNode) < 0 ) return -1;

		if (type == CategoryNode) {
			if( traverse_spink_node_tree(hFeatureNode,level+1,func) < 0 ) return -1;
		} else {
			if( (*func)(QSP_ARG  hFeatureNode,level+1) < 0 )
				return -1;
		}
	}
	return 0;
}
//////////////////////////////////////////////


// This function retrieves and prints the display name and value of all node
// types as value nodes. A value node is a general node type that allows for
// the reading and writing of any node type as a string.

int _print_value_node(QSP_ARG_DECL  spinNodeHandle hNode, unsigned int level)
{
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;
	char value[MAX_BUFF_LEN];
	size_t valueLength = MAX_BUFF_LEN;
	spinNodeType type;

	if( get_node_type(&type,hNode) < 0 ) return -1;

	if( get_display_name(displayName,&displayNameLength,hNode) < 0 ) return -1;

	indent(level);
	if( type == CategoryNode ){
		sprintf(MSG_STR,"%s", displayName);
	} else {
		if( get_node_value_string(value,&valueLength,hNode) < 0 ) return -1;
		sprintf(MSG_STR,"%s:  %s", displayName,value);
	}
	prt_msg(MSG_STR);

	return 0;
}

// This function retrieves and prints the display name and value of a string
// node, limiting the number of printed characters to a maximum defined
// by MAX_NODE_CHARS macro.
int _print_string_node(QSP_ARG_DECL  spinNodeHandle hNode, unsigned int level)
{
	// Retrieve display name
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;
	char stringValue[MAX_BUFF_LEN];
	size_t stringValueLength = MAX_BUFF_LEN;

	if( get_display_name(displayName,&displayNameLength,hNode) < 0 ) return -1;
	if( get_string_node_string(stringValue,&stringValueLength,hNode) < 0 ) return -1;

	// Print value
	indent(level);
	sprintf(MSG_STR,"%s:  %s", displayName,stringValue);
	prt_msg(MSG_STR);
	return 0;
}

#define display_spink_node(hNode, level) _display_spink_node(QSP_ARG  hNode, level)

static int _display_spink_node(QSP_ARG_DECL  spinNodeHandle hNode, int level)
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


int _get_node_map_handle(QSP_ARG_DECL  spinNodeMapHandle *hMap_p, Spink_Map *skm_p, const char *whence)
{
	spinCamera hCam;

	if( get_spink_cam_from_list(&hCam,hCameraList,skm_p->skm_skc_p->skc_sys_idx) < 0 )
		return -1;

	switch( skm_p->skm_type ){
		case INVALID_NODE_MAP:
		case N_NODE_MAP_TYPES:
			sprintf(ERROR_STRING,"get_node_map_handle (%s):  invalid type code!?",whence);
			warn(ERROR_STRING);
			break;
		case CAM_NODE_MAP:
			if( ! IS_CONNECTED(skm_p->skm_skc_p) ){
fprintf(stderr,"init_one_spink_cam:  connecting %s\n",skm_p->skm_skc_p->skc_name);
				if( connect_spink_cam(hCam) < 0 ) return -1;
			} else {
fprintf(stderr,"init_one_spink_cam:  %s is already connected\n",skm_p->skm_skc_p->skc_name);
			}

			if( get_camera_node_map(hMap_p, hCam ) < 0 ){
				sprintf(ERROR_STRING,
			"get_node_map_handle (%s):  error getting camera node map!?",whence);
				error1(ERROR_STRING);
			}
			break;
		case DEV_NODE_MAP:
			if( get_device_node_map(hMap_p, hCam ) < 0 ){
				sprintf(ERROR_STRING,
			"get_node_map_handle (%s):  error getting device node map!?",whence);
				error1(ERROR_STRING);
			}
			break;
		case STREAM_NODE_MAP:
			if( get_stream_node_map(hMap_p, hCam ) < 0 ){
				sprintf(ERROR_STRING,
			"get_node_map_handle (%s):  error getting stream node map!?",whence);
				error1(ERROR_STRING);
			}
			break;
	}
	if( release_spink_cam(hCam) < 0 )
		return -1;
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
	if( get_spink_cam_from_list(&hCam,hCameraList,skc_p->skc_sys_idx) < 0 )
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

void _print_spink_node_info(QSP_ARG_DECL /*Spink_Map *skm_p,*/ Spink_Node *skn_p)
{
	assert(skn_p->skn_handle != NULL);

	// The saved handles seem to go stale!?
	if( ! spink_node_is_available(skn_p->skn_handle) ){
		report_node_access_error(skn_p->skn_handle,"available");
		return;
	}

	display_spink_node(skn_p->skn_handle, 1);
}

#endif // HAVE_LIBSPINNAKER


