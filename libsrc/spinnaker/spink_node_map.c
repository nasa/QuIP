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

#define get_node_type(type_p, hNode) _get_node_type(QSP_ARG  type_p, hNode)

static int _get_node_type(QSP_ARG_DECL  spinNodeType *type_p, spinNodeHandle hNode)
{
	spinError err;

	err = spinNodeGetType(hNode, type_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinNodeGetType");
		return -1;
	}
	return 0;
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
	spinError err;

	// Ensure allocated buffer is large enough for storing the string
	err = spinStringGetValue(hNode, NULL, &n_need);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinStringGetValue");
		return -1;
	}

	if(n_need <= *buflen_p) {
		err = spinNodeToString(hNode, buf, buflen_p);
		if (err != SPINNAKER_ERR_SUCCESS) {
			report_spink_error(err,"spinNodeToString (get_string_node_string)");
			return -1;
		}
	} else {
		strcpy(buf,"(...)");
	}
	return 0;
}

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
	spinError err;
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;
	int64_t integerValue = 0;

	if( get_display_name(displayName,&displayNameLength,hNode) < 0 ) return -1;

	err = spinIntegerGetValue(hNode, &integerValue);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinIntegerGetValue");
		return -1;
	}

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
	spinError err;
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;
	double floatValue = 0.0;

	if( get_display_name(displayName,&displayNameLength,hNode) < 0 ) return -1;

	err = spinFloatGetValue(hNode, &floatValue);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinFloatGetValue");
		return -1;
	}

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
	spinError err;
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;
	bool8_t booleanValue = False;

	if( get_display_name(displayName,&displayNameLength,hNode) < 0 ) return -1;

	err = spinBooleanGetValue(hNode, &booleanValue);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinBooleanGetValue");
		return -1;
	}

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
	spinError err;
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;
	unsigned int i = 0;
	char toolTip[MAX_BUFF_LEN];
	size_t toolTipLength = MAX_BUFF_LEN;
	const unsigned int k_maxChars = MAX_NODE_CHARS;

	if( get_display_name(displayName,&displayNameLength,hNode) < 0 ) return -1;

	err = spinNodeGetToolTip(hNode, toolTip, &toolTipLength);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinNodeGetToolTip");
		return -1;
	}

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

	return err;
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
	spinError err;
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;
	spinNodeHandle hCurrentEntryNode = NULL;
	char currentEntrySymbolic[MAX_BUFF_LEN];
	size_t currentEntrySymbolicLength = MAX_BUFF_LEN;

	if( get_display_name(displayName,&displayNameLength,hEnumerationNode) < 0 ) return -1;

	err = spinEnumerationGetCurrentEntry(hEnumerationNode, &hCurrentEntryNode);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinEnumerationGetCurrentEntry");
		return -1;
	}

	err = spinEnumerationEntryGetSymbolic(hCurrentEntryNode, currentEntrySymbolic, &currentEntrySymbolicLength);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinEnumerationEntryGetSymbolic");
		return -1;
	}

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
	spinError err;

	err = spinNodeGetDisplayName(hdl, buf, len_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinNodeGetDisplayName");
		return -1;
	}
	return 0;
}

int _get_node_name(QSP_ARG_DECL  char *buf, size_t *len_p, spinNodeHandle hdl)
{
	spinError err;

	err = spinNodeGetName(hdl, buf, len_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinNodeGetName");
		return -1;
	}
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

int _traverse_spink_node_tree(QSP_ARG_DECL  spinNodeHandle hCategoryNode, int level, int (*func)(QSP_ARG_DECL spinNodeHandle hNode, int level) )
{
	size_t numberOfFeatures = 0;
	unsigned int i = 0;
	spinError err = SPINNAKER_ERR_SUCCESS;

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

	err = spinCategoryGetNumFeatures(hCategoryNode, &numberOfFeatures);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCategoryGetNumFeatures");
		return -1;
	}

	for (i = 0; i < numberOfFeatures; i++) {
		spinNodeHandle hFeatureNode = NULL;
		spinNodeType type = UnknownNode;

		// Retrieve child
		err = spinCategoryGetFeatureByIndex(hCategoryNode, i, &hFeatureNode);
		if (err != SPINNAKER_ERR_SUCCESS) {
			report_spink_error(err,"spinCategoryGetFeatureByIndex");
			return -1;
		}

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

int _get_device_node_map(QSP_ARG_DECL  spinNodeMapHandle *map_p, spinCamera hCam )
{
	spinError err;

	// Retrieve nodemap from camera
	err = spinCameraGetTLDeviceNodeMap(hCam, map_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraGetTLDeviceNodeMap");
		return -1;
	}
	return 0;
}

int _get_stream_node_map(QSP_ARG_DECL  spinNodeMapHandle *map_p, spinCamera hCam )
{
	spinError err;

	// Retrieve nodemap from camera
	err = spinCameraGetTLStreamNodeMap(hCam, map_p);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraGetTLStreamNodeMap");
		return -1;
	}
	return 0;
}

int _get_camera_node_map(QSP_ARG_DECL  spinNodeMapHandle *map_p, spinCamera hCam )
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

int _refresh_node_map_handle(QSP_ARG_DECL  Spink_Map *skm_p, const char *whence)
{
	spinNodeMapHandle hMap=NULL;

	switch( skm_p->skm_type ){
		case INVALID_NODE_MAP:
		case N_NODE_MAP_TYPES:
			sprintf(ERROR_STRING,"refresh_node_map_handle (%s):  invalid type code!?",whence);
			warn(ERROR_STRING);
			break;
		case CAM_NODE_MAP:
			if( get_camera_node_map(&hMap, skm_p->skm_skc_p->skc_handle ) < 0 ){
				sprintf(ERROR_STRING,
			"refresh_node_map_handle (%s):  error getting camera node map!?",whence);
				error1(ERROR_STRING);
			}
			break;
		case DEV_NODE_MAP:
			if( get_device_node_map(&hMap, skm_p->skm_skc_p->skc_handle ) < 0 ){
				sprintf(ERROR_STRING,
			"refresh_node_map_handle (%s):  error getting device node map!?",whence);
				error1(ERROR_STRING);
			}
			break;
		case STREAM_NODE_MAP:
			if( get_stream_node_map(&hMap, skm_p->skm_skc_p->skc_handle ) < 0 ){
				sprintf(ERROR_STRING,
			"refresh_node_map_handle (%s):  error getting stream node map!?",whence);
				error1(ERROR_STRING);
			}
			break;
	}
	if( skm_p->skm_handle != NULL ){
		if( hMap != skm_p->skm_handle ){
			sprintf(ERROR_STRING,"refresh_node_map_handle (%s):  handle value changed!?",whence);
			advise(ERROR_STRING);
fprintf(stderr,"\tnew_hdl = 0x%lx,   old_hdl = 0x%lx\n",(u_long)hMap,(u_long)skm_p->skm_handle);
fflush(stderr);
fprintf(stderr,"\t*new_hdl = 0x%lx,   *old_hdl = 0x%lx\n",(u_long)(*((void **)hMap)),(u_long)(*((void **)skm_p->skm_handle)));
fflush(stderr);
		}
	}
	skm_p->skm_handle = hMap;
	return 0;
}

// This function acts as the body of the example. First the TL device and
// TL stream nodemaps are retrieved and their nodes printed. Following this,
// the camera is initialized and then the GenICam node is retrieved
// and its nodes printed.

int _get_camera_nodes(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	spinCamera hCam=NULL;
	spinNodeHandle hTLDeviceRoot = NULL;
	spinNodeMapHandle hNodeMapTLDevice = NULL;
	spinNodeMapHandle hNodeMap = NULL;
	spinNodeMapHandle hNodeMapStream = NULL;
	spinError err = SPINNAKER_ERR_SUCCESS;
	spinNodeHandle hStreamRoot = NULL;
	spinNodeHandle hRoot = NULL;



	assert(skc_p!=NULL);
	hCam = skc_p->skc_handle;
	hNodeMapTLDevice = skc_p->skc_TL_dev_node_map;
	hNodeMap = skc_p->skc_genicam_node_map;

	//
	// Retrieve TL device nodemap
	//
	// *** NOTES ***
	// The TL device nodemap is available on the transport layer. As such,
	// camera initialization is unnecessary. It provides mostly immutable
	// information fundamental to the camera such as the serial number,
	// vendor, and model.
	//
	prt_msg("\n*** PRINTING TL DEVICE NODEMAP ***\n");

	// Retrieve root node from nodemap
	if( fetch_spink_node(hNodeMapTLDevice, "Root", &hTLDeviceRoot) < 0 ) return -1;

	// Print values recursively
	if( traverse_spink_node_tree(hTLDeviceRoot,0,_display_spink_node) < 0 ) return -1;


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
	prt_msg("*** PRINTING TL STREAM NODEMAP ***\n");

	if( get_stream_node_map(&hNodeMapStream,hCam) < 0 ) return -1;

	if( fetch_spink_node(hNodeMapStream, "Root", &hStreamRoot) < 0 ) return -1;

	// Print values recursively
	if( traverse_spink_node_tree(hStreamRoot,0,_display_spink_node) < 0 ) return -1;

	//
	// Retrieve GenICam nodemap
	//
	// *** NOTES ***
	// The GenICam nodemap is the primary gateway to customizing and
	// configuring the camera to suit your needs. Configuration options such
	// as image height and width, trigger mode enabling and disabling, and the
	// sequencer are found on this nodemap.
	//
	prt_msg("*** PRINTING GENICAM NODEMAP ***\n");

	hNodeMap = skc_p->skc_genicam_node_map;

	// Retrieve root node from nodemap
	if( fetch_spink_node(hNodeMap, "Root", &hRoot) < 0 ) return -1;

	// Print values recursively
	if( traverse_spink_node_tree(hRoot,0,_display_spink_node) < 0 ) return -1;

	//
	// Deinitialize camera
	//
	// *** NOTES ***
	// Camera deinitialization helps ensure that devices clean up properly
	// and do not need to be power-cycled to maintain integrity.
	//
	err = spinCameraDeInit(hCam);
	if (err != SPINNAKER_ERR_SUCCESS) {
		report_spink_error(err,"spinCameraDeInit");
		return -1;
	}

	return 0;
}

void _list_nodes_from_map(QSP_ARG_DECL  Spink_Map *skm_p)
{
	list_spink_nodes( tell_msgfile() );
}

void _print_spink_node_info(QSP_ARG_DECL /*Spink_Map *skm_p,*/ Spink_Node *skn_p)
{
	// The saved handles seem to go stale!?
	if( ! spink_node_is_available(skn_p->skn_handle) ){
		report_node_access_error(skn_p->skn_handle,"available");
		return;
	}

	display_spink_node(skn_p->skn_handle, 1);
}

#endif // HAVE_LIBSPINNAKER


