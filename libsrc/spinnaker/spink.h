#ifndef _SPINK_H_
#define _SPINK_H_

#ifdef HAVE_CONFIG_H
#include "quip_config.h"
#include "data_obj.h"
#include "fio_api.h"
#else // ! HAVE_CONFIG_H
#endif // ! HAVE_CONFIG_H

//#define MAX_DEBUG	// enables WRAPPER_REPORT

#ifdef HAVE_LIBSPINNAKER
#include "SpinnakerC.h"
#endif // HAVE_LIBSPINNAKER

#define BOOL	bool8_t

#define SPINK_DEBUG_MSG(s)

#include "spink_funcs.h"

// jbm made these limits up...
// We should do something sensible here, but it is difficult
// because we don't necessarily know how many buffers we can
// allocate.  USBFS has a default limit of 16MB, but on euler
// we have increased it to 200MB (in /etc/default/grub, see PGR
// TAN, and pointed out by Brian Cha).
// [the above comment originally from the flycap implementation, but should apply to Spinnaker???]
//
#define MIN_N_BUFFERS 2
//#define MAX_N_BUFFERS 1024
#define MAX_N_BUFFERS 64

// Compiler warning C4996 suppressed due to deprecated strcpy() and sprintf()
// functions on Windows platform.
#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
    #pragma warning(disable : 4996)
#endif

// This macro helps with C-strings.
#define MAX_BUFF_LEN 256

#define MAX_NODE_VALUE_CHARS_TO_PRINT	24	// must be less than LLEN !!

// BUG - if the constant above is changed, these definitions should be changed
// to match...   We could generate them programmatically, but that's extra work
// that we will skip for now.

// 24 chars total, subtract 6 for spaces and parens leaves 18, or 6 digits per... is that enough?
// NOT if we need 6 more characters for hex!?
// used to be %-24d and %-24f
#define INT_NODE_DEC_FMT_STR	"%-6ld (%-6ld - %-6ld)"
#define INT_NODE_HEX_FMT_STR	"0x%-6lx (0x%-6lx - 0x%-6lx)"
#define FLT_NODE_FMT_STR	"%-6f (%-6f - %-6f)"
#define STRING_NODE_FMT_STR	"%-24s"

// a couple of globals...
#ifdef HAVE_LIBSPINNAKER
extern spinCameraList hCameraList;
#endif // HAVE_LIBSPINNAKER
extern size_t numCameras;
extern int current_node_idx;

// forward declarations
struct spink_map;
struct spink_node;

// a global - used for formatting the print-out of nodes
extern int max_display_name_len;

typedef enum {
	INVALID_CHUNK_DATA_TYPE,
	FLOAT_CHUNK_DATA,
	INT_CHUNK_DATA
} Chunk_Data_Type;

#define CHUNK_SELECTOR_ENUM_PREFIX	"EnumEntry_ChunkSelector_"

typedef struct chunk_data {
	const char *	cd_name;
	Chunk_Data_Type	cd_type;
	union {
		double u_fltval;
		int64_t u_intval;
	} cd_u;
} Chunk_Data;

ITEM_INTERFACE_PROTOTYPES(Chunk_Data,chunk_data)

#define pick_chunk_data(s)	_pick_chunk_data(QSP_ARG  s)
#define new_chunk_data(s)	_new_chunk_data(QSP_ARG  s)
#define init_chunk_datas()	_init_chunk_datas(SINGLE_QSP_ARG)
#define list_chunk_datas(fp)	_list_chunk_datas(QSP_ARG  fp)

typedef struct spink_node_type {
	const char *		snt_name;
#ifdef HAVE_LIBSPINNAKER
	spinNodeType		snt_type;
#endif // HAVE_LIBSPINNAKER
	void (*snt_set_func)(QSP_ARG_DECL  struct spink_node *skn_p);
	void (*snt_print_value_func)(QSP_ARG_DECL  struct spink_node *skn_p);
} Spink_Node_Type;

ITEM_INTERFACE_PROTOTYPES(Spink_Node_Type,spink_node_type)
#define init_spink_node_types()	_init_spink_node_types(SINGLE_QSP_ARG)
#define spink_node_type_list()	_spink_node_type_list(SINGLE_QSP_ARG)
#define new_spink_node_type(name)	_new_spink_node_type(QSP_ARG  name)

typedef struct {
	int64_t	min;
	int64_t	max;
} Spink_Int_Node_Data;

typedef struct {
	double	min;
	double	max;
} Spink_Float_Node_Data;

typedef struct {
	// We tried using enum values instead of int values, but not all nodes support that...
	int64_t			enum_ival;	// only used by enum_entry nodes...
} Spink_EnumEntry_Node_Data;

#define INVALID_ENUM_INT_VALUE	(-1)
#define skn_enum_ival	skn_data.u_enum_data.enum_ival

typedef union {
	Spink_Int_Node_Data		u_int_data;
	Spink_Float_Node_Data		u_float_data;
	Spink_EnumEntry_Node_Data	u_enum_data;
} Spink_Node_Data;

#define FLOAT_NODE_MIN_VAL(skn_p)	(skn_p)->skn_data.u_float_data.min
#define FLOAT_NODE_MAX_VAL(skn_p)	(skn_p)->skn_data.u_float_data.max
#define INT_NODE_MIN_VAL(skn_p)		(skn_p)->skn_data.u_int_data.min
#define INT_NODE_MAX_VAL(skn_p)		(skn_p)->skn_data.u_int_data.max

#define SET_FLOAT_NODE_MIN_VAL(skn_p,v)	(skn_p)->skn_data.u_float_data.min = v
#define SET_FLOAT_NODE_MAX_VAL(skn_p,v)	(skn_p)->skn_data.u_float_data.max = v
#define SET_INT_NODE_MIN_VAL(skn_p,v)	(skn_p)->skn_data.u_int_data.min = v
#define SET_INT_NODE_MAX_VAL(skn_p,v)	(skn_p)->skn_data.u_int_data.max = v

// It's kind of wasteful to duplicate information that is present
// in the SDK node structure...  but it's not a lot of storage so
// for now we won't worry about it.

typedef struct spink_node {
	const char *		skn_name;
	struct spink_map *	skn_skm_p;
	struct spink_node *	skn_parent;
	int			skn_idx;	// within the parent
	int			skn_level;	// tree depth
	int			skn_flags;
	Spink_Node_Type *	skn_type_p;
	Spink_Node_Data		skn_data;
	List *			skn_children;

} Spink_Node;

// flag bits
#define NODE_READABLE	1
#define NODE_WRITABLE	2

#define NODE_IS_READABLE(skn_p)	((skn_p)->skn_flags & NODE_READABLE)
#define NODE_IS_WRITABLE(skn_p)	((skn_p)->skn_flags & NODE_WRITABLE)

ITEM_INTERFACE_PROTOTYPES(Spink_Node,spink_node)

#define new_spink_node(s)	_new_spink_node(QSP_ARG  s)
#define del_spink_node(skn_p)	_del_spink_node(QSP_ARG  skn_p)
#define get_spink_node(s)	_get_spink_node(QSP_ARG  s)
#define spink_node_of(s)	_spink_node_of(QSP_ARG  s)
#define init_spink_nodes()	_init_spink_nodes(SINGLE_QSP_ARG)
#define list_spink_nodes(fp)	_list_spink_nodes(QSP_ARG  fp)
#define pick_spink_node(s)	_pick_spink_node(QSP_ARG  s)
#define spink_node_list()	_spink_node_list(SINGLE_QSP_ARG)

typedef struct spink_cat {
	const char *	sct_name;
	Spink_Node *	sct_root_p;
} Spink_Category;

ITEM_INTERFACE_PROTOTYPES(Spink_Category,spink_cat)

#define new_spink_cat(s)	_new_spink_cat(QSP_ARG  s)
#define del_spink_cat(p)	_del_spink_cat(QSP_ARG  p)
#define spink_cat_of(s)		_spink_cat_of(QSP_ARG  s)
#define init_spink_cats()	_init_spink_cats(SINGLE_QSP_ARG)
#define list_spink_cats(fp)	_list_spink_cats(QSP_ARG  fp)
#define pick_spink_cat(s)	_pick_spink_cat(QSP_ARG  s)
#define spink_cat_list()	_spink_cat_list(SINGLE_QSP_ARG)


typedef struct spink_enum_val {
	const char *	sev_name;
	size_t		sev_value;
} Spink_Enum_Val;

ITEM_INTERFACE_PROTOTYPES(Spink_Enum_Val,spink_enum_val)

#define new_spink_enum_val(s)		_new_spink_enum_val(QSP_ARG  s)
#define spink_enum_val_of(s)		_spink_enum_val_of(QSP_ARG  s)
#define init_spink_enum_vals()		_init_spink_enum_vals(SINGLE_QSP_ARG)
#define list_spink_enum_vals(fp)	_list_spink_enum_vals(QSP_ARG  fp)
#define pick_spink_enum_val(s)		_pick_spink_enum_val(QSP_ARG  s)
#define spink_enum_val_list()		_spink_enum_val_list(SINGLE_QSP_ARG)


typedef enum {
	INVALID_NODE_MAP,
	CAM_NODE_MAP,
	DEV_NODE_MAP,
	STREAM_NODE_MAP,
	N_NODE_MAP_TYPES
} Node_Map_Type;

struct spink_cam;

typedef struct spink_map {
	const char *		skm_name;
	struct spink_cam *	skm_skc_p;
	Node_Map_Type		skm_type;
	Spink_Node *		skm_root_p;
	Item_Context *		skm_node_icp;
	Item_Context *		skm_cat_icp;
} Spink_Map;

ITEM_INTERFACE_PROTOTYPES(Spink_Map,spink_map)

#define new_spink_map(s)	_new_spink_map(QSP_ARG  s)
#define del_spink_map(s)	_del_spink_map(QSP_ARG  s)
#define spink_map_of(s)		_spink_map_of(QSP_ARG  s)
#define list_spink_maps(fp)	_list_spink_maps(QSP_ARG  fp)
#define pick_spink_map(s)	_pick_spink_map(QSP_ARG  s)
#define spink_map_list()	_spink_map_list(SINGLE_QSP_ARG)

typedef struct my_event_info {
#ifdef THREAD_SAFE_QUERY
	Query_Stack *	ei_qsp;
#endif // THREAD_SAFE_QUERY
	struct spink_cam *	ei_skc_p;
	int		ei_next_frame;
	int		ei_n_frames;
} Image_Event_Info;

typedef enum {
	FRAME_STATUS_AVAILABLE,
	FRAME_STATUS_IN_USE
} Grab_Frame_Status;

typedef struct grab_frame_info {
	Data_Obj *		gfi_dp;
	Grab_Frame_Status	gfi_status;
	spinImage		gfi_hImage;	// used to be placed in OBJ_EXTRA...
} Grab_Frame_Info;

#define N_FMT7_MODES	5	// PGR has 32, but why bother?

typedef int Framerate_Mask;

typedef struct spink_cam {
	const char *		skc_name;
	int			skc_sys_idx;
	int			skc_iface_idx;
	struct spink_map *	skc_dev_map;
	struct spink_map *	skc_cam_map;
	struct spink_map *	skc_stream_map;
	Item_Context *		skc_chunk_icp;
#ifdef HAVE_LIBSPINNAKER
	spinCamera		skc_current_handle;	// non-NULL if we are holding a handle -
							// set to NULL when released...
#endif /* HAVE_LIBSPINNAKER */

	unsigned int		skc_cols;
	unsigned int		skc_rows;
	unsigned int		skc_depth;	// bytes per pixel
	uint64_t		skc_bytes_per_image;
	int			skc_n_buffers;
	int			skc_newest;
	int			skc_oldest;
	//Data_Obj **		skc_frm_dp_tbl;
	Grab_Frame_Info *	skc_gfi_tbl;

//	Item_Context *		skc_do_icp;	// data_obj context
//	List *			skc_in_use_lp;	// list of frames...
//	List *			skc_feat_lp;
	u_long			skc_flags;

	u_char *		skc_base;	// buffers?
	long			skc_buf_delta;	// assumes evenly spaced buffers???

// information for image event handlers...

	Image_Event_Info	skc_event_info;
} Spink_Cam;

ITEM_INTERFACE_PROTOTYPES(Spink_Cam,spink_cam)

#define new_spink_cam(s)	_new_spink_cam(QSP_ARG  s)
#define del_spink_cam(s)	_del_spink_cam(QSP_ARG  s)
#define spink_cam_of(s)	_spink_cam_of(QSP_ARG  s)
#define list_spink_cams(fp)	_list_spink_cams(QSP_ARG  fp)
#define pick_spink_cam(s)	_pick_spink_cam(QSP_ARG  s)
#define spink_cam_list()	_spink_cam_list(SINGLE_QSP_ARG)

/* flag bits */

#define SPINK_CAM_CONNECTED		1
#define SPINK_CAM_RUNNING		2
#define SPINK_CAM_CAPTURING		4
#define SPINK_CAM_TRANSMITTING		8
#define SPINK_CAM_EVENTS_READY		16
#define SPINK_CAM_CAPT_REQUESTED	32

#define IS_CONNECTED(skc_p)	(skc_p->skc_flags & SPINK_CAM_CONNECTED)
#define IS_RUNNING(skc_p)	(skc_p->skc_flags & SPINK_CAM_RUNNING)
#define IS_CAPTURING(skc_p)	(skc_p->skc_flags & SPINK_CAM_CAPTURING)
#define CAPTURE_REQUESTED(skc_p)	(skc_p->skc_flags & SPINK_CAM_CAPT_REQUESTED)
#define IS_TRANSMITTING(skc_p)	(skc_p->skc_flags & SPINK_CAM_TRANSMITTING)
#define IS_EVENTFUL(skc_p)	(skc_p->skc_flags & SPINK_CAM_EVENTS_READY)

typedef struct spink_interface {
	const char *		ski_name;
	int			ski_idx;
} Spink_Interface;

ITEM_INTERFACE_PROTOTYPES(Spink_Interface,spink_interface)
#define new_spink_interface(s)		_new_spink_interface(QSP_ARG  s)
#define del_spink_interface(p)		_del_spink_interface(QSP_ARG  p)
#define spink_interface_of(s)		_spink_interface_of(QSP_ARG  s)
#define list_spink_interfaces(fp)	_list_spink_interfaces(QSP_ARG  fp)
#define pick_spink_interface(s)		_pick_spink_interface(QSP_ARG  s)
#define spink_interface_list()		_spink_interface_list(SINGLE_QSP_ARG)

// spink_enum.c
#ifdef HAVE_LIBSPINNAKER
extern int _get_camera_model_name(QSP_ARG_DECL  char *buf, size_t buflen, spinCamera hCam);
#define get_camera_model_name(buf, buflen, map) _get_camera_model_name(QSP_ARG  buf, buflen, map)
extern int _get_camera_vendor_name(QSP_ARG_DECL  char *buf, size_t buflen, spinCamera hCam);
#define get_camera_vendor_name(buf, buflen, hCam) _get_camera_vendor_name(QSP_ARG  buf, buflen, hCam)

extern int _get_interface_name(QSP_ARG_DECL  char *buf, size_t buflen, spinInterface hInterface);
#define get_interface_name(buf, buflen, hInterface)	_get_interface_name(QSP_ARG  buf, buflen, hInterface)

extern int _lookup_spink_node(QSP_ARG_DECL Spink_Node *skn_p, spinNodeHandle *hdl_p);
#define lookup_spink_node(skn_p, hdl_p) _lookup_spink_node(QSP_ARG skn_p, hdl_p)
extern void _print_interface_name(QSP_ARG_DECL  spinNodeHandle hInterfaceDisplayName);
#define print_interface_name(hInterfaceDisplayName)	_print_interface_name(QSP_ARG  hInterfaceDisplayName)
extern int _get_spink_cam_list(QSP_ARG_DECL  spinInterface hInterface, spinCameraList *hCamList_p, size_t *num_p);
#define get_spink_cam_list(hInterface, hCamList_p, num_p)	_get_spink_cam_list(QSP_ARG  hInterface, hCamList_p, num_p)
extern int _release_spink_interface_list( QSP_ARG_DECL  spinInterfaceList *hInterfaceList_p );
#define release_spink_interface_list( hInterfaceList_p )	_release_spink_interface_list( QSP_ARG  hInterfaceList_p )
extern int _get_spink_interface_from_list(QSP_ARG_DECL  spinInterface *hInterface_p, spinInterfaceList hInterfaceList, int idx );
extern int _get_spink_model_name(QSP_ARG_DECL   spinNodeMapHandle hNodeMapTLDevice, char *buf, size_t *len_p );
extern int _print_indexed_spink_cam_info(QSP_ARG_DECL   spinCameraList hCameraList, int idx );
#define get_spink_interface_from_list(hInterface_p, hInterfaceList, idx ) _get_spink_interface_from_list(QSP_ARG  hInterface_p, hInterfaceList, idx )
#define get_spink_model_name(hNodeMapTLDevice, buf, len_p ) _get_spink_model_name(QSP_ARG   hNodeMapTLDevice, buf, len_p )
#define print_indexed_spink_cam_info(hCameraList, idx ) _print_indexed_spink_cam_info(QSP_ARG   hCameraList, idx )

extern int _spink_node_is_readable(QSP_ARG_DECL  spinNodeHandle hdl);
#define spink_node_is_readable(hdl)	_spink_node_is_readable(QSP_ARG  hdl)
extern int _spink_node_is_writable(QSP_ARG_DECL  spinNodeHandle hdl);
#define spink_node_is_writable(hdl)	_spink_node_is_writable(QSP_ARG  hdl)
extern int _spink_node_is_available(QSP_ARG_DECL  spinNodeHandle hdl);
#define spink_node_is_available(hdl)	_spink_node_is_available(QSP_ARG  hdl)
extern int _spink_node_is_implemented(QSP_ARG_DECL  spinNodeHandle hdl);
#define spink_node_is_implemented(hdl)	_spink_node_is_implemented(QSP_ARG  hdl)

extern int _spink_release_cam(QSP_ARG_DECL  Spink_Cam *skc_p);
#define spink_release_cam(skc_p) _spink_release_cam(QSP_ARG  skc_p)

extern int _release_spink_cam_list(QSP_ARG_DECL   spinCameraList *hCamList_p );
#define release_spink_cam_list(hCamList_p )	_release_spink_cam_list(QSP_ARG   hCamList_p )
extern int _fetch_spink_map(QSP_ARG_DECL  spinInterface hInterface, spinNodeMapHandle *hMap_p);
#define fetch_spink_map(hInterface, hMap_p) _fetch_spink_map(QSP_ARG  hInterface, hMap_p)

extern int _get_spink_interfaces(QSP_ARG_DECL  spinSystem hSystem, spinInterfaceList *hInterfaceList_p, size_t *numInterfaces_p);
extern int _get_spink_cameras(QSP_ARG_DECL  spinSystem hSystem, spinCameraList *hCameraList_p, size_t *num_p );

#define get_spink_interfaces(hSystem, hInterfaceList_p, numInterfaces_p) _get_spink_interfaces(QSP_ARG  hSystem, hInterfaceList_p, numInterfaces_p)
#define get_spink_cameras(hSystem, hCameraList_p, num_p ) _get_spink_cameras(QSP_ARG  hSystem, hCameraList_p, num_p )

extern int _get_spink_interface_cameras(QSP_ARG_DECL  spinInterface hInterface);
#define get_spink_interface_cameras(hInterface)	_get_spink_interface_cameras(QSP_ARG  hInterface)

extern void _report_spink_error(QSP_ARG_DECL  spinError error, const char *whence );
#define report_spink_error(error,whence)	_report_spink_error(QSP_ARG  error, whence )

#endif // HAVE_LIBSPINNAKER

// spink_node_map.c

extern void _insure_current_camera(QSP_ARG_DECL  Spink_Cam *skc_p);
#define insure_current_camera(skc_p) _insure_current_camera(QSP_ARG  skc_p)

extern int _release_current_camera(QSP_ARG_DECL  int verbose);
#define release_current_camera(v) _release_current_camera(QSP_ARG  v)

extern void _list_nodes_from_map(QSP_ARG_DECL  Spink_Map *skm_p);
#define list_nodes_from_map(skm_p) _list_nodes_from_map(QSP_ARG  skm_p)

extern void _print_spink_node_info(QSP_ARG_DECL Spink_Node *skn_p, int level);
#define print_spink_node_info(skn_p,level) _print_spink_node_info(QSP_ARG skn_p,level)

#ifdef HAVE_LIBSPINNAKER
extern void _report_node_access_error(QSP_ARG_DECL  spinNodeHandle hNode, const char *w);
#define report_node_access_error(hNode, w) _report_node_access_error(QSP_ARG  hNode, w)

extern int _get_display_name_len(QSP_ARG_DECL  spinNodeHandle hdl);
#define get_display_name_len(hdl) _get_display_name_len(QSP_ARG  hdl)


extern int _get_node_map_handle(QSP_ARG_DECL  spinNodeMapHandle *hMap_p,Spink_Map *skm_p, const char *whence);
#define get_node_map_handle(hMap_p,skm_p,w) _get_node_map_handle(QSP_ARG  hMap_p,skm_p,w)

extern int _traverse_by_node_handle(QSP_ARG_DECL  spinNodeHandle hCategoryNode, int level, int (*func)(QSP_ARG_DECL spinNodeHandle hNode, int level) );
#define traverse_by_node_handle(hCategoryNode, level, func ) _traverse_by_node_handle(QSP_ARG  hCategoryNode, level, func )

extern int _print_camera_nodes(QSP_ARG_DECL  Spink_Cam *skc_p);
#define print_camera_nodes(skc_p) _print_camera_nodes(QSP_ARG  skc_p)

extern int _get_node_value_string(QSP_ARG_DECL  char *buf, size_t *buflen_p, spinNodeHandle hNode );
#define get_node_value_string(buf, buflen_p, hNode ) _get_node_value_string(QSP_ARG  buf, buflen_p, hNode )
extern int _print_value_node(QSP_ARG_DECL  spinNodeHandle hNode, unsigned int level);
#define print_value_node(hNode, level) _print_value_node(QSP_ARG  hNode, level)
extern int _get_display_name(QSP_ARG_DECL  char *buf, size_t *len_p, spinNodeHandle hdl);
#define get_display_name(buf, len_p, hdl) _get_display_name(QSP_ARG  buf, len_p, hdl)
extern int _get_node_name(QSP_ARG_DECL  char *buf, size_t *len_p, spinNodeHandle hdl);
#define get_node_name(buf, len_p, hdl) _get_node_name(QSP_ARG  buf, len_p, hdl)
extern int _get_string_node_string(QSP_ARG_DECL  char *buf, size_t *buflen_p, spinNodeHandle hNode );
#define get_string_node_string(buf, buflen_p, hNode ) _get_string_node_string(QSP_ARG  buf, buflen_p, hNode )
extern int _print_string_node(QSP_ARG_DECL  spinNodeHandle hNode, unsigned int level);
#define print_string_node(hNode, level) _print_string_node(QSP_ARG  hNode, level)

// spink_acq.c

extern Grab_Frame_Status _cam_frame_status(QSP_ARG_DECL  Spink_Cam *skc_p, int idx);
extern void _set_cam_frame_status(QSP_ARG_DECL  Spink_Cam *skc_p, int idx, Grab_Frame_Status status);
extern void _set_cam_frame_image(QSP_ARG_DECL  Spink_Cam *skc_p, int idx, spinImage hImage);
#define cam_frame_status(skc_p, idx) _cam_frame_status(QSP_ARG  skc_p, idx)
#define set_cam_frame_status(skc_p, idx, status) _set_cam_frame_status(QSP_ARG  skc_p, idx, status)
#define set_cam_frame_image(skc_p, idx, hImage) _set_cam_frame_image(QSP_ARG  skc_p, idx, hImage)

extern Data_Obj *_cam_frame_with_index(QSP_ARG_DECL  Spink_Cam *skc_p, int idx);

#define cam_frame_with_index(skc_p, idx) _cam_frame_with_index(QSP_ARG  skc_p, idx)

extern void _release_oldest_spink_frame(QSP_ARG_DECL  Spink_Cam *skc_p);
#define release_oldest_spink_frame(skc_p) _release_oldest_spink_frame(QSP_ARG  skc_p)
extern void _release_spink_frame(QSP_ARG_DECL  Spink_Cam *skc_p, int index);
#define release_spink_frame(skc_p,idx) _release_spink_frame(QSP_ARG  skc_p,idx)

extern int _set_acquisition_continuous(QSP_ARG_DECL  Spink_Cam *skc_p);
#define set_acquisition_continuous(skc_p) _set_acquisition_continuous(QSP_ARG  skc_p)

extern void _enable_image_events(QSP_ARG_DECL  Spink_Cam *skc_p, void (*func)(spinImage,void *));
#define enable_image_events(skc_p,f) _enable_image_events(QSP_ARG  skc_p,f)

extern void _set_camera_node(QSP_ARG_DECL  Spink_Cam *skc_p, const char *node_name, const char *entry_name);
#define set_camera_node(skc_p, node_name, entry_name) _set_camera_node(QSP_ARG  skc_p, node_name, entry_name)

extern void _set_n_spink_buffers(QSP_ARG_DECL  Spink_Cam *skc_p, int n);
#define set_n_spink_buffers(skc_p, n) _set_n_spink_buffers(QSP_ARG  skc_p, n)

extern int _next_spink_image(QSP_ARG_DECL  spinImage *img_p, Spink_Cam *skc_p);
#define next_spink_image(img_p, skc_p) _next_spink_image(QSP_ARG  img_p, skc_p)

extern int _spink_test_acq(QSP_ARG_DECL  Spink_Cam *skc_p);
#define spink_test_acq(skc_p) _spink_test_acq(QSP_ARG  skc_p)

#endif // HAVE_LIBSPINNAKER

extern int _spink_start_capture(QSP_ARG_DECL  Spink_Cam *skc_p);
extern int _spink_stop_capture(QSP_ARG_DECL  Spink_Cam *skc_p);
#define spink_start_capture(skc_p) _spink_start_capture(QSP_ARG  skc_p)
#define spink_stop_capture(skc_p) _spink_stop_capture(QSP_ARG  skc_p)

extern Data_Obj * _grab_spink_cam_frame(QSP_ARG_DECL  Spink_Cam * skc_p );
#define grab_spink_cam_frame(skc_p) _grab_spink_cam_frame(QSP_ARG  skc_p)

extern void _show_n_buffers(QSP_ARG_DECL  Spink_Cam *skc_p);
#define show_n_buffers(skc_p) _show_n_buffers(QSP_ARG  skc_p)


// spink_menu.c

extern void _select_spink_map(QSP_ARG_DECL  Spink_Map *skm_p);
#define select_spink_map(skm_p) _select_spink_map(QSP_ARG  skm_p)


// spink_util.c

extern Spink_Cam * _select_spink_cam(QSP_ARG_DECL  Spink_Cam *skc_p);
extern void _deselect_spink_cam(QSP_ARG_DECL  Spink_Cam *skc_p);
#define select_spink_cam(skc_p) _select_spink_cam(QSP_ARG  skc_p)
#define deselect_spink_cam(skc_p) _deselect_spink_cam(QSP_ARG  skc_p)

extern void _init_cam_expr_funcs(SINGLE_QSP_ARG_DECL);
#define init_cam_expr_funcs() _init_cam_expr_funcs(SINGLE_QSP_ARG)

extern int _init_spink_cam_system(SINGLE_QSP_ARG_DECL);
#define init_spink_cam_system() _init_spink_cam_system(SINGLE_QSP_ARG)

extern void _fetch_chunk_data(QSP_ARG_DECL  Chunk_Data *cd_p, Data_Obj *dp);
#define fetch_chunk_data(cd_p, dp) _fetch_chunk_data(QSP_ARG  cd_p, dp)
extern void _enable_chunk_data(QSP_ARG_DECL  Spink_Cam *skc_p, Chunk_Data *cd_p);
#define enable_chunk_data(skc_p, cd_p) _enable_chunk_data(QSP_ARG  skc_p, cd_p)
extern void _format_chunk_data(QSP_ARG_DECL  char *buf, Chunk_Data *cd_p);
#define format_chunk_data(buf, cd_p) _format_chunk_data(QSP_ARG  buf, cd_p)

extern void _pop_map_contexts(SINGLE_QSP_ARG_DECL);
extern void _push_map_contexts(QSP_ARG_DECL  Spink_Map *skm_p);
#define pop_map_contexts() _pop_map_contexts(SINGLE_QSP_ARG)
#define push_map_contexts(skm_p) _push_map_contexts(QSP_ARG  skm_p)

extern void _print_map_tree(QSP_ARG_DECL  Spink_Map *skm_p);
#define print_map_tree(skm_p) _print_map_tree(QSP_ARG  skm_p)

extern void _print_cat_tree(QSP_ARG_DECL  Spink_Category *sct_p);
#define print_cat_tree(sct_p) _print_cat_tree(QSP_ARG  sct_p)

#ifdef HAVE_LIBSPINNAKER
extern Spink_Node_Type *_find_type_by_code(QSP_ARG_DECL  spinNodeType type);
#define find_type_by_code(type) _find_type_by_code(QSP_ARG  type)
#endif // HAVE_LIBSPINNAKER

extern Item_Context * _pop_spink_node_context(SINGLE_QSP_ARG_DECL);
extern void _push_spink_node_context(QSP_ARG_DECL  Item_Context *icp);
#define pop_spink_node_context() _pop_spink_node_context(SINGLE_QSP_ARG)
#define push_spink_node_context(icp) _push_spink_node_context(QSP_ARG  icp)

extern Item_Context * _pop_spink_cat_context(SINGLE_QSP_ARG_DECL);
extern void _push_spink_cat_context(QSP_ARG_DECL  Item_Context *icp);
#define pop_spink_cat_context() _pop_spink_cat_context(SINGLE_QSP_ARG)
#define push_spink_cat_context(icp) _push_spink_cat_context(QSP_ARG  icp)

extern void _release_spink_cam_system(SINGLE_QSP_ARG_DECL);
#define release_spink_cam_system() _release_spink_cam_system(SINGLE_QSP_ARG)

extern int is_fmt7_mode(QSP_ARG_DECL  Spink_Cam *skc_p, int idx );
extern int set_fmt7_mode(QSP_ARG_DECL  Spink_Cam *skc_p, int idx );
extern int set_std_mode(QSP_ARG_DECL  Spink_Cam *skc_p, int idx);

/* stream_spink.c */
extern void _spink_stream_record(QSP_ARG_DECL  Image_File *ifp,int32_t n_frames_wanted,int n_cameras,Spink_Cam **skc_p_tbl);
#define spink_stream_record(ifp,n_frames_wanted,n_cameras,skc_p_tbl) _spink_stream_record(QSP_ARG  ifp,n_frames_wanted,n_cameras,skc_p_tbl)

extern Image_File * _get_file_for_recording(QSP_ARG_DECL  const char *name, uint32_t n_frames,Spink_Cam *skc_p);
#define get_file_for_recording(name,n_f,skc_p)	_get_file_for_recording(QSP_ARG  name,n_f,skc_p)

extern void spink_set_async_record(int flag);
extern int spink_get_async_record(void);
extern void stream_record(QSP_ARG_DECL  Image_File *ifp,int32_t n_frames_wanted,Spink_Cam *skc_p);
extern COMMAND_FUNC( spink_wait_record );
extern COMMAND_FUNC( spink_halt_record );

#endif // _SPINK_H_
