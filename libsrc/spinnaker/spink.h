#ifndef _SPINK_H_
#define _SPINK_H_

#include "quip_config.h"
#include "data_obj.h"
#include "fio_api.h"

#ifdef HAVE_LIBSPINNAKER
#include "SpinnakerC.h"
#endif // HAVE_LIBSPINNAKER

#define BOOL	bool8_t

// Compiler warning C4996 suppressed due to deprecated strcpy() and sprintf()
// functions on Windows platform.
#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
    #pragma warning(disable : 4996)
#endif

// This macro helps with C-strings.
#define MAX_BUFF_LEN 256
#define MAX_NODE_CHARS 35


#define N_FMT7_MODES	5	// PGR has 32, but why bother?

typedef int Framerate_Mask;

typedef struct spink_cam {
	const char *		skc_name;
#ifdef HAVE_LIBSPINNAKER
	spinCamera		skc_handle;
	spinNodeMapHandle	skc_TL_dev_node_map;	// hNodeMapTLDevice
	spinNodeMapHandle	skc_genicam_node_map;	// hNodeMapTLDevice

	/*
	fc2Context		skc_context;
	fc2PGRGuid		skc_guid;
	fc2CameraInfo		skc_cam_info;
	fc2EmbeddedImageInfo	skc_ei_info;
	fc2VideoMode		skc_video_mode;	// current
	fc2FrameRate 		skc_framerate;	// current
	fc2Config		skc_config;
	fc2Format7Info *	skc_fmt7_info_tbl;
	fc2Image *		skc_img_p;
	*/

	unsigned char *		skc_base;	// for captured frames...
	long			skc_buf_delta;
	int	 		skc_framerate_index;	// into all_framerates
	int	 		skc_video_mode_index;	// into all_video_modes
	int	 		skc_fmt7_index;		// of available...

	int	 		skc_my_video_mode_index;	// into private tbl
	int			skc_n_video_modes;
	int			skc_n_fmt7_modes;
	int *			skc_video_mode_indices;
	const char **		skc_video_mode_names;
	Framerate_Mask *	skc_framerate_mask_tbl;	// one for every video mode
	int			skc_n_framerates;
	const char **		skc_framerate_names;
#endif /* HAVE_LIBSPINNAKER */

	unsigned int		skc_cols;
	unsigned int		skc_rows;
	unsigned int		skc_depth;	// bytes per pixel
	int			skc_n_buffers;
	int			skc_newest;
	Data_Obj **		skc_frm_dp_tbl;
	Item_Context *		skc_do_icp;	// data_obj context
	List *			skc_in_use_lp;	// list of frames...
	List *			skc_feat_lp;
	u_long			skc_flags;
} Spink_Cam;

ITEM_INTERFACE_PROTOTYPES(Spink_Cam,spink_cam)

#define new_spink_cam(s)	_new_spink_cam(QSP_ARG  s)
#define spink_cam_of(s)	_spink_cam_of(QSP_ARG  s)
#define list_spink_cams(fp)	_list_spink_cams(QSP_ARG  fp)
#define pick_spink_cam(s)	_pick_spink_cam(QSP_ARG  s)
#define spink_cam_list()	_spink_cam_list(SINGLE_QSP_ARG)

/* flag bits */

#define SPINK_CAM_CONNECTED		1
#define SPINK_CAM_RUNNING		2
#define SPINK_CAM_CAPTURING		4
#define SPINK_CAM_TRANSMITTING		8

#define IS_CONNECTED(skc_p)	(skc_p->skc_flags & SPINK_CAM_CONNECTED)
#define IS_RUNNING(skc_p)	(skc_p->skc_flags & SPINK_CAM_RUNNING)
#define IS_CAPTURING(skc_p)	(skc_p->skc_flags & SPINK_CAM_CAPTURING)
#define IS_TRANSMITTING(skc_p)	(skc_p->skc_flags & SPINK_CAM_TRANSMITTING)

typedef struct spink_interface {
	const char *		ski_name;
#ifdef HAVE_LIBSPINNAKER
	spinInterface		ski_handle;
#endif // HAVE_LIBSPINNAKER
} Spink_Interface;

ITEM_INTERFACE_PROTOTYPES(Spink_Interface,spink_interface)
#define new_spink_interface(s)		_new_spink_interface(QSP_ARG  s)
#define spink_interface_of(s)		_spink_interface_of(QSP_ARG  s)
#define list_spink_interfaces(fp)	_list_spink_interfaces(QSP_ARG  fp)
#define pick_spink_interface(s)		_pick_spink_interface(QSP_ARG  s)
#define spink_interface_list()		_spink_interface_list(SINGLE_QSP_ARG)

// spink_enum.c
#ifdef HAVE_LIBSPINNAKER
extern int _get_camera_model_name(QSP_ARG_DECL  char *buf, size_t buflen, spinNodeMapHandle hNodeMapTLDevice);
#define get_camera_model_name(buf, buflen, map) _get_camera_model_name(QSP_ARG  buf, buflen, map)
extern int _get_camera_vendor_name(QSP_ARG_DECL  char *buf, size_t buflen, spinCamera hCam);
#define get_camera_vendor_name(buf, buflen, hCam) _get_camera_vendor_name(QSP_ARG  buf, buflen, hCam)

extern int _get_interface_name(QSP_ARG_DECL  char *buf, size_t buflen, spinInterface hInterface);
#define get_interface_name(buf, buflen, hInterface)	_get_interface_name(QSP_ARG  buf, buflen, hInterface)

extern int _get_spink_node(QSP_ARG_DECL spinNodeMapHandle hMap, const char *tag, spinNodeHandle *hdl_p);
#define get_spink_node(hMap, tag, hdl_p)	_get_spink_node(QSP_ARG hMap, tag, hdl_p)
extern int _spink_get_string(QSP_ARG_DECL  spinNodeHandle hdl, char *buf, size_t *len_p);
#define spink_get_string(hdl, buf, len_p)	_spink_get_string(QSP_ARG  hdl, buf, len_p)
extern void _print_interface_name(QSP_ARG_DECL  spinNodeHandle hInterfaceDisplayName);
#define print_interface_name(hInterfaceDisplayName)	_print_interface_name(QSP_ARG  hInterfaceDisplayName)
extern int _get_spink_cam_list(QSP_ARG_DECL  spinInterface hInterface, spinCameraList *hCamList_p, size_t *num_p);
#define get_spink_cam_list(hInterface, hCamList_p, num_p)	_get_spink_cam_list(QSP_ARG  hInterface, hCamList_p, num_p)
extern int _release_spink_interface_list( QSP_ARG_DECL  spinInterfaceList *hInterfaceList_p );
#define release_spink_interface_list( hInterfaceList_p )	_release_spink_interface_list( QSP_ARG  hInterfaceList_p )
extern int _release_spink_interface(QSP_ARG_DECL  spinInterface hInterface);
#define release_spink_interface(hInterface)	_release_spink_interface(QSP_ARG  hInterface)
extern int _get_spink_interface_from_list(QSP_ARG_DECL  spinInterface *hInterface_p, spinInterfaceList hInterfaceList, int idx );
extern int _get_spink_cam_from_list(QSP_ARG_DECL  spinCamera *hCam_p, spinCameraList hCameraList, int idx );
extern int _get_spink_transport_level_map(QSP_ARG_DECL   spinNodeMapHandle *mapHdl_p, spinCamera hCam );
extern int _get_spink_vendor_name(QSP_ARG_DECL   spinNodeMapHandle hNodeMapTLDevice, char *buf, size_t *len_p );
extern int _get_spink_model_name(QSP_ARG_DECL   spinNodeMapHandle hNodeMapTLDevice, char *buf, size_t *len_p );
extern int _print_indexed_spink_cam_info(QSP_ARG_DECL   spinCameraList hCameraList, int idx );
#define get_spink_interface_from_list(hInterface_p, hInterfaceList, idx ) _get_spink_interface_from_list(QSP_ARG  hInterface_p, hInterfaceList, idx )
#define get_spink_cam_from_list(hCam_p, hCameraList, idx ) _get_spink_cam_from_list(QSP_ARG  hCam_p, hCameraList, idx )
#define get_spink_transport_level_map(mapHdl_p, hCam ) _get_spink_transport_level_map(QSP_ARG   mapHdl_p, hCam )
#define get_spink_vendor_name(hNodeMapTLDevice, buf, len_p ) _get_spink_vendor_name(QSP_ARG   hNodeMapTLDevice, buf, len_p )
#define get_spink_model_name(hNodeMapTLDevice, buf, len_p ) _get_spink_model_name(QSP_ARG   hNodeMapTLDevice, buf, len_p )
#define print_indexed_spink_cam_info(hCameraList, idx ) _print_indexed_spink_cam_info(QSP_ARG   hCameraList, idx )

extern int _spink_node_is_readable(QSP_ARG_DECL  spinNodeHandle hdl);
#define spink_node_is_readable(hdl)	_spink_node_is_readable(QSP_ARG  hdl)
extern int _spink_node_is_writable(QSP_ARG_DECL  spinNodeHandle hdl);
#define spink_node_is_writable(hdl)	_spink_node_is_writable(QSP_ARG  hdl)
extern int _spink_node_is_available(QSP_ARG_DECL  spinNodeHandle hdl);
#define spink_node_is_available(hdl)	_spink_node_is_available(QSP_ARG  hdl)
extern int _release_spink_cam_list(QSP_ARG_DECL   spinCameraList *hCamList_p );
#define release_spink_cam_list(hCamList_p )	_release_spink_cam_list(QSP_ARG   hCamList_p )
extern int _release_spink_cam(QSP_ARG_DECL  spinCamera hCam);
#define release_spink_cam(hCam)	_release_spink_cam(QSP_ARG  hCam)
extern int _get_spink_map(QSP_ARG_DECL  spinInterface hInterface, spinNodeMapHandle *hMap_p);
#define get_spink_map(hInterface, hMap_p) _get_spink_map(QSP_ARG  hInterface, hMap_p)

extern int _get_spink_system(QSP_ARG_DECL  spinSystem *hSystem_p);
extern int _release_spink_system(QSP_ARG_DECL  spinSystem hSystem);
extern int _get_spink_interfaces(QSP_ARG_DECL  spinSystem hSystem, spinInterfaceList *hInterfaceList_p, size_t *numInterfaces_p);
extern int _get_spink_cameras(QSP_ARG_DECL  spinSystem hSystem, spinCameraList *hCameraList_p, size_t *num_p );

#define get_spink_system(hSystem_p) _get_spink_system(QSP_ARG  hSystem_p)
#define release_spink_system(hSystem) _release_spink_system(QSP_ARG  hSystem)
#define get_spink_interfaces(hSystem, hInterfaceList_p, numInterfaces_p) _get_spink_interfaces(QSP_ARG  hSystem, hInterfaceList_p, numInterfaces_p)
#define get_spink_cameras(hSystem, hCameraList_p, num_p ) _get_spink_cameras(QSP_ARG  hSystem, hCameraList_p, num_p )

extern int _get_spink_interface_cameras(QSP_ARG_DECL  spinInterface hInterface);
#define get_spink_interface_cameras(hInterface)	_get_spink_interface_cameras(QSP_ARG  hInterface)

extern void _report_spink_error(QSP_ARG_DECL  spinError error, const char *whence );
#define report_spink_error(error,whence)	_report_spink_error(QSP_ARG  error, whence )

// spink_node_map.c

extern int _get_camera_node_map(QSP_ARG_DECL  spinNodeMapHandle *map_p, spinCamera hCam );
#define get_camera_node_map(map_p, hCam ) _get_camera_node_map(QSP_ARG  map_p, hCam )

extern int _get_camera_nodes(QSP_ARG_DECL  Spink_Cam *skc_p);
#define get_camera_nodes(skc_p) _get_camera_nodes(QSP_ARG  skc_p)

extern int _get_node_value_string(QSP_ARG_DECL  char *buf, size_t *buflen_p, spinNodeHandle hNode );
#define get_node_value_string(buf, buflen_p, hNode ) _get_node_value_string(QSP_ARG  buf, buflen_p, hNode )
extern int _print_value_node(QSP_ARG_DECL  spinNodeHandle hNode, unsigned int level);
#define print_value_node(hNode, level) _print_value_node(QSP_ARG  hNode, level)
extern int _get_display_name(QSP_ARG_DECL  char *buf, size_t *len_p, spinNodeHandle hdl);
#define get_display_name(buf, len_p, hdl) _get_display_name(QSP_ARG  buf, len_p, hdl)
extern int _get_string_node_string(QSP_ARG_DECL  char *buf, size_t *buflen_p, spinNodeHandle hNode );
#define get_string_node_string(buf, buflen_p, hNode ) _get_string_node_string(QSP_ARG  buf, buflen_p, hNode )
extern int _print_string_node(QSP_ARG_DECL  spinNodeHandle hNode, unsigned int level);
#define print_string_node(hNode, level) _print_string_node(QSP_ARG  hNode, level)

// spink_acq.c

extern int _get_enumeration_entry_by_name(QSP_ARG_DECL  spinNodeHandle hEnum, const char *tag, spinNodeHandle *hdl_p);
#define get_enumeration_entry_by_name(hEnum, tag, hdl_p) _get_enumeration_entry_by_name(QSP_ARG  hEnum, tag, hdl_p)
extern int _get_enumeration_int_val(QSP_ARG_DECL  spinNodeHandle hNode, int64_t *int_ptr);
#define get_enumeration_int_val(hNode, int_ptr) _get_enumeration_int_val(QSP_ARG  hNode, int_ptr)
extern int _set_enumeration_int_val(QSP_ARG_DECL  spinNodeHandle hNode, int64_t v);
extern int _release_spink_image(QSP_ARG_DECL  spinImage hImage);
extern int _next_spink_image(QSP_ARG_DECL  spinImage *img_p, Spink_Cam *skc_p);
#define set_enumeration_int_val(hNode, v) _set_enumeration_int_val(QSP_ARG  hNode, v)
#define release_spink_image(hImage) _release_spink_image(QSP_ARG  hImage)
#define next_spink_image(img_p, skc_p) _next_spink_image(QSP_ARG  img_p, skc_p)
extern int _create_empty_image(QSP_ARG_DECL  spinImage *hImg_p);
extern int _convert_spink_image(QSP_ARG_DECL  spinImage hDestImg, spinImage hSrcImg );
extern int _destroy_spink_image(QSP_ARG_DECL  spinImage hImg);
extern int _spink_start_capture(QSP_ARG_DECL  Spink_Cam *skc_p);
extern int _spink_stop_capture(QSP_ARG_DECL  Spink_Cam *skc_p);

#define create_empty_image(hImg_p) _create_empty_image(QSP_ARG  hImg_p)
#define convert_spink_image(hDestImg, hSrcImg ) _convert_spink_image(QSP_ARG  hDestImg, hSrcImg )
#define destroy_spink_image(hImg) _destroy_spink_image(QSP_ARG  hImg)
#define spink_start_capture(skc_p) _spink_start_capture(QSP_ARG  skc_p)
#define spink_stop_capture(skc_p) _spink_stop_capture(QSP_ARG  skc_p)

extern int _spink_test_acq(QSP_ARG_DECL  Spink_Cam *skc_p);
#define spink_test_acq(skc_p) _spink_test_acq(QSP_ARG  skc_p)

#endif // HAVE_LIBSPINNAKER


// spink_util.c
extern void _release_spink_cam_system(SINGLE_QSP_ARG_DECL);
#define release_spink_cam_system() _release_spink_cam_system(SINGLE_QSP_ARG)

extern int is_fmt7_mode(QSP_ARG_DECL  Spink_Cam *scp, int idx );
extern int set_fmt7_mode(QSP_ARG_DECL  Spink_Cam *scp, int idx );
extern int set_std_mode(QSP_ARG_DECL  Spink_Cam *fcp, int idx);

//////////////////////////
typedef struct pgr_property_type {
	const char *		name;
#ifdef HAVE_LIBSPINNAKER
	/*
	fc2PropertyType		type_code;
	fc2PropertyInfo 	info;
	fc2Property 		prop;
	*/
#endif // HAVE_LIBSPINNAKER
} Spink_Cam_Property_Type;

ITEM_INTERFACE_PROTOTYPES(Spink_Cam_Property_Type,pgr_prop)

#define pick_pgr_prop(p)	_pick_pgr_prop(QSP_ARG  p)
#define get_pgr_prop(p)		_get_pgr_prop(QSP_ARG  p)
#define new_pgr_prop(p)		_new_pgr_prop(QSP_ARG  p)
#define pgr_prop_list()		_pgr_prop_list(SINGLE_QSP_ARG)

typedef struct fly_frame {
	const char *		pf_name;
#ifdef HAVE_LIBSPINNAKER
	//dc1394video_frame_t *	pf_framep;
#endif
	Data_Obj *		pf_dp;
} Fly_Frame;

typedef struct pgr_prop_val {
	int	pv_is_abs;
	union {
		int	u_i;
		float	u_f;
	} pv_u;
} Spink_Cam_Prop_Val;


#define N_EII_PROPERTIES	10

#ifdef HAVE_LIBSPINNAKER

typedef struct _myEmbeddedImageInfo {
	/*
	fc2EmbeddedImageInfoProperty prop_tbl[N_EII_PROPERTIES];
	*/
} myEmbeddedImageInfo;


#define DECLARE_NAMED_PARAM(stem,type,short_stem,cap_stem)	\
								\
typedef struct named_##stem {					\
	const char *		short_stem##_name;		\
	type			short_stem##_value;		\
} Named_##cap_stem;

#else	// ! HAVE_LIBSPINNAKER

#define DECLARE_NAMED_PARAM(stem,type,short_stem,cap_stem)	\
								\
typedef struct named_##stem {					\
	const char *		short_stem##_name;		\
} Named_##cap_stem;

#endif	// ! HAVE_LIBSPINNAKER

typedef struct named_video_mode {
	const char *		nvm_name;
#ifdef HAVE_LIBSPINNAKER
	/*
	fc2VideoMode		nvm_value;
	*/
#endif
	int			nvm_width;
	int			nvm_height;
	int			nvm_depth;
} Named_Video_Mode;

DECLARE_NAMED_PARAM(framerate,/*fc2FrameRate*/int,nfr,Frame_Rate)
DECLARE_NAMED_PARAM(grab_mode,/*fc2GrabMode*/int,ngm,Grab_Mode)
DECLARE_NAMED_PARAM(bus_speed,/*fc2BusSpeed*/int,nbs,Bus_Speed)
DECLARE_NAMED_PARAM(bw_allocation,/*fc2BandwidthAllocation*/int,nba,Bandwidth_Allocation)
DECLARE_NAMED_PARAM(interface,/*fc2InterfaceType*/int,nif,Interface)

typedef struct named_feature {
	const char *		nft_name;
#ifdef HAVE_LIBSPINNAKER
	//dc1394feature_t 	nft_feature;
#endif
} Named_Feature;

typedef struct named_trigger_mode {
	const char *		ntm_name;
#ifdef HAVE_LIBSPINNAKER
	//dc1394trigger_mode_t	ntm_mode;
#endif
} Named_Trigger_Mode;

// jbm made these limits up...
// We should do something sensible here, but it is difficult
// because we don't necessarily know how many buffers we can
// allocate.  USBFS has a default limit of 16MB, but on euler
// we have increased it to 200MB (in /etc/default/grub, see PGR
// TAN, and pointed out by Brian Cha).
//
//
#define MIN_N_BUFFERS 2
#define MAX_N_BUFFERS 1024

extern const char *eii_prop_names[];


/* fly.c */

extern const char *eii_prop_names[];

extern int pick_grab_mode(QSP_ARG_DECL Spink_Cam *fcp, const char *pmpt);

// These can be declared unconditionally if they don't refer to any PGR structs...

extern void show_grab_mode(QSP_ARG_DECL  Spink_Cam *fcp);
extern void set_grab_mode(QSP_ARG_DECL  Spink_Cam *fcp, int idx);
extern void set_buffer_obj(QSP_ARG_DECL  Spink_Cam *fcp, Data_Obj *dp);
extern void set_eii_property(QSP_ARG_DECL  Spink_Cam *fcp, int idx, int yesno );
extern void show_fmt7_modes(QSP_ARG_DECL  Spink_Cam *fcp);
extern int set_n_buffers(QSP_ARG_DECL  Spink_Cam *fcp, int n );
extern void show_n_buffers(QSP_ARG_DECL  Spink_Cam *fcp);
extern void set_fmt7_size(QSP_ARG_DECL  Spink_Cam *fcp, int w, int h );

extern void list_spink_cam_properties(QSP_ARG_DECL  Spink_Cam *fcp);
extern void refresh_spink_cam_properties(QSP_ARG_DECL  Spink_Cam *fcp);

#ifdef HAVE_LIBSPINNAKER
extern void refresh_property_info(QSP_ARG_DECL  Spink_Cam *fcp, Spink_Cam_Property_Type *t);
extern void show_property_info(QSP_ARG_DECL  Spink_Cam *fcp, Spink_Cam_Property_Type *t);
extern void refresh_property_value(QSP_ARG_DECL  Spink_Cam *fcp, Spink_Cam_Property_Type *t);
extern void show_property_value(QSP_ARG_DECL  Spink_Cam *fcp, Spink_Cam_Property_Type *t);

extern void set_prop_value(QSP_ARG_DECL  Spink_Cam *fcp, Spink_Cam_Property_Type *t, Spink_Cam_Prop_Val *vp);
extern void set_prop_auto(QSP_ARG_DECL  Spink_Cam *fcp, Spink_Cam_Property_Type *t, BOOL auto_state);
extern unsigned int read_register(QSP_ARG_DECL  Spink_Cam *fcp, unsigned int addr);
extern void write_register(QSP_ARG_DECL  Spink_Cam *fcp, unsigned int addr, unsigned int val);
//extern void report_fc2_error(QSP_ARG_DECL   fc2Error error, const char *whence );

//extern dc1394video_mode_t pick_video_mode(QSP_ARG_DECL  Spink_Cam *fcp, const char *pmpt);
//extern dc1394video_mode_t pick_fmt7_mode(QSP_ARG_DECL  Spink_Cam *fcp, const char *pmpt);
extern int is_fmt7_mode(QSP_ARG_DECL  Spink_Cam *fcp, int idx);
extern int set_fmt7_mode(QSP_ARG_DECL  Spink_Cam *fcp, int idx );
extern int check_buffer_alignment(QSP_ARG_DECL  Spink_Cam *fcp);
//extern void report_feature_info(QSP_ARG_DECL  Spink_Cam *fcp, dc1394feature_t id );
//extern const char *name_for_trigger_mode(dc1394trigger_mode_t mode);
#endif	// HAVE_LIBSPINNAKER

extern void cleanup_spink_cam(Spink_Cam *fcp);
extern int get_spink_cam_names(QSP_ARG_DECL  Data_Obj *dp );
extern int get_spink_cam_video_mode_strings(QSP_ARG_DECL  Data_Obj *dp, Spink_Cam *fcp);
extern int get_spink_cam_framerate_strings(QSP_ARG_DECL  Data_Obj *dp, Spink_Cam *fcp);
extern void push_spink_cam_context(QSP_ARG_DECL  Spink_Cam *fcp);
extern void pop_spink_cam_context(SINGLE_QSP_ARG_DECL);
extern int init_spink_cam_system(SINGLE_QSP_ARG_DECL);
extern int start_firewire_transmission(QSP_ARG_DECL  Spink_Cam * fcp, int buf_size );
extern Data_Obj * grab_spink_cam_frame(QSP_ARG_DECL  Spink_Cam * fcp );
extern Data_Obj * grab_newest_firewire_frame(QSP_ARG_DECL  Spink_Cam * fcp );
extern void start_firewire_capture(QSP_ARG_DECL  Spink_Cam * fcp);
extern void stop_firewire_capture(QSP_ARG_DECL  Spink_Cam * fcp );
extern int reset_spink_cam(QSP_ARG_DECL  Spink_Cam * fcp );
extern void list_spink_cam_trig( QSP_ARG_DECL  Spink_Cam * fcp );
extern void release_oldest_frame(QSP_ARG_DECL  Spink_Cam *fcp);
extern void report_spink_cam_bandwidth(QSP_ARG_DECL  Spink_Cam *fcp);
extern void list_spink_cam_framerates(QSP_ARG_DECL  Spink_Cam *fcp);
extern int list_spink_cam_video_modes(QSP_ARG_DECL  Spink_Cam *fcp);
extern void show_spink_cam_framerate(QSP_ARG_DECL  Spink_Cam *fcp);
extern void show_spink_cam_video_mode(QSP_ARG_DECL  Spink_Cam *fcp);
extern int pick_spink_cam_framerate(QSP_ARG_DECL  Spink_Cam *fcp, const char *pmpt);
extern int set_framerate(QSP_ARG_DECL  Spink_Cam *fcp, int mode_index);
extern void print_spink_cam_info(QSP_ARG_DECL  Spink_Cam *fcp);
extern int list_spink_cam_features(QSP_ARG_DECL  Spink_Cam *fcp);
extern int get_feature_choices(Spink_Cam *fcp, const char ***chp);
extern void get_spink_cam_features(Spink_Cam *fcp);

/* cam_ctl.c */
#ifdef HAVE_LIBSPINNAKER
//extern void describe_dc1394_error( QSP_ARG_DECL  dc1394error_t e );
//extern int is_auto_capable( dc1394feature_info_t *feat_p );
//extern int set_spink_cam_framerate(Spink_Cam *fcp, dc1394framerate_t framerate );
//extern int set_spink_cam_trigger_polarity(Spink_Cam *fcp,
//	dc1394trigger_polarity_t polarity);
//extern int set_spink_cam_trigger_mode(Spink_Cam *fcp, dc1394trigger_mode_t mode);
//extern int set_spink_cam_trigger_source(Spink_Cam *fcp, dc1394trigger_source_t source);
//extern int set_iso_speed(Spink_Cam *fcp, dc1394speed_t speed);
#endif /* HAVE_LIBSPINNAKER */
extern int set_spink_cam_bmode(Spink_Cam *, int);
extern int power_on_spink_cam(Spink_Cam *fcp);
extern int power_off_spink_cam(Spink_Cam *fcp);
extern int set_spink_cam_temperature(Spink_Cam *fcp, int temp);
extern int set_spink_cam_white_balance(Spink_Cam *fcp, int wb);
extern int set_spink_cam_white_shading(Spink_Cam *fcp, int val);

extern const char *name_of_indexed_video_mode(int idx);

/* stream_fly.c */
extern void fly_set_async_record(int flag);
extern int fly_get_async_record(void);
extern void stream_record(QSP_ARG_DECL  Image_File *ifp,int32_t n_frames, Spink_Cam *fcp);
extern COMMAND_FUNC( flycap_wait_record );
extern COMMAND_FUNC( flycap_halt_record );
extern Image_File * _get_file_for_recording(QSP_ARG_DECL  const char *name,
		int n_frames,Spink_Cam *fcp);
#define get_file_for_recording(name,n_f,fcp)	_get_file_for_recording(QSP_ARG  name,n_f,fcp)

typedef int spinkError;
typedef int spinkPropertyType;
typedef int spinkMode;
typedef int spinkContext;

//#define DEBUG_MSG(m)	fprintf(stderr,"%s\n",#m);
#define DEBUG_MSG(m)

#ifndef HAVE_LIBSPINNAKER
#endif // ! HAVE_LIBSPINNAKER

#endif // _SPINK_H_
