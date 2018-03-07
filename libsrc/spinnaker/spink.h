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


#define N_FMT7_MODES	5	// PGR has 32, but why bother?

typedef int Framerate_Mask;

typedef struct spink_cam {
	const char *		sk_name;
#ifdef HAVE_LIBSPINNAKER
	/*
	fc2Context		sk_context;
	fc2PGRGuid		sk_guid;
	fc2CameraInfo		sk_cam_info;
	fc2EmbeddedImageInfo	sk_ei_info;
	fc2VideoMode		sk_video_mode;	// current
	fc2FrameRate 		sk_framerate;	// current
	fc2Config		sk_config;
	fc2Format7Info *	sk_fmt7_info_tbl;
	fc2Image *		sk_img_p;
	*/

	unsigned char *		sk_base;	// for captured frames...
	long			sk_buf_delta;
	int	 		sk_framerate_index;	// into all_framerates
	int	 		sk_video_mode_index;	// into all_video_modes
	int	 		sk_fmt7_index;		// of available...

	int	 		sk_my_video_mode_index;	// into private tbl
	int			sk_n_video_modes;
	int			sk_n_fmt7_modes;
	int *			sk_video_mode_indices;
	const char **		sk_video_mode_names;
	Framerate_Mask *	sk_framerate_mask_tbl;	// one for every video mode
	int			sk_n_framerates;
	const char **		sk_framerate_names;
#endif /* HAVE_LIBSPINNAKER */

	unsigned int		sk_cols;
	unsigned int		sk_rows;
	unsigned int		sk_depth;	// bytes per pixel
	int			sk_n_buffers;
	int			sk_newest;
	Data_Obj **		sk_frm_dp_tbl;
	Item_Context *		sk_do_icp;	// data_obj context
	List *			sk_in_use_lp;	// list of frames...
	List *			sk_feat_lp;
	u_long			sk_flags;
} Spink_Cam;

ITEM_INTERFACE_PROTOTYPES(Spink_Cam,spink_cam)

#define new_spink_cam(s)	_new_spink_cam(QSP_ARG  s)
#define spink_cam_of(s)	_spink_cam_of(QSP_ARG  s)
#define list_spink_cams(fp)	_list_spink_cams(QSP_ARG  fp)
#define pick_spink_cam(s)	_pick_spink_cam(QSP_ARG  s)
#define spink_cam_list()	_spink_cam_list(SINGLE_QSP_ARG)

/* flag bits */

#define FLY_CAM_USES_BMODE	1
#define FLY_CAM_IS_RUNNING	2
#define FLY_CAM_IS_CAPTURING	4
#define FLY_CAM_IS_TRANSMITTING	8

#define IS_CAPTURING(fcp)	(fcp->sk_flags & FLY_CAM_IS_CAPTURING)
#define IS_TRANSMITTING(fcp)	(fcp->sk_flags & FLY_CAM_IS_TRANSMITTING)

// spink_enum.c
#ifdef HAVE_LIBSPINNAKER
extern int get_spink_node( spinNodeMapHandle hMap, const char *tag, spinNodeHandle *hdl_p);
extern int spink_get_string(spinNodeHandle hdl, char *buf, size_t *len_p);
extern void print_interface_name(spinNodeHandle hInterfaceDisplayName);
extern int get_spink_cam_list(spinInterface hInterface, spinCameraList *hCamList_p, size_t *num_p);
extern int release_spink_interface_list( spinInterfaceList hInterfaceList );
extern int release_spink_interface(spinInterface hInterface);
extern int get_spink_interface_from_list(spinInterface *hInterface_p, spinInterfaceList hInterfaceList, int idx );
extern int get_spink_cam_from_list(spinCamera *hCam_p, spinCameraList hCameraList, int idx );
extern int get_spink_transport_level_map( spinNodeMapHandle *mapHdl_p, spinCamera hCam );
extern int get_spink_vendor_name( spinNodeMapHandle hNodeMapTLDevice, char *buf, size_t *len_p );
extern int get_spink_model_name( spinNodeMapHandle hNodeMapTLDevice, char *buf, size_t *len_p );
//extern int print_spink_cam_info( spinCameraList hCameraList, int idx );
extern int print_indexed_spink_cam_info( spinCameraList hCameraList, int idx );

extern int spink_node_is_readable(spinNodeHandle hdl);
extern int spink_node_is_available(spinNodeHandle hdl);
extern int release_spink_cam_list( spinCameraList *hCamList_p );
extern int release_spink_cam(spinCamera hCam);
extern int get_spink_map( spinInterface hInterface, spinNodeMapHandle *hMap_p);
extern int get_spink_system(spinSystem *hSystem_p);
extern int release_spink_system(spinSystem hSystem);
extern int query_spink_interface(spinInterface hInterface);
extern int get_spink_interfaces(spinSystem hSystem, spinInterfaceList *hInterfaceList_p, size_t *numInterfaces_p);
extern int get_spink_cameras(spinSystem hSystem, spinCameraList *hCameraList_p, size_t *num_p );
#endif // HAVE_LIBSPINNAKER

// spink_util.c
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

#ifndef HAVE_LIBSPINNAKER
#endif // ! HAVE_LIBSPINNAKER

