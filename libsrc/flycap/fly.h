#include "quip_config.h"

#include "query.h"
#include "data_obj.h"
#include "fio_api.h"

#ifdef HAVE_LIBFLYCAP
#include <flycapture/C/FlyCapture2_C.h>
#endif // HAVE_LIBFLYCAP

#define N_FMT7_MODES	5	// PGR has 32, but why bother?

typedef int Framerate_Mask;

typedef struct fly_cam {
	const char *		pc_name;
#ifdef HAVE_LIBFLYCAP
	fc2Context		pc_context;
	fc2PGRGuid		pc_guid;
	fc2CameraInfo		pc_cam_info;
	fc2EmbeddedImageInfo	pc_ei_info;
	fc2VideoMode		pc_video_mode;	// current
	fc2FrameRate 		pc_framerate;	// current
	fc2Config		pc_config;
	fc2Format7Info *	pc_fmt7_info_tbl;
	fc2Image *		pc_img_p;

	unsigned char *		pc_base;	// for captured frames...
	long			pc_buf_delta;
	int	 		pc_framerate_index;	// into all_framerates
	int	 		pc_video_mode_index;	// into all_video_modes
	int	 		pc_fmt7_index;		// of available...

	int	 		pc_my_video_mode_index;	// into private tbl
	int			pc_n_video_modes;
	int			pc_n_fmt7_modes;
	int *			pc_video_mode_indices;
	const char **		pc_video_mode_names;
	Framerate_Mask *	pc_framerate_mask_tbl;	// one for every video mode
	int			pc_n_framerates;
	const char **		pc_framerate_names;
#endif /* HAVE_LIBFLYCAP */

	//dc1394camera_t *	pc_cam_p;
	//dc1394video_mode_t	pc_video_mode;	// current
	//dc1394framerate_t 	pc_framerate;	// current
	//dc1394video_modes_t	pc_video_modes;
	//dc1394framerates_t 	pc_framerates;
	//dc1394featureset_t	pc_features;
	//dc1394capture_policy_t	pc_policy;
#ifdef FOOBAR
	int			pc_ring_buffer_size;
	int			pc_n_avail;
	Item_Context *		pc_pf_icp;	// pframe context
#endif // FOOBAR
	unsigned int		pc_cols;
	unsigned int		pc_rows;
	unsigned int		pc_depth;	// bytes per pixel
	int			pc_n_buffers;
	int			pc_newest;
	Data_Obj **		pc_frm_dp_tbl;
	Item_Context *		pc_do_icp;	// data_obj context
	List *			pc_in_use_lp;	// list of frames...
	List *			pc_feat_lp;
	u_long			pc_flags;
} PGR_Cam;

ITEM_INTERFACE_PROTOTYPES(PGR_Cam,pgc)

/* flag bits */

#define PGR_CAM_USES_BMODE	1
#define PGR_CAM_IS_RUNNING	2
#define PGR_CAM_IS_CAPTURING	4
#define PGR_CAM_IS_TRANSMITTING	8

#define IS_CAPTURING(pgcp)	(pgcp->pc_flags & PGR_CAM_IS_CAPTURING)
#define IS_TRANSMITTING(pgcp)	(pgcp->pc_flags & PGR_CAM_IS_TRANSMITTING)

typedef struct pgr_property_type {
	const char *		name;
#ifdef HAVE_LIBFLYCAP
	fc2PropertyType		type_code;
	fc2PropertyInfo 	info;
	fc2Property 		prop;
#endif // HAVE_LIBFLYCAP
} PGR_Property_Type;

ITEM_INTERFACE_PROTOTYPES(PGR_Property_Type,pgr_prop)

typedef struct fly_frame {
	const char *		pf_name;
#ifdef HAVE_LIBFLYCAP
	//dc1394video_frame_t *	pf_framep;
#endif
	Data_Obj *		pf_dp;
} PGR_Frame;

typedef struct pgr_prop_val {
	int	pv_is_abs;
	union {
		int	u_i;
		float	u_f;
	} pv_u;
} PGR_Prop_Val;


#define N_EII_PROPERTIES	10

#ifdef HAVE_LIBFLYCAP

typedef struct _myEmbeddedImageInfo {
	fc2EmbeddedImageInfoProperty prop_tbl[N_EII_PROPERTIES];
} myEmbeddedImageInfo;


#define DECLARE_NAMED_PARAM(stem,type,short_stem,cap_stem)	\
								\
typedef struct named_##stem {					\
	const char *		short_stem##_name;		\
	type			short_stem##_value;		\
} Named_##cap_stem;

#else	// ! HAVE_LIBFLYCAP

#define DECLARE_NAMED_PARAM(stem,type,short_stem,cap_stem)	\
								\
typedef struct named_##stem {					\
	const char *		short_stem##_name;		\
} Named_##cap_stem;

#endif	// ! HAVE_LIBFLYCAP

typedef struct named_video_mode {
	const char *		nvm_name;
#ifdef HAVE_LIBFLYCAP
	fc2VideoMode		nvm_value;
#endif
	int			nvm_width;
	int			nvm_height;
	int			nvm_depth;
} Named_Video_Mode;

DECLARE_NAMED_PARAM(pixel_format,fc2PixelFormat,npf,Pixel_Format)
DECLARE_NAMED_PARAM(framerate,fc2FrameRate,nfr,Frame_Rate)
DECLARE_NAMED_PARAM(grab_mode,fc2GrabMode,ngm,Grab_Mode)
DECLARE_NAMED_PARAM(bus_speed,fc2BusSpeed,nbs,Bus_Speed)
DECLARE_NAMED_PARAM(bw_allocation,fc2BandwidthAllocation,nba,Bandwidth_Allocation)
DECLARE_NAMED_PARAM(interface,fc2InterfaceType,nif,Interface)

typedef struct named_feature {
	const char *		nft_name;
#ifdef HAVE_LIBFLYCAP
	//dc1394feature_t 	nft_feature;
#endif
} Named_Feature;

typedef struct named_trigger_mode {
	const char *		ntm_name;
#ifdef HAVE_LIBFLYCAP
	//dc1394trigger_mode_t	ntm_mode;
#endif
} Named_Trigger_Mode;

#define NO_PGR_FRAME		((PGR_Frame *)NULL)


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

extern int pick_grab_mode(QSP_ARG_DECL PGR_Cam *pgcp, const char *pmpt);

// These can be declared unconditionally if they don't refer to any PGR structs...

extern void show_grab_mode(QSP_ARG_DECL  PGR_Cam *pgcp);
extern void set_grab_mode(QSP_ARG_DECL  PGR_Cam *pgcp, int idx);
extern void set_buffer_obj(QSP_ARG_DECL  PGR_Cam *pgcp, Data_Obj *dp);
extern void set_eii_property(QSP_ARG_DECL  PGR_Cam *pgcp, int idx, int yesno );
extern void show_fmt7_modes(QSP_ARG_DECL  PGR_Cam *pgcp);
extern int set_n_buffers(QSP_ARG_DECL  PGR_Cam *pgcp, int n );
extern void show_n_buffers(QSP_ARG_DECL  PGR_Cam *pgcp);
extern void set_fmt7_size(QSP_ARG_DECL  PGR_Cam *pgcp, int w, int h );

extern void list_cam_properties(QSP_ARG_DECL  PGR_Cam *pgcp);
extern void refresh_camera_properties(QSP_ARG_DECL  PGR_Cam *pgcp);

#ifdef HAVE_LIBFLYCAP
extern void refresh_property_info(QSP_ARG_DECL  PGR_Cam *pgcp, PGR_Property_Type *t);
extern void show_property_info(QSP_ARG_DECL  PGR_Cam *pgcp, PGR_Property_Type *t);
extern void refresh_property_value(QSP_ARG_DECL  PGR_Cam *pgcp, PGR_Property_Type *t);
extern void show_property_value(QSP_ARG_DECL  PGR_Cam *pgcp, PGR_Property_Type *t);

extern void set_prop_value(QSP_ARG_DECL  PGR_Cam *pgcp, PGR_Property_Type *t, PGR_Prop_Val *vp);
extern void set_prop_auto(QSP_ARG_DECL  PGR_Cam *pgcp, PGR_Property_Type *t, BOOL auto_state);
extern unsigned int read_register(QSP_ARG_DECL  PGR_Cam *pgcp, unsigned int addr);
extern void write_register(QSP_ARG_DECL  PGR_Cam *pgcp, unsigned int addr, unsigned int val);
//extern void report_fc2_error(QSP_ARG_DECL   fc2Error error, const char *whence );

//extern dc1394video_mode_t pick_video_mode(QSP_ARG_DECL  PGR_Cam *pgcp, const char *pmpt);
//extern dc1394video_mode_t pick_fmt7_mode(QSP_ARG_DECL  PGR_Cam *pgcp, const char *pmpt);
extern int set_std_mode(QSP_ARG_DECL  PGR_Cam *pgcp, int idx);
extern int is_fmt7_mode(QSP_ARG_DECL  PGR_Cam *pgcp, int idx);
extern int set_fmt7_mode(QSP_ARG_DECL  PGR_Cam *pgcp, int idx );
extern int check_buffer_alignment(QSP_ARG_DECL  PGR_Cam *pgcp);
//extern void report_feature_info(QSP_ARG_DECL  PGR_Cam *pgcp, dc1394feature_t id );
//extern const char *name_for_trigger_mode(dc1394trigger_mode_t mode);
#endif	// HAVE_LIBFLYCAP

extern void cleanup_cam(PGR_Cam *pgcp);
extern int get_camera_names(QSP_ARG_DECL  Data_Obj *dp );
extern int get_video_mode_strings(QSP_ARG_DECL  Data_Obj *dp, PGR_Cam *pgcp);
extern int get_framerate_strings(QSP_ARG_DECL  Data_Obj *dp, PGR_Cam *pgcp);
extern void push_camera_context(QSP_ARG_DECL  PGR_Cam *pgcp);
extern void pop_camera_context(SINGLE_QSP_ARG_DECL);
extern int init_firewire_system(SINGLE_QSP_ARG_DECL);
extern int start_firewire_transmission(QSP_ARG_DECL  PGR_Cam * pgcp, int buf_size );
extern Data_Obj * grab_firewire_frame(QSP_ARG_DECL  PGR_Cam * pgcp );
extern Data_Obj * grab_newest_firewire_frame(QSP_ARG_DECL  PGR_Cam * pgcp );
extern void start_firewire_capture(QSP_ARG_DECL  PGR_Cam * pgcp);
extern void stop_firewire_capture(QSP_ARG_DECL  PGR_Cam * pgcp );
extern int reset_camera(QSP_ARG_DECL  PGR_Cam * pgcp );
extern void list_trig( QSP_ARG_DECL  PGR_Cam * pgcp );
extern void release_oldest_frame(QSP_ARG_DECL  PGR_Cam *pgcp);
extern void report_bandwidth(QSP_ARG_DECL  PGR_Cam *pgcp);
extern void list_framerates(QSP_ARG_DECL  PGR_Cam *pgcp);
extern int list_video_modes(QSP_ARG_DECL  PGR_Cam *pgcp);
extern void show_framerate(QSP_ARG_DECL  PGR_Cam *pgcp);
extern void show_video_mode(QSP_ARG_DECL  PGR_Cam *pgcp);
extern int pick_framerate(QSP_ARG_DECL  PGR_Cam *pgcp, const char *pmpt);
extern int set_framerate(QSP_ARG_DECL  PGR_Cam *pgcp, int mode_index);
extern void print_camera_info(QSP_ARG_DECL  PGR_Cam *pgcp);
extern int list_camera_features(QSP_ARG_DECL  PGR_Cam *pgcp);
extern int get_feature_choices(PGR_Cam *pgcp, const char ***chp);
extern void get_camera_features(PGR_Cam *pgcp);

/* cam_ctl.c */
#ifdef HAVE_LIBFLYCAP
//extern void describe_dc1394_error( QSP_ARG_DECL  dc1394error_t e );
//extern int is_auto_capable( dc1394feature_info_t *feat_p );
//extern int set_camera_framerate(PGR_Cam *pgcp, dc1394framerate_t framerate );
//extern int set_camera_trigger_polarity(PGR_Cam *pgcp,
//	dc1394trigger_polarity_t polarity);
//extern int set_camera_trigger_mode(PGR_Cam *pgcp, dc1394trigger_mode_t mode);
//extern int set_camera_trigger_source(PGR_Cam *pgcp, dc1394trigger_source_t source);
//extern int set_iso_speed(PGR_Cam *pgcp, dc1394speed_t speed);
#endif /* HAVE_LIBFLYCAP */
extern int set_camera_bmode(PGR_Cam *, int);
extern int power_on_camera(PGR_Cam *pgcp);
extern int power_off_camera(PGR_Cam *pgcp);
extern int set_camera_temperature(PGR_Cam *pgcp, int temp);
extern int set_camera_white_balance(PGR_Cam *pgcp, int wb);
extern int set_camera_white_shading(PGR_Cam *pgcp, int val);

extern const char *name_of_indexed_video_mode(int idx);

/* stream_fly.c */
extern void fly_set_async_record(int flag);
extern int fly_get_async_record(void);
extern void stream_record(QSP_ARG_DECL  Image_File *ifp,int32_t n_frames, PGR_Cam *pgcp);
extern COMMAND_FUNC( flycap_wait_record );
extern COMMAND_FUNC( flycap_halt_record );
extern Image_File * get_file_for_recording(QSP_ARG_DECL  const char *name,
		int n_frames,PGR_Cam *pgcp);

#ifndef HAVE_LIBFLYCAP
#endif // ! HAVE_LIBFLYCAP

