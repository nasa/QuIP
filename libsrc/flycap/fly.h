#include "quip_config.h"

#include "data_obj.h"
#include "fio_api.h"

#ifdef HAVE_LIBFLYCAP
#include <flycapture/C/FlyCapture2_C.h>
#endif // HAVE_LIBFLYCAP

#define N_FMT7_MODES	5	// PGR has 32, but why bother?

typedef int Framerate_Mask;

typedef struct fly_cam {
	const char *		fc_name;
#ifdef HAVE_LIBFLYCAP
	fc2Context		fc_context;
	fc2PGRGuid		fc_guid;
	fc2CameraInfo		fc_cam_info;
	fc2EmbeddedImageInfo	fc_ei_info;
	fc2VideoMode		fc_video_mode;	// current
	fc2FrameRate 		fc_framerate;	// current
	fc2Config		fc_config;
	fc2Format7Info *	fc_fmt7_info_tbl;
	fc2Image *		fc_img_p;

	unsigned char *		fc_base;	// for captured frames...
	long			fc_buf_delta;
	int	 		fc_framerate_index;	// into all_framerates
	int	 		fc_video_mode_index;	// into all_video_modes
	int	 		fc_fmt7_index;		// of available...

	int	 		fc_my_video_mode_index;	// into private tbl
	int			fc_n_video_modes;
	int			fc_n_fmt7_modes;
	int *			fc_video_mode_indices;
	const char **		fc_video_mode_names;
	Framerate_Mask *	fc_framerate_mask_tbl;	// one for every video mode
	int			fc_n_framerates;
	const char **		fc_framerate_names;
#endif /* HAVE_LIBFLYCAP */

	unsigned int		fc_cols;
	unsigned int		fc_rows;
	unsigned int		fc_depth;	// bytes per pixel
	int			fc_n_buffers;
	int			fc_newest;
	Data_Obj **		fc_frm_dp_tbl;
	Item_Context *		fc_do_icp;	// data_obj context
	List *			fc_in_use_lp;	// list of frames...
	List *			fc_feat_lp;
	u_long			fc_flags;
} Fly_Cam;

ITEM_INTERFACE_PROTOTYPES(Fly_Cam,fly_cam)

#define new_fly_cam(s)	_new_fly_cam(QSP_ARG  s)
#define fly_cam_of(s)	_fly_cam_of(QSP_ARG  s)
#define list_fly_cams(fp)	_list_fly_cams(QSP_ARG  fp)
#define pick_fly_cam(s)	_pick_fly_cam(QSP_ARG  s)
#define fly_cam_list()	_fly_cam_list(SINGLE_QSP_ARG)

/* flag bits */

#define FLY_CAM_USES_BMODE	1
#define FLY_CAM_IS_RUNNING	2
#define FLY_CAM_IS_CAPTURING	4
#define FLY_CAM_IS_TRANSMITTING	8

#define IS_CAPTURING(fcp)	(fcp->fc_flags & FLY_CAM_IS_CAPTURING)
#define IS_TRANSMITTING(fcp)	(fcp->fc_flags & FLY_CAM_IS_TRANSMITTING)

typedef struct pgr_property_type {
	const char *		name;
#ifdef HAVE_LIBFLYCAP
	fc2PropertyType		type_code;
	fc2PropertyInfo 	info;
	fc2Property 		prop;
#endif // HAVE_LIBFLYCAP
} Fly_Cam_Property_Type;

ITEM_INTERFACE_PROTOTYPES(Fly_Cam_Property_Type,pgr_prop)

#define pick_pgr_prop(p)	_pick_pgr_prop(QSP_ARG  p)
#define get_pgr_prop(p)		_get_pgr_prop(QSP_ARG  p)
#define new_pgr_prop(p)		_new_pgr_prop(QSP_ARG  p)
#define pgr_prop_list()		_pgr_prop_list(SINGLE_QSP_ARG)

typedef struct fly_frame {
	const char *		pf_name;
#ifdef HAVE_LIBFLYCAP
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
} Fly_Cam_Prop_Val;


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

#define MIN_STROBE_SOURCE	0
#define MAX_STROBE_SOURCE	3

/* fly.c */

extern const char *eii_prop_names[];

extern int pick_grab_mode(QSP_ARG_DECL Fly_Cam *fcp, const char *pmpt);

// These can be declared unconditionally if they don't refer to any PGR structs...

extern void show_grab_mode(QSP_ARG_DECL  Fly_Cam *fcp);
extern void set_grab_mode(QSP_ARG_DECL  Fly_Cam *fcp, int idx);
extern void set_buffer_obj(QSP_ARG_DECL  Fly_Cam *fcp, Data_Obj *dp);
extern void set_eii_property(QSP_ARG_DECL  Fly_Cam *fcp, int idx, int yesno );
extern void show_fmt7_modes(QSP_ARG_DECL  Fly_Cam *fcp);
extern void show_fmt7_info(QSP_ARG_DECL  Fly_Cam *fcp, fc2Mode mode );
extern int set_n_buffers(QSP_ARG_DECL  Fly_Cam *fcp, int n );
extern void show_n_buffers(QSP_ARG_DECL  Fly_Cam *fcp);
extern void set_fmt7_size(QSP_ARG_DECL  Fly_Cam *fcp, int w, int h );

extern void list_fly_cam_properties(QSP_ARG_DECL  Fly_Cam *fcp);
extern void refresh_fly_cam_properties(QSP_ARG_DECL  Fly_Cam *fcp);

#ifdef HAVE_LIBFLYCAP
extern void refresh_property_info(QSP_ARG_DECL  Fly_Cam *fcp, Fly_Cam_Property_Type *t);
extern void show_property_info(QSP_ARG_DECL  Fly_Cam *fcp, Fly_Cam_Property_Type *t);
extern void refresh_property_value(QSP_ARG_DECL  Fly_Cam *fcp, Fly_Cam_Property_Type *t);
extern void show_property_value(QSP_ARG_DECL  Fly_Cam *fcp, Fly_Cam_Property_Type *t);

extern void set_prop_value(QSP_ARG_DECL  Fly_Cam *fcp, Fly_Cam_Property_Type *t, Fly_Cam_Prop_Val *vp);
extern void set_prop_auto(QSP_ARG_DECL  Fly_Cam *fcp, Fly_Cam_Property_Type *t, BOOL auto_state);
extern unsigned int read_register(QSP_ARG_DECL  Fly_Cam *fcp, unsigned int addr);
extern void write_register(QSP_ARG_DECL  Fly_Cam *fcp, unsigned int addr, unsigned int val);
//extern void report_fc2_error(QSP_ARG_DECL   fc2Error error, const char *whence );

//extern dc1394video_mode_t pick_video_mode(QSP_ARG_DECL  Fly_Cam *fcp, const char *pmpt);
//extern dc1394video_mode_t pick_fmt7_mode(QSP_ARG_DECL  Fly_Cam *fcp, const char *pmpt);
extern int set_std_mode(QSP_ARG_DECL  Fly_Cam *fcp, int idx);
extern int is_fmt7_mode(QSP_ARG_DECL  Fly_Cam *fcp, int idx);
extern int set_fmt7_mode(QSP_ARG_DECL  Fly_Cam *fcp, int idx );
extern int check_buffer_alignment(QSP_ARG_DECL  Fly_Cam *fcp);
//extern void report_feature_info(QSP_ARG_DECL  Fly_Cam *fcp, dc1394feature_t id );
//extern const char *name_for_trigger_mode(dc1394trigger_mode_t mode);
#endif	// HAVE_LIBFLYCAP

extern void cleanup_fly_cam(Fly_Cam *fcp);
extern int get_fly_cam_names(QSP_ARG_DECL  Data_Obj *dp );
extern int get_fly_cam_video_mode_strings(QSP_ARG_DECL  Data_Obj *dp, Fly_Cam *fcp);
extern int get_fly_cam_framerate_strings(QSP_ARG_DECL  Data_Obj *dp, Fly_Cam *fcp);
extern void push_fly_cam_context(QSP_ARG_DECL  Fly_Cam *fcp);
extern void pop_fly_cam_context(SINGLE_QSP_ARG_DECL);
extern int init_fly_cam_system(SINGLE_QSP_ARG_DECL);
extern int start_firewire_transmission(QSP_ARG_DECL  Fly_Cam * fcp, int buf_size );
extern Data_Obj * grab_fly_cam_frame(QSP_ARG_DECL  Fly_Cam * fcp );
extern Data_Obj * grab_newest_firewire_frame(QSP_ARG_DECL  Fly_Cam * fcp );
extern void start_firewire_capture(QSP_ARG_DECL  Fly_Cam * fcp);
extern void stop_firewire_capture(QSP_ARG_DECL  Fly_Cam * fcp );
extern int reset_fly_cam(QSP_ARG_DECL  Fly_Cam * fcp );
extern void list_fly_cam_trig( QSP_ARG_DECL  Fly_Cam * fcp );
extern void get_strobe_info( QSP_ARG_DECL  Fly_Cam * fcp, int source );
extern void get_strobe_control( QSP_ARG_DECL  Fly_Cam * fcp, int source );
extern void set_strobe_enable( QSP_ARG_DECL  Fly_Cam * fcp, int source, int enable );
extern void set_strobe_polarity( QSP_ARG_DECL  Fly_Cam * fcp, int source, unsigned int polarity );
extern void set_strobe_delay( QSP_ARG_DECL  Fly_Cam * fcp, int source, int delay );
extern void set_strobe_duration( QSP_ARG_DECL  Fly_Cam * fcp, int source, int duration );
extern void release_oldest_frame(QSP_ARG_DECL  Fly_Cam *fcp);
extern void report_fly_cam_bandwidth(QSP_ARG_DECL  Fly_Cam *fcp);
extern void list_fly_cam_framerates(QSP_ARG_DECL  Fly_Cam *fcp);
extern int list_fly_cam_video_modes(QSP_ARG_DECL  Fly_Cam *fcp);
extern void show_fly_cam_framerate(QSP_ARG_DECL  Fly_Cam *fcp);
extern void show_fly_cam_video_mode(QSP_ARG_DECL  Fly_Cam *fcp);
extern int pick_fly_cam_framerate(QSP_ARG_DECL  Fly_Cam *fcp, const char *pmpt);
extern int set_framerate(QSP_ARG_DECL  Fly_Cam *fcp, int mode_index);
extern void print_fly_cam_info(QSP_ARG_DECL  Fly_Cam *fcp);
extern int list_fly_cam_features(QSP_ARG_DECL  Fly_Cam *fcp);
extern int get_feature_choices(Fly_Cam *fcp, const char ***chp);
extern void get_fly_cam_features(Fly_Cam *fcp);

extern void report_fmt7_modes(QSP_ARG_DECL  Fly_Cam *the_cam_p);

/* cam_ctl.c */
#ifdef HAVE_LIBFLYCAP
//extern void describe_dc1394_error( QSP_ARG_DECL  dc1394error_t e );
//extern int is_auto_capable( dc1394feature_info_t *feat_p );
//extern int set_fly_cam_framerate(Fly_Cam *fcp, dc1394framerate_t framerate );
//extern int set_fly_cam_trigger_polarity(Fly_Cam *fcp,
//	dc1394trigger_polarity_t polarity);
//extern int set_fly_cam_trigger_mode(Fly_Cam *fcp, dc1394trigger_mode_t mode);
//extern int set_fly_cam_trigger_source(Fly_Cam *fcp, dc1394trigger_source_t source);
//extern int set_iso_speed(Fly_Cam *fcp, dc1394speed_t speed);
#endif /* HAVE_LIBFLYCAP */
extern int set_fly_cam_bmode(Fly_Cam *, int);
extern int power_on_fly_cam(Fly_Cam *fcp);
extern int power_off_fly_cam(Fly_Cam *fcp);
extern int set_fly_cam_temperature(Fly_Cam *fcp, int temp);
extern int set_fly_cam_white_balance(Fly_Cam *fcp, int wb);
extern int set_fly_cam_white_shading(Fly_Cam *fcp, int val);

extern const char *name_of_indexed_video_mode(int idx);

/* stream_fly.c */
extern void fly_set_async_record(int flag);
extern int fly_get_async_record(void);
extern void stream_record(QSP_ARG_DECL  Image_File *ifp,int32_t n_frames, Fly_Cam *fcp);
extern COMMAND_FUNC( flycap_wait_record );
extern COMMAND_FUNC( flycap_halt_record );
extern Image_File * _get_file_for_recording(QSP_ARG_DECL  const char *name,
		int n_frames,Fly_Cam *fcp);
#define get_file_for_recording(name,n_f,fcp)	_get_file_for_recording(QSP_ARG  name,n_f,fcp)

#ifndef HAVE_LIBFLYCAP
#endif // ! HAVE_LIBFLYCAP

