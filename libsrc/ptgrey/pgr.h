#include "quip_config.h"

#ifdef HAVE_DC1394_DC1394_H
#include <dc1394/dc1394.h>
#endif

#include "query.h"
#include "data_obj.h"

typedef struct pgr_cam {
	const char *		pc_name;
#ifdef HAVE_LIBDC1394
	dc1394camera_t *	pc_cam_p;
	dc1394video_mode_t	pc_video_mode;	// current
	dc1394framerate_t 	pc_framerate;	// current
	dc1394video_modes_t	pc_video_modes;
	dc1394framerates_t 	pc_framerates;
	dc1394featureset_t	pc_features;
	dc1394capture_policy_t	pc_policy;
#endif /* HAVE_LIBDC1394 */
	unsigned int		pc_nCols;
	unsigned int		pc_nRows;
	int			pc_ring_buffer_size;
	int			pc_n_avail;
	List *			pc_in_use_lp;	// list of frames...
	List *			pc_feat_lp;
	Item_Context *		pc_do_icp;	// data_obj context
	Item_Context *		pc_pf_icp;	// pframe context
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

typedef struct pgr_frame {
	const char *		pf_name;
#ifdef HAVE_LIBDC1394
	dc1394video_frame_t *	pf_framep;
#endif
	Data_Obj *		pf_dp;
} PGR_Frame;

typedef struct named_video_mode {
	const char *		nvm_name;
#ifdef HAVE_LIBDC1394
	dc1394video_mode_t	nvm_mode;
#endif
} Named_Video_Mode;

typedef struct named_color_coding {
	const char *		ncc_name;
#ifdef HAVE_LIBDC1394
	dc1394color_coding_t	ncc_code;
#endif
} Named_Color_Coding;

typedef struct named_framerate {
	const char *		nfr_name;
#ifdef HAVE_LIBDC1394
	dc1394framerate_t 	nfr_framerate;
#endif
} Named_Frame_Rate;

typedef struct named_feature {
	const char *		nft_name;
#ifdef HAVE_LIBDC1394
	dc1394feature_t 	nft_feature;
#endif
} Named_Feature;

typedef struct named_trigger_mode {
	const char *		ntm_name;
#ifdef HAVE_LIBDC1394
	dc1394trigger_mode_t	ntm_mode;
#endif
} Named_Trigger_Mode;

#define NO_PGR_FRAME		((PGR_Frame *)NULL)



/* pgr.c */
#ifdef HAVE_LIBDC1394

#define BAD_VIDEO_MODE ((dc1394video_mode_t)-1)

extern void cleanup_cam(PGR_Cam *pgcp);
extern dc1394video_mode_t pick_video_mode(QSP_ARG_DECL  PGR_Cam *pgcp, const char *pmpt);
extern dc1394video_mode_t pick_fmt7_mode(QSP_ARG_DECL  PGR_Cam *pgcp, const char *pmpt);
extern int set_video_mode(QSP_ARG_DECL  PGR_Cam *pgcp, dc1394video_mode_t mode);
extern void report_feature_info(QSP_ARG_DECL  PGR_Cam *pgcp, dc1394feature_t id );
extern const char *name_for_trigger_mode(dc1394trigger_mode_t mode);
#endif	// HAVE_LIBDC1394
extern int get_camera_names(QSP_ARG_DECL  Data_Obj *dp );
extern int get_video_mode_strings(QSP_ARG_DECL  Data_Obj *dp, PGR_Cam *pgcp);
extern int get_framerate_strings(QSP_ARG_DECL  Data_Obj *dp, PGR_Cam *pgcp);
extern void push_camera_context(QSP_ARG_DECL  PGR_Cam *pgcp);
extern void pop_camera_context(SINGLE_QSP_ARG_DECL);
extern int init_firewire_system(SINGLE_QSP_ARG_DECL);
extern int start_firewire_transmission(QSP_ARG_DECL  PGR_Cam * pgcp, int buf_size );
extern Data_Obj * grab_firewire_frame(QSP_ARG_DECL  PGR_Cam * pgcp );
extern Data_Obj * grab_newest_firewire_frame(QSP_ARG_DECL  PGR_Cam * pgcp );
extern int stop_firewire_capture( PGR_Cam * pgcp );
extern int reset_camera( PGR_Cam * pgcp );
extern void list_trig( QSP_ARG_DECL  PGR_Cam * pgcp );
extern void release_oldest_frame(PGR_Cam *pgcp);
extern void report_bandwidth(QSP_ARG_DECL  PGR_Cam *pgcp);
extern int list_framerates(QSP_ARG_DECL  PGR_Cam *pgcp);
extern int list_video_modes(QSP_ARG_DECL  PGR_Cam *pgcp);
extern void show_framerate(QSP_ARG_DECL  PGR_Cam *pgcp);
extern void show_video_mode(QSP_ARG_DECL  PGR_Cam *pgcp);
extern int pick_framerate(QSP_ARG_DECL  PGR_Cam *pgcp, const char *pmpt);
extern int set_framerate(PGR_Cam *pgcp, int mode_index);
extern void print_camera_info(QSP_ARG_DECL  PGR_Cam *pgcp);
extern int list_camera_features(QSP_ARG_DECL  PGR_Cam *pgcp);
extern int get_feature_choices(PGR_Cam *pgcp, const char ***chp);
extern void get_camera_features(PGR_Cam *pgcp);

/* cam_ctl.c */
#ifdef HAVE_LIBDC1394
extern void describe_dc1394_error( QSP_ARG_DECL  dc1394error_t e );
extern int is_auto_capable( dc1394feature_info_t *feat_p );
extern int set_camera_framerate(PGR_Cam *pgcp, dc1394framerate_t framerate );
extern int set_camera_trigger_polarity(PGR_Cam *pgcp,
	dc1394trigger_polarity_t polarity);
extern int set_camera_trigger_mode(PGR_Cam *pgcp, dc1394trigger_mode_t mode);
extern int set_camera_trigger_source(PGR_Cam *pgcp, dc1394trigger_source_t source);
extern int set_iso_speed(PGR_Cam *pgcp, dc1394speed_t speed);
#endif /* HAVE_LIBDC1394 */
extern int set_camera_bmode(PGR_Cam *, int);
extern int power_on_camera(PGR_Cam *pgcp);
extern int power_off_camera(PGR_Cam *pgcp);
extern int set_camera_temperature(PGR_Cam *pgcp, int temp);
extern int set_camera_white_balance(PGR_Cam *pgcp, int wb);
extern int set_camera_white_shading(PGR_Cam *pgcp, int val);

