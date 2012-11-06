#include "quip_config.h"

#ifdef HAVE_DC1394_DC1394_H
#include <dc1394/dc1394.h>
#endif

#include "query.h"
#include "data_obj.h"

typedef struct pgr_cam {
#ifdef HAVE_LIBDC1394
	dc1394camera_t *	pc_cam_p;
	dc1394video_mode_t	pc_current_video_mode;
	dc1394framerate_t 	pc_current_framerate;
	dc1394video_mode_t *	pc_video_mode_list;
	dc1394featureset_t	pc_features;
#endif /* HAVE_LIBDC1394 */
	unsigned int		pc_nCols;
	unsigned int		pc_nRows;
	int			pc_n_video_modes;
	List *			pc_feat_lp;
	u_long			pc_flags;
} PGR_Cam;

/* flag bits */

#define PGR_CAM_USES_BMODE	1
#define PGR_CAM_IS_RUNNING	2

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
extern void cleanup1394(dc1394camera_t *cam_p);
extern dc1394video_mode_t pick_video_mode(QSP_ARG_DECL  PGR_Cam *pgcp, const char *pmpt);
extern dc1394video_mode_t pick_fmt7_mode(QSP_ARG_DECL  PGR_Cam *pgcp, const char *pmpt);
extern int set_video_mode(PGR_Cam *pgcp, dc1394video_mode_t mode);
extern void report_feature_info( PGR_Cam *pgcp, dc1394feature_t id );
extern const char *name_for_trigger_mode(dc1394trigger_mode_t mode);
#endif
extern PGR_Cam * init_firewire_system(void);
extern int start_firewire_transmission(QSP_ARG_DECL  PGR_Cam * pgcp, int buf_size );
extern PGR_Frame * grab_firewire_frame(QSP_ARG_DECL  PGR_Cam * pgcp );
extern PGR_Frame * grab_newest_firewire_frame(QSP_ARG_DECL  PGR_Cam * pgcp );
extern int stop_firewire_capture( PGR_Cam * pgcp );
extern int reset_camera( PGR_Cam * pgcp );
extern void list_trig( PGR_Cam * pgcp );
extern void release_oldest_frame(PGR_Cam *pgcp);
extern void report_bandwidth(PGR_Cam *pgcp);
extern int list_framerates(PGR_Cam *pgcp);
extern int list_video_modes(PGR_Cam *pgcp);
extern int pick_framerate(QSP_ARG_DECL  PGR_Cam *pgcp, const char *pmpt);
extern int set_framerate(PGR_Cam *pgcp, int mode_index);
extern void print_camera_info(PGR_Cam *pgcp);
extern int list_camera_features(PGR_Cam *pgcp);
extern int get_feature_choices(PGR_Cam *pgcp, const char ***chp);
extern void get_camera_features(PGR_Cam *pgcp);

/* cam_ctl.c */
#ifdef HAVE_LIBDC1394
extern int set_camera_framerate(PGR_Cam *pgcp, dc1394framerate_t framerate );
extern int set_camera_trigger_polarity(PGR_Cam *pgcp,
	dc1394trigger_polarity_t polarity);
extern int set_camera_trigger_mode(PGR_Cam *pgcp, dc1394trigger_mode_t mode);
extern int set_camera_trigger_source(PGR_Cam *pgcp, dc1394trigger_source_t source);
#endif /* HAVE_LIBDC1394 */
extern int set_camera_bmode(PGR_Cam *, int);

