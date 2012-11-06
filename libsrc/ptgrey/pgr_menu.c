#include "quip_config.h"

char VersionId_ptgrey_pgr_menu[] = QUIP_VERSION_STRING;

#include "query.h"
#include "submenus.h"
#include "pgr.h"
#include "data_obj.h"

static PGR_Cam *the_cam_p=NULL;

#ifdef HAVE_LIBDC1394
static dc1394video_mode_t the_fmt7_mode=DC1394_VIDEO_MODE_FORMAT7_0;
#endif

#ifndef HAVE_LIBDC1394

#define NO_LIB_MSG(whence)						\
									\
	sprintf(ERROR_STRING,						\
		"%s:  program built without dc1394 support!?",whence);	\
	ERROR1(ERROR_STRING);

#define EAT_ONE_DUMMY(whence)								\
											\
	const char *s;									\
	s=NAMEOF("dummy word");								\
	NO_LIB_MSG(whence)

#endif /* ! HAVE_LIBDC1394 */

#define CHECK_CAM	if( the_cam_p == NULL ){ \
		WARN("Firewire system not initialized yet..."); \
		return; }

static COMMAND_FUNC(do_list_trig)
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM
	list_trig(the_cam_p);
#else
	NO_LIB_MSG("do_list_trig");
#endif
}

Command trigger_ctbl[]={
{ "list",	do_list_trig,	"report trigger info"			},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL_COMMAND								}
};

static COMMAND_FUNC( do_trigger )
{
	PUSHCMD(trigger_ctbl,"trigger");
}

#ifdef FOOBAR
static COMMAND_FUNC( do_frame )
{
	char *s;
	int i;

	s=NAMEOF( "name for frame object" );
	i=HOW_MANY("capture buffer index");

	CHECK_CAM

	make_frame_obj(s,i);
}
#endif /* FOOBAR */

static COMMAND_FUNC( do_init )
{
#ifdef HAVE_LIBDC1394
	if( the_cam_p != NULL ){
		WARN("Firewire system already initialized!?");
		return;
	}

	if( (the_cam_p=init_firewire_system()) == NULL )
		WARN("Unable to initialize firewire subsystem");
#endif
}

static COMMAND_FUNC( do_list )
{
#ifdef HAVE_LIBDC1394
	/* print out info about this camera */
	int bpp=1; /* BUG determine this from video mode */

	CHECK_CAM
	sprintf(msg_str,"%s %s:  %dx%d, %d %s per pixel",
		the_cam_p->pc_cam_p->vendor,
		the_cam_p->pc_cam_p->model,
		the_cam_p->pc_nRows,the_cam_p->pc_nCols, bpp,
		bpp>1?"bytes":"byte");
	prt_msg(msg_str);
#else
	NO_LIB_MSG("do_list");
#endif
}

static COMMAND_FUNC( do_start )
{
#ifdef HAVE_LIBDC1394
	int rb_size;

	rb_size = HOW_MANY("number of frames in capture ring buffer");
	if( rb_size < 1 ){
		WARN("number of ring buffer frames must be positive");
		return;
	} else if( rb_size > 64 ){
		sprintf(ERROR_STRING,"You have asked for %d ring buffer frames, which is more than 64",
			rb_size);
		WARN(ERROR_STRING);
		/* honor the request! */
	}

	CHECK_CAM

	if( the_cam_p->pc_flags & PGR_CAM_IS_RUNNING ){
		WARN("do_start:  camera is already capturing!?");
		return;
	}

	if( start_firewire_transmission(QSP_ARG  the_cam_p, rb_size ) < 0 ){
		cleanup1394(the_cam_p->pc_cam_p);
		the_cam_p=NULL;
		return;
	}

	the_cam_p->pc_flags |= PGR_CAM_IS_RUNNING;
#else
	NO_LIB_MSG("do_start");
#endif
}

static COMMAND_FUNC( do_grab )
{
#ifdef HAVE_LIBDC1394
	PGR_Frame *pfp;

	CHECK_CAM
	if( (pfp=grab_firewire_frame(QSP_ARG  the_cam_p )) == NULL ){		/* any error */
		cleanup1394(the_cam_p->pc_cam_p);	/* grab error */
		the_cam_p=NULL;
	}
#else
	NO_LIB_MSG("do_grab");
#endif
}

static COMMAND_FUNC( do_grab_newest )
{
#ifdef HAVE_LIBDC1394
	PGR_Frame *pfp;

	CHECK_CAM
	if( (pfp=grab_newest_firewire_frame(QSP_ARG  the_cam_p )) == NULL ){		/* any error */
		cleanup1394(the_cam_p->pc_cam_p);	/* grab error */
		the_cam_p=NULL;
	}
	sprintf(msg_str,"%d",pfp->pf_framep->id);
	ASSIGN_VAR("newest",msg_str);
#else
	NO_LIB_MSG("do_grab_newest");
#endif
}

static COMMAND_FUNC( do_stop )
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM
	stop_firewire_capture( the_cam_p );
#else
	NO_LIB_MSG("do_stop");
#endif
}

static COMMAND_FUNC(do_power)
{
	CHECK_CAM
	WARN("do_power unimplemented");
}

static COMMAND_FUNC(do_reset)
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM
	reset_camera(the_cam_p);
#endif
}

// conflict started here???
static COMMAND_FUNC( do_release )
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM
	release_oldest_frame(the_cam_p);
#endif
}

static COMMAND_FUNC( do_close )
{
	CHECK_CAM
#ifdef HAVE_LIBDC1394
	cleanup1394(the_cam_p->pc_cam_p);
#endif
	the_cam_p=NULL;
}

static COMMAND_FUNC( do_bw )
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM

	report_bandwidth(the_cam_p);
#endif
}

static COMMAND_FUNC( do_list_modes )
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM
	prt_msg("\nAvailable video modes:");
	list_video_modes(the_cam_p);
#endif
}

static COMMAND_FUNC( do_list_framerates )
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM

	prt_msg("\nAvailable framerates:");
	list_framerates(the_cam_p);
#endif
}

static COMMAND_FUNC( do_set_video_mode )
{
#ifdef HAVE_LIBDC1394
	dc1394video_mode_t m;

	m=pick_video_mode(QSP_ARG  the_cam_p,"video mode");
	if( m >= 0 )
		set_video_mode(the_cam_p,m);
#else
	EAT_ONE_DUMMY("do_set_video_mode");
#endif
}

static COMMAND_FUNC( do_set_framerate )
{
#ifdef HAVE_LIBDC1394
	int i;

	i=pick_framerate(QSP_ARG  the_cam_p,"frame rate");
	if( i >= 0 )
		set_framerate(the_cam_p,i);
#else
	EAT_ONE_DUMMY("do_set_framerate");
#endif
}

static COMMAND_FUNC( do_cam_info )
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM
	print_camera_info(the_cam_p);
#else
	NO_LIB_MSG("do_cam_info");
#endif
}

#ifdef HAVE_LIBDC1394
dc1394feature_info_t *curr_feat_p =NULL;
#endif

static COMMAND_FUNC( do_get_frange )
{
#ifdef HAVE_LIBDC1394
	char s[32];

	if( curr_feat_p->id == DC1394_FEATURE_TRIGGER ){
		WARN("trigger feature does not have a range");
		return;
	}

	sprintf(s,"%d",curr_feat_p->value);
	ASSIGN_VAR("fval",s);
	sprintf(s,"%d",curr_feat_p->min);
	ASSIGN_VAR("fmin",s);
	sprintf(s,"%d",curr_feat_p->max);
	ASSIGN_VAR("fmax",s);

	/*
	if( curr_feat_p->auto_capable ){
		ASSIGN_VAR("f_auto_capable","1");
		sprintf(s,"%d",curr_feat_p->auto_active);
		ASSIGN_VAR("f_auto",s);
	} else {
		ASSIGN_VAR("f_auto_capable","0");
		ASSIGN_VAR("f_auto","0");
	}
	*/

	if( curr_feat_p->on_off_capable ){
		ASSIGN_VAR("f_onoff_capable","1");
		sprintf(s,"%d",curr_feat_p->is_on);
		ASSIGN_VAR("f_is_on",s);
	} else {
		ASSIGN_VAR("f_onoff_capable","0");
		ASSIGN_VAR("f_is_on","1");
	}
#else
	NO_LIB_MSG("do_get_frange");
#endif
}

#ifdef HAVE_LIBDC1394
void camera_feature_set_auto(PGR_Cam *pgcp, dc1394feature_info_t *f, int yn)
{
	/* make sure this feature supports auto */
	/*
	if( ! f->auto_capable ){
		sprintf(ERROR_STRING,"%s is not auto-capable",
			//dc1394_feature_desc[f->id - DC1394_FEATURE_MIN]
			dc1394_feature_get_string(f->id) );
		WARN(ERROR_STRING);
		return;
	}
	*/

	if( dc1394_feature_set_mode( pgcp->pc_cam_p, f->id,
		yn ?  DC1394_FEATURE_MODE_AUTO : DC1394_FEATURE_MODE_MANUAL ) !=
		DC1394_SUCCESS ){

		sprintf(DEFAULT_ERROR_STRING,"error setting %s mode for %s",
			yn?"auto":"manual",
			/*dc1394_feature_desc[f->id - DC1394_FEATURE_MIN]*/
			dc1394_feature_get_string(f->id) );
		NWARN(DEFAULT_ERROR_STRING);
	}
}

void camera_feature_set_onoff(PGR_Cam *pgcp, dc1394feature_info_t *f, int yn)
{
	/* make sure this feature supports auto */
	if( ! f->on_off_capable ){
		sprintf(DEFAULT_ERROR_STRING,"%s is not on_off-capable",
			/*dc1394_feature_desc[f->id - DC1394_FEATURE_MIN]*/
			dc1394_feature_get_string(f->id) );
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	if( dc1394_feature_set_power( pgcp->pc_cam_p, f->id,
		yn ?  DC1394_ON : DC1394_OFF ) != DC1394_SUCCESS ){

		sprintf(DEFAULT_ERROR_STRING,"error turning %s %s",
			yn?"on":"off",
			/*dc1394_feature_desc[f->id - DC1394_FEATURE_MIN]*/
			dc1394_feature_get_string(f->id) );
		NWARN(DEFAULT_ERROR_STRING);
	}
}

int camera_feature_set_value(PGR_Cam *pgcp, dc1394feature_info_t *f, int val)
{
	if( dc1394_feature_set_value( pgcp->pc_cam_p, f->id, val ) != DC1394_SUCCESS ){
		sprintf(DEFAULT_ERROR_STRING,"error setting value (%d) for %s",val,
			/* dc1394_feature_desc[f->id - DC1394_FEATURE_MIN]*/
			dc1394_feature_get_string(f->id) );
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	return 0;
}
#endif

static COMMAND_FUNC( do_set_auto )
{
#ifdef HAVE_LIBDC1394
	int yn;
	char pmpt[LLEN];

	sprintf(pmpt,"set %s automatically",
			/* dc1394_feature_desc[curr_feat_p->id - DC1394_FEATURE_MIN]*/
			dc1394_feature_get_string(curr_feat_p->id) );
	yn=ASKIF(pmpt);

	camera_feature_set_auto(the_cam_p,curr_feat_p,yn);
#else
	EAT_ONE_DUMMY("do_set_auto")
#endif
	
}

static COMMAND_FUNC( do_set_onoff )
{
#ifdef HAVE_LIBDC1394
	int yn;
	char pmpt[LLEN];

	sprintf(pmpt,"turn %s on",
			/*dc1394_feature_desc[curr_feat_p->id - DC1394_FEATURE_MIN]*/
			dc1394_feature_get_string(curr_feat_p->id) );
	yn=ASKIF(pmpt);

	camera_feature_set_onoff(the_cam_p,curr_feat_p,yn);
#else
	EAT_ONE_DUMMY("do_set_onoff")
#endif
}

static COMMAND_FUNC( do_set_fval )
{
#ifdef HAVE_LIBDC1394
	char pmpt[LLEN];
	unsigned int val;

	sprintf(pmpt,"%s (%d-%d)",
		/*dc1394_feature_desc[curr_feat_p->id - DC1394_FEATURE_MIN]*/
		dc1394_feature_get_string(curr_feat_p->id),
		curr_feat_p->min,curr_feat_p->max);
	val=HOW_MANY(pmpt);

	if( val < curr_feat_p->min || val > curr_feat_p->max ){
		sprintf(ERROR_STRING,"Value %d is out of range for %s (%d-%d)",
			val,
		/*dc1394_feature_desc[curr_feat_p->id - DC1394_FEATURE_MIN]*/
		dc1394_feature_get_string(curr_feat_p->id),
		curr_feat_p->min,curr_feat_p->max);
		WARN(ERROR_STRING);
		return;
	}

	/* Before we try and set, make sure that this is enabled/manual!? */

	if( camera_feature_set_value(the_cam_p,curr_feat_p,val) < 0 )
		WARN("error setting feature value");
#else
	EAT_ONE_DUMMY("do_set_fval")
#endif
}

static COMMAND_FUNC( change_done )
{
	POPCMD();
	/* refresh all the features so we can display the current values */
	get_camera_features(the_cam_p);

#ifdef HAVE_LIBDC1394
	report_feature_info(the_cam_p,curr_feat_p->id);
#endif
}

static Command fchng_ctbl[]={
{ "fetch",	do_get_frange,	"get current value in $fval, range limits in $fmin $fmax"	},
{ "value",	do_set_fval,	"set value"				},
{ "auto",	do_set_auto,	"set/clear auto flag"			},
{ "on",		do_set_onoff,	"set/clear on_off flag"			},
{ "quit",	change_done,	"exit submenu"				},
{ NULL_COMMAND								}
};


static COMMAND_FUNC( do_feature_change )
{
#ifdef HAVE_LIBDC1394
	const char **choices;
	int n,i;
	Node *np;

	CHECK_CAM

	n = get_feature_choices(the_cam_p,&choices);
	if( n < 1 ) return;

	i=WHICH_ONE("feature",n,choices);
	if( i < 0 ) return;

	np = nth_elt(the_cam_p->pc_feat_lp,i);
	curr_feat_p = (dc1394feature_info_t *) np->n_data;
#else
	EAT_ONE_DUMMY("do_feature_change");
#endif

	PUSHCMD( fchng_ctbl,"feature_change");
}

static COMMAND_FUNC( do_list_cam_features )
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM
	list_camera_features(the_cam_p);
#endif
}

static COMMAND_FUNC( do_feature_info )
{
#ifdef HAVE_LIBDC1394
	const char **choices;
	int n,i;
	Node *np;
	dc1394feature_info_t *f;

	CHECK_CAM

	n = get_feature_choices(the_cam_p,&choices);
	if( n < 1 ) return;

	i=WHICH_ONE("feature",n,choices);
	if( i < 0 ) return;

	np=nth_elt(the_cam_p->pc_feat_lp,i);
	f=(dc1394feature_info_t *)np->n_data;
	report_feature_info(the_cam_p,f->id);
#else
	EAT_ONE_DUMMY("do_feature_info");
#endif
}

static Command feature_ctbl[]={
{ "list",	do_list_cam_features,	"list available features for this camera"	},
{ "info",	do_feature_info,	"report feature info"				},
{ "change",	do_feature_change,	"change feature settings"			},
{ "quit",	popcmd,			"exit submenu"					},
{ NULL_COMMAND										}
};

static COMMAND_FUNC( feature_menu )
{
	PUSHCMD(feature_ctbl,"features");
}

static Command cam_ctbl[]={
{ "list_video_modes",	do_list_modes,		"list all video modes for this camera"	},
{ "list_framerates",	do_list_framerates,	"list all framerates for this camera"	},
{ "set_video_mode",	do_set_video_mode,	"set video mode"			},
{ "set_framerate",	do_set_framerate,	"set framerate"				},
{ "info",		do_cam_info,		"print info about current camera"	},
{ "features",		feature_menu,		"camera feature submenu"		},
{ "quit",		popcmd,			"exit submenu"				},
{ NULL_COMMAND										}
};

static COMMAND_FUNC( cam_menu )
{
	PUSHCMD(cam_ctbl,"camera");
}

static Command capt_ctbl[]={
{ "start",	do_start,	"start capture"			},
{ "grab",	do_grab,	"grab a frame"			},
{ "grab_newest",do_grab_newest,	"grab the newest frame"		},
{ "release",	do_release,	"release a frame"		},
{ "stop",	do_stop,	"stop capture"			},
{ "quit",	popcmd,		"exit submenu"			},
{ NULL_COMMAND							}
};

static COMMAND_FUNC( captmenu )
{
	PUSHCMD( capt_ctbl, "capture" );
}

#define CAM_P	the_cam_p->pc_cam_p

static COMMAND_FUNC( do_fmt7_list )
{
#ifdef HAVE_LIBDC1394
	uint32_t w,h,v;
	uint32_t unit_bytes, max_bytes, packet_bytes, bpp, ppf;

	CHECK_CAM

	if( dc1394_format7_get_max_image_size(CAM_P,the_fmt7_mode,&w,&h) != DC1394_SUCCESS ){
		WARN("error getting max_image_size");
		return;
	}
	sprintf(msg_str,"max image size:  %d x %d",w,h);
	prt_msg(msg_str);

	if( dc1394_format7_get_unit_size(CAM_P,the_fmt7_mode, &h, &v ) != DC1394_SUCCESS ){
		WARN("error getting unit_size");
		return;
	}

	sprintf(msg_str,"unit size:  %d x %d",h,v);
	prt_msg(msg_str);

	if( dc1394_format7_get_image_size(CAM_P,the_fmt7_mode, &w, &h ) != DC1394_SUCCESS ){
		WARN("error getting image_size");
		return;
	}

	sprintf(msg_str,"image size:  %d x %d",w,h);
	prt_msg(msg_str);

	if( dc1394_format7_get_image_position(CAM_P,the_fmt7_mode, &h, &v) != DC1394_SUCCESS ){
		WARN("error getting image_position");
		return;
	}

	sprintf(msg_str,"image position:  %d, %d",h,v);
	prt_msg(msg_str);

	if( dc1394_format7_get_unit_position(CAM_P,the_fmt7_mode, &h, &v ) != DC1394_SUCCESS ){
		WARN("error getting unit_position");
		return;
	}

	sprintf(msg_str,"unit position:  %d, %d",h,v);
	prt_msg(msg_str);

	if( dc1394_format7_get_packet_parameters(CAM_P,the_fmt7_mode, &unit_bytes, &max_bytes) != DC1394_SUCCESS ){
		WARN("error getting packet parameters");
		return;
	}

	sprintf(msg_str,"unit bytes:  %d\nmax bytes %d",unit_bytes,max_bytes);
	prt_msg(msg_str);

	if( /*dc1394_format7_get_byte_per_packet*/
		dc1394_format7_get_packet_size(CAM_P,the_fmt7_mode, &packet_bytes) != DC1394_SUCCESS ){
		WARN("error getting bytes per packet");
		return;
	}

	sprintf(msg_str,"bytes per packet:  %d",packet_bytes);
	prt_msg(msg_str);

	if( /*dc1394_format7_get_recommended_byte_per_packet*/
		dc1394_format7_get_recommended_packet_size(CAM_P,the_fmt7_mode, &bpp) != DC1394_SUCCESS ){
		WARN("error getting recommended bytes per packet");
		return;
	}

	sprintf(msg_str,"recommended bpp:  %d",bpp);
	prt_msg(msg_str);

	if( dc1394_format7_get_packets_per_frame(CAM_P,the_fmt7_mode, &ppf) != DC1394_SUCCESS ){
		WARN("error getting packets per frame");
		return;
	}

	sprintf(msg_str,"packets per frame:  %d",ppf);
	prt_msg(msg_str);

#endif
}

static COMMAND_FUNC( do_fmt7_setsize )
{
	uint32_t w,h;

	w=HOW_MANY("width");
	h=HOW_MANY("height");

	CHECK_CAM

	/* Don't try to set the image size if capture is running... */

	if( the_cam_p->pc_flags & PGR_CAM_IS_RUNNING ){
		WARN("can't set image size while camera is running!?");
		return;
	}

#ifdef HAVE_LIBDC1394
	if( dc1394_format7_set_image_size(CAM_P,the_fmt7_mode, w, h ) != DC1394_SUCCESS ){
		WARN("error setting image size");
		return;
	}
#endif
}

static COMMAND_FUNC( do_fmt7_setposn )
{
	uint32_t h,v;

	h=HOW_MANY("horizontal position (left)");
	v=HOW_MANY("vertical position (top)");

	CHECK_CAM

#ifdef HAVE_LIBDC1394
	if( dc1394_format7_set_image_position( CAM_P,the_fmt7_mode, h, v ) != DC1394_SUCCESS ){
		WARN("error setting image position");
		return;
	}
#endif
}

static COMMAND_FUNC( do_fmt7_select )
{
#ifdef HAVE_LIBDC1394
	dc1394video_mode_t m;

	CHECK_CAM

	m = pick_fmt7_mode(QSP_ARG  /* CAM_P */ the_cam_p,"format7 mode");
	if( m == 0 ) return;

	the_fmt7_mode = m;
#endif
}

static Command fmt7_ctbl[]={
{ "mode",	do_fmt7_select,		"select format7 mode for get/set"	},
{ "list",	do_fmt7_list,		"list format7 settings"			},
{ "image_size",	do_fmt7_setsize,	"set image size"			},
{ "position",	do_fmt7_setposn,	"set image position"			},
{ "quit",	popcmd,			"exit submenu"				},
{ NULL_COMMAND									}
};

static COMMAND_FUNC( fmt7menu )
{
	PUSHCMD( fmt7_ctbl, "format7" );
}

static COMMAND_FUNC( do_bmode )
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM

	if( ASKIF("Use 1394-B mode") ){
		the_cam_p->pc_flags |= PGR_CAM_USES_BMODE;
		set_camera_bmode(the_cam_p,1);
	} else {
		the_cam_p->pc_flags &= ~PGR_CAM_USES_BMODE;
		set_camera_bmode(the_cam_p,0);
	}
#endif
}

static Command pgr_ctbl[]={
{ "init",	do_init,	"initialize subsystem"		},
{ "capture",	captmenu,	"capture submenu"		},
{ "format7",	fmt7menu,	"format7 submenu"		},
{ "list",	do_list,	"list camera"			},
{ "power",	do_power,	"power camera on/off"		},
{ "reset",	do_reset,	"reset camera"			},
/* { "frame",	do_frame,	"create a data object alias for a capture buffer frame"	}, */
{ "trigger",	do_trigger,	"trigger submenu"		},
{ "bandwidth",	do_bw,		"report bandwidth usage"	},
{ "bmode",	do_bmode,	"set/clear B-mode"		},
{ "close",	do_close,	"shutdown firewire subsystem"	},
{ "camera",	cam_menu,	"camera submenu"		},
{ "quit",	popcmd,		"exit submenu"			},
{ NULL_COMMAND							}
};

COMMAND_FUNC( pgr_menu )
{
	PUSHCMD( pgr_ctbl, "pgr" );
}

