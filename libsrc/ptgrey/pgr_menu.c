#include "quip_config.h"

#include "quip_prot.h"
#include "pgr.h"
#include "data_obj.h"
#include "query_bits.h"	// LLEN - BUG

static PGR_Cam *the_cam_p=NULL;

// local prototypes
static COMMAND_FUNC( do_cam_menu );

#ifdef HAVE_LIBDC1394

// BUG - this should be part of the camera struct!?
static dc1394video_mode_t the_fmt7_mode=DC1394_VIDEO_MODE_FORMAT7_0;

static void camera_feature_set_auto(PGR_Cam *pgcp, dc1394feature_info_t *f, int yn);
static void camera_feature_set_onoff(PGR_Cam *pgcp, dc1394feature_info_t *f, int yn);
static int camera_feature_set_value(PGR_Cam *pgcp, dc1394feature_info_t *f, int val);
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
		WARN("No camera selected."); \
		return; }

static COMMAND_FUNC(do_list_trig)
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM
	list_trig(QSP_ARG  the_cam_p);
#else
	NO_LIB_MSG("do_list_trig");
#endif
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(trigger_menu,s,f,h)

MENU_BEGIN(trigger)
ADD_CMD( list,	do_list_trig,	report trigger info )
MENU_END(trigger)

static COMMAND_FUNC( do_trigger )
{
	PUSH_MENU(trigger);
}

static COMMAND_FUNC( do_init )
{
#ifdef HAVE_LIBDC1394
	if( the_cam_p != NULL ){
		WARN("Firewire system already initialized!?");
		return;
	}

	if( init_firewire_system(SINGLE_QSP_ARG) < 0 )
		WARN("Error initializing firewire system.");
#endif
}

static COMMAND_FUNC( do_list_cams )
{
	list_pgcs(QSP_ARG  tell_msgfile(SINGLE_QSP_ARG));
}

static COMMAND_FUNC( do_cam_info )
{
	PGR_Cam *pgcp;

	pgcp = pick_pgc(QSP_ARG  "camera");
	if( pgcp == NULL ) return;

#ifdef HAVE_LIBDC1394
	/* print out info about this camera */
	int bpp=1; /* BUG determine this from video mode */

	CHECK_CAM
	sprintf(msg_str,"%s %s:  %dx%d, %d %s per pixel",
		pgcp->pc_cam_p->vendor,
		pgcp->pc_cam_p->model,
		pgcp->pc_nRows,pgcp->pc_nCols, bpp,
		bpp>1?"bytes":"byte");
	prt_msg(msg_str);

	print_camera_info(QSP_ARG  pgcp);
#else
	NO_LIB_MSG("do_list_cam");
#endif
}

static void select_camera(QSP_ARG_DECL  PGR_Cam *pgcp )
{
	if( the_cam_p != NULL )
		pop_camera_context(SINGLE_QSP_ARG);
	the_cam_p = pgcp;
	push_camera_context(QSP_ARG  pgcp);
}

static COMMAND_FUNC( do_select_cam )
{
	PGR_Cam *pgcp;

	pgcp = pick_pgc(QSP_ARG  "camera");
	if( pgcp == NULL ) return;

	select_camera(QSP_ARG  pgcp);
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
advise("failure, cleaning up...");
		cleanup_cam(the_cam_p);
		the_cam_p=NULL;
		return;
	}

advise("camera is running.");
	the_cam_p->pc_flags |= PGR_CAM_IS_RUNNING;
#else
	NO_LIB_MSG("do_start");
#endif
}

static COMMAND_FUNC( do_grab )
{
#ifdef HAVE_LIBDC1394
	Data_Obj *dp;

	CHECK_CAM
	if( (dp=grab_firewire_frame(QSP_ARG  the_cam_p )) == NULL ){		/* any error */
#ifdef FOOBAR
		cleanup_cam(the_cam_p);	/* grab error */
		the_cam_p=NULL;
#endif // FOOBAR
		// We might fail because we need to release a frame...
		// Don't shut down in that case.
		WARN("do_grab:  failed.");
	}
#else
	NO_LIB_MSG("do_grab");
#endif
}

static COMMAND_FUNC( do_grab_newest )
{
#ifdef HAVE_LIBDC1394
	Data_Obj *dp;

	CHECK_CAM

	// Only try to get the frame if the camera is running...
	if( (the_cam_p->pc_flags & PGR_CAM_IS_RUNNING) == 0 ){
		WARN("do_grab_newest:  camera is not running!?");
		return;
	}

	if( (dp=grab_newest_firewire_frame(QSP_ARG  the_cam_p )) == NULL ){		/* any error */
#ifdef FOOBAR
		cleanup_cam(the_cam_p);	/* grab error */
		the_cam_p=NULL;
#endif // FOOBAR
		if( the_cam_p->pc_policy != DC1394_CAPTURE_POLICY_POLL )
			WARN("do_grab_newest:  failure.");
		// If we are polling, this means there is no newest frame
		// We should set a script variable to indicate this...
		ASSIGN_VAR("newest","-1");
		return;
	}
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
	cleanup_cam(the_cam_p);
#endif
	the_cam_p=NULL;
}

static COMMAND_FUNC( do_bw )
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM

	report_bandwidth(QSP_ARG  the_cam_p);
#endif
}

static COMMAND_FUNC( do_list_modes )
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM
	prt_msg("\nAvailable video modes:");
	list_video_modes(QSP_ARG  the_cam_p);
#endif
}

static COMMAND_FUNC( do_show_video_mode )
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM
	show_video_mode(QSP_ARG  the_cam_p);
#endif
}


static COMMAND_FUNC( do_list_framerates )
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM

	prt_msg("\nAvailable framerates:");
	list_framerates(QSP_ARG  the_cam_p);
#endif
}

static COMMAND_FUNC( do_show_framerate )
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM
	show_framerate(QSP_ARG  the_cam_p);
#endif
}

static COMMAND_FUNC( do_set_video_mode )
{
#ifdef HAVE_LIBDC1394
	dc1394video_mode_t m;

	m=pick_video_mode(QSP_ARG  the_cam_p,"video mode");
	if( m == BAD_VIDEO_MODE )
		set_video_mode(QSP_ARG  the_cam_p,m);
#else /* ! HAVE_LIBDC1394 */
	EAT_ONE_DUMMY("do_set_video_mode");
#endif /* ! HAVE_LIBDC1394 */
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

	//if( curr_feat_p->auto_capable ){
	if( is_auto_capable( curr_feat_p ) ){
		ASSIGN_VAR("f_auto_capable","1");
		//sprintf(s,"%d",curr_feat_p->auto_active);
		//ASSIGN_VAR("f_auto",s);
	} else {
		ASSIGN_VAR("f_auto_capable","0");
		ASSIGN_VAR("f_auto","0");
	}

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
static void camera_feature_set_auto(PGR_Cam *pgcp, dc1394feature_info_t *f, int yn)
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

static void camera_feature_set_onoff(PGR_Cam *pgcp, dc1394feature_info_t *f, int yn)
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

static int camera_feature_set_value(PGR_Cam *pgcp, dc1394feature_info_t *f, int val)
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
	POP_MENU;
	/* refresh all the features so we can display the current values */
	get_camera_features(the_cam_p);

#ifdef HAVE_LIBDC1394
	report_feature_info(QSP_ARG  the_cam_p,curr_feat_p->id);
#endif
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)		ADD_COMMAND(feature_change_menu,s,f,h)

MENU_BEGIN(feature_change)
ADD_CMD( fetch,	do_get_frange,	get current value in $fval with range tlimits in $fmin $fmax )
ADD_CMD( value,	do_set_fval,	set value )
ADD_CMD( auto,	do_set_auto,	set/clear auto flag )
ADD_CMD( on,	do_set_onoff,	set/clear on_off flag )
// BUG? will we get two quit commands?
ADD_CMD( quit,	change_done,	exit submenu )
MENU_END(feature_change)


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

	PUSH_MENU(feature_change);
}

static COMMAND_FUNC( do_list_cam_features )
{
#ifdef HAVE_LIBDC1394
	CHECK_CAM
	list_camera_features(QSP_ARG  the_cam_p);
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
	report_feature_info(QSP_ARG  the_cam_p,f->id);
#else
	EAT_ONE_DUMMY("do_feature_info");
#endif
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(features_menu,s,f,h)

MENU_BEGIN(features)
ADD_CMD( list,		do_list_cam_features,	list available features for this camera )
ADD_CMD( info,		do_feature_info,	report feature info )
ADD_CMD( change,	do_feature_change,	change feature settings )
MENU_END(features)

static COMMAND_FUNC( do_feature_menu )
{
	PUSH_MENU(features);
}

#ifdef HAVE_LIBDC1394
#define N_SPEED_CHOICES	2
static const char *speed_choices[N_SPEED_CHOICES]={"400","800"};
#endif /* HAVE_LIBDC1394 */

static COMMAND_FUNC( do_set_iso_speed )
{
#ifdef HAVE_LIBDC1394
	int i;

	CHECK_CAM

	i=WHICH_ONE("speed",N_SPEED_CHOICES,speed_choices);
	if( i < 0 ) return;

	switch(i){
		case 0:
			if( set_iso_speed(the_cam_p, DC1394_ISO_SPEED_400) < 0 )
				WARN("Error setting iso speed.");
			break;
		case 1:
			if( set_iso_speed(the_cam_p, DC1394_ISO_SPEED_800) < 0 )
				WARN("Error setting iso speed.");
			break;
#ifdef CAUTIOUS
		default:
			ERROR1("CAUTIOUS:  do_set_iso_speed:  wacky speed choice!?");
			break;
#endif /* CAUTIOUS */
	}
#else /* ! HAVE_LIBDC1394 */
	EAT_ONE_DUMMY("speed");
#endif /* ! HAVE_LIBDC1394 */
}

static COMMAND_FUNC( do_power_on )
{
	CHECK_CAM

	if( power_on_camera(the_cam_p) < 0 )
		WARN("Error powering on camera.");
}

static COMMAND_FUNC( do_power_off )
{
	CHECK_CAM

	if( power_off_camera(the_cam_p) < 0 )
		WARN("Error powering off camera.");
}

static COMMAND_FUNC( do_set_temp )
{
	int t;

	t = HOW_MANY("color temperature");	// BUG prompt should display valid range

	CHECK_CAM

	if( set_camera_temperature(the_cam_p, t) < 0 )
		WARN("Error setting color temperature");
}

static COMMAND_FUNC( do_set_white_balance )
{
	int wb;

	wb = HOW_MANY("white balance");	// BUG prompt should display valid range

	CHECK_CAM

	if( set_camera_white_balance(the_cam_p, wb) < 0 )
		WARN("Error setting white balance!?");
}

// WHAT IS WHITE "SHADING" ???

static COMMAND_FUNC( do_set_white_shading )
{
	int val;

	val = HOW_MANY("white shading");	// BUG prompt should display valid range

	CHECK_CAM

	if( set_camera_white_shading(the_cam_p, val) < 0 )
		WARN("Error setting white shading!?");
}

static COMMAND_FUNC( do_get_cams )
{
	Data_Obj *dp;
	int n;

	dp = PICK_OBJ("string table");
	if( dp == NO_OBJ ) return;

	n = get_camera_names( QSP_ARG  dp );
}

static COMMAND_FUNC( do_get_video_modes )
{
	Data_Obj *dp;
	int n;
	char s[8];

	dp = PICK_OBJ("string table");
	if( dp == NO_OBJ ) return;

	CHECK_CAM

	n = get_video_mode_strings( QSP_ARG  dp, the_cam_p );
	sprintf(s,"%d",n);
	// BUG should make this a reserved var...
	ASSIGN_VAR("n_video_modes",s);
}

static COMMAND_FUNC( do_get_framerates )
{
	Data_Obj *dp;
	int n;
	char s[8];

	dp = PICK_OBJ("string table");
	if( dp == NO_OBJ ) return;

	CHECK_CAM

	n = get_framerate_strings( QSP_ARG  dp, the_cam_p );
	sprintf(s,"%d",n);
	// BUG should make this a reserved var...
	ASSIGN_VAR("n_framerates",s);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(camera_menu,s,f,h)

MENU_BEGIN(camera)
ADD_CMD( list_video_modes,	do_list_modes,		list all video modes for this camera )
ADD_CMD( get_video_modes,	do_get_video_modes,	copy video modes strings to an array )
ADD_CMD( set_video_mode,	do_set_video_mode,	set video mode )
ADD_CMD( show_video_mode,	do_show_video_mode,	display current video mode )
ADD_CMD( list_framerates,	do_list_framerates,	list all framerates for this camera )
ADD_CMD( get_framerates,	do_get_framerates,	copy framerate strings to an array )
ADD_CMD( set_framerate,		do_set_framerate,	set framerate )
ADD_CMD( show_framerate,	do_show_framerate,	show current framerate )
ADD_CMD( set_iso_speed,		do_set_iso_speed,	set ISO speed )
ADD_CMD( power_on,		do_power_on,		power on current camera )
ADD_CMD( power_off,		do_power_off,		power off current camera )
ADD_CMD( temperature,		do_set_temp,		set color temperature )
ADD_CMD( white_balance,		do_set_white_balance,	set white balance )
ADD_CMD( white_shading,		do_set_white_shading,	set white shading )
ADD_CMD( features,		do_feature_menu,	camera feature submenu )
MENU_END(camera)

static COMMAND_FUNC( do_cam_menu )
{
	PUSH_MENU(camera);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(capture_menu,s,f,h)

MENU_BEGIN(capture)
ADD_CMD( start,		do_start,	start capture )
ADD_CMD( grab,		do_grab,	grab a frame )
ADD_CMD( grab_newest,	do_grab_newest,	grab the newest frame )
ADD_CMD( release,	do_release,	release a frame )
ADD_CMD( stop,		do_stop,	stop capture )
MENU_END(capture)

static COMMAND_FUNC( captmenu )
{
	PUSH_MENU( capture );
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
#ifdef HAVE_LIBDC1394
	dc1394error_t err;
#endif // HAVE_LIBDC1394

	h=HOW_MANY("horizontal position (left)");
	v=HOW_MANY("vertical position (top)");

	CHECK_CAM

	// What are the constraints as to what this can be???
	// At least on the flea, the position has to be even...

	if( h & 1 ){
		sprintf(ERROR_STRING,"Horizontal position (%d) should be even, rounding down to %d.",h,h&(~1));
		advise(ERROR_STRING);
		h &= ~1;
	}

	if( v & 1 ){
		sprintf(ERROR_STRING,"Vertical position (%d) should be even, rounding down to %d.",v,v&(~1));
		advise(ERROR_STRING);
		v &= ~1;
	}

#ifdef HAVE_LIBDC1394
	if( (err=dc1394_format7_set_image_position( CAM_P,the_fmt7_mode, h, v )) != DC1394_SUCCESS ){
		WARN("error setting image position");
		describe_dc1394_error(QSP_ARG  err);
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

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(format7_menu,s,f,h)

MENU_BEGIN(format7)
ADD_CMD( mode,		do_fmt7_select,		select format7 mode for get/set )
ADD_CMD( list,		do_fmt7_list,		list format7 settings )
ADD_CMD( set_image_size, do_fmt7_setsize,	set image size )
ADD_CMD( position,	do_fmt7_setposn,	set image position )
MENU_END(format7)

static COMMAND_FUNC( fmt7menu )
{
	PUSH_MENU( format7 );
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

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(pgr_menu,s,f,h)

MENU_BEGIN(pgr)
ADD_CMD( init,		do_init,	initialize subsystem )
ADD_CMD( list,		do_list_cams,	list cameras )
ADD_CMD( select,	do_select_cam,	select camera )
ADD_CMD( get_cameras,	do_get_cams,	copy camera names to an array )
ADD_CMD( capture,	captmenu,	capture submenu )
ADD_CMD( format7,	fmt7menu,	format7 submenu )
ADD_CMD( select,	do_select_cam,	select camera )
ADD_CMD( info,		do_cam_info,	print info about current camera )
ADD_CMD( power,		do_power,	power camera on/off )
ADD_CMD( reset,		do_reset,	reset camera )
/* ADD_CMD( frame,	do_frame,	create a data object alias for a capture buffer frame ) */
ADD_CMD( trigger,	do_trigger,	trigger submenu )
ADD_CMD( bandwidth,	do_bw,		report bandwidth usage )
ADD_CMD( bmode,		do_bmode,	set/clear B-mode )
ADD_CMD( close,		do_close,	shutdown firewire subsystem )
ADD_CMD( camera,	do_cam_menu,	camera submenu )
MENU_END(pgr)

COMMAND_FUNC( do_pgr_menu )
{
	PUSH_MENU( pgr );
}

