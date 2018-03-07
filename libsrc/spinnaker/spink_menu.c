#include "quip_config.h"
#include "quip_prot.h"
#include "spink.h"
#include "data_obj.h"
#include "query_bits.h"	// LLEN - BUG

static Spink_Cam *the_cam_p=NULL;	// should this be per-thread?
				// no need yet...

// local prototypes
static COMMAND_FUNC( do_spink_cam_menu );


#define UNIMP_MSG(whence)						\
									\
	sprintf(ERROR_STRING,						\
		"%s:  function not implemented yet!?",whence);		\
	WARN(ERROR_STRING);

#define NO_LIB_MSG(whence)						\
									\
	sprintf(ERROR_STRING,						\
		"%s:  program built without libflycap support!?",whence);	\
	error1(ERROR_STRING);

#define EAT_ONE_DUMMY(whence)						\
									\
	const char *s;							\
	s=NAMEOF("dummy word");						\
	NO_LIB_MSG(whence)


#define CHECK_CAM	if( the_cam_p == NULL ){ \
		WARN("No spink_cam selected."); \
		return; }

static COMMAND_FUNC(do_list_spink_cam_trig)
{
#ifdef HAVE_LIBSPINNAKER
	CHECK_CAM
	list_spink_cam_trig(QSP_ARG  the_cam_p);
#else
	NO_LIB_MSG("do_list_spink_cam_trig");
#endif
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(trigger_menu,s,f,h)

MENU_BEGIN(trigger)
ADD_CMD( list,	do_list_spink_cam_trig,	report trigger info )
MENU_END(trigger)

static COMMAND_FUNC( do_trigger )
{
	CHECK_AND_PUSH_MENU(trigger);
}

static COMMAND_FUNC( do_init )
{
#ifdef HAVE_LIBSPINNAKER
	if( the_cam_p != NULL ){
		WARN("Firewire system already initialized!?");
		return;
	}

	if( init_spink_cam_system(SINGLE_QSP_ARG) < 0 )
		WARN("Error initializing firewire system.");
#endif
}

static COMMAND_FUNC( do_list_spink_cams )
{
	list_spink_cams(tell_msgfile());
}

static COMMAND_FUNC( do_cam_info )
{
	Fly_Cam *fcp;

	fcp = pick_spink_cam("camera");
	if( fcp == NULL ) return;

	if( fcp == the_cam_p ){
		sprintf(MSG_STR,"%s is selected as current camera.",fcp->fc_name);
		prt_msg(MSG_STR);
	}
#ifdef HAVE_LIBSPINNAKER
	print_spink_cam_info(QSP_ARG  fcp);
#else
	NO_LIB_MSG("do_list_spink_cam");
#endif
}

static void select_spink_cam(QSP_ARG_DECL  Fly_Cam *fcp )
{
	if( the_cam_p != NULL )
		pop_spink_cam_context(SINGLE_QSP_ARG);
	the_cam_p = fcp;
	push_spink_cam_context(QSP_ARG  fcp);
#ifdef HAVE_LIBSPINNAKER
	refresh_spink_cam_properties(QSP_ARG  fcp);
#endif // HAVE_LIBSPINNAKER
}

static COMMAND_FUNC( do_select_cam )
{
	Fly_Cam *fcp;

	fcp = pick_spink_cam("camera");
	if( fcp == NULL ) return;

	select_spink_cam(QSP_ARG  fcp);
}

static COMMAND_FUNC( do_start )
{
advise("do_start BEGIN");
	CHECK_CAM
advise("do_start back from CHECK_CAM, calling start_firewire_capture");

	start_firewire_capture(QSP_ARG  the_cam_p);
advise("do_start DONE");
}

static COMMAND_FUNC( do_grab )
{
	Data_Obj *dp;

	CHECK_CAM
	if( (dp=grab_spink_cam_frame(QSP_ARG  the_cam_p )) == NULL ){
		/* any error */
#ifdef FOOBAR
		cleanup_spink_cam(the_cam_p);	/* grab error */
		the_cam_p=NULL;
#endif // FOOBAR
		// We might fail because we need to release a frame...
		// Don't shut down in that case.
		WARN("do_grab:  failed.");
	} else {
		char num_str[32];

		sprintf(num_str,"%d",the_cam_p->fc_newest);
		assign_var("newest",num_str);
	}

}

static COMMAND_FUNC( do_grab_newest )
{
	UNIMP_MSG("do_grab_newest");
}

static COMMAND_FUNC( do_stop )
{
	CHECK_CAM
	stop_firewire_capture(QSP_ARG  the_cam_p );
}

static COMMAND_FUNC(do_power)
{
	CHECK_CAM
	WARN("do_power unimplemented");
}

static COMMAND_FUNC(do_reset)
{
#ifdef HAVE_LIBSPINNAKER
	CHECK_CAM
	reset_spink_cam(QSP_ARG  the_cam_p);
#endif
}

// conflict started here???
static COMMAND_FUNC( do_release )
{
#ifdef HAVE_LIBSPINNAKER
	CHECK_CAM
	release_oldest_frame(QSP_ARG  the_cam_p);
#endif
}

static COMMAND_FUNC( do_close )
{
	CHECK_CAM
#ifdef HAVE_LIBSPINNAKER
	cleanup_spink_cam(the_cam_p);
#endif
	the_cam_p=NULL;
}

static COMMAND_FUNC( do_bw )
{
#ifdef HAVE_LIBSPINNAKER
	CHECK_CAM

	report_spink_cam_bandwidth(QSP_ARG  the_cam_p);
#endif
}

static COMMAND_FUNC( do_list_spink_cam_modes )
{
#ifdef HAVE_LIBSPINNAKER
	CHECK_CAM
	prt_msg("\nAvailable video modes:");
	list_spink_cam_video_modes(QSP_ARG  the_cam_p);
#endif
}

static COMMAND_FUNC( do_show_spink_cam_video_mode )
{
#ifdef HAVE_LIBSPINNAKER
	CHECK_CAM
	show_spink_cam_video_mode(QSP_ARG  the_cam_p);
#endif
}


static COMMAND_FUNC( do_list_spink_cam_framerates )
{
#ifdef HAVE_LIBSPINNAKER
	CHECK_CAM

	prt_msg("\nAvailable framerates:");
	list_spink_cam_framerates(QSP_ARG  the_cam_p);
#endif
}

static COMMAND_FUNC( do_show_spink_cam_framerate )
{
#ifdef HAVE_LIBSPINNAKER
	CHECK_CAM
	show_spink_cam_framerate(QSP_ARG  the_cam_p);
#endif
}

static COMMAND_FUNC( do_set_video_mode )
{
#ifdef HAVE_LIBSPINNAKER
	int i;

	CHECK_CAM
	i = WHICH_ONE("video mode",the_cam_p->fc_n_video_modes,
					the_cam_p->fc_video_mode_names );
	if( i < 0 ) return;

sprintf(ERROR_STRING,"mode %s selected...",
name_of_indexed_video_mode( the_cam_p->fc_video_mode_indices[i] ) );
advise(ERROR_STRING);

	if( is_fmt7_mode(QSP_ARG  the_cam_p, i ) ){
		set_fmt7_mode(QSP_ARG  the_cam_p, the_cam_p->fc_fmt7_index );
	} else {
		set_std_mode( QSP_ARG  the_cam_p, i );
	}

#else // ! HAVE_LIBSPINNAKER
	EAT_ONE_DUMMY("do_set_video_mode");
	UNIMP_MSG("set_video_mode");
#endif // ! HAVE_LIBSPINNAKER
}

static COMMAND_FUNC( do_set_framerate )
{
	int i;

	i = pick_spink_cam_framerate(QSP_ARG  the_cam_p, "frame rate");
	if( i < 0 ) return;

	// CHECK_CAM - not needed: pick_spink_cam_framerate will handle this

}



#ifdef HAVE_LIBSPINNAKER
//#define N_SPEED_CHOICES	2
//static const char *speed_choices[N_SPEED_CHOICES]={"400","800"};
#endif /* HAVE_LIBSPINNAKER */

static COMMAND_FUNC( do_power_on )
{
	CHECK_CAM

	if( power_on_spink_cam(the_cam_p) < 0 )
		WARN("Error powering on camera.");
}

static COMMAND_FUNC( do_power_off )
{
	CHECK_CAM

	if( power_off_spink_cam(the_cam_p) < 0 )
		WARN("Error powering off camera.");
}

static COMMAND_FUNC( do_set_temp )
{
	int t;

	t = HOW_MANY("color temperature");	// BUG prompt should display valid range

	CHECK_CAM

	if( set_spink_cam_temperature(the_cam_p, t) < 0 )
		WARN("Error setting color temperature");
}

static COMMAND_FUNC( do_set_white_balance )
{
	int wb;

	wb = HOW_MANY("white balance");	// BUG prompt should display valid range

	CHECK_CAM

	if( set_spink_cam_white_balance(the_cam_p, wb) < 0 )
		WARN("Error setting white balance!?");
}

// WHAT IS WHITE "SHADING" ???

static COMMAND_FUNC( do_set_white_shading )
{
	int val;

	val = HOW_MANY("white shading");	// BUG prompt should display valid range

	CHECK_CAM

	if( set_spink_cam_white_shading(the_cam_p, val) < 0 )
		WARN("Error setting white shading!?");
}

static COMMAND_FUNC( do_get_cams )
{
	Data_Obj *dp;

	dp = pick_obj("string table");
	if( dp == NULL ) return;

	if( get_spink_cam_names( QSP_ARG  dp ) < 0 )
		WARN("Error getting camera names!?");
}

static COMMAND_FUNC( do_get_spink_cam_video_modes )
{
	Data_Obj *dp;
	int n;
	char s[8];

	dp = pick_obj("string table");
	if( dp == NULL ) return;

	CHECK_CAM

	n = get_spink_cam_video_mode_strings( QSP_ARG  dp, the_cam_p );
	sprintf(s,"%d",n);
	// BUG should make this a reserved var...
	assign_var("n_video_modes",s);
}

static COMMAND_FUNC( do_get_framerates )
{
	Data_Obj *dp;
	int n;
	char s[8];

	dp = pick_obj("string table");
	if( dp == NULL ) return;

	CHECK_CAM

	n = get_spink_cam_framerate_strings( QSP_ARG  dp, the_cam_p );
	sprintf(s,"%d",n);
	// BUG should make this a reserved var...
	assign_var("n_framerates",s);
}

static COMMAND_FUNC( do_read_reg )
{
	unsigned int addr;

	addr = HOW_MANY("register address");
	CHECK_CAM

#ifdef HAVE_LIBSPINNAKER
	{
	unsigned int val;
	val = read_register(QSP_ARG  the_cam_p, addr);
	sprintf(MSG_STR,"0x%x:  0x%x",addr,val);
	prt_msg(MSG_STR);
	}
#else // ! HAVE_LIBSPINNAKER
	UNIMP_MSG("read_register");
#endif // ! HAVE_LIBSPINNAKER
}

static COMMAND_FUNC( do_write_reg )
{
	unsigned int addr;
	unsigned int val;

	addr = HOW_MANY("register address");
	val = HOW_MANY("value");
	CHECK_CAM

#ifdef HAVE_LIBSPINNAKER
	write_register(QSP_ARG  the_cam_p, addr, val);
#else // ! HAVE_LIBSPINNAKER
	UNIMP_MSG("write_register");
#endif // ! HAVE_LIBSPINNAKER
}

static COMMAND_FUNC( do_prop_info )
{
	Fly_Cam_Property_Type *t;

	t = pick_pgr_prop("property type");
	CHECK_CAM

	if( t == NULL ) return;

#ifdef HAVE_LIBSPINNAKER
	refresh_property_info(QSP_ARG  the_cam_p, t );
	show_property_info(QSP_ARG  the_cam_p, t );
#else // ! HAVE_LIBSPINNAKER
	UNIMP_MSG("get_property_info");
#endif // ! HAVE_LIBSPINNAKER
}

static COMMAND_FUNC( do_show_prop )
{
	Fly_Cam_Property_Type *t;

	t = pick_pgr_prop("property type");
	CHECK_CAM

	if( t == NULL ) return;

#ifdef HAVE_LIBSPINNAKER
	refresh_property_value(QSP_ARG  the_cam_p, t );
	show_property_value(QSP_ARG  the_cam_p, t );
#else // ! HAVE_LIBSPINNAKER
	UNIMP_MSG("show_property");
#endif // ! HAVE_LIBSPINNAKER
}

static COMMAND_FUNC( do_set_auto )
{
	Fly_Cam_Property_Type *t;
	int yn;
	char pmpt[LLEN];

	t = pick_pgr_prop("property type");
	if( t != NULL )
		sprintf(pmpt,"Enable automatic setting of %s",t->name);
	else
		sprintf(pmpt,"Dummy boolean");

	yn = ASKIF(pmpt);

	CHECK_CAM

	if( t == NULL ) return;

#ifdef HAVE_LIBSPINNAKER
	set_prop_auto(QSP_ARG   the_cam_p, t, yn );
#else // ! HAVE_LIBSPINNAKER
	UNIMP_MSG("set_prop_auto");
#endif // ! HAVE_LIBSPINNAKER
}

static int use_absolute=1;		// BUG not thread-safe

static COMMAND_FUNC( do_set_absolute )
{
	use_absolute = ASKIF("use physical units to specify property values");
}

//static const char *spec_types[2]={"integer","absolute"};

static COMMAND_FUNC( do_set_prop )
{
	Fly_Cam_Property_Type *t;
	Fly_Cam_Prop_Val pv;

	t = pick_pgr_prop("property type");

	pv.pv_is_abs = use_absolute;
	if( use_absolute ){
		char pmpt[LLEN];

#ifdef HAVE_LIBSPINNAKER
		if( t != NULL ){
			sprintf(pmpt,"%s in %ss",t->name,t->info.pUnits);
		} else {
			sprintf(pmpt,"value (integer)");
		}
#else // ! HAVE_LIBSPINNAKER
		sprintf(pmpt,"value (integer)");
#endif // ! HAVE_LIBSPINNAKER
		pv.pv_u.u_f = HOW_MUCH(pmpt);
	} else {
		pv.pv_u.u_i = HOW_MANY("value (integer)");
	}
	CHECK_CAM
	if( t == NULL ) return;

#ifdef HAVE_LIBSPINNAKER
	set_prop_value(QSP_ARG  the_cam_p, t, &pv );
#else // ! HAVE_LIBSPINNAKER
	UNIMP_MSG("set_prop_value");
#endif // ! HAVE_LIBSPINNAKER
}

static COMMAND_FUNC( do_set_fmt7 )
{
	int i;

	i = HOW_MANY("index of format7 mode");
	CHECK_CAM

#ifdef HAVE_LIBSPINNAKER
	if( i < 0 || i >= the_cam_p->fc_n_fmt7_modes ){
		sprintf(ERROR_STRING,
			"%s:  format7 index must be in the range 0 - %d",
			the_cam_p->fc_name,the_cam_p->fc_n_fmt7_modes-1);
		WARN(ERROR_STRING);
		return;
	}

	set_fmt7_mode(QSP_ARG  the_cam_p, i );
#else // ! HAVE_LIBSPINNAKER
	UNIMP_MSG("set_fmt7_mode");
#endif // ! HAVE_LIBSPINNAKER
}

static COMMAND_FUNC( do_show_n_bufs )
{
	CHECK_CAM

	show_n_buffers(QSP_ARG  the_cam_p);
}

static COMMAND_FUNC( do_set_n_bufs )
{
	int n;

	n=HOW_MANY("number of buffers");

	CHECK_CAM

	if( n < MIN_N_BUFFERS ){
		sprintf(ERROR_STRING,"do_set_n_bufs:  n (%d) must be >= %d",n,MIN_N_BUFFERS);
		WARN(ERROR_STRING);
	} else if ( n > MAX_N_BUFFERS ){
		sprintf(ERROR_STRING,"do_set_n_bufs:  n (%d) must be <= %d",n,MAX_N_BUFFERS);
		WARN(ERROR_STRING);
	} else {
#ifdef HAVE_LIBSPINNAKER
		set_n_buffers(QSP_ARG  the_cam_p, n);
#endif // HAVE_LIBSPINNAKER
	}
}

static COMMAND_FUNC( do_set_eii )
{
	int i;
	int yesno;
	char prompt[LLEN];

	i=WHICH_ONE("embedded image information property",N_EII_PROPERTIES,eii_prop_names);
	if( i < 0 ) strcpy(prompt,"Enter yes or no");
	else sprintf(prompt,"Enable %s",eii_prop_names[i]);
	yesno = ASKIF(prompt);

	if( i < 0 ) return;

	CHECK_CAM

#ifdef HAVE_LIBSPINNAKER
	set_eii_property(QSP_ARG  the_cam_p,i,yesno);
#endif // HAVE_LIBSPINNAKER
}

static COMMAND_FUNC( do_list_spink_cam_props )
{
	CHECK_CAM
#ifdef HAVE_LIBSPINNAKER
	list_spink_cam_properties(QSP_ARG  the_cam_p);
#else // ! HAVE_LIBSPINNAKER
	WARN("No support for libflycap in this build.");
#endif // ! HAVE_LIBSPINNAKER
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(properties_menu,s,f,h)

MENU_BEGIN(properties)
ADD_CMD( list,			do_list_spink_cam_props,		list all properties )
ADD_CMD( info,			do_prop_info,		display property info )
ADD_CMD( show,			do_show_prop,		display property value )
ADD_CMD( set,			do_set_prop,		set property value )
ADD_CMD( auto,			do_set_auto,		enable/disable automatic mode)
ADD_CMD( absolute,		do_set_absolute,	enable/disable physical units for property values )
MENU_END(properties)

static COMMAND_FUNC( do_prop_menu )
{
	CHECK_AND_PUSH_MENU(properties);
}

static COMMAND_FUNC( do_set_iso_speed )
{
	EAT_ONE_DUMMY("speed");
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(spink_cam_menu,s,f,h)

MENU_BEGIN(spink_cam)
ADD_CMD( set_n_buffers,		do_set_n_bufs,		specify number of frames in the ring buffer )
ADD_CMD( show_n_buffers,		do_show_n_bufs,		show number of frames in the ring buffer )
ADD_CMD( set_embedded_image_info,	do_set_eii,	enable/disable embedded image information )
ADD_CMD( read_register,		do_read_reg,		read a camera register )
ADD_CMD( write_register,	do_write_reg,		write a camera register )
ADD_CMD( properties,		do_prop_menu,		camera properties submenu )
ADD_CMD( list_video_modes,	do_list_spink_cam_modes,		list all video modes for this camera )
ADD_CMD( get_video_modes,	do_get_spink_cam_video_modes,	copy video modes strings to an array )
ADD_CMD( set_video_mode,	do_set_video_mode,	set video mode )
ADD_CMD( format7,		do_set_fmt7,		select a format7 mode )
ADD_CMD( show_video_mode,	do_show_spink_cam_video_mode,	display current video mode )
ADD_CMD( list_framerates,	do_list_spink_cam_framerates,	list all framerates for this camera )
ADD_CMD( get_framerates,	do_get_framerates,	copy framerate strings to an array )
ADD_CMD( set_framerate,		do_set_framerate,	set framerate )
ADD_CMD( show_framerate,	do_show_spink_cam_framerate,	show current framerate )
ADD_CMD( set_iso_speed,		do_set_iso_speed,	set ISO speed )
ADD_CMD( power_on,		do_power_on,		power on current camera )
ADD_CMD( power_off,		do_power_off,		power off current camera )
ADD_CMD( temperature,		do_set_temp,		set color temperature )
ADD_CMD( white_balance,		do_set_white_balance,	set white balance )
ADD_CMD( white_shading,		do_set_white_shading,	set white shading )
MENU_END(spink_cam)

static COMMAND_FUNC( do_spink_cam_menu )
{
	CHECK_AND_PUSH_MENU(spink_cam);
}

static COMMAND_FUNC( do_record )
{
	const char *s;
	int n;
	Image_File *ifp;

	s=NAMEOF("name for raw volume recording");
	n=HOW_MANY("number of frames");

	ifp = get_file_for_recording(s,n,the_cam_p);
	if( ifp == NULL ) return;
	
	CHECK_CAM

#ifdef HAVE_LIBSPINNAKER
	stream_record(QSP_ARG  ifp, n, the_cam_p );
#else // ! HAVE_LIBSPINNAKER
	UNIMP_MSG("stream_record");
#endif // ! HAVE_LIBSPINNAKER
}

static COMMAND_FUNC( do_set_bufs )
{
	Data_Obj *dp;

	dp = pick_obj("sequence object to use for capture");
	if( dp == NULL ) return;

	CHECK_CAM
#ifdef HAVE_LIBSPINNAKER
	// make sure the object dimensions match the camera!
	set_buffer_obj(QSP_ARG  the_cam_p, dp);
#endif // HAVE_LIBSPINNAKER
}

static COMMAND_FUNC( do_set_grab_mode )
{
	int idx;

	CHECK_CAM

	idx = pick_grab_mode(QSP_ARG  the_cam_p, "capture mode");
	if( idx < 0 ) return;

#ifdef HAVE_LIBSPINNAKER
	set_grab_mode(QSP_ARG  the_cam_p, idx );
#endif // HAVE_LIBSPINNAKER
}

static COMMAND_FUNC( do_show_grab_mode )
{
	CHECK_CAM

#ifdef HAVE_LIBSPINNAKER
	show_grab_mode(QSP_ARG  the_cam_p);
#endif // HAVE_LIBSPINNAKER
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(capture_menu,s,f,h)

MENU_BEGIN(capture)
ADD_CMD( set_buffer_obj,	do_set_bufs,	specify sequence object to use for capture )
ADD_CMD( set_mode,		do_set_grab_mode,	specify grab mode )
ADD_CMD( show_mode,		do_show_grab_mode,	display current grab mode )
ADD_CMD( start,			do_start,	start capture )
ADD_CMD( grab,			do_grab,	grab a frame )
ADD_CMD( grab_newest,		do_grab_newest,	grab the newest frame )
ADD_CMD( release,		do_release,	release a frame )
ADD_CMD( record,		do_record,	record frames to disk )
ADD_CMD( stop,			do_stop,	stop capture )
MENU_END(capture)

static COMMAND_FUNC( captmenu )
{
	CHECK_AND_PUSH_MENU( capture );
}

#define CAM_P	the_cam_p->fc_cam_p

static COMMAND_FUNC( do_fmt7_list )
{
	UNIMP_MSG("fmt7_list");
}

static COMMAND_FUNC( do_fmt7_setsize )
{
	uint32_t w,h;

	w=HOW_MANY("width");
	h=HOW_MANY("height");

	CHECK_CAM

	/* Don't try to set the image size if capture is running... */

	if( the_cam_p->fc_flags & FLY_CAM_IS_RUNNING ){
		WARN("can't set image size while camera is running!?");
		return;
	}
#ifdef HAVE_LIBSPINNAKER
	set_fmt7_size(QSP_ARG  the_cam_p, w, h );
#else // ! HAVE_LIBSPINNAKER
	UNIMP_MSG("set_fmt7_size");
#endif // ! HAVE_LIBSPINNAKER
}

static COMMAND_FUNC( do_fmt7_setposn )
{
	uint32_t h,v;

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

	UNIMP_MSG("fmt7_posn");
}

static COMMAND_FUNC( do_fmt7_select )
{
	UNIMP_MSG("fmt7_select");
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
	CHECK_AND_PUSH_MENU( format7 );
}

static COMMAND_FUNC(do_quit_fly)
{
	if( the_cam_p != NULL )
		pop_spink_cam_context(SINGLE_QSP_ARG);

	do_pop_menu(SINGLE_QSP_ARG);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(fly_menu,s,f,h)

MENU_BEGIN(fly)
ADD_CMD( init,		do_init,	initialize subsystem )
ADD_CMD( list,		do_list_spink_cams,	list cameras )
ADD_CMD( select,	do_select_cam,	select camera )
ADD_CMD( get_cameras,	do_get_cams,	copy camera names to an array )
ADD_CMD( capture,	captmenu,	capture submenu )
ADD_CMD( format7,	fmt7menu,	format7 submenu )
ADD_CMD( select,	do_select_cam,	select camera )
ADD_CMD( info,		do_cam_info,	print camera info )
ADD_CMD( power,		do_power,	power camera on/off )
ADD_CMD( reset,		do_reset,	reset camera )
/* ADD_CMD( frame,	do_frame,	create a data object alias for a capture buffer frame ) */
ADD_CMD( trigger,	do_trigger,	trigger submenu )
ADD_CMD( bandwidth,	do_bw,		report bandwidth usage )
ADD_CMD( close,		do_close,	shutdown firewire subsystem )
ADD_CMD( camera,	do_spink_cam_menu,	camera submenu )
ADD_CMD( quit,		do_quit_fly,	exit submenu )
MENU_SIMPLE_END(fly)	// doesn't add quit command automatically

COMMAND_FUNC( do_fly_menu )
{
	if( the_cam_p != NULL )
		push_spink_cam_context(QSP_ARG  the_cam_p);
	CHECK_AND_PUSH_MENU( fly );
}

