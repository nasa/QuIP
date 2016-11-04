/* Jeff's interface to the 1394 subsystem usign PGR's libflycap_c */

#include "quip_config.h"

#include "quip_prot.h"
#include "function.h"
#include "data_obj.h"

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* usleep */
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#include "fly.h"

#define TMPSIZE	32	// for temporary object names, e.g. _frame55

ITEM_INTERFACE_DECLARATIONS(PGR_Cam,pgc,0)

#define UNIMP_FUNC(name)						\
	sprintf(ERROR_STRING,"Function %s is not implemented!?",name);	\
	WARN(ERROR_STRING);

#ifdef HAVE_LIBFLYCAP
#define FC2_ENTRY(string,code)				\
{	#string,	code	}
#else // ! HAVE_LIBFLYCAP
#define FC2_ENTRY(string,code)				\
{	#string			}
#endif // ! HAVE_LIBFLYCAP

#ifdef NOT_USED
static Named_Pixel_Format all_pixel_formats[]={
{ "mono8",	FC2_PIXEL_FORMAT_MONO8			},
{ "411yuv8",	FC2_PIXEL_FORMAT_411YUV8		},
{ "422yuv8",	FC2_PIXEL_FORMAT_422YUV8		},
{ "444yuv8",	FC2_PIXEL_FORMAT_444YUV8		},
{ "rgb8",	FC2_PIXEL_FORMAT_RGB8			},
{ "mono16",	FC2_PIXEL_FORMAT_MONO16			},
{ "rgb16",	FC2_PIXEL_FORMAT_RGB16			},
{ "s_mono16",	FC2_PIXEL_FORMAT_S_MONO16		},
{ "s_rgb16",	FC2_PIXEL_FORMAT_S_RGB16		},
{ "raw8",	FC2_PIXEL_FORMAT_RAW8			},
{ "raw16",	FC2_PIXEL_FORMAT_RAW16			},
{ "mono12",	FC2_PIXEL_FORMAT_MONO12			},
{ "raw12",	FC2_PIXEL_FORMAT_RAW12			},
{ "bgr",	FC2_PIXEL_FORMAT_BGR			},
{ "bgru",	FC2_PIXEL_FORMAT_BGRU			},
{ "rgb",	FC2_PIXEL_FORMAT_RGB			},
{ "rgbu",	FC2_PIXEL_FORMAT_RGBU			},
{ "bgr16",	FC2_PIXEL_FORMAT_BGR16			},
{ "bgru16",	FC2_PIXEL_FORMAT_BGRU16			},
{ "jpeg",	FC2_PIXEL_FORMAT_422YUV8_JPEG		},
};

#define N_NAMED_PIXEL_FORMATS	(sizeof(all_pixel_formats)/sizeof(Named_Pixel_Format))
#endif // NOT_USED

#ifdef HAVE_LIBFLYCAP
#define FC2_MODE(string,code,w,h,d)	{#string,code,w,h,d}
#else // ! HAVE_LIBFLYCAP
#define FC2_MODE(string,code,w,h,d)	{#string,w,h,d}
#endif // ! HAVE_LIBFLYCAP

static Named_Video_Mode all_video_modes[]={
FC2_MODE( format7,		FC2_VIDEOMODE_FORMAT7,		0, 0,	0 ),

FC2_MODE( yuv444_160x120,	FC2_VIDEOMODE_160x120YUV444,	160, 120, 3 ),

FC2_MODE( yuv422_320x240,	FC2_VIDEOMODE_320x240YUV422,	320, 240, 2 ),

FC2_MODE( yuv411_640x480,	FC2_VIDEOMODE_640x480YUV411,	640, 480, 0 /*1.5*/),
FC2_MODE( yuv422_640x480,	FC2_VIDEOMODE_640x480YUV422,	640, 480, 2 ),
FC2_MODE( rgb8_640x480,		FC2_VIDEOMODE_640x480RGB,	640, 480, 3 ),
FC2_MODE( mono16_640x480,	FC2_VIDEOMODE_640x480Y16,	640, 480, 2 ),
FC2_MODE( mono8_640x480,	FC2_VIDEOMODE_640x480Y8,	640, 480, 1 ),

FC2_MODE( yuv422_800x600,	FC2_VIDEOMODE_800x600YUV422,	800, 600, 2 ),
FC2_MODE( rgb8_800x600,		FC2_VIDEOMODE_800x600RGB,	800, 600, 3 ),
FC2_MODE( mono16_800x600,	FC2_VIDEOMODE_800x600Y16,	800, 600, 2 ),
FC2_MODE( mono8_800x600,	FC2_VIDEOMODE_800x600Y8,	800, 600, 1 ),

FC2_MODE( yuv422_1024x768,	FC2_VIDEOMODE_1024x768YUV422,	1024, 768, 2 ),
FC2_MODE( rgb8_1024x768,	FC2_VIDEOMODE_1024x768RGB,	1024, 768, 3 ),
FC2_MODE( mono16_1024x768,	FC2_VIDEOMODE_1024x768Y16,	1024, 768, 2 ),
FC2_MODE( mono8_1024x768,	FC2_VIDEOMODE_1024x768Y8,	1024, 768, 1 ),

FC2_MODE( yuv422_1280x960,	FC2_VIDEOMODE_1280x960YUV422,	1280, 960, 2 ),
FC2_MODE( rgb8_1280x960,	FC2_VIDEOMODE_1280x960RGB,	1280, 960, 3 ),
FC2_MODE( mono16_1280x960,	FC2_VIDEOMODE_1280x960Y16,	1280, 960, 2 ),
FC2_MODE( mono8_1280x960,	FC2_VIDEOMODE_1280x960Y8,	1280, 960, 1 ),

FC2_MODE( yuv422_1600x1200,	FC2_VIDEOMODE_1600x1200YUV422,	1600, 1200, 2 ),
FC2_MODE( rgb8_1600x1200,	FC2_VIDEOMODE_1600x1200RGB,	1600, 1200, 3 ),
FC2_MODE( mono16_1600x1200,	FC2_VIDEOMODE_1600x1200Y16,	1600, 1200, 2 ),
FC2_MODE( mono8_1600x1200,	FC2_VIDEOMODE_1600x1200Y8,	1600, 1200, 1 )

};

#define N_NAMED_VIDEO_MODES	(sizeof(all_video_modes)/sizeof(Named_Video_Mode))
const char *name_of_indexed_video_mode(int idx)
{
	if( idx < 0 || idx >= N_NAMED_VIDEO_MODES )
		return "(bad video mode index)";
	return all_video_modes[idx].nvm_name ;
}


/*
static Named_Color_Coding all_color_codes[]={
{ "mono8",	DC1394_COLOR_CODING_MONO8	},
{ "yuv411",	DC1394_COLOR_CODING_YUV411	},
{ "yuv422",	DC1394_COLOR_CODING_YUV422	},
{ "yuv444",	DC1394_COLOR_CODING_YUV444	},
{ "rgb8",	DC1394_COLOR_CODING_RGB8	},
{ "mono16",	DC1394_COLOR_CODING_MONO16	},
{ "rgb16",	DC1394_COLOR_CODING_RGB16	},
{ "mono16s",	DC1394_COLOR_CODING_MONO16S	},
{ "rgb16s",	DC1394_COLOR_CODING_RGB16S	},
{ "raw8",	DC1394_COLOR_CODING_RAW8	},
{ "raw16",	DC1394_COLOR_CODING_RAW16	}
};

#define N_NAMED_COLOR_CODES	(sizeof(all_color_codes)/sizeof(Named_Color_Coding))
*/

static Named_Grab_Mode all_grab_modes[]={
FC2_ENTRY(	drop_frames,	FC2_DROP_FRAMES		),
FC2_ENTRY(	buffer_frames,	FC2_BUFFER_FRAMES	)
};

#define N_NAMED_GRAB_MODES	(sizeof(all_grab_modes)/sizeof(Named_Grab_Mode))


#ifdef HAVE_LIBFLYCAP

static Named_Frame_Rate all_framerates[]={
FC2_ENTRY(	1.875,		FC2_FRAMERATE_1_875	),
FC2_ENTRY(	3.75,		FC2_FRAMERATE_3_75	),
FC2_ENTRY(	7.5,		FC2_FRAMERATE_7_5	),
FC2_ENTRY(	15,		FC2_FRAMERATE_15	),
FC2_ENTRY(	30,		FC2_FRAMERATE_30	),
FC2_ENTRY(	60,		FC2_FRAMERATE_60	),
FC2_ENTRY(	120,		FC2_FRAMERATE_120	),
FC2_ENTRY(	240,		FC2_FRAMERATE_240	),
FC2_ENTRY(	format7,	FC2_FRAMERATE_FORMAT7	)
};

#define N_NAMED_FRAMERATES	(sizeof(all_framerates)/sizeof(Named_Frame_Rate))
#define N_STD_FRAMERATES	(N_NAMED_FRAMERATES-1)

static Named_Bus_Speed all_bus_speeds[]={
FC2_ENTRY(	100,		FC2_BUSSPEED_S100		),
FC2_ENTRY(	200,		FC2_BUSSPEED_S200		),
FC2_ENTRY(	400,		FC2_BUSSPEED_S400		),
FC2_ENTRY(	800,		FC2_BUSSPEED_S800		),
FC2_ENTRY(	1600,		FC2_BUSSPEED_S1600		),
FC2_ENTRY(	3200,		FC2_BUSSPEED_S3200		),
FC2_ENTRY(	5000,		FC2_BUSSPEED_S5000		),
FC2_ENTRY(	10baseT,	FC2_BUSSPEED_10BASE_T		),
FC2_ENTRY(	100baseT,	FC2_BUSSPEED_100BASE_T		),
FC2_ENTRY(	1000baseT,	FC2_BUSSPEED_1000BASE_T		),
FC2_ENTRY(	10000baseT,	FC2_BUSSPEED_10000BASE_T	),
FC2_ENTRY(	fastest,	FC2_BUSSPEED_S_FASTEST		),
FC2_ENTRY(	any,		FC2_BUSSPEED_ANY		)
};

#define N_NAMED_BUS_SPEEDS	(sizeof(all_bus_speeds)/sizeof(Named_Bus_Speed))

static Named_Bandwidth_Allocation all_bw_allocations[]={
FC2_ENTRY(	off,		FC2_BANDWIDTH_ALLOCATION_OFF		),
FC2_ENTRY(	on,		FC2_BANDWIDTH_ALLOCATION_ON		),
FC2_ENTRY(	unsupported,	FC2_BANDWIDTH_ALLOCATION_UNSUPPORTED	),
FC2_ENTRY(	unspecified,	FC2_BANDWIDTH_ALLOCATION_UNSPECIFIED	)
};

#define N_NAMED_BW_ALLOCATIONS	(sizeof(all_bw_allocations)/sizeof(Named_Bandwidth_Allocation))

static Named_Interface all_interfaces[]={
FC2_ENTRY(	ieee1394,	FC2_INTERFACE_IEEE1394	),
FC2_ENTRY(	usb2,		FC2_INTERFACE_USB_2	),
FC2_ENTRY(	usb3,		FC2_INTERFACE_USB_3	),
FC2_ENTRY(	gigE,		FC2_INTERFACE_GIGE	)
};

#define N_NAMED_INTERFACES	(sizeof(all_interfaces)/sizeof(Named_Interface))

#endif // HAVE_LIBFLYCAP


const char *eii_prop_names[N_EII_PROPERTIES]={
	"timestamp",
	"gain",
	"shutter",
	"brightness",
	"exposure",
	"whiteBalance",
	"frameCounter",
	"strobePattern",
	"GPIOPinState",
	"ROIPosition"
};

#ifdef HAVE_LIBFLYCAP

static double get_pgc_size(QSP_ARG_DECL  Item *ip, int dim_index)
{
	switch(dim_index){
		case 0:	return(1.0); /* BUG - not correct for color cameras! */ break;
		case 1: return(((PGR_Cam *)ip)->pc_cols);
		case 2: return(((PGR_Cam *)ip)->pc_rows);
		case 3: return(((PGR_Cam *)ip)->pc_n_buffers);
		case 4: return(1.0);
//#ifdef CAUTIOUS
		default:
//			sprintf(ERROR_STRING,"CAUTIOUS:  Unexpected dimension index (%d) in get_pgc_size!?",dim_index);
//			WARN(ERROR_STRING);
			assert(0);
			break;
//#endif // CAUTIOUS
	}
	return(0.0);
}

static const char * get_pgc_prec_name(QSP_ARG_DECL  Item *ip )
{
	//PGR_Cam *pgcp;

	//pgcp = (PGR_Cam *)ip;

	WARN("get_pgc_prec_name:  need to implement camera-state-based value!?");

	//return def_prec_name(QSP_ARG  ip);
	return("u_byte");
}


static Size_Functions pgc_sf={
	get_pgc_size,
	get_pgc_prec_name

};


static void report_fc2_error(QSP_ARG_DECL  fc2Error error, const char *whence )
{
	const char *msg;

	switch(error){
		case FC2_ERROR_UNDEFINED:
			msg = "Undefined flycap error."; break;
		case FC2_ERROR_OK:
			msg = "Function returned with no errors."; break;
		case FC2_ERROR_FAILED:
			msg = "General failure."; break;
		case FC2_ERROR_NOT_IMPLEMENTED:
			msg = "Function has not been implemented."; break;
		case FC2_ERROR_FAILED_BUS_MASTER_CONNECTION:
			msg = "Could not connect to Bus Master."; break;
		case FC2_ERROR_NOT_CONNECTED:
			msg = "Camera has not been connected."; break;
		case FC2_ERROR_INIT_FAILED:
			msg = "Initialization failed."; break;
		case FC2_ERROR_NOT_INTITIALIZED:
			msg = "Camera has not been initialized."; break;
		case FC2_ERROR_INVALID_PARAMETER:
			msg = "Invalid parameter passed to function."; break;
		case FC2_ERROR_INVALID_SETTINGS:
			msg = "Setting set to camera is invalid."; break;
		case FC2_ERROR_INVALID_BUS_MANAGER:
			msg = "Invalid Bus Manager object."; break;
		case FC2_ERROR_MEMORY_ALLOCATION_FAILED:
			msg = "Could not allocate memory."; break;
		case FC2_ERROR_LOW_LEVEL_FAILURE:
			msg = "Low level error."; break;
		case FC2_ERROR_NOT_FOUND:
			msg = "Device not found."; break;
		case FC2_ERROR_FAILED_GUID:
			msg = "GUID failure."; break;
		case FC2_ERROR_INVALID_PACKET_SIZE:
			msg = "Packet size set to camera is invalid."; break;
		case FC2_ERROR_INVALID_MODE:
			msg = "Invalid mode has been passed to function."; break;
		case FC2_ERROR_NOT_IN_FORMAT7:
			msg = "Error due to not being in Format7."; break;
		case FC2_ERROR_NOT_SUPPORTED:
			msg = "This feature is unsupported."; break;
		case FC2_ERROR_TIMEOUT:
			msg = "Timeout error."; break;
		case FC2_ERROR_BUS_MASTER_FAILED:
			msg = "Bus Master Failure."; break;
		case FC2_ERROR_INVALID_GENERATION:
			msg = "Generation Count Mismatch."; break;
		case FC2_ERROR_LUT_FAILED:
			msg = "Look Up Table failure."; break;
		case FC2_ERROR_IIDC_FAILED:
			msg = "IIDC failure."; break;
		case FC2_ERROR_STROBE_FAILED:
			msg = "Strobe failure."; break;
		case FC2_ERROR_TRIGGER_FAILED:
			msg = "Trigger failure."; break;
		case FC2_ERROR_PROPERTY_FAILED:
			msg = "Property failure."; break;
		case FC2_ERROR_PROPERTY_NOT_PRESENT:
			msg = "Property is not present."; break;
		case FC2_ERROR_REGISTER_FAILED:
			msg = "Register access failed."; break;
		case FC2_ERROR_READ_REGISTER_FAILED:
			msg = "Register read failed."; break;
		case FC2_ERROR_WRITE_REGISTER_FAILED:
			msg = "Register write failed."; break;
		case FC2_ERROR_ISOCH_FAILED:
			msg = "Isochronous failure."; break;
		case FC2_ERROR_ISOCH_ALREADY_STARTED:
			msg = "Isochronous transfer has already been started."; break;
		case FC2_ERROR_ISOCH_NOT_STARTED:
			msg = "Isochronous transfer has not been started."; break;
		case FC2_ERROR_ISOCH_START_FAILED:
			msg = "Isochronous start failed."; break;
		case FC2_ERROR_ISOCH_RETRIEVE_BUFFER_FAILED:
			msg = "Isochronous retrieve buffer failed."; break;
		case FC2_ERROR_ISOCH_STOP_FAILED:
			msg = "Isochronous stop failed."; break;
		case FC2_ERROR_ISOCH_SYNC_FAILED:
			msg = "Isochronous image synchronization failed."; break;
		case FC2_ERROR_ISOCH_BANDWIDTH_EXCEEDED:
			msg = "Isochronous bandwidth exceeded."; break;
		case FC2_ERROR_IMAGE_CONVERSION_FAILED:
			msg = "Image conversion failed."; break;
		case FC2_ERROR_IMAGE_LIBRARY_FAILURE:
			msg = "Image library failure."; break;
		case FC2_ERROR_BUFFER_TOO_SMALL:
			msg = "Buffer is too small."; break;
		case FC2_ERROR_IMAGE_CONSISTENCY_ERROR:
			msg = "There is an image consistency error."; break;

		default:
			sprintf(ERROR_STRING,
		"report_fc2_error (%s):  unhandled error code %d!?\n",
				whence,error);
			WARN(ERROR_STRING);
			msg = "unhandled error code";
			break;
	}
	sprintf(ERROR_STRING,"%s:  %s",whence,msg);
	WARN(ERROR_STRING);
}

ITEM_INTERFACE_DECLARATIONS(PGR_Property_Type,pgr_prop,0)

//  When we change cameras, we have to refresh all properties!

static void _init_one_property(QSP_ARG_DECL const char *name, fc2PropertyType t)
{
	PGR_Property_Type *pgpt;

	pgpt = new_pgr_prop(QSP_ARG  name);
	if( pgpt == NULL ) return;
	pgpt->info.type =
	pgpt->prop.type =
	pgpt->type_code = t;
}

void list_cam_properties(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	List *lp;
	Node *np;
	PGR_Property_Type *pgpt;

	lp = pgr_prop_list(SINGLE_QSP_ARG);	// all properties
	np = QLIST_HEAD(lp);
	if( np != NO_NODE ){
		sprintf(MSG_STR,"\n%s properties",pgcp->pc_name);
		prt_msg(MSG_STR);
	} else {
		sprintf(ERROR_STRING,"%s has no properties!?",pgcp->pc_name);
		WARN(ERROR_STRING);
		return;
	}

	while(np!=NO_NODE){
		pgpt = (PGR_Property_Type *)NODE_DATA(np);
		if( pgpt->info.present ){
			sprintf(MSG_STR,"\t%s",pgpt->name);
			prt_msg(MSG_STR);
		}
		np = NODE_NEXT(np);
	}
	prt_msg("");
}

// We call this after we select a camera

void refresh_camera_properties(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	List *lp;
	Node *np;
	PGR_Property_Type *pgpt;

	lp = pgr_prop_list(SINGLE_QSP_ARG);	// all properties
	np = QLIST_HEAD(lp);
	while(np!=NO_NODE){
		pgpt = (PGR_Property_Type *)NODE_DATA(np);
		refresh_property_info(QSP_ARG  pgcp, pgpt );
		if( pgpt->info.present ){
			refresh_property_value(QSP_ARG  pgcp, pgpt );
		}
		np = NODE_NEXT(np);
	}
}

#define init_one_property(n,t)	_init_one_property(QSP_ARG  n, t)

static void init_property_types(SINGLE_QSP_ARG_DECL)
{
	init_one_property( "brightness",	FC2_BRIGHTNESS		);
	init_one_property( "auto_exposure",	FC2_AUTO_EXPOSURE	);
	init_one_property( "sharpness",		FC2_SHARPNESS		);
	init_one_property( "white_balance",	FC2_WHITE_BALANCE	);
	init_one_property( "hue",		FC2_HUE			);
	init_one_property( "saturation",	FC2_SATURATION		);
	init_one_property( "gamma",		FC2_GAMMA		);
	init_one_property( "iris",		FC2_IRIS		);
	init_one_property( "focus",		FC2_FOCUS		);
	init_one_property( "zoom",		FC2_ZOOM		);
	init_one_property( "pan",		FC2_PAN			);
	init_one_property( "tilt",		FC2_TILT		);
	init_one_property( "shutter",		FC2_SHUTTER		);
	init_one_property( "gain",		FC2_GAIN		);
	init_one_property( "trigger_mode",	FC2_TRIGGER_MODE	);
	init_one_property( "trigger_delay",	FC2_TRIGGER_DELAY	);
	init_one_property( "frame_rate",	FC2_FRAME_RATE		);
	init_one_property( "temperature",	FC2_TEMPERATURE		);
}

void refresh_property_info(QSP_ARG_DECL  PGR_Cam *pgcp, PGR_Property_Type *pgpt )
{
	fc2Error error;

	error = fc2GetPropertyInfo( pgcp->pc_context, &(pgpt->info) );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetPropertyInfo" );
		return;
	}
}

void show_property_info(QSP_ARG_DECL  PGR_Cam *pgcp, PGR_Property_Type *pgpt )
{
	char var_name[32],val_str[32];

	sprintf(MSG_STR,"\n%s %s info:",pgcp->pc_name,pgpt->name);
	prt_msg(MSG_STR);

	// Now print out the property info?
	if( ! pgpt->info.present ){
		sprintf(MSG_STR,"%s is not present.",pgpt->name);
		prt_msg(MSG_STR);
		return;
	}

	if( pgpt->info.autoSupported )
		prt_msg("\tAuto is supported");
	else
		prt_msg("\tAuto is not supported");

	if( pgpt->info.manualSupported )
		prt_msg("\tManual is supported");
	else
		prt_msg("\tManual is not supported");

	if( pgpt->info.onOffSupported )
		prt_msg("\tOn/Off is supported");
	else
		prt_msg("\tOn/Off is not supported");

	if( pgpt->info.onePushSupported )
		prt_msg("\tOne push is supported");
	else
		prt_msg("\tOne push is not supported");

	if( pgpt->info.absValSupported )
		prt_msg("\tAbs. Val. is supported");
	else
		prt_msg("\tAbs. Val. is not supported");

	if( pgpt->info.readOutSupported )
		prt_msg("\tReadout is supported");
	else
		prt_msg("\tReadout is not supported");

	if( pgpt->info.absValSupported ){
		sprintf(MSG_STR,"\tRange:\n\t\t"
		"%d - %d (integer)\n\t\t"
		"%g - %g (absolute)",
			pgpt->info.min,
			pgpt->info.max,
	pgpt->info.absMin,pgpt->info.absMax);
		prt_msg(MSG_STR);

		sprintf(var_name,"%s_abs_min",pgpt->name);	// BUG possible buffer overrun, use snprintf or whatever...
		sprintf(val_str,"%g",pgpt->info.absMin);
		ASSIGN_VAR(var_name,val_str);

		sprintf(var_name,"%s_abs_max",pgpt->name);	// BUG possible buffer overrun, use snprintf or whatever...
		sprintf(val_str,"%g",pgpt->info.absMax);
		ASSIGN_VAR(var_name,val_str);
	} else {
		sprintf(MSG_STR,"\tRange:  %d - %d",
			pgpt->info.min,pgpt->info.max);
		prt_msg(MSG_STR);

		sprintf(var_name,"%s_abs_min",pgpt->name);	// BUG possible buffer overrun, use snprintf or whatever...
		ASSIGN_VAR(var_name,"(undefined)");

		sprintf(var_name,"%s_abs_max",pgpt->name);	// BUG possible buffer overrun, use snprintf or whatever...
		ASSIGN_VAR(var_name,"(undefined)");
	}

	sprintf(var_name,"%s_min",pgpt->name);	// BUG possible buffer overrun, use snprintf or whatever...
	sprintf(val_str,"%d",pgpt->info.min);
	ASSIGN_VAR(var_name,val_str);

	sprintf(var_name,"%s_max",pgpt->name);	// BUG possible buffer overrun, use snprintf or whatever...
	sprintf(val_str,"%d",pgpt->info.max);
	ASSIGN_VAR(var_name,val_str);

	sprintf(MSG_STR,"\tUnits:  %s (%s)",pgpt->info.pUnits,pgpt->info.pUnitAbbr);
	prt_msg(MSG_STR);
}

void refresh_property_value(QSP_ARG_DECL  PGR_Cam *pgcp, PGR_Property_Type *pgpt )
{
	fc2Error error;

	error = fc2GetProperty( pgcp->pc_context, &(pgpt->prop) );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetProperty" );
		return;
	}
}

void show_property_value(QSP_ARG_DECL  PGR_Cam *pgcp, PGR_Property_Type *pgpt )
{
	sprintf(MSG_STR,"\n%s %s:",
		pgcp->pc_name,pgpt->name);
	prt_msg(MSG_STR);

	if( pgpt->info.autoSupported ){
		if( pgpt->prop.autoManualMode )
			sprintf(MSG_STR,"\tAuto mode enabled");
		else
			sprintf(MSG_STR,"\tAuto mode disabled");
	} else if( pgpt->info.manualSupported ){
		if( pgpt->prop.autoManualMode )
			sprintf(MSG_STR,"\tautoManualMode is true");
		else
			sprintf(MSG_STR,"\tautoManualMode is false");
	} else {
		sprintf(MSG_STR,"HUH???  Does not support auto or manual!?");
	}
	prt_msg(MSG_STR);

	if( pgpt->info.onOffSupported ){
		if( pgpt->prop.onOff )
			prt_msg("\tOn");
		else
			prt_msg("\tOff");
	}

	if( pgpt->info.onePushSupported ){
		if( pgpt->prop.onePush )
			prt_msg("\tOne push is true");
		else
			prt_msg("\tOne push is false");
	}
	// What exactly is readOut???  does this tell us whether we can read
	// the value back from the camera???
	if( pgpt->info.readOutSupported ){
		// Now print out the property value itself!
		// Can we see both???

		sprintf(MSG_STR,"\t%s:  %d (integer)",
			pgpt->name,pgpt->prop.valueA);
		prt_msg(MSG_STR);

		// let a script access the value also
		sprintf(MSG_STR,"%d",pgpt->prop.valueA);
		ASSIGN_VAR(pgpt->name,MSG_STR);
		// should this be a reserved var?  I think so!

		if( pgpt->info.absValSupported ){
			sprintf(MSG_STR,"\t%s:  %g %s (absolute)",
				pgpt->name,pgpt->prop.absValue,pgpt->info.pUnitAbbr);
			prt_msg(MSG_STR);

			// let a script access the value also
			sprintf(MSG_STR,"%g",pgpt->prop.absValue);
			sprintf(ERROR_STRING,"%s_abs",pgpt->name);	// using ERROR_STRING as a temporary...
			ASSIGN_VAR(ERROR_STRING,MSG_STR);
			// should this be a reserved var?  I think so!
		} else {
			sprintf(ERROR_STRING,"%s_abs",pgpt->name);	// using ERROR_STRING as a temporary...
			ASSIGN_VAR(ERROR_STRING,"(undefined)");
		}
	} else {
		prt_msg("\t(Readout not supported)");
		sprintf(ERROR_STRING,"%s",pgpt->name);
		ASSIGN_VAR(ERROR_STRING,"(undefined)");
		sprintf(ERROR_STRING,"%s_abs",pgpt->name);
		ASSIGN_VAR(ERROR_STRING,"(undefined)");
	}
}

void set_prop_value(QSP_ARG_DECL  PGR_Cam *pgcp, PGR_Property_Type *pgpt, PGR_Prop_Val *vp )
{
	fc2Error error;

	if( vp->pv_is_abs ){
		if( vp->pv_u.u_f < pgpt->info.absMin || vp->pv_u.u_f > pgpt->info.absMax ){
			sprintf(ERROR_STRING,"Requested %s (%f) out of range (%f - %f)",
				pgpt->name,
				vp->pv_u.u_f,pgpt->info.absMin,pgpt->info.absMax);
			WARN(ERROR_STRING);
			return;
		}
		pgpt->prop.absControl = TRUE;
		pgpt->prop.absValue = vp->pv_u.u_f;
	} else {
		if( vp->pv_u.u_i < pgpt->info.min || vp->pv_u.u_i > pgpt->info.max ){
			sprintf(ERROR_STRING,"Requested %s (%d) out of range (%d - %d)",
				pgpt->name,
				vp->pv_u.u_i,pgpt->info.min,pgpt->info.max);
			WARN(ERROR_STRING);
			return;
		}
		pgpt->prop.absControl = FALSE;
		pgpt->prop.valueA = vp->pv_u.u_i;
	}

	error = fc2SetProperty( pgcp->pc_context, &(pgpt->prop));
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetProperty" );
		return;
	}
}

void set_prop_auto(QSP_ARG_DECL  PGR_Cam *pgcp, PGR_Property_Type *pgpt, BOOL yn )
{
	fc2Error error;

	if( ! pgpt->info.autoSupported ){
		sprintf(ERROR_STRING,"Sorry, auto mode not supported for %s.",
			pgpt->name);
		WARN(ERROR_STRING);
		return;
	}

	pgpt->prop.autoManualMode = yn;
	error = fc2SetProperty( pgcp->pc_context, &(pgpt->prop) );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetProperty" );
		return;
	}
}

static void insure_stopped(QSP_ARG_DECL  PGR_Cam *pgcp, const char *op_desc)
{
	if( (pgcp->pc_flags & PGR_CAM_IS_RUNNING) == 0 ) return;

	sprintf(ERROR_STRING,"Stopping capture on %s prior to %s",
		pgcp->pc_name,op_desc);
	advise(ERROR_STRING);

	stop_firewire_capture(QSP_ARG  pgcp);
}

void
cleanup_cam( PGR_Cam *pgcp )
{
	//if( IS_CAPTURING(pgcp) )
		 //dc1394_capture_stop( pgcp->pc_cam_p );
		 //fly_capture_stop( pgcp->pc_cam_p );
	//if( IS_TRANSMITTING(pgcp) )
		//dc1394_video_set_transmission( pgcp->pc_cam_p, DC1394_OFF );
		//fly_video_set_transmission( pgcp->pc_cam_p, DC1394_OFF );
	/* dc1394_free_camera */
	//dc1394_camera_free( pgcp->pc_cam_p );
	//fly_camera_free( pgcp->pc_cam_p );
}


#define INDEX_SEARCH( stem, type, count, short_stem )			\
									\
static int index_of_##stem( type val )					\
{									\
	unsigned int i;							\
									\
	for(i=0;i<count;i++){						\
		if( all_##stem##s[i].short_stem##_value == val )	\
			return(i);					\
	}								\
	return -1;							\
}

#ifdef NOT_USED
INDEX_SEARCH(index_of_feature,dc1394feature_t,N_NAMED_FEATURES,all_features,nft_feature)
#endif /* NOT_USED */

#ifdef FOOBAR
INDEX_SEARCH(index_of_trigger_mode,dc1394trigger_mode_t,N_NAMED_TRIGGER_MODES,all_trigger_modes,ntm_mode)
#endif // FOOBAR

INDEX_SEARCH(video_mode,fc2VideoMode,N_NAMED_VIDEO_MODES,nvm)
INDEX_SEARCH(framerate,fc2FrameRate,N_NAMED_FRAMERATES,nfr)
INDEX_SEARCH(grab_mode,fc2GrabMode,N_NAMED_GRAB_MODES,ngm)
INDEX_SEARCH(bus_speed,fc2BusSpeed,N_NAMED_BUS_SPEEDS,nbs)
INDEX_SEARCH(bw_allocation,fc2BandwidthAllocation,N_NAMED_BW_ALLOCATIONS,nba)
INDEX_SEARCH(interface,fc2InterfaceType,N_NAMED_INTERFACES,nif)

#define NAME_LOOKUP_FUNC(stem,type,short_stem)		\
							\
static const char *name_for_##stem( type val )		\
{							\
	int i;						\
	i = index_of_##stem(val);			\
	if( i >= 0 )					\
		return(all_##stem##s[i].short_stem##_name);	\
	return NULL;					\
}

NAME_LOOKUP_FUNC(video_mode,fc2VideoMode,nvm)
NAME_LOOKUP_FUNC(framerate,fc2FrameRate,nfr)
NAME_LOOKUP_FUNC(grab_mode,fc2GrabMode,ngm)
NAME_LOOKUP_FUNC(bus_speed,fc2BusSpeed,nbs)
NAME_LOOKUP_FUNC(bw_allocation,fc2BandwidthAllocation,nba)
NAME_LOOKUP_FUNC(interface,fc2InterfaceType,nif)


#ifdef FOOBAR
/* called once at camera initialization... */

void get_camera_features( PGR_Cam *pgcp )
{
	Node *np;
	int i;

	if ( fly_get_cam_features(pgcp) < 0 ) {
		NERROR1("get_camera_features:  unable to get camera feature set");
	}

	/* Now can the table and build the linked list */
#ifdef FUBAR
//#ifdef CAUTIOUS
//	if( pgcp->pc_feat_lp != NO_LIST ) NERROR1("CAUTIOUS:  get_camera_features:  bad list ptr!?");
//#endif /* CAUTIOUS */
	assert( pgcp->pc_feat_lp == NO_LIST );
#endif /* FUBAR */
	/* We may call this again after we have diddled the controls... */
	/* releasing and rebuilding the list is wasteful, but should work... */
	if( pgcp->pc_feat_lp != NO_LIST ){
		while( (np=remHead(pgcp->pc_feat_lp)) != NO_NODE )
			rls_node(np);
	} else {
		pgcp->pc_feat_lp = new_list();
	}

	// BUG figure out how to scan features/capabilies

}
#endif // FOOBAR


#ifdef FOOBAR
static void list_camera_feature(QSP_ARG_DECL  dc1394feature_info_t *feat_p )
{
	sprintf(msg_str,"%s", /*dc1394_feature_desc[feat_p->id - DC1394_FEATURE_MIN]*/
			dc1394_feature_get_string(feat_p->id) );
	prt_msg(msg_str);
}

int list_camera_features(QSP_ARG_DECL  PGR_Cam *pgcp )
{
	Node *np;

//	if( pgcp->pc_feat_lp == NO_LIST ) NERROR1("CAUTIOUS:  list_camera_features:  bad list");
	assert( pgcp->pc_feat_lp != NO_LIST );

	np = pgcp->pc_feat_lp->l_head;
	while(np!=NO_NODE){
		dc1394feature_info_t * f;
		f= (dc1394feature_info_t *) np->n_data;
		list_camera_feature(QSP_ARG  f);
		np=np->n_next;
	}
	return(0);
}

int get_feature_choices( PGR_Cam *pgcp, const char ***chp )
{
	int n;
	const char * /* const */ * sptr;
	Node *np;

	n=eltcount(pgcp->pc_feat_lp);
	if( n <= 0 ) return(0);

	sptr = (const char **) getbuf( n * sizeof(char *) );
	*chp = sptr;

	np=pgcp->pc_feat_lp->l_head;
	while(np!=NO_NODE){
		dc1394feature_info_t *f;
		f= (dc1394feature_info_t *) np->n_data;
		*sptr = /*(char *)dc1394_feature_desc[f->id - DC1394_FEATURE_MIN]*/
			dc1394_feature_get_string(f->id) ;
		sptr++;
		np=np->n_next;
	}
	return n;
}

void report_feature_info(QSP_ARG_DECL  PGR_Cam *pgcp, dc1394feature_t id )
{
	Node *np;
	dc1394feature_info_t *f;
	unsigned int i;
	const char *name;
	char nbuf[32];

	np = pgcp->pc_feat_lp->l_head;
	f=NULL;
	while( np != NO_NODE ){
		f= (dc1394feature_info_t *) np->n_data;

		if( f->id == id )
			np=NO_NODE;
		else
			f=NULL;

		if( np != NO_NODE )
			np = np->n_next;
	}

//#ifdef CAUTIOUS
//	if( f == NULL ){
//		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  report_feature_info:  couldn't find %s",
//			/*dc1394_feature_desc[id - DC1394_FEATURE_MIN]*/
//			dc1394_feature_get_string(id) );
//		NWARN(DEFAULT_ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( f != NULL );

	name=/*dc1394_feature_desc[f->id - DC1394_FEATURE_MIN]*/
		dc1394_feature_get_string(f->id);
	sprintf(nbuf,"%s:",name);
	sprintf(msg_str,"%-16s",nbuf);
	prt_msg_frag(msg_str);

	if (f->on_off_capable) {
		if (f->is_on) 
			prt_msg_frag("ON\t");
		else
			prt_msg_frag("OFF\t");
	} else {
		prt_msg_frag("\t");
	}

	/*
	if (f->one_push){
		if (f->one_push_active)
			prt_msg_frag("  one push: ACTIVE");
		else
			prt_msg_frag("  one push: INACTIVE");
	}
	prt_msg("");
	*/
	/* BUG need to use (new?) feature_get_modes... */
	 /* FIXME */
	/*
	if( f->auto_capable ){
		if (f->auto_active) 
			prt_msg_frag("AUTO\t");
		else
			prt_msg_frag("MANUAL\t");
	} else {
		prt_msg_frag("\t");
	}
	*/

	/*
	prt_msg("");
	*/

	/*
	if( f->id != DC1394_FEATURE_TRIGGER ){
		sprintf(msg_str,"\tmin: %d max %d", f->min, f->max);
		prt_msg(msg_str);
	}
	if( f->absolute_capable){
		sprintf(msg_str,"\tabsolute settings:  value: %f  min: %f  max: %f",
			f->abs_value,f->abs_min,f->abs_max);
		prt_msg(msg_str);
	}
	*/

	switch(f->id){
		case DC1394_FEATURE_TRIGGER:
			switch(f->trigger_modes.num){
				case 0:
					prt_msg("no trigger modes available");
					break;
				case 1:
					sprintf(msg_str,"one trigger mode (%s)",
						name_for_trigger_mode(f->trigger_modes.modes[0]));
					prt_msg(msg_str);
					break;
				default:
					sprintf(msg_str,"%d trigger modes (",f->trigger_modes.num);
					prt_msg_frag(msg_str);
					for(i=0;i<f->trigger_modes.num-1;i++){
						sprintf(msg_str,"%s, ",
					name_for_trigger_mode(f->trigger_modes.modes[i]));
						prt_msg_frag(msg_str);
					}
					sprintf(msg_str,"%s)",
						name_for_trigger_mode(f->trigger_modes.modes[i]));
					prt_msg(msg_str);

					break;
			}
			break;
			/*
    printf("\n\tAvailableTriggerModes: ");
    if (f->trigger_modes.num==0) {
      printf("none");
    }
    else {
      int i;
      for (i=0;i<f->trigger_modes.num;i++) {
	printf("%d ",f->trigger_modes.modes[i]);
      }
    }
    printf("\n\tAvailableTriggerSources: ");
    if (f->trigger_sources.num==0) {
      printf("none");
    }
    else {
      int i;
      for (i=0;i<f->trigger_sources.num;i++) {
	printf("%d ",f->trigger_sources.sources[i]);
      }
    }
    printf("\n\tPolarity Change Capable: ");
    
    if (f->polarity_capable) 
      printf("True");
    else 
      printf("False");
    
    printf("\n\tCurrent Polarity: ");
    
    if (f->trigger_polarity) 
      printf("POS");
    else 
      printf("NEG");
    
    printf("\n\tcurrent mode: %d\n", f->trigger_mode);
    if (f->trigger_sources.num>0) {
      printf("\n\tcurrent source: %d\n", f->trigger_source);
    }
    */
		case DC1394_FEATURE_WHITE_BALANCE: 
		case DC1394_FEATURE_TEMPERATURE:
		case DC1394_FEATURE_WHITE_SHADING: 
			NWARN("unhandled case in feature type switch");
			break;
		default:
			sprintf(msg_str,"value: %-8d  range: %d-%d",f->value,f->min,f->max);
			prt_msg(msg_str);
			break;
	}
}
#endif // FOOBAR



int get_camera_names( QSP_ARG_DECL  Data_Obj *str_dp )
{
	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?
	List *lp;
	Node *np;
	PGR_Cam *pgcp;
	int i, n;

	lp = pgc_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST ){
		WARN("No cameras!?");
		return 0;
	}

	n=eltcount(lp);
	if( OBJ_COLS(str_dp) < n ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d camera names",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),n);
		WARN(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	}
		
	np=QLIST_HEAD(lp);
	i=0;
	while(np!=NO_NODE){
		char *dst;
		pgcp = (PGR_Cam *) NODE_DATA(np);
		dst = OBJ_DATA_PTR(str_dp);
		dst += i * OBJ_PXL_INC(str_dp);
		if( strlen(pgcp->pc_name)+1 > OBJ_COMPS(str_dp) ){
			sprintf(ERROR_STRING,"String object %s has too few components (%ld) to hold camera name \"%s\"",
				OBJ_NAME(str_dp),(long)OBJ_COMPS(str_dp),pgcp->pc_name);
			WARN(ERROR_STRING);
		} else {
			strcpy(dst,pgcp->pc_name);
		}
		i++;
		if( i>=n )
			np=NO_NODE;
		else
			np = NODE_NEXT(np);
	}

	return i;
}

int get_video_mode_strings( QSP_ARG_DECL  Data_Obj *str_dp, PGR_Cam *pgcp )
{
	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?

	int i, n;

	if( OBJ_COLS(str_dp) < pgcp->pc_n_video_modes ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d modes",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),pgcp->pc_n_video_modes);
		WARN(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	} else {
		n=pgcp->pc_n_video_modes;
	}
		
	for(i=0;i<n;i++){
		int k;
		const char *src;
		char *dst;

		k=pgcp->pc_video_mode_indices[i];
		src = all_video_modes[k].nvm_name;
		dst = OBJ_DATA_PTR(str_dp);
		dst += i * OBJ_PXL_INC(str_dp);
		if( strlen(src)+1 > OBJ_COMPS(str_dp) ){
			sprintf(ERROR_STRING,"String object %s has too few components (%ld) to hold mode string \"%s\"",
				OBJ_NAME(str_dp),(long)OBJ_COMPS(str_dp),src);
			WARN(ERROR_STRING);
		} else {
			strcpy(dst,src);
		}
	}
	set_script_var_from_int(QSP_ARG  "n_video_modes",n);
	return n;
}

static int bit_count(Framerate_Mask mask)
{
	int c=0;
	int nbits;

	nbits = sizeof(mask) * 8;
	while(nbits--){
		if( mask & 1 ) c++;
		mask >>= 1;
	}
	return c;
}

static void get_framerate_choices(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	unsigned int i,n,idx;
	Framerate_Mask mask;

	if( pgcp->pc_framerate_names != NULL ){
		givbuf(pgcp->pc_framerate_names);
		pgcp->pc_framerate_names=NULL;	// in case we have an error before finishing
	}

	// format7 doesn't have a framerate!?

	mask = pgcp->pc_framerate_mask_tbl[ pgcp->pc_my_video_mode_index ];
/*
sprintf(ERROR_STRING,"%s:  video mode is %s",
pgcp->pc_name,all_video_modes[pgcp->pc_video_mode_index].nvm_name);
advise(ERROR_STRING);
sprintf(ERROR_STRING,"%s:  my video mode %s (index = %d)",
pgcp->pc_name,pgcp->pc_video_mode_names[pgcp->pc_my_video_mode_index],
pgcp->pc_my_video_mode_index);
advise(ERROR_STRING);
*/

	n = bit_count(mask);
	if( n <= 0 ){
		// this happens for the format7 video mode...
		// Can this ever happen?  If not, should be CAUTIOUS...
		//WARN("no framerates for this video mode!?");
		if( pgcp->pc_video_mode == FC2_VIDEOMODE_FORMAT7 ){
			if( pgcp->pc_framerate == FC2_FRAMERATE_FORMAT7 ){
				advise("No framerates available for format7.");
			}
			  else {
//		WARN("CAUTIOUS:  get_framerate_choices:  video mode is format7, but framerate is not!?");
				assert(0);
			}
		}
		  else {
//	WARN("CAUTIOUS:  get_framerate_choices:  video mode is not format7, but no framerates found!?");
			assert(0);
		}
		return;
	}

	pgcp->pc_framerate_names = getbuf( n * sizeof(char *) );
	pgcp->pc_n_framerates = n ;

	i=(-1);
	idx=0;
	while(mask){
		i++;
		if( mask & 1 ){
			fc2FrameRate r;
			int j;
			r = i ;
			j = index_of_framerate( r );
			pgcp->pc_framerate_names[idx]=all_framerates[j].nfr_name;
			idx++;
		}
			
		mask >>= 1;
	}

//#ifdef CAUTIOUS
//	if( idx != n ) ERROR1("CAUTIOUS:  get_framerate_choices:  count mismatch!?");
//#endif // CAUTIOUS
	assert( idx == n );
}

int get_framerate_strings( QSP_ARG_DECL  Data_Obj *str_dp, PGR_Cam *pgcp )
{
	int i, n;
	const char *src;
	char *dst;

	get_framerate_choices(QSP_ARG  pgcp);

	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?

	n = pgcp->pc_n_framerates;

	if( OBJ_COLS(str_dp) < n ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d framerates",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),n);
		WARN(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	}

	for(i=0;i<pgcp->pc_n_framerates;i++){
		src = pgcp->pc_framerate_names[i];
		dst = OBJ_DATA_PTR(str_dp);
		dst += i * OBJ_PXL_INC(str_dp);
		if( strlen(src)+1 > OBJ_COMPS(str_dp) ){
			sprintf(ERROR_STRING,
"String object %s has too few components (%ld) to hold framerate string \"%s\"",
				OBJ_NAME(str_dp),(long)OBJ_COMPS(str_dp),src);
			WARN(ERROR_STRING);
		} else {
			strcpy(dst,src);
		}
	}
	return n;
}

static void test_setup(QSP_ARG_DECL  PGR_Cam *pgcp,
		fc2VideoMode m, fc2FrameRate r, int *kp, int idx )
{
	fc2Error error;
	BOOL supported;

	// punt if format 7
	if( m == FC2_VIDEOMODE_FORMAT7 && r == FC2_FRAMERATE_FORMAT7 ){
		// BUG?  make sure camera has format7 modes?
		supported = TRUE;
	} else {
		error = fc2GetVideoModeAndFrameRateInfo( pgcp->pc_context,
			m,r,&supported );
		if( error != FC2_ERROR_OK ){
			report_fc2_error(QSP_ARG  error,
				"fc2GetVideoModeAndFrameRateInfo" );
			supported = FALSE;
		}
	}

	if( supported ){
		if( *kp < 0 || pgcp->pc_video_mode_indices[*kp] != idx ){
			*kp = (*kp)+1;
			pgcp->pc_video_mode_indices[*kp] = idx;
			pgcp->pc_video_mode_names[*kp] = all_video_modes[idx].nvm_name;
//fprintf(stderr,"test_setup:  adding video mode %s to %s\n",
//all_video_modes[idx].nvm_name,
//pgcp->pc_name);
			if( pgcp->pc_video_mode == m )
				pgcp->pc_video_mode_index = *kp;
		}
		pgcp->pc_framerate_mask_tbl[*kp] |= 1 << r;
	}
}

static int get_supported_video_modes(QSP_ARG_DECL  PGR_Cam *pgcp )
{
	int i,j,n_so_far;

	n_so_far=(-1);
	//for( i=0;i<N_STD_VIDEO_MODES;i++){
	for( i=0;i<N_NAMED_VIDEO_MODES;i++){
		fc2VideoMode m;
		fc2FrameRate r;

		m=all_video_modes[i].nvm_value;
		if( m == FC2_VIDEOMODE_FORMAT7 ){
			r = FC2_FRAMERATE_FORMAT7;
			test_setup(QSP_ARG  pgcp, m, r, &n_so_far, i );
		} else {
			for(j=0;j<N_STD_FRAMERATES;j++){
				r=all_framerates[j].nfr_value;
//fprintf(stderr,"Testing video mode %d and frame rate %d\n",
//m,pgcp->pc_framerate);
				test_setup(QSP_ARG  pgcp, m, r, &n_so_far, i );
			}
		}
	}
	n_so_far++;
//fprintf(stderr,"get_supported_video_modes:  setting n_video_modes to %d\n",k);
	pgcp->pc_n_video_modes = n_so_far;
	return 0;
}

static fc2FrameRate highest_framerate( QSP_ARG_DECL  Framerate_Mask mask )
{
	int k;

	k=(-1);
	while( mask ){
		mask >>= 1;
		k++;
	}
//#ifdef CAUTIOUS
//	if( k < 0 ){
//		ERROR1("CAUTIOUS:  null framerate mask!?");
//	}
//#endif // CAUTIOUS
	assert( k >= 0 );

	return (fc2FrameRate) k;
}

#ifdef FOOBAR
//#ifdef CAUTIOUS
//
//#define CHECK_IDX(whence)				\
//							\
//if( idx < 0 || idx >= pgcp->pc_n_video_modes ) {	\
//	sprintf(ERROR_STRING,				\
//	"CAUTIOUS:  is_fmt7_mode:  bad index %d!?",idx);\
//	WARN(ERROR_STRING);				\
//	return -1;					\
//}
//
//#else // ! CAUTIOUS
//
//#define CHECK_IDX(whence)
//
//#endif // ! CAUTIOUS
#endif // FOOBAR

int is_fmt7_mode(QSP_ARG_DECL  PGR_Cam *pgcp, int idx )
{
	fc2VideoMode m;

//	CHECK_IDX(is_fmt7_mode)
	assert( idx >= 0 && idx < pgcp->pc_n_video_modes );

	m = all_video_modes[ pgcp->pc_video_mode_indices[idx] ].nvm_value;
	if( m == FC2_VIDEOMODE_FORMAT7 ) return 1;
	return 0;
}

// We might want to call this after changing the video mode - we
// don't know whether the library might change anything (like the
// number of buffers, but it seems possible?

static int refresh_config(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	fc2Error error;

	error = fc2GetConfiguration(pgcp->pc_context,&pgcp->pc_config);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetConfiguration" );
		// should we set a flag to indicate an invalid config?
		return -1;
	}
	pgcp->pc_n_buffers = pgcp->pc_config.numBuffers;
	return 0;
}


int set_std_mode(QSP_ARG_DECL  PGR_Cam *pgcp, int idx )
{
	fc2FrameRate r;
	fc2Error error;
	fc2VideoMode m;

//	CHECK_IDX(set_std_mode)
	assert( idx >= 0 && idx < pgcp->pc_n_video_modes );

	insure_stopped(QSP_ARG  pgcp,"setting video mode");

	m = all_video_modes[ pgcp->pc_video_mode_indices[idx] ].nvm_value;
	if( m == FC2_VIDEOMODE_FORMAT7 ){
		WARN("set_std_mode:  use set_fmt7_mode to select format7!?");
		return -1;
	}

	r = highest_framerate(QSP_ARG  pgcp->pc_framerate_mask_tbl[idx] );

	error = fc2SetVideoModeAndFrameRate(pgcp->pc_context,m,r);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetVideoModeAndFrameRate" );
		return -1;
	}

	pgcp->pc_my_video_mode_index = idx;
	pgcp->pc_video_mode = m;
	pgcp->pc_framerate = r;
	pgcp->pc_base = NULL;	// force init_fly_base to run again
	pgcp->pc_video_mode_index = pgcp->pc_video_mode_indices[idx];
	pgcp->pc_framerate_index = index_of_framerate(r);

	pgcp->pc_cols = all_video_modes[ pgcp->pc_video_mode_index ].nvm_width;
	pgcp->pc_rows = all_video_modes[ pgcp->pc_video_mode_index ].nvm_height;
	pgcp->pc_depth = all_video_modes[ pgcp->pc_video_mode_index ].nvm_depth;

	return refresh_config(QSP_ARG  pgcp);
} // set_std_mode

static void set_highest_fmt7_framerate( QSP_ARG_DECL  PGR_Cam *pgcp )
{
	/* If the frame rate has been set to 60 fps by using a default
	 * video mode at startup, it will not be reset when we switch
	 * to format7.  So here we go to the max by default...
	 */

	fc2Error error;
	fc2Property prop;
#ifdef FOOBAR
	fc2PropertyInfo propInfo;

	propInfo.type = FC2_FRAME_RATE;
	error = fc2GetPropertyInfo( pgcp->pc_context, &propInfo );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetPropertyInfo" );
		return;
	}
#endif // FOOBAR
	PGR_Property_Type *fr_prop_p;

	fr_prop_p = get_pgr_prop(QSP_ARG  "frame_rate" );	// BUG string must match table

//#ifdef CAUTIOUS
//	if( fr_prop_p == NULL )
//		ERROR1("CAUTIOUS:  set_highest_fmt7_framerate:"
//			"  couldn't find frame_rate property!?");
//#endif // CAUTIOUS
	assert( fr_prop_p != NULL );

//#ifdef CAUTIOUS
//	if( ! fr_prop_p->info.absValSupported ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  set_highest_fmt7_framerate:  absolute mode not supported for frame rate!?");
//		WARN(ERROR_STRING);
//		return;
//	}
//#endif // CAUTIOUS
	assert( fr_prop_p->info.absValSupported );

	prop.type = FC2_FRAME_RATE;
	error = fc2GetProperty( pgcp->pc_context, &prop);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetProperty" );
		return;
	}
	prop.absControl = TRUE;
	prop.absValue = fr_prop_p->info.absMax;

	error = fc2SetProperty( pgcp->pc_context, &prop);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetProperty" );
		return;
	}
}

int set_fmt7_mode(QSP_ARG_DECL  PGR_Cam *pgcp, int idx )
{
	fc2Format7ImageSettings settings;
//	fc2Format7PacketInfo pinfo;
//	BOOL is_valid;
	fc2Error error;
	unsigned int packetSize;
	float percentage;

	insure_stopped(QSP_ARG  pgcp,"setting format7 mode");

	if( idx < 0 || idx >= pgcp->pc_n_fmt7_modes ){
		WARN("Format 7 index out of range!?");
		return -1;
	}

	settings.mode = idx;
	settings.offsetX = 0;
	settings.offsetY = 0;
	settings.width = pgcp->pc_fmt7_info_tbl[idx].maxWidth;
	settings.height = pgcp->pc_fmt7_info_tbl[idx].maxHeight;
	if( pgcp->pc_fmt7_info_tbl[idx].pixelFormatBitField &
			FC2_PIXEL_FORMAT_RAW8 )
		settings.pixelFormat = FC2_PIXEL_FORMAT_RAW8;
	else {
		WARN("Camera does not support raw8!?");
		return -1;
	}

fprintf(stderr,"Using size %d x %d\n",settings.width,settings.height);

	percentage = 100.0;
	error = fc2SetFormat7Configuration(pgcp->pc_context,&settings,
			percentage);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetFormat7Configuration" );
		return -1;
	}

	set_highest_fmt7_framerate(QSP_ARG  pgcp);

	// This fails if we are not in format 7 already.
	error = fc2GetFormat7Configuration(pgcp->pc_context,&settings,
			&packetSize,&percentage);

	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetFormat7Configuration" );
		return -1;
	}

	fprintf(stderr,"Percentage = %g (packet size = %d)\n",
		percentage,packetSize);

	pgcp->pc_video_mode = FC2_VIDEOMODE_FORMAT7;
	pgcp->pc_framerate = FC2_FRAMERATE_FORMAT7;
	pgcp->pc_framerate_index = index_of_framerate(FC2_FRAMERATE_FORMAT7);
	pgcp->pc_video_mode_index = index_of_video_mode(FC2_VIDEOMODE_FORMAT7);
	pgcp->pc_my_video_mode_index = (-1);
	pgcp->pc_fmt7_index = idx;
	pgcp->pc_base = NULL;	// force init_fly_base to run again

	// Do we have to set the framerate to FC2_FRAMERATE_FORMAT7???

	pgcp->pc_rows = settings.height;
	pgcp->pc_cols = settings.width;

	{
		long bytes_per_image;
		float est_fps;

		bytes_per_image = settings.width * settings.height;
		// assumes mono8

		est_fps = 8000.0 * packetSize / bytes_per_image;
		fprintf(stderr,"Estimated frame rate:  %g\n",est_fps);
	}

	// refresh_config reads the config from the library...
	return refresh_config(QSP_ARG  pgcp);
} // set_fmt7_mode

void set_eii_property(QSP_ARG_DECL  PGR_Cam *pgcp, int idx, int yesno )
{
	myEmbeddedImageInfo *eii_p;
	fc2Error error;

	eii_p = (myEmbeddedImageInfo *) (&pgcp->pc_ei_info);
	if( ! eii_p->prop_tbl[idx].available ){
		sprintf(ERROR_STRING,"Property %s is not available on %s",
			eii_prop_names[idx],pgcp->pc_name);
		WARN(ERROR_STRING);
		return;
	}
	eii_p->prop_tbl[idx].onOff = yesno ? TRUE : FALSE;

	error = fc2SetEmbeddedImageInfo(pgcp->pc_context,&pgcp->pc_ei_info);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetEmbeddedImageInfo" );
	}
}

void set_grab_mode(QSP_ARG_DECL  PGR_Cam *pgcp, int grabmode_idx )
{
	fc2Error error;
	fc2Config cfg;

//#ifdef CAUTIOUS
//	if( grabmode_idx < 0 || grabmode_idx >= N_NAMED_GRAB_MODES ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  set_grab_mode:  bad index (%d)",grabmode_idx);
//		WARN(ERROR_STRING);
//		return;
//	}
//#endif // CAUTIOUS
	assert( grabmode_idx >= 0 && grabmode_idx < N_NAMED_GRAB_MODES );

	// grab mode is part of the config struct
	cfg = pgcp->pc_config;
	cfg.grabMode = all_grab_modes[grabmode_idx].ngm_value;
	error = fc2SetConfiguration(pgcp->pc_context,&cfg);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetConfiguration" );
		// should we set a flag to indicate an invalid config?
		return;
	}
	pgcp->pc_config.grabMode = cfg.grabMode;
}

void show_grab_mode(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	int idx;

	idx = index_of_grab_mode(pgcp->pc_config.grabMode);
	if( idx < 0 ) return;
	sprintf(MSG_STR,"Current grab mode:  %s",all_grab_modes[idx].ngm_name);
	prt_msg(MSG_STR);
}

int pick_framerate(QSP_ARG_DECL  PGR_Cam *pgcp, const char *pmpt)
{
	int i;

	if( pgcp == NULL ){
		sprintf(ERROR_STRING,"pick_framerate:  no camera selected!?");
		WARN(ERROR_STRING);
		return -1;
	}

	get_framerate_choices(QSP_ARG  pgcp);
	i=WHICH_ONE(pmpt,pgcp->pc_n_framerates,pgcp->pc_framerate_names);
	return i;
}

int set_framerate(QSP_ARG_DECL  PGR_Cam *pgcp, int framerate_index)
{
	fc2FrameRate rate;
	fc2Error error;

	if( pgcp == NULL ) return -1;

	insure_stopped(QSP_ARG  pgcp,"setting frame rate");

	rate = all_framerates[framerate_index].nfr_value;
	error = fc2SetVideoModeAndFrameRate(pgcp->pc_context,
			pgcp->pc_video_mode,rate);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetVideoModeAndFrameRate" );
		return -1;
	}

	pgcp->pc_framerate = rate;

	return refresh_config(QSP_ARG  pgcp);
}

void show_framerate(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	sprintf(MSG_STR,"%s framerate:  %s",
		pgcp->pc_name,all_framerates[pgcp->pc_framerate_index].nfr_name);
	advise(MSG_STR);
}

void show_video_mode(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	sprintf(MSG_STR,"%s video mode:  %s",
		pgcp->pc_name,name_for_video_mode(pgcp->pc_video_mode));
	advise(MSG_STR);
}

int list_video_modes(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	unsigned int i;
	const char *s;

	if( pgcp->pc_n_video_modes <= 0 ){
		WARN("no video modes!?");
		return -1;
	}

	for(i=0;i<pgcp->pc_n_video_modes; i++){
		s=pgcp->pc_video_mode_names[i];
		prt_msg_frag("\t");
		prt_msg(s);
	}
	return 0;
}

void list_framerates(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	int i;

	get_framerate_choices(QSP_ARG  pgcp);

	for(i=0;i<pgcp->pc_n_framerates;i++){
		prt_msg_frag("\t");
		prt_msg(pgcp->pc_framerate_names[i]);
	}
}

static int set_default_video_mode(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	fc2VideoMode m;
	fc2FrameRate r;
	int i,j;
	fc2Error error;
	int _nskip;

	if( get_supported_video_modes(QSP_ARG  pgcp ) < 0 ){
		WARN("set_default_video_mode:  Can't get video modes");
		return -1;
	}

//fprintf(stderr,"get_supported_video_modes found %d modes\n",pgcp->pc_n_video_modes);
	_nskip=0;
	do {
		_nskip++;
		i = pgcp->pc_n_video_modes-(_nskip);	// order the table so that mono8 is last?
		pgcp->pc_my_video_mode_index = i;
		j = pgcp->pc_video_mode_indices[i];
		m = all_video_modes[j].nvm_value;
		// BUG we don't check that nskip is in-bounds, but should be OK
	} while( m == FC2_VIDEOMODE_FORMAT7  && _nskip < pgcp->pc_n_video_modes );

	if( m == FC2_VIDEOMODE_FORMAT7 ){
		/*
		sprintf(ERROR_STRING,"set_default_video_mode:  %s has only format7 modes!?",
			pgcp->pc_name);
		advise(ERROR_STRING);
		*/
		return -1;
	}

//fprintf(stderr,"_nskip = %d, i = %d,  j = %d,  m = %d\n",_nskip,i,j,m);

//fprintf(stderr,"highest non-format7 video mode is %s\n",all_video_modes[j].nvm_name);

	// if this is format7, don't use it!?

	pgcp->pc_video_mode = m;
	pgcp->pc_video_mode_index = j;

	pgcp->pc_cols = all_video_modes[ j ].nvm_width;
	pgcp->pc_rows = all_video_modes[ j ].nvm_height;
	pgcp->pc_depth = all_video_modes[ j ].nvm_depth;

	// Get the hightest frame rate associated with this video mode...
	r = highest_framerate( QSP_ARG  pgcp->pc_framerate_mask_tbl[i] );

//fprintf(stderr,"mode %s, highest supported frame rate is %s\n",
//name_for_video_mode(m),
//name_for_framerate(r));
	pgcp->pc_framerate = r;
	pgcp->pc_framerate_index = index_of_framerate(r);

//sprintf(ERROR_STRING,"set_default_video_mode:  setting to %s", name_for_video_mode(m));
//advise(ERROR_STRING);


	error = fc2SetVideoModeAndFrameRate( pgcp->pc_context,
			pgcp->pc_video_mode, pgcp->pc_framerate );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetVideoModeAndFrameRate" );
		return -1;
	}

	// We might stash the number of video modes here?

	// stash the number of framerates in a script variable
	// in case the user wants to fetch the strings...
	// BUG DO THIS WHEN CAM IS SELECTED!
	//set_script_var_from_int(QSP_ARG
	//		"n_framerates",pgcp->pc_framerates.num);

sprintf(ERROR_STRING,"%s:  %s, %s fps",pgcp->pc_name,name_for_video_mode(m),
						name_for_framerate(r) );
advise(ERROR_STRING);

	return 0;
}

static void fix_string( char *s )
{
	while( *s ){
		if( *s == ' ' ) *s='_';
		// other chars to map also?
		s++;
	}
}

static PGR_Cam *unique_camera_instance( QSP_ARG_DECL  fc2Context context )
{
	int i;
	char cname[80];	// How many chars is enough?
	PGR_Cam *pgcp;
	fc2Error error;
	fc2CameraInfo camInfo;

	error = fc2GetCameraInfo( context, &camInfo );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetCameraInfo" );
		return NULL;
	}

	i=1;
	pgcp=NULL;
	while(pgcp==NULL){
		//sprintf(cname,"%s_%d",cam_p->model,i);
		sprintf(cname,"%s_%d",camInfo.modelName,i);
		fix_string(cname);	// change spaces to underscores
//sprintf(ERROR_STRING,"Checking for existence of %s",cname);
//advise(ERROR_STRING);
		pgcp = pgc_of( QSP_ARG  cname );
		if( pgcp == NULL ){	// This index is free
//sprintf(ERROR_STRING,"%s is not in use",cname);
//advise(ERROR_STRING);
			pgcp = new_pgc( QSP_ARG  cname );
			if( pgcp == NULL ){
				sprintf(ERROR_STRING,
			"Failed to create camera %s!?",cname);
				ERROR1(ERROR_STRING);
			}
		} else {
//sprintf(ERROR_STRING,"%s IS in use",cname);
//advise(ERROR_STRING);
			pgcp = NULL;
		}
		i++;
		if( i>=5 ){
			ERROR1("Too many cameras!?"); 
		}
	}
	pgcp->pc_cam_info = camInfo;
	return pgcp;
}

static void get_fmt7_modes(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	fc2Error error;
	BOOL supported;
	int i, largest=(-1);
	fc2Format7Info fmt7_info_tbl[N_FMT7_MODES];
	size_t nb;

	pgcp->pc_n_fmt7_modes = 0;
	for(i=0;i<N_FMT7_MODES;i++){
		fmt7_info_tbl[i].mode = i;
		error = fc2GetFormat7Info(pgcp->pc_context,
				&fmt7_info_tbl[i],&supported);
		if( error != FC2_ERROR_OK ){
			report_fc2_error(QSP_ARG  error, "fc2GetFormat7Info" );
		}
		if( supported ){
			pgcp->pc_n_fmt7_modes ++ ;
			largest = i;
		}
	}
	// CAUTIOUS?
	if( (largest+1) != pgcp->pc_n_fmt7_modes ){
		sprintf(ERROR_STRING,
	"Unexpected number of format7 modes!?  (largest index = %d, n_modes = %d)",
			largest,pgcp->pc_n_fmt7_modes);
		WARN(ERROR_STRING);
	}

	nb = pgcp->pc_n_fmt7_modes * sizeof(fc2Format7Info);
	pgcp->pc_fmt7_info_tbl = getbuf( nb );
	memcpy(pgcp->pc_fmt7_info_tbl,fmt7_info_tbl,nb);

	pgcp->pc_fmt7_index = 0;
}

#define SHOW_FIELD(desc_str,value)					\
									\
sprintf(MSG_STR,"\t%s:  %d",#desc_str,pgcp->pc_fmt7_info_tbl[mode].value);	\
prt_msg(MSG_STR);

#define SHOW_FIELD_HEX(desc_str,value)					\
									\
sprintf(MSG_STR,"\t%s:  0x%x",#desc_str,pgcp->pc_fmt7_info_tbl[mode].value); \
prt_msg(MSG_STR);

static void show_fmt7_info(QSP_ARG_DECL  PGR_Cam *pgcp, fc2Mode mode )
{
	sprintf(MSG_STR,"Format 7 mode %d:",mode);
	prt_msg(MSG_STR);
	
	SHOW_FIELD(max width,maxWidth)
	SHOW_FIELD(max height,maxHeight)
	SHOW_FIELD(offsetHStepSize,offsetHStepSize)
	SHOW_FIELD(offsetVStepSize,offsetVStepSize)
	SHOW_FIELD(imageHStepSize,imageHStepSize)
	SHOW_FIELD(imageVStepSize,imageVStepSize)
	SHOW_FIELD_HEX(pixelFormatBitField,pixelFormatBitField)
	SHOW_FIELD_HEX(vendorPixelFormatBitField,vendorPixelFormatBitField)
	SHOW_FIELD(packetSize,packetSize)
	SHOW_FIELD(minPacketSize,minPacketSize)
	SHOW_FIELD(maxPacketSize,maxPacketSize)

	prt_msg("");
}

void show_fmt7_modes(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	if( pgcp->pc_n_fmt7_modes <= 0 ){
		prt_msg("\tNo format7 modes available.");
	} else {
		int i;
		for(i=0;i<pgcp->pc_n_fmt7_modes;i++)
			show_fmt7_info( QSP_ARG  pgcp, i );
	}
}

// This should only be called once...

static PGR_Cam *setup_my_camera( QSP_ARG_DECL
				fc2Context context, fc2PGRGuid *guid_p,
				int index )
{
	PGR_Cam *pgcp;
	fc2Error error;
	int i;

	error = fc2Connect(context,guid_p);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2Connect" );
		return NULL;
	}

	// We could have multiple instances of the same model...
	pgcp = unique_camera_instance(QSP_ARG  context);

	error = fc2Disconnect(context);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2Disconnect" );
		// BUG should clean up and return NULL?
		return pgcp;
	}

	//pgcp->pc_cam_p = cam_p;
	error = fc2CreateContext(&pgcp->pc_context);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2CreateContext" );
		// BUG clean up first
		return NULL;
	}
	error=fc2GetCameraFromIndex(pgcp->pc_context,index,&pgcp->pc_guid);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetCameraFromIndex" );
	}

	error = fc2Connect(pgcp->pc_context,&pgcp->pc_guid);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2Connect" );
		// BUG clean up first
		return NULL;
	}

	if( refresh_config(QSP_ARG  pgcp) < 0 )
		// BUG clean up first
		return NULL;

	error = fc2GetEmbeddedImageInfo(pgcp->pc_context,&pgcp->pc_ei_info);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetEmbeddedImageInfo" );
		// BUG clean up first
		return NULL;
	}

	pgcp->pc_feat_lp=NO_LIST;
	pgcp->pc_in_use_lp=NO_LIST;
	pgcp->pc_flags = 0;		/* assume no B-mode unless we are told otherwise... */
	pgcp->pc_base = NULL;

	get_fmt7_modes(QSP_ARG  pgcp);

	error = fc2GetVideoModeAndFrameRate( pgcp->pc_context,
			&pgcp->pc_video_mode, &pgcp->pc_framerate );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetVideoModeAndFrameRate" );
		// BUG clean up first
		return NULL;
	}
/*
fprintf(stderr,"after fetching, video mode = %d (%s), frame rate = %d (%s)\n",
pgcp->pc_video_mode,
name_for_video_mode(pgcp->pc_video_mode),
pgcp->pc_framerate,
name_for_framerate(pgcp->pc_framerate)
);
*/

	// BUG?  could be N_STD_VIDEO_MODES?
	pgcp->pc_video_mode_indices = getbuf( sizeof(int) * N_NAMED_VIDEO_MODES );
	pgcp->pc_video_mode_names = getbuf( sizeof(char *) * N_NAMED_VIDEO_MODES );
	pgcp->pc_framerate_mask_tbl = getbuf( sizeof(Framerate_Mask) * N_NAMED_VIDEO_MODES );
	pgcp->pc_framerate_names = NULL;
	pgcp->pc_n_framerates = 0;

	for(i=0;i<N_NAMED_VIDEO_MODES;i++){
		pgcp->pc_framerate_mask_tbl[i] = 0;
	}

	/* Originally, we set the default video mode here to be
	 * the highest standard video mode...  But for the flea3,
	 * which has an odd-shaped sensor, the good modes are format7
	 * and for no good reason when we change back to format7
	 * under program control the high frame rate is not restored???
	 */

//#ifdef NOT_GOOD
	if( set_default_video_mode(QSP_ARG  pgcp) < 0 ){
		/*
		sprintf(ERROR_STRING,"error setting default video mode for %s",pgcp->pc_name);
		WARN(ERROR_STRING);
		cleanup_cam( pgcp );
		return(NULL);
		*/
		// This can fail for cameras that only support format7...
		// Deal with this better later
		sprintf(ERROR_STRING,"error setting default video mode for %s, only format7?",
			pgcp->pc_name);
		advise(ERROR_STRING);
	}
//#endif // NOT_GOOD

	/* used to set B-mode stuff here... */
	// What if the camera is a usb cam???
	//dc1394_video_set_iso_speed( cam_p, DC1394_ISO_SPEED_400 );

	pgcp->pc_img_p = getbuf( sizeof(*pgcp->pc_img_p) );

        error = fc2CreateImage( pgcp->pc_img_p );
        if ( error != FC2_ERROR_OK ) {
		report_fc2_error(QSP_ARG  error, "fc2CreateImage" );
		// BUG clean up?
		//return NULL;
	}


	// Make a data_obj context for the frames...
	pgcp->pc_do_icp = create_dobj_context( QSP_ARG  pgcp->pc_name );

	pgcp->pc_frm_dp_tbl = NULL;
	pgcp->pc_newest = (-1);

	return(pgcp);
}

void pop_camera_context(SINGLE_QSP_ARG_DECL)
{
	// pop old context...
	Item_Context *icp;
	icp=pop_dobj_context(SINGLE_QSP_ARG);
//#ifdef CAUTIOUS
//	if( icp == NO_ITEM_CONTEXT ){
//		ERROR1("CAUTIOUS:  pop_camera_context popped a null dobj context!?");
//	}
//#endif // CAUTIOUS
	assert( icp != NO_ITEM_CONTEXT );
}

void push_camera_context(QSP_ARG_DECL  PGR_Cam *pgcp)
{
//fprintf(stderr,"pushing camera context for %s (icp = 0x%lx)\n",
//pgcp->pc_name,(long)pgcp->pc_do_icp);
	push_dobj_context(QSP_ARG  pgcp->pc_do_icp);
}

int init_firewire_system(SINGLE_QSP_ARG_DECL)
{
#ifdef HAVE_LIBFLYCAP
	fc2Version version;
	fc2Context context;
	fc2Error error;
	fc2PGRGuid guid;	// BUG should be associated with one camera?
	unsigned int numCameras=0;
	int i;
	static int firewire_system_inited=0;

	if( firewire_system_inited ){
		WARN("Firewire system has already been initialized!?");
		return -1;
	}
	firewire_system_inited=1;
	init_property_types(SINGLE_QSP_ARG);

	fc2GetLibraryVersion(&version);
	sprintf(ERROR_STRING,"FlyCapture2 library version:  %d.%d.%d.%d",
		version.major,version.minor,version.type,version.build);
	advise(ERROR_STRING);

	// BUG?  the call to fc2CreateContext hangs if one is not logged in
	// on the console...  You don't need to RUN from the console,
	// but apparently something gets owned?

	error = fc2CreateContext(&context);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2CreateContext" );
		return -1;
	}

	error = fc2GetNumOfCameras(context,&numCameras);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetNumOfCameras" );
		// BUG? should we destroy the context here?
		return -1;
	}

	if( numCameras == 0 ){
		advise("No cameras detected.");
		return 0;
	}
	sprintf(ERROR_STRING,
		"%d camera%s found.", numCameras, numCameras==1?"":"s" );
	advise(ERROR_STRING);


	//for(i=0;i<numCameras;i++)
	for(i=numCameras-1;i>=0;i--){
		error=fc2GetCameraFromIndex(context,i,&guid);
		if( error != FC2_ERROR_OK ){
			report_fc2_error(QSP_ARG  error, "fc2GetCameraFromIndex" );
		} else {
			setup_my_camera(QSP_ARG   context, &guid, i );
		}
	}
	error = fc2DestroyContext( context );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2DestroyContext" );
	}

	add_sizable(QSP_ARG  pgc_itp, &pgc_sf, (Item *(*)(QSP_ARG_DECL  const char *)) pgc_of );

#endif // HAVE_LIBFLYCAP
	
	return 0;
}

#ifdef HAVE_LIBFLYCAP
static void show_cam_info(QSP_ARG_DECL  fc2CameraInfo *cip)
{
	sprintf(MSG_STR,
                "\n*** CAMERA INFORMATION ***\n"
                "Serial number - %u\n"
                "Camera model - %s (%s)\n"
                "Camera vendor - %s\n"
		"Interface - %s\n"
		"Bus/Node - %d/%d\n"
                "Sensor - %s\n"
                "Resolution - %s\n"
                "Firmware version - %s\n"
                "Firmware build time - %s\n",
                cip->serialNumber,
                cip->modelName,
		cip->isColorCamera ? "color" : "monochrome",
                cip->vendorName,
		name_for_interface(cip->interfaceType),
		cip->busNumber,cip->nodeNumber,
                cip->sensorInfo,
                cip->sensorResolution,
                cip->firmwareVersion,
                cip->firmwareBuildTime );
	prt_msg(MSG_STR);
}

void show_n_buffers(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	sprintf(MSG_STR,"%s:  %d buffers",pgcp->pc_name,pgcp->pc_n_buffers);
	prt_msg(MSG_STR);
}

int set_n_buffers(QSP_ARG_DECL  PGR_Cam *pgcp, int n )
{
	fc2Config cfg;
	fc2Error error;

fprintf(stderr,"set_n_buffers %s %d\n",pgcp->pc_name,n);
	if( n < MIN_N_BUFFERS || n > MAX_N_BUFFERS ){
		sprintf(ERROR_STRING,
"set_n_buffers:  number of buffers must be between %d and %d (%d requested)!?",
			MIN_N_BUFFERS,MAX_N_BUFFERS,n);
		WARN(ERROR_STRING);
		return -1;
	}
	cfg = pgcp->pc_config;
	cfg.numBuffers = n;

	error = fc2SetConfiguration(pgcp->pc_context,&cfg);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetConfiguration" );
		// should we set a flag to indicate an invalid config?
		return -1;
	}
	pgcp->pc_n_buffers =
	pgcp->pc_config.numBuffers = n;
	pgcp->pc_base = NULL;	// force init_fly_base to run again

show_n_buffers(QSP_ARG  pgcp);
	return 0;
}

static void show_cam_cfg(QSP_ARG_DECL  fc2Config *cfp)
{
	prt_msg("*** Configuration ***");
	sprintf(MSG_STR,
		"number of buffers:  %d\n"
		"number of image notifications:  %d\n"
		"min. num. image notifications:  %d\n"
		"grab timeout:  %d\n"
		"grab mode:  %s\n"
		"iso bus speed:  %s\n"
		"async bus speed:  %s\n"
		"bandwidth allocation:  %s\n"
		"register timeout retries:  %d\n"
		"register timeout:  %d\n",
		cfp->numBuffers,
		cfp->numImageNotifications,
		cfp->minNumImageNotifications,
		cfp->grabTimeout,
		name_for_grab_mode(cfp->grabMode),
		name_for_bus_speed(cfp->isochBusSpeed),
		name_for_bus_speed(cfp->asyncBusSpeed),
		name_for_bw_allocation(cfp->bandwidthAllocation),
		cfp->registerTimeoutRetries,
		cfp->registerTimeout);
	prt_msg(MSG_STR);
}

#define SHOW_EI(member)					\
							\
	if( eip->member.available ){			\
		sprintf(MSG_STR,"%s:  %s",#member,	\
		eip->member.onOff ? "on" : "off" );	\
		prt_msg(MSG_STR);			\
	}

static void show_ei_info(QSP_ARG_DECL  fc2EmbeddedImageInfo *eip)
{
	prt_msg("\n*** EMBEDDED IMAGE INFORMATION ***");
	SHOW_EI(timestamp)
	SHOW_EI(gain)
	SHOW_EI(shutter)
	SHOW_EI(brightness)
	SHOW_EI(exposure)
	SHOW_EI(whiteBalance)
	SHOW_EI(frameCounter)
	SHOW_EI(strobePattern)
	SHOW_EI(GPIOPinState)
	SHOW_EI(ROIPosition)
}

#endif // HAVE_LIBFLYCAP

void print_camera_info(QSP_ARG_DECL  PGR_Cam *pgcp)
{
#ifdef HAVE_LIBFLYCAP
	show_cam_info(QSP_ARG  &pgcp->pc_cam_info);
	show_cam_cfg(QSP_ARG  &pgcp->pc_config);
	show_ei_info(QSP_ARG  &pgcp->pc_ei_info);
#endif // HAVE_LIBFLYCAP

	/*
	i=index_of_framerate(pgcp->pc_framerate);
	sprintf(msg_str,"\tframe rate:  %s",all_framerates[i].nfr_name);
	prt_msg(msg_str);
	*/

	//report_camera_features(pgcp);

	// show_fmt7_modes(QSP_ARG  pgcp);
	// Show the current video mode

	sprintf(MSG_STR,"Current video mode:  %s%s",
		pgcp->pc_my_video_mode_index >= 0 ?
		pgcp->pc_video_mode_names[ pgcp->pc_my_video_mode_index ] :
		"format7 mode ",
		pgcp->pc_my_video_mode_index >= 0 ? "": (
			pgcp->pc_fmt7_index == 0 ? "0" : (
			pgcp->pc_fmt7_index == 1 ? "1" : (
			pgcp->pc_fmt7_index == 2 ? "2" : "(>2)" )))
			);
	prt_msg(MSG_STR);

	sprintf(MSG_STR,"Current frame rate:  %s",
		all_framerates[ pgcp->pc_framerate_index ].nfr_name );
	prt_msg(MSG_STR);
}

static void init_one_frame(QSP_ARG_DECL  PGR_Cam *pgcp, int index )
{
	Data_Obj *dp;
	char fname[32];
	Dimension_Set ds1;

	sprintf(fname,"frame%d",index);
	//ASSIGN_VAR("newest",fname+5);

	dp = dobj_of(QSP_ARG  fname);
	if( dp == NO_OBJ ){
		SET_DS_SEQS(&ds1,1);
		SET_DS_FRAMES(&ds1,1);
		SET_DS_ROWS(&ds1,pgcp->pc_img_p->rows);
		SET_DS_COLS(&ds1,pgcp->pc_img_p->cols);
		SET_DS_COMPS(&ds1,1);
		dp = _make_dp(QSP_ARG  fname,&ds1,PREC_FOR_CODE(PREC_UBY));
//#ifdef CAUTIOUS
//		if( dp == NO_OBJ ){
//			sprintf(ERROR_STRING,
//	"CAUTIOUS:  grab_firewire_frame:  Error creating object %s!?",
//				fname);
//			WARN(ERROR_STRING);
//		} else
//#endif // CAUTIOUS
//	       		{
			//SET_OBJ_DATA_PTR(dp,pgcp->pc_img_p->pData);

		assert( dp != NO_OBJ );

		SET_OBJ_DATA_PTR( dp, pgcp->pc_base+index*pgcp->pc_buf_delta );
		pgcp->pc_frm_dp_tbl[index] = dp;

//fprintf(stderr,"init_one_frame %d:  %s, data at 0x%lx\n",index,OBJ_NAME(dp),(long)OBJ_DATA_PTR(dp));
//		}
	} else {
		// CAUTIOUS??
		sprintf(ERROR_STRING,"init_one_frame:  object %s already exists!?",
			fname);
		WARN(ERROR_STRING);
	}
} // end init_one_frame

static void init_cam_frames(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	int index;

//#ifdef CAUTIOUS
//	if( pgcp->pc_n_buffers <= 0 ){
//		sprintf(ERROR_STRING,
//	"CAUTIOUS:  init_cam_frames:  bad n_buffers (%d)!?",pgcp->pc_n_buffers);
//		WARN(ERROR_STRING);
//		return;
//	}
	assert( pgcp->pc_n_buffers > 0 );

//	if( pgcp->pc_frm_dp_tbl != NULL ){
//		sprintf(ERROR_STRING,
//	"CAUTIOUS:  init_cam_frames:  frame table is not NULL!?");
//		WARN(ERROR_STRING);
//		return;
//	}
//#endif // CAUTIOUS
	assert( pgcp->pc_frm_dp_tbl == NULL );

	pgcp->pc_frm_dp_tbl = getbuf( sizeof(Data_Obj) * pgcp->pc_n_buffers );
	for(index=0;index<pgcp->pc_n_buffers;index++)
		init_one_frame(QSP_ARG  pgcp, index);
} // init_cam_frames

// init_fly_base   -   grab frames to get the address
// associated with each frame index.  This wouldn't be necessary
// if we always provided the buffers, but we want this to work
// even when we don't.
//
// We keep track of the largest and smallest address, we save
// those so we can figure out the index of an arbitrary frame...

static void init_fly_base(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	// initialize all the pointers
	int n_buffers_seen=0;
	void *smallest_addr;
	void *largest_addr;
	void *first_addr=NULL;
	void **addr_tbl;

	// pc_base is our flag, reset to NULL when number of buffers
	// is changed, or video mode is changed.
	if( pgcp->pc_base != NULL ){
		return;
	}

	n_buffers_seen = 0;
	addr_tbl = getbuf(sizeof(void *)*pgcp->pc_n_buffers);

	// silence compiler warnings
	largest_addr = NULL;
	smallest_addr = NULL;

	while( n_buffers_seen < pgcp->pc_n_buffers ){
		fc2Error error;
		void *buf_addr;

		error = fc2RetrieveBuffer( pgcp->pc_context, pgcp->pc_img_p );
		if( error != FC2_ERROR_OK ){
			report_fc2_error(QSP_ARG  error, "fc2RetrieveBuffer" );
			return;
		}
/*
sprintf(ERROR_STRING,"pData = 0x%lx",(long)pgcp->pc_img_p->pData);
advise(ERROR_STRING);
*/
		buf_addr = pgcp->pc_img_p->pData;

		if( first_addr == NULL ){
			first_addr = buf_addr;
			smallest_addr = buf_addr;
			largest_addr = buf_addr;
			addr_tbl[n_buffers_seen] = buf_addr;
			n_buffers_seen=1;
		} else {
			int i;
			int new_addr=1;
			for(i=0;i<n_buffers_seen;i++){
				if( buf_addr == addr_tbl[i] )
					new_addr=0;
			}
			if( new_addr ){
				if( buf_addr > largest_addr ) largest_addr = buf_addr;
				if( buf_addr < smallest_addr ) smallest_addr = buf_addr;
				addr_tbl[n_buffers_seen] = buf_addr;
				n_buffers_seen++;
			}
		}
	}

	pgcp->pc_base = smallest_addr;
	pgcp->pc_buf_delta = (largest_addr - smallest_addr) / (n_buffers_seen-1);
	//pgcp->pc_buf_delta = (largest - smallest) / 30;

	if( verbose ){
		sprintf(ERROR_STRING,"%d distinct buffers seen.",
			n_buffers_seen);
		advise(ERROR_STRING);
		sprintf(ERROR_STRING,"largest addr = 0x%lx",
			(long)largest_addr);
		advise(ERROR_STRING);
		sprintf(ERROR_STRING,"smallest addr = 0x%lx",
			(long)smallest_addr);
		advise(ERROR_STRING);

		sprintf(ERROR_STRING,"buf_delta = 0x%lx",
			(long)pgcp->pc_buf_delta);
		advise(ERROR_STRING);
	}

	init_cam_frames(QSP_ARG  pgcp);
}

int check_buffer_alignment(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	int i;

	// alignment requirement is now 1024
	// BUG this should be a parameter...
#define RV_ALIGNMENT_REQ	1024

	for(i=0;i<pgcp->pc_n_buffers;i++){
		if( ((long)(pgcp->pc_base+i*pgcp->pc_buf_delta)) % RV_ALIGNMENT_REQ != 0 ){
			sprintf(ERROR_STRING,"Buffer %d is not aligned - %d byte alignment required for raw volume I/O!?",
				i,RV_ALIGNMENT_REQ);
			WARN(ERROR_STRING);
			return -1;
		}
	}
	return 0;
}

static int index_of_buffer(QSP_ARG_DECL  PGR_Cam *pgcp,fc2Image *ip)
{
	int idx;

	idx = ( ip->pData - pgcp->pc_base ) / pgcp->pc_buf_delta;
	/*
sprintf(ERROR_STRING,
"index_of_buffer:  data at 0x%lx, base = 0x%lx, idx = %d",
(long)ip->pData,(long)pgcp->pc_base,idx);
advise(ERROR_STRING);
*/

//#ifdef CAUTIOUS
//	if( idx < 0 || idx >= pgcp->pc_n_buffers ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  index_of_buffer:  computed index %d is out of range (0-%d)!?",
//			idx,pgcp->pc_n_buffers-1);
//		WARN(ERROR_STRING);
//		idx=0;
//	}
//#endif // CAUTIOUS
	assert( idx >= 0 && idx < pgcp->pc_n_buffers );
	return idx;
}

#ifdef NOT_USED
static const char *name_for_pixel_format(fc2PixelFormat f)
{
	int i;

	for(i=0;i<N_NAMED_PIXEL_FORMATS;i++){
		if( all_pixel_formats[i].npf_value == f )
			return( all_pixel_formats[i].npf_name );
	}
	return("(unrecognixed pixel format code)");
}
#endif // NOT_USED

// libflycap doesn't have a queue mechanism like libdc1394...
// do we get the newest frame???

Data_Obj * grab_firewire_frame(QSP_ARG_DECL  PGR_Cam * pgcp )
{
	fc2Error error;
	int index;

	if( pgcp->pc_base == NULL )
		init_fly_base(QSP_ARG  pgcp);

	error = fc2RetrieveBuffer( pgcp->pc_context, pgcp->pc_img_p );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2RetrieveBuffer" );
		return NULL;
	}
//fprintf(stderr,"pixel format of retrieved images is %s (0x%x)\n",
//name_for_pixel_format(img.format),img.format);

	index = index_of_buffer(QSP_ARG  pgcp, pgcp->pc_img_p );
	pgcp->pc_newest = index;

	return( pgcp->pc_frm_dp_tbl[index] );
}

int reset_camera(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	fc2Error error;

	error=fc2FireBusReset(pgcp->pc_context,&pgcp->pc_guid);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2FireBusReset" );
	}

	return 0;
}

void report_bandwidth(QSP_ARG_DECL  PGR_Cam *pgcp )
{
	UNIMP_FUNC("report_bandwidth");
}

unsigned int read_register( QSP_ARG_DECL  PGR_Cam *pgcp, unsigned int addr )
{
	fc2Error error;
	unsigned int val;

	error = fc2ReadRegister(pgcp->pc_context,addr,&val);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2ReadRegister" );
	}
	return val;
}

void write_register( QSP_ARG_DECL  PGR_Cam *pgcp, unsigned int addr, unsigned int val )
{
	fc2Error error;

	error = fc2WriteRegister(pgcp->pc_context,addr,val);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2WriteRegister" );
	}
}

void start_firewire_capture(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	fc2Error error;

	if( pgcp->pc_flags & PGR_CAM_IS_RUNNING ){
		WARN("start_firewire_capture:  camera is already capturing!?");
		return;
	}

	error = fc2StartCapture(pgcp->pc_context);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2StartCapture" );
	} else {
		pgcp->pc_flags |= PGR_CAM_IS_RUNNING;

		// BUG - we should undo this when we stop capturing, because
		// we might change the video format or something else.
		// Perhaps more efficiently we could only do it when needed?
		init_fly_base(QSP_ARG  pgcp);
	}
}

void stop_firewire_capture(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	fc2Error error;

	if( (pgcp->pc_flags & PGR_CAM_IS_RUNNING) == 0 ){
		WARN("stop_firewire_capture:  camera is not capturing!?");
		return;
	}

	error = fc2StopCapture(pgcp->pc_context);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2StopCapture" );
	} else {
		pgcp->pc_flags &= ~PGR_CAM_IS_RUNNING;
	}
}

void set_fmt7_size(QSP_ARG_DECL  PGR_Cam *pgcp, int w, int h)
{
	UNIMP_FUNC("set_fmt7_size");
}

void release_oldest_frame(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	UNIMP_FUNC("release_oldest_frame");
}

void list_trig(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	fc2Error error;
	fc2TriggerModeInfo tinfo;
	fc2TriggerMode tmode;

	error = fc2GetTriggerModeInfo(pgcp->pc_context,&tinfo);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetTriggerModeInfo" );
		return;
	}
	fprintf(stderr,"Trigger mode info:\n");
#define BOOL_STR(v)	(v?"true":"false")
	fprintf(stderr,"\tpresent:  %s\n",BOOL_STR(tinfo.present));
	fprintf(stderr,"\treadOutSupported:  %s\n",BOOL_STR(tinfo.readOutSupported));
	fprintf(stderr,"\tonOffSupported:  %s\n",BOOL_STR(tinfo.onOffSupported));
	fprintf(stderr,"\tpolaritySupported:  %s\n",BOOL_STR(tinfo.polaritySupported));
	fprintf(stderr,"\tvalueReadable:  %s\n",BOOL_STR(tinfo.valueReadable));
	fprintf(stderr,"\tsourceMask:  0x%x\n",tinfo.sourceMask);
	fprintf(stderr,"\tsoftwareTriggerSupported:  %s\n",BOOL_STR(tinfo.softwareTriggerSupported));
	fprintf(stderr,"\tmodeMask:  0x%x\n",tinfo.modeMask);

	error = fc2GetTriggerMode(pgcp->pc_context,&tmode);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetTriggerMode" );
		return;
	}
	fprintf(stderr,"Trigger mode:\n");
#define ONOFF_STR(v)	(v?"on":"off")
	fprintf(stderr,"\tonOff:  %s\n",ONOFF_STR(tmode.onOff));
#define SHOW_INT_PARAM(p) fprintf(stderr,"\t%s:  %d (0x%x)\n",\
	#p,tmode.p,tmode.p);
	SHOW_INT_PARAM(polarity)
	SHOW_INT_PARAM(source)
	SHOW_INT_PARAM(mode)
	SHOW_INT_PARAM(parameter)
}

void set_buffer_obj(QSP_ARG_DECL  PGR_Cam *pgcp, Data_Obj *dp)
{
	// make sure sizes match
	if( OBJ_COLS(dp) != pgcp->pc_cols || OBJ_ROWS(dp) != pgcp->pc_rows ){
		sprintf(ERROR_STRING,
"set_buffer_obj:  size mismatch between %s (%dx%d) and object %s (%dx%d)",
			pgcp->pc_name,pgcp->pc_cols,pgcp->pc_rows,
			OBJ_NAME(dp),OBJ_COLS(dp),OBJ_ROWS(dp) );
		WARN(ERROR_STRING);
		return;
	}
	if( PREC_CODE(OBJ_MACH_PREC_PTR(dp)) != PREC_UBY ){
		sprintf(ERROR_STRING,"Object %s (%s) should have %s precision!?",
			OBJ_NAME(dp),OBJ_PREC_NAME(dp),NAME_FOR_PREC_CODE(PREC_UBY));
		WARN(ERROR_STRING);
		return;
	}
	{
		fc2Error error;

		error = fc2SetUserBuffers(pgcp->pc_context, OBJ_DATA_PTR(dp),
				OBJ_COLS(dp)*OBJ_ROWS(dp)*OBJ_COMPS(dp),OBJ_FRAMES(dp));
		if( error != FC2_ERROR_OK ){
			report_fc2_error(QSP_ARG  error, "fc2SetUserBuffers" );
			return;
		}
		// refresh the configuration
		refresh_config(QSP_ARG  pgcp);
	}
	pgcp->pc_base = NULL;	// force init_fly_base to run again
}

#endif /* HAVE_LIBFLYCAP */


static const char **grab_mode_names=NULL;

int pick_grab_mode(QSP_ARG_DECL PGR_Cam *pgcp, const char *pmpt)
{
	int idx;

	if( pgcp == NULL ){
		sprintf(ERROR_STRING,"pick_framerate:  no camera selected!?");
		WARN(ERROR_STRING);
		return -1;
	}

	if( grab_mode_names == NULL ){
		grab_mode_names = getbuf( sizeof(char *) * N_NAMED_GRAB_MODES );
		for(idx=0;idx<N_NAMED_GRAB_MODES;idx++)
			grab_mode_names[idx] = all_grab_modes[idx].ngm_name;
	}

	idx=WHICH_ONE(pmpt,N_NAMED_GRAB_MODES,grab_mode_names);
	return idx;
}

