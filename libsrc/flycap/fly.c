/* Jeff's interface to the 1394 subsystem usign PGR's libflycap_c */

#include "quip_config.h"

#include "quip_prot.h"
#include "function.h"
#include "data_obj.h"

#include <stdio.h>
#include <string.h>

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

ITEM_INTERFACE_DECLARATIONS(Fly_Cam,fly_cam,0)

#define UNIMP_FUNC(name)						\
	sprintf(ERROR_STRING,"Function %s is not implemented!?",name);	\
	warn(ERROR_STRING);

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

static double get_fly_cam_size(QSP_ARG_DECL  Item *ip, int dim_index)
{
	switch(dim_index){
		case 0:	return(1.0); /* BUG - not correct for color fly_cams! */ break;
		case 1: return(((Fly_Cam *)ip)->fc_cols);
		case 2: return(((Fly_Cam *)ip)->fc_rows);
		case 3: return(((Fly_Cam *)ip)->fc_n_buffers);
		case 4: return(1.0);
		default:
			// should never happen
			assert(0);
			break;
	}
	return(0.0);
}

static const char * get_fly_cam_prec_name(QSP_ARG_DECL  Item *ip )
{
	//Fly_Cam *fcp;

	//fcp = (Fly_Cam *)ip;

	warn("get_fly_cam_prec_name:  need to implement fly_cam-state-based value!?");

	//return def_prec_name(QSP_ARG  ip);
	return("u_byte");
}


static Size_Functions fly_cam_sf={
	get_fly_cam_size,
	get_fly_cam_prec_name

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
			msg = "Setting set to fly_cam is invalid."; break;
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
			msg = "Packet size set to fly_cam is invalid."; break;
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
			warn(ERROR_STRING);
			msg = "unhandled error code";
			break;
	}
	sprintf(ERROR_STRING,"%s:  %s",whence,msg);
	warn(ERROR_STRING);
}

ITEM_INTERFACE_DECLARATIONS(Fly_Cam_Property_Type,pgr_prop,0)

//  When we change fly_cams, we have to refresh all properties!

static void _init_one_property(QSP_ARG_DECL const char *name, fc2PropertyType t)
{
	Fly_Cam_Property_Type *pgpt;

	pgpt = new_pgr_prop(name);
	if( pgpt == NULL ) return;
	pgpt->info.type =
	pgpt->prop.type =
	pgpt->type_code = t;
}

void list_fly_cam_properties(QSP_ARG_DECL  Fly_Cam *fcp)
{
	List *lp;
	Node *np;
	Fly_Cam_Property_Type *pgpt;

	lp = pgr_prop_list();	// all properties
	np = QLIST_HEAD(lp);
	if( np != NULL ){
		sprintf(MSG_STR,"\n%s properties",fcp->fc_name);
		prt_msg(MSG_STR);
	} else {
		sprintf(ERROR_STRING,"%s has no properties!?",fcp->fc_name);
		warn(ERROR_STRING);
		return;
	}

	while(np!=NULL){
		pgpt = (Fly_Cam_Property_Type *)NODE_DATA(np);
		if( pgpt->info.present ){
			sprintf(MSG_STR,"\t%s",pgpt->name);
			prt_msg(MSG_STR);
		}
		np = NODE_NEXT(np);
	}
	prt_msg("");
}

// We call this after we select a fly_cam

void refresh_fly_cam_properties(QSP_ARG_DECL  Fly_Cam *fcp)
{
	List *lp;
	Node *np;
	Fly_Cam_Property_Type *pgpt;

	lp = pgr_prop_list();	// all properties
	np = QLIST_HEAD(lp);
	while(np!=NULL){
		pgpt = (Fly_Cam_Property_Type *)NODE_DATA(np);
		refresh_property_info(QSP_ARG  fcp, pgpt );
		if( pgpt->info.present ){
			refresh_property_value(QSP_ARG  fcp, pgpt );
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

void refresh_property_info(QSP_ARG_DECL  Fly_Cam *fcp, Fly_Cam_Property_Type *pgpt )
{
	fc2Error error;

	error = fc2GetPropertyInfo( fcp->fc_context, &(pgpt->info) );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetPropertyInfo" );
		return;
	}
}

void show_property_info(QSP_ARG_DECL  Fly_Cam *fcp, Fly_Cam_Property_Type *pgpt )
{
	char var_name[32],val_str[32];

	sprintf(MSG_STR,"\n%s %s info:",fcp->fc_name,pgpt->name);
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
		assign_var(var_name,val_str);

		sprintf(var_name,"%s_abs_max",pgpt->name);	// BUG possible buffer overrun, use snprintf or whatever...
		sprintf(val_str,"%g",pgpt->info.absMax);
		assign_var(var_name,val_str);
	} else {
		sprintf(MSG_STR,"\tRange:  %d - %d",
			pgpt->info.min,pgpt->info.max);
		prt_msg(MSG_STR);

		sprintf(var_name,"%s_abs_min",pgpt->name);	// BUG possible buffer overrun, use snprintf or whatever...
		assign_var(var_name,"(undefined)");

		sprintf(var_name,"%s_abs_max",pgpt->name);	// BUG possible buffer overrun, use snprintf or whatever...
		assign_var(var_name,"(undefined)");
	}

	sprintf(var_name,"%s_min",pgpt->name);	// BUG possible buffer overrun, use snprintf or whatever...
	sprintf(val_str,"%d",pgpt->info.min);
	assign_var(var_name,val_str);

	sprintf(var_name,"%s_max",pgpt->name);	// BUG possible buffer overrun, use snprintf or whatever...
	sprintf(val_str,"%d",pgpt->info.max);
	assign_var(var_name,val_str);

	sprintf(MSG_STR,"\tUnits:  %s (%s)",pgpt->info.pUnits,pgpt->info.pUnitAbbr);
	prt_msg(MSG_STR);
}

void refresh_property_value(QSP_ARG_DECL  Fly_Cam *fcp, Fly_Cam_Property_Type *pgpt )
{
	fc2Error error;

	error = fc2GetProperty( fcp->fc_context, &(pgpt->prop) );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetProperty" );
		return;
	}
}

void show_property_value(QSP_ARG_DECL  Fly_Cam *fcp, Fly_Cam_Property_Type *pgpt )
{
	sprintf(MSG_STR,"\n%s %s:",
		fcp->fc_name,pgpt->name);
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
	// the value back from the fly_cam???
	if( pgpt->info.readOutSupported ){
		// Now print out the property value itself!
		// Can we see both???

		sprintf(MSG_STR,"\t%s:  %d (integer)",
			pgpt->name,pgpt->prop.valueA);
		prt_msg(MSG_STR);

		// let a script access the value also
		sprintf(MSG_STR,"%d",pgpt->prop.valueA);
		assign_var(pgpt->name,MSG_STR);
		// should this be a reserved var?  I think so!

		if( pgpt->info.absValSupported ){
			sprintf(MSG_STR,"\t%s:  %g %s (absolute)",
				pgpt->name,pgpt->prop.absValue,pgpt->info.pUnitAbbr);
			prt_msg(MSG_STR);

			// let a script access the value also
			sprintf(MSG_STR,"%g",pgpt->prop.absValue);
			sprintf(ERROR_STRING,"%s_abs",pgpt->name);	// using ERROR_STRING as a temporary...
			assign_var(ERROR_STRING,MSG_STR);
			// should this be a reserved var?  I think so!
		} else {
			sprintf(ERROR_STRING,"%s_abs",pgpt->name);	// using ERROR_STRING as a temporary...
			assign_var(ERROR_STRING,"(undefined)");
		}
	} else {
		prt_msg("\t(Readout not supported)");
		sprintf(ERROR_STRING,"%s",pgpt->name);
		assign_var(ERROR_STRING,"(undefined)");
		sprintf(ERROR_STRING,"%s_abs",pgpt->name);
		assign_var(ERROR_STRING,"(undefined)");
	}
}

void set_prop_value(QSP_ARG_DECL  Fly_Cam *fcp, Fly_Cam_Property_Type *pgpt, Fly_Cam_Prop_Val *vp )
{
	fc2Error error;

	if( vp->pv_is_abs ){
		if( vp->pv_u.u_f < pgpt->info.absMin || vp->pv_u.u_f > pgpt->info.absMax ){
			sprintf(ERROR_STRING,"Requested %s (%f) out of range (%f - %f)",
				pgpt->name,
				vp->pv_u.u_f,pgpt->info.absMin,pgpt->info.absMax);
			warn(ERROR_STRING);
			return;
		}
		pgpt->prop.absControl = TRUE;
		pgpt->prop.absValue = vp->pv_u.u_f;
	} else {
		if( vp->pv_u.u_i < pgpt->info.min || vp->pv_u.u_i > pgpt->info.max ){
			sprintf(ERROR_STRING,"Requested %s (%d) out of range (%d - %d)",
				pgpt->name,
				vp->pv_u.u_i,pgpt->info.min,pgpt->info.max);
			warn(ERROR_STRING);
			return;
		}
		pgpt->prop.absControl = FALSE;
		pgpt->prop.valueA = vp->pv_u.u_i;
	}

	error = fc2SetProperty( fcp->fc_context, &(pgpt->prop));
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetProperty" );
		return;
	}
}

void set_prop_auto(QSP_ARG_DECL  Fly_Cam *fcp, Fly_Cam_Property_Type *pgpt, BOOL yn )
{
	fc2Error error;

	if( ! pgpt->info.autoSupported ){
		sprintf(ERROR_STRING,"Sorry, auto mode not supported for %s.",
			pgpt->name);
		warn(ERROR_STRING);
		return;
	}

	pgpt->prop.autoManualMode = yn;
	error = fc2SetProperty( fcp->fc_context, &(pgpt->prop) );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetProperty" );
		return;
	}
}

static void insure_stopped(QSP_ARG_DECL  Fly_Cam *fcp, const char *op_desc)
{
	if( (fcp->fc_flags & FLY_CAM_IS_RUNNING) == 0 ) return;

	sprintf(ERROR_STRING,"Stopping capture on %s prior to %s",
		fcp->fc_name,op_desc);
	advise(ERROR_STRING);

	stop_firewire_capture(QSP_ARG  fcp);
}

void
cleanup_fly_cam( Fly_Cam *fcp )
{
	//if( IS_CAPTURING(fcp) )
		 //dc1394_capture_stop( fcp->fc_cam_p );
		 //fly_capture_stop( fcp->fc_cam_p );
	//if( IS_TRANSMITTING(fcp) )
		//dc1394_video_set_transmission( fcp->fc_cam_p, DC1394_OFF );
		//fly_video_set_transmission( fcp->fc_cam_p, DC1394_OFF );
	/* dc1394_free_fly_cam */
	//dc1394_fly_cam_free( fcp->fc_cam_p );
	//fly_fly_cam_free( fcp->fc_cam_p );
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

int get_fly_cam_names( QSP_ARG_DECL  Data_Obj *str_dp )
{
	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?
	List *lp;
	Node *np;
	Fly_Cam *fcp;
	int i, n;

	lp = fly_cam_list();
	if( lp == NULL ){
		warn("No fly_cams!?");
		return 0;
	}

	n=eltcount(lp);
	if( OBJ_COLS(str_dp) < n ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d fly_cam names",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),n);
		warn(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	}
		
	np=QLIST_HEAD(lp);
	i=0;
	while(np!=NULL){
		char *dst;
		fcp = (Fly_Cam *) NODE_DATA(np);
		dst = OBJ_DATA_PTR(str_dp);
		dst += i * OBJ_PXL_INC(str_dp);
		if( strlen(fcp->fc_name)+1 > OBJ_COMPS(str_dp) ){
			sprintf(ERROR_STRING,"String object %s has too few components (%ld) to hold fly_cam name \"%s\"",
				OBJ_NAME(str_dp),(long)OBJ_COMPS(str_dp),fcp->fc_name);
			warn(ERROR_STRING);
		} else {
			strcpy(dst,fcp->fc_name);
		}
		i++;
		if( i>=n )
			np=NULL;
		else
			np = NODE_NEXT(np);
	}

	return i;
}

int get_fly_cam_video_mode_strings( QSP_ARG_DECL  Data_Obj *str_dp, Fly_Cam *fcp )
{
	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?

	int i, n;

	if( OBJ_COLS(str_dp) < fcp->fc_n_video_modes ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d modes",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),fcp->fc_n_video_modes);
		warn(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	} else {
		n=fcp->fc_n_video_modes;
	}
		
	for(i=0;i<n;i++){
		int k;
		const char *src;
		char *dst;

		k=fcp->fc_video_mode_indices[i];
		src = all_video_modes[k].nvm_name;
		dst = OBJ_DATA_PTR(str_dp);
		dst += i * OBJ_PXL_INC(str_dp);
		if( strlen(src)+1 > OBJ_COMPS(str_dp) ){
			sprintf(ERROR_STRING,"String object %s has too few components (%ld) to hold mode string \"%s\"",
				OBJ_NAME(str_dp),(long)OBJ_COMPS(str_dp),src);
			warn(ERROR_STRING);
		} else {
			strcpy(dst,src);
		}
	}
	set_script_var_from_int("n_video_modes",n);
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

static void get_framerate_choices(QSP_ARG_DECL  Fly_Cam *fcp)
{
	unsigned int i,n,idx;
	Framerate_Mask mask;

	if( fcp->fc_framerate_names != NULL ){
		givbuf(fcp->fc_framerate_names);
		fcp->fc_framerate_names=NULL;	// in case we have an error before finishing
	}

	// format7 doesn't have a framerate!?

	mask = fcp->fc_framerate_mask_tbl[ fcp->fc_my_video_mode_index ];
/*
sprintf(ERROR_STRING,"%s:  video mode is %s",
fcp->fc_name,all_video_modes[fcp->fc_video_mode_index].nvm_name);
advise(ERROR_STRING);
sprintf(ERROR_STRING,"%s:  my video mode %s (index = %d)",
fcp->fc_name,fcp->fc_video_mode_names[fcp->fc_my_video_mode_index],
fcp->fc_my_video_mode_index);
advise(ERROR_STRING);
*/

	n = bit_count(mask);
	if( n <= 0 ){
		// this happens for the format7 video mode...
		// Can this ever happen?  If not, should be CAUTIOUS...
		//warn("no framerates for this video mode!?");
		if( fcp->fc_video_mode == FC2_VIDEOMODE_FORMAT7 ){
			if( fcp->fc_framerate == FC2_FRAMERATE_FORMAT7 ){
				advise("No framerates available for format7.");
			}
			  else {
				assert(0);
			}
		}
		  else {
			assert(0);
		}
		return;
	}

	fcp->fc_framerate_names = getbuf( n * sizeof(char *) );
	fcp->fc_n_framerates = n ;

	i=(-1);
	idx=0;
	while(mask){
		i++;
		if( mask & 1 ){
			fc2FrameRate r;
			int j;
			r = i ;
			j = index_of_framerate( r );
			fcp->fc_framerate_names[idx]=all_framerates[j].nfr_name;
			idx++;
		}
			
		mask >>= 1;
	}

	assert( idx == n );
}

int get_fly_cam_framerate_strings( QSP_ARG_DECL  Data_Obj *str_dp, Fly_Cam *fcp )
{
	int i, n;
	const char *src;
	char *dst;

	get_framerate_choices(QSP_ARG  fcp);

	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?

	n = fcp->fc_n_framerates;

	if( OBJ_COLS(str_dp) < n ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d framerates",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),n);
		warn(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	}

	for(i=0;i<fcp->fc_n_framerates;i++){
		src = fcp->fc_framerate_names[i];
		dst = OBJ_DATA_PTR(str_dp);
		dst += i * OBJ_PXL_INC(str_dp);
		if( strlen(src)+1 > OBJ_COMPS(str_dp) ){
			sprintf(ERROR_STRING,
"String object %s has too few components (%ld) to hold framerate string \"%s\"",
				OBJ_NAME(str_dp),(long)OBJ_COMPS(str_dp),src);
			warn(ERROR_STRING);
		} else {
			strcpy(dst,src);
		}
	}
	return n;
}

static void test_setup(QSP_ARG_DECL  Fly_Cam *fcp,
		fc2VideoMode m, fc2FrameRate r, int *kp, int idx )
{
	fc2Error error;
	BOOL supported;

	// punt if format 7
	if( m == FC2_VIDEOMODE_FORMAT7 && r == FC2_FRAMERATE_FORMAT7 ){
		// BUG?  make sure fly_cam has format7 modes?
		supported = TRUE;
	} else {
		error = fc2GetVideoModeAndFrameRateInfo( fcp->fc_context,
			m,r,&supported );
		if( error != FC2_ERROR_OK ){
			report_fc2_error(QSP_ARG  error,
				"fc2GetVideoModeAndFrameRateInfo" );
			supported = FALSE;
		}
	}

	if( supported ){
		if( *kp < 0 || fcp->fc_video_mode_indices[*kp] != idx ){
			*kp = (*kp)+1;
			fcp->fc_video_mode_indices[*kp] = idx;
			fcp->fc_video_mode_names[*kp] = all_video_modes[idx].nvm_name;
//fprintf(stderr,"test_setup:  adding video mode %s to %s\n",
//all_video_modes[idx].nvm_name,
//fcp->fc_name);
			if( fcp->fc_video_mode == m )
				fcp->fc_video_mode_index = *kp;
		}
		fcp->fc_framerate_mask_tbl[*kp] |= 1 << r;
	}
}

static int get_supported_video_modes(QSP_ARG_DECL  Fly_Cam *fcp )
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
			test_setup(QSP_ARG  fcp, m, r, &n_so_far, i );
		} else {
			for(j=0;j<N_STD_FRAMERATES;j++){
				r=all_framerates[j].nfr_value;
//fprintf(stderr,"Testing video mode %d and frame rate %d\n",
//m,fcp->fc_framerate);
				test_setup(QSP_ARG  fcp, m, r, &n_so_far, i );
			}
		}
	}
	n_so_far++;
//fprintf(stderr,"get_supported_video_modes:  setting n_video_modes to %d\n",k);
	fcp->fc_n_video_modes = n_so_far;
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
	assert( k >= 0 );

	return (fc2FrameRate) k;
}

int is_fmt7_mode(QSP_ARG_DECL  Fly_Cam *fcp, int idx )
{
	fc2VideoMode m;

//	CHECK_IDX(is_fmt7_mode)
	assert( idx >= 0 && idx < fcp->fc_n_video_modes );

	m = all_video_modes[ fcp->fc_video_mode_indices[idx] ].nvm_value;
	if( m == FC2_VIDEOMODE_FORMAT7 ) return 1;
	return 0;
}

// We might want to call this after changing the video mode - we
// don't know whether the library might change anything (like the
// number of buffers, but it seems possible?

static int refresh_config(QSP_ARG_DECL  Fly_Cam *fcp)
{
	fc2Error error;

	error = fc2GetConfiguration(fcp->fc_context,&fcp->fc_config);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetConfiguration" );
		// should we set a flag to indicate an invalid config?
		return -1;
	}
	fcp->fc_n_buffers = fcp->fc_config.numBuffers;
	return 0;
}


int set_std_mode(QSP_ARG_DECL  Fly_Cam *fcp, int idx )
{
	fc2FrameRate r;
	fc2Error error;
	fc2VideoMode m;

//	CHECK_IDX(set_std_mode)
	assert( idx >= 0 && idx < fcp->fc_n_video_modes );

	insure_stopped(QSP_ARG  fcp,"setting video mode");

	m = all_video_modes[ fcp->fc_video_mode_indices[idx] ].nvm_value;
	if( m == FC2_VIDEOMODE_FORMAT7 ){
		warn("set_std_mode:  use set_fmt7_mode to select format7!?");
		return -1;
	}

	r = highest_framerate(QSP_ARG  fcp->fc_framerate_mask_tbl[idx] );

	error = fc2SetVideoModeAndFrameRate(fcp->fc_context,m,r);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetVideoModeAndFrameRate" );
		return -1;
	}

	fcp->fc_my_video_mode_index = idx;
	fcp->fc_video_mode = m;
	fcp->fc_framerate = r;
	fcp->fc_base = NULL;	// force init_fly_base to run again
	fcp->fc_video_mode_index = fcp->fc_video_mode_indices[idx];
	fcp->fc_framerate_index = index_of_framerate(r);

	fcp->fc_cols = all_video_modes[ fcp->fc_video_mode_index ].nvm_width;
	fcp->fc_rows = all_video_modes[ fcp->fc_video_mode_index ].nvm_height;
	fcp->fc_depth = all_video_modes[ fcp->fc_video_mode_index ].nvm_depth;

	return refresh_config(QSP_ARG  fcp);
} // set_std_mode

static void set_highest_fmt7_framerate( QSP_ARG_DECL  Fly_Cam *fcp )
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
	error = fc2GetPropertyInfo( fcp->fc_context, &propInfo );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetPropertyInfo" );
		return;
	}
#endif // FOOBAR
	Fly_Cam_Property_Type *fr_prop_p;

	fr_prop_p = get_pgr_prop("frame_rate" );	// BUG string must match table

	assert( fr_prop_p != NULL );

	assert( fr_prop_p->info.absValSupported );

	prop.type = FC2_FRAME_RATE;
	error = fc2GetProperty( fcp->fc_context, &prop);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetProperty" );
		return;
	}
	prop.absControl = TRUE;
	prop.absValue = fr_prop_p->info.absMax;

	error = fc2SetProperty( fcp->fc_context, &prop);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetProperty" );
		return;
	}
}

int set_fmt7_mode(QSP_ARG_DECL  Fly_Cam *fcp, int idx )
{
	fc2Format7ImageSettings settings;
//	fc2Format7PacketInfo pinfo;
//	BOOL is_valid;
	fc2Error error;
	unsigned int packetSize;
	float percentage;

	insure_stopped(QSP_ARG  fcp,"setting format7 mode");

	if( idx < 0 || idx >= fcp->fc_n_fmt7_modes ){
		warn("Format 7 index out of range!?");
		return -1;
	}

	settings.mode = idx;
	settings.offsetX = 0;
	settings.offsetY = 0;
	settings.width = fcp->fc_fmt7_info_tbl[idx].maxWidth;
	settings.height = fcp->fc_fmt7_info_tbl[idx].maxHeight;
	if( fcp->fc_fmt7_info_tbl[idx].pixelFormatBitField &
			FC2_PIXEL_FORMAT_RAW8 )
		settings.pixelFormat = FC2_PIXEL_FORMAT_RAW8;
	else {
		warn("Camera does not support raw8!?");
		return -1;
	}

fprintf(stderr,"Using size %d x %d\n",settings.width,settings.height);

	percentage = 100.0;
	error = fc2SetFormat7Configuration(fcp->fc_context,&settings,
			percentage);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetFormat7Configuration" );
		return -1;
	}

	set_highest_fmt7_framerate(QSP_ARG  fcp);

	// This fails if we are not in format 7 already.
	error = fc2GetFormat7Configuration(fcp->fc_context,&settings,
			&packetSize,&percentage);

	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetFormat7Configuration" );
		return -1;
	}

	fprintf(stderr,"Percentage = %g (packet size = %d)\n",
		percentage,packetSize);

	fcp->fc_video_mode = FC2_VIDEOMODE_FORMAT7;
	fcp->fc_framerate = FC2_FRAMERATE_FORMAT7;
	fcp->fc_framerate_index = index_of_framerate(FC2_FRAMERATE_FORMAT7);
	fcp->fc_video_mode_index = index_of_video_mode(FC2_VIDEOMODE_FORMAT7);
	fcp->fc_my_video_mode_index = (-1);
	fcp->fc_fmt7_index = idx;
	fcp->fc_base = NULL;	// force init_fly_base to run again

	// Do we have to set the framerate to FC2_FRAMERATE_FORMAT7???

	fcp->fc_rows = settings.height;
	fcp->fc_cols = settings.width;

	{
		long bytes_per_image;
		float est_fps;

		bytes_per_image = settings.width * settings.height;
		// assumes mono8

		est_fps = 8000.0 * packetSize / bytes_per_image;
		fprintf(stderr,"Estimated frame rate:  %g\n",est_fps);
	}

	// refresh_config reads the config from the library...
	return refresh_config(QSP_ARG  fcp);
} // set_fmt7_mode

void set_eii_property(QSP_ARG_DECL  Fly_Cam *fcp, int idx, int yesno )
{
	myEmbeddedImageInfo *eii_p;
	fc2Error error;

	eii_p = (myEmbeddedImageInfo *) (&fcp->fc_ei_info);
	if( ! eii_p->prop_tbl[idx].available ){
		sprintf(ERROR_STRING,"Property %s is not available on %s",
			eii_prop_names[idx],fcp->fc_name);
		warn(ERROR_STRING);
		return;
	}
	eii_p->prop_tbl[idx].onOff = yesno ? TRUE : FALSE;

	error = fc2SetEmbeddedImageInfo(fcp->fc_context,&fcp->fc_ei_info);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetEmbeddedImageInfo" );
	}
}

void set_grab_mode(QSP_ARG_DECL  Fly_Cam *fcp, int grabmode_idx )
{
	fc2Error error;
	fc2Config cfg;

	assert( grabmode_idx >= 0 && grabmode_idx < N_NAMED_GRAB_MODES );

	// grab mode is part of the config struct
	cfg = fcp->fc_config;
	cfg.grabMode = all_grab_modes[grabmode_idx].ngm_value;
	error = fc2SetConfiguration(fcp->fc_context,&cfg);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetConfiguration" );
		// should we set a flag to indicate an invalid config?
		return;
	}
	fcp->fc_config.grabMode = cfg.grabMode;
}

void show_grab_mode(QSP_ARG_DECL  Fly_Cam *fcp)
{
	int idx;

	idx = index_of_grab_mode(fcp->fc_config.grabMode);
	if( idx < 0 ) return;
	sprintf(MSG_STR,"Current grab mode:  %s",all_grab_modes[idx].ngm_name);
	prt_msg(MSG_STR);
}

int pick_fly_cam_framerate(QSP_ARG_DECL  Fly_Cam *fcp, const char *pmpt)
{
	int i;

	if( fcp == NULL ){
		sprintf(ERROR_STRING,"pick_fly_cam_framerate:  no fly_cam selected!?");
		warn(ERROR_STRING);
		return -1;
	}

	get_framerate_choices(QSP_ARG  fcp);
	i=WHICH_ONE(pmpt,fcp->fc_n_framerates,fcp->fc_framerate_names);
	return i;
}

int set_framerate(QSP_ARG_DECL  Fly_Cam *fcp, int framerate_index)
{
	fc2FrameRate rate;
	fc2Error error;

	if( fcp == NULL ) return -1;

	insure_stopped(QSP_ARG  fcp,"setting frame rate");

	rate = all_framerates[framerate_index].nfr_value;
	error = fc2SetVideoModeAndFrameRate(fcp->fc_context,
			fcp->fc_video_mode,rate);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetVideoModeAndFrameRate" );
		return -1;
	}

	fcp->fc_framerate = rate;

	return refresh_config(QSP_ARG  fcp);
}

void show_fly_cam_framerate(QSP_ARG_DECL  Fly_Cam *fcp)
{
	sprintf(MSG_STR,"%s framerate:  %s",
		fcp->fc_name,all_framerates[fcp->fc_framerate_index].nfr_name);
	advise(MSG_STR);
}

void show_fly_cam_video_mode(QSP_ARG_DECL  Fly_Cam *fcp)
{
	sprintf(MSG_STR,"%s video mode:  %s",
		fcp->fc_name,name_for_video_mode(fcp->fc_video_mode));
	advise(MSG_STR);
}

int list_fly_cam_video_modes(QSP_ARG_DECL  Fly_Cam *fcp)
{
	unsigned int i;
	const char *s;

	if( fcp->fc_n_video_modes <= 0 ){
		warn("no video modes!?");
		return -1;
	}

	for(i=0;i<fcp->fc_n_video_modes; i++){
		s=fcp->fc_video_mode_names[i];
		prt_msg_frag("\t");
		prt_msg(s);
	}
	return 0;
}

void list_fly_cam_framerates(QSP_ARG_DECL  Fly_Cam *fcp)
{
	int i;

	get_framerate_choices(QSP_ARG  fcp);

	for(i=0;i<fcp->fc_n_framerates;i++){
		prt_msg_frag("\t");
		prt_msg(fcp->fc_framerate_names[i]);
	}
}

static int set_default_video_mode(QSP_ARG_DECL  Fly_Cam *fcp)
{
	fc2VideoMode m;
	fc2FrameRate r;
	int i,j;
	fc2Error error;
	int _nskip;

	if( get_supported_video_modes(QSP_ARG  fcp ) < 0 ){
		warn("set_default_video_mode:  Can't get video modes");
		return -1;
	}

//fprintf(stderr,"get_supported_video_modes found %d modes\n",fcp->fc_n_video_modes);
	_nskip=0;
	do {
		_nskip++;
		i = fcp->fc_n_video_modes-(_nskip);	// order the table so that mono8 is last?
		fcp->fc_my_video_mode_index = i;
		j = fcp->fc_video_mode_indices[i];
		m = all_video_modes[j].nvm_value;
		// BUG we don't check that nskip is in-bounds, but should be OK
	} while( m == FC2_VIDEOMODE_FORMAT7  && _nskip < fcp->fc_n_video_modes );

	if( m == FC2_VIDEOMODE_FORMAT7 ){
		/*
		sprintf(ERROR_STRING,"set_default_video_mode:  %s has only format7 modes!?",
			fcp->fc_name);
		advise(ERROR_STRING);
		*/
		return -1;
	}

//fprintf(stderr,"_nskip = %d, i = %d,  j = %d,  m = %d\n",_nskip,i,j,m);

//fprintf(stderr,"highest non-format7 video mode is %s\n",all_video_modes[j].nvm_name);

	// if this is format7, don't use it!?

	fcp->fc_video_mode = m;
	fcp->fc_video_mode_index = j;

	fcp->fc_cols = all_video_modes[ j ].nvm_width;
	fcp->fc_rows = all_video_modes[ j ].nvm_height;
	fcp->fc_depth = all_video_modes[ j ].nvm_depth;

	// Get the hightest frame rate associated with this video mode...
	r = highest_framerate( QSP_ARG  fcp->fc_framerate_mask_tbl[i] );

//fprintf(stderr,"mode %s, highest supported frame rate is %s\n",
//name_for_video_mode(m),
//name_for_framerate(r));
	fcp->fc_framerate = r;
	fcp->fc_framerate_index = index_of_framerate(r);

//sprintf(ERROR_STRING,"set_default_video_mode:  setting to %s", name_for_video_mode(m));
//advise(ERROR_STRING);


	error = fc2SetVideoModeAndFrameRate( fcp->fc_context,
			fcp->fc_video_mode, fcp->fc_framerate );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetVideoModeAndFrameRate" );
		return -1;
	}

	// We might stash the number of video modes here?

	// stash the number of framerates in a script variable
	// in case the user wants to fetch the strings...
	// BUG DO THIS WHEN CAM IS SELECTED!
	//set_script_var_from_int(QSP_ARG
	//		"n_framerates",fcp->fc_framerates.num);

sprintf(ERROR_STRING,"%s:  %s, %s fps",fcp->fc_name,name_for_video_mode(m),
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

static Fly_Cam *unique_fly_cam_instance( QSP_ARG_DECL  fc2Context context )
{
	int i;
	char cname[LLEN];	// How many chars is enough?
	Fly_Cam *fcp;
	fc2Error error;
	fc2CameraInfo camInfo;

	error = fc2GetCameraInfo( context, &camInfo );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetCameraInfo" );
		return NULL;
	}

	i=1;
	fcp=NULL;
	while(fcp==NULL){
		//sprintf(cname,"%s_%d",cam_p->model,i);
		if( snprintf(cname,LLEN,"%s_%d",camInfo.modelName,i) >= LLEN ){
			error1("unique_fly_cam_instance:  camera name too long for buffer!?");
		}
		fix_string(cname);	// change spaces to underscores
//sprintf(ERROR_STRING,"Checking for existence of %s",cname);
//advise(ERROR_STRING);
		fcp = fly_cam_of( cname );
		if( fcp == NULL ){	// This index is free
//sprintf(ERROR_STRING,"%s is not in use",cname);
//advise(ERROR_STRING);
			fcp = new_fly_cam( cname );
			if( fcp == NULL ){
				sprintf(ERROR_STRING,
			"Failed to create fly_cam %s!?",cname);
				error1(ERROR_STRING);
			}
		} else {
//sprintf(ERROR_STRING,"%s IS in use",cname);
//advise(ERROR_STRING);
			fcp = NULL;
		}
		i++;
		if( i>=5 ){
			error1("Too many fly_cams!?"); 
		}
	}
	fcp->fc_cam_info = camInfo;
	return fcp;
}

static void get_fmt7_modes(QSP_ARG_DECL  Fly_Cam *fcp)
{
	fc2Error error;
	BOOL supported;
	int i, largest=(-1);
	fc2Format7Info fmt7_info_tbl[N_FMT7_MODES];
	size_t nb;

	fcp->fc_n_fmt7_modes = 0;
	for(i=0;i<N_FMT7_MODES;i++){
		fmt7_info_tbl[i].mode = i;
		error = fc2GetFormat7Info(fcp->fc_context,
				&fmt7_info_tbl[i],&supported);
		if( error != FC2_ERROR_OK ){
			report_fc2_error(QSP_ARG  error, "fc2GetFormat7Info" );
		}
		if( supported ){
			fcp->fc_n_fmt7_modes ++ ;
			largest = i;
		}
	}
	if( (largest+1) != fcp->fc_n_fmt7_modes ){
		sprintf(ERROR_STRING,
	"Unexpected number of format7 modes!?  (largest index = %d, n_modes = %d)",
			largest,fcp->fc_n_fmt7_modes);
		warn(ERROR_STRING);
	}

	nb = fcp->fc_n_fmt7_modes * sizeof(fc2Format7Info);
	fcp->fc_fmt7_info_tbl = getbuf( nb );
	memcpy(fcp->fc_fmt7_info_tbl,fmt7_info_tbl,nb);

	fcp->fc_fmt7_index = 0;
}

#define SHOW_FIELD(desc_str,value)					\
									\
sprintf(MSG_STR,"\t%s:  %d",#desc_str,fcp->fc_fmt7_info_tbl[mode].value);	\
prt_msg(MSG_STR);

#define SHOW_FIELD_HEX(desc_str,value)					\
									\
sprintf(MSG_STR,"\t%s:  0x%x",#desc_str,fcp->fc_fmt7_info_tbl[mode].value); \
prt_msg(MSG_STR);

static void show_fmt7_info(QSP_ARG_DECL  Fly_Cam *fcp, fc2Mode mode )
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

void show_fmt7_modes(QSP_ARG_DECL  Fly_Cam *fcp)
{
	if( fcp->fc_n_fmt7_modes <= 0 ){
		prt_msg("\tNo format7 modes available.");
	} else {
		int i;
		for(i=0;i<fcp->fc_n_fmt7_modes;i++)
			show_fmt7_info( QSP_ARG  fcp, i );
	}
}

// This should only be called once...

static Fly_Cam *setup_my_fly_cam( QSP_ARG_DECL
				fc2Context context, fc2PGRGuid *guid_p,
				int index )
{
	Fly_Cam *fcp;
	fc2Error error;
	int i;

	error = fc2Connect(context,guid_p);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2Connect" );
		return NULL;
	}

	// We could have multiple instances of the same model...
	fcp = unique_fly_cam_instance(QSP_ARG  context);

	// Why do we disconnect from the context we were called with -
	// subsequent cameras will still use it...
	// Maybe this just undoes the connection directly above?
	error = fc2Disconnect(context);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2Disconnect" );
		// BUG should clean up and return NULL?
		return fcp;
	}

	//fcp->fc_cam_p = cam_p;
	error = fc2CreateContext(&fcp->fc_context);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2CreateContext" );
		// BUG clean up first
		return NULL;
	}
	error=fc2GetCameraFromIndex(fcp->fc_context,index,&fcp->fc_guid);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetCameraFromIndex" );
	}

	error = fc2Connect(fcp->fc_context,&fcp->fc_guid);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2Connect" );
		// BUG clean up first
		return NULL;
	}

	if( refresh_config(QSP_ARG  fcp) < 0 )
		// BUG clean up first
		return NULL;

	error = fc2GetEmbeddedImageInfo(fcp->fc_context,&fcp->fc_ei_info);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetEmbeddedImageInfo" );
		// BUG clean up first
		return NULL;
	}

	fcp->fc_feat_lp=NULL;
	fcp->fc_in_use_lp=NULL;
	fcp->fc_flags = 0;		/* assume no B-mode unless we are told otherwise... */
	fcp->fc_base = NULL;

	get_fmt7_modes(QSP_ARG  fcp);

	error = fc2GetVideoModeAndFrameRate( fcp->fc_context,
			&fcp->fc_video_mode, &fcp->fc_framerate );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2GetVideoModeAndFrameRate" );
		// BUG clean up first
		return NULL;
	}
/*
fprintf(stderr,"after fetching, video mode = %d (%s), frame rate = %d (%s)\n",
fcp->fc_video_mode,
name_for_video_mode(fcp->fc_video_mode),
fcp->fc_framerate,
name_for_framerate(fcp->fc_framerate)
);
*/

	// BUG?  could be N_STD_VIDEO_MODES?
	fcp->fc_video_mode_indices = getbuf( sizeof(int) * N_NAMED_VIDEO_MODES );
	fcp->fc_video_mode_names = getbuf( sizeof(char *) * N_NAMED_VIDEO_MODES );
	fcp->fc_framerate_mask_tbl = getbuf( sizeof(Framerate_Mask) * N_NAMED_VIDEO_MODES );
	fcp->fc_framerate_names = NULL;
	fcp->fc_n_framerates = 0;

	for(i=0;i<N_NAMED_VIDEO_MODES;i++){
		fcp->fc_framerate_mask_tbl[i] = 0;
	}

	/* Originally, we set the default video mode here to be
	 * the highest standard video mode...  But for the flea3,
	 * which has an odd-shaped sensor, the good modes are format7
	 * and for no good reason when we change back to format7
	 * under program control the high frame rate is not restored???
	 */

//#ifdef NOT_GOOD
	if( set_default_video_mode(QSP_ARG  fcp) < 0 ){
		/*
		sprintf(ERROR_STRING,"error setting default video mode for %s",fcp->fc_name);
		warn(ERROR_STRING);
		cleanup_fly_cam( fcp );
		return(NULL);
		*/
		// This can fail for fly_cams that only support format7...
		// Deal with this better later
		sprintf(ERROR_STRING,"error setting default video mode for %s, only format7?",
			fcp->fc_name);
		advise(ERROR_STRING);
	}
//#endif // NOT_GOOD

	/* used to set B-mode stuff here... */
	// What if the fly_cam is a usb cam???
	//dc1394_video_set_iso_speed( cam_p, DC1394_ISO_SPEED_400 );

	fcp->fc_img_p = getbuf( sizeof(*fcp->fc_img_p) );

        error = fc2CreateImage( fcp->fc_img_p );
        if ( error != FC2_ERROR_OK ) {
		report_fc2_error(QSP_ARG  error, "fc2CreateImage" );
		// BUG clean up?
		//return NULL;
	}


	// Make a data_obj context for the frames...
	fcp->fc_do_icp = create_dobj_context( fcp->fc_name );

	fcp->fc_frm_dp_tbl = NULL;
	fcp->fc_newest = (-1);

	return(fcp);
}

void pop_fly_cam_context(SINGLE_QSP_ARG_DECL)
{
	// pop old context...
	Item_Context *icp;
	icp=pop_dobj_context();
	assert( icp != NULL );
}

void push_fly_cam_context(QSP_ARG_DECL  Fly_Cam *fcp)
{
//fprintf(stderr,"pushing fly_cam context for %s (icp = 0x%lx)\n",
//fcp->fc_name,(long)fcp->fc_do_icp);
	push_dobj_context(fcp->fc_do_icp);
}

int init_fly_cam_system(SINGLE_QSP_ARG_DECL)
{
#ifdef HAVE_LIBFLYCAP
	fc2Version version;
	fc2Context context;
	fc2Error error;
	fc2PGRGuid guid;	// BUG should be associated with one fly_cam?
	unsigned int numCameras=0;
	int i;
	static int firewire_system_inited=0;

	if( firewire_system_inited ){
		warn("Firewire system has already been initialized!?");
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
		advise("No fly_cams detected.");
		return 0;
	}
	sprintf(ERROR_STRING,
		"%d fly_cam%s found.", numCameras, numCameras==1?"":"s" );
	advise(ERROR_STRING);


	//for(i=0;i<numCameras;i++)
	for(i=numCameras-1;i>=0;i--){
		error=fc2GetCameraFromIndex(context,i,&guid);
		if( error != FC2_ERROR_OK ){
			report_fc2_error(QSP_ARG  error, "fc2GetCameraFromIndex" );
		} else {
fprintf(stderr,"Calling setup_my_fly_cam for camera %d\n",i);
			setup_my_fly_cam(QSP_ARG   context, &guid, i );
		}
	}
	error = fc2DestroyContext( context );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2DestroyContext" );
	}

	add_sizable(fly_cam_itp, &fly_cam_sf, (Item *(*)(QSP_ARG_DECL  const char *)) _fly_cam_of );

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

void show_n_buffers(QSP_ARG_DECL  Fly_Cam *fcp)
{
	sprintf(MSG_STR,"%s:  %d buffers",fcp->fc_name,fcp->fc_n_buffers);
	prt_msg(MSG_STR);
}

int set_n_buffers(QSP_ARG_DECL  Fly_Cam *fcp, int n )
{
	fc2Config cfg;
	fc2Error error;

fprintf(stderr,"set_n_buffers %s %d\n",fcp->fc_name,n);
	if( n < MIN_N_BUFFERS || n > MAX_N_BUFFERS ){
		sprintf(ERROR_STRING,
"set_n_buffers:  number of buffers must be between %d and %d (%d requested)!?",
			MIN_N_BUFFERS,MAX_N_BUFFERS,n);
		warn(ERROR_STRING);
		return -1;
	}
	cfg = fcp->fc_config;
	cfg.numBuffers = n;

	error = fc2SetConfiguration(fcp->fc_context,&cfg);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2SetConfiguration" );
		// should we set a flag to indicate an invalid config?
		return -1;
	}
	fcp->fc_n_buffers =
	fcp->fc_config.numBuffers = n;
	fcp->fc_base = NULL;	// force init_fly_base to run again

show_n_buffers(QSP_ARG  fcp);
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

void print_fly_cam_info(QSP_ARG_DECL  Fly_Cam *fcp)
{
#ifdef HAVE_LIBFLYCAP
	show_cam_info(QSP_ARG  &fcp->fc_cam_info);
	show_cam_cfg(QSP_ARG  &fcp->fc_config);
	show_ei_info(QSP_ARG  &fcp->fc_ei_info);
#endif // HAVE_LIBFLYCAP

	/*
	i=index_of_framerate(fcp->fc_framerate);
	sprintf(msg_str,"\tframe rate:  %s",all_framerates[i].nfr_name);
	prt_msg(msg_str);
	*/

	//report_fly_cam_features(fcp);

	// show_fmt7_modes(QSP_ARG  fcp);
	// Show the current video mode

	sprintf(MSG_STR,"Current video mode:  %s%s",
		fcp->fc_my_video_mode_index >= 0 ?
		fcp->fc_video_mode_names[ fcp->fc_my_video_mode_index ] :
		"format7 mode ",
		fcp->fc_my_video_mode_index >= 0 ? "": (
			fcp->fc_fmt7_index == 0 ? "0" : (
			fcp->fc_fmt7_index == 1 ? "1" : (
			fcp->fc_fmt7_index == 2 ? "2" : "(>2)" )))
			);
	prt_msg(MSG_STR);

	sprintf(MSG_STR,"Current frame rate:  %s",
		all_framerates[ fcp->fc_framerate_index ].nfr_name );
	prt_msg(MSG_STR);
}

static void init_one_frame(QSP_ARG_DECL  Fly_Cam *fcp, int index )
{
	Data_Obj *dp;
	char fname[32];
	Dimension_Set ds1;

	sprintf(fname,"frame%d",index);
	//assign_var("newest",fname+5);

	dp = dobj_of(fname);
	if( dp == NULL ){
		SET_DS_SEQS(&ds1,1);
		SET_DS_FRAMES(&ds1,1);
		SET_DS_ROWS(&ds1,fcp->fc_img_p->rows);
		SET_DS_COLS(&ds1,fcp->fc_img_p->cols);
		SET_DS_COMPS(&ds1,1);
		dp = _make_dp(QSP_ARG  fname,&ds1,PREC_FOR_CODE(PREC_UBY));
		assert( dp != NULL );

		SET_OBJ_DATA_PTR( dp, fcp->fc_base+index*fcp->fc_buf_delta );
		fcp->fc_frm_dp_tbl[index] = dp;

//fprintf(stderr,"init_one_frame %d:  %s, data at 0x%lx\n",index,OBJ_NAME(dp),(long)OBJ_DATA_PTR(dp));
//		}
	} else {
		sprintf(ERROR_STRING,"init_one_frame:  object %s already exists!?",
			fname);
		warn(ERROR_STRING);
	}
} // end init_one_frame

static void init_cam_frames(QSP_ARG_DECL  Fly_Cam *fcp)
{
	int index;

	assert( fcp->fc_n_buffers > 0 );
	assert( fcp->fc_frm_dp_tbl == NULL );

	fcp->fc_frm_dp_tbl = getbuf( sizeof(Data_Obj) * fcp->fc_n_buffers );
	for(index=0;index<fcp->fc_n_buffers;index++)
		init_one_frame(QSP_ARG  fcp, index);
} // init_cam_frames

// init_fly_base   -   grab frames to get the address
// associated with each frame index.  This wouldn't be necessary
// if we always provided the buffers, but we want this to work
// even when we don't.
//
// We keep track of the largest and smallest address, we save
// those so we can figure out the index of an arbitrary frame...

static void init_fly_base(QSP_ARG_DECL  Fly_Cam *fcp)
{
	// initialize all the pointers
	int n_buffers_seen=0;
	void *smallest_addr;
	void *largest_addr;
	void *first_addr=NULL;
	void **addr_tbl;

	// fc_base is our flag, reset to NULL when number of buffers
	// is changed, or video mode is changed.
	if( fcp->fc_base != NULL ){
		return;
	}

	n_buffers_seen = 0;
	addr_tbl = getbuf(sizeof(void *)*fcp->fc_n_buffers);

	// silence compiler warnings
	largest_addr = NULL;
	smallest_addr = NULL;

	while( n_buffers_seen < fcp->fc_n_buffers ){
		fc2Error error;
		void *buf_addr;

		error = fc2RetrieveBuffer( fcp->fc_context, fcp->fc_img_p );
		if( error != FC2_ERROR_OK ){
			report_fc2_error(QSP_ARG  error, "fc2RetrieveBuffer" );
			return;
		}
/*
sprintf(ERROR_STRING,"pData = 0x%lx",(long)fcp->fc_img_p->pData);
advise(ERROR_STRING);
*/
		buf_addr = fcp->fc_img_p->pData;

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

	fcp->fc_base = smallest_addr;
	fcp->fc_buf_delta = (largest_addr - smallest_addr) / (n_buffers_seen-1);
	//fcp->fc_buf_delta = (largest - smallest) / 30;

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
			(long)fcp->fc_buf_delta);
		advise(ERROR_STRING);
	}

	init_cam_frames(QSP_ARG  fcp);
}

int check_buffer_alignment(QSP_ARG_DECL  Fly_Cam *fcp)
{
	int i;

	// alignment requirement is now 1024
	// BUG this should be a parameter...
#define RV_ALIGNMENT_REQ	1024

	for(i=0;i<fcp->fc_n_buffers;i++){
		if( ((long)(fcp->fc_base+i*fcp->fc_buf_delta)) % RV_ALIGNMENT_REQ != 0 ){
			sprintf(ERROR_STRING,"Buffer %d is not aligned - %d byte alignment required for raw volume I/O!?",
				i,RV_ALIGNMENT_REQ);
			warn(ERROR_STRING);
			return -1;
		}
	}
	return 0;
}

static int index_of_buffer(QSP_ARG_DECL  Fly_Cam *fcp,fc2Image *ip)
{
	int idx;

	idx = ( ip->pData - fcp->fc_base ) / fcp->fc_buf_delta;
	/*
sprintf(ERROR_STRING,
"index_of_buffer:  data at 0x%lx, base = 0x%lx, idx = %d",
(long)ip->pData,(long)fcp->fc_base,idx);
advise(ERROR_STRING);
*/

	assert( idx >= 0 && idx < fcp->fc_n_buffers );
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

Data_Obj * grab_fly_cam_frame(QSP_ARG_DECL  Fly_Cam * fcp )
{
	fc2Error error;
	int index;

	if( fcp->fc_base == NULL )
		init_fly_base(QSP_ARG  fcp);

	error = fc2RetrieveBuffer( fcp->fc_context, fcp->fc_img_p );
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2RetrieveBuffer" );
		return NULL;
	}
//fprintf(stderr,"pixel format of retrieved images is %s (0x%x)\n",
//name_for_pixel_format(img.format),img.format);

	index = index_of_buffer(QSP_ARG  fcp, fcp->fc_img_p );
	fcp->fc_newest = index;

	return( fcp->fc_frm_dp_tbl[index] );
}

int reset_fly_cam(QSP_ARG_DECL  Fly_Cam *fcp)
{
	fc2Error error;

	error=fc2FireBusReset(fcp->fc_context,&fcp->fc_guid);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2FireBusReset" );
	}

	return 0;
}

void report_fly_cam_bandwidth(QSP_ARG_DECL  Fly_Cam *fcp )
{
	UNIMP_FUNC("report_fly_cam_bandwidth");
}

unsigned int read_register( QSP_ARG_DECL  Fly_Cam *fcp, unsigned int addr )
{
	fc2Error error;
	unsigned int val;

	error = fc2ReadRegister(fcp->fc_context,addr,&val);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2ReadRegister" );
	}
	return val;
}

void write_register( QSP_ARG_DECL  Fly_Cam *fcp, unsigned int addr, unsigned int val )
{
	fc2Error error;

	error = fc2WriteRegister(fcp->fc_context,addr,val);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2WriteRegister" );
	}
}

void start_firewire_capture(QSP_ARG_DECL  Fly_Cam *fcp)
{
	fc2Error error;

advise("start_firewire_capture BEGIN");
	if( fcp->fc_flags & FLY_CAM_IS_RUNNING ){
		warn("start_firewire_capture:  fly_cam is already capturing!?");
		return;
	}
advise("start_firewire_capture cam is not already running");
advise("start_firewire_capture calling fc2StartCapture");

fprintf(stderr,"context = 0x%lx\n",(long)fcp->fc_context);
	error = fc2StartCapture(fcp->fc_context);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2StartCapture" );
	} else {
		fcp->fc_flags |= FLY_CAM_IS_RUNNING;

		// BUG - we should undo this when we stop capturing, because
		// we might change the video format or something else.
		// Perhaps more efficiently we could only do it when needed?
advise("start_firewire_capture calling init_fly_base");
		init_fly_base(QSP_ARG  fcp);
	}
advise("start_firewire_capture DONE");
}

void stop_firewire_capture(QSP_ARG_DECL  Fly_Cam *fcp)
{
	fc2Error error;

	if( (fcp->fc_flags & FLY_CAM_IS_RUNNING) == 0 ){
		warn("stop_firewire_capture:  fly_cam is not capturing!?");
		return;
	}

	error = fc2StopCapture(fcp->fc_context);
	if( error != FC2_ERROR_OK ){
		report_fc2_error(QSP_ARG  error, "fc2StopCapture" );
	} else {
		fcp->fc_flags &= ~FLY_CAM_IS_RUNNING;
	}
}

void set_fmt7_size(QSP_ARG_DECL  Fly_Cam *fcp, int w, int h)
{
	UNIMP_FUNC("set_fmt7_size");
}

void release_oldest_frame(QSP_ARG_DECL  Fly_Cam *fcp)
{
	UNIMP_FUNC("release_oldest_frame");
}

void list_fly_cam_trig(QSP_ARG_DECL  Fly_Cam *fcp)
{
	fc2Error error;
	fc2TriggerModeInfo tinfo;
	fc2TriggerMode tmode;

	error = fc2GetTriggerModeInfo(fcp->fc_context,&tinfo);
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

	error = fc2GetTriggerMode(fcp->fc_context,&tmode);
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

void set_buffer_obj(QSP_ARG_DECL  Fly_Cam *fcp, Data_Obj *dp)
{
	// make sure sizes match
	if( OBJ_COLS(dp) != fcp->fc_cols || OBJ_ROWS(dp) != fcp->fc_rows ){
		sprintf(ERROR_STRING,
"set_buffer_obj:  size mismatch between %s (%dx%d) and object %s (%dx%d)",
			fcp->fc_name,fcp->fc_cols,fcp->fc_rows,
			OBJ_NAME(dp),OBJ_COLS(dp),OBJ_ROWS(dp) );
		warn(ERROR_STRING);
		return;
	}
	if( PREC_CODE(OBJ_MACH_PREC_PTR(dp)) != PREC_UBY ){
		sprintf(ERROR_STRING,"Object %s (%s) should have %s precision!?",
			OBJ_NAME(dp),OBJ_PREC_NAME(dp),NAME_FOR_PREC_CODE(PREC_UBY));
		warn(ERROR_STRING);
		return;
	}
	{
		fc2Error error;

		error = fc2SetUserBuffers(fcp->fc_context, OBJ_DATA_PTR(dp),
				OBJ_COLS(dp)*OBJ_ROWS(dp)*OBJ_COMPS(dp),OBJ_FRAMES(dp));
		if( error != FC2_ERROR_OK ){
			report_fc2_error(QSP_ARG  error, "fc2SetUserBuffers" );
			return;
		}
		// refresh the configuration
		refresh_config(QSP_ARG  fcp);
	}
	fcp->fc_base = NULL;	// force init_fly_base to run again
}

#endif /* HAVE_LIBFLYCAP */


static const char **grab_mode_names=NULL;

int pick_grab_mode(QSP_ARG_DECL Fly_Cam *fcp, const char *pmpt)
{
	int idx;

	if( fcp == NULL ){
		sprintf(ERROR_STRING,"pick_fly_cam_grab_mode:  no fly_cam selected!?");
		warn(ERROR_STRING);
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

