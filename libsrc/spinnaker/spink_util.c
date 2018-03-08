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

#include "spink.h"

#define TMPSIZE	32	// for temporary object names, e.g. _frame55

ITEM_INTERFACE_DECLARATIONS(Spink_Interface,spink_interface,0)
ITEM_INTERFACE_DECLARATIONS(Spink_Cam,spink_cam,0)

#define UNIMP_FUNC(name)						\
	sprintf(ERROR_STRING,"Function %s is not implemented!?",name);	\
	warn(ERROR_STRING);

#ifdef HAVE_LIBSPINNAKER
#define SPINK_ENTRY(string,code)				\
{	#string,	code	}
#else // ! HAVE_LIBSPINNAKER
#define SPINK_ENTRY(string,code)				\
{	#string			}
#endif // ! HAVE_LIBSPINNAKER

#ifdef NOT_USED
static Named_Pixel_Format all_pixel_formats[]={
{ "mono8",	SPINK_PIXEL_FORMAT_MONO8			},
{ "411yuv8",	SPINK_PIXEL_FORMAT_411YUV8		},
{ "422yuv8",	SPINK_PIXEL_FORMAT_422YUV8		},
{ "444yuv8",	SPINK_PIXEL_FORMAT_444YUV8		},
{ "rgb8",	SPINK_PIXEL_FORMAT_RGB8			},
{ "mono16",	SPINK_PIXEL_FORMAT_MONO16			},
{ "rgb16",	SPINK_PIXEL_FORMAT_RGB16			},
{ "s_mono16",	SPINK_PIXEL_FORMAT_S_MONO16		},
{ "s_rgb16",	SPINK_PIXEL_FORMAT_S_RGB16		},
{ "raw8",	SPINK_PIXEL_FORMAT_RAW8			},
{ "raw16",	SPINK_PIXEL_FORMAT_RAW16			},
{ "mono12",	SPINK_PIXEL_FORMAT_MONO12			},
{ "raw12",	SPINK_PIXEL_FORMAT_RAW12			},
{ "bgr",	SPINK_PIXEL_FORMAT_BGR			},
{ "bgru",	SPINK_PIXEL_FORMAT_BGRU			},
{ "rgb",	SPINK_PIXEL_FORMAT_RGB			},
{ "rgbu",	SPINK_PIXEL_FORMAT_RGBU			},
{ "bgr16",	SPINK_PIXEL_FORMAT_BGR16			},
{ "bgru16",	SPINK_PIXEL_FORMAT_BGRU16			},
{ "jpeg",	SPINK_PIXEL_FORMAT_422YUV8_JPEG		},
};

#define N_NAMED_PIXEL_FORMATS	(sizeof(all_pixel_formats)/sizeof(Named_Pixel_Format))
#endif // NOT_USED

#ifdef HAVE_LIBSPINNAKER
#define SPINK_MODE(string,code,w,h,d)	{#string,code,w,h,d}
#else // ! HAVE_LIBSPINNAKER
#define SPINK_MODE(string,code,w,h,d)	{#string,w,h,d}
#endif // ! HAVE_LIBSPINNAKER

static Named_Video_Mode all_video_modes[]={
#ifdef FOO
SPINK_MODE( format7,		SPINK_VIDEOMODE_FORMAT7,		0, 0,	0 ),

SPINK_MODE( yuv444_160x120,	SPINK_VIDEOMODE_160x120YUV444,	160, 120, 3 ),

SPINK_MODE( yuv422_320x240,	SPINK_VIDEOMODE_320x240YUV422,	320, 240, 2 ),

SPINK_MODE( yuv411_640x480,	SPINK_VIDEOMODE_640x480YUV411,	640, 480, 0 /*1.5*/),
SPINK_MODE( yuv422_640x480,	SPINK_VIDEOMODE_640x480YUV422,	640, 480, 2 ),
SPINK_MODE( rgb8_640x480,		SPINK_VIDEOMODE_640x480RGB,	640, 480, 3 ),
SPINK_MODE( mono16_640x480,	SPINK_VIDEOMODE_640x480Y16,	640, 480, 2 ),
SPINK_MODE( mono8_640x480,	SPINK_VIDEOMODE_640x480Y8,	640, 480, 1 ),

SPINK_MODE( yuv422_800x600,	SPINK_VIDEOMODE_800x600YUV422,	800, 600, 2 ),
SPINK_MODE( rgb8_800x600,		SPINK_VIDEOMODE_800x600RGB,	800, 600, 3 ),
SPINK_MODE( mono16_800x600,	SPINK_VIDEOMODE_800x600Y16,	800, 600, 2 ),
SPINK_MODE( mono8_800x600,	SPINK_VIDEOMODE_800x600Y8,	800, 600, 1 ),

SPINK_MODE( yuv422_1024x768,	SPINK_VIDEOMODE_1024x768YUV422,	1024, 768, 2 ),
SPINK_MODE( rgb8_1024x768,	SPINK_VIDEOMODE_1024x768RGB,	1024, 768, 3 ),
SPINK_MODE( mono16_1024x768,	SPINK_VIDEOMODE_1024x768Y16,	1024, 768, 2 ),
SPINK_MODE( mono8_1024x768,	SPINK_VIDEOMODE_1024x768Y8,	1024, 768, 1 ),

SPINK_MODE( yuv422_1280x960,	SPINK_VIDEOMODE_1280x960YUV422,	1280, 960, 2 ),
SPINK_MODE( rgb8_1280x960,	SPINK_VIDEOMODE_1280x960RGB,	1280, 960, 3 ),
SPINK_MODE( mono16_1280x960,	SPINK_VIDEOMODE_1280x960Y16,	1280, 960, 2 ),
SPINK_MODE( mono8_1280x960,	SPINK_VIDEOMODE_1280x960Y8,	1280, 960, 1 ),

SPINK_MODE( yuv422_1600x1200,	SPINK_VIDEOMODE_1600x1200YUV422,	1600, 1200, 2 ),
SPINK_MODE( rgb8_1600x1200,	SPINK_VIDEOMODE_1600x1200RGB,	1600, 1200, 3 ),
SPINK_MODE( mono16_1600x1200,	SPINK_VIDEOMODE_1600x1200Y16,	1600, 1200, 2 ),
SPINK_MODE( mono8_1600x1200,	SPINK_VIDEOMODE_1600x1200Y8,	1600, 1200, 1 )
#endif // FOO

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
/*
SPINK_ENTRY(	drop_frames,	SPINK_DROP_FRAMES		),
SPINK_ENTRY(	buffer_frames,	SPINK_BUFFER_FRAMES	)
*/
};

#define N_NAMED_GRAB_MODES	(sizeof(all_grab_modes)/sizeof(Named_Grab_Mode))


#ifdef HAVE_LIBSPINNAKER

static Named_Frame_Rate all_framerates[]={
/*
SPINK_ENTRY(	1.875,		SPINK_FRAMERATE_1_875	),
SPINK_ENTRY(	3.75,		SPINK_FRAMERATE_3_75	),
SPINK_ENTRY(	7.5,		SPINK_FRAMERATE_7_5	),
SPINK_ENTRY(	15,		SPINK_FRAMERATE_15	),
SPINK_ENTRY(	30,		SPINK_FRAMERATE_30	),
SPINK_ENTRY(	60,		SPINK_FRAMERATE_60	),
SPINK_ENTRY(	120,		SPINK_FRAMERATE_120	),
SPINK_ENTRY(	240,		SPINK_FRAMERATE_240	),
SPINK_ENTRY(	format7,	SPINK_FRAMERATE_FORMAT7	)
*/
};

#define N_NAMED_FRAMERATES	(sizeof(all_framerates)/sizeof(Named_Frame_Rate))
#define N_STD_FRAMERATES	(N_NAMED_FRAMERATES-1)

#ifdef FOOBAR
static Named_Bus_Speed all_bus_speeds[]={
/*
SPINK_ENTRY(	100,		SPINK_BUSSPEED_S100		),
SPINK_ENTRY(	200,		SPINK_BUSSPEED_S200		),
SPINK_ENTRY(	400,		SPINK_BUSSPEED_S400		),
SPINK_ENTRY(	800,		SPINK_BUSSPEED_S800		),
SPINK_ENTRY(	1600,		SPINK_BUSSPEED_S1600		),
SPINK_ENTRY(	3200,		SPINK_BUSSPEED_S3200		),
SPINK_ENTRY(	5000,		SPINK_BUSSPEED_S5000		),
SPINK_ENTRY(	10baseT,	SPINK_BUSSPEED_10BASE_T		),
SPINK_ENTRY(	100baseT,	SPINK_BUSSPEED_100BASE_T		),
SPINK_ENTRY(	1000baseT,	SPINK_BUSSPEED_1000BASE_T		),
SPINK_ENTRY(	10000baseT,	SPINK_BUSSPEED_10000BASE_T	),
SPINK_ENTRY(	fastest,	SPINK_BUSSPEED_S_FASTEST		),
SPINK_ENTRY(	any,		SPINK_BUSSPEED_ANY		)
*/
};

#define N_NAMED_BUS_SPEEDS	(sizeof(all_bus_speeds)/sizeof(Named_Bus_Speed))

static Named_Bandwidth_Allocation all_bw_allocations[]={
/*
SPINK_ENTRY(	off,		SPINK_BANDWIDTH_ALLOCATION_OFF		),
SPINK_ENTRY(	on,		SPINK_BANDWIDTH_ALLOCATION_ON		),
SPINK_ENTRY(	unsupported,	SPINK_BANDWIDTH_ALLOCATION_UNSUPPORTED	),
SPINK_ENTRY(	unspecified,	SPINK_BANDWIDTH_ALLOCATION_UNSPECIFIED	)
*/
};

#define N_NAMED_BW_ALLOCATIONS	(sizeof(all_bw_allocations)/sizeof(Named_Bandwidth_Allocation))

static Named_Interface all_interfaces[]={
/*
SPINK_ENTRY(	ieee1394,	SPINK_INTERFACE_IEEE1394	),
SPINK_ENTRY(	usb2,		SPINK_INTERFACE_USB_2	),
SPINK_ENTRY(	usb3,		SPINK_INTERFACE_USB_3	),
SPINK_ENTRY(	gigE,		SPINK_INTERFACE_GIGE	)
*/
};

#define N_NAMED_INTERFACES	(sizeof(all_interfaces)/sizeof(Named_Interface))
#endif // FOOBAR

#endif // HAVE_LIBSPINNAKER


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

#ifdef HAVE_LIBSPINNAKER

static double get_spink_cam_size(QSP_ARG_DECL  Item *ip, int dim_index)
{
	switch(dim_index){
		case 0:	return(1.0); /* BUG - not correct for color spink_cams! */ break;
		case 1: return(((Spink_Cam *)ip)->sk_cols);
		case 2: return(((Spink_Cam *)ip)->sk_rows);
		case 3: return(((Spink_Cam *)ip)->sk_n_buffers);
		case 4: return(1.0);
		default:
			// should never happen
			assert(0);
			break;
	}
	return(0.0);
}

static const char * get_spink_cam_prec_name(QSP_ARG_DECL  Item *ip )
{
	//Spink_Cam *scp;

	//scp = (Spink_Cam *)ip;

	warn("get_spink_cam_prec_name:  need to implement spink_cam-state-based value!?");

	//return def_prec_name(QSP_ARG  ip);
	return("u_byte");
}


#ifdef FOOBAR
static Size_Functions spink_cam_sf={
	get_spink_cam_size,
	get_spink_cam_prec_name

};
#endif // FOOBAR


void _report_spink_error(QSP_ARG_DECL  spinError error, const char *whence )
{
	const char *msg;

	switch(error){
		case SPINNAKER_ERR_SUCCESS:
			msg = "Success"; break;
		case SPINNAKER_ERR_ERROR:
			msg = "Error"; break;
		case SPINNAKER_ERR_NOT_INITIALIZED:
			msg = "Not initialized"; break;
		case SPINNAKER_ERR_NOT_IMPLEMENTED:
			msg = "Not implemented"; break;
		case SPINNAKER_ERR_RESOURCE_IN_USE:
			msg = "Resource in use"; break;
		case SPINNAKER_ERR_ACCESS_DENIED:
			msg = "Access denied"; break;
		case SPINNAKER_ERR_INVALID_HANDLE:
			msg = "Invalid handle"; break;
		case SPINNAKER_ERR_INVALID_ID:
			msg = "Invalid ID"; break;
		case SPINNAKER_ERR_NO_DATA:
			msg = "No data"; break;
		case SPINNAKER_ERR_INVALID_PARAMETER:
			msg = "Invalid parameter"; break;
		case SPINNAKER_ERR_IO:
			msg = "I/O error"; break;
		case SPINNAKER_ERR_TIMEOUT:
			msg = "Timeout"; break;
		case SPINNAKER_ERR_ABORT:
			msg = "Abort"; break;
		case SPINNAKER_ERR_INVALID_BUFFER:
			msg = "Invalid buffer"; break;
		case SPINNAKER_ERR_NOT_AVAILABLE:
			msg = "Not available"; break;
		case SPINNAKER_ERR_INVALID_ADDRESS:
			msg = "Invalid address"; break;
		case SPINNAKER_ERR_BUFFER_TOO_SMALL:
			msg = "Buffer too small"; break;
		case SPINNAKER_ERR_INVALID_INDEX:
			msg = "Invalid index"; break;
		case SPINNAKER_ERR_PARSING_CHUNK_DATA:
			msg = "Chunk data parsing error"; break;
		case SPINNAKER_ERR_INVALID_VALUE:
			msg = "Invalid value"; break;
		case SPINNAKER_ERR_RESOURCE_EXHAUSTED:
			msg = "Resource exhausted"; break;
		case SPINNAKER_ERR_OUT_OF_MEMORY:
			msg = "Out of memory"; break;
		case SPINNAKER_ERR_BUSY:
			msg = "Busy"; break;

		case GENICAM_ERR_INVALID_ARGUMENT:
			msg = "genicam invalid argument"; break;
		case GENICAM_ERR_OUT_OF_RANGE:
			msg = "genicam range error"; break;
		case GENICAM_ERR_PROPERTY:
			msg = "genicam property error"; break;
		case GENICAM_ERR_RUN_TIME:
			msg = "genicam run time error"; break;
		case GENICAM_ERR_LOGICAL:
			msg = "genicam logical error"; break;
		case GENICAM_ERR_ACCESS:
			msg = "genicam access error"; break;
		case GENICAM_ERR_TIMEOUT:
			msg = "genicam timeout error"; break;
		case GENICAM_ERR_DYNAMIC_CAST:
			msg = "genicam dynamic cast error"; break;
		case GENICAM_ERR_GENERIC:
			msg = "genicam generic error"; break;
		case GENICAM_ERR_BAD_ALLOCATION:
			msg = "genicam bad allocation"; break;

		case SPINNAKER_ERR_IM_CONVERT:
			msg = "image conversion error"; break;
		case SPINNAKER_ERR_IM_COPY:
			msg = "image copy error"; break;
		case SPINNAKER_ERR_IM_MALLOC:
			msg = "image malloc error"; break;
		case SPINNAKER_ERR_IM_NOT_SUPPORTED:
			msg = "image operation not supported"; break;
		case SPINNAKER_ERR_IM_HISTOGRAM_RANGE:
			msg = "image histogram range error"; break;
		case SPINNAKER_ERR_IM_HISTOGRAM_MEAN:
			msg = "image histogram mean error"; break;
		case SPINNAKER_ERR_IM_MIN_MAX:
			msg = "image min/max error"; break;
		case SPINNAKER_ERR_IM_COLOR_CONVERSION:
			msg = "image color conversion error"; break;

//		case SPINNAKER_ERR_CUSTOM_ID = -10000

		default:
			sprintf(ERROR_STRING,
		"report_spink_error (%s):  unhandled error code %d!?\n",
				whence,error);
			warn(ERROR_STRING);
			msg = "unhandled error code";
			break;
	}
	sprintf(ERROR_STRING,"%s:  %s",whence,msg);
	warn(ERROR_STRING);
}

ITEM_INTERFACE_DECLARATIONS(Spink_Cam_Property_Type,pgr_prop,0)

//  When we change spink_cams, we have to refresh all properties!

static void _init_one_property(QSP_ARG_DECL const char *name, spinkPropertyType t)
{
	Spink_Cam_Property_Type *pgpt;

	pgpt = new_pgr_prop(name);
	if( pgpt == NULL ) return;
//	pgpt->info.type =
//	pgpt->prop.type =
//	pgpt->type_code = t;
}

void list_spink_cam_properties(QSP_ARG_DECL  Spink_Cam *scp)
{
	List *lp;
	Node *np;
	//Spink_Cam_Property_Type *pgpt;

	lp = pgr_prop_list();	// all properties
	np = QLIST_HEAD(lp);
	if( np != NULL ){
		sprintf(MSG_STR,"\n%s properties",scp->sk_name);
		prt_msg(MSG_STR);
	} else {
		sprintf(ERROR_STRING,"%s has no properties!?",scp->sk_name);
		warn(ERROR_STRING);
		return;
	}

	while(np!=NULL){
	//	pgpt = (Spink_Cam_Property_Type *)NODE_DATA(np);
		/*
		if( pgpt->info.present ){
			sprintf(MSG_STR,"\t%s",pgpt->name);
			prt_msg(MSG_STR);
		}
		*/
		np = NODE_NEXT(np);
	}
	prt_msg("");
}

// We call this after we select a spink_cam

void refresh_spink_cam_properties(QSP_ARG_DECL  Spink_Cam *scp)
{
	List *lp;
	Node *np;
	Spink_Cam_Property_Type *pgpt;

	lp = pgr_prop_list();	// all properties
	np = QLIST_HEAD(lp);
	while(np!=NULL){
		pgpt = (Spink_Cam_Property_Type *)NODE_DATA(np);
		refresh_property_info(QSP_ARG  scp, pgpt );
		/*
		if( pgpt->info.present ){
			refresh_property_value(QSP_ARG  scp, pgpt );
		}
		*/
		np = NODE_NEXT(np);
	}
}

#define init_one_property(n,t)	_init_one_property(QSP_ARG  n, t)

static void init_property_types(SINGLE_QSP_ARG_DECL)
{
	/*
	init_one_property( "brightness",	SPINK_BRIGHTNESS		);
	init_one_property( "auto_exposure",	SPINK_AUTO_EXPOSURE	);
	init_one_property( "sharpness",		SPINK_SHARPNESS		);
	init_one_property( "white_balance",	SPINK_WHITE_BALANCE	);
	init_one_property( "hue",		SPINK_HUE			);
	init_one_property( "saturation",	SPINK_SATURATION		);
	init_one_property( "gamma",		SPINK_GAMMA		);
	init_one_property( "iris",		SPINK_IRIS		);
	init_one_property( "focus",		SPINK_FOCUS		);
	init_one_property( "zoom",		SPINK_ZOOM		);
	init_one_property( "pan",		SPINK_PAN			);
	init_one_property( "tilt",		SPINK_TILT		);
	init_one_property( "shutter",		SPINK_SHUTTER		);
	init_one_property( "gain",		SPINK_GAIN		);
	init_one_property( "trigger_mode",	SPINK_TRIGGER_MODE	);
	init_one_property( "trigger_delay",	SPINK_TRIGGER_DELAY	);
	init_one_property( "frame_rate",	SPINK_FRAME_RATE		);
	init_one_property( "temperature",	SPINK_TEMPERATURE		);
	*/
}

void refresh_property_info(QSP_ARG_DECL  Spink_Cam *scp, Spink_Cam_Property_Type *pgpt )
{
	//spinkError error;

	/*
	error = spinkGetPropertyInfo( scp->sk_context, &(pgpt->info) );
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetPropertyInfo" );
		return;
	}
	*/
}

void show_property_info(QSP_ARG_DECL  Spink_Cam *scp, Spink_Cam_Property_Type *pgpt )
{
	//char var_name[32],val_str[32];

	sprintf(MSG_STR,"\n%s %s info:",scp->sk_name,pgpt->name);
	prt_msg(MSG_STR);

#ifdef FOOBAR
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
#endif // FOOBAR
}

void refresh_property_value(QSP_ARG_DECL  Spink_Cam *scp, Spink_Cam_Property_Type *pgpt )
{
	/*
	spinkError error;

	error = spinkGetProperty( scp->sk_context, &(pgpt->prop) );
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetProperty" );
		return;
	}
	*/
}

void show_property_value(QSP_ARG_DECL  Spink_Cam *scp, Spink_Cam_Property_Type *pgpt )
{
	sprintf(MSG_STR,"\n%s %s:",
		scp->sk_name,pgpt->name);
	prt_msg(MSG_STR);

#ifdef FOOBAR
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
	// the value back from the spink_cam???
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
#endif // FOOBAR
}

void set_prop_value(QSP_ARG_DECL  Spink_Cam *scp, Spink_Cam_Property_Type *pgpt, Spink_Cam_Prop_Val *vp )
{
#ifdef FOOBAR
	spinkError error;

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

	error = spinkSetProperty( scp->sk_context, &(pgpt->prop));
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkSetProperty" );
		return;
	}
#endif // FOOBAR
}

void set_prop_auto(QSP_ARG_DECL  Spink_Cam *scp, Spink_Cam_Property_Type *pgpt, BOOL yn )
{
#ifdef FOOBAR
	spinkError error;

	if( ! pgpt->info.autoSupported ){
		sprintf(ERROR_STRING,"Sorry, auto mode not supported for %s.",
			pgpt->name);
		warn(ERROR_STRING);
		return;
	}

	pgpt->prop.autoManualMode = yn;
	error = spinkSetProperty( scp->sk_context, &(pgpt->prop) );
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkSetProperty" );
		return;
	}
#endif // FOOBAR
}

static void insure_stopped(QSP_ARG_DECL  Spink_Cam *scp, const char *op_desc)
{
	if( (scp->sk_flags & FLY_CAM_IS_RUNNING) == 0 ) return;

	sprintf(ERROR_STRING,"Stopping capture on %s prior to %s",
		scp->sk_name,op_desc);
	advise(ERROR_STRING);

	stop_firewire_capture(QSP_ARG  scp);
}

void
cleanup_spink_cam( Spink_Cam *scp )
{
	//if( IS_CAPTURING(scp) )
		 //dc1394_capture_stop( scp->sk_cam_p );
		 //fly_capture_stop( scp->sk_cam_p );
	//if( IS_TRANSMITTING(scp) )
		//dc1394_video_set_transmission( scp->sk_cam_p, DC1394_OFF );
		//fly_video_set_transmission( scp->sk_cam_p, DC1394_OFF );
	/* dc1394_free_spink_cam */
	//dc1394_spink_cam_free( scp->sk_cam_p );
	//fly_spink_cam_free( scp->sk_cam_p );
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

INDEX_SEARCH(video_mode,spinkVideoMode,N_NAMED_VIDEO_MODES,nvm)
INDEX_SEARCH(framerate,spinkFrameRate,N_NAMED_FRAMERATES,nfr)
INDEX_SEARCH(grab_mode,spinkGrabMode,N_NAMED_GRAB_MODES,ngm)
INDEX_SEARCH(bus_speed,spinkBusSpeed,N_NAMED_BUS_SPEEDS,nbs)
INDEX_SEARCH(bw_allocation,spinkBandwidthAllocation,N_NAMED_BW_ALLOCATIONS,nba)
INDEX_SEARCH(interface,spinkInterfaceType,N_NAMED_INTERFACES,nif)
#endif // FOOBAR

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

/*
NAME_LOOKUP_FUNC(video_mode,spinkVideoMode,nvm)
NAME_LOOKUP_FUNC(framerate,spinkFrameRate,nfr)
NAME_LOOKUP_FUNC(grab_mode,spinkGrabMode,ngm)
NAME_LOOKUP_FUNC(bus_speed,spinkBusSpeed,nbs)
NAME_LOOKUP_FUNC(bw_allocation,spinkBandwidthAllocation,nba)
NAME_LOOKUP_FUNC(interface,spinkInterfaceType,nif)
*/

int get_spink_cam_names( QSP_ARG_DECL  Data_Obj *str_dp )
{
	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?
	List *lp;
	Node *np;
	Spink_Cam *scp;
	int i, n;

	lp = spink_cam_list();
	if( lp == NULL ){
		warn("No spink_cams!?");
		return 0;
	}

	n=eltcount(lp);
	if( OBJ_COLS(str_dp) < n ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d spink_cam names",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),n);
		warn(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	}
		
	np=QLIST_HEAD(lp);
	i=0;
	while(np!=NULL){
		char *dst;
		scp = (Spink_Cam *) NODE_DATA(np);
		dst = OBJ_DATA_PTR(str_dp);
		dst += i * OBJ_PXL_INC(str_dp);
		if( strlen(scp->sk_name)+1 > OBJ_COMPS(str_dp) ){
			sprintf(ERROR_STRING,"String object %s has too few components (%ld) to hold spink_cam name \"%s\"",
				OBJ_NAME(str_dp),(long)OBJ_COMPS(str_dp),scp->sk_name);
			warn(ERROR_STRING);
		} else {
			strcpy(dst,scp->sk_name);
		}
		i++;
		if( i>=n )
			np=NULL;
		else
			np = NODE_NEXT(np);
	}

	return i;
}

int get_spink_cam_video_mode_strings( QSP_ARG_DECL  Data_Obj *str_dp, Spink_Cam *scp )
{
	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?

	int i, n;

	if( OBJ_COLS(str_dp) < scp->sk_n_video_modes ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d modes",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),scp->sk_n_video_modes);
		warn(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	} else {
		n=scp->sk_n_video_modes;
	}
		
	for(i=0;i<n;i++){
		int k;
		const char *src;
		char *dst;

		k=scp->sk_video_mode_indices[i];
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

static void get_framerate_choices(QSP_ARG_DECL  Spink_Cam *scp)
{
	unsigned int i,n,idx;
	Framerate_Mask mask;

	if( scp->sk_framerate_names != NULL ){
		givbuf(scp->sk_framerate_names);
		scp->sk_framerate_names=NULL;	// in case we have an error before finishing
	}

	// format7 doesn't have a framerate!?

	mask = scp->sk_framerate_mask_tbl[ scp->sk_my_video_mode_index ];
/*
sprintf(ERROR_STRING,"%s:  video mode is %s",
scp->sk_name,all_video_modes[scp->sk_video_mode_index].nvm_name);
advise(ERROR_STRING);
sprintf(ERROR_STRING,"%s:  my video mode %s (index = %d)",
scp->sk_name,scp->sk_video_mode_names[scp->sk_my_video_mode_index],
scp->sk_my_video_mode_index);
advise(ERROR_STRING);
*/

	n = bit_count(mask);
	if( n <= 0 ){
		// this happens for the format7 video mode...
		// Can this ever happen?  If not, should be CAUTIOUS...
		//warn("no framerates for this video mode!?");
#ifdef FOOBAR
		if( scp->sk_video_mode == SPINK_VIDEOMODE_FORMAT7 ){
			if( scp->sk_framerate == SPINK_FRAMERATE_FORMAT7 ){
				advise("No framerates available for format7.");
			}
			  else {
				assert(0);
			}
		}
		  else {
			assert(0);
		}
#endif // FOOBAR
		return;
	}

	scp->sk_framerate_names = getbuf( n * sizeof(char *) );
	scp->sk_n_framerates = n ;

	i=(-1);
	idx=0;
	while(mask){
		i++;
		if( mask & 1 ){
#ifdef FOOBAR
			spinkFrameRate r;
			int j;
			r = i ;
			j = index_of_framerate( r );
			scp->sk_framerate_names[idx]=all_framerates[j].nfr_name;
#endif // FOOBAR
			idx++;
		}
			
		mask >>= 1;
	}

	assert( idx == n );
}

int get_spink_cam_framerate_strings( QSP_ARG_DECL  Data_Obj *str_dp, Spink_Cam *scp )
{
	int i, n;
	const char *src;
	char *dst;

	get_framerate_choices(QSP_ARG  scp);

	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?

	n = scp->sk_n_framerates;

	if( OBJ_COLS(str_dp) < n ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d framerates",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),n);
		warn(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	}

	for(i=0;i<scp->sk_n_framerates;i++){
		src = scp->sk_framerate_names[i];
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

#ifdef FOOBAR
static void test_setup(QSP_ARG_DECL  Spink_Cam *scp,
		spinkVideoMode m, spinkFrameRate r, int *kp, int idx )
{
	spinkError error;
	BOOL supported;

	// punt if format 7
	if( m == SPINK_VIDEOMODE_FORMAT7 && r == SPINK_FRAMERATE_FORMAT7 ){
		// BUG?  make sure spink_cam has format7 modes?
		supported = TRUE;
	} else {
		error = spinkGetVideoModeAndFrameRateInfo( scp->sk_context,
			m,r,&supported );
		if( error != SPINK_ERROR_OK ){
			report_spink_error(QSP_ARG  error,
				"spinkGetVideoModeAndFrameRateInfo" );
			supported = FALSE;
		}
	}

	if( supported ){
		if( *kp < 0 || scp->sk_video_mode_indices[*kp] != idx ){
			*kp = (*kp)+1;
			scp->sk_video_mode_indices[*kp] = idx;
			scp->sk_video_mode_names[*kp] = all_video_modes[idx].nvm_name;
//fprintf(stderr,"test_setup:  adding video mode %s to %s\n",
//all_video_modes[idx].nvm_name,
//scp->sk_name);
			if( scp->sk_video_mode == m )
				scp->sk_video_mode_index = *kp;
		}
		scp->sk_framerate_mask_tbl[*kp] |= 1 << r;
	}
}
#endif // FOOBAR

static int get_supported_video_modes(QSP_ARG_DECL  Spink_Cam *scp )
{
	int i, /*j,*/ n_so_far;

	n_so_far=(-1);
	//for( i=0;i<N_STD_VIDEO_MODES;i++){
	for( i=0;i<N_NAMED_VIDEO_MODES;i++){
#ifdef FOOBAR
		spinkVideoMode m;
		spinkFrameRate r;

		m=all_video_modes[i].nvm_value;
		if( m == SPINK_VIDEOMODE_FORMAT7 ){
			r = SPINK_FRAMERATE_FORMAT7;
			test_setup(QSP_ARG  scp, m, r, &n_so_far, i );
		} else {
			for(j=0;j<N_STD_FRAMERATES;j++){
				r=all_framerates[j].nfr_value;
//fprintf(stderr,"Testing video mode %d and frame rate %d\n",
//m,scp->sk_framerate);
				test_setup(QSP_ARG  scp, m, r, &n_so_far, i );
			}
		}
#endif // FOOBAR
	}
	n_so_far++;
//fprintf(stderr,"get_supported_video_modes:  setting n_video_modes to %d\n",k);
	scp->sk_n_video_modes = n_so_far;
	return 0;
}

#ifdef FOOBAR
static spinkFrameRate highest_framerate( QSP_ARG_DECL  Framerate_Mask mask )
{
	int k;

	k=(-1);
	while( mask ){
		mask >>= 1;
		k++;
	}
	assert( k >= 0 );

	return (spinkFrameRate) k;
}
#endif // FOOBAR

int is_fmt7_mode(QSP_ARG_DECL  Spink_Cam *scp, int idx )
{
#ifdef FOOBAR
	spinkVideoMode m;

//	CHECK_IDX(is_fmt7_mode)
	assert( idx >= 0 && idx < scp->sk_n_video_modes );

	m = all_video_modes[ scp->sk_video_mode_indices[idx] ].nvm_value;
	if( m == SPINK_VIDEOMODE_FORMAT7 ) return 1;
#endif // FOOBAR
	return 0;
}

// We might want to call this after changing the video mode - we
// don't know whether the library might change anything (like the
// number of buffers, but it seems possible?

static int refresh_config(QSP_ARG_DECL  Spink_Cam *scp)
{
#ifdef FOOBAR
	spinkError error;

	error = spinkGetConfiguration(scp->sk_context,&scp->sk_config);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetConfiguration" );
		// should we set a flag to indicate an invalid config?
		return -1;
	}
	scp->sk_n_buffers = scp->sk_config.numBuffers;
#endif // FOOBAR
	return 0;
}


int set_std_mode(QSP_ARG_DECL  Spink_Cam *scp, int idx )
{
#ifdef FOOBAR
	spinkFrameRate r;
	spinkError error;
	spinkVideoMode m;

//	CHECK_IDX(set_std_mode)
	assert( idx >= 0 && idx < scp->sk_n_video_modes );

	insure_stopped(QSP_ARG  scp,"setting video mode");

	m = all_video_modes[ scp->sk_video_mode_indices[idx] ].nvm_value;
	if( m == SPINK_VIDEOMODE_FORMAT7 ){
		warn("set_std_mode:  use set_fmt7_mode to select format7!?");
		return -1;
	}

	r = highest_framerate(QSP_ARG  scp->sk_framerate_mask_tbl[idx] );

	error = spinkSetVideoModeAndFrameRate(scp->sk_context,m,r);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkSetVideoModeAndFrameRate" );
		return -1;
	}

	scp->sk_my_video_mode_index = idx;
	scp->sk_video_mode = m;
	scp->sk_framerate = r;
	scp->sk_base = NULL;	// force init_spink_base to run again
	scp->sk_video_mode_index = scp->sk_video_mode_indices[idx];
	scp->sk_framerate_index = index_of_framerate(r);

	scp->sk_cols = all_video_modes[ scp->sk_video_mode_index ].nvm_width;
	scp->sk_rows = all_video_modes[ scp->sk_video_mode_index ].nvm_height;
	scp->sk_depth = all_video_modes[ scp->sk_video_mode_index ].nvm_depth;

#endif // FOOBAR
	return refresh_config(QSP_ARG  scp);
} // set_std_mode

static void set_highest_fmt7_framerate( QSP_ARG_DECL  Spink_Cam *scp )
{
	/* If the frame rate has been set to 60 fps by using a default
	 * video mode at startup, it will not be reset when we switch
	 * to format7.  So here we go to the max by default...
	 */

#ifdef FOOBAR
	spinkError error;
	spinkProperty prop;
	spinkPropertyInfo propInfo;

	propInfo.type = SPINK_FRAME_RATE;
	error = spinkGetPropertyInfo( scp->sk_context, &propInfo );
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetPropertyInfo" );
		return;
	}
	Spink_Cam_Property_Type *fr_prop_p;

	fr_prop_p = get_pgr_prop("frame_rate" );	// BUG string must match table

	assert( fr_prop_p != NULL );

	assert( fr_prop_p->info.absValSupported );

	prop.type = SPINK_FRAME_RATE;
	error = spinkGetProperty( scp->sk_context, &prop);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetProperty" );
		return;
	}
	prop.absControl = TRUE;
	prop.absValue = fr_prop_p->info.absMax;

	error = spinkSetProperty( scp->sk_context, &prop);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkSetProperty" );
		return;
	}
#endif // FOOBAR
}

int set_fmt7_mode(QSP_ARG_DECL  Spink_Cam *scp, int idx )
{
#ifdef FOOBAR
	spinkFormat7ImageSettings settings;
//	spinkFormat7PacketInfo pinfo;
//	BOOL is_valid;
	spinkError error;
	unsigned int packetSize;
	float percentage;

	insure_stopped(QSP_ARG  scp,"setting format7 mode");

	if( idx < 0 || idx >= scp->sk_n_fmt7_modes ){
		warn("Format 7 index out of range!?");
		return -1;
	}

	settings.mode = idx;
	settings.offsetX = 0;
	settings.offsetY = 0;
	settings.width = scp->sk_fmt7_info_tbl[idx].maxWidth;
	settings.height = scp->sk_fmt7_info_tbl[idx].maxHeight;
	if( scp->sk_fmt7_info_tbl[idx].pixelFormatBitField &
			SPINK_PIXEL_FORMAT_RAW8 )
		settings.pixelFormat = SPINK_PIXEL_FORMAT_RAW8;
	else {
		warn("Camera does not support raw8!?");
		return -1;
	}

fprintf(stderr,"Using size %d x %d\n",settings.width,settings.height);

	percentage = 100.0;
	error = spinkSetFormat7Configuration(scp->sk_context,&settings,
			percentage);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkSetFormat7Configuration" );
		return -1;
	}

	set_highest_fmt7_framerate(QSP_ARG  scp);

	// This fails if we are not in format 7 already.
	error = spinkGetFormat7Configuration(scp->sk_context,&settings,
			&packetSize,&percentage);

	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetFormat7Configuration" );
		return -1;
	}

	fprintf(stderr,"Percentage = %g (packet size = %d)\n",
		percentage,packetSize);

	scp->sk_video_mode = SPINK_VIDEOMODE_FORMAT7;
	scp->sk_framerate = SPINK_FRAMERATE_FORMAT7;
	scp->sk_framerate_index = index_of_framerate(SPINK_FRAMERATE_FORMAT7);
	scp->sk_video_mode_index = index_of_video_mode(SPINK_VIDEOMODE_FORMAT7);
	scp->sk_my_video_mode_index = (-1);
	scp->sk_fmt7_index = idx;
	scp->sk_base = NULL;	// force init_spink_base to run again

	// Do we have to set the framerate to SPINK_FRAMERATE_FORMAT7???

	scp->sk_rows = settings.height;
	scp->sk_cols = settings.width;

	{
		long bytes_per_image;
		float est_fps;

		bytes_per_image = settings.width * settings.height;
		// assumes mono8

		est_fps = 8000.0 * packetSize / bytes_per_image;
		fprintf(stderr,"Estimated frame rate:  %g\n",est_fps);
	}

	// refresh_config reads the config from the library...
#endif // FOOBAR
	return refresh_config(QSP_ARG  scp);
} // set_fmt7_mode

void set_eii_property(QSP_ARG_DECL  Spink_Cam *scp, int idx, int yesno )
{
#ifdef FOOBAR
	spinkError error;
	myEmbeddedImageInfo *eii_p;


	eii_p = (myEmbeddedImageInfo *) (&scp->sk_ei_info);
	if( ! eii_p->prop_tbl[idx].available ){
		sprintf(ERROR_STRING,"Property %s is not available on %s",
			eii_prop_names[idx],scp->sk_name);
		warn(ERROR_STRING);
		return;
	}
	eii_p->prop_tbl[idx].onOff = yesno ? TRUE : FALSE;

	error = spinkSetEmbeddedImageInfo(scp->sk_context,&scp->sk_ei_info);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkSetEmbeddedImageInfo" );
	}
#endif // FOOBAR
}

void set_grab_mode(QSP_ARG_DECL  Spink_Cam *scp, int grabmode_idx )
{
#ifdef FOOBAR
	spinkError error;
	spinkConfig cfg;

	assert( grabmode_idx >= 0 && grabmode_idx < N_NAMED_GRAB_MODES );

	// grab mode is part of the config struct
	cfg = scp->sk_config;
	cfg.grabMode = all_grab_modes[grabmode_idx].ngm_value;
	error = spinkSetConfiguration(scp->sk_context,&cfg);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkSetConfiguration" );
		// should we set a flag to indicate an invalid config?
		return;
	}
	scp->sk_config.grabMode = cfg.grabMode;
#endif // FOOBAR
}

void show_grab_mode(QSP_ARG_DECL  Spink_Cam *scp)
{
#ifdef FOOBAR
	int idx;

	idx = index_of_grab_mode(scp->sk_config.grabMode);
	if( idx < 0 ) return;
	sprintf(MSG_STR,"Current grab mode:  %s",all_grab_modes[idx].ngm_name);
	prt_msg(MSG_STR);
#endif // FOOBAR
}

int pick_spink_cam_framerate(QSP_ARG_DECL  Spink_Cam *scp, const char *pmpt)
{
	int i;

	if( scp == NULL ){
		sprintf(ERROR_STRING,"pick_spink_cam_framerate:  no spink_cam selected!?");
		warn(ERROR_STRING);
		return -1;
	}

	get_framerate_choices(QSP_ARG  scp);
	i=WHICH_ONE(pmpt,scp->sk_n_framerates,scp->sk_framerate_names);
	return i;
}

int set_framerate(QSP_ARG_DECL  Spink_Cam *scp, int framerate_index)
{
#ifdef FOOBAR
	spinkFrameRate rate;
	spinkError error;

	if( scp == NULL ) return -1;

	insure_stopped(QSP_ARG  scp,"setting frame rate");

	rate = all_framerates[framerate_index].nfr_value;
	error = spinkSetVideoModeAndFrameRate(scp->sk_context,
			scp->sk_video_mode,rate);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkSetVideoModeAndFrameRate" );
		return -1;
	}

	scp->sk_framerate = rate;

#endif // FOOBAR
	return refresh_config(QSP_ARG  scp);
}

void show_spink_cam_framerate(QSP_ARG_DECL  Spink_Cam *scp)
{
	sprintf(MSG_STR,"%s framerate:  %s",
		scp->sk_name,all_framerates[scp->sk_framerate_index].nfr_name);
	advise(MSG_STR);
}

void show_spink_cam_video_mode(QSP_ARG_DECL  Spink_Cam *scp)
{
#ifdef FOOBAR
	sprintf(MSG_STR,"%s video mode:  %s",
		scp->sk_name,name_for_video_mode(scp->sk_video_mode));
	advise(MSG_STR);
#endif // FOOBAR
}

int list_spink_cam_video_modes(QSP_ARG_DECL  Spink_Cam *scp)
{
	unsigned int i;
	const char *s;

	if( scp->sk_n_video_modes <= 0 ){
		warn("no video modes!?");
		return -1;
	}

	for(i=0;i<scp->sk_n_video_modes; i++){
		s=scp->sk_video_mode_names[i];
		prt_msg_frag("\t");
		prt_msg(s);
	}
	return 0;
}

void list_spink_cam_framerates(QSP_ARG_DECL  Spink_Cam *scp)
{
#ifdef FOOBAR
	int i;

	get_framerate_choices(QSP_ARG  scp);

	for(i=0;i<scp->sk_n_framerates;i++){
		prt_msg_frag("\t");
		prt_msg(scp->sk_framerate_names[i]);
	}
#endif // FOOBAR
}

static int set_default_video_mode(QSP_ARG_DECL  Spink_Cam *scp)
{
#ifdef FOOBAR
	spinkVideoMode m;
	spinkFrameRate r;
	int i,j;
	spinkError error;
	int _nskip;

	if( get_supported_video_modes(QSP_ARG  scp ) < 0 ){
		warn("set_default_video_mode:  Can't get video modes");
		return -1;
	}

//fprintf(stderr,"get_supported_video_modes found %d modes\n",scp->sk_n_video_modes);
	_nskip=0;
	do {
		_nskip++;
		i = scp->sk_n_video_modes-(_nskip);	// order the table so that mono8 is last?
		scp->sk_my_video_mode_index = i;
		j = scp->sk_video_mode_indices[i];
		m = all_video_modes[j].nvm_value;
		// BUG we don't check that nskip is in-bounds, but should be OK
	} while( m == SPINK_VIDEOMODE_FORMAT7  && _nskip < scp->sk_n_video_modes );

	if( m == SPINK_VIDEOMODE_FORMAT7 ){
		/*
		sprintf(ERROR_STRING,"set_default_video_mode:  %s has only format7 modes!?",
			scp->sk_name);
		advise(ERROR_STRING);
		*/
		return -1;
	}

//fprintf(stderr,"_nskip = %d, i = %d,  j = %d,  m = %d\n",_nskip,i,j,m);

//fprintf(stderr,"highest non-format7 video mode is %s\n",all_video_modes[j].nvm_name);

	// if this is format7, don't use it!?

	scp->sk_video_mode = m;
	scp->sk_video_mode_index = j;

	scp->sk_cols = all_video_modes[ j ].nvm_width;
	scp->sk_rows = all_video_modes[ j ].nvm_height;
	scp->sk_depth = all_video_modes[ j ].nvm_depth;

	// Get the hightest frame rate associated with this video mode...
	r = highest_framerate( QSP_ARG  scp->sk_framerate_mask_tbl[i] );

//fprintf(stderr,"mode %s, highest supported frame rate is %s\n",
//name_for_video_mode(m),
//name_for_framerate(r));
	scp->sk_framerate = r;
	scp->sk_framerate_index = index_of_framerate(r);

//sprintf(ERROR_STRING,"set_default_video_mode:  setting to %s", name_for_video_mode(m));
//advise(ERROR_STRING);


	error = spinkSetVideoModeAndFrameRate( scp->sk_context,
			scp->sk_video_mode, scp->sk_framerate );
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkSetVideoModeAndFrameRate" );
		return -1;
	}

	// We might stash the number of video modes here?

	// stash the number of framerates in a script variable
	// in case the user wants to fetch the strings...
	// BUG DO THIS WHEN CAM IS SELECTED!
	//set_script_var_from_int(QSP_ARG
	//		"n_framerates",scp->sk_framerates.num);

sprintf(ERROR_STRING,"%s:  %s, %s fps",scp->sk_name,name_for_video_mode(m),
						name_for_framerate(r) );
advise(ERROR_STRING);

#endif // FOOBAR
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

static Spink_Cam *unique_spink_cam_instance( QSP_ARG_DECL  spinkContext context )
{
#ifdef FOOBAR
	int i;
	char cname[80];	// How many chars is enough?
	Spink_Cam *scp;
	spinkError error;
	spinkCameraInfo camInfo;

	error = spinkGetCameraInfo( context, &camInfo );
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetCameraInfo" );
		return NULL;
	}

	i=1;
	scp=NULL;
	while(scp==NULL){
		//sprintf(cname,"%s_%d",cam_p->model,i);
		sprintf(cname,"%s_%d",camInfo.modelName,i);
		fix_string(cname);	// change spaces to underscores
//sprintf(ERROR_STRING,"Checking for existence of %s",cname);
//advise(ERROR_STRING);
		scp = spink_cam_of( cname );
		if( scp == NULL ){	// This index is free
//sprintf(ERROR_STRING,"%s is not in use",cname);
//advise(ERROR_STRING);
			scp = new_spink_cam( cname );
			if( scp == NULL ){
				sprintf(ERROR_STRING,
			"Failed to create spink_cam %s!?",cname);
				error1(ERROR_STRING);
			}
		} else {
//sprintf(ERROR_STRING,"%s IS in use",cname);
//advise(ERROR_STRING);
			scp = NULL;
		}
		i++;
		if( i>=5 ){
			error1("Too many spink_cams!?"); 
		}
	}
	scp->sk_cam_info = camInfo;
	return scp;
#else // FOOBAR
	return NULL;
#endif // FOOBAR
}

static void get_fmt7_modes(QSP_ARG_DECL  Spink_Cam *scp)
{
#ifdef FOOBAR
	spinkError error;
	BOOL supported;
	int i, largest=(-1);
	spinkFormat7Info fmt7_info_tbl[N_FMT7_MODES];
	size_t nb;

	scp->sk_n_fmt7_modes = 0;
	for(i=0;i<N_FMT7_MODES;i++){
		fmt7_info_tbl[i].mode = i;
		error = spinkGetFormat7Info(scp->sk_context,
				&fmt7_info_tbl[i],&supported);
		if( error != SPINK_ERROR_OK ){
			report_spink_error(QSP_ARG  error, "spinkGetFormat7Info" );
		}
		if( supported ){
			scp->sk_n_fmt7_modes ++ ;
			largest = i;
		}
	}
	if( (largest+1) != scp->sk_n_fmt7_modes ){
		sprintf(ERROR_STRING,
	"Unexpected number of format7 modes!?  (largest index = %d, n_modes = %d)",
			largest,scp->sk_n_fmt7_modes);
		warn(ERROR_STRING);
	}

	nb = scp->sk_n_fmt7_modes * sizeof(spinkFormat7Info);
	scp->sk_fmt7_info_tbl = getbuf( nb );
	memcpy(scp->sk_fmt7_info_tbl,fmt7_info_tbl,nb);

	scp->sk_fmt7_index = 0;

#endif // FOOBAR
}

#define SHOW_FIELD(desc_str,value)					\
									\
/*sprintf(MSG_STR,"\t%s:  %d",#desc_str,scp->sk_fmt7_info_tbl[mode].value);	\
prt_msg(MSG_STR);*/

#define SHOW_FIELD_HEX(desc_str,value)					\
									\
/*sprintf(MSG_STR,"\t%s:  0x%x",#desc_str,scp->sk_fmt7_info_tbl[mode].value); \
prt_msg(MSG_STR);*/

static void show_fmt7_info(QSP_ARG_DECL  Spink_Cam *scp, spinkMode mode )
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

void show_fmt7_modes(QSP_ARG_DECL  Spink_Cam *scp)
{
	if( scp->sk_n_fmt7_modes <= 0 ){
		prt_msg("\tNo format7 modes available.");
	} else {
		int i;
		for(i=0;i<scp->sk_n_fmt7_modes;i++)
			show_fmt7_info( QSP_ARG  scp, i );
	}
}

#ifdef FOOBAR
// This should only be called once...

static Spink_Cam *setup_my_spink_cam( QSP_ARG_DECL
				spinkContext context, spinkPGRGuid *guid_p,
				int index )
{
	Spink_Cam *scp;
	spinkError error;
	int i;

	error = spinkConnect(context,guid_p);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkConnect" );
		return NULL;
	}

	// We could have multiple instances of the same model...
	scp = unique_spink_cam_instance(QSP_ARG  context);

	// Why do we disconnect from the context we were called with -
	// subsequent cameras will still use it...
	// Maybe this just undoes the connection directly above?
	error = spinkDisconnect(context);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkDisconnect" );
		// BUG should clean up and return NULL?
		return scp;
	}

	//scp->sk_cam_p = cam_p;
	error = spinkCreateContext(&scp->sk_context);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkCreateContext" );
		// BUG clean up first
		return NULL;
	}
	error=spinkGetCameraFromIndex(scp->sk_context,index,&scp->sk_guid);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetCameraFromIndex" );
	}

	error = spinkConnect(scp->sk_context,&scp->sk_guid);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkConnect" );
		// BUG clean up first
		return NULL;
	}

	if( refresh_config(QSP_ARG  scp) < 0 )
		// BUG clean up first
		return NULL;

	error = spinkGetEmbeddedImageInfo(scp->sk_context,&scp->sk_ei_info);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetEmbeddedImageInfo" );
		// BUG clean up first
		return NULL;
	}

	scp->sk_feat_lp=NULL;
	scp->sk_in_use_lp=NULL;
	scp->sk_flags = 0;		/* assume no B-mode unless we are told otherwise... */
	scp->sk_base = NULL;

	get_fmt7_modes(QSP_ARG  scp);

	error = spinkGetVideoModeAndFrameRate( scp->sk_context,
			&scp->sk_video_mode, &scp->sk_framerate );
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetVideoModeAndFrameRate" );
		// BUG clean up first
		return NULL;
	}
/*
fprintf(stderr,"after fetching, video mode = %d (%s), frame rate = %d (%s)\n",
scp->sk_video_mode,
name_for_video_mode(scp->sk_video_mode),
scp->sk_framerate,
name_for_framerate(scp->sk_framerate)
);
*/

	// BUG?  could be N_STD_VIDEO_MODES?
	scp->sk_video_mode_indices = getbuf( sizeof(int) * N_NAMED_VIDEO_MODES );
	scp->sk_video_mode_names = getbuf( sizeof(char *) * N_NAMED_VIDEO_MODES );
	scp->sk_framerate_mask_tbl = getbuf( sizeof(Framerate_Mask) * N_NAMED_VIDEO_MODES );
	scp->sk_framerate_names = NULL;
	scp->sk_n_framerates = 0;

	for(i=0;i<N_NAMED_VIDEO_MODES;i++){
		scp->sk_framerate_mask_tbl[i] = 0;
	}

	/* Originally, we set the default video mode here to be
	 * the highest standard video mode...  But for the flea3,
	 * which has an odd-shaped sensor, the good modes are format7
	 * and for no good reason when we change back to format7
	 * under program control the high frame rate is not restored???
	 */

//#ifdef NOT_GOOD
	if( set_default_video_mode(QSP_ARG  scp) < 0 ){
		/*
		sprintf(ERROR_STRING,"error setting default video mode for %s",scp->sk_name);
		warn(ERROR_STRING);
		cleanup_spink_cam( scp );
		return(NULL);
		*/
		// This can fail for spink_cams that only support format7...
		// Deal with this better later
		sprintf(ERROR_STRING,"error setting default video mode for %s, only format7?",
			scp->sk_name);
		advise(ERROR_STRING);
	}
//#endif // NOT_GOOD

	/* used to set B-mode stuff here... */
	// What if the spink_cam is a usb cam???
	//dc1394_video_set_iso_speed( cam_p, DC1394_ISO_SPEED_400 );

	scp->sk_img_p = getbuf( sizeof(*scp->sk_img_p) );

        error = spinkCreateImage( scp->sk_img_p );
        if ( error != SPINK_ERROR_OK ) {
		report_spink_error(QSP_ARG  error, "spinkCreateImage" );
		// BUG clean up?
		//return NULL;
	}


	// Make a data_obj context for the frames...
	scp->sk_do_icp = create_dobj_context( QSP_ARG  scp->sk_name );

	scp->sk_frm_dp_tbl = NULL;
	scp->sk_newest = (-1);

	return(scp);
}
#endif // FOOBAR

void pop_spink_cam_context(SINGLE_QSP_ARG_DECL)
{
	// pop old context...
	Item_Context *icp;
	icp=pop_dobj_context();
	assert( icp != NULL );
}

void push_spink_cam_context(QSP_ARG_DECL  Spink_Cam *scp)
{
//fprintf(stderr,"pushing spink_cam context for %s (icp = 0x%lx)\n",
//scp->sk_name,(long)scp->sk_do_icp);
	push_dobj_context(scp->sk_do_icp);
}

static spinSystem hSystem = NULL;
static spinInterfaceList hInterfaceList = NULL;
static spinCameraList hCameraList = NULL;
static size_t numCameras = 0;
static size_t numInterfaces = 0;

#define release_spink_interface_structs()	_release_spink_interface_structs(SINGLE_QSP_ARG)

static int _release_spink_interface_structs(SINGLE_QSP_ARG_DECL)
{
	// iterate through the list
	Node *np;
	List *lp;
	Spink_Interface *ski_p;

	lp = spink_interface_list();
	if( lp == NULL ) return 0;
	np = QLIST_HEAD(lp);
	while(np!=NULL){
		ski_p = (Spink_Interface *) NODE_DATA(np);
		if( release_spink_interface(ski_p->ski_handle) < 0 )
			return -1;
		// could delete the struct here too!?!?
		np = NODE_NEXT(np);
	}
	return 0;
}

#define release_spink_cam_structs()	_release_spink_cam_structs(SINGLE_QSP_ARG)

static int _release_spink_cam_structs(SINGLE_QSP_ARG_DECL)
{
	// iterate through the list
	Node *np;
	List *lp;
	Spink_Cam *skc_p;

	lp = spink_cam_list();
	if( lp == NULL ) return 0;
	np = QLIST_HEAD(lp);
	while(np!=NULL){
		skc_p = (Spink_Cam *) NODE_DATA(np);
		if( release_spink_cam(skc_p->skc_handle) < 0 )
			return -1;
		// could delete the struct here too!?!?
		np = NODE_NEXT(np);
	}
	return 0;
}

void _release_spink_cam_system(SINGLE_QSP_ARG_DECL)
{
	assert( hSystem != NULL );
DEBUG_MSG(releast_spink_cam_system BEGIN)
	if( release_spink_interface_structs() < 0 ) return;
	if( release_spink_cam_structs() < 0 ) return;

	if( release_spink_cam_list(&hCameraList) < 0 ) return;
	if( release_spink_interface_list(&hInterfaceList) < 0 ) return;
	if( release_spink_system(hSystem) < 0 ) return;
DEBUG_MSG(releast_spink_cam_system DONE)
}

// Don't we already have this???

static void substitute_char(char *buf,char find, char replace)
{
	char *s;

	s=buf;
	while( *s ){
		if( *s == find )
			*s = replace;
		s++;
	}
}

#define create_spink_camera_structs() _create_spink_camera_structs(SINGLE_QSP_ARG)

static int _create_spink_camera_structs(SINGLE_QSP_ARG_DECL)
{
	int i;
	Spink_Cam *skc_p;
	char buf[MAX_BUFF_LEN];
	size_t len = MAX_BUFF_LEN;


	for(i=0;i<numCameras;i++){
		spinCamera hCam;
		spinNodeMapHandle hNodeMapTLDevice;

		if( get_spink_cam_from_list(&hCam,hCameraList,i) < 0 )
			return -1;

		if( get_spink_transport_level_map(&hNodeMapTLDevice,hCam) < 0 )
			return -1;

		get_camera_model_name(buf,len,hNodeMapTLDevice);
		substitute_char(buf,' ','_');
		skc_p = new_spink_cam(buf);

		skc_p->skc_handle = hCam;
		skc_p->skc_node_map_TL_dev = hNodeMapTLDevice;

		/*
		if( release_spink_interface(hInterface) < 0 )
			return -1;
			*/
	}
	return 0;
}

#define create_spink_interface_structs() _create_spink_interface_structs(SINGLE_QSP_ARG)

static int _create_spink_interface_structs(SINGLE_QSP_ARG_DECL)
{
	int i;
	Spink_Interface *ski_p;
	char buf[MAX_BUFF_LEN];
	size_t len = MAX_BUFF_LEN;


	for(i=0;i<numInterfaces;i++){
		spinInterface hInterface;
		// This call causes releaseSystem to crash!?
		if( get_spink_interface_from_list(&hInterface,hInterfaceList,i) < 0 )
			return -1;

		get_interface_name(buf,len,hInterface);
		substitute_char(buf,' ','_');
		ski_p = new_spink_interface(buf);

		ski_p->ski_handle = hInterface;

		/*
		if( release_spink_interface(hInterface) < 0 )
			return -1;
			*/
	}
	return 0;
}

int init_spink_cam_system(SINGLE_QSP_ARG_DECL)
{
#ifdef HAVE_LIBSPINNAKER
	assert( hSystem == NULL );

	if( get_spink_system(&hSystem) < 0 )
		return -1;

	if( get_spink_interfaces(hSystem,&hInterfaceList,&numInterfaces) < 0 ) return -1;
	if( create_spink_interface_structs() < 0 ) return -1;

	if( get_spink_cameras(hSystem,&hCameraList,&numCameras) < 0 ) return -1;
	if( create_spink_camera_structs() < 0 ) return -1;

	do_on_exit(_release_spink_cam_system);

#endif // HAVE_LIBSPINNAKER
	return 0;

#ifdef FOOBAR
	spinkVersion version;
	spinkContext context;
	spinkError error;
	spinkPGRGuid guid;	// BUG should be associated with one spink_cam?
	unsigned int numCameras=0;
	int i;
	static int firewire_system_inited=0;

	if( firewire_system_inited ){
		warn("Firewire system has already been initialized!?");
		return -1;
	}
	firewire_system_inited=1;
	init_property_types(SINGLE_QSP_ARG);

	spinkGetLibraryVersion(&version);
	sprintf(ERROR_STRING,"FlyCapture2 library version:  %d.%d.%d.%d",
		version.major,version.minor,version.type,version.build);
	advise(ERROR_STRING);

	// BUG?  the call to spinkCreateContext hangs if one is not logged in
	// on the console...  You don't need to RUN from the console,
	// but apparently something gets owned?

	error = spinkCreateContext(&context);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkCreateContext" );
		return -1;
	}

	error = spinkGetNumOfCameras(context,&numCameras);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetNumOfCameras" );
		// BUG? should we destroy the context here?
		return -1;
	}

	if( numCameras == 0 ){
		advise("No spink_cams detected.");
		return 0;
	}
	sprintf(ERROR_STRING,
		"%d spink_cam%s found.", numCameras, numCameras==1?"":"s" );
	advise(ERROR_STRING);


	//for(i=0;i<numCameras;i++)
	for(i=numCameras-1;i>=0;i--){
		error=spinkGetCameraFromIndex(context,i,&guid);
		if( error != SPINK_ERROR_OK ){
			report_spink_error(QSP_ARG  error, "spinkGetCameraFromIndex" );
		} else {
fprintf(stderr,"Calling setup_my_spink_cam for camera %d\n",i);
			setup_my_spink_cam(QSP_ARG   context, &guid, i );
		}
	}
	error = spinkDestroyContext( context );
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkDestroyContext" );
	}

	add_sizable(spink_cam_itp, &spink_cam_sf, (Item *(*)(QSP_ARG_DECL  const char *)) _spink_cam_of );

	return 0;
#endif // FOOBAR
}

#ifdef HAVE_LIBSPINNAKER

#ifdef FOOBAR
static void show_cam_info(QSP_ARG_DECL  spinkCameraInfo *cip)
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
#endif // FOOBAR

void show_n_buffers(QSP_ARG_DECL  Spink_Cam *scp)
{
	sprintf(MSG_STR,"%s:  %d buffers",scp->sk_name,scp->sk_n_buffers);
	prt_msg(MSG_STR);
}

int set_n_buffers(QSP_ARG_DECL  Spink_Cam *scp, int n )
{
#ifdef FOOBAR
	spinkConfig cfg;
	spinkError error;

fprintf(stderr,"set_n_buffers %s %d\n",scp->sk_name,n);
	if( n < MIN_N_BUFFERS || n > MAX_N_BUFFERS ){
		sprintf(ERROR_STRING,
"set_n_buffers:  number of buffers must be between %d and %d (%d requested)!?",
			MIN_N_BUFFERS,MAX_N_BUFFERS,n);
		warn(ERROR_STRING);
		return -1;
	}
	cfg = scp->sk_config;
	cfg.numBuffers = n;

	error = spinkSetConfiguration(scp->sk_context,&cfg);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkSetConfiguration" );
		// should we set a flag to indicate an invalid config?
		return -1;
	}
	scp->sk_n_buffers =
	scp->sk_config.numBuffers = n;
	scp->sk_base = NULL;	// force init_spink_base to run again

show_n_buffers(QSP_ARG  scp);
#endif // FOOBAR
	return 0;
}

#ifdef FOOBAR
static void show_cam_cfg(QSP_ARG_DECL  spinkConfig *cfp)
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
#endif // FOOBAR

#define SHOW_EI(member)					\
							\
	if( eip->member.available ){			\
		sprintf(MSG_STR,"%s:  %s",#member,	\
		eip->member.onOff ? "on" : "off" );	\
		prt_msg(MSG_STR);			\
	}

#ifdef FOOBAR
static void show_ei_info(QSP_ARG_DECL  spinkEmbeddedImageInfo *eip)
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
#endif // FOOBAR

#endif // HAVE_LIBSPINNAKER

void print_spink_cam_info(QSP_ARG_DECL  Spink_Cam *scp)
{
#ifdef HAVE_LIBSPINNAKER
#ifdef FOOBAR
	show_cam_info(QSP_ARG  &scp->sk_cam_info);
	show_cam_cfg(QSP_ARG  &scp->sk_config);
	show_ei_info(QSP_ARG  &scp->sk_ei_info);
#endif // FOOBAR
#endif // HAVE_LIBSPINNAKER

	/*
	i=index_of_framerate(scp->sk_framerate);
	sprintf(msg_str,"\tframe rate:  %s",all_framerates[i].nfr_name);
	prt_msg(msg_str);
	*/

	//report_spink_cam_features(scp);

	// show_fmt7_modes(QSP_ARG  scp);
	// Show the current video mode

	/*
	sprintf(MSG_STR,"Current video mode:  %s%s",
		scp->sk_my_video_mode_index >= 0 ?
		scp->sk_video_mode_names[ scp->sk_my_video_mode_index ] :
		"format7 mode ",
		scp->sk_my_video_mode_index >= 0 ? "": (
			scp->sk_fmt7_index == 0 ? "0" : (
			scp->sk_fmt7_index == 1 ? "1" : (
			scp->sk_fmt7_index == 2 ? "2" : "(>2)" )))
			);
	prt_msg(MSG_STR);

	sprintf(MSG_STR,"Current frame rate:  %s",
		all_framerates[ scp->sk_framerate_index ].nfr_name );
	prt_msg(MSG_STR);
	*/
}

static void init_one_frame(QSP_ARG_DECL  Spink_Cam *scp, int index )
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
#ifdef FOOBAR
		SET_DS_ROWS(&ds1,scp->sk_img_p->rows);
		SET_DS_COLS(&ds1,scp->sk_img_p->cols);
#endif // FOOBAR
		SET_DS_COMPS(&ds1,1);
		dp = _make_dp(QSP_ARG  fname,&ds1,PREC_FOR_CODE(PREC_UBY));
		assert( dp != NULL );

		SET_OBJ_DATA_PTR( dp, scp->sk_base+index*scp->sk_buf_delta );
		scp->sk_frm_dp_tbl[index] = dp;

//fprintf(stderr,"init_one_frame %d:  %s, data at 0x%lx\n",index,OBJ_NAME(dp),(long)OBJ_DATA_PTR(dp));
//		}
	} else {
		sprintf(ERROR_STRING,"init_one_frame:  object %s already exists!?",
			fname);
		warn(ERROR_STRING);
	}
} // end init_one_frame

static void init_cam_frames(QSP_ARG_DECL  Spink_Cam *scp)
{
	int index;

	assert( scp->sk_n_buffers > 0 );
	assert( scp->sk_frm_dp_tbl == NULL );

	scp->sk_frm_dp_tbl = getbuf( sizeof(Data_Obj) * scp->sk_n_buffers );
	for(index=0;index<scp->sk_n_buffers;index++)
		init_one_frame(QSP_ARG  scp, index);
} // init_cam_frames

// init_spink_base   -   grab frames to get the address
// associated with each frame index.  This wouldn't be necessary
// if we always provided the buffers, but we want this to work
// even when we don't.
//
// We keep track of the largest and smallest address, we save
// those so we can figure out the index of an arbitrary frame...

static void init_spink_base(QSP_ARG_DECL  Spink_Cam *scp)
{
#ifdef FOOBAR
	// initialize all the pointers
	int n_buffers_seen=0;
	void *smallest_addr;
	void *largest_addr;
	void *first_addr=NULL;
	void **addr_tbl;

	// sk_base is our flag, reset to NULL when number of buffers
	// is changed, or video mode is changed.
	if( scp->sk_base != NULL ){
		return;
	}

	n_buffers_seen = 0;
	addr_tbl = getbuf(sizeof(void *)*scp->sk_n_buffers);

	// silence compiler warnings
	largest_addr = NULL;
	smallest_addr = NULL;

	while( n_buffers_seen < scp->sk_n_buffers ){
		spinkError error;
		void *buf_addr;

		error = spinkRetrieveBuffer( scp->sk_context, scp->sk_img_p );
		if( error != SPINK_ERROR_OK ){
			report_spink_error(QSP_ARG  error, "spinkRetrieveBuffer" );
			return;
		}
/*
sprintf(ERROR_STRING,"pData = 0x%lx",(long)scp->sk_img_p->pData);
advise(ERROR_STRING);
*/
		buf_addr = scp->sk_img_p->pData;

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

	scp->sk_base = smallest_addr;
	scp->sk_buf_delta = (largest_addr - smallest_addr) / (n_buffers_seen-1);
	//scp->sk_buf_delta = (largest - smallest) / 30;

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
			(long)scp->sk_buf_delta);
		advise(ERROR_STRING);
	}

	init_cam_frames(QSP_ARG  scp);
#endif // FOOBAR
}

int check_buffer_alignment(QSP_ARG_DECL  Spink_Cam *scp)
{
	int i;

	// alignment requirement is now 1024
	// BUG this should be a parameter...
#define RV_ALIGNMENT_REQ	1024

	for(i=0;i<scp->sk_n_buffers;i++){
		if( ((long)(scp->sk_base+i*scp->sk_buf_delta)) % RV_ALIGNMENT_REQ != 0 ){
			sprintf(ERROR_STRING,"Buffer %d is not aligned - %d byte alignment required for raw volume I/O!?",
				i,RV_ALIGNMENT_REQ);
			warn(ERROR_STRING);
			return -1;
		}
	}
	return 0;
}

#ifdef FOOBAR
static int index_of_buffer(QSP_ARG_DECL  Spink_Cam *scp,spinkImage *ip)
{
	int idx;

	idx = ( ip->pData - scp->sk_base ) / scp->sk_buf_delta;
	/*
sprintf(ERROR_STRING,
"index_of_buffer:  data at 0x%lx, base = 0x%lx, idx = %d",
(long)ip->pData,(long)scp->sk_base,idx);
advise(ERROR_STRING);
*/

	assert( idx >= 0 && idx < scp->sk_n_buffers );
	return idx;
}
#endif // FOOBAR

#ifdef NOT_USED
static const char *name_for_pixel_format(spinkPixelFormat f)
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

Data_Obj * grab_spink_cam_frame(QSP_ARG_DECL  Spink_Cam * scp )
{
	int index;

#ifdef FOOBAR
	spinkError error;

	if( scp->sk_base == NULL )
		init_spink_base(QSP_ARG  scp);

	error = spinkRetrieveBuffer( scp->sk_context, scp->sk_img_p );
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkRetrieveBuffer" );
		return NULL;
	}
//fprintf(stderr,"pixel format of retrieved images is %s (0x%x)\n",
//name_for_pixel_format(img.format),img.format);

	index = index_of_buffer(QSP_ARG  scp, scp->sk_img_p );
	scp->sk_newest = index;
#endif // FOOBAR

	return( scp->sk_frm_dp_tbl[index] );
}

int reset_spink_cam(QSP_ARG_DECL  Spink_Cam *scp)
{
#ifdef FOOBAR
	spinkError error;

	error=spinkFireBusReset(scp->sk_context,&scp->sk_guid);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkFireBusReset" );
	}
#endif // FOOBAR

	return 0;
}

void report_spink_cam_bandwidth(QSP_ARG_DECL  Spink_Cam *scp )
{
	UNIMP_FUNC("report_spink_cam_bandwidth");
}

unsigned int read_register( QSP_ARG_DECL  Spink_Cam *scp, unsigned int addr )
{
#ifdef FOOBAR
	spinkError error;
	unsigned int val;

	error = spinkReadRegister(scp->sk_context,addr,&val);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkReadRegister" );
	}
	return val;
#else // FOOBAR
	return 0;
#endif // FOOBAR
}

void write_register( QSP_ARG_DECL  Spink_Cam *scp, unsigned int addr, unsigned int val )
{
#ifdef FOOBAR
	spinkError error;

	error = spinkWriteRegister(scp->sk_context,addr,val);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkWriteRegister" );
	}
#endif // FOOBAR
}

void start_firewire_capture(QSP_ARG_DECL  Spink_Cam *scp)
{
#ifdef FOOBAR
	spinkError error;

advise("start_firewire_capture BEGIN");
	if( scp->sk_flags & FLY_CAM_IS_RUNNING ){
		warn("start_firewire_capture:  spink_cam is already capturing!?");
		return;
	}
advise("start_firewire_capture cam is not already running");
advise("start_firewire_capture calling spinkStartCapture");

fprintf(stderr,"context = 0x%lx\n",(long)scp->sk_context);
	error = spinkStartCapture(scp->sk_context);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkStartCapture" );
	} else {
		scp->sk_flags |= FLY_CAM_IS_RUNNING;

		// BUG - we should undo this when we stop capturing, because
		// we might change the video format or something else.
		// Perhaps more efficiently we could only do it when needed?
advise("start_firewire_capture calling init_spink_base");
		init_spink_base(QSP_ARG  scp);
	}
advise("start_firewire_capture DONE");
#endif // FOOBAR
}

void stop_firewire_capture(QSP_ARG_DECL  Spink_Cam *scp)
{
#ifdef FOOBAR
	spinkError error;

	if( (scp->sk_flags & FLY_CAM_IS_RUNNING) == 0 ){
		warn("stop_firewire_capture:  spink_cam is not capturing!?");
		return;
	}

	error = spinkStopCapture(scp->sk_context);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkStopCapture" );
	} else {
		scp->sk_flags &= ~FLY_CAM_IS_RUNNING;
	}
#endif // FOOBAR
}

void set_fmt7_size(QSP_ARG_DECL  Spink_Cam *scp, int w, int h)
{
	UNIMP_FUNC("set_fmt7_size");
}

void release_oldest_frame(QSP_ARG_DECL  Spink_Cam *scp)
{
	UNIMP_FUNC("release_oldest_frame");
}

void list_spink_cam_trig(QSP_ARG_DECL  Spink_Cam *scp)
{
#ifdef FOOBAR
	spinkError error;
	spinkTriggerModeInfo tinfo;
	spinkTriggerMode tmode;

	error = spinkGetTriggerModeInfo(scp->sk_context,&tinfo);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetTriggerModeInfo" );
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

	error = spinkGetTriggerMode(scp->sk_context,&tmode);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetTriggerMode" );
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
#endif // FOOBAR
}

void set_buffer_obj(QSP_ARG_DECL  Spink_Cam *scp, Data_Obj *dp)
{
	// make sure sizes match
	if( OBJ_COLS(dp) != scp->sk_cols || OBJ_ROWS(dp) != scp->sk_rows ){
		sprintf(ERROR_STRING,
"set_buffer_obj:  size mismatch between %s (%dx%d) and object %s (%dx%d)",
			scp->sk_name,scp->sk_cols,scp->sk_rows,
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
#ifdef FOOBAR
		spinkError error;

		error = spinkSetUserBuffers(scp->sk_context, OBJ_DATA_PTR(dp),
				OBJ_COLS(dp)*OBJ_ROWS(dp)*OBJ_COMPS(dp),OBJ_FRAMES(dp));
		if( error != SPINK_ERROR_OK ){
			report_spink_error(QSP_ARG  error, "spinkSetUserBuffers" );
			return;
		}
		// refresh the configuration
#endif // FOOBAR
		refresh_config(QSP_ARG  scp);
	}
	scp->sk_base = NULL;	// force init_spink_base to run again
}

#endif /* HAVE_LIBSPINNAKER */


static const char **grab_mode_names=NULL;

int pick_grab_mode(QSP_ARG_DECL Spink_Cam *scp, const char *pmpt)
{
	int idx;

	if( scp == NULL ){
		sprintf(ERROR_STRING,"pick_spink_cam_grab_mode:  no spink_cam selected!?");
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

