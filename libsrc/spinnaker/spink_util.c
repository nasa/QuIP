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

// some globals...
static Spink_Map *current_map=NULL;
#define MAX_TREE_DEPTH	4
static Spink_Node *current_parent_p[MAX_TREE_DEPTH]={NULL,NULL,NULL,NULL};
int current_node_idx; 

static spinSystem hSystem = NULL;
static spinInterfaceList hInterfaceList = NULL;
spinCameraList hCameraList = NULL;
size_t numCameras = 0;
static size_t numInterfaces = 0;

#define TMPSIZE	32	// for temporary object names, e.g. _frame55

ITEM_INTERFACE_DECLARATIONS(Spink_Interface,spink_interface,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Cam,spink_cam,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Map,spink_map,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Node,spink_node,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Node_Type,spink_node_type,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Category,spink_cat,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Enum_Val,spink_enum_val,RB_TREE_CONTAINER)

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

ITEM_INTERFACE_DECLARATIONS(Spink_Cam_Property_Type,pgr_prop,0)

//  When we change spink_cams, we have to refresh all properties!

void list_spink_cam_properties(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	List *lp;
	Node *np;
	//Spink_Cam_Property_Type *pgpt;

	lp = pgr_prop_list();	// all properties
	np = QLIST_HEAD(lp);
	if( np != NULL ){
		sprintf(MSG_STR,"\n%s properties",skc_p->skc_name);
		prt_msg(MSG_STR);
	} else {
		sprintf(ERROR_STRING,"%s has no properties!?",skc_p->skc_name);
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

void refresh_spink_cam_properties(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	List *lp;
	Node *np;
	Spink_Cam_Property_Type *pgpt;

	lp = pgr_prop_list();	// all properties
	np = QLIST_HEAD(lp);
	while(np!=NULL){
		pgpt = (Spink_Cam_Property_Type *)NODE_DATA(np);
		refresh_property_info(QSP_ARG  skc_p, pgpt );
		/*
		if( pgpt->info.present ){
			refresh_property_value(QSP_ARG  skc_p, pgpt );
		}
		*/
		np = NODE_NEXT(np);
	}
}

void refresh_property_info(QSP_ARG_DECL  Spink_Cam *skc_p, Spink_Cam_Property_Type *pgpt )
{
	//spinkError error;

	/*
	error = spinkGetPropertyInfo( skc_p->sk_context, &(pgpt->info) );
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetPropertyInfo" );
		return;
	}
	*/
}

void show_property_info(QSP_ARG_DECL  Spink_Cam *skc_p, Spink_Cam_Property_Type *pgpt )
{
	//char var_name[32],val_str[32];

	sprintf(MSG_STR,"\n%s %s info:",skc_p->skc_name,pgpt->name);
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

void refresh_property_value(QSP_ARG_DECL  Spink_Cam *skc_p, Spink_Cam_Property_Type *pgpt )
{
	/*
	spinkError error;

	error = spinkGetProperty( skc_p->sk_context, &(pgpt->prop) );
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkGetProperty" );
		return;
	}
	*/
}

void show_property_value(QSP_ARG_DECL  Spink_Cam *skc_p, Spink_Cam_Property_Type *pgpt )
{
	sprintf(MSG_STR,"\n%s %s:",
		skc_p->skc_name,pgpt->name);
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

void set_prop_value(QSP_ARG_DECL  Spink_Cam *skc_p, Spink_Cam_Property_Type *pgpt, Spink_Cam_Prop_Val *vp )
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

	error = spinkSetProperty( skc_p->sk_context, &(pgpt->prop));
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkSetProperty" );
		return;
	}
#endif // FOOBAR
}

void set_prop_auto(QSP_ARG_DECL  Spink_Cam *skc_p, Spink_Cam_Property_Type *pgpt, BOOL yn )
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
	error = spinkSetProperty( skc_p->sk_context, &(pgpt->prop) );
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkSetProperty" );
		return;
	}
#endif // FOOBAR
}

void
cleanup_spink_cam( Spink_Cam *skc_p )
{
	//if( IS_CAPTURING(skc_p) )
		 //dc1394_capture_stop( skc_p->sk_cam_p );
		 //fly_capture_stop( skc_p->sk_cam_p );
	//if( IS_TRANSMITTING(skc_p) )
		//dc1394_video_set_transmission( skc_p->sk_cam_p, DC1394_OFF );
		//fly_video_set_transmission( skc_p->sk_cam_p, DC1394_OFF );
	/* dc1394_free_spink_cam */
	//dc1394_spink_cam_free( skc_p->sk_cam_p );
	//fly_spink_cam_free( skc_p->sk_cam_p );
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
	Spink_Cam *skc_p;
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
		skc_p = (Spink_Cam *) NODE_DATA(np);
		dst = OBJ_DATA_PTR(str_dp);
		dst += i * OBJ_PXL_INC(str_dp);
		if( strlen(skc_p->skc_name)+1 > OBJ_COMPS(str_dp) ){
			sprintf(ERROR_STRING,"String object %s has too few components (%ld) to hold spink_cam name \"%s\"",
				OBJ_NAME(str_dp),(long)OBJ_COMPS(str_dp),skc_p->skc_name);
			warn(ERROR_STRING);
		} else {
			strcpy(dst,skc_p->skc_name);
		}
		i++;
		if( i>=n )
			np=NULL;
		else
			np = NODE_NEXT(np);
	}

	return i;
}

int get_spink_cam_video_mode_strings( QSP_ARG_DECL  Data_Obj *str_dp, Spink_Cam *skc_p )
{
	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?

	int i, n;

	if( OBJ_COLS(str_dp) < skc_p->skc_n_video_modes ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d modes",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),skc_p->skc_n_video_modes);
		warn(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	} else {
		n=skc_p->skc_n_video_modes;
	}
		
	for(i=0;i<n;i++){
		int k;
		const char *src;
		char *dst;

		k=skc_p->skc_video_mode_indices[i];
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

static void get_framerate_choices(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	unsigned int i,n,idx;
	Framerate_Mask mask;

	if( skc_p->skc_framerate_names != NULL ){
		givbuf(skc_p->skc_framerate_names);
		skc_p->skc_framerate_names=NULL;	// in case we have an error before finishing
	}

	// format7 doesn't have a framerate!?

	mask = skc_p->skc_framerate_mask_tbl[ skc_p->skc_my_video_mode_index ];
/*
sprintf(ERROR_STRING,"%s:  video mode is %s",
skc_p->skc_name,all_video_modes[skc_p->skc_video_mode_index].nvm_name);
advise(ERROR_STRING);
sprintf(ERROR_STRING,"%s:  my video mode %s (index = %d)",
skc_p->skc_name,skc_p->skc_video_mode_names[skc_p->skc_my_video_mode_index],
skc_p->skc_my_video_mode_index);
advise(ERROR_STRING);
*/

	n = bit_count(mask);
	if( n <= 0 ){
		// this happens for the format7 video mode...
		// Can this ever happen?  If not, should be CAUTIOUS...
		//warn("no framerates for this video mode!?");
#ifdef FOOBAR
		if( skc_p->skc_video_mode == SPINK_VIDEOMODE_FORMAT7 ){
			if( skc_p->skc_framerate == SPINK_FRAMERATE_FORMAT7 ){
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

	skc_p->skc_framerate_names = getbuf( n * sizeof(char *) );
	skc_p->skc_n_framerates = n ;

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
			skc_p->skc_framerate_names[idx]=all_framerates[j].nfr_name;
#endif // FOOBAR
			idx++;
		}
			
		mask >>= 1;
	}

	assert( idx == n );
}

int get_spink_cam_framerate_strings( QSP_ARG_DECL  Data_Obj *str_dp, Spink_Cam *skc_p )
{
	int i, n;
	const char *src;
	char *dst;

	get_framerate_choices(QSP_ARG  skc_p);

	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?

	n = skc_p->skc_n_framerates;

	if( OBJ_COLS(str_dp) < n ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d framerates",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),n);
		warn(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	}

	for(i=0;i<skc_p->skc_n_framerates;i++){
		src = skc_p->skc_framerate_names[i];
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
static void test_setup(QSP_ARG_DECL  Spink_Cam *skc_p,
		spinkVideoMode m, spinkFrameRate r, int *kp, int idx )
{
	spinkError error;
	BOOL supported;

	// punt if format 7
	if( m == SPINK_VIDEOMODE_FORMAT7 && r == SPINK_FRAMERATE_FORMAT7 ){
		// BUG?  make sure spink_cam has format7 modes?
		supported = TRUE;
	} else {
		error = spinkGetVideoModeAndFrameRateInfo( skc_p->sk_context,
			m,r,&supported );
		if( error != SPINK_ERROR_OK ){
			report_spink_error(QSP_ARG  error,
				"spinkGetVideoModeAndFrameRateInfo" );
			supported = FALSE;
		}
	}

	if( supported ){
		if( *kp < 0 || skc_p->skc_video_mode_indices[*kp] != idx ){
			*kp = (*kp)+1;
			skc_p->skc_video_mode_indices[*kp] = idx;
			skc_p->skc_video_mode_names[*kp] = all_video_modes[idx].nvm_name;
//fprintf(stderr,"test_setup:  adding video mode %s to %s\n",
//all_video_modes[idx].nvm_name,
//skc_p->skc_name);
			if( skc_p->skc_video_mode == m )
				skc_p->skc_video_mode_index = *kp;
		}
		skc_p->skc_framerate_mask_tbl[*kp] |= 1 << r;
	}
}
#endif // FOOBAR

// init_spink_base   -   grab frames to get the address
// associated with each frame index.  This wouldn't be necessary
// if we always provided the buffers, but we want this to work
// even when we don't.
//
// We keep track of the largest and smallest address, we save
// those so we can figure out the index of an arbitrary frame...


int check_buffer_alignment(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	int i;

	// alignment requirement is now 1024
	// BUG this should be a parameter...
#define RV_ALIGNMENT_REQ	1024

	for(i=0;i<skc_p->skc_n_buffers;i++){
		if( ((long)(skc_p->skc_base+i*skc_p->skc_buf_delta)) % RV_ALIGNMENT_REQ != 0 ){
			sprintf(ERROR_STRING,"Buffer %d is not aligned - %d byte alignment required for raw volume I/O!?",
				i,RV_ALIGNMENT_REQ);
			warn(ERROR_STRING);
			return -1;
		}
	}
	return 0;
}

#ifdef FOOBAR
static int index_of_buffer(QSP_ARG_DECL  Spink_Cam *skc_p,spinkImage *ip)
{
	int idx;

	idx = ( ip->pData - skc_p->skc_base ) / skc_p->skc_buf_delta;
	/*
sprintf(ERROR_STRING,
"index_of_buffer:  data at 0x%lx, base = 0x%lx, idx = %d",
(long)ip->pData,(long)skc_p->skc_base,idx);
advise(ERROR_STRING);
*/

	assert( idx >= 0 && idx < skc_p->skc_n_buffers );
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

Data_Obj * grab_spink_cam_frame(QSP_ARG_DECL  Spink_Cam * skc_p )
{
#ifdef FOOBAR
	int index;

	spinkError error;

	if( skc_p->skc_base == NULL )
		init_spink_base(QSP_ARG  skc_p);

	error = spinkRetrieveBuffer( skc_p->sk_context, skc_p->sk_img_p );
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkRetrieveBuffer" );
		return NULL;
	}
//fprintf(stderr,"pixel format of retrieved images is %s (0x%x)\n",
//name_for_pixel_format(img.format),img.format);

	index = index_of_buffer(QSP_ARG  skc_p, skc_p->sk_img_p );
	skc_p->skc_newest = index;

	return( skc_p->skc_frm_dp_tbl[index] );
#endif // FOOBAR
	return NULL;
}

int reset_spink_cam(QSP_ARG_DECL  Spink_Cam *skc_p)
{
#ifdef FOOBAR
	spinkError error;

	error=spinkFireBusReset(skc_p->sk_context,&skc_p->sk_guid);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkFireBusReset" );
	}
#endif // FOOBAR

	return 0;
}

void report_spink_cam_bandwidth(QSP_ARG_DECL  Spink_Cam *skc_p )
{
	UNIMP_FUNC("report_spink_cam_bandwidth");
}

unsigned int read_register( QSP_ARG_DECL  Spink_Cam *skc_p, unsigned int addr )
{
#ifdef FOOBAR
	spinkError error;
	unsigned int val;

	error = spinkReadRegister(skc_p->sk_context,addr,&val);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkReadRegister" );
	}
	return val;
#else // FOOBAR
	return 0;
#endif // FOOBAR
}

void write_register( QSP_ARG_DECL  Spink_Cam *skc_p, unsigned int addr, unsigned int val )
{
#ifdef FOOBAR
	spinkError error;

	error = spinkWriteRegister(skc_p->sk_context,addr,val);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkWriteRegister" );
	}
#endif // FOOBAR
}

void start_firewire_capture(QSP_ARG_DECL  Spink_Cam *skc_p)
{
#ifdef FOOBAR
	spinkError error;

advise("start_firewire_capture BEGIN");
	if( skc_p->skc_flags & FLY_CAM_IS_RUNNING ){
		warn("start_firewire_capture:  spink_cam is already capturing!?");
		return;
	}
advise("start_firewire_capture cam is not already running");
advise("start_firewire_capture calling spinkStartCapture");

//fprintf(stderr,"context = 0x%lx\n",(long)skc_p->sk_context);
	error = spinkStartCapture(skc_p->sk_context);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkStartCapture" );
	} else {
		skc_p->skc_flags |= FLY_CAM_IS_RUNNING;

		// BUG - we should undo this when we stop capturing, because
		// we might change the video format or something else.
		// Perhaps more efficiently we could only do it when needed?
advise("start_firewire_capture calling init_spink_base");
		init_spink_base(QSP_ARG  skc_p);
	}
advise("start_firewire_capture DONE");
#endif // FOOBAR
}

void stop_firewire_capture(QSP_ARG_DECL  Spink_Cam *skc_p)
{
#ifdef FOOBAR
	spinkError error;

	if( (skc_p->skc_flags & FLY_CAM_IS_RUNNING) == 0 ){
		warn("stop_firewire_capture:  spink_cam is not capturing!?");
		return;
	}

	error = spinkStopCapture(skc_p->sk_context);
	if( error != SPINK_ERROR_OK ){
		report_spink_error(QSP_ARG  error, "spinkStopCapture" );
	} else {
		skc_p->skc_flags &= ~FLY_CAM_IS_RUNNING;
	}
#endif // FOOBAR
}

void set_fmt7_size(QSP_ARG_DECL  Spink_Cam *skc_p, int w, int h)
{
	UNIMP_FUNC("set_fmt7_size");
}

void release_oldest_frame(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	UNIMP_FUNC("release_oldest_frame");
}

void list_spink_cam_trig(QSP_ARG_DECL  Spink_Cam *skc_p)
{
#ifdef FOOBAR
	spinkError error;
	spinkTriggerModeInfo tinfo;
	spinkTriggerMode tmode;

	error = spinkGetTriggerModeInfo(skc_p->sk_context,&tinfo);
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

	error = spinkGetTriggerMode(skc_p->sk_context,&tmode);
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

void set_buffer_obj(QSP_ARG_DECL  Spink_Cam *skc_p, Data_Obj *dp)
{
	// make sure sizes match
	if( OBJ_COLS(dp) != skc_p->skc_cols || OBJ_ROWS(dp) != skc_p->skc_rows ){
		sprintf(ERROR_STRING,
"set_buffer_obj:  size mismatch between %s (%dx%d) and object %s (%dx%d)",
			skc_p->skc_name,skc_p->skc_cols,skc_p->skc_rows,
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

		error = spinkSetUserBuffers(skc_p->sk_context, OBJ_DATA_PTR(dp),
				OBJ_COLS(dp)*OBJ_ROWS(dp)*OBJ_COMPS(dp),OBJ_FRAMES(dp));
		if( error != SPINK_ERROR_OK ){
			report_spink_error(QSP_ARG  error, "spinkSetUserBuffers" );
			return;
		}
#endif // FOOBAR
	}
	skc_p->skc_base = NULL;	// force init_spink_base to run again
}

#endif /* HAVE_LIBSPINNAKER */


static const char **grab_mode_names=NULL;

int pick_grab_mode(QSP_ARG_DECL Spink_Cam *skc_p, const char *pmpt)
{
	int idx;

	if( skc_p == NULL ){
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

void pop_spink_cam_context(SINGLE_QSP_ARG_DECL)
{
	// pop old context...
	Item_Context *icp;
	icp=pop_dobj_context();
	assert( icp != NULL );
}

void push_spink_cam_context(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	push_dobj_context(skc_p->skc_do_icp);
}

Item_Context * _pop_spink_node_context(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp;
	if( spink_node_itp == NULL ) init_spink_nodes();
	icp = pop_item_context(spink_node_itp);
	return icp;
}

void _push_spink_node_context(QSP_ARG_DECL  Item_Context *icp)
{
	if( spink_node_itp == NULL ) init_spink_nodes();
	push_item_context(spink_node_itp,icp);
}

Item_Context * _pop_spink_cat_context(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp;
	if( spink_cat_itp == NULL ) init_spink_cats();
	icp = pop_item_context(spink_cat_itp);
	return icp;
}

void _push_spink_cat_context(QSP_ARG_DECL  Item_Context *icp)
{
	if( spink_cat_itp == NULL ) init_spink_cats();
	push_item_context(spink_cat_itp,icp);
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

#define get_unique_cam_name(buf, buflen) _get_unique_cam_name(QSP_ARG  buf, buflen)

static int _get_unique_cam_name(QSP_ARG_DECL  char *buf, int buflen)
{
	int i=2;
	int orig_len;
	Spink_Cam *skc_p;

	orig_len = strlen(buf);
	if( orig_len+3 > buflen ){
		sprintf(ERROR_STRING,
			"Camera name buffer needs to be enlarged to accomodate multiple instances of '%s'!?",
			buf);
		error1(ERROR_STRING);
	}
	buf[orig_len]='_';
	buf[orig_len+2]=0;
	while(i<10){
		buf[orig_len+1]='0'+i;	// 2-9
		skc_p = spink_cam_of(buf);
		if( skc_p == NULL ) return 0;
	}
	return -1;
}

#define INVALID_SET_FUNC(name)									\
static void _set_##name##_node(QSP_ARG_DECL  Spink_Node *skn_p)					\
{												\
	sprintf(ERROR_STRING,"set_%s_node:  %s nodes should never be set!?",#name,#name);	\
	error1(ERROR_STRING);									\
}

INVALID_SET_FUNC(category)
INVALID_SET_FUNC(register)
INVALID_SET_FUNC(port)
INVALID_SET_FUNC(base)
INVALID_SET_FUNC(unknown)
INVALID_SET_FUNC(command)
INVALID_SET_FUNC(enum_entry)

// These need to be implemented...
INVALID_SET_FUNC(value)
INVALID_SET_FUNC(string)
INVALID_SET_FUNC(integer)
INVALID_SET_FUNC(float)
INVALID_SET_FUNC(boolean)
//INVALID_SET_FUNC(enumeration)


#define init_enum_val(skn_p) _init_enum_val(QSP_ARG  skn_p)

static void _init_enum_val(QSP_ARG_DECL  Spink_Node *skn_p)
{
	char *s, *copy;
	const char *enum_name, *entry_name;

	// copy the name to break it up into components
	s=copy=(char *)savestr(skn_p->skn_name);
	while( *s && *s!='_' ) s++;
	assert(*s=='_');
	s++;
	assert(*s!=0);
	enum_name = s;
	while( *s && *s!='_' ) s++;
	assert(*s=='_');
	*s = 0;
	s++;
	assert(*s!=0);
	entry_name = s;
fprintf(stderr,"enum = %s     entry = %s\n",enum_name,entry_name);
	if( !strcmp(enum_name,"GainAuto") ){
		if( !strcmp(entry_name,"Once") ){
			skn_p->skn_enum_val = GainAuto_Once;
		} else if( !strcmp(entry_name,"Off") ){
			skn_p->skn_enum_val = GainAuto_Off;
		} else if( !strcmp(entry_name,"Continuous") ){
			skn_p->skn_enum_val = GainAuto_Continuous;
		} else {
			sprintf(ERROR_STRING,"init_enum_val:  Unhandled entry %s, enumeration %s",entry_name,enum_name);
			warn(ERROR_STRING);
		}
	} else {
		sprintf(ERROR_STRING,"init_enum_val:  Unhandled enumeration %s",enum_name);
		warn(ERROR_STRING);
	}
}

// The enumeration entry nodes have names like EnumerationEntry_GainAuto_Once - but we
// want to make the choice be "Once", so we skip ahead past the second underscore...

static void make_enumeration_choices(const char ***tbl_ptr, int *nc_p, Spink_Node *skn_p)
{
	int n;
	Node *np;
	const char **tbl;
	const char *s;

	assert(skn_p!=NULL);
	assert(skn_p->skn_children!=NULL);
	assert(skn_p->skn_type_p->snt_type == EnumerationNode);

	n = eltcount(skn_p->skn_children);
	assert(n>0);
	*nc_p = n;
	tbl = getbuf( n * sizeof(char *) );
	*tbl_ptr = tbl;
	np = QLIST_HEAD(skn_p->skn_children);
	
	while(np!=NULL){
		Spink_Node *child;
		child = NODE_DATA(np);
		s = child->skn_name;
		while( *s && *s != '_' ) s++;
		assert(*s=='_');
		s++;	// skip first
		while( *s && *s != '_' ) s++;
		assert(*s=='_');
		s++;	// skip second
		assert(*s!=0);
		*tbl = s;
		tbl++;
		np = NODE_NEXT(np);
	}
}

static void _set_enumeration_node(QSP_ARG_DECL  Spink_Node *skn_p)
{
	int idx;
	const char **choices;
	int n_choices;
	Spink_Node *child;
	Node *np;
	spinNodeHandle hNode;

	make_enumeration_choices(&choices,&n_choices,skn_p);
	idx = which_one(skn_p->skn_name,n_choices,choices);
	if( idx < 0 ){
		givbuf(choices);
		return;
	}

	// now find the enum node
	np = nth_elt(skn_p->skn_children,idx);
	assert(np!=NULL);
	child = NODE_DATA(np);

	if( child->skn_enum_val == INVALID_ENUM_VAL )
		init_enum_val(child);
	assert( child->skn_enum_val != INVALID_ENUM_VAL );

	fprintf(stderr,"Child node is %s, enum value is %ld\n",child->skn_name,child->skn_enum_val);
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( set_enum_enum_value(hNode,child->skn_enum_val) < 0 )
		warn("Error setting enum value!?");
//SPINNAKERC_API spinEnumerationSetEnumValue(spinNodeHandle hNode, size_t value);
	// EnumerationSetEnumValue
}



#define INVALID_PRINT_VALUE_FUNC(name)								\
static void _print_##name##_node_value(QSP_ARG_DECL  Spink_Node *skn_p)				\
{												\
	sprintf(ERROR_STRING,"print_%s_node_value:  %s nodes cannot be printed!?",#name,#name);	\
	error1(ERROR_STRING);									\
}


//INVALID_PRINT_VALUE_FUNC(register)
//INVALID_PRINT_VALUE_FUNC(enum_entry)
INVALID_PRINT_VALUE_FUNC(port)
INVALID_PRINT_VALUE_FUNC(base)
INVALID_PRINT_VALUE_FUNC(unknown)

static void _print_register_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	sprintf(MSG_STR,STRING_NODE_FMT_STR,"(unhandled case!?)");
	prt_msg_frag(MSG_STR);
}

static void _print_enum_entry_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	sprintf(MSG_STR,STRING_NODE_FMT_STR,"");
	prt_msg_frag(MSG_STR);
}

static void _print_category_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	sprintf(MSG_STR,STRING_NODE_FMT_STR,"");
	prt_msg_frag(MSG_STR);
}

static void _print_value_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	char val_buf[MAX_BUFF_LEN];
	size_t buf_len = MAX_BUFF_LEN;
	spinNodeHandle hNode;
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_node_value_string(val_buf,&buf_len,hNode) < 0 ) return;
	sprintf(MSG_STR,STRING_NODE_FMT_STR,val_buf);
	prt_msg_frag(MSG_STR);
}

static void _print_string_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	char val_buf[MAX_BUFF_LEN];
	size_t buf_len = MAX_BUFF_LEN;
	spinNodeHandle hNode;
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_string_node_string(val_buf,&buf_len,hNode) < 0 ) return;
	sprintf(MSG_STR,STRING_NODE_FMT_STR,val_buf);
	prt_msg_frag(MSG_STR);
}

static void _print_integer_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	int64_t integerValue = 0;
	spinNodeHandle hNode;
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_int_value(hNode, &integerValue) < 0 ) return;
	sprintf(MSG_STR,INT_NODE_FMT_STR, integerValue);
	prt_msg_frag(MSG_STR);
}

static void _print_float_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	double floatValue = 0.0;
	spinNodeHandle hNode;
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_float_value(hNode,&floatValue) < 0 ) return;
	sprintf(MSG_STR,FLT_NODE_FMT_STR, floatValue);
	prt_msg_frag(MSG_STR);
}

static void _print_boolean_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	bool8_t booleanValue = False;
	spinNodeHandle hNode;
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_bool_value(hNode,&booleanValue) < 0 ) return;
	sprintf(MSG_STR,STRING_NODE_FMT_STR, (booleanValue ? "true" : "false"));
	prt_msg_frag(MSG_STR);
}

static void _print_command_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	char val_buf[MAX_BUFF_LEN];
	size_t buf_len = MAX_BUFF_LEN;
	spinNodeHandle hNode;
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_tip_value(hNode,val_buf,&buf_len) < 0 ) return;
	if( buf_len > MAX_NODE_VALUE_CHARS_TO_PRINT) {
		int i;
		for (i = 0; i < MAX_NODE_VALUE_CHARS_TO_PRINT-3; i++) {
			MSG_STR[i] = val_buf[i];
		}
		MSG_STR[i++]='.';
		MSG_STR[i++]='.';
		MSG_STR[i++]='.';
		MSG_STR[i++]=0;
	} else {
		sprintf(MSG_STR,STRING_NODE_FMT_STR, val_buf);
	}
	prt_msg_frag(MSG_STR);
}

static void _print_enumeration_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	char val_buf[MAX_BUFF_LEN];
	size_t buf_len = MAX_BUFF_LEN;
	spinNodeHandle hCurrentEntryNode = NULL;
	spinNodeHandle hNode;
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_current_entry(hNode,&hCurrentEntryNode) < 0 ) return;
	if( get_entry_symbolic(hCurrentEntryNode, val_buf, &buf_len) < 0 ) return;
	sprintf(MSG_STR,STRING_NODE_FMT_STR,val_buf);
	prt_msg_frag(MSG_STR);
}

#define INIT_NODE_TYPE(name,code)					\
	snt_p = new_spink_node_type(#name);				\
	snt_p->snt_type = code;						\
	snt_p->snt_set_func = _set_##name##_node;			\
	snt_p->snt_print_value_func = _print_##name##_node_value;	\


#define init_default_node_types() _init_default_node_types(SINGLE_QSP_ARG)

static void _init_default_node_types(SINGLE_QSP_ARG_DECL)
{
	Spink_Node_Type *snt_p;

	INIT_NODE_TYPE(category,CategoryNode)
	INIT_NODE_TYPE(register,RegisterNode)
	INIT_NODE_TYPE(port,PortNode)
	INIT_NODE_TYPE(base,BaseNode)
	INIT_NODE_TYPE(unknown,UnknownNode)
	INIT_NODE_TYPE(value,ValueNode)
	INIT_NODE_TYPE(string,StringNode)
	INIT_NODE_TYPE(integer,IntegerNode)
	INIT_NODE_TYPE(float,FloatNode)
	INIT_NODE_TYPE(boolean,BooleanNode)
	INIT_NODE_TYPE(command,CommandNode)
	INIT_NODE_TYPE(enumeration,EnumerationNode)
	INIT_NODE_TYPE(enum_entry,EnumEntryNode)
}

Spink_Node_Type *_find_type_by_code(QSP_ARG_DECL  spinNodeType type)
{
	List *lp;
	Node *np;

	if( spink_node_type_itp == NULL )
		init_spink_node_types();

	lp = spink_node_type_list();
	if( lp == NULL || eltcount(lp) == 0 ){
		init_default_node_types();
		lp = spink_node_type_list();
	}
	assert( lp != NULL && eltcount(lp) != 0 );

	np = QLIST_HEAD(lp);
	while( np != NULL ){
		Spink_Node_Type *snt_p;
		snt_p = NODE_DATA(np);
		if( snt_p->snt_type == type ) return snt_p;
		np = NODE_NEXT(np);
	}
	// Should we create the new type here???
	warn("Node type not found!?");
	return NULL;
}

// This helper function deals with output indentation, of which there is a lot.

#define indent(level) _indent(QSP_ARG  level)

static void _indent(QSP_ARG_DECL  int level)
{
	int i = 0;

	for (i = 0; i < level; i++) {
		prt_msg_frag("   ");
	}
}

#define print_display_name(hNode) _print_display_name(QSP_ARG  hNode)

static void _print_display_name(QSP_ARG_DECL  Spink_Node * skn_p)
{
	char fmt_str[16];
	char displayName[MAX_BUFF_LEN];
	size_t displayNameLength = MAX_BUFF_LEN;
	spinNodeHandle hNode;

	assert(max_display_name_len>0);
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_display_name(displayName,&displayNameLength,hNode) < 0 ) return;
	sprintf(fmt_str,"%%-%ds",max_display_name_len+3);
	sprintf(MSG_STR,fmt_str,displayName);
	prt_msg_frag(MSG_STR);
}

#define print_node_type(snt_p) _print_node_type(QSP_ARG  snt_p)

static void _print_node_type(QSP_ARG_DECL  Spink_Node_Type * snt_p)
{
	sprintf(MSG_STR,"%-16s",snt_p->snt_name);
	prt_msg_frag(MSG_STR);
}

#define show_rw_status(hNode) _show_rw_status(QSP_ARG  hNode)

static void _show_rw_status(QSP_ARG_DECL  Spink_Node *skn_p)
{
	if( NODE_IS_READABLE(skn_p) ){
		if( NODE_IS_WRITABLE(skn_p) ){
			prt_msg("   (read/write)");
		} else {
			prt_msg("   (read-only)");
		}
	} else if( NODE_IS_WRITABLE(skn_p) ){
			prt_msg("   (write-only)");
	} else {
		prt_msg("   (no read or write access!?)");
	}
}

#define MAX_LEVEL	3

void _print_spink_node_info(QSP_ARG_DECL  Spink_Node *skn_p, int level)
{
	Spink_Node_Type *snt_p;

	//assert(level>0);	// don't print root node
	indent(level-1);
	print_display_name(skn_p);
	indent(MAX_LEVEL-level);
	snt_p = skn_p->skn_type_p;
	assert(snt_p!=NULL);
	print_node_type(snt_p);
	(*(snt_p->snt_print_value_func))(QSP_ARG  skn_p);
	show_rw_status(skn_p);
}

static void _print_node_from_tree(QSP_ARG_DECL  Spink_Node *skn_p)
{
	if( skn_p->skn_level == 0 ) return;	// don't print root node
	print_spink_node_info(skn_p,skn_p->skn_level);
}

#define traverse_node_tree(skn_p, func) _traverse_node_tree(QSP_ARG  skn_p, func)

static void _traverse_node_tree(QSP_ARG_DECL  Spink_Node *skn_p, void (*func)(QSP_ARG_DECL  Spink_Node *))
{
	Node *np;
	Spink_Node *child_p;

	assert(skn_p!=NULL);
	(*func)(QSP_ARG  skn_p);
	if( skn_p->skn_children == NULL ) return;
	np = QLIST_HEAD(skn_p->skn_children);
	if( np == NULL ) return;
	while(np!=NULL){
		child_p = NODE_DATA(np);
		traverse_node_tree(child_p,func);
		np = NODE_NEXT(np);
	}
}

void _print_map_tree(QSP_ARG_DECL  Spink_Map *skm_p)
{
	assert(skm_p!=NULL);
	assert(skm_p->skm_root_p!=NULL);
	traverse_node_tree(skm_p->skm_root_p,_print_node_from_tree);
}

void _print_cat_tree(QSP_ARG_DECL  Spink_Category *sct_p)
{
	assert(sct_p!=NULL);
	assert(sct_p->sct_root_p!=NULL);
	traverse_node_tree(sct_p->sct_root_p,_print_node_from_tree);
}

#define init_enums() _init_enums(SINGLE_QSP_ARG)

#define ADD_ENUM_VAL(group,value)						\
	sev_p = new_spink_enum_val("EnumerationEntry_" #group "_" #value);	\
	assert(sev_p!=NULL);							\
	sev_p->sev_value = group##_##value;

static int enums_inited=0;

static void _init_enums(SINGLE_QSP_ARG_DECL)
{
	Spink_Enum_Val *sev_p;


//ADD_ENUM_VAL(AcquisitionFrameRateAuto,Continuous)
//ADD_ENUM_VAL(AcquisitionFrameRateAuto,Off)
ADD_ENUM_VAL(AcquisitionMode,Continuous)
ADD_ENUM_VAL(AcquisitionMode,MultiFrame)
ADD_ENUM_VAL(AcquisitionMode,SingleFrame)
ADD_ENUM_VAL(AdcBitDepth,Bit10)
ADD_ENUM_VAL(AutoAlgorithmSelector,Ae)
ADD_ENUM_VAL(AutoAlgorithmSelector,Awb)
ADD_ENUM_VAL(AutoExposureControlPriority,ExposureTime)
ADD_ENUM_VAL(AutoExposureControlPriority,Gain)
ADD_ENUM_VAL(AutoExposureLightingMode,Backlight)
ADD_ENUM_VAL(AutoExposureLightingMode,Frontlight)
ADD_ENUM_VAL(AutoExposureLightingMode,Normal)
ADD_ENUM_VAL(AutoExposureMeteringMode,Average)
ADD_ENUM_VAL(AutoExposureMeteringMode,Partial)
ADD_ENUM_VAL(AutoExposureMeteringMode,Spot)
ADD_ENUM_VAL(AutoExposureTargetGreyValueAuto,Continuous)
ADD_ENUM_VAL(AutoExposureTargetGreyValueAuto,Off)
ADD_ENUM_VAL(BinningHorizontalMode,Average)
ADD_ENUM_VAL(BinningHorizontalMode,Sum)
ADD_ENUM_VAL(BinningSelector,All)
ADD_ENUM_VAL(BinningSelector,ISP)
ADD_ENUM_VAL(BinningSelector,Sensor)
ADD_ENUM_VAL(BinningVerticalMode,Average)
ADD_ENUM_VAL(BinningVerticalMode,Sum)
ADD_ENUM_VAL(BlackLevelSelector,All)
ADD_ENUM_VAL(BlackLevelSelector,Analog)
ADD_ENUM_VAL(BlackLevelSelector,Digital)
//ADD_ENUM_VAL(ChunkBlackLevelSelector,Off)
//ADD_ENUM_VAL(ChunkGainSelector,Counter0)
//ADD_ENUM_VAL(ChunkGainSelector,Counter1)
//ADD_ENUM_VAL(ChunkGainSelector,Off)
//ADD_ENUM_VAL(ChunkGainSelector,On)
//ADD_ENUM_VAL(ChunkSelector,All)
ADD_ENUM_VAL(ChunkSelector,BlackLevel)
//ADD_ENUM_VAL(ChunkSelector,Blue)
ADD_ENUM_VAL(ChunkSelector,CRC)
//ADD_ENUM_VAL(ChunkSelector,Error)
//ADD_ENUM_VAL(ChunkSelector,ExposureEnd)
ADD_ENUM_VAL(ChunkSelector,ExposureTime)
//ADD_ENUM_VAL(ChunkSelector,FrameCounter)
ADD_ENUM_VAL(ChunkSelector,Gain)
//ADD_ENUM_VAL(ChunkSelector,Green)
ADD_ENUM_VAL(ChunkSelector,Height)
ADD_ENUM_VAL(ChunkSelector,Image)
//ADD_ENUM_VAL(ChunkSelector,LUT1)
ADD_ENUM_VAL(ChunkSelector,OffsetX)
ADD_ENUM_VAL(ChunkSelector,OffsetY)
ADD_ENUM_VAL(ChunkSelector,PixelFormat)
//ADD_ENUM_VAL(ChunkSelector,Red)
ADD_ENUM_VAL(ChunkSelector,SequencerSetActive)
ADD_ENUM_VAL(ChunkSelector,Timestamp)
ADD_ENUM_VAL(ChunkSelector,Width)
ADD_ENUM_VAL(CounterEventSource,Counter0End)
ADD_ENUM_VAL(CounterEventSource,Counter0Start)
ADD_ENUM_VAL(CounterEventSource,Counter1End)
ADD_ENUM_VAL(CounterEventSource,Counter1Start)
ADD_ENUM_VAL(CounterEventSource,ExposureEnd)
ADD_ENUM_VAL(CounterEventSource,ExposureStart)
ADD_ENUM_VAL(CounterEventSource,FrameTriggerWait)
ADD_ENUM_VAL(CounterEventSource,Line0)
ADD_ENUM_VAL(CounterEventSource,Line2)
ADD_ENUM_VAL(CounterEventSource,Line3)
ADD_ENUM_VAL(CounterEventSource,LogicBlock0)
ADD_ENUM_VAL(CounterEventSource,LogicBlock1)
ADD_ENUM_VAL(CounterEventSource,Off)
ADD_ENUM_VAL(CounterEventSource,UserOutput0)
ADD_ENUM_VAL(CounterEventSource,UserOutput1)
ADD_ENUM_VAL(CounterEventSource,UserOutput2)
ADD_ENUM_VAL(CounterEventSource,UserOutput3)
//ADD_ENUM_VAL(CounterSelector,UserOutput1)
//ADD_ENUM_VAL(CounterSelector,UserOutput2)
//ADD_ENUM_VAL(CounterStatus,Enable)
//ADD_ENUM_VAL(CounterStatus,Input0)
//ADD_ENUM_VAL(CounterStatus,Input1)
//ADD_ENUM_VAL(CounterStatus,Input2)
//ADD_ENUM_VAL(CounterStatus,Zero)
//ADD_ENUM_VAL(CounterTriggerActivation,CounterCompleted)
//ADD_ENUM_VAL(CounterTriggerActivation,CounterOverflow)
//ADD_ENUM_VAL(CounterTriggerActivation,LogicBlock0)
//ADD_ENUM_VAL(CounterTriggerActivation,LogicBlock1)
//ADD_ENUM_VAL(CounterTriggerActivation,Value)
//ADD_ENUM_VAL(CounterTriggerSource,AnyEdge)
ADD_ENUM_VAL(CounterTriggerSource,Counter0End)
ADD_ENUM_VAL(CounterTriggerSource,Counter0Start)
ADD_ENUM_VAL(CounterTriggerSource,Counter1End)
ADD_ENUM_VAL(CounterTriggerSource,Counter1Start)
//ADD_ENUM_VAL(CounterTriggerSource,CounterActive)
//ADD_ENUM_VAL(CounterTriggerSource,CounterIdle)
//ADD_ENUM_VAL(CounterTriggerSource,CounterTriggerWait)
ADD_ENUM_VAL(CounterTriggerSource,ExposureEnd)
ADD_ENUM_VAL(CounterTriggerSource,ExposureStart)
//ADD_ENUM_VAL(CounterTriggerSource,FallingEdge)
ADD_ENUM_VAL(CounterTriggerSource,FrameTriggerWait)
//ADD_ENUM_VAL(CounterTriggerSource,LevelHigh)
//ADD_ENUM_VAL(CounterTriggerSource,LevelLow)
ADD_ENUM_VAL(CounterTriggerSource,LogicBlock0)
ADD_ENUM_VAL(CounterTriggerSource,LogicBlock1)
//ADD_ENUM_VAL(CounterTriggerSource,RisingEdge)
ADD_ENUM_VAL(DecimationHorizontalMode,Discard)
ADD_ENUM_VAL(DecimationSelector,All)
ADD_ENUM_VAL(DecimationSelector,Sensor)
ADD_ENUM_VAL(DecimationVerticalMode,Discard)
//ADD_ENUM_VAL(DefectCorrectionMode,Average)
//ADD_ENUM_VAL(DefectCorrectionMode,Highlight)
ADD_ENUM_VAL(DeviceAccessStatus,NoAccess)
ADD_ENUM_VAL(DeviceAccessStatus,ReadOnly)
ADD_ENUM_VAL(DeviceAccessStatus,ReadWrite)
ADD_ENUM_VAL(DeviceAccessStatus,Unknown)
ADD_ENUM_VAL(DeviceCurrentSpeed,FullSpeed)
ADD_ENUM_VAL(DeviceCurrentSpeed,HighSpeed)
ADD_ENUM_VAL(DeviceCurrentSpeed,LowSpeed)
ADD_ENUM_VAL(DeviceCurrentSpeed,SuperSpeed)
ADD_ENUM_VAL(DeviceCurrentSpeed,UnknownSpeed)
ADD_ENUM_VAL(DeviceEndianessMechanism,Legacy)
ADD_ENUM_VAL(DeviceEndianessMechanism,Standard)
ADD_ENUM_VAL(DeviceIndicatorMode,Active)
ADD_ENUM_VAL(DeviceIndicatorMode,ErrorStatus)
ADD_ENUM_VAL(DeviceIndicatorMode,Inactive)
ADD_ENUM_VAL(DevicePowerSupplySelector,External)
ADD_ENUM_VAL(DeviceScanType,Areascan)
ADD_ENUM_VAL(DeviceTLType,CameraLink)
ADD_ENUM_VAL(DeviceTLType,CameraLinkHS)
ADD_ENUM_VAL(DeviceTLType,CoaXPress)
ADD_ENUM_VAL(DeviceTLType,Custom)
ADD_ENUM_VAL(DeviceTLType,GigEVision)
ADD_ENUM_VAL(DeviceTLType,USB3Vision)
ADD_ENUM_VAL(DeviceType,CL)
ADD_ENUM_VAL(DeviceType,CLHS)
ADD_ENUM_VAL(DeviceType,Custom)
ADD_ENUM_VAL(DeviceType,CXP)
ADD_ENUM_VAL(DeviceType,ETHERNET)
ADD_ENUM_VAL(DeviceType,GEV)
ADD_ENUM_VAL(DeviceType,IIDC)
ADD_ENUM_VAL(DeviceType,Mixed)
ADD_ENUM_VAL(DeviceType,PCI)
ADD_ENUM_VAL(DeviceType,U3V)
ADD_ENUM_VAL(DeviceType,UVC)
//ADD_ENUM_VAL(EventNotification,Line3)
//ADD_ENUM_VAL(EventNotification,UserOutput0)
//ADD_ENUM_VAL(EventSelector,Line0)
//ADD_ENUM_VAL(EventSelector,Line2)
ADD_ENUM_VAL(ExposureAuto,Continuous)
ADD_ENUM_VAL(ExposureAuto,Off)
ADD_ENUM_VAL(ExposureAuto,Once)
ADD_ENUM_VAL(ExposureMode,Timed)
ADD_ENUM_VAL(ExposureMode,TriggerWidth)
//ADD_ENUM_VAL(FfcMode,Calibration)
//ADD_ENUM_VAL(FfcMode,Factory)
//ADD_ENUM_VAL(FfcMode,User)
//ADD_ENUM_VAL(FileOpenMode,)
//ADD_ENUM_VAL(FileOperationSelector,Automatic)
//ADD_ENUM_VAL(FileOperationSelector,Basic)
//ADD_ENUM_VAL(FileOperationSelector,Failure)
//ADD_ENUM_VAL(FileOperationSelector,Success)
//ADD_ENUM_VAL(FileOperationSelector,UserControlled)
//ADD_ENUM_VAL(FileOperationStatus,)
//ADD_ENUM_VAL(FileSelector,Delete)
//ADD_ENUM_VAL(FileSelector,Read)
//ADD_ENUM_VAL(FileSelector,ReadWrite)
//ADD_ENUM_VAL(FileSelector,Write)
ADD_ENUM_VAL(GainAuto,Continuous)
ADD_ENUM_VAL(GainAuto,Off)
ADD_ENUM_VAL(GainAuto,Once)
ADD_ENUM_VAL(GainSelector,All)
ADD_ENUM_VAL(GenICamXMLLocation,Device)
ADD_ENUM_VAL(GenICamXMLLocation,Host)
ADD_ENUM_VAL(GUIXMLLocation,Device)
ADD_ENUM_VAL(GUIXMLLocation,Host)
//ADD_ENUM_VAL(LineFormat,UserFile1)
//ADD_ENUM_VAL(LineInputFilterSelector,UserSet0)
//ADD_ENUM_VAL(LineInputFilterSelector,UserSetDefault)
ADD_ENUM_VAL(LineMode,Input)
//ADD_ENUM_VAL(LineMode,Trigger)
//ADD_ENUM_VAL(LineMode,UserOutput3)
ADD_ENUM_VAL(LineSelector,Line0)
ADD_ENUM_VAL(LineSelector,Line1)
ADD_ENUM_VAL(LineSelector,Line2)
ADD_ENUM_VAL(LineSelector,Line3)
//ADD_ENUM_VAL(LineSelector,OptoCoupled)
//ADD_ENUM_VAL(LineSelector,UserOutput0)
//ADD_ENUM_VAL(LineSelector,UserOutput1)
//ADD_ENUM_VAL(LineSelector,UserOutput2)
//ADD_ENUM_VAL(LineSource,UserSet1)
//ADD_ENUM_VAL(LogicBlockLUTInputActivation,Debounce)
//ADD_ENUM_VAL(LogicBlockLUTInputActivation,Deglitch)
//ADD_ENUM_VAL(LogicBlockLUTInputActivation,Input)
//ADD_ENUM_VAL(LogicBlockLUTInputActivation,Line3)
//ADD_ENUM_VAL(LogicBlockLUTInputActivation,Off)
//ADD_ENUM_VAL(LogicBlockLUTInputSelector,UserOutput1)
//ADD_ENUM_VAL(LogicBlockLUTInputSelector,UserOutput2)
//ADD_ENUM_VAL(LogicBlockLUTInputSelector,UserOutput3)
ADD_ENUM_VAL(LogicBlockLUTInputSource,AcquisitionActive)
//ADD_ENUM_VAL(LogicBlockLUTInputSource,AnyEdge)
ADD_ENUM_VAL(LogicBlockLUTInputSource,Counter0End)
ADD_ENUM_VAL(LogicBlockLUTInputSource,Counter0Start)
ADD_ENUM_VAL(LogicBlockLUTInputSource,Counter1End)
ADD_ENUM_VAL(LogicBlockLUTInputSource,Counter1Start)
ADD_ENUM_VAL(LogicBlockLUTInputSource,ExposureEnd)
ADD_ENUM_VAL(LogicBlockLUTInputSource,ExposureStart)
//ADD_ENUM_VAL(LogicBlockLUTInputSource,FallingEdge)
ADD_ENUM_VAL(LogicBlockLUTInputSource,FrameTriggerWait)
//ADD_ENUM_VAL(LogicBlockLUTInputSource,LevelHigh)
//ADD_ENUM_VAL(LogicBlockLUTInputSource,LevelLow)
ADD_ENUM_VAL(LogicBlockLUTInputSource,Line0)
ADD_ENUM_VAL(LogicBlockLUTInputSource,Line1)
ADD_ENUM_VAL(LogicBlockLUTInputSource,Line2)
ADD_ENUM_VAL(LogicBlockLUTInputSource,LogicBlock0)
ADD_ENUM_VAL(LogicBlockLUTInputSource,LogicBlock1)
//ADD_ENUM_VAL(LogicBlockLUTInputSource,RisingEdge)
//ADD_ENUM_VAL(LogicBlockLUTSelector,Line3)
//ADD_ENUM_VAL(LogicBlockLUTSelector,UserOutput0)
//ADD_ENUM_VAL(LogicBlockSelector,Line0)
//ADD_ENUM_VAL(LogicBlockSelector,Line2)
//ADD_ENUM_VAL(LUTSelector,MHzTick)
//ADD_ENUM_VAL(PixelCoding,Mono)
//ADD_ENUM_VAL(PixelCoding,MonoSigned)
//ADD_ENUM_VAL(PixelCoding,Raw)
//ADD_ENUM_VAL(PixelCoding,RGBPacked)
//ADD_ENUM_VAL(PixelCoding,YUV411Packed)
//ADD_ENUM_VAL(PixelCoding,YUV422Packed)
//ADD_ENUM_VAL(PixelCoding,YUV444Packed)
ADD_ENUM_VAL(PixelColorFilter,BayerBG)
ADD_ENUM_VAL(PixelColorFilter,BayerGB)
ADD_ENUM_VAL(PixelColorFilter,BayerGR)
ADD_ENUM_VAL(PixelColorFilter,BayerRG)
ADD_ENUM_VAL(PixelColorFilter,None)
ADD_ENUM_VAL(PixelFormat,Mono12p)
ADD_ENUM_VAL(PixelFormat,Mono12Packed)
ADD_ENUM_VAL(PixelFormat,Mono16)
ADD_ENUM_VAL(PixelFormat,Mono8)
ADD_ENUM_VAL(PixelSize,Bpp1)
ADD_ENUM_VAL(PixelSize,Bpp10)
ADD_ENUM_VAL(PixelSize,Bpp12)
ADD_ENUM_VAL(PixelSize,Bpp14)
ADD_ENUM_VAL(PixelSize,Bpp16)
ADD_ENUM_VAL(PixelSize,Bpp2)
ADD_ENUM_VAL(PixelSize,Bpp20)
ADD_ENUM_VAL(PixelSize,Bpp24)
ADD_ENUM_VAL(PixelSize,Bpp30)
ADD_ENUM_VAL(PixelSize,Bpp32)
ADD_ENUM_VAL(PixelSize,Bpp36)
ADD_ENUM_VAL(PixelSize,Bpp4)
ADD_ENUM_VAL(PixelSize,Bpp48)
ADD_ENUM_VAL(PixelSize,Bpp64)
ADD_ENUM_VAL(PixelSize,Bpp8)
ADD_ENUM_VAL(PixelSize,Bpp96)
ADD_ENUM_VAL(SensorShutterMode,Global)
ADD_ENUM_VAL(SequencerConfigurationMode,Off)
ADD_ENUM_VAL(SequencerConfigurationMode,On)
ADD_ENUM_VAL(SequencerConfigurationValid,No)
ADD_ENUM_VAL(SequencerConfigurationValid,Yes)
ADD_ENUM_VAL(SequencerFeatureSelector,ExposureTime)
ADD_ENUM_VAL(SequencerFeatureSelector,Gain)
ADD_ENUM_VAL(SequencerFeatureSelector,Height)
ADD_ENUM_VAL(SequencerFeatureSelector,OffsetX)
ADD_ENUM_VAL(SequencerFeatureSelector,OffsetY)
ADD_ENUM_VAL(SequencerFeatureSelector,Width)
ADD_ENUM_VAL(SequencerMode,Off)
ADD_ENUM_VAL(SequencerMode,On)
ADD_ENUM_VAL(SequencerSetValid,No)
ADD_ENUM_VAL(SequencerSetValid,Yes)
ADD_ENUM_VAL(SequencerTriggerSource,FrameStart)
ADD_ENUM_VAL(SequencerTriggerSource,Off)
//ADD_ENUM_VAL(TestImageSelector,Off)
//ADD_ENUM_VAL(TestImageSelector,TestImage1)
//ADD_ENUM_VAL(TestImageSelector,TestImage2)
ADD_ENUM_VAL(TestPatternGeneratorSelector,PipelineStart)
ADD_ENUM_VAL(TestPatternGeneratorSelector,Sensor)
ADD_ENUM_VAL(TestPattern,Off)
ADD_ENUM_VAL(TestPattern,SensorTestPattern)
//ADD_ENUM_VAL(TransferControlMode,)
ADD_ENUM_VAL(TriggerActivation,FallingEdge)
ADD_ENUM_VAL(TriggerActivation,RisingEdge)
ADD_ENUM_VAL(TriggerMode,Off)
ADD_ENUM_VAL(TriggerMode,On)
ADD_ENUM_VAL(TriggerOverlap,Off)
ADD_ENUM_VAL(TriggerOverlap,ReadOut)
ADD_ENUM_VAL(TriggerSelector,AcquisitionStart)
//ADD_ENUM_VAL(TriggerSelector,ExposureActive)
ADD_ENUM_VAL(TriggerSelector,FrameBurstStart)
ADD_ENUM_VAL(TriggerSelector,FrameStart)
ADD_ENUM_VAL(TriggerSource,Counter0End)
ADD_ENUM_VAL(TriggerSource,Counter0Start)
ADD_ENUM_VAL(TriggerSource,Counter1End)
ADD_ENUM_VAL(TriggerSource,Counter1Start)
ADD_ENUM_VAL(TriggerSource,Line0)
ADD_ENUM_VAL(TriggerSource,Line2)
ADD_ENUM_VAL(TriggerSource,Line3)
ADD_ENUM_VAL(TriggerSource,LogicBlock0)
ADD_ENUM_VAL(TriggerSource,LogicBlock1)
ADD_ENUM_VAL(TriggerSource,Software)
ADD_ENUM_VAL(TriggerSource,UserOutput0)
ADD_ENUM_VAL(TriggerSource,UserOutput1)
ADD_ENUM_VAL(TriggerSource,UserOutput2)
ADD_ENUM_VAL(TriggerSource,UserOutput3)
ADD_ENUM_VAL(U3VCurrentSpeed,FullSpeed)
ADD_ENUM_VAL(U3VCurrentSpeed,HighSpeed)
ADD_ENUM_VAL(U3VCurrentSpeed,LowSpeed)
ADD_ENUM_VAL(U3VCurrentSpeed,SuperSpeed)
//ADD_ENUM_VAL(UserOutputSelector,Close)
//ADD_ENUM_VAL(UserOutputSelector,Open)
//ADD_ENUM_VAL(UserOutputSelector,Read)
//ADD_ENUM_VAL(UserOutputSelector,Write)
ADD_ENUM_VAL(UserSetDefault,Default)
//ADD_ENUM_VAL(UserSetDefaultSelector,Default)
//ADD_ENUM_VAL(UserSetDefaultSelector,UserSet1)
//ADD_ENUM_VAL(UserSetDefaultSelector,UserSet2)
ADD_ENUM_VAL(UserSetDefault,UserSet0)
ADD_ENUM_VAL(UserSetDefault,UserSet1)
ADD_ENUM_VAL(UserSetFeatureSelector,AasRoiEnableAe)
ADD_ENUM_VAL(UserSetFeatureSelector,AasRoiEnableAwb)
ADD_ENUM_VAL(UserSetFeatureSelector,AasRoiHeightAe)
ADD_ENUM_VAL(UserSetFeatureSelector,AasRoiHeightAwb)
ADD_ENUM_VAL(UserSetFeatureSelector,AasRoiOffsetXAe)
ADD_ENUM_VAL(UserSetFeatureSelector,AasRoiOffsetXAwb)
ADD_ENUM_VAL(UserSetFeatureSelector,AasRoiOffsetYAe)
ADD_ENUM_VAL(UserSetFeatureSelector,AasRoiOffsetYAwb)
ADD_ENUM_VAL(UserSetFeatureSelector,AasRoiWidthAe)
ADD_ENUM_VAL(UserSetFeatureSelector,AasRoiWidthAwb)
ADD_ENUM_VAL(UserSetFeatureSelector,AcquisitionBurstFrameCount)
ADD_ENUM_VAL(UserSetFeatureSelector,AcquisitionFrameCount)
ADD_ENUM_VAL(UserSetFeatureSelector,AcquisitionFrameRate)
ADD_ENUM_VAL(UserSetFeatureSelector,AcquisitionFrameRateEnable)
ADD_ENUM_VAL(UserSetFeatureSelector,AcquisitionLineRate)
ADD_ENUM_VAL(UserSetFeatureSelector,AcquisitionMode)
ADD_ENUM_VAL(UserSetFeatureSelector,AdcBitDepth)
ADD_ENUM_VAL(UserSetFeatureSelector,AutoExposureControlLoopDamping)
ADD_ENUM_VAL(UserSetFeatureSelector,AutoExposureControlPriority)
ADD_ENUM_VAL(UserSetFeatureSelector,AutoExposureEVCompensation)
ADD_ENUM_VAL(UserSetFeatureSelector,AutoExposureExposureTimeLowerLimit)
ADD_ENUM_VAL(UserSetFeatureSelector,AutoExposureExposureTimeUpperLimit)
ADD_ENUM_VAL(UserSetFeatureSelector,AutoExposureGainLowerLimit)
ADD_ENUM_VAL(UserSetFeatureSelector,AutoExposureGainUpperLimit)
ADD_ENUM_VAL(UserSetFeatureSelector,AutoExposureGreyValueLowerLimit)
ADD_ENUM_VAL(UserSetFeatureSelector,AutoExposureGreyValueUpperLimit)
ADD_ENUM_VAL(UserSetFeatureSelector,AutoExposureLightingMode)
ADD_ENUM_VAL(UserSetFeatureSelector,AutoExposureMeteringMode)
ADD_ENUM_VAL(UserSetFeatureSelector,AutoExposureTargetGreyValue)
ADD_ENUM_VAL(UserSetFeatureSelector,AutoExposureTargetGreyValueAuto)
ADD_ENUM_VAL(UserSetFeatureSelector,BalanceRatioBlue)
ADD_ENUM_VAL(UserSetFeatureSelector,BalanceRatioRed)
ADD_ENUM_VAL(UserSetFeatureSelector,BalanceWhiteAuto)
ADD_ENUM_VAL(UserSetFeatureSelector,BalanceWhiteAutoDamping)
ADD_ENUM_VAL(UserSetFeatureSelector,BalanceWhiteAutoLowerLimit)
ADD_ENUM_VAL(UserSetFeatureSelector,BalanceWhiteAutoProfile)
ADD_ENUM_VAL(UserSetFeatureSelector,BalanceWhiteAutoUpperLimit)
ADD_ENUM_VAL(UserSetFeatureSelector,BinningHorizontalAll)
ADD_ENUM_VAL(UserSetFeatureSelector,BinningHorizontalMode)
ADD_ENUM_VAL(UserSetFeatureSelector,BinningVerticalAll)
ADD_ENUM_VAL(UserSetFeatureSelector,BinningVerticalMode)
ADD_ENUM_VAL(UserSetFeatureSelector,BlackLevelAll)
ADD_ENUM_VAL(UserSetFeatureSelector,ChunkEnableAll)
ADD_ENUM_VAL(UserSetFeatureSelector,ChunkModeActive)
ADD_ENUM_VAL(UserSetFeatureSelector,ColorTransformationEnable)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterDelayCounter0)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterDelayCounter1)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterDurationCounter0)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterDurationCounter1)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterEventActivationCounter0)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterEventActivationCounter1)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterEventSourceCounter0)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterEventSourceCounter1)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterResetActivationCounter0)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterResetActivationCounter1)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterResetSourceCounter0)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterResetSourceCounter1)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterTriggerActivationCounter0)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterTriggerActivationCounter1)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterTriggerSourceCounter0)
ADD_ENUM_VAL(UserSetFeatureSelector,CounterTriggerSourceCounter1)
//ADD_ENUM_VAL(UserSetFeatureSelector,CRC)
ADD_ENUM_VAL(UserSetFeatureSelector,DecimationHorizontalAll)
ADD_ENUM_VAL(UserSetFeatureSelector,DecimationVerticalAll)
ADD_ENUM_VAL(UserSetFeatureSelector,DefectCorrectionMode)
ADD_ENUM_VAL(UserSetFeatureSelector,DefectCorrectStaticEnable)
ADD_ENUM_VAL(UserSetFeatureSelector,DeviceIndicatorMode)
ADD_ENUM_VAL(UserSetFeatureSelector,DeviceLinkBandwidthReserve)
ADD_ENUM_VAL(UserSetFeatureSelector,DeviceLinkThroughputLimit)
ADD_ENUM_VAL(UserSetFeatureSelector,EvCompensationRaw)
ADD_ENUM_VAL(UserSetFeatureSelector,EventNotificationError)
ADD_ENUM_VAL(UserSetFeatureSelector,EventNotificationExposureEnd)
ADD_ENUM_VAL(UserSetFeatureSelector,ExposureActiveMode)
ADD_ENUM_VAL(UserSetFeatureSelector,ExposureAuto)
ADD_ENUM_VAL(UserSetFeatureSelector,ExposureMode)
ADD_ENUM_VAL(UserSetFeatureSelector,ExposureTime)
ADD_ENUM_VAL(UserSetFeatureSelector,FfcEnable)
ADD_ENUM_VAL(UserSetFeatureSelector,FfcMode)
ADD_ENUM_VAL(UserSetFeatureSelector,GainAll)
ADD_ENUM_VAL(UserSetFeatureSelector,GainAuto)
ADD_ENUM_VAL(UserSetFeatureSelector,Gamma)
ADD_ENUM_VAL(UserSetFeatureSelector,GammaEnable)
ADD_ENUM_VAL(UserSetFeatureSelector,Height)
ADD_ENUM_VAL(UserSetFeatureSelector,IspEnable)
ADD_ENUM_VAL(UserSetFeatureSelector,LineFilterWidthLine0Debounce)
ADD_ENUM_VAL(UserSetFeatureSelector,LineFilterWidthLine0Deglitch)
ADD_ENUM_VAL(UserSetFeatureSelector,LineFilterWidthLine1Debounce)
ADD_ENUM_VAL(UserSetFeatureSelector,LineFilterWidthLine1Deglitch)
ADD_ENUM_VAL(UserSetFeatureSelector,LineFilterWidthLine2Debounce)
ADD_ENUM_VAL(UserSetFeatureSelector,LineFilterWidthLine2Deglitch)
ADD_ENUM_VAL(UserSetFeatureSelector,LineFilterWidthLine3Debounce)
ADD_ENUM_VAL(UserSetFeatureSelector,LineFilterWidthLine3Deglitch)
ADD_ENUM_VAL(UserSetFeatureSelector,LineInverterLine0)
ADD_ENUM_VAL(UserSetFeatureSelector,LineInverterLine1)
ADD_ENUM_VAL(UserSetFeatureSelector,LineInverterLine2)
ADD_ENUM_VAL(UserSetFeatureSelector,LineInverterLine3)
ADD_ENUM_VAL(UserSetFeatureSelector,LineModeLine0)
ADD_ENUM_VAL(UserSetFeatureSelector,LineModeLine1)
ADD_ENUM_VAL(UserSetFeatureSelector,LineModeLine2)
ADD_ENUM_VAL(UserSetFeatureSelector,LineModeLine3)
ADD_ENUM_VAL(UserSetFeatureSelector,LineSourceLine0)
ADD_ENUM_VAL(UserSetFeatureSelector,LineSourceLine1)
ADD_ENUM_VAL(UserSetFeatureSelector,LineSourceLine2)
ADD_ENUM_VAL(UserSetFeatureSelector,LineSourceLine3)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputActivationLogicBlock0Input0)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputActivationLogicBlock0Input1)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputActivationLogicBlock0Input2)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputActivationLogicBlock0Input3)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputActivationLogicBlock1Input0)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputActivationLogicBlock1Input1)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputActivationLogicBlock1Input2)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputActivationLogicBlock1Input3)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputSourceLogicBlock0Input0)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputSourceLogicBlock0Input1)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputSourceLogicBlock0Input2)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputSourceLogicBlock0Input3)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputSourceLogicBlock1Input0)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputSourceLogicBlock1Input1)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputSourceLogicBlock1Input2)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTInputSourceLogicBlock1Input3)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTOutputValueAllLogicBlock0Enable)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTOutputValueAllLogicBlock0Value)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTOutputValueAllLogicBlock1Enable)
ADD_ENUM_VAL(UserSetFeatureSelector,LogicBlockLUTOutputValueAllLogicBlock1Value)
ADD_ENUM_VAL(UserSetFeatureSelector,LUTEnable)
ADD_ENUM_VAL(UserSetFeatureSelector,OffsetX)
ADD_ENUM_VAL(UserSetFeatureSelector,OffsetY)
ADD_ENUM_VAL(UserSetFeatureSelector,PixelFormat)
ADD_ENUM_VAL(UserSetFeatureSelector,ReverseX)
ADD_ENUM_VAL(UserSetFeatureSelector,ReverseY)
ADD_ENUM_VAL(UserSetFeatureSelector,RgbTransformLightSource)
ADD_ENUM_VAL(UserSetFeatureSelector,Saturation)
ADD_ENUM_VAL(UserSetFeatureSelector,SaturationEnable)
ADD_ENUM_VAL(UserSetFeatureSelector,SensorShutterMode)
ADD_ENUM_VAL(UserSetFeatureSelector,Sharpening)
ADD_ENUM_VAL(UserSetFeatureSelector,SharpeningAuto)
ADD_ENUM_VAL(UserSetFeatureSelector,SharpeningEnable)
ADD_ENUM_VAL(UserSetFeatureSelector,SharpeningThreshold)
ADD_ENUM_VAL(UserSetFeatureSelector,TestPatternPipelineStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TestPatternSensor)
ADD_ENUM_VAL(UserSetFeatureSelector,TransferBlockCount)
ADD_ENUM_VAL(UserSetFeatureSelector,TransferControlMode)
ADD_ENUM_VAL(UserSetFeatureSelector,TransferOperationMode)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerActivationAcquisitionStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerActivationFrameBurstStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerActivationFrameStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerDelayAcquisitionStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerDelayFrameBurstStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerDelayFrameStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerModeAcquisitionStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerModeFrameBurstStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerModeFrameStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerOverlapAcquisitionStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerOverlapFrameBurstStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerOverlapFrameStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerSourceAcquisitionStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerSourceFrameBurstStart)
ADD_ENUM_VAL(UserSetFeatureSelector,TriggerSourceFrameStart)
ADD_ENUM_VAL(UserSetFeatureSelector,UserOutputValueAll)
ADD_ENUM_VAL(UserSetFeatureSelector,Width)
ADD_ENUM_VAL(UserSetSelector,Default)
ADD_ENUM_VAL(UserSetSelector,UserSet0)
ADD_ENUM_VAL(UserSetSelector,UserSet1)
//ADD_ENUM_VAL(UserSetSelector,UserSet2)
//ADD_ENUM_VAL(V3,FrameID)
//ADD_ENUM_VAL(V3,Height)
//ADD_ENUM_VAL(V3,Image)
//ADD_ENUM_VAL(V3,OffsetY)
//ADD_ENUM_VAL(VideoMode,Mode0)
//ADD_ENUM_VAL(VideoMode,Mode1)

#ifdef FOOBAR
ADD_ENUM_VAL(DeviceType,Mixed)
ADD_ENUM_VAL(DeviceType,Custom)
ADD_ENUM_VAL(DeviceType,GEV)
ADD_ENUM_VAL(DeviceType,CL)
ADD_ENUM_VAL(DeviceType,IIDC)
ADD_ENUM_VAL(DeviceType,UVC)
ADD_ENUM_VAL(DeviceType,CXP)
ADD_ENUM_VAL(DeviceType,CLHS)
ADD_ENUM_VAL(DeviceType,U3V)
ADD_ENUM_VAL(DeviceType,ETHERNET)
ADD_ENUM_VAL(DeviceType,PCI)
ADD_ENUM_VAL(DeviceAccessStatus,Unknown)
ADD_ENUM_VAL(DeviceAccessStatus,ReadWrite)
ADD_ENUM_VAL(DeviceAccessStatus,ReadOnly)
ADD_ENUM_VAL(DeviceAccessStatus,NoAccess)
ADD_ENUM_VAL(GUIXMLLocation,Device)
ADD_ENUM_VAL(GUIXMLLocation,Host)
ADD_ENUM_VAL(GenICamXMLLocation,Device)
ADD_ENUM_VAL(GenICamXMLLocation,Host)
ADD_ENUM_VAL(DeviceCurrentSpeed,UnknownSpeed)
ADD_ENUM_VAL(DeviceCurrentSpeed,LowSpeed)
ADD_ENUM_VAL(DeviceCurrentSpeed,FullSpeed)
ADD_ENUM_VAL(DeviceCurrentSpeed,HighSpeed)
ADD_ENUM_VAL(DeviceCurrentSpeed,SuperSpeed)
ADD_ENUM_VAL(DeviceEndianessMechanism,Legacy)
ADD_ENUM_VAL(DeviceEndianessMechanism,Standard)
ADD_ENUM_VAL(GainAuto,Off)
ADD_ENUM_VAL(GainAuto,Once)
ADD_ENUM_VAL(GainAuto,Continuous)
ADD_ENUM_VAL(DeviceScanType,Areascan)
ADD_ENUM_VAL(TriggerSelector,FrameStart)
//ADD_ENUM_VAL(TriggerSelector,ExposureActive)
ADD_ENUM_VAL(TriggerMode,Off)
ADD_ENUM_VAL(TriggerMode,On)
ADD_ENUM_VAL(TriggerSource,Software)
ADD_ENUM_VAL(TriggerSource,Line0)
ADD_ENUM_VAL(TriggerSource,Line2)
ADD_ENUM_VAL(TriggerSource,Line3)
ADD_ENUM_VAL(TriggerActivation,RisingEdge)
ADD_ENUM_VAL(TriggerActivation,FallingEdge)
ADD_ENUM_VAL(ExposureMode,Timed)
ADD_ENUM_VAL(ExposureMode,TriggerWidth)
ADD_ENUM_VAL(ExposureAuto,Off)
ADD_ENUM_VAL(ExposureAuto,Once)
ADD_ENUM_VAL(ExposureAuto,Continuous)
ADD_ENUM_VAL(AcquisitionMode,Continuous)
ADD_ENUM_VAL(AcquisitionMode,SingleFrame)
ADD_ENUM_VAL(AcquisitionMode,MultiFrame)
//ADD_ENUM_VAL(AcquisitionFrameRateAuto,Off)
//ADD_ENUM_VAL(AcquisitionFrameRateAuto,Continuous)
ADD_ENUM_VAL(PixelFormat,Mono8)
ADD_ENUM_VAL(PixelFormat,Mono12p)
ADD_ENUM_VAL(PixelFormat,Mono16)
//ADD_ENUM_VAL(VideoMode,Mode0)
//ADD_ENUM_VAL(VideoMode,Mode1)
//ADD_ENUM_VAL(PixelCoding,Mono)
//ADD_ENUM_VAL(PixelCoding,MonoSigned)
//ADD_ENUM_VAL(PixelCoding,RGBPacked)
//ADD_ENUM_VAL(PixelCoding,YUV411Packed)
//ADD_ENUM_VAL(PixelCoding,YUV422Packed)
//ADD_ENUM_VAL(PixelCoding,YUV444Packed)
//ADD_ENUM_VAL(PixelCoding,Raw)
ADD_ENUM_VAL(PixelSize,Bpp8)
ADD_ENUM_VAL(PixelSize,Bpp10)
ADD_ENUM_VAL(PixelSize,Bpp12)
ADD_ENUM_VAL(PixelSize,Bpp16)
ADD_ENUM_VAL(PixelSize,Bpp24)
ADD_ENUM_VAL(PixelSize,Bpp32)
ADD_ENUM_VAL(PixelColorFilter,BayerRG)
ADD_ENUM_VAL(PixelColorFilter,BayerGB)
ADD_ENUM_VAL(PixelColorFilter,BayerGR)
ADD_ENUM_VAL(PixelColorFilter,BayerBG)
ADD_ENUM_VAL(PixelColorFilter,None)
//ADD_ENUM_VAL(TestImageSelector,Off)
//ADD_ENUM_VAL(TestImageSelector,TestImage1)
//ADD_ENUM_VAL(TestImageSelector,TestImage2)
ADD_ENUM_VAL(UserSetSelector,Default)
ADD_ENUM_VAL(UserSetSelector,UserSet1)
//ADD_ENUM_VAL(UserSetSelector,UserSet2)
//ADD_ENUM_VAL(UserSetDefaultSelector,Default)
//ADD_ENUM_VAL(UserSetDefaultSelector,UserSet1)
//ADD_ENUM_VAL(UserSetDefaultSelector,UserSet2)
ADD_ENUM_VAL(LineSelector,Line0)
ADD_ENUM_VAL(LineSelector,Line1)
ADD_ENUM_VAL(LineSelector,Line2)
ADD_ENUM_VAL(LineSelector,Line3)
ADD_ENUM_VAL(LineMode,Input)
//ADD_ENUM_VAL(LineMode,Trigger)
ADD_ENUM_VAL(U3VCurrentSpeed,LowSpeed)
ADD_ENUM_VAL(U3VCurrentSpeed,FullSpeed)
ADD_ENUM_VAL(U3VCurrentSpeed,HighSpeed)
ADD_ENUM_VAL(U3VCurrentSpeed,SuperSpeed)
ADD_ENUM_VAL(ChunkSelector,Image)
ADD_ENUM_VAL(ChunkSelector,CRC)
//ADD_ENUM_VAL(ChunkSelector,FrameCounter)
ADD_ENUM_VAL(ChunkSelector,OffsetX)
ADD_ENUM_VAL(ChunkSelector,OffsetY)
ADD_ENUM_VAL(ChunkSelector,Width)
ADD_ENUM_VAL(ChunkSelector,Height)
ADD_ENUM_VAL(ChunkSelector,ExposureTime)
ADD_ENUM_VAL(ChunkSelector,Gain)
ADD_ENUM_VAL(ChunkSelector,BlackLevel)
#endif // FOOBAR

#ifdef FOOBAR
ADD_ENUM_VAL(GenICamXMLLocation,Device)
ADD_ENUM_VAL(GenICamXMLLocation,Host)
ADD_ENUM_VAL(DeviceCurrentSpeed,UnknownSpeed)
ADD_ENUM_VAL(DeviceCurrentSpeed,LowSpeed)
ADD_ENUM_VAL(DeviceCurrentSpeed,FullSpeed)
ADD_ENUM_VAL(DeviceCurrentSpeed,HighSpeed)
ADD_ENUM_VAL(DeviceCurrentSpeed,SuperSpeed)
ADD_ENUM_VAL(DeviceEndianessMechanism,Legacy)
ADD_ENUM_VAL(DeviceEndianessMechanism,Standard)
ADD_ENUM_VAL(AcquisitionMode,Continuous)
ADD_ENUM_VAL(AcquisitionMode,SingleFrame)
ADD_ENUM_VAL(AcquisitionMode,MultiFrame)
ADD_ENUM_VAL(ExposureMode,Timed)
ADD_ENUM_VAL(ExposureMode,TriggerWidth)
ADD_ENUM_VAL(ExposureAuto,Off)
ADD_ENUM_VAL(ExposureAuto,Once)
ADD_ENUM_VAL(ExposureAuto,Continuous)
ADD_ENUM_VAL(TriggerSelector,AcquisitionStart)
ADD_ENUM_VAL(TriggerSelector,FrameStart)
ADD_ENUM_VAL(TriggerSelector,FrameBurstStart)
ADD_ENUM_VAL(TriggerMode,Off)
ADD_ENUM_VAL(TriggerMode,On)
ADD_ENUM_VAL(TriggerSource,Software)
ADD_ENUM_VAL(TriggerSource,Line0)
ADD_ENUM_VAL(TriggerSource,Line2)
ADD_ENUM_VAL(TriggerSource,Line3)
ADD_ENUM_VAL(TriggerSource,UserOutput0)
ADD_ENUM_VAL(TriggerSource,UserOutput1)
ADD_ENUM_VAL(TriggerSource,UserOutput2)
ADD_ENUM_VAL(TriggerSource,UserOutput3)
ADD_ENUM_VAL(TriggerSource,Counter0Start)
ADD_ENUM_VAL(TriggerSource,Counter1Start)
ADD_ENUM_VAL(TriggerSource,Counter0End)
ADD_ENUM_VAL(TriggerSource,Counter1End)
ADD_ENUM_VAL(TriggerSource,LogicBlock0)
ADD_ENUM_VAL(TriggerSource,LogicBlock1)
ADD_ENUM_VAL(TriggerOverlap,Off)
ADD_ENUM_VAL(TriggerOverlap,ReadOut)
ADD_ENUM_VAL(SensorShutterMode,Global)
ADD_ENUM_VAL(GainSelector,All)
ADD_ENUM_VAL(GainAuto,Off)
ADD_ENUM_VAL(GainAuto,Once)
ADD_ENUM_VAL(GainAuto,Continuous)
ADD_ENUM_VAL(BlackLevelSelector,All)
ADD_ENUM_VAL(BlackLevelSelector,Analog)
ADD_ENUM_VAL(BlackLevelSelector,Digital)
ADD_ENUM_VAL(PixelFormat,Mono8)
ADD_ENUM_VAL(PixelFormat,Mono16)
ADD_ENUM_VAL(PixelFormat,Mono12Packed)
ADD_ENUM_VAL(PixelFormat,Mono12p)
ADD_ENUM_VAL(PixelSize,Bpp1)
ADD_ENUM_VAL(PixelSize,Bpp2)
ADD_ENUM_VAL(PixelSize,Bpp4)
ADD_ENUM_VAL(PixelSize,Bpp8)
ADD_ENUM_VAL(PixelSize,Bpp10)
ADD_ENUM_VAL(PixelSize,Bpp12)
ADD_ENUM_VAL(PixelSize,Bpp14)
ADD_ENUM_VAL(PixelSize,Bpp16)
ADD_ENUM_VAL(PixelSize,Bpp20)
ADD_ENUM_VAL(PixelSize,Bpp24)
ADD_ENUM_VAL(PixelSize,Bpp30)
ADD_ENUM_VAL(PixelSize,Bpp32)
ADD_ENUM_VAL(PixelSize,Bpp36)
ADD_ENUM_VAL(PixelSize,Bpp48)
ADD_ENUM_VAL(PixelSize,Bpp64)
ADD_ENUM_VAL(PixelSize,Bpp96)
ADD_ENUM_VAL(PixelColorFilter,None)
ADD_ENUM_VAL(PixelColorFilter,BayerRG)
ADD_ENUM_VAL(PixelColorFilter,BayerGB)
ADD_ENUM_VAL(PixelColorFilter,BayerGR)
ADD_ENUM_VAL(PixelColorFilter,BayerBG)
ADD_ENUM_VAL(BinningSelector,All)
ADD_ENUM_VAL(BinningSelector,Sensor)
ADD_ENUM_VAL(BinningSelector,ISP)
ADD_ENUM_VAL(BinningHorizontalMode,Sum)
ADD_ENUM_VAL(BinningHorizontalMode,Average)
ADD_ENUM_VAL(BinningVerticalMode,Sum)
ADD_ENUM_VAL(BinningVerticalMode,Average)
ADD_ENUM_VAL(DecimationSelector,All)
ADD_ENUM_VAL(DecimationSelector,Sensor)
ADD_ENUM_VAL(DecimationHorizontalMode,Discard)
#endif // FOOBAR

	enums_inited = 1;
}

static int _register_one_node(QSP_ARG_DECL  spinNodeHandle hNode, int level)
{
	char name[LLEN];
	size_t l=LLEN;
	Spink_Node *skn_p;
	spinNodeType type;
	int n;

	if( get_node_name(name,&l,hNode) < 0 )
		error1("register_one_node:  error getting node name!?");

	skn_p = new_spink_node(name);
	assert(skn_p!=NULL);

	assert(current_map!=NULL);
	skn_p->skn_skm_p = current_map;
	if( level == 0 ){
		assert(current_map->skm_root_p == NULL);
		assert(!strcmp(name,"Root"));
		current_map->skm_root_p = skn_p;
		skn_p->skn_parent = NULL;
		skn_p->skn_idx = (-1);	// because no parent
		//assert(current_parent_p==NULL);
	} else {
		int idx;
		idx=level-1;
		assert(idx<MAX_TREE_DEPTH);
		skn_p->skn_parent = current_parent_p[idx];
		skn_p->skn_idx = current_node_idx;	// set in traverse...
	}
	current_parent_p[level] = skn_p;

	skn_p->skn_flags = 0;
	if( spink_node_is_readable(hNode) ){
		skn_p->skn_flags |= NODE_READABLE;
	}
	if( spink_node_is_writable(hNode) ){
		skn_p->skn_flags |= NODE_WRITABLE;
	}

	n = get_display_name_len(hNode);
	if( n > max_display_name_len )
		max_display_name_len = n;

	if( get_node_type(hNode,&type) < 0 ) return -1;
	skn_p->skn_type_p = find_type_by_code(type);
	assert(skn_p->skn_type_p!=NULL);

	// don't make a category for the root node
	if( level > 0 && type == CategoryNode ){
		Spink_Category *sct_p;
		sct_p = spink_cat_of(skn_p->skn_name);
		assert(sct_p==NULL); 
		sct_p = new_spink_cat(skn_p->skn_name);
		assert(sct_p!=NULL); 
		sct_p->sct_root_p = skn_p;
	}

	skn_p->skn_enum_val = INVALID_ENUM_VAL;
	if( type == EnumEntryNode ){
		Spink_Enum_Val *sev_p;

		if( ! enums_inited ) init_enums();

		sev_p = spink_enum_val_of(skn_p->skn_name);
		//assert(sev_p!=NULL);

		if( sev_p == NULL ){
			sprintf(ERROR_STRING,"No enum value defined for %s!?",skn_p->skn_name);
			warn(ERROR_STRING);
			skn_p->skn_enum_val = INVALID_ENUM_VAL;
		} else {
			skn_p->skn_enum_val = sev_p->sev_value;
		}
	}

	skn_p->skn_children = NULL;
	skn_p->skn_level = level;

//fprintf(stderr,"register_one_node:  %s   flags = %d\n",skn_p->skn_name,skn_p->skn_flags);

	//skn_p->skn_handle = hNode;
	return 0;
}

// We are duplicating the node tree with our own structures,
// but it is difficult to link the structs as we build the tree,
// so we do it after the fact using the parent pointers.

static void add_child_to_parent(Spink_Node *skn_p)
{
	Node *np;

	np = mk_node(skn_p);
	if( skn_p->skn_parent->skn_children == NULL )
		skn_p->skn_parent->skn_children = new_list();
	assert( skn_p->skn_parent->skn_children != NULL );

	addHead(skn_p->skn_parent->skn_children,np);
}

#define build_child_lists() _build_child_lists(SINGLE_QSP_ARG)

static void _build_child_lists(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;
	
	lp = spink_node_list();
	assert(lp!=NULL);
	assert(eltcount(lp)>0);

	np = QLIST_HEAD(lp);
	while(np!=NULL){
		Spink_Node *skn_p;
		skn_p = NODE_DATA(np);
		if( skn_p->skn_parent != NULL ){
			add_child_to_parent(skn_p);
		} else {
			assert( skn_p->skn_idx == (-1) );	// root node
		}
		np = NODE_NEXT(np);
	}
}

void _pop_map_contexts(SINGLE_QSP_ARG_DECL)
{
	pop_spink_node_context();
	pop_spink_cat_context();
}

void _push_map_contexts(QSP_ARG_DECL  Spink_Map *skm_p)
{
	push_spink_node_context(skm_p->skm_node_icp);
	push_spink_cat_context(skm_p->skm_cat_icp);
}

#define register_map_nodes(hMap,skm_p) _register_map_nodes(QSP_ARG  hMap,skm_p)

static void _register_map_nodes(QSP_ARG_DECL  spinNodeMapHandle hMap, Spink_Map *skm_p)
{
	spinNodeHandle hRoot=NULL;

//fprintf(stderr,"register_map_nodes BEGIN   hMap = 0x%lx\n",(u_long)hMap);

	//push_spink_node_context(skm_p->skm_icp);
	push_map_contexts(skm_p);
//fprintf(stderr,"register_map_nodes fetching root node   hMap = 0x%lx\n",(u_long)hMap);
	if( fetch_spink_node(hMap, "Root", &hRoot) < 0 )
		error1("register_map_nodes:  error fetching map root node");

//fprintf(stderr,"register_map_nodes:  root node fetched, traversing...\n");
	current_map = skm_p;
	//current_parent_p = NULL;
	skm_p->skm_root_p = NULL;
	if( traverse_by_node_handle(hRoot,0,_register_one_node) < 0 )
		error1("error traversing node map");
	current_map = NULL;

	// Do this before popping the context!!!
	build_child_lists();

	pop_map_contexts();

}


#define register_one_map(skc_p, code, name) _register_one_map(QSP_ARG  skc_p, code, name)

static void _register_one_map(QSP_ARG_DECL  Spink_Cam *skc_p, Node_Map_Type type, const char *name)
{
	Spink_Map *skm_p;
	spinNodeMapHandle hMap = NULL;

//fprintf(stderr,"register_one_map %s BEGIN, type = %d\n",name,type);
	insure_current_camera(skc_p);
	assert( skc_p->skc_current_handle != NULL );
//fprintf(stderr,"register_one_map:  %s has current handle 0x%lx\n", skc_p->skc_name,(u_long)skc_p->skc_current_handle);

	skm_p = new_spink_map(name);
	if( skm_p == NULL ) error1("Unable to create map struct!?");
//fprintf(stderr,"Created new map struct %s at 0x%lx\n", skm_p->skm_name,(u_long)skm_p);


	if( spink_node_itp == NULL ) init_spink_nodes();
	skm_p->skm_node_icp = create_item_context(spink_node_itp,name);
	assert(skm_p->skm_node_icp!=NULL);

	if( spink_cat_itp == NULL ) init_spink_cats();
	skm_p->skm_cat_icp = create_item_context(spink_cat_itp,name);
	assert(skm_p->skm_cat_icp!=NULL);

	// do we need to push the context too???

	//skm_p->skm_handle = NULL;
	skm_p->skm_type = type;
	skm_p->skm_skc_p = skc_p;

//	fetch_map_handle(skm_p);
//fprintf(stderr,"register_one_map calling get_node_map_handle...\n");
	get_node_map_handle(&hMap,skm_p,"register_one_map");	// first time just sets
//fprintf(stderr,"register_one_map:  hMap = 0x%lx, *hMap = 0x%lx \n",(u_long)hMap, (u_long)*((void **)hMap));

	register_map_nodes(hMap,skm_p);
}

#define register_cam_nodemaps(skc_p) _register_cam_nodemaps(QSP_ARG  skc_p)

static void _register_cam_nodemaps(QSP_ARG_DECL  Spink_Cam *skc_p)
{
//fprintf(stderr,"register_cam_nodemaps BEGIN\n");
//fprintf(stderr,"register_cam_nodemaps registering device map\n");
	sprintf(MSG_STR,"%s.device_TL",skc_p->skc_name);
	register_one_map(skc_p,DEV_NODE_MAP,MSG_STR);
//fprintf(stderr,"register_cam_nodemaps registering camera map\n");
	sprintf(MSG_STR,"%s.genicam",skc_p->skc_name);
	register_one_map(skc_p,CAM_NODE_MAP,MSG_STR);
//fprintf(stderr,"register_cam_nodemaps DONE\n");
}

#define init_one_spink_cam(idx) _init_one_spink_cam(QSP_ARG  idx)

static int _init_one_spink_cam(QSP_ARG_DECL  int idx)
{
	spinCamera hCam;
//	spinNodeMapHandle hNodeMap;
	Spink_Cam *skc_p;
	char buf[MAX_BUFF_LEN];
	size_t len = MAX_BUFF_LEN;

	if( get_cam_from_list(hCameraList,idx,&hCam) < 0 )
		return -1;
//fprintf(stderr,"init_one_spink_cam:  get_cam_from_list returned 0x%lx\n",(u_long)hCam);

	if( get_camera_model_name(buf,len,hCam) < 0 ) return -1;
	substitute_char(buf,' ','_');
	// Check and see if another camera of this type has already
	// been detected...
	skc_p = spink_cam_of(buf);
	if( skc_p != NULL ){
		if( get_unique_cam_name(buf,MAX_BUFF_LEN) < 0 )
			return -1;
	}
	skc_p = new_spink_cam(buf);
	if( skc_p == NULL ) return -1;
	skc_p->skc_current_handle = hCam;
//fprintf(stderr,"init_one_spink_cam:  setting current handle to 0x%lx\n",(u_long)hCam);

	//skc_p->skc_handle = hCam;
	skc_p->skc_sys_idx = idx;
	skc_p->skc_iface_idx = -1;	// invalid value

	// register_cam_nodemaps will get the camera handle again...
//	if( release_spink_cam(hCam) < 0 )
//		return -1;

	//skc_p->skc_TL_dev_node_map = hNodeMapTLDevice;
	//skc_p->skc_genicam_node_map = hNodeMap;

	register_cam_nodemaps(skc_p);
	//skc_p->skc_flags = SPINK_CAM_CONNECTED;

	// Make a data_obj context for the frames...
	skc_p->skc_do_icp = create_dobj_context( QSP_ARG  skc_p->skc_name );
	assert(skc_p->skc_do_icp != NULL);

	// We have to explicitly release here, as we weren't able to call
	// insure_current_camera at the beginning...
	//spink_release_cam(skc_p);
	release_current_camera();

	return 0;
}

#define create_spink_camera_structs() _create_spink_camera_structs(SINGLE_QSP_ARG)

static int _create_spink_camera_structs(SINGLE_QSP_ARG_DECL)
{
	int i;

	for(i=0;i<numCameras;i++){
		if( init_one_spink_cam(i) < 0 )
			return -1;
	}
	return 0;
} // end create_spink_camera_structs

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

		//ski_p->ski_handle = hInterface;
		ski_p->ski_idx = i;

		if( release_interface(hInterface) < 0 )
			return -1;
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

	// We get the cameras from the system, not from individual interfaces...
	if( get_spink_cameras(hSystem,&hCameraList,&numCameras) < 0 ) return -1;
	if( create_spink_camera_structs() < 0 ) return -1;

	do_on_exit(_release_spink_cam_system);

#endif // HAVE_LIBSPINNAKER
	return 0;
}


#define release_spink_interface_structs()	_release_spink_interface_structs(SINGLE_QSP_ARG)

static int _release_spink_interface_structs(SINGLE_QSP_ARG_DECL)
{
	// iterate through the list
	Node *np;
	List *lp;
	Spink_Interface *ski_p;

	lp = spink_interface_list();
	if( lp == NULL ) return 0;

	while( (np=remHead(lp)) != NULL ){
		ski_p = (Spink_Interface *) NODE_DATA(np);
		/*
		if( release_spink_interface(ski_p->ski_handle) < 0 )
			return -1;
			*/
		// could delete the struct here too!?!?
		del_spink_interface(ski_p);
		np = NODE_NEXT(np);
	}
	return 0;
}

void _release_spink_cam_system(SINGLE_QSP_ARG_DECL)
{
	assert( hSystem != NULL );
DEBUG_MSG(releast_spink_cam_system BEGIN)
	release_current_camera();

	if( release_spink_interface_structs() < 0 ) return;
	//if( release_spink_cam_structs() < 0 ) return;

	if( release_spink_cam_list(&hCameraList) < 0 ) return;
	if( release_spink_interface_list(&hInterfaceList) < 0 ) return;
	if( release_spink_system(hSystem) < 0 ) return;
DEBUG_MSG(releast_spink_cam_system DONE)
}

int _spink_release_cam(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	assert(skc_p->skc_current_handle!=NULL);
//fprintf(stderr,"spink_release_cam:  old handle was 0x%lx, will set to NULL\n",(u_long)skc_p->skc_current_handle);
	if( release_spink_cam(skc_p->skc_current_handle) < 0 )
		return -1;
	skc_p->skc_current_handle=NULL;
	return 0;
}

