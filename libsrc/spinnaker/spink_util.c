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
#define MAX_TREE_DEPTH	5
static Spink_Node *current_parent_p[MAX_TREE_DEPTH]={NULL,NULL,NULL,NULL};
int current_node_idx; 

static spinSystem hSystem = NULL;
static spinInterfaceList hInterfaceList = NULL;
spinCameraList hCameraList = NULL;
size_t numCameras = 0;
static size_t numInterfaces = 0;

ITEM_INTERFACE_DECLARATIONS(Spink_Interface,spink_interface,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Cam,spink_cam,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Map,spink_map,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Node,spink_node,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Node_Type,spink_node_type,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Category,spink_cat,RB_TREE_CONTAINER)

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

int _get_spink_cam_names( QSP_ARG_DECL  Data_Obj *str_dp )
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

#ifdef FOOBAR
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
#endif // FOOBAR

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

#ifdef NOT_USED

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

void set_fmt7_size(QSP_ARG_DECL  Spink_Cam *skc_p, int w, int h)
{
	UNIMP_FUNC("set_fmt7_size");
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

#ifdef FOOBAR
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
#endif // FOOBAR

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

#ifdef FOOBAR
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
#endif // FOOBAR

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
INVALID_SET_FUNC(enum_entry)

// These need to be implemented...
INVALID_SET_FUNC(value)

//get_node_max_value_float		FloatGetMax
//get_node_max_value_int		IntegerGetMax
//set_node_value_float		FloatSetValue
//set_node_value_int		IntegerSetValue
//set_node_value_string		StringSetValue

//set_node_value_string		StringSetValue

//command_is_done		CommandIsDone
//exec_spink_command		CommandExecute
static void _set_command_node(QSP_ARG_DECL  Spink_Node *skn_p)
{
	spinNodeHandle hNode;
	bool8_t done;

	assert(skn_p->skn_type_p->snt_type == CommandNode);

	if( lookup_spink_node(skn_p, &hNode) < 0 || exec_spink_command(hNode) < 0 ){
		sprintf(ERROR_STRING,"Error executing %s",skn_p->skn_name);
		warn(ERROR_STRING);
	}
	// wait here for command to finish
	do {
		if( command_is_done(hNode,&done) < 0 ) return;
	} while( ! done );
}

static void _set_string_node(QSP_ARG_DECL  Spink_Node *skn_p)
{
	spinNodeHandle hNode;
	const char *s;

	assert(skn_p->skn_type_p->snt_type == StringNode);

	s = nameof(skn_p->skn_name);

	// Does StringSetValue make a deep copy???  If not, we need to save the string before passing!?
	if( lookup_spink_node(skn_p, &hNode) < 0 || set_node_value_string(hNode,s) < 0 ){
		sprintf(ERROR_STRING,"Error setting %s",skn_p->skn_name);
		warn(ERROR_STRING);
	}
}

#define get_float_range(skn_p, min_p, max_p) _get_float_range(QSP_ARG  skn_p, min_p, max_p)

static void _get_float_range(QSP_ARG_DECL  Spink_Node *skn_p, double *min_p, double *max_p)
{
	spinNodeHandle hNode;

	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_node_min_value_float(hNode,min_p) < 0 ) return;
	if( get_node_max_value_float(hNode,max_p) < 0 ) return;
}

#define get_int_range(skn_p, min_p, max_p) _get_int_range(QSP_ARG  skn_p, min_p, max_p)

static void _get_int_range(QSP_ARG_DECL  Spink_Node *skn_p, int64_t *min_p, int64_t *max_p)
{
	spinNodeHandle hNode;

	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_node_min_value_int(hNode,min_p) < 0 ) return;
	if( get_node_max_value_int(hNode,max_p) < 0 ) return;
}

static void _set_float_node(QSP_ARG_DECL  Spink_Node *skn_p)
{
	double minv, maxv;
	double dval;
	char pmpt[LLEN];
	spinNodeHandle hNode;

	assert(skn_p->skn_type_p->snt_type == FloatNode);

	get_float_range(skn_p,&minv,&maxv);
	sprintf(pmpt,"%s (%g-%g)",skn_p->skn_name,minv,maxv);
	dval = how_much(pmpt);

	if( lookup_spink_node(skn_p, &hNode) < 0 || set_node_value_float(hNode,dval) < 0 ){
		sprintf(ERROR_STRING,"Error setting %s",skn_p->skn_name);
		warn(ERROR_STRING);
	}
}

static void _set_integer_node(QSP_ARG_DECL  Spink_Node *skn_p)
{
	int64_t minv, maxv;
	int64_t ival;
	char pmpt[LLEN];
	spinNodeHandle hNode;

	assert(skn_p->skn_type_p->snt_type == IntegerNode);

	get_int_range(skn_p,&minv,&maxv);
	sprintf(pmpt,"%s (%ld-%ld)",skn_p->skn_name,minv,maxv);
	ival = how_many(pmpt);

	if( lookup_spink_node(skn_p, &hNode) < 0 || set_node_value_int(hNode,ival) < 0 ){
		sprintf(ERROR_STRING,"Error setting %s",skn_p->skn_name);
		warn(ERROR_STRING);
	}
}

// set_node_value_bool		BooleanSetValue
static void _set_boolean_node(QSP_ARG_DECL  Spink_Node *skn_p)
{
	char pmpt[LLEN];
	spinNodeHandle hNode;
	bool8_t flag;

	assert(skn_p->skn_type_p->snt_type == BooleanNode);
	
	sprintf(pmpt,"%s",skn_p->skn_name);
	if( askif(pmpt) )
		flag = TRUE;
	else
		flag = FALSE;

	if( lookup_spink_node(skn_p, &hNode) < 0 || set_node_value_bool(hNode,flag) < 0 ){
		sprintf(ERROR_STRING,"Error setting %s",skn_p->skn_name);
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

	assert( child->skn_enum_ival != INVALID_ENUM_INT_VALUE );

	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( set_enum_int_val(hNode,child->skn_enum_ival) < 0 )
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
	if( skn_p->skn_level == 0 ){
		return;	// don't print root node
	}
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

static int _register_one_node(QSP_ARG_DECL  spinNodeHandle hNode, int level)
{
	char name[LLEN];
	size_t l=LLEN;
	Spink_Node *skn_p;
	spinNodeType type;
	int n;

	if( get_node_name(name,&l,hNode) < 0 )
		error1("register_one_node:  error getting node name!?");
	assert(strlen(name)<(LLEN-1));

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
		assert(idx>=0&&idx<MAX_TREE_DEPTH);
		skn_p->skn_parent = current_parent_p[idx];
		skn_p->skn_idx = current_node_idx;	// set in traverse...
	}
	assert(level>=0&&level<MAX_TREE_DEPTH);
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
		//assert(sct_p==NULL); 
		// This should always be NULL the first time we initialize the system,
		// but currently we aren't cleaning up when we shut it down and reinitialize...
		if( sct_p == NULL ){
			sct_p = new_spink_cat(skn_p->skn_name);
		}
		assert(sct_p!=NULL); 
		sct_p->sct_root_p = skn_p;
	}

	skn_p->skn_enum_ival = INVALID_ENUM_INT_VALUE;
	if( type == EnumEntryNode ){
		int64_t ival;
		if( get_enum_int_value(hNode,&ival) < 0 ){
			skn_p->skn_enum_ival = INVALID_ENUM_INT_VALUE;
		} else {
			skn_p->skn_enum_ival = ival;
			assert(ival>=0);
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

static Spink_Map * _register_one_map(QSP_ARG_DECL  Spink_Cam *skc_p, Node_Map_Type type, const char *name)
{
	Spink_Map *skm_p;
	spinNodeMapHandle hMap = NULL;

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
	get_node_map_handle(&hMap,skm_p,"register_one_map");	// first time just sets
//fprintf(stderr,"register_one_map:  hMap = 0x%lx, *hMap = 0x%lx \n",(u_long)hMap, (u_long)*((void **)hMap));

	register_map_nodes(hMap,skm_p);

	return skm_p;
}

#define register_cam_nodemaps(skc_p) _register_cam_nodemaps(QSP_ARG  skc_p)

static void _register_cam_nodemaps(QSP_ARG_DECL  Spink_Cam *skc_p)
{
//fprintf(stderr,"register_cam_nodemaps BEGIN\n");
//fprintf(stderr,"register_cam_nodemaps registering device map\n");
	sprintf(MSG_STR,"%s.stream_TL",skc_p->skc_name);
	skc_p->skc_stream_map = register_one_map(skc_p,STREAM_NODE_MAP,MSG_STR);

	sprintf(MSG_STR,"%s.device_TL",skc_p->skc_name);
	skc_p->skc_dev_map = register_one_map(skc_p,DEV_NODE_MAP,MSG_STR);
//fprintf(stderr,"register_cam_nodemaps registering camera map\n");

	sprintf(MSG_STR,"%s.genicam",skc_p->skc_name);
	skc_p->skc_cam_map = register_one_map(skc_p,CAM_NODE_MAP,MSG_STR);

	// Now get width and height from the map...

//fprintf(stderr,"register_cam_nodemaps DONE\n");
}

#define int_node_value(s) _int_node_value(QSP_ARG  s)

static int64_t _int_node_value(QSP_ARG_DECL  const char *s)
{
	int64_t integerValue = 0;
	spinNodeHandle hNode;
	Spink_Node *skn_p;
	Spink_Node_Type *snt_p;

	skn_p = get_spink_node(s);
	if( skn_p == NULL ){
		sprintf(ERROR_STRING,"int_node_value:  Node '%s' not found!?",s);
		warn(ERROR_STRING);
		return 0;
	}
	snt_p = skn_p->skn_type_p;
	assert(snt_p!=NULL);
	if(snt_p->snt_type != IntegerNode){
		sprintf(ERROR_STRING,"int_node_value:  Node '%s' is not an integer node!?",s);
		warn(ERROR_STRING);
		return 0;
	}
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return 0;
	if( get_int_value(hNode, &integerValue) < 0 ) return 0;

	return integerValue;
}


#define get_cam_dimensions(skc_p) _get_cam_dimensions(QSP_ARG  skc_p)

static void _get_cam_dimensions(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	select_spink_map(skc_p->skc_cam_map);
	skc_p->skc_cols = int_node_value("Width");
	skc_p->skc_rows = int_node_value("Height");
	select_spink_map(NULL);
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

	skc_p->skc_flags = 0;
	//skc_p->skc_handle = hCam;
	skc_p->skc_sys_idx = idx;
	skc_p->skc_iface_idx = -1;	// invalid value
	skc_p->skc_n_buffers = 0;

	// register_cam_nodemaps will get the camera handle again...
//	if( release_spink_cam(hCam) < 0 )
//		return -1;

	//skc_p->skc_TL_dev_node_map = hNodeMapTLDevice;
	//skc_p->skc_genicam_node_map = hNodeMap;

	skc_p->skc_dev_map=NULL;
	skc_p->skc_cam_map=NULL;
	skc_p->skc_stream_map=NULL;

	// The camera has to be connected to get the genicam node map!
	register_cam_nodemaps(skc_p);

	get_cam_dimensions(skc_p);

#ifdef FOOBAR
	// Make a data_obj context for the frames...
	skc_p->skc_do_icp = create_dobj_context( QSP_ARG  skc_p->skc_name );
	assert(skc_p->skc_do_icp != NULL);
#endif // FOOBAR

	// We have to explicitly release here, as we weren't able to call
	// insure_current_camera at the beginning...
	//spink_release_cam(skc_p);

	release_current_camera(1);

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

#define stop_all_cameras() _stop_all_cameras(SINGLE_QSP_ARG)

static void _stop_all_cameras(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;
	Spink_Cam *skc_p;

	lp = spink_cam_list();
	if( lp == NULL ) return;
	np = QLIST_HEAD(lp);
	if( np == NULL ) return;

	while(np!=NULL){
		skc_p = NODE_DATA(np);
		if( IS_CAPTURING(skc_p) ) spink_stop_capture(skc_p);
		np = NODE_NEXT(np);
	}
}

void _release_spink_cam_system(SINGLE_QSP_ARG_DECL)
{
	if( hSystem == NULL ) return;	// may already be shut down?

DEBUG_MSG(releast_spink_cam_system BEGIN)

	// make sure that no cameras are running...
fprintf(stderr,"release_spink_cam_system stopping all cameras\n");
	stop_all_cameras();

fprintf(stderr,"release_spink_cam_system releasing current camera\n");
	release_current_camera(0);

fprintf(stderr,"release_spink_cam_system releasing interface structs\n");
	if( release_spink_interface_structs() < 0 ) return;
	//if( release_spink_cam_structs() < 0 ) return;

fprintf(stderr,"release_spink_cam_system releasing camera list\n");
	if( release_spink_cam_list(&hCameraList) < 0 ) return;
fprintf(stderr,"release_spink_cam_system releasing interface list\n");
	if( release_spink_interface_list(&hInterfaceList) < 0 ) return;
fprintf(stderr,"release_spink_cam_system releasing system\n");
	if( release_spink_system(hSystem) < 0 ) return;
fprintf(stderr,"release_spink_cam_system DONE\n");
	hSystem = NULL;
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

