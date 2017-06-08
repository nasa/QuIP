#include "quip_config.h"
//#include "CameraViewController.h"
#include "quipAppDelegate.h"
#include "quip_prot.h"
#include "ios_prot.h"

#include "camera.h"
#include "camera_api.h"
#include "avCap.h"
#include "viewer.h"

static IOS_Item_Type *camera_itp;
IOS_ITEM_INIT_FUNC(Camera,camera,0)
IOS_ITEM_NEW_FUNC(Camera,camera)
IOS_ITEM_CHECK_FUNC(Camera,camera)
IOS_ITEM_PICK_FUNC(Camera,camera)
IOS_ITEM_LIST_FUNC(Camera,camera)

static double get_camera_size(QSP_ARG_DECL  IOS_Item *ip,int index)
{
	double d;
	Camera *cam_p;

	cam_p = (Camera *)ip;

	AVCaptureDeviceFormat *cdf;
	cdf = cam_p.dev.activeFormat;
	CMFormatDescriptionRef fdr;
	fdr = cdf.formatDescription;
	CMVideoDimensions vdims;
	vdims = CMVideoFormatDescriptionGetDimensions( fdr );
	switch(index){
		case 1:	d = vdims.width; break;		// columns
		case 2:	d = vdims.height; break;	// rows
		default: d=1.0; break;
	}
	return(d);
}

static const char * get_camera_prec(QSP_ARG_DECL  IOS_Item *ip )
{
	return "unimp";
}

IOS_Size_Functions camera_sf={
		get_camera_size,
		get_camera_prec
};

@implementation Camera

@synthesize dev;

+(void) initClass
{
	camera_itp =[[IOS_Item_Type alloc] initWithName: STRINGOBJ("Camera") ];
}

@end

static void init_formats(Camera *cam)
{
	int i,n;

	NSArray *fmt_list=cam.dev.formats;

	n= (int) fmt_list.count;
	sprintf(MSG_STR,"\t%d formats supported",n);
	prt_msg(MSG_STR);

	// Now scan the list:
	for(i=0;i<n;i++){
		AVCaptureDeviceFormat *cdf;
		CMVideoDimensions vdims;
		cdf = [fmt_list objectAtIndex:i];
#ifdef CAUTIOUS
		if( cdf == NULL ){
			NERROR1("CAUTIOUS:  init_formats:  null format!?");
			return;
		}
#endif // CAUTIOUS

		if( [ cdf.mediaType compare:AVMediaTypeVideo ] == NSOrderedSame ){ 
			int w, h;

// Only available in iOS 8 and later!?
#ifdef IOS8_ONLY
			vdims = cdf.highResolutionStillImageDimensions;
			w = vdims.width;
			h = vdims.height;
//fprintf(stderr,"Highest still resolution:  %d x %d\n",w,h);
#endif // IOS8_ONLY

			// We'd like to have a name for this format?
			CMFormatDescriptionRef fdr;
			fdr = cdf.formatDescription;
			vdims = CMVideoFormatDescriptionGetDimensions( fdr );
			w = vdims.width;
			h = vdims.height;
		} else {
			fprintf(stderr,"init_formats:  Unhandled media type!?\n");
		}
	}
}

static void init_camera_subsystem(SINGLE_QSP_ARG_DECL)
{
	Camera *cam;
	NSArray *cam_list;
	int i;

	//cam_list = [AVCaptureDevice devices];
	cam_list = [AVCaptureDevice devicesWithMediaType: AVMediaTypeVideo ];

	if( cam_list == NULL ){
		WARN("init_camera_subsystem:  no cameras found!?");
		return;
	}

	for(i=0;i<cam_list.count;i++){
		AVCaptureDevice *avcd;

		avcd = [cam_list objectAtIndex:i];
#ifdef CAUTIOUS
		if( avcd == NULL ) {
			WARN("CAUTIOUS:  init_camera_subsystem:  Null list element!?");
			return;
		}
#endif /* CAUTIOUS */

		cam = new_camera(QSP_ARG  avcd.localizedName.UTF8String );
		if( cam == NULL ){
			sprintf(ERROR_STRING,"Error creating camera %s!?",
				avcd.localizedName.UTF8String);
			WARN(ERROR_STRING);
			return;
		}

		cam.dev = avcd;
		// Now set up the formats too!
		init_formats(cam);
	}

	//fprintf(stderr,"%d cameras found:\n",cam_list.count);
	//list_cameras(SINGLE_QSP_ARG);

	// BUG should be reserved var?
	set_script_var_from_int(QSP_ARG  "n_cameras",
				/*camera_list_p->num*/ cam_list.count);

	add_ios_sizable(QSP_ARG  camera_itp,&camera_sf, NULL );
}

static void print_cam_info(QSP_ARG_DECL  Camera *cam)
{
	sprintf(MSG_STR,"%s:",cam.name.UTF8String);
	prt_msg(MSG_STR);

	// could list possible focus modes?

//	if( [cam.dev isTorchModeSupported:AVCaptureTorchModeOn] )
	if( [cam.dev hasFlash] )
		prt_msg("\tFlash is available");
	else
		prt_msg("\tNo flash");

	if( [cam.dev hasTorch] )
		prt_msg("\tTorch is available");
	else
		prt_msg("\tNo torch");

	if( [cam.dev isWhiteBalanceModeSupported:AVCaptureWhiteBalanceModeAutoWhiteBalance ] )
		prt_msg("\tAuto white balance is available");
	else
		prt_msg("\tNo auto white balance");

	NSArray *fmt_list=cam.dev.formats;
	sprintf(MSG_STR,"\t%d formats supported",(int)fmt_list.count);
	prt_msg(MSG_STR);
}

static COMMAND_FUNC( do_list_cams )
{
	prt_msg("A/V Capture Devices:");
	list_cameras(QSP_ARG  tell_msgfile(SINGLE_QSP_ARG));
}

static COMMAND_FUNC( do_cam_info )
{
	Camera *cam;

	cam = pick_camera(QSP_ARG  "");
	if( cam == NULL ) return;

	print_cam_info(QSP_ARG  cam);
}

static int get_ios_item_names( QSP_ARG_DECL  Data_Obj *str_dp, IOS_Item_Type *itp )
{
	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?
	IOS_List *lp;
	IOS_Node *np;
	IOS_Item *ip;
	int i, n;

	lp = ios_item_list(QSP_ARG  itp);
	if( lp == NULL ){
		WARN("get_item_names:  No item list!?");
		return 0;
	}

	n=ios_eltcount(lp);
	if( OBJ_COLS(str_dp) < n ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d %s names",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),n,IOS_ITEM_TYPE_NAME(itp));
		WARN(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	}
		
	np=IOS_LIST_HEAD(lp);
	i=0;
	while(np!=NO_IOS_NODE){
		char *dst;
		ip = (IOS_Item *) IOS_NODE_DATA(np);
		dst = OBJ_DATA_PTR(str_dp);
		dst += i * OBJ_PXL_INC(str_dp);
		if( strlen(IOS_ITEM_NAME(ip))+1 > OBJ_COMPS(str_dp) ){
			sprintf(ERROR_STRING,"String object %s has too few components (%ld) to hold item name \"%s\"",
				OBJ_NAME(str_dp),(long)OBJ_COMPS(str_dp),IOS_ITEM_NAME(ip));
			WARN(ERROR_STRING);
		} else {
			strcpy(dst,IOS_ITEM_NAME(ip));
		}
		i++;
		if( i>=n )
			np=NO_IOS_NODE;
		else
			np = IOS_NODE_NEXT(np);
	}

	return i;
}

static int get_camera_names( QSP_ARG_DECL  Data_Obj *str_dp )
{

	return get_ios_item_names(QSP_ARG  str_dp, camera_itp);
}

static COMMAND_FUNC( do_get_cams )
{
	Data_Obj *dp;

	dp = PICK_OBJ("string table");
	if( dp == NO_OBJ ) return;

	if( get_camera_names( QSP_ARG  dp ) < 0 )
		WARN("Error getting camera names!?");
}

static COMMAND_FUNC( do_mon_cam )
{
	Viewer *vp;

	vp = PICK_VWR("");
	if( vp == NO_VIEWER ) return;

	monitor_av_session(vp);
}

static COMMAND_FUNC( do_stop_mon )
{
	monitor_av_session(NULL);
}


static COMMAND_FUNC( do_chk_session )
{
	check_av_session();
}

static COMMAND_FUNC( do_stop_session )
{ stop_av_capture(); }

static COMMAND_FUNC( do_pause_session )
{ pause_av_capture(); }

static COMMAND_FUNC( do_restart_session )
{ restart_av_capture(); }

static COMMAND_FUNC( do_start_session )
{
	Camera *cam;

	cam = pick_camera(QSP_ARG  "");
	if( cam == NULL ) return;

	start_av_capture( cam.dev );
}

static COMMAND_FUNC( do_grab_cam )
{
	Data_Obj *dp;

	dp = PICK_OBJ("target image object");
	if( dp == NO_OBJ ) return;

	grab_next_frame(dp);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(camera_menu,s,f,h)
MENU_BEGIN(camera)
ADD_CMD( list,		do_list_cams,		list available cameras )
ADD_CMD( info,		do_cam_info,		print info about a camera )
ADD_CMD( monitor,	do_mon_cam,		monitor capture )
ADD_CMD( stop_mon,	do_stop_mon,		stop monitoring capture )
ADD_CMD( grab,		do_grab_cam,		grab a frame )
ADD_CMD( get_cameras,	do_get_cams,		copy camera names to an array )
ADD_CMD( check,		do_chk_session,		check session state )
ADD_CMD( start,		do_start_session,	start capture )
ADD_CMD( pause,		do_pause_session,	pause capture )
ADD_CMD( restart,	do_restart_session,	restart capture )
ADD_CMD( stop,		do_stop_session,	stop capture )
MENU_END(camera)

COMMAND_FUNC(do_cam_menu)
{
	static int inited=0;

	if( ! inited ){
		init_camera_subsystem(SINGLE_QSP_ARG);
		inited=1;
	}

	PUSH_MENU(camera);
}

