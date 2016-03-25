#include "quip_config.h"
//#include "CameraViewController.h"
#include "quipAppDelegate.h"
#include "quip_prot.h"
#include "ios_prot.h"

#include "camera.h"
#include "camera_api.h"

static IOS_Item_Type *camera_itp;
IOS_ITEM_INIT_FUNC(Camera,camera)
IOS_ITEM_NEW_FUNC(Camera,camera)
IOS_ITEM_CHECK_FUNC(Camera,camera)
IOS_ITEM_PICK_FUNC(Camera,camera)
IOS_ITEM_LIST_FUNC(Camera,camera)

@implementation Camera

@synthesize dev;

+(void) initClass
{
	camera_itp =[[IOS_Item_Type alloc] initWithName: STRINGOBJ("Camera") ];
}

@end

static void init_camera_subsystem(SINGLE_QSP_ARG_DECL)
{
	Camera *cam;
	NSArray *cam_list;
	int i;

	cam_list = [AVCaptureDevice devices];

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
	}

	fprintf(stderr,"%d cameras found:\n",cam_list.count);
	list_cameras(SINGLE_QSP_ARG);
}

static void print_cam_info(QSP_ARG_DECL  Camera *cam)
{
	// First list the input sources
	NSArray *inputs;
	inputs = cam.inputSources;
	if( inputs.count > 1 ){
		int i;
		prt_msg("Input sources:");
		for(i=0;i<inputs.count;i++){
			AVCaptureDeviceInputSource *cdis;
			cdis = [inputs objectAtIndex:i];
			prt_msg_frag("\t");
			prt_msg(cdis.localizedName.UTF8String);
		}
	}

	// could list possible focus modes?

	if( [cam isTorchModeSupported:AVCaptureTorchModeOn] )
		prt_msg("Torch is available");
	else
		prt_msg("No torch");

	if( [cam isWhiteBalanceModeSupported:AVCaptureWhiteBalanceModeAutoWhiteBalance ] )
		prt_msg("Auto white balance is available");
	else
		prt_msg("No auto white balance");

}

#ifdef FOOBAR
static int camera_init(void)
{
    CameraViewController *my_view_ctl=
    [[CameraViewController alloc]init];
    
    [my_view_ctl startCameraControllerFromViewController: (UIViewController *)globalAppDelegate.qvc
                                           usingDelegate: globalAppDelegate];
    
    /*if( [UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypeCamera]
       == NO ){
        NWARN("Camera is not available!?");
        return -1;
    }
    
    UIImagePickerController *my_cam_ctl=[[UIImagePickerController alloc] init];
    
    my_cam_ctl.sourceType = UIImagePickerControllerSourceTypeCamera;
    
    my_cam_ctl.mediaTypes = [UIImagePickerController
            availableMediaTypesForSourceType:UIImagePickerControllerSourceTypeCamera];

    my_cam_ctl.allowsEditing = NO;
    
    my_cam_ctl.delegate = my_view_ctl;
    */
    
    
    /*
	//UIImagePicker *uipp;

	AVCaptureSession *session = [[AVCaptureSession alloc] init];

	// Add inputs and outputs.

	AVCaptureStillImageOutput *stillImageOutput = [[AVCaptureStillImageOutput alloc] init];

	NSDictionary *outputSettings = @{ AVVideoCodecKey : AVVideoCodecJPEG};

	[stillImageOutput setOutputSettings:outputSettings];

	//
	AVCaptureConnection *videoConnection = nil;

	for (AVCaptureConnection *connection in stillImageOutput.connections) {
		for (AVCaptureInputPort *port in [connection inputPorts]) {
 		 	if ([[port mediaType] isEqual:AVMediaTypeVideo] ) {
				videoConnection = connection;
				break;
			}
		}
		if (videoConnection) { break; }
	}

	//[session startRunning];

     */

    return 0;
}

static COMMAND_FUNC( do_start )
{
	static int cam_inited=0;

	if( ! cam_inited ){
		if( camera_init() < 0 ) return;
		cam_inited=1;
	}
}
ADD_CMD( start,	do_start,	start acquisition from camera)
#endif /* FOOBAR */

static COMMAND_FUNC( do_list_cams )
{
	list_cameras(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_cam_info )
{
	Camera *cam;

	cam = pick_camera(QSP_ARG  "");
	if( cam == NULL ) return;

	print_cam_info(QSP_ARG  cam);
}


#define ADD_CMD(s,f,h)	ADD_COMMAND(camera_menu,s,f,h)
MENU_BEGIN(camera)
ADD_CMD( list,	do_list_cams,	list available cameras			)
ADD_CMD( info,	do_cam_info,	print information about a camera	)
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

