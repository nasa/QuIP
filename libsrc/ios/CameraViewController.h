#include "quip_config.h"
//#include <UIKit/UIKit.h>
#include <AVFoundation/AVCaptureSession.h>
#include <AVFoundation/AVCaptureOutput.h>
#include <AVFoundation/AVCaptureInput.h>

#include <MobileCoreServices/UTCoreTypes.h>

#include "quipAppDelegate.h"
#include "quip_prot.h"
#include "ios_prot.h"

@interface CameraViewController : UIViewController
- (BOOL) startCameraControllerFromViewController: (UIViewController*) controller
	usingDelegate: (id <UIImagePickerControllerDelegate, UINavigationControllerDelegate>) delegate;
@end

