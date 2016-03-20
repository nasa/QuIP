#include "quip_config.h"
#include "CameraViewController.h"

@implementation CameraViewController /* (CameraDelegateMethods) */

// This makes the status bar go away...

-(BOOL) prefersStatusBarHidden {
	return YES;
}



// For responding to the user tapping Cancel.

- (void) imagePickerControllerDidCancel: (UIImagePickerController *) picker
{
	[[picker parentViewController] dismissModalViewControllerAnimated: YES];
	//[picker release];
	
}



// For responding to the user accepting a newly-captured picture or movie

- (void) imagePickerController: (UIImagePickerController *) picker
 didFinishPickingMediaWithInfo: (NSDictionary *) info
{
	NSString *mediaType = [info objectForKey: UIImagePickerControllerMediaType];
	UIImage *originalImage, *editedImage, *imageToSave;
	
	// Handle a still image capture
	
	if (CFStringCompare ((__bridge CFStringRef) mediaType, kUTTypeImage, 0)
		== kCFCompareEqualTo) {
		editedImage = (UIImage *) [info objectForKey:
								UIImagePickerControllerEditedImage];
		originalImage = (UIImage *) [info objectForKey:
									UIImagePickerControllerOriginalImage];
		if (editedImage) {
			imageToSave = editedImage;
		} else {
			imageToSave = originalImage;
		}
		
		// Save the new image (original or edited) to the Camera Roll
		
		UIImageWriteToSavedPhotosAlbum (imageToSave, nil, nil , nil);
		
	}
	
	
	
	// Handle a movie capture
	
	if (CFStringCompare ((__bridge CFStringRef) mediaType, kUTTypeMovie, 0)
		== kCFCompareEqualTo) {
		
		NSString *moviePath = [[info objectForKey: UIImagePickerControllerMediaURL] path];
		
		if (UIVideoAtPathIsCompatibleWithSavedPhotosAlbum (moviePath)) {
			UISaveVideoAtPathToSavedPhotosAlbum ( moviePath, nil, nil, nil);
		}
	}
	
	[[picker parentViewController] dismissModalViewControllerAnimated: YES];
	//[picker release];
}

- (BOOL) startCameraControllerFromViewController: (UIViewController*) controller
	usingDelegate: (id <UIImagePickerControllerDelegate, UINavigationControllerDelegate>) delegate
{
	if (([UIImagePickerController isSourceTypeAvailable:
		UIImagePickerControllerSourceTypeCamera] == NO)
		|| (delegate == nil)
		|| (controller == nil))
		
		return NO;
	
	UIImagePickerController *cameraUI = [[UIImagePickerController alloc] init];
	cameraUI.sourceType = UIImagePickerControllerSourceTypeCamera;
	
	// Displays a control that allows the user to choose picture or
	// movie capture, if both are available:
	cameraUI.mediaTypes =
	[UIImagePickerController availableMediaTypesForSourceType:
	UIImagePickerControllerSourceTypeCamera];
	
	// Hides the controls for moving & scaling pictures, or for
	// trimming movies. To instead show the controls, use YES.
	cameraUI.allowsEditing = NO;
	
	cameraUI.delegate = delegate;
	
	[controller presentModalViewController: cameraUI animated: YES];
	return YES;
	
}



@end
