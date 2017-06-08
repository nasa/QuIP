#include "quip_config.h"

#include <AVFoundation/AVCaptureDevice.h>
#include <AVFoundation/AVCaptureInput.h>
#include "avCap.h"

#include "quip_prot.h"
#include "viewer.h"
#include "quipImageView.h"

// the_cap is a static global that holds the value of the current a/v session
// used for camera capture
static avCap *the_cap=NULL;

// mon_vp is a static globl that represents the viewer
// we are displaying live camera data in...
static Viewer *mon_vp=NULL;

// grab_dp is a static global
// When the user wants to grab a frame, they pass a pointer to
// their image buffer, which is stored in grab_dp
static Data_Obj *grab_dp=NULL;

@implementation avCap

@synthesize _session;

- (avCap *) init
{
	//[super init];
	return self;
}

// this is a callback delegate?

//#define MAX_IMAGES_FOR_NOW	4

- (void)captureOutput:(AVCaptureOutput *)captureOutput
	didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
	fromConnection:(AVCaptureConnection *)connection
{
	CVImageBufferRef pixelBuffer = NULL;
fprintf(stderr,"captureOutput BEGIN\n");
	if( mon_vp != NULL ){
		pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
		CGImageRef newImage;
		// This crops instead of scaling...
		newImage = createImageFromBuffer(pixelBuffer,0,0,
			VW_HEIGHT(mon_vp),VW_WIDTH(mon_vp));
			// BUG?  args are declared width,height???
#ifdef BUILD_FOR_IOS
		UIImage *img = [UIImage imageWithCGImage:newImage
			scale:1.0 orientation:UIImageOrientationRight ];
		CGImageRelease(newImage);
		
		(VW_QV(mon_vp)).bgImageView.image = img;
#else
		NWARN("captureOutput:  Need to implement for MAC OS!?");
		return;
#endif // BUILD_FOR_IOS
			}
	if( grab_dp != NULL ){
		if( pixelBuffer == NULL )
			pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
		// How do we dereference a CVImageBufferRef???

		// Make sure the dimensions of the rectangle match that
		// of the data object.
		/*
		CGRect r = CVImageBufferGetCleanRect(pixelBuffer);
		if( r == NULL ){
			NWARN("error getting clean rect!?");
			grab_dp=NULL;
			return;
		}
		*/
		CGSize s = CVImageBufferGetDisplaySize(pixelBuffer);
		if( s.width == 0 ){
			NWARN("error getting display size!?");
			grab_dp=NULL;
			return;
		}

		if( s.width != OBJ_COLS(grab_dp) || s.height != OBJ_ROWS(grab_dp) ){
			sprintf(DEFAULT_ERROR_STRING,
		"grab buffer size (%dx%d) does not match image %s (%dx%d)",
				(int)s.width,(int)s.height,OBJ_NAME(grab_dp),
				OBJ_COLS(grab_dp),OBJ_ROWS(grab_dp));
			NWARN(DEFAULT_ERROR_STRING);
			grab_dp=NULL;
			return;
		}

#ifdef FOOBAR
		int size = s.width * s.height * 4;	// assume 4 bpp
		CVPixelBufferLockBaseAddress(pixelBuffer,0);
		int8_t *baseAddress = (int8_t *)
			CVPixelBufferGetBaseAddress(pixelBuffer);
		memcpy(OBJ_DATA_PTR(grab_dp), baseAddress, size);
		CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
#endif // FOOBAR
		copyImageFromBuffer(grab_dp,pixelBuffer,0,0,(size_t)s.width,(size_t)s.height);

		grab_dp=NULL;
	}
} // end captureOutput

- (void)setupAVCaptureFromDevice: (AVCaptureDevice *) videoDevice
{
	//-- Setup Capture Session.
	_session = [[AVCaptureSession alloc] init];
	[_session beginConfiguration];

	//-- Set preset session size.
	// what is the default, full res?
	// presets are high, medium, low, 640x480 (VGA), 1280x720 (720p HD),
	// and photo

	//[_session setSessionPreset:_sessionPreset];
	[_session setSessionPreset:AVCaptureSessionPresetHigh];

	if(videoDevice == nil) assert(0);

	//-- Add the device to the session.
	NSError *error;
	AVCaptureDeviceInput *input =
		[AVCaptureDeviceInput deviceInputWithDevice:videoDevice
			error:&error];
	if(error)
		assert(0);

	[_session addInput:input];

	//-- Create the output for the capture session.
	// can also be StillImageOutput, AudioDataOutput, etc
	AVCaptureVideoDataOutput * dataOutput =
		[[AVCaptureVideoDataOutput alloc] init];

	// Probably want to set this to NO when recording
	[dataOutput setAlwaysDiscardsLateVideoFrames:YES];

	/*
	//-- Set to YUV420.
	[dataOutput setVideoSettings:[NSDictionary
		dictionaryWithObject:[NSNumber
			numberWithInt:
				kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
		forKey:(id)kCVPixelBufferPixelFormatTypeKey]
		// Necessary for manual preview
	];
	*/
	//-- Set to RGB.
	[dataOutput setVideoSettings:[NSDictionary
		dictionaryWithObject:[NSNumber
			numberWithInt:
				kCVPixelFormatType_32BGRA]
		forKey:(id)kCVPixelBufferPixelFormatTypeKey]
	];
	/*
	dataOutput.videoSettings =
		@{ (NSString *)kCVPixelBufferPixelFormatTypeKey :
		@(kCVPixelFormatType_32BGRA) };
	*/

	// Set dispatch to be on the main thread
	// so OpenGL can do things with the data

#define MAIN_QUEUE

#ifdef MAIN_QUEUE

	[dataOutput setSampleBufferDelegate:self
			queue: dispatch_get_main_queue() ];

#else // ! MAIN_QUEUE

	dispatch_queue_t my_q = dispatch_queue_create("MyQueue",DISPATCH_QUEUE_SERIAL);

	[dataOutput setSampleBufferDelegate:self
			queue: my_q ];
	dispatch_release(my_q);

#endif // ! MAIN_QUEUE

	[_session addOutput:dataOutput];

	[_session commitConfiguration];
fprintf(stderr,"setupAVCapture:  configuration committed...\n");

	[_session startRunning];
fprintf(stderr,"setupAVCapture:  session started...\n");
}

@end

void start_av_capture( AVCaptureDevice *dev )
{
	if( the_cap != NULL ){
		advise("capture already started...");
		return;
	}

	the_cap = [[avCap alloc] init];
	[the_cap setupAVCaptureFromDevice:dev];
}

void check_av_session(void)
{
	if( the_cap == NULL ){
		advise("No capture session!?");
		return;
	}

	if( the_cap._session.running )
		advise("Session is running.");
	else
		advise("Session is NOT running.");
}


void pause_av_capture( void )
{
	if( the_cap == NULL ){
		advise("No capture session!?");
		return;
	}

	if( the_cap._session.running )
		[the_cap._session stopRunning];
	else
		advise("pause:  session is already stopped!?");
}

void stop_av_capture( void )
{
	if( the_cap == NULL ){
		advise("No capture session!?");
		return;
	}

	pause_av_capture();

	the_cap = NULL;		// does removing the reference destroy it?
}

void restart_av_capture( void )
{
	if( the_cap == NULL ){
		advise("No capture session!?");
		return;
	}

	if( the_cap._session.running )
		advise("restart:  session is already started!?");
	else
		[the_cap._session startRunning];
}

void monitor_av_session(Viewer *vp)
{
	if( the_cap == NULL ){
		advise("monitor_av_session:  no capture session!?");
		return;
	}
	if( vp == NULL ){	// null arg is code to stop monitoring
		mon_vp=vp;
		return;
	}

	// push the viewer...
	show_viewer(vp);

	// How do we tell the delegate to display?
	//
	// If mon_vp is non-null, then the capture callback
	// will send the image to the viewer.
	mon_vp = vp;
}

void grab_next_frame(Data_Obj *dp)
{
	if( the_cap == NULL ){
		advise("grab_next_frame:  no capture session!?");
		return;
	}

	// This would be a place to do some checks?
fprintf(stderr,"grab_next_frame:  will store to %s\n",OBJ_NAME(dp));
	grab_dp = dp;
}


// What if we want to do some scaling???

CGImageRef createImageFromBuffer( CVImageBufferRef buffer,
	size_t left, size_t top, size_t width, size_t height )
{
	int bytesPerRow = (int)CVPixelBufferGetBytesPerRow(buffer);
	int dataWidth = (int)CVPixelBufferGetWidth(buffer);
	int dataHeight = (int)CVPixelBufferGetHeight(buffer);
	if (left + width > dataWidth ){
fprintf(stderr,"left = %zu   width = %zu   dataWidth = %d\n",
left,width,dataWidth);
fprintf(stderr,"createImageFromBuffer:  Requested width (%zu+%zu) is larger than data (%d), shrinking to fit.\n",left,width,dataWidth);
		// try to honor offset first
		if( left < dataWidth ){
			width = dataWidth - left;
		} else {
fprintf(stderr,"Left offset out of range, setting to 0\n");
			left = 0;
			if( width > dataWidth )
				width = dataWidth;
		}
	}
	if( top + height > dataHeight) {
fprintf(stderr,"top = %zu   height = %zu   dataHeight = %d\n",
top,height,dataHeight);
fprintf(stderr,"createImageFromBuffer:  Requested height (%zu+%zu) is larger than data (%d), shrinking to fit.\n",top,height,dataHeight);
		// try to honor offset first
		if( top < dataHeight ){
			height = dataHeight - top;
		} else {
fprintf(stderr,"createImageFromBuffer:  Top offset out of range, setting to 0\n");
			top = 0;
			if( height > dataHeight )
				height = dataHeight;
		}
	}
	/*
		[NSException raise:NSInvalidArgumentException
format:@"createImageFromBuffer:  crop rectangle does not fit within image data"
		];
		*/

	// make row bytes a multiple of 16...
	unsigned long newBytesPerRow = ((width*4+0xf)>>4)<<4;
	CVPixelBufferLockBaseAddress(buffer,0);
	int8_t *baseAddress = (int8_t *)CVPixelBufferGetBaseAddress(buffer);
	unsigned long size = newBytesPerRow*height;
	int8_t *bytes = (int8_t*)malloc(size);
	if (newBytesPerRow == bytesPerRow) {
		memcpy(bytes, baseAddress+top*bytesPerRow, size);
	} else {
		for(int y=0; y<height; y++) {
			memcpy(bytes+y*newBytesPerRow,
				baseAddress+left*4+(top+y)*bytesPerRow,
				newBytesPerRow);
		}
	}
	CVPixelBufferUnlockBaseAddress(buffer, 0);
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef newContext = CGBitmapContextCreate(bytes, width, height,
		8,
		newBytesPerRow,
		colorSpace,
		kCGBitmapByteOrder32Little| kCGImageAlphaNoneSkipFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef result = CGBitmapContextCreateImage(newContext);
	CGContextRelease(newContext);
	free(bytes);
	return result;
} // createImageFromBuffer

int copyImageFromBuffer( Data_Obj *dp, CVImageBufferRef buffer,
	size_t left, size_t top, size_t width, size_t height )
{
	int bytesPerRow = (int)CVPixelBufferGetBytesPerRow(buffer);
	int dataWidth = (int)CVPixelBufferGetWidth(buffer);
	int dataHeight = (int)CVPixelBufferGetHeight(buffer);
	if (left + width > dataWidth || top + height > dataHeight) {
		[NSException raise:NSInvalidArgumentException
format:@"copyImageFromBuffer:  crop rectangle does not fit within image data"
		];
	}
	// make row bytes a multiple of 16...
	unsigned long newBytesPerRow = ((width*4+0xf)>>4)<<4;
	unsigned long size = newBytesPerRow*height;

	//int8_t *bytes = (int8_t*)malloc(size);
	if( OBJ_N_MACH_ELTS(dp) != size ){
		sprintf(DEFAULT_ERROR_STRING,
"copyImageFromBuffer:  size of object %s byte count (%d) does not match request (%lu)",
			OBJ_NAME(dp),
			OBJ_N_MACH_ELTS(dp),size);
		NWARN(DEFAULT_ERROR_STRING);
		return -1;
	}
	if( !IS_CONTIGUOUS(dp) ){
		sprintf(DEFAULT_ERROR_STRING,
"copyImageFromBuffer:  object %s must be contiguous!?",
			OBJ_NAME(dp) );
		NWARN(DEFAULT_ERROR_STRING);
		return -1;
	}
		
	CVPixelBufferLockBaseAddress(buffer,0);
	int8_t *baseAddress = (int8_t *)CVPixelBufferGetBaseAddress(buffer);

	if (newBytesPerRow == bytesPerRow) {
		memcpy(OBJ_DATA_PTR(dp), baseAddress+top*bytesPerRow, size);
	} else {
		for(int y=0; y<height; y++) {
			memcpy(((char *)OBJ_DATA_PTR(dp))+y*newBytesPerRow,
				baseAddress+left*4+(top+y)*bytesPerRow,
				newBytesPerRow);
		}
	}
	CVPixelBufferUnlockBaseAddress(buffer, 0);

	return 0;
} // copyImageFromBuffer

