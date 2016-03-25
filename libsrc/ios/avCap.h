#include <CoreMedia/CMSampleBuffer.h>

#include <AVFoundation/AVCaptureSession.h>
#include <AVFoundation/AVCaptureOutput.h>

#include "viewer.h"

@interface avCap:NSObject  <AVCaptureVideoDataOutputSampleBufferDelegate> 

@property (retain) AVCaptureSession	*_session;

@end

extern void start_av_capture(AVCaptureDevice *dev);
extern void stop_av_capture(void);
extern void restart_av_capture(void);
extern void pause_av_capture(void);
extern void check_av_session(void);
extern void monitor_av_session(Viewer *vp);
extern void grab_next_frame(Data_Obj *dp);

extern CGImageRef createImageFromBuffer( CVImageBufferRef buffer,
	size_t left, size_t top, size_t width, size_t height );

extern int copyImageFromBuffer( Data_Obj *dp, CVImageBufferRef buffer,
	size_t left, size_t top, size_t width, size_t height );

