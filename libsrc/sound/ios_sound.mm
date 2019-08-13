
#include "quip_config.h"
#include "function.h"

#ifdef BUILD_FOR_IOS

#import <AVFoundation/AVAudioSession.h>
#import <AudioToolbox/AudioToolbox.h>

#include <unistd.h>	// usleep
#include <mach/mach_time.h>	// mach_absolute_time()	// ticks since boot...
#include "quip_prot.h"
#include "sound.h"
#include "debug.h"
#include "CAStreamBasicDescription.h"
#include "BufferManager.h"
#include "DCRejectionFilter.h"


//static unsigned int nchannels=2;		/* default */
static int halting=0;
static int streaming=0;
int audio_state = AUDIO_UNINITED;
static int audio_stream_fd=(-1);	// for streaming to/from disk???
static int the_sample_rate=44100;
//static u_long host_time0=0;
static uint64_t host_time0=0;

//typedef int32_t	audio_sample_type;
typedef float	audio_sample_type;
#define EXPECTED_SOUND_PREC	PREC_SP

static int32_t n_to_record=(-1);	// stream
static audio_sample_type *record_data;
static int rec_stop_requested=0;
static AudioQueueRef audio_input_queue=NULL;
#define N_RECORD_BUFFERS	8
static AudioQueueBufferRef record_bufp[N_RECORD_BUFFERS];;

static int32_t n_to_play=(-1);	// stream
static audio_sample_type *playback_data;
static int play_stop_requested=0;
static AudioQueueRef audio_output_queue=NULL;
#define N_PLAYBACK_BUFFERS	8
static AudioQueueBufferRef playback_bufp[N_PLAYBACK_BUFFERS];;




//	PaError		err;

static Timestamp_Functions dobj_tsf={
	{
		get_sound_seconds,
		get_sound_milliseconds,
		get_sound_microseconds
	}
};

// We use this struct to pass information to the play callback...

typedef struct {
	void *		src_data;
	uint32_t	src_idx;	// index of next sample
	Precision *	src_prec_p;
	int		src_n_channels;	// usually 1 or 2
	int32_t		src_frames_to_go;
	int		src_dup_flag;
} Sound_Data;

//static Sound_Data the_sound, *the_sdp=NULL;

#define CHECK_ERROR(msg)				\
							\
	if( stat != 0 ){				\
		warn(msg);				\
		return;					\
	}


int set_playback_nchan(QSP_ARG_DECL  int channels)
{
#ifdef FOOBAR
	return(0);
#else // ! FOOBAR
	warn("set output n_channels not implemented yet for iOS!?");
	return -1;
#endif // ! FOOBAR
}

static struct mach_timebase_info *mtbi_p=NULL;
static double hTime2nsFactor=1;

static void init_timebase_info(SINGLE_QSP_ARG_DECL)
{
	kern_return_t kerror;
	static struct mach_timebase_info mtbi;
	kerror = mach_timebase_info(&mtbi);
	if( kerror != KERN_SUCCESS ){
		warn("init_mach_time:  failed to fetch timebase info!?");
	} else {
		mtbi_p = (&mtbi);
		hTime2nsFactor = ((double)mtbi.numer)/mtbi.denom;
fprintf(stderr,"hTime2nsFactor = %g\n",hTime2nsFactor);
	}
}

#ifdef FOOBAR

#define COPY_SOUND(type,silence)				\
{								\
	type *dst = (type *) dest;				\
	type *src = (type *) sdp->src_data;			\
	if( sdp->src_dup_flag )					\
		src += sdp->src_idx;				\
	else							\
		src += 2 * sdp->src_idx;			\
								\
	sdp->src_idx += frames_to_copy;				\
								\
	while( frames_to_copy -- ){				\
		*dst++ = *src;					\
		if( sdp->src_dup_flag == 0 )			\
			src++;					\
		*dst++ = *src;					\
		src++;						\
	}							\
	/* zero remainder of final buffer */			\
	while( frames_to_zero -- ){				\
		*dst++ = silence;				\
		*dst++ = silence;				\
	}							\
}

static void _copy_sound_data(QSP_ARG_DECL  void *dest, Sound_Data *sdp, int frames_to_copy, int frames_to_zero )
{
	switch( PREC_CODE( sdp->src_prec_p ) ){
		case PREC_BY:  COPY_SOUND(char,0) break;
		case PREC_IN:  COPY_SOUND(short,0) break;
		default:
			sprintf(ERROR_STRING,
				"copy_sound_data:  unsupported sound precision %s!?",
					PREC_NAME(sdp->src_prec_p));
			warn(ERROR_STRING);
			break;
	}
}
#endif // FOOBAR

/********************** new stuff for ios audio queue interface ******************/


// SMPTE time code flag bits
/*
kSMPTETimeValid
kSMPTETimeRunning
*/

// SMPTE time code types

/*
static void process_smpte_time()
{
	switch(stp->mType){
		case kSMPTETimeType24:
			break;
		case kSMPTETimeType25:
			break;
		case kSMPTETimeType30Drop:
			break;
		case kSMPTETimeType30:
			break;
		case kSMPTETimeType2997:
			break;
		case kSMPTETimeType2997Drop:
			break;
		case kSMPTETimeType60:
			break;
		case kSMPTETimeType5994:
			break;
		case kSMPTETimeType60Drop:
			break;
		case kSMPTETimeType5994Drop:
			break;
		case kSMPTETimeType50:
			break;
		case kSMPTETimeType2398:
			break;
		default:
			warn("Unexpected SMPTE time code type!?");
			break;
	}
}
*/

static void report_audio_error(QSP_ARG_DECL  OSStatus code)
{
	switch(code){
		case kAudioQueueErr_InvalidBuffer:
			warn("invalid audio buffer"); break;
		case kAudioQueueErr_BufferEmpty:
			warn("audio buffer empty"); break;
		case kAudioQueueErr_DisposalPending:
			warn("audio disposal pending"); break;
		case kAudioQueueErr_InvalidProperty:
			warn("invalid audio property"); break;
		case kAudioQueueErr_InvalidPropertySize:
			warn("invalid audio property size"); break;
		case kAudioQueueErr_InvalidParameter:
			warn("invalid audio parameter"); break;
		case kAudioQueueErr_CannotStart:
			warn("audio cannot start"); break;
		case kAudioQueueErr_InvalidDevice:
			warn("invalid audio device"); break;
		case kAudioQueueErr_BufferInQueue:
			warn("audio buffer in queue"); break;
		case kAudioQueueErr_InvalidRunState:
			warn("invalid audio run state"); break;
		case kAudioQueueErr_InvalidQueueType:
			warn("invalid audio queue type"); break;
		case kAudioQueueErr_Permissions:
			warn("audio permission error"); break;
		case kAudioQueueErr_InvalidPropertyValue:
			warn("invalid audio property value"); break;
		case kAudioQueueErr_PrimeTimedOut:
			warn("audio prime timeout"); break;
		case kAudioQueueErr_CodecNotFound:
			warn("audio codec not found"); break;
		case kAudioQueueErr_InvalidCodecAccess:
			warn("invalid audio codec access"); break;
		case kAudioQueueErr_QueueInvalidated:
			warn("audio queue invalidated"); break;
		case kAudioQueueErr_RecordUnderrun:
			warn("audio record underrun"); break;
		case kAudioQueueErr_EnqueueDuringReset:
			warn("audio enqueue during reset"); break;
		case kAudioQueueErr_InvalidOfflineMode:
			warn("invalid audio offline mode"); break;
		case kAudioFormatUnsupportedDataFormatError:
			warn("unsupported audio data format"); break;
		case -50:
			warn("error in user parameter list"); break;
		default:
			advise("audio error not handled in switch");
			NSError *error = [NSError errorWithDomain:NSOSStatusErrorDomain code:code userInfo:nil];
			if( error == NULL ){
				fprintf(stderr,"code = %d, unable to find NSError\n",(int)code);
			} else {
				NSString *s;
				//s=[error localizedFailureReason];
				s=[error localizedDescription];
				if( s == NULL ) {
					//fprintf(stderr,"code = %d, unable to find localizedFailureReason\n",(int)code);
					fprintf(stderr,"code = %d, unable to find localizedDescription\n",(int)code);
				} else {
					fprintf(stderr,"%s\n",s.UTF8String);
				}
			}
			break;
	}
}

#ifdef NOT_USED
static void init_remote_io(SINGLE_QSP_ARG_DECL)
{
	OSStatus stat;
	AudioComponentInstance _rioUnit;
	AudioComponentDescription desc;
	desc.componentType = kAudioUnitType_Output;
	desc.componentSubType = kAudioUnitSubType_RemoteIO;
	desc.componentManufacturer = kAudioUnitManufacturer_Apple;
	desc.componentFlags = 0;
	desc.componentFlagsMask = 0;
	BufferManager *_buffer_manager;
	DCRejectionFilter *_dc_rejection_filter;

	AudioComponent comp = AudioComponentFindNext(NULL, &desc);
	stat=AudioComponentInstanceNew(comp, &_rioUnit);
	CHECK_ERROR("couldn't create a new instance of AURemoteIO");

	//  Enable input and output on AURemoteIO
	//  Input is enabled on the input scope of the input element
	//  Output is enabled on the output scope of the output element

	UInt32 one = 1;
	stat=AudioUnitSetProperty(_rioUnit,				// unit
				kAudioOutputUnitProperty_EnableIO,	// property id
				kAudioUnitScope_Input,			// scope
				1,					// element
				&one,					// data ptr
				sizeof(one)				// size
				);
	CHECK_ERROR("could not enable input on AURemoteIO");
	stat=AudioUnitSetProperty(_rioUnit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &one, sizeof(one));
	CHECK_ERROR("could not enable output on AURemoteIO");

	// Explicitly set the input and output client formats
	// sample rate = 44100, num channels = 1, format = 32 bit floating point

	CAStreamBasicDescription ioFormat = CAStreamBasicDescription(44100, 1, CAStreamBasicDescription::kPCMFormatFloat32, false);
	stat=AudioUnitSetProperty(_rioUnit,
				kAudioUnitProperty_StreamFormat,
				kAudioUnitScope_Output,
				1,
				&ioFormat,
				sizeof(ioFormat)
				);
	CHECK_ERROR("couldn't set the input client format on AURemoteIO");
	stat=AudioUnitSetProperty(_rioUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &ioFormat, sizeof(ioFormat));
	CHECK_ERROR("couldn't set the output client format on AURemoteIO");

	// Set the MaximumFramesPerSlice property. This property is used to describe to an audio unit the maximum number
	// of samples it will be asked to produce on any single given call to AudioUnitRender
	UInt32 maxFramesPerSlice = 4096;
	stat=AudioUnitSetProperty(_rioUnit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0, &maxFramesPerSlice, sizeof(UInt32));
	CHECK_ERROR("couldn't set max frames per slice on AURemoteIO");

	// Get the property value back from AURemoteIO. We are going to use this value to allocate buffers accordingly
	UInt32 propSize = sizeof(UInt32);
	stat=AudioUnitGetProperty(_rioUnit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0, &maxFramesPerSlice, &propSize);
	CHECK_ERROR("couldn't get max frames per slice on AURemoteIO");

	_buffer_manager = new BufferManager(maxFramesPerSlice);
	_dc_rejection_filter = new DCRejectionFilter;

	// We need references to certain data in the render callback
	// This simple struct is used to hold that information
#ifdef FOOBAR
	cd.rioUnit = _rioUnit;
	cd.bufferManager = _bufferManager;
	cd.dcRejectionFilter = _dcRejectionFilter;
	cd.muteAudio = &_muteAudio;
	cd.audioChainIsBeingReconstructed = &_audioChainIsBeingReconstructed;
#endif // FOOBAR

	// Set the render callback on AURemoteIO
	AURenderCallbackStruct renderCallback;
//	renderCallback.inputProc = performRender;
	renderCallback.inputProcRefCon = NULL;
	stat=AudioUnitSetProperty(_rioUnit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &renderCallback, sizeof(renderCallback));
	CHECK_ERROR("couldn't set render callback on AURemoteIO");

	// Initialize the AURemoteIO instance
	stat=AudioUnitInitialize(_rioUnit);
	CHECK_ERROR("couldn't initialize AURemoteIO instance");
}
#endif // NOT_USED

#ifdef FOOBAR

/* This routine will be called by the PortAudio engine when audio is needed.
** It may called at interrupt level on some machines so don't do anything
** that could mess up the system like calling malloc() or free().
*/

static int play_dp_callback( const void *inputBuffer, void *outputBuffer,
						unsigned long frames_per_buffer,
						const PaStreamCallbackTimeInfo* timeInfo,
						PaStreamCallbackFlags statusFlags,
						void *userData )
{
	Sound_Data *sdp = (Sound_Data*)userData;
	int frames_to_copy;
	int frames_to_zero;
	int finished;
	//(void) inputBuffer; /* Prevent unused variable warnings. */

#ifdef CAUTIOUS
	if( sdp->src_frames_to_go < 0 )
		error1(DEFAULT_QSP_ARG  "CAUTIOUS:  play_dp_callback:  source frames_to_go less than zero!?");
#endif // CAUTIOUS

	if( sdp->src_frames_to_go < frames_per_buffer ) {
		frames_to_copy = sdp->src_frames_to_go;
		finished = 1;
	} else {
		frames_to_copy = frames_per_buffer;
		finished = 0;
	}
	sdp->src_frames_to_go -= frames_to_copy;
	frames_to_zero = frames_per_buffer - frames_to_copy;

	copy_sound_data(outputBuffer,sdp,frames_to_copy,frames_to_zero);

	return finished;

} // play_dp_callback
#endif // FOOBAR

extern "C" {

void set_sound_volume(QSP_ARG_DECL  int g)
{
	warn("set_sound_volume not implemented yet for iOS!?");
}

void set_samp_freq(QSP_ARG_DECL  unsigned int req_rate)
{
	CHECK_AUDIO(AUDIO_PLAY);

	// BUG?  should we insure validity?
	the_sample_rate = req_rate;
}

} // end extern "C"

#ifdef FOOBAR
static int init_ios_audio_session(QSP_ARG_DECL  int mode)
{
	static AVAudioSession *my_session=NULL;
	NSError *error = nil;

	if( my_session != NULL ){
fprintf(stderr,"init_ios_audio_session:  already initialized!?\n");
		return -1;
	}

	my_session = [AVAudioSession sharedInstance];
	[my_session setCategory:AVAudioSessionCategoryPlayAndRecord error:&error];

	if( error ){
		sprintf(ERROR_STRING,"Error setting session category");
		warn(ERROR_STRING);
		return -1;
	}

	// set the buffer duration to 5 ms
	NSTimeInterval bufferDuration = .005;
	[my_session setPreferredIOBufferDuration:bufferDuration error:&error];
	if( error ){
		sprintf(ERROR_STRING,"Error setting buffer duration");
		warn(ERROR_STRING);
		return -1;
	}
	[my_session setPreferredSampleRate:44100 error:&error];
	if( error ){
		sprintf(ERROR_STRING,"Error setting sample rate");
		warn(ERROR_STRING);
		return -1;
	}
/*
	// add interruption handler
	[[NSNotificationCenter defaultCenter] addObserver:self
						selector:@selector(handleInterruption:)
						name:AVAudioSessionInterruptionNotification
						object:sessionInstance];
 
	// we don't do anything special in the route change notification
	[[NSNotificationCenter defaultCenter] addObserver:self
						selector:@selector(handleRouteChange:)
						name:AVAudioSessionRouteChangeNotification
						object:sessionInstance];

	// if media services are reset, we need to rebuild our audio chain
	[[NSNotificationCenter defaultCenter]	addObserver:	self
						selector:	@selector(handleMediaServerReset:)
						name:	AVAudioSessionMediaServicesWereResetNotification
						object:	sessionInstance];
  
 */
	// activate the audio session
	[[AVAudioSession sharedInstance] setActive:YES error:&error];
	if( error ){
		sprintf(ERROR_STRING,"Error activating audio session");
		warn(ERROR_STRING);
		return -1;
	}

advise("audio session activated...");
	return 0;
}
#endif // FOOBAR

static void stop_audio_input(SINGLE_QSP_ARG_DECL)
{
	OSStatus status;

	if( audio_input_queue == NULL ){
		warn("stop_audio_input:  audio system not initialized!?");
		return;
	}

advise("Calling AudioQueueStop!");
	status = AudioQueueStop(audio_input_queue,true);
	if( status != 0 )
		report_audio_error(QSP_ARG  status);

	// Even after stopping, previously queued buffers will be delivered...
	rec_stop_requested=1;
}

static void stop_audio_output(SINGLE_QSP_ARG_DECL)
{
	OSStatus status;

	if( audio_output_queue == NULL ){
		warn("stop_audio_output:  audio system not initialized!?");
		return;
	}

advise("Calling AudioQueueStop!");
	status = AudioQueueStop(audio_output_queue,false);
	if( status != 0 )
		report_audio_error(QSP_ARG  status);

	// Even after stopping, previously queued buffers will be delivered...
	play_stop_requested=1;
}

// The output callback is called when the queue needs more data

static void my_output_callback(	void *inUserData,
			AudioQueueRef inAQ,
			AudioQueueBufferRef inBuffer
			)
{
	OSStatus status;
	int n_avail;
	uint32_t n_to_copy=0;

	n_avail = inBuffer->mAudioDataBytesCapacity / sizeof(audio_sample_type);	// BUG get sample size from buffer?

//fprintf(stderr,"my_output_callback:  n_to_play = %d\n",n_to_play);

	// Save the data here before releasing buffer!
	if( n_to_play > 0 ){

		if( n_avail <= n_to_play )
			n_to_copy = n_avail;
		else
			n_to_copy = n_to_play;

		memcpy( inBuffer->mAudioData, playback_data, n_to_copy*sizeof(audio_sample_type) );

		n_to_play -= n_to_copy;
		playback_data += n_to_copy;
	}

	if( n_to_play == 0 ){
		// Don't stop the playback queue until all of the queued buffers have played...
//fprintf(stderr,"my_output_callback:  stopping\n");
		if( ! play_stop_requested )
			stop_audio_output(SINGLE_QSP_ARG);
	} else {
		inBuffer->mAudioDataByteSize = n_to_copy * sizeof(audio_sample_type);	// BUG get sample size from buffer?
		status = AudioQueueEnqueueBuffer ( inAQ, inBuffer, 0, NULL );
		if( status != 0 )
			report_audio_error(QSP_ARG  status);
	}
}

// audio time stamp flag bits
// With simple recording we are seeing a value of 7, which indicates
// the first 3...
// TimeStampSample, TimeStampHostTime and TimeStampRateScalar...

/*
kAudioTimeStampSampleTimeValid
kAudioTimeStampHostTimeValid
kAudioTimeStampRateScalarValid
kAudioTimeStampWordClockTimeValid
kAudioTimeStampSMPTETimeValid
*/

// The callback seems to be called as long as we are queueing buffers, even after we have stopped the queue!?

static void my_input_callback(	void *inUserData,
			AudioQueueRef inAQ,
			AudioQueueBufferRef inBuffer,
			const AudioTimeStamp *inStartTime,
			UInt32 inNumberPacketDescriptions,
			const AudioStreamPacketDescription *inPacketDescs )
{
	OSStatus status;
	int n_avail;
	double d_delta;

	//fprintf(stderr,"my_input_callback, buffer at 0x%lx\n",(u_long) inBuffer);

	n_avail = inBuffer->mAudioDataByteSize / sizeof(audio_sample_type);	// BUG get sample size from buffer?

if( host_time0 == 0 ) host_time0 = inStartTime->mHostTime;

//fprintf(stderr,"mFlags = 0x%x\n",(unsigned int)inStartTime->mFlags);
/*fprintf(stderr,"mSampleTime = %g, mHostTime 0x%llx, host delta = %llu,  mRateScalar = %g\n",
inStartTime->mSampleTime,inStartTime->mHostTime,inStartTime->mHostTime-host_time0,inStartTime->mRateScalar);
*/
	d_delta = inStartTime->mHostTime - host_time0;
	fprintf(stderr,"mSampleTime = %g, %g milleseconds\n",
		inStartTime->mSampleTime,(d_delta * hTime2nsFactor)/1000000) ;

/*
fprintf(stderr,"%d packet descriptions, %d samples available, %ld samples remain to record\n",
(unsigned int)inNumberPacketDescriptions,
n_avail,
n_to_record);
*/

	// Save the data here before releasing buffer!
	if( n_to_record > 0 ){
		int n_to_copy;

		if( n_avail <= n_to_record )
			n_to_copy = n_avail;
		else
			n_to_copy = n_to_record;

		memcpy( record_data, inBuffer->mAudioData, n_to_copy*sizeof(audio_sample_type) );

		n_to_record -= n_to_copy;
		record_data += n_to_copy;
	}

	if( n_to_record == 0 ){
		if( ! rec_stop_requested )
			stop_audio_input(SINGLE_QSP_ARG);
	} else {
		status = AudioQueueEnqueueBuffer ( inAQ, inBuffer, 0, NULL );
		if( status != 0 )
			report_audio_error(QSP_ARG  status);
	}
}

static void init_ios_audio_output(SINGLE_QSP_ARG_DECL)
{
	OSStatus status;
	AudioStreamBasicDescription in_fmt={0};
	void *user_data=NULL;			// for use with callback function
	CFRunLoopRef callback_run_loop=NULL;	// use audio queue internal thread
	CFStringRef callback_run_loop_mode=NULL;	// equivalent to kCFRunLoopCommonModes
	UInt32 flags=0;				// reserved for future use

	if( audio_output_queue != NULL ){
		advise("init_ios_audio_output:  already initialized!?");
		return;
	}

	//if( init_ios_audio_session(QSP_ARG  mode) < 0 ) return;

//advise("calling init_remote_io");
//	init_remote_io(SINGLE_QSP_ARG);

	in_fmt.mSampleRate = 44100;
	in_fmt.mFormatID = kAudioFormatLinearPCM;
	in_fmt.mFormatFlags = /*kAudioFormatFlagsCanonical*/ kAudioFormatFlagIsFloat | kAudioFormatFlagsNativeEndian | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved;
	in_fmt.mFramesPerPacket = 1;
	in_fmt.mChannelsPerFrame = 1;
	in_fmt.mBytesPerFrame = in_fmt.mChannelsPerFrame * sizeof(SInt32);
	in_fmt.mBytesPerPacket = in_fmt.mFramesPerPacket * in_fmt.mBytesPerFrame;
	in_fmt.mBitsPerChannel = 8 * sizeof (SInt32);

advise("Calling AudioQueueNewOutput...");
	status = AudioQueueNewOutput(	&in_fmt,
					my_output_callback,
					user_data,
					callback_run_loop,
					callback_run_loop_mode,
					flags,
					&audio_output_queue );
	if( status != 0 )
		report_audio_error(QSP_ARG  status);

	for (int i = 0; i < N_PLAYBACK_BUFFERS; ++i) {
		status=AudioQueueAllocateBuffer ( audio_output_queue, 8192, &playback_bufp[i] );
		if( status != 0 )
			report_audio_error(QSP_ARG  status);
fprintf(stderr,"allocated playback buffer at 0x%lx\n",(u_long)playback_bufp[i]);
	}
}

static void init_ios_audio_input(SINGLE_QSP_ARG_DECL)
{
	OSStatus status;
	AudioStreamBasicDescription in_fmt={0};
	//AudioQueueInputCallback callback_func=my_input_callback;	// called when buffer is full
	void *user_data=NULL;			// for use with callback function
	CFRunLoopRef callback_run_loop=NULL;	// use audio queue internal thread
	CFStringRef callback_run_loop_mode=NULL;	// equivalent to kCFRunLoopCommonModes
	UInt32 flags=0;				// reserved for future use

	if( audio_input_queue != NULL ){
		advise("init_ios_audio_input:  already initialized!?");
		return;
	}

	//if( init_ios_audio_session(QSP_ARG  mode) < 0 ) return;

//advise("calling init_remote_io");
//	init_remote_io(SINGLE_QSP_ARG);

	in_fmt.mSampleRate = 44100;
	in_fmt.mFormatID = kAudioFormatLinearPCM;
	in_fmt.mFormatFlags = /*kAudioFormatFlagsCanonical*/ kAudioFormatFlagIsFloat | kAudioFormatFlagsNativeEndian | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved;
	in_fmt.mFramesPerPacket = 1;
	in_fmt.mChannelsPerFrame = 1;
	in_fmt.mBytesPerFrame = in_fmt.mChannelsPerFrame * sizeof(SInt32);
	in_fmt.mBytesPerPacket = in_fmt.mFramesPerPacket * in_fmt.mBytesPerFrame;
	in_fmt.mBitsPerChannel = 8 * sizeof (SInt32);

advise("Calling AudioQueueNewInput...");
	status = AudioQueueNewInput(	&in_fmt,
					/*callback_func*/my_input_callback,
					user_data,
					callback_run_loop,
					callback_run_loop_mode,
					flags,
					&audio_input_queue );
	if( status != 0 )
		report_audio_error(QSP_ARG  status);

	for (int i = 0; i < N_RECORD_BUFFERS; ++i) {
		status=AudioQueueAllocateBuffer ( audio_input_queue, 8192, &record_bufp[i] );
		if( status != 0 )
			report_audio_error(QSP_ARG  status);
fprintf(stderr,"allocated buffer at 0x%lx\n",(u_long)record_bufp[i]);
	}
}

static void init_ios_audio(QSP_ARG_DECL  int mode)
{
	if( mtbi_p == NULL ) init_timebase_info(SINGLE_QSP_ARG);

	if( mode == AUDIO_RECORD )
		init_ios_audio_input(SINGLE_QSP_ARG);
	else if( mode == AUDIO_PLAY )
		init_ios_audio_output(SINGLE_QSP_ARG);
	else
		warn("init_ios_audio:  undexpected mode!?");
}


static void start_audio_input(SINGLE_QSP_ARG_DECL)
{
	OSStatus status;

	if( audio_input_queue == NULL ){
		warn("start_audio_input:  audio system not initialized!?");
		init_ios_audio_input(SINGLE_QSP_ARG);
	}

	rec_stop_requested=0;

	for (int i = 0; i < N_RECORD_BUFFERS; ++i) {
		status = AudioQueueEnqueueBuffer ( audio_input_queue, record_bufp[i], 0, NULL );
		if( status != 0 )
			report_audio_error(QSP_ARG  status);
	}

advise("calling AudioQueueStart for input queue...");
	status = AudioQueueStart(audio_input_queue,NULL);
	if( status != 0 )
		report_audio_error(QSP_ARG  status);

	// We probably need to relinquish control before we will receive packets?

}

static void start_audio_output(SINGLE_QSP_ARG_DECL)
{
	OSStatus status;

	if( audio_output_queue == NULL ){
		warn("start_audio_output:  audio system not initialized!?");
		return;
	}

	play_stop_requested=0;

	for (int i = 0; i < N_PLAYBACK_BUFFERS; ++i) {
		if( n_to_play > 0 )
			my_output_callback(NULL,audio_output_queue,playback_bufp[i]);
	}

advise("calling AudioQueueStart for output queue...");
	// Maybe should not call if already started!?
	status = AudioQueueStart(audio_output_queue,NULL);
	if( status != 0 )
		report_audio_error(QSP_ARG  status);
advise("start_audio_output will return.");
}

void audio_init(QSP_ARG_DECL  int mode)
{
	//int channels;
	static int ts_class_inited=0;

advise("audio_init BEGIN");
	init_ios_audio(QSP_ARG  mode);

	if( ! ts_class_inited ){
		add_tsable(dobj_itp,&dobj_tsf,(Item * (*)(QSP_ARG_DECL  const char *))_hunt_obj);
		ts_class_inited++;
	}

#ifdef DEBUG
	if( debug & sound_debug ){
		sprintf(ERROR_STRING,"audio_init:  mode = %d",mode);
		advise(ERROR_STRING);
	}
#endif /* DEBUG */

	if(audio_state == mode) return;

	if(audio_state != AUDIO_UNINITED) {
		/* should we reset the interface??? */
	}

	if(mode == AUDIO_RECORD)
	{
		/* what do we need to do here??? */
		//warn("audio_init:  don't know how to record!?");

	} else if( mode == AUDIO_PLAY ) {
		/* open the device for playback */
		//portaudio_playback_init(SINGLE_QSP_ARG);
	} else if( mode == AUDIO_UNINITED ){	/* de-initialize */
		audio_state = mode;
		return;
	}
#ifdef CAUTIOUS
	else {
		warn("unexpected audio mode requested!?");
	}
#endif	/* CAUTIOUS */



	//channels = nchannels;

	audio_state = mode;

#ifdef FOOBAR
	/* This is a little BUGGY:  if we request a sample rate before a mode has been
	 * initialized, we init for playback.  We'd like to get that same rate when
	 * we subsequently do a recording...
	 */
	if( samp_freq > 0 )
		set_samp_freq(samp_freq);
	else
		set_samp_freq(DEFAULT_SAMP_FREQ);
#endif /* FOOBAR */

}

void halt_play_stream(SINGLE_QSP_ARG_DECL)
{
	if( halting )
		warn("halt_play_stream:  already halting!?");
	if( !streaming )
		warn("halt_play_stream:  not streaming!?");
	halting=1;

	/* wait for disk_reader & audio_writer to finish - should call pthread_join (BUG)! */

	while( streaming && audio_stream_fd != (-1) )
		usleep(100000);
}

void play_stream(QSP_ARG_DECL  int fd)
{
	warn("unimplemented for iOS:  play_stream");
}

int _sound_seek(QSP_ARG_DECL  index_t idx)
{
	warn("unimplemented for iOS:  _sound_seek");
    return 0;
}

int _async_play_sound(QSP_ARG_DECL  Data_Obj *dp)
{
	warn("unimplemented for iOS:  _async_play_sound");
    return 0;
}

void set_stereo_output(QSP_ARG_DECL  int is_stereo)
{
	warn("unimplemented for iOS:  set_stereo_output");
}

void pause_sound(SINGLE_QSP_ARG_DECL)
{
	warn("unimplemented for iOS:  pause_sound");
}

static int good_for_sound(QSP_ARG_DECL  Data_Obj *dp)
{
	if( OBJ_PREC(dp) != EXPECTED_SOUND_PREC ){
		sprintf(ERROR_STRING,"good_for_sound:  object %s (%s) should have %s precision!?",
			OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)),NAME_FOR_PREC_CODE(EXPECTED_SOUND_PREC) );
		warn(ERROR_STRING);
		return 0;
	}
	if( OBJ_COMPS(dp) != 1 ){
		sprintf(ERROR_STRING,"good_for_sound:  object %s should have 1 components!?",
			OBJ_NAME(dp) );
		warn(ERROR_STRING);
		return 0;
	}
	if( !IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"good_for_sound:  object %s should be contiguous!?",
			OBJ_NAME(dp) );
		warn(ERROR_STRING);
		return 0;
	}
	return 1;
}

//
// We start the record queue when we create it...
// For the sake of efficiency, we should only run it when we want
// to record.  So we need to have a more complex sense of state?

void _record_sound(QSP_ARG_DECL  Data_Obj *dp)
{
	if(audio_state!=AUDIO_RECORD)
		audio_init(QSP_ARG  AUDIO_RECORD);

	if( ! good_for_sound(QSP_ARG  dp) ) return;

	n_to_record = OBJ_N_MACH_ELTS(dp);
	record_data = (audio_sample_type *) OBJ_DATA_PTR(dp);

	start_audio_input(SINGLE_QSP_ARG);
}


void play_sound(QSP_ARG_DECL  Data_Obj *dp)
{
	if(audio_state!=AUDIO_PLAY)
		audio_init(QSP_ARG  AUDIO_PLAY);

	if( ! good_for_sound(QSP_ARG  dp) ) return;

	n_to_play = OBJ_N_MACH_ELTS(dp);
	playback_data = (audio_sample_type *) OBJ_DATA_PTR(dp);

	start_audio_output(SINGLE_QSP_ARG);
}

void record_stream(QSP_ARG_DECL  int sound_fd, int timestamp_fd)
{
	if(audio_state!=AUDIO_RECORD) audio_init(QSP_ARG  AUDIO_RECORD);
	advise("not sure what to do after initialization to record a stream in iOS...");
}

#endif /* BUILD_FOR_IOS */

#ifdef MAY_BE_USEFUL

UInt32 audioInputIsAvailable;

UInt32 propertySize = sizeof (audioInputIsAvailable);

 

	// A nonzero value on output means that
	// audio input is available
	AudioSessionGetProperty ( kAudioSessionProperty_AudioInputAvailable,
				&propertySize, &audioInputIsAvailable );



// This code initializes an audio player?
NSString *soundFilePath = [[NSBundle mainBundle] pathForResource: @"sound" ofType: @"wav"];
NSURL *fileURL = [[NSURL alloc] initFileURLWithPath: soundFilePath];

AVAudioPlayer *newPlayer = [[AVAudioPlayer alloc] initWithContentsOfURL: fileURL error: nil];
[fileURL release];
self.player = newPlayer;
[newPlayer release];
[self.player prepareToPlay];
[self.player setDelegate: self];



// example delegate method, shows changing the legend on the play button

- (void) audioPlayerDidFinishPlaying: (AVAudioPlayer *) player successfully: (BOOL) flag
{
	if (flag == YES) {
		[self.button setTitle: @"Play" forState: UIControlStateNormal];
	}
}


o
- (IBAction) playOrPause: (id) sender {
	// if already playing, then pause
	if (self.player.playing) {
		[self.button setTitle: @"Play" forState: UIControlStateHighlighted];
		[self.button setTitle: @"Play" forState: UIControlStateNormal];
		[self.player pause];
	// if stopped or paused, start playing
	} else {
		[self.button setTitle: @"Pause" forState: UIControlStateHighlighted];
		[self.button setTitle: @"Pause" forState: UIControlStateNormal];
		[self.player play];
	}
}

[self.player setVolume: 1.0];	// available range is 0.0 through 1.0





// audio queue stuff



static const int kNumberBuffers = 3;

// Create a data structure to manage information needed by the audio queue

struct myAQStruct {
	AudioFileID					mAudioFile;
	CAStreamBasicDescription		mDataFormat;
	AudioQueueRef				mQueue;
	AudioQueueBufferRef			mBuffers[kNumberBuffers];
	SInt64						mCurrentPacket;
	UInt32						mNumPacketsToRead;
	AudioStreamPacketDescription	*mPacketDescs;
	bool							mDone;
};

// Define a playback audio queue callback function



Float32 volume = 1;

AudioQueueSetParameter ( myAQstruct.audioQueueObject, kAudioQueueParam_Volume, volume );




typedef struct AudioQueueLevelMeterState {
	Float32	mAveragePower;
	Float32	mPeakPower;
};  AudioQueueLevelMeterState;




// from aurioTouch

	catch (CAXException &e) {
	NSLog(@"Error returned from setupIOUnit: %d: %s", (int)e.mError, e.mOperation);
	}
	catch (...) {
	NSLog(@"Unknown error returned from setupIOUnit");
	}

	return;
#endif // MAY_BE_USEFUL
