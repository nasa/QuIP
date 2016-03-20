
#include "quip_config.h"

#ifdef HAVE_PORTAUDIO

#ifdef HAVE_PORTAUDIO_H
#include <portaudio.h>
#endif

#include <unistd.h>	// usleep

#include "quip_prot.h"
#include "sound.h"

static unsigned int nchannels=2;		/* default */
static int halting=0;
static int streaming=0;
int audio_state = AUDIO_UNINITED;
static int audio_stream_fd=(-1);	// for streaming to/from disk???
static PaStream * playback_stream=NULL;
static int the_sample_rate=44100;

//    PaError             err;

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

static Sound_Data the_sound, *the_sdp=NULL;

int set_playback_nchan(QSP_ARG_DECL  int channels)
{
#ifdef FOOBAR
	return(0);
#else // ! FOOBAR
	WARN("set output n_channels not implemented yet for portaudio!?");
	return -1;
#endif // ! FOOBAR
}

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

static void copy_sound_data(void *dest, Sound_Data *sdp, int frames_to_copy, int frames_to_zero )
{
	switch( PREC_CODE( sdp->src_prec_p ) ){
		case PREC_BY:  COPY_SOUND(char,0) break;
		case PREC_IN:  COPY_SOUND(short,0) break;
		default:
			sprintf(DEFAULT_ERROR_STRING,
				"copy_sound_data:  unsupported sound precision %s!?",
					PREC_NAME(sdp->src_prec_p));
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}
}



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

void play_sound(QSP_ARG_DECL  Data_Obj *dp)
{
	PaStreamParameters  outputParameters;
	PaTime              streamOpened;
	PaError             err;

	if(audio_state!=AUDIO_PLAY) audio_init(QSP_ARG  AUDIO_PLAY);
#ifdef FOOBAR
	if( OBJ_MACH_PREC(dp) != PREC_IN ){
		sprintf(ERROR_STRING,"Object %s has precision %s, should be %s for sounds",OBJ_NAME(dp),
			PREC_NAME(OBJ_MACH_PREC_PTR(dp)),NAME_FOR_PREC_CODE(PREC_IN));
		NWARN(ERROR_STRING);
		return;
	}
#endif // FOOBAR
	// what precisions can be played - any?

	if( OBJ_COMPS(dp) > 2 ){
		sprintf(ERROR_STRING,"Sound %s has %ld components, expected 1 or 2!?",
			OBJ_NAME(dp),(long)OBJ_COMPS(dp));
		advise(ERROR_STRING);
		return;
	}

	/* write interleaved */

	the_sdp = &the_sound;

	the_sdp->src_data = OBJ_DATA_PTR(dp);
	the_sdp->src_frames_to_go = OBJ_N_MACH_ELTS(dp) / OBJ_COMPS(dp) ;
	the_sdp->src_n_channels = OBJ_COMPS(dp) ;
	the_sdp->src_idx = 0;
	the_sdp->src_prec_p = OBJ_PREC_PTR(dp);
	if( OBJ_COMPS(dp) == 1 )
		the_sdp->src_dup_flag = 1;
	else
		the_sdp->src_dup_flag = 0;

	outputParameters.device = Pa_GetDefaultOutputDevice(); /* Default output device. */
	if (outputParameters.device == paNoDevice) {
		WARN("No default output audio device!?");
		return;
	}
	outputParameters.channelCount = 2;			/* Stereo output. */
	switch( PREC_MACH_CODE( OBJ_PREC_PTR(dp) ) ){
		case PREC_BY:
			outputParameters.sampleFormat = paInt8 ;
			break;
		case PREC_IN:
			outputParameters.sampleFormat = paInt16 ;
			break;
		default:
			sprintf(ERROR_STRING,"play_sound:  unhandled obj precision %s!?",
				PREC_NAME(OBJ_PREC_PTR(dp)));
			WARN(ERROR_STRING);
			outputParameters.sampleFormat = paInt32 ;
			break;
	}

	outputParameters.suggestedLatency =
		Pa_GetDeviceInfo( outputParameters.device )->defaultLowOutputLatency;
	outputParameters.hostApiSpecificStreamInfo = NULL;

	err = Pa_OpenStream( &playback_stream,
		NULL,      /* No input. */
		&outputParameters,
		the_sample_rate,
		256,       /* Frames per buffer. */
		paClipOff, /* We won't output out of range samples so don't bother clipping them. */
		play_dp_callback,
		the_sdp );

	streamOpened = Pa_GetStreamTime( playback_stream ); /* Time in seconds when stream was opened (approx). */

	err = Pa_StartStream( playback_stream );
	if( err != paNoError ){
		WARN("Error starting playback stream!?");
		goto close_it;
	}

	// Could make this a "wait_sound" function to allow
	// asynchronous playback???

	while( ( err = Pa_IsStreamActive( playback_stream ) ) == 1 ){
		Pa_Sleep(100);
	}

	if( err < 0 ){
		WARN("Error occurred during sound playback!?");
	}

close_it:
	err = Pa_CloseStream( playback_stream );
	if( err != paNoError ){
		WARN("Error closing playback stream!?");
	}
}

void set_sound_volume(QSP_ARG_DECL  int g)
{
#ifdef FOOBAR
	int pcm_gain;

	CHECK_AUDIO(AUDIO_PLAY);

	pcm_gain = 25700;

#ifdef HAVE_IOCTL
	ioctl(mfd, SOUND_MIXER_WRITE_VOLUME, &g);
	ioctl(mfd, SOUND_MIXER_WRITE_PCM, &pcm_gain);
#else
	WARN("set_sound_volume:  don't know how to do this!?");
#endif
#else // ! FOOBAR
	WARN("set_sound_volume not implemented yet for portaudio!?");
#endif // ! FOOBAR
}

void set_samp_freq(QSP_ARG_DECL  unsigned int req_rate)
{
	CHECK_AUDIO(AUDIO_PLAY);

	// BUG?  should we insure validity?
	the_sample_rate = req_rate;
}

static void portaudio_playback_init(SINGLE_QSP_ARG_DECL)
{
	if( Pa_Initialize() != paNoError )
		WARN("Error initializing portaudio for playback!?");
}

void audio_init(QSP_ARG_DECL  int mode)
{
	int channels;
	static int ts_class_inited=0;


	if( ! ts_class_inited ){
		add_tsable(QSP_ARG  dobj_itp,&dobj_tsf,(Item * (*)(QSP_ARG_DECL  const char *))hunt_obj);
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
		WARN("audio_init:  don't know how to record!?");

	} else if( mode == AUDIO_PLAY ) {
		/* open the device for playback */
		portaudio_playback_init(SINGLE_QSP_ARG);
	} else if( mode == AUDIO_UNINITED ){	/* de-initialize */
		audio_state = mode;
		return;
	}
#ifdef CAUTIOUS
	else {
		WARN("unexpected audio mode requested!?");
	}
#endif	/* CAUTIOUS */



	channels = nchannels;

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
		WARN("halt_play_stream:  already halting!?");
	if( !streaming )
		WARN("halt_play_stream:  not streaming!?");
	halting=1;

	/* wait for disk_reader & audio_writer to finish - should call pthread_join (BUG)! */

	while( streaming && audio_stream_fd != (-1) )
		usleep(100000);
}

void play_stream(QSP_ARG_DECL  int fd)
{
	WARN("unimplemented for portaudio:  play_stream");
}

void set_stereo_output(QSP_ARG_DECL  int is_stereo)
{
	WARN("unimplemented for portaudio:  set_stereo_output");
}

void pause_sound(SINGLE_QSP_ARG_DECL)
{
	WARN("unimplemented for portaudio:  pause_sound");
}

#endif /* HAVE_PORTAUDIO */

