
#include "quip_config.h"

#ifdef HAVE_PORTAUDIO

#ifdef HAVE_PORTAUDIO_H
#include <portaudio.h>
#endif

#include <unistd.h>	// usleep

#include "quip_prot.h"
#include "sound.h"
#include "function.h"

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
	int32_t		src_n_frames;	// so we can seek...
	int		src_dup_flag;
} Sound_Data;

static Sound_Data the_sound, *the_sdp=NULL;

int set_playback_nchan(QSP_ARG_DECL  int channels)
{
#ifdef FOOBAR
	return(0);
#else // ! FOOBAR
	warn("set output n_channels not implemented yet for portaudio!?");
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

#define copy_sound_data(dest,sdp,frames_to_copy,frames_to_zero ) _copy_sound_data(QSP_ARG  dest,sdp,frames_to_copy,frames_to_zero )

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
	Query_Stack *qsp;

	qsp = userData;

	//(void) inputBuffer; /* Prevent unused variable warnings. */

	assert( sdp->src_frames_to_go >= 0 );

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

int _sound_seek(QSP_ARG_DECL  index_t idx)
{
	// index_t is unsigned so we don't need to check for negative!
	if( idx >= the_sdp->src_n_frames ){
		sprintf(ERROR_STRING,
	"sound_seek:  bad index (%d), should be in the range 0-%d!?",
			idx,the_sdp->src_n_frames);
		warn(ERROR_STRING);
		return -1;
	}
	the_sdp->src_idx = idx;
	the_sdp->src_frames_to_go = the_sdp->src_n_frames - idx;
	return 0;
}

int _async_play_sound(QSP_ARG_DECL  Data_Obj *dp)
{
	PaStreamParameters  outputParameters;
	PaTime              streamOpened;
	PaError             err;

	if(audio_state!=AUDIO_PLAY) audio_init(QSP_ARG  AUDIO_PLAY);

	if( OBJ_MACH_PREC(dp) != PREC_IN ){
		sprintf(ERROR_STRING,"Object %s has precision %s, should be %s for sounds",OBJ_NAME(dp),
			PREC_NAME(OBJ_MACH_PREC_PTR(dp)),NAME_FOR_PREC_CODE(PREC_IN));
		warn(ERROR_STRING);
		return -1;
	}

	// what precisions can be played - any?

	if( OBJ_COMPS(dp) > 2 ){
		sprintf(ERROR_STRING,"Sound %s has %ld components, expected 1 or 2!?",
			OBJ_NAME(dp),(long)OBJ_COMPS(dp));
		advise(ERROR_STRING);
		return -1;
	}

	/* write interleaved */

	the_sdp = &the_sound;

	the_sdp->src_data = OBJ_DATA_PTR(dp);
	the_sdp->src_n_frames = OBJ_N_MACH_ELTS(dp) / OBJ_COMPS(dp) ;
	the_sdp->src_frames_to_go = the_sdp->src_n_frames;
fprintf(stderr,"play_sound:  object %s has %d frames\n",
OBJ_NAME(dp),the_sdp->src_frames_to_go);
fprintf(stderr,"play_sound:  the_sdp = 0x%lx\n",(long)the_sdp);
	the_sdp->src_n_channels = OBJ_COMPS(dp) ;
	the_sdp->src_idx = 0;	// start at the beginning!
	the_sdp->src_prec_p = OBJ_PREC_PTR(dp);
	if( OBJ_COMPS(dp) == 1 )
		the_sdp->src_dup_flag = 1;
	else
		the_sdp->src_dup_flag = 0;

	outputParameters.device = Pa_GetDefaultOutputDevice(); /* Default output device. */
	if (outputParameters.device == paNoDevice) {
		warn("No default output audio device!?");
		return -1;
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
			warn(ERROR_STRING);
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
		warn("Error starting playback stream!?");
		err = Pa_CloseStream( playback_stream );
		if( err != paNoError ){
			warn("Error closing playback stream!?");
		}
		return -1;
	}
	return 0;
} // end async_play_sound

void _play_sound(QSP_ARG_DECL  Data_Obj *dp)
{
	PaError             err;

	if( async_play_sound(dp) < 0 )
		return;

	// Could make this a "wait_sound" function to allow
	// asynchronous playback???

	while( ( err = Pa_IsStreamActive( playback_stream ) ) == 1 ){
		Pa_Sleep(100);
	}

	if( err < 0 ){
		warn("Error occurred during sound playback!?");
	}

	err = Pa_CloseStream( playback_stream );
	if( err != paNoError ){
		warn("Error closing playback stream!?");
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
	warn("set_sound_volume:  don't know how to do this!?");
#endif
#else // ! FOOBAR
	warn("set_sound_volume not implemented yet for portaudio!?");
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
		warn("Error initializing portaudio for playback!?");
}

void audio_init(QSP_ARG_DECL  int mode)
{
	int channels;
	static int ts_class_inited=0;


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

	if(mode == AUDIO_RECORD) {
		/* what do we need to do here??? */
		warn("audio_init:  don't know how to record!?");

	} else if( mode == AUDIO_PLAY ) {
		/* open the device for playback */
		portaudio_playback_init(SINGLE_QSP_ARG);
	} else {
		assert( mode == AUDIO_UNINITED );
		/* de-initialize */
		audio_state = mode;
		return;
	}

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
	warn("unimplemented for portaudio:  play_stream");
}

void set_stereo_output(QSP_ARG_DECL  int is_stereo)
{
	warn("unimplemented for portaudio:  set_stereo_output");
}

void pause_sound(SINGLE_QSP_ARG_DECL)
{
	warn("unimplemented for portaudio:  pause_sound");
}

#endif /* HAVE_PORTAUDIO */

