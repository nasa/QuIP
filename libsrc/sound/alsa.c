#include "quip_config.h"

#ifdef HAVE_ALSA

#ifndef USE_OSS_SOUND

#include <stdio.h>
#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_IOCTL_H
#include <sys/ioctl.h>
#endif
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif
#ifdef HAVE_TIME_H
#include <time.h>		/* localtime() */
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_ALSA_ASOUNDLIB_H
#include <alsa/asoundlib.h>
#endif

#include "quip_prot.h"
#include "function.h"
#include "debug.h"
#include "sound.h"

static snd_pcm_t *playback_handle=NULL;
static snd_pcm_hw_params_t *hw_params=NULL;
static snd_pcm_sw_params_t *sw_params=NULL;

#define MIXER_NAME "/dev/mixer"

#define AUDIO_UNINITED	0
#define AUDIO_PLAY	1
#define AUDIO_RECORD	2

#define MONO	1
#define STEREO 2
#define QUAD	4

static unsigned int nchannels=2;		/* default */

int audio_state = AUDIO_UNINITED;

static int mfd;
#define DEFAULT_SAMP_FREQ	48000
static int samp_freq=(-1);


#define N_SAMPS_PER_BUF	0x4000			/* 16K, about 1 second's worth */
#define N_BUFFERS	4

static int streaming=0;
static int halting=0;
static int audio_stream_fd=(-1);

static Timestamp_Functions dobj_tsf={
	{
		get_sound_seconds,
		get_sound_milliseconds,
		get_sound_microseconds
	}
};

#ifdef NOT_USED
void show_playback_state(SINGLE_QSP_ARG_DECL)
{
	snd_pcm_state_t state;

	if( playback_handle == NULL ){
		warn("show_playback_state:  playback_handle not initialized");
		return;
	}
	state = snd_pcm_state(playback_handle);

	/* now report the state */
	switch(state){
		case SND_PCM_STATE_OPEN:  advise("playback device state is OPEN"); break;
		case SND_PCM_STATE_SETUP:  advise("playback device state is SETUP, waiting to prepare"); break;
		case SND_PCM_STATE_PREPARED:  advise("playback device state is PREPARED, waiting to start"); break;
		case SND_PCM_STATE_RUNNING:  advise("playback device state is RUNNING"); break;
		case SND_PCM_STATE_XRUN:  advise("playback device state is XRUN (overrun or underrun"); break;
		case SND_PCM_STATE_DRAINING:  advise("playback device state is DRAINING"); break;
		case SND_PCM_STATE_PAUSED:  advise("playback device state is PAUSED"); break;
		case SND_PCM_STATE_SUSPENDED: advise("playback device state is SUSPENDED"); break;
		case SND_PCM_STATE_DISCONNECTED: advise("playback device state is DISCONNECTED"); break;
		default: warn("unexpected state code"); break;
	}
} /* end show_playback_state */
#endif /* NOT_USED */

int set_playback_nchan(QSP_ARG_DECL  int channels)
{
	int err;

	if( audio_state != AUDIO_PLAY ){
		warn("set_playback_nchan:  playback mode must be initialized before setting n_channels");
		return(-1);
	}
	if( channels < 1 || channels > 2 ){
		/* Does card support quad sound??? */
		if( channels != 4 ){
			warn("Sorry, number of channels must be 1,2, or 4");
			return(-1);
		} else {
			advise("Assuming audio card is quad-capable!");
		}
	}
	nchannels = channels;

	if ((err = snd_pcm_hw_params_set_channels (playback_handle, hw_params, nchannels)) < 0) {
		sprintf(ERROR_STRING,"Cannot set channel count (%s)\n", snd_strerror (err));
		warn(ERROR_STRING);
		return(-1);
	}

	if ((err = snd_pcm_hw_params (playback_handle, hw_params)) < 0) {
		sprintf(ERROR_STRING,"Cannot set hw parameters (%s)\n", snd_strerror (err));
		warn(ERROR_STRING);
		return(-1);
	}
	return(0);
}

/*
 *	Underrun and suspend recovery
  */

static int xrun_recovery(snd_pcm_t *handle, int err)
{
	if (err == -EPIPE) {/* under-run */
NADVISE("underrun, calling snd_pcm_prepare()");
		err = snd_pcm_prepare(handle);
		if (err < 0)
			printf("Can't recovery from underrun, prepare failed: %s\n", snd_strerror(err));
		return 0;
	} else if (err == -ESTRPIPE) {
		while ((err = snd_pcm_resume(handle)) == -EAGAIN)
			sleep(1);	/* wait until the suspend flag is released */
		if (err < 0) {
			err = snd_pcm_prepare(handle);
			if (err < 0)
				printf("Can't recovery from suspend, prepare failed: %s\n", snd_strerror(err));
		}
		return 0;
	}
	return err;
}

void play_sound(QSP_ARG_DECL  Data_Obj *dp)
{
	int err;
	short *data_ptr;
	long n_req, n_written;
	snd_pcm_state_t state;

	if(audio_state!=AUDIO_PLAY) audio_init(QSP_ARG  AUDIO_PLAY);	

	if( OBJ_MACH_PREC(dp) != PREC_IN ){
		sprintf(ERROR_STRING,"Object %s has precision %s, should be %s for sounds",OBJ_NAME(dp),
			PREC_NAME(OBJ_MACH_PREC_PTR(dp)),NAME_FOR_PREC_CODE(PREC_IN));
		warn(ERROR_STRING);
		return;
	}

	if( OBJ_COMPS(dp) != nchannels ){
		sprintf(ERROR_STRING,
	"Sound %s has %ld components, output configured for %d channels!?",  
			OBJ_NAME(dp),(long)OBJ_COMPS(dp),nchannels);
		advise(ERROR_STRING);

		if( set_playback_nchan(QSP_ARG  OBJ_COMPS(dp)) < 0 ){
			sprintf(ERROR_STRING,
	"Sound %s has illegal number of channels (%ld)",
				OBJ_NAME(dp),(long)OBJ_COMPS(dp));
			warn(ERROR_STRING);
			return;
		}
	}

	/*
	sprintf(ERROR_STRING,"audio_state=%d",audio_state);
	advise(ERROR_STRING);
	*/

	/* Before we start writing, apparently we need to call snd_pcm_start first?
	 * should we preload the buffer?
	 */
	state = snd_pcm_state(playback_handle);
	if( state == SND_PCM_STATE_XRUN ){	/* from a previous write? */
		err = snd_pcm_prepare(playback_handle);
		if (err < 0){
			sprintf(ERROR_STRING,"play_sound:  couldn't restore prepared state after underrun");
			warn(ERROR_STRING);
			return;
		}
	}
	/*
	if( (err=snd_pcm_start(playback_handle)) < 0 ){
		warn("some error with snd_pcm_start() !? ");
	}
	*/

	/* write interleaved */

	n_req = OBJ_COLS(dp);
	n_written=0;
	data_ptr = (short *)OBJ_DATA_PTR(dp);
	while( n_req > 0 ){
		err = snd_pcm_writei(playback_handle, data_ptr, n_req);
		if( err < 0 ){
			if( err == -EAGAIN ){
				/* this is ok? */
			} else {
				if( xrun_recovery(playback_handle,err) < 0 ){
					sprintf (ERROR_STRING,
						"write to audio interface failed (%s)\n", snd_strerror (err));
					warn(ERROR_STRING);
					return;
				}
				err = 0;	/* signal that no bytes were written */
			}
		}
		/* if no error, the return value is number of samples written? */
		if( err != n_req ){
			sprintf(ERROR_STRING,"%ld samples requested, %d written",n_req,err);
			advise(ERROR_STRING);
		}
		n_req -= err;
		/* advance pointer over what has already played */
		data_ptr += err * OBJ_COMPS(dp);
	}
	/* Now we are probably in the underrun state - if we will play again, prepare here... */
}

void set_sound_volume(QSP_ARG_DECL  int g)
{
	int pcm_gain;

	CHECK_AUDIO(AUDIO_PLAY);

	pcm_gain = 25700;

#ifdef HAVE_IOCTL
	ioctl(mfd, SOUND_MIXER_WRITE_VOLUME, &g);
	ioctl(mfd, SOUND_MIXER_WRITE_PCM, &pcm_gain); 
#else
	warn("set_sound_volume:  don't know how to do this!?");
#endif
}

void set_samp_freq(QSP_ARG_DECL  unsigned int req_rate)
{
	unsigned int prev_rate;
	int err;

	CHECK_AUDIO(AUDIO_PLAY);

	prev_rate = samp_freq;
	samp_freq = req_rate;

	if ((err = snd_pcm_hw_params_set_rate_near (playback_handle, hw_params, &req_rate, 0)) < 0) {
		sprintf (ERROR_STRING, "cannot set sample rate near %d (%s)\n", req_rate,snd_strerror (err));
 		warn (ERROR_STRING);
	}
}

static void set_hw_params()
{
	int err;
	unsigned int rate;
 
 	if ((err = snd_pcm_hw_params_any (playback_handle, hw_params)) < 0) {
		fprintf(stderr, "cannot initialize hardware parameter structure (%s)\n", snd_strerror (err));
		exit (1);
	}

	if ((err = snd_pcm_hw_params_set_access (playback_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
		fprintf(stderr, "cannot set access type (%s)\n", snd_strerror (err));
 		exit (1);
	}

	/* set the format to 16bit little-endian */
	if ((err = snd_pcm_hw_params_set_format (playback_handle, hw_params, SND_PCM_FORMAT_S16_LE)) < 0) {
		fprintf (stderr, "cannot set sample format (%s)\n", snd_strerror (err));
	 	exit (1);
	}

	rate=48000;
	if ((err = snd_pcm_hw_params_set_rate_near (playback_handle, hw_params, &rate, 0)) < 0) {
		fprintf (stderr, "cannot set sample rate near 48000 (%s)\n", snd_strerror (err));
 		exit (1);
	}

	if ((err = snd_pcm_hw_params_set_channels (playback_handle, hw_params, nchannels)) < 0) {
		fprintf (stderr, "cannot set channel count (%s)\n", snd_strerror (err));
		exit (1);
	}

	if ((err = snd_pcm_hw_params (playback_handle, hw_params)) < 0) {
		fprintf (stderr, "cannot set parameters (%s)\n", snd_strerror (err));
		exit (1);
	}
	
	/* don't free, might need later... */
	/*
	snd_pcm_hw_params_free (hw_params);
	*/
}

static int set_sw_params(SINGLE_QSP_ARG_DECL)
{
        int err;

#ifdef CAUTIOUS
	if( playback_handle == NULL ){
		warn("CAUTIOUS:  set_sw_params:  playback handle has not been set!?");
		return(-1);
	}
	if( sw_params == NULL ){
		warn("CAUTIOUS:  set_sw_params:  sw_params pointer has not been set!?");
		return(-1);
	}
#endif /* CAUTIOUS */


        /* get the current sw_params */
        err = snd_pcm_sw_params_current(playback_handle, sw_params);
        if (err < 0) {
                printf("Unable to determine current sw_params for playback: %s\n", snd_strerror(err));
                return err;
        }
        /* start the transfer when the buffer is almost full: */
        /* (buffer_size / avail_min) * avail_min */
	/* jbm:  Not sure which buffer this refers to, but probably the driver's internal ring buffer? */
        err = snd_pcm_sw_params_set_start_threshold(playback_handle, sw_params, 512);
        if (err < 0) {
                printf("Unable to set start threshold mode for playback: %s\n", snd_strerror(err));
                return err;
        }
#ifdef FOOBAR
        /* allow the transfer when at least period_size samples can be processed */
	/* jbm:  what should this be??? */
        err = snd_pcm_sw_params_set_avail_min(playback_handle, sw_params, 512);
        if (err < 0) {
                printf("Unable to set avail min for playback: %s\n", snd_strerror(err));
                return err;
        }
#endif
        /* align all transfers to 1 sample */
        err = snd_pcm_sw_params_set_xfer_align(playback_handle, sw_params, 1);
        if (err < 0) {
                printf("Unable to set transfer align for playback: %s\n", snd_strerror(err));
                return err;
        }
        /* write the parameters to the playback device */
        err = snd_pcm_sw_params(playback_handle, sw_params);
        if (err < 0) {
                printf("Unable to set sw params for playback: %s\n", snd_strerror(err));
                return err;
        }
        return 0;
}

#define ALSA_NAME	"plughw:0,0"

static void alsa_playback_init(SINGLE_QSP_ARG_DECL)
{
	int err;
	if ((err = snd_pcm_open (&playback_handle, ALSA_NAME, SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
		fprintf(stderr, "cannot open audio device %s (%s)\n", ALSA_NAME, snd_strerror (err));
		exit (1);
	}
	/* we pass the address of the pointer, which gets sets if successful... */
	if ((err = snd_pcm_hw_params_malloc (&hw_params)) < 0) {
		fprintf(stderr, "cannot allocate hardware parameter structure (%s)\n", snd_strerror (err));
	 	exit (1);
	}
 
	if ((err = snd_pcm_sw_params_malloc (&sw_params)) < 0) {
		fprintf(stderr, "cannot allocate software parameter structure (%s)\n", snd_strerror (err));
	 	exit (1);
	}

	set_hw_params();
	set_sw_params(SINGLE_QSP_ARG);
	
	if ((err = snd_pcm_prepare (playback_handle)) < 0) {
		fprintf (stderr, "cannot prepare audio interface for use (%s)\n", snd_strerror (err));
		exit (1);
	}
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

	if( mfd == 0 ){
		if( (mfd = open(MIXER_NAME,O_RDWR,0)) < 0 ){
			sprintf(ERROR_STRING,"open(%s)",MIXER_NAME);
			perror(ERROR_STRING);
			sprintf(ERROR_STRING,"error opening mixer device %s",
				MIXER_NAME);
			warn(ERROR_STRING);
		}
else {
if( debug & sound_debug ){
advise("mixer opened");
}
}
	}

	if(audio_state != AUDIO_UNINITED) {
		/* should we reset the interface??? */
	}

	if(mode == AUDIO_RECORD)
	{
		/* what do we need to do here??? */

	} else if( mode == AUDIO_PLAY ) {
		/* open the ALSA device for playback */
		alsa_playback_init(SINGLE_QSP_ARG);
	} else if( mode == AUDIO_UNINITED ){	/* de-initialize */
		audio_state = mode;
		return;
	}
#ifdef CAUTIOUS
	else {
		warn("unexpected audio mode requested!?");
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
		warn("halt_play_stream:  already halting!?");
	if( !streaming )
		warn("halt_play_stream:  not streaming!?");
	halting=1;

	/* wait for disk_reader & audio_writer to finish - should call pthread_join (BUG)! */

	while( streaming && audio_stream_fd != (-1) )
		usleep(100000);
}


void play_stream(QSP_ARG_DECL  int fd) { warn("unimplemented for ALSA:  play_stream"); }
void set_stereo_output(QSP_ARG_DECL  int is_stereo) { warn("unimplemented for ALSA:  set_stereo_output"); }
void pause_sound(SINGLE_QSP_ARG_DECL) { warn("unimplemented for ALSA:  pause_sound"); }

#endif /* ! USE_OSS_SOUND */


#endif /* HAVE_ALSA */
