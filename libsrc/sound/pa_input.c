#include "quip_config.h"

#ifdef HAVE_PORTAUDIO

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

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#ifdef HAVE_TIME_H
#include <time.h>		/* localtime() */
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "sound.h"

/* assume 40kHz sample rate, 400 samples is 10msec worth... */
/* #define N_SAMPS_PER_BUF		512 */
#define N_SAMPS_PER_BUF		2048

/* If 1 buffer is around 12 msec, then 80 buffers would be about 1 second */
/* #define N_BUFFERS	256 */		/* 3 sec? */
#define N_BUFFERS	(0x20000/N_SAMPS_PER_BUF)		/* 3 sec? */

static pthread_t audio_thr;
static pthread_t writer_thr;
static int n_record_channels=0;
static int streaming=0;

/* audio_reader sets newest to a nonnegative integer when
 * data is ready; we want to have enough buffers available that
 * the reader can get a bit ahead.  oldest gets set the first
 * time newest is set, and thereafter is updated only by disk_writer.
 */
static int active_buf=0;		/* currently receiving audion data */
static int wait_usecs=1;
static int newest=(-1), oldest=(-1);

static int halting=0;
static Data_Obj *audio_stream_dp=NULL;
static int audio_stream_fd=(-1);
static int timestamp_stream_fd=(-1);
static struct timeval tv_tbl[N_BUFFERS];

typedef struct sound_device {
	char *			sd_name;
//	snd_pcm_t *		sd_capture_handle;
//	snd_pcm_hw_params_t *	sd_hw_params;
} Sound_Device;

typedef struct {
	Query_Stack *	ara_qsp;
} Audio_Reader_Args;

static Sound_Device *the_sdp=NULL;
#define DEFAULT_SOUND_DEVICE		"default"	/* capture device */

//ITEM_INTERFACE_DECLARATIONS_STATIC( Sound_Device, snddev )
static Item_Type *snddev_itp=NULL;
static ITEM_INIT_FUNC(Sound_Device,snddev,0)
static ITEM_CHECK_FUNC(Sound_Device,snddev)
static ITEM_NEW_FUNC(Sound_Device,snddev)

#define init_snddevs()	_init_snddevs(SINGLE_QSP_ARG)
#define snddev_of(s)	_snddev_of(QSP_ARG  s)
#define new_snddev(s)	_new_snddev(QSP_ARG  s)



static Sound_Device * init_sound_device(QSP_ARG_DECL  const char *devname);

static int _record_sound(QSP_ARG_DECL  Data_Obj *dp, Sound_Device *sdp);
static int init_sound_hardware(QSP_ARG_DECL  Sound_Device *sdp);

void set_sound_gain(QSP_ARG_DECL  int g)
{
	int recsrc;
	recsrc = 0;

	WARN("set_sound_gain:  not implemented for portaudio");
}

#ifdef NOT_USED_YET
static void select_input(QSP_ARG_DECL  int mask )
{
	int recsrc;
	recsrc = 0;

	WARN("select_input:  not yet implemented for portaudio");
}
#endif // NOT_USED_YET

void select_mic_input(SINGLE_QSP_ARG_DECL)
{
	//select_input(QSP_ARG  SOUND_MASK_MIC);
	WARN("select_mic_input not implemented for portaudio!?");
}

void select_line_input(SINGLE_QSP_ARG_DECL)
{
	//select_input(QSP_ARG  SOUND_MASK_LINE);
	WARN("select_line_input not implemented for portaudio!?");
}

// BUG these functions don't appear to be interface-specific,
// yet they are duplicated here and in alsa_input.c ...

double get_sound_seconds(QSP_ARG_DECL  Item *ip,dimension_t frame)
{
	Data_Obj *dp;
	u_long sec;
	struct tm tm1;
	Timestamp_Data *tm_p;

	dp = (Data_Obj *)ip;

	if( ! object_is_sound(QSP_ARG  dp) ) return(-1.0);

	/* convert time stamp to broken-down time */

	tm_p = (Timestamp_Data *)OBJ_DATA_PTR(dp);

	tm1.tm_sec = tm_p->tsd_sec;
	tm1.tm_min = tm_p->tsd_min;
	tm1.tm_hour = tm_p->tsd_hour;
	tm1.tm_mday = tm_p->tsd_mday;
	tm1.tm_mon = tm_p->tsd_mon;
	tm1.tm_year = tm_p->tsd_year;

	sec = mktime(&tm1);

	return((double)sec);
}

double get_sound_microseconds(QSP_ARG_DECL  Item *ip,dimension_t frame)
{
	Data_Obj *dp;
	u_long usec;
	Timestamp_Data *tm_p;

	dp = (Data_Obj *)ip;

	if( ! object_is_sound(QSP_ARG  dp) ) return(-1.0);

	tm_p = (Timestamp_Data *)OBJ_DATA_PTR(dp);

	usec = tm_p->tsd_csec;
	usec *= 100;
	usec += tm_p->tsd_ccsec;
	usec *= 100;
	usec += tm_p->tsd_usec;

	return((double)usec);
}

double get_sound_milliseconds(QSP_ARG_DECL  Item *ip,dimension_t frame)
{
	if( ! object_is_sound(QSP_ARG  (Data_Obj *)ip) ) return(-1.0);
	return( get_sound_microseconds(QSP_ARG  ip,frame) / 1000.0 );
}

void halt_rec_stream(SINGLE_QSP_ARG_DECL)
{
	if( halting ){
		WARN("halt_rec_stream:  already halting!?");
		return;
	}

	if( !streaming ){
		WARN("halt_rec_stream:  not streaming!?");
		return;
	}

	halting=1;

	/* wait for disk writer to finish - should call pthread_join! */

	/*
	while( streaming && audio_stream_fd != (-1) )
		usleep(100000);
		*/

	if( pthread_join(audio_thr,NULL) < 0 )
		WARN("halt_rec_stream:  error in pthread_join");

	streaming = 0;
	halting = 0;
}

void set_stereo_input(QSP_ARG_DECL  int is_stereo) { WARN("unimplemented for portaudio:  set_stereo_input"); }

void record_sound(QSP_ARG_DECL  Data_Obj *dp)
{
	CHECK_AUDIO(AUDIO_RECORD);

	if( the_sdp == NULL ){
		the_sdp = init_sound_device(QSP_ARG  DEFAULT_SOUND_DEVICE);
		if( the_sdp == NULL ) return;
	}
	_record_sound(QSP_ARG  dp,the_sdp);
}

#define FRAMES_PER_CHUNK	128		/* how should we set this?? */

static int setup_record(QSP_ARG_DECL  Sound_Device *sdp)
{
#ifdef FOOBAR
	snd_pcm_uframes_t n_frames;
	int dir=0;
	int err;

	/* Set period size to 32 frames. */
	n_frames = FRAMES_PER_CHUNK;
	snd_pcm_hw_params_set_period_size_near(sdp->sd_capture_handle,
		sdp->sd_hw_params, &n_frames, &dir);

	if((err = snd_pcm_hw_params(sdp->sd_capture_handle, sdp->sd_hw_params)) < 0) {
		sprintf(ERROR_STRING, "setup_record:  cannot set parameters (%s)\n",
			snd_strerror(err));
		WARN(ERROR_STRING);
		return(-1);
	}

	if((err = snd_pcm_prepare(sdp->sd_capture_handle)) < 0) {
		sprintf(ERROR_STRING,"record_sound:  cannot prepare audio interface %s for use (%s)\n",
			sdp->sd_name,snd_strerror(err));
		WARN(ERROR_STRING);
		return(-1);
	}
	return(0);
#else // ! FOOBAR
	WARN("setup_record not implemented for portaudio!?");
	return -1;
#endif // ! FOOBAR

}

static int _record_sound(QSP_ARG_DECL  Data_Obj *dp, Sound_Device *sdp)
{
#ifdef FOOBAR
	snd_pcm_uframes_t n;
	char *ptr;
	
	if( setup_record(QSP_ARG  sdp) < 0 ) return(-1);

	n = OBJ_N_TYPE_ELTS(dp)/OBJ_COMPS(dp);		/* assume tdim =2 if stereo... */
sprintf(ERROR_STRING,"_record_sound:  n_frames = %ld, tdim = %ld, size = %d",
n, (long)OBJ_COMPS(dp), PREC_SIZE(OBJ_MACH_PREC_PTR(dp)));
advise(ERROR_STRING);

	ptr = (char *)OBJ_DATA_PTR(dp);
	return read_sound_frames(QSP_ARG  sdp,ptr,n,OBJ_COMPS(dp)*PREC_SIZE(OBJ_MACH_PREC_PTR(dp)));
#else // ! FOOBAR
	WARN("_record_sound not implemented for portaudio!?");
	return -1;
#endif // ! FOOBAR
}

static int init_sound_hardware(QSP_ARG_DECL  Sound_Device *sdp)
{
	WARN("init_sound_hardware:  not implemented for portaudio!?");
	return(0);
}

static Sound_Device * init_sound_device(QSP_ARG_DECL  const char *devname)
{
	Sound_Device *sdp;

	sdp = snddev_of(devname);
	if( sdp != NULL ){
		sprintf(ERROR_STRING,"init_sound_device:  device %s is already initialized",devname);
		WARN(ERROR_STRING);
		return(sdp);
	}

	sdp = new_snddev(devname);
	if( sdp == NULL ){
		sprintf(ERROR_STRING,"init_sound_device:  unable to create struct for device %s",devname);
		WARN(ERROR_STRING);
		return(NULL);
	}

	if( init_sound_hardware(QSP_ARG  sdp) < 0 ){
		sprintf(ERROR_STRING,"init_sound_device:  Unable to initialize sound hardware for device %s",sdp->sd_name);
		WARN(ERROR_STRING);
		/* BUG cleanup here */
		return(NULL);
	}

	return(sdp);
}

/* If we are streaming to disk, we assume that we will record asynchronously until a halt command
 * is given.  We check to see if we are stereo or mono, and create a pair of sound vectors accordingly.
 */

static Data_Obj * init_stream_obj(SINGLE_QSP_ARG_DECL)
{
	Dimension_Set ds1;

	ds1.ds_dimension[0] = n_record_channels;
	ds1.ds_dimension[1] = N_SAMPS_PER_BUF;
	ds1.ds_dimension[2] = N_BUFFERS;	/* for double buffering */
	ds1.ds_dimension[3] = 1;
	ds1.ds_dimension[4] = 1;

	audio_stream_dp = make_dobj("_audio_stream_obj",&ds1,PREC_FOR_CODE(PREC_IN));
	return( audio_stream_dp );
}

// disk writer is a separate thread, but should inherit a qsp from the caller

static  void *disk_writer(void *arg)
{
	short *ptr;
	int n_want, n_written;
	struct timeval *tvp;
	Query_Stack *qsp;

	qsp = arg;

	n_want = OBJ_COMPS(audio_stream_dp) * OBJ_COLS(audio_stream_dp) * sizeof(short);
	while(streaming){
		/* wait for first buffer */
		while( oldest == (-1) ){
			if( !streaming ) goto writer_done;
			usleep(wait_usecs);		/* one frame of data */
		}
		/* wait for an extra buffer to complete */
		while( oldest == newest ){
			if( !streaming ) goto writer_done;
			usleep(wait_usecs);		/* one frame of data */
		}


		ptr = (short *)OBJ_DATA_PTR(audio_stream_dp);
		ptr += oldest * OBJ_ROW_INC(audio_stream_dp);

		if( (n_written=write(audio_stream_fd,ptr,n_want)) < 0 ){
			tell_sys_error("write");
			WARN("error writing audio stream file");
		} else if( n_written != n_want ){
			sprintf(ERROR_STRING,"disk_writer:  %d audio bytes requested, %d actually written",
					n_want,n_written);
			WARN(ERROR_STRING);
		}

		tvp = &tv_tbl[oldest];
		if( (n_written=write(timestamp_stream_fd,tvp,sizeof(*tvp))) < 0 ){
			tell_sys_error("write");
			WARN("error writing audio timestamp stream file");
		} else if( n_written != sizeof(*tvp) ){
	sprintf(ERROR_STRING,"disk_writer:  %d timestamp bytes requested, %d actually written",
				n_want,n_written);
			WARN(ERROR_STRING);
		}

		oldest++;
		if( oldest >= N_BUFFERS )
			oldest = 0;
	}
writer_done:

	close(audio_stream_fd);
	delvec(audio_stream_dp);
	audio_stream_fd=(-1);

	close(timestamp_stream_fd);
	timestamp_stream_fd=(-1);

	return(NULL);
}

static int read_sound_frames(QSP_ARG_DECL  Sound_Device *sdp, char *ptr, int32_t frames_remaining, int frame_size )
{
	WARN("read_sound_frames not implemented for portaudio!?");
	return -1;
}

static void *audio_reader(void *arg)
{
	int warned_once=0;
	struct timeval tv;
	struct timezone tz;
	int framesize;
	//snd_pcm_uframes_t frames_requested;
	int32_t frames_requested;
	Audio_Reader_Args *ara_p;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *qsp;
#endif

	ara_p = arg;
#ifdef THREAD_SAFE_QUERY
	qsp = ara_p->ara_qsp;
#endif
	framesize = OBJ_COMPS(audio_stream_dp)*PREC_SIZE(OBJ_MACH_PREC_PTR(audio_stream_dp));
	frames_requested = OBJ_COLS(audio_stream_dp);

	while( !halting ){
		short *ptr;

		ptr = (short *) OBJ_DATA_PTR(audio_stream_dp);
		ptr += active_buf * OBJ_ROW_INC(audio_stream_dp);

		/* now fill this buffer with data */
		if( read_sound_frames(QSP_ARG  the_sdp,(char *)ptr,frames_requested,framesize) < 0 ){
			WARN("error reading audio data");
			halting=1;
		}

		/* get a timestamp for this bufferful */
		if( gettimeofday(&tv,&tz) < 0 && ! warned_once ){
			perror("gettimeofday");
			WARN("error getting time stamp for sound stream");
			warned_once++;
		}
		tv_tbl[ active_buf ] = tv;

		/* we hope that read doesn't return asynchronously!? */

		newest = active_buf;
		if( oldest < 0 ) oldest=newest;

		active_buf ++;
		if( active_buf >= N_BUFFERS ) active_buf=0;

		while( active_buf == oldest ){
			sprintf(ERROR_STRING,"audio_reader:  disk writer not keeping up (active_buf = %d, oldest = %d)!?",active_buf,oldest);
			WARN(ERROR_STRING);
			usleep(wait_usecs);	/* wait one buffer */
		}

		/* yield the processor here */
		usleep(10);

	}

	/* Wait for disk writer to finish */
	usleep(500000);

	/* We close the file here, otherwise the sound driver
	 * continues to record and buffer data.
	 */

	audio_init(QSP_ARG  AUDIO_UNINITED);

	return(NULL);

} /* end audio_reader */

static int stream_record_init(SINGLE_QSP_ARG_DECL)
{
	if( streaming ){
		WARN("stream_record_init:  already streaming, need to halt before initiating another recording");
		return(-1);
	}

	if( init_stream_obj(SINGLE_QSP_ARG) == NULL ){
		WARN("stream_record_init:  error creating audio stream object");
		return(-1);
	}
	streaming=1;
	return(0);
}


void record_stream(QSP_ARG_DECL  int sound_fd, int timestamp_fd)
{
	pthread_attr_t attr1;

	audio_stream_fd = sound_fd;
	timestamp_stream_fd = timestamp_fd;

	if( the_sdp == NULL ){
		the_sdp = init_sound_device(QSP_ARG  DEFAULT_SOUND_DEVICE);
		if( the_sdp == NULL ) return;
	}

	if( setup_record(QSP_ARG  the_sdp) < 0 ) return;

	if( stream_record_init(SINGLE_QSP_ARG) < 0 ){
		WARN("error initializing stream recorder");
		return;
	}

	wait_usecs = 25 /* usecs per sample @ 40 kHz */ * N_SAMPS_PER_BUF;


	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);
	pthread_create(&audio_thr,&attr1,audio_reader,NULL);
	pthread_create(&writer_thr,&attr1,disk_writer,THIS_QSP);
}


#endif /* HAVE_PORTAUDIO */

