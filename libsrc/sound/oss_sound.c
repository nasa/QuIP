/* this file provides the interface to the OSS sound module... */
#include "quip_config.h"

char VersionId_sound_oss_sound[] = QUIP_VERSION_STRING;

#ifdef USE_OSS_SOUND

#include <stdio.h>

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#ifdef HAVE_FCTNL_H
#include <fcntl.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_SYS_IOCTL_H
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

#include "getbuf.h"
#include "debug.h"
#include "sound.h"
#include "function.h"		/* add_tsable */

#define DEVICE_NAME "/dev/dsp"
#define MIXER_NAME "/dev/mixer"

#define AUDIO_UNINITED	0
#define AUDIO_PLAY	1
#define AUDIO_RECORD	2

#define MONO   1
#define STEREO 2
#define QUAD   4

static int nchannels=1;

static int audio_state = AUDIO_UNINITED;

static short *buf = NULL;
static int mfd;
static int afd=(-1);
#define DEFAULT_SAMP_FREQ	16000
static int samp_freq=(-1);

#define CHECK_AUDIO(state)	if( audio_state == AUDIO_UNINITED ) audio_init(state);	

static pthread_t audio_thr;
static pthread_t writer_thr;

#define N_SAMPS_PER_BUF	0x4000			/* 16K, about 1 second's worth */
#define N_BUFFERS	4

static int streaming=0;
static int active_buf=0;
static int buf_to_write=(-1);
static int halting=0;
static Data_Obj *audio_stream_dp;
static int audio_stream_fd=(-1);
static int timestamp_stream_fd=(-1);

static struct timeval tv_tbl[N_BUFFERS];

void set_stereo_input(int is_stereo)
{
	CHECK_AUDIO(AUDIO_RECORD);

	if( ioctl(afd, SNDCTL_DSP_STEREO, &is_stereo) == -1 ){
		perror("ioctl");
		close(afd);
		fprintf(stderr, "ioctl error (DSP_STEREO)\n");
		return;
	}
}

void set_stereo_output(int is_stereo)
{
	CHECK_AUDIO(AUDIO_PLAY);

	if( ioctl(afd, SNDCTL_DSP_STEREO, &is_stereo) == -1 ){
		perror("ioctl");
		close(afd);
		fprintf(stderr, "ioctl error (DSP_STEREO)\n");
		return;
	}
}

int set_playback_nchan(int channels)
{
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

	/* Why set state to record??? We can only record one channel... */
	/*
	if(audio_state != AUDIO_RECORD) audio_init(AUDIO_RECORD);
	*/

	return(0);
}

void record_sound(Data_Obj *dp)
{ 
	struct timeval tv;
	struct timezone tz;
	struct tm *tmp;
	int mode = AUDIO_RECORD;
	dimension_t n_reserve, n_record;
	int i,n_read;
	Timestamp_Data *tp;
	u_short *sp;
	
	/*
	sprintf(error_string,"audio_state=%d",audio_state);
	advise(error_string);
	*/
	
	if(audio_state!=AUDIO_RECORD) audio_init(mode);

	/* Do we need to reinitialize to get set up??? */
	
	if( MACHINE_PREC(dp) != PREC_IN ){
		sprintf(error_string,
	"Object %s has precision %s, should be %s for sounds",dp->dt_name,
			prec_name[MACHINE_PREC(dp)],prec_name[PREC_IN]);
		warn(error_string);
		return;
	}

	/* We need to reserve some space at the beginning of the data vector for
	 * the time stamp;  We record:
	 *	centiseconds
	 *	centi-centi-seconds (modulo 100 usec)
	 *	residual microseconds
	 *	seconds
	 *	minutes
	 *	hours
	 *	day-of-month
	 *	month
	 *	year
	 *	
	 * That is 9 numbers...
	 */

	/* BUG we ought to make sure that our data vector has more than 9 samples...*/

	/* We want to round up to a multiple of the number of channels */

	n_reserve = ( ( N_TIMESTAMP_WORDS + dp->dt_tdim - 1 ) / dp->dt_tdim ) * dp->dt_tdim;

	if( gettimeofday(&tv,&tz) < 0 ){
		perror("gettimeofday");
		sprintf(error_string,"error getting time stamp for sound %s",dp->dt_name);
		warn(error_string);
	} else {
		/* We convert the seconds from the epoch into month,day
		 * minute, etc.
		 * This requires a little more storage, but has the advantage
		 * of making all the numbers small, so that if we play it as
		 * a sound we don't hear too much...
		 *
		 * We might want to do the same thing with the numbers of usecs...
		 */

		/* correct for local time */
		tmp=localtime(&tv.tv_sec);
	}

	/* Write the timestamp data into the data block.  What we are doing here is not
	 * portable across endian-ness BUG.
	 */
	tp = dp->dt_data;

	tp->tsd_csec = tv.tv_usec / 10000;
	tp->tsd_ccsec = (tv.tv_usec / 100) % 100;
	tp->tsd_usec = tv.tv_usec % 100;

	tp->tsd_sec = tmp->tm_sec;
	tp->tsd_min = tmp->tm_min;
	tp->tsd_hour = tmp->tm_hour;
	tp->tsd_mday = tmp->tm_mday;
	tp->tsd_mon = tmp->tm_mon;
	tp->tsd_year = tmp->tm_year;

	/* we ought to zero the rest... */
	i = n_reserve - N_TIMESTAMP_WORDS;
	sp = (u_short *)(tp+1);
	while(i--) *sp++ = 0;

	/* Now read the sound device */

	n_record = dp->dt_nelts - n_reserve;
	if( (n_read=read(afd, (((short *)dp->dt_data)+n_reserve), n_record * sizeof(short) )) < 0 ){
		perror("read");
		sprintf(error_string,"Error recording sound %s",dp->dt_name);
		warn(error_string);
	} else if( n_read != n_record*sizeof(short)){
		sprintf(error_string,"record_sound %s:  %ld bytes requested, %d actually read",
			dp->dt_name,n_record,n_read);
		warn(error_string);
	}

	/* We close the file here, otherwise the sound driver
	 * continues to record and buffer data.
	 */

	audio_init(AUDIO_UNINITED);
}

void play_sound(Data_Obj *dp)
{
	if( MACHINE_PREC(dp) != PREC_IN ){
		sprintf(error_string,"Object %s has precision %s, should be %s for sounds",dp->dt_name,
			prec_name[MACHINE_PREC(dp)],prec_name[PREC_IN]);
		warn(error_string);
		return;
	}

	if( dp->dt_tdim != nchannels ){
		sprintf(error_string,
	"Sound %s has %ld components, output configured for %d channels!?",  
			dp->dt_name,dp->dt_tdim,nchannels);
		advise(error_string);

		if( set_playback_nchan(dp->dt_tdim) < 0 ){
			sprintf(error_string,
	"Sound %s has illegal number of channels (%ld)",
				dp->dt_name,dp->dt_tdim);
			warn(error_string);
			return;
		}
       }

	if(audio_state!=AUDIO_PLAY) audio_init(AUDIO_PLAY);	

	/*
	sprintf(error_string,"audio_state=%d",audio_state);
	advise(error_string);
	*/

	if(write(afd, dp->dt_data, dp->dt_nelts * sizeof(*buf)) != dp->dt_nelts * sizeof(*buf)){
		perror("write");
		warn("playback error");
	}
}

void pause_sound()
{
	CHECK_AUDIO(AUDIO_PLAY);

	ioctl(afd, SNDCTL_DSP_RESET, 0);
}	

void set_sound_gain(int g)
{
	int recsrc;
	recsrc = 0;

	CHECK_AUDIO(AUDIO_RECORD);

	ioctl(mfd, SOUND_MIXER_READ_RECSRC, &recsrc);
	if(recsrc & SOUND_MASK_LINE) {
advise("sound recording souce is line...");
		ioctl(mfd, SOUND_MIXER_WRITE_LINE, &g);
	} else {
advise("sound recording souce is the microphone...");
		ioctl(mfd, SOUND_MIXER_WRITE_MIC, &g);
	}
}

static void select_input( int mask )
{
	int recsrc;
	recsrc = 0;

	CHECK_AUDIO(AUDIO_RECORD);

	if( ioctl(mfd, SOUND_MIXER_READ_RECSRC, &recsrc) < 0 ){
		tell_sys_error("ioctl(SOUND_MIXER_READ_RECSRC)");
		warn("error fetching input source");
		return;
	}
	recsrc &= ~ (SOUND_MASK_LINE|SOUND_MASK_MIC);
	recsrc |= mask;
	if( ioctl(mfd, SOUND_MIXER_WRITE_RECSRC, &recsrc) < 0 ){
		tell_sys_error("ioctl(SOUND_MIXER_WRITE_RECSRC)");
		warn("error selecting microphone input");
	}
}

void select_mic_input()
{
	select_input(SOUND_MASK_MIC);
}

void select_line_input()
{
	select_input(SOUND_MASK_LINE);
}


void set_sound_volume(int g)
{
	int pcm_gain;

	CHECK_AUDIO(AUDIO_PLAY);

	pcm_gain = 25700;

	ioctl(mfd, SOUND_MIXER_WRITE_VOLUME, &g);
	ioctl(mfd, SOUND_MIXER_WRITE_PCM, &pcm_gain); 
}

void set_samp_freq(unsigned int req_rate)
{
	int prev_rate;

	CHECK_AUDIO(AUDIO_PLAY);

	prev_rate = samp_freq;
	samp_freq = req_rate;
	if(ioctl(afd, SNDCTL_DSP_SPEED, &samp_freq) == -1 ){
		perror("ioctl");
		sprintf(error_string,"error setting sample frequency to %d",req_rate);
		warn(error_string);
		if( prev_rate > 0 ){
			sprintf(error_string,"reverting to previous rate %d",prev_rate);
			advise(error_string);
		}
		/* close(afd); */
		return;
	}
	if( req_rate != samp_freq){
		sprintf(error_string,"frequency %d requested, %d actually set",
			req_rate,samp_freq);
		warn(error_string);
		return;
	}
}

double get_sound_seconds(Item *ip,dimension_t frame)
{
	Data_Obj *dp;
	u_long sec;
	struct tm tm1;
	Timestamp_Data *tm_p;

	dp = (Data_Obj *)ip;

	if( ! object_is_sound(dp) ) return(-1.0);

	/* convert time stamp to broken-down time */

	tm_p = dp->dt_data;

	tm1.tm_sec = tm_p->tsd_sec;
	tm1.tm_min = tm_p->tsd_min;
	tm1.tm_hour = tm_p->tsd_hour;
	tm1.tm_mday = tm_p->tsd_mday;
	tm1.tm_mon = tm_p->tsd_mon;
	tm1.tm_year = tm_p->tsd_year;

	sec = mktime(&tm1);

	return((double)sec);
}

double get_sound_microseconds(Item *ip,dimension_t frame)
{
	Data_Obj *dp;
	u_long usec;
	Timestamp_Data *tm_p;

	dp = (Data_Obj *)ip;

	if( ! object_is_sound(dp) ) return(-1.0);

	tm_p = dp->dt_data;

	usec = tm_p->tsd_csec;
	usec *= 100;
	usec += tm_p->tsd_ccsec;
	usec *= 100;
	usec += tm_p->tsd_usec;

	return((double)usec);
}

double get_sound_milliseconds(Item *ip,dimension_t frame)
{
	if( ! object_is_sound((Data_Obj *)ip) ) return(-1.0);
	return( get_sound_microseconds(ip,frame) / 1000.0 );
}



static Timestamp_Functions dobj_tsf={
	{
		get_sound_seconds,
		get_sound_milliseconds,
		get_sound_microseconds
	}
};


void audio_init(int mode)
{
	int format;
	int nformat;
	int recsrc;
	int mask;
	int channels;
	static int ts_class_inited=0;

	if( ! ts_class_inited ){
		add_tsable(dobj_itp,&dobj_tsf,(Item * (*)(char *))hunt_obj);
		ts_class_inited++;
	}

#ifdef DEBUG
	if( debug & sound_debug ){
		sprintf(error_string,"audio_init:  mode = %d",mode);
		advise(error_string);
	}
#endif /* DEBUG */

	if(audio_state == mode) return;

	if( mfd == 0 ){
		if( (mfd = open(MIXER_NAME,O_RDWR,0)) < 0 ){
			perror("open");
			sprintf(error_string,"error opening mixer device %s",
				MIXER_NAME);
			warn(error_string);
		}
else {
if( debug & sound_debug ){
advise("mixer opened");
}
}
	}

	if(audio_state != AUDIO_UNINITED) {
#ifdef DEBUG
		if( debug & sound_debug ){
			sprintf(error_string,
				"closing file descriptor, afd=%d",afd);
			advise(error_string);
		}
#endif /* DEBUG */
		close(afd);
	}

	if(mode == AUDIO_RECORD)
	{
		if((afd = open(DEVICE_NAME, O_RDONLY, 0)) < 0) {
			perror("open");
			fprintf(stderr, "error opening audio device %s\n", DEVICE_NAME);
			fflush(stderr);
			return; 
		}
else {
if( debug & sound_debug ){
advise("audio device opened for reading");
}
}

		//recsrc = SOUND_MASK_MIC;
		recsrc = SOUND_MASK_LINE;
		ioctl(mfd, SOUND_MIXER_WRITE_RECSRC, &recsrc);
	
	/*mfd = */
	} else if( mode == AUDIO_PLAY ) {
		if((afd = open(DEVICE_NAME, O_WRONLY, 0)) < 0){
			perror("open");
			fprintf(stderr, "error opening audio device %s\n", DEVICE_NAME);
			fflush(stderr);
			return;
		}
	/*mfd = */
	} else if( mode == AUDIO_UNINITED ){	/* de-initialize */
		audio_state = mode;
		return;
	}
#ifdef CAUTIOUS
	else {
		warn("unexpected audio mode requested!?");
	}
#endif	/* CAUTIOUS */

	
	fcntl(afd, F_SETFD, FD_CLOEXEC);

	ioctl(afd, SNDCTL_DSP_GETFMTS, &mask);
	
	format = AFMT_S16_LE;

	nformat = format;
	if(ioctl(afd, SNDCTL_DSP_SETFMT, &format) == -1 || format != nformat) {
		perror("ioctl");
		close(afd);
		fprintf(stderr,"ioctl error (DSP_SETFMT)\n");
		return;
	}

	channels = nchannels;

	if(ioctl(afd, SNDCTL_DSP_CHANNELS, &channels) == -1 ){
		perror("ioctl");
		close(afd);
		fprintf(stderr, "ioctl error (SNDCTL_DSP_CHANNELS)\n");
		return;
	}

	audio_state = mode;

	/* This is a little BUGGY:  if we request a sample rate before a mode has been
	 * initialized, we init for playback.  We'd like to get that same rate when
	 * we subsequently do a recording...
	 */
	if( samp_freq > 0 )
		set_samp_freq(samp_freq);
	else
		set_samp_freq(DEFAULT_SAMP_FREQ);
}

/* If we are streaming to disk, we assume that we will record asynchronously until a halt command
 * is given.  We check to see if we are stereo or mono, and create a pair of sound vectors accordingly.
 */

static Data_Obj * init_stream_obj(int n_channels)
{
	Dimension_Set ds1;

	ds1.ds_dimension[0] = n_channels;
	ds1.ds_dimension[1] = N_SAMPS_PER_BUF;
	ds1.ds_dimension[2] = N_BUFFERS;	/* for double buffering */
	ds1.ds_dimension[3] = 1;
	ds1.ds_dimension[4] = 1;

	audio_stream_dp = make_dobj("_audio_stream_obj",&ds1,PREC_IN);
	return( audio_stream_dp );
}

static int stream_record_init()
{
	int recsrc;
	int n_channels;

	if( streaming ){
		warn("stream_record_init:  already streaming, need to halt before initiating another recording");
		return(-1);
	}

	if(audio_state!=AUDIO_RECORD) audio_init(AUDIO_RECORD);

	if( ioctl(mfd, SOUND_MIXER_READ_RECSRC, &recsrc) < 0 ){
		tell_sys_error("ioctl(SOUND_MIXER_READ_RECSRC)");
		warn("stream_record_init:  error fetching input source");
		return(-1);
	}
	if( recsrc & SOUND_MASK_LINE )
		n_channels=2;
	else
		n_channels=1;

	if( init_stream_obj(n_channels) == NO_OBJ ){
		warn("stream_record_init:  error creating audio stream object");
		return(-1);
	}
	streaming=1;
	return(0);
}


static void *audio_reader(void *arg)
{
	int n_channels;
	int n_want, n_read;
	int warned_once=0;
	struct timeval tv;
	struct timezone tz;

	n_channels = audio_stream_dp->dt_tdim;

	n_want = n_channels * N_SAMPS_PER_BUF * sizeof(short);

	while( !halting ){
		short *ptr;

		ptr = (short *) audio_stream_dp->dt_data;
		ptr += active_buf * audio_stream_dp->dt_rowinc;

		if( (n_read=read(afd, ptr, n_want)) < 0 ){
			perror("read");
			warn("Error recording sound stream");
		} else if( n_read != n_want ){
			sprintf(error_string,"audio_reader:  %d bytes requested, %d actually read",
				n_want,n_read);
			warn(error_string);
		}
		/* get a timestamp for this bufferful */
		if( gettimeofday(&tv,&tz) < 0 && ! warned_once ){
			perror("gettimeofday");
			warn("error getting time stamp for sound stream");
			warned_once++;
		}
		tv_tbl[ active_buf ] = tv;

		/* we hope that read doesn't return asynchronously!? */

		while( buf_to_write != (-1) ){
			warn("audio_reader:  disk writer not keeping up!?");
			usleep(500000);		/* 500 msec */
		}

		buf_to_write = active_buf;
		active_buf ++;
		if( active_buf >= N_BUFFERS ) active_buf=0;

		/* yield the processor here */
		usleep(10);
	}

	/* Wait for disk writer to finish */
	usleep(500000);
	streaming=0;

	/* We close the file here, otherwise the sound driver
	 * continues to record and buffer data.
	 */

	audio_init(AUDIO_UNINITED);

	return(NULL);
}

static  void *disk_writer(void *arg)
{
	short *ptr;
	int n_want, n_written;
	struct timeval *tvp;

	n_want = audio_stream_dp->dt_tdim * audio_stream_dp->dt_cols * sizeof(short);

	while(streaming){
		/* wait for a complete buffer */
		while( buf_to_write == (-1) ){
			if( !streaming ) goto writer_done;
			usleep(100000);		/* 100 msec */
		}

		ptr = audio_stream_dp->dt_data;
		ptr += buf_to_write * audio_stream_dp->dt_rowinc;

		if( (n_written=write(audio_stream_fd,ptr,n_want)) < 0 ){
			tell_sys_error("write");
			warn("error writing audio stream file");
		} else if( n_written != n_want ){
			sprintf(error_string,"disk_writer:  %d audio bytes requested, %d actually written",
					n_want,n_written);
			warn(error_string);
		}

		tvp = &tv_tbl[buf_to_write];
		if( (n_written=write(timestamp_stream_fd,tvp,sizeof(*tvp))) < 0 ){
			tell_sys_error("write");
			warn("error writing audio timestamp stream file");
		} else if( n_written != sizeof(*tvp) ){
	sprintf(error_string,"disk_writer:  %d timestamp bytes requested, %d actually written",
				n_want,n_written);
			warn(error_string);
		}

		buf_to_write = (-1);	/* signal done writing */
	}
writer_done:

	close(audio_stream_fd);
	delvec(audio_stream_dp);
	audio_stream_fd=(-1);

	close(timestamp_stream_fd);
	timestamp_stream_fd=(-1);

	return(NULL);
}

void record_stream(int sound_fd, int timestamp_fd)
{
	pthread_attr_t attr1;

	audio_stream_fd = sound_fd;
	timestamp_stream_fd = timestamp_fd;

	if( stream_record_init() < 0 ){
		warn("error initializing stream recorder");
		return;
	}

	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);
	pthread_create(&audio_thr,&attr1,audio_reader,NULL);
	pthread_create(&writer_thr,&attr1,disk_writer,NULL);
}

void halt_rec_stream()
{
	if( halting )
		warn("halt_rec_stream:  already halting!?");
	if( !streaming )
		warn("halt_rec_stream:  not streaming!?");
	halting=1;

	/* wait for disk writer to finish - should call pthread_join! */

	while( streaming && audio_stream_fd != (-1) )
		usleep(100000);
	halting=0;
}

void halt_play_stream()
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

static int stream_play_init()
{
	/* BUG n_channels is hard-coded to 2 here, should determine from file,
	 * but currently the files don't have headers...
	 *
	 * We use the global that can be set from the menu...
	 */

	if(audio_state!=AUDIO_PLAY) audio_init(AUDIO_PLAY);

	if( init_stream_obj(nchannels) == NO_OBJ ){
		warn("stream_record_init:  error creating audio stream object");
		return(-1);
	}

	halting=0;
	streaming=1;
	active_buf=0;
	buf_to_write=(-1);
	return(0);
}

static void *audio_writer(void *arg)
{
	short *ptr;
	int n_want, n_written;

	n_want = audio_stream_dp->dt_tdim * audio_stream_dp->dt_cols * sizeof(short);
	while( !halting ){
		/* wait for a buffer to be read */
		while( buf_to_write==(-1) )
			usleep(1000);	/* 1 msec */
		ptr = audio_stream_dp->dt_data;
		ptr += buf_to_write * audio_stream_dp->dt_rowinc;

		if( (n_written=write(afd, ptr, n_want)) < 0 ){
			tell_sys_error("write");
			warn("error writing audio stream data to audio device");
		} else if( n_written != n_want ){
			sprintf(error_string,"audio_writer:  %d bytes requested, %d actually written",
					n_want,n_written);
			warn(error_string);
		}

		buf_to_write = (-1);
	}
	streaming=0;
	halting=0;
	delvec(audio_stream_dp);
	return(NULL);
}

static void *disk_reader(void *arg)
{
	short *ptr;
	int n_want, n_read;

	n_want = audio_stream_dp->dt_tdim * audio_stream_dp->dt_cols * sizeof(short);

	while( !halting ){
		ptr = audio_stream_dp->dt_data;
		ptr += active_buf * audio_stream_dp->dt_rowinc;

		if( (n_read=read(audio_stream_fd,ptr,n_want)) < 0 ){
			tell_sys_error("read");
			warn("error reading audio stream file");
		} else if( n_read == 0 ){
			/* EOF */
			halting=1;
			/* BUG should zero the buffer or goto */
		} else if( n_read != n_want ){
			sprintf(error_string,"disk_reader:  %d audio bytes requested, %d actually read",
					n_want,n_read);
			warn(error_string);
		}

		if( !halting ){
			while( buf_to_write != (-1) )		/* wait for playback of last section to finish */
				usleep(10000);			/* 10 msec */

			buf_to_write = active_buf;
			active_buf++;
			if( active_buf >= N_BUFFERS )
				active_buf = 0;
		}
	}
	close(audio_stream_fd);
	audio_stream_fd=(-1);

	return(NULL);
}

void play_stream(int fd)
{
	pthread_attr_t attr1;

	audio_stream_fd = fd;

	if( stream_play_init() < 0 ){
		warn("error initializing stream recorder");
		return;
	}

	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);
	pthread_create(&audio_thr,&attr1,audio_writer,NULL);
	pthread_create(&writer_thr,&attr1,disk_reader,NULL);
}


#endif /* USE_OSS_SOUND */

