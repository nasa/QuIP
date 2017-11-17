#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_TIME_H
#include <time.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* usleep */
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>		/* open() */
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>		/* open() */
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>		/* struct timeval? */
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>		/* open() */
#endif

#include "quip_prot.h"


//#ifndef BUILD_FOR_IOS

/* #include "warmprot.h" */

#include "sound.h"

typedef struct {
	Data_Obj *ra_dp;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *ra_qsp;
#endif
} Recorder_Args;


#include <pthread.h>

debug_flag_t sound_debug=0;


int object_is_sound(QSP_ARG_DECL  Data_Obj *dp)
{
	if( OBJ_MACH_PREC(dp) != PREC_IN ){
		sprintf(ERROR_STRING,"Object %s has %s precision, should be %s for sound",
			OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)),NAME_FOR_PREC_CODE(PREC_IN));
		WARN(ERROR_STRING);
		return(0);
	}
	return(1);
}

static COMMAND_FUNC( do_inputgain );

static COMMAND_FUNC( do_close )
{
	audio_init(QSP_ARG  AUDIO_UNINITED);
}

static COMMAND_FUNC( do_stereo_input )
{
	int is_stereo;

	is_stereo = ASKIF("stereo input");
	set_stereo_input(QSP_ARG  is_stereo);
}

static COMMAND_FUNC( do_stereo_output )
{
	int is_stereo;

	is_stereo = ASKIF("stereo output");
	set_stereo_output(QSP_ARG  is_stereo);
}

static COMMAND_FUNC( do_set_nchan )
{
	int channels;

	channels = (int)HOW_MANY("number of output channels (1=MONO 2=STEREO 4=QUAD)");

	set_playback_nchan(QSP_ARG  channels);
}

static int async_record=0;
static int recording_in_progress=0;

static pthread_t rec_thr;

static COMMAND_FUNC( do_setasync )
{
	async_record = ASKIF("record sounds asynchronously");
}

static void *sound_recorder(void *argp)
{
	Data_Obj *dp;
	Recorder_Args *rap;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *qsp;
#endif

	rap=argp;

	dp=rap->ra_dp;
#ifdef THREAD_SAFE_QUERY
	qsp=rap->ra_qsp;
#endif

	dp=(Data_Obj *)argp;
	sprintf(ERROR_STRING,"sound_recorder:  sound is %s",OBJ_NAME(dp));
	advise(ERROR_STRING);
	recording_in_progress=1;
	record_sound(QSP_ARG  dp);
	recording_in_progress=0;
	return(NULL);
}

static COMMAND_FUNC( do_recsound )
{
	Data_Obj *dp;
	Recorder_Args ra;

	/*
	sprintf(ERROR_STRING,"audio_state=%d",audio_state);
	advise(ERROR_STRING);
	*/

	dp = pick_obj("sound object");
	if( dp == NULL ) return;
	ra.ra_dp = dp;
#ifdef THREAD_SAFE_QUERY
	ra.ra_qsp = qsp;
#endif

	if( async_record ){
		/* fork a thread to do the recording */
		pthread_attr_t attr1;
		pthread_attr_init(&attr1);	/* initialize to default values */
		pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);
		pthread_create(&rec_thr,&attr1,sound_recorder,&ra);
	} else {
		/* synchronous record */
advise("beginning synchronous record");
		record_sound(QSP_ARG  dp);
	}
	//record_sound(QSP_ARG  dp);
}

static COMMAND_FUNC( do_waitrec )
{
	if( ! async_record ){
		WARN("waitrec:  not in asynchronous sound record mode");
		return;
	}
	while(recording_in_progress) usleep(1000);
}

static COMMAND_FUNC( do_playsound )
{
	Data_Obj *dp;

	dp = pick_obj("sound object");
	if( dp == NULL ) return;

	play_sound(QSP_ARG  dp);
}

static COMMAND_FUNC( do_inputgain )
{
	int g;

	g = (int)HOW_MANY("input gain (0-100)");

	g = g < 0 ? 0 : g;
	g = g > 100 ? 100 : g;

	g = g * 256 + g;
	set_sound_gain(QSP_ARG  g);
}

static COMMAND_FUNC( do_volume )
{
	int g;

	g = (int)HOW_MANY("playback volume (0-100)");

	g = g * 256 + g;
	set_sound_volume(QSP_ARG  g);
}

static COMMAND_FUNC( do_set_alert )
{
	Data_Obj *dp;

	dp = pick_obj("warning sound");
	if( dp == NULL ) return;

	use_audio_warning(QSP_ARG  dp);
}

#define MIN_SAMP_FREQ	4000

static COMMAND_FUNC( do_set_samp_freq )
{
	int rate;

	rate = (int)HOW_MANY("sample frequency");
	if( rate <= 0 ){
		sprintf(ERROR_STRING,"sample frequency must be greater than or equal to %d",MIN_SAMP_FREQ);
		WARN(ERROR_STRING);
		return;
	}
	set_samp_freq(QSP_ARG  rate);
}

static COMMAND_FUNC( do_sound_info )
{
	Data_Obj *dp;
	u_long usec;
	long sec;
	char *s;

	dp=pick_obj("sound object");
	if( dp == NULL ) return;

	if( OBJ_MACH_PREC(dp) != PREC_IN ){
		sprintf(ERROR_STRING,"sound_info:  object %s has %s precision, expect %s for sounds",
			OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)),NAME_FOR_PREC_CODE(PREC_IN) );
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_ROWS(dp) > 1 || OBJ_FRAMES(dp) > 1 || OBJ_SEQS(dp) > 1 ){
		sprintf(ERROR_STRING,"sound_info:  object %s is not a row vector!?",OBJ_NAME(dp));
		WARN(ERROR_STRING);
	}
	if( OBJ_N_TYPE_ELTS(dp) < N_TIMESTAMP_WORDS ){
		sprintf(ERROR_STRING,"sound_info:  object %s has too few elements to contain a timestamp",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}

	sec = (long) get_sound_seconds(QSP_ARG  (Item *)dp,0);
	usec = (u_long) get_sound_microseconds(QSP_ARG  (Item *)dp,0);

	s=ctime(&sec);
	s[strlen(s)-1]=0;	/* get rid of final newline */
	sprintf(msg_str,"%s:  %s, %ld microseconds",OBJ_NAME(dp),s,usec);
	prt_msg(msg_str);
}

#define N_INPUTS	2
static const char *input_names[N_INPUTS]={"microphone","line"};

static COMMAND_FUNC( do_insel )
{
	int n;

	n=WHICH_ONE("input channel",N_INPUTS,input_names);
	if( n < 0 ) return;

	if( n == 0 )
		select_mic_input(SINGLE_QSP_ARG);
	else
		select_line_input(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_recstream )
{
	int sound_fd, timestamp_fd;
	const char *s;

	s=NAMEOF("audio stream file");
	sound_fd=open(s,O_WRONLY|O_CREAT|O_TRUNC,0644);

	s=NAMEOF("audio timestamp stream file");
	timestamp_fd=open(s,O_WRONLY|O_CREAT|O_TRUNC,0644);

	if( sound_fd < 0 ){
		sprintf(ERROR_STRING,"open(%s)",s);
		tell_sys_error(ERROR_STRING);
		WARN("error opening audio stream file");
		return;
	}

	if( timestamp_fd < 0 ){
		sprintf(ERROR_STRING,"open(%s)",s);
		tell_sys_error(ERROR_STRING);
		WARN("error opening audio timestamp stream file");
		return;
	}

	record_stream(QSP_ARG  sound_fd,timestamp_fd);
}

static COMMAND_FUNC( do_pb_stream )
{
	int fd;
	const char *s;

	s=NAMEOF("audio stream file");
	fd=open(s,O_RDONLY);
	if( fd < 0 ){
		sprintf(ERROR_STRING,"open(%s)",s);
		tell_sys_error(ERROR_STRING);
		WARN("error opening audio stream file");
		return;
	}

	play_stream(QSP_ARG  fd);
}

static COMMAND_FUNC( do_show_stamps )
{
	FILE *fp;
	const char *fn;
	struct timeval tv;

	fn=NAMEOF("timestamp filename");
	fp=try_open(fn,"r");
	if( !fp ) return;

	while( fread(&tv,sizeof(tv),1,fp) == 1 ){
		char *s;
		s=ctime(&tv.tv_sec);
		if( s[strlen(s)-1] == '\n' )
			s[strlen(s)-1] = 0;
		sprintf(msg_str,"%s\t\t%ld",s,(long)tv.tv_usec/1000);
		prt_msg(msg_str);
	}
	fclose(fp);
}


static COMMAND_FUNC(do_halt_rec_stream){halt_rec_stream(SINGLE_QSP_ARG);}

#define ADD_CMD(s,f,h)	ADD_COMMAND(audio_record_menu,s,f,h)

MENU_BEGIN(audio_record)
ADD_CMD( select_input,	do_insel,		select input channel )
ADD_CMD( input_gain,	do_inputgain,		set input gain level )
ADD_CMD( stereo,	do_stereo_input,	select stereo/mono input )
ADD_CMD( stream,	do_recstream,		record audio stream to disk )
ADD_CMD( halt,		do_halt_rec_stream,	halt stream recording )
ADD_CMD( async,		do_setasync,		enable/disable asynchronous recording )
ADD_CMD( record,	do_recsound,		record a sound clip )
ADD_CMD( wait,		do_waitrec,		wait for async record to finish )
MENU_END(audio_record)

static COMMAND_FUNC(do_pause_sound){pause_sound(SINGLE_QSP_ARG);}
static COMMAND_FUNC(do_halt_play_stream){halt_play_stream(SINGLE_QSP_ARG);}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(audio_playback_menu,s,f,h)

MENU_BEGIN(audio_playback)
ADD_CMD( info,		do_sound_info,		display timestamp )
ADD_CMD( play,		do_playsound,		play a sound )
ADD_CMD( pause,		do_pause_sound,		pause a sound )
ADD_CMD( stream,	do_pb_stream,		stream audio from disk )
ADD_CMD( halt,		do_halt_play_stream,	halt stream playback )
ADD_CMD( nchannels,	do_set_nchan,		set number of output channels )
ADD_CMD( stereo,	do_stereo_output,	select stereo/mono output )
ADD_CMD( volume,	do_volume,		set playback volume )
ADD_CMD( alert_sound,	do_set_alert,		set warning sound )
ADD_CMD( timestamps,	do_show_stamps,		display recorded timestamps )
MENU_END(audio_playback)

static COMMAND_FUNC( do_rec_menu )
{
	CHECK_AND_PUSH_MENU(audio_record);
}

static COMMAND_FUNC( do_pb_menu )
{
	CHECK_AND_PUSH_MENU(audio_playback);
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(sound_menu,s,f,h)

MENU_BEGIN(sound)
ADD_CMD( acquire,	do_rec_menu,		sound recording submenu )
ADD_CMD( playback,	do_pb_menu,		sound playback submenu )
ADD_CMD( sample_freq,	do_set_samp_freq,	set sample frequency )
ADD_CMD( close,		do_close,		close audio device )
MENU_END(sound)

//#endif // ! BUILD_FOR_IOS

COMMAND_FUNC( do_sound_menu )
{
//#ifdef BUILD_FOR_IOS
//    WARN("Sorry, no sound support for iOS yet...");
//#else // ! BUILD_FOR_IOS
	static int sound_inited=0;
    
	if( ! sound_inited ){
#ifdef DEBUG
		if( sound_debug == 0 )
			sound_debug = add_debug_module("sound");
#endif /* DEBUG */
		sound_inited=1;
	}
    
	CHECK_AND_PUSH_MENU(sound);	/* load the initial menu */
//#endif // ! BUILD_FOR_IOS
}

