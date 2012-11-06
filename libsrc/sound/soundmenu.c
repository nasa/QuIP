#include "quip_config.h"

char VersionId_sound_soundmenu[] = QUIP_VERSION_STRING;


#ifdef HAVE_SOUND

/* we reference these symbols to force the files to load even when they're not needed,
 * so as to keep our version updating scripts from being confused - they might think
 * the extra file represented a version difference...
 */

extern char *VersionId_sound_alsa;
extern char *VersionId_sound_alsa_input;
/* extern char *VersionId_sound_oss_sound; */

/* This used to be local to force_load_sound_version_strings, but on durer (alsa), we
 * got no undef version id strings, even though the same things seemed to work on oss sound
 * machines, running the same compiler version???
 * Making the string pointer a global fixed it only for the last one - in other words,
 * the compiler was smart enough to see that the first two assignments did nothing
 * and so they were pruned.  So we have 3 dummy string pointers now!?
 */

char *dummy_string_ptr1;
char *dummy_string_ptr2;
char *dummy_string_ptr3;

/* all this does is to get the file version strings included in the executable */
void force_load_sound_version_strings()
{
	//char *s;
	dummy_string_ptr1=VersionId_sound_alsa;
	dummy_string_ptr2=VersionId_sound_alsa_input;
	/*
	dummy_string_ptr3=VersionId_sound_oss_sound;
	*/
}

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

#include "query.h"
#include "getbuf.h"
#include "submenus.h"
/* #include "warmprot.h" */
#include "debug.h"
#include "version.h"

#include "sound.h"

typedef struct {
	Data_Obj *ra_dp;
#ifdef THREAD_SAFE_QUERY
	Query_Stream *ra_qsp;
#endif
} Recorder_Args;


#include <pthread.h>

debug_flag_t sound_debug=0;


int object_is_sound(QSP_ARG_DECL  Data_Obj *dp)
{
	if( MACHINE_PREC(dp) != PREC_IN ){
		sprintf(ERROR_STRING,"Object %s has %s precision, should be %s for sound",
			dp->dt_name,prec_name[MACHINE_PREC(dp)],prec_name[PREC_IN]);
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

	channels = HOW_MANY("number of output channels (1=MONO 2=STEREO 4=QUAD)");

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
	Query_Stream *qsp;
#endif

	rap=argp;

	dp=rap->ra_dp;
#ifdef THREAD_SAFE_QUERY
	qsp=rap->ra_qsp;
#endif

	dp=(Data_Obj *)argp;
	sprintf(ERROR_STRING,"sound_recorder:  sound is %s",dp->dt_name);
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

	dp = PICK_OBJ("sound object");
	if( dp == NO_OBJ ) return;
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

	dp = PICK_OBJ("sound object");
	if( dp == NO_OBJ ) return;

	play_sound(QSP_ARG  dp);
}

static COMMAND_FUNC( do_inputgain )
{
	int g;

	g = HOW_MANY("input gain (0-100)");

	g = g < 0 ? 0 : g;
	g = g > 100 ? 100 : g;

	g = g * 256 + g;
	set_sound_gain(QSP_ARG  g);
}

static COMMAND_FUNC( do_volume )
{
	int g;

	g = HOW_MANY("playback volume (0-100)");

	g = g * 256 + g;
	set_sound_volume(QSP_ARG  g);
}

static COMMAND_FUNC( do_set_alert )
{
	Data_Obj *dp;

	dp = PICK_OBJ("warning sound");
	if( dp == NO_OBJ ) return;

	use_audio_warning(QSP_ARG  dp);
}

#define MIN_SAMP_FREQ	4000

static COMMAND_FUNC( do_set_samp_freq )
{
	int rate;

	rate = HOW_MANY("sample frequency");
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

	dp=PICK_OBJ("sound object");
	if( dp == NO_OBJ ) return;

	if( MACHINE_PREC(dp) != PREC_IN ){
		sprintf(ERROR_STRING,"sound_info:  object %s has %s precision, expect %s for sounds",
			dp->dt_name,prec_name[MACHINE_PREC(dp)],prec_name[PREC_IN]);
		WARN(ERROR_STRING);
		return;
	}
	if( dp->dt_rows > 1 || dp->dt_frames > 1 || dp->dt_seqs > 1 ){
		sprintf(ERROR_STRING,"sound_info:  object %s is not a row vector!?",dp->dt_name);
		WARN(ERROR_STRING);
	}
	if( dp->dt_n_type_elts < N_TIMESTAMP_WORDS ){
		sprintf(ERROR_STRING,"sound_info:  object %s has too few elements to contain a timestamp",
			dp->dt_name);
		WARN(ERROR_STRING);
		return;
	}

	sec = get_sound_seconds((Item *)dp,0);
	usec = get_sound_microseconds((Item *)dp,0);

	s=ctime(&sec);
	s[strlen(s)-1]=0;	/* get rid of final newline */
	sprintf(msg_str,"%s:  %s, %ld microseconds",dp->dt_name,s,usec);
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
	fp=TRY_OPEN(fn,"r");
	if( !fp ) return;

	while( fread(&tv,sizeof(tv),1,fp) == 1 ){
		char *s;
		s=ctime(&tv.tv_sec);
		if( s[strlen(s)-1] == '\n' )
			s[strlen(s)-1] = 0;
		sprintf(msg_str,"%s\t\t%ld",s,tv.tv_usec/1000);
		prt_msg(msg_str);
	}
	fclose(fp);
}


static COMMAND_FUNC(do_halt_rec_stream){halt_rec_stream(SINGLE_QSP_ARG);}

static Command rec_ctbl[]={
{ "select_input",do_insel,		"select input channel"		},
{ "input_gain",	do_inputgain,		"set input gain level"		},
{ "stereo",	do_stereo_input,	"select stereo/mono input"	},
{ "stream",	do_recstream,		"record audio stream to disk"	},
{ "halt",	do_halt_rec_stream,	"halt stream recording"		},
{ "async",	do_setasync,		"enable/disable asynchronous recording"},
{ "record",	do_recsound,		"record a sound clip"		},
{ "wait",	do_waitrec,		"wait for async record to finish"},
{ "quit",	popcmd,			"exit submenu"			},
{ NULL_COMMAND								}
};

static COMMAND_FUNC(do_pause_sound){pause_sound(SINGLE_QSP_ARG);}
static COMMAND_FUNC(do_halt_play_stream){halt_play_stream(SINGLE_QSP_ARG);}

static Command pb_ctbl[]={
{ "info",	do_sound_info,	"display timestamp"			},
{ "play",	do_playsound,	"play a sound"				},
{ "pause",	do_pause_sound,	"pause a sound"				},
{ "stream",	do_pb_stream,	"stream audio from disk"		},
{ "halt",	do_halt_play_stream,"halt stream playback"		},
{ "nchannels",	do_set_nchan,	"set number of output channels"		},
{ "stereo",	do_stereo_output,"select stereo/mono output"		},
{ "volume",	do_volume,	"set playback volume"			},
{ "alert_sound",do_set_alert,	"set warning sound"			},
{ "timestamps",	do_show_stamps,	"display recorded timestamps"		},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL_COMMAND								}
};

static COMMAND_FUNC( rec_menu )
{
	PUSHCMD(rec_ctbl,"audio_record");
}

static COMMAND_FUNC( pb_menu )
{
	PUSHCMD(pb_ctbl,"audio_playback");
}

static Command sound_ctbl[]={
	{ "acquire",		rec_menu,		"sound recording submenu"	},
	{ "playback",		pb_menu,		"sound playback submenu"	},
	{ "sample_freq",	do_set_samp_freq,	"set sample frequency"		},
	{ "close",		do_close,		"close audio device"		},
	{ "quit",		popcmd,			"exit submenu"			},
	{ NULL_COMMAND									}
};


COMMAND_FUNC( soundmenu )
{
	static int sound_inited=0;

	if( ! sound_inited ){
#ifdef DEBUG
		if( sound_debug == 0 )
			sound_debug = add_debug_module(QSP_ARG  "sound");
#endif /* DEBUG */
		auto_version(QSP_ARG  "SOUND","VersionId_sound");
		sound_inited=1;
	}

	PUSHCMD(sound_ctbl,"sound");	/* load the initial menu */
}

#endif /* HAVE_SOUND */
