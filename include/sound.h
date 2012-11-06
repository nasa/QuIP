
#include "quip_config.h"

#ifdef HAVE_SOUND

/* BUG?  is this OSS only??? */
#ifdef HAVE_SYS_SOUNDCARD_H
#include <sys/soundcard.h>
#endif

#include "data_obj.h"
#include "query.h"

extern debug_flag_t sound_debug;

/* This is the linux sound interface... should move into alsa stuff? */
#define DEVICE_NAME "/dev/dsp"
#define MIXER_NAME "/dev/mixer"

#define AUDIO_UNINITED	0
#define AUDIO_PLAY	1
#define AUDIO_RECORD	2

#define MONO   1
#define STEREO 2
#define QUAD   4

extern int audio_state;
#define CHECK_AUDIO(state)	if( audio_state == AUDIO_UNINITED ) audio_init(QSP_ARG  state);	

typedef struct timestamp_data {
	u_short	tsd_csec;	/* centiseconds */
	u_short	tsd_ccsec;	/* centicentiseconds */
	u_short	tsd_usec;	/* microseconds */
	/* from broken-down time */
	u_short	tsd_sec;
	u_short	tsd_min;
	u_short	tsd_hour;
	u_short	tsd_mday;
	u_short	tsd_mon;
	u_short	tsd_year;
} Timestamp_Data;

#define N_TIMESTAMP_WORDS	(sizeof(Timestamp_Data)/sizeof(u_short))

/* soundmenu.c */
extern int object_is_sound(QSP_ARG_DECL  Data_Obj *dp);

/* sound.c */

extern void halt_rec_stream(SINGLE_QSP_ARG_DECL);
extern void halt_play_stream(SINGLE_QSP_ARG_DECL);
extern void record_stream(QSP_ARG_DECL  int sound_fd, int time_fd);
extern void play_stream(QSP_ARG_DECL  int fd);
extern int set_playback_nchan(QSP_ARG_DECL  int);
extern void audio_init(QSP_ARG_DECL  int);
extern void set_stereo_output(QSP_ARG_DECL  int);
extern void set_stereo_input(QSP_ARG_DECL  int);
extern void record_sound(QSP_ARG_DECL  Data_Obj *);
extern void play_sound(QSP_ARG_DECL  Data_Obj *);
extern void set_sound_gain(QSP_ARG_DECL  int);
extern void set_sound_volume(QSP_ARG_DECL  int);
extern void pause_sound(SINGLE_QSP_ARG_DECL);
extern void set_samp_freq(QSP_ARG_DECL  unsigned int);
extern double get_sound_seconds(Item *dp,dimension_t frame);
extern double get_sound_milliseconds(Item *dp,dimension_t frame);
extern double get_sound_microseconds(Item *dp,dimension_t frame);
extern void select_mic_input(SINGLE_QSP_ARG_DECL);
extern void select_line_input(SINGLE_QSP_ARG_DECL);


/* alert.c */

extern void use_audio_warning(QSP_ARG_DECL  Data_Obj *);

#endif /* HAVE_SOUND */
