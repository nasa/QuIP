
#include "quip_config.h"

#ifndef HAVE_SOUND

#include "quip_prot.h"
#include "sound.h"

#define advise_no_sound() _advise_no_sound(SINGLE_QSP_ARG)

static void _advise_no_sound(SINGLE_QSP_ARG_DECL)
{
	static int no_sound_advised=0;

	if( no_sound_advised ) return;

	warn("Sorry, no sound support in this build.");
	no_sound_advised=1;
}

int _sound_seek(QSP_ARG_DECL  index_t idx)
{
	advise_no_sound();
	return -1;
}

int _async_play_sound(QSP_ARG_DECL  Data_Obj *dp)
{
	advise_no_sound();
	return -1;
}

#ifndef BUILD_FOR_IOS

void _play_sound(QSP_ARG_DECL  Data_Obj *dp)
{
	advise_no_sound();
}


void set_sound_volume(QSP_ARG_DECL  int g)
{
	advise_no_sound();
}

int set_playback_nchan(QSP_ARG_DECL  int channels)
{
	advise_no_sound();
	return 0;
}

void set_samp_freq(QSP_ARG_DECL  unsigned int req_rate)
{
	advise_no_sound();
}

void audio_init(QSP_ARG_DECL  int mode)
{
	advise_no_sound();
}

void set_stereo_output(QSP_ARG_DECL  int is_stereo)
{
	advise_no_sound();
}

void halt_play_stream(SINGLE_QSP_ARG_DECL)
{
	advise_no_sound();
}

void play_stream(QSP_ARG_DECL  int fd)
{
	advise_no_sound();
}

void pause_sound(SINGLE_QSP_ARG_DECL)
{
	advise_no_sound();
}

void _record_sound(QSP_ARG_DECL  Data_Obj *dp)
{
	advise_no_sound();
}

void record_stream(QSP_ARG_DECL  int sound_fd, int timestamp_fd)
{
	advise_no_sound();
}
#endif // BUILD_FOR_IOS


void set_sound_gain(QSP_ARG_DECL  int g)
{
	advise_no_sound();
}

void select_mic_input(SINGLE_QSP_ARG_DECL)
{
	advise_no_sound();
}

void select_line_input(SINGLE_QSP_ARG_DECL)
{
	advise_no_sound();
}


double get_sound_seconds(QSP_ARG_DECL  Item *ip,dimension_t frame)
{
	advise_no_sound();
	return 0.0;
}

double get_sound_microseconds(QSP_ARG_DECL  Item *ip,dimension_t frame)
{
	advise_no_sound();
	return 0.0;
}

double get_sound_milliseconds(QSP_ARG_DECL  Item *ip,dimension_t frame)
{
	advise_no_sound();
	return 0.0;
}

void halt_rec_stream(SINGLE_QSP_ARG_DECL)
{
	advise_no_sound();
}

void set_stereo_input(QSP_ARG_DECL  int is_stereo)
{
	advise_no_sound();
}

#endif // ! HAVE_SOUND

