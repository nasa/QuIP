
#include "quip_config.h"

#ifdef HAVE_APPLE_CORE_AUDIO

#include "quip_prot.h"
#include "sound.h"

int set_playback_nchan(QSP_ARG_DECL  int channels)
{
	return 0;
}

void play_sound(QSP_ARG_DECL  Data_Obj *dp)
{}

void set_sound_volume(QSP_ARG_DECL  int g)
{}

void set_samp_freq(QSP_ARG_DECL  unsigned int req_rate)
{}

void audio_init(QSP_ARG_DECL  int mode)
{}

void halt_play_stream(SINGLE_QSP_ARG_DECL)
{}

void play_stream(QSP_ARG_DECL  int fd) {}
void set_stereo_output(QSP_ARG_DECL  int is_stereo) {}
void pause_sound(SINGLE_QSP_ARG_DECL) {}

void set_sound_gain(QSP_ARG_DECL  int g)
{}

void select_mic_input(SINGLE_QSP_ARG_DECL)
{}

void select_line_input(SINGLE_QSP_ARG_DECL)
{}


double get_sound_seconds(QSP_ARG_DECL  Item *ip,dimension_t frame)
{
	return 0.0;
}

double get_sound_microseconds(QSP_ARG_DECL  Item *ip,dimension_t frame)
{
	return 0.0;
}

double get_sound_milliseconds(QSP_ARG_DECL  Item *ip,dimension_t frame)
{
	return 0.0;
}

void halt_rec_stream(SINGLE_QSP_ARG_DECL)
{}

void set_stereo_input(QSP_ARG_DECL  int is_stereo) {}

void _record_sound(QSP_ARG_DECL  Data_Obj *dp)
{}

void record_stream(QSP_ARG_DECL  int sound_fd, int timestamp_fd)
{}

int _sound_seek(QSP_ARG_DECL  index_t idx)
{
    return 0;
}

int _async_play_sound(QSP_ARG_DECL  Data_Obj *dp)
{
    return 0;
}

#endif // HAVE_APPLE_CORE_AUDIO

