/* we might like to give audio warnings... */

#include "quip_config.h"
char VersionId_sound_alert[] = QUIP_VERSION_STRING;

#include "data_obj.h"
#include "sound.h"

static Data_Obj *warning_dp=NO_OBJ;

void audio_warn(QSP_ARG_DECL  const char *s)
{
	tty_warn(QSP_ARG  s);
#ifdef HAVE_SOUND
	if( warning_dp != NO_OBJ )
		play_sound(QSP_ARG  warning_dp);
#else
	WARN("audio_warn:  no sound support in this configuration");
#endif
}

void use_audio_warning(QSP_ARG_DECL  Data_Obj *dp)
{
	set_warn_func(audio_warn);
	warning_dp = dp;
}


