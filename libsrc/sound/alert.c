/* we might like to give audio warnings... */

#include "quip_config.h"

//#ifdef HAVE_SOUND

#include "quip_prot.h"
#include "data_obj.h"
#include "sound.h"

static Data_Obj *warning_dp=NO_OBJ;

static void audio_warn(QSP_ARG_DECL  const char *s)
{
	tty_warn(QSP_ARG  s);
	if( warning_dp != NO_OBJ )
		play_sound(QSP_ARG  warning_dp);
}

void use_audio_warning(QSP_ARG_DECL  Data_Obj *dp)
{
	set_warn_func(audio_warn);
	warning_dp = dp;
}

//#endif /* HAVE_SOUND */

