/* we might like to give audio warnings... */

#include "quip_config.h"

//#ifdef HAVE_SOUND

#include "quip_prot.h"
#include "data_obj.h"
#include "sound.h"

static Data_Obj *warning_dp=NULL;

static void audio_warn(QSP_ARG_DECL  const char *s)
{
	tty_warn(QSP_ARG  s);
	if( warning_dp != NULL )
		play_sound(warning_dp);
}

void use_audio_warning(QSP_ARG_DECL  Data_Obj *dp)
{
	set_warn_func(audio_warn);
	warning_dp = dp;
}

//#endif /* HAVE_SOUND */

