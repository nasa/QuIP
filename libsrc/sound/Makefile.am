all:	libsound.a
	cp libsound.a ../../libs

noinst_LIBRARIES = libsound.a

libsound_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

libsound_a_SOURCES =  soundmenu.c alert.c alsa.c alsa_input.c no_sound.c pa_sound.c pa_input.c

noinst_HEADERS =	sound.h


#ALSA_OFILES=	alsa.o alsa_input.o
#OSS_OFILES=	oss_sound.o
#OFILES=	soundmenu.o alert.o $(ALSA_OFILES) $(OSS_OFILES)
