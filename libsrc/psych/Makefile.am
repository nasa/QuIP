all:	libpsych.a
	cp libpsych.a ../../libs

noinst_LIBRARIES = libpsych.a

libpsych_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

libpsych_a_SOURCES = 				\
						\
		xvalmenu.c			\
		stc_menu.c	exp.c		\
		stair.c				\
		asc_data.c	lookmenu.c	\
		mlfit.c		lump.c		\
		weibull.c			\
		errbars.c	stc_edit.c	\
		clrdat.c

noinst_HEADERS = stc.h
