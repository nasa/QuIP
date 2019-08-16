all:	libpsych.a
	cp libpsych.a ../../libs

noinst_LIBRARIES = libpsych.a

libpsych_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

libpsych_a_SOURCES = 				\
						\
		xvalmenu.c			\
		stair.c				\
		asc_data.c	lookmenu.c	\
		mlfit.c		lump.c		\
		weibull.c			\
		chngp_menu.c			\
		class_menu.c			\
		stc_menu.c			\
		exp_menu.c			\
		stc_util.c	exp.c		\
		errbars.c	stc_edit.c

noinst_HEADERS = stc.h
