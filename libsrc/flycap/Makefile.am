all:	libfly.a
	cp libfly.a ../../libs
noinst_LIBRARIES = libfly.a

libfly_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

libfly_a_SOURCES = fly_menu.c stream_fly.c fly.c cam_ctl.c

noinst_HEADERS = fly.h

