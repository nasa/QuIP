all:	libdvmenu.a
	cp libdvmenu.a ../../libs

noinst_LIBRARIES = libdvmenu.a

libdvmenu_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

libdvmenu_a_SOURCES = dvmenu.c frame.c ieee1394io.c

noinst_HEADERS = dv_globals.h frame.h

