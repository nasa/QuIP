
all:	librawvol.a
	cp librawvol.a ../../libs

noinst_LIBRARIES = librawvol.a

librawvol_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes -D_LARGEFILE64_SOURCE

librawvol_a_SOURCES = rvmenu.c rawvol.c llseek.c getsize.c rawvol_player.c

#noinst_HEADERS = llseek.h rawvol.h off64_t.h
noinst_HEADERS = rawvol.h

