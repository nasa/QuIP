all:	libspink.a
	cp libspink.a ../../libs
noinst_LIBRARIES = libspink.a

libspink_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

libspink_a_SOURCES = spink_menu.c stream_spink.c spink.c

noinst_HEADERS = spink.h

test:	spink_enum.o
	$(CC) -o test spink_enum.o -lSpinnaker_C

