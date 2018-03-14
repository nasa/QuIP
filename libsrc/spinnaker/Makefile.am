all:	libspink.a
	cp libspink.a ../../libs
noinst_LIBRARIES = libspink.a

libspink_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

libspink_a_SOURCES = spink_menu.c stream_spink.c spink_util.c spink_enum.c spink_cam_ctl.c spink_node_map.c spink_acq.c spink_funcs.c

noinst_HEADERS = spink.h

test:	spink_enum.o spink_test.o
	$(CC) -o test spink_test.o spink_enum.o -lSpinnaker_C

