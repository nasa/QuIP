all:	libocvq.a
	cp libocvq.a ../../libs

noinst_LIBRARIES = libocvq.a

libocvq_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

libocvq_a_SOURCES = opencv_menu.c opencv_glue.c

noinst_HEADERS = opencv_glue.h

