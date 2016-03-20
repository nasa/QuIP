all:	libdas1602.a
	cp libdas1602.a ../../libs

noinst_LIBRARIES = libdas1602.a

libdas1602_a_CPPFLAGS = -I$(top_srcdir)/include -D_GNU_SOURCE -Wall -Wmissing-prototypes

libdas1602_a_SOURCES = 					\
				dastst.c		\
				pacer.c

noinst_HEADERS =		ioctl_das1602.h		\
				pacer.h

