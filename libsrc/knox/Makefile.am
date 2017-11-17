all:	libknox.a
	cp libknox.a ../../libs

noinst_LIBRARIES = libknox.a

libknox_a_CPPFLAGS = -I$(top_srcdir)/include -D_GNU_SOURCE -Wall -Wmissing-prototypes

libknox_a_SOURCES = 					\
				knox.c			\
				knox_tbl.c

