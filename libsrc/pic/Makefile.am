
all:	libpic.a
	cp libpic.a ../../libs

noinst_LIBRARIES = libpic.a

libpic_a_CPPFLAGS = -I$(top_srcdir)/include -D_GNU_SOURCE -Wall -Wmissing-prototypes

libpic_a_SOURCES = 					\
				pic.c			\
				pic_tbl.c

