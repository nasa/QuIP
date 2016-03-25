all:	libgslmenu.a
	cp libgslmenu.a ../../libs

noinst_LIBRARIES = libgslmenu.a

libgslmenu_a_CPPFLAGS = -I$(top_srcdir)/include -D_GNU_SOURCE -Wall -Wmissing-prototypes

libgslmenu_a_SOURCES = 					\
				gsl.c			\
				gslmenu.c

noinst_HEADERS =		gslprot.h

