all:	libvisca.a
	cp libvisca.a ../../libs

viscadatadir	=	$(datadir)/quip/params/visca

noinst_LIBRARIES = libvisca.a

libvisca_a_CPPFLAGS = -I$(top_srcdir)/include -D_GNU_SOURCE -Wall -Wmissing-prototypes

libvisca_a_SOURCES = 					\
				visca.c			\
				glue.c

noinst_HEADERS =		cam_params.h visca.h

# temporarily disable
viscadata_DATA	=	zoom_coeffs.asc

EXTRA_DIST	=	$(viscadata_DATA)

