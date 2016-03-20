all:	libmeteor.a
	cp libmeteor.a ../../libs

noinst_LIBRARIES = libmeteor.a

libmeteor_a_CPPFLAGS = -D_GNU_SOURCE -I$(top_srcdir)/include -Wall -Wmissing-prototypes

libmeteor_a_SOURCES = 					\
				mmenu.c			\
				mcont.c			\
				mmodule.c		\
				mviewer.c		\
				pupfind.c		\
				mcapt.c			\
				meteor_flow.c		\
				stream.c		\
				mgeo.c			\
				fg_routines.c		\
				mhw.c

noinst_HEADERS =		pupfind.h		\
				curv_setup1.h		\
				curv_setup2.h		\
				curv_setup3.h		\
				fg_routines.h		\
				ioctl_himemfb.h		\
				ioctl_meteor.h		\
				memchunk.h		\
				meteor.h		\
				mmenu.h			\
				mmvi.h

