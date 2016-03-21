all:	libxsupp.a
	cp libxsupp.a ../../libs

noinst_LIBRARIES = libxsupp.a

libxsupp_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

libxsupp_a_SOURCES =				\
						\
			xsync.c			\
			glxhelper.c		\
			which_display.c		\
			dpy.c			\
			event.c			\
			check_display.c		\
			lut_xlib.c		\
			view_xlib.c		\
			vbl.c

#			handle.c

noinst_HEADERS =	handle.h		\
			vgat_ioctl.h		\
			xsupp_prot.h

