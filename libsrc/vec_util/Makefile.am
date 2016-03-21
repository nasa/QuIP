all:	libvec_util.a
	cp libvec_util.a ../../libs

noinst_LIBRARIES = libvec_util.a

libvec_util_a_CPPFLAGS = -I$(top_srcdir)/include -DNOWRAP -Wall -Wmissing-prototypes

#				bessel.c	\
#				oddeven.c	\
#				mkwheel.c	\
#				qt.c		\
#				radavg.c	\
#				legendre.c	\
#				quads.c		\
#

#				thinz.c		\
#

libvec_util_a_SOURCES =				\
				dptone.c	\
				resample.c	\
				convolve.c	\
				conv3d.c	\
				dither.c	\
				dpinvert.c	\
				sample.c	\
				median.c	\
				morph.c		\
				dct8.c		\
				SeedFill.c	\
				lutmap.c	\
				vinterp.c	\
				krast.c		\
				cumsum.c	\
				scramble.c	\
				yuv2rgb.c	\
				hough.c		\
				local_max.c	\
				integral.c	\
				histo.c		\
				graph_path.c	\
				scale.c		\
				size.c		\
				wrap.c		\
				gj.c

#noinst_HEADERS = quadtree.h ggem.h polynm.h
noinst_HEADERS = ggem.h

