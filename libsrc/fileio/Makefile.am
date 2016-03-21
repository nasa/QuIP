all:	libfileio.a
	cp libfileio.a ../../libs

noinst_LIBRARIES = libfileio.a

# GNU_SOURCE is needed to elim warnings about round?
libfileio_a_CPPFLAGS = -D_GNU_SOURCE -I$(top_srcdir)/include -Wall -Wmissing-prototypes

#			viff.c		\
#			sunras.c	\
#			vl.c		\
#			vista.c		\
#			qt.c		\
#			bmp.c		\
#			lumisys.c	\
#			bdf.c		\
#			dsk.c		\
#			rgb.c		\
#			filetype.c	\
#

libfileio_a_SOURCES = 			\
			get_hdr.c	\
			img_file.c	\
			tiff.c		\
			ppm.c		\
			bmp.c		\
			rv.c		\
			hips1.c		\
			hips2.c		\
			wav.c		\
			png.c		\
			jpeg.c		\
			raw.c		\
			fileport.c	\
			read_raw.c	\
			fio_menu.c	\
			copts.c		\
			jp_opts.c	\
			rdcolmap.c	\
			png_menu.c	\
			mpeg.c		\
			fioasc.c	\
			matio.c		\
			avi.c		\
			avi_info_file.c

#			avcodec_hdr.h		\
#			bdf.h			\
#			bdf_hdr.h		\
#			bmp.h			\
#			bmp_hdr.h		\
#

noinst_HEADERS =				\
			cderror.h		\
			cdjpeg.h		\
			fio_prot.h		\
			fiojpeg.h		\
			jinclude.h		\
			jpeg_private.h		\
			jpegint.h		\
			markers.h		\
			bmp.h			\
			bmp_hdr.h		\
			matio_private.h		\
			my_avi.h

