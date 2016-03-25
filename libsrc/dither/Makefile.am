all:	libdither.a
	cp libdither.a ../../libs

noinst_LIBRARIES = libdither.a

libdither_a_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/include -D_GNU_SOURCE -Wall -Wmissing-prototypes

libdither_a_SOURCES = 					\
				cdiff.c			\
				cspread.c		\
				ctone.c			\
				dspread.c		\
				getbest.c		\
				getwhite.c		\
				gspread.c		\
				rb2rgb.c		\
				requant.c		\
				spread.c		\
				tspread.c


noinst_HEADERS =					\
				cie.h			\
				ctone.h			\
				phosmax.h		\
				requant.h		\
				qlevel.h



