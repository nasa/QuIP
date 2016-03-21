
all:	libhips.a
	cp libhips.a ../../libs

noinst_LIBRARIES = libhips.a

# GNU_SOURCE is needed to elim warnings about round?
libhips_a_CPPFLAGS = -D_GNU_SOURCE -I$(top_srcdir)/include -Wall -Wmissing-prototypes

# jbm removed herrs.c
#			herrs.c

libhips_a_SOURCES =			\
			hsizepix.c	\
			perr.c		\
			rdoldhdr.c	\
			readhdr.c	\
			writehdr.c	\
			xparam.c



