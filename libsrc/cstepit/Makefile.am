all:	libcstepit.a
	cp libcstepit.a ../../libs

noinst_LIBRARIES = libcstepit.a

libcstepit_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

# fitsine not used now?
#			fitsine.c	\
#

libcstepit_a_SOURCES = 			\
			stepit.c	\
			stepmenu.c	\
			stepsupp.c	\
			pkg.c		\
			cs_supp.c	\
			am_supp.c	\
			pr_supp.c	\
			sparse.c

noinst_HEADERS = cstepit.h sparse.h

