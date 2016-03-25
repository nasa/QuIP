
all:	libvl2.a
	cp libvl2.a ../../libs

noinst_LIBRARIES = libvl2.a

libvl2_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

#libmvimenu_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

#			vl2_func_tbl.c

libvl2_a_SOURCES =			\
			vl2_veclib.c	\
			vl2_utils.c	\
			vl2.c		\
			slow_len.c	\
			fftsupp.c	\
			vl2_platform.c

#			linear_funcs.c
#			lin_util.c
#			vl2_streams.c
#			vl2_viewer.c
#			vl2_func_tbl.c

#noinst_HEADERS = my_vl2.h

vl2_veclib.debug:	vl2_veclib.c
	gcc -E -I../.. -I$(top_srcdir)/include vl2_veclib.c > vl2_veclib.debug_
	indent < vl2_veclib.debug_  > vl2_veclib.debug
#./strip_it2.sh

#vl2.debug:	vl2.c
#	gcc -E -I../.. -I$(top_srcdir)/include vl2.c > vl2.debug_

#			vl2_func_tbl.h

#			vl2_host_call_defs.h

noinst_HEADERS =					\
			linear.c			\
			vfft.c				\
			vl2_kernels.c			\
			vl2_typtbl.c			\
			cpu_call_defs.h			\
			vl2_host_funcs.c		\
			vl2_host_call_defs.h		\
			vl2_host_untyped_call_defs.h


#clean-local:
#	-rm -rf vl2_veclib_expanded.c vl2_veclib_stripped.c

