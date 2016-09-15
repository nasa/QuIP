SUBDIRS =	Random123

all:	libocl.a
	cp libocl.a ../../libs

noinst_LIBRARIES = libocl.a

libocl_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

# BUG this is mac only...
#AM_CFLAGS= -framework OpenGL
AM_CFLAGS= -I/System/Library/Frameworks/OpenGL.framework/Headers

#libmvimenu_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

#	ocl_viewer.c

libocl_a_SOURCES =			\
			ocl_fft.c	\
			ocl_streams.c	\
			ocl_menu.c	\
			ocl_rand.c	\
			ocl_utils.c	\
			ocl_veclib.c	\
			ocl.c

ocl_rand.c:	vuni.i

vuni.i:	$(srcdir)/vuni.ocl
	$(srcdir)/gencl.sh $(srcdir)/vuni.ocl vuni.i

#			ocl_func_tbl.c

#noinst_HEADERS = my_ocl.h

#ocl_veclib_expanded.c:	_ocl_veclib.c
#	gcc -E -I../.. -I$(top_srcdir)/include `test -f _ocl_veclib.c || echo $(srcdir)/`_ocl_veclib.c > ocl_veclib_expanded.c || rm ocl_veclib_expanded.c
#
#ocl_veclib_stripped.c:		ocl_veclib_expanded.c
#	./strip_it.sh
#
#ocl_veclib.c:	ocl_veclib_stripped.c

#./strip_it2.sh
#			ocl_api.h

noinst_HEADERS =					\
			ocl_host_funcs.c		\
			ocl_kernels.c			\
			ocl_typtbl.c			\
			ocl_typed_fft.c			\
			gencl.sh			\
			vuni.ocl			\
			conversions.h			\
			enum_menu_calls.h		\
			host_conv_prot.h		\
			kern_src.h			\
			ocl_host_call_defs.h		\
			ocl_host_untyped_call_defs.h	\
			ocl_kern_args.h			\
			ocl_kern_call_defs.h		\
			ocl_kernel_src.h		\
			type_undefs.h			\
			undefs.h


clean-local:
	-rm -rf vuni.i

#	-rm -rf ocl_veclib_expanded.c ocl_veclib_stripped.c vuni.i

