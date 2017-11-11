all:	libdobj.a
	cp libdobj.a ../../libs

noinst_LIBRARIES = libdobj.a

libdobj_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes
# -D__STRICT_ANSI__  this breaks u_long in sys/types.h...

#			platform.c
#			pf_viewer.c

libdobj_a_SOURCES =				\
			areas.c			\
			arrays.c		\
			ascii.c			\
			contig.c		\
			data_fns.c		\
			data_obj.c		\
			dfuncs.c		\
			dobj_util.c		\
			dplist.c		\
			get_obj.c		\
			index.c			\
			makedobj.c		\
			memops.c		\
			shape_info.c		\
			dobj_menu.c		\
			ascmenu.c		\
			ops_menu.c		\
			dobj_expr.c		\
			sub_obj.c

noinst_HEADERS =				\
			dobj_private.h

