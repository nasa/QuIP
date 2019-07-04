all:	libvectree.a
	cp libvectree.a ../../libs

noinst_LIBRARIES = libvectree.a

libvectree_a_CPPFLAGS = -I$(top_srcdir)/include -I$(top_builddir)/include -Wall -Wmissing-prototypes
# -D__STRICT_ANSI__  this breaks u_long in sys/types.h...

libvectree_a_SOURCES =			\
			vt_menu.c	\
			vectree.y	\
			vectree.c	\
			vecnodes.c	\
			evaltree.c	\
			identifier.c	\
			undef_sym.c	\
			resolve.c	\
			comptree.c	\
			costtree.c	\
			dumptree.c	\
			opt_tree.c	\
			nodetbl.c	\
			subrt.c		\
			subrt_funcs.c	\
			mseq.c		\
			vt_native.c	\
			string_ref.c	\
			fusion.c	\
			vector_parser_data.c	\
			vt_native.h

noinst_HEADERS =	subrt.h

#noinst_HEADERS = vectree.h vt_native.h \
#			ctx_pair.h	\
#			identifier.h	\
#			pointer.h	\
#			reference.h	\
#			subrt.h		\
#			undef_sym.h	\
#			vec_expr_node.h

AM_YFLAGS=-v

