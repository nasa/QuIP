
noinst_LIBRARIES = libvectree.a

libvectree_a_CPPFLAGS = -I$(top_srcdir)/include -Wall
# -D__STRICT_ANSI__  this breaks u_long in sys/types.h...

libvectree_a_SOURCES =			\
			vt_menu.c	\
			vectree.y	\
			vectree.c	\
			vecnodes.c	\
			evaltree.c	\
			resolve.c	\
			comptree.c	\
			costtree.c	\
			dumptree.c	\
			opt_tree.c	\
			nodetbl.c	\
			subrt.c		\
			mseq.c		\
			vt_native.c	\
			yacc_hack.h	\
			vt_native.h

noinst_HEADERS = vectree.h vt_native.h yacc_hack.h

AM_YFLAGS=-v

