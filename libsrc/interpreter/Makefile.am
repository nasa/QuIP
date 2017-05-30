
all:	libinterpreter.a
	cp libinterpreter.a ../../libs

noinst_LIBRARIES = libinterpreter.a

# define _GNU_SOURCE to quiet warnings about round...
#
libinterpreter_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes -D_GNU_SOURCE

#		dict.c

libinterpreter_a_SOURCES =		\
					\
		fake_cam.c		\
		start_quip.c		\
		bi_menu.c		\
		container.c		\
		rbtree.c		\
		class.c			\
		command.c		\
		complete.c		\
		encrypt_funcs.c		\
		features.c		\
		gcrypt.c		\
		debug.c			\
		error.c			\
		fileck.c		\
		freel.c			\
		function.c		\
		getbuf.c		\
		hash.c			\
		history.c		\
		item_type.c		\
		list.c			\
		macro.c			\
		menu.c			\
		param.c			\
		pathnm.c		\
		query.c			\
		pipe_support.c		\
		questions.c		\
		nexpr.y			\
		ptoz.c			\
		query_funcs.c		\
		quip_exec.c		\
		callback.c		\
		query_stack.c		\
		typed_scalars.c		\
		rcfile.c		\
		rn.c			\
		stack.c			\
		strbuf.c		\
		substr.c		\
		expect.c		\
		termio.c		\
		tryhard.c		\
		variable.c		\
		which1.c

#		expr_node.h

# moved to ../../include
#		func_helper.h

noinst_HEADERS =			\
					\
		strbuf.h		\
		query_private.h		\
		query_prot.h

win_nexpr.c:	nexpr.c
	cp nexpr.c win_nexpr.c

