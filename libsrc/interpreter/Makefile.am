
all:	libinterpreter.a
	cp libinterpreter.a ../../libs

noinst_LIBRARIES = libinterpreter.a

# define _GNU_SOURCE to quiet warnings about round...
#
libinterpreter_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes -D_GNU_SOURCE


libinterpreter_a_SOURCES =		\
					\
		fake_cam.c		\
		bi_menu.c		\
		typed_scalars.c		\
		class.c			\
		command.c		\
		complete.c		\
		encrypt_funcs.c		\
		features.c		\
		gcrypt.c		\
		debug.c			\
		dict.c			\
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
		nexpr.y			\
		param.c			\
		pathnm.c		\
		query.c			\
		query_funcs.c		\
		query_stack.c		\
		start_quip.c		\
		quip_main.c		\
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


noinst_HEADERS =			\
					\
		chewtext.h		\
		func_helper.h		\
		query_prot.h

win_nexpr.c:	nexpr.c
	cp nexpr.c win_nexpr.c

