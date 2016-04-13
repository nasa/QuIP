
all:	libunix_util.a
	cp libunix_util.a ../../libs

noinst_LIBRARIES = libunix_util.a

# define _GNU_SOURCE to quiet warnings about round...
#
libunix_util_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes -D_GNU_SOURCE

libunix_util_a_SOURCES =		\
					\
		fork.c			\
		mouse.c			\
		my_stty.c		\
		serial.c		\
		stamps.c		\
		sched.c			\
		sigpush.c		\
		threads.c		\
		timex.c			\
		alarm.c			\
		unix_menu.c		\
		pipe_menu.c

noinst_HEADERS =			\
		my_stty.h		\
		stamps.h		\
		unix_prot.h

