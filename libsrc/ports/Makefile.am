all:	libports.a
	cp libports.a ../../libs

noinst_LIBRARIES = libports.a

libports_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

libports_a_SOURCES =			\
			client.c	\
			dataport.c	\
			packets.c	\
			portmenu.c	\
			ports.c		\
			server.c	\
			sockopts.c	\
			xmitrecv.c


noinst_HEADERS = ports.h


