all:	libgui.a
	cp libgui.a ../../libs

noinst_LIBRARIES = libgui.a

libgui_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

libgui_a_SOURCES = screen_objs.c protomenu.c motif.c

noinst_HEADERS = my_motif.h gui_cmds.h gui_prot.h

# NEED TO SET -DXVIEW OR -DMOTIF !!!!

#XVOFILES=	screen_objs.o protomenu.o xview.o



