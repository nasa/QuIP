
all:	libglmenu.a
	cp libglmenu.a ../../libs

noinst_LIBRARIES = libglmenu.a

# The main project includes the cuda inc dirs, but that breaks
# nvstusb.c...

# These seem to get appended to inherited flags???
libglmenu_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

# This seems to fix the problem on craik...
# But breaks it on mach, MBP!?
#CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

libglmenu_a_SOURCES = glmenu.c dl.c tilemenu.c gl_util.c glx_supp.c tile.c stereo.c nvstusb.c usb_libusb.c glfb.c

noinst_HEADERS =	dl.h tile.h glx_supp.h glut_supp.h gl_util.h glfb.h nvstusb.h usb.h

#3ds.c
#Init.c
#Main.c
#dl.c
#facemain.c
#facemenu.c
#gl_util.c
#glmain.c
#glmenu.c
#glut_supp.c
#glx_supp.c
#teapot.c
#tile.c
#tilemenu.c
