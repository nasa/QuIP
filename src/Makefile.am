#bin_PROGRAMS = quip datamenu quip_test1
#bin_PROGRAMS = quip math_drill
# math_drill doesn't compile with cuda on system...

bin_PROGRAMS = quip

EXTRA_DIST =	quip_test.sh


#quip_test1_SOURCES = quip_test1.c
#quip_test1_CPPFLAGS = -I$(top_srcdir)/include
#quip_test1_LDADD = ../libsrc/interpreter/libinterpreter.a ../libsrc/qutil/libqutil.a ../libsrc/dobj/libdobj.a ../libsrc/nports/libnports.a

quip_SOURCES = quip_main.c
quip_CPPFLAGS = -I$(top_srcdir)/include

quip_LDFLAGS = -L../libs -Wl,-rpath,/usr/local/cuda/lib
#quip_LDFLAGS = -rpath /usr/local/cuda/lib

#math_drill_SOURCES = math.c
#math_drill_CPPFLAGS = -I$(top_srcdir)/include

# BUG - how do we figure out what Xlib?

# If we specify the .a files instead of using the -l option form,
# then the program will be relinked when the libraries are updated

CUDALIBS= ../libs/libqcuda.a ../libs/libcu2.a
VLIBS=	../libs/libpport.a ../libs/libview.a ../libs/libxsupp.a
MVILIBS= ../libsrc/mvimenu/libmvimenu.a ../libsrc/seq/libseq.a
VECLIBS= ../libs/libvectree.a ../libs/libvec.a ../libs/libvl2.a ../libs/libocl.a ../libs/libvec_util.a
DLIBS= ../libs/libfileio.a ../libs/libhips.a ../libs/libdobj.a ../libs/libports.a ../libs/librawvol.a
#JLIBS= -linterpreter -lunix_util
JLIBS= ../libs/libinterpreter.a ../libs/libunix_util.a
PLIB= ../libs/libpsych.a ../libs/libcstepit.a
SVRLIB= ../libs/libserver.a
GUILIB=	../libs/libgui.a
OCVLIB=	../libs/libocvq.a
V4LIB=	../libs/libvmenu.a
SOUNDLIB= ../libs/libsound.a
RQLIB= ../libs/libdither.a
KNOXLIB= ../libs/libknox.a
METEORLIB= ../libs/libmeteor.a
VISCALIB= ../libs/libvisca.a
GSLLIB=	../libs/libgslmenu.a
DASLIB=	../libs/libdas1602.a
FLYLIB=	../libs/libfly.a
MYFANNLIB=	../libs/libmyfann.a


DVLIB=	../libsrc/ieee1394/libdvmenu.a
OGL_LIB=	../libsrc/opengl/libglmenu.a
PGRLIB=	../libsrc/ptgrey/libpgr.a


LIBS=	@LIBS@

quip_LDADD = $(SOUNDLIB) ../libs/libnrmenu.a

if HAVE_NUMREC
  quip_LDADD += -lnumrec
endif

quip_LDADD += $(MYFANNLIB) $(DASLIB) $(DVLIB) $(PGRLIB) $(FLYLIB) $(OCVLIB) $(OGL_LIB) $(CUDALIBS) $(V4LIB) $(RQLIB) $(VISCALIB) $(KNOXLIB) $(METEORLIB) $(GUILIB) $(VLIBS) $(MVILIBS) $(PLIB) $(VECLIBS) $(GSLLIB) $(DLIBS) $(JLIBS) $(SVRLIB)

# this doesn't work any more because lib dobj has dependencies on libcuda - should FIXME?  BUG?
#math_drill_LDADD = $(VLIBS) $(VECLIBS) $(RQLIB) $(DLIBS) $(JLIBS)

check_SCRIPTS = quip_test.sh
TESTS = $(check_SCRIPTS)


#CLEANFILES = quip_test.sh


