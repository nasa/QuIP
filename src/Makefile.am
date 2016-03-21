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
CUDALIBS= -lqcuda
VLIBS=	-lpport -lview -lxsupp
MVILIBS= ../libsrc/mvimenu/libmvimenu.a ../libsrc/seq/libseq.a
VECLIBS= -lvectree -lvec -lvl2 -locl -lvec_util
DLIBS= -lfileio -lhips -ldobj -lports -lrawvol
JLIBS= -linterpreter -lunix_util
PLIB= -lpsych -lcstepit
SVRLIB= -lserver
GUILIB=	-lgui
OCVLIB=	-locvq
V4LIB=	-lvmenu
SOUNDLIB= -lsound
RQLIB= -ldither
KNOXLIB= -lknox
METEORLIB= -lmeteor
VISCALIB= -lvisca
GSLLIB=	-lgslmenu
DASLIB=	-ldas1602


DVLIB=	../libsrc/ieee1394/libdvmenu.a
OGL_LIB=	../libsrc/opengl/libglmenu.a
PGRLIB=	../libsrc/ptgrey/libpgr.a


LIBS=	@LIBS@


quip_LDADD = $(SOUNDLIB) -lnrmenu

if HAVE_NUMREC
  quip_LDADD += -lnumrec
endif

quip_LDADD += $(DASLIB) $(DVLIB) $(PGRLIB) $(OCVLIB) $(OGL_LIB) $(CUDALIBS) $(V4LIB) $(RQLIB) $(VISCALIB) $(KNOXLIB) $(METEORLIB) $(GUILIB) $(VLIBS) $(MVILIBS) $(PLIB) $(VECLIBS) $(GSLLIB) $(DLIBS) $(JLIBS) $(SVRLIB)

# this doesn't work any more because lib dobj has dependencies on libcuda - should FIXME?  BUG?
#math_drill_LDADD = $(VLIBS) $(VECLIBS) $(RQLIB) $(DLIBS) $(JLIBS)

check_SCRIPTS = quip_test.sh
TESTS = $(check_SCRIPTS)


#CLEANFILES = quip_test.sh


