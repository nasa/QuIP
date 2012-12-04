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

#math_drill_SOURCES = math.c
#math_drill_CPPFLAGS = -I$(top_srcdir)/include

# BUG - how do we figure out what Xlib?
CUDALIBS= ../libsrc/cuda/libcuda.a
VLIBS=	../libsrc/viewmenu/libviewmenu.a ../libsrc/parport/libpport.a ../libsrc/genwin/libgenwin.a ../libsrc/lutmenu/liblutmenu.a ../libsrc/luts/libluts.a ../libsrc/view/libview.a ../libsrc/xsupp/libxsupp.a
MVILIBS= ../libsrc/mvimenu/libmvimenu.a ../libsrc/seq/libseq.a
VECLIBS= ../libsrc/vectree/libvectree.a ../libsrc/newvec/libnewvec.a ../libsrc/vec_util/libvec_util.a
DLIBS= ../libsrc/datamenu/libdatamenu.a ../libsrc/fileio/libfileio.a ../libsrc/dobj/libdobj.a ../libsrc/nports/libnports.a ../libsrc/rawvol/librawvol.a
JLIBS= ../libsrc/interpreter/libinterpreter.a ../libsrc/qutil/libqutil.a
PLIB= ../libsrc/psych/libpsych.a ../libsrc/cstepit/libcstepit.a
SVRLIB= ../libsrc/www/libserver.a
GUILIB=	../libsrc/gui/libgui.a
OCVLIB=	../libsrc/opencv/libocvq.a
V4LIB=	../libsrc/v4l2/libvmenu.a
SOUNDLIB= ../libsrc/sound/libsound.a
RQLIB= ../libsrc/dither/libdither.a
KNOXLIB= ../libsrc/knox/libknox.a
METEORLIB= ../libsrc/meteor/libmeteor.a
GPSLIB= ../libsrc/gps/libgps.a
PICLIB=	../libsrc/pic/libpicmenu.a
VISCALIB= ../libsrc/visca/libvisca.a
GSLLIB=	../libsrc/gsl/libgslmenu.a
ATCLIB=	../libsrc/atc/libatc.a
DASLIB=	../libsrc/das1602/libdas1602.a


DVLIB=	../libsrc/ieee1394/libdvmenu.a
OGL_LIB=	../libsrc/opengl/libglmenu.a
PGRLIB=	../libsrc/ptgrey/libpgr.a


LIBS=	@LIBS@

quip_LDADD = $(SOUNDLIB) ../libsrc/nrmenu/libnrmenu.a

if HAVE_NUMREC
  quip_LDADD += -lnumrec
endif

quip_LDADD += $(DASLIB) $(ATCLIB) $(DVLIB) $(PGRLIB) $(OCVLIB) $(OGL_LIB) $(GPSLIB) $(PICLIB) $(CUDALIBS) $(V4LIB) $(RQLIB) $(VISCALIB) $(KNOXLIB) $(METEORLIB) $(GUILIB) $(VLIBS) $(MVILIBS) $(PLIB) $(VECLIBS) $(GSLLIB) $(DLIBS) $(JLIBS) $(SVRLIB)

# this doesn't work any more because lib dobj has dependencies on libcuda - should FIXME?  BUG?
#math_drill_LDADD = $(VLIBS) $(VECLIBS) $(RQLIB) $(DLIBS) $(JLIBS)

check_SCRIPTS = quip_test.sh
TESTS = $(check_SCRIPTS)


#CLEANFILES = quip_test.sh


