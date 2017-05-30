# OLD COMMENT:
# We put the .o files from .cu files here to make sure to recompile when needed
# BUT that breaks things on non-cuda systems...
# NEW COMMET:
# All .cu files moved to ../cu2

all:	libqcuda.a
	cp libqcuda.a ../../libs

# where to find config.h?  I thought it would be $top_srcdir, but on make distcheck
# that generated ../../.. instead of ../.. , and it doesn't work!?
#
#NVCCFLAGS =	-I/usr/local/cuda/include -I$(HOME)/NVIDIA_CUDA_SDK_2.2/common/inc -I$(top_srcdir) -I$(top_srcdir)/include -DHAVE_CONFIG_H

# The sm_13 flag makes double precision warnings from the assembler
# go away, but we need a way to determine whether or not our hardware
# has the capability!?
# This should be done in configure...
# -arch sm_13 is good for wheatstone but breaks things on craik...
# -arch sm_11 on craik

# This file has posed problems because it needs to work on both cuda and non-cuda systems.
# Non-cuda systems have no nvcc, so we don't really want to include the .cu files in the list
# of sources all the time.
# We have tried to address this by conditionally adding the .cu files to

# Tried adding '-ccbin clang' to work on mac, but didn't help...

#NVCCFLAGS =	-ccbin clang @NVCCFLAGS@ @NVCC_ARCH_FLAGS@ $(CPPFLAGS) --machine 64 $(NVCPPFLAGS) -DHAVE_CONFIG_H

NVCCFLAGS =	@NVCCFLAGS@ @NVCC_ARCH_FLAGS@ $(CPPFLAGS) $(NVCPPFLAGS) -DHAVE_CONFIG_H

NVCPPFLAGS =	-I/usr/local/cuda/include -I$(builddir) -I../.. -I$(top_srcdir)/include
#-I../.. -I$(top_srcdir)/include -DHAVE_CONFIG_H

.cu.o:	$(NVCC) -o $@ -c `test -f $< || echo $(srcdir)/`$< $(NVCCFLAGS)

# The dependency building stuff doesn't seem to know about .cu files...

#		../../include/veclib/fast_test.h	\
#		../../include/veclib/xfer_args.h

CU_DEPS =						\
		$(top_srcdir)/include/my_cuda.h				\
		$(top_srcdir)/include/cuda_supp.h 		\
		$(top_builddir)/config.h 				\
		$(top_srcdir)/include/quip_config.h 		\
		$(top_srcdir)/include/quip_version.h 		\
		$(top_srcdir)/include/veclib/vecgen.h

host_calls.h:	host_call_utils.h ../../include/veclib/fast_test.h	\
		../../include/veclib/xfer_args.h
	touch host_calls.h

# this rule doesn't seem to fire in make distcheck...


noinst_LIBRARIES = libqcuda.a

#CC = g++

# CPPFLAGS is for /bin/cpp (C preprocessor)
#libqcuda_a_CPPFLAGS = -I$(top_srcdir)/include -D_GNU_SOURCE -Wall -Wmissing-prototypes
libqcuda_a_CPPFLAGS = -I$(top_srcdir)/include -D_GNU_SOURCE
libqcuda_a_CXXFLAGS = -Wall -Wswitch-enum
libqcuda_a_CFLAGS = -Wall -Wmissing-prototypes -Wswitch-enum

if HAVE_NVCC

libqcuda_a_CPPFLAGS += -I/usr/local/cuda/include $(NVCPPFLAGS)

# This file appears to be here when we build in the cvs tree,
# but not in the distribution!?!?
# We'll comment this out for now...
#clean-local:
#	rm myproject.linkinfo

endif

# -D__STRICT_ANSI__  this breaks u_long in sys/types.h...

# Should the shell scripts be listed here, or in EXTRA_DIST???
#				root_cmds.sh		\
#

noinst_HEADERS =					\
				cuda_viewer.h		\
				glx_hack.h

libqcuda_a_SOURCES =					\
				cuda.cpp		\
				cuda_npp.cpp		\
				cuda_curand.cpp		\
				simpleCUBLAS.cpp	\
				deviceQuery.cpp		\
				cuda_error.c		\
				cuda_streams.c		\
				cuda_menu.cpp

#libqcuda_a_SOURCES += fakit.cpp


#else
#
#libqcuda_a_SOURCES += fakit.cpp
#

EXTRA_DIST	=	root_cmds.sh

clean-local:
	-rm -rf *expanded.c *stripped.c

