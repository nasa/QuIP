
all:	libcu2.a
	cp libcu2.a ../../libs

noinst_LIBRARIES = libcu2.a

libcu2_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

libcu2_a_SOURCES =			\
			cu2.c		\
			cu2_curand.c	\
			cu2_fft.c	\
			cu2_streams.c	\
			cu2_utils.c


NVCCFLAGS =	@NVCCFLAGS@ @NVCC_ARCH_FLAGS@ $(CPPFLAGS) --machine 64 $(NVCPPFLAGS) -DHAVE_CONFIG_H

NVCPPFLAGS =	-I/usr/local/cuda/include -I$(builddir) -I../.. -I$(top_srcdir)/include

.cu.o:	$(NVCC) -o $@ -c `test -f $< || echo $(srcdir)/`$< $(NVCCFLAGS)

#clean-local:
#	-rm -rf cu2_veclib_expanded.c cu2_veclib_stripped.c

NVCC_SOURCES =			cu2_funcs.cu

# more dependencies here, but what are they?  BUG
cu2_funcs.cu:	cu2_veclib.c cu2_host_untyped_call_defs.h

# the dependencies above don't seem to trigger the recompile!?

# May not have added ALL dependencies!?  BUG
cu2_funcs.o:		cu2_funcs.cu			\
			cu2_veclib.c			\
			cu2_kern_call_defs.h		\
			cu2_kernels.c			\
			cu2_host_call_defs.h		\
			cu2_host_funcs.c		\
			cu2_typtbl.c			\
			cu2_host_untyped_call_defs.h	\
			../../include/veclib/cu2_port.h	\
			../../include/veclib/cu2_veclib_prot.h	\
			../../include/veclib/host_typed_call_defs.h	\
			../../include/veclib/gen_gpu_calls.h	\
			../../include/veclib/gpu_args.h	\
			../../include/veclib/gen_entries.h	\
			../../include/veclib/gen_entries.c	\
			../../include/veclib/gen_host_calls.c	\
			../../include/veclib/dp_func_defs.h	\
			../../include/veclib/gpu_call_utils.h	\
			../../include/veclib/slow_len.h	\
			../../include/veclib/eqsp_incs.h	\
			../../include/veclib/slow_incs.h	\
			../../include/veclib/all_vec.c	\
			../../include/veclib/vecgen.h	\
			../../include/veclib/report_args.h	\
			cu2_host_funcs.c		\
			cu2_host_call_defs.h
	$(NVCC) -o $@ -c `test -f $< || echo $(srcdir)/`$< $(NVCCFLAGS)

noinst_HEADERS =					\
			cu2_kern_call_defs.h		\
			cu2_host_call_defs.h		\
			cu2_func_tbl.h			\
			cu2_host_funcs.c		\
			cu2_host_untyped_call_defs.h	\
			cu2_kernels.c			\
			cu2_veclib.c			\
			cu2_typtbl.c			\
			cu2_typtbl.h

if HAVE_NVCC
libcu2_a_SOURCES += $(NVCC_SOURCES)
else
noinst_HEADERS += $(NVCC_SOURCES)
endif

#else
#
