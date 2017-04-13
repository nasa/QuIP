
all:	libcu2.a
	cp libcu2.a ../../libs

# cu2_funcs.o gets built from cu2_funcs.cu with nvcc, but how does
# it get put into the library???

M4OPTS=	-I $(top_srcdir)/include -I $(srcdir)

GENERATED_SRCS=						\
			cu2_utils_expanded.c		\
			cu2_veclib_expanded.c		\
			cu2_fill_expanded.c		\
			cu2_yuv2rgb_expanded.c		\
			cu2_centroid_expanded.c

clean-local:
	rm -rf $(GENERATED_SRCS)

cu2_utils_expanded.c:	$(LOCAL_M4FILES) $(srcdir)/_cu2_utils.c
	m4 $(M4OPTS) < $(srcdir)/cu2_utils.m4 > $(builddir)/cu2_utils_expanded.c

cu2_veclib_expanded.c:	$(LOCAL_M4FILES) $(INC_M4FILES)
	m4 $(M4OPTS) $(srcdir)/cu2_veclib.m4 | sed -e 's/	*\\//' > $(builddir)/cu2_veclib_expanded.c

cu2_fill_expanded.c:	$(LOCAL_M4FILES) $(INC_M4FILES)
	m4 $(M4OPTS) < $(srcdir)/cu2_fill.m4 > $(builddir)/cu2_fill_expanded.c

cu2_yuv2rgb_expanded.c:	$(LOCAL_M4FILES) $(INC_M4FILES)
	m4 $(M4OPTS) < $(srcdir)/cu2_yuv2rgb.m4 > $(builddir)/cu2_yuv2rgb_expanded.c

cu2_centroid_expanded.c:	$(LOCAL_M4FILES) $(INC_M4FILES)
	m4 $(M4OPTS) < $(srcdir)/cu2_centroid.m4 > $(builddir)/cu2_centroid_expanded.c



$(srcdir)/cu2_centroid.cu:	$(builddir)/cu2_centroid_expanded.c
	touch $(srcdir)/cu2_centroid.cu

$(srcdir)/cu2_yuv2rgb.cu:	$(builddir)/cu2_yuv2rgb_expanded.c
	touch $(srcdir)/cu2_yuv2rgb.cu

$(srcdir)/cu2_fill.cu:	$(builddir)/cu2_fill_expanded.c
	touch $(srcdir)/cu2_fill.cu

$(srcdir)/cu2_utils.cu:	$(builddir)/cu2_utils_expanded.c
	touch $(srcdir)/cu2_utils.cu


# Why does the generic rule not work?

cu2_yuv2rgb.o:	$(srcdir)/cu2_yuv2rgb.cu
	$(NVCC) -o $@ -c `test -f $< || echo $(srcdir)/`$< $(NVCCFLAGS)

cu2_fill.o:	$(srcdir)/cu2_fill.cu
	$(NVCC) -o $@ -c `test -f $< || echo $(srcdir)/`$< $(NVCCFLAGS)

cu2_utils.o:	$(srcdir)/cu2_utils.cu
	$(NVCC) -o $@ -c `test -f $< || echo $(srcdir)/`$< $(NVCCFLAGS)

cu2_centroid.o:	$(srcdir)/cu2_centroid.cu
	$(NVCC) -o $@ -c `test -f $< || echo $(srcdir)/`$< $(NVCCFLAGS)

#	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

# May not have added ALL dependencies!?  BUG
cu2_funcs.o:		$(srcdir)/cu2_funcs.cu			\
			$(srcdir)/m4_cu2_veclib.c			\
			$(top_builddir)/include/veclib/cu2_veclib_prot.h	\
			$(top_srcdir)/include/veclib/slow_len.h	\
			$(top_srcdir)/include/veclib/vecgen.h
	$(NVCC) -o $@ -c `test -f $< || echo $(srcdir)/`$< $(NVCCFLAGS)

noinst_LIBRARIES = libcu2.a

libcu2_a_CPPFLAGS = -I$(top_srcdir)/include -I$(top_builddir)/include -Wall -Wmissing-prototypes

#			cu2.c

libcu2_a_SOURCES =			\
			cu2_curand.c	\
			cu2_fft.c	\
			cu2_streams.c	\
			cu2_utils.c


NVCCFLAGS =	@NVCCFLAGS@ @NVCC_ARCH_FLAGS@ --machine 64 $(CPPFLAGS) $(NVCPPFLAGS) -DHAVE_CONFIG_H

NVCPPFLAGS =	-I/usr/local/cuda/include -I$(builddir) -I$(top_builddir) -I$(top_srcdir)/include -I$(top_builddir)/include

.cu.o:	$(NVCC) -o $@ -c `test -f $< || echo $(srcdir)/`$< $(NVCCFLAGS)

#clean-local:
#	-rm -rf cu2_veclib_expanded.c cu2_veclib_stripped.c

NVCC_SOURCES =						\
				cu2_funcs.cu		\
				cu2_fill.cu		\
				cu2_centroid.cu		\
				cu2_yuv2rgb.cu


# more dependencies here, but what are they?  BUG
$(srcdir)/cu2_funcs.cu:	m4_cu2_veclib.c
		touch $(srcdir)/cu2_funcs.cu

# cu2_host_untyped_call_defs.h

$(srcdir)/m4_cu2_veclib.c:	cu2_veclib_expanded.c
	touch $(srcdir)/m4_cu2_veclib.c

#			cu2_kern_args.m4
#			cu2_kernel_src.m4

LOCAL_M4FILES=							\
			$(srcdir)/cu2_fft_funcs.m4		\
			$(srcdir)/cu2_typed_fft_funcs.m4	\
			$(srcdir)/cu2_host_call_defs.m4		\
			$(srcdir)/cu2_host_funcs.m4		\
			$(srcdir)/cu2_kern_call_defs.m4		\
			$(srcdir)/cu2_kernels.m4		\
			$(srcdir)/cu2_typtbl.m4			\
			$(srcdir)/cu2_utils.m4			\
			$(srcdir)/cu2_fill.m4			\
			$(srcdir)/cu2_centroid.m4		\
			$(srcdir)/cu2_centroid_defs.m4		\
			$(srcdir)/cu2_yuv2rgb.m4			\
			$(srcdir)/cu2_veclib.m4 


M4_INCDIR=$(srcdir)/../../include/veclib

INC_M4FILES=							\
			$(M4_INCDIR)/all_bit_kernels.m4		\
			$(M4_INCDIR)/all_float_kernels.m4	\
			$(M4_INCDIR)/all_int_kernels.m4		\
			$(M4_INCDIR)/all_mixed_float_kernels.m4	\
			$(M4_INCDIR)/all_mixed_int_kernels.m4	\
			$(M4_INCDIR)/all_mixed_uint_kernels.m4	\
			$(M4_INCDIR)/all_same_prec_vec.m4	\
			$(M4_INCDIR)/all_uint_kernels.m4	\
			$(M4_INCDIR)/all_vec.m4			\
			$(M4_INCDIR)/bit_defs.m4		\
			$(M4_INCDIR)/by_defs.m4			\
			$(M4_INCDIR)/cpx_args.m4		\
			$(M4_INCDIR)/cpx_vec.m4			\
			$(M4_INCDIR)/di_defs.m4			\
			$(M4_INCDIR)/dp_defs.m4			\
			$(M4_INCDIR)/dp_func_defs.m4		\
			$(M4_INCDIR)/eqsp_defs.m4		\
			$(M4_INCDIR)/eqsp_incs.m4		\
			$(M4_INCDIR)/fast_defs.m4		\
			$(M4_INCDIR)/fast_eqsp_defs.m4		\
			$(M4_INCDIR)/fast_test.m4		\
			$(M4_INCDIR)/fft_funcs.m4		\
			$(M4_INCDIR)/flen_defs.m4		\
			$(M4_INCDIR)/flush_output.m4		\
			$(M4_INCDIR)/gen_bit_calls.m4		\
			$(M4_INCDIR)/gen_float_calls.m4		\
			$(M4_INCDIR)/gen_func_array.m4		\
			$(M4_INCDIR)/gen_gpu_calls.m4		\
			$(M4_INCDIR)/gen_host_calls.m4		\
			$(M4_INCDIR)/gen_int_calls.m4		\
			$(M4_INCDIR)/gen_kern_args.m4		\
			$(M4_INCDIR)/gen_kernel_calls.m4	\
			$(M4_INCDIR)/gen_mixed_float_calls.m4	\
			$(M4_INCDIR)/gen_mixed_int_calls.m4	\
			$(M4_INCDIR)/gen_mixed_uint_calls.m4	\
			$(M4_INCDIR)/gen_port.m4		\
			$(M4_INCDIR)/gen_typtbl.m4		\
			$(M4_INCDIR)/gen_uint_calls.m4		\
			$(M4_INCDIR)/gpu_args.m4		\
			$(M4_INCDIR)/gpu_call_utils.m4		\
			$(M4_INCDIR)/gpu_special_defs.m4	\
			$(M4_INCDIR)/gpu_host_call_defs.m4	\
			$(M4_INCDIR)/host_calls_special.m4	\
			$(M4_INCDIR)/host_typed_call_defs.m4	\
			$(M4_INCDIR)/in_defs.m4			\
			$(M4_INCDIR)/inby_defs.m4		\
			$(M4_INCDIR)/int_bit_vec.m4		\
			$(M4_INCDIR)/intvec.m4			\
			$(M4_INCDIR)/li_defs.m4			\
			$(M4_INCDIR)/math_funcs.m4		\
			$(M4_INCDIR)/new_conv.m4		\
			$(M4_INCDIR)/cu2_port.m4		\
			$(M4_INCDIR)/cu2_veclib_prot.m4		\
			$(M4_INCDIR)/quat_args.m4		\
			$(M4_INCDIR)/real_args.m4		\
			$(M4_INCDIR)/report_args.m4		\
			$(M4_INCDIR)/signed_vec.m4		\
			$(M4_INCDIR)/slow_defs.m4		\
			$(M4_INCDIR)/slow_incs.m4		\
			$(M4_INCDIR)/slow_len.m4		\
			$(M4_INCDIR)/slow_vars.m4		\
			$(M4_INCDIR)/sp_defs.m4			\
			$(M4_INCDIR)/sp_func_defs.m4		\
			$(M4_INCDIR)/spdp_defs.m4		\
			$(M4_INCDIR)/uby_defs.m4		\
			$(M4_INCDIR)/ubyin_defs.m4		\
			$(M4_INCDIR)/udi_defs.m4		\
			$(M4_INCDIR)/uin_defs.m4		\
			$(M4_INCDIR)/uindi_defs.m4		\
			$(M4_INCDIR)/uli_defs.m4		\
			$(M4_INCDIR)/unsigned_vec.m4		\
			$(M4_INCDIR)/vecgen.m4			\
			$(M4_INCDIR)/xfer_args.m4

# the dependencies above don't seem to trigger the recompile!?

#			cu2_kern_call_defs.h		\
#			cu2_kernels.c			\
#			cu2_host_call_defs.h		\
#			cu2_host_funcs.c		\
#			cu2_typtbl.c			\
#			../../include/veclib/all_vec.c	\
#			../../include/veclib/report_args.h	\
#			cu2_host_funcs.c		\
#			cu2_host_funcs.c		\
#			cu2_host_call_defs.h
#			../../include/veclib/gen_host_calls.c	\
#			../../include/veclib/dp_func_defs.h	\
#			../../include/veclib/gpu_call_utils.h	\
#			../../include/veclib/host_typed_call_defs.h	\
#			../../include/veclib/gen_gpu_calls.h	\
#			../../include/veclib/gpu_args.h	\
#			../../include/veclib/gen_entries.h	\
#			../../include/veclib/cu2_port.h	\
#			../../include/veclib/eqsp_incs.h	\
#			../../include/veclib/slow_incs.h	\
#			cu2_host_untyped_call_defs.h


#			cu2_kern_call_defs.h
#			cu2_host_call_defs.h
#			cu2_host_untyped_call_defs.h
#			cu2_typtbl.h

noinst_HEADERS =					\
			$(LOCAL_M4FILES)		\
							\
			cu2_fill.cu			\
			cu2_func_tbl.h			\
			cu2.c				\
			_cu2_utils.c			\
			m4_cu2_veclib.c

if HAVE_NVCC
libcu2_a_SOURCES += $(NVCC_SOURCES)
else
noinst_HEADERS += $(NVCC_SOURCES)
endif

