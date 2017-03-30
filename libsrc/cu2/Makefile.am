
all:	libcu2.a
	cp libcu2.a ../../libs

# cu2_funcs.o gets built from cu2_funcs.cu with nvcc, but how does
# it get put into the library???

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
cu2_funcs.cu:	m4_cu2_veclib.c cu2_host_untyped_call_defs.h

m4_cu2_veclib.c:	cu2_veclib_expanded.c

#			cu2_kern_args.m4
#			cu2_kernel_src.m4

LOCAL_M4FILES=					\
			cu2_fft_funcs.m4	\
			cu2_host_call_defs.m4	\
			cu2_host_funcs.m4	\
			cu2_kern_call_defs.m4	\
			cu2_kernels.m4		\
			cu2_typtbl.m4		\
			cu2_veclib.m4 


INCDIR=../../include/veclib
INC_M4FILES=							\
			$(INCDIR)/all_bit_kernels.m4		\
			$(INCDIR)/all_float_kernels.m4		\
			$(INCDIR)/all_int_kernels.m4		\
			$(INCDIR)/all_mixed_float_kernels.m4	\
			$(INCDIR)/all_mixed_int_kernels.m4	\
			$(INCDIR)/all_mixed_uint_kernels.m4	\
			$(INCDIR)/all_same_prec_vec.m4		\
			$(INCDIR)/all_uint_kernels.m4		\
			$(INCDIR)/all_vec.m4			\
			$(INCDIR)/bit_defs.m4			\
			$(INCDIR)/by_defs.m4			\
			$(INCDIR)/cpx_args.m4			\
			$(INCDIR)/cpx_vec.m4			\
			$(INCDIR)/di_defs.m4			\
			$(INCDIR)/dp_defs.m4			\
			$(INCDIR)/dp_func_defs.m4		\
			$(INCDIR)/eqsp_defs.m4			\
			$(INCDIR)/eqsp_incs.m4			\
			$(INCDIR)/fast_defs.m4			\
			$(INCDIR)/fast_eqsp_defs.m4		\
			$(INCDIR)/fast_test.m4			\
			$(INCDIR)/fft_funcs.m4			\
			$(INCDIR)/flen_defs.m4			\
			$(INCDIR)/flush_output.m4		\
			$(INCDIR)/gen_bit_calls.m4		\
			$(INCDIR)/gen_float_calls.m4		\
			$(INCDIR)/gen_func_array.m4		\
			$(INCDIR)/gen_gpu_calls.m4		\
			$(INCDIR)/gen_host_calls.m4		\
			$(INCDIR)/gen_int_calls.m4		\
			$(INCDIR)/gen_kern_args.m4		\
			$(INCDIR)/gen_kernel_calls.m4		\
			$(INCDIR)/gen_mixed_float_calls.m4	\
			$(INCDIR)/gen_mixed_int_calls.m4	\
			$(INCDIR)/gen_mixed_uint_calls.m4	\
			$(INCDIR)/gen_port.m4			\
			$(INCDIR)/gen_typtbl.m4			\
			$(INCDIR)/gen_uint_calls.m4		\
			$(INCDIR)/gpu_args.m4			\
			$(INCDIR)/gpu_call_utils.m4		\
			$(INCDIR)/host_calls_cpu.m4		\
			$(INCDIR)/host_calls_gpu.m4		\
			$(INCDIR)/host_calls_special.m4		\
			$(INCDIR)/host_typed_call_defs.m4	\
			$(INCDIR)/in_defs.m4			\
			$(INCDIR)/inby_defs.m4			\
			$(INCDIR)/int_bit_vec.m4		\
			$(INCDIR)/intvec.m4			\
			$(INCDIR)/li_defs.m4			\
			$(INCDIR)/math_funcs.m4			\
			$(INCDIR)/new_conv.m4			\
			$(INCDIR)/cu2_port.m4			\
			$(INCDIR)/cu2_veclib_prot.m4		\
			$(INCDIR)/quat_args.m4			\
			$(INCDIR)/real_args.m4			\
			$(INCDIR)/report_args.m4		\
			$(INCDIR)/signed_vec.m4			\
			$(INCDIR)/slow_defs.m4			\
			$(INCDIR)/slow_incs.m4			\
			$(INCDIR)/slow_len.m4			\
			$(INCDIR)/slow_vars.m4			\
			$(INCDIR)/sp_defs.m4			\
			$(INCDIR)/sp_func_defs.m4		\
			$(INCDIR)/spdp_defs.m4			\
			$(INCDIR)/uby_defs.m4			\
			$(INCDIR)/ubyin_defs.m4			\
			$(INCDIR)/udi_defs.m4			\
			$(INCDIR)/uin_defs.m4			\
			$(INCDIR)/uindi_defs.m4			\
			$(INCDIR)/uli_defs.m4			\
			$(INCDIR)/unsigned_vec.m4		\
			$(INCDIR)/vecgen.m4			\
			$(INCDIR)/xfer_args.m4

cu2_veclib_expanded.c:	$(LOCAL_M4FILES) $(INC_M4FILES)
	m4 cu2_veclib.m4 | sed -e 's/	*\\//' > cu2_veclib_expanded.c

# the dependencies above don't seem to trigger the recompile!?

# May not have added ALL dependencies!?  BUG
cu2_funcs.o:		cu2_funcs.cu			\
			m4_cu2_veclib.c			\
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
			m4_cu2_veclib.c			\
			cu2_typtbl.c			\
			cu2_typtbl.h

if HAVE_NVCC
libcu2_a_SOURCES += $(NVCC_SOURCES)
else
noinst_HEADERS += $(NVCC_SOURCES)
endif

