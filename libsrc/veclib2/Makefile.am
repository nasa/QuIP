## Process this file with automake to produce Makefile.in
## (comments with ## are stripped by automake!)

# this generates a warning when using BUILT_SOURCES???

all:	libvl2.a
	cp libvl2.a ../../libs

clean-local:
	rm -rf $(GENERATED_SRCS)

GENERATED_SRCS=							\
			$(builddir)/vl2_veclib_expanded.c	\
			$(builddir)/vl2_utils_expanded.c


noinst_LIBRARIES = libvl2.a

libvl2_a_CPPFLAGS = -I$(top_srcdir)/include -I$(top_builddir)/include -Wall -Wmissing-prototypes

libvl2_a_CXXFLAGS = -O3
libvl2_a_CFLAGS = -O3

# Now include fast_sp.c in vl2_veclib.c...
#			fast_sp.c

M4OPTS=	-I $(srcdir) -I $(top_srcdir)/include

$(srcdir)/vl2_utils.c:	$(builddir)/vl2_utils_expanded.c
	touch $(srcdir)/vl2_utils.c

$(srcdir)/vl2_veclib.c:	$(builddir)/vl2_veclib_expanded.c
	touch $(srcdir)/vl2_veclib.c


vl2_utils_expanded.c:	vl2_utils.m4 _vl2_utils.c $(LOCAL_M4FILES) $(INC_M4FILES)
		m4 $(M4OPTS) < $(srcdir)/vl2_utils.m4 > $(builddir)/vl2_utils_expanded.c

# the sed command removes escaped newlines in functions that need
# to be quoted for opencl...

vl2_veclib_expanded.c:	$(srcdir)/vl2_veclib.m4 $(LOCAL_M4FILES) $(INC_M4FILES)
	m4 $(M4OPTS) $(srcdir)/vl2_veclib.m4 | sed -e 's/	*\\//' > $(builddir)/vl2_veclib_expanded.c

## OOPS - BUILT_SOURCES seem to be added to sources and compiled by themselves,
## although we intend for them just to be included... should we call them .h instead?
#BUILT_SOURCES=							\
#			$(builddir)/vl2_veclib_expanded.c	\
#			$(builddir)/vl2_utils_expanded.c

# Might be better to distribute?  but would not work in vpath build???
#nodist_libvl2_a_SOURCES =	$(GENERATED_SRCS)

libvl2_a_SOURCES =			\
			vl2_veclib.c	\
			vl2_utils.c	\
			vl2.c		\
			slow_len.c	\
			fftsupp.c	\
			vl2_platform.c

#			linear_funcs.c
#			lin_util.c
#			vl2_streams.c
#			vl2_viewer.c
#			vl2_func_tbl.c

#			vfft.m4
#			vl2_veclib.m4

LOCAL_M4FILES=							\
			cpu_call_defs.m4			\
			linear.m4				\
			vl2_host_call_defs.m4			\
			vl2_host_untyped_call_defs.m4			\
			vl2_host_funcs.m4			\
			vl2_typed_fft_funcs.m4			\
			vl2_typtbl.m4				\
			vl2_utils.m4				\
			vl2_kernels.m4

INCDIR=$(top_srcdir)/include/veclib
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
			$(INCDIR)/cpu_host_call_defs.m4		\
			$(INCDIR)/host_calls_special.m4		\
			$(INCDIR)/host_typed_call_defs.m4	\
			$(INCDIR)/host_fft_funcs.m4	\
			$(INCDIR)/in_defs.m4			\
			$(INCDIR)/inby_defs.m4			\
			$(INCDIR)/int_bit_vec.m4		\
			$(INCDIR)/intvec.m4			\
			$(INCDIR)/li_defs.m4			\
			$(INCDIR)/math_funcs.m4			\
			$(INCDIR)/new_conv.m4			\
			$(INCDIR)/vl2_port.m4			\
			$(INCDIR)/vl2_veclib_prot.m4		\
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

#noinst_HEADERS = my_vl2.h

# OLD for cpp macros...
#vl2_veclib.debug:	vl2_veclib.c
#	gcc -E -I../.. -I$(top_srcdir)/include vl2_veclib.c > vl2_veclib.debug_
#	indent < vl2_veclib.debug_  > vl2_veclib.debug
##./strip_it2.sh
#
##vl2.debug:	vl2.c
##	gcc -E -I../.. -I$(top_srcdir)/include vl2.c > vl2.debug_

noinst_HEADERS =					\
			$(LOCAL_M4FILES)		\
							\
			_vl2_utils.c			\
			vfft.c


