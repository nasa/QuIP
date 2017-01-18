SUBDIRS =	Random123

all:	libocl.a
	cp libocl.a ../../libs

noinst_LIBRARIES = libocl.a

libocl_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

# BUG this is mac only...
#AM_CFLAGS= -framework OpenGL
AM_CFLAGS= -I/System/Library/Frameworks/OpenGL.framework/Headers

#libmvimenu_a_CPPFLAGS = -I$(top_srcdir)/include -Wall -Wmissing-prototypes

#	ocl_viewer.c

#	ocl_veclib.c

libocl_a_SOURCES =			\
			ocl_fft.c	\
			ocl_streams.c	\
			ocl_menu.c	\
			ocl_rand.c	\
			ocl_utils.c	\
			m4_ocl_veclib.c	\
			ocl.c

m4_ocl_veclib.c:	ocl_veclib_expanded.c

#../../include/veclib/vecgen.m4:

LOCAL_M4FILES=					\
			ocl_fft_funcs.m4	\
			ocl_host_call_defs.m4	\
			ocl_host_funcs.m4	\
			ocl_kern_args.m4	\
			ocl_kern_call_defs.m4	\
			ocl_kernel_src.m4	\
			ocl_kernels.m4		\
			ocl_typtbl.m4		\
			ocl_veclib.m4 

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
			$(INCDIR)/ocl_port.m4			\
			$(INCDIR)/ocl_veclib_prot.m4		\
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

ocl_veclib_expanded.c:	$(LOCAL_M4FILES) $(INC_M4FILES)
	m4 ocl_veclib.m4 > ocl_veclib_expanded.c


ocl_rand.c:	vuni.i

vuni.i:	$(srcdir)/vuni.ocl
	$(srcdir)/gencl.sh $(srcdir)/vuni.ocl vuni.i

#			ocl_func_tbl.c

#noinst_HEADERS = my_ocl.h

#ocl_veclib_expanded.c:	_ocl_veclib.c
#	gcc -E -I../.. -I$(top_srcdir)/include `test -f _ocl_veclib.c || echo $(srcdir)/`_ocl_veclib.c > ocl_veclib_expanded.c || rm ocl_veclib_expanded.c
#
#ocl_veclib_stripped.c:		ocl_veclib_expanded.c
#	./strip_it.sh
#
#ocl_veclib.c:	ocl_veclib_stripped.c

#./strip_it2.sh
#			ocl_api.h

noinst_HEADERS =					\
			ocl_host_funcs.c		\
			ocl_kernels.c			\
			ocl_typtbl.c			\
			ocl_typed_fft.c			\
			gencl.sh			\
			vuni.ocl			\
			conversions.h			\
			enum_menu_calls.h		\
			host_conv_prot.h		\
			kern_src.h			\
			ocl_host_call_defs.h		\
			ocl_host_untyped_call_defs.h	\
			ocl_kern_args.h			\
			ocl_kern_call_defs.h		\
			ocl_kernel_src.h		\
			type_undefs.h			\
			undefs.h


clean-local:
	-rm -rf vuni.i

#	-rm -rf ocl_veclib_expanded.c ocl_veclib_stripped.c vuni.i

