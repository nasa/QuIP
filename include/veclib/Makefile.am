
GENERATED_SOURCE=	vl2_veclib_prot.h ocl_veclib_prot.h cu2_veclib_prot.h

all:	$(GENERATED_SOURCE)

clean-local:
	rm -rf $(GENERATED_SOURCE)

#			enum_menu_calls.h
#			gen_menu_prot.h
#			cu2_menu_prot.h
#			menu_call_defs.h
#			menu_prot_defs.h
#			menu_undefs.h
#			ocl_menu_prot.h
#			vl2_menu_prot.h

#		gen_entries.c

#			cpx_args.h
#			cpx_vec.c
#			fft_funcs.c
#			gen_func_array.c
#			gen_host_calls.c
#			gen_kernel_calls.c
#			host_fft_funcs.c
#			intvec.c
#			int_bit_vec.c
#			math_funcs.c
#			new_conv.c
#			signed_vec.c
#			unsigned_vec.c
#			all_bit_kernels.h		\
#			bit_defs.h			\
#			bitmap_ops.h			\
#			by_defs.h			\
#			cu2_port.h			\
#			cu2_veclib_prot.h		\
#			di_defs.h			\
#			dp_defs.h			\
#			dp_func_defs.h			\
#			eqsp_incs.h			\
#			fast_test.h			\
#			fftsupp.h			\
#			func_undefs.h			\
#			gen_bit_calls.h			\
#			gen_entries.h			\
#			gen_gpu_calls.h			\
#			gen_kern_args.h			\
#			gen_port.h			\
#			gen_typtbl.h			\
#			gpu_args.h			\
#			gpu_call_utils.h		\
#			host_calls_cpu.h		\
#			host_calls_gpu.h		\
#			host_calls_special.h		\
#			host_typed_call_defs.h		\
#			inby_defs.h			\
#			in_defs.h			\
#			li_defs.h			\
#			method_undefs.h			\
#			obj_args.h			\
#			ocl_port.h			\
#			ocl_veclib_prot.h		\
#			quat_args.h			\
#			real_args.h			\
#			report_args.h			\
#			slow_incs.h			\
#			slow_vars.h			\
#			sp_defs.h			\
#			spdp_defs.h			\
#			sp_func_defs.h			\
#			type_undefs.h			\
#			uby_defs.h			\
#			ubyin_defs.h			\
#			udi_defs.h			\
#			uin_defs.h			\
#			uindi_defs.h			\
#			uli_defs.h			\
#			vec_func.h			\
#			vecgen.h			\
#			vl2_port.h			\
#			vl2_veclib_prot.h		\
#			xfer_args.h			\
#			eqsp_defs.h			\
#			fast_defs.h			\
#			flen_defs.h			\
#			fast_eqsp_defs.h		\
#			gen_float_calls.h		\
#			gpu_call_defs.h			\
#			slow_defs.h			\
#			speed_undefs.h			\
#			all_float_kernels.h		\
#			all_int_kernels.h		\
#			all_mixed_float_kernels.h	\
#			all_mixed_int_kernels.h		\
#			all_mixed_uint_kernels.h	\
#			all_uint_kernels.h		\
#			gen_int_calls.h			\
#			gen_mixed_float_calls.h		\
#			gen_mixed_int_calls.h		\
#			gen_mixed_uint_calls.h		\
#			gen_uint_calls.h


cu2_veclib_prot.h:	cu2_veclib_prot.m4 cu2_port.m4 gen_port.m4 cu2_func_prot.m4
			m4 -I $(srcdir) -I$(top_srcdir)/include < $(srcdir)/cu2_veclib_prot.m4 > $(builddir)/cu2_veclib_prot.h

vl2_veclib_prot.h:	vl2_veclib_prot.m4 vl2_port.m4 gen_port.m4 vl2_func_prot.m4
			m4 -I $(srcdir) -I$(top_srcdir)/include < $(srcdir)/vl2_veclib_prot.m4 > $(builddir)/vl2_veclib_prot.h

ocl_veclib_prot.h:	ocl_veclib_prot.m4 ocl_port.m4 gen_port.m4 ocl_func_prot.m4
			m4 -I $(srcdir) -I$(top_srcdir)/include < $(srcdir)/ocl_veclib_prot.m4 > $(builddir)/ocl_veclib_prot.h

#			platform_funcs.h

# Files that need to be included in the distribution archive,
# but not installed as system includes...

noinst_HEADERS =					\
			vl2_veclib_prot.m4		\
			ocl_veclib_prot.m4		\
			cu2_veclib_prot.m4		\
			vl2_func_prot.m4		\
			ocl_func_prot.m4		\
			cu2_func_prot.m4		\
			gen_port.m4			\
			vl2_port.m4			\
			ocl_port.m4			\
			cu2_port.m4			\
			vecgen.m4			\
			platform_funcs.m4		\
							\
			all_bit_kernels.m4		\
			all_float_kernels.m4		\
			all_int_kernels.m4		\
			all_mixed_float_kernels.m4	\
			all_mixed_int_kernels.m4	\
			all_mixed_uint_kernels.m4	\
			all_uint_kernels.m4		\
			bit_defs.m4			\
			by_defs.m4			\
			di_defs.m4			\
			dp_defs.m4			\
			inby_defs.m4			\
			in_defs.m4			\
			li_defs.m4			\
			sp_defs.m4			\
			spdp_defs.m4			\
			uby_defs.m4			\
			ubyin_defs.m4			\
			udi_defs.m4			\
			uin_defs.m4			\
			uindi_defs.m4			\
			uli_defs.m4			\
							\
			gen_kern_args.m4		\
			gpu_args.m4			\
			gpu_special_defs.m4		\
			gen_gpu_calls.m4		\
			gen_kernel_calls.m4		\
			slow_len.m4			\
			slow_incs.m4			\
			slow_vars.m4			\
			eqsp_incs.m4			\
			host_typed_call_defs.m4		\
			gen_host_calls.m4		\
			gen_typtbl.m4			\
			gen_func_array.m4		\
							\
			all_same_prec_vec.m4		\
			all_vec.m4			\
							\
			new_conv.m4			\
			math_funcs.m4			\
			fft_funcs.m4			\
			cpx_vec.m4			\
			signed_vec.m4			\
			unsigned_vec.m4			\
			intvec.m4			\
			int_bit_vec.m4			\
			gen_bit_calls.m4			\
			gen_int_calls.m4			\
			gen_uint_calls.m4			\
			gen_float_calls.m4			\
			gen_mixed_float_calls.m4			\
			gen_mixed_int_calls.m4			\
			gen_mixed_uint_calls.m4			\
			gpu_call_utils.m4			\
							\
			real_args.m4			\
			cpx_args.m4			\
			quat_args.m4			\
			report_args.m4			\
			flush_output.m4			\
							\
			gen_kern_args.m4			\
			sp_func_defs.m4			\
			dp_func_defs.m4			\
			gpu_host_call_defs.m4			\
			cpu_host_call_defs.m4			\
			host_calls_special.m4			\
			xfer_args.m4			\
							\
			fast_defs.m4			\
			flen_defs.m4			\
			eqsp_defs.m4			\
			fast_eqsp_defs.m4			\
			slow_defs.m4			\
			fast_test.m4			\
			gen_entries.m4			\
			host_fft_funcs.m4			\
							\
			asc_classes.c			\
			bitmap_ops.c			\
			conv_vec.c			\
			dispatch_tbl.c			\
			slow_len.h			\
							\
			vec_func.h			\
			vecgen.h			\
			obj_args.h			\
			ocl_veclib_prot.h		\
			fftsupp.h			\
			dim3.h				\
			dim5.h

