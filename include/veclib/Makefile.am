
all:	vl2_veclib_prot.h ocl_veclib_prot.h cu2_veclib_prot.h

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
			m4 < cu2_veclib_prot.m4 > cu2_veclib_prot.h

vl2_veclib_prot.h:	vl2_veclib_prot.m4 vl2_port.m4 gen_port.m4 vl2_func_prot.m4
			m4 < vl2_veclib_prot.m4 > vl2_veclib_prot.h

ocl_veclib_prot.h:	ocl_veclib_prot.m4 ocl_port.m4 gen_port.m4 ocl_func_prot.m4
			m4 < ocl_veclib_prot.m4 > ocl_veclib_prot.h

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
							\
			asc_classes.c			\
			bitmap_ops.c			\
			conv_vec.c			\
			dispatch_tbl.c			\
			slow_len.h			\
							\
			ocl_veclib_prot.h		\
			dim3.h				\
			dim5.h

