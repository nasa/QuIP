
noinst_LIBRARIES = libnewvec.a

libnewvec_a_CPPFLAGS = -I$(top_srcdir)/include -D_GNU_SOURCE -Wall
# -D__STRICT_ANSI__  this breaks u_long in sys/types.h...

libnewvec_a_SOURCES =				\
				vec_args.c	\
				vec_call.c	\
				vectbl.c	\
				convert.c	\
				entries.c	\
				bm_funcs.c	\
				spdp_funcs.c	\
				spfft.c		\
				dpfft.c		\
				obj_args.c	\
				warmenu.c	\
				vl_menu.c	\
				wrap.c		\
				vec_chn.c	\
				fftsupp.c	\
				chn_menu.c	\
				size.c		\
				lin_util.c	\
				scale.c		\
				sampmenu.c	\
				dispatch.c	\
				cksiz.c		\
				cpu_flags.c	\
				by_funcs.c	\
				uby_funcs.c	\
				in_funcs.c	\
				uin_funcs.c	\
				di_funcs.c	\
				udi_funcs.c	\
				li_funcs.c	\
				uli_funcs.c	\
				ubyin_funcs.c	\
				uindi_funcs.c	\
				inby_funcs.c	\
				sp_funcs.c	\
				dp_funcs.c	\
				bitmap.c	\
				typtbl.c	\
				fast_sp.c



# included files (not compiled themselves

EXTRA_libnewvec_a_SOURCES =			\
				inby_funcs.c	\
				bit_vec.c	\
				conv_vec.c	\
				vfft.c		\
				linear.c

noinst_HEADERS =				\
				new_ops.h	\
				spdp_prot.h	\
				inby_prot.h	\
				ubyin_prot.h	\
				uindi_prot.h	\
				bm_prot.h	\
				bitmap.h	\
				all_prot.h	\
				by_prot.h	\
				uby_prot.h	\
				in_prot.h	\
				uin_prot.h	\
				di_prot.h	\
				li_prot.h	\
				udi_prot.h	\
				uli_prot.h	\
				sp_prot.h	\
				dp_prot.h	\
				sp_common.h	\
				dp_common.h	\
				my_cpuid.h	\
				fast_sp.h	\
				fp_prot.h	\
				convert.h	\
				new_chains.h	\
				nvf.h		\
				vec_args.h	\
				nvproto.h	\
				method_prot.h	\
				conv_prot.h	\
				calling_args.h	\
				nv_undefs.h	\
				os_check.h


fast_sp.s:	fast_sp.c
	$(CC) -I../../include -I../.. $(CPPFLAGS) $(CFLAGS) -S fast_sp.c

