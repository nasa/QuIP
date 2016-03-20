all:	libvec.a
	cp libvec.a ../../libs

noinst_LIBRARIES = libvec.a

libvec_a_CPPFLAGS = -I$(top_srcdir)/include -D_GNU_SOURCE -Wall -Wmissing-prototypes
# -D__STRICT_ANSI__  this breaks u_long in sys/types.h...

#				sampmenu.c	\
#

#				bm_funcs.c

#				convert.c

libvec_a_SOURCES =				\
				sampmenu.c	\
				vec_args.c	\
				vec_call.c	\
				obj_args.c	\
				warmenu.c	\
				vectbl.c	\
				vl_menu.c	\
				vec_chn.c	\
				chn_menu.c	\
				cksiz.c		\
				cpu_flags.c	\
				lin_util.c	\
				bitmap.c	\
				vec_func.c	\
				dispatch.c	\
				platform.c	\
				platform_menu.c	\
				vec_chain.c


#				entries.c	\
#				fast_sp.c
#				typtbl.c	\
#				spfft.c		\
#				dpfft.c		\
#				sp_funcs.c	\
#				by_funcs.c	\
#				uby_funcs.c	\
#				in_funcs.c	\
#				uin_funcs.c	\
#				di_funcs.c	\
#				udi_funcs.c	\
#				li_funcs.c	\
#				uli_funcs.c	\
#				ubyin_funcs.c	\
#				uindi_funcs.c	\
#				inby_funcs.c	\
#				spdp_funcs.c	\
#				inby_funcs.c	\
#				simd_funcs.c	\
#				dp_funcs.c

# included files (not compiled themselves

EXTRA_libvec_a_SOURCES =			\
				bit_vec.c	\
				conv_vec.c	\
				linear.c

#				bm_prot.h
#				by_prot.h
#				di_prot.h
#				dp_prot.h
#				fast_prot.h
#				fp_prot.h
#				conv_prot.h
#				in_prot.h
#				inby_prot.h
#				li_prot.h
#				sp_prot.h
#				spdp_prot.h
#				uby_prot.h
#				ubyin_prot.h
#				udi_prot.h
#				uin_prot.h
#				uindi_prot.h
#				uli_prot.h

noinst_HEADERS =				\
				bitmap.h	\
				calling_args.h	\
				convert.h	\
				dp_common.h	\
				fast_sp.h	\
				method_prot.h	\
				my_cpuid.h	\
				new_ops.h	\
				nv_undefs.h	\
				nvf.h		\
				nvproto.h	\
				os_check.h	\
				sp_common.h	\
				vec_chain.h	\
				veclib_prot.h


fast_sp.s:	fast_sp.c
	$(CC) -I../../include -I../.. $(CPPFLAGS) $(CFLAGS) -S fast_sp.c

