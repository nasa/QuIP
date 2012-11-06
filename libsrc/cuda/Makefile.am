
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

NVCCFLAGS =	@NVCC_ARCH_FLAGS@ $(CPPFLAGS) -I/usr/local/cuda/include -I../.. -I$(top_srcdir)/include -DHAVE_CONFIG_H

.cu.o:	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

# The dependency building stuff doesn't seem to know about .cu files...

CU_DEPS =	host_call_utils.h			\
		host_calls.h				\
		gpu_call_utils.h			\
		my_cuda.h				\
		my_vector_functions.h			\
		../../include/veclib/fast_test.h	\
		../../include/veclib/xfer_args.h	\
		../../include/cuda_supp.h 		\
		../../config.h 				\
		../../include/quip_config.h 		\
		../../include/quip_version.h 		\
		../../include/vecgen.h

host_calls.h:	host_call_utils.h ../../include/veclib/fast_test.h	\
		../../include/veclib/xfer_args.h
	touch host_calls.h

myproject.cu:	myproject_kernel.cu					\
		$(CU_DEPS)						\
		all_prot.h float_prot.h int_prot.h signed_prot.h	\
		all_funcs.h bit_prot.h					\
		undefs.h undefs2.h					\
		../../include/veclib/intvec.c				\
		../../include/veclib/all_vec.c				\
		../../include/veclib/cpx_vec.c				\
		../../include/veclib/signed_vec.c			\
		../../include/veclib/unsigned_vec.c			\
		../../include/veclib/math_funcs.c			\
		gpu_all.cu gpu_int.cu gpu_signed.cu gpu_bit.cu
	touch myproject.cu

conversions.cu:	$(CU_DEPS)
	touch conversions.cu

fill.cu:	$(CU_DEPS)
	touch fill.cu

cuda_centroid.cu:	gpu_cent.cu		\
			$(CU_DEPS)
	touch cuda_centroid.cu

yuv2rgb.cu:	$(CU_DEPS)
	touch yuv2rgb.cu


myproject.o:	myproject.cu
	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

yuv2rgb.o:	yuv2rgb.cu
	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

conversions.o:	conversions.cu
	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

fill.o:	fill.cu
	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

cuda_centroid.o:	cuda_centroid.cu
	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

noinst_LIBRARIES = libcuda.a

#CC = g++

libcuda_a_CPPFLAGS = -I$(top_srcdir)/include -D_GNU_SOURCE -Wall

if HAVE_NVCC

libcuda_a_CPPFLAGS += -I/usr/local/cuda/include $(NVCPPFLAGS)

# This file appears to be here when we build in the cvs tree,
# but not in the distribution!?!?
# We'll comment this out for now...
#clean-local:
#	rm myproject.linkinfo

endif

# -D__STRICT_ANSI__  this breaks u_long in sys/types.h...

# The headers section includes the true headers (.h), plus
# code that is included in other .c or .cu files...

noinst_HEADERS =		my_cuda.h		\
				my_vector_functions.h	\
				cuda_viewer.h		\
				all_prot.h		\
				bit_prot.h		\
				int_prot.h		\
				undefs.h		\
				undefs2.h		\
				signed_prot.h		\
				float_prot.h		\
				prot_defs.h		\
				hcall_defs.h		\
				call_defs.h		\
				host_calls.h		\
				host_call_utils.h	\
				gpu_calls.h		\
				gpu_call_utils.h	\
				myproject_kernel.cu	\
				gpu_bit.cu		\
				gpu_all.cu		\
				gpu_int.cu		\
				gpu_cent.cu		\
				gpu_signed.cu		\
				conversions.h		\
				all_funcs.h		\
				menu_calls.h

#				bit_prec_funcs.cu	\
#				all_prec_funcs.cu	\
#				float_prec_funcs.cu	\
#				int_prec_funcs.cu	\
#				signed_prec_funcs.cu

libcuda_a_SOURCES =					\
				cuda.cpp		\
				cuda_npp.cpp		\
				cuda_curand.cpp		\
				cuda_viewer.cpp		\
				simpleCUBLAS.cpp	\
				deviceQuery.cpp		\
				cuda_func_tbl.cpp	\
				cuda_error.c		\
				cuda_streams.c		\
				cuda_menu.cpp

# This business seems to break distclean, the .cu files don't get removed...

if HAVE_NVCC

libcuda_a_SOURCES += myproject.cu conversions.cu cuda_centroid.cu fill.cu yuv2rgb.cu

else

libcuda_a_SOURCES += fakit.cpp

noinst_HEADERS += myproject.cu conversions.cu cuda_centroid.cu fill.cu yuv2rgb.cu

endif


