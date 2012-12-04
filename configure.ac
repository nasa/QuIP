#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

# BUG - need to set or clear HAVE_VBL
# BUG - should we set MY_INTR???  (builtin.c, do_cmd.c)

# How can we turn off optimization??

#AC_PREREQ([2.63])
AC_INIT([quip], [0.3], [jeffrey.b.mulligan@nasa.gov])


AC_CONFIG_SRCDIR([src/quip_main.c])
AC_CONFIG_HEADERS([config.h])

# hack to locate expat/iconv in /usr/local on BSD systems
#CPPFLAGS="$CPPFLAGS -I/usr/local/include"
#LIBS="$LIBS -L/usr/local/lib"

# Check for odd include directories
AC_CHECK_FILE([/usr/X11/include],[CPPFLAGS="$CPPFLAGS -I/usr/X11/include"])
AC_CHECK_FILE([/sw/include],[CPPFLAGS="$CPPFLAGS -I/sw/include"])

# AC_FUNC_MALLOC causes problems on Mac OSX, defines malloc to be rpl_malloc!?
AC_FUNC_MALLOC

# CUDA checks

AC_CHECK_FILE([/usr/local/cuda/include],[CPPFLAGS="$CPPFLAGS -I/usr/local/cuda/include"])

# Normally the SDK will be installed in the user's home directory...
# But in the nfs environment where we might have different
# versions on different systems, this can be problematic.
# The solution here is to put a symbolic link in /usr/local/cuda.


AC_CHECK_FILE([/usr/local/cuda/NVIDIA_GPU_Computing_SDK],[gpu_sdk_dir="/usr/local/cuda/NVIDIA_GPU_Computing_SDK"],[gpu_sdk_dir="no"])
if test "x${gpu_sdk_dir}" = xno; then
  # BUG - jbm creating a symbolic link with an underscore replacing the space,
  # because the space screwed up some of the scripts later on...
  AC_CHECK_FILE([/Developer/GPU_Computing],[gpu_sdk_dir="/Developer/GPU_Computing"],[gpu_sdk_dir="no"])
  if test "x${gpu_sdk_dir}" = xno; then
    AC_CHECK_FILE([/usr/local/cuda/NVIDIA_CUDA_SDK],[cuda_sdk_dir="/usr/local/cuda/NVIDIA_CUDA_SDK"],[cuda_sdk_dir="no"])
    if test "x${cuda_sdk_dir}" = xno; then
      AC_CHECK_FILE([$HOME/NVIDIA_GPU_Computing_SDK],[gpu_sdk_dir="$HOME/NVIDIA_GPU_Computing_SDK"],[gpu_sdk_dir="no"])
    fi
  fi
fi

if test "x${gpu_sdk_dir}" = xno; then
  if test "x${cuda_sdk_dir}" = xno; then
    have_nv_sdk=no
  else
    nv_sdk_inc_dir="$cuda_sdk_dir/common/inc"
    CPPFLAGS="$CPPFLAGS -I$nv_sdk_inc_dir"
    NVCCFLAGS="-I$nv_sdk_inc_dir"
    have_nv_sdk=yes
    AC_CHECK_FILE([$cuda_sdk_dir/common/lib/linux],
    			[have_cuda_sdk_libdir=yes],
    			[have_cuda_sdk_libdir=no])
    if test "x${have_cuda_sdk_libdir}" = xyes; then
      LDFLAGS="$LDFLAGS -L$cuda_sdk_dir/common/lib/linux"
    fi
  fi
else
  nv_sdk_inc_dir="$gpu_sdk_dir/C/common/inc"
  CPPFLAGS="$CPPFLAGS -I$nv_sdk_inc_dir"
  NVCCFLAGS="-I$nv_sdk_inc_dir"
  have_nv_sdk=yes
  # BUG - when we found $gpu_sdk_dir, we should know enough to know whether
  # we are on a mac or not...
  AC_CHECK_FILE([$gpu_sdk_dir/shared/lib/linux],
    			[have_cuda_sdk_libdir=yes],
    			[have_cuda_sdk_libdir=no])
  if test "x${have_cuda_sdk_libdir}" = xyes; then
    LDFLAGS="$LDFLAGS -L$gpu_sdk_dir/shared/lib/linux"
  else
    AC_CHECK_FILE([$gpu_sdk_dir/shared/lib/darwin],
    			[have_cuda_sdk_libdir=yes],
    			[have_cuda_sdk_libdir=no])
    if test "x${have_cuda_sdk_libdir}" = xyes; then
      LDFLAGS="$LDFLAGS -L$gpu_sdk_dir/shared/lib/darwin"
    fi
  fi
fi

# Check for odd lib directories
AC_CHECK_FILE([/usr/X11/lib],[LDFLAGS="$LDFLAGS -L/usr/X11/lib"])
AC_CHECK_FILE([/sw/lib],[LDFLAGS="$LDFLAGS -L/sw/lib"])
AC_CHECK_FILE([/usr/X11R6/lib],[LDFLAGS="$LDFLAGS -L/usr/X11R6/lib"])
AC_CHECK_FILE([/usr/lib64],[LDFLAGS="$LDFLAGS -L/usr/lib64"])
AC_CHECK_FILE([/usr/local/lib],[LDFLAGS="$LDFLAGS -L/usr/local/lib"])
AC_CHECK_FILE([/usr/local/cuda/lib64],[have_usr_local_cuda_lib64=yes],
		[have_usr_local_cuda_lib64=no])
if test "x${have_usr_local_cuda_lib64}" = xyes; then
  LDFLAGS="$LDFLAGS -L/usr/local/cuda/lib64"
else
  AC_CHECK_FILE([/usr/local/cuda/lib],[LDFLAGS="$LDFLAGS -L/usr/local/cuda/lib"])
fi

AC_CHECK_FILE([/usr/local/include],[CPPFLAGS="$CPPFLAGS -I/usr/local/include"])

# check for das1602
# BUG - /dev/adc0 could represent a different type of interface,
# but now this is the only one we support so we don't worry...
AC_CHECK_FILE([/dev/adc0],[have_das1602=yes],[have_das1602=no])

# check for v4l2
AC_CHECK_FILE([/dev/video0],[have_v4l2=yes],[have_v4l2=no])

# check for pic
AC_CHECK_FILE([/dev/pic],[have_pic=yes],[have_pic=no])


# check for meteor
AC_CHECK_FILE([/dev/meteor0],[have_meteor=yes],[have_meteor=no])

# check for symbolic link to visca tty port
AC_CHECK_FILE([/dev/visca],[have_visca=yes],[have_visca=no])

# check for /proc/cpuinfo, a linux-ism
AC_CHECK_FILE([/proc/cpuinfo],[have_proc_cpuinfo=yes],[have_proc_cpuinfo=no])

# check for frame buffer device
AC_CHECK_FILE([/dev/fb0],[have_fb_dev=yes],[have_fb_dev=no])

# check for parallel port
AC_CHECK_FILE([/dev/parport0],[have_parport=yes],[have_parport=no])

# BUG g++ required for cuda even when HW not present!?

# Checks for programs.
AC_PROG_YACC
AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_RANLIB
AM_PROG_CC_C_O

# We need to set something, or report something in the doc Makefile
# if texi2dvi is not present!?
# When the program is not present, it puts 'texi2dvi not found'
# in the variable TEXI2DVI...
AC_CHECK_PROG(TEXI2DVI,texi2dvi,[/usr/bin/texi2dvi],[texi2dvi not found],[/usr/bin])

if test "x$TEXI2DVI" == "xtexi2dvi not found"; then
  have_texi2dvi=no
else
  have_texi2dvi=yes
fi


AC_ARG_VAR([NVCC], [nvcc compiler to use])
AC_PATH_PROG([NVCC], [nvcc], [no])
working_nvcc=no
if test "x$NVCC" != "xno"
then
    AC_MSG_CHECKING([whether nvcc works])
    cat>conftest.cu<<EOF
    __global__ void dummy_func(float *v) {
		int index1;
		index1 = blockIdx.x * blockDim.x + threadIdx.x;
	   v[[index1]]=0;
       }
       int main(int ac,char **av)
       {
	void *cuda_mem;
	int n_blocks=8, n_threads_per_block=128;
	cuda_mem = (float *) cudaMalloc(&cuda_mem, sizeof(float) * n_threads_per_block * n_blocks );
	dummy_func<<< n_blocks, n_threads_per_block >>>((float *)cuda_mem);
	}
EOF
       cp conftest.cu /tmp/conftest.cu
#       if $NVCC conftest.cu && test_whether_output_files_are_ok
       if $NVCC conftest.cu
       then
               working_nvcc=yes
       fi
       rm -f conftest.cu a.out
       AC_MSG_RESULT([$working_nvcc])
fi

# Create a variable we can test in Makefile.am:
AM_CONDITIONAL([HAVE_NVCC], [test "x$working_nvcc" = "xyes"])

# determine cuda compute capability
if test "x$working_nvcc" == xyes; then
  AC_MSG_CHECKING([for CUDA-capable device])
  cat>conftest.c<<EOF
#include <stdio.h>
#include <cuda_runtime.h>
int main() {
    int deviceCount, device;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess) 
        deviceCount = 0;
    /* machines with no GPUs can still report one emulation device */
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999){ /* 9999 means emulation only */
            ++gpuDeviceCount;
            printf("%d.%d\n",
		properties.major,properties.minor);
	    return 0;	// just print out for first device
	}
    }
    printf("0.0\n");

    /* don't just return the number of gpus, because other runtime cuda
       errors can also yield non-zero return values */
    if (gpuDeviceCount > 0)
        return 0; /* success */
    else
        return 1; /* failure */
}
EOF

  # BUG should have variables with inc and lib dirs?
  # LDFLAGS instead of hard-coded -L/usr/local/cuda/lib64
  # no lib64 on mac
  gcc -o /tmp/gpu_test -I/usr/local/cuda/include $LDFLAGS conftest.c -lcudart
  comp_cap=`/tmp/gpu_test`
  rm -f conftest.c /tmp/gpu_test
  AC_MSG_RESULT([compute capability $comp_cap])
  arch_set=no
  if test "x${comp_cap}" = x1.0; then
    AC_DEFINE([CUDA_COMP_CAP],10,[CUDA Compute Capability 1.0])
    AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_10'])
    arch_set=yes
  fi
  if test "x${comp_cap}" = x1.1; then
    AC_DEFINE([CUDA_COMP_CAP],11,[CUDA Compute Capability 1.1])
    AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_11'])
    arch_set=yes
  fi
  if test "x${comp_cap}" = x1.2; then
    AC_DEFINE([CUDA_COMP_CAP],12,[CUDA Compute Capability 1.2])
    AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_12'])
    arch_set=yes
  fi
  if test "x${comp_cap}" = x1.3; then
    AC_DEFINE([CUDA_COMP_CAP],13,[CUDA Compute Capability 1.3])
    AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_13'])
    arch_set=yes
  fi
  if test "x${comp_cap}" = x2.0; then
    AC_DEFINE([CUDA_COMP_CAP],20,[CUDA Compute Capability 2.0])
    AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_20'])
    arch_set=yes
  fi
  if test "x${comp_cap}" = x2.1; then
    AC_DEFINE([CUDA_COMP_CAP],21,[CUDA Compute Capability 2.1])
    AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_21'])
    arch_set=yes
  fi
  if test "x${comp_cap}" = x0.0; then
    AC_DEFINE([CUDA_COMP_CAP],10,[CUDA Compute Capability 0.0])
    arch_set=yes
  fi
  if test "x${arch_set}" = xno; then
    AC_SUBST(NVCC_ARCH_FLAGS,[])
    AC_MSG_WARN([
  ------------------------------------------------------
  Missing case for CUDA compute capability $comp_cap
  in configure script - Please fix in configure.ac,
  or notify the maintainer (jeffrey.b.mulligan@nasa.gov).
  ------------------------------------------------------])
  fi
fi	# end working_nvcc loop

# User settable features:

# Enable async by default:
AC_ARG_ENABLE([async-exec],
	[AS_HELP_STRING([--disable-async-exec],
		[disable asynchronous execution @<:@default: no@:>@])],
	[ async_exec=${enableval}], [async_exec=yes])

# OR Disable by default:
#AC_ARG_ENABLE(([async-exec],
#	[ --enable-async-exec	enable async exec ],
#	[ async_exec=${enableval}], [async_exec=no])


# disable thread-safe query by default

AC_ARG_ENABLE([thread-safe-query],
	[AS_HELP_STRING([--enable-thread-safe-query],
		[enable thread-safe interpreter input @<:@default: yes@:>@])],
	[ thread_safe_query=${enableval}], [thread_safe_query=no])


# This flag causes escape to custom assembler code - not tested with
# optimizer, probably better to get the compiler to do this?

# disable sse extensions by default (although compiler may use them?)

AC_ARG_ENABLE([sse-extensions],
	[AS_HELP_STRING([--enable-sse-extensions],
		[enable use of SSE extensions@<:@default: yes@:>@])],
	[ use_sse=${enableval}], [use_sse=no])

# disable multiprocessing by default

AC_ARG_ENABLE([n-processors],
	[AS_HELP_STRING([--enable-n-processors=N],
		[enable multiprocessing with N processors @<:@default: 2@:>@])],
	[ n_processors=${enableval}], [n_processors=no])

# Devices

# Enable v4l2 by default:
AC_ARG_ENABLE([use-v4l2],
	[AS_HELP_STRING([--disable-v4l2],[disable V4L2 video subsystem @<:@default: no@:>@])],
	[ use_v4l2=${enableval}], [use_v4l2=yes])

# Enable meteor by default:
AC_ARG_ENABLE([use-meteor],
	[AS_HELP_STRING([--disable-meteor],[disable Matrox meteor video subsystem @<:@default: no@:>@])],
	[ use_meteor=${enableval}], [use_meteor=yes])

# Enable aio by default:
AC_ARG_ENABLE([use-das1602],
	[AS_HELP_STRING([--disable-das1602],[disable Measurement Computing analog I/O subsystem @<:@default: no@:>@])],
	[ use_das1602=${enableval}], [use_das1602=yes])

# Enable visca by default:
AC_ARG_ENABLE([use-visca],
	[AS_HELP_STRING([--disable-visca],[disable Sony VISCA protocol support @<:@default: no@:>@])],
	[ use_visca=${enableval}], [use_visca=yes])


# Enable fb_dev by default:
AC_ARG_ENABLE([use-fb-dev],
	[AS_HELP_STRING([--disable-fb-dev],[disable /dev/fb subsystem @<:@default: no@:>@])],
	[ use_fb_dev=${enableval}], [use_fb_dev=yes])

# disable PIC microcontroller LED controller by default

AC_ARG_ENABLE([pic-led-controller],
	[AS_HELP_STRING([--enable-pic-led-controller],
		[enable PIC LED controller @<:@default: yes@:>@])],
	[ use_pic=${enableval}], [use_pic=no])


# Enable tty_ctl by default:
AC_ARG_ENABLE([use_tty_ctl],
	[AS_HELP_STRING([--disable-tty-ctl],[disable low level tty control @<:@default: no@:>@])],
	[ use_tty_ctl=${enableval}], [use_tty_ctl=yes])

# Enable viewers by default:
AC_ARG_ENABLE([use_viewers],
	[AS_HELP_STRING([--disable-viewers],[disable graphical viewing window support @<:@default: no@:>@])],
	[ use_viewers=${enableval}], [use_viewers=yes])

# Enable history by default:
AC_ARG_ENABLE([use_history],
	[AS_HELP_STRING([--disable-history],[disable history features @<:@default: no@:>@])],
	[ use_history=${enableval}], [use_history=yes])

# Enable rt_sched by default:
AC_ARG_ENABLE([use_rt_sched],
	[AS_HELP_STRING([--disable-rt-sched],[disable real-time scheduler control @<:@default: no@:>@])],
	[ use_rt_sched=${enableval}], [use_rt_sched=yes])

# Enable gui stuff by default:
AC_ARG_ENABLE([use_motif_gui],
	[AS_HELP_STRING([--disable-gui],[disable GUI features @<:@default: no@:>@])],
	[ use_motif_gui=${enableval}], [use_motif_gui=yes])

# Enable cstepit stuff by default:
AC_ARG_ENABLE([use_stepit],
	[AS_HELP_STRING([--disable-stepit],[disable STEPIT optimization @<:@default: no@:>@])],
	[ use_stepit=${enableval}], [use_stepit=yes])

# Enable numrec support by default:
AC_ARG_ENABLE([use_numrec],
	[AS_HELP_STRING([--disable-numrec],[disable Numerical Recipes support@<:@default: no@:>@])],
	[ use_numrec=${enableval}], [use_numrec=yes])

# Enable sound support by default:
AC_ARG_ENABLE([use_sound],
	[AS_HELP_STRING([--disable-sound],[disable sound support@<:@default: no@:>@])],
	[ use_sound=${enableval}], [use_sound=yes])

# Enable debug by default:
AC_ARG_ENABLE([use_debug],
	[AS_HELP_STRING([--disable-debug],[disable debug features @<:@default: no@:>@])],
	[ use_debug=${enableval}], [use_debug=yes])

## Enable getbuf by default:
#AC_ARG_ENABLE([use_getbuf],
#	[AS_HELP_STRING([--disable-getbuf],[disable internal memory manager @<:@default: no@:>@])],
#	[ use_getbuf=${enableval}], [use_getbuf=yes])

# OR Disable by default:
#AC_ARG_ENABLE(([async-exec],
#	[ --enable-async-exec	enable async exec ],
#	[ async_exec=${enableval}], [async_exec=no])

# Disable getbuf by default:
AC_ARG_ENABLE([use_getbuf],
	[AS_HELP_STRING([--enable-getbuf],[enable internal memory manager @<:@default: yes@:>@])],
	[ use_getbuf=${enableval}], [use_getbuf=no])

# Enable cautious checks by default:
AC_ARG_ENABLE([use_cautious],
	[AS_HELP_STRING([--disable-cautious],[disable cautious checks @<:@default: no@:>@])],
	[ use_cautious=${enableval}], [use_cautious=yes])

# Enable avi support by default:
AC_ARG_ENABLE([use_avi],
	[AS_HELP_STRING([--disable-avi],[disable ffmpeg avi support @<:@default: no@:>@])],
	[ use_avi=${enableval}], [use_avi=yes])

# Enable cuda by default:
AC_ARG_ENABLE([use_cuda],
	[AS_HELP_STRING([--disable-cuda],[disable GPU operations with CUDA @<:@default: no@:>@])],
	[ use_cuda=${enableval}], [use_cuda=yes])

## Disable debug by default:
#AC_ARG_ENABLE(([use_debug],
#	[AS_HELP_STRING([--enable-debug],[enable cautious checking @<:@default: yes@:>@])],
#	[ use_debug=${enableval}], [use_debug=no])

# this test bombs if we perform it after testing for libGLEW !?!?!?
AC_CHECK_SIZEOF([long])

# Checks for libraries.

AC_SEARCH_LIBS([memalign],[],[have_memalign=yes],[have_memalign=no])

AC_SEARCH_LIBS([adjtimex],[],[have_adjtimex=yes],[have_adjtimex=no])

#have_llseek=no
#AC_SEARCH_LIBS([_llseek],[],[have_llseek=yes])

# mandatory libraries
# -lm

AC_SEARCH_LIBS([cos],[m],[have_math=yes],[have_math=no])
AC_SEARCH_LIBS([cosf],[m],[have_fmath=yes],[have_fmath=no])
AC_SEARCH_LIBS([cosl],[m],[have_lmath=yes],[have_lmath=no])
AC_SEARCH_LIBS([round],[m],[have_round=yes],[have_round=no])

# should always have this on unix, but not on iOS?
have_stat=no
AC_SEARCH_LIBS([stat],[c],[have_stat=yes])

# optional libraries
# -lX11		X
# -lpthread	pthreads
# -lXm		motif
# -lXt		X intrinsics, needed by Xm
# -lnumrec	numerical recipes
# -lcurl	www lib
# -lasound	ALSA sound library
# -lmatio	matlab i/o
# -ltiff	TIFF image library
# -lpng		PNG image library

AC_SEARCH_LIBS([tgetent],[termcap],[have_libtermcap=yes],[have_libtermcap=no])

AC_SEARCH_LIBS([cblas_cgemv],[gslcblas],[have_libgslcblas=yes],[have_libgslcblas=no])

AC_SEARCH_LIBS([gsl_vector_alloc],[gsl],[have_libgsl=yes],[have_libgsl=no])

# On a fresh CentOS 5.6 installation, /usr/lib64/libX11.so.6 exists (a link),
# but no file /usr/lib64/libX11.so - without the latter, the linker does
# not find a library when given -lX11 - the devel package needs to
# be installed for this link to be created, and for the header files
# to be installed...

AC_SEARCH_LIBS([XOpenDisplay],[X11],[have_libx11=yes],[have_libx11=no])
AC_SEARCH_LIBS([XShmCreateImage],[Xext],[have_xext=yes],[have_xext=no])


AC_SEARCH_LIBS([glVertex2d],[GL],[have_libgl=yes],[have_libgl=no])
AC_SEARCH_LIBS([glXGetVideoSyncSGI],[GL],[have_sgisync=yes],[have_sgisync=no])
AC_SEARCH_LIBS([gluLookAt],[GLU],[have_libglu=yes],[have_libglu=no])
AC_SEARCH_LIBS([glutInit],[glut],[have_libglut=yes],[have_libglut=no])

AC_SEARCH_LIBS([pthread_create],[pthread],[have_pthreads=yes],[have_pthreads=no])
AC_SEARCH_LIBS([sched_setscheduler],[],[have_sched=yes],[have_sched=no])

# On Mac, we need Xt (not needed on linux???), but the order matters!
# libXm has to come first, but if it comes first here, it comes later
# in the link line...
AC_SEARCH_LIBS([XtManageChild],[Xt],[have_libxt=yes],[have_libxt=no])
AC_SEARCH_LIBS([XmCreateForm],[Xm],[have_libmotif=yes],[have_libmotif=no])

AC_SEARCH_LIBS([float_svdcmp],[numrec],[have_numrec=yes],[have_numrec=no])
AC_SEARCH_LIBS([curl_global_init],[curl],[have_curl=yes],[have_curl=no])
AC_SEARCH_LIBS([snd_pcm_open],[asound],[have_alsa=yes],[have_alsa=no])
AC_SEARCH_LIBS([inflate],[z],[have_libz=yes],[have_libz=no])
AC_SEARCH_LIBS([Mat_Open],[matio],[have_libmatio=yes],[have_libmatio=no])
AC_SEARCH_LIBS([TIFFGetField],[tiff],[have_libtiff=yes],[have_libtiff=no])
AC_SEARCH_LIBS([png_read_image],[png],[have_libpng=yes],[have_libpng=no])

# later libs come first... OpenCV seems to include a copy of libjpeg,
# but only if built when libjpeg was not already installed!

AC_SEARCH_LIBS([jpeg_start_decompress],[jpeg],[have_libjpeg=yes],[have_libjpeg=no])

AC_SEARCH_LIBS([qt_read_scanlines],[qt],[have_libqt=yes],[have_libqt=no])
AC_SEARCH_LIBS([av_crc],[avutil],[have_libavutil=yes],[have_libavutil=no])
AC_SEARCH_LIBS([sws_scale],[swscale],[have_libswscale=yes],[have_libswscale=no])
AC_SEARCH_LIBS([avcodec_open],[avcodec],[have_libavcodec=yes],[have_libavcodec=no])
#have_libavutil=no
#AC_SEARCH_LIBS([av_open_input_file],[avutil],[have_libavutil=yes])
AC_SEARCH_LIBS([BZ2_bzDecompressInit],[bz2],[have_libbz2=yes],[have_libbz2=no])
AC_SEARCH_LIBS([av_open_input_file],[avformat],[have_libavformat=yes],[have_libavformat=no])
AC_SEARCH_LIBS([initscr],[ncurses],[have_ncurses=yes],[have_ncurses=no])

AC_SEARCH_LIBS([cvSetZero],[opencv_core cxcore],[have_opencvcore=yes],[have_opencvcore=no])
AC_SEARCH_LIBS([cvContourArea],[opencv_imgproc cv],[have_opencvimgproc=yes],[have_opencvimgproc=no])
AC_SEARCH_LIBS([cvHaarDetectObjects],[opencv_objdetect],[have_opencvobjdetect=yes],[have_opencvobjdetect=no])
AC_SEARCH_LIBS([cvSaveImage],[opencv_highgui highgui],[have_opencvhighgui=yes],[have_opencvhighgui=no])

AC_SEARCH_LIBS([dv_decode_full_frame],[dv],[have_libdv=yes],[have_libdv=no])

# _new is a function...
AC_SEARCH_LIBS([dc1394_new],[dc1394],[have_libdc1394=yes],[have_libdc1394=no])

# cuda libraries
AC_SEARCH_LIBS([cudaUnbindTexture],[cudart],[have_libcudart=yes],[have_libcudart=no])
# If we have the cudart lib, then we can determine the compute
# capability of the device.

AC_SEARCH_LIBS([cufftPlan1d],[cufft],[have_libcufft=yes],[have_libcufft=no])
AC_SEARCH_LIBS([cublasGetError],[cublas],[have_libcublas=yes],[have_libcublas=no])
AC_SEARCH_LIBS([curandCreateGenerator],[curand],[have_libcurand=yes],[have_libcurand=no])
AC_SEARCH_LIBS([nppiErode_8u_C1R],[npp],[have_libnpp=yes],[have_libnpp=no])
# BUG need to determine whether we're compiling for 64 bit!
AC_SEARCH_LIBS([glewInit],[GLEW_x86_64 GLEW],[have_libglew=yes],[have_libglew=no])
#if test "x${have_libglew}" = xno; then
#  AC_SEARCH_LIBS([glewInit],[GLEW],[have_libglew=yes],[have_libglew=no])
#fi

# Checks for header files.

AC_PATH_X
AC_FUNC_ALLOCA

AC_HEADER_STDBOOL

AC_CHECK_HEADERS([ioctl.h])
AC_CHECK_HEADERS([ctype.h])
AC_CHECK_HEADERS([pwd.h])
AC_CHECK_HEADERS([math.h])
AC_CHECK_HEADERS([signal.h])
AC_CHECK_HEADERS([grp.h])
AC_CHECK_HEADERS([time.h])
AC_CHECK_HEADERS([errno.h])
AC_CHECK_HEADERS([asm/errno.h])
AC_CHECK_HEADERS([ncurses.h])
AC_CHECK_HEADERS([curses.h])
AC_CHECK_HEADERS([term.h])
AC_CHECK_HEADERS([assert.h])

# on old SUN
AC_CHECK_HEADERS([rasterfile.h])

# used by usb2000 module
AC_CHECK_HEADERS([usb.h])

# BUG if this header is not present, shouldn't try to use v4l2
AC_CHECK_HEADERS([linux/videodev2.h])

# BUG if this header is not present, shouldn't try to use parport
AC_CHECK_HEADERS([linux/ppdev.h])

# BUG if this header is not present, shouldn't try to use /dev/fb
AC_CHECK_HEADERS([linux/fb.h])

AC_CHECK_HEADERS([asm/types.h])
AC_CHECK_HEADERS([sys/types.h])
AC_CHECK_HEADERS([sys/wait.h])
AC_CHECK_HEADERS([sys/io.h])

AC_CHECK_HEADERS([sched.h],[have_sched_h=yes],[have_sched_h=no])
AC_CHECK_HEADERS([termios.h unistd.h])
AC_CHECK_HEADERS([sys/soundcard.h])
AC_CHECK_HEADERS([sys/timex.h],[have_sys_timex_h=yes],[have_sys_timex_h=no])
AC_CHECK_HEADERS([alsa/asoundlib.h])

AC_CHECK_HEADERS([curl/curl.h])

AC_CHECK_HEADERS([fcntl.h inttypes.h libintl.h limits.h malloc.h])
AC_CHECK_HEADERS([memory.h netdb.h netinet/in.h])
AC_CHECK_HEADERS([stddef.h stdint.h stdlib.h string.h strings.h])
AC_CHECK_HEADERS([ieeefp.h])

AC_CHECK_HEADERS([sys/resource.h])
AC_CHECK_HEADERS([sys/file.h sys/ioctl.h sys/mount.h sys/param.h])
AC_CHECK_HEADERS([sys/filio.h])
AC_CHECK_HEADERS([sys/mman.h])
AC_CHECK_HEADERS([sys/socket.h sys/statvfs.h sys/time.h sys/timeb.h])
AC_CHECK_HEADERS([sys/disklabel.h sys/mount.h])

# used in hips, perr.c...
AC_CHECK_HEADERS([stdarg.h])
AC_CHECK_HEADERS([varargs.h])

AC_CHECK_HEADERS([linux/fd.h linux/fs.h])

AC_CHECK_HEADERS([sys/ipc.h])

# X11 headers
AC_CHECK_HEADERS([X11/Intrinsic.h])
AC_CHECK_HEADERS([X11/Xlib.h])
AC_CHECK_HEADERS([X11/Xutil.h])
AC_CHECK_HEADERS([X11/Xmd.h])
AC_CHECK_HEADERS([sys/shm.h])
AC_CHECK_HEADERS([X11/extensions/XShm.h],[],[],[
#ifdef HAVE_X11_XLIB_H
#include <X11/Xlib.h>
#endif
])

AC_CHECK_HEADERS([Xm/Xm.h])
AC_CHECK_HEADERS([Xm/XmAll.h])

# opengl, GLUT

# on Mac, these headers are in /usr/X11/include...
AC_CHECK_HEADERS([GL/gl.h])

AC_CHECK_HEADERS([GL/glx.h])
AC_CHECK_HEADERS([GL/glu.h])
AC_CHECK_HEADERS([GL/glut.h])
# on wheatstone, this one is in the nvidia distribution...
AC_CHECK_HEADERS([GL/glew.h])

# libdv
AC_CHECK_HEADERS([libdv/dv.h])
AC_CHECK_HEADERS([libdv/dv_types.h])
AC_CHECK_HEADERS([libdv/dv1394.h])

# libdc1394
AC_CHECK_HEADERS([dc1394/dc1394.h])

# OpenCV  headers
# old version?
AC_CHECK_HEADERS([opencv/cvver.h opencv/cv.h opencv/highgui.h])
# newer version?
AC_CHECK_HEADERS([opencv2/core/version.hpp])

# png.h is not where expected on MacOS
AC_CHECK_HEADERS([png.h])

# BUG If this file is not found, then we should unset HAVE_JPEG
AC_CHECK_HEADERS([jpeglib.h],[have_jpeglib_h=yes],[have_jpeglib_h=no])

# cuda headers
# These won't be found if the include directories aren't in CPPFLAGS!?
AC_CHECK_HEADERS([curand.h])
AC_CHECK_HEADERS([nppi.h])

if test $ac_cv_sizeof_long -eq 8; then
  AC_DEFINE([LONG_64_BIT], 1, [64 bit longs])
fi

if test $ac_cv_sizeof_long -eq 4; then
  AC_DEFINE([LONG_32_BIT], 1, [32 bit longs])
fi

#AC_CHECK_SIZEOF([off64_t])

# Automake stuff

# consider silent-rules option, see p. 122
AM_INIT_AUTOMAKE
AM_PROG_CC_C_O

# Checks for typedefs, structures, and compiler characteristics.

AC_TYPE_UID_T
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_MODE_T
AC_TYPE_OFF_T
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T

# Checks for library functions.
AC_FUNC_FORK
AC_FUNC_MKTIME
AC_FUNC_MMAP
AC_FUNC_STRTOD

#AC_CHECK_FUNCS([floor ftime getcwd gethostbyname gettimeofday time sleep drand48 srand48 random srandom rand srand memset pow rint select socket sqrt strchr strdup strerror strrchr strstr strtol])

# check for seek funcs
AC_CHECK_FUNCS([llseek lseek64])
# check for scheduler funcs
AC_CHECK_FUNCS([sched_setscheduler sched_setparam])
# check for general utility funcs
AC_CHECK_FUNCS([popen getcwd memset flock])
AC_CHECK_FUNCS([munmap])
# check for time-related funcs
AC_CHECK_FUNCS([gettimeofday ftime time sleep])
AC_CHECK_FUNCS([alarm])
# check for math funcs
AC_CHECK_FUNCS([floor pow rint sqrt])
# check for string funcs
AC_CHECK_FUNCS([bcopy])
AC_CHECK_FUNCS([strchr strdup strerror strrchr strstr strtol])
AC_CHECK_FUNCS([strncasecmp])
# check for networking funcs
AC_CHECK_FUNCS([gethostbyname select socket])
# check for random number funcs
AC_CHECK_FUNCS([drand48 srand48 random srandom rand srand])

AC_CONFIG_FILES([Makefile
                 macros/Makefile
                 macros/system/Makefile
                 macros/compute/Makefile
                 macros/startup/Makefile
                 macros/data/Makefile
                 macros/view/Makefile
                 macros/numrec/Makefile
                 macros/knox/Makefile
                 macros/visca/Makefile
                 macros/atc/Makefile
                 macros/v4l2/Makefile
                 macros/gui/Makefile
                 macros/ctrack/Makefile
                 macros/eyemov/Makefile
                 macros/analysis/Makefile
                 macros/movie/Makefile
                 doc/Makefile
                 include/Makefile
                 include/veclib/Makefile
                 libsrc/Makefile
                 libsrc/qutil/Makefile
                 libsrc/gsl/Makefile
                 libsrc/interpreter/Makefile
                 libsrc/dobj/Makefile
                 libsrc/nports/Makefile
                 libsrc/fileio/Makefile
                 libsrc/datamenu/Makefile
                 libsrc/rawvol/Makefile
                 libsrc/newvec/Makefile
                 libsrc/vec_util/Makefile
                 libsrc/vectree/Makefile
                 libsrc/mvimenu/Makefile
                 libsrc/seq/Makefile
                 libsrc/view/Makefile
                 libsrc/viewmenu/Makefile
                 libsrc/xsupp/Makefile
                 libsrc/genwin/Makefile
                 libsrc/lutmenu/Makefile
                 libsrc/luts/Makefile
                 libsrc/psych/Makefile
                 libsrc/cstepit/Makefile
                 libsrc/opencv/Makefile
                 libsrc/visca/Makefile
                 libsrc/pic/Makefile
                 libsrc/opengl/Makefile
                 libsrc/ieee1394/Makefile
                 libsrc/ptgrey/Makefile
                 libsrc/cuda/Makefile
                 libsrc/dither/Makefile
                 libsrc/das1602/Makefile
                 libsrc/usb2000/Makefile
                 libsrc/meteor/Makefile
                 libsrc/knox/Makefile
                 libsrc/v4l2/Makefile
                 libsrc/parport/Makefile
                 libsrc/gps/Makefile
                 libsrc/www/Makefile
                 libsrc/gui/Makefile
                 libsrc/nrmenu/Makefile
                 libsrc/atc/Makefile
                 libsrc/sound/Makefile
                 src/Makefile])

# See p. 111 if we only need the library for the feature...
#
# Test lib presences against explicit configuration flags

if test "x${have_pthreads}" = xyes; then
  AC_CHECK_HEADERS([pthread.h],[],[have_pthreads=no])
fi

if test "x${have_pthreads}" = xno; then
  if test "x${async_exec}" = xyes; then
    AC_MSG_WARN([
  ------------------------------------------------------
  NOT building with asynchronous execution, because
  the pthreads library was not found.  Building a
  single-threaded version.
  ------------------------------------------------------])
  fi
  async_exec=no
fi

if test "x${have_libmotif}" = xyes; then
  if test "x${have_libxt}" = xyes; then
    AC_CHECK_HEADERS([Xm/Xm.h],[],[have_libmotif=no])
  else
    AC_MSG_WARN([
    -------------------------------------------------
    libXm is present, but libXt is not.
    Not building with GUI support.
    -------------------------------------------------])
    have_libmotif=no
  fi
fi

if test "x${have_alsa}" = xno; then
  if test "x${use_sound}" = xyes; then
    AC_MSG_WARN([
  ---------------------------------
  ALSA sound library not found.
  Building without sound support.
  ---------------------------------])
    use_sound=no
  fi
fi

if test "x${have_v4l2}" = xno; then
  if test "x${use_v4l2}" = xyes; then
    AC_MSG_WARN([
  ---------------------------------
  /dev/video0 not found.
  Building without V4L2 support.
  ---------------------------------])
    use_v4l2=no
  fi
fi


if test "x${have_meteor}" = xno; then
  if test "x${use_meteor}" = xyes; then
    AC_MSG_WARN([
  ---------------------------------
  /dev/meteor0 not found.
  Building without Matrox meteor support.
  ---------------------------------])
    use_meteor=no
  fi
fi


if test "x${have_visca}" = xno; then
  if test "x${use_visca}" = xyes; then
    AC_MSG_WARN([
  ---------------------------------
  /dev/visca not found.
  Building without Sony VISCA support.
  ---------------------------------])
    use_visca=no
  fi
fi


if test "x${have_fb_dev}" = xno; then
  if test "x${use_fb_dev}" = xyes; then
    AC_MSG_WARN([
  --------------------------------------------
  /dev/fb0 not found.
  Building without frame buffer device support.
  --------------------------------------------])
    use_fb_dev=no
  fi
fi

if test "x${have_das1602}" = xno; then
  if test "x${use_das1602}" = xyes; then
    AC_MSG_WARN([
  ---------------------------------
  /dev/adc0 not found.
  Building without DAS1602 support.
  ---------------------------------])
    use_das1602=no
  fi
fi


if test "x${have_parport}" = xno; then
  AC_MSG_WARN([
  ---------------------------------
  /dev/parport0 not found.
  Building without parallel port support.
  ---------------------------------])
fi


if test "x${have_sched_h}" = xno; then
  if test "x${use_rt_sched}" = xyes; then
    AC_MSG_WARN([
  -------------------------------------------------
  NOT building with real-time scheduler support,
  because sched.h was not found.
  -------------------------------------------------])
  fi
  use_rt_sched=no
fi
if test "x${have_sched}" = xno; then
  if test "x${use_rt_sched}" = xyes; then
    AC_MSG_WARN([
  -------------------------------------------------------
  NOT building with real-time scheduler support, because
  a library containing sched_setscheduler was not found.
  -------------------------------------------------------])
  fi
  use_rt_sched=no
fi

if test "x${have_stat}" = xyes; then
  AC_DEFINE([HAVE_STAT], 1, [stat system call available to check file status])
fi

if test "x${async_exec}" = xyes; then
  AC_DEFINE([ASYNC_EXEC], 1, [asynchronous execution enabled])
fi

if test "x${thread_safe_query}" = xyes; then
  AC_DEFINE([THREAD_SAFE_QUERY], 1, [thread-safe query features enabled])
fi

if test "x${use_sse}" = xyes; then
  AC_DEFINE([USE_SSE], 1, [use of SSE* extensions enabled])
fi

if test "x${n_processors}" != xno; then
  AC_DEFINE_UNQUOTED([N_PROCESSORS], ${n_processors}, [multi-processing with $n_processors processors enabled])
fi

if test "x${have_pic}" = xno; then
  if test "x${use_pic}" = xyes; then
  AC_MSG_WARN([
  -------------------------------------------
  Device /dev/pic not found.
  Building without PIC microcontroller support.
  -------------------------------------------])
    use_pic=no
  fi
fi

# Here we enable PIC if /dev/pic is present, regardless of the state of use_pic
if test "x${have_pic}" = xyes; then
  AC_DEFINE([HAVE_PIC], 1, [PIC LED controller support enabled])
fi

if test "x${have_numrec}" = xno; then
  if test "x${use_numrec}" = xyes; then
    AC_MSG_WARN([
  -------------------------------------------
  Numerical Recipes library not found.
  Building without numrec support.
  -------------------------------------------])
    use_numrec=no
  fi
fi
if test "x${have_numrec}" = xyes; then
  if test "x${use_numrec}" = xyes; then
    AC_DEFINE([HAVE_NUMREC], 1, [Numerical Recipes support enabled])
  fi
fi
AM_CONDITIONAL([HAVE_NUMREC],[test x${use_numrec} = xyes])


# Set config variables

if test "x${memalign}" = xyes; then
  AC_DEFINE([HAVE_MEMALIGN], 1, [memory alignment with memalign supported])
fi

if test "x${have_adjtimex}" = xyes; then
  if test "x${have_sys_timex_h}" = xyes; then
    AC_DEFINE([HAVE_ADJTIMEX], 1, [clock adjustment with adjtimex supported])
  fi
fi

if test "x${curl}" = xyes; then
  AC_DEFINE([HAVE_CURL], 1, [www communication supported])
fi

if test "x${have_libglut}" = xyes; then
  AC_DEFINE([HAVE_GLUT], 1, [GLUT library present])
fi

if test "x${use_tty_ctl}" = xyes; then
  AC_DEFINE([TTY_CTL], 1, [low-level tty control enabled])
fi

if test "x${use_viewers}" = xyes; then
  AC_DEFINE([VIEWERS], 1, [graphical viewing windows enabled])
fi

if test "x${use_cuda}" = xyes; then
  if test "x${working_nvcc}" = xyes ;  then
    if test "x${have_nv_sdk}" = xyes ; then
      # We could conceivably have all the software installed,
      # but not have an actual cuda-capable device???
      # Check for devices when determining compute capability!
      AC_DEFINE([HAVE_CUDA], 1, [GPU operations with CUDA enabled])
      if test "x${have_libcurand}" = xyes ; then
        AC_DEFINE([HAVE_LIBCURAND], 1, [GPU random number generation with libcurand enabled])
      fi
      if test "x${have_libnpp}" = xyes ; then
        AC_DEFINE([HAVE_LIBNPP], 1, [GPU operations libnpp enabled])
      fi
    else
      AC_MSG_WARN([
  -------------------------------------------------
  Working nvcc found, but unable to find SDK.
  Not building with CUDA support.
  -------------------------------------------------])
    fi
  else
    AC_MSG_WARN([
  -------------------------------------------------
  CUDA support requested, but unable to find nvcc.
  Not building with CUDA support.
  -------------------------------------------------])
  fi
fi

if test "x${use_history}" = xyes; then
  AC_DEFINE([HAVE_HISTORY], 1, [response history enabled])
fi

if test "x${use_rt_sched}" = xyes; then
  AC_DEFINE([ALLOW_RT_SCHED], 1, [real-time scheduler control enabled])
fi

if test "x${use_motif_gui}" = xyes; then
  if test "x${have_libmotif}" == xyes; then
    AC_DEFINE([HAVE_MOTIF], 1, [MOTIF gui support enabled])
    AC_DEFINE([INTERFACE], 1, [GUI features enabled])
  else
    AC_MSG_WARN([
  -------------------------------------------------
  libXm is not present.
  Not building with GUI support.
  -------------------------------------------------])
  fi
fi

if test "x${use_stepit}" = xyes; then
  AC_DEFINE([STEPIT], 1, [STEPIT optimization enabled])
fi

if test "x${use_sound}" = xyes; then
  AC_DEFINE([HAVE_SOUND], 1, [Sound support enabled])
fi

if test "x${use_v4l2}" = xyes; then
  AC_DEFINE([HAVE_V4L2], 1, [V4L2 video interface support enabled])
fi

if test "x${use_meteor}" = xyes; then
  AC_DEFINE([HAVE_METEOR], 1, [Matrox meteor video interface support enabled])
fi

if test "x${use_visca}" = xyes; then
  AC_DEFINE([HAVE_VISCA], 1, [Sony VISCA camera control protocol support enabled])
fi

if test "x${have_proc_cpuinfo}" = xyes; then
  AC_DEFINE([HAVE_PROC_CPUINFO], 1, [CPU information available via /proc/cpuinfo])
fi

if test "x${use_fb_dev}" = xyes; then
  AC_DEFINE([HAVE_FB_DEV], 1, [Frame buffer device support enabled])
fi

if test "x${use_das1602}" = xyes; then
  AC_DEFINE([HAVE_DAS1602], 1, [Measurement Computing DAS1602 interface support enabled])
fi

if test "x${have_parport}" = xyes; then
  AC_DEFINE([HAVE_PARPORT], 1, [parallel port support enabled])
fi

if test "x${use_debug}" = xyes; then
  AC_DEFINE([DEBUG], 1, [debug features enabled])
fi

if test "x${use_getbuf}" = xyes; then
  AC_DEFINE([USE_GETBUF], 1, [use internal memory manager])
fi

if test "x${use_cautious}" = xyes; then
  AC_DEFINE([CAUTIOUS], 1, [cautious checking enabled])
fi

if test "x${have_round}" = xyes; then
  AC_DEFINE([HAVE_ROUND], 1, [library implementation of round() available])
fi

if test "x${have_xext}" = xyes; then
  AC_DEFINE([HAVE_X11_EXT], 1, [X11 extensions enabled (w/ libXext)])
fi

if test "x${have_libgl}" = xyes; then
  if test "x${ac_cv_header_GL_gl_h}" = xno; then
    AC_MSG_WARN([
  -------------------------------------------------
  libgl is present, but header file GL/gl.h is not.
  Building without OpenGL support.
  -------------------------------------------------])
  else
    if test "x${ac_cv_header_GL_glu_h}" = xno; then
      AC_MSG_WARN([
  -------------------------------------------------
  libgl is present, but header file GL/glu.h is not.
  Building without OpenGL support.
  -------------------------------------------------])
    else
      AC_DEFINE([HAVE_OPENGL], 1, [OpenGL graphics support enabled])
    fi
  fi
fi

if test "x${have_sgisync}" = xyes; then
  AC_DEFINE([HAVE_VIDEOSYNCSGI], 1, [OpenGL graphics support enabled])
fi

if test "x${have_libx11}" = xyes; then
  if test "$x${ac_cv_header_X11_Xlib_h}" = xno; then
    AC_MSG_WARN([
  -------------------------------------------------
  libX11 is present, but header file X11/Xlib.h is not.
  Building without X11 support.
  -------------------------------------------------])
  else
    AC_DEFINE([HAVE_X11], 1, [X11 window system support enabled])
  fi
fi

if test "x${have_libtermcap}" = xyes; then
  AC_DEFINE( [HAVE_TERMCAP], 1, [Termcap terminal capability library])
fi

have_gsl=no
if test "x${have_libgslcblas}" = xyes; then
  if test "x${have_libgsl}" = xyes; then
    have_gsl=yes
    AC_DEFINE( [HAVE_GSL], 1, [GNU Scientific Library support enabled])
  fi
fi

if test "x${have_libtiff}" = xyes; then
  AC_DEFINE([HAVE_TIFF], 1, [TIFF file support enabled])
fi

if test "x${have_libpng}" = xyes; then
  AC_DEFINE([HAVE_PNG], 1, [PNG file support enabled])
fi

if test "x${have_libjpeg}" = xyes; then
  if test "x${have_jpeglib_h}" = xyes; then
    AC_DEFINE([HAVE_JPEG_SUPPORT], 1, [JPEG file support enabled])
  fi
fi

if test "x${have_libswscale}" = xyes; then
  AC_DEFINE([HAVE_LIBSWSCALE], 1, [AVI file support enabled (w/ libswscale)])
fi

if test "x${have_libavcodec}" = xyes; then
  AC_DEFINE([HAVE_LIBAVCODEC], 1, [AVI file support enabled (w/ libavcodec)])
fi

# Need other libs as well here?
if test "x${have_libavformat}" = xyes; then
  AC_DEFINE([HAVE_LIBAVFORMAT], 1, [AVI file support enabled (w/ libavformat)])
fi

if test "x${have_libmatio}" = xyes; then
  AC_DEFINE([HAVE_MATIO], 1, [Matlab i/o enabled])
fi

if test "x${have_ncurses}" = xyes; then
  AC_DEFINE([HAVE_NCURSES], 1, [Screen drawing with ncurses])
fi

if test "x${have_pthreads}" = xyes; then
  AC_DEFINE([HAVE_PTHREADS], 1, [Multi-threading with pthreads])
fi


have_opencv=no
if test "x${have_opencvcore}" = xyes; then
  if test "x${have_opencvimgproc}" = xyes; then
    if test "x${have_opencvobjdetect}" = xyes; then
      if test "x${have_opencvhighgui}" = xyes; then
        have_opencv=yes
        # Should make sure that the user has not disabled...
        AC_DEFINE([HAVE_OPENCV], 1, [Linkage to OpenCV library])
      fi
    fi
  fi
fi

if test "x${have_libdv}" = xyes; then
  AC_DEFINE([HAVE_LIBDV], 1, [Support for ieee1394 cameras])
fi

if test "x${have_libdc1394}" = xyes; then
  AC_DEFINE([HAVE_LIBDC1394], 1, [Support for Point Grey Research ieee1394 cameras])
fi

# do we need to do anything about the doc subdir here???

AC_OUTPUT

# Warn about missing optional libraries.

if test "x${use_cuda}" = xyes; then
  if test "x${working_nvcc}" = xno ;  then
    AC_MSG_WARN([
  ---------------------------------------------
  Unable to find a working CUDA compiler nvcc.
  Building without CUDA support...
  ---------------------------------------------])
  fi
fi

if test "x${have_pthreads}" = xno; then
  AC_MSG_WARN([
  ----------------------------------------
  Unable to find the pthreads library.
  Building a single-threaded version...
  ----------------------------------------])
fi

if test "x${have_libmatio}" = xno; then
  AC_MSG_WARN([
  -----------------------------------------
  Unable to find the matio library.
  Building without matio support...
  -----------------------------------------])
fi

if test "x${have_libmotif}" = xno; then
  AC_MSG_WARN([
  --------------------------------------------------------------------
  NOT building with gui support, because the Xm library was not found.
  --------------------------------------------------------------------])
fi

if test "x${have_memalign}" = xno; then
  AC_MSG_WARN([
  -----------------------------------------
  Unable to find memalign.
  May have alignment-related problems...
  -----------------------------------------])
fi

if test "x${have_curl}" = xno; then
  AC_MSG_WARN([
  ------------------------------------------
  Unable to find libcurl.
  Building without www support.
  ------------------------------------------])
fi

if test "x${have_libtiff}" = xno; then
  AC_MSG_WARN([
  ------------------------------------------
  Unable to find libtiff.
  Building without TIFF file support.
  ------------------------------------------])
fi

if test "x${have_libjpeg}" = xno; then
  AC_MSG_WARN([
  ------------------------------------------
  Unable to find libjpeg.
  Building without JPEG file support.
  ------------------------------------------])
fi

if test "x${have_libavformat}" = xno; then
  AC_MSG_WARN([
  ------------------------------------------
  Unable to find libavformat.
  Building without AVI file support.
  ------------------------------------------])
fi

if test "x${have_opencv}" = xno; then
  AC_MSG_WARN([
  ------------------------------------------
  Unable to find one or more OpenCV libraries.
  Building without OpenCV support.
  ------------------------------------------])
fi

if test "x${have_ncurses}" = xno; then
  AC_MSG_WARN([
  ------------------------------------------
  Unable to find ncurses library.
  Building without ncurses support.
  ------------------------------------------])
fi

if test "x${have_gsl}" = xno; then
  AC_MSG_WARN([
  ------------------------------------------
  Unable to find GNU scientific library (libgsl,libgslcblas).
  Building without GSL support.
  ------------------------------------------])
fi

if test "x${have_libdc1394}" = xno; then
  AC_MSG_WARN([
  ------------------------------------------
  Unable to find libdc1394.
  Building without support for IEEE1394 digital cameras.
  ------------------------------------------])
fi

if test "x${have_libtermcap}" = xno; then
  AC_MSG_WARN([
  ------------------------------------------
  Unable to find libtermcap.
  Building without termcap functionality.
  ------------------------------------------])
fi

if test "x${have_libglut}" = xno; then
  AC_MSG_WARN([
  ------------------------------------------
  Unable to find libglut.
  Building without GLUT functionality for openGL.
  ------------------------------------------])
fi


if test "x${have_texi2dvi}" = xno; then
  AC_MSG_WARN([
  ------------------------------------------
  Unable to find texi2dvi.
  You will not be able to build the pdf form of the documentation.
  ------------------------------------------])
fi


