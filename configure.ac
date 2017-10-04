#					       -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

# BUG - need to set or clear HAVE_VBL
# BUG - should we set MY_INTR???  (builtin.c, do_cmd.c)

# BUG - need to test for makeinfo

# How can we turn off optimization??

#AC_PREREQ([2.63])
AC_INIT([quip], [1.0], [jeffrey.b.mulligan@nasa.gov])

# this line prevents -O2...
#: ${CXXFLAGS=-g}
# didn't work - but CFLAGS="-g" on configure command line does.
# Maybe it would work here with CFLAGS instead of CXXFLAGS?
# see def_config.csh...

AC_CONFIG_SRCDIR([src/quip_main.c])
AC_CONFIG_HEADERS([config.h])

AC_CANONICAL_HOST
#echo Canonical host is $host, cpu $host_cpu, vendor $host_vendor, os $host_os

#echo host_os = $host_os
case "$host_os" in
darwin* ) os_is_darwin=yes ;;
* ) os_is_darwin=no ;;
esac

echo Host OS is darwin:  $os_is_darwin

# User settable features:

# Disable minimal build by default:
AC_ARG_ENABLE([minimal-build],
	[ --enable-minimal-build	disable ALL optional features ],
	[ minimal_build=${enableval}], [minimal_build=no])

# Disable ioperm-based vbl_wait by default:
AC_ARG_ENABLE([use-vbl],
	[ --enable-use-vbl	enable ioperm-based vbl_wait ],
	[ use_vbl=${enableval}], [use_vbl=no])

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


# disable multiprocessing by default

AC_ARG_ENABLE([n-processors],
	[AS_HELP_STRING([--enable-n-processors=N],
		[enable multiprocessing with N processors @<:@default: 2@:>@])],
	[ n_processors=${enableval}], [n_processors=1])

# Devices

# Enable raw volume by default:
AC_ARG_ENABLE([use-rawvol],
	[AS_HELP_STRING([--disable-rawvol],[disable raw disk volume subsystem @<:@default: no@:>@])],
	[ use_rawvol=${enableval}], [use_rawvol=yes])

# Enable v4l2 by default:
AC_ARG_ENABLE([use-v4l2],
	[AS_HELP_STRING([--disable-v4l2],[disable V4L2 video subsystem @<:@default: no@:>@])],
	[ use_v4l2=${enableval}], [use_v4l2=yes])

# Enable parport by default:
AC_ARG_ENABLE([use-parport],
	[AS_HELP_STRING([--disable-parport],[disable parallel port support @<:@default: no@:>@])],
	[ use_parport=${enableval}], [use_parport=yes])

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


# disable Knox switcher support by default

AC_ARG_ENABLE([knox-switcher],
	[AS_HELP_STRING([--enable-knox-switcher],
		[enable Knox video switcher support @<:@default: yes@:>@])],
	[ use_knox=${enableval}], [use_knox=no])


# Enable tty_ctl by default:
AC_ARG_ENABLE([tty-ctl],
	[AS_HELP_STRING([--disable-tty-ctl],[disable low level tty control @<:@default: no@:>@])],
	[ use_tty_ctl=${enableval}], [use_tty_ctl=yes])

# Enable script interrupt handler by default:
AC_ARG_ENABLE([use_my_intr],
	[AS_HELP_STRING([--disable-my-intr],[disable scriptable interrupt handler @<:@default: no@:>@])],
	[ use_my_intr=${enableval}], [use_my_intr=yes])

# Enable viewers by default:
AC_ARG_ENABLE([use_viewers],
	[AS_HELP_STRING([--disable-use_viewers],[disable graphical viewing window support @<:@default: no@:>@])],
	[ use_viewers=${enableval}], [use_viewers=yes])

# Enable history by default:
AC_ARG_ENABLE([use_history],
	[AS_HELP_STRING([--disable-use_history],[disable history features @<:@default: no@:>@])],
	[ use_history=${enableval}], [use_history=yes])

# Disable rt_sched by default:
AC_ARG_ENABLE([rt-sched],
	[AS_HELP_STRING([--enable-rt-sched],[enable real-time scheduler control @<:@default: yes@:>@])],
	[ use_rt_sched=${enableval}], [use_rt_sched=no])

# Enable rt_sched by default:
#AC_ARG_ENABLE([use_rt_sched],
#	[AS_HELP_STRING([--disable-rt-sched],[disable real-time scheduler control @<:@default: no@:>@])],
#	[ use_rt_sched=${enableval}], [use_rt_sched=yes])

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

# Enable sparselm support by default:
AC_ARG_ENABLE([use_sparselm],
	[AS_HELP_STRING([--disable-sparselm],[disable Sparse L-M support@<:@default: no@:>@])],
	[ use_sparselm=${enableval}], [use_sparselm=yes])

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

# Enable opencl by default:
AC_ARG_ENABLE([use_opencl],
	[AS_HELP_STRING([--disable-opencl],[disable GPU operations with OpenCL @<:@default: no@:>@])],
	[ use_opencl=${enableval}], [use_opencl=yes])

# Metal seems to only work with objective C, so we can't test for it
# with a regular C program...
## Enable metal by default:
#AC_ARG_ENABLE([use_metal],
#	[AS_HELP_STRING([--disable-metal],[disable GPU operations with Metal @<:@default: no@:>@])],
#	[ use_metal=${enableval}], [use_metal=yes])

# Enable X11 by default:
AC_ARG_ENABLE([use_x11],
	[AS_HELP_STRING([--disable-x11],[disable X11 @<:@default: no@:>@])],
	[ use_x11=${enableval}], [use_x11=yes])

# Enable GSL by default:
AC_ARG_ENABLE([use_gsl],
	[AS_HELP_STRING([--disable-gsl],[disable GNU Scientific Library @<:@default: no@:>@])],
	[ use_gsl=${enableval}], [use_gsl=yes])

# Enable pthreads by default:
AC_ARG_ENABLE([use_pthreads],
	[AS_HELP_STRING([--disable-pthreads],[disable pthread support@<:@default: no@:>@])],
	[ use_pthreads=${enableval}], [use_pthreads=yes])

# Enable curl by default:
AC_ARG_ENABLE([use_curl],
	[AS_HELP_STRING([--disable-curl],[disable curl support@<:@default: no@:>@])],
	[ use_curl=${enableval}], [use_curl=yes])

# Enable matio by default:
AC_ARG_ENABLE([use_matio],
	[AS_HELP_STRING([--disable-matio],[disable Matlab I/O support@<:@default: no@:>@])],
	[ use_matio=${enableval}], [use_matio=yes])

# Enable opencv by default:
AC_ARG_ENABLE([opencv],
	[AS_HELP_STRING([--disable-opencv],[disable OpenCV support@<:@default: no@:>@])],
	[ use_opencv=${enableval}], [use_opencv=yes])

# Enable FANN by default:
AC_ARG_ENABLE([use_fann],
	[AS_HELP_STRING([--disable-fann],[disable support for neural networks w/ libfann@<:@default: no@:>@])],
	[ use_fann=${enableval}], [use_fann=yes])

# Enable fileio by default:
AC_ARG_ENABLE([use_fileio],
	[AS_HELP_STRING([--disable-fileio],[disable support for image file I/O@<:@default: no@:>@])],
	[ use_fileio=${enableval}], [use_fileio=yes])

# Enable ncurses by default:
AC_ARG_ENABLE([use_ncurses],
	[AS_HELP_STRING([--disable-ncurses],[disable support terminal control w/ ncurses@<:@default: no@:>@])],
	[ use_ncurses=${enableval}], [use_ncurses=yes])

# Enable flycap by default:
AC_ARG_ENABLE([use_flycap],
	[AS_HELP_STRING([--disable-flycap],[disable support for PGR cameras with libflycap@<:@default: no@:>@])],
	[ use_flycap=${enableval}], [use_flycap=yes])

# Enable dc1394 by default:
AC_ARG_ENABLE([dc1394],
	[AS_HELP_STRING([--disable-dc1394],[disable support terminal control w/ dc1394@<:@default: no@:>@])],
	[ use_dc1394=${enableval}], [use_dc1394=yes])

# Enable opengl by default:
AC_ARG_ENABLE([use_opengl],
	[AS_HELP_STRING([--disable-opengl],[disable OpenGL support@<:@default: no@:>@])],
	[ use_opengl=${enableval}], [use_opengl=yes])

# Enable libdv by default:
AC_ARG_ENABLE([use_dv],
	[AS_HELP_STRING([--disable-dv],[disable digital video support with libdv@<:@default: no@:>@])],
	[ use_dv=${enableval}], [use_dv=yes])

## Disable debug by default:
#AC_ARG_ENABLE(([use_debug],
#	[AS_HELP_STRING([--enable-debug],[enable cautious checking @<:@default: yes@:>@])],
#	[ use_debug=${enableval}], [use_debug=no])
# hack to locate expat/iconv in /usr/local on BSD systems
#CPPFLAGS="$CPPFLAGS -I/usr/local/include"
#LIBS="$LIBS -L/usr/local/lib"

if test "x${minimal_build}" = xyes; then
	# BUG? - if any of these are set differently from
	# their defaults, we might print a warning?
	async_exec=no
	thread_safe_query=no
	n_processors=1
	use_rawvol=no
	use_v4l2=no
	use_parport=no
	use_meteor=no
	use_das1602=no
	use_visca=no
	use_fb_dev=no
	use_knox=no
	use_tty_ctl=no
	use_my_intr=no
	use_viewers=no
	use_history=no
	use_rt_sched=no
	use_motif_gui=no
	use_stepit=no
	use_numrec=no
	use_sparselm=no
	use_sound=no
	use_debug=yes
	use_getbuf=no
	use_cautious=yes
	use_avi=no
	use_cuda=no
	use_opencl=no
#	use_metal=no
	use_x11=no
	use_gsl=no
	use_pthreads=no
	use_curl=no
	use_matio=no
	use_opencv=no
	use_fileio=no
	use_ncurses=no
	use_dc1394=no
	use_flycap=no
	use_opengl=no
	use_dv=no
fi

#################################################
#
# cuda CUDA checks
#
###################################################


# For older versions, the SDK may be installed in
# the user's home directory...
# But in the nfs environment where we might have different
# versions on different systems, this can be problematic.
# The solution here is to put a symbolic link in /usr/local/cuda.

# On the mac, it is installed in /Developer/GPU Computing
# On iMac with CUDA 5.5, /Developer/NVIDIA/CUDA-5.5
# We need to make a link to /usr/local/cuda/NVIDIA_GPU_Computing_SDK

# As of cuda 5, it seems to be installed in /usr/local/cuda,
# along with other components?
#
# On craik, (cuda4) there is /usr/local/cuda...


if test "x${use_cuda}" = xyes; then
  AC_CHECK_FILE([/Developer/NVIDIA],[cuda_on_mac="yes"],[cuda_on_mac="no"])
  if test "x${cuda_on_mac}" = xyes; then
    AC_CHECK_FILE([/Developer/NVIDIA/CUDA-8.0],[cuda_dir="/Developer/NVIDIA/CUDA-8.0"],[cuda_dir="no"])
    if test "x${cuda_dir}" = xno; then
      AC_CHECK_FILE([/Developer/NVIDIA/CUDA-7.5],[cuda_dir="/Developer/NVIDIA/CUDA-7.5"],[cuda_dir="no"])
      if test "x${cuda_dir}" = xno; then
        AC_CHECK_FILE([/Developer/NVIDIA/CUDA-7.0],[cuda_dir="/Developer/NVIDIA/CUDA-7.0"],[cuda_dir="no"])
        if test "x${cuda_dir}" = xno; then
          AC_CHECK_FILE([/Developer/NVIDIA/CUDA-6.5],[cuda_dir="/Developer/NVIDIA/CUDA-6.5"],[cuda_dir="no"])
          if test "x${cuda_dir}" = xno; then
            AC_CHECK_FILE([/Developer/NVIDIA/CUDA-5.5],[cuda_dir="/Developer/NVIDIA/CUDA-5.5"],[cuda_dir="no"])
          fi
        fi
      fi
    fi
echo Found cuda software in $cuda_dir
    if test "x${cuda_dir}" != xno; then
      AC_CHECK_FILE([$cuda_dir/include],[CPPFLAGS="$CPPFLAGS -I$cuda_dir/include"])
      PATH=$PATH:$cuda_dir/bin
      export PATH
      AC_CHECK_FILE([$cuda_dir/samples],[nv_sdk_dir="$cuda_dir/samples"],[nv_sdk_dir="no"])

      nv_sdk_incdir="$nv_sdk_dir/common/inc"
echo Including $nv_sdk_incdir
      CPPFLAGS="$CPPFLAGS -I$nv_sdk_incdir"

      AC_CHECK_FILE([$cuda_dir/samples/common/lib/darwin],
	  [nv_sdk_libdir="$cuda_dir/samples/common/lib/darwin"],
          [nv_sdk_libdir="no"])

      if test "x${nv_sdk_libdir}" = xno; then
        have_nv_sdk=no
      else
echo Adding nvidia SDK libdir $nv_sdk_libdir - 1
        LDFLAGS="$LDFLAGS -L$nv_sdk_libdir"
        have_nv_sdk=yes
      fi

    fi
  else # ! cuda_on_mac

    AC_CHECK_FILE([/usr/local/cuda],[have_cuda="yes"],[have_cuda="no"])


    if test "x${have_cuda}" = xyes; then
      AC_CHECK_FILE([/usr/local/cuda/include],[CPPFLAGS="$CPPFLAGS -I/usr/local/cuda/include"])
      PATH=$PATH:/usr/local/cuda/bin
      export PATH
      AC_CHECK_FILE([/usr/local/cuda/samples],[nv_sdk_dir="/usr/local/cuda/samples"],[nv_sdk_dir="no"])
      if test "x${nv_sdk_dir}" = xno; then
        have_nv_sdk=no
      else	# sdk dir found
        AC_CHECK_FILE([/usr/local/cuda/samples/common/lib/linux/x86_64],[nv_sdk_libdir="/usr/local/cuda/samples/common/lib/linux/x86_64"],[nv_sdk_libdir="no"])
        AC_CHECK_FILE([/usr/local/cuda/samples/common/inc],[nv_sdk_incdir="/usr/local/cuda/samples/common/inc"],[nv_sdk_incdir="no"])
        if test "x${nv_sdk_incdir}" = xno; then
          have_nv_sdk=no
	else
          CPPFLAGS="$CPPFLAGS -I$nv_sdk_incdir"
          if test "x${nv_sdk_libdir}" = xno; then
            have_nv_sdk=no
          else
echo Adding nvidia SDK libdir $nv_sdk_libdir - 2
            LDFLAGS="$LDFLAGS -L$nv_sdk_libdir"
            have_nv_sdk=yes
	  fi
        fi
      fi
    fi # have_cuda
  fi		# ! cuda_on_mac
else		# ! use_cuda
  have_cuda_sdk_libdir=no
  have_nv_sdk=no
  nv_sdk_libdir=no
fi

#################################################
#						#
#		End of CUDA checks		#
#						#
#################################################

# These fink checks used to come after the X11 checks, but
# we had a version mismatch problem with libpng (which seems
# to exist in /opt/X11...).

# use homebrew instead of fink?
#AC_CHECK_FILE([/sw/include],[CPPFLAGS="$CPPFLAGS -I/sw/include"])
#AC_CHECK_FILE([/sw/lib/ffmpeg-2.4/include],[CPPFLAGS="$CPPFLAGS -I/sw/lib/ffmpeg-2.4/include"])

# Check for odd include directories
if test "x${use_x11}" = xyes; then
  AC_CHECK_FILE([/usr/X11/include],[CPPFLAGS="$CPPFLAGS -I/usr/X11/include"])
  AC_CHECK_FILE([/opt/X11/include],[CPPFLAGS="$CPPFLAGS -I/opt/X11/include"])
fi

# must come before compiler stuff
# This allows us to use GNU stuff, e.g. O_DIRECT...

echo calling ac_gnu_source
AC_GNU_SOURCE
echo DONE calling ac_gnu_source

#################################################
#
# opencl OpenCL checks
#
###################################################

# how to do this on Mac?
# until we figure this out, we enable by default...

if test "x${use_opencl}" = xyes; then
  have_libopencl=yes
fi

##################################################
##
## metal Metal checks
##
####################################################
#
## how to do this on Mac?
## until we figure this out, we enable by default...
#
#if test "x${use_metal}" = xyes; then
#  have_libmetal=yes
#fi
#
##################################################

#AC_CHECK_FILE([/sw/lib],[LDFLAGS="$LDFLAGS -L/sw/lib"])
#AC_CHECK_FILE([/sw/lib/ffmpeg-2.4/lib],[LDFLAGS="$LDFLAGS -L/sw/lib/ffmpeg-2.4/lib"])
## fink puts this stuff here...
## We should probably not use both!?
#AC_CHECK_FILE([/sw/lib/ffmpeg-1.2/lib],[LDFLAGS="$LDFLAGS -L/sw/lib/ffmpeg-1.2/lib"])
## We haven't ported to the newer library yet, so we prefer the old one

#if test "x${ac_cv_file__sw_lib_ffmpeg_1_2_lib}" = xno; then
#  AC_CHECK_FILE([/sw/lib/ffmpeg-2.0/lib],[LDFLAGS="$LDFLAGS -L/sw/lib/ffmpeg-2.0/lib"])
#fi

# this is for Ubuntu...
AC_CHECK_FILE([/usr/lib/x86_64-linux-gnu],[LDFLAGS="$LDFLAGS -L/usr/lib/x86_64-linux-gnu"])

AC_CHECK_FILE([/usr/lib64],[LDFLAGS="$LDFLAGS -L/usr/lib64"])
AC_CHECK_FILE([/usr/local/lib],[LDFLAGS="$LDFLAGS -L/usr/local/lib"])
AC_CHECK_FILE([/usr/local/include],[CPPFLAGS="$CPPFLAGS -I/usr/local/include"])
AC_CHECK_FILE([/usr/include/ffmpeg],[CPPFLAGS="$CPPFLAGS -I/usr/include/ffmpeg"])
#AC_CHECK_FILE([/sw/lib/ffmpeg-1.2/include],[CPPFLAGS="$CPPFLAGS -I/sw/lib/ffmpeg-1.2/include"])

# ffmpeg headers may be installed under /usr/include OR /usr/include/ffmpeg...
AC_CHECK_FILE([/usr/include/ffmpeg],[CPPFLAGS="$CPPFLAGS -I/usr/include/ffmpeg"])

if test "x${use_x11}" = xyes; then
  AC_CHECK_FILE([/usr/X11/lib],[LDFLAGS="$LDFLAGS -L/usr/X11/lib"])
  AC_CHECK_FILE([/usr/X11R6/lib],[LDFLAGS="$LDFLAGS -L/usr/X11R6/lib"])
  AC_CHECK_FILE([/opt/X11/lib],[LDFLAGS="$LDFLAGS -L/opt/X11/lib"])
fi

if test "x${use_cuda}" = xyes; then
  AC_CHECK_FILE([/usr/local/cuda/lib64],[have_usr_local_cuda_lib64=yes],
		[have_usr_local_cuda_lib64=no])
  if test "x${have_usr_local_cuda_lib64}" = xyes; then
    LDFLAGS="$LDFLAGS -L/usr/local/cuda/lib64"
  else
    AC_CHECK_FILE([/usr/local/cuda/lib],[LDFLAGS="$LDFLAGS -L/usr/local/cuda/lib"])
  fi
fi

AC_CHECK_FILE([./libsrc/interpreter/secret_key.c],[have_secret_key=yes],[have_secret_key=no])

# check for das1602
# BUG - /dev/adc0 could represent a different type of interface,
# but now this is the only one we support so we don't worry...
AC_CHECK_FILE([/dev/adc0],[have_das1602=yes],[have_das1602=no])

# check for v4l2
# This fails if /dev/video0 is present but not readable...
# We should check for this and give a fatal error so that root can fix it!
if test "x${use_v4l2}" = xyes; then
  AC_CHECK_FILE([/dev/video0],[have_video_device=yes],[have_video_device=no])
else
  have_video_device=no
fi

# check for parallel port
if test "x${use_parport}" = xyes; then
  AC_CHECK_FILE([/dev/parport0],[have_parport=yes],[have_parport=no])
else
  have_parport=no
fi

# check for raw volume
if test "x${use_rawvol}" = xyes; then
  AC_CHECK_FILE([/dev/rawvol],[have_rawvol=yes],[have_rawvol=no])
  if test "x${have_rawvol}" = xno; then
    AC_CHECK_FILE([/dev/rawvol-disk1],[have_rawvol=yes],[have_rawvol=no])
  fi
fi

# check for knox video switcher
AC_CHECK_FILE([/dev/knox],[have_knox=yes],[have_knox=no])


# check for meteor
if test "x${use_meteor}" = xyes; then
  AC_CHECK_FILE([/dev/meteor0],[have_meteor=yes],[have_meteor=no])
fi

# check for symbolic link to visca tty port
if test "x${use_visca}" = xyes; then
  AC_CHECK_FILE([/dev/visca],[have_visca=yes],[have_visca=no])
fi

# check for /proc/cpuinfo, a linux-ism
AC_CHECK_FILE([/proc/cpuinfo],[have_proc_cpuinfo=yes],[have_proc_cpuinfo=no])

# check for frame buffer device
AC_CHECK_FILE([/dev/fb0],[fb_dev_present=yes],[fb_dev_present=no])

# Automake stuff

# consider silent-rules option, see p. 122
AM_INIT_AUTOMAKE
#export TAR_OPTIONS = --owner=0 --group=0 --numeric-owner
TAR_OPTIONS=--owner=0
TAR_OPTIONS+=--group=0
TAR_OPTIONS+=--numeric-owner

# BUG g++ required for cuda even when HW not present!?

# Checks for programs.

# Check for yacc/bison
#AC_PROG_YACC
# bison -y ?
AC_CHECK_PROGS(YACC,'bison -y' byacc yacc,[AM_MISSING_PROG(yacc)])

# these didn't work
#: ${CFLAGS="-g -O0"}
#: ${CXXFLAGS="-g -O0"}

# Check for compilers
AC_PROG_CC
AC_PROG_CC_C99
AC_PROG_CXX

AM_PROG_CC_C_O

# Check for install
AC_PROG_INSTALL

# Check for ranlib and ar
AC_PROG_RANLIB
# AM_PROG_AR not recognized on CentOS 6 ???
#AM_PROG_AR

# Check for - what?
AM_PROG_CC_C_O

# BUG we might want to set a config var to indicate if we have c99,
# so that we can know if we have round(3)...
#if test "x{$ac_cv_prog_cc_c99}" = xno; then
#    AC_MSG_WARN([
#------------------------------------------------------
#Compiler does not have a c99 mode.
#------------------------------------------------------])
#else
#  AC_MSG_RESULT([C99 support present.])
#fi

# trying to get this to generate version.texi in doc...
#AM_MAINTAINER_MODE([enable])

AC_CHECK_PROG(MAKEINFO,makeinfo,[/usr/bin/makeinfo],[makeinfo not found],[/usr/bin])
if test "x$MAKEINFO" == "xmakeinfo not found"; then
  have_makeinfo=no
else
  have_makeinfo=yes
fi

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


working_nvcc=no
NVCC=ls	# to prevent a syntax error later, if use_cuda is no
if test "x${use_cuda}" = xyes; then
  AC_ARG_VAR([NVCC], [nvcc compiler to use])
  AC_PATH_PROG([NVCC], [nvcc], [no])
echo NVCC = $NVCC
  if test "x$NVCC" != "xno"; then
    AC_MSG_CHECKING([whether nvcc works])
    cat>nvcc_test.cu<<EOF
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
    # so we can look at it later...
    #cp nvcc_test.cu /tmp/nvcc_test.cu
#       if $NVCC nvcc_test.cu && test_whether_output_files_are_ok
    if $NVCC nvcc_test.cu
    then
      working_nvcc=yes
    else
      # On mac with xcode 5, nvcc is broken out of the box, but can
      # be fixed with -ccbin=/usr/bin/clang option...
      # But this option doesn't work on linux...
      if test "x$os_is_darwin" == xyes; then
        NVCCFLAGS+=" -ccbin=/usr/bin/clang"
      fi
      echo testing nvcc with flags $NVCCFLAGS
      if $NVCC $NVCCFLAGS nvcc_test.cu
      then
        AC_SUBST(NVCCFLAGS,["$NVCCFLAGS"])
        working_nvcc=yes
      fi
    fi

    rm -f nvcc_test.cu a.out

    AC_MSG_RESULT([$working_nvcc])
  fi
#else
#  NVCC=ls	# to prevent a syntax error later, if use_cuda is no
fi

# Create a variable we can test in Makefile.am:
AM_CONDITIONAL([HAVE_NVCC], [test "x$working_nvcc" = "xyes"])

# determine cuda compute capability
#
# BUG when someone else is logged in on the console,
# they become owners of /dev/nvidiactl and /dev/nvidia0,
# with permissions 400 - this causes the test program to
# return 0 devices!?

# On mac OSX 10.12 (Sierra), we don't seem to have the /dev/nvidia files,
# but everything seems to work OK...

have_nvidia0=no
if test "x$working_nvcc" == xyes; then
  AC_MSG_CHECKING([for /dev/nvidia0])
  if test -e /dev/nvidia0; then
    if test -r /dev/nvidia0; then
      if test -w /dev/nvidia0; then
        AC_MSG_RESULT([present and accessible])
        have_nvidia0=yes
      else
        AC_MSG_RESULT([present but is not writable])
      fi
    else
      AC_MSG_RESULT([present but is not readable])
    fi
  else
    AC_MSG_RESULT([not present])
    # Don't need the dev file on Mac OSX ?
    if test "x$os_is_darwin" == xyes; then
      AC_MSG_NOTICE([Assuming CUDA will work without /dev/nvidia0 on Mac OSX])
      have_nvidia0=yes
    fi
  fi
fi
have_nvidiactl=no
if test "x$have_nvidia0" == xyes; then
  AC_MSG_CHECKING([for /dev/nvidiactl])
  if test -e /dev/nvidiactl; then
    if test -r /dev/nvidiactl; then
      if test -w /dev/nvidiactl; then
        AC_MSG_RESULT([present and accessible])
        have_nvidiactl=yes
      else
        AC_MSG_RESULT([present but is not writable])
      fi
    else
      AC_MSG_RESULT([present but is not readable])
    fi
  else
    AC_MSG_RESULT([not present])
    if test "x$os_is_darwin" == xyes; then
      AC_MSG_NOTICE([Assuming CUDA will work without /dev/nvidiactl on Mac OSX])
      have_nvidiactl=yes
    fi
  fi
fi
have_cuda_device=no

if test "x$working_nvcc" == xyes; then
  if test "x$have_nvidiactl" == xyes; then
    AC_MSG_CHECKING([for CUDA-capable device])
    cat>gpu_check.c<<EOF
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

    /* dont just return the number of gpus, because other runtime cuda
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
  #
  # as of cuda 6.5 (on brewster laptop), /usr/local/cuda/lib seems good,
  # but after upgrading to 10.11 (El Capitan), the test program fails
  # when run from /bin/sh, because LD_LIBRARY_PATH is not set.  This
  # can be fixed by adding -rpath /usr/local/cuda/lib to the command,
  # but rpath may be a Mac-ism, and not work on linux?
  # -rpath is not recognized on linux, try -R...

    # On a new CentOS installation, can't execute the program in /tmp!?
    #tmp_dir = /tmp
    tmp_dir=.		# BUG - build_dir???

    if test "x${host_vendor}" = xapple; then
      if $CC -o $tmp_dir/gpu_test $CPPFLAGS $LDFLAGS -rpath /usr/local/cuda/lib gpu_check.c -lcudart # 2> /dev/null
      then
        comp_cap=`$tmp_dir/gpu_test`
        rm -f $tmp_dir/gpu_test
      else
        comp_cap=0.0
      fi
    else
      # Not apple
      if $CC $CPPFLAGS $LDFLAGS -o $tmp_dir/gpu_test gpu_check.c -lcudart # 2> /dev/null
      then
        comp_cap=`$tmp_dir/gpu_test`
        rm -f $tmp_dir/gpu_test
      else
        comp_cap=0.0
      fi
    fi
    rm -f gpu_check.c
    AC_MSG_RESULT([compute capability $comp_cap])
    arch_set=no
    if test "x${comp_cap}" = x1.0; then
      AC_DEFINE([CUDA_COMP_CAP],10,[CUDA Compute Capability 1.0])
      AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_10'])
      arch_set=yes
      have_cuda_device=yes
    fi
    if test "x${comp_cap}" = x1.1; then
      AC_DEFINE([CUDA_COMP_CAP],11,[CUDA Compute Capability 1.1])
      AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_11'])
      arch_set=yes
      have_cuda_device=yes
    fi
    if test "x${comp_cap}" = x1.2; then
      AC_DEFINE([CUDA_COMP_CAP],12,[CUDA Compute Capability 1.2])
      AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_12'])
      arch_set=yes
      have_cuda_device=yes
    fi
    if test "x${comp_cap}" = x1.3; then
      AC_DEFINE([CUDA_COMP_CAP],13,[CUDA Compute Capability 1.3])
      AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_13'])
      arch_set=yes
      have_cuda_device=yes
    fi
    if test "x${comp_cap}" = x2.0; then
      AC_DEFINE([CUDA_COMP_CAP],20,[CUDA Compute Capability 2.0])
      AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_20'])
      arch_set=yes
      have_cuda_device=yes
    fi
    if test "x${comp_cap}" = x2.1; then
      AC_DEFINE([CUDA_COMP_CAP],21,[CUDA Compute Capability 2.1])
      AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_21'])
      arch_set=yes
      have_cuda_device=yes
    fi
    if test "x${comp_cap}" = x3.0; then
      AC_DEFINE([CUDA_COMP_CAP],30,[CUDA Compute Capability 3.0])
      AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_30'])
      arch_set=yes
      have_cuda_device=yes
    fi
    # BUG - fill in the missing cases for cards we don't own!
    if test "x${comp_cap}" = x6.1; then
      AC_DEFINE([CUDA_COMP_CAP],61,[CUDA Compute Capability 6.1])
      AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_61'])
      arch_set=yes
      have_cuda_device=yes
    fi
    if test "x${comp_cap}" = x0.0; then
    AC_MSG_WARN([
------------------------------------------------------
CUDA development tools are present, but can't find
any CUDA devices.  Make sure that the correct driver
is installed.
------------------------------------------------------])
      # I assume that the 10 here is not a typo, but that we have to provide
      # some sort of valid value???  BUG?
      AC_DEFINE([CUDA_COMP_CAP],00,[CUDA Compute Capability 0.0])
      arch_set=yes
      have_cuda_device=no
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
  else	# have working nvcc but can't access device
    AC_MSG_WARN([
------------------------------------------------------
CUDA development tools are present, but the device files
are not accessible.  Making a guess at the compute capability,
please reconfigure with access to the device files
/dev/nvidia0 and /dev/nvidiactl.
------------------------------------------------------])
    AC_DEFINE([CUDA_COMP_CAP],21,[CUDA Compute Capability 2.1])
    AC_SUBST(NVCC_ARCH_FLAGS,['-arch sm_21'])
  fi
fi	# end working_nvcc loop


# this test bombs if we perform it after testing for libGLEW !?!?!?
AC_CHECK_SIZEOF([long])

# Checks for libraries.

AC_SEARCH_LIBS([posix_memalign],[],[have_pmemalign=yes],[have_pmemalign=no])

AC_SEARCH_LIBS([adjtimex],[],[have_adjtimex=yes],[have_adjtimex=no])

#have_llseek=no
#AC_SEARCH_LIBS([_llseek],[],[have_llseek=yes])

# # AC_FUNC_MALLOC causes problems on Mac OSX, defines malloc to be rpl_malloc!?
# # On mac mini, the AC_FUNC_MALLOC line seems to cause automake to look
# # in home dir for "missing" instead of here???
# the problem may have been the result of bad ordering of AM_INIT_AUTOMAKE
AC_FUNC_MALLOC


# mandatory libraries
# -lm

AC_SEARCH_LIBS([cos],[m],[have_math=yes],[have_math=no])
AC_SEARCH_LIBS([cosf],[m],[have_fmath=yes],[have_fmath=no])
AC_SEARCH_LIBS([cosl],[m],[have_lmath=yes],[have_lmath=no])
AC_SEARCH_LIBS([round],[m],[have_round=yes],[have_round=no])
AC_SEARCH_LIBS([erf],[m],[have_erf=yes],[have_erf=no])

# on ubuntu, the numrec test fails because sqrt can't be found!?
AC_SEARCH_LIBS([sqrt],[m],[have_math=yes],[have_math=no])
AC_SEARCH_LIBS([sqrtf],[m],[have_fmath=yes],[have_fmath=no])

# should always have this on unix, but not on iOS?
have_stat=no
AC_SEARCH_LIBS([stat],[c],[have_stat=yes],[have_stat=no])

AC_SEARCH_LIBS([ctime_r],[c],[have_ctime_r=yes],[have_ctime_r=no])
AC_SEARCH_LIBS([ctime],[c],[have_ctime=yes],[have_ctime=no])
AC_SEARCH_LIBS([gmtime_r],[c],[have_gmtime_r=yes],[have_gmtime_r=no])
AC_SEARCH_LIBS([gmtime],[c],[have_gmtime=yes],[have_gmtime=no])
AC_SEARCH_LIBS([asctime_r],[c],[have_asctime_r=yes],[have_asctime_r=no])
AC_SEARCH_LIBS([asctime],[c],[have_asctime=yes],[have_asctime=no])
AC_SEARCH_LIBS([localtime_r],[c],[have_localtime_r=yes],[have_localtime_r=no])
AC_SEARCH_LIBS([posix_memalign],[c],[have_posix_memalign=yes],[have_posix_memalign=no])

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

AC_SEARCH_LIBS([libusb_init],[usb-1.0],[have_libusb=yes],[have_libusb=no])

AC_SEARCH_LIBS([gcry_cipher_encrypt],[gcrypt],[have_libgcrypt=yes],[have_libgcrypt=no])

if test "x${use_tty_ctl}" = xyes; then
  AC_SEARCH_LIBS([tgetent],[termcap],[have_libtermcap=yes],[have_libtermcap=no])
fi

if test "x${use_gsl}" = xyes; then
  AC_SEARCH_LIBS([cblas_cgemv],[gslcblas],[have_libgslcblas=yes],[have_libgslcblas=no])

  AC_SEARCH_LIBS([gsl_vector_alloc],[gsl],[have_libgsl=yes],[have_libgsl=no])
fi

# On a fresh CentOS 5.6 installation, /usr/lib64/libX11.so.6 exists (a link),
# but no file /usr/lib64/libX11.so - without the latter, the linker does
# not find a library when given -lX11 - the devel package needs to
# be installed for this link to be created, and for the header files
# to be installed...

if test "x${use_x11}" = xyes; then
  AC_SEARCH_LIBS([XOpenDisplay],[X11],[have_libx11=yes],[have_libx11=no])
  AC_SEARCH_LIBS([XShmCreateImage],[Xext],[have_xext=yes],[have_xext=no])
fi


# On ubuntu, including -lGLU seems to break loading from libm!?

if test "x${use_opengl}" = xyes; then
  AC_SEARCH_LIBS([glVertex2d],[GL],[have_libgl=yes],[have_libgl=no])
  ## # This function is provided by cuda!?
  ## AC_SEARCH_LIBS([glXGetVideoSyncSGI],[GL],[have_sgisync=yes],[have_sgisync=no])
  AC_SEARCH_LIBS([gluLookAt],[GLU],[have_libglu=yes],[have_libglu=no])
  AC_SEARCH_LIBS([glutInit],[glut],[have_libglut=yes],[have_libglut=no])
  AC_SEARCH_LIBS([glewInit],[GLEW],[have_libglew=yes],[have_libglew=no])
fi


if test "x${use_pthreads}" = xyes; then
  AC_SEARCH_LIBS([pthread_create],[pthread],[have_pthreads=yes],[have_pthreads=no])
fi

if test "x${use_rt_sched}" = xyes; then
  AC_SEARCH_LIBS([sched_setscheduler],[],[have_sched=yes],[have_sched=no])
  # check for scheduler funcs
  # BUG - some redundancy here!?
  AC_CHECK_FUNCS([sched_setscheduler sched_setparam sched_getparam])
fi

# On Mac, we need Xt (not needed on linux???), but the order matters!
# libXm has to come first, but if it comes first here, it comes later
# in the link line...
if test "x${use_motif_gui}" = xyes; then
  AC_SEARCH_LIBS([XtManageChild],[Xt],[have_libxt=yes],[have_libxt=no])
  AC_SEARCH_LIBS([XmCreateForm],[Xm],[have_libmotif=yes],[have_libmotif=no])
fi

# This inserts -lnumrec into the linker options, but we need the full path
# if the program is to be setuid root...

if test "x${use_numrec}" = xyes; then
  AC_SEARCH_LIBS([float_svdcmp],[numrec],[have_numrec=yes],[have_numrec=no])
fi

# sparselm has dependencies on SuiteSparse (cholmod)...
# cholmod depends on suitesparse config
# AND things like dgemm - from BLAS???
# Library order in demo program:
# -lsplm  -lcholmod -lsuitesparseconfig -lccolamd -lcolamd -lcamd -lamd -lmetis -llapack -lblas -lf2c 

if test "x${use_sparselm}" = xyes; then
  # used to be dgemm, but on Ubuntu it's dgemm_ ???
  AC_SEARCH_LIBS([dgemm_],[blas openblas],[have_blas=yes],[have_blas=no])
  if test "x${have_blas}" = xyes; then
    AC_SEARCH_LIBS([dpotrf_],[lapack],[have_lapack=yes],[have_lapack=no])
  else
    have_lapack=no
  fi
  if test "x${have_lapack}" = xyes; then
    AC_SEARCH_LIBS([clock_gettime],[rt],[have_librt=yes],[have_librt=no])
    # on Ubuntu, this fails because suitesparseconfig needs clock_gettime, from lrt
    AC_SEARCH_LIBS([SuiteSparse_calloc],[suitesparseconfig],[have_suitesparse=yes],[have_suitesparse=no])
  else
    have_suitesparse=no
  fi
  if test "x${have_suitesparse}" = xyes; then
    AC_SEARCH_LIBS([amd_2],[amd],[have_libamd=yes],[have_libamd=no])
  else
    have_libamd=no
  fi
  if test "x${have_libamd}" = xyes; then
    AC_SEARCH_LIBS([colamd],[colamd],[have_colamd=yes],[have_colamd=no])
  else
    have_colamd=no
  fi
  if test "x${have_colamd}" = xyes; then
    AC_SEARCH_LIBS([cholmod_solve],[cholmod],[have_cholmod=yes],[have_cholmod=no])
  else
    have_cholmod=no
  fi
  if test "x${have_cholmod}" = xyes; then
    AC_SEARCH_LIBS([sparselm_difcrs],[splm],[have_sparselm=yes],[have_sparselm=no])
  else
    have_sparselm=no
  fi
fi

if test "x${use_curl}" = xyes; then
  AC_SEARCH_LIBS([curl_global_init],[curl],[have_libcurl=yes],[have_libcurl=no])
fi

# Sound - first check for portaudio (most portable)
if test "x${use_sound}" = xyes; then
  AC_SEARCH_LIBS([Pa_Initialize],[portaudio],[have_portaudio=yes],[have_portaudio=no])
  if test "x${have_portaudio}" = xno; then
    AC_SEARCH_LIBS([snd_pcm_open],[asound],[have_alsa=yes],[have_alsa=no])
    if test "x${have_alsa}" = xno; then
      have_sound=no
    else
      have_sound=yes
    fi
  else
    have_sound=yes
  fi
fi

# what do we need libz for???
#AC_SEARCH_LIBS([inflate],[z],[have_libz=yes],[have_libz=no])

if test "x${use_matio}" = xyes; then
  AC_SEARCH_LIBS([Mat_Open],[matio],[have_libmatio=yes],[have_libmatio=no])
  if test "x${have_libmatio}" = xyes; then
    AC_SEARCH_LIBS([H5Gclose],[hdf5 hdf5.9],[have_libhdf5=yes],[have_libhdf5=no])
    # on the mac, with libmatio2, we have dependencies on libhdf5,
    # but this dependency is not present with the linux version???
    #if test "x${have_libhdf5}" = no; then
    #  have_libmatio=no
    #fi
  fi
fi

if test "x${use_opencl}" = xyes; then
# On pavlov w/ 2.8, we need clfftSetup
  AC_SEARCH_LIBS([clfftSetup],[clFFT],[have_clfft=yes],[have_clfft=no])
#  AC_SEARCH_LIBS([clfftInitSetupData],[clFFT],[have_clfft=yes],[have_clfft=no])
fi

if test "x${use_fann}" = xyes; then
  AC_SEARCH_LIBS([fann_create_standard],[fann],[have_fann=yes],[have_fann=no])
fi

if test "x${use_fileio}" = xyes; then
  AC_SEARCH_LIBS([TIFFGetField],[tiff],[have_libtiff=yes],[have_libtiff=no])
  AC_SEARCH_LIBS([png_read_image],[png],[have_libpng=yes],[have_libpng=no])

  # later libs come first... OpenCV seems to include a copy of libjpeg,
  # but only if built when libjpeg was not already installed!

  AC_SEARCH_LIBS([jpeg_start_decompress],[jpeg],[have_libjpeg=yes],[have_libjpeg=no])

  AC_SEARCH_LIBS([qt_read_scanlines],[qt],[have_libqt=yes],[have_libqt=no])
  AC_SEARCH_LIBS([av_crc],[avutil],[have_libavutil=yes],[have_libavutil=no])
  AC_SEARCH_LIBS([sws_scale],[swscale],[have_libswscale=yes],[have_libswscale=no])
  # with fink, libavformat is installed separately from lbiavcodec,
  # and so should be tested separately!  BUG
  # ditto for libavutil - BUG
  # BUG this is true on ubuntu also!
  #AC_SEARCH_LIBS([avcodec_open],[avcodec],[have_libavcodec=yes],[have_libavcodec=no])
  # libavformat depends on libavcodec , so libavcodec has to come first...
  AC_SEARCH_LIBS([avcodec_open2],[avcodec],[have_libavcodec=yes],[have_libavcodec=no])
  # old
  #AC_SEARCH_LIBS([av_open_input_file],[avformat],[have_libavformat=yes],[have_libavformat=no])
  AC_SEARCH_LIBS([avformat_open_input],[avformat],[have_libavformat=yes],[have_libavformat=no])
#  AC_SEARCH_LIBS([av_close_input_file],[avformat],[have_libavformat=yes],[have_libavformat=no])
  #have_libavutil=no
  #AC_SEARCH_LIBS([av_open_input_file],[avutil],[have_libavutil=yes])
  AC_SEARCH_LIBS([BZ2_bzDecompressInit],[bz2],[have_libbz2=yes],[have_libbz2=no])
fi

if test "x${use_ncurses}" = xyes; then
  AC_SEARCH_LIBS([initscr],[ncurses],[have_ncurses=yes],[have_ncurses=no])
fi

if test "x${use_opencv}" = xyes; then
  AC_SEARCH_LIBS([cvSetZero],[opencv_core cxcore],[have_opencvcore=yes],[have_opencvcore=no])
  AC_SEARCH_LIBS([cvContourArea],[opencv_imgproc cv],[have_opencvimgproc=yes],[have_opencvimgproc=no])
  AC_SEARCH_LIBS([cvHaarDetectObjects],[opencv_objdetect],[have_opencvobjdetect=yes],[have_opencvobjdetect=no])
  AC_SEARCH_LIBS([cvSaveImage],[opencv_highgui highgui opencv_imgcodecs],[have_opencvhighgui=yes],[have_opencvhighgui=no])
fi

#
## libraw1394 is needed by libflycapture-c as well as libdv...
#
# BUT - we don't need it if libflycapture-c is not present or unusable.
# And the binary that we have can't be used on CentOS 5...
#
#AC_SEARCH_LIBS([raw1394_new_handle],[raw1394],[have_libraw1394=yes],[have_libraw1394=no])

# Our old libdv program was based on dvgrab - the new
# version is not compativle with the current libraries
# But we can probably fix it up by looking at the current srcs...
#
## raw1394 check has to come before dv for correct lib order
#AC_SEARCH_LIBS([raw1394_new_handle],[raw1394],[have_libraw1394=yes],[have_libraw1394=no])
#AC_SEARCH_LIBS([dv_decode_full_frame],[dv],[have_libdv=yes],[have_libdv=no])

# _new is a function...

# 
if test "x${use_flycap}" = xyes; then
  AC_SEARCH_LIBS([fc2CreateContext],[flycapture-c],[have_libflycap=yes],[have_libflycap=no])
  # we also need to include our interface library???
  if test "x${have_libflycap}" = xyes; then
    use_dc1394=no
  fi
fi

if test "x${use_dc1394}" = xyes; then
  AC_CHECK_HEADERS([dc1394/dc1394.h])
  #echo ac_cv_header_dc1394_dc1394_h = $ac_cv_header_dc1394_dc1394_h
  if test "x${ac_cv_header_dc1394_dc1394_h}" = xyes; then
    have_dc1394_header=yes
    AC_SEARCH_LIBS([dc1394_new],[dc1394],[have_libdc1394=yes],[have_libdc1394=no])
  else
    have_dc1394_header=no
    have_libdc1394=no	# might really be present???
  fi
fi

if test "x${use_opencl}" = xyes; then
  # opencl libraries
  # On mac, this is not in user lib, rather it's included with a -framework option!?
  # EOF
  #AC_SEARCH_LIBS([cudaUnbindTexture],[cudart],[have_libcudart=yes],[have_libcudart=no])
  # If we have the cudart lib, then we can determine the compute
  # capability of the device.
  # what about linux?
  AC_MSG_CHECKING([for OpenCL])
  cat>ocl_test.c<<EOF
#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
  int main(int ac,char **av)
  {
    cl_platform_id platform_id = NULL;
    cl_int ret; //accepts return values for APIs
    cl_uint ret_num_platforms;
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  }
EOF
  if $CC $CPPFLAGS $LDFLAGS ocl_test.c $LIBS 2> /dev/null
  then
    have_libopencl=yes
  else
    # try adding -framework option
    #AC_MSG_CHECKING([for OpenCL with -framework option])
    if $CC $CPPFLAGS $LDFLAGS -framework OpenCL ocl_test.c $LIBS 2> /dev/null
    then
      LDFLAGS="$LDFLAGS -framework OpenCL -framework OpenGL"
      have_libopencl=yes
      /bin/rm a.out
    else
      have_libopencl=no
    fi
  fi
  /bin/rm ocl_test.c
  #echo Done checking for OpenCL, have_libopencl=$have_libopencl
  echo $have_libopencl
fi


#if test "x${use_metal}" = xyes; then
#  # metal libraries
#  # On mac, this is not in user lib, rather it's included with a -framework option!?
#  # EOF
#  AC_MSG_CHECKING([for Metal])
#  cat>mtl_test.c<<EOF
##ifdef __APPLE__ //Mac OSX has a different name for the header file
##include <Metal/Metal.h>
##endif
# 
#  int main(int ac,char **av)
#  {
#  }
#EOF
#  echo $CC $CPPFLAGS $LDFLAGS mtl_test.c $LIBS
#  if $CC $CPPFLAGS $LDFLAGS mtl_test.c $LIBS 2> /tmp/mtl_errs # /dev/null
#  then
#    have_libmetal=yes
#  else
#    # try adding -framework option
#    #AC_MSG_CHECKING([for OpenCL with -framework option])
#    if $CC $CPPFLAGS $LDFLAGS -framework Metal mtl_test.c $LIBS 2> /dev/null
#    then
#      LDFLAGS="$LDFLAGS -framework Metal"
#      have_libmetal=yes
#      /bin/rm a.out
#    else
#      have_libmetal=no
#    fi
#  fi
#  /bin/rm mtl_test.c
#  echo $have_libmetal
#fi

if test "x${use_cuda}" = xyes; then
  # cuda libraries
  AC_SEARCH_LIBS([cudaUnbindTexture],[cudart],[have_libcudart=yes],[have_libcudart=no])
  # If we have the cudart lib, then we can determine the compute
  # capability of the device.

  AC_SEARCH_LIBS([cufftPlan1d],[cufft],[have_libcufft=yes],[have_libcufft=no])
  AC_SEARCH_LIBS([cublasGetError],[cublas],[have_libcublas=yes],[have_libcublas=no])
  AC_SEARCH_LIBS([curandCreateGenerator],[curand],[have_libcurand=yes],[have_libcurand=no])

  AC_SEARCH_LIBS([nppiErode_8u_C1R],[nppi],[have_libnppi=yes],[have_libnppi=no])
  # older versions have libnpp
  # if that is satisfied by libnppi, then we may need other npp libs too!
  # BUG should check more carefully the status of this...
  if test "x${have_libnppi}" = xyes; then
    AC_SEARCH_LIBS([nppsFree],[npps])
  else
    # need to look for a different subroutine to avoid cached value???
    #AC_SEARCH_LIBS([nppiErode_8u_C1R],[npp],[have_libnpp=yes],[have_libnpp=no])
    AC_SEARCH_LIBS([nppiDilate_8u_C1R],[npp],[have_libnpp=yes],[have_libnpp=no])
  fi

  # BUG need to determine whether we're compiling for 64 bit!
  AC_SEARCH_LIBS([glewInit],[GLEW_x86_64 GLEW],[have_libglew=yes],[have_libglew=no])
  #if test "x${have_libglew}" = xno; then
  #  AC_SEARCH_LIBS([glewInit],[GLEW],[have_libglew=yes],[have_libglew=no])
  #fi
fi

if test "x${use_opengl}" = xyes; then
  # This function is provided by cuda, in libGLEW!?
  # but what header is needed to define the macro???
  #
  # This doesn't work in cuda 5, but does work for cuda 4, even though it apparently is also
  # a macro there!?!?
  if test "x${have_cuda5}" = xyes; then
    AC_MSG_CHECKING([for glXGetVideoSyncSGI (cuda 5)])
    cat>vsync_test.c<<EOF
    #include <GL/glx.h>
    //#include <GL/glxext.h>
    int main(int ac,char **av)
    {
    	unsigned int count;
    	glXGetVideoSyncSGI(&count);
    }
EOF
    echo Compiling vsync_test.c ...
    if $CC $CPPFLAGS $LDFLAGS vsync_test.c $LIBS
    then
      echo test succeeded
      have_sgisync=yes
    else
      echo test failed
      have_sgisync=no
    fi
    echo Done checking for glXGetVideoSyncSGI
  else
    AC_SEARCH_LIBS([glXGetVideoSyncSGI],[GL GLEW],[have_sgisync=yes],[have_sgisync=no])
  fi
else
  have_sgisync=no
fi
# BUG - need to remove vsync_test.c - and a.out?

# Checks for header files.


if test "x${use_x11}" = xyes; then
  AC_PATH_X
fi

AC_FUNC_ALLOCA

AC_HEADER_STDBOOL

AC_CHECK_HEADERS([assert.h])
AC_CHECK_HEADERS([ioctl.h])
AC_CHECK_HEADERS([linux/ioctl.h])
AC_CHECK_HEADERS([ctype.h])
AC_CHECK_HEADERS([pwd.h])
AC_CHECK_HEADERS([math.h])
AC_CHECK_HEADERS([signal.h])
AC_CHECK_HEADERS([grp.h])
AC_CHECK_HEADERS([time.h])
AC_CHECK_HEADERS([errno.h])
AC_CHECK_HEADERS([sys/errno.h])
AC_CHECK_HEADERS([asm/errno.h])
AC_CHECK_HEADERS([ncurses.h])
AC_CHECK_HEADERS([curses.h])
AC_CHECK_HEADERS([term.h])

AC_CHECK_HEADERS([gpg-error.h],[have_gpg_error_h=yes],[have_gpg_error_h=no])
if test "x${have_gpg_error_h}" = xyes; then 
  # on Mavericks, this fails if it can't find gpg-error.h
  AC_CHECK_HEADERS([gcrypt.h],[have_gcrypt_h=yes],[have_gcrypt_h=no])
else
  have_gcrypt_h=no
fi

# on old SUN
if test "x${use_fileio}" = xyes; then
  AC_CHECK_HEADERS([rasterfile.h])
fi

# used by usb2000 module
AC_CHECK_HEADERS([usb.h])

AC_CHECK_HEADERS([linux/videodev2.h],[have_v4l2_header=yes],[have_v4l2_header=no])

if test "x${use_parport}" = xyes; then
  # BUG if this header is not present, shouldn't try to use parport
  AC_CHECK_HEADERS([linux/ppdev.h])
fi

# BUG if this header is not present, shouldn't try to use /dev/fb
AC_CHECK_HEADERS([linux/fb.h],[have_linux_fb_h=yes],[have_linux_fb_h=no])

AC_CHECK_HEADERS([asm/types.h])
AC_CHECK_HEADERS([sys/types.h])
AC_CHECK_HEADERS([sys/wait.h])
AC_CHECK_HEADERS([sys/io.h])

if test "x${use_rt_sched}" = xyes; then
  AC_CHECK_HEADERS([sched.h],[have_sched_h=yes],[have_sched_h=no])
fi

AC_CHECK_HEADERS([termios.h unistd.h])
AC_CHECK_HEADERS([sys/timex.h],[have_sys_timex_h=yes],[have_sys_timex_h=no])

if test "x${have_sound}" = xyes; then
  if test "x${have_alsa}" = xyes; then
    AC_CHECK_HEADERS([sys/soundcard.h])
    AC_CHECK_HEADERS([alsa/asoundlib.h])
  fi
  if test "x${have_portaudio}" = xyes; then
    # check for portaudio headers here
    AC_CHECK_HEADERS([portaudio.h])
  fi
fi

if test "x${use_curl}" = xyes; then
  # BUG if libcurl is found but this file is missing,
  # then it still tries to compile
  AC_CHECK_HEADERS([curl/curl.h])
fi

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
if test "x${use_x11}" = xyes; then
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
fi

# opengl, GLUT
if test "x${use_opengl}" = xyes; then

  # on Mac, these headers are in /usr/X11/include...
  AC_CHECK_HEADERS([GL/gl.h])
  AC_CHECK_HEADERS([GL/glext.h], [], [],
[[#ifdef HAVE_GL_GL_H
# include <GL/gl.h>
#endif
  ]])
  AC_CHECK_HEADERS([GL/glx.h])
  AC_CHECK_HEADERS([GL/glu.h])
  AC_CHECK_HEADERS([GL/glut.h])
  AC_CHECK_HEADERS([GL/freeglut.h])
  # on wheatstone, this one is in the nvidia distribution...
  AC_CHECK_HEADERS([GL/glew.h])
fi

if test "x${use_opencl}" = xyes; then
  AC_CHECK_HEADERS([clFFT.h])
fi

if test "x${use_dv}" = xyes; then
  # libdv
  AC_CHECK_HEADERS([libdv/dv.h])
  AC_CHECK_HEADERS([libdv/dv_types.h])
  AC_CHECK_HEADERS([libdv/dv1394.h])
fi

# OpenCV  headers
# old version?
if test "x${use_opencv}" = xyes; then
  # used to check for cvver.h...
  AC_CHECK_HEADERS([opencv/cvver.h opencv/cv.h opencv/highgui.h])
  # newer version?
  AC_CHECK_HEADERS([opencv2/core/core_c.h opencv2/core/version.hpp])
  AC_CHECK_HEADERS([opencv2/imgproc/imgproc_c.h opencv2/objdetect/objdetect.hpp])
  AC_CHECK_HEADERS([opencv2/highgui/highgui_c.h])
fi

if test "x${use_fileio}" = xyes; then
  # png.h is not where expected on MacOS
  AC_CHECK_HEADERS([png.h])
  # BUG If this file is not found, then we should unset HAVE_JPEG
  AC_CHECK_HEADERS([jpeglib.h],[have_jpeglib_h=yes],[have_jpeglib_h=no])

fi

if test "x${use_cuda}" = xyes; then
  # cuda headers
  # These won't be found if the include directories aren't in CPPFLAGS!?
  AC_CHECK_HEADERS([curand.h])
  AC_CHECK_HEADERS([cufft.h])
  AC_CHECK_HEADERS([npp.h])
fi

if test $ac_cv_sizeof_long -eq 8; then
  AC_DEFINE([LONG_64_BIT], 1, [64 bit longs])
fi

if test $ac_cv_sizeof_long -eq 4; then
  AC_DEFINE([LONG_32_BIT], 1, [32 bit longs])
fi

#AC_CHECK_SIZEOF([off64_t])

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

#AC_CHECK_FUNCS([floor ftime getcwd gethostbyname gettimeofday time sleep drand48 srand48 random srandom rand srand memset pow rint select socket sqrt strchr strdup strerror strrchr strstr strtol strtoll])

# We ought to organize these by what library they are in,
# or what header file declares them (unistd.h etc)

# check for seek funcs
AC_CHECK_FUNCS([llseek lseek64])
AC_CHECK_FUNCS([getuid geteuid])
# check for general utility funcs
AC_CHECK_FUNCS([popen getcwd memset flock])
AC_CHECK_FUNCS([getpid])
AC_CHECK_FUNCS([munmap])
# check for time-related funcs
AC_CHECK_FUNCS([gettimeofday ftime time sleep])
AC_CHECK_FUNCS([alarm])
AC_CHECK_FUNCS([usleep])
AC_CHECK_FUNCS([setitimer])
AC_CHECK_FUNCS([tcgetattr tcsetattr cfsetspeed cfgetispeed cfsetispeed cfgetospeed cfsetospeed])
AC_CHECK_FUNCS([isatty])
# check for math funcs
AC_CHECK_FUNCS([floor pow rint sqrt])
# check for string funcs
AC_CHECK_FUNCS([bcopy])
AC_CHECK_FUNCS([memcpy memmove])
AC_CHECK_FUNCS([strchr strdup strerror strrchr strstr])
AC_CHECK_FUNCS([strtol strtoll strtoul strtoull])
AC_CHECK_FUNCS([strncasecmp])
# check for networking funcs
AC_CHECK_FUNCS([gethostbyname select socket])
# check for random number funcs
AC_CHECK_FUNCS([drand48 srand48 random srandom rand srand])
AC_CHECK_FUNCS([ioperm],[have_ioperm=yes],[have_ioperm=no])

#		 macros/colors/Makefile

AC_CONFIG_FILES([Makefile
		 macros/Makefile
		 macros/system/Makefile
		 macros/compute/Makefile
		 macros/startup/Makefile
		 macros/data/Makefile
		 macros/view/Makefile
		 macros/numrec/Makefile
		 macros/gsl/Makefile
		 macros/knox/Makefile
		 macros/flycap/Makefile
		 macros/visca/Makefile
		 macros/v4l2/Makefile
		 macros/gui/Makefile
		 macros/ctrack/Makefile
		 macros/em_exp/Makefile
		 macros/eyemov/Makefile
		 macros/postproc/Makefile
		 macros/analysis/Makefile
		 macros/movie/Makefile
		 macros/opengl/Makefile
		 doc/Makefile
		 libs/Makefile
		 include/Makefile
		 include/veclib/Makefile
		 include/img_file/Makefile
		 include/hips/Makefile
		 libsrc/Makefile
		 libsrc/www/Makefile
		 libsrc/gsl/Makefile
		 libsrc/unix/Makefile
		 libsrc/interpreter/Makefile
		 libsrc/dobj/Makefile
		 libsrc/ports/Makefile
		 libsrc/fann/Makefile
		 libsrc/fileio/Makefile
		 libsrc/hips/Makefile
		 libsrc/rawvol/Makefile
		 libsrc/veclib/Makefile
		 libsrc/veclib2/Makefile
		 libsrc/vec_util/Makefile
		 libsrc/vectree/Makefile
		 libsrc/mvimenu/Makefile
		 libsrc/seq/Makefile
		 libsrc/view/Makefile
		 libsrc/xsupp/Makefile
		 libsrc/psych/Makefile
		 libsrc/cstepit/Makefile
		 libsrc/opencv/Makefile
		 libsrc/visca/Makefile
		 libsrc/knox/Makefile
		 libsrc/opengl/Makefile
		 libsrc/polhemus/Makefile
		 libsrc/ieee1394/Makefile
		 libsrc/flycap/Makefile
		 libsrc/ptgrey/Makefile
		 libsrc/cuda/Makefile
		 libsrc/dither/Makefile
		 libsrc/das1602/Makefile
		 libsrc/meteor/Makefile
		 libsrc/v4l2/Makefile
		 libsrc/gui/Makefile
		 libsrc/nrmenu/Makefile
		 libsrc/sound/Makefile
		 libsrc/parport/Makefile
		 libsrc/opencl/Makefile
		 libsrc/opencl/Random123/Makefile
		 libsrc/opencl/Random123/features/Makefile
		 libsrc/cu2/Makefile
		 src/Makefile])

# These modules should really be external???
#		 libsrc/usb2000/Makefile
#		 libsrc/smarteye/Makefile
#		 libsrc/gps/Makefile

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
Unable to find libpthread.
Building without multi-threading.
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
Unable to find libXt (although libXm is present).
Building without GUI support.
-------------------------------------------------])
    have_libmotif=no
  fi
fi

if test "x${have_sound}" = xno; then
  if test "x${use_sound}" = xyes; then
    AC_MSG_WARN([
---------------------------------
Unable to find portaudio or ALSA sound library.
Building without sound support.
---------------------------------])
    use_sound=no
  fi
fi

if test "x${have_rawvol}" = xno; then
  if test "x${use_rawvol}" = xyes; then
    AC_MSG_WARN([
---------------------------------
Device /dev/rawvol not found.
Building without raw disk volume support.
---------------------------------])
    use_rawvol=no
  fi
fi


if test "x${have_video_device}" = xyes; then
  if test "x${have_v4l2_header}" = xyes; then
    have_v4l2=yes
  else
    have_v4l2=no
  fi
else
  have_v4l2=no
fi

if test "x${have_v4l2}" = xno; then
  if test "x${use_v4l2}" = xyes; then
    AC_MSG_WARN([
---------------------------------
Device /dev/video0 and/or linux/videodev2.h not found.
Building without V4L2 support.
---------------------------------])
    use_v4l2=no
  fi
fi


if test "x${have_parport}" = xno; then
  if test "x${use_parport}" = xyes; then
    AC_MSG_WARN([
---------------------------------
Device /dev/parport0 not found.
Building without parallel port support.
---------------------------------])
    use_parport=no
  fi
fi


if test "x${have_meteor}" = xno; then
  if test "x${use_meteor}" = xyes; then
    AC_MSG_WARN([
---------------------------------
Device /dev/meteor0 not found.
Building without Matrox meteor support.
---------------------------------])
    use_meteor=no
  fi
fi


if test "x${have_visca}" = xno; then
  if test "x${use_visca}" = xyes; then
    AC_MSG_WARN([
---------------------------------
Device /dev/visca not found.
Building without Sony VISCA support.
---------------------------------])
    use_visca=no
  fi
fi

if test "x${fb_dev_present}" = xyes; then
  if test "x${have_linux_fb_h}" = xyes; then
    have_fb_dev=yes
  else
    have_fb_dev=no
  fi
else
  have_fb_dev=no
fi

if test "x${have_fb_dev}" = xno; then
  if test "x${use_fb_dev}" = xyes; then
    AC_MSG_WARN([
--------------------------------------------
Device /dev/fb0 and/or linux/fb.h not found.
Building without frame buffer device support.
--------------------------------------------])
  fi
fi

if test "x${have_das1602}" = xno; then
  if test "x${use_das1602}" = xyes; then
    AC_MSG_WARN([
---------------------------------
Device /dev/adc0 not found.
Building without DAS1602 support.
---------------------------------])
    use_das1602=no
  fi
fi


if test "x${use_rt_sched}" = xyes; then
  if test "x${have_sched_h}" = xno; then
    AC_MSG_WARN([
-------------------------------------------------
Unable to find sched.h.
Building without real-time scheduler support.
-------------------------------------------------])
    use_rt_sched=no
  fi
fi

if test "x${use_rt_sched}" = xyes; then
  if test "x${have_sched}" = xno; then
    AC_MSG_WARN([
-------------------------------------------------------
Unable to find a library containing sched_setscheduler.
Building without real-time scheduler support.
-------------------------------------------------------])
    use_rt_sched=no
  fi
fi

if test "x${have_secret_key}" = xno; then
  AC_MSG_WARN([
---------------------------------
File ./libsrc/interpreter/secret_key.c not found.
Building without default encryption key.
---------------------------------])
fi


if test "x${have_secret_key}" = xyes; then
  AC_DEFINE([HAVE_SECRET_KEY], 1, [custom secret encryption key provided])
fi

if test "x${have_stat}" = xyes; then
  AC_DEFINE([HAVE_STAT], 1, [stat system call available to check file status])
fi

if test "x${have_ctime_r}" = xyes; then
  AC_DEFINE([HAVE_CTIME_R], 1, [ctime_r available for date formatting])
fi

if test "x${have_ctime}" = xyes; then
  AC_DEFINE([HAVE_CTIME], 1, [ctime available for date formatting])
fi

if test "x${have_gmtime_r}" = xyes; then
  AC_DEFINE([HAVE_GMTIME_R], 1, [gmtime_r available for Grenwich Mean Time])
fi

if test "x${have_gmtime}" = xyes; then
  AC_DEFINE([HAVE_GMTIME], 1, [gmtime available for Grenwich Mean Time])
fi

if test "x${have_asctime_r}" = xyes; then
  AC_DEFINE([HAVE_ASCTIME_R], 1, [asctime_r available for date formatting])
fi

if test "x${have_asctime}" = xyes; then
  AC_DEFINE([HAVE_ASCTIME], 1, [asctime available for date formatting])
fi

if test "x${have_localtime_r}" = xyes; then
  AC_DEFINE([HAVE_LOCALTIME_R], 1, [localtime_r available])
fi

if test "x${have_posix_memalign}" = xyes; then
  AC_DEFINE([HAVE_POSIX_MEMALIGN], 1, [posix_memalign available])
fi

if test "x${async_exec}" = xyes; then
  AC_DEFINE([ASYNC_EXEC], 1, [asynchronous execution enabled])
fi

if test "x${thread_safe_query}" = xyes; then
  AC_DEFINE([THREAD_SAFE_QUERY], 1, [thread-safe query features enabled])
fi

if test "x${n_processors}" != xno; then
  AC_DEFINE_UNQUOTED([N_PROCESSORS], ${n_processors}, [multi-processing with $n_processors processors enabled])
fi

if test "x${have_knox}" = xno; then
  if test "x${use_knox}" = xyes; then
  AC_MSG_WARN([
-------------------------------------------
Device /dev/knox not found.
Building without Knox video switcher support.
-------------------------------------------])
    use_knox=no
  fi
fi

# Here we enable Knox support if /dev/knox is present, regardless of the state of use_knox
if test "x${have_knox}" = xyes; then
#  if test "x${use_knox}" = xyes; then
    AC_DEFINE([HAVE_KNOX], 1, [Knox video switcher support enabled])
#  fi
fi

if test "x${have_numrec}" = xno; then
  if test "x${use_numrec}" = xyes; then
    AC_MSG_WARN([
-------------------------------------------
Unable to find libnumrec.
Building without numrec support.
-------------------------------------------])
    use_numrec=no
  fi
#fi
#if test "x${have_numrec}" = xyes; then
else
  if test "x${use_numrec}" = xyes; then
    AC_DEFINE([HAVE_NUMREC], 1, [Numerical Recipes support present])
    AC_DEFINE([USE_NUMREC], 1, [Numerical Recipes support enabled])
  fi
fi
# what does this do, let use test HAVE_NUMREC in makefile's?
AM_CONDITIONAL([HAVE_NUMREC],[test x${use_numrec} = xyes])

if test "x${have_sparselm}" = xno; then
  if test "x${use_sparselm}" = xyes; then
    AC_MSG_WARN([
-------------------------------------------
Unable to find libsparselm.
Building without sparselm support.
-------------------------------------------])
    use_sparselm=no
  fi
else
  if test "x${use_sparselm}" = xyes; then
    AC_DEFINE([HAVE_SPARSELM], 1, [Sparse L-M support enabled])
  fi
fi

# Set config variables

if test "x${have_pmemalign}" = xyes; then
  AC_DEFINE([HAVE_POSIX_MEMALIGN], 1, [memory alignment with posix_memalign supported])
fi

if test "x${have_adjtimex}" = xyes; then
  if test "x${have_sys_timex_h}" = xyes; then
    AC_DEFINE([HAVE_ADJTIMEX], 1, [clock adjustment with adjtimex supported])
  fi
fi

# no code here for older w3c library???

if test "x${have_libcurl}" = xyes; then
  AC_DEFINE([HAVE_LIBCURL], 1, [www communication supported])
fi

if test "x${have_libglut}" = xyes; then
  AC_DEFINE([HAVE_GLUT], 1, [GLUT library present])
fi

if test "x${use_tty_ctl}" = xyes; then
  AC_DEFINE([TTY_CTL], 1, [low-level tty control enabled])
fi

if test "x${use_my_intr}" = xyes; then
  AC_DEFINE([MY_INTR], 1, [scriptable interrupt handler enabled])
fi

if test "x${use_viewers}" = xyes; then
  AC_DEFINE([VIEWERS], 1, [graphical viewing windows enabled])
fi

if test "x${use_cuda}" = xyes; then
  if test "x${working_nvcc}" = xyes ;  then
    if test "x${have_nv_sdk}" = xyes ; then
      if test "x${have_cuda_device}" = xyes ; then
        # We could conceivably have all the software installed,
        # but not have an actual cuda-capable device???
        # Check for devices when determining compute capability!
        AC_DEFINE([HAVE_CUDA], 1, [GPU operations with CUDA enabled])
        if test "x${have_libcurand}" = xyes ; then
	  AC_DEFINE([HAVE_LIBCURAND], 1, [GPU random number generation with libcurand enabled])
        fi
        if test "x${have_libnpp}" = xyes ; then
	  AC_DEFINE([HAVE_LIBNPP], 1, [CUDA GPU operations with libnpp enabled])
        else
          if test "x${have_libnppi}" = xyes ; then
	    AC_DEFINE([HAVE_LIBNPP], 1, [CUDA GPU operations with libnpp enabled])
          fi
        fi
      else
        AC_MSG_WARN([
-------------------------------------------------
Unable to find any cuda devices (although working nvcc found).
Building without CUDA support.
-------------------------------------------------])
      fi
    else
      AC_MSG_WARN([
-------------------------------------------------
Unable to find cuda SDK (although working nvcc found).
Building without CUDA support.
-------------------------------------------------])
    fi
  else
    AC_MSG_WARN([
-------------------------------------------------
Unable to find nvcc (although CUDA support requested).
Building without CUDA support.
-------------------------------------------------])
  fi
fi

if test "x${use_opencl}" = xyes; then
  if test "x${have_libopencl}" = xyes ;  then
    AC_DEFINE([HAVE_OPENCL], 1, [GPU operations with OpenCL enabled])
  else
    AC_MSG_WARN([
-------------------------------------------------
Unable to find libopencl (although requested).
Building without OpenCL support.
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
    AC_DEFINE([HAVE_GUI_INTERFACE], 1, [GUI features enabled])
  else
    AC_MSG_WARN([
-------------------------------------------------
Unable to find libXm.
Building without GUI support.
-------------------------------------------------])
  fi
fi

if test "x${use_stepit}" = xyes; then
  AC_DEFINE([STEPIT], 1, [STEPIT optimization enabled])
fi

if test "x${have_sound}" = xyes; then
  AC_DEFINE([HAVE_SOUND], 1, [Sound support enabled])
  if test "x${have_portaudio}" = xyes; then
    AC_DEFINE([HAVE_PORTAUDIO], 1, [Sound provided with libportaudio])
  fi
  if test "x${have_alsa}" = xyes; then
    AC_DEFINE([HAVE_ALSA], 1, [Sound provided with libasound (ALSA)])
  fi
fi
AM_CONDITIONAL([HAVE_SOUND],[test x${have_sound} = xyes])

if test "x${use_rawvol}" = xyes; then
  AC_DEFINE([HAVE_RAWVOL], 1, [Raw disk volume support enabled])
fi

if test "x${have_v4l2}" = xyes; then
  AC_DEFINE([HAVE_V4L2], 1, [V4L2 video interface support enabled])
fi

if test "x${use_parport}" = xyes; then
  AC_DEFINE([HAVE_PARPORT], 1, [parallel port support enabled])
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

if test "x${have_fb_dev}" = xyes; then
  AC_DEFINE([HAVE_FB_DEV], 1, [Frame buffer device support enabled])
fi

if test "x${use_das1602}" = xyes; then
  AC_DEFINE([HAVE_DAS1602], 1, [Measurement Computing DAS1602 interface support enabled])
fi

if test "x${use_debug}" = xyes; then
  AC_DEFINE([QUIP_DEBUG], 1, [debug features enabled])
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

if test "x${have_erf}" = xyes; then
  AC_DEFINE([HAVE_ERF], 1, [library implementation of erf() available])
fi

if test "x${have_xext}" = xyes; then
  AC_DEFINE([HAVE_X11_EXT], 1, [X11 extensions enabled (w/ libXext)])
fi

if test "x${have_libgl}" = xyes; then
  if test "x${ac_cv_header_GL_gl_h}" = xno; then
    AC_MSG_WARN([
-------------------------------------------------
Unable to find header file GL/gl.h (although libgl is present).
Building without OpenGL support.
-------------------------------------------------])
  else
    if test "x${ac_cv_header_GL_glu_h}" = xno; then
      AC_MSG_WARN([
-------------------------------------------------
Unable to find header file GL/glu.h (although libgl is present).
Building without OpenGL support.
-------------------------------------------------])
    else
      AC_DEFINE([HAVE_OPENGL], 1, [OpenGL graphics support enabled])
    fi
  fi
fi

if test "x${have_sgisync}" = xyes; then
  AC_DEFINE([HAVE_VIDEOSYNCSGI], 1, [OpenGL video sync support enabled])
fi

if test "x${have_libx11}" = xyes; then
  if test "x${ac_cv_header_X11_Xlib_h}" = xno; then
    AC_MSG_WARN([
-------------------------------------------------
Unable to find header file X11/Xlib.h (although libX11 is present).
Building without X11 support.
-------------------------------------------------])
  else
    AC_DEFINE([HAVE_X11], 1, [X11 window system support enabled])
    if test "x${use_vbl}" = xyes; then
      if test "x${have_ioperm}" = xyes; then
        AC_DEFINE([HAVE_VBL], 1, [Video blanking support via i/o registers])
      else
        AC_MSG_WARN([
-------------------------------------------------
Unable to access i/o registers.
Building without vertical blanking support.
-------------------------------------------------])
      fi
    fi
  fi
fi

if test "x${have_libusb}" = xyes; then
  AC_DEFINE( [HAVE_LIBUSB], 1, [USB communication library])
fi

if test "x${have_libtermcap}" = xyes; then
  AC_DEFINE( [HAVE_TERMCAP], 1, [Termcap terminal capability library])
fi

if test "x${have_libgcrypt}" = xyes; then
  if test "x${have_gcrypt_h}" = xyes; then
    AC_DEFINE( [HAVE_LIBGCRYPT], 1, [GNU Encryption library])
    AC_DEFINE( [HAVE_ENCRYPTION], 1, [Encryption support])
  else
    if test "x${have_gpg_error_h}" = xno; then
      AC_MSG_WARN([
-------------------------------------------------
Unable to find header file gpg-error.h (although libgcrypt is present).
Building without gcrypt support.
-------------------------------------------------])
    else
      AC_MSG_WARN([
-------------------------------------------------
Unable to find header file gcrypt.h (although libgcrypt is present).
Building without gcrypt support.
-------------------------------------------------])
    fi
  fi
fi

have_gsl=no
if test "x${have_libgslcblas}" = xyes; then
  if test "x${have_libgsl}" = xyes; then
    have_gsl=yes
    AC_DEFINE( [HAVE_GSL], 1, [GNU Scientific Library support enabled])
  fi
fi

if test "x${have_fann}" = xyes; then
  AC_DEFINE([HAVE_FANN], 1, [FANN neural network support enabled])
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

# BUG?  we need to have all the ffmpeg libs to compile...

if test "x${have_libswscale}" = xyes; then
  AC_DEFINE([HAVE_LIBSWSCALE], 1, [AVI file support enabled (w/ libswscale)])
fi

if test "x${have_libavcodec}" = xyes; then
  AC_DEFINE([HAVE_LIBAVCODEC], 1, [AVI file support enabled (w/ libavcodec)])
fi

#if test "x${have_libavcodec2}" = xyes; then
#  AC_DEFINE([HAVE_LIBAVCODEC2], 1, [AVI file support enabled (w/ libavcodec v2)])
#fi

# Need other libs as well here?
if test "x${have_libavformat}" = xyes; then
  AC_DEFINE([HAVE_LIBAVFORMAT], 1, [AVI file support enabled (w/ libavformat)])
fi

if test "x${have_libmatio}" = xyes; then
  # hdf5 lib is required on mac only...
  # how to test here???
  #if test "x${have_libhdf5}" = xyes; then
    AC_DEFINE([HAVE_MATIO], 1, [Matlab i/o enabled])
  #fi
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
  if test "x${have_libraw1394}" = xyes; then
    AC_DEFINE([HAVE_LIBDV], 1, [Support for ieee1394 cameras])
  fi
fi

if test "x${have_libflycap}" = xyes; then
  AC_DEFINE([HAVE_LIBFLYCAP], 1, [Support for Point Grey Research cameras w/ libflycap])
fi

if test "x${have_libdc1394}" = xyes; then
  if test "x${have_dc1394_header}" = xno; then
    AC_MSG_WARN([
---------------------------------------------
libdc1394 is present, but header file dc1394.h is not...
Building without libdc1394 support.
---------------------------------------------])
  else
    AC_DEFINE([HAVE_LIBDC1394], 1, [Support for Point Grey Research ieee1394 cameras])
  fi
fi

if test "x${have_libglew}" = xyes; then
  AC_DEFINE([HAVE_LIBGLEW],1,[utility routines for openGL & CUDA])
fi

if test "x${have_clfft}" = xyes; then
  AC_DEFINE([HAVE_CLFFT],1,[OpenCL FFT library])
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

if test "x${use_opencl}" = xyes; then
  if test "x${have_libopencl}" = xno ;  then
    AC_MSG_WARN([
---------------------------------------------
Unable to find OpenCL library.
Building without OpenCL support...
---------------------------------------------])
  fi
fi

if test "x${have_pthreads}" = xno; then
  AC_MSG_WARN([
----------------------------------------
Unable to find the pthreads library.
Building without multi-threading...
----------------------------------------])
fi

if test "x${use_fann}" = xyes; then
  if test "x${have_fann}" != xyes; then
    AC_MSG_WARN([
-----------------------------------------
Unable to find the fann library.
Building without FANN neural network support...
-----------------------------------------])
  fi
fi

if test "x${use_matio}" = xyes; then
  if test "x${have_libmatio}" != xyes; then
  AC_MSG_WARN([
-----------------------------------------
Unable to find the matio library.
Building without matio support...
-----------------------------------------])
  else
    if test "x${have_libhdf5}" != xyes; then
  AC_MSG_WARN([
-----------------------------------------
Unable to find the hdf5 library.
Building without matio support...
-----------------------------------------])
    fi
  fi
fi

if test "x${have_libmotif}" != xyes; then
  if test "x${use_motif_gui}" = xyes; then
  AC_MSG_WARN([
--------------------------------------------------------------------
Unable to find libXm.
Building without gui support...
--------------------------------------------------------------------])
  fi
fi


# At the moment, posix_memalign is only used in the rawvol module,
# although it probably should be used in dobj...

if test "x${have_pmemalign}" = xno; then
  AC_MSG_WARN([
-----------------------------------------
Unable to find posix_memalign.
May have alignment-related problems...
-----------------------------------------])
fi

if test "x${have_libcurl}" != xyes; then
  if test "x${use_curl}" = xyes; then
  AC_MSG_WARN([
------------------------------------------
Unable to find libcurl.
Building without www support.
------------------------------------------])
  fi
fi

if test "x${use_fileio}" = xyes; then
  if test "x${have_libtiff}" != xyes; then
    AC_MSG_WARN([
------------------------------------------
Unable to find libtiff.
Building without TIFF file support.
------------------------------------------])
  fi

  if test "x${have_libpng}" != xyes; then
    AC_MSG_WARN([
------------------------------------------
Unable to find libpng.
Building without PNG file support.
------------------------------------------])
  fi

  if test "x${have_libjpeg}" != xyes; then
    AC_MSG_WARN([
------------------------------------------
Unable to find libjpeg.
Building without JPEG file support.
------------------------------------------])
  fi

  if test "x${have_libavformat}" != xyes; then
    AC_MSG_WARN([
------------------------------------------
Unable to find libavformat.
Building without AVI file support.
------------------------------------------])
  fi
fi

if test "x${have_opencv}" != xyes; then
  if test "x${use_opencv}" = xyes; then
    AC_MSG_WARN([
------------------------------------------
Unable to find one or more OpenCV libraries.
Building without OpenCV support.
------------------------------------------])
  fi
fi

if test "x${have_ncurses}" != xyes; then
  if test "x${use_ncurses}" = xyes; then
    AC_MSG_WARN([
------------------------------------------
Unable to find ncurses library.
Building without ncurses support.
------------------------------------------])
  fi
fi

if test "x${have_gsl}" != xyes; then
  if test "x${use_gsl}" = xyes; then
    AC_MSG_WARN([
------------------------------------------
Unable to find GNU scientific library (libgsl,libgslcblas).
Building without GSL support.
------------------------------------------])
  fi
fi

if test "x${have_libflycap}" != xyes; then
  if test "x${use_flycap}" = xyes; then
    AC_MSG_WARN([
------------------------------------------
Unable to find libflycap.
Building without support for Point Grey digital cameras.
------------------------------------------])
  fi
fi

if test "x${have_libdc1394}" != xyes; then
  if test "x${use_dc1394}" = xyes; then
    AC_MSG_WARN([
------------------------------------------
Unable to find libdc1394.
Building without support for IEEE1394 digital cameras.
------------------------------------------])
  fi
fi

  if test "x${have_libusb}" != xyes; then
    AC_MSG_WARN([
------------------------------------------
Unable to find libusb1.
Building without USB functionality.
------------------------------------------])
  fi

if test "x${use_tty_ctl}" = xyes; then
  if test "x${have_libtermcap}" != xyes; then
    AC_MSG_WARN([
------------------------------------------
Unable to find libtermcap.
Building without termcap functionality.
------------------------------------------])
  fi
fi

if test "x${have_libglut}" != xyes; then
  if test "x${use_opengl}" = xyes; then
    AC_MSG_WARN([
------------------------------------------
Unable to find libglut.
Building without GLUT functionality for openGL.
------------------------------------------])
  fi
fi

# Create a variable we can test in Makefile.am:
AM_CONDITIONAL([HAVE_MAKEINFO], [test "x$have_makeinfo" = "xyes"])

if test "x${have_makeinfo}" != xyes; then
  AC_MSG_WARN([
------------------------------------------
Unable to find makeinfo.
You will not be able to build the documentation.
(Install texinfo package?).
------------------------------------------])
else

  if test "x${have_texi2dvi}" != xyes; then
    AC_MSG_WARN([
------------------------------------------
Unable to find texi2dvi.
You will not be able to build the pdf form of the documentation.
------------------------------------------])
  fi

fi

