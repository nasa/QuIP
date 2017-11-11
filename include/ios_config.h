// This version hand edited to work with iOS
#ifndef _IOS_CONFIG_H_
#define _IOS_CONFIG_H_

/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

//#undef USE_OPEN_UDID
//#define USE_OPEN_UDID	1	// jbm added this file

#define HAVE_FILEIO	1
#define HAVE_ENCRYPTION	1
#define HAVE_SECRET_KEY	1

#define HAVE_ASCTIME_R	1	// new for unix - present in iOS???

/* real-time scheduler control enabled */
/* #undef ALLOW_RT_SCHED */

/* asynchronous execution enabled */
//#define ASYNC_EXEC 1

/* cautious checking enabled */
#define CAUTIOUS 1

/* Define to one of `_getb67', `GETB67', `getb67' for Cray-2 and Cray-YMP
   systems. This function is required for `alloca.c' support on those systems.
   */
/* #undef CRAY_STACKSEG_END */

/* CUDA Compute Capability 0.0 */
/* #undef CUDA_COMP_CAP */

/* Define to 1 if using `alloca.c'. */
/* #undef C_ALLOCA */

/* debug features enabled */
#define QUIP_DEBUG 1

/* clock adjustment with adjtimex supported */
/* #undef HAVE_ADJTIMEX */

/* Define to 1 if you have the `alarm' function. */
//#define HAVE_ALARM 1

/* Define to 1 if you have `alloca', as a function or macro. */
//#define HAVE_ALLOCA 1

/* Define to 1 if you have <alloca.h> and it should be used (not on Ultrix).
   */
//#define HAVE_ALLOCA_H 1

/* Define to 1 if you have the <alsa/asoundlib.h> header file. */
/* #undef HAVE_ALSA_ASOUNDLIB_H */

/* Define to 1 if you have the <asm/errno.h> header file. */
/* #undef HAVE_ASM_ERRNO_H */

/* Define to 1 if you have the <asm/types.h> header file. */
/* #undef HAVE_ASM_TYPES_H */

/* Define to 1 if you have the <assert.h> header file. */
#define HAVE_ASSERT_H 1

/* Define to 1 if you have the `bcopy' function. */
#define HAVE_BCOPY 1

/* Define to 1 if you have the <ctype.h> header file. */
#define HAVE_CTYPE_H 1

/* GPU operations with CUDA enabled */
/* #undef HAVE_CUDA */

/* Define to 1 if you have the <curand.h> header file. */
/* #undef HAVE_CURAND_H */

/* www communication supported */
/* #undef HAVE_CURL */

/* Define to 1 if you have the <curl/curl.h> header file. */
//#define HAVE_CURL_CURL_H 1

/* Define to 1 if you have the <curses.h> header file. */
#define HAVE_CURSES_H 1

/* Measurement Computing DAS1602 interface support enabled */
/* #undef HAVE_DAS1602 */

/* Define to 1 if you have the <dc1394/dc1394.h> header file. */
//#define HAVE_DC1394_DC1394_H 1

/* Define to 1 if you have the `drand48' function. */
#define HAVE_DRAND48 1

/* Define to 1 if you have the <errno.h> header file. */
#define HAVE_ERRNO_H 1

/* Frame buffer device support enabled */
/* #undef HAVE_FB_DEV */

/* Define to 1 if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define to 1 if you have the `flock' function. */
//#define HAVE_FLOCK 1

/* Define to 1 if you have the `floor' function. */
#define HAVE_FLOOR 1

/* Define to 1 if you have the `fork' function. */
//#define HAVE_FORK 1

/* Define to 1 if you have the `ftime' function. */
#define HAVE_FTIME 1

/* Define to 1 if you have the `getcwd' function. */
#define HAVE_GETCWD 1

/* Define to 1 if you have the `gethostbyname' function. */
#define HAVE_GETHOSTBYNAME 1

/* Define to 1 if you have the `getpagesize' function. */
#define HAVE_GETPAGESIZE 1

/* Define to 1 if you have the `gettimeofday' function. */
#define HAVE_GETTIMEOFDAY 1

/* Define to 1 if you have the `getuid' function. */
#define HAVE_GETUID 1

/* GLUT library present */
//#define HAVE_GLUT 1

/* Define to 1 if you have the <GL/glew.h> header file. */
/* #undef HAVE_GL_GLEW_H */

/* Define to 1 if you have the <GL/glut.h> header file. */
//#define HAVE_GL_GLUT_H 1

/* Define to 1 if you have the <GL/glu.h> header file. */
//#define HAVE_GL_GLU_H 1

/* Define to 1 if you have the <GL/glx.h> header file. */
//#define HAVE_GL_GLX_H 1

/* Define to 1 if you have the <GL/gl.h> header file. */
//#define HAVE_GL_GL_H 1

/* Define to 1 if you have the <grp.h> header file. */
//#define HAVE_GRP_H 1

/* GNU Scientific Library support enabled */
//#define HAVE_GSL 1

/* response history enabled */
#define HAVE_HISTORY 1

/* Define to 1 if you have the <ieeefp.h> header file. */
/* #undef HAVE_IEEEFP_H */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <ioctl.h> header file. */
/* #undef HAVE_IOCTL_H */

/* Define to 1 if you have the <jpeglib.h> header file. */
//#define HAVE_JPEGLIB_H 1

/* JPEG file support enabled */
//#define HAVE_JPEG_SUPPORT 1

/* AVI file support enabled (w/ libavcodec) */
//#define HAVE_LIBAVCODEC 1

/* AVI file support enabled (w/ libavformat) */
//#define HAVE_LIBAVFORMAT 1

/* GPU random number generation with libcurand enabled */
/* #undef HAVE_LIBCURAND */

/* Support for Point Grey Research ieee1394 cameras */
//#define HAVE_LIBDC1394 1

/* Support for ieee1394 cameras */
/* #undef HAVE_LIBDV */

/* Define to 1 if you have the <libdv/dv1394.h> header file. */
/* #undef HAVE_LIBDV_DV1394_H */

/* Define to 1 if you have the <libdv/dv.h> header file. */
/* #undef HAVE_LIBDV_DV_H */

/* Define to 1 if you have the <libdv/dv_types.h> header file. */
/* #undef HAVE_LIBDV_DV_TYPES_H */

/* Define to 1 if you have the <libintl.h> header file. */
//#define HAVE_LIBINTL_H 1

/* GPU operations libnpp enabled */
/* #undef HAVE_LIBNPP */

/* AVI file support enabled (w/ libswscale) */
//#define HAVE_LIBSWSCALE 1

/* Define to 1 if you have the <limits.h> header file. */
#define HAVE_LIMITS_H 1

/* Define to 1 if you have the <linux/fb.h> header file. */
/* #undef HAVE_LINUX_FB_H */

/* Define to 1 if you have the <linux/fd.h> header file. */
/* #undef HAVE_LINUX_FD_H */

/* Define to 1 if you have the <linux/fs.h> header file. */
/* #undef HAVE_LINUX_FS_H */

/* Define to 1 if you have the <linux/ppdev.h> header file. */
/* #undef HAVE_LINUX_PPDEV_H */

/* Define to 1 if you have the <linux/videodev2.h> header file. */
/* #undef HAVE_LINUX_VIDEODEV2_H */

/* Define to 1 if you have the `llseek' function. */
/* #undef HAVE_LLSEEK */

/* Define to 1 if you have the `lseek64' function. */
/* #undef HAVE_LSEEK64 */

/* Define to 1 if your system has a GNU libc compatible `malloc' function, and
   to 0 otherwise. */
#define HAVE_MALLOC 1

/* Define to 1 if you have the <malloc.h> header file. */
/* #undef HAVE_MALLOC_H */

/* Define to 1 if you have the <math.h> header file. */
#define HAVE_MATH_H 1

/* Matlab i/o enabled */
//#define HAVE_MATIO 1

/* memory alignment with memalign supported */
/* #undef HAVE_MEMALIGN */

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the `memset' function. */
#define HAVE_MEMSET 1

/* Matrox meteor video interface support enabled */
/* #undef HAVE_METEOR */

/* Define to 1 if you have a working `mmap' system call. */
#define HAVE_MMAP 1

/* MOTIF gui support enabled */
//#define HAVE_MOTIF 1

/* Define to 1 if you have the `munmap' function. */
#define HAVE_MUNMAP 1

/* Screen drawing with ncurses */
//#define HAVE_NCURSES 1

/* Define to 1 if you have the <ncurses.h> header file. */
#define HAVE_NCURSES_H 1

/* Define to 1 if you have the <netdb.h> header file. */
#define HAVE_NETDB_H 1

/* Define to 1 if you have the <netinet/in.h> header file. */
#define HAVE_NETINET_IN_H 1

/* Define to 1 if you have the <nppi.h> header file. */
/* #undef HAVE_NPPI_H */

/* Numerical Recipes support enabled */
#define HAVE_NUMREC 1

/* Linkage to OpenCV library */
//#define HAVE_OPENCV 1

/* Define to 1 if you have the <opencv2/core/version.hpp> header file. */
//#define HAVE_OPENCV2_CORE_VERSION_HPP 1

/* Define to 1 if you have the <opencv/cvver.h> header file. */
/* #undef HAVE_OPENCV_CVVER_H */

/* Define to 1 if you have the <opencv/cv.h> header file. */
//#define HAVE_OPENCV_CV_H 1

/* Define to 1 if you have the <opencv/highgui.h> header file. */
//#define HAVE_OPENCV_HIGHGUI_H 1

/* OpenGL graphics support enabled */
//#define HAVE_OPENGL 1

/* parallel port support enabled */
/* #undef HAVE_PARPORT */

/* PIC LED controller support enabled */
/* #undef HAVE_PIC */

/* PNG file support enabled */
//#define HAVE_PNG 1

/* Define to 1 if you have the <png.h> header file. */
//#define HAVE_PNG_H 1

/* Define to 1 if you have the `popen' function. */
#define HAVE_POPEN 1

/* Define to 1 if you have the `pow' function. */
#define HAVE_POW 1

/* CPU information available via /proc/cpuinfo */
/* #undef HAVE_PROC_CPUINFO */

/* Multi-threading with pthreads */
//#define HAVE_PTHREADS 1

/* Define to 1 if you have the <pthread.h> header file. */
//#define HAVE_PTHREAD_H 1

/* Define to 1 if you have the <pwd.h> header file. */
//#define HAVE_PWD_H 1

/* Define to 1 if you have the `rand' function. */
#define HAVE_RAND 1

/* Define to 1 if you have the `random' function. */
#define HAVE_RANDOM 1

/* Define to 1 if you have the <rasterfile.h> header file. */
/* #undef HAVE_RASTERFILE_H */

/* Define to 1 if you have the `rint' function. */
#define HAVE_RINT 1

/* library implementation of round() available */
#define HAVE_ROUND 1

/* Define to 1 if you have the <sched.h> header file. */
//#define HAVE_SCHED_H 1

/* Define to 1 if you have the `sched_setparam' function. */
/* #undef HAVE_SCHED_SETPARAM */

/* Define to 1 if you have the `sched_setscheduler' function. */
/* #undef HAVE_SCHED_SETSCHEDULER */

/* Define to 1 if you have the `select' function. */
#define HAVE_SELECT 1

/* Define to 1 if you have the <signal.h> header file. */
//#define HAVE_SIGNAL_H 1

/* Define to 1 if you have the `sleep' function. */
#define HAVE_SLEEP 1

/* Define to 1 if you have the `socket' function. */
#define HAVE_SOCKET 1

/* Sound support enabled */
/* #undef HAVE_SOUND */

/* Define to 1 if you have the `sqrt' function. */
#define HAVE_SQRT 1

/* Define to 1 if you have the `srand' function. */
#define HAVE_SRAND 1

/* Define to 1 if you have the `srand48' function. */
#define HAVE_SRAND48 1

/* Define to 1 if you have the `srandom' function. */
#define HAVE_SRANDOM 1

/* stat system call available to check file status */
#define HAVE_STAT 1

#define HAVE_CTIME 1
#define HAVE_CTIME_R 1
#define HAVE_GMTIME_R 1

/* Define to 1 if you have the <stdarg.h> header file. */
#define HAVE_STDARG_H 1

/* Define to 1 if stdbool.h conforms to C99. */
/* #undef HAVE_STDBOOL_H */

/* Define to 1 if you have the <stddef.h> header file. */
#define HAVE_STDDEF_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the `strchr' function. */
#define HAVE_STRCHR 1

/* Define to 1 if you have the `strdup' function. */
#define HAVE_STRDUP 1

/* Define to 1 if you have the `strerror' function. */
#define HAVE_STRERROR 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the `strncasecmp' function. */
#define HAVE_STRNCASECMP 1

/* Define to 1 if you have the `strrchr' function. */
#define HAVE_STRRCHR 1

/* Define to 1 if you have the `strstr' function. */
#define HAVE_STRSTR 1

/* Define to 1 if you have the `strtol' function. */
#define HAVE_STRTOL 1

/* Define to 1 if you have the <sys/disklabel.h> header file. */
/* #undef HAVE_SYS_DISKLABEL_H */

/* Define to 1 if you have the <sys/file.h> header file. */
//#define HAVE_SYS_FILE_H 1

/* Define to 1 if you have the <sys/filio.h> header file. */
//#define HAVE_SYS_FILIO_H 1

/* Define to 1 if you have the <sys/ioctl.h> header file. */
#define HAVE_SYS_IOCTL_H 1

/* Define to 1 if you have the <sys/io.h> header file. */
/* #undef HAVE_SYS_IO_H */

/* Define to 1 if you have the <sys/ipc.h> header file. */
//#define HAVE_SYS_IPC_H 1

/* Define to 1 if you have the <sys/mman.h> header file. */
//#define HAVE_SYS_MMAN_H 1

/* Define to 1 if you have the <sys/mount.h> header file. */
//#define HAVE_SYS_MOUNT_H 1

/* Define to 1 if you have the <sys/param.h> header file. */
//#define HAVE_SYS_PARAM_H 1

/* Define to 1 if you have the <sys/resource.h> header file. */
//#define HAVE_SYS_RESOURCE_H 1

/* Define to 1 if you have the <sys/shm.h> header file. */
//#define HAVE_SYS_SHM_H 1

/* Define to 1 if you have the <sys/socket.h> header file. */
#define HAVE_SYS_SOCKET_H 1

/* Define to 1 if you have the <sys/soundcard.h> header file. */
/* #undef HAVE_SYS_SOUNDCARD_H */

/* Define to 1 if you have the <sys/statvfs.h> header file. */
//#define HAVE_SYS_STATVFS_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/timeb.h> header file. */
//#define HAVE_SYS_TIMEB_H 1

/* Define to 1 if you have the <sys/timex.h> header file. */
/* #undef HAVE_SYS_TIMEX_H */

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/wait.h> header file. */
//#define HAVE_SYS_WAIT_H 1

/* Termcap terminal capability library */
#define HAVE_TERMCAP 1

/* Define to 1 if you have the <termios.h> header file. */
//#define HAVE_TERMIOS_H 1

/* Define to 1 if you have the <term.h> header file. */
#define HAVE_TERM_H 1

/* TIFF file support enabled */
//#define HAVE_TIFF 1

/* Define to 1 if you have the `time' function. */
#define HAVE_TIME 1

/* Define to 1 if you have the <time.h> header file. */
#define HAVE_TIME_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the <usb.h> header file. */
/* #undef HAVE_USB_H */

/* V4L2 video interface support enabled */
/* #undef HAVE_V4L2 */

/* Define to 1 if you have the <varargs.h> header file. */
/* #undef HAVE_VARARGS_H */

/* Define to 1 if you have the `vfork' function. */
#define HAVE_VFORK 1

/* Define to 1 if you have the <vfork.h> header file. */
/* #undef HAVE_VFORK_H */

/* OpenGL graphics support enabled */
//#define HAVE_VIDEOSYNCSGI 1

/* Sony VISCA camera control protocol support enabled */
/* #undef HAVE_VISCA */

/* Define to 1 if `fork' works. */
//#define HAVE_WORKING_FORK 1

/* Define to 1 if `vfork' works. */
//#define HAVE_WORKING_VFORK 1

/* X11 window system support enabled */
//#define HAVE_X11 1

/* X11 extensions enabled (w/ libXext) */
//#define HAVE_X11_EXT 1

/* Define to 1 if you have the <X11/extensions/XShm.h> header file. */
//#define HAVE_X11_EXTENSIONS_XSHM_H 1

/* Define to 1 if you have the <X11/Intrinsic.h> header file. */
//#define HAVE_X11_INTRINSIC_H 1

/* Define to 1 if you have the <X11/Xlib.h> header file. */
//#define HAVE_X11_XLIB_H 1

/* Define to 1 if you have the <X11/Xmd.h> header file. */
//#define HAVE_X11_XMD_H 1

/* Define to 1 if you have the <X11/Xutil.h> header file. */
//#define HAVE_X11_XUTIL_H 1

/* Define to 1 if you have the <Xm/XmAll.h> header file. */
//#define HAVE_XM_XMALL_H 1

/* Define to 1 if you have the <Xm/Xm.h> header file. */
//#define HAVE_XM_XM_H 1

/* Define to 1 if the system has the type `_Bool'. */
#define HAVE__BOOL 1

/* GUI features enabled */
#define HAVE_GUI_INTERFACE 1

/* 32 bit longs */
#define LONG_32_BIT 1

/* 64 bit longs */
//#define LONG_64_BIT 1

/* Define to 1 if your C compiler doesn't accept -c and -o together. */
/* #undef NO_MINUS_C_MINUS_O */

/* multi-processing with $n_processors processors enabled */
//#define N_PROCESSORS 16
#define N_PROCESSORS 1

/* Name of package */
#define PACKAGE "quip"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "jeffrey.b.mulligan@nasa.gov"

/* Define to the full name of this package. */
#define PACKAGE_NAME "quip"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "quip 0.4"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "quip"

/* Define to the version of this package. */
#define PACKAGE_VERSION "0.4"

/* The size of `long', as computed by sizeof. */
#define SIZEOF_LONG 4

/* If using the C implementation of alloca, define if you know the
   direction of stack growth for your system; otherwise it will be
   automatically deduced at runtime.
	STACK_DIRECTION > 0 => grows toward higher addresses
	STACK_DIRECTION < 0 => grows toward lower addresses
	STACK_DIRECTION = 0 => direction of growth unknown */
/* #undef STACK_DIRECTION */

/* Define to 1 if you have the ANSI C header files. */
//#define STDC_HEADERS 1

/* STEPIT optimization enabled */
#define STEPIT 1

/* thread-safe query features enabled */
//#define THREAD_SAFE_QUERY 1

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
#define TIME_WITH_SYS_TIME 1

/* low-level tty control enabled */
//#define TTY_CTL 1

/* use internal memory manager */
/* #undef USE_GETBUF */
//#define USE_GETBUF 1

/* Version number of package */
#define VERSION "0.4"

/* graphical viewing windows enabled */
#define VIEWERS 1

/* Define for Solaris 2.5.1 so the uint32_t typedef from <sys/synch.h>,
   <pthread.h>, or <semaphore.h> is not used. If the typedef was allowed, the
   #define below would cause a syntax error. */
/* #undef _UINT32_T */

/* Define for Solaris 2.5.1 so the uint64_t typedef from <sys/synch.h>,
   <pthread.h>, or <semaphore.h> is not used. If the typedef was allowed, the
   #define below would cause a syntax error. */
/* #undef _UINT64_T */

/* Define for Solaris 2.5.1 so the uint8_t typedef from <sys/synch.h>,
   <pthread.h>, or <semaphore.h> is not used. If the typedef was allowed, the
   #define below would cause a syntax error. */
/* #undef _UINT8_T */

/* Define to `int' if <sys/types.h> doesn't define. */
/* #undef gid_t */

/* Define to the type of a signed integer type of width exactly 16 bits if
   such a type exists and the standard includes do not define it. */
/* #undef int16_t */

/* Define to the type of a signed integer type of width exactly 32 bits if
   such a type exists and the standard includes do not define it. */
/* #undef int32_t */

/* Define to the type of a signed integer type of width exactly 64 bits if
   such a type exists and the standard includes do not define it. */
/* #undef int64_t */

/* Define to rpl_malloc if the replacement function should be used. */
/* #undef malloc */

/* Define to `int' if <sys/types.h> does not define. */
/* #undef mode_t */

/* Define to `long int' if <sys/types.h> does not define. */
/* #undef off_t */

/* Define to `int' if <sys/types.h> does not define. */
/* #undef pid_t */

/* Define to `unsigned int' if <sys/types.h> does not define. */
/* #undef size_t */

/* Define to `int' if <sys/types.h> doesn't define. */
/* #undef uid_t */

/* Define to the type of an unsigned integer type of width exactly 16 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint16_t */

/* Define to the type of an unsigned integer type of width exactly 32 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint32_t */

/* Define to the type of an unsigned integer type of width exactly 64 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint64_t */

/* Define to the type of an unsigned integer type of width exactly 8 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint8_t */

/* Define as `fork' if `vfork' does not work. */
/* #undef vfork */

#define VERY_CAUTIOUS	// just for debugging

#endif // ! _IOS_CONFIG_H_
