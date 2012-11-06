
/* fg_routines.h
 *
 *  9/02/97 mjw
 *  9/03/97 mjw added fg_put_roi()
 *  9/19/97 mjw added several routines
 *  3/22/99 mjw v2.2
 */

#ifndef _FG_ROUTINES
#define _FG_ROUTINES

#include "quip_config.h"
#include "query.h"

#ifdef HAVE_SYS_FCNTL_H
#include <sys/fcntl.h>
#endif

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <unistd.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_MEMORY_H
#include <memory.h>
#endif

#include "ioctl_meteor.h"

#if defined(ATI_FG)

/* define ATI_FG if want to write directly to ATI frame buffer */
# define DISPLAY_FG
# include "ioctl_ati_fb.h"
# define DEF_DISPLAY_DEVICE		"/dev/ati_fb"
# define DEF_SCREEN_WIDTH	1024
# define SCREEN_OFFSET          0
/*# define SCREEN_OFFSET          (20*DEF_SCREEN_WIDTH+20) */

#elif defined(MGA_FG)

/* define MGA_FG if want to write directly to Matrox frame buffer */
# define DISPLAY_FG
# include "ioctl_mga_fb.h"
# define DEF_DISPLAY_DEVICE		"/dev/mga_fb"
# define DEF_SCREEN_WIDTH	1024
# define SCREEN_OFFSET            0

#endif

#define HIMEM_FG
/* define HIMEM_FG if want to write to RAM above high_memory */
#ifdef HIMEM_FG
# define DEF_HIMEM_DEVICE	"/dev/himemfb"
# include "ioctl_himemfb.h"
#endif

#define	MAX_ALLOW_CONSEC_ERRORS	10
#define	TRUE	1
#define	FALSE	0

#define DEF_METEOR_DEVICE	"/dev/meteor0"
#define DEF_NUM_COLS		640
#define DEF_NUM_ROWS		480
#define DEF_SCREEN_BYTE_DEPTH	4

/* video_source arguments for fg_open() */
#define SOURCE_NTSC	 	METEOR_INPUT_DEV0
#define SOURCE_SVIDEO	 	METEOR_INPUT_DEV_SVIDEO

/* video_format arguments for fg_open() */
#define MERGE_FIELDS	 	METEOR_GEO_RGB24
#define SEPARATE_FIELDS	 	METEOR_GEO_RGB24_2FLD

/* ram_location arguments for fg_open() */
#define DISPLAY_RAM	0
#define BIGPHYS_RAM	1
#define HIMEM_RAM	2

/* values for fg_put_box() */
#define FG_RED		0x00ff0000
#define	FG_GREEN	0x0000ff00
#define	FG_BLUE		0x000000ff
#define	FG_WHITE	0x00ffffff
#define	FG_BLACK	0x00000000

/* prototypes */
/*
 * sets up meteor frame grabber in rgb888 mode
 * returns 0 on success; neg number on failure
 * example arguments:
 *      video_source: SOURCE_NTSC, SOURCE_SVIDEO
 *      video_format: MERGE_FIELDS, SEPARATE_FIELDS
 *      ram_location: DISPLAY_RAM, BIGPHYS_RAM, HIMEM_RAM
 */
int fg_open(QSP_ARG_DECL  int video_source, int video_format, int ram_location);
void fg_close();

/*
 * captures one frame
 * returns 0 on success -1 on failure
 */
int fg_capture();

/*
 * turn on/off frame_grabber for continuous capture
 * returns 0 on success -1 on failure
 */
int fg_run_continuous();
void fg_stop();

/*
 * get a region of interest from video ram to buf
 * xoffset is from left
 * yoffset is from top
 * pixel format in destination buf is rgbrgbrgb ...
 */
void fg_get_roi(int xoffset, int yoffset, int height, int width, u_char* buf);

/*
 * put a region of interest from a buf to video ram
 * xoffset is from left
 * yoffset is from top
 * pixel format in source buf is rgbrgbrgb ...
 *
 */
void fg_put_roi(int xoffset, int yoffset, int height, int width, u_char* buf);

/*
 * put a grey region of interest from a buf to video ram
 * xoffset is from left
 * yoffset is from top
 * 8 bits per pixel in buffer
 *
 */
void 
fg_put_grey_roi(int xoffset, int yoffset, int height, int width, u_char* buf);

/*
 * place a rectangle on the image
 * xoffset is from left
 * yoffset is from top
 * value is of format xrgb (so red would be 0x00ff0000)
 */
void fg_put_box(int xoffset, int yoffset, int height, int width, uint32_t value);

/* set various parameters
 */
int fg_rows( int rows );  /* 20 .. 480 */
int fg_cols( int cols );  /* 40 .. 640 */
int fg_hue( int hue );    /* -128 .. 127 def 0 */
int fg_brightness( int brightness ); /* 0 .. 255 def 128 */
int fg_contrast( int contrast ); /* 0 .. 255 def 64 */
int fg_gamma( int gamma ); /* TRUE/FALSE def FALSE */

int assoc_pids(pid_t,pid_t);
int unassoc_pids(pid_t,pid_t);

/* use this only if doing low level operations */
u_char *fg_buf_addr();

#endif
