
#include "quip_config.h"

char VersionId_meteor_mgeo[] = QUIP_VERSION_STRING;

#ifdef HAVE_METEOR

#include <stdio.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>	/* floor() */
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifdef HAVE_SYS_FCNTL_H
#include <sys/fcntl.h>
#endif

#ifdef HAVE_SYS_SIGNAL_H
#include <sys/signal.h>
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#ifdef HAVE_SYS_RESOURCE_H
#include <sys/resource.h>
#endif

/* #include <sys/shm.h> */

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#include "ioctl_meteor.h"

#include "mmenu.h"
#include "debug.h"
#include "query.h"

#include "mmvi.h"

/* We need the defn of METEOR_FIELD_MODE */
#include "meteor.h"

/* NTSC */
#define MAXROWS 480
#define MAXCOLS 640
#define DEFAULT_FORMAT METEOR_FMT_NTSC
#define MAXFPS 30

#define MAXPIXELSIZE 4

//extern int errno;

/* device stuff */

struct meteor_geomet my_geo;
static int curr_ofmt=METEOR_GEO_RGB24;
int usePacked;


/* geometry stuff */
int	meteor_columns = MAXCOLS,
	meteor_rows = MAXROWS,
	meteor_pixelsize = MAXPIXELSIZE,	/* set but never used!?!? */
	meteor_bytes_per_pixel = 4,
	meteor_field_mode = 0x0,
	which_fields = 0x0,
	sigmode = 0x0;

int num_meteor_frames = DEFAULT_METEOR_FRAMES;

/* local prototypes */

static void meteor_set_geometry(QSP_ARG_DECL  struct meteor_geomet *);

#define N_OFORMATS	5
static const char *ofmt_names[N_OFORMATS]={
	"rgb24",
	"rgb16",
	"yuv_planar",
	"yuv_packed",
	"yuv_422"
};

void meteor_set_size(QSP_ARG_DECL  int r,int c,int nf)
{
	num_meteor_frames = nf;
	meteor_rows = r & 0x7fe;
	meteor_columns = c & 0x3fc;

	my_geo.rows = meteor_rows;
	my_geo.columns = meteor_columns;
	my_geo.frames = num_meteor_frames;

	meteor_set_geometry(QSP_ARG  &my_geo);
}


void meteor_set_num_frames(QSP_ARG_DECL  int n)
{
	/*
	if( meteor_field_mode == METEOR_FIELD_MODE )
		num_meteor_frames=2*n;
	else
		num_meteor_frames=n;
	*/

	num_meteor_frames=n;

	my_geo.frames = num_meteor_frames;
	meteor_set_geometry(QSP_ARG  &my_geo);
}

/* The limit on the number of frames USED to be determined by
 * the bigphysarea memory size defined in /etc/lilo.conf
 *
 * With the new himemfb driver, the amount of memory is set
 * in /etc/lilo.conf.  We ought to have the difference between
 * the actual physical memory and what is specified in the file.
 *
 * For testing, we assume that we have 64 Mb.
 *
 * 640 * 480 * 4 / 1024 = 640 * 480 / 256 = 10 * 480 / 4 = 1200 1k blocks
 *
 * Now, each frame is 1200 1k blocks (1.2Mb), an extra page seems to get
 * added on, plus extra spacing so individual frames don't cross
 * 4Mb boundaries...
 *
 * So we need 4M for each 3 frames...
 *
 * So if we have 64M, we ought to be able to have 3*16 = 48 frames...
 */

/* BUG this definition assumes 64M reserved - need to get the true number
 * from himemfb...  on dirac we now reseve 128M
 */

/* On dirac we supposedly have reserved 128MB, but the last frame
 * (actually mtr->mem, which is the last page...)
 * mem->frame_size keeps getting reset to 0xffffffff, but
 * we can't figure out where it's coming from.  04-09-2007
 * It seems ok if nframes is 95...
 */

static int32_t max_frames=95;

static COMMAND_FUNC( do_meteor_set_num_frames )
{
	int n;

	n=HOW_MANY("number of frames");
	if( n<1 || n>max_frames ){
		sprintf(error_string,
	"Number of frames must be positive and less than or equal to %ld",
			(long)max_frames);
		WARN(error_string);
		return;
	}
	meteor_set_num_frames(QSP_ARG  n);
}

static COMMAND_FUNC( do_meteor_set_size )
{
	int c,r;

	c=HOW_MANY("number of columns");
	r=HOW_MANY("number of rows");
	meteor_set_size(QSP_ARG  r,c,num_meteor_frames);
}

int get_bytes_per_pixel(QSP_ARG_DECL  int fmt)
{
	int ofmt_index;

	ofmt_index = get_ofmt_index(QSP_ARG  fmt);
	if( ofmt_index < 0 ) return(0);
	return(meteor_bytes_per_pixel);
}

int get_ofmt_index(QSP_ARG_DECL  int fmt)
{
	int ofmt_index=(-1);

/*
sprintf(error_string,"get_ofmt_index:  fmt = %d",fmt);
advise(error_string);
*/
	switch(fmt&METEOR_GEO_OUTPUT_MASK){
		case METEOR_GEO_RGB24:  ofmt_index=0; meteor_bytes_per_pixel=4; break;
		case METEOR_GEO_RGB16:  ofmt_index=1; meteor_bytes_per_pixel=2; break;
		case METEOR_GEO_YUV_PLANAR:  ofmt_index=2; meteor_bytes_per_pixel=2; break;
		case METEOR_GEO_YUV_PACKED:  ofmt_index=3; meteor_bytes_per_pixel=2; break;
		case METEOR_GEO_YUV_422:  ofmt_index=4; meteor_bytes_per_pixel=2; break;
		default:
			sprintf(error_string,"get_ofmt_index:  bad format 0x%x",
				fmt&METEOR_GEO_OUTPUT_MASK);
			WARN(error_string);
			break;
	}
	return(ofmt_index);
}

static void show_meteor_geometry(QSP_ARG_DECL  struct meteor_geomet *gp)
{
	int depth;
	int ofmt_index;

	ofmt_index=get_ofmt_index(QSP_ARG   gp->oformat );
	if( ofmt_index < 0 ) return;

	depth = meteor_bytes_per_pixel*8;

	sprintf(msg_str,"Meteor geometry:\n\t%d rows x %d columns",
		gp->rows, gp->columns);
	prt_msg(msg_str);
	sprintf(msg_str,"\t%d frames, depth = %d",
		gp->frames, depth);
	prt_msg(msg_str);
	sprintf(msg_str,"\toutput format %s", ofmt_names[ofmt_index]);
	prt_msg(msg_str);
}

static void meteor_set_geometry(QSP_ARG_DECL  struct meteor_geomet *gp)
{
	if( verbose ) show_meteor_geometry(QSP_ARG  gp);

#ifndef FAKE_METEOR_HARDWARE
	if (ioctl(meteor_fd, METEORSETGEO, gp) < 0){
		perror("ioctl METEORSETGEO failed");
		ERROR1("meteor_set_geometry:  Unable to allocate meteor memory");
		return;
	}
#endif /* ! FAKE_METEOR_HARDWARE */

	/* after we change the geometry, we need to re-mmap! */

	meteor_mmap(SINGLE_QSP_ARG);
}

int meteor_get_geometry(struct meteor_geomet *gp)
{
#ifndef FAKE_METEOR_HARDWARE

	if (ioctl(meteor_fd, METEORGETGEO, gp) < 0){
		perror("ioctl GetGeometry failed");
		return(-1);
	}

#else /* FAKE_METEOR_HARDWARE */

	gp->rows=480;
	gp->columns=640;
	gp->frames=8;	/* BUG shuld be N_FAKE_FRAMES */
	gp->oformat = METEOR_GEO_RGB24;

#endif /* FAKE_METEOR_HARDWARE */

	return(0);
}

static COMMAND_FUNC( do_meteor_get_geometry )
{
	struct meteor_geomet _geo;

	if( meteor_get_geometry(&_geo) < 0 ) return;

	show_meteor_geometry(QSP_ARG  &_geo);
}

void meteor_set_field_mode(QSP_ARG_DECL  int flag)
{
	if( flag && meteor_field_mode == METEOR_FIELD_MODE ){
		WARN("meteor_set_field_mode(1):  already in field mode!?");
		return;
	} else if ( (!flag) && meteor_field_mode == 0 ){
		WARN("meteor_set_field_mode(0):  already NOT in field mode!?");
		return;
	}

	if( meteor_get_geometry(&my_geo) < 0 ) return;


	/*
	num_meteor_frames *= 2;
	*/

	if( flag ){
		meteor_field_mode = METEOR_FIELD_MODE;
		/* sigmode = METEOR_SIG_FIELD; */
		meteor_rows /= 2;
	} else {
		meteor_field_mode = 0;
		meteor_rows *= 2;
	}

	my_geo.rows = meteor_rows;
	my_geo.frames = num_meteor_frames;
	my_geo.oformat &= ~METEOR_FIELD_MODE;
	my_geo.oformat |= meteor_field_mode;

	meteor_set_geometry(QSP_ARG  &my_geo);
}

void meteor_set_oformat(QSP_ARG_DECL  int fmt)
{
	/* set global variable meteor_bytes_per_pixel */

	/* leave other geo params alone */

	my_geo.oformat = fmt | meteor_field_mode | which_fields;
	meteor_set_geometry(QSP_ARG  &my_geo);
}

static COMMAND_FUNC( do_meteor_set_oformat )
{
	int i,fmt;

	i=WHICH_ONE("output format",N_OFORMATS,ofmt_names);
	if( i<0 ) return;

	switch(i){
		case 0:  fmt=METEOR_GEO_RGB24; break;
		case 1:  fmt=METEOR_GEO_RGB16; break;
		case 2:  fmt=METEOR_GEO_YUV_PLANAR; break;
		case 3:  fmt=METEOR_GEO_YUV_PACKED; break;
		case 4:  fmt=METEOR_GEO_YUV_422; break;
#ifdef CAUTIOUS
		default:  WARN("wacky output format!?"); return;
#endif /* CAUTIOUS */
	}

	curr_ofmt = fmt;

	meteor_set_oformat(QSP_ARG  fmt);
}

static COMMAND_FUNC( do_meteor_set_field_mode )
{
	if( ASKIF("capture fields") )
		meteor_set_field_mode(QSP_ARG  1);
	else
		meteor_set_field_mode(QSP_ARG  0);
}

void set_grab_depth(QSP_ARG_DECL  int depth)
{
	switch (depth) {
		case 24:
		case 32:
			if( verbose ) prt_msg("Grabbing in 24 bit.");
			meteor_pixelsize = 4;
			meteor_bytes_per_pixel = 4;
			my_geo.oformat = METEOR_GEO_RGB24;
			break;
		case 16:
		case 15:
			if( verbose ) prt_msg("Grabbing in 16 bit.");
			meteor_pixelsize = 2;
			meteor_bytes_per_pixel = 2;
			my_geo.oformat = METEOR_GEO_RGB16;
			break;
		default:
			if( verbose ) prt_msg("Grabbing in monochrome");
			meteor_pixelsize = 1;
			meteor_bytes_per_pixel = 2;
			if ((usePacked = checkChips(SINGLE_QSP_ARG))) {
				my_geo.oformat = METEOR_GEO_YUV_PACKED;
				advise("Format set to YUV_PACKED");
			} else {
				my_geo.oformat = METEOR_GEO_YUV_PLANAR;
				advise("Format set to YUV_PLANAR");
			}
			break;
	}
	/* BUG should set geometry here? */
}

static COMMAND_FUNC( do_set_grab_depth )
{
	int depth;

	depth=HOW_MANY("grab bit depth");
	set_grab_depth(QSP_ARG  depth);
}

static COMMAND_FUNC( do_even_only )
{
	if( ASKIF("capture only even fields") ){
		which_fields=METEOR_GEO_EVEN_ONLY;
	} else {
		which_fields=0;
	}
	
	meteor_set_oformat(QSP_ARG  curr_ofmt);
}

static COMMAND_FUNC( do_odd_only )
{
	if( ASKIF("capture only odd fields") ){
		which_fields=METEOR_GEO_ODD_ONLY;
	} else {
		which_fields=0;
	}
	
	meteor_set_oformat(QSP_ARG  curr_ofmt);
}

static Command geo_ctbl[]={
{ "size",	do_meteor_set_size,	"specify # rows & columns"	},
{ "depth",	do_set_grab_depth,	"specify grab bit depth"	},
{ "field_mode",	do_meteor_set_field_mode,"set/clear field mode"		},
{ "odd_only",	do_odd_only,		"capture only odd fields"	},
{ "even_only",	do_even_only,		"capture only even fields"	},
{ "get_geom",	do_meteor_get_geometry,	"display meteor geometry"	},
{ "set_oformat",do_meteor_set_oformat,	"set output format"		},
{ "nframes",	do_meteor_set_num_frames,"set number of buffered frames"},
{ "quit",	popcmd,			"exit submenu"			},
{ NULL_COMMAND								}
};

COMMAND_FUNC( do_geometry )
{
	PUSHCMD(geo_ctbl,"geometry");
}


#endif /* HAVE_METEOR */

