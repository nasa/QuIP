#ifndef FAKE_METEOR_HARDWARE

#include "quip_config.h"

/* fg_routines
 *
 *  8/15/97 mjw extracted from snap.c
 *  9/02/97 mjw major re-write
 *  9/03/97 mjw add fg_put_roi()
 *  9/18/97 mjw add many routines
 * 10/14/97 mjw added gamma routine
 * 10/31/97 mjw added hooks for screen offset
 * 03/22/99 mjw v2.2
 * 03/29/99 mjw fix HIMEM access in fg_open
 * 04/23/99 mjw change fg_rows and fg_cols to check for valid argument
 */

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif


#include "fg_routines.h"
#include "quip_prot.h"	/* ERROR_STRING */

#ifdef HAVE_METEOR
#include "ioctl_meteor.h"

//#ifdef LINUX
//#define MAP_OPTIONS	MAP_FILE|MAP_PRIVATE
//#else
//#define MAP_OPTIONS	MAP_PRIVATE
//#endif
#define MAP_OPTIONS	MAP_FILE|MAP_PRIVATE

/* globals */
/* (these should be in private area) */
unsigned int	fifoerrcnts;
unsigned int	dmaerrcnts;
int	meteor_fd=(-1);
int	fbfd=(-1);
struct fb_info {
	int ram_location;
	int ram_width;
	u_char *ram_addr;
	size_t	ram_size;
} fgbuf;

static int meteor_is_mapped=0, fbdev_is_mapped=0;


/*
 * sets up meteor frame grabber in rgb888 mode
 * initializes global variables
 * returns 0 on success; neg number on failure
 * example arguments:
 *	video_source: SOURCE_NTSC, SOURCE_SVIDEO
 *	video_format: MERGE_FIELDS, SEPARATE_FIELDS
 *	ram_location: DISPLAY_RAM, BIGPHYS_RAM, KERNEL_RAM
 */

int fg_open(QSP_ARG_DECL  int video_source, int video_format, int ram_location)
{
	const char *		MeteorDev = DEF_METEOR_DEVICE;
	u_char *		framebuf;
	int			numcols = DEF_NUM_COLS;
	int			numrows = DEF_NUM_ROWS;
	int			framesize;
	int			bytesperpixel;
	uint32_t		iformat = METEOR_FMT_NTSC;
	struct meteor_counts	err_cnts;
	struct meteor_geomet	geo;

advise("fg_open BEGIN");


	/* open the digitizer */
	/* jbm:  was RDONLY, changed to RDWR for mmap */
	if ((meteor_fd = open(MeteorDev, O_RDWR)) < 0) {
		sprintf(ERROR_STRING,"open(%s)",MeteorDev);
		perror(ERROR_STRING);
		return( -1 );
	}

	if( verbose ){
		sprintf(ERROR_STRING,"MeteorDev %s opened, fd=%d",MeteorDev,meteor_fd);
		advise(ERROR_STRING);
	}


	/* reset all error counts */
	fifoerrcnts = 0;
	dmaerrcnts = 0;
	err_cnts.fifo_errors = 0;
	err_cnts.dma_errors = 0;
	err_cnts.n_frames_captured = 0;
	err_cnts.even_fields_captured = 0;
	err_cnts.odd_fields_captured = 0;
sprintf(ERROR_STRING,"fg_open:  ioctl METEORSCOUNT (0x%x)",METEORSCOUNT);
advise(ERROR_STRING);
	if (ioctl(meteor_fd, (int)(METEORSCOUNT), &err_cnts) < 0) {
		perror("fg_open:  ioctl(METEORSCOUNT)");
		fg_close();
		return( -1 );
	}

	if( numcols > 640 )
		numcols = 640;
	if( (numcols % 8) != 0 )
		numcols = numcols & 0x3f8;

sprintf(ERROR_STRING,"fg_open:  ioctl METEORSINPUT");
advise(ERROR_STRING);
	if (ioctl(meteor_fd, METEORSINPUT, &video_source) < 0) {
		perror("ioctl(METEORSINPUT)");
		fg_close();
		return( -1 );
	}

sprintf(ERROR_STRING,"fg_open:  ioctl METEORSFMT");
advise(ERROR_STRING);
	if (ioctl(meteor_fd, METEORSFMT, &iformat) < 0) {
		perror("ioctl(METEORSFMT)");
		fg_close();
		return( -1 );
	}

	geo.columns = numcols;
	geo.rows = numrows;

	/* why is geo.frames = 1 ? maybe we should have a larger default,
	 * or get a default from an environment variable...   (jbm)
	 */

	geo.frames = 1;
	geo.oformat = video_format;

	switch( video_format ) {
		case METEOR_GEO_YUV_PLANAR:
		case METEOR_GEO_YUV_2FLD:
		case METEOR_GEO_YUV_PACKED:
		case METEOR_GEO_RGB16:
			bytesperpixel = 2;
			break;
		default:
			bytesperpixel = 4;
			break;
	}

fprintf(stderr,"progress, switching on ram_location %d...",ram_location);
	switch( ram_location ) {

#ifdef DISPLAY_FG
		case DISPLAY_RAM: {
			/* open the display ram */
			if ((fbfd = open(DEF_DISPLAY_DEVICE, O_RDWR)) < 0) {
				perror("open(DisplayDev)");
				fg_close();
				return( -1 );
			}

			/* tell the meteor driver where linear ram is located */
			if (ioctl(fbfd, FB_GETADDR, &phys_fb_addr) < 0) {
				perror("ioctl(FB_GETADDR)");
				fg_close();
				return( -1 );
			}
			vid.addr = (uint32_t)phys_fb_addr + (uint32_t)SCREEN_OFFSET;
			vid.width = DEF_SCREEN_WIDTH * DEF_SCREEN_BYTE_DEPTH;
			vid.ramsize = vid.width * (geo.rows + 16);
			if (ioctl(meteor_fd, METEORSVIDEO, &vid) < 0) {
				perror("ioctl(METEORSVIDEO)");
				fg_close();
				return( -1 );
			}

			/* set geometry only after set video */
			if (ioctl(meteor_fd, METEORSETGEO, &geo) < 0) {
				perror("ioctl(METEORSETGEO)");
				fg_close();
				return( -1 );
			}

			framesize = vid.width * geo.rows;
			framebuf = (u_char *) mmap( (caddr_t)0, (size_t)framesize,
				PROT_READ | PROT_WRITE,
				MAP_OPTIONS,
				fbfd, (off_t)SCREEN_OFFSET );
			if (framebuf == (u_char *)-1) {
				perror("fb mmap failed");
				fg_close();
				return( -1 );
			}
printf("framebuf %x, vid.addr %x\n",framebuf, vid.addr);
			fbdev_is_mapped=1;

			fgbuf.ram_location = ram_location;
			fgbuf.ram_width = vid.width;
			fgbuf.ram_addr = framebuf;
			fbbuf.ram_size = framesize;

			break;
		}
#endif

#ifdef HIMEM_FG
			/*
			framebuf = (u_char *) mmap( (caddr_t)0, (size_t)framesize,
				PROT_READ|PROT_WRITE,
				MAP_OPTIONS,
				fbfd, (off_t)0 );
			if (framebuf == (u_char *)-1) {
				perror("fb mmap failed");
				fg_close();
				return( -1 );
			}
			fbdev_is_mapped=1;
			*/

#endif

		case KERNEL_RAM: {
fprintf(stderr,"calling ioctl METEORSETGEO for kernel ram...\n");
			if (ioctl(meteor_fd, METEORSETGEO, &geo) < 0) {
				perror("ioctl(METEORSETGEO)");
				fg_close();
				return( -1 );
			}
			// Need to map frames individually!
fprintf(stderr,"fg_open not yet mapping kernel ram!?\n");

			break;
		}

		case BIGPHYS_RAM: {

			if (ioctl(meteor_fd, METEORSETGEO, &geo) < 0) {
				perror("ioctl(METEORSETGEO)");
				fg_close();
				return( -1 );
			}

			framesize = bytesperpixel * geo.columns * geo.rows;
			framebuf=(u_char *)mmap((caddr_t)0, framesize, PROT_READ,
				MAP_OPTIONS, meteor_fd, (off_t)0);
			if (framebuf == (u_char *)-1) {
				perror("meteor mmap failed");
				fg_close();
				return( -1 );
			}
			meteor_is_mapped=1;

			fgbuf.ram_location = ram_location;
			fgbuf.ram_width = bytesperpixel * geo.columns;
			fgbuf.ram_addr = framebuf;
			fgbuf.ram_size = framesize;

			break;
		}

		default: {
			fprintf(stderr, "illegal argument fg_open\n");
			fgbuf.ram_location = 0;
			fgbuf.ram_width = 0;
			fgbuf.ram_addr = NULL;
			fgbuf.ram_size = 0;
			fg_close();
			return( -1 );
			break;
		}
	}

	return( 0 );
}

#ifdef FOOBAR

/* jbm added this routine to make autorelease of chunks work properly
 * with multiple threads that exit in random order.  The drivers release
 * routine is only called after the last thread exits, and the pid may
 * not be the same as the one which allocated the chunk.
 */

int assoc_pids(pid1,pid2)
{
	Mem_Chunk mc1;

	mc1.mc_pid[0]=pid1;
	mc1.mc_pid[1]=pid2;

	assert( fbfd != (-1) );

	if( ioctl(fbfd,HM_ADDPID,&mc1) < 0 ){
		perror("ioctl HM_ADDPID");
		error1("error associating process id");
	}
/*
sprintf(ERROR_STRING,"associating pid's %d and %d",pid1,pid2);
advise(ERROR_STRING);
*/

	return(0);
}

int unassoc_pids(pid1,pid2)
{
	Mem_Chunk mc1;

	mc1.mc_pid[0]=pid1;
	mc1.mc_pid[1]=pid2;

	assert( fbfd != (-1) );

	if( ioctl(fbfd,HM_SUBPID,&mc1) < 0 ){
		perror("ioctl HM_SUBPID");
		error1("error unassociating process id");
	}
/*
sprintf(ERROR_STRING,"unassociating pid's %d and %d",pid1,pid2);
advise(ERROR_STRING);
*/

	return(0);
}

#endif /* FOOBAR */

/*
 * captures one frame
 * returns 0 on success -1 on failure
 */
int fg_capture()
{
	int capcmd = METEOR_CAP_SINGLE;
	int done = FALSE;
	int consecutive_errors = 0;
	struct meteor_counts  cnts;

	if (ioctl(meteor_fd, METEORCAPTUR, &capcmd)) {
		perror("ioctl(start_capture)");
		return(-1);
	}

	/* repeat until no errors */
	while( done == FALSE) {
		if (ioctl(meteor_fd, METEORGCOUNT, &cnts) < 0) {
			perror("ioctl(METEORGCOUNT)");
			return(-1);
		}
		if( (cnts.fifo_errors > fifoerrcnts) ||
			(cnts.dma_errors > dmaerrcnts) ) {

			fifoerrcnts = cnts.fifo_errors;
			dmaerrcnts = cnts.dma_errors;
			consecutive_errors += 1;
			if (consecutive_errors > MAX_ALLOW_CONSEC_ERRORS) {
				fprintf(stderr, "too many FIFO errors --\n");
				return(-1);
			}
			if (ioctl(meteor_fd, METEORCAPTUR, &capcmd)) {
				perror("ioctl(start_capture)");
				return(-1);
			}
		} else {
			done = TRUE;
		}
	}
	return( 0 );
}

/*
 * turns on frame_grabber for continuous capture
 * use fg_open() once before calling this routine
 * companion routine is fg_stop()
 * returns 0 on success -1 on failure
 */

int fg_run_continuous()
{
	int capcmd = METEOR_CAP_CONTINOUS;

	if (ioctl(meteor_fd, METEORCAPTUR, &capcmd)) {
		perror("ioctl(start_capture)");
		return(-1);
	}

	return( 0 );
}


/*
 * get a region of interest from video ram to buf
 * xoffset is from left
 * yoffset is from top
 * pixel format in destination buf is rgbrgbrgb ...
 *
 */

void fg_get_roi(int xoffset, int yoffset, int height, int width, u_char* buf)
{
	int i, j;
	int	row_width = fgbuf.ram_width;
	u_char *vram;

	vram = fgbuf.ram_addr;
	vram += yoffset * row_width;
	vram += xoffset * 4;

	for( i=0; i<height; i++ ) {
		for( j=0; j<width; j++ ) {
			*buf++ = vram[4*j + 2];
			*buf++ = vram[4*j + 1];
			*buf++ = vram[4*j];
		}
		vram += row_width;
	}
}


/*
 * put a region of interest from a buf to video ram
 * xoffset is from left
 * yoffset is from top
 * pixel format in source buf is rgbrgbrgb ...
 *
 */

void fg_put_roi(int xoffset, int yoffset, int height, int width, u_char* buf)
{
	int i, j;
	int	row_width = fgbuf.ram_width;
	u_char *vram;

	vram = fgbuf.ram_addr;
	vram += yoffset * row_width;
	vram += xoffset * 4;

	for( i=0; i<height; i++ ) {
		for( j=0; j<width; j++ ) {
			vram[4*j + 2] = *buf++;
			vram[4*j + 1] = *buf++;
			vram[4*j] = *buf++;
		}
		vram += row_width;
	}
}

/*
 * put a grey region of interest from a buf to video ram
 * xoffset is from left
 * yoffset is from top
 * 8 bits per pixel in buffer
 *
 */

void
fg_put_grey_roi(int xoffset, int yoffset, int height, int width, u_char* buf)
{
	int i, j;
	int	row_width = fgbuf.ram_width;
	u_char *vram;

	vram = fgbuf.ram_addr;
	vram += yoffset * row_width;
	vram += xoffset * 4;

	for( i=0; i<height; i++ ) {
		for( j=0; j<width; j++ ) {
			vram[4*j + 2] = *buf;
			vram[4*j + 1] = *buf;
			vram[4*j] = *buf++;
		}
		vram += row_width;
	}
}


/*
 * place a rectangle on the image
 * xoffset is from left
 * yoffset is from top
 * value is of format xrgb (so red would be 0x00ff0000)
 */

void fg_put_box(int xoffset, int yoffset, int height, int width, uint32_t value)
{
	int i;
	int row_width = fgbuf.ram_width >> 2;
	uint32_t *image;
	uint32_t *starth1, *starth2, *startv1, *startv2;

	image = (uint32_t *)fgbuf.ram_addr;
	image += yoffset * row_width;
	image += xoffset;

	starth1 = starth2 = image;
	starth2 += (height - 1) * row_width;
	startv1 = startv2 = image;
	startv2 += width - 1;

	for( i=0; i<width; i++ ) {
		*starth1++ = value;
	}
	for( i=0; i<height; i++ ) {
		*startv1 = value;
		startv1 += row_width;
	}
	for( i=0; i<width; i++ ) {
		*starth2++ = value;
	}
	for( i=0; i<height; i++ ) {
		*startv2 = value;
		startv2 += row_width;
	}
}



void fg_stop()
{
	int capcmd = METEOR_CAP_STOP_CONT;

	if (ioctl(meteor_fd, METEORCAPTUR, &capcmd)) {
		perror("ioctl(stop_capture)");
	}
}

void fg_close()
{
	/* BUG?  under 2.4, the system doesn't seem to call
	 * the release method when the object is still mapped...
	 */


	if( meteor_is_mapped || fbdev_is_mapped ){
#ifdef SOLARIS
		if( munmap( (void *) fgbuf.ram_addr, (size_t) fgbuf.ram_size ) < 0 )
#else
		if( munmap( fgbuf.ram_addr, fgbuf.ram_size ) < 0 )
#endif
			perror("munmap meteor/fbdev");
	} else {
		NADVISE("fg_close:  nothing has been mmap'd");
	}

	close(meteor_fd);
	close(fbfd);
	fbfd = -1;
	fgbuf.ram_location = 0;
	fgbuf.ram_width = 0;
	fgbuf.ram_addr = NULL;
}

/* sets number of rows
 * returns actual number of rows or -1 for failure
 */
int fg_rows( int rows )
{
	struct meteor_geomet  geo;

	if (ioctl(meteor_fd, METEORGETGEO, &geo) < 0) {
		perror("ioctl(METEORGETGEO)");
		return( -1 );
	}

	if( rows % 2 )
		rows &= 0x7fe;
	geo.rows = rows;

	if (ioctl(meteor_fd, METEORSETGEO, &geo) < 0) {
		perror("ioctl(METEORSETGEO)");
		return( -1 );
	}

	return( rows );
}


/* sets number of columns
 * makes sure cols is divisible by 8
 * returns actual number of columns for success -1 for failure
 */
int fg_cols( int cols )
{
	struct meteor_geomet  geo;

	if (ioctl(meteor_fd, METEORGETGEO, &geo) < 0) {
		perror("ioctl(METEORGETGEO)");
		return( -1 );
	}

	if( cols % 8 )
		cols = cols & 0x3f8;
	geo.columns = cols;

	if (ioctl(meteor_fd, METEORSETGEO, &geo) < 0) {
		perror("ioctl(METEORSETGEO)");
		return( -1 );
	}

	return( cols );
}


/* returns 0 for success -1 for failure
 */
int fg_hue( int hue )
{
	if (ioctl(meteor_fd, METEORSHUE, &hue) < 0) {
		perror("ioctl(METEORSHUE)");
		return( -1 );
	}
	return( 0 );
}


/* returns 0 for success -1 for failure
 */
int fg_brightness( int brightness )
{
	if (ioctl(meteor_fd, METEORSBRIG, &brightness) < 0) {
		perror("ioctl(METEORSBRIG)");
		return( -1 );
	}
	return( 0 );
}


/* returns 0 for success -1 for failure
 */
int fg_contrast( int contrast )
{
	if (ioctl(meteor_fd, METEORSCONT, &contrast) < 0) {
		perror("ioctl(METEORSCONT)");
		return( -1 );
	}
	return( 0 );
}

/* returns 0 for success -1 for failure
 */
int fg_gamma( int gamma )
{
	int  arg;

	if( gamma == TRUE )
		arg = 0;
	else
		arg = 1;

	if (ioctl(meteor_fd, METEORSGAMMA, &arg) < 0) {
		perror("ioctl(METEORSGAMMA)");
		return( -1 );
	}
	return( 0 );
}

/* use this only if doing low level operations */
u_char *fg_buf_addr()
{
	return( fgbuf.ram_addr );
}

#else /* FAKE_METEOR_HARDWARE */

int meteor_fd=(-1);
int fbfd=(-1);

int fg_open(QSP_ARG_DECL  int video_source, int video_format, int ram_location)
{
	return(0);
}

#endif /* FAKE_METEOR_HARDWARE */

#endif /* HAVE_METEOR */

