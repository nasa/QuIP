#ifndef FAKE_METEOR_HARDWARE

#include "quip_config.h"

char VersionId_meteor_fg_routines[] = QUIP_VERSION_STRING;

#ifdef HAVE_METEOR

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
#include "query.h"	/* error_string */
#include "memchunk.h"
#include "ioctl_meteor.h"
#include "debug.h"	/* error_string? */

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
int	meteorfd;
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
 *	ram_location: DISPLAY_RAM, BIGPHYS_RAM, HIMEM_RAM
 */

int fg_open(QSP_ARG_DECL  int video_source, int video_format, int ram_location)
{
	const char *		MeteorDev = DEF_METEOR_DEVICE;
	u_char *		framebuf;
	u_char *		phys_fb_addr;
	u_int			hm_size;
	int			numcols = DEF_NUM_COLS;
	int			numrows = DEF_NUM_ROWS;
	int			framesize;
	int			bytesperpixel;
	uint32_t		iformat = METEOR_FMT_NTSC;
	struct meteor_counts	err_cnts;
	struct meteor_geomet	geo;
	struct meteor_fbuf	vid;

advise("fg_open BEGIN");


	/* open the digitizer */
	/* jbm:  was RDONLY, changed to RDWR for mmap */
	if ((meteorfd = open(MeteorDev, O_RDWR)) < 0) {
		sprintf(error_string,"open(%s)",MeteorDev);
		perror(error_string);
		return( -1 );
	}

	if( verbose ){
		sprintf(error_string,"MeteorDev %s opened, fd=%d",MeteorDev,meteorfd);
		advise(error_string);
	}


	/* reset all error counts */
	fifoerrcnts = 0;
	dmaerrcnts = 0;
	err_cnts.fifo_errors = 0;
	err_cnts.dma_errors = 0;
	err_cnts.frames_captured = 0;
	err_cnts.even_fields_captured = 0;
	err_cnts.odd_fields_captured = 0;
//sprintf(error_string,"fg_open:  sizeof(struct meteor_counts) = 0x%lx",sizeof(struct meteor_counts));
//advise(error_string);
//sprintf(error_string,"fg_open:  METEORSCOUNT = 0x%lx",METEORSCOUNT);
//advise(error_string);
	if (ioctl(meteorfd, (int)(METEORSCOUNT), &err_cnts) < 0) {
		perror("fg_open:  ioctl(METEORSCOUNT)");
		fg_close();
		return( -1 );
	}

	if( numcols > 640 )
		numcols = 640;
	if( (numcols % 8) != 0 )
		numcols = numcols & 0x3f8;

	if (ioctl(meteorfd, METEORSINPUT, &video_source) < 0) {
		perror("ioctl(METEORSINPUT)");
		fg_close();
		return( -1 );
	}

	if (ioctl(meteorfd, METEORSFMT, &iformat) < 0) {
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

advise("progress...");
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
			if (ioctl(meteorfd, METEORSVIDEO, &vid) < 0) {
				perror("ioctl(METEORSVIDEO)");
				fg_close();
				return( -1 );
			}

			/* set geometry only after set video */
			if (ioctl(meteorfd, METEORSETGEO, &geo) < 0) {
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
		case HIMEM_RAM: {
			Mem_Chunk mc1;

advise("HIMEM_FG");
			if( verbose )
				advise("allocating frame buffer using himemfb device");

			if ((fbfd = open(DEF_HIMEM_DEVICE, O_RDWR)) < 0) {
				sprintf(error_string,"open(%s)",DEF_HIMEM_DEVICE);
				perror(error_string);
				fg_close();
				return( -1 );
			}

			/* jbm:
			 * The original code here used HM_GETADDR and HM_GETSIZE
			 * to determine the address and size of the *entire*
			 * himem area...  now we would like to use himemfb's
			 * new allocator to take just a portion of the memory...
			 * The purpose of this is to allow other frame grabber
			 * drivers to share himem peacefully.
			 *
			 * We still need some guidance for how much memory to take...
			 * We can still use GETSIZE to get the total amount of RAM,
			 * and then query for the largest chunk - if these are equal,
			 * then we know that noone else is using any of the memory,
			 * and conversely if they are not equal then we know that
			 * someone else is holding some of the memory.
			 *
			 * A basic problem is that the driver is handed all the memory,
			 * and then frames are placed in response to SETGEO requests...
			 * It would be better to know now what geometry we are going
			 * to want to use, and make an appropriate request to himemfb.
			 *
			 * For 640x480, we can fit 3 frames into 4 MB... (approx 1.2 ea).
			 * So for 16 frames, we need 22 MB...  for 32 we need 44 MB...
			 * It is unlikely that we will ever want more than 48 MB.
			 * We will make the default request for 64 MB (48 frames?).
			 * It would be better to have several seconds worth...
			 *
			 * We reserve 64Mb, enough for 3 * 16 = 48 frames.
			 *
			 * the default number of frames is set in ../include/mmenu.h.
			 */
#define DEFAULT_HIMEM_REQ_SIZE	(128*1024*1024)

			if (ioctl(fbfd, HM_GETADDR, &phys_fb_addr) < 0) {
				perror("ioctl(HM_GETADDR)");
				fg_close();
				return( -1 );
			}
			if (ioctl(fbfd, HM_GETSIZE, &hm_size) < 0) {
				perror("ioctl(HM_GETSIZE)");
				fg_close();
				return( -1 );
			}

			/* we now know the location and total size, check on availability */
			mc1.mc_size = 0;	/* flag size req */
			if( ioctl(fbfd, HM_REQCHUNK, &mc1) < 0 ){
				perror("ioctl(HM_REQCHUNK)");
				fg_close();
				return(-1);
			}
			/* the size of the largest chunk should be returned in mc1.mc_size */
			if( verbose ){
sprintf(error_string,"largest himemfb chunk is %d bytes",mc1.mc_size);
advise(error_string);
			}

			if( mc1.mc_size <= 0 ){
				WARN("no memory for frame buffer!?");
				fg_close();
				return(-1);
			}

			if( mc1.mc_size > DEFAULT_HIMEM_REQ_SIZE ){
				mc1.mc_size = DEFAULT_HIMEM_REQ_SIZE ;
			}

			if( verbose ){
sprintf(error_string,"requesting %d bytes from himemfb",mc1.mc_size);
advise(error_string);
			}
			mc1.mc_flags = AUTO_RELEASE;	/* release the chunk when himemfb is closed */

			if( ioctl(fbfd, HM_REQCHUNK, &mc1) < 0 ){
				perror("ioctl(HM_REQCHUNK)");
				fg_close();
				return(-1);
			}

			/* vid.addr = (uint32_t)phys_fb_addr; */
			/* vid.ramsize = hm_size; */

			vid.addr = (int_for_addr)mc1.mc_addr;
			vid.ramsize = mc1.mc_size;

			vid.width = 0;
			if (ioctl(meteorfd, METEORSVIDEO, &vid) < 0) {
				perror("ioctl(METEORSVIDEO)");
				fg_close();
				return( -1 );
			}

			/* set geometry only after set video */
			if (ioctl(meteorfd, METEORSETGEO, &geo) < 0) {
				perror("ioctl(METEORSETGEO)");
				fg_close();
				return( -1 );
			}

			framesize = geo.columns * geo.rows * bytesperpixel;

			/* We can use either device as mmap source, but the himemfb
			 * device always maps from the beginning, it doesn't know that
			 * the meteor driver might have shifted the frame buffer up
			 * because of the 4MB boundary problem...
			 */

/* can use either device as mmap source */
/*			framebuf=(u_char *)mmap((caddr_t)0, framesize, PROT_READ,
				MAP_FILE|MAP_PRIVATE, meteorfd, (off_t)0);
*/
advise("calling mmap");
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

			fgbuf.ram_location = ram_location;
			fgbuf.ram_width = bytesperpixel * geo.columns;
			fgbuf.ram_addr = framebuf;
			fgbuf.ram_size = framesize;

			break;
		}
#endif

		case BIGPHYS_RAM: {

			if (ioctl(meteorfd, METEORSETGEO, &geo) < 0) {
				perror("ioctl(METEORSETGEO)");
				fg_close();
				return( -1 );
			}

			framesize = bytesperpixel * geo.columns * geo.rows;
			framebuf=(u_char *)mmap((caddr_t)0, framesize, PROT_READ,
				MAP_OPTIONS, meteorfd, (off_t)0);
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

#ifdef CAUTIOUS
	if( fbfd == (-1) )
		error1("CAUTIOUS:  assoc_pids:  himemfb is not open");
#endif /* CAUTIOUS */

	if( ioctl(fbfd,HM_ADDPID,&mc1) < 0 ){
		perror("ioctl HM_ADDPID");
		error1("error associating process id");
	}
/*
sprintf(error_string,"associating pid's %d and %d",pid1,pid2);
advise(error_string);
*/

	return(0);
}

int unassoc_pids(pid1,pid2)
{
	Mem_Chunk mc1;

	mc1.mc_pid[0]=pid1;
	mc1.mc_pid[1]=pid2;

#ifdef CAUTIOUS
	if( fbfd == (-1) )
		error1("CAUTIOUS:  unassoc_pids:  himemfb is not open");
#endif /* CAUTIOUS */

	if( ioctl(fbfd,HM_SUBPID,&mc1) < 0 ){
		perror("ioctl HM_SUBPID");
		error1("error unassociating process id");
	}
/*
sprintf(error_string,"unassociating pid's %d and %d",pid1,pid2);
advise(error_string);
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

	if (ioctl(meteorfd, METEORCAPTUR, &capcmd)) {
		perror("ioctl(start_capture)");
		return(-1);
	}

	/* repeat until no errors */
	while( done == FALSE) {
		if (ioctl(meteorfd, METEORGCOUNT, &cnts) < 0) {
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
			if (ioctl(meteorfd, METEORCAPTUR, &capcmd)) {
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

	if (ioctl(meteorfd, METEORCAPTUR, &capcmd)) {
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

	if (ioctl(meteorfd, METEORCAPTUR, &capcmd)) {
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
		advise("fg_close:  nothing has been mmap'd");
	}

	close(meteorfd);
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

	if (ioctl(meteorfd, METEORGETGEO, &geo) < 0) {
		perror("ioctl(METEORGETGEO)");
		return( -1 );
	}

	if( rows % 2 )
		rows &= 0x7fe;
	geo.rows = rows;

	if (ioctl(meteorfd, METEORSETGEO, &geo) < 0) {
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

	if (ioctl(meteorfd, METEORGETGEO, &geo) < 0) {
		perror("ioctl(METEORGETGEO)");
		return( -1 );
	}

	if( cols % 8 )
		cols = cols & 0x3f8;
	geo.columns = cols;

	if (ioctl(meteorfd, METEORSETGEO, &geo) < 0) {
		perror("ioctl(METEORSETGEO)");
		return( -1 );
	}

	return( cols );
}


/* returns 0 for success -1 for failure
 */
int fg_hue( int hue )
{
	if (ioctl(meteorfd, METEORSHUE, &hue) < 0) {
		perror("ioctl(METEORSHUE)");
		return( -1 );
	}
	return( 0 );
}


/* returns 0 for success -1 for failure
 */
int fg_brightness( int brightness )
{
	if (ioctl(meteorfd, METEORSBRIG, &brightness) < 0) {
		perror("ioctl(METEORSBRIG)");
		return( -1 );
	}
	return( 0 );
}


/* returns 0 for success -1 for failure
 */
int fg_contrast( int contrast )
{
	if (ioctl(meteorfd, METEORSCONT, &contrast) < 0) {
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

	if (ioctl(meteorfd, METEORSGAMMA, &arg) < 0) {
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

int meteorfd=(-1);
int fbfd=(-1);

int fg_open(int video_source, int video_format, int ram_location)
{
	return(0);
}

#endif /* FAKE_METEOR_HARDWARE */

#endif /* HAVE_METEOR */

