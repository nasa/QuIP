#include "quip_config.h"

char VersionId_v4l2_ezstream[] = QUIP_VERSION_STRING;

#include "query.h"

#ifdef HAVE_V4L2

/* Stream video to disk.
 *
 * Simplified version:
 *   No multithreading, assume the v4l2 ring buffer provides adequate buffering.
 *
 * Unlike the old meteor (where the rgb channels were used to grab 3 synchronized
 * monochrome cameras), the lml bt44 cards have 4 asynchronous inputs.  We want to avoid
 * seeking, so we interleave images from all the cameras in the recording.  But because
 * the cameras are acynchronous, we don't necessarily know the source of the nth frame...
 * We would like to save this information and save it, either as a separate file or a
 * header or tailer...
 *
 */


#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>		/* these two are for getpriority */
#endif

#ifdef HAVE_SYS_RESOURCE_H
#include <sys/resource.h>
#endif

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif

#ifdef HAVE_SCHED_H
#include <sched.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "rv_api.h"
//#include "fg_routines.h"		// meteor stuff?

#include "debug.h"
/* #include "xsupp.h" */
#include "data_obj.h"
#include "fio_api.h"
#include "my_video_dev.h"
#include "my_v4l2.h"
#include "thread_util.h"


/* on wheatstone, we get 6 buffers per camera... */

static int really_writing=1;	/* just for testing */

//static int capture_rows=480;
//static int capture_cols=640;
//static int capture_bytes_per_pixel=2;

/* data passed to the video reader */

#define MAX_CAMERAS	4

typedef struct {
	long		vr_n_frames;
	int		vr_n_disks;
	Image_File *	vr_ifp;
	int		vr_n_cameras;
	Video_Device *	vr_vdp[MAX_CAMERAS];
} vr_args;

struct itimerval tmr1;

/* globals */

#define NOT_RECORDING		1
#define RECORDING		2
#define RECORD_HALTING		4		/* async halt requested */
#define RECORD_FINISHING	8	/* halt request has been processed */

static int which_device;
static int record_state=NOT_RECORDING;
static int recording_in_process = 0;

//#define DEFAULT_BYTES_PER_PIXEL		4

/* raw video from the grabber is 422 YUYV */
#define DEFAULT_BYTES_PER_PIXEL		2

/* disk writer stuff */
static long n_so_far;			/* # frames written */
static long n_to_write;			/* number of bytes per write op */

static int async_capture=0;

/* local prototypes */
static struct v4l2_buffer *next_frame(QSP_ARG_DECL  int n_devices, Video_Device **vdp_tbl);
static void v4l2_finish_recording(QSP_ARG_DECL  Image_File *ifp);


#ifdef RECORD_TIMESTAMPS

typedef struct ts_data {
	int		which_dev;
	struct timeval	grab_time;
} TS_Data;

static TS_Data *ts_array=NULL;

static int stamping=1;
static unsigned int n_stored_times=0;
static unsigned int ts_array_size=0;
#endif /* RECORD_TIMESTAMPS */

static uint32_t get_blocks_per_frame()
{
	uint32_t blocks_per_frame, bytes_per_frame;

	bytes_per_frame = 640*480*2;	/* BUG don't hard code! */

#ifdef RECORD_TIMESTAMPS
	/* The meteor driver puts timestamps at the end of frames, but v4l2 doesn't??? */
	//if( stamping ) bytes_per_frame += TIMESTAMP_SIZE;
#endif /* RECORD_TIMESTAMPS */
	
	blocks_per_frame = ( bytes_per_frame + BLOCK_SIZE - 1 ) / BLOCK_SIZE;
	return(blocks_per_frame);
}

#ifdef RECORD_TIMESTAMPS
static void init_stamps(uint32_t n_frames)
{
	if( ts_array != NULL ){
		if( ts_array_size == n_frames ){
			n_stored_times=0;
			return;
		}
		givbuf( ts_array );
	}

	ts_array = (TS_Data *) getbuf( sizeof(TS_Data) * n_frames );
	ts_array_size = n_frames;
	n_stored_times=0;
}
#endif /* RECORD_TIMESTAMPS */

#ifdef DEBUG_TIMERS

static void show_tmr(struct itimerval *tmrp)
{
	sprintf(error_string,"interval:  %ld   %ld",
			tmrp->it_interval.tv_sec,
			tmrp->it_interval.tv_usec);
	advise(error_string);

	sprintf(error_string,"value:  %ld   %ld",
			tmrp->it_value.tv_sec,
			tmrp->it_value.tv_usec);
	advise(error_string);
}

static void show_tmrs()
{
advise("real time timer:");
getitimer(ITIMER_REAL,&tmr1);
show_tmr(&tmr1);
}

#endif /* DEBUG_TIMERS */

/* These don't seem to be used anywhere in this module??? */

void set_v4l2_async_record(int flag)
{ async_capture = flag; }

int get_v4l2_async_record()
{ return async_capture; }

/*
 */

void v4l2_stream_record(QSP_ARG_DECL  Image_File *ifp,long n_frames,int n_cameras, Video_Device **vd_tbl)
{
	int fd_arr[MAX_DISKS];
	int ndisks, which_disk;
	uint32_t blocks_per_frame;
	Shape_Info shape;
	RV_Inode *inp;
	struct v4l2_buffer *bufp;
	int i;

	if( record_state != NOT_RECORDING ){
		sprintf(error_string,
	"v4l2_stream_record:  can't record file %s until previous record completes",
			ifp->if_name);
		WARN(error_string);
		return;
	}

/* grabber dependent? */
	blocks_per_frame = get_blocks_per_frame();

	n_to_write = blocks_per_frame * BLOCK_SIZE;
//n_to_write>>=2;		/* write quarter for testing */


	if( ifp->if_type != IFT_RV ){
		sprintf(error_string,
	"stream record:  image file %s (type %s) should be type %s",
			ifp->if_name,
			ft_tbl[ifp->if_type].ft_name,
			ft_tbl[IFT_RV].ft_name);
		WARN(error_string);
		return;
	}


	inp = (RV_Inode *) ifp->if_hd;
	ndisks = queue_rv_file(inp,fd_arr);

#ifdef CAUTIOUS
	if( ndisks < 1 ){
		sprintf(error_string,
			"Bad number (%d) of raw volume disks",ndisks);
		WARN(error_string);
		return;
	}
#endif /* CAUTIOUS */


	/* meteor_get_geometry(&_geo); */
	shape.si_flags = 0;
	shape.si_rows = 480;	/* BUG don't hard code */
	shape.si_cols = 640; /* BUG don't hard code */
	/* should get bytes per pixel from _geo... */
	shape.si_comps =  DEFAULT_BYTES_PER_PIXEL;
	shape.si_frames = n_frames;
	shape.si_seqs = 1;
	shape.si_prec = PREC_UBY;
	set_shape_flags(&shape,NO_OBJ);

	rv_set_shape(QSP_ARG  ifp->if_name,&shape);


	/* We write an entire frame to each disk in turn... */

#ifdef RECORD_TIMESTAMPS
	if( stamping ) init_stamps(n_frames);
#endif /* RECORD_TIMESTAMPS */

	record_state = RECORDING;

	/* stuff from video_reader */

	for(i=0;i<n_cameras;i++)
		start_capturing(QSP_ARG  vd_tbl[i]);

	n_so_far = 0;
	which_disk=0;

	bufp=next_frame(QSP_ARG  n_cameras,vd_tbl);
	while( bufp != NULL ){
		int n_written;

		/* write the frame to disk */
if( really_writing ){
		if( (n_written = write(fd_arr[which_disk],vd_tbl[which_device]->vd_buf_tbl[ bufp->index ].mb_start,n_to_write))
			!= n_to_write ){
			sprintf(error_string,"write (frm %ld, fd=%d)",n_so_far,ifp->if_fd);
			perror(error_string);
			sprintf(error_string,
				"%ld requested, %d written",
				n_to_write,n_written);
			WARN(error_string);
			return;
		}

		which_disk = (which_disk+1) % ndisks;
}
		n_so_far++;

		/* QBUG releases this buffer to be used again */
		if( xioctl(vd_tbl[which_device]->vd_fd, VIDIOC_QBUF, bufp) < 0 )
			ERRNO_WARN ("v4l2_stream_record:  error queueing frame");

		if( n_so_far >= n_frames )
			bufp = NULL;
		else
			bufp=next_frame(QSP_ARG  n_cameras,vd_tbl);
	}
	if( bufp != NULL ){
		if( xioctl(vd_tbl[which_device]->vd_fd, VIDIOC_QBUF, bufp) < 0 )
			ERRNO_WARN ("v4l2_stream_record:  error queueing frame");
	}

	for(i=0;i<n_cameras;i++)
		stop_capturing(QSP_ARG  vd_tbl[i]);

	rv_sync(SINGLE_QSP_ARG);

	/* we used to disable real-time scheduling here, but
	 * when video_reader executes as a separate thread there
	 * is no point, because it won't affect the main process!
	 */

	recording_in_process = 0;
	record_state=NOT_RECORDING;

#ifdef RECORD_TIMESTAMPS
	n_stored_times = n_so_far;
#endif

#ifdef CAUTIOUS
	if( ifp == NO_IMAGE_FILE ){
		WARN("CAUTIOUS:  v4l2_stream_record:  ifp is NULL!?");
		return;
	}
#endif /* CAUTIOUS */

	v4l2_finish_recording( QSP_ARG  ifp );

	/* Because the disk writers don't use the fileio library,
	 * the ifp doesn't know how many frames have been written.
	 */

	ifp->if_nfrms = n_frames;	/* BUG? is this really what happened? */
} /* end v4l2_stream_record */

/* Get the next frame from whichever device might be ready.  We want to store
 * the frames we get in device order, so we don't necessarily deliver up the frame
 * we get here next...
 *
 * This seems to fail, so instead we wait for the frame we want, accepting
 * that frames may pile up in the other channels...
 */

static struct v4l2_buffer nf_buf;

static struct v4l2_buffer *next_frame(QSP_ARG_DECL  int n_devices, Video_Device **vdp_tbl)
{
	fd_set fds;
	struct timeval tv;
	int r;
	//int i;
	int fd_limit;
	Video_Device *ready_vdp;
#ifdef WRITE_SEQUENTIAL
	int where_to_write;
#endif /* WRITE_SEQUENTIAL */

#define CHECK_ALL_DEVICES

	/* We CHECK ALL DEVICES when the cameras are not sychronized, and so we
	 * don't know which one will be ready next
	 */

#ifdef CHECK_ALL_DEVICES
	int i;

	FD_ZERO(&fds);
	fd_limit = vdp_tbl[0]->vd_fd;
	for(i=0;i<n_devices;i++){
		FD_SET(vdp_tbl[i]->vd_fd, &fds);
		if( vdp_tbl[i]->vd_fd > fd_limit )
			fd_limit = vdp_tbl[i]->vd_fd;
	}
	/* what is fd+1 all about??? see select(2) man page... */
	fd_limit++;

	/* Timeout. */
	tv.tv_sec = 2;
	tv.tv_usec = 0;

	r = select(fd_limit, &fds, NULL, NULL, &tv);

	if(r == -1) {
		/* original code repeated if EINTR */
		ERRNO_WARN("select");
		return(NULL);
	}

	if( r == 0 ) {
		sprintf(error_string, "select timeout");
		WARN(error_string);
		return(NULL);
	}

	/* Something is readable on one of the fd's - but which one??? */
	which_device = (-1);
	for(i=0;i<n_devices;i++){
		if( FD_ISSET(vdp_tbl[i]->vd_fd,&fds) ){
			if( which_device != -1 ) {
				if( verbose )
					advise("more than one device is ready!");
				/* Because we assign which_device every
				 * time we encounter a ready device, this
				 * has the effect of giving a preference
				 * to higher-numbered devices.  We ought
				 * to do something smarter, like noting
				 * how much time has elapsed since the
				 * device was last serviced, to avoid
				 * having a device's ring buffer fill up.
				 * In practice, this doesn't seem to be a
				 * problem, even though the system only
				 * gives each device 6 buffers (200 msec).
				 * With the * new SATA disks, write times
				 * are only 1-2 msec.
				 * Still, we might consider this a BUG.
				 */
			}
			which_device = i;
		}
	}
#ifdef CAUTIOUS
	if( which_device < 0 ){
		sprintf(error_string,"CAUTIOUS:  next_frame:  no ready device found!?");
		WARN(error_string);
		return(NULL);
	}
#endif /* CAUTIOUS */

#else /* ! CHECK_ALL_DEVICES */

	/* figure out which device we want to read next... */
	which_device = (newest+1) % n_devices;
	FD_ZERO(&fds);
	FD_SET(vdp_tbl[which_device]->vd_fd, &fds);
	fd_limit = vdp_tbl[which_device]->vd_fd;
	fd_limit++;
	/* Timeout. */
	tv.tv_sec = 2;
	tv.tv_usec = 0;

	r = select(fd_limit, &fds, NULL, NULL, &tv);

	if(r == -1) {
		/* original code repeated if EINTR */
		ERRNO_WARN("select");
		return(NULL);
	}

	if( r == 0 ) {
		sprintf(error_string, "select timeout");
		WARN(error_string);
		return(NULL);
	}

#endif /* ! CHECK_ALL_DEVICES */

	ready_vdp = vdp_tbl[which_device];

	//CLEAR (nf_buf);

	nf_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	nf_buf.memory = V4L2_MEMORY_MMAP;

	/* DQBUF pulls a buffer out and locks it up while we use it... */
	if( xioctl(ready_vdp->vd_fd, VIDIOC_DQBUF, &nf_buf) < 0 ) {
		/* original code had special cases for EAGAIN and EIO */
		ERRNO_WARN ("VIDIOC_DQBUF #1");
		return(NULL);
	}

#ifdef CAUTIOUS
	if( nf_buf.index < 0 || nf_buf.index >= (unsigned int) ready_vdp->vd_n_buffers ){
		sprintf(error_string,"CAUTIOUS:  Unexpected buffer number (%d) from VIDIOC_DQBUF, expected 0-%d",
			nf_buf.index,ready_vdp->vd_n_buffers-1);
		WARN(error_string);
		return(NULL);
	}
#endif /* CAUTIOUS */

#ifdef RECORD_TIMESTAMPS
	/* Store the timestamp for this buffer */
#ifdef CAUTIOUS
	if( n_stored_times >= ts_array_size ){
		sprintf(error_string,"CAUTIOUS:  n_stored_times (%d) should be less than ts_array_size (%d) when storing a new timestamp",n_stored_times,ts_array_size);
		ERROR1(error_string);
	}
#endif /* CAUTIOUS */
	ts_array[n_stored_times].grab_time = nf_buf.timestamp;
	ts_array[n_stored_times].which_dev = which_device;
	n_stored_times++;
#endif /* RECORD_TIMESTAMPS */


	/* note that newest & oldest are one thing for the video device,
	 * and another completely different thing for our ring buffer.
	 * Here we update the information for the video dev's private ring buffer...
	 */

	return(&nf_buf);
} /* end next_frame */

void v4l2_finish_recording(QSP_ARG_DECL  Image_File *ifp)
{
	RV_Inode *inp;

//sprintf(error_string,"v4l2_finish_recording %s",ifp->if_name);
//advise(error_string);
	inp = get_rv_inode(QSP_ARG  ifp->if_name);
#ifdef CAUTIOUS
	if( inp == NO_INODE ){
		sprintf(error_string,"CAUTIOUS: v4l2_finish_recording:  missing rv inode %s",ifp->if_name);
		ERROR1(error_string);
	}
#endif

	close_image_file(QSP_ARG  ifp);		/* close write file	*/

	/* FIXME - when we do the full moviemenu interface, we need to make
	 * sure our new recording gets incorporated into the database here...
	 */

	/* update_movie_database(inp); */
	//warn("NOT updating movie database");
	/* note_error_frames(inp); */
}

#ifdef RECORD_CAPTURE_COUNT
void dump_ccount(int index,FILE* fp)
{
	int i;

	for(i=0;i<ppi[index].ppi_nf;i++){
		fprintf(fp,"%d\n",ppi[index].ppi_ccount[i]);
	}
	fclose(fp);
}
#endif	/* RECORD_CAPTURE_COUNT */

#ifdef RECORD_TIMESTAMPS
void dump_timestamps(const char *filename)
{
	WARN("dump_timestamps not implemented in ezstream.c");
}

COMMAND_FUNC( print_grab_times )
{
#ifdef RECORD_TIMESTAMPS
	unsigned int i;
	char *s;

	for(i=0;i<n_stored_times;i++){
		s=ctime(&ts_array[i].grab_time.tv_sec);
		/* remove trailing newline */
		if( s[ strlen(s) - 1 ] == '\n' ) s[ strlen(s) - 1 ] = 0;
		sprintf(msg_str,"%d\t%s\t%ld\t%3ld.%03ld",
				ts_array[i].which_dev,
				s,
				ts_array[i].grab_time.tv_sec,
				ts_array[i].grab_time.tv_usec/1000,
				ts_array[i].grab_time.tv_usec%1000
				);
		prt_msg(msg_str);
	}
#else /* ! RECORD_TIMESTAMPS */
	WARN("print_grab_times:  RECORD_TIMESTAMPS not enabled in this build!?");
#endif /* ! RECORD_TIMESTAMPS */
}

COMMAND_FUNC( print_store_times )
{
	WARN("print_store_times not implemented in ezstream.c");
}

#endif /* RECORD_TIMESTAMPS */

/*
 * Frames from the device are encoded 422 (YUYV)
 */
/*
 * do_record()
 *
 * saves the frames to a disk file, no header, just raw rgbx...
 */

COMMAND_FUNC( do_stream_record )
{
	long n_frames;
	Image_File *ifp;
	const char *name;
	uint32_t n_blocks;
	Video_Device *vd_tbl[MAX_CAMERAS];
	int i;
	int nc;

	/* BUG use fileio library here??? */

	name = NAMEOF("image file name");
	// total frames or frames per camera??
	n_frames=HOW_MANY("number of frames (or fields if field mode)");
	nc=HOW_MANY("number of cameras");
	for(i=0;i<nc;i++)
		vd_tbl[i] = PICK_VIDEO_DEV("");
	/* BUG should make sure that all are distinct */

	for(i=0;i<nc;i++)
		if( vd_tbl[i] == NO_VIDEO_DEVICE ) return;

	ifp = img_file_of(QSP_ARG  name);

	if( ifp != NO_IMAGE_FILE ){
		sprintf(error_string,"Clobbering existing image file %s",name);
		advise(error_string);
		image_file_clobber(1);	/* not necessary !? */
		delete_image_file(QSP_ARG  ifp);
	}

	set_filetype(QSP_ARG  IFT_RV);
	ifp = write_image_file(QSP_ARG  name,n_frames);	/* nf stored in if_frms_to_write */

	/* sets nframes in ifp, but doesn't allocate rv blocks properly...
	 * (WHY NOT??? maybe because the image dimensions are not known???)
	 * we could know them, however, because at this point the geometry is set.
	 */

	if( ifp == NO_IMAGE_FILE ){
		sprintf(error_string,"Error creating movie file %s",name);
		WARN(error_string);
		return;
	}

	n_blocks = FRAMES_TO_ALLOCATE(n_frames,rv_get_ndisks()) * get_blocks_per_frame();

	/* n_blocks is the total number of blocks, not the number per disk(?) */

	if( rv_realloc(QSP_ARG  name,n_blocks) < 0 ){
		sprintf(error_string,"error reallocating %d blocks for rv file %s",
			n_blocks,name);
		WARN(error_string);
		return;
	}

	if( recording_in_process ){
		WARN("do_record:  async recording is already in process, need to wait!?");
		wait_record(SINGLE_QSP_ARG);
	}

	recording_in_process = 1;

	v4l2_stream_record(QSP_ARG  ifp,n_frames,nc,vd_tbl);
}
 /* end do_record() */

#else /* ! HAVEV4L2 */

COMMAND_FUNC( do_stream_record )
{
	ERROR1("do_stream_record:  Program not configured with V4L2 support.");
}

#endif /* ! HAVE_V4L2 */

COMMAND_FUNC( halt_record )
{
	WARN("halting not implemented in ezstream.c");
}

COMMAND_FUNC( wait_record )
{
	WARN("waiting not implemented in ezstream.c");
}


