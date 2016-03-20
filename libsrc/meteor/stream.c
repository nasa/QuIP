
#include "quip_config.h"
#include "mmenu.h"

#ifdef HAVE_METEOR

/* Stream video to disk.
 *
 * This code was written for the seagate cheetah's on lagrange.
 *
 * Here is the data:
 *	individual disk		19-20 Mb/sec (measured w/ sio)
 *	/dev/md0		26 Mb/sec (w/ 2 or 4 striped disks)
 *	4 disks			60 Mb/sec (4 sio procs running in parallel
 *
 * So the raid0 implementation is not too good yet...
 *
 * Now here is some interesting data:
 *	with this program writing a quarter of a frame to a file,
 *	frames sometimes get dropped...   Here are 3 patterns that
 *	have been observed:
 *
 *	1)  periods of keeping up, punctuated (at 6 sec intervals?) by
 *	loooooong delays, followed by some catch up.  This behavior
 *	is seen on a new file system.
 *
 *	2)  on an older file system, frames are stored at a uniform
 *	(but inadequate) rate
 *
 *	3)  When the amount of memory reserved for bigphysarea [is increased?]
 *	(see /etc/lilo.conf),  then we see steady writing, with
 *	no gaps!?  Could this be because the number of kernel i/o
 *	buffers as smaller, so they get flushed to disk right away?
 *
 * I never could get this to work on a unix file system, hence the "raw volume"
 * file system implemented in librawvol, and fileio/rv.c.
 * We read and write the raw disks directly, maintaining our own database
 * of what is where...
 */


#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

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

#include "quip_prot.h"
#include "rv_api.h"
#include "mmvi.h"
#include "ioctl_meteor.h"
#include "fg_routines.h"

#include "mmenu.h"
#include "debug.h"
/* #include "xsupp.h" */
#include "data_obj.h"
#include "fio_api.h"
#include "rt_sched.h"


/* data passed to the video reader */

typedef struct {
	int32_t		vr_n_frames;
	int		vr_n_disks;
	Image_File *	vr_ifp;
	Query_Stack *	vr_qsp;		// needed for thread-safe-query
} vr_args;

#ifdef FOOBAR
/* This was used to make sure that himemfb resources were freed, but
 * it didn't work, because pthreads creates another process that never
 * executes our code, but which perversely was the last to exit...
 */
static pid_t grabber_pid=0;
#endif /* FOOBAR */

struct itimerval tmr1;

static unsigned int old_alarm;

static vr_args vra1;	/* can't be on the stack */

static pid_t master_pid;

/* globals */

#define NOT_RECORDING		1
#define RECORDING		2
#define RECORD_HALTING		4		/* async halt requested */
#define RECORD_FINISHING	8	/* halt request has been processed */

static int record_state=NOT_RECORDING;

#define DEFAULT_BYTES_PER_PIXEL		4

/* disk writer stuff */
static int32_t n_so_far;			/* # frames written */
static int32_t n_stream_frames;		/* a global in case we want to print the size of the request */
#ifdef FOOBAR
static int n_discard_lines=0;		/* just for testing so far */
#endif /* FOOBAR */
static int32_t starting_count;
static int32_t n_to_write;			/* number of bytes per write op */
#define N_FRAGMENTS 1			/* blocks_per_frame is 1200 */
static pthread_t dw_thr[MAX_DISKS];
static pthread_t grab_thr;

static int async_capture=0;

/* local prototypes */
static void clear_buffers(SINGLE_QSP_ARG_DECL);
static void *disk_writer(void *);
static void *video_reader(void *);
static void *video_reader_thread(void *);
static void start_grab_thread(QSP_ARG_DECL  vr_args *);

/* status codes */

#define MS_INIT		0		/* MASTER CODE  I	init */
#define MS_WAITING	1		/* MASTER CODE	w	waiting */
#define MS_CHECKING	2		/* MASTER CODE	c	checking */
#define MS_RELEASING	3		/* MASTER CODE	r	releasing */
#define MS_BOT		4		/* MASTER CODE	b	bottom */
#define MS_DONE		5		/* MASTER CODE	d	done */
#define MS_ALARM	6		/* MASTER CODE	A	alarm */

#define DW_INIT		0		/* DISK CODE	i	init */
#define DW_TOP		1		/* DISK CODE	t	top of loop */
#define DW_WAIT		2		/* DISK CODE	w	waiting */
#define DW_WRITE	3		/* DISK CODE	W	writing */
#define DW_DONE		4		/* DISK CODE	D	done writing */
#define DW_BOT		5		/* DISK CODE	b	bottom */
#define DW_EXIT		6		/* DISK CODE	x	exit */

#define TRACE_FLOW

#ifdef TRACE_FLOW

static int m_status=0;
static char mstatstr[]="IwcrbdA";
static char statstr[]="itwWDbx";

static char estring[MAX_DISKS][128];

static const char *master_codes=
	"Master Codes:\n"
	"I	initializing\n"
	"w	waiting\n"
	"c	checking\n"
	"r	releasing\n"
	"b	bottom of loop\n"
	"d	done\n"
	"A	alarm\n";

static const char *dw_codes=
	"DiskWriter codes:\n"
	"i	initializing\n"
	"t	top of loop\n"
	"w	waiting\n"
	"W	writing\n"
	"D	done writing\n"
	"b	bottom of loop\n"
	"x	exiting\n";

#define STATUS_HELP(str)					\
	if( verbose ){						\
		NADVISE(str);					\
	}

static const char *column_doc=
	"thread status oldest newest status next_to_wt ...\n";

#define MSTATUS(code)						\
m_status = code;						\
if( verbose ){							\
sprintf(ERROR_STRING,						\
"M %c\t%d\t%d\t%c %d\t%c %d\t%c %d\t%c %d",			\
mstatstr[m_status],oldest,newest,				\
statstr[ppi[0].ppi_status],					\
ppi[0].ppi_next_to_write,					\
statstr[ppi[1].ppi_status],					\
ppi[1].ppi_next_to_write,					\
statstr[ppi[2].ppi_status],					\
ppi[2].ppi_next_to_write,					\
statstr[ppi[3].ppi_status],					\
ppi[3].ppi_next_to_write);					\
advise(ERROR_STRING);						\
}

#define STATUS(code)						\
pip->ppi_status = code;						\
if( verbose ){							\
sprintf(estring[pip->ppi_index],				\
"%c %c\t%d\t%d\t%c %d\t%c %d\t%c %d\t%c %d",			\
'0'+pip->ppi_index,mstatstr[m_status],oldest,newest,		\
statstr[ppi[0].ppi_status],					\
ppi[0].ppi_next_to_write,					\
statstr[ppi[1].ppi_status],					\
ppi[1].ppi_next_to_write,					\
statstr[ppi[2].ppi_status],					\
ppi[2].ppi_next_to_write,					\
statstr[ppi[3].ppi_status],					\
ppi[3].ppi_next_to_write);					\
NADVISE(estring[pip->ppi_index]);				\
}

/* RSTATUS doesn't seem to be used anywere??? */

#define RSTATUS(code)						\
pip->ppi_status = code;						\
if( verbose ){							\
sprintf(estring[pip->ppi_index],				\
"%c %c\t%d\t%c %d\t%c %d\t%c %d\t%c %d",			\
'0'+pip->ppi_index,rmstatstr[m_status],				\
read_frame_want,						\
rstatstr[ppi[0].ppi_status],					\
ppi[0].ppi_next_to_read,					\
rstatstr[ppi[1].ppi_status],					\
ppi[1].ppi_next_to_read,					\
rstatstr[ppi[2].ppi_status],					\
ppi[2].ppi_next_to_read,					\
rstatstr[ppi[3].ppi_status],					\
ppi[3].ppi_next_to_read);					\
NADVISE(estring[pip->ppi_index]);				\
}

#define RMSTATUS(code)						\
m_status = code;						\
if( verbose ){							\
sprintf(ERROR_STRING,						\
"M %c\t%d\t%c %d\t%c %d\t%c %d\t%c %d",				\
rmstatstr[m_status],						\
read_frame_want,						\
rstatstr[ppi[0].ppi_status],					\
ppi[0].ppi_next_to_write,					\
rstatstr[ppi[1].ppi_status],					\
ppi[1].ppi_next_to_write,					\
rstatstr[ppi[2].ppi_status],					\
ppi[2].ppi_next_to_write,					\
rstatstr[ppi[3].ppi_status],					\
ppi[3].ppi_next_to_write);					\
NADVISE(ERROR_STRING);						\
}
#else /* ! TRACE_FLOW */

#define MSTATUS(code)		/* nop */
#define STATUS(code)		/* nop */
#define RSTATUS(code)		/* nop */
#define RMSTATUS(code)		/* nop */

#endif /* ! TRACE_FLOW */



#define MAXCAPT 100000
#define MAX_RINGBUF_FRAMES	300

#ifdef RECORD_TIMESTAMPS
/* What is the most timestamps we might want?
 * in field mode, 60 /sec, 3600/min...
 * 10 minutes would be 40000 - with two longs per timeval, and two
 * per sample, that is 640k - is that too much memory??
 *
 * sni recordings are closer to an hour, but because of (previous) disk
 * limitations on purkinje, we record 30 minute sections, 108000 frames.
 */
#define MAX_TIMESTAMPS	120000
typedef struct ts_pair {
	struct timeval grab_time;
	struct timeval stor_time;
} TS_Pair;

TS_Pair ts_array[MAX_TIMESTAMPS];

int stamping=0;
int n_stored_times;
#endif /* RECORD_TIMESTAMPS */

#define MAX_DISK_WRITER_THREADS		4	/* could be larger, but performance drops... */

#define DEFAULT_DISK_WRITER_THREADS	2	/* 2 should be sufficient, but on craik this causes fifo errors??? */
						/* on wheatstone (3 disks)
						 * having this be 2 triggers
						 * an error...
						 */

#define MAX_DISKS_PER_THREAD		4	/* should be ceil(((float)MAX_DISKS)/n_disk_writer_threads) */

static int n_disk_writer_threads=DEFAULT_DISK_WRITER_THREADS;

/* These are the data to be passed to each disk writer thread */

typedef struct per_proc_info {
	int	ppi_index;		/* thread index */
	pid_t	ppi_pid;
	int	ppi_fd[MAX_DISKS_PER_THREAD];			/* file descriptors */
	int32_t	ppi_nf;			/* number of frames this proc will write */
	int	ppi_flags;
	int	ppi_next_frm;	/* index into ring buffer */
#ifdef RECORD_CAPTURE_COUNT
	int	ppi_ccount[MAXCAPT];	/* capture count after each write */
#endif	/* RECORD_CAPTURE_COUNT */
	int	ppi_tot_disks;
	int	ppi_my_disks;		/* number of disks that this thread uses */
#ifdef TRACE_FLOW
	int	ppi_status;		/* holds disk_writer state */
#endif /* TRACE_FLOW */
} Proc_Info;

#define ppi_next_to_write	ppi_next_frm
#define ppi_next_to_read	ppi_next_frm


/* flag bits */
#define DW_READY_TO_GO	1
#define DW_EXITING	2

#define READY_TO_GO(pip)	( (pip)->ppi_flags & DW_READY_TO_GO )
#define EXITING(pip)		( (pip)->ppi_flags & DW_EXITING )


/* These global variables are used for inter-thread communication... */

static Proc_Info ppi[MAX_DISKS];
static int thread_write_enabled[MAX_DISKS]={1,1,1,1,1,1,1,1};

static int oldest, newest;	/* indices into ringbuf
				 *
				 * In the absence of wrap-around,
				 * (1+newest-oldest) = n_active.
				 */
static int n_ready_bufs;

void thread_write_enable(QSP_ARG_DECL  int index, int flag)
{
//#ifdef CAUTIOUS
//	if( index < 0 || index >= MAX_DISKS ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  thread_write_enable:  index %d out of range",index);
//		WARN(ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( index >= 0 && index < MAX_DISKS );
	
	thread_write_enabled[index]=flag;
}

#ifdef DEBUG_TIMERS

static void show_tmr(QSP_ARG_DECL  struct itimerval *tmrp)
{
	sprintf(ERROR_STRING,"interval:  %d   %d",
			tmrp->it_interval.tv_sec,
			tmrp->it_interval.tv_usec);
	advise(ERROR_STRING);

	sprintf(ERROR_STRING,"value:  %d   %d",
			tmrp->it_value.tv_sec,
			tmrp->it_value.tv_usec);
	advise(ERROR_STRING);
}

static void show_tmrs(SINGLE_QSP_ARG_DECL)
{
advise("real time timer:");
getitimer(ITIMER_REAL,&tmr1);
show_tmr(QSP_ARG  &tmr1);
}

#endif /* DEBUG_TIMERS */


#ifdef RECORD_TIMESTAMPS
static void reset_stamps()
{
	int i;
	struct timezone tz;

	/* This is a hack so that the values aren't
	 * completely wacky if an entry gets skipped because
	 * of the race problem.
	 */

	gettimeofday(&ts_array[0].stor_time,&tz);
	for(i=1;i<MAX_TIMESTAMPS;i++){
		ts_array[i].stor_time.tv_sec = ts_array[0].stor_time.tv_sec;
		ts_array[i].stor_time.tv_usec = ts_array[0].stor_time.tv_usec;
	}
}
#endif /* RECORD_TIMESTAMPS */

void set_async_record(int flag)
{ async_capture = flag; }

int get_async_record(void)
{ return async_capture; }

static void start_dw_threads(QSP_ARG_DECL  int32_t nf,int ndisks,int* fd_arr)
{
	int i;
	pthread_attr_t attr1;
	int disks_per_thread;

	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);

	/* Having one thread per disk is kind of inefficient, we really want
	 * to have the minimum number that are required for overlapping writes
	 * to finish in time...  With the new hardware, this is 2!
	 * The threads should alternate, so each thread should skip n_disk_writer_threads disks.
	 */

if( (ndisks % n_disk_writer_threads) != 0 ){
	sprintf(ERROR_STRING,"n_disk_writer_threads (%d) must evenly divide ndisks (%d)",
			n_disk_writer_threads,ndisks);
	ERROR1(ERROR_STRING);
}

	disks_per_thread = ndisks / n_disk_writer_threads;

	for(i=0;i<n_disk_writer_threads;i++){
		int j;

		ppi[i].ppi_index=i;
		ppi[i].ppi_nf = (nf+(n_disk_writer_threads-1)-i)/n_disk_writer_threads;
		ppi[i].ppi_flags = 0;
		for(j=0;j<disks_per_thread;j++)
			ppi[i].ppi_fd[j] = fd_arr[ i + j*n_disk_writer_threads ];
		ppi[i].ppi_tot_disks = ndisks;
		ppi[i].ppi_my_disks = disks_per_thread;

		pthread_create(&dw_thr[i],&attr1,disk_writer,&ppi[i]);
		/* pthread_create(&dw_thr[i],NULL,disk_writer,&ppi[i]); */
	}
}

static void start_grab_thread(QSP_ARG_DECL  vr_args *vrap)
{
	pthread_attr_t attr1;

	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);

	pthread_create(&grab_thr,&attr1,video_reader_thread,vrap);

#ifdef DEBUG_TIMERS
advise("master thread:");
show_tmrs(SINGLE_QSP_ARG);
#endif /* DEBUG_TIMERS */

}

/* We call this if we haven't terminated 5 seconds after we think we should...
 */

static void stream_wakeup(int unused)
{
	NWARN("stream_wakeup:  record failed!? (alarm went off before recording finished)");
	verbose=1;
	sprintf(DEFAULT_ERROR_STRING,"%d of %d frames captured",n_so_far,n_stream_frames);
	NADVISE(DEFAULT_ERROR_STRING);

#ifdef DEBUG_TIMERS
do_date();
show_tmrs(SGL_DEFAULT_QSP_ARG);
#endif /* DEBUG_TIMERS */

	/* MSTATUS(MS_ALARM); */

	exit(1);
}

/* We have 5 processes:  a master process which keeps track of the frame
 * buffer, and ndisks disk writer processes.  The disk writers need to wait
 * until there is at least one frame in the buffer.
 *
 * The first version of this program synchronized on each frame,
 * but this resulted in dropped frames...  so we can't wait for
 * each frame to be written before queueing the next request...
 *
 * So we need a data structure to communicate the state of the ring
 * bufffer to the disk writers (so they known when and what to write),
 * and the state of the disk writers to the master process
 * ( so it knows when to release elements of the ring buffer).
 *
 * We can have two variables to hold the state of the ring buffer:
 * oldest_available, and newest_available.
 * Each disk writer will have a variable last_written.
 * When last_written matches or exceeds oldest_available for all processes,
 * then we can release the buffer...  but what about wrap around???
 * This complicates the "exceeds"...
 */

void stream_record(QSP_ARG_DECL  Image_File *ifp,int32_t n_frames)
{
	int32_t npix;
	int fd_arr[MAX_DISKS];
	int ndisks;
	uint32_t total_blocks, blocks_per_frame;
	struct meteor_geomet _geo;
	Shape_Info shape;
	Shape_Info *shpp=(&shape);
	RV_Inode *inp;

	if( record_state != NOT_RECORDING ){
		sprintf(ERROR_STRING,
	"stream_record:  can't record file %s until previous record completes",
			ifp->if_name);
		WARN(ERROR_STRING);
		return;
	}

//	INSURE_MM("stream_record");
	assert( _mm != NULL );


	/* set_rt(); */

	if( MAX_RINGBUF_FRAMES < num_meteor_frames ){
		WARN("Need to recompile mcapt.c with a larger value of MAX_RINGBUF_FRAMES");
		return;
	}

	clear_buffers(SINGLE_QSP_ARG);	/* not really necessary */

//#ifdef CAUTIOUS
//	if( _mm->frame_size != meteor_bytes_per_pixel*meteor_columns*meteor_rows ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  _mm->frame_size = 0x%x, but bpp (%d) * cols (%d) * rows (%d) = 0x%x !?",
//			n_to_write,meteor_bytes_per_pixel,meteor_columns,meteor_rows,
//			meteor_bytes_per_pixel*meteor_columns*meteor_rows);
//		WARN(ERROR_STRING);
//		meteor_status(SINGLE_QSP_ARG);
//	}
//#endif /* CAUTIOUS */
	assert( _mm->frame_size == meteor_bytes_per_pixel*meteor_columns*meteor_rows );

	/* allow space for the timestamp, if we are recording timestamps... */

	blocks_per_frame = get_blocks_per_frame();

	n_to_write = blocks_per_frame * BLOCK_SIZE;

	n_to_write /= N_FRAGMENTS;
//#ifdef CAUTIOUS
//	if( (n_to_write % BLOCK_SIZE) != 0 ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  N_FRAGMENTS (%d) does not divide blocks_per_frame (%d) evenly",
//			N_FRAGMENTS,blocks_per_frame);
//		ERROR1(ERROR_STRING);
//	}
//#endif /* CAUTIOUS */
	assert( (n_to_write % BLOCK_SIZE) == 0 );


	total_blocks = n_frames * blocks_per_frame;

	if( FT_CODE(IF_TYPE(ifp)) != IFT_RV ){
		sprintf(ERROR_STRING,
	"stream record:  image file %s (type %s) should be type %s",
			ifp->if_name,
			FT_NAME(IF_TYPE(ifp)),
			FT_NAME(FILETYPE_FOR_CODE(IFT_RV)) );
		WARN(ERROR_STRING);
		return;
	}


	inp = (RV_Inode *)ifp->if_hdr_p;
	ndisks = queue_rv_file(QSP_ARG  inp,fd_arr);

//#ifdef CAUTIOUS
//	if( ndisks < 1 ){
//		sprintf(ERROR_STRING,
//			"Bad number (%d) of raw volume disks",ndisks);
//		WARN(ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( ndisks > 0 );

	if( num_meteor_frames < (2*ndisks) ){
		sprintf(ERROR_STRING,
	"buffer frames (%d) must be >= 2 x number of disks (%d)",
			num_meteor_frames,ndisks);
		WARN(ERROR_STRING);
		return;
	}


	/* set the shape info */
	meteor_get_geometry(&_geo);
	SET_SHP_FLAGS(shpp,0);
#ifdef FOOBAR
	shape.si_rows = _geo.rows - n_discard_lines;
#endif /* FOOBAR */

	/* what about the timestamp information that may be tacked onto
	 * the frame?  Does this not get written to the raw volume?
	 */

	SET_SHP_ROWS(shpp, _geo.rows);
	SET_SHP_COLS(shpp,_geo.columns);
	if( get_ofmt_index(QSP_ARG  _geo.oformat ) < 0 ){
		WARN("error determining bytes per pixel");
		SET_SHP_COMPS(shpp,DEFAULT_BYTES_PER_PIXEL);
	} else {
		SET_SHP_COMPS(shpp,meteor_bytes_per_pixel);
	}
	SET_SHP_FRAMES(shpp,n_frames);
	SET_SHP_SEQS(shpp, 1);
	SET_SHP_PREC_PTR(shpp,PREC_FOR_CODE(PREC_UBY) );
	//set_shape_flags(&shape,NO_OBJ);
	if( !meteor_field_mode )
		SET_SHP_FLAG_BITS(shpp,DT_INTERLACED);

	rv_set_shape(QSP_ARG  ifp->if_name,shpp);

	/* We write an entire frame to each disk in turn... */

	npix=n_to_write/meteor_bytes_per_pixel;

	/* For the sake of symmetry, we'll create ndisks child threads,
	 * and have the parent wait for them.
	 */

	oldest= 0;
	newest= (-1);
	n_ready_bufs=0;

#ifdef RECORD_TIMESTAMPS
	if( stamping ) reset_stamps();
#endif /* RECORD_TIMESTAMPS */

#ifdef ALLOW_RT_SCHED
	/* real-time scheduling priority
	 * This doesn't seem to make a whole lot of difference.
	 */
	if( try_rt_sched ) rt_sched(QSP_ARG  1);
#endif /* ALLOW_RT_SCHED */

	master_pid = getpid();

	start_dw_threads(QSP_ARG  n_frames,ndisks,fd_arr);

	vra1.vr_n_frames = n_frames;
	vra1.vr_n_disks = ndisks;
	vra1.vr_ifp = ifp;
#ifdef THREAD_SAFE_QUERY
	vra1.vr_qsp = qsp;
#endif

	record_state = RECORDING;

	n_stream_frames = n_frames;	/* remember for later printing... */

	if( async_capture ){
		start_grab_thread(QSP_ARG  &vra1);
	} else {
		video_reader(&vra1);
#ifdef ALLOW_RT_SCHED
		if( rt_is_on ) rt_sched(QSP_ARG  0);
#endif /* ALLOW_RT_SCHED */
		/* Because the disk writers don't use the fileio library,
		 * the ifp doesn't know how many frames have been written.
		 */
		ifp->if_nfrms = n_frames;	/* BUG? is this really what happened? */
	}
} /* end stream_record */

COMMAND_FUNC( meteor_wait_record )
{
#ifdef DEBUG_TIMERS
advise("meteor_wait_record:");
show_tmrs(SINGLE_QSP_ARG);
#endif /* DEBUG_TIMERS */

	if( ! recording_in_process ){
		advise("meteor_wait_record:  no recording currently in process!?");
		return;
	}

	/* BUG make sure this thread is still running! */

#ifdef FOOBAR
//#ifdef CAUTIOUS
//	/* Sometimes execution can reach this point before the grab thread
//	 * has executed... BUG
//	 */
//	if( grabber_pid == 0 )
//		ERROR1("CAUTIOUS:  meteor_wait_record:  no grabber thread");
//#endif /* CAUTIOUS */
	assert( grabber_pid != 0 );

	if( unassoc_pids(master_pid,grabber_pid) < 0 )
		ERROR1("error unassociating grabber pid");
#endif /* FOOBAR */

advise("joining w/ grab thread...");
	if( pthread_join(grab_thr,NULL) != 0 ){
		WARN("Error joining video reader thread");
	}
advise("joined!");

#ifdef ALLOW_RT_SCHED
	if( rt_is_on ) rt_sched(QSP_ARG  0);
#endif /* ALLOW_RT_SCHED */

}

COMMAND_FUNC( meteor_halt_record )
{
	if( record_state & RECORDING ){
		/* If we are here, we ought to be able to take it for granted
		 * that we are in fact in async mode...
		 */
		if( record_state & (RECORD_HALTING|RECORD_FINISHING) ){
			sprintf(ERROR_STRING,"meteor_halt_record:  halt already in progress!?");
			WARN(ERROR_STRING);
		} else {
			record_state |= RECORD_HALTING;
		}
	} else {
		/* We make this an advisory instead of a warning because
		 * the record might just have finished...
		 */
		sprintf(ERROR_STRING,"meteor_halt_record:  not currently recording!?");
		advise(ERROR_STRING);
		return;
	}

	meteor_wait_record(SINGLE_QSP_ARG);
}

static void *video_reader_thread(void *argp)
{
#ifdef FOOBAR
	grabber_pid=getpid();

sprintf(ERROR_STRING,"video_reader_thread:  grabber_pid = %d",grabber_pid);
advise(ERROR_STRING);

	if( assoc_pids(master_pid,grabber_pid) < 0 )
		ERROR1("video_reader_thread:  error associating pid");
#endif /* FOOBAR */

	return( video_reader(argp) );
}

static void *video_reader(void *argp)
{
	int all_ready;
	int need_oldest;
	int i;
	int32_t ending_count;
int last_newest=(-1);
	struct meteor_counts cnt;
	vr_args *vrap;
	int32_t n_frames, orig_n_frames;
	uint32_t final_size;
	struct drop_info di;
	int real_time_ok=1;
	int n_hw_errors=0;
	int ndisks;
	int seconds_before_alarm;
	Image_File *stream_ifp;
	RV_Inode *inp;
	Query_Stack *qsp; // needed for thread-safe-query

	STATUS_HELP(master_codes);
	STATUS_HELP(dw_codes);
	STATUS_HELP(column_doc);

	vrap = (vr_args*) argp;

	orig_n_frames = n_frames = vrap->vr_n_frames;
	ndisks = vrap->vr_n_disks;
	stream_ifp = vrap->vr_ifp;
	qsp = vrap->vr_qsp;

	inp = (RV_Inode *)stream_ifp->if_hdr_p;


#ifdef RECORD_TIMESTAMPS
	n_stored_times = 0;
#endif /* RECORD_TIMESTAMPS */

	/* Wait for all threads to get ready before starting grabber */

	all_ready = 0;
	while( ! all_ready ){
		all_ready=1;
		for(i=0;i<n_disk_writer_threads;i++){
			if( ! READY_TO_GO(&ppi[i]) ){
				YIELD_PROC(1)
				all_ready=0;
			}
		}
	}

MSTATUS(MS_INIT)

	/* Now start the meteor capturing */

	/* set synchronous mode */
	capture_code = METEORCAPFRM;
	_hiwat=num_meteor_frames-2;	/* stop when this many bufs are full */
	_lowat=_hiwat-1;		/* restart asap */

	/* zero the error counts */
	meteor_clear_counts();

	if( meteor_capture(SINGLE_QSP_ARG) < 0 )
		ERROR1("meteor_capture() failed");


	/* wait for the first frame */

	/* Apparently, a buffer is 'active' while it is being filled...
	 * therefore, we wait for this to be 2, so we know the first
	 * frame has been completely captured before we start writing.
	 * But we can do a little housekeeping while the first frame
	 * is filling...
	 */

	while( _mm->num_active_bufs < 1 ){
MSTATUS(MS_WAITING)
/*
sprintf(ERROR_STRING,"num_activ_bufs = %d",_mm->num_active_bufs);
advise(ERROR_STRING);
*/
		usleep(1000);
	}


	/* remember the starting frame count */
	starting_count = _mm->frames_captured;

	n_so_far = 0;

	/* schedule a wakeup for 30 seconds after we think we should finish */

#define FRAMES_PER_SECOND	30
#define FIELDS_PER_SECOND	60

#define N_EXTRA_SECONDS		30

	/* this worked well when we were not in field mode... */
	if( !meteor_field_mode )
		seconds_before_alarm = N_EXTRA_SECONDS + (n_frames+FRAMES_PER_SECOND-1)/FRAMES_PER_SECOND;
	else
		seconds_before_alarm = N_EXTRA_SECONDS + (n_frames+FIELDS_PER_SECOND-1)/FIELDS_PER_SECOND;

#ifdef DEBUG_TIMERS
do_date();
sprintf(ERROR_STRING,"calling alarm(%d)",seconds_before_alarm);
advise(ERROR_STRING);
#endif /* DEBUG_TIMERS */

	old_alarm = alarm(seconds_before_alarm);

#ifdef DEBUG_TIMERS
if( old_alarm > 0 ){
sprintf(ERROR_STRING,"old alarm would have occurred in %d seconds",
old_alarm);
advise(ERROR_STRING);
} else advise("no old alarm was pending");
show_tmrs(SINGLE_QSP_ARG);
#endif /* DEBUG_TIMERS */

	/* BUG should we call alarm(0) to cancel a previously pending alarm? */
	signal(SIGALRM,stream_wakeup);

	/* BUG??? do we need to cancel the alarm at the end of the recording??? */

	while( n_so_far < n_frames ){

		/* We know we have a frame ready when we first enter,
		 * but later we might need to wait.
		 *
		 * Also, what about that comment that num_active_bufs is 1
		 * when the first frame is filling?  Wait for it to be 2 to be
		 * safe...
		 */

		while( _mm->num_active_bufs <= 1 ){
MSTATUS(MS_WAITING)
			usleep(10000);
		}

		/* See if all the disk writers have written another frame */

		/* In the older version, where each disk had a piece of each frame,
		 * we really needed to check all four disk writers...
		 * Now we could figure out which one is responsible for writing
		 * oldest, and only check it, but this way will still work...
		 *
		 * We used to only throw out the single oldest frame here,
		 * but it turns out that sometimes by the time this code executes
		 * there can be several frames that are ready to go.
		 */

MSTATUS(MS_CHECKING)
		need_oldest=0;
		while( (!need_oldest) && oldest != newest ){
			/* see if any disk writers want this frame */
			for(i=0;i<n_disk_writer_threads;i++){
				if( ( ! EXITING(&ppi[i]) ) &&
					ppi[i].ppi_next_to_write == oldest )

					need_oldest=1;
			}
			if( !need_oldest ){
MSTATUS(MS_RELEASING)
				/* release this buffer,
				 * and increment oldest.
				 */
				_mm->num_active_bufs--;

				oldest++;
				n_so_far++;
				if( oldest >= num_meteor_frames )
					oldest=0;
			}
		}

		/* update newest */
		/* this used to be num_active_bufs-1, we change to -2
		 * because of the suspicion that a buf is "active"
		 * while being filled...
		 */
		newest = (oldest + _mm->num_active_bufs - 2) % num_meteor_frames;
		/* video_reader updating n_ready_bufs... */
		if( newest >= oldest ){
			n_ready_bufs = newest-oldest+1;
		} else {
			/* BUG (fixed?) when we start out, newest is initialized to -1! */
			if( newest >= 0 )
				n_ready_bufs = newest+(num_meteor_frames-oldest);
			else
				n_ready_bufs = 0;
		}

MSTATUS(MS_BOT)

last_newest = newest;
		/* give up processor */
		YIELD_PROC(100)

		/* shut things down if the user has given a halt command */

		if( record_state & RECORD_HALTING ){	/* halt requested */
advise("video_reader HALTING");
			/* we capture ndisks more frames so that none of
			 * the disk writer processes hang waiting for data.
			 * We round up to a multiple of ndisks, so that each
			 * rawvol disk has something...
			 * There is a possibility of a race condition here,
			 * because a disk writer process could get swapped in
			 * before we update the ppi's, so we add another
			 * 4 frames for good measure.  what's 120 ms
			 * between friends, anyway?
			 */

#define ADDITIONAL_FRAMES_PER_DISK	3
#define ADDITIONAL_FRAMES	(ndisks*ADDITIONAL_FRAMES_PER_DISK)

/* NEAREST_MULTIPLE rounds down... */

#define NEAREST_MULTIPLE( num , factor )			\
								\
		( ( ( num + (factor)/2 ) / ( factor ) ) * ( factor ) )

			if( n_frames >
		NEAREST_MULTIPLE( (n_so_far+(ADDITIONAL_FRAMES-1)), ndisks ) ){
				int i;
				n_frames = NEAREST_MULTIPLE( (n_so_far+(ADDITIONAL_FRAMES-1)), ndisks );
				for(i=0;i<n_disk_writer_threads;i++)
					ppi[i].ppi_nf = n_frames/n_disk_writer_threads;
advise("ppi_nf changed");
			}
else
advise("ppi_nf NOT changed");
			/* If n_frames is small, we just let nature take its course */

			record_state |= RECORD_FINISHING;
			record_state &= ~RECORD_HALTING;

		} /* end halt check section */
	} /* end main video read loop */
MSTATUS(MS_DONE)

	signal(SIGALRM,SIG_IGN);

	/* We have to cancel the alarm...  This was revealed when we had
	 * some synchronous (non-threaded) recordings followed by some
	 * asynchronous (multi-threaded) recordings...  In the
	 * multi-threaded case, the new thread gets its own timers,
	 * so if an old timer from the master thread is still ticking,
	 * it will go off and call the wakeup routine, even if the video
	 * reader thread has reset its own timer.
	 *
	 * This behavior is not clearly documented anywhere...
	 */

	old_alarm = alarm(0);

#ifdef DEBUG_TIMER
if( old_alarm > 0 ){
sprintf(ERROR_STRING,"old alarm would have occurred in %d seconds",
old_alarm);
advise(ERROR_STRING);
} else advise("no old alarm was pending");
#endif /* DEBUG_TIMER */

if( verbose ) advise("main thread stopping capture");
	meteor_stop_capture(SINGLE_QSP_ARG);

	/* check the total number of dropped frames */
	ending_count = _mm->frames_captured;

	/* wait for disk writer threads to finish */

	for(i=0;i<n_disk_writer_threads;i++){
#ifdef FOOBAR
		if( unassoc_pids(master_pid,ppi[i].ppi_pid) < 0 )
			ERROR1("error unassociating disk writer process id");
#endif /* FOOBAR */

		if( pthread_join(dw_thr[i],NULL) != 0 ){
			sprintf(ERROR_STRING,"Error joining disk writer thread %d",i);
			WARN(ERROR_STRING);
		}
	}

	/* There is some confusion about what the ending count should be...
	 * This used to be the test for real time recording success,
	 * but now we just trust in the driver to tell us this.
	 */

	/*
	if( (ending_count-starting_count) != n_frames ){
		sprintf(ERROR_STRING,
	"Wanted %d frames, captured %d (%d-%d-1)",n_frames,
			(ending_count-starting_count)-1,ending_count,starting_count);
		WARN(ERROR_STRING);
	} else {
		advise("Recording done in real time");
	}
	*/

	/* it looks like ther might be a race condition here when recording
	 * asynchronously - in one experimental run, a stimulus timed out,
	 * (e.g. the record finished before or around the time the halt cmd
	 * would have been sent) and all the subsequent trials were recorded w/
	 * len 12 ~!? (ash.1.8)
	 *
	 * A problem would occur if record_state were tested right before this
	 * assignment, and then another thread set the state to halting right
	 * after...
	 */

	record_state = NOT_RECORDING;

	if (ioctl(meteor_fd, METEORGCOUNT, &cnt)) {
		perror("ioctl GetCount failed");
	} else {
		if( cnt.fifo_errors > 0 ){
#ifdef IA64
			sprintf(ERROR_STRING,"Movie %s:  %d fifo errors",
				stream_ifp->if_name,cnt.fifo_errors);
#else
			sprintf(ERROR_STRING,"Movie %s:  %d fifo errors",
				stream_ifp->if_name,cnt.fifo_errors);
#endif
			WARN(ERROR_STRING);
			real_time_ok = 0;
		}
		if( cnt.dma_errors > 0 ){
#ifdef IA64
			sprintf(ERROR_STRING,"Movie %s:  %d dma errors",
				stream_ifp->if_name,cnt.dma_errors);
#else
			sprintf(ERROR_STRING,"Movie %s:  %d dma errors",
				stream_ifp->if_name,cnt.dma_errors);
#endif
			WARN(ERROR_STRING);
			real_time_ok = 0;
		}
		n_hw_errors = cnt.fifo_errors + cnt.dma_errors;
	}

	/* BUG here we should get the indices of the bad
	 * frames and store them in the RV record...
	 * maybe we should do something similar for dropped frames?
	 */

	if( ioctl(meteor_fd, METEORGNDROP, &di)) {
		perror("ioctl GNDROP failed");
	} else if( di.n_total > 0 ) {
#ifdef IA64
		sprintf(ERROR_STRING,"Movie %s not NOT recorded in real time (%d frames dropped)",
			stream_ifp->if_name,di.n_total);
#else
		sprintf(ERROR_STRING,"Movie %s not NOT recorded in real time (%d frames dropped)",
			stream_ifp->if_name,di.n_total);
#endif
		WARN(ERROR_STRING);
		real_time_ok = 0;
	}

	if( real_time_ok ){
		sprintf(ERROR_STRING,"video_reader:  Movie %s recorded successfully in real time.",stream_ifp->if_name);
		advise(ERROR_STRING);
	}
	if( n_hw_errors > 0 ){
		WARN("Some frames may have corrupt data!?");
	}

	if( orig_n_frames != n_frames ){
		/* recompute the size of the file in case we were halted */

		final_size = meteor_rows * meteor_columns * meteor_bytes_per_pixel;

		/* we used to divide size by 2 here if in field mode,
		 * but that was a bug, because field mode reduces meteor_rows...
		 *
		 * if( meteor_field_mode ) final_size /= 2;
		 */

		final_size += (BLOCK_SIZE-1);
		final_size /= BLOCK_SIZE;	/* frame size in blocks */
						/* (rounded up to block boundary) */
		final_size *= n_frames;		/* total blocks */

		rv_truncate( inp, final_size );

		/* rv_truncate corrects the size, but doesn't know about nframes... */

		SET_SHP_FRAMES( RV_MOVIE_SHAPE(inp), n_frames);

		/* Have to do it in stream_ifp->if_dp too, in order to get a correct
		 * answer from the nframes() expression function...
		 */

		stream_ifp->if_nfrms = n_frames;
#ifdef FOO
		assert( stream_ifp->if_dp != NO_OBJ );
//#ifdef CAUTIOUS
//		if( stream_ifp->if_dp == NO_OBJ )
//			WARN("CAUTIOUS:  stream_ifp has NULL if_dp!?");
//		else
//#endif /* CAUTIOUS */
			stream_ifp->if_dp->dt_frames = n_frames;
#endif /* FOO */

	}

	rv_sync(SINGLE_QSP_ARG);

	/* we used to disable real-time scheduling here, but
	 * when video_reader executes as a separate thread there
	 * is no point, because it won't affect the main process!
	 */

	recording_in_process = 0;

//#ifdef CAUTIOUS
//	if( stream_ifp == NO_IMAGE_FILE ){
//		WARN("CAUTIOUS:  video_reader:  stream_ifp is NULL!?");
//		return(NULL);
//	}
//#endif /* CAUTIOUS */
	assert( stream_ifp != NO_IMAGE_FILE );

	finish_recording( QSP_ARG  stream_ifp );

	return(NULL);

} /* end video_reader */

/* disk_writer needs to have its own ERROR_STRING, but we fork these threads
 * from a single command, so passing the qsp won't help...
 */

void *disk_writer(void *argp)
{
	Proc_Info *pip;
	int fd;
	int32_t j;
	int i_frag;
	int next;
#ifdef FOOBAR
struct timeval tmp_time1,tmp_time2;
struct timezone tmp_tz;
#endif /* FOOBAR */

	pip = (Proc_Info*) argp;
STATUS(DW_INIT)

	pip->ppi_pid = getpid();
#ifdef FOOBAR
	if( assoc_pids(master_pid,pip->ppi_pid) < 0 )
		ERROR1("disk_writer:  error associating pid");
#endif /* FOOBAR */

	/* tell the parent that we're ready, and wait for siblings */
	pip->ppi_flags |= DW_READY_TO_GO;

	pip->ppi_next_to_write = pip->ppi_index;	/* 0,1 ( ,2,3) */

	/* wait for the first frame */
	while( newest < pip->ppi_next_to_write )
		usleep(10000);

	for(j=0;j<pip->ppi_nf;j++){
		int n_written;		/* number of bytes written */
		char *buf;
		int ready;
		int nwaits;

		/* See if we have data available to write.
		 */
STATUS(DW_TOP)
		ready=0;
		nwaits=1;
		next = pip->ppi_next_to_write;

		fd = pip->ppi_fd[ j % pip->ppi_my_disks ];
#ifdef SPECIAL_TEST
		fd = open("/dev/null",O_RDWR);
		if( fd < 0 ){
			perror("open");
			ERROR1("error opening /dev/null");
		}
#endif /* SPECIAL_TEST */

		while( ! ready ){
#ifndef FOOBAR
			if( oldest <= newest ){		/* Not wrapped around... */
				if( next <= newest && next >= oldest )
					ready=1;
			} else {			/* newest has wrapped around */
				if( next <= newest || next >= oldest )
					ready=1;
			}
#else
			/* This logic is so that the buffers will fill up,
			 * and writing will be happening far from reading in
			 * memory...  But this will make a problem with the last
			 * few frames...
			 */
			/*
			if( n_ready_bufs > num_meteor_frames/2 ){
				ready=1;
			}
			*/
			/* Start as soon as we know there will be a frame for us... */
			if( n_ready_bufs > n_disk_writer_threads ){
				ready=1;
			}
#endif

			if( ! ready ){
STATUS(DW_WAIT)
				/* YIELD_PROC here does not work...
				 * we get into a deadlock situation,
				 * where some writes never complete...
				 */
				usleep(nwaits);
				nwaits *= 2;
			}
		}

		buf = mmbuf + meteor_off.frame_offset[next];
//sprintf(ERROR_STRING,"disk_writer:  next %d   addr 0x%lx",next,(int_for_addr)buf);
//advise(ERROR_STRING);

		/* write out the next frame */

STATUS(DW_WRITE)

		/* n_really_to_write was used for testing disk
		 * performance and bus saturation...
		 */


#ifdef RECORD_TIMESTAMPS
		/* We don't really want to just increment i_stamp, because the threads can execute in
		 * variable order...
		 * How do we know which frame this is???
		 * j      is the index of the frame for this thread...
		 * pip->ppi_index	is the index of the frame for j=0
		 * SO
		 * i_stamp = pip->ppi_index + j * n_disk_writer_threads
		 */

		if( stamping ){
			int i_stamp;

			i_stamp = pip->ppi_index + j * n_disk_writer_threads;
			if( i_stamp < MAX_TIMESTAMPS ){
				struct timezone tz;
				struct timeval *tsp;

				if( gettimeofday(&ts_array[i_stamp].stor_time,&tz) < 0 )
					perror("gettimeofday");

				tsp = (struct timeval *)(buf + _mm->frame_size);
				ts_array[i_stamp].grab_time = *tsp;

				if( i_stamp >= n_stored_times )
					n_stored_times = i_stamp + 1;
			}
		}
#endif /* RECORD_TIMESTAMPS */

		if( thread_write_enabled[pip->ppi_index] ){
#ifdef FOOBAR
/* for debugging, time the disk write operation.  On wheatstone (a new machine with fast disks), we are getting
 * fifo error, suggesting that the meteor is not getting enough bus bandwidth.  Maybe we can slow down the disk writes?
 */
if( gettimeofday(&tmp_time1,&tmp_tz) < 0 )
perror("gettimeofday");
#endif /* FOOBAR */
			for(i_frag=0;i_frag<N_FRAGMENTS;i_frag++){
				if( (n_written = write(fd,buf+i_frag*n_to_write,n_to_write))
					!= n_to_write ){
					sprintf(DEFAULT_ERROR_STRING,"write (frm %d, fd=%d)",next,fd);
					perror(DEFAULT_ERROR_STRING);
					sprintf(DEFAULT_ERROR_STRING,
						"%d requested, %d written",
						n_to_write,n_written);
					NWARN(DEFAULT_ERROR_STRING);
					return(NULL);
				}
#ifdef FOOBAR
if( gettimeofday(&tmp_time2,&tmp_tz) < 0 )
perror("gettimeofday");
/* calculate elapsed time */
sprintf(DEFAULT_ERROR_STRING,"write time frame %d frag %d:  %d msec",j+1,i_frag+1,1000*(tmp_time2.tv_sec-tmp_time1.tv_sec)+
(tmp_time2.tv_usec-tmp_time1.tv_usec)/1000);
advise(DEFAULT_ERROR_STRING);
#endif /* FOOBAR */

			}
		}

STATUS(DW_DONE)

		/* Now we've written a frame.
		 * Let the parent know, so that when all threads
		 * have so signalled the frame can be released to
		 * the ring buffer.
		 */

		pip->ppi_next_to_write += n_disk_writer_threads;

		if( pip->ppi_next_to_write >= num_meteor_frames )	/* wrap around */
			pip->ppi_next_to_write -= num_meteor_frames;

#ifdef RECORD_CAPTURE_COUNT
		/* record # frames caputured while writing. */
		pip->ppi_ccount[j]=_mm->frames_captured - starting_count;
#endif	/* RECORD_CAPTURE_COUNT */
/*
if( verbose ){
sprintf(estring[pip->ppi_index],"%c\tpri = %d",'0'+pip->ppi_index,pri);
advise(estring[pip->ppi_index]);
}
*/

STATUS(DW_BOT)
	}
STATUS(DW_EXIT)

	pip->ppi_flags |= DW_EXITING;

	return(NULL);

} /* end disk_writer() */

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

/* write 0's to all frames */
static void clear_buffers(SINGLE_QSP_ARG_DECL)
{
	uint32_t *p,npix;
	int i;
	unsigned int j;

	if( meteor_bytes_per_pixel != DEFAULT_BYTES_PER_PIXEL ) {
		sprintf(ERROR_STRING,
			"clear_buffers:  meteor_bytes_per_pixel = %d (expected %d)",
			meteor_bytes_per_pixel,DEFAULT_BYTES_PER_PIXEL);
		WARN(ERROR_STRING);
		return;
	}

	npix = meteor_columns * meteor_rows ;
	for(i=0;i<num_meteor_frames;i++){
		p = (uint32_t *)(mmbuf + meteor_off.frame_offset[i]);
		for(j=0;j<npix;j++){
			*p++ = 0;
		}
	}
}

#ifdef RECORD_TIMESTAMPS
void dump_timestamps(const char *filename)
{
	int i;
	FILE *fp;
	int32_t ds,dus,ds2,dus2;

	fp=try_open(filename,"w");
	if( !fp ) return;

#ifdef RELATIVE_TIME
	ds=dus=0;
	fprintf(fp,"%d\t%d\n",ds,dus);
	for(i=1;i<n_stored_times;i++){
		ds = (ts_array[i].grab_time.tv_sec - ts_array[0].grab_time.tv_sec);
		dus = (ts_array[i].grab_time.tv_usec - ts_array[0].grab_time.tv_usec);
		if( dus < 0 ){
			dus += 1000000;
			ds -= 1;
		}
		fprintf(fp,"%d\t%d\n",ds,dus);
	}
#else
	for(i=0;i<n_stored_times;i++){
		ds = ts_array[i].grab_time.tv_sec;
		dus = ts_array[i].grab_time.tv_usec;
		ds2 = ts_array[i].stor_time.tv_sec;
		dus2 = ts_array[i].stor_time.tv_usec;

		fprintf(fp,"%d\t%d\t%d\t%d\n",ds,dus,ds2,dus2);
	}
#endif
	fclose(fp);
}

void print_grab_times()
{
	int i;
	char *s;

	for(i=0;i<n_stored_times;i++){
		s=ctime(&ts_array[i].grab_time.tv_sec);
		/* remove trailing newline */
		if( s[ strlen(s) - 1 ] == '\n' ) s[ strlen(s) - 1 ] = 0;
		sprintf(msg_str,"%s\t%ld\t%3ld.%03ld",s,
				ts_array[i].grab_time.tv_sec,
				ts_array[i].grab_time.tv_usec/1000,
				ts_array[i].grab_time.tv_usec%1000
				);
		prt_msg(msg_str);
	}
}

void print_store_times()
{
	int i;
	char *s;

	for(i=0;i<n_stored_times;i++){
		s=ctime(&ts_array[i].stor_time.tv_sec);
		/* remove trailing newline */
		if( s[ strlen(s) - 1 ] == '\n' ) s[ strlen(s) - 1 ] = 0;
		sprintf(msg_str,"%s\t%ld\t%3ld.%03ld",s,
				ts_array[i].stor_time.tv_sec,
				ts_array[i].stor_time.tv_usec/1000,
				ts_array[i].stor_time.tv_usec%1000
				);
		prt_msg(msg_str);
	}
}

#endif /* RECORD_TIMESTAMPS */

#endif /* HAVE_METEOR */

