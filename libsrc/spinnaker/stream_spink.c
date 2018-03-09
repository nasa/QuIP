
#include "quip_config.h"

#ifdef HAVE_LIBSPINNAKER

/* Stream video to disk.
 *
 * adapted from meteor library for libflycap.
 *
 * First implementation on euler, two 1 TB sata drives, each over 100 Mb/sec
 * Camera is Flea3 USB, 1.3 MB at 150 fps
 *
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
#include "spink.h"
#include "gmovie.h"

//#include "debug.h"
//#include "data_obj.h"
#include "fio_api.h"

#ifdef ALLOW_RT_SCHED

#include "rt_sched.h"
#define YIELD_PROC(time)	{ if( rt_is_on ) sched_yield(); else usleep(time); }

#else /* ! ALLOW_RT_SCHED */

#define YIELD_PROC(time)	usleep(time);

#endif /* ALLOW_RT_SCHED */


// New local variables introduced for
// porting from meteor to flycap
static int recording_in_process;
static int n_enqueued;
static int n_frames_read=0;
static int mem_locked=0;

/* data passed to the video reader */

typedef struct {
	int32_t		vr_n_frames;
	int		vr_n_disks;
	Image_File *	vr_ifp;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *	vr_qsp;		// needed for thread-safe-query
#endif // THREAD_SAFE_QUERY
	Spink_Cam *	vr_cam_p;
} vr_args;

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

#define DEFAULT_BYTES_PER_PIXEL		1

/* disk writer stuff */
static int32_t n_stream_frames;		/* a global in case we want to print the size of the request */
static int32_t starting_count;
static int32_t n_to_write;		/* number of bytes per write op */
#define N_FRAGMENTS 1			/* blocks_per_frame is ? */
static pthread_t dw_thr[MAX_DISKS];
static pthread_t grab_thr;

static int async_capture=0;

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
	"thread status newest n_read status next_to_wt ...\n";

#define MSTATUS(code)						\
m_status = code;						\
if( verbose ){							\
sprintf(ERROR_STRING,						\
"M %c\t%d\t%d\t%c %d/%d\t%c %d/%d",				\
mstatstr[m_status],newest,n_frames_read,			\
statstr[ppi[0].ppi_status],					\
ppi[0].ppi_n_frames_written,					\
ppi[0].ppi_n_enqueued,						\
statstr[ppi[1].ppi_status],					\
ppi[1].ppi_n_frames_written,					\
ppi[1].ppi_n_enqueued						\
);								\
advise(ERROR_STRING);						\
}

#define STATUS(code)						\
pip->ppi_status = code;						\
if( verbose ){							\
sprintf(estring[pip->ppi_index],				\
"%c %c\t%d\t%d\t%c %d/%d\t%c %d/%d",			\
'0'+pip->ppi_index,mstatstr[m_status],newest,n_frames_read,	\
statstr[ppi[0].ppi_status],					\
ppi[0].ppi_n_frames_written,					\
ppi[0].ppi_n_enqueued,					\
statstr[ppi[1].ppi_status],					\
ppi[1].ppi_n_frames_written,					\
ppi[1].ppi_n_enqueued					\
);								\
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
ppi[0].ppi_n_enqueued,					\
rstatstr[ppi[1].ppi_status],					\
ppi[1].ppi_n_enqueued,					\
rstatstr[ppi[2].ppi_status],					\
ppi[2].ppi_n_enqueued,					\
rstatstr[ppi[3].ppi_status],					\
ppi[3].ppi_n_enqueued);					\
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
ppi[0].ppi_n_enqueued,					\
rstatstr[ppi[1].ppi_status],					\
ppi[1].ppi_n_enqueued,					\
rstatstr[ppi[2].ppi_status],					\
ppi[2].ppi_n_enqueued,					\
rstatstr[ppi[3].ppi_status],					\
ppi[3].ppi_n_enqueued);					\
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

#define DW_QUEUE_LEN	64	// needs to be at least the number of buffers

typedef struct per_proc_info {
	int	ppi_index;		/* thread index */
	pid_t	ppi_pid;
	int	ppi_fd[MAX_DISKS_PER_THREAD];			/* file descriptors */
	int32_t	ppi_n_frames_to_write;	/* number of frames this proc will write */
	int32_t	ppi_n_frames_written;	/* number of frames already written */
	int	ppi_flags;

	int	ppi_queue[DW_QUEUE_LEN];	// indices of buffers to write

	// the video_reader enqueues a frame writing the index of the buffer
	// into the queue at queue_wt_idx, which it then increment,
	// finally incrementing n_enqueued.
	//
	// The disk_writers waits for n_enqueued to be larger than n_dequeued.
	// Then it writes the buffer from the queue at queue_rd_idx, then
	// increments n_dequeued.
	int	ppi_queue_wt_idx;		// video reader controls this
	int	ppi_queue_rd_idx;		// disk writer controls this
	// rather than keep a single counter (which would be subject to race
	// conditions), we keep separate counters for video_reader and
	// disk_writer to manipulate independently
	int	ppi_n_enqueued;
	int	ppi_n_dequeued;

#ifdef RECORD_CAPTURE_COUNT
	int	ppi_ccount[MAXCAPT];	/* capture count after each write */
#endif	/* RECORD_CAPTURE_COUNT */
	int	ppi_tot_disks;
	int	ppi_my_disks;		/* number of disks that this thread uses */
#ifdef TRACE_FLOW
	int	ppi_status;		/* holds disk_writer state */
#endif /* TRACE_FLOW */
	Spink_Cam *	ppi_cam_p;
} Proc_Info;

//#define ppi_next_to_write	ppi_next_frm
//#define ppi_next_to_read	ppi_next_frm


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

#ifdef NOT_USED
static void thread_write_enable(QSP_ARG_DECL  int index, int flag)
{
	assert( index >= 0 && index < MAX_DISKS );
	thread_write_enabled[index]=flag;
}
#endif // NOT_USED

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

/* disk_writer needs to have its own ERROR_STRING, but we fork these threads
 * from a single command, so passing the qsp won't help...
 *
 * Running the Flea3 at 500 fps, we see alternate frames offset by 10,
 * which might be explained by one of the disk writers getting 10 frames behind...
 *
 * We would like to know more about how libflycap handles the ring buffer...
 */

static void *disk_writer(void *argp)
{
	Proc_Info *pip;
	int fd;
	int32_t j;
	int i_frag;
#ifdef FOOBAR
struct timeval tmp_time1,tmp_time2;
struct timezone tmp_tz;
#endif /* FOOBAR */

	pip = (Proc_Info*) argp;
STATUS(DW_INIT)

	pip->ppi_pid = getpid();
#ifdef FOOBAR
	if( assoc_pids(master_pid,pip->ppi_pid) < 0 )
		error1("disk_writer:  error associating pid");
#endif /* FOOBAR */

	/* tell the parent that we're ready, and wait for siblings */
	pip->ppi_flags |= DW_READY_TO_GO;

	/* wait for the first frame */
	while( pip->ppi_n_enqueued <= pip->ppi_n_dequeued ){
		//usleep(10000);	// this is too long at high frame rates
//fprintf(stderr,"disk_writer sleeping before first frame...\n");
		usleep(1000);
	}

	for(j=0;j<pip->ppi_n_frames_to_write;j++){
		int n_written;		/* number of bytes written */
		char *buf;
		int ready;
		int queue_idx, buf_idx;

		/* See if we have data available to write.
		 */
STATUS(DW_TOP)
		ready=0;
		//next = pip->ppi_next_to_write;

		fd = pip->ppi_fd[ j % pip->ppi_my_disks ];

		while( ! ready ){
			if( pip->ppi_n_enqueued > pip->ppi_n_dequeued )
				ready=1;

			if( ! ready ){
STATUS(DW_WAIT)
				/* YIELD_PROC here does not work...
				 * we get into a deadlock situation,
				 * where some writes never complete...
				 */
				// 512 was sometimes needed twice when capturing
				// at 500 fps w/ PGR usb3 camera...
				// We used to have a variable sleep that we would make
				// longer each time we woke up and still weren't ready...
				usleep(512);
			}
		}

		// we need to figure out which queue entry to look at...
		queue_idx = j % DW_QUEUE_LEN;
		buf_idx = pip->ppi_queue[ queue_idx ];


		buf = OBJ_DATA_PTR( pip->ppi_cam_p->skc_frm_dp_tbl[buf_idx] );

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
					sprintf(DEFAULT_ERROR_STRING,"write (frm %d, fd=%d, i_frag = %d, buf = 0x%lx, n = %d )",buf_idx,fd,i_frag,(long)buf,n_to_write);
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

		pip->ppi_n_frames_written ++;

		/* Now we've written a frame.
		 * Let the parent know, so that when all threads
		 * have so signalled the frame can be released to
		 * the ring buffer.
		 */

		pip->ppi_n_dequeued ++;
		//pip->ppi_next_to_write += n_disk_writer_threads;

		//if( pip->ppi_next_to_write >= pip->ppi_cam_p->skc_n_buffers )	/* wrap around */
			//pip->ppi_next_to_write -= pip->ppi_cam_p->skc_n_buffers;

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

void fly_set_async_record(int flag)
{ async_capture = flag; }

int fly_get_async_record(void)
{ return async_capture; }

static void start_dw_threads(QSP_ARG_DECL  int32_t nf,int n_disks,int* fd_arr, Spink_Cam *scp)
{
	int i;
	pthread_attr_t attr1;
	int disks_per_thread;

	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);

	/* Having one thread per disk is kind of inefficient, we really want
	 * to have the minimum number that are required for overlapping writes
	 * to finish in time...  With the new hardware, this is 2!
	 * The threads should alternate, so each thread should skip
	 * n_disk_writer_threads disks.
	 */

if( (n_disks % n_disk_writer_threads) != 0 ){
	sprintf(ERROR_STRING,"n_disk_writer_threads (%d) must evenly divide n_disks (%d)",
			n_disk_writer_threads,n_disks);
	error1(ERROR_STRING);
}

	disks_per_thread = n_disks / n_disk_writer_threads;

	for(i=0;i<n_disk_writer_threads;i++){
		int j;

		ppi[i].ppi_index=i;
		ppi[i].ppi_n_frames_to_write = (nf+(n_disk_writer_threads-1)-i)/n_disk_writer_threads;
		ppi[i].ppi_n_frames_written = 0;

		ppi[i].ppi_queue_rd_idx = (-1);	// nothing to send to disk
		ppi[i].ppi_queue_wt_idx = 0;	// use first location
		ppi[i].ppi_n_enqueued = 0;
		ppi[i].ppi_n_dequeued = 0;

		ppi[i].ppi_flags = 0;
		for(j=0;j<disks_per_thread;j++){
			ppi[i].ppi_fd[j] = fd_arr[ i + j*n_disk_writer_threads ];
		}
		ppi[i].ppi_tot_disks = n_disks;
		ppi[i].ppi_my_disks = disks_per_thread;

		ppi[i].ppi_cam_p = scp;

		pthread_create(&dw_thr[i],&attr1,disk_writer,&ppi[i]);
		/* pthread_create(&dw_thr[i],NULL,disk_writer,&ppi[i]); */
	}
} // start_dw_threads

/* We call this if we haven't terminated 5 seconds after we think we should...
 */

static void stream_wakeup(int unused)
{
	NWARN("stream_wakeup:  record failed!? (alarm went off before recording finished)");
	verbose=1;
	sprintf(DEFAULT_ERROR_STRING,"%d of %d frames captured",n_enqueued,n_stream_frames);
	NADVISE(DEFAULT_ERROR_STRING);

#ifdef DEBUG_TIMERS
do_date();
show_tmrs(SGL_DEFAULT_QSP_ARG);
#endif /* DEBUG_TIMERS */

	/* MSTATUS(MS_ALARM); */

	exit(1);
}

#ifdef MOVED

// This should be moved to lib rawvol or libmvimenu???

/*static*/ void _make_movie_from_inode(QSP_ARG_DECL  RV_Inode *inp)
{
	Movie *mvip;
	Image_File *ifp;

	if( is_rv_directory(inp) || is_rv_link(inp) ){
		if( verbose ){
			sprintf(ERROR_STRING,"make_movie_from_inode:  rv inode %s is not a movie",rv_name(inp));
			advise(ERROR_STRING);
		}
		return;
	}

	mvip = create_movie(rv_name(inp));
	if( mvip == NULL ){
		sprintf(ERROR_STRING,
			"error creating movie %s",rv_name(inp));
		warn(ERROR_STRING);
	} else {
		ifp = img_file_of(rv_name(inp));
		if( ifp == NULL ){
			sprintf(ERROR_STRING,
	"image file struct for rv file %s does not exist!?",rv_name(inp));
			warn(ERROR_STRING);
		} else {
			mvip->mvi_data = ifp;
			// the object has type dimensions set, but
			// not mach dimensions!?
			SET_MOVIE_FRAMES(mvip, OBJ_FRAMES(ifp->if_dp));
			SET_MOVIE_HEIGHT(mvip, OBJ_ROWS(ifp->if_dp));
			SET_MOVIE_WIDTH(mvip, OBJ_COLS(ifp->if_dp));
			SET_MOVIE_DEPTH(mvip, OBJ_COMPS(ifp->if_dp));
		}
	}
}


/* after recording an RV file, automatically create image_file and movie
 * structs to read/playback...
 *
 * Can this be moved to librawvol or libmvimenu?
 */

/*static*/ void _update_movie_database(QSP_ARG_DECL  RV_Inode *inp)
{
	if( is_rv_directory(inp) || is_rv_link(inp) ){
		if( verbose ){
			sprintf(ERROR_STRING,"update_movie_database:  rv inode %s is not a movie",rv_name(inp));
			advise(ERROR_STRING);
		}
		return;
	}

	setup_rv_iofile(inp);		/* open read file	*/
	make_movie_from_inode(inp);	/* make movie struct	*/
}

#endif // MOVED

static void finish_recording(QSP_ARG_DECL  Image_File *ifp)
{
	RV_Inode *inp;

	inp = get_rv_inode(ifp->if_name);
	assert( inp != NULL );

	close_image_file(ifp);		/* close write file	*/

	update_movie_database(inp);

	// do we have error frames for PGR??
	//note_error_frames(inp);
}

// The video reader enques frames to one of the disk writers...
//
// The disk writer sets next_to_enqueue to -1 to indicate it is ready.
//
// Other grabbers consistently use the buffers in a cyclic order,
// but the flycap library doesn't seem to always do this!?

static void *video_reader(void *argp)
{
	int all_ready;
	//int need_oldest;
	int i;
//	int32_t ending_count;
//int last_newest=(-1);
//int expected_newest;
	//struct meteor_counts cnt;
	vr_args *vrap;
	Spink_Cam *scp;
	int32_t n_frames_wanted, orig_n_frames_wanted;
	uint32_t final_size;
	//struct drop_info di;
	//int real_time_ok=1;
	int n_hw_errors=0;
	int n_disks;
	int seconds_before_alarm;
	Image_File *stream_ifp;
	RV_Inode *inp;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *qsp; // needed for thread-safe-query
#endif // THREAD_SAFE_QUERY

	STATUS_HELP(master_codes);
	STATUS_HELP(dw_codes);
	STATUS_HELP(column_doc);

	vrap = (vr_args*) argp;
	scp = vrap->vr_cam_p;

	orig_n_frames_wanted = n_frames_wanted = vrap->vr_n_frames;
	n_disks = vrap->vr_n_disks;
	stream_ifp = vrap->vr_ifp;
#ifdef THREAD_SAFE_QUERY
	qsp = vrap->vr_qsp;
#endif // THREAD_SAFE_QUERY

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

	// metor capture was started here...
	// Now capture is started in main record func, to determine
	// number of buffers...

	starting_count = 0;
	n_enqueued = 0;

	/* schedule a wakeup for 30 seconds after we think we should finish */

#define FRAMES_PER_SECOND	30
#define FIELDS_PER_SECOND	60

#define N_EXTRA_SECONDS		30

	/* this worked well when we were not in field mode... */
	seconds_before_alarm = N_EXTRA_SECONDS +
		(n_frames_wanted+FRAMES_PER_SECOND-1)/FRAMES_PER_SECOND;

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

	while( n_enqueued < n_frames_wanted ){
		Data_Obj *dp;
		int dw_idx;
		int queue_idx;

		// get the next frame
		dp = grab_spink_cam_frame(QSP_ARG  scp);
n_frames_read++;

		assert( dp != NULL );

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

		// We don't need to release PGR frames, but we might check to
		// make sure that both disk writers are keeping up?

		int total_frames_written=0;
		int min_frames_written=n_frames_wanted;
		int max_frames_written=0;
		int min_i;

		for(i=0;i<n_disk_writer_threads;i++){
			int n;
			n = ppi[i].ppi_n_frames_written;
			total_frames_written+=n;
			if(n<min_frames_written){
				min_frames_written=n;
				min_i=i;
			}
			if(n>max_frames_written){
				max_frames_written=n;
			}
		}
		// Do we need to do this test???
		// This difference should really be a function
		// of the number of buffers...  But when the number of
		// buffers is large (e.g. 500), then 10 frames is not too many.
		//
		// This should probably be a soft var - in any case,
		// it should never be larger than the number of buffers minus 2!
#define MAX_DW_ASYNCHRONY	20
		if( (max_frames_written-min_frames_written) > MAX_DW_ASYNCHRONY ){
fprintf(stderr,"video_reader %d:  Disk writer %d not keeping up, %d written, max = %d\n",
n_frames_read,min_i,min_frames_written,max_frames_written);
		}
		if( n_frames_read-total_frames_written >
				(scp->skc_n_buffers-n_disks) ){
fprintf(stderr,"Disk writers not keeping up:  %d frames read, %d written, %d buffers\n",
n_frames_read,total_frames_written,scp->skc_n_buffers);
		}

//expected_newest=newest+1;
//if( expected_newest >= scp->skc_n_buffers ) expected_newest=0;

		newest = scp->skc_newest;

//if( newest != expected_newest )
//fprintf(stderr,"newest = %d, expected %d!? (n_frames_read = %d)\n",
//newest,expected_newest,n_frames_read);

		// Determine which disk writer should receive this frame

		dw_idx = (n_frames_read-1) % n_disk_writer_threads;

if( (ppi[dw_idx].ppi_n_enqueued - ppi[dw_idx].ppi_n_dequeued) >
					(scp->skc_n_buffers/n_disk_writer_threads) )
fprintf(stderr,"video_reader %d:  dw_idx = %d, %d enqueued   %d dequeued\n",
n_frames_read,dw_idx,ppi[dw_idx].ppi_n_enqueued,ppi[dw_idx].ppi_n_dequeued);

		// make sure the disk_writer is ready to enqueue

		queue_idx = ( (n_frames_read-1) / n_disk_writer_threads ) % DW_QUEUE_LEN;
		ppi[dw_idx].ppi_queue[queue_idx] = newest;
		ppi[dw_idx].ppi_queue_wt_idx = queue_idx;
		ppi[dw_idx].ppi_n_enqueued ++;

		n_enqueued ++;

MSTATUS(MS_BOT)

//last_newest = newest;
		/* give up processor */
		YIELD_PROC(100)		// 100 usec sleep if not rt_sched

		/* shut things down if the user has given a halt command */

		if( record_state & RECORD_HALTING ){	/* halt requested */
advise("video_reader HALTING");
			/* we capture n_disks more frames so that none of
			 * the disk writer processes hang waiting for data.
			 * We round up to a multiple of n_disks, so that each
			 * rawvol disk has something...
			 * There is a possibility of a race condition here,
			 * because a disk writer process could get swapped in
			 * before we update the ppi's, so we add another
			 * 4 frames for good measure.  what's 120 ms
			 * between friends, anyway?
			 */

#define ADDITIONAL_FRAMES_PER_DISK	3
#define ADDITIONAL_FRAMES	(n_disks*ADDITIONAL_FRAMES_PER_DISK)

/* NEAREST_MULTIPLE rounds down... */

#define NEAREST_MULTIPLE( num , factor )			\
								\
		( ( ( num + (factor)/2 ) / ( factor ) ) * ( factor ) )

			if( n_frames_wanted >
		NEAREST_MULTIPLE( (n_enqueued+(ADDITIONAL_FRAMES-1)), n_disks ) ){
				int i;
				n_frames_wanted = NEAREST_MULTIPLE( (n_enqueued+(ADDITIONAL_FRAMES-1)), n_disks );
				for(i=0;i<n_disk_writer_threads;i++)
					ppi[i].ppi_n_frames_to_write = n_frames_wanted/n_disk_writer_threads;
advise("ppi_n_frames_to_write changed");
			}
else
advise("ppi_n_frames_to_write NOT changed");
			/* If n_frames_wanted is small, we just let nature take its course */

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
	//meteor_stop_capture(SINGLE_QSP_ARG);
	stop_firewire_capture(QSP_ARG  scp);

	/* check the total number of dropped frames */
	//ending_count = _mm->frames_captured;
//fprintf(stderr,"OOPS - not determining ending count!?\n");

	/* wait for disk writer threads to finish */

	for(i=0;i<n_disk_writer_threads;i++){
#ifdef FOOBAR
		if( unassoc_pids(master_pid,ppi[i].ppi_pid) < 0 )
			error1("error unassociating disk writer process id");
#endif /* FOOBAR */

		if( pthread_join(dw_thr[i],NULL) != 0 ){
			sprintf(ERROR_STRING,"Error joining disk writer thread %d",i);
			warn(ERROR_STRING);
		}
	}

	/* There is some confusion about what the ending count should be...
	 * This used to be the test for real time recording success,
	 * but now we just trust in the driver to tell us this.
	 */

	/*
	if( (ending_count-starting_count) != n_frames_wanted ){
		sprintf(ERROR_STRING,
	"Wanted %d frames, captured %d (%d-%d-1)",n_frames_wanted,
			(ending_count-starting_count)-1,ending_count,starting_count);
		warn(ERROR_STRING);
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

	// in the meteor program, this is where
	// we query the driver for dropped frames & fifo errors
	//
	// Until we know if we dropped frames or not, this is not
	// meaningful...

	/*
	if( real_time_ok ){
		sprintf(ERROR_STRING,"video_reader:  Movie %s recorded successfully in real time.",stream_ifp->if_name);
		advise(ERROR_STRING);
	}
	*/
	if( n_hw_errors > 0 ){
		warn("Some frames may have corrupt data!?");
	}

	if( orig_n_frames_wanted != n_frames_wanted ){
		/* recompute the size of the file in case we were halted */

		final_size = scp->skc_rows * scp->skc_cols * scp->skc_depth ;

		/* we used to divide size by 2 here if in field mode,
		 * but that was a bug, because field mode reduces meteor_rows...
		 *
		 * if( meteor_field_mode ) final_size /= 2;
		 */

		final_size += (BLOCK_SIZE-1);
		final_size /= BLOCK_SIZE;	/* frame size in blocks */
						/* (rounded up to block boundary) */
		final_size *= n_frames_wanted;		/* total blocks */

		rv_truncate( inp, final_size );

		/* rv_truncate corrects the size, but doesn't know about nframes... */

		SET_SHP_FRAMES( rv_movie_shape(inp), n_frames_wanted);

		/* Have to do it in stream_ifp->if_dp too, in order to get a correct
		 * answer from the nframes() expression function...
		 */

		stream_ifp->if_nfrms = n_frames_wanted;
		// used to set stream_ifp->if_dp here also?
	}

	rv_sync(SINGLE_QSP_ARG);

	/* we used to disable real-time scheduling here, but
	 * when video_reader executes as a separate thread there
	 * is no point, because it won't affect the main process!
	 */

	recording_in_process = 0;

	assert( stream_ifp != NULL );

	finish_recording( QSP_ARG  stream_ifp );

	if( mem_locked ){
		if( munlockall() < 0 ){
			tell_sys_error("munlockall");
			warn("Failed to unlock memory!?");
		}
		mem_locked=0;
	}

	return(NULL);

} /* end video_reader */

static void *video_reader_thread(void *argp)
{
#ifdef FOOBAR
	grabber_pid=getpid();

sprintf(ERROR_STRING,"video_reader_thread:  grabber_pid = %d",grabber_pid);
advise(ERROR_STRING);

	if( assoc_pids(master_pid,grabber_pid) < 0 )
		error1("video_reader_thread:  error associating pid");
#endif /* FOOBAR */

	return( video_reader(argp) );
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

#ifdef NOT_USED
/* write 0's to all frames */
static void clear_buffers(SINGLE_QSP_ARG_DECL)
{
	//uint32_t *p,npix;
	//int i;
	//unsigned int j;

	/*
	if( meteor_bytes_per_pixel != DEFAULT_BYTES_PER_PIXEL ) {
		sprintf(ERROR_STRING,
			"clear_buffers:  meteor_bytes_per_pixel = %d (expected %d)",
			meteor_bytes_per_pixel,DEFAULT_BYTES_PER_PIXEL);
		warn(ERROR_STRING);
		return;
	}
	*/
	// BUG - need to check camera bytes per pixel???

	/*
	npix = meteor_columns * meteor_rows ;
	for(i=0;i<scp->skc_n_buffers;i++){
		p = (uint32_t *)(mmbuf + meteor_off.frame_offset[i]);
		for(j=0;j<npix;j++){
			*p++ = 0;
		}
	}
	*/
//fprintf(stderr,"OOPS - not clearing buffers!?\n");
} // clear_buffers
#endif // NOT_USED

static uint32_t get_blocks_per_frame(Spink_Cam *scp)
{
	uint32_t blocks_per_frame, bytes_per_frame;

	bytes_per_frame = scp->skc_cols * scp->skc_rows * scp->skc_depth;
//fprintf(stderr,"bytes_per_frame = %d\n",bytes_per_frame);


#ifdef RECORD_TIMESTAMPS
	if( stamping ) bytes_per_frame += TIMESTAMP_SIZE;
#endif /* RECORD_TIMESTAMPS */
	
	blocks_per_frame = ( bytes_per_frame + BLOCK_SIZE - 1 ) / BLOCK_SIZE;
	return(blocks_per_frame);
}


/* We have 5 processes:  a master process which keeps track of the frame
 * buffer, and n_disks disk writer processes.  The disk writers need to wait
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

void stream_record(QSP_ARG_DECL  Image_File *ifp,int32_t n_frames_wanted,Spink_Cam *scp)
{
	//int32_t npix;
	int fd_arr[MAX_DISKS];
	int n_disks;
	//uint32_t total_blocks;
	uint32_t blocks_per_frame;
	//struct meteor_geomet _geo;
	Shape_Info *shpp;
	RV_Inode *inp;

	if( record_state != NOT_RECORDING ){
		sprintf(ERROR_STRING,
	"stream_record:  can't record file %s until previous record completes",
			ifp->if_name);
		warn(ERROR_STRING);
		return;
	}

	// Before starting, make sure that all of the buffers are aligned
	// for writing with O_DIRECT

	if( check_buffer_alignment(QSP_ARG  scp) < 0 ) return;

	// We need to start the camera to get the number of buffers
	// and their addresses...
//advise("calling start_firewire_capture...");
	start_firewire_capture(QSP_ARG  scp);
//advise("Back from start_firewire_capture.");

// meteor stuff??
//	INSURE_MM("stream_record");


	/* set_rt(); */

	/*
	if( MAX_RINGBUF_FRAMES < scp->skc_n_buffers ){
		warn("Need to recompile mcapt.c with a larger value of MAX_RINGBUF_FRAMES");
		return;
	}
	*/

	//clear_buffers(SINGLE_QSP_ARG);	// not really necessary,
						// but helpful if debugging
						//

	/* allow space for the timestamp, if we are recording timestamps... */

	blocks_per_frame = get_blocks_per_frame(scp);

	n_to_write = blocks_per_frame * BLOCK_SIZE;

	n_to_write /= N_FRAGMENTS;
	assert( (n_to_write % BLOCK_SIZE) == 0 );


	//total_blocks = n_frames_wanted * blocks_per_frame;

	if( FT_CODE(IF_TYPE(ifp)) != IFT_RV ){
		sprintf(ERROR_STRING,
	"stream record:  image file %s (type %s) should be type %s",
			ifp->if_name,
			FT_NAME(IF_TYPE(ifp)),
			FT_NAME(FILETYPE_FOR_CODE(IFT_RV)) );
		warn(ERROR_STRING);
		return;
	}


	inp = (RV_Inode *)ifp->if_hdr_p;
	n_disks = queue_rv_file(inp,fd_arr);
	assert( n_disks > 1 );
	assert( scp->skc_n_buffers > 0 );

	if( scp->skc_n_buffers < (2*n_disks) ){
		sprintf(ERROR_STRING,
	"buffer frames (%d) must be >= 2 x number of disks (%d)",
			scp->skc_n_buffers,n_disks);
		warn(ERROR_STRING);
		return;
	}


	/* set the shape info */
	//meteor_get_geometry(&_geo);

	shpp = ALLOC_SHAPE;
	SET_SHP_FLAGS(shpp,0);

	/* what about the timestamp information that may be tacked onto
	 * the frame?  Does this not get written to the raw volume?
	 */

	SET_SHP_ROWS(shpp, scp->skc_rows );
	SET_SHP_COLS(shpp, scp->skc_cols );
	SET_SHP_COMPS(shpp,scp->skc_depth);

	SET_SHP_FRAMES(shpp,n_frames_wanted);
	SET_SHP_SEQS(shpp, 1);
	SET_SHP_PREC_PTR(shpp,PREC_FOR_CODE(PREC_UBY) );
	SET_SHP_FLAGS(shpp, DT_IMAGE );

	rv_set_shape(ifp->if_name,shpp);
	// does this copy or point?
	// can we free the shape?
	// BUG memory leak

	/* We write an entire frame to each disk in turn... */

	//npix=n_to_write/meteor_bytes_per_pixel;
	//npix=n_to_write/scp->skc_depth;

	/* For the sake of symmetry, we'll create n_disks child threads,
	 * and have the parent wait for them.
	 */

	oldest= 0;
	newest= (-1);	// So disk writers don't think they can go
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

	// stream_record starts the disk writers first, so they
	// will be ready...
	start_dw_threads(QSP_ARG  n_frames_wanted,n_disks,fd_arr,scp);

	vra1.vr_n_frames = n_frames_wanted;
	vra1.vr_n_disks = n_disks;
	vra1.vr_ifp = ifp;
#ifdef THREAD_SAFE_QUERY
	vra1.vr_qsp = qsp;
#endif
	vra1.vr_cam_p = scp;
	// Have we initialized vra1 completely??
	n_frames_read = 0;

	record_state = RECORDING;

	n_stream_frames = n_frames_wanted;	/* remember for later printing... */

	if( mlockall(MCL_FUTURE) < 0 ){
		tell_sys_error("mlockall");
		warn("Failed to lock process memory!?");
	} else {
		mem_locked=1;
	}

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
		ifp->if_nfrms = n_frames_wanted;	/* BUG? is this really what happened? */
	}
} /* end stream_record */

COMMAND_FUNC( flycap_wait_record )
{
#ifdef DEBUG_TIMERS
advise("flycap_wait_record:");
show_tmrs(SINGLE_QSP_ARG);
#endif /* DEBUG_TIMERS */

	if( ! recording_in_process ){
		advise("flycap_wait_record:  no recording currently in process!?");
		return;
	}

	/* BUG make sure this thread is still running! */

#ifdef FOOBAR
	assert( grabber_pid != 0 );

	if( unassoc_pids(master_pid,grabber_pid) < 0 )
		error1("error unassociating grabber pid");
#endif /* FOOBAR */

advise("joining w/ grab thread...");
	if( pthread_join(grab_thr,NULL) != 0 ){
		warn("Error joining video reader thread");
	}
advise("joined!");

#ifdef ALLOW_RT_SCHED
	if( rt_is_on ) rt_sched(QSP_ARG  0);
#endif /* ALLOW_RT_SCHED */

}

COMMAND_FUNC( flycap_halt_record )
{
	if( record_state & RECORDING ){
		/* If we are here, we ought to be able to take it for granted
		 * that we are in fact in async mode...
		 */
		if( record_state & (RECORD_HALTING|RECORD_FINISHING) ){
			sprintf(ERROR_STRING,"flycap_halt_record:  halt already in progress!?");
			warn(ERROR_STRING);
		} else {
			record_state |= RECORD_HALTING;
		}
	} else {
		/* We make this an advisory instead of a warning because
		 * the record might just have finished...
		 */
		sprintf(ERROR_STRING,"flycap_halt_record:  not currently recording!?");
		advise(ERROR_STRING);
		return;
	}

	flycap_wait_record(SINGLE_QSP_ARG);
}

#ifdef RECORD_CAPTURE_COUNT
void dump_ccount(int index,FILE* fp)
{
	int i;

	for(i=0;i<ppi[index].ppi_n_frames_to_write;i++){
		fprintf(fp,"%d\n",ppi[index].ppi_ccount[i]);
	}
	fclose(fp);
}
#endif	/* RECORD_CAPTURE_COUNT */

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

Image_File * _get_file_for_recording(QSP_ARG_DECL  const char *name,
		int n_frames_wanted,Spink_Cam *scp)
{
	Image_File *ifp;
	long n_blocks;

	ifp = img_file_of(name);

	if( ifp != NULL ){
		RV_Inode *inp;
		// is the existing file an RV file?
		if( IF_TYPE_CODE(ifp) != IFT_RV ){
			sprintf(ERROR_STRING,
	"Existing file %s is not a raw volume file, not clobbering.",
				IF_NAME(ifp));
			warn(ERROR_STRING);
			return NULL;
		}

		inp = (RV_Inode *) ifp->if_hdr_p;

		if( ! rv_access_allowed(QSP_ARG  inp) ){
			sprintf(ERROR_STRING,
	"No permission to clobber existing raw volume file %s.",
				IF_NAME(ifp));
			warn(ERROR_STRING);
			return NULL;
		}

		if( verbose ){
			sprintf(ERROR_STRING,"Clobbering existing image file %s",name);
			advise(ERROR_STRING);
		}

		image_file_clobber(1);	/* enable clobbering - not necessary !? */
		// Now we are pretty sure there will be no permission errors
		delete_image_file(ifp);
	}

	set_filetype(FILETYPE_FOR_CODE(IFT_RV));
	ifp = write_image_file(name,n_frames_wanted);	/* nf stored in if_frms_to_write */

	// BUG?  where do we set the shape of ifp->if_dp???

	/* sets nframes in ifp, but doesn't allocate rv blocks properly...
	 * (WHY NOT??? maybe because the image dimensions are not known???)
	 * we could know them, however, because at this point the geometry is set.
	 */

	if( ifp == NULL ){
		sprintf(ERROR_STRING,"Error creating movie file %s",name);
		warn(ERROR_STRING);
		return NULL;	// BUG clean up
	}

	n_blocks = rv_frames_to_allocate(n_frames_wanted) * get_blocks_per_frame(scp);

	/* n_blocks is the total number of blocks, not the number per disk(?) */

	if( rv_realloc(name,n_blocks) < 0 ){
		sprintf(ERROR_STRING,"error reallocating %ld blocks for rv file %s",
			n_blocks,name);
		warn(ERROR_STRING);
		return NULL;	// BUG clean up
	}
	return ifp;
}

#endif /* HAVE_LIBSPINNAKER */

