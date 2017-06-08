#include "quip_config.h"

// C includes

#include <stdio.h>

#ifdef HAVE_ASSERT_H
#include <assert.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>		// memcpy()
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>		// ioctl()
#endif

#ifdef HAVE_LIBDV_DV1394_H
#include <libdv/dv1394.h>
#endif


// local includes

#include "frame.h"
#include "quip_prot.h"
#include "data_obj.h"
#include "img_file.h"
//#include "filehandler.h"

#define RAW_BUF_SIZE	(10240)

bool	g_buffer_underrun;
volatile int	g_alldone;
char	*g_dst_file_name;

int	g_autosplit;
int	g_timestamp;
int	g_card;
int	g_channel;
int	g_file_format;
bool	g_open_dml;
int	g_frame_count;
int	g_frame_every;
int	g_progress;
int	g_max_file_size;
char	*g_dv1394 = NULL;
int	g_jpeg_quality;
bool	g_jpeg_deinterlace;
int	g_jpeg_width;
int	g_jpeg_height;
bool	g_jpeg_overwrite;

//volatile bool	g_reader_active;
bool	g_reader_active;
Frame	*g_current_frame=NULL;

pthread_mutex_t g_mutex;
pthread_t	g_thread;

List *g_output_queue=NULL;
List *g_buffer_queue=NULL;
List *grab_lp=NULL;

static void create_frames(void)
{
	Frame *frmp;
	int i;

	if( g_buffer_queue == NULL )
		g_buffer_queue = new_list();

	i=100;
	while(i--){
		frmp = new_frame();
		// BUG initialize fields here
		queue_push_back(g_buffer_queue,frmp);
	}
}

/* read_frames appears in the audio support library??? */

static void setup_capture()
{
	//int incomplete_frames = 0;
	//Frame *frmp = NULL;

	g_alldone = FALSE;
	g_buffer_underrun = FALSE;

	/*
	for (int i = 0; i < 100; ++i) {
		fail_null(frmp = new Frame);
		g_buffer_queue.push_back(frmp);
	}
	*/
	create_frames();

//advise("initializing mutex...");
	pthread_mutex_init(&g_mutex, NULL);
	g_reader_active = TRUE;
//advise("creating read_frames thread...");
	pthread_create(&g_thread, NULL, read_frames, NULL);
}

static void DoneWithFrame(Frame *frame)
{

	pthread_mutex_lock(&g_mutex);
//advise("DoneWithFrame:  pushing back on buffer queue");

	// old code, before brollyx change
	queue_push_back(g_buffer_queue,frame);

	// printf("writer > buf: buffer %d, output %d\n",g_buffer_queue.size(), g_output_queue.size());
	// fflush(stdout);
	pthread_mutex_unlock(&g_mutex);
}

static Frame* GetNextFrame(void)
{
	Frame *	frmp = NULL;

	while (frmp == NULL) {

//advise("GetNextFrame:  locking mutex");
		pthread_mutex_lock(&g_mutex);

		// printf("  BEFORE   buffer: %d, output: %d\n",queue_size(g_buffer_queue),queue_size(g_output_queue));

		if (queue_size(g_output_queue) > 0) {
			Node *np;
//advise("GetNextFrame:  queue size is greater than 0");

			// Each new frame from the buffer_queue pushes back the output_queue,
			// and is thus placed at the head of the output_queue.
			// So we will find the oldest frame at the tail of the output_queue. - brollyx
			//np = remHead(g_output_queue);
			np = remTail(g_output_queue);

			//np = QLIST_HEAD(g_output_queue);
			frmp = (Frame *)np->n_data;
			//printf("writer < out: buffer %d, output %d\n",queue_size(g_buffer_queue), queue_size(g_output_queue));
			// fflush(stdout);
		}

//advise("GetNextFrame:  unlocking mutex");
		pthread_mutex_unlock(&g_mutex);

		if (!g_reader_active)
			break;

		if (frmp == NULL){
//advise("GetNextFrame:  frmp is null, sleeping");
			usleep (100000);
		}

	}
	if (frmp != NULL)
		ExtractHeader(frmp);
	return frmp;
}

static void queue_add_elt( List *lp, void *obj )
{
	Node *np;

#ifdef CAUTIOUS
	if( lp == NULL )
		NERROR1("CAUTIOUS:  Oops - missing queue");
#endif /* CAUTIOUS */

	np = mk_node(obj);
	addTail(lp,np);
}


static void IncreaseBufferQueue(void)
{
	Frame *frmp;

	pthread_mutex_lock(&g_mutex);

        // To increase the length of the buffer_queue without pushing it back, we need to use addTail()
        // with a new_frame. -brollyx
	frmp = new_frame();
	queue_add_elt(g_buffer_queue,frmp);

	pthread_mutex_unlock(&g_mutex);
}

static void dv_grab(QSP_ARG_DECL  int n)
{
	Frame *frmp = NULL;

	if( grab_lp == NULL ) grab_lp = new_list();

//sprintf(DEFAULT_ERROR_STRING,"Setting up to capture %d frames",n);
//advise(DEFAULT_ERROR_STRING);
	setup_capture();
	while( (n > 0) && (g_reader_active == TRUE) ){
//advise("getting next frame");
//	  if  ( !queue_size(g_buffer_queue) ) {
//		  create_frames();
//		  printf("Created 100 frames in buffer queue. new length: %d\n",queue_size(g_buffer_queue));
//		}


		frmp = GetNextFrame();
		if (frmp == NULL){
advise("no next frame!?");
			break;
		}
//sprintf(DEFAULT_ERROR_STRING,"after GetNextFrame, reader_active == %d",g_reader_active);
//advise(DEFAULT_ERROR_STRING);
		if (IsComplete(frmp)) {
			Node *np;
//advise("complete frame received");
			n--;
			np=mk_node(frmp);
			// "We" want the first element of grab_lp to be the first temporal 
			// frame. Then for the extracting phase, extract(1) would be
			// extracting the first frame. That's why the newest/latest frame
			// should be added at the tail of the grab_lp
			//addHead(grab_lp,np);
			addTail(grab_lp,np);

			//break;
		} else {
advise("incomplete frame, done...");

//                        Commented out by brollyx 
                        DoneWithFrame(frmp);

		}

		// When GetNextFrame() is called, if the output_queue is empty, several frames (between 1 
		// and 6 usually) from the buffer_queue will be moved to the output_queue, resulting in a
		// decrease of the buffer_queue length.
		// So we need to add some new elements to the buffer_queue (otherwise it will become empty 
		// once 100 frames have been read). The function IncreaseBufferQueue() actually only adds
		// 1 frame to the buffer_queue. But it's fine, as it will be called several times while
		// the output_buffer is not empty. - brollyx
		IncreaseBufferQueue();

//sprintf(DEFAULT_ERROR_STRING,"bottom of loop, reader_active == %d",g_reader_active);
//advise(DEFAULT_ERROR_STRING);
	}
	if( n == 0 ) {
		g_reader_active=FALSE;
		g_alldone = TRUE;
	} else {
		sprintf(DEFAULT_ERROR_STRING,"dv_grab:  reader_active set FALSE, with n = %d",n);
		advise(DEFAULT_ERROR_STRING);
	}
	/* BUG need to stop reader here */
}

void queue_push_back( List *lp, void *obj )
{
	Node *np;

#ifdef CAUTIOUS
	if( lp == NULL )
		NERROR1("CAUTIOUS:  Oops - missing queue");
#endif /* CAUTIOUS */

	np = mk_node(obj);
	addHead(lp,np);
}


void *queue_front(List *lp)
{
	if( lp == NULL ) return NULL;
	if( QLIST_HEAD(lp) == NO_NODE ) return NULL;
	return( QLIST_HEAD(lp)->n_data );
}

void queue_pop_front ( List *lp )
{
	if( lp == NULL ) return;
	if( QLIST_HEAD(lp) == NO_NODE ) return;
	remHead(lp);
}

int queue_size(List *lp)
{
	if( lp == NULL ) return 0;
	return eltcount(lp);
}



static COMMAND_FUNC( do_dv_capture )
{
	Frame *frmp = NULL;
	int incomplete_frames = 0;

	setup_capture();

advise("watching reader_active flag...");
	while (g_reader_active == TRUE) {
advise("getting next frame");
		frmp = GetNextFrame();
		if (frmp == NULL){
advise("no next frame!?");
			break;
		}
		if (IsComplete(frmp)) {
advise("complete frame received");
			break;
		} else {
advise("incomplete frame, done...");
			DoneWithFrame(frmp);
		}
	}

advise("reader is not active...");
	/* report a buffer underrun. */

	if (g_buffer_underrun == TRUE) {

		TimeCode timeCode;
		struct tm recDate;

		GetTimeCode(frmp,&timeCode);
		GetRecordingDateTime(frmp,&recDate);
		sprintf( ERROR_STRING, "buffer underrun near: timecode %2.2d:%2.2d:%2.2d.%2.2d date %4.4d.%2.2d.%2.2d %2.2d:%2.2d:%2.2d\n",
			timeCode.hour, timeCode.min, timeCode.sec, timeCode.frame,
			recDate.tm_year + 1900, recDate.tm_mon + 1, recDate.tm_mday,
			recDate.tm_hour, recDate.tm_min, recDate.tm_sec);
		NWARN(ERROR_STRING);
		NWARN("This error means that the frames could not be written fast enough.\n");
		g_buffer_underrun = FALSE;
	}

	/* report a broken frame. */

	if (IsComplete(frmp) == FALSE) {

		TimeCode timeCode;
		struct tm recDate;

		GetTimeCode(frmp,&timeCode);
		GetRecordingDateTime(frmp,&recDate);
		sprintf( ERROR_STRING, "frame dropped: timecode %2.2d:%2.2d:%2.2d.%2.2d date %4.4d.%2.2d.%2.2d %2.2d:%2.2d:%2.2d\n",
			timeCode.hour, timeCode.min, timeCode.sec, timeCode.frame,
			recDate.tm_year + 1900, recDate.tm_mon + 1, recDate.tm_mday,
			recDate.tm_hour, recDate.tm_min, recDate.tm_sec);
		NWARN(ERROR_STRING);
		NWARN( "This error means that the ieee1394 driver received an incomplete frame.\n");
			incomplete_frames++;
	} else {
		/* Here is were we might want to write the frame - jbm */
		//g_file->WriteFrame(*frmp);
	}

	DoneWithFrame(frmp);
}

static void set_defaults(void)
{
	g_autosplit = FALSE;
	g_timestamp = FALSE;
	g_card = 0;
	g_channel = 63;
	g_open_dml = FALSE;
	g_frame_count = 100000;
	g_frame_every = 1;
	g_dst_file_name = NULL;
	g_progress = FALSE;
	g_max_file_size = 1024;
	g_jpeg_quality = 75;
	g_jpeg_deinterlace = FALSE;
	g_jpeg_width = -1;
	g_jpeg_height = -1;
	g_jpeg_overwrite = FALSE;
}

#ifdef NOT_USED
void signal_handler(int sig)
{
	/* replace this signal handler with the default (which aborts) */

	signal(SIGINT, SIG_IGN);
	signal(SIGHUP, SIG_IGN);

	/* setting these variables will let us fall out of the main loop */

advise("signal handler stopping reader");
	g_reader_active = FALSE;
	g_alldone = TRUE;
}
#endif /* NOT_USED */

static COMMAND_FUNC( do_dv_grab )
{
	int n;

	n=HOW_MANY("number of frames to grab");

	if( n<=0 ){
		WARN("number of frames must be positive");
		return;
	}
	dv_grab(QSP_ARG  n);

	sprintf(msg_str,"%d frames grabbed",eltcount(grab_lp));
	prt_msg(msg_str);
}

static COMMAND_FUNC( do_dv_info )
{
	int n;

	n=eltcount(g_output_queue);
	sprintf(msg_str,"Output queue has %d frames",n);
	prt_msg(msg_str);

	n=eltcount(grab_lp);
	sprintf(msg_str,"Grab list has %d frames",n);
	prt_msg(msg_str);
}

static COMMAND_FUNC( do_dv_extract )
{
	Data_Obj *dp;
	int n;
	Node *np;
	Frame *frmp;

	dp=PICK_OBJ("destination data object");
	n=HOW_MANY("index of stored frame");
	np = nth_elt(grab_lp,n);
	if( dp == NO_OBJ ) return;
	if( np == NO_NODE ) return;
	frmp = (Frame *)np->n_data;
	// BUG check the size, type, contiguity here
//sprintf(ERROR_STRING,"do_dv_extract:  frame %d, frmp = 0x%lx",n,(u_long)frmp);
//advise(ERROR_STRING);

	ExtractHeader(frmp);
	ExtractRGB(frmp,OBJ_DATA_PTR(dp));
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(dv_menu,s,f,h)

MENU_BEGIN(dv)
ADD_CMD( capture,	do_dv_capture,	continuously capture frames )
ADD_CMD( grab,		do_dv_grab,	grab N frames )
ADD_CMD( info,		do_dv_info,	give info about the output queue )
ADD_CMD( extract,	do_dv_extract,	convert stored DV frame to a memory object )
MENU_END(dv)

static int dv_inited=0;

COMMAND_FUNC( do_dv_menu )
{
	if( !dv_inited ){
		set_defaults();
		g_output_queue = new_list();
		dv_inited=1;
	}
	PUSH_MENU(dv);
}

