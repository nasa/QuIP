/*
 * ieee1394io.c -- asynchronously grabbing DV data
 *
 */

#ifdef HAVE_LIBDV

#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
//#include <pthread.h>

#include <string.h>		// memcpy

#include <libraw1394/raw1394.h>
#include <libraw1394/csr.h>

#include <sys/ioctl.h>
#include <sys/mman.h>
/* #include <libdv/dv1394.h> */ /* jbm disabled for centOS compile */
#include "dv1394.h"			/* from kernel src tree, copied here */

#include "frame.h"
//#include "dvmenu.h"
//#include "myerror.h"
#include "debug.h"

#define RAW_BUF_SIZE	(10240)

/* global variables shared with the main thread */

//extern volatile bool	g_reader_active;
//extern bool	g_buffer_underrun;
//extern int	g_card;
//extern int	g_channel;
//extern pthread_mutex_t	g_mutex;
//extern char *g_dv1394;

/* private global variables */

//int	g_testdata;
Frame	*g_current_frame;
unsigned char *g_frame;
static unsigned char *g_dv1394_map;

static int n_calls=0;

static int avi_iso_handler(raw1394handle_t handle, int channel, size_t length,
			quadlet_t *data)
{
	/* The PAL/NTSC DV data has the following format:

		- packets of 496 bytes length
		- packets of 16 bytes length.

		The long packets contain the actual video and audio
		contents that goes into the AVI file. The contents of the
		other packets and the meaning of the header is unknown. If
		you know something about it, please let me know.

		The actual DV data starts at quadlet 4 of the long packet,
		so we have 480 bytes of DV data per packet.  For PAL, each
		rame is made out of 300 packets and there are 25 frames
		per second.  That is 144000 bytes per frame and 3600000
		bytes per second.  For NTSC we have 250 packages per frame
		and 30 frames per second.  That is 120000 bytes per frame
		and 3600000 bytes per second too.

		We also attempt to detect incomplete frames. The idea is:
		If we hit the begin of frame indicator and this is not the
		very first packet for this frame, then at least one packed
		has been dropped by the driver. This does not guarantee
		that no incomplete frames are saved, because if we miss the
		frame header of the next frame, we can´t tell whether the
		last one is incomplete.  */


	n_calls++;
	if (length > 16) {

		unsigned char *p = (unsigned char*) & data[3];
		int section_type = p[0] >> 5;		/* section type is in bits 5 - 7 */
		int dif_sequence = p[1] >> 4;		/* dif sequence number is in bits 4 - 7 */
		int dif_block = p[2];

		/* if we are at the beginning of a frame, we put the previous frame in our output_queue.
			Then we try to get an unused frame_buffer from the buffer_queue for the current frame.
			We must lock the queues because they are shared between this thread and the main thread. */

		if (section_type == 0 && dif_sequence == 0) {
			pthread_mutex_lock(&g_mutex);
			if (g_current_frame != NULL) {
				queue_push_back(g_output_queue,g_current_frame);
				g_current_frame = NULL;
				// printf("reader > out: buffer %d, output %d\n", g_buffer_queue.size(), g_output_queue.size());
				// fflush(stdout);
			}


			if( queue_size(g_buffer_queue) > 0) {
				g_current_frame = queue_front(g_buffer_queue);
				g_current_frame->frm_bytesInFrame = 0;
				queue_pop_front(g_buffer_queue);
				// printf("reader < buf: buffer %d, output %d\n",g_buffer_queue.size(), g_output_queue.size());
				// fflush(stdout);
			}
			else {
NWARN("buffer underrun...");
				g_buffer_underrun = TRUE;
			}
			pthread_mutex_unlock(&g_mutex);
		}

		if (g_current_frame != NULL) {

			switch (section_type) {
			case 0:		// 1 Header block
if( verbose ) advise("copying header block");
				memcpy(g_current_frame->frm_data + dif_sequence * 150 * 80, p, 480);
				break;

			case 1:		// 2 Subcode blocks
if( verbose ) advise("copying subcode blocks");
				memcpy(g_current_frame->frm_data + dif_sequence * 150 * 80 + (1 + dif_block) * 80, p, 480);
				break;

			case 2:		// 3 VAUX blocks
if( verbose ) advise("copying VAUX blocks");
				memcpy(g_current_frame->frm_data + dif_sequence * 150 * 80 + (3 + dif_block) * 80, p, 480);
				break;

			case 3:		// 9 Audio blocks interleaved with video
if( verbose ) advise("copying interleaved audio blocks");
				memcpy(g_current_frame->frm_data + dif_sequence * 150 * 80 + (6 + dif_block * 16) * 80, p, 480);
				break;

			case 4:		// 135 Video blocks interleaved with audio
if( verbose ) advise("copying interleaved video blocks");
				memcpy(g_current_frame->frm_data + dif_sequence * 150 * 80 + (7 + (dif_block / 15) + dif_block) * 80, p, 480);
				break;

			default:		// we can´t handle any other data
NWARN("unknown block type");
				break;
			}
			g_current_frame->frm_bytesInFrame += 480;
		}
	}
	return 0;
}


static int my_reset_handler(raw1394handle_t handle, unsigned int generation)
{
	static int i = 0;

	fprintf( stderr, "reset %d\n", i++);
	if (i == 100){
		g_reader_active = FALSE;
	}
	return 0;
}


static raw1394handle_t open_1394_driver(int channel, iso_handler_t handler)
{
	int numcards;
	struct raw1394_portinfo g_pinf[16];
	iso_handler_t	g_oldhandler;
	raw1394handle_t handle;
	int dv1394_fd = -1;
	int n_frames = DV1394_MAX_FRAMES/4;

	struct dv1394_init init = {
		DV1394_API_VERSION, g_channel, n_frames, DV1394_PAL, 0, 0, 0
	};

	if (g_dv1394 != NULL) {
		dv1394_fd = open(g_dv1394, O_RDWR);
		if(dv1394_fd == -1) {
			perror("dv1394 open");
			exit(EXIT_FAILURE);
		}

		if(ioctl(dv1394_fd, /* DV1394_INIT */ DV1394_IOC_INIT, &init)) {
			perror("dv1394 INIT ioctl");
			close(dv1394_fd);
			exit(EXIT_FAILURE);
		}

		g_dv1394_map = (unsigned char *) mmap( NULL, DV1394_PAL_FRAME_SIZE * n_frames,
				PROT_READ|PROT_WRITE, MAP_SHARED, dv1394_fd, 0);
		if(g_dv1394_map == MAP_FAILED) {
			perror("mmap frame buffers");
			close(dv1394_fd);
			exit(EXIT_FAILURE);
		}


		if(ioctl(dv1394_fd, DV1394_IOC_START_RECEIVE, NULL)) {
			perror("dv1394 IOC_START_RECEIVE ioctl");
			close(dv1394_fd);
			exit(EXIT_FAILURE);
		}

		// BUG - why return a file descriptor here???
		return (raw1394handle_t) ((long)dv1394_fd);
	}

	if (!(handle = raw1394_new_handle())) {
		perror("raw1394 - couldn't get handle");
		fprintf( stderr, "This error usually means that the ieee1394 driver is not loaded or that /dev/raw1394 does not exist.\n");
		exit(EXIT_FAILURE);
	}

	if ((numcards = raw1394_get_port_info(handle, g_pinf, 16)) < 0) {
		perror("raw1394 - couldn't get card info");
		exit(EXIT_FAILURE);
	}

	if (raw1394_set_port(handle, g_card) < 0) {
		perror("raw1394 - couldn't set port");
		exit(EXIT_FAILURE);
	}

	g_oldhandler = raw1394_set_iso_handler(handle, g_channel, handler);
	/* raw1394_set_tag_handler(handle, my_tag_handler);	*/
	raw1394_set_bus_reset_handler(handle, my_reset_handler);

	/* Starting iso receive */

	if (raw1394_start_iso_rcv(handle, channel) < 0) {
		perror("raw1394 - couldn't start iso receive");
		exit(EXIT_FAILURE);
	}
	return handle;
}


static void close_1394_driver(int channel, raw1394handle_t handle)
{
	if (g_dv1394 != NULL)
	{
		if (g_dv1394_map != NULL)
			munmap(g_dv1394_map, DV1394_PAL_FRAME_SIZE * DV1394_MAX_FRAMES/4);
		// BUG before we case the file descriptor to a handle...
		close((int) handle);	// should we do this here???
	}
	else
	{
		raw1394_stop_iso_rcv(handle, channel);
			raw1394_destroy_handle(handle);
	}
}


/*

void open_testdata()
{
	g_testdata = open("testdata.raw", O_RDONLY);
	assert(g_testdata != -1);
}

void read_testdata()
{
	quadlet_t	data[RAW_BUF_SIZE];
	size_t	length;
	size_t	count;

	pthread_mutex_lock(&g_mutex);
	int size = queue_size(g_buffer_queue);
	pthread_mutex_unlock(&g_mutex);

	if (size > 0) {
		count = read(g_testdata, &length, sizeof(length));
		if (count == sizeof(length)) {
		count = read(g_testdata, data, length);
		if (count == length) {
			avi_iso_handler(NULL, 0, length, data);
		} else
			g_reader_active = FALSE;
	} else
		g_reader_active = FALSE;
	}
}

void close_testdata()
{
	close(g_testdata);
}

*/




static bool dv1394_loop_iterate(int handle)
{
	struct dv1394_status dvst;
	int n_frames = DV1394_MAX_FRAMES/4;
	unsigned int i;

	if(ioctl(handle, DV1394_IOC_WAIT_FRAMES, n_frames - 1))
	{
		perror("error: ioctl IOC_WAIT_FRAMES");
		return FALSE;
	}

	if (ioctl(handle, DV1394_IOC_GET_STATUS, &dvst))
	{
		perror("ioctl IOC_GET_STATUS");
		return FALSE;
	}

	if (dvst.dropped_frames > 0)
	{
		fprintf( stderr, "dv1394 reported %d dropped frames. Stopping.\n", dvst.dropped_frames);
		return FALSE;
	}

	for (i = 0; i < dvst.n_clear_frames; i++)
	{
		pthread_mutex_lock(&g_mutex);
		if (g_current_frame != NULL)
		{
			queue_push_back(g_output_queue,g_current_frame);
			g_current_frame = NULL;
			// printf("reader > out: buffer %d, output %d\n", g_buffer_queue.size(), g_output_queue.size());
			// fflush(stdout);
		}


		if (queue_size(g_buffer_queue) > 0)
		{
			g_current_frame = queue_front(g_buffer_queue);
			g_current_frame->frm_bytesInFrame = 0;
			queue_pop_front(g_buffer_queue);
			// printf("reader < buf: buffer %d, output %d\n",g_buffer_queue.size(), g_output_queue.size());
			// fflush(stdout);
		}
		else
			g_buffer_underrun = TRUE;
		pthread_mutex_unlock(&g_mutex);

		if ( g_current_frame != NULL )
		{
			memcpy( g_current_frame->frm_data,
				(g_dv1394_map + (dvst.first_clear_frame * DV1394_PAL_FRAME_SIZE)),
				DV1394_PAL_FRAME_SIZE );
			g_current_frame->frm_bytesInFrame = GetFrameSize(g_current_frame);
		}


		if(ioctl(handle, DV1394_IOC_RECEIVE_FRAMES, 1))
		{
			perror("error: ioctl IOC_RECEIVE_FRAMES");
			return FALSE;
		}

		if (ioctl(handle, DV1394_IOC_GET_STATUS, &dvst))
		{
			perror("ioctl IOC_GET_STATUS");
			return FALSE;
		}

		if (dvst.dropped_frames > 0)
		{
			fprintf( stderr, "dv1394 reported %d dropped frames. Stopping.\n", dvst.dropped_frames);
			return FALSE;
		}

	}
	return TRUE;
}

/* read_frames() is the entry point of the reader thread
 */

void* read_frames(void* arg)
{
	raw1394handle_t handle;
	sigset_t sigset;

	sigfillset(&sigset);
	pthread_sigmask(SIG_BLOCK, &sigset, 0);

	g_reader_active = TRUE;
	g_current_frame = NULL;
	handle = NULL;

	handle = open_1394_driver(g_channel, avi_iso_handler);

	while (g_reader_active == TRUE) {

		/* Poll the 1394 interface */

		if (g_dv1394 != NULL){
			g_reader_active = dv1394_loop_iterate( (int) handle);
		}
		else {
//advise("calling raw1394_loop_iterate");
			raw1394_loop_iterate(handle);
		}
	}
	close_1394_driver(g_channel, handle);

	/* if we are still using a frame, put it back */

	if (g_current_frame != NULL) {
		pthread_mutex_lock(&g_mutex);
		queue_push_back(g_buffer_queue,g_current_frame);
		pthread_mutex_unlock(&g_mutex);
	}
	return NULL;
}

/* TEST format functions */

extern char *g_dst_file_name;
extern int g_frame_count;
extern volatile int	g_alldone;
extern int usage();

#ifdef FOOBAR
static int test_iso_handler(raw1394handle_t handle, int channel, size_t length,
					quadlet_t *data)
{
	if (length < RAW_BUF_SIZE) {
		*(int*)g_frame = length;
		memcpy(g_frame + 4, data, length);
		}
	return 0;
}

int capture_test()
{
	int	frames_read;
	int	length;
	int	dst_fd;
	raw1394handle_t handle;

	if ((dst_fd = open(g_dst_file_name, O_WRONLY | O_CREAT, 00644)) < 0) {
		perror("problem with output file");
		usage();
		exit(1);
	}

	g_frame = (unsigned char*)malloc(RAW_BUF_SIZE);
	if( g_frame == NULL ) error1("capture_test:  unable to allocate frame");

	handle = open_1394_driver(g_channel, test_iso_handler);

	frames_read = 0;
	while ((!g_alldone) && (frames_read < g_frame_count)) {
		raw1394_loop_iterate(handle);
		length = *(int*)g_frame;
		write(dst_fd, g_frame, length + 4);
		frames_read++;
	}
	close_1394_driver(g_channel, handle);
	free(g_frame);
	close(dst_fd);

	return 0;
}
#endif/* FOOBAR */

#endif /* HAVE_LIBDV */


