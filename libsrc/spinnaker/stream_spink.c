
#include "quip_config.h"

#ifdef HAVE_LIBSPINNAKER

/* Stream video to disk.
 *
 * Now euler has 3 drives, and we'd like to stream 2 BlackFly cameras, each at 170 fps
 * 1.3 MB per frame...
 * 2 * 1.3 * 170 = 442 MB/sec - required disk write speed is 150 MB/sec!?
 *
 * Older implementation used video reader and disk writer threads...  with image
 * events, we can let the event handlers to the disk writing, and no longer need
 * to explicitly read.  However, we have to figure out which disk to write to, and
 * make sure that we don't write until the previous write has finished.  It would be
 * nice if we could do this without a mutex, but I'm not sure that that is possible...
 *
 * We get the index of the disk as the camera index plus the frame index times the number of cameras,
 * modulo n_disks.
 *
 * Example:  2 cameras, 3 disks:
 * For this to work, the time to write a frame must be less than 3/2 the frame time.  So
 * the time to record 6 frames (3 from each camera) will be 6*4/3 = 8 frames, = 2.7 per disk
 *
 *	Frame	Camera	Index	Disk		starts at	should finish at
 *	0	0	0	0		0		1.5
 *	0	1	1	1		0		1.5
 *	1	0	2	2		1		2.5
 *	1	1	3	0 		1.5		3
 *	2	0	4	1 		2		3.5
 *	2	1	5	2		2.5		4
 *	3	0	6	0		3		4.5
 *	3	1	7	1		3.5		5
 *	4	0	8	2		4		5.5
 *	4	1	9	0		4.5		6
 *	5	0	10	1		5		6.5
 *	5	1	11	2		5.5		7
 *	6	0	12	0		6		7.5
 *	6	1	13	1		6.5		8
 *	...
 *
 * We can look at it from the point of view of the disks:
 *
 * time		0	0.5	1.0	1.5	2.0	2.5	3.0	3.5	4.0	4.5	5.0	5.5
 * disk0	0 c0_f0----------------- 3 c1_f1---------------- 6 c0_f3------------------
 * disk1	1 c1_f0-----------------        4 c0_f2----------------- 7 c1_f3-----------------	    ...
 * disk2                        2 c0_f1------------------ 5 c1_f2---------------- 8 c0_f4----------------
 *
 * If things run smoothly, at any given time we will be using 4 buffers, 3 writing to disk, and one waiting for its
 * disk to become available.
 *
 * For each disk, we create a variable to hold the index of the next frame/cam index allowed to access the disk.
 * The handler computes its own index, then checks the variable.  If it is a match, then it can take the file
 * descriptor and write the data.  When the write finishes, it then advances the index var.  If the var holds a smaller
 * value, then it needs to wait, either polling or sleeping or a combination of the two.  The sleep time can be
 * adjusted based on what happens...  The var should never hold a larger value, that is some kind of internal error.
 */

#include <unistd.h>	// write
#include <sys/mman.h>	// mlockall
#include "quip_prot.h"
#include "rv_api.h"
#include "spink.h"
#include "gmovie.h"
#include "fio_api.h"

#ifdef ALLOW_RT_SCHED

#include "rt_sched.h"
#define YIELD_PROC(time)	{ if( rt_is_on ) sched_yield(); else usleep(time); }

#else /* ! ALLOW_RT_SCHED */

#define YIELD_PROC(time)	usleep(time);

#endif /* ALLOW_RT_SCHED */

typedef struct recording_disk_info {
	int		rdi_fd;		// file descriptor
	int		rdi_idx;	// the disk index
	uint64_t	rdi_next;	// frame/cam index next to write
	uint64_t	rdi_last;	// frame/cam index next to write
	int		rdi_state;
	uint64_t	rdi_n_stored;
} Recording_Disk_Info;

// flag bits
#define DISK_IDLE	1
#define DISK_WRITING	2


static Recording_Disk_Info disk_info_tbl[MAX_DISKS];	// MAX_DISKS is defined in rv_api.h
static int n_disks=0;
static int n_cameras=0;
static int n_stream_frames=0;

typedef enum {
	IDLE,
	RECORDING,
	RECORD_HALTING
} Recording_State;

static int record_state=IDLE;
static int mem_locked=0;

static void init_recording_disk_info(void)
{
	int i;

	for(i=0;i<MAX_DISKS;i++){
		disk_info_tbl[i].rdi_fd = (-1);
		disk_info_tbl[i].rdi_idx = (-1);
		disk_info_tbl[i].rdi_next = (uint64_t) (-1);
		disk_info_tbl[i].rdi_state = DISK_IDLE;
		disk_info_tbl[i].rdi_n_stored = 0;
	}
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

static void store_grab_time(uint64_t this_idx)
{
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
}

#endif /* RECORD_TIMESTAMPS */

static void write_image_to_disk(spinImage hImg, void *vp)
{
	Image_Event_Info *inf_p;
	Spink_Cam *skc_p;
	uint64_t this_idx;
	int64_t frame_id;
	int disk_idx;
	void *data_ptr;
	ssize_t n_written,n_to_write;
	Recording_Disk_Info *rdi_p;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *qsp;
#endif // THREAD_SAFE_QUERY

	inf_p = (Image_Event_Info *) vp;

	assert(inf_p!=NULL);
	skc_p = inf_p->ei_skc_p;
	assert(skc_p!=NULL);
#ifdef THREAD_SAFE_QUERY
	qsp = inf_p->ei_qsp;
	assert(qsp!=NULL);
#endif // THREAD_SAFE_QUERY

	// Get the index of this frame
	// Fetch FrameID
	if( get_image_chunk_int(hImg,"ChunkFrameID",&frame_id) < 0 )
		warn("write_image_to_disk:  error fetching frame_id!?");

	// We need to determine whether frame_id's start at zero each time we start capturing...
fprintf(stderr,"write_image_to_disk:  frame %ld received (%s)\n",frame_id,skc_p->skc_name);

	// BUG - using skc_sys_idx assumes that we are recording from all cameras!
	this_idx = frame_id * n_cameras + skc_p->skc_sys_idx;
	disk_idx = this_idx % n_disks;
	rdi_p = (& disk_info_tbl[disk_idx]);

if( rdi_p->rdi_state == DISK_WRITING )
fprintf(stderr,"write_image_to_disk:  disk %d is already writing...\n",disk_idx);
if( rdi_p->rdi_next != this_idx )
fprintf(stderr,"write_image_to_disk:  next frame is %ld, will wait for %ld...\n",rdi_p->rdi_next,this_idx);
	while( rdi_p->rdi_state != DISK_IDLE && rdi_p->rdi_next != this_idx ){
		usleep(100000);		// sleep 100 usecs
	}

	rdi_p->rdi_state = DISK_WRITING;

	if( get_image_data(hImg,&data_ptr) < 0 )
		error1("write_image_to_disk:  error getting image data pointer!?");

#ifdef RECORD_TIMESTAMPS
	store_grab_time(this_idx);
#endif /* RECORD_TIMESTAMPS */

	n_to_write = skc_p->skc_bytes_per_image;
fprintf(stderr,"Will attempt to write %ld bytes\n",n_to_write);
	if( (n_written = write(rdi_p->rdi_fd,data_ptr,n_to_write)) != n_to_write ){
		sprintf(ERROR_STRING,"write (frm %ld, fd=%d, buf = 0x%lx, n = %ld )",
			frame_id,rdi_p->rdi_fd,(long)data_ptr,n_to_write);
		perror(ERROR_STRING);
		sprintf(ERROR_STRING,"%ld requested, %ld written", n_to_write,n_written);
		warn(DEFAULT_ERROR_STRING);
		return;
	}

	rdi_p->rdi_next += n_disks;
	rdi_p->rdi_state = DISK_IDLE;

} /* end write_image_to_disk() */

#define finish_recording(ifp) _finish_recording(QSP_ARG  ifp)

static void _finish_recording(QSP_ARG_DECL  Image_File *ifp)
{
	RV_Inode *inp;

	inp = get_rv_inode(ifp->if_name);
	assert( inp != NULL );

	close_image_file(ifp);		/* close write file	*/

	update_movie_database(inp);

	// do we have error frames for PGR??
	//note_error_frames(inp);
}

static uint32_t get_blocks_per_frame(Spink_Cam *skc_p)
{
	uint32_t blocks_per_frame, bytes_per_frame;

	bytes_per_frame = skc_p->skc_cols * skc_p->skc_rows * skc_p->skc_depth;
fprintf(stderr,"%s:  bytes_per_frame = %d\n",skc_p->skc_name,bytes_per_frame);


#ifdef RECORD_TIMESTAMPS
	if( stamping ) bytes_per_frame += TIMESTAMP_SIZE;
#endif /* RECORD_TIMESTAMPS */
	
	blocks_per_frame = ( bytes_per_frame + BLOCK_SIZE - 1 ) / BLOCK_SIZE;
	return(blocks_per_frame);
}

static void wait_for_recording(void)
{
	int waiting;
int told=0;

	// Now we wait until all of the frames have been stored
	// We could sleep if we knew the frame rate...
	do {
		int i;

		waiting=0;
		for( i=0; i<n_disks && waiting == 0; i++ ){
			if( disk_info_tbl[i].rdi_next != disk_info_tbl[i].rdi_last ){
if( ! told ){
fprintf(stderr,"waiting for disk %d (next = %ld   last = %ld)\n",i,
disk_info_tbl[i].rdi_next,disk_info_tbl[i].rdi_last);
told=1;
}
				waiting=1;
			}
		}
	} while(waiting);
}

static void setup_disk_info(uint32_t n_frames_wanted)
{
	uint32_t min_per_disk,n_extra;
	int i;

	min_per_disk = n_frames_wanted / n_disks;
	n_extra = n_frames_wanted % n_disks;
	for(i=0;i<n_disks;i++){
		disk_info_tbl[i].rdi_next = i;
		disk_info_tbl[i].rdi_last = min_per_disk * n_disks + i;
		if( i < n_extra )
			disk_info_tbl[i].rdi_last += n_disks ;
	}
}

#define setup_recording_file(ifp, skc_p, n ) _setup_recording_file(QSP_ARG  ifp, skc_p, n )

static void _setup_recording_file(QSP_ARG_DECL  Image_File *ifp, Spink_Cam *skc_p, uint32_t n_frames_wanted )
{
	int fd_arr[MAX_DISKS];
	Shape_Info *shpp;
	RV_Inode *inp;
	int i;

	inp = (RV_Inode *)ifp->if_hdr_p;
	n_disks = queue_rv_file(inp,fd_arr);
	assert( n_disks > 1 );
	for(i=0;i<n_disks;i++)
		disk_info_tbl[i].rdi_fd = fd_arr[i];	// disks should be seek'd to proper spot!

	/* set the shape info */
	//meteor_get_geometry(&_geo);

	shpp = ALLOC_SHAPE;
	SET_SHP_FLAGS(shpp,0);

	/* what about the timestamp information that may be tacked onto
	 * the frame?  Does this not get written to the raw volume?
	 */

	SET_SHP_ROWS(shpp, skc_p->skc_rows );
	SET_SHP_COLS(shpp, skc_p->skc_cols );
	SET_SHP_COMPS(shpp,skc_p->skc_depth);

	SET_SHP_FRAMES(shpp,n_frames_wanted);
	SET_SHP_SEQS(shpp, 1);
	SET_SHP_PREC_PTR(shpp,PREC_FOR_CODE(PREC_UBY) );
	SET_SHP_FLAGS(shpp, DT_IMAGE );


	// does this copy or point?
	// can we free the shape?
	// BUG memory leak
	rv_set_shape(ifp->if_name,shpp);
}

void _spink_stream_record(QSP_ARG_DECL  Image_File *ifp,int32_t n_frames_wanted,int n_cameras,Spink_Cam **skc_p_tbl)
{
	//int32_t npix;
	//uint32_t total_blocks;
	uint32_t blocks_per_frame;
	//struct meteor_geomet _geo;
	int i;

	if( record_state != IDLE ){
		sprintf(ERROR_STRING,
	"stream_record:  can't record file %s until previous record completes",
			ifp->if_name);
		warn(ERROR_STRING);
		return;
	}

	assert(n_cameras>0&&n_cameras<3);	// What is the real limit on cameras???
	blocks_per_frame = get_blocks_per_frame(skc_p_tbl[0]);
	for(i=1;i<n_cameras;i++){
		if( blocks_per_frame != get_blocks_per_frame(skc_p_tbl[i]) ){
			sprintf(ERROR_STRING,"spink_stream_record:  camera frame size mismatch!?");
			warn(ERROR_STRING);
			sprintf(ERROR_STRING,"%s:  %d blocks per frame",skc_p_tbl[0]->skc_name,blocks_per_frame);
			advise(ERROR_STRING);
			sprintf(ERROR_STRING,"%s:  %d blocks per frame",skc_p_tbl[i]->skc_name,
				get_blocks_per_frame(skc_p_tbl[i]));
			advise(ERROR_STRING);
			advise("Cameras must have identical frame sizes for multi-camera capture.");
			return;
		}
	}

	// the frame size should be an integral number of blocks...
fprintf(stderr,"spink_stream_record:  bytes per image is %ld\n", skc_p_tbl[0]->skc_bytes_per_image);
	assert( (skc_p_tbl[0]->skc_bytes_per_image % BLOCK_SIZE) == 0 );

	if( FT_CODE(IF_TYPE(ifp)) != IFT_RV ){
		sprintf(ERROR_STRING,
	"stream record:  image file %s (type %s) should be type %s",
			ifp->if_name,
			FT_NAME(IF_TYPE(ifp)),
			FT_NAME(FILETYPE_FOR_CODE(IFT_RV)) );
		warn(ERROR_STRING);
		return;
	}

	init_recording_disk_info();


	setup_recording_file(ifp,skc_p_tbl[0],n_frames_wanted);

#ifdef RECORD_TIMESTAMPS
	if( stamping ) reset_stamps();
#endif /* RECORD_TIMESTAMPS */

	n_stream_frames = n_frames_wanted;	/* remember for later printing... */

	if( mlockall(MCL_FUTURE) < 0 ){
		tell_sys_error("mlockall");
		warn("Failed to lock process memory!?");
	} else {
		mem_locked=1;
	}

	setup_disk_info(n_frames_wanted);
fprintf(stderr,"stream_record:  enabling image events\n");
	for(i=0;i<n_cameras;i++)
		enable_image_events(skc_p_tbl[i], write_image_to_disk );

	record_state = RECORDING;

fprintf(stderr,"stream_record:  starting cameras (%d)\n",n_cameras);
	for(i=0;i<n_cameras;i++)
		spink_start_capture(skc_p_tbl[i]);

fprintf(stderr,"stream_record:  waiting\n");
	wait_for_recording();
	finish_recording(ifp);

} /* end stream_record */


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

Image_File * _get_file_for_recording(QSP_ARG_DECL  const char *name, uint32_t n_frames_wanted,Spink_Cam *skc_p)
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

	n_blocks = rv_frames_to_allocate(n_frames_wanted) * get_blocks_per_frame(skc_p);

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

