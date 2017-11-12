
#include "quip_config.h"

#ifdef HAVE_METEOR

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>		/* open() */
#endif

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#include "quip_prot.h"
#include "mmvi.h"
#include "ioctl_meteor.h"
#include "mmenu.h"
#include "rv_api.h"
#include "fio_api.h"

#include "data_obj.h"

/* global vars */
struct meteor_mem *_mm=NULL;
int _hiwat=0, _lowat=0;
/* how do we know what size the ioctl codes are? */
int64_t capture_code = METEORCAPTUR ;

/* local prototypes */
static void mem_record(QSP_ARG_DECL  Image_File *ifp,uint32_t n_frames);


static COMMAND_FUNC( do_meteor_status )
{
	meteor_status(SINGLE_QSP_ARG);
}

void meteor_status(SINGLE_QSP_ARG_DECL)
{

//	INSURE_MM("meteor_status");
	assert( _mm != NULL );

	sprintf(msg_str,"_mm = 0x%"PRIxPTR,(uintptr_t)_mm);
	prt_msg(msg_str);
	sprintf(msg_str,"frame size: %d (0x%x)",_mm->frame_size,_mm->frame_size);
	prt_msg(msg_str);
	sprintf(msg_str,"num bufs: %d",_mm->num_bufs);
	prt_msg(msg_str);
	sprintf(msg_str,"%d frames captured",_mm->n_frames_captured);
	prt_msg(msg_str);
	sprintf(msg_str,"current frame %d, field %d",_mm->cur_frame,_mm->cur_field);
	prt_msg(msg_str);
	sprintf(msg_str,"lowat %d, hiwat %d",_mm->lowat,_mm->hiwat);
	prt_msg(msg_str);
	sprintf(msg_str,"active mask 0x%lx",_mm->active);
	prt_msg(msg_str);
	sprintf(msg_str,"%d active bufs",_mm->num_active_bufs);
	prt_msg(msg_str);
}

static void meteor_get_errors(SINGLE_QSP_ARG_DECL)
{
	struct meteor_counts cnt;

	if (ioctl(meteor_fd, METEORGCOUNT, &cnt)) {
		perror("ioctl GetCount failed");
		exit(1);
	}
	sprintf(ERROR_STRING, "Frames: %d\nEven:   %d\nOdd:	%d\n", 
		cnt.n_frames_captured,
		cnt.even_fields_captured,
		cnt.odd_fields_captured);
	advise(ERROR_STRING);
	sprintf(ERROR_STRING, "Fifo errors: %d\n", cnt.fifo_errors);
	advise(ERROR_STRING);
	sprintf(ERROR_STRING, "DMA errors:  %d\n", cnt.dma_errors);
	advise(ERROR_STRING);
}

void meteor_clear_counts()
{
	struct meteor_counts cnt;

	if (ioctl(meteor_fd, METEORGCOUNT, &cnt)) {
		perror("ioctl GetCount failed");
		return;
	}

	cnt.fifo_errors			= 0;
	cnt.dma_errors			= 0;
	cnt.n_frames_captured		= 0;
	cnt.even_fields_captured	= 0;
	cnt.odd_fields_captured		= 0;

	if (ioctl(meteor_fd, METEORSCOUNT, &cnt)) {
		perror("ioctl SetCount failed");
		return;
	}
}

#ifdef USE_SIGS
//void gotframe(int signum)
//{
//	struct meteor_mem *mm = (struct meteor_mem *)(mmbuf + meteor_off.mem_off);
//	unsigned char *src;
//	unsigned int count;
//
//	src = (u_char *)
//	( (num_meteor_frames>1)	? mmbuf+meteor_off.frame_offset[mm->cur_frame-1]
//				: mmbuf + meteor_off.frame_offset[0] );
//	count = my_geo.rows*my_geo.columns*meteor_bytes_per_pixel;
//}
#endif // USE_SIGS

static void meteor_check_capture_control(SINGLE_QSP_ARG_DECL)
{
	uint32_t cap;

	if (ioctl(meteor_fd, METEORGCAPT, &cap)) {
		perror("ioctl GetCapt failed");
		exit(1);
	}
	sprintf(ERROR_STRING, "Capture control: 0x%x\n", cap);
	advise(ERROR_STRING);
}

static COMMAND_FUNC( do_meteor_check )
{
	meteor_get_errors(SINGLE_QSP_ARG);
	/*
	meteor_check_frame();
	*/
	meteor_check_capture_control(SINGLE_QSP_ARG);
}

static int capture_mode = METEOR_CAP_SINGLE ;
static int mm_flags=0;
#define CAPTURING	1
#define IS_CAPTURING	(mm_flags&CAPTURING)

static COMMAND_FUNC( do_meteor_capture )
{
advise("do_meteor_capture BEGIN");
	meteor_capture(SINGLE_QSP_ARG);
}

/* There are two capture modes (codes) implemented in the driver.
 * METEORCAPTUR and METEORCAPFRM
 *
 * For METEORCAPTUR, there are several modes:
 *	METEOR_CAP_SINGLE
 *	METEOR_CAP_CONT_ONCE
 *	METEOR_CAP_CONTINOUS
 *
 * For METEORCAPFRM, capture will proceed depending on the "water"
 * levels...  we set this so that we will not stop capturing until
 * all the buffers are full, and we will restart when there is 1 or more
 * unused buffers...
 *
 */

int meteor_capture(SINGLE_QSP_ARG_DECL)
{
	if( IS_CAPTURING ){
		warn("meteor_capture:  already capturing!?");
		return(-1);
	}

#ifndef FAKE_METEOR_HARDWARE
	if( capture_code == METEORCAPTUR ){
/*
sprintf(ERROR_STRING,"capture_code = METEORCAPTUR, mode = %d (0x%x)",capture_mode,capture_mode);
advise(ERROR_STRING);
*/
		if (ioctl(meteor_fd, METEORCAPTUR, &capture_mode)){
			perror("ioctl Capture failed");
			return(-1);
		}
	} else {
		struct meteor_capframe frame;

		frame.command = METEOR_CAP_N_FRAMES;
		frame.lowat = _lowat;
		frame.hiwat = _hiwat;
		if (ioctl(meteor_fd, METEORCAPFRM, &frame)){
			perror("ioctl CAPFRM/CAP_N_FRAMES failed");
			return(-1);
		}
	}
#endif /* ! FAKE_METEOR_HARDWARE */
	mm_flags |= CAPTURING;
	return(0);
}

static void meteor_disable_signal()
{
#ifndef FAKE_METEOR_HARDWARE
	int c;

	c = 0;
	if (ioctl(meteor_fd, METEORSSIGNAL, &c) < 0) {
		perror("ioctl SetSignal failed");
		exit(1);
	}
#endif
}

COMMAND_FUNC( meteor_stop_capture )
{
	int c;

	if( !IS_CAPTURING ){
		warn("Not capturing, can't stop!?");
		return;
	}

	/* first disable application signals */
	meteor_disable_signal();

#ifndef FAKE_METEOR_HARDWARE
	/* now stop continuous capture */
	if( capture_code == METEORCAPTUR ){
		c = METEOR_CAP_STOP_CONT;
		if (ioctl(meteor_fd, METEORCAPTUR, &c)) {
			perror("ioctl CaptContinuous failed");
			exit(1);
		}

	} else {
		struct meteor_capframe frame;

	/* METEOR_CAP_STOP_FRAMES is used w/ METEOR_SYNCAP */

		frame.command = METEOR_CAP_STOP_FRAMES;
		if (ioctl(meteor_fd, METEORCAPFRM, &frame)) {
			perror("ioctl CAPFRM/CAP_STOP_FRAMES failed");
			exit(1);
		}
	}
#endif

	usleep(80000);  /* wait for 80ms to make sure no more frames are arriving */

	mm_flags &= ~CAPTURING;
}

#define N_CAPTURE_MODES	4
static const char *mode_names[N_CAPTURE_MODES]={
	"single frame",
	"single sequence",
	"continuous",
	"synchronous"
};

static int meteor_mode[N_CAPTURE_MODES] ={
	METEOR_CAP_SINGLE,
	METEOR_CAP_CONT_ONCE,
	METEOR_CAP_CONTINOUS,
	METEORCAPFRM
} ;

static int index_of_mode(QSP_ARG_DECL  int mode )
{
	int i;

	for(i=0;i<N_CAPTURE_MODES;i++)
		if( meteor_mode[i] == mode ) return(i);
	assert( ! "mode missing from table" );
	return(-1);
}

static void set_capture_mode(int64_t mode)
{
	if( mode == METEORCAPFRM ){
		capture_code=mode;
	} else {
		capture_mode = mode;
		capture_code = METEORCAPTUR;	/* default */
	}
}

void setup_monitor_capture(SINGLE_QSP_ARG_DECL)
{
	if( IS_CAPTURING ){
		/* are we in the right mode??? */
		if( capture_mode == METEOR_CAP_CONTINOUS ){
			if( verbose ) advise("Meteor will continue to capture in continuous mode.");
		} else {
			int i;
			i=index_of_mode(QSP_ARG  capture_mode);
			sprintf(ERROR_STRING,"Meteor is already capturing in %s mode!?",
					i>=0 ?  mode_names[ i ] : "(unknown)" );
			warn(ERROR_STRING);
		}
		return;
	}
	set_capture_mode(METEOR_CAP_CONTINOUS);
	meteor_capture(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_set_capture_mode )
{
	int i;
	int mode=(-1);

	i=WHICH_ONE("capture mode",N_CAPTURE_MODES,mode_names);
	if( i<0 ) return;

	switch(i){
		case 0:  mode = METEOR_CAP_SINGLE; break;
		case 1:  mode = METEOR_CAP_CONT_ONCE; break;
		case 2:  mode = METEOR_CAP_CONTINOUS; break;
		case 3:  mode = METEORCAPFRM; break;
	}
	if( mode < 0 ) return;

	set_capture_mode(mode);
}

static void point_object_to_frame(Data_Obj *dp, int index)
{
	SET_OBJ_DATA_PTR(dp,frame_address(index));
}

Data_Obj *make_frame_object(QSP_ARG_DECL  const char* name,int index)
{
	Dimension_Set dimset;
	Data_Obj *dp;

	if( index<0 || index>= num_meteor_frames ){
		warn("frame index out of range");
		return(NULL);
	}

	dimset.ds_dimension[0] = meteor_bytes_per_pixel;
	dimset.ds_dimension[1] = my_geo.columns;
	dimset.ds_dimension[2] = my_geo.rows;
	dimset.ds_dimension[3] = 1;
	dimset.ds_dimension[4] = 1;

	dp = _make_dp(QSP_ARG  name,&dimset,PREC_FOR_CODE(PREC_UBY));

	if( dp != NULL )
		point_object_to_frame(dp,index);

	return(dp);
}

static COMMAND_FUNC( do_make_object )
{
	Data_Obj *dp;
	const char *s;
	int n;

	s=NAMEOF("name for object");
	n=HOW_MANY("index of frame");

	dp = make_frame_object(QSP_ARG  s,n);
}

static COMMAND_FUNC( do_lowat )
{
	int i;

	i=HOW_MANY("lowat");
	/* check for legal value here BUG */
	_lowat=i;
}


static COMMAND_FUNC( do_hiwat )
{
	int i;

	i=HOW_MANY("hiwat");
	/* check for legal value here BUG */
	_hiwat=i;
}

static COMMAND_FUNC( do_release )
{
//	INSURE_MM("do_release");
	assert( _mm != NULL );

	_mm->num_active_bufs--;
}

static int32_t error_code_tbl[3]={
	METEORGERR0FRMS,
	METEORGERR1FRMS,
	METEORGERR2FRMS
};


static void note_error_frames(RV_Inode *inp)
{
	struct error_info ei1;
	struct drop_info di;
	uint32_t *tmp_buf;
	int i,nerrs;

	/* BUG it would be cleaner to merge dropped frames with the other error
	 * types in the driver.
	 */

#ifndef FAKE_METEOR_HARDWARE
	if (ioctl(meteor_fd, METEORGNDROP, &di)) {
		perror("ioctl GNDROP failed");
		exit(1);
	}
#else
	di.n_total = di.n_saved = 0;
#endif

	nerrs = di.n_saved;
	if( nerrs > 0 ){
		tmp_buf =(uint32_t*)  getbuf( nerrs * sizeof(uint32_t) );
		if( ioctl(meteor_fd,METEORGDRPFRMS,tmp_buf) ){
			perror("ioctl METEORGDRPFRMS failed");
		}
		remember_frame_info(inp,0,nerrs,tmp_buf);
		givbuf(tmp_buf);
	}

#ifndef FAKE_METEOR_HARDWARE
	if (ioctl(meteor_fd, METEORGNERR, &ei1)) {
		perror("ioctl GNERR failed");
		exit(1);
	}
#else
	for(i=0;i<3;i++){
		ei1.ei[i].n_total = ei1.ei[i].n_saved = 0;
	}
#endif

	for(i=0;i<3;i++){
		nerrs = ei1.ei[i].n_saved;
		if( nerrs > 0 ){
			tmp_buf = (uint32_t*) getbuf( nerrs * sizeof(uint32_t) );
			if( ioctl(meteor_fd,error_code_tbl[i],tmp_buf) ){
				perror("ioctl METEORGERRFRMS failed");
			}
			remember_frame_info(inp,1+i,nerrs,tmp_buf);
			givbuf(tmp_buf);
		}
	}
}

uint32_t get_blocks_per_frame()
{
	uint32_t blocks_per_frame, bytes_per_frame;

	bytes_per_frame = meteor_columns*meteor_rows*meteor_bytes_per_pixel;

#ifdef RECORD_TIMESTAMPS
	if( stamping ) bytes_per_frame += TIMESTAMP_SIZE;
#endif /* RECORD_TIMESTAMPS */
	
	blocks_per_frame = ( bytes_per_frame + BLOCK_SIZE - 1 ) / BLOCK_SIZE;
	return(blocks_per_frame);
}

/*
 * do_record()
 *
 * saves the frames to a disk file, no header, just raw rgbx...
 */

static COMMAND_FUNC( do_record )
{
	int32_t n_frames;
	Image_File *ifp;
	const char *name;
	uint32_t n_blocks;

	/* BUG use fileio library here??? */

	name = NAMEOF("image file name");
	n_frames=HOW_MANY("number of frames (or fields if field mode)");

	ifp = img_file_of(QSP_ARG  name);

	if( ifp != NULL ){
		sprintf(ERROR_STRING,"Clobbering existing image file %s",name);
		advise(ERROR_STRING);
		image_file_clobber(1);	/* not necessary !? */
		delete_image_file(QSP_ARG  ifp);
	}

	set_filetype(QSP_ARG  FILETYPE_FOR_CODE(IFT_RV));
	ifp = write_image_file(QSP_ARG  name,n_frames);	/* nf stored in if_frms_to_write */

	/* sets nframes in ifp, but doesn't allocate rv blocks properly...
	 * (WHY NOT??? maybe because the image dimensions are not known???)
	 * we could know them, however, because at this point the geometry is set.
	 */

	if( ifp == NULL ){
		sprintf(ERROR_STRING,"Error creating movie file %s",name);
		warn(ERROR_STRING);
		return;
	}

	n_blocks = rv_frames_to_allocate(n_frames) * get_blocks_per_frame();

	/* n_blocks is the total number of blocks, not the number per disk(?) */

	if( rv_realloc(QSP_ARG  name,n_blocks) < 0 ){
		sprintf(ERROR_STRING,"error reallocating %d blocks for rv file %s",
			n_blocks,name);
		warn(ERROR_STRING);
		return;
	}

	meteor_record_clip(QSP_ARG  ifp,n_frames);
} /* end do_record() */

void finish_recording(QSP_ARG_DECL  Image_File *ifp)
{
	RV_Inode *inp;

	inp = get_rv_inode(QSP_ARG  ifp->if_name);
	assert(inp!=NULL);

	close_image_file(QSP_ARG  ifp);		/* close write file	*/
	update_movie_database(QSP_ARG  inp);
	note_error_frames(inp);
}

int recording_in_process = 0;
Image_File *record_ifp=NULL;

void meteor_record_clip(QSP_ARG_DECL  Image_File *ifp,int32_t n_frames)
{
	if( recording_in_process ){
		warn("meteor_record_clip:  async recording is already in process, need to wait!?");
		meteor_wait_record(SINGLE_QSP_ARG);	/* isn't really used... */
	}

	recording_in_process = 1;
	record_ifp = ifp;

	/* If we want async recording, use stream record,
	 * even if number of frames is small.
	 */
	if( n_frames > num_meteor_frames || get_async_record() ){
		stream_record(QSP_ARG  ifp,n_frames);
	} else {
		/* all frames can fit in mem */
		mem_record(QSP_ARG  ifp,n_frames);
	}

	/* We used to check for async record here, and call finish_recording()
	 * if false, but finish_recording() is also called at the end of video_reader()...
	 * This works for both async and synchronous recordings.
	 */
}

/* Record a short movie, where we have enough ram allocated to hold
 * the whole thing.  In this case, we acquire the whole thing before
 * we initiate any disk i/o.
 */

static void mem_record(QSP_ARG_DECL  Image_File *ifp,uint32_t n_frames)
{
	int32_t n_remaining;
	int32_t frames_requested;
	int mem_frm_to_wt;
	int32_t npix;
	int32_t n_to_write;
	struct meteor_geomet gp;
	unsigned int i=0;
	Dimension_Set dimset;
	Data_Obj *dp;
	RV_Inode *inp;

//advise("mem_record BEGIN");
	if( n_frames > MAX_NUM_FRAMES )
		error1("mem_record:  Fix MAX_NUM_FRAMES");

	mem_frm_to_wt=0;

	if( _mm == NULL ){
fprintf(stderr,"calling map_mem_data...\n");
		_mm = map_mem_data(SINGLE_QSP_ARG);
		assert( _mm != NULL );
	}

	/* set synchronous mode */
	capture_code = METEORCAPTUR ;	/* mem_record() */
	capture_mode = METEOR_CAP_CONT_ONCE ;

	n_to_write = _mm->frame_size;
	npix=n_to_write/4;

	/* start capturing */

	_mm->n_frames_captured=0;

	if( meteor_capture(SINGLE_QSP_ARG) < 0 ){
		warn("error starting capture");
		return;
	}

	frames_requested=n_frames;
	n_remaining=n_frames;

#ifndef FAKE_METEOR_HARDWARE
	while(_mm->n_frames_captured< n_frames){
/*
sprintf(ERROR_STRING,"%d of %d frames captured",_mm->n_frames_captured,n_frames);
advise(ERROR_STRING);
*/
		usleep(16000);
	}
#else
advise("faking hardware");
	_mm->n_frames_captured = n_frames;
#endif

	meteor_stop_capture(SINGLE_QSP_ARG);

	if( ifp == NULL ) return;

	meteor_get_geometry(&gp);

	dimset.ds_dimension[0]=get_bytes_per_pixel(QSP_ARG  gp.oformat );
	dimset.ds_dimension[1]=gp.columns;	
	dimset.ds_dimension[2]=gp.rows;	
	dimset.ds_dimension[3]=1;
	dimset.ds_dimension[4]=1;
	dp = _make_dp(QSP_ARG  "tmp_dp",&dimset,PREC_FOR_CODE(PREC_UBY));
	if( !meteor_field_mode )
		SET_SHP_FLAG_BITS(OBJ_SHAPE(dp),DT_INTERLACED);

	if( dp == NULL ){
		warn("mem_record:  error creating tmp dp");
		return;
	}

	/* Now write the frames to disk */

	/* This does not properly handle timestamps, so call out an error */
	assert( FT_CODE(IF_TYPE(ifp)) == IFT_RV );

	inp = (RV_Inode *)ifp->if_hdr_p;
	if( rv_movie_extra(inp) != 0 ){
		sprintf(ERROR_STRING,"File %s, rvi_extra_bytes = %d!?",ifp->if_name,rv_movie_extra(inp));
		warn(ERROR_STRING);
		error1("Sorry, can't record timestamps in memory recordings at present...");
	}

	for(i=0;i<n_frames;i++){
		point_object_to_frame(dp,i);
		write_image_to_file(QSP_ARG  ifp,dp);
	}
	SET_OBJ_FLAG_BITS(dp, DT_NO_DATA);
	delvec(QSP_ARG  dp);

	/* the image file should have been close automatically by
	 * write_image_to_file().  At this point, we might like to
	 * automatically open it for reading to make it available
	 * for playback.  (BUG?)
	 */

	rv_sync(SINGLE_QSP_ARG);

	recording_in_process = 0;
}

#ifdef RECORD_CAPTURE_COUNT
static COMMAND_FUNC( do_dump_cc )
{
	FILE *fp;
	const char *s;
	int index;

	index = HOW_MANY("index");
	s=NAMEOF("filename");
	fp = try_open(s,"w");
	if( !fp ) return;

	dump_ccount(index,fp);
}
#endif /* RECORD_CAPTURE_COUNT */


#ifdef FOOBAR
static COMMAND_FUNC( do_set_ndiscard )
{
	int n;

	n = HOW_MANY("number of lines to discard");
	set_n_discard(n);
}
#endif /* FOOBAR */

static COMMAND_FUNC( do_playback )
{
	Image_File *ifp;

	ifp = pick_img_file("");
	if( ifp == NULL ) return;

#ifdef HAVE_X11_EXT
	play_meteor_movie(QSP_ARG  ifp);
#else
	warn("do_playback:  Program was configured without X11 Extensions.");
#endif
}

static void get_error_counts(QSP_ARG_DECL  const char* s,const char* s2,int index)
{
	struct error_info ei1;
	char val[32];
#ifndef FAKE_METEOR_HARDWARE
	if (ioctl(meteor_fd, METEORGNERR, &ei1)) {
		perror("ioctl GNERR failed");
		exit(1);
	}
#else
	ei1.ei[0].n_total = ei1.ei[0].n_saved = 0;
	ei1.ei[1].n_total = ei1.ei[1].n_saved = 0;
	ei1.ei[2].n_total = ei1.ei[2].n_saved = 0;
#endif

	sprintf(val,"%d",ei1.ei[index].n_total);
	assign_var(s,val);
	sprintf(val,"%d",ei1.ei[index].n_saved);

	assign_var(s2,val);
}

static COMMAND_FUNC( do_get_n_fifo_errors )
{
	const char *s,*s2;

	s = NAMEOF("variable name for total errors");
	s2 = NAMEOF("variable name for saved errors");

	get_error_counts(QSP_ARG  s,s2,0);
}

static COMMAND_FUNC( do_get_n_dma_errors )
{
	const char *s,*s2;

	s = NAMEOF("variable name for total errors");
	s2 = NAMEOF("variable name for saved errors");

	get_error_counts(QSP_ARG  s,s2,1);
}

static COMMAND_FUNC( do_get_n_fifodma_errors )
{
	const char *s,*s2;

	s = NAMEOF("variable name for total errors");
	s2 = NAMEOF("variable name for saved errors");

	get_error_counts(QSP_ARG  s,s2,2);
}

static COMMAND_FUNC( do_get_ndrops )
{
	struct drop_info di;
	const char *s,*s2;
	char val[32];

	s = NAMEOF("variable name for total drops");
	s2 = NAMEOF("variable name for saved drops");

#ifndef FAKE_METEOR_HARDWARE
	if (ioctl(meteor_fd, METEORGNDROP, &di)) {
		perror("ioctl GNDROP failed");
		exit(1);
	}
#else
	di.n_total = di.n_saved = 0;
#endif

	sprintf(val,"%d",di.n_total);
	assign_var(s,val);
	sprintf(val,"%d",di.n_saved);
	assign_var(s2,val);
}

static void get_err_fields(QSP_ARG_DECL  Data_Obj *dp,int index)
{
	struct error_info ei;
	unsigned int nerrs;

	if( index < 0 || index > 2 ){
		sprintf(ERROR_STRING,
			"error index (%d) must be between 0 and 2",index);
		warn(ERROR_STRING);
	}

	if( OBJ_MACH_PREC(dp) != PREC_UDI ){
		sprintf(ERROR_STRING,
	"Vector %s has precision %s, should be %s for METEORGERRFRMS",
			OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)),
			PREC_UDI_NAME);
		warn(ERROR_STRING);
		return;
	}

	if (ioctl(meteor_fd, METEORGNERR, &ei)) {
		perror("ioctl GNERR failed");
		return;
	}
	nerrs = ei.ei[index].n_saved;

	if( OBJ_N_TYPE_ELTS(dp) < nerrs ){
		sprintf(ERROR_STRING,
	"Vector %s has %d elements, not big enough to store %d errors",
			OBJ_NAME(dp),OBJ_N_TYPE_ELTS(dp),nerrs);
		warn(ERROR_STRING);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,
	"Vector %s should be contiguous for METEORGERRFRMS",
			OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}

	if( ioctl(meteor_fd,error_code_tbl[index],OBJ_DATA_PTR(dp)) ){
		perror("ioctl METEORGERRFRMS failed");
	}
}

static COMMAND_FUNC( do_get_fifo_errors )
{
	Data_Obj *dp;


	dp = pick_obj("vector for error frame indices");
	if( dp== NULL ) return;

	get_err_fields(QSP_ARG  dp,0);	/* see /usr/src/matrox/meteor.c */
}

static COMMAND_FUNC( do_get_fifodma_errors )
{
	Data_Obj *dp;


	dp = pick_obj("vector for error frame indices");
	if( dp== NULL ) return;

	get_err_fields(QSP_ARG  dp,2);	/* see /usr/src/matrox/meteor.c */
}

static COMMAND_FUNC( do_get_dma_errors )
{
	Data_Obj *dp;


	dp = pick_obj("vector for error frame indices");
	if( dp== NULL ) return;

	get_err_fields(QSP_ARG  dp,1);	/* see /usr/src/matrox/meteor.c */
}


static COMMAND_FUNC( do_get_drops )
{
	Data_Obj *dp;
	struct drop_info di;

	dp = pick_obj("vector for drop frame indices");

	if( dp== NULL ) return;

	if( OBJ_MACH_PREC(dp) != PREC_UDI ){
		sprintf(ERROR_STRING,
	"Vector %s has precision %s, should be %s for METEORGDRPFRMS",
			OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)),
			PREC_UDI_NAME);
		warn(ERROR_STRING);
		return;
	}

#ifndef FAKE_METEOR_HARDWARE
	if (ioctl(meteor_fd, METEORGNDROP, &di)) {
		perror("ioctl GNDROP failed");
		return;
	}
#else
	di.n_total = di.n_saved = 0;
#endif

	if( OBJ_N_TYPE_ELTS(dp) < di.n_saved ){
		sprintf(ERROR_STRING,
	"Vector %s has %d elements, not big enough to store %d drops",
			OBJ_NAME(dp),OBJ_N_TYPE_ELTS(dp),di.n_saved);
		warn(ERROR_STRING);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,
	"Vector %s should be contiguous for METEORGDRPFRMS",
			OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}

	if( ioctl(meteor_fd,METEORGDRPFRMS,OBJ_DATA_PTR(dp)) ){
		perror("ioctl METEORGDRPFRMS failed");
	}
}

static COMMAND_FUNC( do_set_async )
{
	if( ASKIF("record movies asynchronously") )
		set_async_record(1);
	else
		set_async_record(0);
}

#ifdef RECORD_TIMESTAMPS
static COMMAND_FUNC(  do_ts_enable )
{
	stamping=ASKIF("record timestamp data during recording");

	enable_meteor_timestamps(QSP_ARG  stamping);	/* set driver flag */

	if( stamping ){
		rv_set_extra(sizeof(struct timeval));
	} else {
		rv_set_extra(0);
	}
}

static COMMAND_FUNC( do_dump )
{
	const char *s;
	s=NAMEOF("filename");
	if( !stamping ) {
		warn("not collecting timestamps, can't dump");
		return;
	}
	dump_timestamps(s);
}

static const char *ts_names[2]={"grab_time","store_time"};

static COMMAND_FUNC( do_print )
{
	int i;

	i=WHICH_ONE("type of timestamp (grab_time/store_time)",2,ts_names);
	if( i < 0 ) return;
	if( i==0 ) print_grab_times();
	else if( i==1 ) print_store_times();
}

#endif /* RECORD_TIMESTAMPS */

static COMMAND_FUNC( do_write_enable )
{
	int index,flag;
	char pmpt[LLEN];

	index = HOW_MANY("index of disk writer thread");
	sprintf(pmpt,"Write data to disk-writer thread %d",index);
	flag = ASKIF(pmpt);

	thread_write_enable(QSP_ARG  index,flag);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(capture_menu,s,f,h)

MENU_BEGIN(capture)
ADD_CMD( record,		do_record,		stream video to disk )
ADD_CMD( async,			do_set_async,		set/clear asynchronous recording )
ADD_CMD( halt,			meteor_halt_record,	halt asynchronous recording )
ADD_CMD( wait,			meteor_wait_record,	wait for asynchronous recording to finish )
ADD_CMD( playback,		do_playback,		playback a previously recorded movie )
#ifdef RECORD_CAPTURE_COUNT
ADD_CMD( ccount,		do_dump_cc,		dump ccount array to file )
#endif /* RECORD_CAPTURE_COUNT */
#ifdef FOOBAR
ADD_CMD( ndiscard,		do_set_ndiscard,	set number of lines to discard )
#endif /* FOOBAR */
ADD_CMD( n_fifo_errors,		do_get_n_fifo_errors,	get the number of corrupted frames )
ADD_CMD( n_dma_errors,		do_get_n_dma_errors,	get the number of corrupted frames )
ADD_CMD( n_fifodma_errors,	do_get_n_fifodma_errors,	get the number of corrupted frames )
ADD_CMD( ndrops,		do_get_ndrops,		get the number of dropped frames )
ADD_CMD( get_fifo_frms,		do_get_fifo_errors,	get the corrupted frame indices )
ADD_CMD( get_fifodma_frms,	do_get_fifodma_errors,	get the corrupted frame indices )
ADD_CMD( get_dma_frms,		do_get_dma_errors,	get the corrupted frame indices )
ADD_CMD( get_drops,		do_get_drops,		get the corrupted frame indices )
ADD_CMD( write_data,		do_write_enable,	enable/disable writing to particular disks )
#ifdef RECORD_TIMESTAMPS
ADD_CMD( enable_timestamps,	do_ts_enable,		enable/disable timestamping )
ADD_CMD( dump_timestamps,	do_dump,		dump timestamp data to a file )
ADD_CMD( print_timestamps,	do_print,		print timestamp data )
#endif /* RECORD_TIMESTAMPS */
MENU_END(capture)

COMMAND_FUNC( do_capture )
{
	CHECK_AND_PUSH_MENU(capture);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(cap_tst_menu,s,f,h)

MENU_BEGIN(cap_tst)
ADD_CMD( mode,		do_set_capture_mode,	set capture mode )
ADD_CMD( capture,	do_meteor_capture,	capture frame(s) )
ADD_CMD( stop,		meteor_stop_capture,	stop capture )
ADD_CMD( check,		do_meteor_check,	check capture status )
ADD_CMD( monitor,	do_meteor_status,	monitor capture status )
ADD_CMD( object,	do_make_object,		create object for frame )
ADD_CMD( hiwat,		do_hiwat,		set hiwat variable )
ADD_CMD( lowat,		do_lowat,		set lowat variable )
ADD_CMD( release,	do_release,		decrement n_active_bufs )
MENU_END(cap_tst)

COMMAND_FUNC( do_captst )
{
	CHECK_AND_PUSH_MENU(cap_tst);
}


#endif /* HAVE_METEOR */
