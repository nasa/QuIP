#include "quip_config.h"

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif


#include "quip_prot.h"
#include "data_obj.h"
#include "fio_api.h"
#include "rv_api.h"
#include "veclib_api.h"
#include "vec_util.h"		/* yuv422_to_rgb24 */
#include "my_video_dev.h"
#include "my_v4l2.h"


Video_Device *curr_vdp=NO_VIDEO_DEVICE;

ITEM_INTERFACE_DECLARATIONS(Video_Device,video_dev,0)

/* Call ioctl, repeating if interrupted */

int xioctl(int fd, int request, void *arg)
{
	int r;

	do {
		r = ioctl(fd, request, arg);
	} while( r == -1 && errno== EINTR );

	return r;
}

void errno_warn(QSP_ARG_DECL  const char *s)
{
	sprintf(ERROR_STRING,"%s:  %s",s,strerror(errno));
	WARN(ERROR_STRING);
}



typedef struct {
	const char *	vfmt_name;
	int		vfmt_bpp;	// bytes per pixel
	uint32_t	vfmt_code;	// list of codes in videodev2.h
} Video_Format;

#define N_VIDEO_FORMATS 6

#ifndef HAVE_V4L2
#define	V4L2_PIX_FMT_YUYV	-1
#define	V4L2_PIX_FMT_RGB24	-1
#define	V4L2_PIX_FMT_BGR24	-1
#define	V4L2_PIX_FMT_RGB32	-1
#define	V4L2_PIX_FMT_BGR32	-1
#define	V4L2_PIX_FMT_GREY	-1
#endif // ! HAVE_V4L2

static Video_Format vfmt_list[N_VIDEO_FORMATS]={
{ "yuyv",	2,	V4L2_PIX_FMT_YUYV	},
{ "rgb24",	3,	V4L2_PIX_FMT_RGB24	},
{ "bgr24",	3,	V4L2_PIX_FMT_BGR24	},
{ "rgb32",	4,	V4L2_PIX_FMT_RGB32	},
{ "bgr32",	4,	V4L2_PIX_FMT_BGR32	},
{ "gray",	1,	V4L2_PIX_FMT_GREY	}
};

#define DEFAULT_VFMT_INDEX 0	// YUYV

static int vfmt_index=DEFAULT_VFMT_INDEX;
static const char *vfmt_names[N_VIDEO_FORMATS];
static int vfmt_names_inited=0;

static COMMAND_FUNC(set_vfmt)
{
	int i;

	if( ! vfmt_names_inited ){
		for(i=0;i<N_VIDEO_FORMATS;i++)
			vfmt_names[i] = vfmt_list[i].vfmt_name;
		vfmt_names_inited=1;
	}

	i=WHICH_ONE("Video pixel format",N_VIDEO_FORMATS,vfmt_names);
	if( i >= 0 )
		vfmt_index = i;
}

typedef struct {
	const char *	vfld_name;
	int		vfld_height;
	int		vfld_code;
} Video_Field_Mode;

#define N_FIELD_MODES 6

#ifndef HAVE_V4L2
#define	V4L2_FIELD_INTERLACED	-1
#define	V4L2_FIELD_ALTERNATE	-1
#define	V4L2_FIELD_TOP		-1
#define	V4L2_FIELD_BOTTOM	-1
#define	V4L2_FIELD_SEQ_TB	-1
#define	V4L2_FIELD_SEQ_BT	-1
#endif // HAVE_V4L2

static Video_Field_Mode vfld_tbl[N_FIELD_MODES]={
{	"interlaced",	480,	V4L2_FIELD_INTERLACED	},
{	"alternate",	240,	V4L2_FIELD_ALTERNATE	},
{	"top",		240,	V4L2_FIELD_TOP		},
{	"bottom",	240,	V4L2_FIELD_BOTTOM	},
{	"seq_tb",	480,	V4L2_FIELD_SEQ_TB	},
{	"seq_bt",	480,	V4L2_FIELD_SEQ_BT	}
};

static const char * field_mode_choices[N_FIELD_MODES];
static int field_mode_choices_inited=0;
static int vfld_index=0;	// default is interlaced

static COMMAND_FUNC( set_field_mode )
{
	int i;

	if( ! field_mode_choices_inited ){
		for(i=0;i<N_FIELD_MODES;i++)
			field_mode_choices[i] = vfld_tbl[i].vfld_name;
	}

	i=WHICH_ONE("field mode",N_FIELD_MODES,field_mode_choices);
	if( i>=0 ) vfld_index=i;
}


/* based on init_device from capture.c */

/* CHECK_CROPCAP is maybe here to retain some code we didn't orignally need? */

static int init_video_device(QSP_ARG_DECL  Video_Device *vdp)
{
#ifdef HAVE_V4L2

	struct v4l2_capability cap;
#ifdef 	CHECK_CROPCAP
	struct v4l2_cropcap cropcap;
	struct v4l2_crop crop;
#endif
	struct v4l2_format fmt;
	unsigned int min;
	unsigned int i_buffer;
	int bytes_per_pixel;

	struct v4l2_requestbuffers req;		/* for init_mmap() */

	if(-1 == xioctl(vdp->vd_fd, VIDIOC_QUERYCAP, &cap)) {
		if( errno == EINVAL ){
			sprintf(ERROR_STRING, "%s is not a V4L2 device!?", vdp->vd_name);
		} else {
			sprintf(ERROR_STRING,"VIDIOC_QUERYCAP:  %s",strerror(errno));
		}
		WARN(ERROR_STRING);
		return(-1);
	}

	if(!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
		sprintf(ERROR_STRING,"%s does not have video capture capability",vdp->vd_name);
		WARN(ERROR_STRING);
		return(-1);
	}

	if(!(cap.capabilities & V4L2_CAP_STREAMING)) {
		sprintf(ERROR_STRING,"%s does not support streaming i/o",vdp->vd_name);
		WARN(ERROR_STRING);
		return(-1);
	}

	if( cap.capabilities & V4L2_CAP_READWRITE ){
		advise("Device supports read/write");
	} else {
		advise("Device does NOT support read/write");
	}


	/* Select video input, video standard and tune here. */

#ifdef CHECK_CROPCAP
	/* What is cropcap?? */

	CLEAR(cropcap);

	cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	if( xioctl(vdp->vd_fd, VIDIOC_CROPCAP, &cropcap) == 0 ) {
		crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		crop.c = cropcap.defrect; /* reset to default */

		if(-1 == xioctl(vdp->vd_fd, VIDIOC_S_CROP, &crop)) {
			switch(errno) {
			case EINVAL:
				/* Cropping not supported. */
				WARN("cropping not supported");
				break;
			default:
				errno_warn("VIDIOC_S_CROP");
				/* Errors ignored. */
				break;
			}
		}
	} else {	
		/* with lml bt44, this generates EINVAL */
		//errno_warn("VIDIOC_CROPCAP");
		/* Errors ignored. */
	}
#endif /* CHECK_CROPCAP */


	CLEAR(fmt);

	fmt.type		= V4L2_BUF_TYPE_VIDEO_CAPTURE;
	/* BUG?  wouldn't it be better to query the card for the frame size? */
	fmt.fmt.pix.width	= 640; 
	fmt.fmt.pix.height	= vfld_tbl[vfld_index].vfld_height;
	// Can we query which pixel formats are allowed?

	fmt.fmt.pix.pixelformat = vfmt_list[vfmt_index].vfmt_code;
	bytes_per_pixel = vfmt_list[vfmt_index].vfmt_bpp;

	// Can we set the other field types???
	//fmt.fmt.pix.field	= V4L2_FIELD_INTERLACED;
	// this generates an invalid argument error...
	//fmt.fmt.pix.field	= V4L2_FIELD_ALTERNATE;
	fmt.fmt.pix.field	= vfld_tbl[vfld_index].vfld_code;

	if(-1 == xioctl(vdp->vd_fd, VIDIOC_S_FMT, &fmt)){
		sprintf(ERROR_STRING,"VIDIOC_S_FMT:  %s",strerror(errno));
		WARN(ERROR_STRING);
		return(-1);
	}

	/* Note VIDIOC_S_FMT may change width and height. */

	/* Buggy driver paranoia. */
sprintf(ERROR_STRING,"pix.width = %d",
fmt.fmt.pix.width);
advise(ERROR_STRING);
	min = fmt.fmt.pix.width * bytes_per_pixel;
	if(fmt.fmt.pix.bytesperline < min){
sprintf(ERROR_STRING,"bytesperline = %d, setting to min (%d)",
fmt.fmt.pix.bytesperline,min);
advise(ERROR_STRING);

		fmt.fmt.pix.bytesperline = min;
	}
	min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
	if(fmt.fmt.pix.sizeimage < min){
sprintf(ERROR_STRING,"sizeimage = %d, setting to min (%d)",
fmt.fmt.pix.sizeimage,min);
advise(ERROR_STRING);
		fmt.fmt.pix.sizeimage = min;
	}

	/* the next section came from init_mmap, capture.c */

	CLEAR(req);

	//req.count		= 8;			/* only 8??? */
	req.count		= MAX_BUFFERS_PER_DEVICE;		/* one second of buffering... */
	req.type		= V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory		= V4L2_MEMORY_MMAP;

	if( xioctl(vdp->vd_fd, VIDIOC_REQBUFS, &req) < 0 ){
		if(EINVAL == errno) {
			sprintf(ERROR_STRING, "%s does not support memory mapping", vdp->vd_name);
		} else {
			sprintf(ERROR_STRING,"VIDIOC_REQBUFS:  %s",strerror(errno));
		}
		WARN(ERROR_STRING);
		return(-1);
	}

	if(req.count < 2) {
		sprintf(ERROR_STRING, "Insufficient buffer memory on %s\n",
			vdp->vd_name);
		WARN(ERROR_STRING);
		return(-1);
	}
sprintf(ERROR_STRING,"Requested %d buffers, got %d",MAX_BUFFERS_PER_DEVICE,req.count);
advise(ERROR_STRING);

	vdp->vd_n_buffers = 0;

	// make sure data area is set to ram...
	curr_ap = ram_area_p;

	for(i_buffer = 0; i_buffer < req.count; ++i_buffer) {
		struct v4l2_buffer buf;
		char name[128];
		Dimension_Set dimset;
		Data_Obj *dp;

		CLEAR(buf);

		buf.type	= V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory	= V4L2_MEMORY_MMAP;
		buf.index	= i_buffer;

		if(-1 == xioctl( vdp->vd_fd, VIDIOC_QUERYBUF, &buf)){
			errno_warn(QSP_ARG  "VIDIOC_QUERYBUF");
			return(-1);
		}

		vdp->vd_buf_tbl[i_buffer].mb_length = buf.length;
		vdp->vd_buf_tbl[i_buffer].mb_vdp = vdp;
		vdp->vd_buf_tbl[i_buffer].mb_start =
			mmap(NULL /* start anywhere */,
				buf.length,
				PROT_READ | PROT_WRITE /* required */,
				MAP_SHARED /* recommended */,
				vdp->vd_fd, buf.m.offset);

		if(MAP_FAILED == vdp->vd_buf_tbl[i_buffer].mb_start){
			ERRNO_WARN("mmap");
			return(-1);
		}

		/* We create an object that points to this buffer in case
		 * we want to perform scripted operations involving vector
		 * expressions (e.g. real-time tracking); see flow.c
		 */

		sprintf(name,"%s.buffer%d",vdp->vd_name,i_buffer);
		/* BUG need to do these dynamically, might be using scaler */
		// Use the current pixel format
		switch( vfmt_list[vfmt_index].vfmt_code ){
			case V4L2_PIX_FMT_YUYV:
#ifdef FOOBAR
				dimset.ds_dimension[0]=4;	/* four bytes per pixel pair */
				dimset.ds_dimension[1]=320;	/* pixel pairs per row */
#endif // FOOBAR
				dimset.ds_dimension[0]=2;	/* two bytes per pixel - YU or YV */
				dimset.ds_dimension[1]=640;	/* pixels row */
				break;
			case V4L2_PIX_FMT_GREY:
				dimset.ds_dimension[0]=1;
				dimset.ds_dimension[1]=640;
				break;
			case V4L2_PIX_FMT_RGB24:
			case V4L2_PIX_FMT_BGR24:
				dimset.ds_dimension[0]=3;
				dimset.ds_dimension[1]=640;
				break;
			case V4L2_PIX_FMT_RGB32:
			case V4L2_PIX_FMT_BGR32:
				dimset.ds_dimension[0]=4;
				dimset.ds_dimension[1]=640;
				break;
			default:
				sprintf(ERROR_STRING,"Oops, haven't implemented buffer creation for %s pixel format!?",
					vfmt_list[vfmt_index].vfmt_name);
				WARN(ERROR_STRING);
				// default to YUYV
				dimset.ds_dimension[0]=4;	/* four bytes per pixel pair */
				dimset.ds_dimension[1]=320;	/* pixel pairs per row */
				break;
		}
		/* rows */
		dimset.ds_dimension[2]=vfld_tbl[vfld_index].vfld_height;

		dimset.ds_dimension[3]=1;
		dimset.ds_dimension[4]=1;
		dp = _make_dp(QSP_ARG  name,&dimset,PREC_FOR_CODE(PREC_UBY));
#ifdef CAUTIOUS
		if( dp == NO_OBJ ) ERROR1("CAUTIOUS:  error creating data_obj for video buffer");
#endif /* CAUTIOUS */
		SET_OBJ_DATA_PTR(dp, vdp->vd_buf_tbl[i_buffer].mb_start );

		vdp->vd_n_buffers ++;
	}
#endif // HAVE_V4L2
	return 0;
}

#define REPORT_FLAG(bit,set_string,clr_string)			\
	if( flags & bit )					\
		sprintf(msg_str,"\t\t0x%x\t%s",bit,set_string);	\
	else							\
		sprintf(msg_str,"\t\t0x0\t%s",clr_string);	\
	prt_msg(msg_str);					\
	flags &= ~(bit);

static void report_status(QSP_ARG_DECL  Video_Device *vdp)
{
	int flags;

	sprintf(msg_str,"Device %s:",vdp->vd_name);
	prt_msg(msg_str);
	sprintf(msg_str,"\tFlags:\t0x%x",vdp->vd_flags);
	prt_msg(msg_str);
	flags = vdp->vd_flags;

	REPORT_FLAG(VD_CAPTURING,"CAPTURING","not capturing")

#ifdef CAUTIOUS
	if( flags != 0 ){
		sprintf(ERROR_STRING,
	"CAUTIOUS:  report_status:  Flags for video device %s (0x%x) corrupted by unknown bits 0x%x!?",
			vdp->vd_name,vdp->vd_flags,flags);
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

}

/* based on start_capturing() from capture.c */

/* BUG?  should we test for the memory-mapped capability before doing this?? */

int start_capturing(QSP_ARG_DECL  Video_Device *vdp)
{
#ifdef HAVE_V4L2
	int i;
	enum v4l2_buf_type type;

	if( IS_CAPTURING( vdp ) ){
		sprintf(ERROR_STRING,"start_capturing:  Video device %s is already capturing!?",vdp->vd_name);
		WARN(ERROR_STRING);
		return(-1);
	}

	if( verbose ){
		sprintf(ERROR_STRING,"start_capturing:  starting video device %s.",vdp->vd_name);
		advise(ERROR_STRING);
	}
	/* Queue all the buffers, then start streaming... */

	for(i = 0; i < vdp->vd_n_buffers; ++i) {
		struct v4l2_buffer buf;

		CLEAR(buf);

		buf.type	= V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory	= V4L2_MEMORY_MMAP;
		buf.index	= i;

		if(-1 == xioctl(vdp->vd_fd, VIDIOC_QBUF, &buf))
			ERRNO_WARN("VIDIOC_QBUF #1");
	}

	vdp->vd_newest = vdp->vd_oldest = (-1);
		
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	if(-1 == xioctl(vdp->vd_fd, VIDIOC_STREAMON, &type))
		ERRNO_WARN("VIDIOC_STREAMON");

	vdp->vd_flags |= VD_CAPTURING;

#endif // HAVE_V4L2
	return 0;
}

#ifdef HAVE_V4L2
int dq_buf(QSP_ARG_DECL  Video_Device *vdp,struct v4l2_buffer *bufp)
{
	fd_set fds;
	struct timeval tv;
	int r;

	FD_ZERO(&fds);
	FD_SET(vdp->vd_fd, &fds);

	/* Timeout. */
	tv.tv_sec = 2;
	tv.tv_usec = 0;

	/* what is fd+1 all about??? */
	/* from the select man page:  "nfds is the number of the highest
	 * file descriptor, plus one"
	 */
	r = select(vdp->vd_fd + 1, &fds, NULL, NULL, &tv);

	if(r == -1) {
		/* original code repeated if EINTR */
		ERRNO_WARN("select");
		return(-1);
	}

	if( r == 0 ) {
		sprintf(ERROR_STRING, "select timeout");
		WARN(ERROR_STRING);
		return(-1);
	}


	//CLEAR (buf);

	bufp->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	bufp->memory = V4L2_MEMORY_MMAP;

	if( xioctl (vdp->vd_fd, VIDIOC_DQBUF, bufp) < 0 ) {
		/* original code had special cases for EAGAIN and EIO */
		ERRNO_WARN ("VIDIOC_DQBUF #2");		/* dq_buf */
		return(-1);
	}

#ifdef CAUTIOUS
	if( bufp->index >= (unsigned int) vdp->vd_n_buffers ){
		sprintf(ERROR_STRING,"CAUTIOUS:  Unexpected buffer number (%d) from VIDIOC_DQBUF, expected 0-%d",
			bufp->index,vdp->vd_n_buffers-1);
		WARN(ERROR_STRING);
		return(-1);
	}
#endif /* CAUTIOUS */

sprintf(ERROR_STRING,"Buffer %d de-queued",bufp->index);
advise(ERROR_STRING);

	return(0);
}

/* based on main_loop() and read_frame() in capture.c */

static void get_next_frame(QSP_ARG_DECL  Video_Device *vdp)
{
	struct v4l2_buffer buf;

	if( ! IS_CAPTURING(vdp) ){
		sprintf(ERROR_STRING,"get_next_frame:  Video device %s is not capturing!?",
			vdp->vd_name);
		WARN(ERROR_STRING);
		return;
	}

	if( dq_buf(QSP_ARG  vdp,&buf) < 0 ) return;	/* de-queue a buffer - release? */

	/* here is where we use the data... */

	/* presumably this call asks the driver to refill the buffer,
	 * Old comment said:
	 * this will be our "release" function...
	 * But this is not the release function, dq is the release func???
	 */
	if( xioctl(vdp->vd_fd, VIDIOC_QBUF, &buf) < 0 )
		ERRNO_WARN ("VIDIOC_QBUF #2");

} /* end get_next_frame */


int check_queue_status(QSP_ARG_DECL  Video_Device *vdp)
{
	int i;
	struct timeval oldest_time;
	struct timeval newest_time;
	int n_ready=0;

	if( ! IS_CAPTURING(vdp) ){
		sprintf(ERROR_STRING,"check_queue_status:  Video device %s is not capturing!?",vdp->vd_name);
		WARN(ERROR_STRING);
		return(-1);
	}

	/* flag that these have not been set yet */
	oldest_time.tv_sec = 0;
	newest_time.tv_sec = 0;
	/* quiet compiler */
	oldest_time.tv_usec = 0;
	newest_time.tv_usec = 0;


	for (i = 0; i < vdp->vd_n_buffers; ++i){
		struct v4l2_buffer buf;

		buf.type	= V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory	= V4L2_MEMORY_MMAP;
		buf.index	= i;

		if(-1 == xioctl( vdp->vd_fd, VIDIOC_QUERYBUF, &buf)){
			ERRNO_WARN("VIDIOC_QUERYBUF");
			return(-1);
		}

		if( buf.flags & V4L2_BUF_FLAG_DONE ){	/* data is ready */
			if( oldest_time.tv_sec == 0 ){
				oldest_time = buf.timestamp;
				newest_time = buf.timestamp;
				vdp->vd_oldest = vdp->vd_newest = i;
			} else {
				if( buf.timestamp.tv_sec < oldest_time.tv_sec ){
					oldest_time = buf.timestamp;
					vdp->vd_oldest = i;
				} else if( buf.timestamp.tv_sec == oldest_time.tv_sec &&
						buf.timestamp.tv_usec < oldest_time.tv_usec ){
					oldest_time = buf.timestamp;
					vdp->vd_oldest = i;
				} else if( buf.timestamp.tv_sec > newest_time.tv_sec ){
					newest_time = buf.timestamp;
					vdp->vd_newest = i;
				} else if( buf.timestamp.tv_sec == newest_time.tv_sec &&
						buf.timestamp.tv_usec > newest_time.tv_usec ){
					newest_time = buf.timestamp;
					vdp->vd_newest = i;
				}
			}
			n_ready++;
		}
	}
//sprintf(ERROR_STRING,"check_queue_status:  %d buffers are ready",n_ready);
//advise(ERROR_STRING);
	return(n_ready);
} // end check_queue_status

#ifdef NOT_YET
/* code from uninit_device() */

		for (i = 0; i < vdp->vd_n_buffers; ++i)
			if (-1 == munmap (buffers[i].mb_start, buffers[i].mb_length))
				errno_exit ("munmap");
#endif /* NOT_YET */

/* based on stop_capturing() */

int stop_capturing(QSP_ARG_DECL  Video_Device *vdp)
{
	enum v4l2_buf_type type;

	if( ! IS_CAPTURING(vdp) ){
		sprintf(ERROR_STRING,"stop_capturing:  Video device %s is not capturing!?",vdp->vd_name);
		WARN(ERROR_STRING);
		return(-1);
	}

	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	if( xioctl(vdp->vd_fd, VIDIOC_STREAMOFF, &type) < 0 ){
		ERRNO_WARN("VIDIOC_STREAMOFF");
		return(-1);
	}
	vdp->vd_flags &= ~VD_CAPTURING;

	if( verbose ){
		sprintf(ERROR_STRING,"Video device %s has been stopped.",
			vdp->vd_name);
		advise(ERROR_STRING);
	}
	return 0;
}
#endif // HAVE_V4L2

/* based on open_device() from capture.c */

static int open_video_device(QSP_ARG_DECL  const char *dev_name)
{
	struct stat st; 
	Video_Device *vdp;
	int fd;

	/* first make sure this device is not already open */
	vdp = video_dev_of(QSP_ARG  dev_name);
	if( vdp != NO_VIDEO_DEVICE ){
		sprintf(ERROR_STRING,"open_video_device:  device %s is already open!?",dev_name);
		WARN(ERROR_STRING);
		return(-1);
	}

	if( stat(dev_name, &st) < 0 ) {
		sprintf(ERROR_STRING, "Cannot identify '%s': %d, %s\n",
			dev_name, errno, strerror( errno));
		WARN(ERROR_STRING);
		return -1;
	}

	if( !S_ISCHR( st.st_mode)) {
		sprintf(ERROR_STRING, "%s is no device\n", dev_name);
		WARN(ERROR_STRING);
		return -1;
	}

	fd = open( dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

	if( -1 == fd) {
		sprintf(ERROR_STRING, "Cannot open '%s': %d, %s\n",
			dev_name, errno, strerror( errno));
		WARN(ERROR_STRING);
		return -1;
	}

	vdp = new_video_dev(QSP_ARG  dev_name);
#ifdef CAUTIOUS
	if( vdp == NO_VIDEO_DEVICE ){
		sprintf(ERROR_STRING,"CAUTIOUS:  open_video_device:  unable to create new Video_Device struct for %s",dev_name);
		WARN(ERROR_STRING);
		return(-1);
	}
#endif /* CAUTIOUS */

	vdp->vd_name = savestr(dev_name);
	vdp->vd_fd = fd;

	vdp->vd_flags = 0;
	vdp->vd_n_inputs = 0;
	vdp->vd_n_standards = 0;
	vdp->vd_input_choices = NULL;
	vdp->vd_std_choices = NULL;

	/* init_video_device does the v4l2 initializations */
	init_video_device(QSP_ARG  vdp);

	curr_vdp = vdp;

	return 0;
}


static COMMAND_FUNC( do_open )
{
	const char *s;

	s=NAMEOF("video device");
	if( open_video_device(QSP_ARG  s) < 0 ){
		sprintf(ERROR_STRING,"Error opening video device %s",s);
		WARN(ERROR_STRING);
	}
}

static COMMAND_FUNC( do_select )
{
	Video_Device *vdp;

	vdp = PICK_VIDEO_DEV("");
	if( vdp == NO_VIDEO_DEVICE ) return;

	curr_vdp = vdp;
}

/* The status function was introduced when we ran into problems resulting
 * from a failure to initialize the flags after creating the device struct.
 * Now all it does is report the flags, and die if any unknown bits are set.
 * Ideally it should report everything...
 */

static COMMAND_FUNC( do_status )
{
	CHECK_DEVICE
	report_status(QSP_ARG  curr_vdp);
}

static COMMAND_FUNC( do_start )
{
	CHECK_DEVICE
	start_capturing(QSP_ARG  curr_vdp);
}

static COMMAND_FUNC( do_stop )
{
	CHECK_DEVICE
#ifdef HAVE_V4L2
	stop_capturing(QSP_ARG  curr_vdp);
#endif // HAVE_V4L2
}

static COMMAND_FUNC( do_next )
{
	CHECK_DEVICE
#ifdef HAVE_V4L2
	get_next_frame(QSP_ARG  curr_vdp);
#endif // HAVE_V4L2
}

static COMMAND_FUNC( do_yuv2gray )
{
	Data_Obj *dst_dp, *src_dp;

	dst_dp = PICK_OBJ("destination GRAY image");
	src_dp = PICK_OBJ("source YUYV image");

	if( dst_dp == NO_OBJ || src_dp == NO_OBJ )
		return;

	/* BUG Here we need to check sizes, etc */

	yuv422_to_gray(QSP_ARG  dst_dp,src_dp);
}

#ifdef HAVE_V4L2
static int query_control(QSP_ARG_DECL  struct v4l2_queryctrl *ctlp)
{
	if( ioctl(curr_vdp->vd_fd,VIDIOC_QUERYCTRL,ctlp) < 0 ){
		WARN("error querying control");
		return(-1);
	}

	/* Should be ctrl_info or something... */

	/* the structure should now have the range of values... */
	switch(ctlp->type){
		case V4L2_CTRL_TYPE_INTEGER:
			sprintf(ERROR_STRING,"%s, integer control %d - %d",ctlp->name,ctlp->minimum,ctlp->maximum); break;
		case V4L2_CTRL_TYPE_MENU:
			sprintf(ERROR_STRING,"%s, menu control",ctlp->name); break;
		case V4L2_CTRL_TYPE_BOOLEAN:
			sprintf(ERROR_STRING,"%s, boolean control",ctlp->name); break;
		case V4L2_CTRL_TYPE_BUTTON:
			sprintf(ERROR_STRING,"%s, button control",ctlp->name); break;
#ifdef CAUTIOUS
		default: sprintf(ERROR_STRING,"CAUTIOUS:  unknown control"); break;
#endif /* CAUTIOUS */
	}
	advise(ERROR_STRING);
	return(0);
}

static void set_integer_control(QSP_ARG_DECL uint32_t id)
{
	uint32_t v;
	char prompt[LLEN];
	struct v4l2_queryctrl qry;
	struct v4l2_control ctrl;

	qry.id = id;

	if( query_control(QSP_ARG  &qry) < 0 ) return;

	ctrl.id = id;
	if( ioctl(curr_vdp->vd_fd,VIDIOC_G_CTRL,&ctrl) < 0 ){
		sprintf(ERROR_STRING,"error getting current %s setting",qry.name);
		WARN(ERROR_STRING);
		return;
	}
	sprintf(ERROR_STRING,"Current value of %s is %d",qry.name,ctrl.value);
	advise(ERROR_STRING);

	/* BUG assumes integer control */
	sprintf(prompt,"%s (%d - %d)",qry.name,qry.minimum,qry.maximum);
	v = HOW_MANY(prompt);

	ctrl.value = v;

	if( ioctl(curr_vdp->vd_fd,VIDIOC_S_CTRL,&ctrl) < 0 ){
		sprintf(ERROR_STRING,"error getting current %s setting",qry.name);
		WARN(ERROR_STRING);
		return;
	}
}

static int get_integer_control(QSP_ARG_DECL uint32_t id)
{
	struct v4l2_queryctrl qry;
	struct v4l2_control ctrl;
	int r;

	qry.id = id;

	if( query_control(QSP_ARG  &qry) < 0 ) return(0);

	ctrl.id = id;
	if( ioctl(curr_vdp->vd_fd,VIDIOC_G_CTRL,&ctrl) < 0 ){
		sprintf(ERROR_STRING,"error getting current %s setting",qry.name);
		WARN(ERROR_STRING);
		return(0);
	}
	r= ctrl.value;

	return r;
}
#endif // HAVE_V4L2

#ifdef HAVE_V4L2

#define SET_INTEGER_CONTROL(control)					\
	set_integer_control(QSP_ARG  control);

#else // ! HAVE_V4L2
#define SET_INTEGER_CONTROL(control)					\
									\
{									\
	int i;								\
	i = HOW_MANY("dummy control value");				\
	sprintf(ERROR_STRING,						\
	"program not configured with V4L2 support, can't set %s!?",#control);	\
	WARN(ERROR_STRING);						\
}

#endif // ! HAVE_V4L2

static COMMAND_FUNC( do_set_hue )
{
	CHECK_DEVICE
	SET_INTEGER_CONTROL(V4L2_CID_HUE);
}

static COMMAND_FUNC( do_set_bright )
{
	CHECK_DEVICE
	SET_INTEGER_CONTROL(V4L2_CID_BRIGHTNESS);
}

static COMMAND_FUNC( do_set_contrast )
{
	CHECK_DEVICE
	SET_INTEGER_CONTROL(V4L2_CID_CONTRAST);
}

static COMMAND_FUNC( do_set_saturation )
{
	CHECK_DEVICE
	SET_INTEGER_CONTROL(V4L2_CID_SATURATION);
}


#ifdef HAVE_V4L2
static void do_get_control( QSP_ARG_DECL const char *varname, int ctl_index )
{
	int v;

	CHECK_DEVICE
	v=get_integer_control(QSP_ARG ctl_index);
	sprintf(msg_str,"%d",v);
	ASSIGN_VAR(varname,msg_str);
}
#endif // HAVE_V4L2

static COMMAND_FUNC( do_get_hue )
{
	const char *s;

	s=NAMEOF("variable name for hue setting");
#ifdef HAVE_V4L2
	do_get_control(QSP_ARG s,V4L2_CID_HUE);
#else // ! HAVE_V4L2
	ASSIGN_VAR(s,"0");
#endif // ! HAVE_V4L2
}

static COMMAND_FUNC( do_get_bright )
{
	const char *s;

	s=NAMEOF("variable name for brightness setting");
#ifdef HAVE_V4L2
	do_get_control(QSP_ARG s,V4L2_CID_BRIGHTNESS);
#else // ! HAVE_V4L2
	ASSIGN_VAR(s,"0");
#endif // ! HAVE_V4L2
}

static COMMAND_FUNC( do_get_contrast )
{
	const char *s;

	s=NAMEOF("variable name for contrast setting");
#ifdef HAVE_V4L2
	do_get_control(QSP_ARG s,V4L2_CID_CONTRAST);
#else // ! HAVE_V4L2
	ASSIGN_VAR(s,"0");
#endif // ! HAVE_V4L2
}

static COMMAND_FUNC( do_get_saturation )
{
	const char *s;

	s=NAMEOF("variable name for saturation setting");
#ifdef HAVE_V4L2
	do_get_control(QSP_ARG s,V4L2_CID_SATURATION);
#else // ! HAVE_V4L2
	ASSIGN_VAR(s,"0");
#endif // ! HAVE_V4L2
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(video_controls_menu,s,f,h)

MENU_BEGIN(video_controls)
ADD_CMD( set_hue,		do_set_hue,		adjust hue )
ADD_CMD( set_brightness,	do_set_bright,		adjust picture brightness )
ADD_CMD( set_contrast,		do_set_contrast,	adjust picture contrast )
ADD_CMD( set_saturation,	do_set_saturation,	adjust picture saturation )
ADD_CMD( get_hue,		do_get_hue,		fetch hue )
ADD_CMD( get_brightness,	do_get_bright,		fetch picture brightness )
ADD_CMD( get_contrast,		do_get_contrast,	fetch picture contrast )
ADD_CMD( get_saturation,	do_get_saturation,	fetch picture saturation )
MENU_END(video_controls)

static COMMAND_FUNC( do_vctl_menu )
{
	PUSH_MENU(video_controls);
}

static COMMAND_FUNC( do_report_input )
{
#ifdef HAVE_V4L2
	struct v4l2_input input;
	int index;

	CHECK_DEVICE

	if (-1 == ioctl (curr_vdp->vd_fd, VIDIOC_G_INPUT, &index)) {
        	perror ("VIDIOC_G_INPUT");
	        exit (EXIT_FAILURE);
	}

	memset (&input, 0, sizeof (input));
	input.index = index;

	if (-1 == ioctl (curr_vdp->vd_fd, VIDIOC_ENUMINPUT, &input)) {
	        perror ("VIDIOC_ENUMINPUT");
	        exit (EXIT_FAILURE);
	}

	printf ("Current input: %s\n", input.name);
#endif // HAVE_V4L2
}

static COMMAND_FUNC( do_list_inputs )
{
#ifdef HAVE_V4L2
	struct v4l2_input input;

	CHECK_DEVICE

	memset (&input, 0, sizeof (input));
	input.index = 0;

	printf ("Inputs:\n");
	while( ioctl (curr_vdp->vd_fd, VIDIOC_ENUMINPUT, &input) == 0 ) {
		printf ("\t%s\n", input.name);
		input.index ++;
	}
	if( errno != EINVAL ){
	        perror ("VIDIOC_ENUMINPUT");
	        exit (EXIT_FAILURE);
	}
#endif // HAVE_V4L2
}
				      
static COMMAND_FUNC( do_list_stds )
{
#ifdef HAVE_V4L2
	struct v4l2_input input;
	struct v4l2_standard standard;

	CHECK_DEVICE

	memset (&input, 0, sizeof (input));

	if (-1 == ioctl (curr_vdp->vd_fd, VIDIOC_G_INPUT, &input.index)) {
		perror ("VIDIOC_G_INPUT");
		exit (EXIT_FAILURE);
	}

	if (-1 == ioctl (curr_vdp->vd_fd, VIDIOC_ENUMINPUT, &input)) {
		perror ("VIDIOC_ENUM_INPUT");
		exit (EXIT_FAILURE);
	}

	printf ("Current input %s supports:\n", input.name);

	memset (&standard, 0, sizeof (standard));
	standard.index = 0;

	while (0 == ioctl (curr_vdp->vd_fd, VIDIOC_ENUMSTD, &standard)) {
		if (standard.id & input.std)
			printf ("%s\n", standard.name);
		standard.index++;
	}

	/* EINVAL indicates the end of the enumeration, which cannot be
	   empty unless this device falls under the USB exception. */

	if (errno != EINVAL || standard.index == 0) {
		perror ("VIDIOC_ENUMSTD");
		exit (EXIT_FAILURE);
	}
#endif // HAVE_V4L2
}


#ifdef HAVE_V4L2
static int count_standards(QSP_ARG_DECL  Video_Device *vdp)
{
	struct v4l2_standard standard;

	CHECK_DEVICE2

	memset (&standard, 0, sizeof (standard));
	standard.index = 0;

	while (0 == ioctl (curr_vdp->vd_fd, VIDIOC_ENUMSTD, &standard)) {
		standard.index++;
	}

	/* EINVAL indicates the end of the enumeration, which cannot be
	   empty unless this device falls under the USB exception. */

	if (errno != EINVAL || standard.index == 0) {
		perror ("VIDIOC_ENUMSTD");
		exit (EXIT_FAILURE);
	}

	return(standard.index);
}

static void init_std_choices(QSP_ARG_DECL  Video_Device *vdp)
{
	int i;
	struct v4l2_standard standard;
	Choice *cp;

	vdp->vd_n_standards = count_standards(QSP_ARG  vdp);

	if( vdp->vd_std_choices != NULL )
		givbuf(vdp->vd_std_choices);

	vdp->vd_std_choices = (Choice *)getbuf( sizeof(Choice)*vdp->vd_n_standards );

	memset (&standard, 0, sizeof (standard));

	cp = vdp->vd_std_choices;
	for(i=0;i<vdp->vd_n_standards;i++){
		standard.index = i;
		if( ioctl (curr_vdp->vd_fd, VIDIOC_ENUMSTD, &standard) < 0 ) {
			perror("VIDIOC_ENUMSTD");
		} else {
			cp->ch_name = savestr((const char *)standard.name);
			cp->ch_id = standard.id;
			cp++;
		}
	}
}

static void set_standard( QSP_ARG_DECL  int id )
{
	v4l2_std_id std_id;
	struct v4l2_input input;

	memset (&input, 0, sizeof (input));

	if (-1 == ioctl (curr_vdp->vd_fd, VIDIOC_G_INPUT, &input.index)) {
        	perror ("VIDIOC_G_INPUT");
	        exit (EXIT_FAILURE);
	}

	if (-1 == ioctl (curr_vdp->vd_fd, VIDIOC_ENUMINPUT, &input)) {
	        perror ("VIDIOC_ENUM_INPUT");
	        exit (EXIT_FAILURE);
	}

//sprintf(ERROR_STRING,"set_standard:  input.std = 0x%lx",input.std);
////advise(ERROR_STRING);

	if( (input.std & id) == 0 ){
		sprintf(ERROR_STRING,"Oops, %s input does not support requested standard",
			input.name);
		WARN(ERROR_STRING);
		return;
	}

	std_id = id;

	if (-1 == ioctl (curr_vdp->vd_fd, VIDIOC_S_STD, &std_id)) {
        	perror ("VIDIOC_S_STD");
	        exit (EXIT_FAILURE);
	}
}
#endif // HAVE_V4L2

static COMMAND_FUNC( do_set_std )
{
#ifdef HAVE_V4L2
	int i;
	const char **choices;

	CHECK_DEVICE

	if( curr_vdp->vd_n_standards <= 0 )
		init_std_choices(QSP_ARG  curr_vdp);
	
	choices = (const char **)getbuf( sizeof(char *) *
					curr_vdp->vd_n_standards );

	for(i=0;i<curr_vdp->vd_n_standards;i++)
		choices[i] = savestr(curr_vdp->vd_std_choices[i].ch_name);

	i = WHICH_ONE("video standard",curr_vdp->vd_n_standards,choices);

	set_standard(QSP_ARG  curr_vdp->vd_std_choices[i].ch_id );

	for(i=0;i<curr_vdp->vd_n_standards;i++)
		rls_str(choices[i]);
	givbuf(choices);
#else // ! HAVE_V4L2
	const char *s;
	s=NAMEOF("standard");	// dummy word to throw away
	// print warning
#endif // ! HAVE_V4L2
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(standards_menu,s,f,h)

MENU_BEGIN(standards)
//ADD_CMD( info,		do_std_info,		report current info about input and standard )
//ADD_CMD( set_input,		do_set_input,		select video input )
ADD_CMD( set_standard,	do_set_std,		select video standard )
ADD_CMD( list,		do_list_stds,		list available standards )
ADD_CMD( report_input,	do_report_input,	report current input device )
ADD_CMD( list_inputs,	do_list_inputs,		list all input devices )
MENU_END(standards)

static COMMAND_FUNC( do_std_menu )
{
	PUSH_MENU(standards);
}

static COMMAND_FUNC( do_downsample )
{
	Data_Obj *dst_dp, *src_dp;

	dst_dp = PICK_OBJ("destination object");
	src_dp = PICK_OBJ("source object");
	if( dst_dp == NO_OBJ || src_dp == NO_OBJ ) return;
	fast_downsample(dst_dp,src_dp);
}

#ifdef RECORD_TIMESTAMPS
static COMMAND_FUNC( do_dump_ts )
{
	const char *s;

	s=NAMEOF("filename for timestamp data");
	dump_timestamps(s);
}
#endif /* RECORD_TIMESTAMPS */


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(stream_menu,s,f,h)

MENU_BEGIN(stream)
ADD_CMD( record,	do_stream_record,	record video to raw volume )
ADD_CMD( wait,		wait_record,		wait for current recording to finish )
ADD_CMD( halt,		halt_record,		halt current recording )
#ifdef RECORD_TIMESTAMPS
ADD_CMD( timestamps,	do_dump_ts,		dump timestamps )
ADD_CMD( grab_times,	print_grab_times,	print grab times )
ADD_CMD( store_times,	print_grab_times,	print store times )
#endif /* RECORD_TIMESTAMPS */
MENU_END(stream)

debug_flag_t stream_debug=0;

static COMMAND_FUNC( do_stream_menu )
{
	PUSH_MENU(stream);
}

static COMMAND_FUNC( do_list_devs )
{
	List *lp;
	Node *np;
	Video_Device *vdp;

	if( video_dev_itp == NO_ITEM_TYPE ){
		advise("do_list_devs:  no video devices have been opened.");
		return;
	}
	lp = item_list(QSP_ARG  video_dev_itp);
	np = lp->l_head;
	while( np != NO_NODE ){
		vdp = (Video_Device *)np->n_data;
		report_status(QSP_ARG  vdp);
		np=np->n_next;
	}
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(v4l2_menu,s,f,h)

MENU_BEGIN(v4l2)
ADD_CMD( format,	set_vfmt,	set pixel format )
ADD_CMD( field_mode,	set_field_mode,	set field mode )
ADD_CMD( open,		do_open,	open video device )
ADD_CMD( list,		do_list_devs,	list open video devices & statuses )
ADD_CMD( status,	do_status,	report status of  current video device )
ADD_CMD( start,		do_start,	start capturing )
ADD_CMD( stop,		do_stop,	stop capturing )
ADD_CMD( next,		do_next,	capture next frame )
ADD_CMD( select,	do_select,	select device )
ADD_CMD( yuv2rgb,	do_yuv2rgb,	convert from YUYV to RGB )
ADD_CMD( yuv2gray,	do_yuv2gray,	convert from YUYV to GRAY )
ADD_CMD( downsample,	do_downsample,	fast downsampling )
ADD_CMD( controls,	do_vctl_menu,	video controls submenu )
ADD_CMD( standards,	do_std_menu,	video standards submenu )
ADD_CMD( flow,		do_flow_menu,	frame-by-frame capture submenu )
ADD_CMD( stream,	do_stream_menu,	streaming capture submenu )
MENU_END(v4l2)

COMMAND_FUNC( do_v4l2_menu )
{
	static int inited=0;

	if( ! inited ){
		stream_debug=add_debug_module(QSP_ARG  "stream_record");
#ifdef HAVE_RAWVOL
		if( insure_default_rv(SINGLE_QSP_ARG) < 0 ){
			WARN("error opening default raw volume");
		}
#endif // HAVE_RAWVOL

		/* FIXME put analogous stuff here for lml board,
		 * someday write movie module...
		 */
		//init_rv_movies();
		//meteor_init();
		//load_movie_module(&meteor_movie_module);

		inited=1;

	}

	PUSH_MENU(v4l2);
}

