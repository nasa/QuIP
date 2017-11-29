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


Video_Device *curr_vdp=NULL;

ITEM_INTERFACE_DECLARATIONS(Video_Device,video_dev,0)

/* Call ioctl, repeating if interrupted */

int xioctl(int fd, int request, void *arg)
{
	int r;

	do {
//fprintf(stderr,"xiotcl calling ioctl\n");
		r = ioctl(fd, request, arg);
	} while( r == -1 && errno== EINTR );
//fprintf(stderr,"xiotcl returning %d\n",r);

	return r;
}

void errno_warn(QSP_ARG_DECL  const char *s)
{
	sprintf(ERROR_STRING,"%s:  %s",s,strerror(errno));
	warn(ERROR_STRING);
}

void print_buf_info(const char *msg, My_Buffer *mbp)
{
	fprintf(stderr,"%s  buf %d   flags =",msg,mbp->mb_index);
	if( mbp->mb_flags == 0 ){
		fprintf(stderr," 0\n");
		return;
	}

#ifdef HAVE_V4L2
	if( mbp->mb_flags & V4L2_BUF_FLAG_MAPPED )
		fprintf(stderr," mapped");
	if( mbp->mb_flags & V4L2_BUF_FLAG_QUEUED )
		fprintf(stderr," queued");
	if( mbp->mb_flags & V4L2_BUF_FLAG_DONE )
		fprintf(stderr," done");
#else // ! HAVE_V4L2
	fprintf(stderr," (v4l2 not present)");
#endif // ! HAVE_V4L2
	fprintf(stderr,"\n");
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

static int vfmt_index=DEFAULT_VFMT_INDEX;	// BUG why is this global???
static const char *vfmt_names[N_VIDEO_FORMATS];
static int vfmt_names_inited=0;

#ifdef HAVE_V4L2
static const char *name_for_type(int t)
{
	char *s;
	switch(t){
		case V4L2_BUF_TYPE_VIDEO_CAPTURE:
			return "video_capture";
			break;
		default:
			s="unhandled type code";
			break;
	}
	return s;
}

static const char *name_for_mem(int t)
{
	char *s;
	switch(t){
		case V4L2_MEMORY_MMAP:
			return "memory_map";
			break;
		default:
			s="unhandled mem code";
			break;
	}
	return s;
}

static void print_v4l2_buf_info(struct v4l2_buffer *bufp)
{
	fprintf(stderr,"buf at 0x%lx, type = %s, mem = %s, index = %d\n",
		(long)bufp,name_for_type(bufp->type),name_for_mem(bufp->memory),bufp->index);
}

static inline void get_name_for_buffer(char *name,My_Buffer *mbp)
{
	sprintf(name,"%s.buffer%d",mbp->mb_vdp->vd_name,mbp->mb_index);
}
#endif // HAVE_V4L2

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

#ifdef HAVE_V4L2

#define setup_buffer_dimensions(dsp) _setup_buffer_dimensions(QSP_ARG  dsp)

static void _setup_buffer_dimensions(QSP_ARG_DECL  Dimension_Set *dsp)
{
	/* BUG need to do these dynamically, might be using scaler */
	// Also, this assume NTSC standard!?!?
	// Use the current pixel format
	switch( vfmt_list[vfmt_index].vfmt_code ){
		case V4L2_PIX_FMT_YUYV:
			dsp->ds_dimension[0]=2;	/* two bytes per pixel - YU or YV */
			dsp->ds_dimension[1]=640;	/* pixels row */
			break;
		case V4L2_PIX_FMT_GREY:
			dsp->ds_dimension[0]=1;
			dsp->ds_dimension[1]=640;
			break;
		case V4L2_PIX_FMT_RGB24:
		case V4L2_PIX_FMT_BGR24:
			dsp->ds_dimension[0]=3;
			dsp->ds_dimension[1]=640;
			break;
		case V4L2_PIX_FMT_RGB32:
		case V4L2_PIX_FMT_BGR32:
			dsp->ds_dimension[0]=4;
			dsp->ds_dimension[1]=640;
			break;
		default:
			sprintf(ERROR_STRING,"Oops, haven't implemented buffer creation for %s pixel format!?",
				vfmt_list[vfmt_index].vfmt_name);
			warn(ERROR_STRING);
			// default to YUYV
			dsp->ds_dimension[0]=4;	/* four bytes per pixel pair */
			dsp->ds_dimension[1]=320;	/* pixel pairs per row */
			break;
	}
	/* rows */
	dsp->ds_dimension[2]=vfld_tbl[vfld_index].vfld_height;

	dsp->ds_dimension[3]=1;
	dsp->ds_dimension[4]=1;
}


// init_video_buffer handles the device driver calls

#define init_video_buffer(mbp) _init_video_buffer(QSP_ARG  mbp )

static inline int _init_video_buffer(QSP_ARG_DECL  My_Buffer *mbp )
{
	CLEAR(mbp->mb_buf);

	mbp->mb_buf.type	= V4L2_BUF_TYPE_VIDEO_CAPTURE;
	mbp->mb_buf.memory	= V4L2_MEMORY_MMAP;
	mbp->mb_buf.index	= mbp->mb_index;

	if(-1 == xioctl( mbp->mb_vdp->vd_fd, VIDIOC_QUERYBUF, &(mbp->mb_buf))){
		errno_warn(QSP_ARG  "VIDIOC_QUERYBUF");
		return -1;
	}

	mbp->mb_start =
		mmap(	NULL,				/* start anywhere */
			mbp->mb_buf.length,
			PROT_READ | PROT_WRITE,		/* required */
			MAP_SHARED,			/* recommended */
			mbp->mb_vdp->vd_fd,
			mbp->mb_buf.m.offset);

	if(MAP_FAILED == mbp->mb_start){
		ERRNO_WARN("mmap");
		return -1;
	}
	return 0;
}

#define create_object_for_buffer(mbp, dsp ) _create_object_for_buffer(QSP_ARG  mbp, dsp )

static void _create_object_for_buffer(QSP_ARG_DECL  My_Buffer *mbp, Dimension_Set *dsp )
{
	char name[128];
	Data_Obj *dp;

	get_name_for_buffer(name,mbp);
	dp = _make_dp(QSP_ARG  name,dsp,PREC_FOR_CODE(PREC_UBY));
#ifdef CAUTIOUS
	if( dp == NULL ) error1("CAUTIOUS:  error creating data_obj for video buffer");
#endif /* CAUTIOUS */
	point_obj_to_ext_data(dp, mbp->mb_start );
	mbp->mb_dp = dp;
}

#define setup_one_buffer(vdp, i_buffer, dsp ) _setup_one_buffer(QSP_ARG  vdp, i_buffer, dsp )

static int _setup_one_buffer(QSP_ARG_DECL  Video_Device *vdp, int i_buffer, Dimension_Set *dsp )
{
	My_Buffer *mbp;

	mbp = &(vdp->vd_buf_tbl[i_buffer]);
	mbp->mb_vdp = vdp;
	mbp->mb_index = i_buffer;
	mbp->mb_flags = 0;

	if( init_video_buffer(mbp) < 0 )
		return -1;

	/* We create an object that points to this buffer in case
	 * we want to perform scripted operations involving vector
	 * expressions (e.g. real-time tracking); see flow.c
	 */

	create_object_for_buffer(mbp,dsp);
	return 0;
}

#define dealloc_one_buffer(mbp) _dealloc_one_buffer(QSP_ARG  mbp)

static int _dealloc_one_buffer(QSP_ARG_DECL  My_Buffer *mbp )
{
	int status;

fprintf(stderr,"Deallocating buffer %d\n",mbp->mb_index);
	assert(mbp->mb_dp!=NULL);
	delvec(mbp->mb_dp);

	status = munmap(mbp->mb_start,mbp->mb_buf.length);
	if( status < 0 ){
		tell_sys_error("munmap");
		return -1;
	}
	return 0;
}

#define try_buffer_request(msg,vdp,reqp) _try_buffer_request(QSP_ARG  msg,vdp,reqp)

static inline int _try_buffer_request(QSP_ARG_DECL  const char *msg, Video_Device *vdp, struct v4l2_requestbuffers *reqp)
{
fprintf(stderr,"try_buffer_request, mem = 0x%x   type = 0x%x    count = %d\n",reqp->memory,reqp->type,reqp->count);
	if( xioctl(vdp->vd_fd, VIDIOC_REQBUFS, reqp) < 0 ){
		if(EINVAL == errno) {
			sprintf(ERROR_STRING, "%s does not support %s", vdp->vd_name,msg);
			advise(ERROR_STRING);
		} else {
			sprintf(ERROR_STRING,"VIDIOC_REQBUFS (try_buffer_request %s):  %s",
				msg,strerror(errno));
			warn(ERROR_STRING);
		}
		return -1;
	}
fprintf(stderr,"try_buffer_request, returned count = %d\n",reqp->count);
	return 0;
}

#define release_mmap_buffers(vdp) _release_mmap_buffers(QSP_ARG  vdp)

static inline void _release_mmap_buffers(QSP_ARG_DECL  Video_Device *vdp)
{
	struct v4l2_requestbuffers req;

	CLEAR(req);
	req.type		= V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory		= V4L2_MEMORY_MMAP;
	req.count		= 0;	// release the buffers
fprintf(stderr,"release_mmap_buffers calling try_buffer_request\n");
	if( try_buffer_request("memory-mapped buffers",vdp,&req) < 0 )
		warn("Error releasing memory-mapped buffers!?");
	else {
		//vdp->vd_flags &= ~VD_USING_MMAP_BUFFERS;
		vdp->vd_n_buffers = 0;
	}
}

#define request_mmap_buffers(vdp, nreq) _request_mmap_buffers(QSP_ARG  vdp, nreq)

static inline int _request_mmap_buffers(QSP_ARG_DECL  Video_Device *vdp, int nreq)
{
	struct v4l2_requestbuffers req;

	assert(CAN_USE_MMAP_BUFFERS(vdp));

	CLEAR(req);
	req.type		= V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory		= V4L2_MEMORY_MMAP;
	req.count		= nreq;

fprintf(stderr,"request_mmap_buffers calling try_buffer_request\n");
	if( try_buffer_request("memory-mapped buffers",vdp,&req) == 0 ){
		vdp->vd_flags |= VD_USING_MMAP_BUFFERS;
		vdp->vd_flags &= ~VD_USING_USERSPACE_BUFFERS;
		if( req.count == nreq ){
			sprintf(MSG_STR,"%d buffers allocated",nreq);
		} else {
			sprintf(MSG_STR,"%d buffers requested, but %d allocated",nreq,req.count);
		}
		advise(MSG_STR);
		return req.count;
	}
	return -1;
}

#define request_userspace_buffers(vdp ) _request_userspace_buffers(QSP_ARG  vdp )

static inline int _request_userspace_buffers(QSP_ARG_DECL  Video_Device *vdp )
{
	struct v4l2_requestbuffers req;

	assert(CAN_USE_USERSPACE_BUFFERS(vdp));

	if( IS_USING_MMAP_BUFFERS(vdp) ){
fprintf(stderr,"request_userspace_buffers calling release_mmap_buffers\n");
		release_mmap_buffers(vdp);
		assert( ! IS_USING_MMAP_BUFFERS(vdp) );
	}

	CLEAR(req);
	req.type		= V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory		= V4L2_MEMORY_USERPTR;

fprintf(stderr,"request_userspace_buffers calling try_buffer_request\n");
	if( try_buffer_request("user-space buffers",vdp,&req) == 0 ){
		vdp->vd_flags |= VD_USING_USERSPACE_BUFFERS;
		vdp->vd_flags &= ~VD_USING_MMAP_BUFFERS;
	}

	return -1;
}

#ifdef NOT_USED

// Find out if we can use user-space buffers...

#define discover_buffer_options(vdp) _discover_buffer_options(QSP_ARG  vdp)

static void _discover_buffer_options(QSP_ARG_DECL  Video_Device *vdp)
{
	// The devices we have don't seem to support this, so why bother?
//	vdp->vd_flags |= VD_SUPPORTS_USERSPACE_BUFFERS;	// trial to pass assertion
//	if( request_userspace_buffers(vdp) < 0 )
//		vdp->vd_flags &= ~VD_SUPPORTS_USERSPACE_BUFFERS;	// remember this

	vdp->vd_flags |= VD_SUPPORTS_MMAP_BUFFERS;	// trial to pass assertion
fprintf(stderr,"discover_buffer_options calling request_mmap_buffers\n");
	if( request_mmap_buffers(vdp,DEFAULT_N_VIDEO_BUFFERS) < 0 )
		vdp->vd_flags &= ~VD_SUPPORTS_MMAP_BUFFERS;	// remember this

	if( IS_USING_MMAP_BUFFERS(vdp) ){
fprintf(stderr,"discover_buffer_options NOT calling release_mmap_buffers\n");
		//release_mmap_buffers(vdp);
	}
}
#endif // NOT_USED

#define check_capabilities(vdp) _check_capabilities(QSP_ARG  vdp)

static int _check_capabilities(QSP_ARG_DECL  Video_Device *vdp)
{
	struct v4l2_capability cap;
#ifdef 	CHECK_CROPCAP
	struct v4l2_cropcap cropcap;
	struct v4l2_crop crop;
#endif

	if(-1 == xioctl(vdp->vd_fd, VIDIOC_QUERYCAP, &cap)) {
		if( errno == EINVAL ){
			sprintf(ERROR_STRING, "%s is not a V4L2 device!?", vdp->vd_name);
		} else {
			sprintf(ERROR_STRING,"VIDIOC_QUERYCAP:  %s",strerror(errno));
		}
		warn(ERROR_STRING);
		return -1;
	}

	if(!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
		sprintf(ERROR_STRING,"%s does not have video capture capability",vdp->vd_name);
		warn(ERROR_STRING);
		return -1;
	}

	if(!(cap.capabilities & V4L2_CAP_STREAMING)) {
		sprintf(ERROR_STRING,"%s does not support streaming i/o",vdp->vd_name);
		warn(ERROR_STRING);
		return -1;
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
				warn("cropping not supported");
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

	return 0;
} // end check_capabilities

#define dealloc_buffers(vdp) _dealloc_buffers(QSP_ARG  vdp)

static void _dealloc_buffers(QSP_ARG_DECL  Video_Device *vdp)
{
	int i_buffer;

	for(i_buffer=0;i_buffer<vdp->vd_n_buffers;i_buffer++){
		My_Buffer *mbp;
		mbp = &(vdp->vd_buf_tbl[i_buffer]);
		dealloc_one_buffer(mbp);
	}
	vdp->vd_n_buffers = 0;
}

#define init_buffer_objects(vdp, n) _init_buffer_objects(QSP_ARG  vdp, n)

static inline int _init_buffer_objects(QSP_ARG_DECL  Video_Device *vdp, int n)
{
	Dimension_Set dimset;
	unsigned int i_buffer;

	if( vdp->vd_n_buffers > 0 )
		dealloc_buffers(vdp);

	vdp->vd_n_buffers = n;

	// make sure data area is set to ram...
	// BUG should push and pop!?
	curr_ap = ram_area_p;

	setup_buffer_dimensions(&dimset);

	for(i_buffer = 0; i_buffer < n; ++i_buffer) {
		if( setup_one_buffer(vdp,i_buffer,&dimset) < 0 ){
			warn("error setting up buffer");
			return -1;
		}
	}
	return 0;
}

#define setup_buffers(vdp,n) _setup_buffers(QSP_ARG  vdp,n)

static int _setup_buffers(QSP_ARG_DECL  Video_Device *vdp, int nreq)
{
	int n;

	if( (n=request_mmap_buffers(curr_vdp, nreq)) < 0 )
		return -1;

	if( init_buffer_objects(curr_vdp,n) < 0 )
		return -1;

	// We know how many buffers we have, make an array of buffer info
	// structs

	return 0;
}


#define setup_video_format(vdp) _setup_video_format(QSP_ARG  vdp)

static int _setup_video_format(QSP_ARG_DECL  Video_Device *vdp)
{
	struct v4l2_format fmt;
	int bytes_per_pixel;
	unsigned int min;


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
		warn(ERROR_STRING);
		return -1;
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
	return 0;
}

#endif // HAVE_V4L2

/* based on init_device from capture.c */

/* CHECK_CROPCAP is maybe here to retain some code we didn't orignally need? */

#define init_video_device(vdp,fd) _init_video_device(QSP_ARG  vdp,fd)

static int _init_video_device(QSP_ARG_DECL  Video_Device *vdp, int fd)
{
#ifdef HAVE_V4L2

	vdp->vd_fd = fd;
	vdp->vd_flags = 0;
	vdp->vd_n_inputs = 0;
	vdp->vd_n_standards = 0;
	vdp->vd_n_buffers = 0;
	vdp->vd_input_choices = NULL;
	vdp->vd_std_choices = NULL;

	if( check_capabilities(vdp) < 0 )
		return -1;

	if( setup_video_format(vdp) < 0 )
		return -1;

//fprintf(stderr,"NOT calling discover_buffer_options!\n");
//	discover_buffer_options(vdp);

//	if( setup_buffers(vdp) < 0 )
//		return -1;
	// Don't call this here, make the user set the number of buffers in the script.
	// That is because we are getting EBUSY when trying to reset after changing
	// the interlace mode or pixel format!?

	// Instead, assume all is well
	vdp->vd_flags |= VD_SUPPORTS_MMAP_BUFFERS;	// trial to pass assertion

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
		error1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

}

#ifdef HAVE_V4L2

#define enqueue_buffer(mbp) _enqueue_buffer(QSP_ARG  mbp)

static inline int _enqueue_buffer(QSP_ARG_DECL  My_Buffer *mbp)
{
	struct v4l2_buffer *bufp;

//	Buffer fields should already be set???
//	CLEAR(mbp->mb_buf);
	bufp = &(mbp->mb_buf);

	assert(bufp->type	== V4L2_BUF_TYPE_VIDEO_CAPTURE );
	assert(bufp->memory	== V4L2_MEMORY_MMAP );
	assert(bufp->index	== mbp->mb_index );

	if(-1 == xioctl(mbp->mb_vdp->vd_fd, VIDIOC_QBUF, bufp)){
		ERRNO_WARN("VIDIOC_QBUF (enqueue_buffer)");
		return -1;
	}
	// documentation says MAPPED and QUEUED flags should be set,
	// but this does not seem to be the case!?!?
if( debug & v4l2_debug ){
fprintf(stderr,"enqueue_buffer:  buffer %d queued, flgs = 0x%x\n",
bufp->index,bufp->flags);
}

	// We keep our own flags because the driver doesn't seem
	// to do this correctly!?
	mbp->mb_flags |= V4L2_BUF_FLAG_QUEUED;
if( debug & v4l2_debug ){
print_buf_info("enqueue_buffer",mbp);
}

	return 0;
}
#endif // HAVE_V4L2

/* based on start_capturing() from capture.c */

/* BUG?  should we test for the memory-mapped capability before doing this?? */

int _start_capturing(QSP_ARG_DECL  Video_Device *vdp)
{
#ifdef HAVE_V4L2
	int i;
	enum v4l2_buf_type type;

	if( IS_CAPTURING( vdp ) ){
		sprintf(ERROR_STRING,"start_capturing:  Video device %s is already capturing!?",vdp->vd_name);
		warn(ERROR_STRING);
		return -1;
	}

	if( verbose ){
		sprintf(ERROR_STRING,"start_capturing:  starting video device %s.",vdp->vd_name);
		advise(ERROR_STRING);
	}
if( debug & v4l2_debug ){
fprintf(stderr,"start_capturing:  starting video device %s.",vdp->vd_name);
}
	/* Queue all the buffers, then start streaming... */

	for(i = 0; i < vdp->vd_n_buffers; ++i) {
		My_Buffer *mbp;
		mbp = &(vdp->vd_buf_tbl[i]);
		if( enqueue_buffer(mbp) < 0 )
			return -1;
	}

	vdp->vd_newest_mbp = vdp->vd_oldest_mbp = NULL;
		
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	if(-1 == xioctl(vdp->vd_fd, VIDIOC_STREAMON, &type))
		ERRNO_WARN("VIDIOC_STREAMON");

	vdp->vd_flags |= VD_CAPTURING;

#endif // HAVE_V4L2
	return 0;
} // start_capturing

#ifdef HAVE_V4L2

#define wait_for_video_data(vdp) _wait_for_video_data(QSP_ARG  vdp)

static inline int _wait_for_video_data(QSP_ARG_DECL  Video_Device *vdp)
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
		return -1;
	}

	if( r == 0 ) {
		sprintf(ERROR_STRING, "select timeout");
		warn(ERROR_STRING);
		return -1;
	}
	return 0;
}

int dq_buf(QSP_ARG_DECL  Video_Device *vdp,struct v4l2_buffer *bufp)
{
	if( wait_for_video_data(vdp) < 0 )
		return -1;

	bufp->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	bufp->memory = V4L2_MEMORY_MMAP;

if( debug & v4l2_debug ){
fprintf(stderr,"dq_buf:  dequeueing buffer %d at 0x%lx, flags = 0x%x\n",
bufp->index,(long)bufp,bufp->flags);
}
	if( xioctl (vdp->vd_fd, VIDIOC_DQBUF, bufp) < 0 ) {
		/* original code had special cases for EAGAIN and EIO */
		ERRNO_WARN ("VIDIOC_DQBUF #2");		/* dq_buf */
		return -1;
	}
if( debug & v4l2_debug ){
fprintf(stderr,"dq_buf:  dequeued buffer %d at 0x%lx, flags = 0x%x\n",
bufp->index,(long)bufp,bufp->flags);
}

#ifdef CAUTIOUS
	if( bufp->index >= (unsigned int) vdp->vd_n_buffers ){
		sprintf(ERROR_STRING,"CAUTIOUS:  Unexpected buffer number (%d) from VIDIOC_DQBUF, expected 0-%d",
			bufp->index,vdp->vd_n_buffers-1);
		warn(ERROR_STRING);
		return -1;
	}
#endif /* CAUTIOUS */

sprintf(ERROR_STRING,"Buffer %d (of %d)  de-queued",bufp->index,vdp->vd_n_buffers);
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
		warn(ERROR_STRING);
		return;
	}

	if( dq_buf(QSP_ARG  vdp,&buf) < 0 ) return;	/* de-queue a buffer - release? */
print_v4l2_buf_info(&buf);
}

static void enqueue_indexed_buffer(QSP_ARG_DECL  Video_Device *vdp, int idx)
{
	struct v4l2_buffer buf;

	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.index = idx;

	if( xioctl(vdp->vd_fd, VIDIOC_QBUF, &buf) < 0 )
		ERRNO_WARN ("VIDIOC_QBUF (enqueue_indexed_buffer)");

}

static inline int buffer_is_ready(My_Buffer *mbp)
{
	if( mbp->mb_flags & V4L2_BUF_FLAG_DONE )
		return 1;
	else
		return 0;
}

static inline void note_dequeued(Video_Device *vdp, int idx)
{
	My_Buffer *mbp;

	assert(idx>=0 && idx < vdp->vd_n_buffers);
	mbp = &(vdp->vd_buf_tbl[idx]);
	mbp->mb_flags |= V4L2_BUF_FLAG_DONE;
if( debug & v4l2_debug ){
print_buf_info("note_dequeued",mbp);
}

}

#define dequeue_ready_buffers(mbp) _dequeue_ready_buffers(QSP_ARG  mbp)

static inline int _dequeue_ready_buffers(QSP_ARG_DECL  My_Buffer *mbp)
{
	struct v4l2_buffer buf;
	int n=0;

	do {
		CLEAR(buf);
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE ;
		buf.memory = V4L2_MEMORY_MMAP ;
		if( xioctl (mbp->mb_vdp->vd_fd, VIDIOC_DQBUF, &buf) < 0 ) {
			/* original code had special cases for EAGAIN and EIO */
			ERRNO_WARN ("VIDIOC_DQBUF (dequeue_ready_buffers)");		/* dq_buf */
			return -1;
		}
if( debug & v4l2_debug ){
fprintf(stderr,"update_buf_status:  Buffer %d is done, dequeued buffer %d\n",mbp->mb_index,buf.index);
}
		// The dequeued buffer may not be the one that is done now!?
		note_dequeued(mbp->mb_vdp,buf.index);
		n++;
	} while( buf.index != mbp->mb_index );
	return n;
}

// update_buf_status:
// return values:
//      N	buffer ready and N buffers dequeued
// 	1	buffer ready and dequeued
// 	0	buffer not ready
// 	-1	error

#define update_buf_status(mbp) _update_buf_status(QSP_ARG  mbp)

static inline int _update_buf_status(QSP_ARG_DECL  My_Buffer *mbp )
{
	struct v4l2_buffer *bufp;

	bufp = &(mbp->mb_buf);

//	bufp->type	= V4L2_BUF_TYPE_VIDEO_CAPTURE;
//	bufp->memory	= V4L2_MEMORY_MMAP;
//	bufp->index	= buf_idx;
	assert(	bufp->type	== V4L2_BUF_TYPE_VIDEO_CAPTURE );
	assert(	bufp->memory	== V4L2_MEMORY_MMAP );
	assert(	bufp->index	== mbp->mb_index );

	if(-1 == xioctl( mbp->mb_vdp->vd_fd, VIDIOC_QUERYBUF, bufp)){
		ERRNO_WARN("VIDIOC_QUERYBUF");
		return -1;
	}
//fprintf(stderr,"update_buf_status:  buffer %d flags = 0x%x\n",bufp->index,bufp->flags);
	if( bufp->flags & V4L2_BUF_FLAG_DONE ){	/* data is ready */
		mbp->mb_flags |= V4L2_BUF_FLAG_DONE;	// ready flag
if( debug & v4l2_debug ){
print_buf_info("update_buf_status",mbp);
}
		return dequeue_ready_buffers(mbp);
	} else {
		return 0;
	}
}

static inline int is_before( struct timeval *tsp1, struct timeval *tsp2 )
{
	if( tsp1->tv_sec < tsp2->tv_sec )
		return 1;
	if( tsp1->tv_sec == tsp2->tv_sec && tsp1->tv_usec < tsp2->tv_usec )
		return 1;
	return 0;
}

static inline int is_after( struct timeval *tsp1, struct timeval *tsp2 )
{
	if( tsp1->tv_sec > tsp2->tv_sec )
		return 1;
	if( tsp1->tv_sec == tsp2->tv_sec && tsp1->tv_usec > tsp2->tv_usec )
		return 1;
	return 0;
}

static inline void check_newest(My_Buffer *mbp)
{
	struct v4l2_buffer *bufp;
	Video_Device *vdp;

	bufp = &(mbp->mb_buf);
	assert(bufp->index == mbp->mb_index);

	vdp = mbp->mb_vdp;

//	assert( (bufp->flags & V4L2_BUF_FLAG_DONE) != 0 );
//	The done flag here is cleared when the buffer is dequeued...
	assert( (mbp->mb_flags & V4L2_BUF_FLAG_DONE) != 0 );

	if( vdp->vd_oldest_mbp == NULL ){
		vdp->vd_oldest_mbp = vdp->vd_newest_mbp = mbp;
		return;
	}

	if( mbp == vdp->vd_oldest_mbp || mbp == vdp->vd_newest_mbp ) return;

if( debug & v4l2_debug ){
fprintf(stderr,"check_newest:  comparing old newest buffer %d with buffer %d\n",
vdp->vd_newest_mbp->mb_index,bufp->index);
}
	if( is_after(&(bufp->timestamp),&(vdp->vd_newest_mbp->mb_buf.timestamp)) ){
		vdp->vd_newest_mbp = mbp;
if( debug & v4l2_debug ){
fprintf(stderr,"check_newest:  newest buffer reset to %d\n",mbp->mb_index);
}
	}
}

static inline void check_oldest(My_Buffer *mbp)
{
	struct v4l2_buffer *bufp;
	Video_Device *vdp;

	bufp = &(mbp->mb_buf);
	assert(bufp->index == mbp->mb_index);
	vdp = mbp->mb_vdp;

	//assert( (bufp->flags & V4L2_BUF_FLAG_DONE) != 0 );
	assert( (mbp->mb_flags & V4L2_BUF_FLAG_DONE) != 0 );

	if( vdp->vd_oldest_mbp == NULL ){
		vdp->vd_oldest_mbp = vdp->vd_newest_mbp = mbp;
		return;
	}

	if( mbp == vdp->vd_oldest_mbp || mbp == vdp->vd_newest_mbp ) return;

if( debug & v4l2_debug ){
fprintf(stderr,"check_oldest:  comparing old oldest buffer %d with buffer %d\n",
vdp->vd_oldest_mbp->mb_index,bufp->index);
}
	if( is_before(&(bufp->timestamp),&(vdp->vd_oldest_mbp->mb_buf.timestamp)) ){
		vdp->vd_oldest_mbp = mbp;
if( debug & v4l2_debug ){
fprintf(stderr,"check_oldest:  oldest buffer reset to %d\n",mbp->mb_index);
}
	}
}

static void find_newest_buffer(Video_Device *vdp)
{
	int buf_idx;

	for (buf_idx = 0; buf_idx < vdp->vd_n_buffers; ++buf_idx){
		My_Buffer *mbp;
		mbp = &(vdp->vd_buf_tbl[buf_idx]);
		if( buffer_is_ready(mbp) )
			check_newest(mbp);
	}
if( debug & v4l2_debug ){
if( vdp->vd_newest_mbp != NULL )
fprintf(stderr,"find_newest_buffer:  newest buffer is %d\n",vdp->vd_newest_mbp->mb_index);
}
//else
//fprintf(stderr,"find_newest_buffer:  no newest buffer\n");
}

void _find_oldest_buffer(QSP_ARG_DECL  Video_Device *vdp)
{
	int buf_idx;

	for (buf_idx = 0; buf_idx < vdp->vd_n_buffers; ++buf_idx){
		My_Buffer *mbp;
		mbp = &(vdp->vd_buf_tbl[buf_idx]);
		if( buffer_is_ready(mbp) )
			check_oldest(mbp);
	}
if( debug & v4l2_debug ){
if( vdp->vd_oldest_mbp != NULL )
fprintf(stderr,"find_oldest_buffer:  oldest buffer is %d\n",vdp->vd_oldest_mbp->mb_index);
//else
//fprintf(stderr,"find_oldest_buffer:  no oldest buffer\n");
}

}

// returns the number of available (dequeued) frames

int check_queue_status(QSP_ARG_DECL  Video_Device *vdp)
{
	int buf_idx;
	int n_ready=0;
	int status;

	if( ! IS_CAPTURING(vdp) ){
		sprintf(ERROR_STRING,"check_queue_status:  Video device %s is not capturing!?",vdp->vd_name);
		warn(ERROR_STRING);
		return -1;
	}


	for (buf_idx = 0; buf_idx < vdp->vd_n_buffers; ++buf_idx){
		My_Buffer *mbp;
		mbp = &(vdp->vd_buf_tbl[buf_idx]);
		if( (status=update_buf_status(mbp)) < 0 )
			return -1;
		n_ready += status;
	}

	find_newest_buffer(vdp);
	find_oldest_buffer(vdp);

	return n_ready;
} // end check_queue_status

#ifdef NOT_YET
/* code from uninit_device() */

		for (i = 0; i < vdp->vd_n_buffers; ++i)
			if (-1 == munmap (buffers[i].mb_start, buffers[i].mb_length))
				errno_exit ("munmap");
#endif /* NOT_YET */

/* based on stop_capturing() */

int _stop_capturing(QSP_ARG_DECL  Video_Device *vdp)
{
	enum v4l2_buf_type type;

	if( ! IS_CAPTURING(vdp) ){
		sprintf(ERROR_STRING,"stop_capturing:  Video device %s is not capturing!?",vdp->vd_name);
		warn(ERROR_STRING);
		return -1;
	}

	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	if( xioctl(vdp->vd_fd, VIDIOC_STREAMOFF, &type) < 0 ){
		ERRNO_WARN("VIDIOC_STREAMOFF");
		return -1;
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
	vdp = video_dev_of(dev_name);
	if( vdp != NO_VIDEO_DEVICE ){
		sprintf(ERROR_STRING,"open_video_device:  device %s is already open!?",dev_name);
		warn(ERROR_STRING);
		return -1;
	}

	if( stat(dev_name, &st) < 0 ) {
		sprintf(ERROR_STRING, "Cannot identify '%s': %d, %s\n",
			dev_name, errno, strerror( errno));
		warn(ERROR_STRING);
		return -1;
	}

	if( !S_ISCHR( st.st_mode)) {
		sprintf(ERROR_STRING, "%s is no device\n", dev_name);
		warn(ERROR_STRING);
		return -1;
	}

	fd = open( dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

	if( -1 == fd) {
		sprintf(ERROR_STRING, "Cannot open '%s': %d, %s\n",
			dev_name, errno, strerror( errno));
		warn(ERROR_STRING);
		return -1;
	}

	vdp = new_video_dev(dev_name);
#ifdef CAUTIOUS
	if( vdp == NO_VIDEO_DEVICE ){
		sprintf(ERROR_STRING,"CAUTIOUS:  open_video_device:  unable to create new Video_Device struct for %s",dev_name);
		warn(ERROR_STRING);
		return -1;
	}
#endif /* CAUTIOUS */

	/* init_video_device does the v4l2 initializations */
	init_video_device(vdp,fd);

	curr_vdp = vdp;

	return 0;
}


#ifdef HAVE_V4L2

#define close_video_device(vdp) _close_video_device(QSP_ARG  vdp)

static int _close_video_device(QSP_ARG_DECL  Video_Device *vdp)
{
	// Release the buffer objects
	dealloc_buffers(vdp);

	if( close(vdp->vd_fd) < 0 ){
		tell_sys_error("close");
		return -1;
	}

	del_video_dev(vdp);

	return 0;
}

#endif // HAVE_V4L2


static COMMAND_FUNC( do_open )
{
	const char *s;

	s=NAMEOF("video device");
	if( open_video_device(QSP_ARG  s) < 0 ){
		sprintf(ERROR_STRING,"Error opening video device %s",s);
		warn(ERROR_STRING);
	}
}

static COMMAND_FUNC( do_close )
{
	Video_Device *vdp;

	vdp = pick_video_dev("");
	if( vdp == NULL ) return;

#ifdef HAVE_V4L2
	close_video_device(vdp);
#endif // HAVE_V4L2
}

static COMMAND_FUNC( do_select_device )
{
	Video_Device *vdp;

	vdp = pick_video_dev("");
	if( vdp == NULL ) return;

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
	start_capturing(curr_vdp);
}

static COMMAND_FUNC( do_stop )
{
	CHECK_DEVICE
#ifdef HAVE_V4L2
	stop_capturing(curr_vdp);
#endif // HAVE_V4L2
}

static COMMAND_FUNC( do_dq_next )
{
	CHECK_DEVICE
#ifdef HAVE_V4L2
	get_next_frame(QSP_ARG  curr_vdp);
#endif // HAVE_V4L2
}

static COMMAND_FUNC( do_q_buf )
{
	int idx;

	CHECK_DEVICE

	idx = how_many("buffer index");
	// BUG check for valid value
#ifdef HAVE_V4L2
	enqueue_indexed_buffer(QSP_ARG  curr_vdp, idx);
#endif // HAVE_V4L2
}

static COMMAND_FUNC( do_yuv2gray )
{
	Data_Obj *dst_dp, *src_dp;

	dst_dp = pick_obj("destination GRAY image");
	src_dp = pick_obj("source YUYV image");

	if( dst_dp == NULL || src_dp == NULL )
		return;

	/* BUG Here we need to check sizes, etc */

	yuv422_to_gray(QSP_ARG  dst_dp,src_dp);
}

#ifdef HAVE_V4L2
static int query_control(QSP_ARG_DECL  struct v4l2_queryctrl *ctlp)
{
	if( ioctl(curr_vdp->vd_fd,VIDIOC_QUERYCTRL,ctlp) < 0 ){
		warn("error querying control");
		return -1;
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
		warn(ERROR_STRING);
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
		warn(ERROR_STRING);
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
		warn(ERROR_STRING);
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
	NO_V4L2_MSG(#control,i)						\
}

#define NO_V4L2_MSG(label,value)	_NO_V4L2_MSG(label,value)

#define _NO_V4L2_MSG(label,value)					\
									\
	sprintf(ERROR_STRING,						\
	"program not configured with V4L2 support, can't set %s to %d!?",label,value);	\
	warn(ERROR_STRING);

#define NO_V4L2_MSG2(label,string)					\
									\
	sprintf(ERROR_STRING,						\
	"program not configured with V4L2 support, can't set %s to %s!?",label,string);	\
	warn(ERROR_STRING);

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
	assign_var(varname,msg_str);
}
#endif // HAVE_V4L2

static COMMAND_FUNC( do_get_hue )
{
	const char *s;

	s=NAMEOF("variable name for hue setting");
#ifdef HAVE_V4L2
	do_get_control(QSP_ARG s,V4L2_CID_HUE);
#else // ! HAVE_V4L2
	assign_var(s,"0");
#endif // ! HAVE_V4L2
}

static COMMAND_FUNC( do_get_bright )
{
	const char *s;

	s=NAMEOF("variable name for brightness setting");
#ifdef HAVE_V4L2
	do_get_control(QSP_ARG s,V4L2_CID_BRIGHTNESS);
#else // ! HAVE_V4L2
	assign_var(s,"0");
#endif // ! HAVE_V4L2
}

static COMMAND_FUNC( do_get_contrast )
{
	const char *s;

	s=NAMEOF("variable name for contrast setting");
#ifdef HAVE_V4L2
	do_get_control(QSP_ARG s,V4L2_CID_CONTRAST);
#else // ! HAVE_V4L2
	assign_var(s,"0");
#endif // ! HAVE_V4L2
}

static COMMAND_FUNC( do_get_saturation )
{
	const char *s;

	s=NAMEOF("variable name for saturation setting");
#ifdef HAVE_V4L2
	do_get_control(QSP_ARG s,V4L2_CID_SATURATION);
#else // ! HAVE_V4L2
	assign_var(s,"0");
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
	CHECK_AND_PUSH_MENU(video_controls);
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
		warn(ERROR_STRING);
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
	NO_V4L2_MSG2("standard",s)
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
	CHECK_AND_PUSH_MENU(standards);
}

static COMMAND_FUNC( do_downsample )
{
	Data_Obj *dst_dp, *src_dp;

	dst_dp = pick_obj("destination object");
	src_dp = pick_obj("source object");
	if( dst_dp == NULL || src_dp == NULL ) return;
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

debug_flag_t v4l2_debug=0;

static COMMAND_FUNC( do_stream_menu )
{
	CHECK_AND_PUSH_MENU(stream);
}

static COMMAND_FUNC( do_list_devs )
{
	List *lp;
	Node *np;
	Video_Device *vdp;

	if( video_dev_itp == NULL ){
		advise("do_list_devs:  no video devices have been opened.");
		return;
	}
	lp = item_list(video_dev_itp);
	np = QLIST_HEAD(lp);
	while( np != NULL ){
		vdp = (Video_Device *)np->n_data;
		report_status(QSP_ARG  vdp);
		np=np->n_next;
	}
}

static COMMAND_FUNC(do_set_n_buffers)
{
	int n;

	n = how_many("number of buffers");

#ifdef HAVE_V4L2
	if( curr_vdp->vd_n_buffers > 0 )
		dealloc_buffers(curr_vdp);

	if( setup_buffers(curr_vdp,n) < 0 )
		warn("Unable to allocate buffers!?");
#endif // HAVE_V4L2
}



#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(v4l2_menu,s,f,h)

MENU_BEGIN(v4l2)
ADD_CMD( format,	set_vfmt,	set pixel format )
ADD_CMD( field_mode,	set_field_mode,	set field mode )
ADD_CMD( set_n_buffers,	do_set_n_buffers,	request buffer pool )
ADD_CMD( open,		do_open,	open video device )
ADD_CMD( close,		do_close,	close video device )
ADD_CMD( list,		do_list_devs,	list open video devices & statuses )
ADD_CMD( status,	do_status,	report status of  current video device )
ADD_CMD( start,		do_start,	start capturing )
ADD_CMD( stop,		do_stop,	stop capturing )
ADD_CMD( dequeue_next,	do_dq_next,	get next captured frame )
ADD_CMD( enqueue_buf,	do_q_buf,	release a buffer to be filled )
ADD_CMD( select,	do_select_device,	select device )
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
		v4l2_debug=add_debug_module("v4l2");
#ifdef HAVE_RAWVOL
		if( insure_default_rv(SINGLE_QSP_ARG) < 0 ){
			warn("error opening default raw volume");
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

	CHECK_AND_PUSH_MENU(v4l2);
}

