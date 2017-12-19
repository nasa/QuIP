#include "quip_config.h"

/* Like continuous capture to memory, but we use the handshaking
 * (like in stream record)
 * to make sure our application is synchronized.
 *
 * This file was hacked from the version in new_meteor.  We use flow control
 * when we need to interact with the data in real time (e.g., tracking with
 * a robotic camera).  Otherwise we might just stream to disk...
 */

#ifdef HAVE_UNISTD_H
#include <unistd.h>			/* usleep */
#endif

#ifdef HAVE_STRING_H
#include <string.h>			/* memset */
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#include "quip_prot.h"
#include "data_obj.h"
#include "img_file.h"
#include "debug.h"
#include "my_video_dev.h"
#include "my_v4l2.h"
#include "veclib_api.h"	// do_yuv2rgb

#ifdef HAVE_V4L2

#define oldest	curr_vdp->vd_oldest
#define newest	curr_vdp->vd_newest

#endif /* HAVE_V4L2 */

static COMMAND_FUNC( do_stop_flow )
{
	CHECK_DEVICE

	if( ! IS_CAPTURING(curr_vdp) ){
		warn("do_stop_flow:  Current video device is not capturing!?");
		return;
	}

#ifdef HAVE_V4L2
	if( stop_capturing(curr_vdp) < 0 )
		WARN("error stopping capture");
#endif
}

#ifdef HAVE_V4L2

#define update_index_var(stem,mbp) _update_index_var(QSP_ARG  stem,mbp)

static inline void _update_index_var(QSP_ARG_DECL  const char *stem, My_Buffer *mbp)
{
	char varname[128];
	char val[32];

	sprintf(varname,"%s.%s",stem,curr_vdp->vd_name);
	if( mbp == NULL ){
		assign_var(varname,"-1");
	} else {
		sprintf(val,"%d",mbp->mb_index);
		assign_var(varname,val);
	}
}

static COMMAND_FUNC( update_vars )
{
	char val[32];
	char varname[128];
	int n;

	n=check_queue_status(QSP_ARG  curr_vdp);

	sprintf(val,"%d",n);
	sprintf(varname,"n_ready.%s",curr_vdp->vd_name);
	assign_var(varname,val);

	update_index_var("oldest",curr_vdp->vd_oldest_mbp);
	update_index_var("newest",curr_vdp->vd_newest_mbp);
}

#endif // HAVE_V4L2

static COMMAND_FUNC( do_start_flow )
{
	CHECK_DEVICE

	if( IS_CAPTURING(curr_vdp) ){
		warn("do_start_flow:  Current video device is already capturing!?");
		return;
	}
	if( ! HAS_BUFFERS(curr_vdp) ){
		warn("do_start_flow:  Current video device has no buffers!?");
		return;
	}
	/* start capturing - oldest & newest are initialized withing start_capturing now */

#ifdef HAVE_V4L2
	if( start_capturing(curr_vdp) < 0 )
		WARN("error starting capture");

	update_vars(SINGLE_QSP_ARG);
#endif // HAVE_V4L2
}

static COMMAND_FUNC( wait_next )	/* wait til we have another frame */
{
#ifdef HAVE_V4L2
	int n,m;

	CHECK_DEVICE

	if( ! IS_CAPTURING(curr_vdp) ){
		sprintf(ERROR_STRING,"wait_next:  Video device %s is not capturing!?",
			curr_vdp->vd_name);
		WARN(ERROR_STRING);
		return;
	}

	n = check_queue_status(QSP_ARG  curr_vdp);
	if( n == curr_vdp->vd_n_buffers ){
		sprintf(ERROR_STRING,
	"wait_next:  device %s already has all of its %d buffers ready!?",
			curr_vdp->vd_name,curr_vdp->vd_n_buffers);
		WARN(ERROR_STRING);
		return;
	}

	do {
		m = check_queue_status(QSP_ARG  curr_vdp);
	} while( m == n );

if( debug & v4l2_debug ){
fprintf(stderr,"wait_next:  %d buffers available\n",m);
}
	update_vars(SINGLE_QSP_ARG);
#endif
} // end wait_next

static COMMAND_FUNC( wait_drip )	/* wait til we have at least one frame */
{
#ifdef HAVE_V4L2
	int n=0;

	CHECK_DEVICE

	if( ! IS_CAPTURING(curr_vdp) ){
		sprintf(ERROR_STRING,"wait_drip:  Video device %s is not capturing!?",
			curr_vdp->vd_name);
		WARN(ERROR_STRING);
		return;
	}

	do {
		n = check_queue_status(QSP_ARG  curr_vdp);
	} while( n == 0 );

	update_vars(SINGLE_QSP_ARG);
#endif
} // end wait_drip

#ifdef HAVE_V4L2

#define release_buffer(mbp) _release_buffer(QSP_ARG  mbp)

static inline void _release_buffer(QSP_ARG_DECL  My_Buffer *mbp)
{
	struct v4l2_buffer *bufp;

if( debug & v4l2_debug ){
print_buf_info("release_buffer",mbp);
}
	assert(mbp->mb_flags & V4L2_BUF_FLAG_DONE);
	bufp = &(mbp->mb_vdp->vd_oldest_mbp->mb_buf);

	assert(bufp->type == V4L2_BUF_TYPE_VIDEO_CAPTURE);
	assert(bufp->memory == V4L2_MEMORY_MMAP);
	assert(bufp->index == mbp->mb_index);

//	if( xioctl(mbp->mb_vdp->vd_fd, VIDIOC_DQBUF, bufp ) < 0 )
//		ERRNO_WARN ("VIDIOC_DQBUF (release_oldest_buffer)");
//fprintf(stderr,"release_buffer:  after dequeueing buffer %d, flags = 0x%x\n",
//bufp->index,bufp->flags);
	if( xioctl(mbp->mb_vdp->vd_fd, VIDIOC_QBUF, bufp ) < 0 )
		ERRNO_WARN ("VIDIOC_QBUF (release_oldest_buffer)");
//if( debug & v4l2_debug ){
//fprintf(stderr,"release_buffer:  after enqueued buffer %d, flags = 0x%x\n",
//bufp->index,bufp->flags);
//}

	mbp->mb_flags &= ~V4L2_BUF_FLAG_DONE;
}

void release_oldest_buffer(QSP_ARG_DECL  Video_Device *vdp)
{
	if( vdp->vd_oldest_mbp == NULL ){
		WARN("release_oldest_buffer:  no oldest buffer to release!?");
		return;
	}
if( debug & v4l2_debug ){
fprintf(stderr,"release_oldest_buffer releasing buffer %d\n",vdp->vd_oldest_mbp->mb_index);
}
	release_buffer(vdp->vd_oldest_mbp);

	if( vdp->vd_newest_mbp == vdp->vd_oldest_mbp ){
		vdp->vd_oldest_mbp=vdp->vd_newest_mbp=NULL;
	} else {
if( debug & v4l2_debug ){
fprintf(stderr,"release_oldest_buffer calling find_oldest_buffer\n");
}
		vdp->vd_oldest_mbp=vdp->vd_newest_mbp;
		find_oldest_buffer(vdp);
if( debug & v4l2_debug ){
assert(vdp->vd_oldest_mbp!=NULL);
fprintf(stderr,"release_oldest_buffer new oldest_buffer is %d\n",vdp->vd_oldest_mbp->mb_index);
}
	}
}
#endif /* HAVE_V4L2 */

static COMMAND_FUNC( do_release_buffer )
{
#ifdef HAVE_V4L2
	CHECK_DEVICE

	if( ! IS_CAPTURING(curr_vdp) ){
		sprintf(ERROR_STRING,"wait_next:  Video device %s is not capturing!?",
			curr_vdp->vd_name);
		WARN(ERROR_STRING);
		return;
	}

	release_oldest_buffer(QSP_ARG  curr_vdp);
	update_vars(SINGLE_QSP_ARG);
#endif /* HAVE_V4L2 */
}


#define ADD_CMD(s,f,h)	ADD_COMMAND(flow_menu,s,f,h)

MENU_BEGIN(flow)
ADD_CMD( start,		do_start_flow,		start capture )
ADD_CMD( stop,		do_stop_flow,		start capture )
ADD_CMD( wait,		wait_drip,		wait for at least one frame in memory )
ADD_CMD( next,		wait_next,		wait for the next frame )
ADD_CMD( release,	do_release_buffer,	release oldest buffer )
ADD_CMD( yuv2rgb,	do_yuv2rgb,		convert from YUYV to RGB )
MENU_END(flow)

COMMAND_FUNC( do_flow_menu )
{
	CHECK_AND_PUSH_MENU(flow);
}

