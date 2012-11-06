#include "quip_config.h"

char VersionId_v4l2_flow[] = QUIP_VERSION_STRING;


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

#include "query.h"
#include "data_obj.h"
#include "img_file.h"
#include "my_video_dev.h"
#include "my_v4l2.h"

#include "debug.h"	/* verbose */

#ifdef HAVE_V4L2

#define oldest	curr_vdp->vd_oldest
#define newest	curr_vdp->vd_newest

#else

#define CHECK_DEVICE

#endif /* ! HAVE_V4L2 */

COMMAND_FUNC( start_flow )
{
	CHECK_DEVICE

	/* start capturing - oldest & newest are initialized withing start_capturing now */

#ifdef HAVE_V4L2
	if( start_capturing(QSP_ARG  curr_vdp) < 0 )
		WARN("error starting capture");
#endif
}

COMMAND_FUNC( stop_flow )
{
	CHECK_DEVICE

	/* start capturing - oldest & newest are initialized withing start_capturing now */

#ifdef HAVE_V4L2
	if( stop_capturing(QSP_ARG  curr_vdp) < 0 )
		WARN("error stopping capture");
#endif
}

#ifdef HAVE_V4L2
static COMMAND_FUNC( update_vars )
{

	char s[32];
	char varname[64];
	int n;

	n=check_queue_status(QSP_ARG  curr_vdp);

	sprintf(s,"%d",n);
	sprintf(varname,"n_ready.%s",curr_vdp->vd_name);
	ASSIGN_VAR(varname,s);

	sprintf(s,"%d",oldest);
	sprintf(varname,"oldest.%s",curr_vdp->vd_name);
	ASSIGN_VAR(varname,s);

	sprintf(s,"%d",newest);
	sprintf(varname,"newest.%s",curr_vdp->vd_name);
	ASSIGN_VAR(varname,s);
}
#endif

COMMAND_FUNC( wait_next )	/* wait til we have another frame */
{
#ifdef HAVE_V4L2
	int n,m;

	CHECK_DEVICE

	if( ! IS_CAPTURING(curr_vdp) ){
		sprintf(error_string,"wait_next:  Video device %s is not capturing!?",
			curr_vdp->vd_name);
		WARN(error_string);
		return;
	}

	n = check_queue_status(QSP_ARG  curr_vdp);
	if( n == curr_vdp->vd_n_buffers ){
		sprintf(error_string,
	"wait_next:  device %s already has all of its %d buffers ready!?",
			curr_vdp->vd_name,curr_vdp->vd_n_buffers);
		WARN(error_string);
		return;
	}

	do {
		m = check_queue_status(QSP_ARG  curr_vdp);
	} while( m == n );

	update_vars(SINGLE_QSP_ARG);
#endif
}

COMMAND_FUNC( wait_drip )	/* wait til we have at least one frame */
{
#ifdef HAVE_V4L2
	int n=0;

	CHECK_DEVICE

	if( ! IS_CAPTURING(curr_vdp) ){
		sprintf(error_string,"wait_drip:  Video device %s is not capturing!?",
			curr_vdp->vd_name);
		WARN(error_string);
		return;
	}

	do {
		n = check_queue_status(QSP_ARG  curr_vdp);
	} while( n == 0 );

	update_vars(SINGLE_QSP_ARG);
#endif
}

#ifdef HAVE_V4L2
void release_oldest_buffer(QSP_ARG_DECL  Video_Device *vdp)
{
	struct v4l2_buffer buf;

//sprintf(error_string,"release_oldest_buffer %s BEGIN (newest = %d, oldest = %d)",
//vdp->vd_name,vdp->vd_oldest,vdp->vd_newest);
//advise(error_string);

	if( vdp->vd_oldest < 0 ){
		WARN("release_oldest_buffer:  no oldest buffer to release!?");
		return;
	}

	CLEAR(buf);
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.index = vdp->vd_oldest;

	if( xioctl(vdp->vd_fd, VIDIOC_QBUF, &buf) < 0 )
		ERRNO_WARN ("VIDIOC_QBUF #3");

	if( vdp->vd_newest == vdp->vd_oldest ){
		vdp->vd_oldest=vdp->vd_newest=(-1);
	} else {
		vdp->vd_oldest++;
		if( vdp->vd_oldest >= vdp->vd_n_buffers ) vdp->vd_oldest=0;
	}
}
#endif /* HAVE_V4L2 */

COMMAND_FUNC( do_release_buffer )
{
#ifdef HAVE_V4L2
	CHECK_DEVICE

	if( ! IS_CAPTURING(curr_vdp) ){
		sprintf(error_string,"wait_next:  Video device %s is not capturing!?",
			curr_vdp->vd_name);
		WARN(error_string);
		return;
	}

	release_oldest_buffer(QSP_ARG  curr_vdp);
	update_vars(SINGLE_QSP_ARG);
#endif /* HAVE_V4L2 */
}


static Command flow_ctbl[]={
{ "start",	start_flow,	"start capture"		},
{ "stop",	stop_flow,	"start capture"		},
{ "wait",	wait_drip,	"wait for at least one frame in memory"	},
{ "next",	wait_next,	"wait for the next frame"	},
{ "release",	do_release_buffer,	"release oldest buffer"	},
{ "quit",	popcmd,		"exit submenu"		},
{ NULL_COMMAND						}
};

COMMAND_FUNC( flow_menu )
{
	PUSHCMD(flow_ctbl,"flow");
}

