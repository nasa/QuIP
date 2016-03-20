
#ifndef NO_VIDEO_DEVICE

#include "quip_config.h"

#include "query.h"


#ifdef HAVE_ASM_TYPES_H
#include <asm/types.h>	  /* for videodev2.h */
#endif

#define __user			/* something for kernel modules? */
/* end of [unnecessary] defns */

#ifdef HAVE_V4L2
#ifdef HAVE_LINUX_VIDEODEV2_H
#include <linux/videodev2.h>
#endif
#endif // HAVE_V4L2

#define CLEAR(x) memset(&(x), 0, sizeof(x))

typedef struct my_buffer {
	void *			mb_start;
	size_t			mb_length;
#ifdef HAVE_V4L2
	struct video_device *	mb_vdp;
#endif // HAVE_V4L2
} My_Buffer;

typedef struct choice {
	const char *	ch_name;
	int		ch_id;
} Choice;

#define MAX_BUFFERS_PER_DEVICE	32

typedef struct video_device {
	const char *	vd_name;
	int		vd_fd;
	int		vd_flags;

	/* stuff for stream record */
	int		vd_oldest;
	int		vd_newest;
	int		vd_n_buffers;
	My_Buffer 	vd_buf_tbl[MAX_BUFFERS_PER_DEVICE];

	/* stuff for setting controls */
	int		vd_n_standards;
	int		vd_n_inputs;
	Choice * 	vd_input_choices;
	Choice * 	vd_std_choices;
} Video_Device;

#define NO_VIDEO_DEVICE	((Video_Device *)NULL)

/* flag bits */
#define VD_CAPTURING		1

#define IS_CAPTURING( vdp )			( (vdp)->vd_flags & VD_CAPTURING )

#ifdef HAVE_V4L2

#define CHECK_DEVICE						\
								\
	if( curr_vdp == NO_VIDEO_DEVICE ){			\
		WARN("No video device selected");		\
		return;						\
	}


#define CHECK_DEVICE2						\
								\
	if( curr_vdp == NO_VIDEO_DEVICE ){			\
		WARN("No video device selected");		\
		return -1;					\
	}

#else // ! HAVE_V4L2

#define CHECK_DEVICE
#define CHECK_DEVICE2

#endif // ! HAVE_V4L2

/* globals */

extern Video_Device * curr_vdp;
extern unsigned int n_buffers;


/* prototypes */

extern int start_capturing(QSP_ARG_DECL  Video_Device *);
#ifdef HAVE_V4L2
extern int dq_buf(QSP_ARG_DECL  Video_Device *vdp,struct v4l2_buffer *bufp);
#endif /* HAVE_V4L2 */
extern int xioctl(int fd, int request, void *arg);
extern void errno_warn(QSP_ARG_DECL  const char *s);

#define ERRNO_WARN(s)	errno_warn(QSP_ARG  s)


extern COMMAND_FUNC( do_flow_menu );

#endif /* undef NO_VIDEO_DEVICE */

