
#ifndef NO_VIDEO_DEVICE

#include "quip_config.h"
#include "fio_api.h"


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

extern int v4l2_field_mode;	// BUG - avoid global

typedef struct my_buffer {
	void *			mb_start;
	Data_Obj *		mb_dp;
	int			mb_index;
	int			mb_flags;
#ifdef HAVE_V4L2
	struct v4l2_buffer	mb_buf;
	struct video_device *	mb_vdp;
#endif // HAVE_V4L2
} My_Buffer;

typedef struct choice {
	const char *	ch_name;
	int		ch_id;
} Choice;

// One device gives 27, while another gives 6!?
#define MAX_BUFFERS_PER_DEVICE	32
#define DEFAULT_N_VIDEO_BUFFERS	6	// smallest number we are given by an interface?

typedef struct video_device {
	const char *	vd_name;
	int		vd_fd;
	int		vd_flags;

	/* stuff for stream record */
	int		vd_n_buffers;

	// Better to allocate this table dynamically?
	My_Buffer 	vd_buf_tbl[MAX_BUFFERS_PER_DEVICE];
	My_Buffer * 	vd_oldest_mbp;
	My_Buffer * 	vd_newest_mbp;

	/* stuff for setting controls */
	int		vd_n_standards;
	int		vd_n_inputs;
	Choice * 	vd_input_choices;
	Choice * 	vd_std_choices;
} Video_Device;

#define NO_VIDEO_DEVICE	((Video_Device *)NULL)

/* flag bits */
#define VD_CAPTURING				1
#define VD_SUPPORTS_USERSPACE_BUFFERS		2
#define VD_SUPPORTS_MMAP_BUFFERS		4
#define VD_USING_USERSPACE_BUFFERS		8
#define VD_USING_MMAP_BUFFERS			16
#define VD_HAS_BUFFERS				32

#define IS_USING_MMAP_BUFFERS(vdp)		((vdp)->vd_flags & VD_USING_MMAP_BUFFERS)
#define IS_USING_USERSPACE_BUFFERS(vdp)		((vdp)->vd_flags & VD_USING_USERSPACE_BUFFERS)
#define CAN_USE_MMAP_BUFFERS(vdp)		((vdp)->vd_flags & VD_SUPPORTS_MMAP_BUFFERS)
#define CAN_USE_USERSPACE_BUFFERS(vdp)		((vdp)->vd_flags & VD_SUPPORTS_USERSPACE_BUFFERS)
#define HAS_BUFFERS(vdp)			((vdp)->vd_flags & VD_HAS_BUFFERS)

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

extern int _start_capturing(QSP_ARG_DECL  Video_Device *);
#define start_capturing(vdp) _start_capturing(QSP_ARG  vdp)

#ifdef HAVE_V4L2
extern int dq_buf(QSP_ARG_DECL  Video_Device *vdp,struct v4l2_buffer *bufp);
#endif /* HAVE_V4L2 */
extern int xioctl(int fd, int request, void *arg);
extern void errno_warn(QSP_ARG_DECL  const char *s);

extern void print_buf_info(const char *msg, My_Buffer *mbp);

#define ERRNO_WARN(s)	errno_warn(QSP_ARG  s)


extern COMMAND_FUNC( do_flow_menu );

// ezstream.c
extern void v4l2_record_clip(Image_File *ifp,int n_frames_to_request);

#endif /* undef NO_VIDEO_DEVICE */

