
#ifndef _MY_V4L2_H_
#define _MY_V4L2_H_

#include "quip_config.h"
#include "fio_api.h"

#include "my_video_dev.h"

#ifdef RECORD_TIMESTAMPS
#define TIMESTAMP_SIZE	sizeof(struct timeval)
#endif

extern debug_flag_t v4l2_debug;

/* vmenu.c */

#ifdef HAVE_V4L2

extern int _stop_capturing(QSP_ARG_DECL  Video_Device *);
extern void _find_oldest_buffer(QSP_ARG_DECL  Video_Device *vdp);
extern int _check_queue_status(QSP_ARG_DECL  Video_Device *);

#define stop_capturing(vdp) _stop_capturing(QSP_ARG  vdp)
#define find_oldest_buffer(vdp) _find_oldest_buffer(QSP_ARG  vdp)
#define check_queue_status(vdp) _check_queue_status(QSP_ARG  vdp)

#endif /* HAVE_V4L2 */

ITEM_INTERFACE_PROTOTYPES(Video_Device,video_dev)
#define pick_video_dev(p)	_pick_video_dev(QSP_ARG  p)
#define video_dev_of(s)		_video_dev_of(QSP_ARG  s)
#define new_video_dev(s)	_new_video_dev(QSP_ARG  s)
#define del_video_dev(s)	_del_video_dev(QSP_ARG  s)

/* ezstream.c */
extern uint32_t get_blocks_per_v4l2_frame(void);
extern int get_async_record(void);

/* stream.c */
extern COMMAND_FUNC( print_grab_times );
extern COMMAND_FUNC( print_store_times );
extern COMMAND_FUNC( wait_record );
extern COMMAND_FUNC( halt_record );

extern void dump_timestamps(const char *filename);
extern COMMAND_FUNC( do_stream_record );

#ifdef HAVE_V4L2
extern void _v4l2_stream_record(QSP_ARG_DECL  Image_File *ifp, long nf, int nc, Video_Device **vd_tbl);
#define v4l2_stream_record(ifp,nf,nc,vd_tbl) _v4l2_stream_record(QSP_ARG  ifp,nf,nc,vd_tbl)

extern void _release_oldest_buffer(QSP_ARG_DECL  Video_Device *);

#define release_oldest_buffer(vdp) _release_oldest_buffer(QSP_ARG  vdp)

#endif /* HAVE_V4L2 */

/* fastdown.c */
extern void _fast_downsample(QSP_ARG_DECL  Data_Obj *,Data_Obj *);
#define fast_downsample(dp1,dp2) _fast_downsample(QSP_ARG  dp1,dp2)


#endif /* _MY_V4L2_H_ */
