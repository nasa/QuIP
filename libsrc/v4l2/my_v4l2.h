
#ifndef _MY_V4L2_H_
#define _MY_V4L2_H_

#include "quip_config.h"
#include "fio_api.h"

#include "my_video_dev.h"
#include "query.h"

#ifdef RECORD_TIMESTAMPS
#define TIMESTAMP_SIZE	sizeof(struct timeval)
#endif

extern debug_flag_t stream_debug;

/* vmenu.c */

#ifdef HAVE_V4L2
extern int stop_capturing(QSP_ARG_DECL  Video_Device *);
extern int check_queue_status(QSP_ARG_DECL  Video_Device *);
#endif /* HAVE_V4L2 */

ITEM_INTERFACE_PROTOTYPES(Video_Device,video_dev)
#define PICK_VIDEO_DEV(p)	pick_video_dev(QSP_ARG  p)

/* stream.c */
extern COMMAND_FUNC( print_grab_times );
extern COMMAND_FUNC( print_store_times );
extern COMMAND_FUNC( wait_record );
extern COMMAND_FUNC( halt_record );

extern void dump_timestamps(const char *filename);
extern COMMAND_FUNC( do_stream_record );

#ifdef HAVE_V4L2
extern void v4l2_stream_record(QSP_ARG_DECL  Image_File *ifp, long nf, int nc, Video_Device **vd_tbl);
extern void release_oldest_buffer(QSP_ARG_DECL  Video_Device *);
#endif /* HAVE_V4L2 */

/* fastdown.c */
extern void fast_downsample(Data_Obj *,Data_Obj *);


#endif /* _MY_V4L2_H_ */
