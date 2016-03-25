
#include "quip_config.h"


#ifdef INC_VERSION
char VersionId_inc_mmenu[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */


#include <stdio.h>
#include "img_file.h"
#include "rv_api.h"

#define DEFAULT_METEOR_WIDTH	640
#define DEFAULT_METEOR_HEIGHT	480
//#define DEFAULT_METEOR_FRAMES	96
#define DEFAULT_METEOR_FRAMES	48

/* globals */

extern int usePacked;
extern char *mmbuf;
extern int32_t last_mm_size;
extern int meteor_fd;
extern int num_meteor_frames;
extern int bytes_per_pixel;

#ifdef ALLOW_RT_SCHED
extern int rt_is_on;
#endif /* ALLOW_RT_SCHED */

#ifdef RECORD_TIMESTAMPS
extern int stamping;
#define TIMESTAMP_SIZE	sizeof(struct timeval)
#endif /* RECORD_TIMESTAMPS */

extern struct meteor_frame_offset meteor_off;
extern struct meteor_geomet my_geo;
extern struct meteor_mem *_mm;
extern int _hiwat, _lowat;
extern int64_t capture_code;
extern int meteor_rows, meteor_columns, meteor_bytes_per_pixel;
extern int meteor_field_mode;

extern int recording_in_process;
extern Image_File *record_ifp;


/* prototypes */

/* mhw.c */
extern void meteor_mmap(SINGLE_QSP_ARG_DECL);
extern int checkChips(SINGLE_QSP_ARG_DECL);

/* mcapt.c */

extern void gotframe(int signum);
extern void	meteor_status(SINGLE_QSP_ARG_DECL);
extern uint32_t	get_blocks_per_frame(void);
extern void	update_movie_database(QSP_ARG_DECL  RV_Inode *inp);
extern void	mm_init(void);
extern int	meteor_capture(SINGLE_QSP_ARG_DECL);
extern COMMAND_FUNC( meteor_stop_capture );
extern void	meteor_clear_counts(void);
extern void	meteor_record_clip(QSP_ARG_DECL  Image_File *ifp,int32_t n_frames);
extern Data_Obj	*make_frame_object(QSP_ARG_DECL  const char *name, int index);
extern void	setup_monitor_capture(SINGLE_QSP_ARG_DECL);
extern void	finish_recording(QSP_ARG_DECL  Image_File *);

/* mgeo.c */
extern void meteor_set_size(QSP_ARG_DECL  int,int,int);
extern void set_grab_depth(QSP_ARG_DECL  int);
extern COMMAND_FUNC( do_geometry );
extern COMMAND_FUNC( do_capture );
extern COMMAND_FUNC( do_captst );


extern COMMAND_FUNC( do_video_controls );

/* stream.c */
extern void thread_write_enable(QSP_ARG_DECL  int index, int flag);
extern void dump_timestamps(const char *);
extern void print_grab_times(void);
extern void print_store_times(void);
extern void set_async_record(int);
extern int get_async_record(void);
extern COMMAND_FUNC( meteor_halt_record );
extern COMMAND_FUNC( meteor_wait_record );
extern void stream_record(QSP_ARG_DECL  Image_File *,int32_t);
extern void dump_ccount(int,FILE *);
extern void set_fudge(int);
extern void set_ndisks(int);
extern void set_n_discard(int);
extern void monitor_meteor_video(QSP_ARG_DECL  Data_Obj *dp);
extern void play_meteor_movie(QSP_ARG_DECL  Image_File *ifp);
extern void play_meteor_frame(QSP_ARG_DECL  Image_File *ifp, uint32_t frame);
extern void set_disp_comp(int);


/* mmenu.c */

void enable_meteor_timestamps(QSP_ARG_DECL  uint32_t flag);
void make_movie_from_inode(QSP_ARG_DECL  RV_Inode *inp);

/* flow.c */
extern COMMAND_FUNC( meteor_flow_menu );
