
/* extern defns for the v4l2 movie module */

#include "gmovie.h"

/* vmodule.c */

extern void v4l2_init(SINGLE_QSP_ARG_DECL);

extern void v4l2_record(QSP_ARG_DECL  uint32_t n_fields,Movie *mvip);
extern void v4l2_open_movie(QSP_ARG_DECL  const char *filename);
extern int v4l2_setup_movie(QSP_ARG_DECL  Movie *mvip,uint32_t n_fields);
extern void v4l2_movie_info(QSP_ARG_DECL  Movie *mvip);
extern void v4l2_add_frame(QSP_ARG_DECL  Movie *mvip,Data_Obj *dp);
extern void v4l2_end_assemble(QSP_ARG_DECL  Movie *mvip);
extern void v4l2_get_frame(QSP_ARG_DECL  Movie *mvip, uint32_t n, Data_Obj *dp);
extern void v4l2_get_framec(QSP_ARG_DECL  Movie *mvip, uint32_t n, Data_Obj *dp, int comp);
extern void v4l2_get_field(QSP_ARG_DECL  Movie *mvip,uint32_t f,Data_Obj *dp);
extern void v4l2_get_fieldc(QSP_ARG_DECL  Movie *mvip,uint32_t f,Data_Obj *dp,int comp);

extern void init_v4l2_viewer(QSP_ARG_DECL  int width, int height, int depth);

extern void v4l2_play_movie(QSP_ARG_DECL  Movie *mvip);
extern void v4l2_reverse_movie(QSP_ARG_DECL  Movie *mvip);
extern void v4l2_shuttle_movie(QSP_ARG_DECL  Movie *mvip,uint32_t frame);
extern void v4l2_close_movie(QSP_ARG_DECL  Movie *mvip);
extern int v4l2_setup_play(QSP_ARG_DECL  Movie *mvip);
extern void v4l2_monitor(SINGLE_QSP_ARG_DECL);
extern void v4l2_wait_play(SINGLE_QSP_ARG_DECL);

extern void _monitor_v4l2_video(QSP_ARG_DECL  Data_Obj *dp);
#define monitor_v4l2_video(dp) _monitor_v4l2_video(QSP_ARG  dp)

