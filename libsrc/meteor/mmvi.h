
/* extern defns for the meteor movie module */

#include "gmovie.h"
#include "ioctl_meteor.h"
#include "query.h"

#ifdef ALLOW_RT_SCHED
extern int rt_is_on;
#define YIELD_PROC(time)	{ if( rt_is_on ) sched_yield(); else usleep(time); }
#else /* ! ALLOW_RT_SCHED */
#define YIELD_PROC(time)	usleep(time);
#endif /* ALLOW_RT_SCHED */

//#ifdef CAUTIOUS
//#define INSURE_MM(s)	if(_mm==NULL) {								\
//				sprintf(ERROR_STRING,"CAUTIOUS: %s:  _mm not initialized",(s));	\
//				ERROR1(ERROR_STRING);						\
//			}
//#else
//#define INSURE_MM(s)
//#endif

/* global vars */
extern int displaying_color;


/* mmodule.c */

extern void meteor_init(SINGLE_QSP_ARG_DECL);

extern void meteor_record(QSP_ARG_DECL  uint32_t n_fields,Movie *mvip);
extern void meteor_open_movie(QSP_ARG_DECL  const char *filename);
extern int meteor_setup_movie(QSP_ARG_DECL  Movie *mvip,uint32_t n_fields);
extern void meteor_movie_info(QSP_ARG_DECL  Movie *mvip);
extern void meteor_add_frame(QSP_ARG_DECL  Movie *mvip,Data_Obj *dp);
extern void meteor_end_assemble(QSP_ARG_DECL  Movie *mvip);
extern void meteor_get_frame(QSP_ARG_DECL  Movie *mvip, uint32_t n, Data_Obj *dp);
extern void meteor_get_framec(QSP_ARG_DECL  Movie *mvip, uint32_t n, Data_Obj *dp, int comp);
extern void meteor_get_field(QSP_ARG_DECL  Movie *mvip,uint32_t f,Data_Obj *dp);
extern void meteor_get_fieldc(QSP_ARG_DECL  Movie *mvip,uint32_t f,Data_Obj *dp,int comp);

extern void init_meteor_viewer(QSP_ARG_DECL  int width, int height, int depth);

extern void meteor_play_movie(QSP_ARG_DECL  Movie *mvip);
extern void meteor_reverse_movie(Movie *mvip);
extern void meteor_shuttle_movie(QSP_ARG_DECL  Movie *mvip,uint32_t frame);
extern void meteor_close_movie(QSP_ARG_DECL  Movie *mvip);
extern int meteor_setup_play(Movie *mvip);
extern void meteor_monitor(SINGLE_QSP_ARG_DECL);
extern void meteor_wait_play(void);

/* mcapt.c */

/* mgeo.c */

extern int meteor_get_geometry(struct meteor_geomet *gp);
extern int get_bytes_per_pixel(QSP_ARG_DECL  int fmt);
extern int get_ofmt_index(QSP_ARG_DECL  int fmt);

