
#include "gmovie.h"
#include "data_obj.h"

/* xmvi.c */


extern void xmvi_init(SINGLE_QSP_ARG_DECL);

#ifdef FOOBAR
extern int x_setup_movie(QSP_ARG_DECL  Movie *mvip,uint32_t);
extern void x_add_frame(QSP_ARG_DECL  Movie *mvip,Data_Obj *dp);
extern void x_end_assemble(QSP_ARG_DECL  Movie *mvip);
extern void x_record_movie(QSP_ARG_DECL  uint32_t n,Movie *mvip);
extern void x_movie_info(QSP_ARG_DECL  Movie *mvip);
extern void x_open_movie(QSP_ARG_DECL  const char *filename);
extern void x_play_movie(QSP_ARG_DECL  Movie *mvip);
extern void x_reverse_movie(Movie *mvip);
extern void x_get_frame(QSP_ARG_DECL  Movie *mvip,uint32_t n,Data_Obj *dp);
extern void x_get_field(QSP_ARG_DECL  Movie *mvip,uint32_t n,Data_Obj *dp);
extern void x_get_frame_comp(QSP_ARG_DECL  Movie *mvip,uint32_t n,Data_Obj *dp,int);
extern void x_get_field_comp(QSP_ARG_DECL  Movie *mvip,uint32_t n,Data_Obj *dp,int);
extern void x_close_movie(QSP_ARG_DECL  Movie *mvip);
extern void x_movie_init(SINGLE_QSP_ARG_DECL);
extern int  x_setup_play(Movie *);
extern void x_wait_play(void);
extern void x_monitor(SINGLE_QSP_ARG_DECL);
#endif /* FOOBAR */



