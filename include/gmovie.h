#ifndef _GMOVIE_H_
#define _GMOVIE_H_

#include "quip_config.h"

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>		/* time_t */
#endif

#include "data_obj.h"
#include "node.h"
#include "items.h"
#include "query.h"

typedef struct movie {
	Item		mvi_item;
	Shape_Info	mvi_shape;
	int		mvi_flags;
	time_t		mvi_time;
	void *		mvi_data;		/* device specific */
	const char *	mvi_filename;
	int32_t		mvi_offset;		/* for sirius shuttle */
} Movie;

#define mvi_name	mvi_item.item_name
#define mvi_depth	mvi_shape.si_mach_dim[0]
#define mvi_width	mvi_shape.si_mach_dim[1]
#define mvi_height	mvi_shape.si_mach_dim[2]
#define mvi_nframes	mvi_shape.si_mach_dim[3]

#define NO_MOVIE	((Movie *)NULL)

/* flag bits */
#define MVI_PLAYABLE		1
#define MVI_ASSEMBLING		2
#define MVI_RECORDING		4
#define MVI_REFERENCED		8
#define N_GEN_MOVIE_FLAGS	4
#define FIRST_MODULE_FLAG	16

#define IS_PLAYABLE(mvip)		( (mvip) ->mvi_flags & MVI_PLAYABLE)
#define IS_ASSEMBLING(mvip)		( (mvip) ->mvi_flags & MVI_ASSEMBLING)
#define IS_RECORDING(mvip)		( (mvip) ->mvi_flags & MVI_RECORDING)

#define IS_REFERENCED(mvip)		( (mvip) ->mvi_flags & MVI_REFERENCED )

#define FRAME_OBJ_NAME	"movie_frame"

typedef struct mvi_module {
	const char *	mm_name;

	/* BUG - other objects have struct item here,
	 * but that makes it a hassle to initialize with tables.
	 * This will work as long as the Item structure doesn't
	 * have other entries that get initialized...
	 */


	/* record functions */
	int (*setup_func)(QSP_ARG_DECL  Movie *,uint32_t);	/* to record or assemble */
	void (*add_func)(QSP_ARG_DECL  Movie *,Data_Obj *);	/* append a frame */
	void (*end_func)(QSP_ARG_DECL  Movie *);		/* close movie after assembly */
	void (*record_func)(QSP_ARG_DECL  uint32_t,Movie *);	/* record input video */
	void (*monitor_func)(SINGLE_QSP_ARG_DECL);	/* display video input */

	/* misc */
	void (*menu_func)(SINGLE_QSP_ARG_DECL);	/* menu of device commands */
	void (*info_func)(QSP_ARG_DECL  Movie *);		/* print extra info */
	void (*init_func)(SINGLE_QSP_ARG_DECL);

	/* playback functions */
	void (*open_func)(QSP_ARG_DECL  const char *);		/* open movie by name, create object */
	int (*setup_play_func)(Movie *);
	void (*play_func)(QSP_ARG_DECL  Movie *);		/* play back movie */
	void (*wait_func)(void);		/* wait for playback to finish */
	void (*reverse_func)(Movie *);		/* play back movie in reverse */
	void (*frame_func)(QSP_ARG_DECL  Movie *,uint32_t,Data_Obj *);/* read a frame to mem */
	void (*field_func)(QSP_ARG_DECL  Movie *,uint32_t,Data_Obj *);/* read a field to mem */
	void (*framec_func)(QSP_ARG_DECL  Movie *,uint32_t,Data_Obj *,int);/* read component */
	void (*fieldc_func)(QSP_ARG_DECL  Movie *,uint32_t,Data_Obj *,int);/* read component */
	void (*close_func)(QSP_ARG_DECL  Movie *);		/* companion to open_func, close file open for playing */
	void (*shuttle_func)(QSP_ARG_DECL  Movie *, uint32_t);	/* play a single frame */


} Movie_Module;


#define NO_MOVIE_MODULE	((Movie_Module *)NULL)



/* global vars */
extern int n_refresh;
extern Movie_Module x_movie_module;



/* prototypes */

/* mvimenu.c */

extern List *movie_list(SINGLE_QSP_ARG_DECL);
extern int movie_ok(void);
extern Movie *create_movie(QSP_ARG_DECL  const char *moviename);
extern void close_movie(QSP_ARG_DECL  Movie *mvip);
extern void delete_movie(QSP_ARG_DECL  Movie *mvip);
extern void add_playable(Item_Type * itp,void *vp);
extern void load_movie_module(QSP_ARG_DECL  Movie_Module *mmp);
extern const char *movie_pathname(const char *filename);


#endif /* ! _GMOVIE_H_ */
