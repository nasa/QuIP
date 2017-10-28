#include "quip_config.h"

#ifdef HAVE_X11

#include "quip_prot.h"
#include "xmvi.h"
#include "xsupp.h"		/* needs to come before img_file.h */
#include "fio_api.h"
#include "gmovie.h"
#include "viewer.h"


static int x_setup_movie(QSP_ARG_DECL  Movie *mvip,uint32_t n_fields)
{
	const char *pathname;
	Image_File *ifp;

	pathname = movie_pathname(MOVIE_NAME(mvip));
	if( pathname == NULL ) return(-1);

	ifp = open_image_file(QSP_ARG  pathname,"w");
	if( ifp == NULL )
		return(-1);

	SET_MOVIE_DATA(mvip, ifp);
	return(0);
}

static void x_add_frame(QSP_ARG_DECL  Movie *mvip,Data_Obj *dp)
{
	/* make sure dp & mvip have same dimensions if not 1st frame */
	write_image_to_file(QSP_ARG  (Image_File *)mvip->mvi_data,dp);
}

static void x_end_assemble(QSP_ARG_DECL  Movie *mvip)
{
	close_image_file(QSP_ARG  (Image_File *)mvip->mvi_data);
	delete_movie(QSP_ARG  mvip);
}

static void x_monitor(SINGLE_QSP_ARG_DECL)
{
	WARN("Sorry, no video input in X11");
}

static void x_record_movie(QSP_ARG_DECL  uint32_t n,Movie *mvip)
{
	WARN("Sorry, no video input in X11");
}

static COMMAND_FUNC( x_menu )
{
	WARN("movie/control:  Sorry, no X11-specific movie control commands");
}

static void x_movie_info(QSP_ARG_DECL  Movie *mvip)
{
	advise("movie/control:  Sorry, no X11-specific movie information");
}

static int x_setup_play(Movie *mvip)
{ return(0); }

static void x_wait_play(void)
{}

static void x_open_movie(QSP_ARG_DECL  const char *filename)
{
	/* prepare to play */
	/* open file, create data object, read in */

	Image_File *ifp;
	Data_Obj *dp;
	Movie *mvip;
	const char *pathname;

	pathname = movie_pathname(filename);
	ifp=open_image_file(QSP_ARG  pathname,"r");
	if( ifp == NULL ) return;

	if( ram_area_p == NULL ) dataobj_init(SINGLE_QSP_ARG);

	dp = make_dobj(filename, OBJ_TYPE_DIMS(ifp->if_dp),
			OBJ_PREC_PTR(ifp->if_dp));

	if( dp == NULL ) return;

	read_object_from_file(QSP_ARG  dp,ifp);
	/* should close file automatically!? */


	/* make the movie object */

	mvip = create_movie(QSP_ARG  filename);
	if( mvip == NULL ){
		/* BUG free dobj here */
		return;
	}
	mvip->mvi_flags = 0;

	mvip->mvi_data = dp;
}

#define MOVIE_VIEWER_NAME	"Movie_Viewer"

static void x_play_movie(QSP_ARG_DECL  Movie *mvip)
{
	Data_Obj *dp;
	Viewer *vp;

	dp = (Data_Obj *)mvip->mvi_data;
	/* longlist(dp); */

	vp = vwr_of(MOVIE_VIEWER_NAME);

mk_win:
	if( vp == NULL ){
		vp = viewer_init(MOVIE_VIEWER_NAME,OBJ_COLS(dp),OBJ_ROWS(dp),0);
		if( vp == NULL ){
			WARN("couldn't create viewer");
			return;
		}
		default_cmap(QSP_ARG  VW_DPYABLE(vp) );
		show_viewer(vp);	/* default state is to be shown */
		select_viewer(vp);
	} else {
		if( vp->vw_width != OBJ_COLS(dp) ||
			vp->vw_height != OBJ_ROWS(dp) ){
			sprintf(ERROR_STRING,
				"Resizing movie viewer for movie %s",
				OBJ_NAME(dp));
			advise(ERROR_STRING);
			delete_viewer(vp);
			vp=NULL;
			goto mk_win;
		}
	}

	/* load_viewer got rewritten, no longer show all frames!? */
	old_load_viewer(QSP_ARG  vp,dp);
}

static void x_reverse_movie(Movie *mvip)
{
	/* display in X viewer */
	NWARN("Sorry, x_reverse_movie not implemented");
}

static void x_get_field(QSP_ARG_DECL  Movie *mvip,uint32_t n,Data_Obj *dp)
{
	WARN("Sorry, can't yet get fields from X images.");
}

static void x_get_field_comp(QSP_ARG_DECL  Movie *mvip,uint32_t n,Data_Obj *dp,int nc)
{
	WARN("Sorry, can't yet get fields from X images.");
}

static void x_get_frame(QSP_ARG_DECL  Movie *mvip,uint32_t n,Data_Obj *dp)
{
	Data_Obj *src_dp;

	src_dp = (Data_Obj *)mvip->mvi_data;

	if( OBJ_FRAMES(src_dp) > 1 ){
		/* BUG won't work right if nseqs > 1 ... */
		char index[16];
		sprintf(index,"[%d]",n);
		src_dp = index_data(QSP_ARG  src_dp,index);
	}

	if( dp_same(src_dp,dp,"x_get_frame") ) return;

	dp_copy(dp,src_dp);
}

static void x_get_frame_comp(QSP_ARG_DECL  Movie *mvip,uint32_t n,Data_Obj *dp,int nc)
{
	WARN("Sorry, x_get_frame_comp() not implemented");
}

static void x_close_movie(QSP_ARG_DECL  Movie *mvip)
{
	delvec((Data_Obj *)mvip->mvi_data);
}


static void x_movie_init(SINGLE_QSP_ARG_DECL)
{
	/* initialize window system? */
	/* initialize data objects? */
}

static Movie_Module x_movie_module ={
	"x",
	x_setup_movie,
	x_add_frame,
	x_end_assemble,
	x_record_movie,
	x_monitor,

	x_menu,
	x_movie_info,
	x_movie_init,

	x_open_movie,
	x_setup_play,
	x_play_movie,
	x_wait_play,
	x_reverse_movie,
	x_get_frame,
	x_get_field,
	x_get_frame_comp,
	x_get_field_comp,
	x_close_movie
};

static int x_loaded=0;

void xmvi_init(SINGLE_QSP_ARG_DECL)
{
	if( !x_loaded ){
		load_movie_module(QSP_ARG  &x_movie_module);
		x_loaded++;
	}
	else WARN("x movie menu already loaded!?");
}


#endif /* HAVE_X11 */

