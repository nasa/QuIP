#include "quip_config.h"

char VersionId_mvimenu_xmvi[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

#include "xmvi.h"
#include "xsupp.h"		/* needs to come before img_file.h */
#include "fio_api.h"
#include "gmovie.h"
#include "viewer.h"
#include "getbuf.h"


int x_setup_movie(QSP_ARG_DECL  Movie *mvip,uint32_t n_fields)
{
	const char *pathname;
	Image_File *ifp;

	pathname = movie_pathname(mvip->mvi_name);
	if( pathname == NULL ) return(-1);

	ifp = open_image_file(QSP_ARG  pathname,"w");
	if( ifp == NO_IMAGE_FILE )
		return(-1);

	mvip->mvi_data = ifp;
	return(0);
}

void x_add_frame(QSP_ARG_DECL  Movie *mvip,Data_Obj *dp)
{
	/* make sure dp & mvip have same dimensions if not 1st frame */
	write_image_to_file(QSP_ARG  (Image_File *)mvip->mvi_data,dp);
}

void x_end_assemble(QSP_ARG_DECL  Movie *mvip)
{
	close_image_file(QSP_ARG  (Image_File *)mvip->mvi_data);
	delete_movie(QSP_ARG  mvip);
}

void x_monitor(SINGLE_QSP_ARG_DECL)
{
	WARN("Sorry, no video input in X11");
}

void x_record_movie(QSP_ARG_DECL  uint32_t n,Movie *mvip)
{
	WARN("Sorry, no video input in X11");
}

COMMAND_FUNC( x_menu )
{
	WARN("movie/control:  Sorry, no X11-specific movie control commands");
}

void x_movie_info(QSP_ARG_DECL  Movie *mvip)
{
	advise("movie/control:  Sorry, no X11-specific movie information");
}

int x_setup_play(Movie *mvip)
{ return(0); }

void x_wait_play(void)
{}

void x_open_movie(QSP_ARG_DECL  const char *filename)
{
	/* prepare to play */
	/* open file, create data object, read in */

	Image_File *ifp;
	Data_Obj *dp;
	Movie *mvip;
	const char *pathname;

	pathname = movie_pathname(filename);
	ifp=open_image_file(QSP_ARG  pathname,"r");
	if( ifp == NO_IMAGE_FILE ) return;

	if( ram_area == NO_AREA ) dataobj_init(SINGLE_QSP_ARG);

	dp = make_dobj(QSP_ARG  filename, &ifp->if_dp->dt_type_dimset,
			ifp->if_dp->dt_prec);

	if( dp == NO_OBJ ) return;

	read_object_from_file(QSP_ARG  dp,ifp);
	/* should close file automatically!? */


	/* make the movie object */

	mvip = create_movie(QSP_ARG  filename);
	if( mvip == NO_MOVIE ){
		/* BUG free dobj here */
		return;
	}
	mvip->mvi_flags = 0;

	mvip->mvi_data = dp;
}

#define MOVIE_VIEWER_NAME	"Movie_Viewer"

void x_play_movie(QSP_ARG_DECL  Movie *mvip)
{
	Data_Obj *dp;
	Viewer *vp;

	dp = (Data_Obj *)mvip->mvi_data;
	/* longlist(dp); */

	vp = vwr_of(QSP_ARG  MOVIE_VIEWER_NAME);

mk_win:
	if( vp == NO_VIEWER ){
		vp = viewer_init(QSP_ARG  MOVIE_VIEWER_NAME,dp->dt_cols,dp->dt_rows,0);
		if( vp == NO_VIEWER ){
			WARN("couldn't create viewer");
			return;
		}
		default_cmap(&vp->vw_top);
		show_viewer(QSP_ARG  vp);	/* default state is to be shown */
		select_viewer(QSP_ARG  vp);
	} else {
		if( vp->vw_width != dp->dt_cols ||
			vp->vw_height != dp->dt_rows ){
			sprintf(error_string,
				"Resizing movie viewer for movie %s",
				dp->dt_name);
			advise(error_string);
			delete_viewer(QSP_ARG  vp);
			vp=NO_VIEWER;
			goto mk_win;
		}
	}

	/* load_viewer got rewritten, no longer show all frames!? */
	old_load_viewer(QSP_ARG  vp,dp);
}

void x_reverse_movie(Movie *mvip)
{
	/* display in X viewer */
	NWARN("Sorry, x_reverse_movie not implemented");
}

void x_get_field(QSP_ARG_DECL  Movie *mvip,uint32_t n,Data_Obj *dp)
{
	WARN("Sorry, can't yet get fields from X images.");
}

void x_get_field_comp(QSP_ARG_DECL  Movie *mvip,uint32_t n,Data_Obj *dp,int nc)
{
	WARN("Sorry, can't yet get fields from X images.");
}

void x_get_frame(QSP_ARG_DECL  Movie *mvip,uint32_t n,Data_Obj *dp)
{
	Data_Obj *src_dp;

	src_dp = (Data_Obj *)mvip->mvi_data;

	if( src_dp->dt_frames > 1 ){
		/* BUG won't work right if nseqs > 1 ... */
		char index[16];
		sprintf(index,"[%d]",n);
		src_dp = index_data(QSP_ARG  src_dp,index);
	}

	if( dp_same(QSP_ARG  src_dp,dp,"x_get_frame") ) return;

	dp_copy(QSP_ARG  dp,src_dp);
}

void x_get_frame_comp(QSP_ARG_DECL  Movie *mvip,uint32_t n,Data_Obj *dp,int nc)
{
	WARN("Sorry, x_get_frame_comp() not implemented");
}

void x_close_movie(QSP_ARG_DECL  Movie *mvip)
{
	delvec(QSP_ARG  (Data_Obj *)mvip->mvi_data);
}


void x_movie_init(SINGLE_QSP_ARG_DECL)
{
	/* initialize window system? */
	/* initialize data objects? */
}

Movie_Module x_movie_module ={
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

