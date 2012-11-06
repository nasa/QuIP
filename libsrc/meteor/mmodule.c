#include "quip_config.h"

char VersionId_meteor_mmodule[] = QUIP_VERSION_STRING;

#ifdef HAVE_METEOR

#include <stdio.h>

#include "xsupp.h"		/* has to come before jmorecfg.h */
#include "fio_api.h"

#include "ioctl_meteor.h"
#include "mmvi.h"
#include "mmenu.h"
#include "debug.h"
#include "viewer.h"
#include "rv_api.h"

#ifdef DEBUG
int meteor_debug=0;
#endif /* DEBUG */

void meteor_record(QSP_ARG_DECL  uint32_t n_fields,Movie *mvip)
{
	Image_File *ifp;
	uint32_t n_frames_to_request;

	ifp = (Image_File*) mvip->mvi_data;	/* file is opened in meteor_setup_movie */

	/* have we written the header out yet?
	 * If not, then we can go in and set the number of frames now!
	 */

	if( ifp == NO_IMAGE_FILE ){
		WARN("Image file has not been opened");
		return;
	}
	/* we assume the request is for a number of fields...
	 * Unfortunately, or stupidly, or whatever, meteor_record_clip() takes
	 * its agrument in fields only if the meteor is in field mode, otherwise
	 * it's in frames...
	 */

	if( meteor_field_mode ){
		/* specify record length in fields, so nothing to do */
		n_frames_to_request = n_fields;
	} else {
		if( n_fields % 2 ){
			WARN("meteor records even number of fields, rounding up");
			n_fields++;
		}
		n_frames_to_request = n_fields / 2;
	}

	meteor_record_clip(QSP_ARG  ifp,n_frames_to_request);

	/* meteor_record_clip closes the ifp, so we need to eradicate the
	 * reference to it now.
	 */
	mvip->mvi_data = NULL;
}


/* open a file for playpack */

void meteor_open_movie(QSP_ARG_DECL  const char *filename)
{
	advise("meteor open_movie is a NOP");
}

void meteor_play_movie(QSP_ARG_DECL  Movie *mvip)
{
	Image_File *ifp;

	ifp= (Image_File *) mvip->mvi_data;
#ifdef HAVE_X11_EXT
	play_meteor_movie(QSP_ARG  ifp);
#else
	WARN("meteor_play_movie:  Program was configured without X11 Extensions.");
#endif
}

void meteor_reverse_movie(Movie *mvip)
{
	sprintf(DEFAULT_ERROR_STRING,
		"meteor_reverse_movie:  Sorry, meteor can't reverse movie %s",
		mvip->mvi_name);

	NWARN(DEFAULT_ERROR_STRING);
}

void meteor_shuttle_movie(QSP_ARG_DECL  Movie* mvip,uint32_t frame)
{
	Image_File *ifp;

sprintf(error_string,"meteor_shuttle_movie %s %d",mvip->mvi_name,frame);
advise(error_string);
	ifp= (Image_File *) mvip->mvi_data;

#ifdef HAVE_X11_EXT
	play_meteor_frame(QSP_ARG  ifp, frame);
#else
	WARN("meteor_shuttle_movie:  Program was configured without X11 Extensions.");
#endif
}


void meteor_close_movie(QSP_ARG_DECL  Movie *mvip)
{
}

int meteor_setup_play(Movie *mvip)
{
	return(0);
}


void meteor_monitor(SINGLE_QSP_ARG_DECL)
{
	Data_Obj *dp;

	dp = make_frame_object(QSP_ARG  "tmpfrm",0);

#ifdef HAVE_X11_EXT
	setup_monitor_capture(SINGLE_QSP_ARG);
	monitor_meteor_video(QSP_ARG  dp);
#else
	WARN("meteor_monitor:  Program was configured without X11 Extensions.");
#endif
}

void meteor_wait_play()
{
}

void meteor_movie_info(QSP_ARG_DECL  Movie *mvip)
{
	Image_File *ifp;

	ifp= (Image_File *) mvip->mvi_data;
	if_info(QSP_ARG  ifp);
}


/* setup to record a movie */

int meteor_setup_movie(QSP_ARG_DECL  Movie *mvip,uint32_t n_fields)
{
	int ft;
	Image_File *ifp;
	struct meteor_geomet _geo;
	uint32_t blocks_per_frame,n_frames;
	int n_allocframes ;
	int n_disks = rv_get_ndisks();

	/* BUG shouldn't insist upon even field count in field mode */
	if( n_fields % 2 ){
		sprintf(error_string,
	"meteor_setup_movie:  requested number of fields (%d) is odd, rounding up",
			n_fields);
		WARN(error_string);
		n_fields++;
	}
	if( ! meteor_field_mode )
		n_frames = n_fields / 2;
	else
		n_frames = n_fields ;

#ifdef DEBUG
if( debug & meteor_debug )
advise("meteor_setup_movie");
#endif /* DEBUG */

	/* used to set mvi_filename here.... */

	/* BUG? set filetype to IFT_DISK? */
	/* maybe more flexible to let the user specify */

	if( (ft=get_filetype()) != IFT_RV ){
		if( verbose ){
			sprintf(error_string,
	"meteor_setup_movie:  default filetype is %s, resetting to type %s",
				ft_tbl[ft].ft_name,
				ft_tbl[IFT_RV].ft_name);
			advise(error_string);
		}
		set_filetype(QSP_ARG  IFT_RV);
	}

	ifp = write_image_file(QSP_ARG  mvip->mvi_name,n_frames);

	if( ifp == NO_IMAGE_FILE ) return(-1);

	mvip->mvi_data = ifp;

	/* Need to set up an if_dp here...  BUG?  setup_dummy() ?? */

	if( meteor_get_geometry(&_geo) < 0 ){
		/* BUG close file */
		return(-1);
	}

	mvip->mvi_height = _geo.rows;
	mvip->mvi_width = _geo.columns;
	mvip->mvi_nframes = n_frames;

	blocks_per_frame = get_blocks_per_frame();

	/* Make sure that the number of frames requested is an integral multiple of n_disks */

	n_allocframes = FRAMES_TO_ALLOCATE( n_frames, n_disks );

	if( rv_realloc(QSP_ARG  mvip->mvi_name,n_allocframes*blocks_per_frame) < 0 ){
		sprintf(error_string,"meteor_setup_movie:  failed to allocate rawvol storage for %s",
			mvip->mvi_name);
		WARN(error_string);
		return(-1);
	}

	return(0);
}

void meteor_add_frame(QSP_ARG_DECL  Movie *mvip,Data_Obj *dp)
{

#ifdef DEBUG
if( debug & meteor_debug )
advise("meteor_add_frame");
#endif /* DEBUG */

	/* BUG make sure dp & mvip have same dimensions if not 1st frame */
	write_image_to_file(QSP_ARG  (Image_File *) mvip->mvi_data,dp);
}

void meteor_end_assemble(QSP_ARG_DECL  Movie *mvip)
{
	RV_Inode *inp;

#ifdef DEBUG
if( debug & meteor_debug )
advise("meteor_end_assemble");
#endif /* DEBUG */

	inp = rv_inode_of(QSP_ARG  mvip->mvi_name);	/* save this for later */

	/* don't do this stuff until async record is finished */

	if( get_async_record() ){
		/* BUG should arrange for this to be called later */
		return;
	}

	if( mvip->mvi_data != NULL ){
		/* Do we really need to do this???
		 * We know it is an rv file, so this is really pretty much
		 * of a no-op???
		 */
sprintf(error_string,"meteor_end_assemble calling close_image_file, movie is %s, if is %s",
mvip->mvi_name,((Image_File *)mvip->mvi_data)->if_name);
advise(error_string);
		close_image_file(QSP_ARG  (Image_File *) mvip->mvi_data);
	}

	/* We delete the movie because it was open for recording,
	 * and we're too lazy to change the flags...
	 */

	delete_movie(QSP_ARG  mvip);

#ifdef CAUTIOUS
	if( inp == NO_INODE ){
		WARN("CAUTIOUS:  meteor_end_assemble:  missing rv inode!?");
		return;
	}
#endif /* CAUTIOUS */

	update_movie_database(QSP_ARG  inp);
}

void meteor_get_frame(QSP_ARG_DECL  Movie *mvip, uint32_t n, Data_Obj *dp)
{
	Image_File *ifp;

	ifp = (Image_File *) mvip->mvi_data;

	if( image_file_seek(QSP_ARG  ifp,n) < 0 )
		return;

	read_object_from_file(QSP_ARG  dp,ifp);
}

void meteor_get_framec(QSP_ARG_DECL  Movie *mvip, uint32_t n, Data_Obj *dp, int comp)
{
	WARN("Sorry, meteor_get_framec not implemented yet");
}

void meteor_get_field(QSP_ARG_DECL  Movie *mvip,uint32_t f,Data_Obj *dp)
{
	WARN("Sorry, meteor_get_field not implemented yet");
}


void meteor_get_fieldc(QSP_ARG_DECL  Movie *mvip,uint32_t f,Data_Obj* Datadp,int comp)
{
	WARN("Sorry, meteor_get_fieldc not implemented yet");
}

/***************************/

#endif /* HAVE_METEOR */

