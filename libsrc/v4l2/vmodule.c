#include "quip_config.h"


#include <stdio.h>

#include "quip_prot.h"		/* has to come before jmorecfg.h */
#include "xsupp.h"		/* has to come before jmorecfg.h */
#include "fio_api.h"
#include "vmvi.h"
#include "viewer.h"
#include "rv_api.h"
#include "debug.h"
#include "my_video_dev.h"
#include "my_v4l2.h"

#ifdef QUIP_DEBUG
//int v4l2_debug=0;
#endif /* QUIP_DEBUG */

#ifdef HAVE_V4L2

void v4l2_record(QSP_ARG_DECL  uint32_t n_fields,Movie *mvip)
{
	Image_File *ifp;
	uint32_t n_frames_to_request;

	ifp = (Image_File*) mvip->mvi_data;	/* file is opened in v4l2_setup_movie */

	/* have we written the header out yet?
	 * If not, then we can go in and set the number of frames now!
	 */

	if( ifp == NULL ){
		warn("Image file has not been opened");
		return;
	}
	/* we assume the request is for a number of fields...
	 * Unfortunately, or stupidly, or whatever, v4l2_record_clip() takes
	 * its agrument in fields only if the meteor is in field mode, otherwise
	 * it's in frames...
	 */

	if( v4l2_field_mode ){
		/* specify record length in fields, so nothing to do */
		n_frames_to_request = n_fields;
	} else {
		if( n_fields % 2 ){
			warn("meteor records even number of fields, rounding up");
			n_fields++;
		}
		n_frames_to_request = n_fields / 2;
	}

	v4l2_record_clip(ifp,n_frames_to_request);

	/* v4l2_record_clip closes the ifp, so we need to eradicate the
	 * reference to it now.
	 */
	mvip->mvi_data = NULL;
}

#endif /* HAVE_V4L2 */

/* open a file for playpack */

void v4l2_open_movie(QSP_ARG_DECL  const char *filename)
{
	advise("meteor open_movie is a NOP");
}

void v4l2_play_movie(QSP_ARG_DECL  Movie *mvip)
{
	Image_File *ifp;

	ifp= (Image_File *) mvip->mvi_data;
#ifdef HAVE_X11_EXT
	play_rawvol_movie(ifp);
#else
	warn("v4l2_play_movie:  Program was configured without X11 Extensions.");
#endif
}

void v4l2_reverse_movie(QSP_ARG_DECL  Movie *mvip)
{
	sprintf(ERROR_STRING,
		"v4l2_reverse_movie:  Sorry, meteor can't reverse movie %s",
		MOVIE_NAME(mvip));

	warn(ERROR_STRING);
}

void v4l2_shuttle_movie(QSP_ARG_DECL  Movie* mvip,uint32_t frame)
{
	Image_File *ifp;

sprintf(ERROR_STRING,"v4l2_shuttle_movie %s %d",MOVIE_NAME(mvip),frame);
advise(ERROR_STRING);
	ifp= (Image_File *) mvip->mvi_data;

#ifdef HAVE_X11_EXT
	play_rawvol_frame(ifp, frame);
#else
	warn("v4l2_shuttle_movie:  Program was configured without X11 Extensions.");
#endif
}


void v4l2_close_movie(QSP_ARG_DECL  Movie *mvip)
{
}

int v4l2_setup_play(QSP_ARG_DECL  Movie *mvip)
{
	return(0);
}

void v4l2_monitor(SINGLE_QSP_ARG_DECL)
{
	//Data_Obj *dp;
	//dp = make_frame_object("tmpfrm",0);

	My_Buffer *mbp;

	assert(curr_vdp!=NULL);
	mbp = &(curr_vdp->vd_buf_tbl[0]);
	assert(mbp!=NULL);
	assert(mbp->mb_dp!=NULL);

#ifdef HAVE_X11_EXT
	//setup_monitor_capture(SINGLE_QSP_ARG);
	// HERE WE NEED TO TURN ON CAPTURE!!!
	fprintf(stderr,"v4l2_monitor:  Need to turn on capture!?\n");
	monitor_v4l2_video(mbp->mb_dp);
#else
	warn("v4l2_monitor:  Program was configured without X11 Extensions.");
#endif
}

void v4l2_wait_play(SINGLE_QSP_ARG_DECL)
{
}

void v4l2_movie_info(QSP_ARG_DECL  Movie *mvip)
{
	Image_File *ifp;

	ifp= (Image_File *) mvip->mvi_data;
	if_info(ifp);
}


/* setup to record a movie */

int v4l2_setup_movie(QSP_ARG_DECL  Movie *mvip,uint32_t n_fields)
{
	Filetype * ftp;
	Image_File *ifp;
	//struct v4l2_geomet _geo;
	uint32_t blocks_per_frame,n_frames;
	int n_allocframes ;
	//int n_disks = rv_get_ndisks();

	/* BUG shouldn't insist upon even field count in field mode */
	if( n_fields % 2 ){
		sprintf(ERROR_STRING,
	"v4l2_setup_movie:  requested number of fields (%d) is odd, rounding up",
			n_fields);
		warn(ERROR_STRING);
		n_fields++;
	}
	if( ! v4l2_field_mode )
		n_frames = n_fields / 2;
	else
		n_frames = n_fields ;

#ifdef QUIP_DEBUG
if( debug & v4l2_debug )
advise("v4l2_setup_movie");
#endif /* QUIP_DEBUG */

	/* used to set mvi_filename here.... */

	/* BUG? set filetype to IFT_DISK? */
	/* maybe more flexible to let the user specify */

	ftp = current_filetype();
	if( FT_CODE(ftp) != IFT_RV ){
		Filetype *new_ftp;

		new_ftp = FILETYPE_FOR_CODE(IFT_RV);
		if( verbose ){
			sprintf(ERROR_STRING,
	"v4l2_setup_movie:  default filetype is %s, resetting to type %s",
				FT_NAME(ftp),
				FT_NAME(new_ftp) );
			advise(ERROR_STRING);
		}
		set_filetype(new_ftp);
	}

	ifp = write_image_file(MOVIE_NAME(mvip),n_frames);

	if( ifp == NULL ) return(-1);

	mvip->mvi_data = ifp;

	/* Need to set up an if_dp here...  BUG?  setup_dummy() ?? */

	//if( v4l2_get_geometry(&_geo) < 0 ){
	//	/* BUG close file */
	//	return(-1);
	//}

	//SET_MOVIE_HEIGHT(mvip, _geo.rows);
	//SET_MOVIE_WIDTH(mvip, _geo.columns);
	// BUG don't assume NTSC
	SET_MOVIE_HEIGHT(mvip, 480);
	SET_MOVIE_WIDTH(mvip, 640);
	SET_MOVIE_FRAMES(mvip, n_frames);

	blocks_per_frame = get_blocks_per_v4l2_frame();

	/* Make sure that the number of frames requested is an integral multiple of n_disks */

	n_allocframes = rv_frames_to_allocate( n_frames );

	if( rv_realloc(MOVIE_NAME(mvip),n_allocframes*blocks_per_frame) < 0 ){
		sprintf(ERROR_STRING,"v4l2_setup_movie:  failed to allocate rawvol storage for %s",
			MOVIE_NAME(mvip));
		warn(ERROR_STRING);
		return(-1);
	}

	return(0);
}

void v4l2_add_frame(QSP_ARG_DECL  Movie *mvip,Data_Obj *dp)
{

#ifdef QUIP_DEBUG
if( debug & v4l2_debug )
advise("v4l2_add_frame");
#endif /* QUIP_DEBUG */

	/* BUG make sure dp & mvip have same dimensions if not 1st frame */
	write_image_to_file((Image_File *) mvip->mvi_data,dp);
}

void v4l2_end_assemble(QSP_ARG_DECL  Movie *mvip)
{
	RV_Inode *inp;

#ifdef QUIP_DEBUG
if( debug & v4l2_debug )
advise("v4l2_end_assemble");
#endif /* QUIP_DEBUG */

	inp = rv_inode_of(MOVIE_NAME(mvip));	/* save this for later */

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
sprintf(ERROR_STRING,"v4l2_end_assemble calling close_image_file, movie is %s, if is %s",
MOVIE_NAME(mvip),((Image_File *)mvip->mvi_data)->if_name);
advise(ERROR_STRING);
		close_image_file((Image_File *) mvip->mvi_data);
	}

	/* We delete the movie because it was open for recording,
	 * and we're too lazy to change the flags...
	 */

	delete_movie(mvip);
	assert( inp != NULL );

	update_movie_database(inp);
}

void v4l2_get_frame(QSP_ARG_DECL  Movie *mvip, uint32_t n, Data_Obj *dp)
{
	Image_File *ifp;

	ifp = (Image_File *) mvip->mvi_data;

	if( image_file_seek(ifp,n) < 0 )
		return;

	read_object_from_file(dp,ifp);
}

void v4l2_get_framec(QSP_ARG_DECL  Movie *mvip, uint32_t n, Data_Obj *dp, int comp)
{
	warn("Sorry, v4l2_get_framec not implemented yet");
}

void v4l2_get_field(QSP_ARG_DECL  Movie *mvip,uint32_t f,Data_Obj *dp)
{
	warn("Sorry, v4l2_get_field not implemented yet");
}


void v4l2_get_fieldc(QSP_ARG_DECL  Movie *mvip,uint32_t f,Data_Obj* Datadp,int comp)
{
	warn("Sorry, v4l2_get_fieldc not implemented yet");
}

void _monitor_v4l2_video(QSP_ARG_DECL  Data_Obj *dp)
{
	init_rawvol_viewer(OBJ_COLS(dp), OBJ_ROWS(dp), OBJ_COMPS(dp));

	/* BUG should make this a background process or something... */
	/* This routine just displays over and over again, no synchronization
	 * with interrupts...
	 */
	while(1){
		display_to_rawvol_viewer(dp);
	}
}

/***************************/


