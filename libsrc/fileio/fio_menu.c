#include "quip_config.h"

#include "fio_prot.h"
#include "quip_prot.h"
#include "query_bits.h"	// LLEN - BUG

//#include "../datamenu/dataprot.h"
//#include "submenus.h"
//#include "debug.h"
//#include "data_obj.h"
//#include "submenus.h"
#include "img_file.h"
#include "fio_api.h"
//#include "filetype.h"
//
//#include "rv_api.h"
//
//#include "raw.h"	/* set_raw_sizes() ? */
//#include "bdf.h"	/* bdf_menu */
//#include "menuname.h"

#include "warn.h"


#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif


#include "quip_menu.h"
#include "dobj_prot.h"


static COMMAND_FUNC( do_set_filetype );
static COMMAND_FUNC( do_read_image_file );
static COMMAND_FUNC( do_write_image_file );
static COMMAND_FUNC( rd_next );
static COMMAND_FUNC( rd_obj );
static COMMAND_FUNC( do_close_hips );
static COMMAND_FUNC( wt_next );
static COMMAND_FUNC( do_if_info );
static COMMAND_FUNC( do_set_raw_sizes );

static COMMAND_FUNC( do_set_iofdir )
{
	const char *s;

	s=NAMEOF("directory for image/data files");
	set_iofile_directory(QSP_ARG  s);
}

static COMMAND_FUNC( do_set_filetype )
{
	Filetype *ftp;

	ftp = pick_file_type(QSP_ARG  "file format");

	if( ftp != NULL )
		set_filetype(QSP_ARG  ftp);
}

static COMMAND_FUNC( do_read_image_file )	/** open file for reading */
{
	char prompt[256];
	const char *s;
	Image_File *ifp;
	Filetype *ftp;

	ftp = current_filetype();
	sprintf(prompt,"input %s file",FT_NAME(ftp));
	s = NAMEOF(prompt);

	if( s == NULL || *s == 0 ){
		WARN("Null filename!?");
		return;
	}

	ifp = img_file_of(QSP_ARG  s);
	if( ifp != NULL ){
		sprintf(ERROR_STRING,"do_read_image_file:  file %s is already open",ifp->if_name);
		WARN(ERROR_STRING);
		return;
	}

	if(  read_image_file(QSP_ARG  s) == NULL )
		WARN("error reading image file");
}

static COMMAND_FUNC( do_write_image_file )
{
	const char *s;
	u_int n;
	Image_File *ifp;

	s=NAMEOF("output image file");
	n=(u_int) HOW_MANY("number of frames");

	if( s == NULL || *s == 0 ){
		WARN("Null filename specified");
		return;
	}

	ifp = img_file_of(QSP_ARG  s);
	if( ifp != NULL ){
		sprintf(ERROR_STRING,"do_write_image_file:  file %s is already open",ifp->if_name);
		WARN(ERROR_STRING);
		return;
	}

	if( write_image_file(QSP_ARG  s,n) == NULL )
		WARN("error writing image file");
}

static COMMAND_FUNC( rd_next )
{
	Data_Obj *dp;
	Image_File *ifp;

	dp=PICK_OBJ("name of image data object" );
	ifp=PICK_IMG_FILE("");

	read_object_from_file(QSP_ARG  dp,ifp);
}

static COMMAND_FUNC( rd_obj )
{
	Data_Obj *dp;
	Image_File *ifp;

	dp=PICK_OBJ( "name of image data object" );
	ifp=PICK_IMG_FILE("");
	if( dp == NULL ) return;
	if( ifp == NULL ) return;
	read_object(QSP_ARG  dp, ifp);
}

static COMMAND_FUNC( do_close_all_hips )
{
	Image_File *ifp;
	List *lp;
	Node *np;

	lp = image_file_list(SINGLE_QSP_ARG);
	if( lp==NULL ) return;

	np=QLIST_HEAD(lp);
	while(np!=NULL){
		ifp=(Image_File *)np->n_data;
		np=np->n_next;
		close_image_file(QSP_ARG  ifp);
	}
}

static COMMAND_FUNC( do_close_hips )
{
	Image_File *ifp;

	ifp=PICK_IMG_FILE("");

	close_image_file(QSP_ARG  ifp);
}

static COMMAND_FUNC( wt_next )
{
	Data_Obj *dp;
	Image_File *ifp;

	dp=PICK_OBJ( "name of image or sequence" );
	ifp=PICK_IMG_FILE("");
	write_image_to_file(QSP_ARG  ifp,dp);
}

static COMMAND_FUNC( do_if_info )
{
	Image_File *ifp;

	ifp = PICK_IMG_FILE("");
	if( ifp == NULL ) return;
	if_info(QSP_ARG  ifp);
}

static COMMAND_FUNC( do_set_raw_sizes )
{
	dimension_t arr[N_DIMENSIONS];
	int i;

	for(i=0;i<N_DIMENSIONS;i++){
		char pmt[LLEN];

		sprintf(pmt,"number of raw %ss",dimension_name[i]);
		arr[i] = (dimension_t)HOW_MANY(pmt);
	}
	set_raw_sizes(arr);
}

static COMMAND_FUNC( do_set_raw_prec )
{
	Precision * prec_p;

	prec_p = get_precision(SINGLE_QSP_ARG);
	set_raw_prec( prec_p );
}

static COMMAND_FUNC( do_seek_frm )
{
	Image_File *ifp;
	dimension_t n;

	ifp = PICK_IMG_FILE("");
	n = (dimension_t)HOW_MANY("frame index");

	if( ifp == NULL ) return;

	image_file_seek(QSP_ARG  ifp,n);
}

static COMMAND_FUNC( do_set_clobber )
{
	image_file_clobber(ASKIF("Allow overwrites of exisiting files") );
}

static COMMAND_FUNC( do_set_direct )
{
	int flag;

	flag=ASKIF("use direct i/o");
#ifdef HAVE_DIRECT_IO
	set_direct_io(flag);
#else /* ! HAVE_DIRECT_IO */
	if( flag )
		advise("ignoring direct i/o request on non-sgi system");
#endif /* ! HAVE_DIRECT_IO */
}

static COMMAND_FUNC( do_delete_imgfile )
{
	Image_File *ifp;

	ifp = PICK_IMG_FILE("");
	if( ifp == NULL ) return;

	if( FT_CODE(IF_TYPE(ifp)) != IFT_RV ){
		sprintf(ERROR_STRING,
	"Use \"close\" for image file %s, not \"delete\"",
			ifp->if_name);
		WARN(ERROR_STRING);
		return;
	}

	delete_image_file(QSP_ARG  ifp);
}

static COMMAND_FUNC( do_set_autoclose )
{
	Image_File *ifp;
	int yn;

	ifp = PICK_IMG_FILE("");
	yn = ASKIF("close automatically after reading last frame");

	if( ifp == NULL ) return;

	if( !yn ) ifp->if_flags |= NO_AUTO_CLOSE;
	else ifp->if_flags &= ~NO_AUTO_CLOSE;
}

static COMMAND_FUNC( do_list_image_files ) { list_img_files(QSP_ARG  tell_msgfile(SINGLE_QSP_ARG)); }

#define ADD_CMD(s,f,h)	ADD_COMMAND(file_io_menu,s,f,h)

MENU_BEGIN(file_io)

ADD_CMD( filetype,	do_set_filetype,	specify file format			)
ADD_CMD( read,		do_read_image_file,	open image file for reading		)
ADD_CMD( load,		rd_obj,			load data object from file		)
ADD_CMD( write,		do_write_image_file,	open image file for writing		)
ADD_CMD( get,		rd_next,		read next frame from image file		)
ADD_CMD( put,		wt_next,		write next frame to image file		)
ADD_CMD( close,		do_close_hips,		close an open image file		)
ADD_CMD( close_all,	do_close_all_hips,	close all open image files		)
ADD_CMD( raw_sizes,	do_set_raw_sizes,	specify dimensions of raw file image	)
ADD_CMD( raw_prec,	do_set_raw_prec,	specify precision of raw file image	)
ADD_CMD( seek,		do_seek_frm,		seek to specified frame			)
ADD_CMD( list,		do_list_image_files,	list currently open image files		)
ADD_CMD( directory,	do_set_iofdir,		set directory for image/data files	)
ADD_CMD( info,		do_if_info,		give info about an open image file	)
ADD_CMD( delete,	do_delete_imgfile,	delete file struct			)
ADD_CMD( autoclose,	do_set_autoclose,	disable/enable automatic closure	)
ADD_CMD( clobber,	do_set_clobber,		enable/disable file overwrites		)
ADD_CMD( direct_io,	do_set_direct,		enable/disable direct i/o (for local disks)	)
#ifdef NOT_YET
ADD_CMD( rawvol,	do_rv_menu,		raw volume submenu			)
#endif /* NOT_YET */

#ifdef HAVE_JPEG_SUPPORT
ADD_CMD( jpeg,		do_jpeg_menu,		JPEG codec submenu			)
#endif /* HAVE_JPEG_SUPPORT */

#ifdef HAVE_BDF
ADD_CMD( bdf,		bdf_menu,		BDF submenu				)
#endif /* HAVE_BDF */

#ifdef HAVE_PNG
ADD_CMD( png,		do_png_menu,		PNG submenu				)
#endif /* HAVE_PNG */

MENU_END(file_io)



COMMAND_FUNC( do_fio_menu )
{
	static int inited=0;

	if( !inited ){
		image_file_init(SINGLE_QSP_ARG);
		dm_init(SINGLE_QSP_ARG);/* initialize data menu */
		//verfio(SINGLE_QSP_ARG);	/* Version control */
		inited=1;
	}

	PUSH_MENU(file_io);
} 


