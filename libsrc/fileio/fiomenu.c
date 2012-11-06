
#include "quip_config.h"

char VersionId_fio_fiomenu[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "fio_prot.h"
#include "../datamenu/dataprot.h"
#include "submenus.h"
#include "debug.h"
#include "data_obj.h"
#include "submenus.h"
#include "img_file.h"
#include "filetype.h"

#include "rv_api.h"

#include "raw.h"	/* set_raw_sizes() ? */
#include "bdf.h"	/* bdf_menu */
#include "menuname.h"

#ifdef HAVE_JPEG_SUPPORT
extern COMMAND_FUNC( jpeg_menu );
#endif /* HAVE_JPEG_SUPPORT */

#ifdef HAVE_PNG 
extern COMMAND_FUNC( png_menu );
#endif /* HAVE_PNG */


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
	int n;
	const char *filetypes[N_FILETYPE];

	for(n=0;n<N_FILETYPE;n++)
		filetypes[n]=ft_tbl[n].ft_name;

	n = (filetype_code) WHICH_ONE("file format",N_FILETYPE,filetypes);
	if( n < 0 ) return;

	set_filetype(QSP_ARG  (filetype_code)n);
}

static COMMAND_FUNC( do_read_image_file )	/** open file for reading */
{
	char prompt[256];
	const char *s;
	Image_File *ifp;

	sprintf(prompt,"input %s file",ft_tbl[get_filetype()].ft_name);
	s = NAMEOF(prompt);

	ifp = img_file_of(QSP_ARG  s);
	if( ifp != NO_IMAGE_FILE ){
		sprintf(ERROR_STRING,"do_read_image_file:  file %s is already open",ifp->if_name);
		WARN(ERROR_STRING);
		return;
	}

	if(  read_image_file(QSP_ARG  s) == NO_IMAGE_FILE )
		WARN("error reading image file");
}

static COMMAND_FUNC( do_write_image_file )
{
	const char *s;
	u_int n;
	Image_File *ifp;

	s=NAMEOF("output image file");
	n=(u_int) HOW_MANY("number of frames");

	ifp = img_file_of(QSP_ARG  s);
	if( ifp != NO_IMAGE_FILE ){
		sprintf(ERROR_STRING,"do_write_image_file:  file %s is already open",ifp->if_name);
		WARN(ERROR_STRING);
		return;
	}

	if( write_image_file(QSP_ARG  s,n) == NO_IMAGE_FILE )
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
	if( dp == NO_OBJ ) return;
	if( ifp == NO_IMAGE_FILE ) return;
	read_object(QSP_ARG  dp, ifp);
}

COMMAND_FUNC( close_all_hips )
{
	Image_File *ifp;
	List *lp;
	Node *np;

	lp = image_file_list(SINGLE_QSP_ARG);
	if( lp==NO_LIST ) return;

	np=lp->l_head;
	while(np!=NO_NODE){
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
	if( ifp == NO_IMAGE_FILE ) return;
	if_info(QSP_ARG  ifp);
}

static COMMAND_FUNC( do_set_raw_sizes )
{
	dimension_t arr[N_DIMENSIONS];
	int i;

	for(i=0;i<N_DIMENSIONS;i++){
		char pmt[LLEN];

		sprintf(pmt,"number of raw %ss",dimension_name[i]);
		arr[i] = HOW_MANY(pmt);
	}
	set_raw_sizes(arr);
}

static COMMAND_FUNC( do_set_raw_prec )
{
	prec_t p;

	p = get_precision(SINGLE_QSP_ARG);
	raw_prec = p;
}

static COMMAND_FUNC( do_seek_frm )
{
	Image_File *ifp;
	u_long n;

	ifp = PICK_IMG_FILE("");
	n = HOW_MANY("frame index");

	if( ifp == NO_IMAGE_FILE ) return;

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
	if( ifp == NO_IMAGE_FILE ) return;

	if( ifp->if_type != IFT_RV ){
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

	if( ifp == NO_IMAGE_FILE ) return;

	if( !yn ) ifp->if_flags |= NO_AUTO_CLOSE;
	else ifp->if_flags &= ~NO_AUTO_CLOSE;
}

static COMMAND_FUNC( do_list_image_files ) { list_img_files(SINGLE_QSP_ARG); }

Command fio_ctbl[]={
{ "filetype",	do_set_filetype,	"specify file format"		},
{ "read",	do_read_image_file,	"open image file for reading"	},
{ "load",	rd_obj,			"load data object from file"	},
{ "write",	do_write_image_file,	"open image file for writing"	},
{ "get",	rd_next,		"read next frame from image file"},
{ "put",	wt_next,		"write next frame to image file"},
{ "close",	do_close_hips,		"close an open image file"	},
{ "close_all",	close_all_hips,		"close all open image files"	},
{ "raw_sizes",	do_set_raw_sizes,	"specify dimensions of raw file image"},
{ "raw_prec",	do_set_raw_prec,	"specify precision of raw file image"},
{ "seek",	do_seek_frm,		"seek to specified frame"	},
{ "list",	do_list_image_files,	"list currently open image files"},
{ "directory",	do_set_iofdir,		"set directory for image/data files"},
{ "info",	do_if_info,		"give info about an open image file"},
{ "delete",	do_delete_imgfile,	"delete file struct"		},
{ "autoclose",	do_set_autoclose,	"disable/enable automatic closure" },
{ "clobber",	do_set_clobber,		"enable/disable file overwrites"},
{ "direct_io",	do_set_direct,		"enable/disable direct i/o (for local disks)"},
{ "rawvol",	rv_menu,		"raw volume submenu"		},

#ifdef HAVE_JPEG_SUPPORT
{ "jpeg",	jpeg_menu,		"JPEG codec submenu"		},
#endif /* HAVE_JPEG_SUPPORT */

{ "bdf",	bdf_menu,		"BDF submenu"			},

#ifdef HAVE_PNG
{ "png",	png_menu,		"PNG submenu"			},
#endif /* HAVE_PNG */

#ifndef MAC
{ "quit",	popcmd,			"quit"				},
#endif /* ! MAC */
{ NULL_COMMAND								}
};

COMMAND_FUNC( fiomenu )
{
	static int inited=0;

	if( !inited ){
		image_file_init(SINGLE_QSP_ARG);
		dm_init(SINGLE_QSP_ARG);/* initialize data menu */
		verfio(SINGLE_QSP_ARG);	/* Version control */
		inited=1;
	}

	PUSHCMD(fio_ctbl,FIO_MENU_NAME);
}

