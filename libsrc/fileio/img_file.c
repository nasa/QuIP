#include "quip_config.h"

#include <stdio.h>

/* these next two includes used to be ifdef SGI */
/* For the old sgi system, we used iopen(), and O_DIRECT ... */
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STATVFS_H
#include <sys/statvfs.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* close()  (on mac) */
#endif

#ifdef HAVE_STRING_H
#include <string.h>		/* strcmp() */
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#include "fio_prot.h"
#include "quip_prot.h"
#include "debug.h"
#include "getbuf.h"
#include "item_prot.h"
//#include "filetype.h"
#include "function.h"	/* prototype for add_sizable() */
#include "img_file.h"

#ifdef HAVE_PNG
#include "img_file/fio_png.h"
#else /* !HAVE_PNG */

#ifdef BUILD_FOR_IOS
#include "img_file/fio_png.h"
#endif // BUILD_FOR_IOS

#endif /* !HAVE_PNG */

#ifdef HAVE_TIFF
#include "img_file/fiotiff.h"
#endif /* HAVE_TIFF */

#include "img_file/wav.h"
//#include "fiojpeg.h"
#include "item_type.h"
//#include "raw.h"
//#include "uio.h"
#include "fileck.h"
#include "warn.h"
//#include "pathnm.h"

#ifdef HAVE_AVI_SUPPORT
#include "my_avi.h"
#endif /* HAVE_AVI_SUPPORT */

#ifdef HAVE_IOPEN
#include "glimage.h"			/* prototype for iopen() */
#endif /* HAVE_IOPEN */

#ifdef QUIP_DEBUG
debug_flag_t debug_fileio=0;
#endif /* QUIP_DEBUG */

static Filetype *curr_ftp=NO_FILETYPE;	// BUG set to default (was HIPS1)
static int no_clobber=1;		// BUG not thread-safe!?

static const char *iofile_directory=NULL;

ITEM_INTERFACE_DECLARATIONS(Image_File,img_file)


static int direct_flag=1;

#ifdef HAVE_GETTIMEOFDAY

static double get_if_seconds(QSP_ARG_DECL  Item *ip,dimension_t frame);
static double get_if_milliseconds(QSP_ARG_DECL  Item *ip,dimension_t frame);
static double get_if_microseconds(QSP_ARG_DECL  Item *ip,dimension_t frame);

static Timestamp_Functions if_tsf={
	{	get_if_seconds,
		get_if_milliseconds,
		get_if_microseconds
	}
};
#endif /* HAVE_GETTIMEOFDAY */

static Item_Type *file_type_itp=NO_ITEM_TYPE;
ITEM_INIT_FUNC(Filetype,file_type)
ITEM_NEW_FUNC(Filetype,file_type)
ITEM_CHECK_FUNC(Filetype,file_type)
ITEM_PICK_FUNC(Filetype,file_type)
ITEM_ENUM_FUNC(Filetype,file_type)

void set_direct_io(int flag)
{
	direct_flag=flag;
}

List *image_file_list(SINGLE_QSP_ARG_DECL)
{
	if( img_file_itp==NO_ITEM_TYPE ) return(NO_LIST);

	return( item_list(QSP_ARG  img_file_itp) );
}

static void update_pathname(Image_File *ifp)
{
	if( ifp->if_pathname != ifp->if_name ){
		rls_str((char *)ifp->if_pathname);
	}

	/* BUG? don't require UNIX delimiters... */

	if( iofile_directory != NULL && *ifp->if_name != '/' ){
		char str[LLEN];
		sprintf(str,"%s/%s",iofile_directory,ifp->if_name);
		ifp->if_pathname = savestr(str);
	} else {
		ifp->if_pathname = ifp->if_name;
	}
}

void set_iofile_directory(QSP_ARG_DECL  const char *dirname)
{
	if( !directory_exists(QSP_ARG  dirname) ){
		sprintf(ERROR_STRING,
	"Directory %s does not exist or is not a directory", dirname);
		WARN(ERROR_STRING);
		return;
	}

	if( iofile_directory != NULL ){
		rls_str(iofile_directory);
	}

	iofile_directory = savestr(dirname);
}

static FIO_WT_FUNC( dummy )
{
fprintf(stderr,"dummy_wt:  doing nothing!?\n");
	return(0);
}

static FIO_INFO_FUNC( null )
{}

static FIO_CONV_FUNC(null)
{ return(-1); }

static FIO_UNCONV_FUNC(null)
{ return(-1); }

static FIO_RD_FUNC( null ) {}
static FIO_OPEN_FUNC( null ) { return(NO_IMAGE_FILE); }
static FIO_CLOSE_FUNC( null ) {}

static FIO_SEEK_FUNC(null)
{
	sprintf(ERROR_STRING,"Sorry, can't seek on file %s",ifp->if_name);
	WARN(ERROR_STRING);
	return(-1);
}

#define DECLARE_FILETYPE(stem,code,flags)			\
								\
	ftp = new_file_type(QSP_ARG  #stem);			\
	SET_FT_CODE(ftp,code);					\
	SET_FT_FLAGS(ftp,flags);				\
	SET_FT_OPEN_FUNC(ftp,FIO_OPEN_FUNC_NAME(stem));		\
	SET_FT_READ_FUNC(ftp,FIO_RD_FUNC_NAME(stem));		\
	SET_FT_WRITE_FUNC(ftp,FIO_WT_FUNC_NAME(stem));		\
	SET_FT_CLOSE_FUNC(ftp,FIO_CLOSE_FUNC_NAME(stem));	\
	SET_FT_CONV_FUNC(ftp,FIO_CONV_FUNC_NAME(stem));		\
	SET_FT_UNCONV_FUNC(ftp,FIO_UNCONV_FUNC_NAME(stem));	\
	SET_FT_SEEK_FUNC(ftp,FIO_SEEK_FUNC_NAME(stem));		\
	SET_FT_INFO_FUNC(ftp,FIO_INFO_FUNC_NAME(stem));

#define network_open	null_open
#define network_rd	null_rd
#define network_wt	dummy_wt
#define network_close	null_close
#define network_conv	null_conv
#define network_unconv	null_unconv
#define network_seek	null_seek
#define network_info_func	null_info_func
#define network_seek_frame	null_seek_frame

#define raw_close	generic_imgfile_close
#define raw_seek	uio_seek
#define raw_info_func	null_info_func
#define raw_seek_frame	uio_seek_frame

#define hips1_rd		raw_rd
#define hips1_info_func		null_info_func
#define hips1_seek_frame	uio_seek_frame

#define hips2_info_func		null_info_func
#define hips2_seek_frame	std_seek_frame

#define bmp_wt		dummy_wt
#define bmp_seek_frame	null_seek_frame

static void create_filetypes(SINGLE_QSP_ARG_DECL)
{
	Filetype *ftp;

	//init_filetypes(SINGLE_QSP_ARG);		// init the item type

	DECLARE_FILETYPE(network,IFT_NETWORK,0)
	DECLARE_FILETYPE(raw,IFT_RAW,USE_UNIX_IO|CAN_DO_FORMAT)
	DECLARE_FILETYPE(hips1,IFT_HIPS1,USE_UNIX_IO|CAN_DO_FORMAT)
	DECLARE_FILETYPE(hips2,IFT_HIPS2,USE_STDIO|CAN_DO_FORMAT)
	DECLARE_FILETYPE(ppm,IFT_PPM,USE_STDIO|CAN_DO_FORMAT)
	DECLARE_FILETYPE(ascii,IFT_ASC,USE_STDIO|CAN_DO_FORMAT)
	DECLARE_FILETYPE(wav,IFT_WAV,USE_STDIO|CAN_DO_FORMAT)

#ifdef HAVE_RAWVOL
	DECLARE_FILETYPE(rvfio,IFT_RV,CAN_DO_FORMAT)
#endif // HAVE_RAWVOL

#ifdef HAVE_JPEG_SUPPORT
	DECLARE_FILETYPE(jpeg,IFT_JPEG,USE_STDIO|CAN_DO_FORMAT)
	DECLARE_FILETYPE(lml,IFT_LML,USE_STDIO|CAN_DO_FORMAT)
#endif /* HAVE_JPEG_SUPPORT */

#ifdef HAVE_PNG
	DECLARE_FILETYPE(pngfio,IFT_PNG,USE_STDIO|CAN_DO_FORMAT)
#else /* !HAVE_PNG */
#ifdef BUILD_FOR_IOS
	DECLARE_FILETYPE(pngfio,IFT_PNG,CAN_DO_FORMAT)

#endif // BUILD_FOR_IOS
#endif /* !HAVE_PNG */

#ifdef HAVE_TIFF
	DECLARE_FILETYPE(tiff,IFT_TIFF,CAN_DO_FORMAT)
#endif /* HAVE_TIFF */

#ifdef HAVE_MATIO
	DECLARE_FILETYPE(mat,IFT_MATLAB,CAN_DO_FORMAT)
#endif /* HAVE_MATIO */

#ifdef HAVE_AVI_SUPPORT
	DECLARE_FILETYPE(avi,IFT_AVI,CAN_DO_FORMAT)
#endif /* HAVE_AVI_SUPPORT */

	DECLARE_FILETYPE(bmp,IFT_BMP,USE_STDIO|CAN_READ_FORMAT)
}

static double get_if_size(QSP_ARG_DECL  Item *ip,int index)
{
	Image_File *ifp;

	ifp = (Image_File *)ip;
//#ifdef CAUTIOUS
//	if( ifp->if_dp == NO_OBJ ){
//		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  image file %s (type %s) has no associated data object!?",
//				ifp->if_name,FT_NAME(IF_TYPE(ifp)));
//		NERROR1(DEFAULT_ERROR_STRING);
//	}
//#endif /* CAUTIOUS */
	assert( ifp->if_dp != NO_OBJ );

	return( get_dobj_size(QSP_ARG  ifp->if_dp,index) );
}

static const char * get_if_prec_name(QSP_ARG_DECL  Item *ip)
{
	Image_File *ifp;

	ifp = (Image_File *)ip;
//#ifdef CAUTIOUS
//	if( ifp->if_dp == NO_OBJ ){
//		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  image file %s (type %s) has no associated data object!?",
//				ifp->if_name,FT_NAME(IF_TYPE(ifp)));
//		NERROR1(DEFAULT_ERROR_STRING);
//	}
//#endif /* CAUTIOUS */
	assert( ifp->if_dp != NO_OBJ );

	return( get_dobj_prec_name(QSP_ARG  ifp->if_dp) );
}

static Size_Functions imgfile_sf={
		get_if_size,
		get_if_prec_name
};

static double get_if_il_flg(QSP_ARG_DECL  Item *ip)
{
	Image_File *ifp;

	ifp = (Image_File *)ip;
//#ifdef CAUTIOUS
//	if( ifp->if_dp == NO_OBJ ){
//		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  get_if_il_flg:  image file %s (type %s) has no associated data object!?",
//				ifp->if_name,FT_NAME(IF_TYPE(ifp)));
//		NERROR1(DEFAULT_ERROR_STRING);
//	}
//#endif /* CAUTIOUS */
	assert( ifp->if_dp != NO_OBJ );

	return( get_dobj_il_flg(QSP_ARG  ifp->if_dp) );
}

static Interlace_Functions imgfile_if={
		get_if_il_flg
};


// image_file_init initializes the fileio subsystem

void image_file_init(SINGLE_QSP_ARG_DECL)
{
	static int inited=0;

	if( inited ) return;

//#ifdef CAUTIOUS
//	/* We don't use a table any more */
//	//ft_tbl_check(SINGLE_QSP_ARG);
//#endif /* CAUTIOUS */

#ifdef QUIP_DEBUG
	debug_fileio = add_debug_module(QSP_ARG  "fileio");
#endif /* QUIP_DEBUG */

	/* This may have already been called - e.g. mmvi */
	if( img_file_itp == NO_ITEM_TYPE )
		//img_file_init(SINGLE_QSP_ARG);
		init_img_files(SINGLE_QSP_ARG);

	create_filetypes(SINGLE_QSP_ARG);	// used to be a static table

	// Set the default filetype
	curr_ftp = FILETYPE_FOR_CODE(IFT_HIPS2);

#ifdef HAVE_GETTIMEOFDAY
	add_tsable( QSP_ARG   img_file_itp, &if_tsf, (Item * (*)(QSP_ARG_DECL  const char *))img_file_of);
#endif

	add_sizable(QSP_ARG  img_file_itp,&imgfile_sf,NULL);
	add_interlaceable(QSP_ARG  img_file_itp,&imgfile_if,NULL);

	//setstrfunc("iof_exists",iof_exists);
	DECLARE_STR1_FUNCTION(	iof_exists,	iof_exists )

	define_port_data_type(QSP_ARG  P_IMG_FILE,"image_file","name of image file",
		recv_img_file,
		/* null_proc, */
		(const char *(*)(QSP_ARG_DECL  const char *))pick_img_file,
		(void (*)(QSP_ARG_DECL  Port *,const void *,int)) xmit_img_file);

	inited=1;
}

/*
 * Delete the file struct, assume the file itself has already
 * been closed, if necessary.
 */

void delete_image_file(QSP_ARG_DECL  Image_File *ifp)
{
	// BUG - this can fail for a raw volume file because
	// of permissions mismatch...
	// We need to check BEFORE we get here...

	if( ifp->if_dp != NO_OBJ ){
		/* Do we know that the other resources are already freed? */
		/* If we are reading from a file, then this is a dummy object
		 * and should be freed... but what about writing???
		 */
		givbuf(ifp->if_dp);
		ifp->if_dp = NULL;
	}
	if( ifp->if_pathname != ifp->if_name ){
		rls_str((char *)ifp->if_pathname);
	}
	DEL_IMG_FILE(ifp);
	rls_str((char *)ifp->if_name);

	/* don't free the struct pointer, it's marked available
	 * for reuse by del_item (called from del_img_file)...
	 */
}

/*
 * Close the file associated with this image file structure.
 * Also delete the image file structure.  May appear
 * as tabled close routine for simple filetypes, also
 * may be called from filetype-specific close routine.
 */

void generic_imgfile_close(QSP_ARG_DECL  Image_File *ifp)
{
	if( USES_STDIO(ifp) ){
		if( ifp->if_fp != NULL ) {
			fclose(ifp->if_fp);
		}
	} else if( USES_UNIX_IO(ifp) ){
		if( ifp->if_fd != -1 ){
			close(ifp->if_fd);
		}
	}

	if( HAD_ERROR(ifp) && IS_WRITABLE(ifp) ){
		/* BUG this should only apply to file-system files */
		if( USES_STDIO(ifp) || USES_UNIX_IO(ifp) )
			unlink(ifp->if_pathname);	/* remove file */
	}
	delete_image_file(QSP_ARG  ifp);
}

static Data_Obj *new_temp_dobj(void)
{
	Data_Obj *dp;

	dp = getbuf(sizeof(Data_Obj));
	SET_OBJ_SHAPE( dp, ALLOC_SHAPE );

	return dp;
}

static void rls_temp_dobj(Data_Obj *dp)
{
	givbuf( OBJ_TYPE_INCS(dp) );
	givbuf( OBJ_MACH_INCS(dp) );
	givbuf( OBJ_TYPE_DIMS(dp) );
	givbuf( OBJ_MACH_DIMS(dp) );
	givbuf( OBJ_SHAPE(dp) );
	givbuf(dp);
}

void setup_dummy(Image_File *ifp)
{

//#ifdef CAUTIOUS
//	if( ifp->if_dp != NO_OBJ ){
//		sprintf(DEFAULT_ERROR_STRING,
//	"CAUTIOUS:  image file %s has already been set up with object %s",
//			ifp->if_name,OBJ_NAME(ifp->if_dp));
//		NWARN(DEFAULT_ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( ifp->if_dp == NO_OBJ );

	/* these are not in the database... */

	ifp->if_dp = new_temp_dobj();
	// BUG?  do we need to lock this object?

	SET_OBJ_NAME(ifp->if_dp, ifp->if_name);
	ifp->if_dp->dt_ap = NO_AREA;
	ifp->if_dp->dt_parent = NO_OBJ;
	ifp->if_dp->dt_children = NO_LIST;
	SET_OBJ_DATA_PTR(ifp->if_dp, NULL);
	SET_OBJ_FLAGS(ifp->if_dp, 0);

	SET_OBJ_PREC_PTR(ifp->if_dp,NULL);
}

int open_fd(QSP_ARG_DECL  Image_File *ifp)
{
	int o_direct=0;

#ifdef HAVE_RGB
	if(  FT_CODE(IF_TYPE(ifp))  == IFT_RGB )
		return(0);
#endif /* HAVE_RGB */

#ifdef HAVE_DIRECT_IO

	/* for SGI video disk files,
	 * use sgi extension "direct i/o"
	 * which causes the disk driver to dma directly to user space.
	 *
	 * This only works for a local disk, however,
	 * so we need to stat() the file first and find if it is local.
	 *
	 * BUG - this works ok for read files,
	 * but for writing a file, you can't stat()
	 * a file that doesn't exist!?
	 *
	 * We should get the directory and stat it...
	 */

	if(  FT_CODE(IF_TYPE(ifp))  == IFT_DISK ){
		if( IS_READABLE(ifp) ){
			struct statvfs vfsbuf;

			if( statvfs(ifp->if_pathname,&vfsbuf)< 0 ){
				sprintf(ERROR_STRING,"statvfs (%s):",
					ifp->if_pathname);
				tell_sys_error(ERROR_STRING);
				NWARN("Couldn't determine fs type, not using O_DIRECT");
			} else {
				if( vfsbuf.f_flag & ST_LOCAL ){
					o_direct = O_DIRECT;
				} else {
					o_direct = 0;
				}
			}
		} else {
			/* BUG - should stat the directory... */
			if( direct_flag ){
advise("writing file using DIRECT_IO based on flag (NEED TO FIX!)");
advise(ifp->if_pathname);
				o_direct = O_DIRECT;
			}
		}
	}
		
retry:

#endif /* HAVE_DIRECT_IO */


	if( IS_READABLE(ifp) )
		ifp->if_fd = open(ifp->if_pathname,O_RDONLY|o_direct);
	else

		/* open read-write so can rewrite header nframes */

		ifp->if_fd = open(ifp->if_pathname,
					O_RDWR|O_CREAT|O_TRUNC|o_direct,0644);

	if( ifp->if_fd < 0 ){

#ifdef HAVE_DIRECT_IO
		/* can't do O_DIRECT across NFS */
		if( o_direct && errno==EINVAL ){
			o_direct = 0;
sprintf(ERROR_STRING,"Couldn't open file \"%s\" with direct i/o.",
ifp->if_pathname);
NWARN(ERROR_STRING);
advise("retrying to open write file w/o DIRECT_IO");
			goto retry;
		}
#endif /* HAVE_DIRECT_IO */

		tell_sys_error("open");
		sprintf(DEFAULT_ERROR_STRING,
			"open_fd:  error getting descriptor for %s file %s",
			IS_READABLE(ifp)?"read":"write",ifp->if_pathname);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	return(0);
}

#ifdef HAVE_IOPEN
int ifp_iopen(Image_File *ifp)
{
sprintf(ERROR_STRING,"name %s  rows %d cols %d",
ifp->if_name,OBJ_ROWS(ifp->if_dp),OBJ_COLS(ifp->if_dp));
advise(ERROR_STRING);

	/* BUG maybe this 3 can be set to 2 if tdim==1 */
	ifp->if_hd.rgb_ip = iopen(ifp->if_pathname,"w",VERBATIM(1),3,
		OBJ_ROWS(ifp->if_dp),OBJ_COLS(ifp->if_dp),OBJ_COMPS(ifp->if_dp));

	if( ifp->if_hd.rgb_ip == NULL ){
		sprintf(ERROR_STRING,"Error iopening file %s",ifp->if_pathname);
		NWARN(ERROR_STRING);
		return(-1);
	} else return(0);
}
#endif /* HAVE_IOPEN */

int open_fp(Image_File *ifp)
{
#ifndef HAVE_RGB

	if( IS_READABLE(ifp) ){
		ifp->if_fp = try_open(DEFAULT_QSP_ARG  ifp->if_pathname,"r");
	} else {
		/* open read-write so we can read back
		 * the header if necessary...  (see hips2.c)
		 */
		ifp->if_fp = try_open(DEFAULT_QSP_ARG  ifp->if_pathname,"w+");
	}
	if( ! ifp->if_fp ) return(-1);
	return(0);

#else /* HAVE_RGB */

	if( IS_READABLE(ifp) ){
		if(  FT_CODE(IF_TYPE(ifp))  == IFT_RGB ){
			ifp->if_hd.rgb_ip = iopen(ifp->if_pathname,"r");
			if( ifp->if_hd.rgb_ip == NULL ){
				sprintf(ERROR_STRING,
			"open_fp:  error getting RGB descriptor for file %s",
					ifp->if_pathname);
				NWARN(ERROR_STRING);
				return(-1);
			} else return(0);
		} else {
			ifp->if_fp = TRY_OPEN(ifp->if_pathname,"r");
		}
	} else {
		/* if .rgb defer the iopen until we know the image size */
		if(  FT_CODE(IF_TYPE(ifp))  != IFT_RGB ){
			/* open read-write so we can read back
			 * the header if necessary...  (see hips2.c)
			 */
			ifp->if_fp = TRY_OPEN(ifp->if_pathname,"w+");
		}
	}
	if(  FT_CODE(IF_TYPE(ifp))  != IFT_RGB ){
		if( ! ifp->if_fp ) return(-1);
	}
	return(0);

#endif /* HAVE_RGB */
}

static int check_clobber(QSP_ARG_DECL  Image_File *ifp)
{
	const char *dir;

	dir=parent_directory_of(ifp->if_pathname);


	/* now see if the file already exists, and if
	 * our application is permitting clobbers, then
	 * check the file system permissions
	 */

	if( file_exists(QSP_ARG  ifp->if_pathname) ){
		if( no_clobber ){
			sprintf(DEFAULT_ERROR_STRING,
				"Not clobbering existing file \"%s\"",
				ifp->if_pathname);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
		} else if( !can_write_to(QSP_ARG  ifp->if_pathname) )
			return(-1);
	} else {
		/* We may have write permissions to the file even if we don't have
		 * write permissions on the directory, e.g. /dev/null
		 */
		if( !can_write_to(QSP_ARG  dir) ){
			sprintf(DEFAULT_ERROR_STRING, "Can't write to directory \"%s\"", dir);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
		}
	}

	return(0);
}

void image_file_clobber(int flag)
{
	if( flag ) no_clobber=0;
	else no_clobber=1;
}

/*
 * This routine creates and initializes an image file struct,
 * and then opens the file using a method appropriate to the file type.
 *
 * Typically called from module open routines.
 *
 * It used to be called image_file_open, but that was too confusing
 * given the existence of a different function called open_image_file...
 */

Image_File *img_file_creat(QSP_ARG_DECL  const char *name,int rw,Filetype * ftp)
{
	Image_File *ifp;
	int had_error=0;

	if( rw == FILE_READ && CANNOT_READ(ftp) ){
		sprintf(ERROR_STRING,"Sorry, don't know how to read %s files",
			FT_NAME(ftp));
		NWARN(ERROR_STRING);
		return(NO_IMAGE_FILE);
	} else if( rw == FILE_WRITE && CANNOT_WRITE(ftp) ){
		sprintf(ERROR_STRING,"Sorry, don't know how to write %s files",
			FT_NAME(ftp));
		NWARN(ERROR_STRING);
		return(NO_IMAGE_FILE);
	}

	ifp = new_img_file(QSP_ARG  name);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->if_flags = (short) rw;
	ifp->if_nfrms = 0;

	ifp->if_pathname = ifp->if_name;	/* default */
	update_pathname(ifp);

	ifp->if_dp = NO_OBJ;

	// what is the "dummy" used for, and when do we release it?
	if( IS_READABLE(ifp) ) setup_dummy(ifp);

	if( IS_WRITABLE(ifp) ){
		if( check_clobber(QSP_ARG  ifp) < 0 ){
			had_error=1;
			goto dun;
		}
	}

	ifp->if_ftp = ftp;

	if( USES_STDIO(ifp) ){
		if( open_fp(ifp) < 0 ) had_error=1;
	} else if( USES_UNIX_IO(ifp) ){
		if( open_fd(QSP_ARG  ifp) < 0 ) had_error=1;
	} else {
		// This used to be a CAUTIOUS error, but some
		// filetypes (tiff, matlab) have their own
		// open routines...  So we do - nothing?
	}

#ifdef FOOBAR
//#ifdef CAUTIOUS
	  else {
//	  	sprintf(ERROR_STRING,
//"CAUTIOUS:  img_file_creat:  file type %s doesn't specify i/o type?",
//			FT_NAME(ftp));
//	  	ERROR1(ERROR_STRING);
		assert( AERROR("img_file_creat:  file type %s doesn't specify i/o type?") );
	}
//#endif // CAUTIOUS
#endif // FOOBAR

dun:
	if( had_error ){
		if( IS_READABLE(ifp) ){
			//givbuf(ifp->if_dp);
			rls_temp_dobj(ifp->if_dp);
		}
		DEL_IMG_FILE(ifp);
		/* BUG? should also rls_str(name) here???
		 * Or does del_item release the naem???
		 */
		return(NO_IMAGE_FILE);
	}

	return(ifp);
}

int same_size(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp)
{
	if( ifp->if_dp == NO_OBJ ){
		sprintf(ERROR_STRING,"No size/prec info about image file %s",
			ifp->if_name);
		WARN(ERROR_STRING);
		return(0);
	}

	if(
		( OBJ_ROWS(ifp->if_dp) != 0 && OBJ_ROWS(ifp->if_dp) != OBJ_ROWS(dp) )	||
		( OBJ_COLS(ifp->if_dp) != 0 && OBJ_COLS(ifp->if_dp) != OBJ_COLS(dp) )
		){

		sprintf(ERROR_STRING,"size mismatch, object %s and file %s",
			OBJ_NAME(dp),ifp->if_name);
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,"      %-24s %-24s",
			OBJ_NAME(dp),ifp->if_name);
		advise(ERROR_STRING);
		sprintf(ERROR_STRING,"rows: %-24d %-24d",OBJ_ROWS(dp),
			OBJ_ROWS(ifp->if_dp));
		advise(ERROR_STRING);
		sprintf(ERROR_STRING,"cols: %-24d %-24d",OBJ_COLS(dp),
			OBJ_COLS(ifp->if_dp));
		advise(ERROR_STRING);
		return(0);
	}
	return(1);
}

int same_type(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp)
{
	int retval=1;

	if( ifp->if_dp == NO_OBJ ){
		sprintf(ERROR_STRING,"No size/prec info about image file %s",
			ifp->if_name);
		WARN(ERROR_STRING);
		return(0);
	}

	if( OBJ_PREC(dp) != OBJ_PREC(ifp->if_dp) ){
		/* special case for unsigned (hips doesn't record this) */
		if(
		    (OBJ_PREC(dp) == PREC_UDI && OBJ_PREC(ifp->if_dp) == PREC_DI) ||
		    (OBJ_PREC(dp) == PREC_UIN && OBJ_PREC(ifp->if_dp) == PREC_IN) ||
		    (OBJ_PREC(dp) == PREC_UBY && OBJ_PREC(ifp->if_dp) == PREC_BY)
		){
			/* it's ok */
		} else {
			sprintf(ERROR_STRING,
	"Pixel format (%s) for file %s\n\tdoes not match object %s precision (%s)",
	PREC_NAME(OBJ_PREC_PTR(ifp->if_dp)),ifp->if_name,
	OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)));
			WARN(ERROR_STRING);
			retval=0;
		}
	}

	if( OBJ_COMPS(dp) != OBJ_COMPS(ifp->if_dp) ){
		sprintf(ERROR_STRING,
	"Pixel dimension (%d) for file %s\n\tdoes not match pixel dimension (%d) for object %s",
	OBJ_COMPS(ifp->if_dp),ifp->if_name,OBJ_COMPS(dp),OBJ_NAME(dp));
		WARN(ERROR_STRING);
		retval=0;
	}
	return(retval);
}

void copy_dimensions(Data_Obj *dpto,Data_Obj *dpfr)	/* used by write routines... */
{
	int i;

	for(i=0;i<N_DIMENSIONS;i++){
		SET_OBJ_TYPE_DIM(dpto,i,OBJ_TYPE_DIM(dpfr,i));
		SET_OBJ_TYPE_INC(dpto,i,OBJ_TYPE_INC(dpfr,i));
	}
	SET_OBJ_PREC_PTR(dpto, OBJ_PREC_PTR(dpfr));
	SET_OBJ_FLAGS(dpto, OBJ_FLAGS(dpfr));
	SET_OBJ_MAXDIM(dpto, OBJ_MAXDIM(dpfr) );
	SET_OBJ_MINDIM(dpto, OBJ_MINDIM(dpfr) );
	SET_OBJ_N_TYPE_ELTS(dpto,OBJ_N_TYPE_ELTS(dpfr) );
}

void if_info(QSP_ARG_DECL  Image_File *ifp)
{
	sprintf(msg_str,"File %s:",ifp->if_name);
	prt_msg(msg_str);
	sprintf(msg_str,"\tpathname:  %s:",ifp->if_pathname);
	prt_msg(msg_str);
	sprintf(msg_str,"\t%s format (%d)",FT_NAME(IF_TYPE(ifp)),FT_CODE(IF_TYPE(ifp)) );
	prt_msg(msg_str);
	if( ifp->if_dp != NO_OBJ ){
		sprintf(msg_str,"\t%s pixels",PREC_NAME(OBJ_PREC_PTR(ifp->if_dp)));
		prt_msg(msg_str);
		if( OBJ_SEQS(ifp->if_dp) > 1 ){
			sprintf(msg_str,
			"\t%d sequences, ",OBJ_SEQS(ifp->if_dp));
			prt_msg_frag(msg_str);
		} else {
			sprintf(msg_str,"\t");
			prt_msg_frag(msg_str);
		}

#define INFO_ARGS( n )	n , n==1?"":"s"

		sprintf(msg_str,
		"%d frame%s, %d row%s, %d column%s, %d component%s",
			INFO_ARGS( OBJ_FRAMES(ifp->if_dp) ),
			INFO_ARGS( OBJ_ROWS(ifp->if_dp) ),
			INFO_ARGS( OBJ_COLS(ifp->if_dp) ),
			INFO_ARGS( OBJ_COMPS(ifp->if_dp) )
			);
		prt_msg(msg_str);
	}
	if( IS_READABLE(ifp) ){
		prt_msg("\topen for reading");
		sprintf(msg_str,"\t%d frame%s already read",
			INFO_ARGS(ifp->if_nfrms));
		prt_msg(msg_str);
	} else if( IS_WRITABLE(ifp) ){
		prt_msg("\topen for writing");
		sprintf(msg_str,"\t%d frame%s already written",
			INFO_ARGS(ifp->if_nfrms));
		prt_msg(msg_str);
	}
//#ifdef CAUTIOUS
	else
//		prt_msg("Wacky RW mode!?");
		assert( AERROR("Wacky RW mode!?") );
//#endif // CAUTIOUS

	/* print format-specific info, if any */
	(*FT_INFO_FUNC(IF_TYPE(ifp)))(QSP_ARG  ifp);
}

/* typical usage:
 *	dump_image_file("foo.viff",IFT_VIFF,buffer,width,height,PREC_BY);
 */

void dump_image_file(QSP_ARG_DECL  const char *filename,Filetype *ftp,void *data,dimension_t width,dimension_t height,Precision *prec_p)
{
	Data_Obj *dp;
	Image_File *ifp;

	dp = new_temp_dobj();

	SET_OBJ_DATA_PTR(dp, data);
	SET_OBJ_NAME(dp, "dump_image");

	SET_OBJ_COMPS(dp,1);
	SET_OBJ_COMP_INC(dp,1);
	SET_OBJ_COLS(dp,width);
	SET_OBJ_PXL_INC(dp,1);
	SET_OBJ_ROWS(dp,height);
	SET_OBJ_ROW_INC(dp,(incr_t)width);
	SET_OBJ_FRAMES(dp,1);
	SET_OBJ_FRM_INC(dp,(incr_t)(width*height));
	SET_OBJ_SEQS(dp,1);
	SET_OBJ_SEQ_INC(dp,(incr_t)(width*height));

	SET_OBJ_N_TYPE_ELTS(dp,width*height);

	SET_OBJ_PREC_PTR(dp,prec_p);

	ifp=(*FT_OPEN_FUNC(ftp))(QSP_ARG  filename,FILE_WRITE);
	if( ifp == NO_IMAGE_FILE ){
		rls_temp_dobj(dp);
		return;
	}
	write_image_to_file(QSP_ARG  ifp,dp);
	rls_temp_dobj(dp);
}

#ifdef FOOBAR

/*
 * this one should allocate it's own storage
 */

void *load_image_file(char *name,filetype_code input_file_type,filetype_code desired_hdr_type)
{
	Image_File *ifp;
	Data_Obj *dp;
	void *new_hdp;

	ifp=open_image_file(name,"r");
	if( ifp == NO_IMAGE_FILE ) return(NULL);

	/*
	dp = make_dobj(name,ifp->if_dp->dt_type_dim,OBJ_PREC(ifp->if_dp));
	*/
	/* originally we gave this the filename, but use localname() to
	 * work with Carlo's hack
	 */
	dp = make_dobj(localname(),
		ifp->if_dp->dt_type_dim,OBJ_PREC(ifp->if_dp));

	if( dp == NO_OBJ ) return(NULL);

	/* read the data */
	(*FT_READ_FUNC(ftp))(dp,ifp,0,0,0);

	/* convert to desired type */
	if((*FT_UNCONV_FUNC(ftp))(&new_hdp,dp) < 0 )
		return(NULL);

	return(new_hdp);
}
#endif /* FOOBAR */

/* filetype independent stuff lifted from fiomenu.c */

Filetype * current_filetype(void)
{
	return(curr_ftp);
}

void set_filetype(QSP_ARG_DECL  Filetype *ftp)
{
	curr_ftp=ftp;
}

static Item_Type *suffix_itp=NO_ITEM_TYPE;

typedef struct known_suffix {
	const char *	sfx_name;
	Filetype *	sfx_ftp;
} Known_Suffix;

#define NO_SUFFIX		((Known_Suffix *)NULL)

static ITEM_INIT_FUNC(Known_Suffix,suffix)
static ITEM_NEW_FUNC(Known_Suffix,suffix)
static ITEM_CHECK_FUNC(Known_Suffix,suffix)

Filetype *filetype_for_code(QSP_ARG_DECL  filetype_code code)
{
	List *lp;
	Node *np;
	Filetype *ftp;

	lp = file_type_list(SINGLE_QSP_ARG);
	np=QLIST_HEAD(lp);
	while(np!=NO_NODE){
		ftp = (Filetype *) NODE_DATA(np);
		if( FT_CODE(ftp) == code ) return ftp;
		np = NODE_NEXT(np);
	}

	sprintf(ERROR_STRING,"filetype_for_code:  no filetype defined for code %d!?",code);
	WARN(ERROR_STRING);

	return NO_FILETYPE;
}

static void init_suffix( QSP_ARG_DECL  const char *name, filetype_code code )
{
	Known_Suffix *sfx_p;
	Filetype *ftp;

	ftp = filetype_for_code(QSP_ARG  code);
//#ifdef CAUTIOUS
//	if( ftp == NO_FILETYPE ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  init_suffix:  No filetype found for suffix \"%s\"!?",name);
//		WARN(ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( ftp != NO_FILETYPE );

	sfx_p = new_suffix(QSP_ARG  name);
//#ifdef CAUTIOUS
//	if( sfx_p == NO_SUFFIX ) {
//		ERROR1("CAUTIOUS:  init_suffix:  Error creating filetype suffix!?");
//		IOS_RETURN
//	}
//#endif /* CAUTIOUS */
	assert( sfx_p != NO_SUFFIX );

	/* Now file the filetype struct with this code... */
	sfx_p->sfx_ftp = ftp;
}

#define INIT_SUFFIX(name,code)	init_suffix(QSP_ARG  name, code)

static void init_suffixes(SINGLE_QSP_ARG_DECL)
{
	INIT_SUFFIX( "hips2",	IFT_HIPS2);
	INIT_SUFFIX( "raw",	IFT_RAW	);
#ifdef HAVE_KHOROS
	INIT_SUFFIX( "viff",	IFT_VIFF);
#endif /* HAVE_KHOROS */
#ifdef HAVE_TIFF
	INIT_SUFFIX( "tiff",	IFT_TIFF);
	INIT_SUFFIX( "tif",	IFT_TIFF);
#endif /* HAVE_TIFF */
#ifdef HAVE_RGB
	INIT_SUFFIX( "rgb",	IFT_RGB	);
#endif /* HAVE_RGB */
	INIT_SUFFIX( "ppm",	IFT_PPM	);
#ifdef HAVE_JPEG_SUPPORT
	INIT_SUFFIX( "jpg",	IFT_JPEG);
	INIT_SUFFIX( "JPG",	IFT_JPEG);
	INIT_SUFFIX( "lml",	IFT_LML	);
	INIT_SUFFIX( "asc",	IFT_ASC	);
	INIT_SUFFIX( "jpeg",	IFT_JPEG);
	INIT_SUFFIX( "mjpg",	IFT_JPEG);
#endif /* HAVE_JPEG_SUPPORT */

#ifdef HAVE_PNG
	INIT_SUFFIX( "png",	IFT_PNG	);
#else /* !HAVE_PNG */
#ifdef BUILD_FOR_IOS
	INIT_SUFFIX( "png",	IFT_PNG	);
#endif // BUILD_FOR_IOS
#endif /* !HAVE_PNG */

#ifdef HAVE_MATIO
	INIT_SUFFIX( "mat",	IFT_MATLAB);
#endif /* HAVE_MATIO */

#ifdef HAVE_MPEG
	INIT_SUFFIX( "mpeg",	IFT_MPEG);
	INIT_SUFFIX( "mpg",	IFT_MPEG);
#endif /* HAVE_MPEG */

#ifdef HAVE_QUICKTIME
	INIT_SUFFIX( "mov",	IFT_QT	);
#endif
	INIT_SUFFIX( "hips1",	IFT_HIPS1);

#ifdef HAVE_AVI_SUPPORT
	INIT_SUFFIX( "avi",	IFT_AVI	);
#endif /* HAVE_AVI_SUPPORT */

	INIT_SUFFIX( "bmp",	IFT_BMP	);
	INIT_SUFFIX( "BMP",	IFT_BMP	);
	INIT_SUFFIX( "wav",	IFT_WAV	);

#ifdef NOT_YET

	INIT_SUFFIX( "vst",	IFT_VISTA);
	INIT_SUFFIX( "dsk",	IFT_DISK);
	INIT_SUFFIX( "vl",	IFT_VL	);
	INIT_SUFFIX( "ras",	IFT_SUNRAS);
	INIT_SUFFIX( "WAV",	IFT_WAV	);
	INIT_SUFFIX( "bdf",	IFT_BDF	);


#endif /* NOT_YET */

}

static Filetype* infer_filetype_from_name(QSP_ARG_DECL  const char *name)
{
	Known_Suffix *sfx_p;
	const char *suffix=NULL;
	const char *s;
	static int suffixes_inited=0;

	/* First make sure that we've initialized the table of suffixes */
	if( ! suffixes_inited ){
		init_suffixes(SINGLE_QSP_ARG);
		suffixes_inited=1;
	}

	/* set the suffix to the string following the last '.' */

	s=name;
	while(*s!=0){
		if( *s == '.' ) suffix = s+1;
		s++;
	}
	if( suffix == NULL || *suffix == 0 ) return(NO_FILETYPE);

	sfx_p = suffix_of(QSP_ARG  suffix);
	if( sfx_p == NO_SUFFIX ){
		// Print an advisory
		sprintf(ERROR_STRING,"File suffix \"%s\" is not known!?",
			suffix);
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,"Using current file type %s",
			FT_NAME(curr_ftp) );
		advise(ERROR_STRING);
		return(NO_FILETYPE);
	}
	return( sfx_p->sfx_ftp );
}

/*
 * Call type-specific function to open the file
 */

Image_File *read_image_file(QSP_ARG_DECL  const char *name)
{
	Image_File *ifp;
	Filetype * ftp;

	ftp = infer_filetype_from_name(QSP_ARG  name);
	if( ftp == NO_FILETYPE ){
//#ifdef CAUTIOUS
//		if( curr_ftp == NO_FILETYPE ) ERROR1("CAUTIOUS:  read_image_file:  default filetype is not set!?");
//#endif /* CAUTIOUS */
		assert( curr_ftp != NO_FILETYPE );

		ftp=curr_ftp;	/* use default if can't figure it out */
	} else if( verbose && ftp!=curr_ftp ){
		sprintf(ERROR_STRING,"Inferring filetype %s from filename %s, overriding default %s",
			FT_NAME(ftp),name,FT_NAME(curr_ftp));
		advise(ERROR_STRING);
	}

	if( CANNOT_READ(ftp) ){
		sprintf(ERROR_STRING,"Sorry, can't read files of type %s",
			FT_NAME(ftp));
		NWARN(ERROR_STRING);
		return(NO_IMAGE_FILE);
	}

	/* pathname hasn't been set yet... */
	ifp=(*ftp->op_func)( QSP_ARG  name, FILE_READ );

	if( ifp == NO_IMAGE_FILE ) {
		sprintf(ERROR_STRING,
			"error reading %s file \"%s\"",FT_NAME(ftp),name);
		NWARN(ERROR_STRING);
	}
	return(ifp);
}

/* Open a file for writing */

Image_File *write_image_file(QSP_ARG_DECL  const char *filename,dimension_t n)
{
	Image_File *ifp;
	Filetype * ftp;

	ftp = infer_filetype_from_name(QSP_ARG  filename);
	if( ftp == NO_FILETYPE ) {	/* use default if can't figure it out */
//#ifdef CAUTIOUS
//		if( curr_ftp == NO_FILETYPE ) ERROR1("CAUTIOUS:  write_image_file:  default filetype is not set!?");
//#endif /* CAUTIOUS */
		assert( curr_ftp != NO_FILETYPE );

		ftp=curr_ftp;	/* use default if can't figure it out */
	} else if( ftp != curr_ftp ){
		sprintf(ERROR_STRING,"Inferring filetype %s from filename %s, overriding default %s",
			FT_NAME(ftp),filename,FT_NAME(curr_ftp));
		advise(ERROR_STRING);
		// Should we make this filetype the new current default???
	}

	if( CANNOT_WRITE(ftp) ){
		sprintf(ERROR_STRING,"Sorry, can't write files of type %s",
			FT_NAME(ftp));
		NWARN(ERROR_STRING);
		return(NO_IMAGE_FILE);
	}

	ifp = (*FT_OPEN_FUNC(ftp))( QSP_ARG  filename, FILE_WRITE ) ;
	if( ifp != NO_IMAGE_FILE )
		ifp->if_frms_to_wt = n;

	return(ifp);
}

/* Should we impose that the objects have the same size?? */

void read_object_from_file(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp)
{
	if( dp == NO_OBJ ) return;
	if( ifp == NO_IMAGE_FILE ) return;

	if( !IS_READABLE(ifp) ){
		sprintf(ERROR_STRING,"File %s is not readable",ifp->if_name);
		WARN(ERROR_STRING);
		return;
	}

	if( OBJ_ROWS(ifp->if_dp) != 0 && OBJ_ROWS(dp) != OBJ_ROWS(ifp->if_dp) ){
		sprintf(ERROR_STRING,"Row count mismatch, object %s (%d) and file %s (%d)",
				OBJ_NAME(dp),OBJ_ROWS(dp),ifp->if_name,OBJ_ROWS(ifp->if_dp));
		WARN(ERROR_STRING);
	}

	if( OBJ_COLS(ifp->if_dp) != 0 && OBJ_COLS(dp) != OBJ_COLS(ifp->if_dp) ){
		sprintf(ERROR_STRING,"Column count mismatch, object %s (%d) and file %s (%d)",
				OBJ_NAME(dp),OBJ_COLS(dp),ifp->if_name,OBJ_COLS(ifp->if_dp));
		WARN(ERROR_STRING);
	}

	/* was this nfrms a BUG, or was there a reason??? */
	
	(*FT_READ_FUNC(IF_TYPE(ifp)))(QSP_ARG  dp,ifp,0,0,/* ifp->hf_nfrms */ 0 );
}

/*
 * Filetype independent way to close an image file.
 * Calls routine from table.
 */

void close_image_file(QSP_ARG_DECL  Image_File *ifp)
{
	if( ifp == NO_IMAGE_FILE ) return;
	(*FT_CLOSE_FUNC(IF_TYPE(ifp)))(QSP_ARG  ifp);
}

/*
 * Open for reading or writing
 *
 * High level routine, calls r/w specific routine, which may
 * call vectored module-specific routine...
 */

Image_File * open_image_file(QSP_ARG_DECL  const char *filename,const char *rw)
{
	Image_File *ifp;

sprintf(ERROR_STRING,"open_image_file %s",filename);
advise(ERROR_STRING);
	if( *rw == 'r' )
		ifp = read_image_file(QSP_ARG  filename);

	/* BUG 4096 is an arbitrary big number.  Originally we
	 * passed the number of frames to write to the open routine
	 * so we could write the header; Now for hips1 we can go
	 * back and edit it later...  need to support this feature
	 * for hips2, viff and ???
	 */

	else if( *rw == 'w' )
		ifp = write_image_file(QSP_ARG  filename,4096);

//#ifdef CAUTIOUS
	else {
//		WARN("CAUTIOUS:  bad r/w string passed to open_image_file");
//		ifp = NO_IMAGE_FILE;
		assert( AERROR("bad r/w string passed to open_image_file") );
	}
//#endif /* CAUTIOUS */

	return(ifp);
}

/* put an image out to a writable file */

void write_image_to_file(QSP_ARG_DECL  Image_File *ifp,Data_Obj *dp)
{
	/* take filetype from image file */
	if( dp == NO_OBJ ) return;
	if( ifp == NO_IMAGE_FILE ) return;

	if( ! IS_WRITABLE(ifp) ){
		sprintf(ERROR_STRING,"File %s is not writable",ifp->if_name);
		WARN(ERROR_STRING);
		return;
	}
    
	(*FT_WRITE_FUNC(IF_TYPE(ifp)))(QSP_ARG  dp,ifp);
}

static off_t fio_seek_setup(Image_File *ifp, index_t n)
{
	off_t frms_to_seek;
	index_t frmsiz;

	// if_nfrms holds the number of frames already read...
	// index_t is unsigned, so we have to cast this to be sure to get
	// the correct sign...
	frms_to_seek = ((off_t)n) - ifp->if_nfrms;

	/* figure out frame size */

	frmsiz = OBJ_COLS(ifp->if_dp) * OBJ_ROWS(ifp->if_dp)
		* OBJ_COMPS(ifp->if_dp) * PREC_SIZE(OBJ_PREC_PTR(ifp->if_dp));

	if( FT_CODE(IF_TYPE(ifp)) == IFT_DISK ){
		/* round up to block size */
		frmsiz += 511;
		frmsiz &= ~511;
		/* BUG? add a 1?? */
	}

	/* although IFT_RV frames are also rounded up to blocksize,
	 * we pass the frame number to rv_seek_frame() and let it worry
	 * about it...
	 */

	return( frms_to_seek * frmsiz );
}

int uio_seek_frame(QSP_ARG_DECL  Image_File *ifp, index_t n)
{
	off_t offset;

	offset = fio_seek_setup(ifp,n);

	if( lseek(ifp->if_fd,offset,SEEK_CUR) < 0 ){
		WARN("lseek error");
		return(-1);
	}
	return(0);
}

int std_seek_frame(QSP_ARG_DECL  Image_File *ifp, index_t n)
{
	off_t offset;

	offset = fio_seek_setup(ifp,n);

	if( offset == 0 ){
		if( verbose ) advise("Seek to current location requested!?");
		return(0);	/* nothing to do */
	}
	if( fseek(ifp->if_fp,(long)offset,/*1*/ SEEK_CUR) != 0 ){
		WARN("fseek error");
		return(-1);
	}
	return(0);
}

int image_file_seek(QSP_ARG_DECL  Image_File *ifp,dimension_t n)
{
	/* BUG?  off_t is long long on new sgi!? */

#ifdef QUIP_DEBUG
if( debug & debug_fileio ){
sprintf(ERROR_STRING,"image_file_seek %s %d",
		ifp->if_name,n);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	if( ! IS_READABLE(ifp) ){
		sprintf(ERROR_STRING,"File %s is not readable, can't seek",
			ifp->if_name);
		WARN(ERROR_STRING);
		return(-1);
	}
	if( n >= OBJ_FRAMES(ifp->if_dp) ){
		sprintf(ERROR_STRING,
	"Frame index %d is out of range for file %s (%d frames)",
			n,ifp->if_name,OBJ_FRAMES(ifp->if_dp));
		WARN(ERROR_STRING);
		return(-1);
	}

	/* how do we figure out what frame we are at currently?
	 * It looks like if_nfrms holds the index of the next frame
	 * we will read...
	 *
	 * For most of our formats, the frame size is fixed so we can
	 * just calculate the offset.  For jpeg, we have to use a seek
	 * table that we construct when we first scan the file for frames.
	 *
	 * This would be cleaner if we vectored the seek routine...
	 * rv is a special case, since we have multiple disks to worry about.
	 */

	if( (*FT_SEEK_FUNC(IF_TYPE(ifp)))(QSP_ARG  ifp,n) < 0 ){
		sprintf(ERROR_STRING,"Error seeking frame %d on file %s",n,ifp->if_name);
		WARN(ERROR_STRING);
		return(-1);
	}
	ifp->if_nfrms = n;
	return(0);
}

void check_auto_close(QSP_ARG_DECL  Image_File *ifp)
{
	if( ifp->if_nfrms >= ifp->if_frms_to_wt ){
		if( verbose ){
	sprintf(ERROR_STRING, "closing file \"%s\" after writing %d frames",
			ifp->if_name,ifp->if_nfrms);
			advise(ERROR_STRING);
		}
		close_image_file(QSP_ARG  ifp);
	}
}

double iof_exists(QSP_ARG_DECL  const char *s)
{
	Image_File *ifp;

	ifp=img_file_of(QSP_ARG  s);
	if( ifp==NO_IMAGE_FILE ) return(0.0);
	else return(1.0);
}

static double get_if_seconds(QSP_ARG_DECL  Item *ip,dimension_t frame)
{
	Image_File *ifp;

	ifp = (Image_File *) ip;

	switch( FT_CODE(IF_TYPE(ifp)) ){

#ifdef HAVE_JPEG_SUPPORT
		case IFT_LML: return( get_lml_seconds(QSP_ARG  ifp,frame) ); break;
#endif /* HAVE_JPEG_SUPPORT */

#ifdef HAVE_RAWVOL
		case IFT_RV: return( get_rv_seconds(QSP_ARG  ifp,frame)); break;
#endif /* HAVE_RAWVOL */

		default:
			WARN("Timestamp functions are only supported for file types LML and RV");
			sprintf(ERROR_STRING,"(file %s is type %s)",ifp->if_name,
					FT_NAME(IF_TYPE(ifp)));
			advise(ERROR_STRING);
			return(-1);
	}
}

static double get_if_milliseconds(QSP_ARG_DECL  Item *ip,dimension_t frame)
{
	Image_File *ifp;

	ifp = (Image_File *) ip;

	switch( FT_CODE(IF_TYPE(ifp)) ){

#ifdef HAVE_JPEG_SUPPORT
		case IFT_LML: return( get_lml_milliseconds(QSP_ARG  ifp,frame) ); break;
#endif /* HAVE_JPEG_SUPPORT */

#ifdef HAVE_RAWVOL
		case IFT_RV: return( get_rv_milliseconds(QSP_ARG  ifp,frame) ); break;
#endif /* HAVE_RAWVOL */

		default:
			WARN("Timestamp functions are only supported for file types LML and RV");
			sprintf(ERROR_STRING,"(file %s is type %s)",ifp->if_name,
					FT_NAME(IF_TYPE(ifp)));
			advise(ERROR_STRING);
			return(-1);
	}
}

static double get_if_microseconds(QSP_ARG_DECL  Item *ip,dimension_t frame)
{
	Image_File *ifp;

	ifp = (Image_File *) ip;

	switch( FT_CODE(IF_TYPE(ifp)) ){
#ifdef HAVE_JPEG_SUPPORT
		case IFT_LML: return( get_lml_microseconds(QSP_ARG  ifp,frame) ); break;
#endif /* HAVE_JPEG_SUPPORT */

#ifdef HAVE_RAWVOL
		case IFT_RV: return( get_rv_microseconds(QSP_ARG  ifp,frame) ); break;
#endif /* HAVE_RAWVOL */

		default:
			WARN("Timestamp functions are only supported for file types LML and RV");
			sprintf(ERROR_STRING,"(file %s is type %s)",ifp->if_name,
					FT_NAME(IF_TYPE(ifp)) );
			advise(ERROR_STRING);
			return(-1);
	}
}

