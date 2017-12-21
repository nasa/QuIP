
#ifndef _IMG_FILE_H_
#define _IMG_FILE_H_

#include "quip_config.h"

#include "data_obj.h"

#include "nports_api.h"

#ifdef FOOBAR
#ifdef HAVE_TIFF
#include <tiffio.h>
#endif /* HAVE_TIFF */
#endif // FOOBAR

// We used to have some of these ifdef'd, but
// maybe it's better for the values to be constant?

typedef enum {
	IFT_NETWORK,		/* 0 */
	IFT_RAW,		/* 1 */
	IFT_HIPS1,		/* 2 */
	IFT_HIPS2,		/* 3 */
	IFT_SUNRAS,		/* 4 */
	IFT_PPM,		/* 5 */
	IFT_DIS,		/* 6 */
	IFT_VISTA,		/* 7 */
	IFT_VL,			/* 8 */
	IFT_RGB,		/* 9 */
	IFT_DISK,		/* 10 */
	IFT_RV,			/* 11 */
	IFT_LUM,		/* 12 */
	IFT_WAV,		/* 13 */
	IFT_BMP,		/* 14 */
	IFT_ASC,		/* 15 */
	IFT_BDF,		/* 16 */
	IFT_AVI,		/* 17 */
	IFT_MATLAB,		/* 18 */
	IFT_LML,		/* 19 */
	IFT_JPEG,		/* 20 */
	IFT_MPEG,		/* 21 */
	IFT_TIFF,		/* 22 */
	IFT_PNG,		/* 23 */
	IFT_VIFF,		/* 24 */
	IFT_QT,			/* 25 */
	N_FILETYPE
} filetype_code;

struct image_file;

typedef struct filetype {
	Item		ft_item;

	struct image_file *	(*op_func)(QSP_ARG_DECL  const char *,int rw);
	void		(*rd_func)(QSP_ARG_DECL  Data_Obj *,struct image_file *,
				index_t,index_t,index_t);
	int		(*wt_func)(QSP_ARG_DECL  Data_Obj *,struct image_file *);
	void		(*close_func)(QSP_ARG_DECL  struct image_file *);
	int		(*unconv_func)(QSP_ARG_DECL  void *,Data_Obj *);
					/* from dp to whatever */
	int		(*conv_func)(QSP_ARG_DECL  Data_Obj *, void *);
					/* from whatever to dp */
	void		(*info_func)(QSP_ARG_DECL  struct image_file *);
	int		(*seek_func)(QSP_ARG_DECL  struct image_file *,dimension_t);	/* might need to be 64 bit... */
	short		ft_flags;
	filetype_code	ft_code;
} Filetype;


/* Filetype */
#define FT_NAME(ftp)		(ftp)->ft_item.item_name
#define FT_CODE(ftp)		(ftp)->ft_code
#define FT_FLAGS(ftp)		(ftp)->ft_flags
#define FT_OPEN_FUNC(ftp)	(ftp)->op_func
#define FT_READ_FUNC(ftp)	(ftp)->rd_func
#define FT_WRITE_FUNC(ftp)	(ftp)->wt_func
#define FT_CLOSE_FUNC(ftp)	(ftp)->close_func
#define FT_CONV_FUNC(ftp)	(ftp)->conv_func
#define FT_UNCONV_FUNC(ftp)	(ftp)->unconv_func
#define FT_SEEK_FUNC(ftp)	(ftp)->seek_func
#define FT_INFO_FUNC(ftp)	(ftp)->info_func

#define SET_FT_CODE(ftp,c)		(ftp)->ft_code = c
#define SET_FT_FLAGS(ftp,v)		(ftp)->ft_flags = v
#define SET_FT_OPEN_FUNC(ftp,f)		(ftp)->op_func = f
#define SET_FT_READ_FUNC(ftp,f)		(ftp)->rd_func = f
#define SET_FT_WRITE_FUNC(ftp,f)	(ftp)->wt_func = f
#define SET_FT_CLOSE_FUNC(ftp,f)	(ftp)->close_func = f
#define SET_FT_CONV_FUNC(ftp,f)		(ftp)->conv_func = f
#define SET_FT_UNCONV_FUNC(ftp,f)	(ftp)->unconv_func = f
#define SET_FT_SEEK_FUNC(ftp,f)		(ftp)->seek_func = f
#define SET_FT_INFO_FUNC(ftp,f)		(ftp)->info_func = f


ITEM_INIT_PROT(Filetype,file_type)
ITEM_NEW_PROT(Filetype,file_type)
ITEM_CHECK_PROT(Filetype,file_type)
ITEM_PICK_PROT(Filetype,file_type)
ITEM_ENUM_PROT(Filetype,file_type)
ITEM_LIST_PROT(Filetype,file_type)	// added for debugging

#define init_file_types()	_init_file_types(SINGLE_QSP_ARG)
#define new_file_type(s)	_new_file_type(QSP_ARG  s)
#define file_type_of(s)		_file_type_of(QSP_ARG  s)
#define pick_file_type(s)	_pick_file_type(QSP_ARG  s)
#define file_type_list()	_file_type_list(SINGLE_QSP_ARG)
#define list_file_types(fp)	_list_file_types(QSP_ARG  fp)

struct image_file {
	Item		if_item;
#define if_name		if_item.item_name
	union {
		FILE *	u_fp;		/* file pointer */
		int	u_fd;		/* file descriptor */
		void *	u_tiff;
	} if_file_u;
	dimension_t	if_nfrms;	/* frames read/written	*/
	dimension_t	if_frms_to_wt;	/* used to keep this in the header... */
	Filetype *	if_ftp;
	void *		if_hdr_p;	/* pointer to header */
					/* an Img_File_Hdr */
	short		if_flags;
	Data_Obj *	if_dp;		/* to remember width, etc */
	const char *	if_pathname;	/* full pathname */
// BUG see png.c - we need to have an iOS version of this struct...
//
//#ifdef BUILD_FOR_IOS
//	UIImage *	if_imgp;	// for iOS png...
//#endif // BUILD_FOR_IOS
} ;


/* Image_File */
#define IF_NAME(ifp)		(ifp)->if_name
#define IF_TYPE(ifp)		(ifp)->if_ftp
#define SET_IF_TYPE(ifp,ftp)	(ifp)->if_ftp = ftp
#define IF_TYPE_CODE(ifp)	FT_CODE( IF_TYPE(ifp) )

#define if_fp	if_file_u.u_fp
#define if_fd	if_file_u.u_fd
#ifdef HAVE_TIFF
#define if_tiff	if_file_u.u_tiff
#endif /* HAVE_TIFF */

#ifdef FOOBAR
// Moved to fio_api.h
ITEM_INIT_PROT(Image_File,img_file)
ITEM_NEW_PROT(Image_File,img_file)
ITEM_CHECK_PROT(Image_File,img_file)
ITEM_PICK_PROT(Image_File,img_file)

#define init_img_files(s)	_init_img_files(SINGLE_QSP_ARG)
#define new_img_file(s)		_new_img_file(QSP_ARG  s)
#define img_file_of(s)		_img_file_of(QSP_ARG  s)
#define pick_img_file(pmpt)	_pick_img_file(QSP_ARG  pmpt)
#endif // FOOBAR

/* flag values for ifp's & filetype's */

#define FILE_READ	1
#define FILE_WRITE	2
#define USE_STDIO	4
#define FILE_ERROR	8
#define USE_UNIX_IO	16

#define FILE_TALL	16
#define FILE_SHORT	32
#define FILE_THICK	64
#define FILE_THIN	128

#define NO_AUTO_CLOSE	256

#define IS_WRITABLE(ifp)	(ifp->if_flags & FILE_WRITE)
#define IS_READABLE(ifp)	(ifp->if_flags & FILE_READ)

#define CAN_READ_FORMAT		FILE_READ
#define CAN_WRITE_FORMAT	FILE_WRITE
#define CAN_DO_FORMAT		(CAN_READ_FORMAT|CAN_WRITE_FORMAT)

#define USES_STDIO(ifp)		(FT_FLAGS(IF_TYPE(ifp)) & USE_STDIO)
#define USES_UNIX_IO(ifp)	(FT_FLAGS(IF_TYPE(ifp)) & USE_UNIX_IO)

#define CANNOT_READ(ftp)	((FT_FLAGS(ftp)&CAN_READ_FORMAT)==0)
#define CANNOT_WRITE(ftp)	((FT_FLAGS(ftp)&CAN_WRITE_FORMAT)==0)

#define HAD_ERROR(ifp)		(ifp->if_flags & FILE_ERROR)
#define SET_ERROR(ifp)		ifp->if_flags |= FILE_ERROR

#define WANT_FRAMES(ifp)	(OBJ_FRAMES(ifp->if_dp)*OBJ_SEQS(ifp->if_dp))
#define FILE_FINISHED(ifp)	( ifp->if_nfrms == WANT_FRAMES(ifp) && (ifp->if_flags&NO_AUTO_CLOSE)==0 )




#endif /* ! _IMG_FILE_H_ */

