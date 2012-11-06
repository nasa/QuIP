
#ifndef _IMG_FILE_H_
#define _IMG_FILE_H_

#include "quip_config.h"

#include "query.h"

//#include "rawvol.h"
#include "data_obj.h"

#include "nports_api.h"

#ifdef HAVE_TIFF
#include <tiffio.h>
#endif /* HAVE_TIFF */


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
#ifdef HAVE_RGB
	IFT_RGB,		/* 9 */
#endif /* HAVE_RGB */
	IFT_DISK,		/* 10 */
	IFT_RV,			/* 11 */
	IFT_LUM,		/* 12 */
	IFT_WAV,		/* 13 */
	IFT_BMP,		/* 14 */
	IFT_ASC,		/* 15 */
	IFT_BDF,		/* 16 */
#ifdef HAVE_LIBAVCODEC
	IFT_AVI,		/* 17 */
#endif /* HAVE_LIBAVCODEC */

#ifdef HAVE_MATIO
	IFT_MATLAB,		/* 18 */
#endif /* HAVE_MATIO */

#ifdef HAVE_JPEG_SUPPORT
	IFT_LML,		/* 19 */
	IFT_JPEG,		/* 20 */
#endif /* HAVE_JPEG_SUPPORT */

#ifdef HAVE_MPEG
	IFT_MPEG,		/* 21 */
#endif /* HAVE_MPEG */

#ifdef HAVE_TIFF
	IFT_TIFF,		/* 22 */
#endif /* HAVE_TIFF */

#ifdef HAVE_PNG
	IFT_PNG,		/* 23 */
#endif /* HAVE_PNG */

#ifdef HAVE_KHOROS
	IFT_VIFF,		/* 24 */
#endif /* HAVE_KHOROS */

#ifdef HAVE_QUICKTIME
	IFT_QT,			/* 25 */
#endif /* HAVE_QUICKTIME */

	N_FILETYPE
} filetype_code;



typedef struct image_file {
	Item		if_item;
#define if_name		if_item.item_name
	union {
		FILE *	u_fp;		/* file pointer */
		int	u_fd;		/* file descriptor */
#ifdef HAVE_TIFF
		TIFF *	u_tiff;
#endif /* HAVE_TIFF */
	} if_file_u;
	dimension_t	if_nfrms;	/* frames read/written	*/
	dimension_t	if_frms_to_wt;	/* used to keep this in the header... */
	filetype_code	if_type;
	void *		if_hd;		/* pointer to header */
	short		if_flags;
	Data_Obj *	if_dp;		/* to remember width, etc */
	const char *	if_pathname;	/* full pathname */
} Image_File;

#define if_fp	if_file_u.u_fp
#define if_fd	if_file_u.u_fd
#ifdef HAVE_TIFF
#define if_tiff	if_file_u.u_tiff
#endif /* HAVE_TIFF */

#define NO_IMAGE_FILE		((Image_File*) NULL )

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

#define USES_STDIO(ifp)		(ft_tbl[ifp->if_type].ft_flags & USE_STDIO)
#define USES_UNIX_IO(ifp)	(ft_tbl[ifp->if_type].ft_flags & USE_UNIX_IO)

#define CANNOT_READ(i)		((ft_tbl[i].ft_flags&CAN_READ_FORMAT)==0)
#define CANNOT_WRITE(i)		((ft_tbl[i].ft_flags&CAN_WRITE_FORMAT)==0)

#define HAD_ERROR(ifp)		(ifp->if_flags & FILE_ERROR)
#define SET_ERROR(ifp)		ifp->if_flags |= FILE_ERROR

#define WANT_FRAMES(ifp)	(ifp->if_dp->dt_frames*ifp->if_dp->dt_seqs)
#define FILE_FINISHED(ifp)	( ifp->if_nfrms == WANT_FRAMES(ifp) && (ifp->if_flags&NO_AUTO_CLOSE)==0 )



/* Public prototypes */


/* img_file.c */
extern void	set_iofile_directory(QSP_ARG_DECL  const char *);
extern		Image_File *read_image_file(QSP_ARG_DECL  const char *name);
extern void	read_object_from_file(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp);


#endif /* ! _IMG_FILE_H_ */

