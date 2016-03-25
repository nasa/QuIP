/* This file contains stuff which is private to the fileio module */

#ifndef _FIO_PROT_H_
#define _FIO_PROT_H_

#include "quip_config.h"


#include "img_file/img_file_hdr.h"
#include "fio_api.h"
#include "fiojpeg.h"
#include "hips/hipl_fmt.h"


/* globals */

extern Item_Type *if_itp;

#ifdef QUIP_DEBUG
extern debug_flag_t debug_fileio;
#endif /* QUIP_DEBUG */


/* prototypes */


/* img_file.c */
extern int	std_seek_frame(QSP_ARG_DECL  Image_File *, uint32_t );
extern int	uio_seek_frame(QSP_ARG_DECL  Image_File *, uint32_t );
extern void	check_auto_close(QSP_ARG_DECL  Image_File *);
// moved to fio_api.h
//extern void	delete_image_file(QSP_ARG_DECL  Image_File *);
//extern void generic_imgfile_close(QSP_ARG_DECL  Image_File *ifp);
extern List *	image_file_list(SINGLE_QSP_ARG_DECL);
extern void	set_direct_io(int);
ITEM_INTERFACE_PROTOTYPES(Image_File,img_file)
#define DEL_IMG_FILE(s)	del_img_file(QSP_ARG  s)

extern void setup_dummy(Image_File *ifp);
extern int open_fd(QSP_ARG_DECL  Image_File *ifp);
#ifndef PC
extern int ifp_iopen(Image_File *ifp);
#endif /* ! PC */
extern int open_fp(Image_File *ifp);
extern void image_file_clobber(int);
extern void image_file_init(SINGLE_QSP_ARG_DECL);

extern Image_File *img_file_creat(QSP_ARG_DECL  const char *,int rw,Filetype * ftp);
#define IMG_FILE_CREAT(fn,m,t)	img_file_creat(QSP_ARG  fn,m,t)
extern int same_dimensions(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp);
extern int same_size(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp);
extern int same_type(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp);
extern void copy_dimensions(Data_Obj *dpto,Data_Obj *dpfr);
extern void dump_image_file(QSP_ARG_DECL  const char *filename,Filetype *ftp,void *data,
			dimension_t width,dimension_t height,Precision * prec_p);
extern void *load_image_file(const char *name,
	filetype_code input_file_type,filetype_code desired_hdr_type);

//extern Image_File *write_image_file(QSP_ARG_DECL  const char *filename,dimension_t n);
//extern void close_image_file(QSP_ARG_DECL  Image_File *ifp);
//extern Image_File * open_image_file(QSP_ARG_DECL  const char *filename, const char *rw);
//extern void write_image_to_file(QSP_ARG_DECL  Image_File *ifp,Data_Obj *dp);
//extern int image_file_seek(QSP_ARG_DECL  Image_File *ifp,dimension_t n);
extern double iof_exists(QSP_ARG_DECL  const char *);

/* lml.c ? */
extern double get_lml_seconds(QSP_ARG_DECL  Image_File *,dimension_t);
extern double get_lml_milliseconds(QSP_ARG_DECL  Image_File *,dimension_t);
extern double get_lml_microseconds(QSP_ARG_DECL  Image_File *,dimension_t);

#ifdef HAVE_RAWVOL
/* rv.c */
extern int	rv_seek_frame(Image_File *ifp, uint32_t n );
extern double	get_rv_seconds(QSP_ARG_DECL  Image_File *ifp,dimension_t);
extern double	get_rv_milliseconds(QSP_ARG_DECL  Image_File *ifp,dimension_t);
extern double	get_rv_microseconds(QSP_ARG_DECL  Image_File *ifp,dimension_t);
#endif /* HAVE_RAWVOL */

/* avi.c */

//#include "verfio.h"

/* dsk.c */		/* not in sir_disk.h because need typedef Img_File */

FIO_INTERFACE_PROTOTYPES( dsk , Sir_Disk_Hdr )

#ifdef HAVE_PNG
FIO_INTERFACE_PROTOTYPES( png , Png_Hdr )
#endif /* HAVE_PNG */

//#ifdef HAVE_TIFF
//FIO_INTERFACE_PROTOTYPES( tiff , Tiff_Hdr )
//#endif /* HAVE_TIFF */


/* fileport.c */
extern void xmit_img_file(QSP_ARG_DECL  Port *mpp,Image_File *ifp,int flag);
extern long recv_img_file(QSP_ARG_DECL  Port *mpp, Packet *pkp);

#ifdef HAVE_AVI_SUPPORT
extern int check_avi_info(Image_File *);
extern void save_avi_info(Image_File *ifp);
#endif /* HAVE_AVI_SUPPORT */

// read_raw.c
extern void read_object(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp);
extern FIO_OPEN_FUNC( raw );
extern FIO_RD_FUNC( raw );
extern FIO_WT_FUNC( raw );
extern FIO_CONV_FUNC(raw);	// not really implemented
extern FIO_UNCONV_FUNC(raw);	// not implemented

// rv.c
extern FIO_OPEN_FUNC( rvfio );
extern FIO_RD_FUNC( rvfio );
extern FIO_WT_FUNC( rvfio );
extern FIO_CONV_FUNC( rvfio );	// not really implemented
extern FIO_UNCONV_FUNC( rvfio );	// not implemented
extern FIO_CLOSE_FUNC( rvfio );
extern FIO_SEEK_FUNC( rvfio );
extern FIO_INFO_FUNC( rvfio );

// ppm.c
extern FIO_OPEN_FUNC( ppm );
extern FIO_RD_FUNC( ppm );
extern FIO_WT_FUNC( ppm );
extern FIO_CONV_FUNC( ppm );	// not really implemented
extern FIO_UNCONV_FUNC( ppm );	// not implemented
extern FIO_CLOSE_FUNC( ppm );
extern FIO_SEEK_FUNC( ppm );
extern FIO_INFO_FUNC( ppm );

// fioasc.c
extern FIO_OPEN_FUNC( ascii );
extern FIO_RD_FUNC( ascii );
extern FIO_WT_FUNC( ascii );
extern FIO_CLOSE_FUNC( ascii );
extern FIO_CONV_FUNC( ascii );	// not really implemented
extern FIO_UNCONV_FUNC( ascii );	// not implemented
extern FIO_SEEK_FUNC( ascii );
extern FIO_INFO_FUNC( ascii );


// raw.c
extern void set_raw_sizes( dimension_t arr[N_DIMENSIONS] );
extern void set_raw_prec(Precision * prec_p);

// hips1.c
extern FIO_OPEN_FUNC( hips1 );
extern FIO_WT_FUNC( hips1 );
extern FIO_CLOSE_FUNC( hips1 );
extern FIO_CONV_FUNC(hips1);	// not implemented
extern FIO_UNCONV_FUNC(hips1);

// hips2.c
extern FIO_OPEN_FUNC(hips2);
extern FIO_RD_FUNC( hips2 );
extern FIO_WT_FUNC( hips2 );
extern FIO_CLOSE_FUNC( hips2 );
extern FIO_CONV_FUNC(hips2);	// not implemented
extern FIO_UNCONV_FUNC(hips2);

// bmp.c
extern FIO_OPEN_FUNC(bmp);
extern FIO_CLOSE_FUNC(bmp);
extern FIO_CONV_FUNC(bmp);
extern FIO_UNCONV_FUNC(bmp);
extern FIO_RD_FUNC(bmp);
extern FIO_INFO_FUNC(bmp);

#ifdef HAVE_MATIO
extern FIO_OPEN_FUNC(mat);
extern FIO_CLOSE_FUNC(mat);
extern FIO_CONV_FUNC(mat);
extern FIO_UNCONV_FUNC(mat);
extern FIO_RD_FUNC(mat);
extern FIO_WT_FUNC(mat);
extern FIO_INFO_FUNC(mat);
extern FIO_SEEK_FUNC( mat );
#endif // HAVE_MATIO

// get_hdr.c
extern int ftch_header(QSP_ARG_DECL  int fd,Header *hd);
#endif /* ! _FIO_PROT_H_ */
