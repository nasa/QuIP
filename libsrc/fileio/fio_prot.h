/* This file contains stuff which is private to the fileio module */

#ifndef _FIO_PROT_H_
#define _FIO_PROT_H_

#include "quip_config.h"

#include "img_file_hdr.h"
#include "fio_api.h"

#define FIO_OPEN_FUNC( funcname )				\
Image_File * funcname(QSP_ARG_DECL  const char *name,int rw)

#define FIO_CLOSE_FUNC( funcname )				\
void funcname(QSP_ARG_DECL  Image_File *ifp)

#define FIO_WT_FUNC( funcname )					\
int funcname(QSP_ARG_DECL  Data_Obj *dp, Image_File *ifp )

#define FIO_RD_FUNC( funcname )					\
void funcname(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp,index_t x_offset,index_t y_offset,index_t t_offset)

#define FIO_SETHDR_FUNC( funcname )				\
int funcname(QSP_ARG_DECL  Image_File *ifp)

#define FIO_SEEK_FUNC( funcname )				\
int funcname( QSP_ARG_DECL  Image_File *ifp, dimension_t n )

#define FIO_INFO_FUNC( funcname )			\
void funcname( QSP_ARG_DECL  Image_File *ifp )

#define FIO_INTERFACE_PROTOTYPES( stem , header_type )	\
							\
extern FIO_OPEN_FUNC( stem##_open );			\
extern FIO_CLOSE_FUNC( stem##_close );			\
extern FIO_SETHDR_FUNC( set_##stem##_hdr );		\
extern FIO_WT_FUNC( stem##_wt );			\
extern FIO_RD_FUNC( stem##_rd );			\
extern FIO_INFO_FUNC( stem##_info );			\
extern FIO_SEEK_FUNC( stem##_seek_frame );		\
extern int stem##_to_dp(Data_Obj *dp,header_type *hd_p);	\
extern int dp_to_##stem(header_type *hd_p,Data_Obj *dp);	\
extern int stem##_unconv(void *hd_pp ,Data_Obj *dp);	\
extern int stem##_conv(Data_Obj *dp, void *hd_pp);

/* globals */

extern Item_Type *if_itp;
extern prec_t raw_prec;

#ifdef DEBUG
extern debug_flag_t debug_fileio;
#endif /* DEBUG */


/* prototypes */


/* img_file.c */
extern int	std_seek(QSP_ARG_DECL  Image_File *, uint32_t );
extern int	uio_seek(QSP_ARG_DECL  Image_File *, uint32_t );
extern void	check_auto_close(QSP_ARG_DECL  Image_File *);
// moved to fio_api.h
//extern void	delete_image_file(QSP_ARG_DECL  Image_File *);
//extern void generic_imgfile_close(QSP_ARG_DECL  Image_File *ifp);
extern List *	image_file_list(SINGLE_QSP_ARG_DECL);
extern void	set_direct_io(int);
ITEM_INTERFACE_PROTOTYPES(Image_File,img_file)
#define DEL_IMG_FILE(s)	del_img_file(QSP_ARG  s)

extern void setup_dummy(Image_File *ifp);
extern int open_fd(Image_File *ifp);
#ifndef PC
extern int ifp_iopen(Image_File *ifp);
#endif /* ! PC */
extern int open_fp(Image_File *ifp);
extern void image_file_clobber(int);
extern void image_file_init(SINGLE_QSP_ARG_DECL);

extern Image_File *image_file_open(QSP_ARG_DECL  const char *,int rw,filetype_code type);
#define IMAGE_FILE_OPEN(fn,m,t)	image_file_open(QSP_ARG  fn,m,t)
extern int same_dimensions(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp);
extern int same_size(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp);
extern int same_type(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp);
extern void copy_dimensions(Data_Obj *dpto,Data_Obj *dpfr);
extern void dump_image_file(QSP_ARG_DECL  const char *filename,filetype_code filetype,void *data,
			dimension_t width,dimension_t height,prec_t prec);
extern void *load_image_file(const char *name,
	filetype_code input_file_type,filetype_code desired_hdr_type);

extern filetype_code get_filetype(void);
//extern Image_File *write_image_file(QSP_ARG_DECL  const char *filename,dimension_t n);
//extern void close_image_file(QSP_ARG_DECL  Image_File *ifp);
//extern Image_File * open_image_file(QSP_ARG_DECL  const char *filename, const char *rw);
//extern void write_image_to_file(QSP_ARG_DECL  Image_File *ifp,Data_Obj *dp);
//extern int image_file_seek(QSP_ARG_DECL  Image_File *ifp,dimension_t n);
extern double iof_exists(QSP_ARG_DECL  const char *);

/* lml.c ? */
extern double get_lml_seconds(Image_File *,dimension_t);
extern double get_lml_milliseconds(Image_File *,dimension_t);
extern double get_lml_microseconds(Image_File *,dimension_t);

/* rv.c */
extern int	rv_seek(Image_File *ifp, uint32_t n );
extern double	get_rv_seconds(Image_File *ifp,dimension_t);
extern double	get_rv_milliseconds(Image_File *ifp,dimension_t);
extern double	get_rv_microseconds(Image_File *ifp,dimension_t);

/* avi.c */

#include "verfio.h"

/* dsk.c */		/* not in sir_disk.h because need typedef Img_File */

FIO_INTERFACE_PROTOTYPES( dsk , Sir_Disk_Hdr )


/* fileport.c */
extern void xmit_file(QSP_ARG_DECL  Port *mpp,Image_File *ifp,int flag);
extern Image_File * recv_file(QSP_ARG_DECL  Port *mpp);

#ifdef HAVE_LIBAVCODEC
extern int check_avi_info(Image_File *);
extern void save_avi_info(Image_File *ifp);
#endif

#endif /* ! _FIO_PROT_H_ */
