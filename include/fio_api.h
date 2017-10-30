
#ifndef _FIO_API_H_
#define _FIO_API_H_


#include "quip_config.h"
#include "img_file.h"

#define FIO_OPEN_FUNC_NAME(stem)	stem##_open
#define FIO_OPEN_FUNC( stem )				\
Image_File * FIO_OPEN_FUNC_NAME(stem)(QSP_ARG_DECL  const char *name,int rw)

#define FIO_CLOSE_FUNC_NAME( stem )	stem##_close
#define FIO_CLOSE_FUNC( stem )				\
void FIO_CLOSE_FUNC_NAME(stem)(QSP_ARG_DECL  Image_File *ifp)

#define FIO_WT_FUNC_NAME( stem )	stem##_wt
#define FIO_WT_FUNC( stem )					\
int FIO_WT_FUNC_NAME(stem)(QSP_ARG_DECL  Data_Obj *dp, Image_File *ifp )

#define FIO_RD_FUNC_NAME( stem )	stem##_rd
#define FIO_RD_FUNC( stem )					\
void FIO_RD_FUNC_NAME(stem)(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp,index_t x_offset,index_t y_offset,index_t t_offset)

#define FIO_SETHDR_FUNC_NAME( stem )	set_##stem##_hdr
#define FIO_SETHDR_FUNC( stem )				\
int FIO_SETHDR_FUNC_NAME(stem)(QSP_ARG_DECL  Image_File *ifp)

#define FIO_SEEK_FUNC_NAME( stem )	stem##_seek_frame
#define FIO_SEEK_FUNC( stem )				\
int FIO_SEEK_FUNC_NAME(stem)( QSP_ARG_DECL  Image_File *ifp, dimension_t n )

#define FIO_INFO_FUNC_NAME( stem )	stem##_info_func
#define FIO_INFO_FUNC( stem )			\
void FIO_INFO_FUNC_NAME(stem)( QSP_ARG_DECL  Image_File *ifp )

#define FIO_FT_TO_DP_FUNC_NAME(stem)		_##stem##_to_dp
#define FIO_FT_TO_DP_FUNC(stem,header_type)					\
int FIO_FT_TO_DP_FUNC_NAME(stem)(QSP_ARG_DECL  Data_Obj *dp,header_type *hd_p)

#define FIO_DP_TO_FT_FUNC_NAME(stem)		_dp_to_##stem
#define FIO_DP_TO_FT_FUNC(stem,header_type)					\
int FIO_DP_TO_FT_FUNC_NAME(stem)(QSP_ARG_DECL  header_type *hd_p,Data_Obj *dp)

#define FIO_UNCONV_FUNC_NAME(stem)		_##stem##_unconv
#define FIO_UNCONV_FUNC(stem)						\
int FIO_UNCONV_FUNC_NAME(stem)(QSP_ARG_DECL  void *hd_pp ,Data_Obj *dp)

#define FIO_CONV_FUNC_NAME(stem)		_##stem##_conv
#define FIO_CONV_FUNC(stem)						\
int FIO_CONV_FUNC_NAME(stem)(QSP_ARG_DECL  Data_Obj *dp, void *hd_pp)

#define FIO_INTERFACE_PROTOTYPES( stem , header_type )			\
									\
extern FIO_OPEN_FUNC( stem );						\
extern FIO_CLOSE_FUNC( stem );						\
extern FIO_WT_FUNC( stem );						\
extern FIO_RD_FUNC( stem );						\
extern FIO_SETHDR_FUNC( stem );						\
extern FIO_INFO_FUNC( stem );						\
extern FIO_SEEK_FUNC( stem );						\
extern FIO_FT_TO_DP_FUNC(stem,header_type);				\
extern FIO_DP_TO_FT_FUNC(stem,header_type);				\
extern FIO_UNCONV_FUNC(stem);						\
extern FIO_CONV_FUNC(stem);

/* Public prototypes */

/* img_file.c */
extern Filetype *	current_filetype(void);
extern void close_image_file(QSP_ARG_DECL  Image_File *ifp);
extern void generic_imgfile_close(QSP_ARG_DECL  Image_File *ifp);
#define GENERIC_IMGFILE_CLOSE(ifp)	generic_imgfile_close(QSP_ARG  ifp)
extern void if_info(QSP_ARG_DECL  Image_File *ifp);
extern Image_File *write_image_file(QSP_ARG_DECL  const char *filename,dimension_t n);
extern void write_image_to_file(QSP_ARG_DECL  Image_File *ifp,Data_Obj *dp);
extern int image_file_seek(QSP_ARG_DECL  Image_File *ifp,dimension_t n);
extern void image_file_clobber(int);
extern void	delete_image_file(QSP_ARG_DECL  Image_File *);
extern Image_File * open_image_file(QSP_ARG_DECL  const char *filename, const char *rw);

/* img_file.c */
extern void		set_iofile_directory(QSP_ARG_DECL  const char *);
extern	Image_File *	read_image_file(QSP_ARG_DECL  const char *name);
extern void		read_object_from_file(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp);
extern void		set_filetype(QSP_ARG_DECL  Filetype *ftp);


ITEM_INTERFACE_PROTOTYPES(Image_File,img_file)

#define init_img_files()	_init_img_files(SINGLE_QSP_ARG)
#define img_file_of(s)		_img_file_of(QSP_ARG  s)
#define new_img_file(s)		_new_img_file(QSP_ARG  s)
#define list_img_files(fp)	_list_img_files(QSP_ARG  fp)
#define pick_img_file(s)	_pick_img_file(QSP_ARG  s)
#define del_img_file(s)		_del_img_file(QSP_ARG  s)

extern Filetype *filetype_for_code(QSP_ARG_DECL  filetype_code code);
#define FILETYPE_FOR_CODE(code)	filetype_for_code(QSP_ARG  code)

#ifdef HAVE_JPEG_SUPPORT
extern COMMAND_FUNC( do_jpeg_menu );
#endif /* HAVE_JPEG_SUPPORT */

#ifdef HAVE_PNG 
extern COMMAND_FUNC( do_png_menu );
#endif /* HAVE_PNG */

#endif /* ! _FIO_API_H_ */

