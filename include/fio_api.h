

#ifndef _FIO_API_H_
#define _FIO_API_H_


#include "quip_config.h"
#include "img_file.h"

/* These are NOT items... (why not?) */
typedef struct filetype {
	const char *	ft_name;

	Image_File *	(*op_func)(QSP_ARG_DECL  const char *,int rw);
	void		(*rd_func)(QSP_ARG_DECL  Data_Obj *,Image_File *,
				index_t,index_t,index_t);
	int		(*wt_func)(QSP_ARG_DECL  Data_Obj *,Image_File *);
	void		(*close_func)(QSP_ARG_DECL  Image_File *);
	int		(*unconv_func)(void *,Data_Obj *);
					/* from dp to whatever */
	int		(*conv_func)(Data_Obj *, void *);
					/* from whatever to dp */
	void		(*info_func)(QSP_ARG_DECL  Image_File *);
	int		(*seek_func)(QSP_ARG_DECL  Image_File *,dimension_t);	/* might need to be 64 bit... */
	short		ft_flags;
	filetype_code	ft_type;
} Filetype;


/* Public prototypes */

/* img_file.c */
extern void set_filetype(QSP_ARG_DECL  filetype_code n);
extern filetype_code get_filetype(void);
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
ITEM_INTERFACE_PROTOTYPES(Image_File,img_file)


/* Public data structures */

extern Filetype ft_tbl[];

#endif /* ! _FIO_API_H_ */
