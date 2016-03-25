
#ifdef HAVE_RASTERFILE_H
#include <rasterfile.h>
#else
#include "rasterf.h"
#endif

#include "data_obj.h"
#include "img_file.h"

extern void sunras_close(QSP_ARG_DECL  Image_File *ifp);
extern void sunras_rd(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp,
	index_t x_offset, index_t y_offset, index_t t_offset);
extern Image_File *sunras_open(QSP_ARG_DECL  const char *name,int rw);
extern int sunras_unconv(void *hd_pp,Data_Obj *dp);
extern int sunras_conv(Data_Obj *dp,void *hd_pp);

