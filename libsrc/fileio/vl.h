
#ifdef INC_VERSION
char VersionId_inc_vl[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#include "img_file.h"

/* prototypes from vl.c */

extern void vl_close(QSP_ARG_DECL  Image_File *ifp);
extern Image_File * vl_open(QSP_ARG_DECL  const char *name,int rw);
extern int vl_unconv(void *hdr_pp,Data_Obj *dp);
extern int vl_conv(Data_Obj *dp,void *hd_pp);


