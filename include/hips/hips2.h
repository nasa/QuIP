
#ifndef NO_HIPS2
#define NO_HIPS2
 
#include <stdio.h>
#include "hipbasic.h"
#include "hiperror.h"
#include "hip2hdr.h"
#include "data_obj.h"
#include "img_file.h"
 
/* which file ??? */
extern struct extpar *findparam(Hips2_Header *hd,char *name);

/* hips2.c */
extern int		hips2_to_dp(Data_Obj *dp,Hips2_Header *hd_p);
extern void		hdr2_strs(Hips2_Header *hdp);
//extern FIO_OPEN_FUNC( hips2_open );
//extern FIO_CLOSE_FUNC( hips2_close );
extern FIO_OPEN_FUNC( hips2 );
extern FIO_CLOSE_FUNC( hips2 );
extern int		dp_to_hips2(Hips2_Header *hd_p,Data_Obj *dp);
extern int		set_hips2_hdr(QSP_ARG_DECL  Image_File *ifp);
//extern FIO_WT_FUNC( hips2_wt );
//extern FIO_RD_FUNC( hips2_rd );
extern FIO_WT_FUNC( hips2 );
extern FIO_RD_FUNC( hips2 );
//extern int		hips2_unconv(void *hd_pp ,Data_Obj *dp);
//extern int		hips2_conv(Data_Obj *dp, void *hd_pp);
extern FIO_CONV_FUNC( hips2 );
extern FIO_UNCONV_FUNC( hips2 );

/* writehdr.c */
extern int		wt_hips2_hdr(FILE *fp,Hips2_Header *hd,const Filename fname);

#endif /* NO_HIPS2 */

