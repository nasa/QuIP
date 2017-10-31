
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
extern void		hdr2_strs(Hips2_Header *hdp);
extern int		set_hips2_hdr(QSP_ARG_DECL  Image_File *ifp);

//extern int		hips2_to_dp(Data_Obj *dp,Hips2_Header *hd_p);
//extern int		dp_to_hips2(Hips2_Header *hd_p,Data_Obj *dp);
extern FIO_FT_TO_DP_FUNC(hips2,Hips2_Header);
extern FIO_DP_TO_FT_FUNC(hips2,Hips2_Header);
extern FIO_OPEN_FUNC( hips2 );
extern FIO_CLOSE_FUNC( hips2 );
extern FIO_WT_FUNC( hips2 );
extern FIO_RD_FUNC( hips2 );
extern FIO_CONV_FUNC( hips2 );
extern FIO_UNCONV_FUNC( hips2 );

#define hips2_to_dp(a,b)	_hips2_to_dp(QSP_ARG  a,b)
#define dp_to_hips2(a,b)	_dp_to_hips2(QSP_ARG  a,b)

/* writehdr.c */
extern int		wt_hips2_hdr(FILE *fp,Hips2_Header *hd,const Filename fname);

#endif /* NO_HIPS2 */

