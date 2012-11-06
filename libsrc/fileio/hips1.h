
#ifndef NO_HIPS1
#define NO_HIPS1

#include "data_obj.h"
#include "hipl_fmt.h"
#include "hip1hdr.h"
#include "img_file.h"

extern   void hips1_close(QSP_ARG_DECL  Image_File *ifp);
extern   int hips1_to_dp(Data_Obj *dp,Hips1_Header *hd_p);
extern   void hdr1_strs(Hips1_Header *hdp);
extern   FIO_OPEN_FUNC( hips1_open );
extern   int dp_to_hips1(Hips1_Header *hd_p,Data_Obj *dp);
extern   FIO_SETHDR_FUNC( set_hdr );
extern	FIO_WT_FUNC(hips1_wt);
//extern   int hips1_wt(Data_Obj *dp,Image_File *ifp);
extern   int hips1_unconv(void *hd_pp ,Data_Obj *dp);
extern   int hips1_conv(Data_Obj *dp, void *hd_pp);

#endif /* NO_HIPS1 */

