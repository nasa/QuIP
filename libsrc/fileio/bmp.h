#ifndef NO_BMP
#define NO_BMP
 
#include <stdio.h>
//#include "hip2hdr.h"
#include "hips/hips2.h"
#include "data_obj.h"
#include "img_file.h"
 
/* bmp.c */
extern int		bmp_to_dp(Data_Obj *dp,BMP_Header *hd_p);
extern int		dp_to_bmp(Hips2_Header *hd_p,Data_Obj *dp);
extern int		set_bmp_hdr(Image_File *ifp);
extern void		bmp_info(QSP_ARG_DECL  Image_File *ifp);

 
#endif /* NO_HIPS2 */

