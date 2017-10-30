
#ifndef NO_HIPS1
#define NO_HIPS1

#include "data_obj.h"
#include "hipl_fmt.h"
#include "hip1hdr.h"
#include "img_file.h"

extern   void hdr1_strs(Hips1_Header *hdp);

//extern   int hips1_to_dp(Data_Obj *dp,Hips1_Header *hd_p);
//extern   int dp_to_hips1(Hips1_Header *hd_p,Data_Obj *dp);
extern FIO_FT_TO_DP_FUNC(hips1,Hips1_Header);
extern FIO_DP_TO_FT_FUNC(hips1,Hips1_Header);

#define hips1_to_dp(a,b)	_hips1_to_dp(QSP_ARG  a,b)
#define dp_to_hips1(a,b)	_dp_to_hips1(QSP_ARG  a,b)

extern   FIO_SETHDR_FUNC( set_hdr );
extern FIO_CLOSE_FUNC( hips1 );
extern FIO_OPEN_FUNC( hips1 );
extern FIO_WT_FUNC( hips1 );
extern FIO_CONV_FUNC( hips1 );
extern FIO_UNCONV_FUNC( hips1 );

#endif /* NO_HIPS1 */

