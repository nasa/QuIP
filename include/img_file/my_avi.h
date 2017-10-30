
#include "img_file.h"

/* prototypes */

FIO_INTERFACE_PROTOTYPES( avi , AVCodec_Hdr )

#define avi_conv(a,b)	_avi_conv(QSP_ARG  a,b)
#define avi_unconv(a,b)	_avi_unconv(QSP_ARG  a,b)
#define avi_to_dp(a,b)	_avi_to_dp(QSP_ARG  a,b)
#define dp_to_avi(a,b)	_dp_to_avi(QSP_ARG  a,b)

