
#ifdef INC_VERSION
char VersionId_inc_rv[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#include "img_file.h"

FIO_INTERFACE_PROTOTYPES( rvfio, RV_Inode )

#define set_rvfio_hdr(ifp)	_set_rvfio_hdr(QSP_ARG  ifp)

