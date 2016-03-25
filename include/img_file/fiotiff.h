
#include "quip_config.h"

#ifdef HAVE_TIFF


#ifdef INC_VERSION
char VersionId_inc_fiotiff[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#include <tiffio.h>

#include "img_file_hdr.h"

/* tiff.c */
FIO_INTERFACE_PROTOTYPES( tiff , TIFF )

/* writehdr.c */
extern int		wt_tiff_hdr(FILE *fp,Hips2_Header *hd,Filename fname);

#endif /* HAVE_TIFF */

