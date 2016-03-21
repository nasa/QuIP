

#ifndef PPM_DEFS

#include "img_file.h"

#ifdef INC_VERSION
char VersionId_inc_ppm[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#include "ppmhdr.h"


#define PPM_DEFS


FIO_INTERFACE_PROTOTYPES( ppm , Ppm_Header )
extern int rd_ppm_hdr(FILE *fp,Ppm_Header *,const char *filename);
extern void wt_ppm_hdr(FILE *fp,Ppm_Header *hdp,const char *filename);

FIO_INTERFACE_PROTOTYPES( dis, Dis_Header )
extern int rd_dis_hdr(FILE *fp,Dis_Header *,const char *filename);
extern void wt_dis_hdr(FILE *fp,Dis_Header *hdp,const char *filename);



#endif /* PPM_DEFS */

