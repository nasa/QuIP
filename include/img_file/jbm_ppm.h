

#ifndef PPM_DEFS

#include "img_file.h"

#ifdef INC_VERSION
char VersionId_inc_ppm[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#include "ppmhdr.h"


#define PPM_DEFS


FIO_INTERFACE_PROTOTYPES( ppm , Ppm_Header )

extern int _rd_ppm_hdr(QSP_ARG_DECL  FILE *fp,Ppm_Header *,const char *filename);
extern void _wt_ppm_hdr(QSP_ARG_DECL  FILE *fp,Ppm_Header *hdp,const char *filename);

#define rd_ppm_hdr(fp,hdp,filename) _rd_ppm_hdr(QSP_ARG  fp,hdp,filename)
#define wt_ppm_hdr(fp,hdp,filename) _wt_ppm_hdr(QSP_ARG  fp,hdp,filename)

#define ppm_to_dp(a,b)	_ppm_to_dp(QSP_ARG  a,b)
#define dp_to_ppm(a,b)	_dp_to_ppm(QSP_ARG  a,b)

FIO_INTERFACE_PROTOTYPES( dis, Dis_Header )

extern int _rd_dis_hdr(QSP_ARG_DECL  FILE *fp,Dis_Header *,const char *filename);
extern void _wt_dis_hdr(QSP_ARG_DECL  FILE *fp,Dis_Header *hdp,const char *filename);

#define rd_dis_hdr(fp,hdp,filename) _rd_dis_hdr(QSP_ARG fp,hdp,filename)
#define wt_dis_hdr(fp,hdp,filename) _wt_dis_hdr(QSP_ARG fp,hdp,filename)

#define dis_to_dp(a,b)	_dis_to_dp(QSP_ARG  a,b)
#define dp_to_dis(a,b)	_dp_to_dis(QSP_ARG  a,b)


#endif /* PPM_DEFS */

