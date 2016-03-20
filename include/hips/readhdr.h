
#ifndef NO_READHDR
#define NO_READHDR

#include <stdio.h>
#include "hipbasic.h"
#include "img_file.h"
#include "hip2hdr.h"

extern int rd_hips2_hdr(FILE *fp,Hips2_Header *hd,const Filename fname);
extern void rls_hips2_hd(Hips2_Header *hd);
extern void null_hips2_hd(Hips2_Header *hd);

#endif /* NO_READHDR */
