
#ifndef NO_RDOLDHDR
#define NO_RDOLDHDR

#include <stdio.h>
#include "hip2hdr.h"

extern   int fread_oldhdr(FILE *fp,Hips2_Header *hd,char *firsts,const char *fname);
extern Bool swallownl(FILE *);
extern Bool hfgets(char *s,int n,FILE *fp);

#endif /* NO_RDOLDHDR */

