
#ifndef NO_XPARAM
#define NO_XPARAM

#include "hip2hdr.h"
#include "hips2.h"


#ifdef SUN
extern   int setparam(/* va_dcl */ );    
extern   int setparamd(/* va_dcl */ );
extern   int getparam(/* va_dcl */ );
#else /* ! SUN */
extern   int setparam(Hips2_Header *hd, ...);    
extern   int setparamd(Hips2_Header *hd, ...);
extern   int getparam(Hips2_Header *hd, ...);
#endif /* ! SUN */

extern   int clearparam(Hips2_Header *hd,char *name);
//extern   struct extpar *findparam(Hips2_Header *hd,char *name);
extern   int checkname(char *name);
extern   int mergeparam(Hips2_Header *hd1,Hips2_Header *hd2);

#endif /* NO_XPARAM */
