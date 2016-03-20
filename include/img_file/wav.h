
/* we need something here... */

#ifndef NO_WAV
#define NO_WAV
 
#include <stdio.h>
#include "wav_hdr.h"
#include "data_obj.h"
#include "img_file.h"
 
/* wav.c */
FIO_INTERFACE_PROTOTYPES( wav, Wav_Header )

/* writehdr.c */
extern int		wt_wav_hdr(FILE *fp,Wav_Header *hd,Filename fname);

#endif /* NO_WAV */

