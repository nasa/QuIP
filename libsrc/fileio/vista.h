
#ifndef _VISTA_H_
#define _VISTA_H_

#include "vistahdr.h"

FIO_INTERFACE_PROTOTYPES( vista , Vista_Hdr )

extern void		bswap(short *sp);
extern void		swap_hdr(Vista_Hdr *hd_p);

#endif /* _VISTA_H_ */

