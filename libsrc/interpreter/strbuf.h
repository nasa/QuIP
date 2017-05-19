

#ifndef _STRBUF_H_
#define _STRBUF_H_

#include "quip_fwd.h"
//#include "typedefs.h"

/* string buffer structure */

struct string_buf {
	char *		sb_buf;
	size_t		sb_size;
} ;


/* String_Buf */
/* For now this is not an object... */
//#define SB_BUF(sbp)		sbp->sb_buf
#define SB_SIZE(sbp)		sbp->sb_size
#define SET_SB_BUF(sbp,s)	sbp->sb_buf = s
#define SET_SB_SIZE(sbp,n)	sbp->sb_size = n


#endif /* ! _STRBUF_H_ */

