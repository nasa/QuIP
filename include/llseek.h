
#ifdef HAVE_LINUX_UNISTD_H
#include <linux/unistd.h>
#else

#ifdef HAVE_UNISTD_H		// defines lseek/off_t on Mac
#include <unistd.h>
#endif

#endif /* ! HAVE_LINUX_UNISTD */

#include "off64_t.h"

/* On 64bit Mac, there is no off64_t, but off_t is 64bits,
 * so we can just use lseek...
 */

// Shouldn't off64_t be defined in a system file somewhere?


/* Don't need this when _LARGEFILE64_SOURCE is defined... */
/* #include <stdint.h> */

typedef u_long blk_t;
typedef int errcode_t;

extern off64_t my_lseek64(int,off64_t,int);

extern errcode_t get_device_size(QSP_ARG_DECL  const char *,int,blk_t *);


