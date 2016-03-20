
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif /* HAVE_SYS_TIME_H */

#include "quip_prot.h"

typedef struct stamped_char {
	struct timeval		sc_tv;
	int			sc_n;
	char *			sc_buf;
} Stamped_Char;

extern COMMAND_FUNC( stamp_menu );

