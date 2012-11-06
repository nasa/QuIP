
#include <sys/time.h>
#include "query.h"

typedef struct stamped_char {
	struct timeval		sc_tv;
	int			sc_n;
	char *			sc_buf;
} Stamped_Char;

extern COMMAND_FUNC( stamp_menu );

