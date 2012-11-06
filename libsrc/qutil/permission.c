#include "quip_config.h"

char VersionId_qutil_permission[] = QUIP_VERSION_STRING;

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#include "query.h"

/* if the program is setuid root, make the effective user id the same as
 * the real userid...
 */

void check_suid_root(void)
{
	uid_t ruid, euid;

	ruid = getuid();
	euid = geteuid();

	if( euid == 0 ){
		NADVISE("Effective uid is 0");
		seteuid(ruid);
	}
}

