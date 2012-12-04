#ifndef QUIP_CONFIG_H
#define QUIP_CONFIG_H

#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

#define MY_INTR	1	// a kludge...

/* When we used CVS (and before that SCCS) to manage
 * the source code, strings like the following held the version
 * data.  But now that we are using git, there are no longer
 * individual versions on the files.  Nevertheless, we still
 * can put the version string in each file to insure that
 * all were compiled at the same time.
 */

/* quip_version.h is generated automatically by a shell script */
#include "quip_version.h"

#endif

