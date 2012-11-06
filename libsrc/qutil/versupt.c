#include "quip_config.h"

char VersionId_qutil_versupt[] = QUIP_VERSION_STRING;

#define INC_VERSION

/* these files are all included in the support lib */
/* #include "error.h" */ /* included from somewhere else!? */
#include "freel.h"
#include "function.h"
#include "getbuf.h"
#include "hash.h"
//#include "is_unix.h"	/* really need this one ! */
#include "items.h"
#include "query.h"
#include "macros.h"
#include "node.h"
#include "query.h"
#include "rn.h"
#include "savestr.h"
#include "sigpush.h"
#include "strbuf.h"
#include "substr.h"
#include "typedefs.h"
#include "query.h"
#include "version.h"	/* really need this one ! */
#include "vecgen.h"

#ifdef FOOBAR

#include "supfiles.h"		/* ext decls of sccs strings */

/* we had termio.c here, but no termio on non-unix systems - right? */

#define N_SUPPORT_FILES	23
/* BUG - NOT UP-TO-DATE !!! */

FileVersion support_files[N_SUPPORT_FILES] = {
	{	VersionId_qutil_debug,		"debug.c"	},
	{	VersionId_qutil_error,		"error.c"	},
	{	VersionId_qutil_expr,		"expr.c"	},
	{	VersionId_qutil_freel,		"freel.c"	},
	{	VersionId_qutil_function,	"functio.c"	},
	{	VersionId_qutil_getbuf,		"getbuf.c"	},
	{	VersionId_qutil_hash,		"hash.c"	},
	{	VersionId_qutil_items,		"items.c"	},
	{	VersionId_qutil_mkver,		"mkver.c"	},
	{	VersionId_qutil_node,		"node.c"	},
	{	VersionId_qutil_rn,		"rn.c"		},
	{	VersionId_qutil_savestr,	"savestr.c"	},
	{	VersionId_qutil_sigpush,	"sigpush.c"	},
	{	VersionId_qutil_substr,		"substr.c"	},
	{	VersionId_qutil_tkspace,	"tkspace.c"	},
	{	VersionId_qutil_tryhard,	"tryhard.c"	},
	{	VersionId_qutil_strbuf,		"strbuf.c"	},
	{	VersionId_qutil_versupt,	"versupt.c"	},
	{	VersionId_qutil_class,		"class.c"	},
	{	VersionId_qutil_handle,		"handle.c"	},
	{	VersionId_qutil_pathnm,		"pathnm.c"	}
};

void versupt()
{
	mkver ("SUPPORT", support_files, N_SUPPORT_FILES);
}

#else /* ! FOOBAR */

void versupt(SINGLE_QSP_ARG_DECL)
{
	auto_version(QSP_ARG  "SUPPORT","VersionId_qutil");
	/* We used to use strings here too, but that wasn't
	 * very well thought out...
	 */
	/*
	auto_version(QSP_ARG  "INCLUDE","VersionId_inc");
	*/
}

#endif /* ! FOOBAR */


