#include "quip_config.h"

char VersionId_lutmenu_verlutm[] = QUIP_VERSION_STRING;

#include "version.h"
#include "verlutm.h"

#ifdef FOOBAR
extern char VersionId_lutmenu_bitmenu[];
extern char VersionId_lutmenu_cmmenu[];
extern char VersionId_lutmenu_linmenu[];
extern char VersionId_lutmenu_lutmenu[];

#define N_LUTMENU_FILES	5

FileVersion lutm_files[N_LUTMENU_FILES] = {
	{	VersionId_lutmenu_bitmenu,	"bitmenu.c"	},
	{	VersionId_lutmenu_cmmenu,	"cmmenu.c"	},
	{	VersionId_lutmenu_linmenu,	"linmenu.c"	},
	{	VersionId_lutmenu_lutmenu,	"lutmenu.c"	},
	{	VersionId_lutmenu_verlutm,	"verlutm.c"	}
};


void verlutm()
{
	mkver("LUTMENU", lutm_files, N_LUTMENU_FILES);
}

#else /* ! FOOBAR */

void verlutm(SINGLE_QSP_ARG_DECL)
{
	auto_version(QSP_ARG  "LUTMENU","VersionId_lutmenu");
}

#endif /* ! FOOBAR */
