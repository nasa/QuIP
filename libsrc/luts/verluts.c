#include "quip_config.h"

char VersionId_luts_verluts[] = QUIP_VERSION_STRING;

#include "version.h"
#include "cmaps.h"

#ifdef FOOBAR
extern char VersionId_luts_alpha[];
extern char VersionId_luts_bplanes[];
extern char VersionId_luts_cmfuncs[];
extern char VersionId_luts_linear[];
extern char VersionId_luts_lutbuf[];
extern char VersionId_luts_funcvec[];

#define N_LUTS_FILES	7

FileVersion luts_files[N_LUTS_FILES] = {
	{	VersionId_luts_alpha,	"alpha.c"	},
	{	VersionId_luts_bplanes,	"bplanes.c"	},
	{	VersionId_luts_cmfuncs,	"cmfuncs.c"	},
	{	VersionId_luts_linear,	"linear.c"	},
	{	VersionId_luts_lutbuf,	"lutbuf.c"	},
	{	VersionId_luts_funcvec,	"funcvec.c"	},
	{	VersionId_luts_verluts,	"verluts.c"	}
};

void verluts()
{
	mkver("LUTS", luts_files, N_LUTS_FILES);
}

#else /* ! FOOBAR */

void verluts(SINGLE_QSP_ARG_DECL)
{
	auto_version(QSP_ARG  "LUTS","VersionId_luts");
}

#endif /* ! FOOBAR */
