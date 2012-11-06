#include "quip_config.h"

char VersionId_datamenu_verdatam[] = QUIP_VERSION_STRING;

#include "version.h"
#include "dataprot.h"

#ifdef FOOBAR

extern char	VersionId_datamenu_ascmenu[];
extern char	VersionId_datamenu_datamenu[];
#ifndef PC
extern char	VersionId_datamenu_fiomenu[];
#endif /* PC */
extern char	VersionId_datamenu_ops_menu[];

FileVersion datam_files[] = 
{
	{	VersionId_datamenu_ascmenu,	"ascmenu.c"	},
	{	VersionId_datamenu_datamenu,	"datamenu.c"	},
#ifndef PC
	{	VersionId_datamenu_fiomenu,	"fiomenu.c"	},
#endif /* PC */
	{	VersionId_datamenu_ops_menu,	"ops_menu.c"	},
	{	VersionId_datamenu_verdatam,	"verdatam.c"	}
};

#define MAX_DATAMENU_FILES (sizeof(datam_files)/sizeof(FileVersion))

void verdatam()
{
	mkver("DATAMENU", datam_files, MAX_DATAMENU_FILES);
}
#else /* ! FOOBAR */

void verdatam(SINGLE_QSP_ARG_DECL)
{
	auto_version(QSP_ARG  "DATAMENU","VersionId_datamenu");
}

#endif /* ! FOOBAR */
