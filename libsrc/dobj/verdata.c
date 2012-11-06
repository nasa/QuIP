#include "quip_config.h"

char VersionId_dataf_verdata[] = QUIP_VERSION_STRING;

#include "version.h"
#include "data_obj.h"

#ifdef FOOBAR

extern char    VersionId_dataf_areas[];
extern char    VersionId_dataf_arrays[];
extern char    VersionId_dataf_ascii[];
extern char    VersionId_dataf_contig[];
extern char    VersionId_dataf_data_fns[];
extern char    VersionId_dataf_data_obj[];
extern char    VersionId_dataf_dplist[];
extern char    VersionId_dataf_dataport[];
extern char    VersionId_dataf_dfuncs[];
extern char    VersionId_dataf_get_obj[];
extern char    VersionId_dataf_index[];
extern char    VersionId_dataf_makedobj[];
extern char    VersionId_dataf_memops[];
extern char    VersionId_dataf_sub_obj[];

FileVersion data_files[] = {
	{	VersionId_dataf_areas,	"areas.c"	},
	{	VersionId_dataf_arrays,	"arrays.c"	},
	{	VersionId_dataf_ascii,	"ascii.c"	},
	{	VersionId_dataf_contig,	"contig.c"	},
	{	VersionId_dataf_data_fns,	"data_fns.c"	},
	{	VersionId_dataf_data_obj,	"data_obj.c"	},
	{	VersionId_dataf_dplist,	"dplist.c"	},
	{	VersionId_dataf_dataport,	"dataport.c"	},
	{	VersionId_dataf_dfuncs,	"dfuncs.c"	},
	{	VersionId_dataf_get_obj,	"get_obj.c"	},
	{	VersionId_dataf_index,	"index.c"	},
	{	VersionId_dataf_makedobj,	"makedobj.c"	},
	{	VersionId_dataf_memops,	"memops.c"	},
	{	VersionId_dataf_sub_obj,	"sub_obj.c"	},
	{	VersionId_dataf_verdata,	"verdata.c"	}
};

#define MAX_DATA_FILES (sizeof(data_files)/sizeof(FileVersion))

void verdata()
{
	mkver ("DATA", data_files, MAX_DATA_FILES);
}

#else /* ! FOOBAR */

void verdata(SINGLE_QSP_ARG_DECL)
{
	auto_version(QSP_ARG  "DATA","VersionId_dataf");
}

#endif /* ! FOOBAR */
