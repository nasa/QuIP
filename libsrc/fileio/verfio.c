#include "quip_config.h"

char VersionId_fio_verfio[] = QUIP_VERSION_STRING;

#include "query.h"
#include "version.h"

/* We might like to define INC_VERSION here, but we get a multiply defined error
 * because data_obj.h is used in support (expr.y)...  the real solution would
 * be to explicitly include the files needed, in the unix section, making sure
 * that the version string declarations come before any conditional reading of
 * the rest of the .h file.
 */

#include "img_file.h"
#include "verfio.h"


#ifdef FOOBAR
extern char    VersionId_fio_filetype[];
extern char    VersionId_fio_rdoldhdr[];
extern char    VersionId_fio_get_hdr[];
extern char    VersionId_fio_herrs[];
extern char    VersionId_fio_hips1[];
extern char    VersionId_fio_hips2[];
extern char    VersionId_fio_hsizepix[];
extern char    VersionId_fio_img_file[];
extern char    VersionId_fio_perr[];
extern char    VersionId_fio_raw[];
extern char    VersionId_fio_readhdr[];
extern char    VersionId_fio_read_raw[];
extern char    VersionId_fio_rgb[];
extern char    VersionId_fio_sunras[];
#ifndef PC
extern char    VersionId_fio_viff[];
#endif /* !PC */
extern char    VersionId_fio_writehdr[];
extern char    VersionId_fio_wsubs[];
extern char    VersionId_fio_xparam[];
extern char    VersionId_fio_vista[];

FileVersion fio_files[] = {
	{	VersionId_fio_filetype,	"filetype.c"	},
	{	VersionId_fio_rdoldhdr,	"rdoldhdr.c"	},
	{	VersionId_fio_get_hdr,	"get_hdr.c"	},
	{	VersionId_fio_herrs,	"herrs.c"	},
	{	VersionId_fio_hips1,	"hips1.c"	},
	{	VersionId_fio_hips2,	"hips2.c"	},
	{	VersionId_fio_hsizepix,	"hsizepix.c"	},
	{	VersionId_fio_img_file,	"img_file.c"	},
	{	VersionId_fio_perr,	"perr.c"	},
	{	VersionId_fio_raw,		"fio_raw.c"	},
	{	VersionId_fio_readhdr,	"readhdr.c"	},
	{	VersionId_fio_read_raw,	"read_raw.c"	},
	{	VersionId_fio_rgb,		"rgb.c"		},
	{	VersionId_fio_sunras,	"sunras.c"	},
#ifndef PC
	{	VersionId_fio_viff,	"viff.c"	},
#endif /* !PC */
	{	VersionId_fio_writehdr,	"writehdr.c"	},
	{	VersionId_fio_wsubs,	"wsubs.c"	},
	{	VersionId_fio_xparam,	"xparam.c"	},
	{	VersionId_fio_vista,	"vista.c"	},
};

#define MAX_FILEIO_FILES (sizeof(fio_files)/sizeof(FileVersion))

void verfio(SINGLE_QSP_ARG_DECL)
{
	mkver("FILEIO", fio_files, MAX_FILEIO_FILES);
}

#else /* ! FOOBAR */

void verfio(SINGLE_QSP_ARG_DECL)
{
	auto_version(QSP_ARG  "FILEIO","VersionId_fio");
#ifdef HAVE_JJPEG
	auto_version(QSP_ARG  "JJPEG","VersionId_jjpeg");
#endif
}

#endif /* ! FOOBAR */

