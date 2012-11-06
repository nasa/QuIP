
#ifdef __cplusplus
extern "C" {
#endif


/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Filename:	version.h

Purpose:	Information about a files parent, name, sccs revision,
				and delta date.

Author:	Philippe A. Stassart (Sterling Federal Systems)

Revisions:
11-22-93	P. Stassart		Original release
10-18-94	P. Stassart		Use VersionId where ever possible
12-18-94	jbm			move lib_name field to beginning
					of structure for item pkg.
					Changed version structure to use lists.
									
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#include "query.h"

#define VERSION_ITEM_NAME	"module"

typedef struct date {
	int mo;
	int day;
	int yr;
} Date;

typedef struct file_version {
	const char *	vers_string;
	const char *	vers_symname;
} FileVersion;

typedef struct my_version
{
	Item		ver_item;
	List *		file_list;
} Version;

#define lib_name	ver_item.item_name
#define NO_VERSION	((Version *)NULL)


typedef struct version_request {
	const char *module_name;
	const char *id_string;
} Version_Request;


/* mkver.c */
extern void list_build_version(void);
extern void list_versions(SINGLE_QSP_ARG_DECL);

extern void list_all_files(SINGLE_QSP_ARG_DECL);
extern void list_libs(SINGLE_QSP_ARG_DECL);
extern void list_files(QSP_ARG_DECL  const char *);

#ifdef HAVE_HISTORY
extern void init_version_hist(QSP_ARG_DECL  const char *prompt);
#endif /* HAVE_HISTORY */

extern const char *get_progfile(void);
extern const char * get_symname(void);
extern void get_deferred_requests(SINGLE_QSP_ARG_DECL);
extern void auto_version(QSP_ARG_DECL  const char *module_name,const char *prefix);

extern void get_version_info(void);
extern void versupt(SINGLE_QSP_ARG_DECL);




#ifdef __cplusplus
}
#endif
