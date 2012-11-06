/* version information submenu */

#include "quip_config.h"

char VersionId_interpreter_ver_cmds[] = QUIP_VERSION_STRING;


#include <stdio.h>
#include <string.h>
#include "version.h"
#include "bi_cmds.h"
#include "menuname.h"
#include "submenus.h"
#include "query.h"	/* prototype for intractive() */

static void do_list_files(SINGLE_QSP_ARG_DECL);

#define MORE 1
#define NO_MORE !MORE

static void do_list_files(SINGLE_QSP_ARG_DECL)
{
#ifdef HAVE_HISTORY
	if( intractive(SINGLE_QSP_ARG) ) init_version_hist(QSP_ARG  VERSION_ITEM_NAME);
#endif /* HAVE_HISTORY */

	list_files( QSP_ARG  nameof(QSP_ARG  VERSION_ITEM_NAME) );
}

static COMMAND_FUNC( do_list_all_files )
{ list_all_files(SINGLE_QSP_ARG); }

static COMMAND_FUNC( do_list_libs )
{
	prt_msg("");
	list_build_version();
	prt_msg("");
	list_libs(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_list_build ){ list_build_version(); }

Command ver_ctbl[]={
{ "build",	do_list_build,		"display build version"	},
{ "list",	do_list_libs,		"list modules with registered version info"	},
{ "files",	do_list_files,		"list module files & version numbers"		},
{ "all",	do_list_all_files,	"list all files"				},
{ "features",	do_list_features,	"show compile-time flags"			},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"						},
#endif /* !MAC */
{ NULL_COMMAND										}
};

COMMAND_FUNC( vermenu )
{
	static int	inited_ver = 0;

	if (!inited_ver) {
		versupt(SINGLE_QSP_ARG);
		inited_ver = 1;
	}

	get_deferred_requests(SINGLE_QSP_ARG);

	PUSHCMD(ver_ctbl,VERSION_MENU_NAME);
}

