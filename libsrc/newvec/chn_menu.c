
#include "quip_config.h"

char VersionId_newvec_chn_menu[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include <string.h>
#include "debug.h"
#include "getbuf.h"
#include "items.h"
#include "nvf.h"
#include "new_chains.h"
#include "getbuf.h"

/* local prototypes */
static COMMAND_FUNC( do_chain_info );
static COMMAND_FUNC( do_start_chain );
static COMMAND_FUNC( do_end_chain );
static COMMAND_FUNC( do_exec_chain );

#define PICK_CHAIN(pmpt)	pick_vec_chain(QSP_ARG  pmpt)


static COMMAND_FUNC( do_chain_info )
{
	Chain *cp;

	cp=PICK_CHAIN( "chain buffer" );
	if( cp==NO_CHAIN ) return;
	chain_info(cp);
}

static COMMAND_FUNC( do_start_chain )
{
	const char *s;
	Chain *cp;

	s=NAMEOF("new chain");
	cp=vec_chain_of( QSP_ARG   s );
	if( cp != NO_CHAIN ){
		WARN("chain name already in use");
		return;
	}

	start_chain(QSP_ARG  s);
}

static COMMAND_FUNC( do_end_chain )
{
	end_chain();
}

static COMMAND_FUNC( do_exec_chain )
{
	Chain *cp;

	cp=PICK_CHAIN( "name of chain buffer" );
	if( cp==NO_CHAIN ) return;
	exec_chain(cp);
}

static COMMAND_FUNC( do_list_chains )
{ list_vec_chains(SINGLE_QSP_ARG); }

Command chn_ctbl[]={
{ "new_chain",	do_start_chain,	"open a chain buffer",			},
{ "end_chain",	do_end_chain,	"close current chain buffer",		},
{ "execute",	do_exec_chain,	"execute a chain buffer",		},
{ "list",	do_list_chains,	"list current chain buffers",		},
{ "info",	do_chain_info,	"give info about a chain buffer",	},
{ "quit",	popcmd,		"exit submenu",				},
{ NULL,		NULL,		NULL					},
};

COMMAND_FUNC( do_chains )
{
	PUSHCMD(chn_ctbl,"chains");
}



