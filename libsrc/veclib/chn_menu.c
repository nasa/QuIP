
#include "quip_config.h"

#include <stdio.h>
#include <string.h>
#include "quip_prot.h"
#include "vec_chain.h"
//#include "nvf.h"
//#include "new_chains.h"
#include "getbuf.h"

static COMMAND_FUNC( do_chain_info )
{
	Chain *cp;

	cp=PICK_CHAIN( "chain buffer" );
	if( cp==NO_CHAIN ) return;
	chain_info(QSP_ARG  cp);
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

#define ADD_CMD(s,f,h)	ADD_COMMAND(chains_menu,s,f,h)

MENU_BEGIN(chains)
ADD_CMD( new_chain,	do_start_chain,	open a chain buffer		)
ADD_CMD( end_chain,	do_end_chain,	close current chain buffer	)
ADD_CMD( execute,	do_exec_chain,	execute a chain buffer		)
ADD_CMD( list,		do_list_chains,	list current chain buffers	)
ADD_CMD( info,		do_chain_info,	give info about a chain buffer	)
MENU_END(chains)

COMMAND_FUNC( do_chains )
{
	PUSH_MENU(chains);
}



