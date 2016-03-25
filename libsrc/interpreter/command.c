
#include "quip_config.h"
#include "quip_prot.h"

void list_command(QSP_ARG_DECL  Command *cp)
{
	sprintf(MSG_STR,"%-24s\t%s",cp->cmd_selector,cp->cmd_help);
	prt_msg(MSG_STR);
}

