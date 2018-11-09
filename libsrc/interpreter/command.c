
#include "quip_config.h"
#include "quip_prot.h"
#include "command.h"

void list_command(QSP_ARG_DECL  Command *cp)
{
	sprintf(MSG_STR,"%-24s\t%s",cp->cmd_selector,cp->cmd_help);
	advise(MSG_STR);
}

