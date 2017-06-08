
#ifndef _COMMAND_H_
#define _COMMAND_H_

#include "quip_fwd.h"

struct command {
	const char *	cmd_selector;
	const char *	cmd_help;
	int		cmd_serial;
	void		(*cmd_action)(SINGLE_QSP_ARG_DECL);
} ;

#define CMD_SELECTOR(cp)	cp->cmd_selector

extern void list_command(QSP_ARG_DECL  Command *cp);

#endif /* !  _COMMAND_H_ */

