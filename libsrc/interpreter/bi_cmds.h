
#ifndef _BI_CMDS_H_
#define _BI_CMDS_H_

#include "query.h"

#define RPT_CMD_WORD		"repeat"
#define FOR_CMD_WORD		"foreach"
#define END_CMD_WORD		"end"
#define DO_CMD_WORD		"do"
#define NOP_CMD_WORD		"nop"
#define WHILE_CMD_WORD		"while"
#define DATE_CMD_WORD		"date"

#ifdef MAC

#define ECHO_CMD_WORD		"Echo"
#define WARN_CMD_WORD		"Warn"
#define MAX_WARN_CMD_WORD	"Limit"
#define CLR_WARN_CMD_WORD	"Clear"
#define CNT_WARN_CMD_WORD	"Count"
#define DEBUG_CMD_WORD		"Debug"
#define VERBOSE_CMD_WORD	"Verbose"
#define ADVISE_CMD_WORD		"Advise"
#define IF_CMD_WORD		"if"
#define ID_CMD_WORD		"Identify"
#define POP_CMD_WORD		"popfile"

#define REDIR_CMD_STR		"Input"
#define XSCR_CMD_STR		"Transcribe"

#else /* !MAC */

#define ECHO_CMD_WORD		"echo"
#define WARN_CMD_WORD		"warn"
#define MAX_WARN_CMD_WORD	"max_warnings"
#define CLR_WARN_CMD_WORD	"clear_warnings"
#define CNT_WARN_CMD_WORD	"count_warnings"
#define DEBUG_CMD_WORD		"debug"
#define VERBOSE_CMD_WORD	"verbose"
#define ADVISE_CMD_WORD		"advise"
#define IF_CMD_WORD		"If"
#define ID_CMD_WORD		"identify"
#define POP_CMD_WORD		"PopFile"

#define REDIR_CMD_STR		"<"
#define XSCR_CMD_STR		">"

#endif /* !MAC */

/*
 * these are command functions which are declared in other
 * files besides builtin.c
 */

extern COMMAND_FUNC( vermenu );
extern void bi_init(SINGLE_QSP_ARG_DECL);
extern void my_onintr(int);		/* int arg for linux only? */


#endif /* !_BI_CMDS_H_ */

