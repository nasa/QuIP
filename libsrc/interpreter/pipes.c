#include "quip_config.h"

char VersionId_interpreter_pipes[] = QUIP_VERSION_STRING;

#ifdef HAVE_POPEN

/** open a subprocess */

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "debug.h"
#include "items.h"
#include "savestr.h"
#include "items.h"
#include "menuname.h"
#include "submenus.h"

static void creat_pipe(QSP_ARG_DECL  const char *name,const char *command,const char *rw);
//static void close_pipe(Pipe *pp);
static void sendto_pipe(QSP_ARG_DECL  Pipe *pp,const char *text);
static void readfr_pipe(QSP_ARG_DECL  Pipe *pp,const char *varname);

static COMMAND_FUNC( do_newpipe );
static COMMAND_FUNC( do_sendpipe );
static COMMAND_FUNC( do_pipe_info );
static COMMAND_FUNC( do_closepipe );

ITEM_INTERFACE_DECLARATIONS(Pipe,pipe)

static void creat_pipe(QSP_ARG_DECL  const char *name, const char* command, const char* rw)
{
	Pipe *pp;
	int flg;

	if( *rw == 'r' ) flg=READ_PIPE;
	else if( *rw == 'w' ) flg=WRITE_PIPE;
	else {
		sprintf(ERROR_STRING,"create_pipe:  bad r/w string \"%s\"",rw);
		WARN(ERROR_STRING);
		return;
	}

	pp = new_pipe(QSP_ARG  name);
	if( pp == NO_PIPE ) return;

	pp->p_cmd = savestr(command);
	pp->p_flgs = flg;
	pp->p_fp = popen(command,rw);

	if( pp->p_fp == NULL ){
		sprintf(ERROR_STRING,
			"unable to execute command \"%s\"",command);
		WARN(ERROR_STRING);
		close_pipe(QSP_ARG  pp);
	}
}

void close_pipe(QSP_ARG_DECL  Pipe *pp)
{
	if( pp->p_fp != NULL && pclose(pp->p_fp) == -1 ){
		sprintf(ERROR_STRING,"Error closing pipe \"%s\"!?",pp->p_name);
		WARN(ERROR_STRING);
	}
	pp = del_pipe(QSP_ARG  pp->p_name);
	rls_str(pp->p_name);
	rls_str(pp->p_cmd);
}

static void sendto_pipe(QSP_ARG_DECL  Pipe *pp,const char* text)
{
	if( (pp->p_flgs & WRITE_PIPE) == 0 ){
		sprintf(ERROR_STRING,"Can't write to read pipe %s",pp->p_name);
		WARN(ERROR_STRING);
		return;
	}

	if( fprintf(pp->p_fp,"%s\n",text) == EOF ){
		sprintf(ERROR_STRING,
			"write failed on pipe \"%s\"",pp->p_name);
		WARN(ERROR_STRING);
		close_pipe(QSP_ARG  pp);
	} else if( fflush(pp->p_fp) == EOF ){
		sprintf(ERROR_STRING,
			"fflush failed on pipe \"%s\"",pp->p_name);
		WARN(ERROR_STRING);
		close_pipe(QSP_ARG  pp);
	}
#ifdef DEBUG
	else if( debug ) advise("pipe flushed");
#endif /* DEBUG */
}

static void readfr_pipe(QSP_ARG_DECL  Pipe *pp,const char* varname)
{
	char buf[LLEN];

	if( (pp->p_flgs & READ_PIPE) == 0 ){
		sprintf(ERROR_STRING,"Can't read from  write pipe %s",pp->p_name);
		WARN(ERROR_STRING);
		return;
	}

	if( fgets(buf,LLEN,pp->p_fp) == NULL ){
		sprintf(ERROR_STRING,
			"read failed on pipe \"%s\"",pp->p_name);
		WARN(ERROR_STRING);
		close_pipe(QSP_ARG  pp);
	}

	if( buf[strlen(buf)-1] == '\n' )
		buf[strlen(buf)-1] = 0;		/* remove trailing newline */

	ASSIGN_VAR(varname,buf);
}

#define N_RW_CHOICES	2
static const char *rw_choices[N_RW_CHOICES]={"r","w"};

static COMMAND_FUNC( do_newpipe )
{
	const char *s;
	char name[LLEN], cmdbuf[LLEN];
	int n;

	s=NAMEOF("name for pipe");
	strcpy(name,s);

	n=WHICH_ONE("read/write",N_RW_CHOICES,rw_choices);

	s=NAMEOF("command");
	strcpy(cmdbuf,s);

	if( n< 0 ) return;

	creat_pipe(QSP_ARG  name,cmdbuf,rw_choices[n]);
}

static COMMAND_FUNC( do_sendpipe )
{
	Pipe *pp;
	const char *s;

	pp=PICK_PIPE("");
	s=NAMEOF("text to send");

	if( pp == NO_PIPE ) return;
	sendto_pipe(QSP_ARG  pp,s);
}

static COMMAND_FUNC( do_readpipe )
{
	Pipe *pp;
	const char *s;

	pp=PICK_PIPE("");
	s=NAMEOF("variable for text storage");

	if( pp == NO_PIPE ) return;
	readfr_pipe(QSP_ARG  pp,s);
}

static COMMAND_FUNC( do_pipe_info )
{
	Pipe *pp;
	int i;

	pp = PICK_PIPE("");
	if( pp == NO_PIPE ) return;

	if( pp->p_flgs & READ_PIPE ) i=0;
	else if( pp->p_flgs & READ_PIPE ) i=1;
#ifdef CAUTIOUS
	else {
		WARN("CAUTIOUS:  bad pipe r/w flag");
		return;
	}
#endif /* CAUTIOUS */

	sprintf(msg_str,"Pipe:\t\t\"%s\", %s",pp->p_name,rw_choices[i]);
	prt_msg(msg_str);
	sprintf(msg_str,"Command:\t\t\"%s\"",pp->p_cmd);
	prt_msg(msg_str);
}

static COMMAND_FUNC( do_closepipe )
{
	Pipe *pp;

	pp = PICK_PIPE("");
	if( pp == NO_PIPE ) return;

	close_pipe(QSP_ARG  pp);
}

static COMMAND_FUNC( do_list_pipes ){ list_pipes(SINGLE_QSP_ARG); }

/* fp should be null - but where do we specify the pipe handle? */

static COMMAND_FUNC( do_pipe_redir )
{
	Pipe *pp;

	pp = PICK_PIPE("");
	if( pp == NO_PIPE ) return;

	if( (pp->p_flgs & READ_PIPE) == 0 ) {
		sprintf(ERROR_STRING,"do_pipe_redir:  pipe %s is not readable!?",pp->p_name);
		WARN(ERROR_STRING);
		return;
	}

	sprintf(msg_str,"Pipe \"%s\"",pp->p_cmd);
	push_input_file(QSP_ARG  msg_str);
	redir(QSP_ARG pp->p_fp);
	THIS_QSP->qs_query[QLEVEL].q_dupfile = (FILE *) pp;
	THIS_QSP->qs_query[QLEVEL].q_flags |= Q_PIPE;
}

#endif /* HAVE_POPEN */

Command pipe_ctbl[]={
{ "open",	do_newpipe,	"create new pipe"		},
{ "sendto",	do_sendpipe,	"send input text to pipe"	},
{ "read",	do_readpipe,	"read output text from pipe"	},
{ "close",	do_closepipe,	"close pipe"			},
{ "list",	do_list_pipes,	"list currently open pipes"	},
{ "info",	do_pipe_info,	"give information about a pipe"	},
{ "redir",	do_pipe_redir,	"redirect interpreter to a readable pipe"	},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"			},
#endif /* !MAC */
{ NULL,		NULL,		NULL				}
};

COMMAND_FUNC( pipemenu )
{
	PUSHCMD(pipe_ctbl,PIPE_MENU_NAME);
}


