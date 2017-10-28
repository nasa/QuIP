#include "quip_config.h"

#ifdef HAVE_POPEN

/** open a subprocess */

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "query_bits.h"	// for some reason that is where my_pipe is layed out...
#include "pipe_support.h"
#include "function.h"
#include "query_stack.h"	// BUG eliminate dependency

#define N_RW_CHOICES	2

// we have to pass "r" or "w" to popen, but it is nice to have
// the more human-readable complete strings in the scripts...

static const char *rw_choices[N_RW_CHOICES]={"read","write"};

static COMMAND_FUNC( do_newpipe )
{
	const char *pipe_name;
	const char *cmd;
	const char *mode;
	int n;

	pipe_name=NAMEOF("name for pipe");

	n=WHICH_ONE("read/write",N_RW_CHOICES,rw_choices);

	cmd=NAMEOF("command");

	if( n < 0 || n >= 2 ) return;
	if( n == 0 ) mode="r";
	else mode="w";
	
	creat_pipe(QSP_ARG  pipe_name,cmd,mode);
}

static COMMAND_FUNC( do_sendpipe )
{
	Pipe *pp;
	const char *s;

	pp=pick_pipe("");
	s=NAMEOF("text to send");

	if( pp == NULL ) return;
	sendto_pipe(QSP_ARG  pp,s);
}

static COMMAND_FUNC( do_readpipe )
{
	Pipe *pp;
	const char *s;

	pp=pick_pipe("");
	s=NAMEOF("variable for text storage");

	if( pp == NULL ) {
		assign_var(s,"error_missing_pipe");
		return;
	}
	readfr_pipe(QSP_ARG  pp,s);
}

static COMMAND_FUNC( do_pipe_info )
{
	Pipe *pp;
	int i;

	pp = pick_pipe("");
	if( pp == NULL ) return;

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

	pp = pick_pipe("");
	if( pp == NULL ) return;

	close_pipe(QSP_ARG  pp);
}

static COMMAND_FUNC( do_list_pipes ){ list_pipes(tell_msgfile()); }

/* fp should be null - but where do we specify the pipe handle? */

static COMMAND_FUNC( do_pipe_redir )
{
	Pipe *pp;

	pp = pick_pipe("");
	if( pp == NULL ) return;

	if( (pp->p_flgs & READ_PIPE) == 0 ) {
		sprintf(ERROR_STRING,
	"do_pipe_redir:  pipe %s is not readable!?",pp->p_name);
		WARN(ERROR_STRING);
		return;
	}

	//push_input_file(QSP_ARG  msg_str);
	sprintf(msg_str,"Pipe \"%s\"",pp->p_cmd);
	redir_from_pipe(QSP_ARG pp, msg_str);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(pipes_menu,s,f,h)

MENU_BEGIN(pipes)
ADD_CMD( open,		do_newpipe,	create new pipe )
ADD_CMD( sendto,	do_sendpipe,	send input text to pipe )
ADD_CMD( read,		do_readpipe,	read output text from pipe )
ADD_CMD( close,		do_closepipe,	close pipe )
ADD_CMD( list,		do_list_pipes,	list currently open pipes )
ADD_CMD( info,		do_pipe_info,	give information about a pipe )
ADD_CMD( redir,		do_pipe_redir,	redirect interpreter to a readable pipe )
MENU_END(pipes)

static double pipe_exists(QSP_ARG_DECL  const char *s)
{
	Pipe *pp;

	pp=pipe_of(s);
	if( pp==NULL ) return(0.0);
	else return(1.0);
}

COMMAND_FUNC( do_pipe_menu )
{
	static int inited=0;
	if( ! inited ){
		DECLARE_STR1_FUNCTION(	pipe_exists,	pipe_exists )
		inited=1;
	}
	CHECK_AND_PUSH_MENU(pipes);
}


#endif /* HAVE_POPEN */

