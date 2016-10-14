#include "quip_config.h"

#ifdef HAVE_POPEN

/** open a subprocess */

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "query_prot.h"
#include "pipe_support.h"
#include "query_bits.h"		// my_pipe struct declared here???
#include "item_type.h"

ITEM_INTERFACE_DECLARATIONS(Pipe,pipe,0)

void creat_pipe(QSP_ARG_DECL  const char *name, const char* command, const char* rw)
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
	del_pipe(QSP_ARG  pp);
	rls_str(pp->p_name);
	rls_str(pp->p_cmd);
}

void sendto_pipe(QSP_ARG_DECL  Pipe *pp,const char* text)
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

void readfr_pipe(QSP_ARG_DECL  Pipe *pp,const char* varname)
{
	char buf[LLEN];

	if( (pp->p_flgs & READ_PIPE) == 0 ){
		sprintf(ERROR_STRING,"Can't read from  write pipe %s",pp->p_name);
		WARN(ERROR_STRING);
		return;
	}

	if( fgets(buf,LLEN,pp->p_fp) == NULL ){
		if( verbose ){
			sprintf(ERROR_STRING,
		"read failed on pipe \"%s\"",pp->p_name);
			advise(ERROR_STRING);
		}
		close_pipe(QSP_ARG  pp);
		ASSIGN_VAR(varname,"pipe_read_error");
	} else {
		/* remove trailing newline */
		if( buf[strlen(buf)-1] == '\n' )
			buf[strlen(buf)-1] = 0;
		ASSIGN_VAR(varname,buf);
	}
}

#endif /* HAVE_POPEN */

