#include "quip_config.h"

char VersionId_interpreter_ncmds[] = QUIP_VERSION_STRING;

/**/
/**			cmd tables with word selectors		**/
/**/

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* qsort() */
#endif
#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#include "history.h"
#include "bi_cmds.h"
#include "debug.h"
#include "node.h"
#include "items.h"
#include "substr.h"
#include "query.h"
#include "savestr.h"
#include "help.h"
#include "submenus.h"

/* globals */


const char *	_help_p="";
Command *	_bctbl;
Command *	_hctbl;


/* local vars */

static const char *		_bpmpt;
const char *top_level_pmpt=NULL;

#ifdef THREAD_SAFE_QUERY

static pthread_mutex_t	cmd_mutex=PTHREAD_MUTEX_INITIALIZER;
static int cmd_flags=0;

#define LOCK_CMDS						\
								\
	if( n_active_threads > 1 )				\
	{							\
		int status;					\
								\
		status = pthread_mutex_lock(&cmd_mutex);	\
		if( status != 0 )				\
			report_mutex_error(QSP_ARG  status,"LOCK_CMDS");\
		cmd_flags |= LIST_LOCKED;			\
	}

#define UNLOCK_CMDS						\
								\
	if( cmd_flags & LIST_LOCKED )				\
	{							\
		int status;					\
								\
		cmd_flags &= ~LIST_LOCKED;			\
		status = pthread_mutex_unlock(&cmd_mutex);	\
		if( status != 0 )				\
			report_mutex_error(QSP_ARG  status,"UNLOCK_CMDS");\
	}

#else /* ! THREAD_SAFE_QUERY */

#define LOCK_CMDS
#define UNLOCK_CMDS

#endif /* ! THREAD_SAFE_QUERY */

/* local prototypes */


static void print_context_help(QSP_ARG_DECL  Item_Context *icp);
static char *get_cmd_ctx_name(QSP_ARG_DECL  const char *pmpt);

static Item_Context * load_menu(QSP_ARG_DECL  const char *pmpt,Command *ctbl);
static void del_ci(TMP_QSP_ARG_DECL  Item *cip);

static void cmd_init(Query_Stream *qsp);
static Command_Item *cmd_of(QSP_ARG_DECL  const char *name);
static Command_Item *new_cmd(QSP_ARG_DECL  const char *name);
static Command_Item *get_cmd(QSP_ARG_DECL  const char *name);
static Command_Item *del_cmd(Query_Stream *qsp, const char *name);


static void make_cmd_items(QSP_ARG_DECL  Command *ctbl);

#ifdef HAVE_HISTORY
static void load_defaults(QSP_ARG_DECL  const char *pmpt);
#endif /* HAVE_HISTORY */


#define CMD_ITEM_NAME	"CmdItem"

void cmd_init(Query_Stream *qsp)
{
	char classname[32];

#ifdef CAUTIOUS
	if( qsp->qs_cmd_itp != NO_ITEM_TYPE ){
		sprintf(ERROR_STRING,"CAUTIOUS:  cmd_init:  %s object class already initialized\n",qsp->qs_cmd_itp->it_name);
		WARN(ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */

	sprintf(classname,"%s_%d",CMD_ITEM_NAME,qsp->qs_serial);
	qsp->qs_cmd_itp = new_item_type(QSP_ARG  classname);
}

Command_Item * cmd_of(QSP_ARG_DECL  const char* s)
{
	if( THIS_QSP->qs_cmd_itp == NO_ITEM_TYPE )
		cmd_init(THIS_QSP);
	return( ( Command_Item * )item_of(QSP_ARG  THIS_QSP->qs_cmd_itp,s) );
}

Command_Item * get_cmd(QSP_ARG_DECL  const char* s)
{
	if( THIS_QSP->qs_cmd_itp == NO_ITEM_TYPE ) cmd_init(THIS_QSP);
	return( ( Command_Item * )get_item(QSP_ARG  THIS_QSP->qs_cmd_itp,s) );
}

Command_Item * new_cmd(QSP_ARG_DECL  const char* item_name)
{
	Command_Item *cip;

	if( THIS_QSP->qs_cmd_itp == NO_ITEM_TYPE ) cmd_init(THIS_QSP);
	cip = (Command_Item *) new_item( QSP_ARG  THIS_QSP->qs_cmd_itp, item_name, sizeof(*cip) );
	return(cip);
}

Command_Item * del_cmd(Query_Stream *qsp, const char* s)
{
	Command_Item * cip;

	cip=get_cmd(QSP_ARG  s);
	if( cip == NO_COMMAND_ITEM ){
		sprintf(ERROR_STRING,"Can't delete \"%s\" (doesn't exist)\n",s);
		WARN(ERROR_STRING);
		return(NO_COMMAND_ITEM);
	}
	del_item(QSP_ARG  THIS_QSP->qs_cmd_itp, cip);
	return(cip);
}


/* push_prompt - concatenate the new prompt fragment onto the existing prompt
 */

static void push_prompt(QSP_ARG_DECL  const char *pmpt)
{
	int i;

	if( (i=strlen(QUERY_PROMPT)) >= 2 ){
		i-=2;	/* assume prompt ends in "> " */
		QUERY_PROMPT[i]=0;
		strcat(QUERY_PROMPT,"/");
	}
	if( strlen(QUERY_PROMPT)+strlen(pmpt)+2 >= LLEN ){
		sprintf(ERROR_STRING,"Attempting to append prompt \"%s\" to previous prompt \"%s\"",
			pmpt,QUERY_PROMPT);
		advise(ERROR_STRING);
		ERROR1("prompt overflow!?");
	}

	strcat(QUERY_PROMPT,pmpt);

#ifdef CAUTIOUS
	if( !strcmp(&pmpt[strlen(pmpt)-2],"> ") ){
		/*
		sprintf(ERROR_STRING,"CAUTIOUS:  Prompt \"%s\" does not need to have \"> \" appended",
			pmpt);
		WARN(ERROR_STRING);
		QUERY_PROMPT[strlen(QUERY_PROMPT)-2]=0;
		*/
	}
#endif /* CAUTIOUS */

	strcat(QUERY_PROMPT,"> ");
}
/* BUG things don't work properly if different submenus use the same prompt */

static char *get_cmd_ctx_name(QSP_ARG_DECL  const char *pmpt)
{
#ifdef CAUTIOUS
if( strlen(CMD_ITEM_NAME)+strlen(pmpt) > LLEN-20 ){
sprintf(ERROR_STRING,"CAUTIOUS:  get_cmd_ctx_name:  string overflow (%ld+%ld)",
(u_long)strlen(CMD_ITEM_NAME),(u_long)strlen(pmpt));
WARN(ERROR_STRING);
sprintf(ERROR_STRING,"pmpt = \"%s\"",pmpt);
advise(ERROR_STRING);
ERROR1("CAUTIOUS:  exiting to prevent buffer overflow");
}
#endif /* CAUTIOUS */

	sprintf(THIS_QSP->qs_ctxname,"%s_%d.%s",CMD_ITEM_NAME,THIS_QSP->qs_serial,pmpt);
	return(THIS_QSP->qs_ctxname);
}

void set_bis(Command *ctbl,const char *pmpt,Command *hctbl,const char *hpmpt)		/** set builtin vectors */
{
	_bctbl=ctbl;
	_bpmpt=pmpt;
	_hctbl=hctbl;
	_help_p=hpmpt;

#ifdef HELPFUL
	builtin_help(_bpmpt);
#endif /* HELPFUL */
}

static void make_cmd_items(QSP_ARG_DECL  Command *ctbl)
{
	while( ctbl->cmd_sel != NULL ){
		Command_Item *cip;

		cip=new_cmd(QSP_ARG  ctbl->cmd_sel);
		if( cip != NO_COMMAND_ITEM ){
			cip->ci_cmdp = ctbl;
			cip->ci_qsp = THIS_QSP;
		}
		ctbl++;
	}
}

/* load_menu
 * Creates an item context for the commands...
 *
 * For multi-threading this gets tricky - we should not need to duplicate
 * the context for use by separate threads, but there is a potential
 * race condition if two threads try to create the same one...
 */

static Item_Context * load_menu(QSP_ARG_DECL  const char *pmpt,Command *ctbl)
{
	/* Command_Menu *mp; */
	Item_Context *icp;

#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  prompt is \"%s\"",
WHENCE(load_menu),pmpt);
advise(ERROR_STRING);
}
#endif	/* DEBUG */

	/* called from pushcmd() */

	if( THIS_QSP->qs_cmd_itp==NO_ITEM_TYPE ){
		cmd_init(THIS_QSP);		/* gets called 1st time */
		set_del_method(QSP_ARG  THIS_QSP->qs_cmd_itp,del_ci);
	}

	if( ! ( QUERY_FLAGS & QS_BUILTINS_INITED ) ){
		/* reset the flag before a recursive call */
		QUERY_FLAGS |= QS_BUILTINS_INITED;
		icp = load_menu(QSP_ARG  _help_p,_hctbl);
		PUSH_ITEM_CONTEXT(THIS_QSP->qs_cmd_itp,icp);
		icp=load_menu(QSP_ARG  _bpmpt,_bctbl);
		PUSH_ITEM_CONTEXT(THIS_QSP->qs_cmd_itp,icp);
	}

	icp = create_item_context(QSP_ARG  THIS_QSP->qs_cmd_itp,pmpt);
	PUSH_ITEM_CONTEXT(THIS_QSP->qs_cmd_itp,icp);
	make_cmd_items(QSP_ARG  ctbl);
	pop_item_context(QSP_ARG  THIS_QSP->qs_cmd_itp);

	return(icp);

} // end load_menu

/* Delete a command item...
 * We need this to be able to reload menus.
 *
 * Now that the command items are in per-query_stream item_type's, we need to have
 * the item type pointer here - which we normally get from the query stream.  But because
 * the delete routine has to conform to the item package (which doesn't pass around qsp's),
 * we have a problem...  unless the items themselves carry around the itp???
 */

static void del_ci(TMP_QSP_ARG_DECL  Item *ip)
{
	Command_Item *cip;

	cip = (Command_Item *) ip;

	del_cmd(cip->ci_qsp,cip->ci_item.item_name);	/* remove from database */
	rls_str((char *)cip->ci_item.item_name);
	/* don't free cmdp, because it points to a static table */
}

/* This command is used if we have changed the selector
 * words (as in adjust.c)
 *
 * BUG what will happen if the redefinition function is
 * in the menu that is being redefined?  In that case, the
 * context will already be pushed on the stack, and so if
 * the icp comes up different we could have a dangling ptr!?
 */

void reload_menu(QSP_ARG_DECL  const char *pmpt,Command *ctbl)
{
	Item_Context *icp;

	pmpt = get_cmd_ctx_name(QSP_ARG  pmpt);

	icp = ctx_of(QSP_ARG  pmpt);

#ifdef CAUTIOUS
	if( icp == NO_ITEM_CONTEXT ){
		WARN("couldn't find menu to reload");
		return;
	}
#endif /* CAUTIOUS */

	PUSH_ITEM_CONTEXT(THIS_QSP->qs_cmd_itp,icp);
	delete_item_context(QSP_ARG   icp);
	/* BUG do we need to free the storage??? */
	pop_item_context(QSP_ARG  THIS_QSP->qs_cmd_itp);

	/* redo it */
	icp = create_item_context(QSP_ARG  THIS_QSP->qs_cmd_itp,pmpt);
	PUSH_ITEM_CONTEXT(THIS_QSP->qs_cmd_itp,icp);
	make_cmd_items(QSP_ARG  ctbl);
	pop_item_context(QSP_ARG  THIS_QSP->qs_cmd_itp);
} /* end reload_menu */

void pushcmd(QSP_ARG_DECL  Command *ctbl,const char *pmpt)
{
	char *ctxname;
	Item_Context *icp;

#ifdef MAC

	create_menu(ctbl,pmpt);

#else /* ! MAC */

	if( THIS_QSP->qs_cmd_itp == NO_ITEM_TYPE ){
		top_level_pmpt = savestr(pmpt);
	}

	/* append the new prompt to the current prompt */
	push_prompt(QSP_ARG  pmpt);

	/* what does this do? */
	bi_init(SINGLE_QSP_ARG);

	/* find the context */
	ctxname = get_cmd_ctx_name(QSP_ARG  QUERY_PROMPT);
#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  looking for context %s",
WHENCE(pushcmd),ctxname);
advise(ERROR_STRING);
}
#endif	/* DEBUG */

	LOCK_CMDS

	icp = ctx_of(QSP_ARG  ctxname);

	if( icp == NO_ITEM_CONTEXT ){
#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  context %s not found, prompt is \"%s\"",
WHENCE(pushcmd),ctxname,pmpt);
advise(ERROR_STRING);
}
#endif	/* DEBUG */
		icp = load_menu(QSP_ARG  QUERY_PROMPT,ctbl);
#ifdef CAUTIOUS
		if( icp == NO_ITEM_CONTEXT ){
list_ctxs(SINGLE_QSP_ARG);
sprintf(ERROR_STRING,"menu context still not found for prompt %s, loading",pmpt);
advise(ERROR_STRING);
			ERROR1("CAUTIOUS:  couldn't make menu context");
		}
#endif /* CAUTIOUS */
	}
	UNLOCK_CMDS

	/* We have a single context stack for items - but we should
	 * have one per qsp???
	 */

	PUSH_ITEM_CONTEXT(THIS_QSP->qs_cmd_itp,icp);

#endif /* ! MAC */
} /* end pushcmd() */

void popcmd(SINGLE_QSP_ARG_DECL)
{
	int i;

	/* fix the prompt... */
	/* BUG we should only bother with this when we're running interactive! */
	i=strlen(QUERY_PROMPT);
	i--;
	while(i>=0 && QUERY_PROMPT[i] != '/' )
		i--;
	if( i>=0 && QUERY_PROMPT[i]=='/' ){
		QUERY_PROMPT[i]=0;
		strcat(QUERY_PROMPT,"> ");
	}

#ifdef CAUTIOUS
	if( THIS_QSP->qs_cmd_itp==NO_ITEM_TYPE )
		ERROR1("popcmd:  No command item type");
#endif /* CAUTIOUS */

	pop_item_context(QSP_ARG  THIS_QSP->qs_cmd_itp);
}

COMMAND_FUNC( top_menu )
{
	int n;
#ifdef CAUTIOUS
	int n2;
#endif /* CAUTIOUS */

	/* get the number of contexts */
	n = eltcount( CONTEXT_LIST(THIS_QSP->qs_cmd_itp) );
	/* pop all but one */

	while(n>MIN_CMD_DEPTH){
		popcmd(SINGLE_QSP_ARG);
		n--;
#ifdef CAUTIOUS
		n2 = eltcount( CONTEXT_LIST(THIS_QSP->qs_cmd_itp) );
		if( n != n2 )
			ERROR1("CAUTIOUS:  top_menu:  popcmd did not reduce number of command contexts!?");
#endif /* CAUTIOUS */
	}
}

/* there was a function popall() which seemed to be the same as top_menu()
 */

COMMAND_FUNC( push_top_menu )
{
	Node *np;
	Item_Context *icp;

	np= CONTEXT_LIST(THIS_QSP->qs_cmd_itp)->l_tail;	/* default context */
	np=np->n_last;				/* builtin help */

#ifdef CAUTIOUS
	if( np == NO_NODE ){
		WARN("CAUTIOUS:  push_top_menu:  missing bi help menu!?");
		return;
	}
#endif /* CAUTIOUS */

	np=np->n_last;				/* builtin menu */

#ifdef CAUTIOUS
	if( np == NO_NODE ){
		WARN("CAUTIOUS:  push_top_menu:  missing bi menu!?");
		return;
	}
#endif /* CAUTIOUS */

	np=np->n_last;				/* root menu */

#ifdef CAUTIOUS
	if( np == NO_NODE ){
		WARN("CAUTIOUS:  push_top_menu:  missing root menu!?");
		return;
	}
#endif /* CAUTIOUS */

	/* BUG should adjust the prompt here */
	push_prompt(QSP_ARG  top_level_pmpt);	/* CAUTIOUS?  check for non-null? */

	icp = (Item_Context *) np->n_data;
	PUSH_ITEM_CONTEXT(THIS_QSP->qs_cmd_itp,icp);
} /* end push_top_menu */

void hhelpme(QSP_ARG_DECL  const char *pmpt)		/* print out the current help items */
{
	Node *np;
	Item_Context *icp;
	char *ctxname;

	if( pmpt == NULL ){		/* current menu */
		icp = CURRENT_CONTEXT(THIS_QSP->qs_cmd_itp);
	} else {			/* builtin menu */
		ctxname = get_cmd_ctx_name(QSP_ARG  pmpt);
		icp = ctx_of(QSP_ARG  ctxname);
#ifdef CAUTIOUS
		if( icp == NO_ITEM_CONTEXT )
			ERROR1("CAUTIOUS:  hhelpme:  prompt context not found");
#endif /* CAUTIOUS */
	}

	print_context_help(QSP_ARG  icp);

	/* the help menu was pushed after the default,
	 * so it is back one from the tail */
	np = CONTEXT_LIST(THIS_QSP->qs_cmd_itp)->l_tail;
	np = np->n_last;
	icp = (Item_Context *) np->n_data;
	print_context_help(QSP_ARG  icp);
}

static void print_context_help(QSP_ARG_DECL  Item_Context *icp)
{
	List *lp;
	Node *np;

	lp = namespace_list(icp->ic_nsp);
#ifdef CAUTIOUS
	if( lp == NO_LIST )
		ERROR1("CAUTIOUS: couldn't get menu item list");
#endif /* CAUTIOUS */

	np=lp->l_head;

	while( np != NO_NODE ){
		Command_Item *cip;
		Command *cmdp;
		cip = (Command_Item *) np->n_data;
		cmdp = cip->ci_cmdp;
		sprintf(ERROR_STRING,
			"%-24s%s",cmdp->cmd_sel,cmdp->cmd_help);
		advise(ERROR_STRING);
		np = np->n_next;
	}

}

void getwcmd(SINGLE_QSP_ARG_DECL)		/** get command from ctbl */
{
	const char *s;
	Node *np;
	List *lp;
	Command_Item *cip;
	Item_Context *icp;
	const char *pmpt;

	lp=CONTEXT_LIST(THIS_QSP->qs_cmd_itp);
#ifdef CAUTIOUS
	if( lp==NO_LIST ) ERROR1("CAUTIOUS:  no command context!?");
#endif /* CAUTIOUS */

	np=lp->l_head;
	icp=(Item_Context *) np->n_data;
	pmpt = icp->ic_name;
	/* skip over "CmdItem_%d" - we used to do this with a fixed length,
	 * and most of the time we probably still could, because the serial number
	 * will rarely be more than 1 digit.  But to be safe, we check.
	 */

	/* pmpt += strlen(CMD_ITEM_NAME)+1; */	/* skip over CmdItem. */
	pmpt += strlen(CMD_ITEM_NAME)+3;	/* skip over CmdItem_1. */
	if( *pmpt == '.' ) pmpt++;		/* broken if more than 99 Query_Streams */

#ifdef HELPFUL
	if( intractive() ) command_help(pmpt);
#endif /* HELPFUL */

#ifdef HAVE_HISTORY

	if( intractive(SINGLE_QSP_ARG) && HISTORY_FLAG ){
		load_defaults(QSP_ARG  pmpt);
		load_defaults(QSP_ARG  _bpmpt);
	}

#endif /* HAVE_HISTORY */

	s=nameof2(QSP_ARG  pmpt);

	if( (*s) == 0 ) return;

	cip = cmd_of(QSP_ARG  s);
	if( cip != NO_COMMAND_ITEM ){	/* an exact match! */
		Command *cmdp;
		cmdp = cip->ci_cmdp;
		(*cmdp->cmd_func)(SINGLE_QSP_ARG);
		return;
	}

	/* here we do old linear scan for substrings ! */
	/* (for hashing to work, we require an exact match) */

	lp=item_list(QSP_ARG  THIS_QSP->qs_cmd_itp);
	np=lp->l_head;
	while(np!=NO_NODE){
		Command_Item *cip;
		Command *cmdp;

		cip=(Command_Item *) np->n_data;
		cmdp=cip->ci_cmdp;
		if( is_a_substring( s, cmdp->cmd_sel ) ){
/* BUG?  we should have a flag which inhibits this warning message... */
sprintf(ERROR_STRING,"matching substring \"%s\" to command word \"%s\"",
s,cmdp->cmd_sel);
WARN(ERROR_STRING);
			(*(cmdp->cmd_func))(SINGLE_QSP_ARG);
			return;
		}
		np=np->n_next;
	}
	
	/* BUG need to search the builtin commands for substrings too!? */

#ifdef HAVE_HISTORY
	/* make sure that a bad command doesn't get saved */
	if( intractive(SINGLE_QSP_ARG) && HISTORY_FLAG ){
		rem_def(QSP_ARG  pmpt,s);
	}
#endif /* HAVE_HISTORY */

	/* note that the string gets printed later */
	sprintf(ERROR_STRING,"unrecognized command \"");
	while( *s ){
		char str[8];
		if( isprint(*s) ) sprintf(str,"%c",*s);
		else sprintf(str,"\\%o",*s);
		strcat(ERROR_STRING,str);
		s++;
	}
	strcat(ERROR_STRING,"\"");
	WARN(ERROR_STRING);

	if( intractive(SINGLE_QSP_ARG) )
		advise("type '?' for a list of valid commands");
}

#ifdef HAVE_HISTORY

/* Initialize the history list for this prompt
 */

static void load_defaults(QSP_ARG_DECL  const char *pmpt)
{
	/* Command_Menu *mp; */
	Item_Context *cmd_icp;
	Item_Context *hist_icp;
	char *ctxname;

	ctxname = get_cmd_ctx_name(QSP_ARG  pmpt);

	cmd_icp = ctx_of( QSP_ARG  ctxname );
#ifdef CAUTIOUS
	if( cmd_icp == NO_ITEM_CONTEXT ){
		sprintf(ERROR_STRING,
			"CAUTIOUS:  missing cmd context \"%s\" in load_defaults",
			ctxname);
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	hist_icp = find_hist(QSP_ARG  pmpt);

	PUSH_ITEM_CONTEXT(choice_itp,hist_icp);

	if( hist_icp != NO_ITEM_CONTEXT )
		init_hist_from_list(QSP_ARG  pmpt,namespace_list(cmd_icp->ic_nsp));

	pop_item_context(QSP_ARG  choice_itp);

	/* BUG need to initialize the builtins too!? */
}

#endif /* HAVE_HISTORY */

int cmd_depth(SINGLE_QSP_ARG_DECL)
{
	if( THIS_QSP->qs_cmd_itp == NULL ) return(0);
	return( eltcount( CONTEXT_LIST(THIS_QSP->qs_cmd_itp)) );
}


