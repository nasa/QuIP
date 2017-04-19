
#include <string.h>
#include "quip_config.h"
#include "quip_prot.h"
#include "query_stack.h"
#include "quip_menu.h"
#include "macro.h"
#include "variable.h"
//#include "qwarn.h"
#include "warn.h"
#include "nexpr.h"
#include "query_prot.h"
#include "getbuf.h"
#include "history.h"

Query_Stack *_defaultQSP=NULL;
static Item_Type *query_stack_itp=NULL;
ITEM_INIT_FUNC(Query_Stack,query_stack,0)
ITEM_LIST_FUNC(Query_Stack,query_stack)
ITEM_NEW_FUNC(Query_Stack,query_stack)
ITEM_PICK_FUNC(Query_Stack,query_stack)
// new_query_stack is defined here, not using the standard template...

#define IS_LEGAL_VARNAME_CHAR(c)	(isalnum(c) || c=='_')

//static List *qstack_free_list=NO_LIST;

static void push_prompt(QSP_ARG_DECL  const char *pmpt);
static void pop_prompt(SINGLE_QSP_ARG_DECL);

void push_menu(QSP_ARG_DECL  Menu *mp)
{
	push_item(QS_MENU_STACK(THIS_QSP),mp);

	// Fix the prompt
	push_prompt(QSP_ARG  MENU_PROMPT(mp));
}

COMMAND_FUNC( do_exit_prog )
{
#ifdef BUILD_FOR_OBJC
	ios_exit_program();	// doesn't exit right away???
#else // ! BUILD_FOR_OBJC
	nice_exit(QSP_ARG  0);
#endif	// ! BUILD_FOR_OBJC
}

Menu *pop_menu(SINGLE_QSP_ARG_DECL)
{
	Menu *mp;

	pop_prompt(SINGLE_QSP_ARG);
	mp = pop_item(QS_MENU_STACK(THIS_QSP));

	// If the last menu was popped, then exit the program.
	if( STACK_IS_EMPTY(QS_MENU_STACK(THIS_QSP)) ){
#ifdef BUILD_FOR_CMD_LINE
        
        do_exit_prog(SINGLE_QSP_ARG);
        
#else // ! BUILD_FOR_CMD_LINE

	WARN("unexpected call to pop_menu!?");
        
fprintf(stderr,"pop_menu:  menu stack empty, exiting program...\n");
abort();    // for debugging...
            // need to call do_exit_prog if not debugging!
        
#endif // ! BUILD_FOR_CMD_LINE
	}

	return mp;
}

void show_menu_stack(SINGLE_QSP_ARG_DECL)
{
	// a stack is implemented as a list...
	Node *np;
	List *lp;
	Menu *mp;
	int i=0;

	lp = QS_MENU_STACK(THIS_QSP);
	if( lp == NO_LIST ){
		WARN("show_menu_stack:  no menu stack!?");
		return;
	}
	np = QLIST_HEAD(lp);
	prt_msg("Menu stack:");
	while( np != NO_NODE ){
		mp = (Menu *) NODE_DATA(np);
		sprintf(MSG_STR,"\tmenu %d: %s",i++,MENU_PROMPT(mp));
		prt_msg(MSG_STR);

		np = NODE_NEXT(np);
	}
}

Query_Stack * init_first_query_stack(void)
{
	default_qsp = new_qstk(NULL_QSP_ARG  "First_Query_Stack");
	return(default_qsp);
}

#ifdef FOOBAR
Query_Stack *new_query_stack(QSP_ARG_DECL  const char *name)
{
	Query_Stack *new_qsp;
	Node *np;

	np = QLIST_HEAD(qstack_free_list);
	if( np == NO_NODE ){
		new_qsp=(Query_Stack *)getbuf(sizeof(Query_Stack));
//fprintf(stderr,"new_query_stack calling savestr #1\n");
		new_qsp->qs_item.item_name = savestr(name);
	} else {
		np = remHead(qstack_free_list);
		new_qsp = (Query_Stack *) NODE_DATA(np);
//fprintf(stderr,"new_query_stack calling savestr #2\n");
		new_qsp->qs_item.item_name = savestr(name);
	}

	return(new_qsp);
}
#endif // FOOBAR

/* push_prompt - concatenate the new prompt fragment onto the existing prompt
 *
 * This is the old version which may have assumed a statically allocated string...
 * Now we use a String_Buf...
 */

static void push_prompt(QSP_ARG_DECL  const char *pmpt)
{
	long n;
	String_Buf *sbp;

	sbp = QS_PROMPT_SB(THIS_QSP);
	if( sbp == NO_STRINGBUF )
		SET_QS_PROMPT_SB(THIS_QSP,(sbp=new_stringbuf()) );

	if( SB_SIZE(sbp) == 0 ){
		enlarge_buffer(sbp,strlen(pmpt)+4);
		// does this clear the string?
	}

	n =(strlen(SB_BUF(sbp))+strlen(pmpt)+4) ;
	if( SB_SIZE(sbp) < n ){
//#ifdef CAUTIOUS
//		if( n > LLEN ){
//			sprintf(ERROR_STRING,
//		"CAUTIOUS:  push_prompt:  Attempting to append prompt \"%s\" to previous prompt \"%s\"",
//				pmpt,SB_BUF(sbp));
//			advise(ERROR_STRING);
//			ERROR1("prompt overflow!?");
//		}
//#endif /* CAUTIOUS */

		// BUG?  Because we are now using string buffers,
		// we probably don't need this assertion.
		// However, if the prompt gets too big it is
		// probably a script error (macro recursion)
		// causing a prompt overflow...
		// So we might want to perform a check here,
		// but it should probably not be an assertion...

		//assert( n < LLEN );
		if( n >= LLEN ){
			sprintf(ERROR_STRING,
"push_prompt:  Attempting to append prompt \"%s\" to previous prompt:\n\"%s\"",
				pmpt,SB_BUF(sbp));
			advise(ERROR_STRING);
			advise("Probable script problem?");
		}
		enlarge_buffer(sbp,n);
	}

	if( (n=strlen(SB_BUF(sbp))) >= 2 ){
		n-=2;	/* assume prompt ends in "> " */
		SB_BUF(sbp)[n]=0;
		strcat(SB_BUF(sbp),"/");
	}

	strcat(SB_BUF(sbp),pmpt);
	strcat(SB_BUF(sbp),"> ");
}

/* fix the prompt when exiting a menu... */
/* BUG we should only bother with this when we're running interactive! */

static void pop_prompt(SINGLE_QSP_ARG_DECL)
{
	String_Buf *sbp;
	long n;

	sbp=QS_PROMPT_SB(THIS_QSP);
	n=strlen(SB_BUF(sbp));
	n--;
	while(n>=0 && QS_PROMPT_STR(THIS_QSP)[n] != '/' )
		n--;
	if( SB_BUF(sbp)[n]=='/' ){
		SB_BUF(sbp)[n]=0;
		strcat(SB_BUF(sbp),"> ");
	}
}

char *error_string(SINGLE_QSP_ARG_DECL)
{
	return THIS_QSP->qs_error_string;
}

char *message_string(SINGLE_QSP_ARG_DECL)
{
	return THIS_QSP->qs_msg_str;
}

#ifdef HAVE_LIBCURL
Curl_Info *qs_curl_info(SINGLE_QSP_ARG_DECL)
{
	return THIS_QSP->qs_curl_info;
}
#endif // HAVE_LIBCURL

Vec_Expr_Node * qs_top_node( SINGLE_QSP_ARG_DECL  )
{
	return THIS_QSP->qs_top_enp;
}

void set_top_node( QSP_ARG_DECL  Vec_Expr_Node *enp )
{
	THIS_QSP->qs_top_enp = enp;
}

const char * qs_filename( SINGLE_QSP_ARG_DECL )
{
	return QRY_FILENAME(CURR_QRY(THIS_QSP));
}

int qs_serial_func(SINGLE_QSP_ARG_DECL)
{
	return _QS_SERIAL(THIS_QSP);
}

String_Buf *qs_scratch_buffer(SINGLE_QSP_ARG_DECL)
{
	return THIS_QSP->qs_scratch;
}

void set_curr_string(QSP_ARG_DECL  const char *s)
{
	SET_QS_CURR_STRING(THIS_QSP,s);
}

const char *qs_curr_string(SINGLE_QSP_ARG_DECL)
{
	return _QS_CURR_STRING(THIS_QSP);
}

char *qs_expr_string(SINGLE_QSP_ARG_DECL)
{
	return THIS_QSP->_qs_expr_string;
}

/*
Input_Format_Spec *qs_ascii_input_format(SINGLE_QSP_ARG_DECL)
{
	return THIS_QSP->qs_dai_p->dai_input_fmt;
}
*/

int qs_level(SINGLE_QSP_ARG_DECL)
{
	return QS_LEVEL(THIS_QSP);
}

FILE *qs_msg_file(SINGLE_QSP_ARG_DECL)
{
	return QS_MSG_FILE(THIS_QSP);
}

