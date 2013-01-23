#include "quip_config.h"

char VersionId_interpreter_chewtext[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include <stdlib.h>

#include "query.h"
#include "savestr.h"
#include "chewtext.h"
#include "debug.h"
#include "query.h"

void digest(QSP_ARG_DECL  const char *text)
{
	int level;
	/* int la_level; */

#ifdef OLD_LOOKAHEAD
	/* enable lookahead at this level and higher */
	push_lookahead(SINGLE_QSP_ARG);
	/* la_level= */ enable_lookahead(tell_qlevel(SINGLE_QSP_ARG));
#endif /* OLD_LOOKAHEAD */

	push_top_menu(SINGLE_QSP_ARG);		/* make sure at root menu */
	push_input_file(QSP_ARG  "-");
	PUSHTEXT(text);			/* or fullpush? */
	level=tell_qlevel(SINGLE_QSP_ARG);

	do {
		do_cmd(SINGLE_QSP_ARG);

		/* The command may have disabled lookahead;
		 * We need to lookahead here to make sure
		 * the end of the text is properly detected.
		 */

		lookahead(SINGLE_QSP_ARG);

	} while( tell_qlevel(SINGLE_QSP_ARG) >= level );

	popcmd(SINGLE_QSP_ARG);			/* back to where we were (undo push_top_menu) */
#ifdef OLD_LOOKAHEAD
	pop_lookahead(SINGLE_QSP_ARG);
	/* enable_lookahead(la_level); */
#endif /* OLD_LOOKAHEAD */

}

void swallow(QSP_ARG_DECL  const char *text)
{
	CHEWING=1;

	/* set_busy_cursor(); */

	digest(QSP_ARG  text);

	if( CHEW_LIST != NO_LIST ){
		Node *np;

		while( (np=remHead(CHEW_LIST)) != NO_NODE ){
			digest( QSP_ARG  (char*) np->n_data );
			rls_str((char *)np->n_data);
			rls_node(np);
		}
	}
	/*
	else {
WARN("no chew_list");
	}
	*/

	/* set_std_cursor(); */

	CHEWING=0;
}

void chew_text(QSP_ARG_DECL  const char *text)
{
	if( text == NULL ) return;
#ifdef DEBUG
//if( debug ){
//sprintf(ERROR_STRING,"chewing \"%s\"",text);
//advise(ERROR_STRING);
//}
#endif /* DEBUG */

	if( CHEWING ){
		Node *np;

		np = mk_node( (void *) savestr(text) );
		if( CHEW_LIST == NO_LIST ){
			CHEW_LIST = new_list();
#ifdef CAUTIOUS
			if( CHEW_LIST==NO_LIST )
				ERROR1("couldn't make chew list");
#endif /* CAUTIOUS */
		}
		// Is this thread-safe?
		addTail(CHEW_LIST,np);
	} else swallow(QSP_ARG  text);
}

