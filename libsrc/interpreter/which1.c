#include "quip_config.h"

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "query_stack.h"
#include "query_prot.h"
#include "warn.h"
#include "history.h"

#ifdef HAVE_HISTORY
static void check_preload(QSP_ARG_DECL  const char *prompt, int n, const char **choices)
{
	if( ! IS_COMPLETING(THIS_QSP) ) return;
	if( ! intractive(SINGLE_QSP_ARG) ) return;
	if( *prompt == 0 ) return;

	preload_history_list(prompt,n,choices);
}
#endif /* HAVE_HISTORY */

// prompt should already be pre-loaded??

int _which_one(QSP_ARG_DECL  const char *prompt, int n, const char** choices)
{
	int i;
	int nmatches=0;
	int lastmatch=(-1);	/* init to elim warning */
	const char *user_response;

#ifdef HAVE_HISTORY
	check_preload(QSP_ARG  prompt, n, choices);
#endif /* HAVE_HISTORY */

	user_response = nameof(prompt);

	for(i=0;i<n;i++){
		assert(choices[i]!=NULL);
		if( !strcmp( user_response, choices[i] ) ){
			return(i);
		}
	}
	
	/* if no exact match check for substring match */
	for(i=0;i<n;i++)
		if( is_a_substring( user_response, choices[i] ) ){
			lastmatch=i;
			nmatches++;
		}
	if( nmatches==1 ){
		sprintf(ERROR_STRING,"Unambiguous substring match of \"%s\" to \"%s\"",
			user_response,choices[lastmatch]);
		//advise(ERROR_STRING);
		WARN(ERROR_STRING);
		return(lastmatch);
	}
	else if( nmatches > 1 ){
		sprintf(ERROR_STRING,"ambiguous choice \"%s\"",user_response);
		WARN(ERROR_STRING);
		return(-1);
	}

	sprintf(ERROR_STRING,"invalid choice \"%s\"",user_response);
	WARN(ERROR_STRING);
	sprintf(ERROR_STRING,"valid selections for %s are:",prompt);
	advise(ERROR_STRING);
	for(i=0;i<n;i++){
		sprintf(ERROR_STRING,"\t%s",choices[i]);
		advise(ERROR_STRING);
	}
#ifdef HAVE_HISTORY
	if( intractive(SINGLE_QSP_ARG) ) rem_def(prompt,user_response) ;
#endif /* HAVE_HISTORY */

	return(-1);
}

