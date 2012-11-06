#include "quip_config.h"

char VersionId_interpreter_askif[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include "query.h"

#define N_BOOL_CHOICES	4
static const char *bool_choices[N_BOOL_CHOICES]={"no","yes","false","true"};

#define YES	1
#define NO	0

int askif(QSP_ARG_DECL  const char *prompt)
{
	char pline[LLEN];
	int n;

	if( prompt[0] != 0 ) sprintf(pline,"%s? (y/n) ",prompt);
	else pline[0]=0;

	do {
		n = which_one2(QSP_ARG  pline,N_BOOL_CHOICES,bool_choices);
	} while( n < 0 && intractive( SINGLE_QSP_ARG ) );


	switch(n){
		case 0:				/* no */
		case 2:				/* false */
			return(0);
		case 1:				/* yes */
		case 3:				/* true */
			return(1);
	}
	return( -1 );
}

int confirm(QSP_ARG_DECL  const char *s)
{
	if( !intractive( SINGLE_QSP_ARG ) ) return(1);
	return(askif(QSP_ARG  s));
}


