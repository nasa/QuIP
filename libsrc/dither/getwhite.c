#include "quip_config.h"

char VersionId_dither_getwhite[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "debug.h"
#include "cie.h"
#include "query.h"
#include "ctone.h"

#define W_DIR	"./"

#define NAMELEN	80

float _white[3];

void rmnl(char *s)		/* remove a final newline */
{
	while( *s && *s != '\n' ) s++;
	*s=0;
}

int rwhite(char *name)
{
	FILE *fp;

	fp=try_open(DEFAULT_QSP_ARG  name,"r");
	if( !fp ) return(0);
	else {
		fprintf(stderr,"reading white values from %s\n",name);
		if( fscanf(fp,"%f %f %f",&_white[0],&_white[1],&_white[2])
			!= 3 ) NERROR1("bad white file");
	}
	return(1);
}

COMMAND_FUNC( getwhite )		/** initialze white value (from file) */
{
	_white[0]=HOW_MANY("red component");
	_white[1]=HOW_MANY("green component");
	_white[2]=HOW_MANY("blue component");
	know_white=1;
}

