#include "quip_config.h"

char VersionId_newvec_os_check[] = QUIP_VERSION_STRING;

/* even if the cpu has the mmx functions, we can't use them unless the OS is 2.4 or higher */

#include <stdio.h>
#include "myerror.h"

int os_supports_mmx(void)
{
	FILE *fp;
	int vno,major,minor;

	fp=popen("uname -r","r");
	if( fp == NULL ){
		NWARN("Unable to open uname pipe to determine OS release");
		return(0);
	}
	if( fscanf(fp,"%d.%d.%d",&vno,&major,&minor) != 3 ){
		NWARN("error scanning release number");
		return(0);
	}
	if( vno < 2 ) return(0);
	if( vno == 2 && major < 3 ) return(0);

	/* 2.4 or higher is OK */
	return(1);
}

