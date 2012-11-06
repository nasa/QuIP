#include "quip_config.h"

char VersionId_fio_wsubs[] = QUIP_VERSION_STRING;

/*
 * Copyright (c) 1991 Michael Landy
 *
 * Disclaimer:  No guarantees of performance accompany this software,
 * nor is any responsibility assumed on the part of the authors.  All the
 * software has been tested extensively and every effort has been made to
 * insure its reliability.
 */

/*
 * wsubs.c - HIPS image header write header subroutines
 *
 * Michael Landy - 2/1/82
 * modified to use read/write - 4/26/82
 * modified to return #chars - msl - 9/21/90
 * modified for HIPS 2 - msl - 1/3/91
 */

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "fio_prot.h"
#include "hips2.h"
#include "perr.h"
#include "wsubs.h"

int wnocr(FILE *fp,const char *s)
{
	const char *t;
	int i;

	t = s;
	i = 0;
	while (*t != '\n' && *t != '\0') {
		putc(*t++,fp);
		i++;
	}
	putc('\n',fp);
	return(i+1);
}

int dfprintf(FILE *fp,int i,Filename fname)
{
	char s[30];
	int j;

	sprintf(s,"%6d\n",i);	/* jbm, %d to %6d to be able to edit header */
	j = strlen(s);
	if (fwrite(s,j,1,fp) != 1)
		return(perr(HE_HDRWRT,fname));
	return(j);
}
