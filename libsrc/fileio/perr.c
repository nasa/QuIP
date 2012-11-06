#include "quip_config.h"

char VersionId_fio_perr[] = QUIP_VERSION_STRING;

/*
 * Copyright (c) 1991 Michael Landy
 *
 * Disclaimer:  No guarantees of performance accompany this software,
 * nor is any responsibility assumed on the part of the authors.  All the
 * software has been tested extensively and every effort has been made to
 * insure its reliability.
 */

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_STDARG_H
#include <stdarg.h>
#endif

#ifdef FOOBAR
#ifdef SUN
#include <varargs.h>
#else /* ! SUN */
#include <stdarg.h>
#endif /* ! SUN  */
#endif /* FOOBAR */

#include "hips2.h"

#ifdef PC
#include <process.h>		/* exit() */
#endif /* PC */

extern struct h_errstruct h_errors[];
char badfmt[] = "invalid error format code %d";

int j_hipserrlev = HEL_ERROR;
int j_hipserrprt = HEL_INFORM;	/* default: print&return for all others */
int j_hipserrno;
char j_hipserr[200];

#define hipserrlev	j_hipserrlev
#define hipserrprt	j_hipserrprt
#define hipserrno	j_hipserrno
#define hipserr		j_hipserr


#ifndef SUN
int perr(int,...);
#endif

/* perr(errorcode,errorprintargs...) */

#ifdef SUN
int perr(va_alist)
va_dcl	
#else /* ! SUN */
int perr(int first, ...)
#endif /* ! SUN */
{
	va_list ap;
	int n,i1,i2,i3,i4;
	char *s1,*s2,*s3,*s4;

#ifdef SUN
	va_start(ap);
#else
	va_start(ap,first);
#endif /* ! SUN  */
	n = va_arg(ap,int);

	/*
	hipserrno = n;
	sprintf(hipserr,"%s: ",tell_progname());
	*/

	hipserr[0]=0;		/* clear string; errors are not necessarily fatal */

	if (n <=0 || n > MAXERR) {
		n = MAXERR+1;
		sprintf(hipserr+strlen(hipserr),h_errors[n-1].h_errstr,n);
	}
	else {
		switch(h_errors[n-1].h_errfmt) {
		case HEP_N:
			sprintf(hipserr+strlen(hipserr),h_errors[n-1].h_errstr);
			break;
		case HEP_D:
			i1 = va_arg(ap,int);
			sprintf(hipserr+strlen(hipserr),h_errors[n-1].h_errstr,
				i1);
			break;
		case HEP_S:
			s1 = va_arg(ap,char *);
			sprintf(hipserr+strlen(hipserr),h_errors[n-1].h_errstr,
				s1);
			break;
		case HEP_SD:
			s1 = va_arg(ap,char *);
			i1 = va_arg(ap,int);
			sprintf(hipserr+strlen(hipserr),h_errors[n-1].h_errstr,
				s1,i1);
			break;
		case HEP_DS:
			i1 = va_arg(ap,int);
			s1 = va_arg(ap,char *);
			sprintf(hipserr+strlen(hipserr),h_errors[n-1].h_errstr,
				i1,s1);
			break;
		case HEP_SS:
			s1 = va_arg(ap,char *);
			s2 = va_arg(ap,char *);
			sprintf(hipserr+strlen(hipserr),h_errors[n-1].h_errstr,
				s1,s2);
			break;
		case HEP_SDD:
			s1 = va_arg(ap,char *);
			i1 = va_arg(ap,int);
			i2 = va_arg(ap,int);
			sprintf(hipserr+strlen(hipserr),h_errors[n-1].h_errstr,
				s1,i1,i2);
			break;
		case HEP_SDS:
			s1 = va_arg(ap,char *);
			i1 = va_arg(ap,int);
			s2 = va_arg(ap,char *);
			sprintf(hipserr+strlen(hipserr),h_errors[n-1].h_errstr,
				s1,i1,s2);
			break;
		case HEP_SSS:
			s1 = va_arg(ap,char *);
			s2 = va_arg(ap,char *);
			s3 = va_arg(ap,char *);
			sprintf(hipserr+strlen(hipserr),h_errors[n-1].h_errstr,
				s1,s2,s3);
			break;
		case HEP_DDDD:
			i1 = va_arg(ap,int);
			i2 = va_arg(ap,int);
			i3 = va_arg(ap,int);
			i4 = va_arg(ap,int);
			sprintf(hipserr+strlen(hipserr),h_errors[n-1].h_errstr,
				i1,i2,i3,i4);
			break;
		case HEP_SDDDDS:
			s1 = va_arg(ap,char *);
			i1 = va_arg(ap,int);
			i2 = va_arg(ap,int);
			i3 = va_arg(ap,int);
			i4 = va_arg(ap,int);
			s2 = va_arg(ap,char *);
			sprintf(hipserr+strlen(hipserr),h_errors[n-1].h_errstr,
				s1,i1,i2,i3,i4,s2);
			break;
		case HEP_SSSS:
			s1 = va_arg(ap,char *);
			s2 = va_arg(ap,char *);
			s3 = va_arg(ap,char *);
			s4 = va_arg(ap,char *);
			sprintf(hipserr+strlen(hipserr),h_errors[n-1].h_errstr,
				s1,s2,s3,s4);
			break;
		default:
			sprintf(hipserr+strlen(hipserr),badfmt,
				h_errors[n-1].h_errfmt);
			break;
		}
	}
	va_end(ap);
	if (h_errors[n-1].h_errsev >= hipserrprt ||
	    h_errors[n-1].h_errsev >= hipserrlev){
		sprintf(DEFAULT_ERROR_STRING,"%s",hipserr);
		NWARN(DEFAULT_ERROR_STRING);
	}
	if (h_errors[n-1].h_errsev >= hipserrlev){
		advise("NOT exiting despite severe HIPS error");
		/* exit(h_errors[n-1].h_errsev); */
	}
	return(HIPS_ERROR);
}

