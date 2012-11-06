#include "quip_config.h"

char VersionId_interpreter_howmany[] = QUIP_VERSION_STRING;

#include <stdio.h>
#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif
#ifdef HAVE_MATH_H
#include <math.h>		/* floor() */
#endif

#include "query.h"
#include "nexpr.h"
#include "debug.h"


#ifdef HAVE_ROUND
#define round_func	round
#elif HAVE_RINT
#define round_func	rint
#else
#error 'Neither round nor rint is present, no rounding function.'
#endif


long how_many(QSP_ARG_DECL  const char *prompt)
{
        const char *s;
	char pline[LLEN];
        long n;
	double dn;

	if( prompt[0] != 0 ) sprintf(pline,PROMPT_FORMAT,prompt);
	else pline[0]=0;
        s=qword(QSP_ARG  pline);

	dn=pexpr(QSP_ARG  s);

	/* Check that the returned double is within the range of a long integer */

	if( dn > (double)((u_long) 0xffffffff) ){	/* largest unsigned positive int */
		/* what about signed int? BUG - jbm */
		sprintf(ERROR_STRING,"how_many:  dn = %g (expression was \"%s\"",dn,s);
		advise(ERROR_STRING);
		WARN("long integer overflow in how_many()");
		/* this case seems to happen after a -1 returned by
		 * pexpr has been cast to an unsigned long...
		 */

		return(0);
	}

	if( dn > 0x80000000 ){
		u_long *ulp;
		ulp = (u_long *) &n;
		*ulp = round_func(dn);
	} else {
		n=round_func(dn);
	}


	/* we used to check for rounding error here... */
	/* We still ought to check for gross error
	 * caused by going to a signed int...
	 */

	/* The idea is that we want to warn the user if
	 * a number like 4.5 is entered when an
	 * integer is needed...  But this test was giving
	 * false positives because some integers
	 * were not represented as pure integers in
	 * the double representation  -  how can that be?
	 *
	 * We need to know something about the machine precision
	 * and the rounding error...
	 */

#ifdef FOOBAR

	/* if( dn != n ){ */ 			/* } so showmatch works */
	if( resid != 0.0 ){
		long ln;
		/* maybe a fractional number was entered? */
		ln = (long) floor(dn+0.5);
		if( ((double) ln) == floor(dn+0.5) ){
			if( verbose ){
				sprintf(ERROR_STRING, "how_many:  resid = %g, double number %g rounded to long int %ld (==%ld?) !?",resid,dn,n,ln);
				advise(ERROR_STRING);
			}
			return(ln);
		}

		if( dn > 0 ){
			/* maybe this is an unsigned int? */
			u_long un;

			un = (u_long) dn;
			if( dn != un ){
				sprintf(ERROR_STRING,
	"how_many:  error converting double %g to long %ld (0x%lx)",
					dn,n,n);
				WARN(ERROR_STRING);
				return(0);
			}
			n = * ((long *)&un);
			return(n);
		}
		
		sprintf(ERROR_STRING,
	"how_many:  error converting double %g to long %ld (0x%lx)",
			dn,n,n);
		WARN(ERROR_STRING);
		return(0);
	}
#endif /* FOOBAR */

        return(n);
}

