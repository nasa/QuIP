char VersionId_gps_fmtgps[] = "$RCSfile$ $Revision$ $Date$";
/* take a file captured from the GPS3 time code reader, output it as 10 bytes */

#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* exit */
#endif

#define TC_MARKER	0xf7	/* we get f7 when using vitc?? */



#define NEXT( c )			\
					\
	c = getchar();			\
	posn++;				\
	if( (c) == EOF ) exit(1);


int main(int ac, char **av)
{
	int c;
	int posn=(-1);
	int c1,c2,c3,c4;
	int dropped;

	/* synchronize to first marker */
	do {
top:
		if( dropped!=2 ){
			do {
				NEXT(c)
				if( c != TC_MARKER ){
					printf("0%-5o   %5d:    %2x\n",posn,posn,c);
				}
			} while( c != TC_MARKER );
		}
		dropped=0;

		printf("0%-5o   %5d:    %2x",posn,posn,c);

		NEXT(c1);
		NEXT(c2);
		NEXT(c3);
		NEXT(c4);

		if( c4 == 0xff ){	/* char dropped!? */
			dropped=1;
			printf("   %2x   %2x   %2x     ",c1,c2,c3);
		} else if( c3 == 0xff ){
			dropped=1;
			printf("   %2x   %2x          ",c1,c2);
		} else {
			printf("   %2x   %2x   %2x   %2x",c1,c2,c3,c4);
		}

		if( !dropped ){
			NEXT(c)

			if( c != 0xff ){
				printf("\n0%-5o   %5d:    %2x\n",posn,posn,c);
				goto top;
			}
		} else {
			c=0xff;
		}

		printf("      %2x",c);

		NEXT(c1)
		NEXT(c2)
		NEXT(c3)
		NEXT(c4)

		if( c4 == TC_MARKER ){	/* char dropped!? */
			dropped=2;
			printf("   %2x   %2x   %2x     ",c1,c2,c3);
			c=c4;
		} else {
			printf("   %2x   %2x   %2x   %2x",c1,c2,c3,c4);
		}

		if( dropped ){
			printf("     *\n");
		} else {
			printf("\n");
		}
	} while(1);




}

