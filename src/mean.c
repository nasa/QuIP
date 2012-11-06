char VersionId_psych_mean[] = "$RCSfile$ $Revision$ $Date$";

/* For large data sets, we get junky values for the stddev, maybe an overflow??? */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

static int chatty=0;

int main(ac,av)
char **av;
{
	int in;
	double sum, sos;
	float data;
	double n, ldata, std_err, mean, std_dev, var;

	if( ac > 1 ){
		if( ac==2 && !strcmp(av[1],"-v") ) chatty=1;
		else {
			fprintf(stderr,"usage:  %s [-v]\n",av[0]);
			exit(1);
		}
	}

	sum=sos=0.0;
	n=0.0;
	while( scanf("%f",&data)==1 ){
		ldata = data;
		sum+=ldata;
		sos+=ldata*ldata;
		n+=1.0;
	}
	if( n == 0.0 ){
		fprintf(stderr,"mean:  no input data\n");
		exit(1);
	}

	/* variance is the average squared deviation */
	/* compute from mean-square minus square of mean */

	mean = sum / n;
	var = sos - n * mean * mean;
	/* var /= (n-1); */		/* n-1 or n??? */
	var /= (n);		/* n-1 or n??? */
	std_dev = sqrt( var ) ;

	/* S.E.M. is std_dev/sqrt(n)? */

	std_err = std_dev / sqrt( n );

	in=n;
	if( chatty )
		printf("mean of %d samples:\t%g\t\tstd. error:  %g\tstd. deviation:  %g\n",
			in,mean,std_err,std_dev);
	else printf("%g\t%g\t%d\t%g\n",mean,std_err,in,std_dev);

	return(0);
}

