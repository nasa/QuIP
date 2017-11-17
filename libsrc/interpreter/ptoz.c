#include "quip_config.h"

#include <stdio.h>
#include <math.h>
#include "quip_prot.h"
#include "function.h"

static double sqrt_of_two=0.0;

double ztop(double zscore)
{
	double p;
	
	if( sqrt_of_two == 0.0 )
		sqrt_of_two = sqrt(2);

	p = (1+erf(zscore/sqrt_of_two))/2.0;
	return p;
}

// p = (1+erf(z/sqrt(2)))/2
// 2p = 1 + erf(z/sqrt(2))
// 2p-1 = erf(z/sqrt(2))
// erfinv(2p-1) = z / sqrt(2)
// z = sqrt(2) * erfinv(2p-1)

double ptoz(double p)
{
	if( sqrt_of_two == 0.0 )
		sqrt_of_two = sqrt(2);

	return sqrt_of_two * erfinv( 2 * p - 1 ); 
}

