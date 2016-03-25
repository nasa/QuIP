#include "quip_config.h"

#include <stdio.h>
#include <math.h>
#include "quip_prot.h"
//#include "function.h"

#ifdef FOOBAR
// BUG instead of a table, we should compute using erf!?

double   ztable[] = {
.5000, .5040, .5080, .5120, .5160, .5199, .5239, .5279, .5319, .5359,
.5398, .5438, .5478, .5517, .5557, .5596, .5636, .5675, .5714, .5753,
.5793, .5832, .5871, .5910, .5948, .5987, .6026, .6064, .6103, .6141,
.6179, .6217, .6255, .6293, .6331, .6368, .6406, .6443, .6480, .6517,
.6554, .6591, .6628, .6664, .6700, .6736, .6772, .6808, .6844, .6779,
.6915, .6950, .6985, .7019, .7054, .7088, .7123, .7157, .7190, .7224,
.7257, .7291, .7324, .7357, .7389, .7422, .7454, .7456, .7517, .7549,
.7580, .7611, .7642, .7673, .7704, .7734, .7764, .7794, .7823, .7852,
.7881, .7910, .7939, .7967, .7995, .8023, .8051, .8078, .8106, .8133,
.8159, .8186, .8212, .8238, .8264, .8289, .8315, .8340, .8365, .8389,
.8413, .8438, .8461, .8485, .8508, .8531, .8554, .8577, .8599, .8621,
.8643, .8665, .8686, .8708, .8729, .8749, .8770, .8790, .8810, .8830,
.8849, .8869, .8888, .8907, .8925, .8944, .8962, .8980, .8997, .9015,
.9032, .9049, .9066, .9082, .9099, .9115, .9131, .9147, .9162, .9177,
.9192, .9207, .9222, .9236, .9251, .9265, .9279, .9292, .9306, .9319, 
.9332, .9345, .9357, .9370, .9382, .9394, .9406, .9418, .9429, .9441,
.9452, .9463, .9474, .9484, .9495, .9505, .9515, .9525, .9535, .9545,
.9554, .9564, .9573, .9582, .9591, .9599, .9608, .9616, .9625, .9633,
.9641, .9649, .9656, .9664, .9671, .9678, .9686, .9693, .9699, .9706,
.9713, .9719, .9726, .9732, .9738, .9744, .9750, .9756, .9761, .9767,
.9773, .9778, .9783, .9788, .9793, .9798, .9803, .9808, .9812, .9817,
.9821, .9826, .9830, .9834, .9838, .9842, .9846, .9850, .9854, .9857,
.9861, .9864, .9868, .9871, .9875, .9878, .9881, .9884, .9887, .9890,
.9893, .9896, .9898, .9901, .9904, .9906, .9909, .9911, .9913, .9916,
.9918, .9920, .9922, .9925, .9927, .9929, .9931, .9932, .9934, .9936,
1.000 };
#define ZTSIZ	251
#define ZTBL_SCALE	100	// 100 bins is a step of 1 z...

// Given a probability, find the table entry that is closest.

double ptoz(double prob)
{
        double  factor = 1.0;
        int     z_idx = 0;
        if (prob < .5) {
                factor =  -factor;
                prob = 1.0 - prob;
        }
        while (ztable[z_idx] != 1.0 && ztable[z_idx] < prob) z_idx++;
	// Now the table points to an entry the is >= the requested p
        if (ztable[z_idx] == 1.0){
                if( prob == 1.0 && verbose  )
			NWARN("warning: infinite z score");
                else {
			sprintf (DEFAULT_ERROR_STRING,
			"%f is too high a probability\n", prob);
			NWARN(DEFAULT_ERROR_STRING);
		}
        }
        return (factor*z_idx/((double) ZTBL_SCALE));
}

double ztop(double zscore)
{
	int zi, zneg=0;
	double prob;

	if( zscore < 0 ){
		zneg=1;
		zscore*=(-1);
	}
	zscore*=ZTBL_SCALE;
	zscore+=.499;
	zi=(int)zscore;
	if( zi >= ZTSIZ ){
		if( verbose ){
			NWARN("zscore outside of table");
			sprintf(DEFAULT_ERROR_STRING,
				"zscore: %f\tzi: %d",(zscore-.499)/ZTBL_SCALE,zi);
			NADVISE(DEFAULT_ERROR_STRING);
		}
		zi=ZTSIZ-1;
	}
	prob=ztable[zi];
	if( zneg ) return(1.0-prob);
	else return(prob);
}

#endif // FOOBAR

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

