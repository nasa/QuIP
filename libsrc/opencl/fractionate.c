#include <stdio.h>
#include <math.h>

#define MAX_RESOLUTION 16
#define RESOLUTION 16

struct frac {
	float num;
	float inum;
	int numerator;
	int denominator;
};

struct frac frac_tbl[MAX_RESOLUTION+1];

void init_frac_tbl(int n)
{
	// BUG validate n

	int i;

	for(i=0;i<n+1;i++){
		frac_tbl[i].num = i * (1.0/(float)n);
		frac_tbl[i].inum = 0;
		frac_tbl[i].numerator = i;
		frac_tbl[i].denominator = n;
		if( frac_tbl[i].numerator == n ){
			frac_tbl[i].inum = 1;
			frac_tbl[i].numerator = 0;
		}
//printf("%d\t%f\t%d / %d\n",i,frac_tbl[i].num,frac_tbl[i].numerator,frac_tbl[i].denominator);
		if( frac_tbl[i].numerator != 0 ){
			while( (frac_tbl[i].numerator & 1) == 0 ){
				frac_tbl[i].numerator /= 2;
				frac_tbl[i].denominator /= 2;
			}
		}
//printf("%d\t%5.3f\t%d / %d\n",i,frac_tbl[i].num,frac_tbl[i].numerator,frac_tbl[i].denominator);
	}
}

int main(int ac, char **av)
{
	float num;
	float inum;
	float rem;
	int i, i_min;
	float v_this, v_min;

	init_frac_tbl(RESOLUTION);

	while( scanf("%f",&num) == 1 ){
		inum = floor(num);
		rem = num - inum;
		// we could use an efficient algorithm to insert the remainder,
		// but why bother...
		v_min = 2;
		i_min = (-1);
		i=0;
//printf("num = %f, searching for min\n",num);
		do {
			v_this = fabs( frac_tbl[i].num - rem );
			if( v_this < v_min ){
				i_min = i;
				v_min = v_this;
			}
			i++;
		} while( v_min == v_this );
		printf("%g ~ %g + %d / %d  = %5.3f\n",num,inum,frac_tbl[i_min].numerator,
			frac_tbl[i_min].denominator,frac_tbl[i_min].num);
	}
}


