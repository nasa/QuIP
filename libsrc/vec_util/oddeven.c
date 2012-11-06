#include "quip_config.h"

char VersionId_vec_util_oddeven[] = QUIP_VERSION_STRING;

#include <math.h>
#include "vec_util.h"
#include "data_obj.h"

void mkodd(Data_Obj *dp)
{
	int i,j;
	int n,n2;
	float val, *buf;

	if( (n=whchk(dp)) == (-1) ) return;

	n2=n/2;
	buf = (float *) dp->dt_data;

	/* zero the Nyquist freqs */

	for(j=0;j<n;j++) *(buf + j ) = 0.0;
	for(j=0;j<n;j++) *(buf + n*j ) = 0.0;

	for(i=1;i<=n2;i++){
		for(j=1;j<n;j++){
			if( i==n2 && j == n2 )
				*(buf+i*n+j) = 0.0;
			else {
				val = (*(buf+i*n+j));
				*(buf+(n-i)*n + (n-j) ) = ( - val );
			}
		}
	}
}

void mkeven(Data_Obj *dp)
{
	int i,j;
	int n,n2;
	float val, *buf;

	if( (n=whchk(dp)) == (-1) ) return;

	n2=n/2;
	buf = (float *) dp->dt_data;

	for(i=1;i<=n2;i++){
		for(j=1;j<n;j++){
			if( i==n2 && j == n2 )
				;
			else {
				val = (*(buf+i*n+j));
				*(buf+(n-i)*n + (n-j) ) = ( val );
			}
		}
	}
}

