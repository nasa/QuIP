#include "quip_config.h"

char VersionId_vec_util_mkwheel[] = QUIP_VERSION_STRING;

#include <math.h>
#include <stdio.h>
#include "vec_util.h"
#include "data_obj.h"

int whchk(Data_Obj *dp)
{
	if( ! IS_IMAGE(dp) ) {
		NWARN("must be an image");
		return(-1);
	}
	if( dp->dt_rows != dp->dt_cols ){
		NWARN("image must be square");
		return(-1);
	}
	if( dp->dt_prec != PREC_SP ){
		NWARN("must be float precision");
		return(-1);
	}
	if( dp->dt_comps != 1 ){
		NWARN("must be have a single component");
		return(-1);
	}
	return(0);
}

void make_axle(Data_Obj *dp)
{
	float *buf;
	int n,n2;

	if( (n=whchk(dp)) == (-1) ) return;
	buf = (float *) dp->dt_data;
	n2=n/2;
	*(buf + n * n2 + n2 ) = 1.0;
}


void mkwheel(Data_Obj *dp,int nspokes,double arg0)
{
	int i,j;
	int n,n2;
	float x,y,arg, val, *buf;

	if( (n=whchk(dp)) == (-1) ) return;

	n2=n/2;
	buf = (float *) dp->dt_data;

	for(j=0;j<n;j++) *(buf + j ) = 0.0;
	for(j=0;j<n;j++) *(buf + n*j ) = 0.0;

	if( nspokes & 1 ){		/* odd # spokes */
		for(i=1;i<=n2;i++){
			for(j=1;j<n;j++){
				x=i-n2;
				y=j-n2;
				if( x==0 && y == 0 ) val=0.0;
				else {
					arg=atan2(y,x);
					arg*=nspokes;
					arg+=arg0;
					if( sin(arg) > 0.0 ) val=0.5;
					else val=(-0.5);
				}
				*(buf+i*n+j) = val;
				*(buf+(n-i)*n + (n-j) ) = ( - val );
			}
		}
	} else {			/* even # spokes */
		for(i=1;i<=n2;i++){
			for(j=1;j<n;j++){
				x=i-n2;
				y=j-n2;
				if( x==0 && y == 0 ) val=0.0;
				else {
					arg=atan2(y,x);
					arg*=nspokes;
					arg+=arg0;
					if( sin(arg) > 0.0 ) val=0.5;
					else val=(-0.5);
				}
				*(buf+i*n+j) = val;
				*(buf+(n-i)*n + (n-j) ) = val ;
			}
		}
	}
}

