#include "quip_config.h"

char VersionId_vec_util_krast[] = QUIP_VERSION_STRING;

#include "vec_util.h"

static int base_size;
static int base_log;


/* See koenderink & van doorn, proc ieee v67 Oct 1979 1465-1466.
 *
 * In their equations, the subscript i runs over the bits of the binary representation.
 */

static void getkpt( u_long a, u_short *px, u_short *py )
{
	int sigma,eta;
	int prev_sigma,prev_eta;
	int a2;		/*	a_i / 2		*/
	int a32;	/*	3 a_i / 2 	*/
	int a2a;
	int atmp;
	int atmp2;
	unsigned int bit;
	unsigned int new_bit;
	int i;

	atmp=a;
	bit=1;
	a2=a32=a2a=0;
	/* hard-coded for 256^2? */
	/* travesrse the bit-pairs of a, building up a32, a2, and a2a */
	for(i=0;i<base_log;i++){
		atmp2=atmp>>1;
		if( ( atmp2 ^ atmp ) & 1 ) a32 |= bit;
		if( atmp & 2 ){
			a2 |= bit;
			if( atmp & 1 ) a2a |= bit;
		}
		atmp>>=2;
		bit<<=1;
	}
	sigma=eta=0;

#define SPECIAL_ITERATION	0
	bit = 1 << (base_log-2);
	for(i=0;i<base_log-1;i++){
		prev_sigma = sigma >> 1;
		prev_eta = eta >> 1;
		if( i == SPECIAL_ITERATION ){	/* do this to make pattern close */
			/* new_bit = ( prev_sigma ^ a32 ^ bit ) & bit;*/
			new_bit = prev_sigma & bit;
			sigma |= new_bit;
			new_bit = (prev_eta ^ ~(a32>>1) ) & bit;
			eta |= new_bit;
		} else {
			new_bit = ( prev_sigma ^ ~(a32>>1) ) & bit;
			sigma |= new_bit;
			new_bit = ( prev_eta ^ (a2a>>1) ) & bit;
			eta |= new_bit;
		}
		bit >>= 1;
	}
		
	*px = ( a2 & ~sigma ) ^ ( a32 & sigma ) ^ eta;
	*py = ( a2 & sigma ) ^ ( a32 & ~sigma ) ^ eta;
/*
sprintf(error_string,
"a = 0x%x, a2 = 0x%x, a32 = 0x%x, a2a = 0x%x, sigma = 0x%x, eta = 0x%x, x = 0x%x, y = 0x%x",
a,a2,a32,a2a,sigma,eta,*px,*py);
advise(error_string);
*/

}

static int my_log2( u_long n )
{
	int l=0;

	do {
		if( n == 1 ) return(l);
		l++;
		n>>=1;
	} while(n);
	NERROR1("my_log2:  shouldn't happen");
	return(-1);
}

void mk_krast(QSP_ARG_DECL  Data_Obj *dp)
{
	u_long i;
	u_short *sp;
	int l;

	if( dp->dt_prec != PREC_UIN ){
		sprintf(error_string,"mk_krast:  object %s (%s) should have precision %s",
			dp->dt_name,name_for_prec(dp->dt_prec),name_for_prec(PREC_UIN));
		WARN(error_string);
		return;
	}
	if( dp->dt_comps != 2 ){
		sprintf(error_string,"mk_krast:  object %s (%d) should have depth 2",
			dp->dt_name,dp->dt_comps);
		WARN(error_string);
		return;
	}
	l = my_log2( dp->dt_cols ) ;
	if( 1<<l != (int) dp->dt_cols ){
		sprintf(error_string,"mk_krast:  %s length (%d) is not a power of two",
			dp->dt_name,dp->dt_cols);
		WARN(error_string);
		l++;
	}

	if( l & 1 ){
		sprintf(error_string,"mk_krast:  %s length (%d) is not the square of a power of two",
			dp->dt_name,dp->dt_cols);
		WARN(error_string);
		l++;
	}

	base_log = l / 2;
	base_size = 1 << l;

	sp=(u_short *)dp->dt_data;
	for(i=0;i<dp->dt_cols;i++){
		getkpt(i,sp,sp+dp->dt_cinc);
		sp += dp->dt_pinc;
	}
}


