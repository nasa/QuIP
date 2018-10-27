#include "quip_config.h"

#include "vec_util.h"
#include "quip_prot.h"

static int base_size;
static int base_log=4;	// initialize to quiet compiler?


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

	atmp=(int)a;
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
		
	*px = (u_short)(( a2 & ~sigma ) ^ ( a32 & sigma ) ^ eta);
	*py = (u_short)(( a2 & sigma ) ^ ( a32 & ~sigma ) ^ eta);
/*
sprintf(ERROR_STRING,
"a = 0x%x, a2 = 0x%x, a32 = 0x%x, a2a = 0x%x, sigma = 0x%x, eta = 0x%x, x = 0x%x, y = 0x%x",
a,a2,a32,a2a,sigma,eta,*px,*py);
advise(ERROR_STRING);
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

void _mk_krast(QSP_ARG_DECL  Data_Obj *dp)
{
	u_long i;
	u_short *sp;
	int l;

	INSIST_RAM_OBJ(dp,mk_krast);

	if( OBJ_PREC(dp) != PREC_UIN ){
		sprintf(ERROR_STRING,"mk_krast:  object %s (%s) should have precision %s",
			OBJ_NAME(dp),OBJ_PREC_NAME(dp),NAME_FOR_PREC_CODE(PREC_UIN));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_COMPS(dp) != 2 ){
		sprintf(ERROR_STRING,"mk_krast:  object %s (%d) should have depth 2",
			OBJ_NAME(dp),OBJ_COMPS(dp));
		WARN(ERROR_STRING);
		return;
	}
	l = my_log2( OBJ_COLS(dp) ) ;
	if( 1<<l != (int) OBJ_COLS(dp) ){
		sprintf(ERROR_STRING,"mk_krast:  %s length (%d) is not a power of two",
			OBJ_NAME(dp),OBJ_COLS(dp));
		WARN(ERROR_STRING);
		l++;
	}

	if( l & 1 ){
		sprintf(ERROR_STRING,"mk_krast:  %s length (%d) is not the square of a power of two",
			OBJ_NAME(dp),OBJ_COLS(dp));
		WARN(ERROR_STRING);
		l++;
	}

	base_log = l / 2;
	base_size = 1 << l;

	sp=(u_short *)OBJ_DATA_PTR(dp);
	for(i=0;i<OBJ_COLS(dp);i++){
		getkpt(i,sp,sp+OBJ_COMP_INC(dp));
		sp += OBJ_PXL_INC(dp);
	}
}


