
#include "quip_config.h"

/*
 * this is a dummy implementation of the device-specific
 * routines needed for a lutbuffer interface.  Something
 * like what is here will be needed to emulate lut buffers
 * for a device with no dedicated storage for lut buffers
 */

#include <stdio.h>
#include "cmaps.h"
#include "getbuf.h"

void dispose_lb(Lutbuf *lbp)
{
}

void init_lb_data(Lutbuf *lbp)
{
	int i,j;
	unsigned char *ucp;

#ifdef CAUTIOUS
	if( lbp->lb_data != NULL ){
		warn("lutbuffer data already initialized");
		return;
	}
#endif
	lbp->lb_data = (void *)
		getbuf(N_COMPS*lbp->lb_len*sizeof(unsigned char));

	/* initialize to zeroes */

	ucp = lbp->lb_data;
	for(i=0;i<N_COMPS;i++){
		for(j=0;j<lbp->lb_len;j++){
			*ucp++ = 0;
		}
	}
}

void assign_lutbuf( Lutbuf *lbp, unsigned char cmap[N_COMPS][N_COLORS] )
{
	int i,j;
	unsigned char *ucp;

	if( lbp == NO_LUTBUF ){
		warn("assign_lutbuf(): lbp = NULL");
		return;
	}
#ifdef CAUTIOUS
	if( lbp->lb_data == NULL ){
		warn("lutbuffer data not initialized");
		init_lb_data(lbp);
	}
#endif
	ucp = lbp->lb_data;
	for(i=0;i<N_COMPS;i++){
		for(j=0;j<lbp->lb_len;j++){
			*ucp++ = cmap[i][lbp->lb_base+j];
		}
	}
}

void read_lutbuf( unsigned char data[N_COMPS][N_COLORS], Lutbuf *lbp )
{
	int i,j;
	unsigned char *ucp;

	if( lbp == NO_LUTBUF ){
		warn("read_lutbuf():  lbp = NULL");
		return;
	}
	if( lbp->lb_data == NULL ){
		warn("read_lutbuf():  no lutbuf data");
		return;
	}

	ucp = lbp->lb_data;

	for(i=0;i<N_COMPS;i++){
		for(j=0;j<lbp->lb_len;j++){
			data[i][lbp->lb_base+j] = *ucp++ ;
		}
	}
}

void show_lb_value( Lutbuf *lbp, int index )
{
	unsigned char *ucp;
	int i;

	if (index < 0 || index > 255) {
		warn ( "show_lb_value(): Illegal lut index");
		return;
	}

	if( lbp->lb_data == NULL ) {
		warn("lut buffer has no data");
		return;
	}

	ucp = lbp->lb_data;

	printf("%d:\t",index);

	ucp+=index;
	for(i=0;i<N_COMPS;i++){
		printf("  %3d",*ucp);
		ucp+=lbp->lb_len;
	}
	printf("\n");
}

void dump_lut( Lutbuf *lbp )
{
	if( lbp==NO_LUTBUF ) return;
	/* test program, no hw to dump to */
}

void lb_extra_info( Lutbuf *lbp )
{
}

void set_n_protect(int n)
{
}

