#include "quip_config.h"

char VersionId_luts_machdep[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include "cmaps.h"
#include "getbuf.h"

/*
 * This file is not part of the luts library.  It is a file of dummy routines
 * to be used for testing and to serve as an example when writing an
 * interface for a new display system.  The functions in this file are
 * those which must be implemented for a device to interface with
 * the high-level software.
 */

/*
 * Free the private resources used by a lutbuf.
 * General things (the name and the structure itself ) are freed by del_lb().
 */


/*
 * Initialize (allocate?) the resources needed by a lut buffer.
 * Called by make_lb().
 */

void ram_init_lb_data(lbp)
Lutbuf *lbp;
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

/*
 * Copy colormap data from the array cmap to the lutbuf's private storage.
 */

void ram_assign_lutbuf(lbp,cmap)
Lutbuf *lbp;
unsigned char cmap[N_COMPS][N_COLORS];
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

/*
 * Fill the array with colormap data from the lutbuf's private storage.
 */

void ram_read_lutbuf(data,lbp)
unsigned char data[N_COMPS][N_COLORS]; Lutbuf *lbp;
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

/*
 * Print the values for one lutbuf entry.
 */

void ram_show_lb_value(lbp,index)
Lutbuf *lbp; int index;
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

void ram_lut_init()
{
	Lut_Module *lmp;

	lmp=init_lut_module();
	if( lmp == NO_LUT_MODULE ) return;

	lmp->init_func = ram_init_lb_data;
	lmp->ass_func = ram_assign_lutbuf;
	lmp->read_func = ram_read_lutbuf;
	lmp->show_func = ram_show_lb_value;

	load_lut_module(lmp);
}

