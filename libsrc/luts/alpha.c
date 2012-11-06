#include "quip_config.h"

char VersionId_luts_alpha[] = QUIP_VERSION_STRING;

/*
 * alpha.c	alpha colormap stuff
 */

#include <stdio.h>
#include "items.h"
#include "cmaps.h"

void set_alpha( int index,int alpha )
{
	if( ALPHA_INDEX >= N_COMPS ){
		NWARN("Sorry, no alpha color map");
		return;
	}
#ifdef HAVE_X11
	CM_DATA(current_dpyp->c_cm_dp,ALPHA_INDEX,index) = alpha;	/* no linearization ! */
#endif
	update_if();
}


/* make an alpha color map which encodes a binary number */
/* written for eye tracker synchonization track */

#define N_INDEX_BITS	15

void index_alpha( int index, int lv, int hv )
{
	int color,bit;

	push_cm_state();
	CLR_CM_STATE(IMMEDIATE);
	bit=1;
	for(color=1;color<=N_INDEX_BITS;color++){
		if( index & bit )
			set_alpha(color,hv);
		else
			set_alpha(color,lv);
		bit <<= 1;
	}
	pop_cm_state();
	update_if();
}
	
void const_alpha(int value)
{
	int i;

	push_cm_state();
	CLR_CM_STATE(IMMEDIATE);
	for(i=0;i<256;i++)
		set_alpha(i,value);
	pop_cm_state();

	update_if();
}


