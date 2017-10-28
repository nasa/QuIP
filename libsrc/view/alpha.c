#include "quip_config.h"

/*
 * alpha.c	alpha colormap stuff
 */

#include <stdio.h>
#include "quip_prot.h"
#include "cmaps.h"

void _set_alpha(QSP_ARG_DECL  int index,int alpha )
{
	if( ALPHA_INDEX >= N_COMPS ){
		warn("Sorry, no alpha color map");
		return;
	}
#ifdef HAVE_X11
	CM_DATA( DPA_CMAP_OBJ(current_dpyp),ALPHA_INDEX,index) = alpha;	/* no linearization ! */
#endif
	update_if();
}


/* make an alpha color map which encodes a binary number */
/* written for eye tracker synchonization track */

#define N_INDEX_BITS	15

void _index_alpha(QSP_ARG_DECL  int index, int lv, int hv )
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
	
void _const_alpha(QSP_ARG_DECL  int value)
{
	int i;

	push_cm_state();
	CLR_CM_STATE(IMMEDIATE);
	for(i=0;i<256;i++)
		set_alpha(i,value);
	pop_cm_state();

	update_if();
}


