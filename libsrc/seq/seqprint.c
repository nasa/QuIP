/** seqprint.c	routines to print out sequence information */

#include "quip_config.h"

char VersionId_seq_seqprint[] = QUIP_VERSION_STRING;


#include <stdio.h>

#include "data_obj.h"
#include "seq.h"
#include "items.h"

void pseq(Seq *seqptr)		/** print out a sequence */
{
	sprintf(msg_str,"%s: ",seqptr->seq_name);
	prt_msg_frag(msg_str);
	pfull(seqptr);
	prt_msg("");
}

void pframe( Seq *seqptr)
{
	Item *ip;

	ip = (Item *)seqptr->seq_data;
	sprintf(msg_str,"%d * %s",seqptr->seq_count,ip->item_name);
	prt_msg_frag(msg_str);
}

void pfull( Seq *seqptr )
{
	if( seqptr->seq_first == NO_SEQ && seqptr->seq_next==NO_SEQ )	/* frame */
		pframe(seqptr);
	else {						/* concatentation */
		if( seqptr->seq_count == -1 ){	/* reversal */
			prt_msg_frag("reverse( ");
			pone(seqptr->seq_first);
			prt_msg_frag(" )");
			return;
		}
		if( seqptr->seq_count != 1 ){
			sprintf(msg_str,"%d ( ",seqptr->seq_count);
			prt_msg_frag(msg_str);
		}
		pone(seqptr->seq_first);
		if( seqptr->seq_next != NO_SEQ ){
			prt_msg_frag(" + ");
			pone(seqptr->seq_next);
		}
		if( seqptr->seq_count != 1 )
			prt_msg_frag(" ) ");
	}
}

void pone( Seq *seqptr )
{
	if( seqptr==NO_SEQ ){ /* end of the chain */
		/* warn("pone:  null sequence"); */
		return;
	}
	if( seqptr->seq_name!=NULL ){	/* frames don't have seqnames ... */
		sprintf(msg_str,"%s",seqptr->seq_name);
		prt_msg_frag(msg_str);
	} else if( seqptr->seq_first == NO_SEQ && seqptr->seq_next==NO_SEQ ) /* frame */
		pframe(seqptr);
	else {						/* concatentation */
		if( seqptr->seq_count == -1 ){
			prt_msg_frag("reverse( ");
			pone(seqptr->seq_first);
			prt_msg_frag(" )");
			return;
		}
		if( seqptr->seq_count != 1 ){
			sprintf(msg_str,"%d ( ",seqptr->seq_count);
			prt_msg_frag(msg_str);
		}
		pone(seqptr->seq_first);
		if( seqptr->seq_next != NO_SEQ ){
			prt_msg_frag(" + ");
			pone(seqptr->seq_next);
		}
		if( seqptr->seq_count != 1 )
			prt_msg_frag(" ) ");
	}
}

