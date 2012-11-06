#include "quip_config.h"

char VersionId_vectree_mlabmisc[] = QUIP_VERSION_STRING;

#include "mlab.h"

void mc_save()
{
	advise("Sorry, save not implemented yet");
}

void mc_lookfor()
{
	advise("Sorry, lookfor not implemented yet");
}

void mc_help()
{
	advise("Sorry, help not implemented yet");
}

void mc_who()
{
	prt_msg("\nYour variables are:\n");

	list_dobjs();
}

void mc_whos()
{
	List *lp;
	Node *np;
	Data_Obj *dp;
	long tot_elts=0,tot_bytes=0;


	lp =  dobj_list();
	if( lp == NO_LIST ){
		advise("No variables in existence");
		return;
	}
	lp = alpha_sort(lp);

	prt_msg("\n              Name      Size      Elements    Bytes   Density    Complex\n");

	np=lp->l_head;
	while(np!=NO_NODE){
		long ne,nb;

		dp = np->n_data;

		sprintf(msg_str,"%16s",dp->dt_name);
		prt_msg_frag(msg_str);

		sprintf(msg_str,"   %5ld by %-5ld",dp->dt_rows,dp->dt_cols);
		prt_msg_frag(msg_str);

		sprintf(msg_str,"   %5ld",ne=(dp->dt_rows*dp->dt_cols*dp->dt_tdim));
		prt_msg_frag(msg_str);

		sprintf(msg_str,"   %5ld",nb=(ne*siztbl[ MACHINE_PREC(dp) ]) );
		prt_msg_frag(msg_str);

		prt_msg_frag("       Full");

		if( IS_COMPLEX(dp) )
			prt_msg("      Yes");
		else
			prt_msg("      No");


		np=np->n_next;

		tot_elts += ne;
		tot_bytes += nb;
	}

	sprintf(msg_str,"\nGrand total is %ld elements using %ld bytes\n",tot_elts,tot_bytes);
	prt_msg(msg_str);

	dellist(lp);
}


