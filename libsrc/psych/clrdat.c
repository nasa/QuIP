#include "quip_config.h"

char VersionId_psych_clrdat[] = QUIP_VERSION_STRING;

#include "stc.h"
#include "node.h"
#include "debug.h"

void clrdat(SINGLE_QSP_ARG_DECL)	/* just clears data tables */
{
	List *lp;
	Node *np;
	Trial_Class *clp;
	int val;

	lp=class_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST ) return;

	np=lp->l_head;
	while(np!=NO_NODE){
		clp = (Trial_Class *) np->n_data;
		clp->cl_dtp->d_npts=0;
		for(val=0;val<MAXVALS;val++)
			clp->cl_dtp->d_data[val].ntotal=
			clp->cl_dtp->d_data[val].ncorr=0;
		np=np->n_next;
	}
}

void note_trial(Trial_Class *clp,int val,int rsp,int crct)
{
	if( rsp == crct )
		clp->cl_dtp->d_data[val].ncorr ++;
	if( rsp != REDO && rsp != ABORT )
		clp->cl_dtp->d_data[val].ntotal ++;
}

