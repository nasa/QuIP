#include "quip_config.h"

#include "stc.h"
#include "node.h"
#include "debug.h"

void clrdat(SINGLE_QSP_ARG_DECL)	/* just clears data tables */
{
	List *lp;
	Node *np;
	Trial_Class *tcp;
	Data_Tbl *dtp;
	int i;

	lp=class_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST ) return;

	np=lp->l_head;
	while(np!=NO_NODE){
		tcp = (Trial_Class *) np->n_data;
		dtp = CLASS_DATA_TBL(tcp);
		if( dtp == NULL ){
			sprintf(ERROR_STRING,
		"Stimulus class %s has null data table, initializing...",
				CLASS_NAME(tcp) );
			advise(ERROR_STRING);

			dtp = alloc_data_tbl(tcp,_nvals);
		}
fprintf(stderr,"clrdat:  clearing data table for class %s\n",CLASS_NAME(tcp));
		SET_DTBL_N(dtp,0);
		for(i=0;i<DTBL_SIZE(dtp);i++){
			SET_DATUM_NTOTAL(DTBL_ENTRY(dtp,i),0);
			SET_DATUM_NCORR(DTBL_ENTRY(dtp,i),0);
		}
		np=np->n_next;
	}
fprintf(stderr,"clrdat:  DONE\n");
}

void note_trial(Trial_Class *tcp,int val,int rsp,int crct)
{
	Data_Tbl *dtp;

	dtp = CLASS_DATA_TBL(tcp);

	assert( dtp != NULL );

	if( rsp == crct )
		SET_DATUM_NCORR( DTBL_ENTRY(dtp,val),
			1 + DATUM_NCORR( DTBL_ENTRY(dtp,val) ) );
	if( rsp != REDO && rsp != ABORT )
		SET_DATUM_NTOTAL( DTBL_ENTRY(dtp,val),
			1 + DATUM_NTOTAL( DTBL_ENTRY(dtp,val) ) );
}

