#include "quip_config.h"

#include "stc.h"
#include "node.h"
#include "debug.h"
#include "list.h"

void clrdat(SINGLE_QSP_ARG_DECL)	/* just clears data tables */
{
	List *lp;
	Node *np;
	Trial_Class *tcp;
	Summary_Data_Tbl *dtp;
	int i;

	lp=class_list();
	if( lp == NULL ) return;

	np=QLIST_HEAD(lp);
	while(np!=NULL){
		tcp = (Trial_Class *) np->n_data;
		dtp = CLASS_SUMM_DATA_TBL(tcp);
		if( dtp == NULL ){
			sprintf(ERROR_STRING,
		"Stimulus class %s has null data table, initializing...",
				CLASS_NAME(tcp) );
			advise(ERROR_STRING);

			dtp = alloc_data_tbl(tcp,_nvals);
		}
fprintf(stderr,"clrdat:  clearing data table for class %s\n",CLASS_NAME(tcp));
		SET_SUMM_DTBL_N(dtp,0);
		for(i=0;i<SUMM_DTBL_SIZE(dtp);i++){
			SET_DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp,i),0);
			SET_DATUM_NCORR(SUMM_DTBL_ENTRY(dtp,i),0);
		}
		np=np->n_next;
	}
fprintf(stderr,"clrdat:  DONE\n");
}

void note_trial(Summary_Data_Tbl *sdtp,int val,int rsp,int crct)
{
	assert( sdtp != NULL );

	if( rsp == crct )
		SET_DATUM_NCORR( SUMM_DTBL_ENTRY(sdtp,val),
			1 + DATUM_NCORR( SUMM_DTBL_ENTRY(sdtp,val) ) );
	if( rsp != REDO && rsp != ABORT )
		SET_DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdtp,val),
			1 + DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdtp,val) ) );
}

