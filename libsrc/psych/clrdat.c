#include "quip_config.h"

#include "stc.h"
#include "node.h"
#include "debug.h"
#include "list.h"

static void clear_one_class(QSP_ARG_DECL  Trial_Class *tc_p, void *arg )
{
	Summary_Data_Tbl *sdt_p;

	sdt_p = CLASS_SUMM_DTBL(tc_p);
	if( sdt_p == NULL ){
		sprintf(ERROR_STRING,
	"Stimulus class %s has null data table, initializing...",
			CLASS_NAME(tc_p) );
		advise(ERROR_STRING);

		sdt_p = new_summary_data_tbl(CLASS_XVAL_OBJ(tc_p));
		SET_SUMM_DTBL_CLASS(sdt_p,tc_p);
	} else {
		clear_summary_data(sdt_p);
	}
}

void clrdat(SINGLE_QSP_ARG_DECL)	/* just clears data tables */
{
	iterate_over_classes(clear_one_class,NULL);
}

void note_trial(Summary_Data_Tbl *sdt_p,int val,int rsp,int crct)
{
	assert( sdt_p != NULL );

	if( rsp == crct )
		SET_DATUM_NCORR( SUMM_DTBL_ENTRY(sdt_p,val),
			1 + DATUM_NCORR( SUMM_DTBL_ENTRY(sdt_p,val) ) );
	if( rsp != REDO && rsp != ABORT )
		SET_DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdt_p,val),
			1 + DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdt_p,val) ) );
}

