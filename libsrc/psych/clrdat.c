#include "quip_config.h"

#include "stc.h"
#include "node.h"
#include "debug.h"
#include "list.h"

static void clear_one_class(QSP_ARG_DECL  Trial_Class *tc_p, void *arg )
{
	Summary_Data_Tbl *sdt_p;

	sdt_p = CLASS_SUMM_DTBL(tc_p);
	assert(sdt_p!=NULL);
	clear_summary_data(sdt_p);
}

void clrdat(SINGLE_QSP_ARG_DECL)	/* just clears data tables */
{
	iterate_over_classes(clear_one_class,NULL);
}

void update_summary(Summary_Data_Tbl *sdt_p,Staircase *st_p,int rsp)
{
	int val;

	assert( sdt_p != NULL );
fprintf(stderr,"update_summary:  data table at 0x%lx, size = %d\n",(long)sdt_p,SUMM_DTBL_SIZE(sdt_p));

	val = STAIR_VAL(st_p);
	assert( SUMM_DTBL_SIZE(sdt_p) > 0 );
	assert( val >= 0 && val < SUMM_DTBL_SIZE(sdt_p) );

	if( rsp == STAIR_CRCT_RSP(st_p) )
		SET_DATUM_NCORR( SUMM_DTBL_ENTRY(sdt_p,val),
			1 + DATUM_NCORR( SUMM_DTBL_ENTRY(sdt_p,val) ) );
	if( rsp != REDO && rsp != ABORT )
		SET_DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdt_p,val),
			1 + DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdt_p,val) ) );
}

void append_trial( Sequential_Data_Tbl *qdt_p, Staircase *st_p , int rsp )
{
	Node *np;
	Sequence_Datum *qd_p;

	qd_p = getbuf(sizeof(Sequence_Datum));

	SET_SEQ_DATUM_CLASS_IDX(qd_p, CLASS_INDEX( STAIR_CLASS(st_p) ) );
	SET_SEQ_DATUM_STAIR_IDX(qd_p, STAIR_INDEX(st_p) );
	SET_SEQ_DATUM_XVAL_IDX(qd_p, STAIR_VAL(st_p) );
	SET_SEQ_DATUM_RESPONSE(qd_p, rsp );
	SET_SEQ_DATUM_CRCT_RSP(qd_p, STAIR_CRCT_RSP(st_p) );

	np = mk_node(qd_p);
	addTail( SEQ_DTBL_LIST(qdt_p), np );
}

