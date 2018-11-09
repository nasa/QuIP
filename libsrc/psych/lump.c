#include "quip_config.h"

#include "stc.h"

#define lump(dst_tcp,src_tcp) _lump(QSP_ARG  dst_tcp,src_tcp)

static void _lump(QSP_ARG_DECL  Trial_Class *dst_tcp,Trial_Class *src_tcp)
{
	Summary_Data_Tbl *dtp_to;
	Summary_Data_Tbl *dtp_fr;
	int j;

	dtp_to=CLASS_SUMM_DTBL(dst_tcp);
	dtp_fr=CLASS_SUMM_DTBL(src_tcp);

	if( SUMM_DTBL_SIZE(dtp_to) != SUMM_DTBL_SIZE(dtp_fr) ){
		sprintf(ERROR_STRING,
			"lump:  data table sizes (%d,%d) do not match!?",
			SUMM_DTBL_SIZE(dtp_to),SUMM_DTBL_SIZE(dtp_fr));
		warn(ERROR_STRING);
		return;
	}

	for(j=0;j<SUMM_DTBL_SIZE(dtp_fr);j++){
		if( DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp_fr,j)) > 0 ||
		DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp_to,j)) > 0 ){
			DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp_to,j)) +=
				DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp_fr,j));
			DATUM_NCORR(SUMM_DTBL_ENTRY(dtp_to,j)) +=
				DATUM_NCORR(SUMM_DTBL_ENTRY(dtp_fr,j));
		}
	}
}


COMMAND_FUNC( do_lump )
{
	Trial_Class *dst_tcp, *src_tcp;

	dst_tcp = find_class_from_index( (int) how_many("destination class") );
	src_tcp = find_class_from_index( (int) how_many("source class") );

	if( dst_tcp == NULL || src_tcp == NULL )
		return;

	lump(dst_tcp,src_tcp);
}

