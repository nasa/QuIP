#include "quip_config.h"

#include "stc.h"
#include "query_stack.h"

static void _lump(Trial_Class *,Trial_Class *);

COMMAND_FUNC( lump )
{
	Trial_Class *dst_tcp, *src_tcp;

	dst_tcp=index_class( QSP_ARG   (int) HOW_MANY("destination class") );
	src_tcp=index_class( QSP_ARG   (int) HOW_MANY("source class") );

	if( dst_tcp == NO_CLASS || src_tcp==NO_CLASS )
		return;

	_lump(dst_tcp,src_tcp);
}

static void _lump(Trial_Class *dst_tcp,Trial_Class *src_tcp)
{
	Data_Tbl *dtp_to;
	Data_Tbl *dtp_fr;
	int j;

	dtp_to=CLASS_DATA_TBL(dst_tcp);
	dtp_fr=CLASS_DATA_TBL(src_tcp);

	if( DTBL_SIZE(dtp_to) != DTBL_SIZE(dtp_fr) ){
		sprintf(DEFAULT_ERROR_STRING,
			"_lump:  data table sizes (%d,%d) do not match!?",
			DTBL_SIZE(dtp_to),DTBL_SIZE(dtp_fr));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	for(j=0;j<DTBL_SIZE(dtp_fr);j++){
		if( DATUM_NTOTAL(DTBL_ENTRY(dtp_fr,j)) > 0 ||
		DATUM_NTOTAL(DTBL_ENTRY(dtp_to,j)) > 0 ){
			DATUM_NTOTAL(DTBL_ENTRY(dtp_to,j)) +=
				DATUM_NTOTAL(DTBL_ENTRY(dtp_fr,j));
			DATUM_NCORR(DTBL_ENTRY(dtp_to,j)) +=
				DATUM_NCORR(DTBL_ENTRY(dtp_fr,j));
		}
	}
}

