#include "quip_config.h"

#include "stc.h"

#define lump(dst_tcp,src_tcp) _lump(QSP_ARG  dst_tcp,src_tcp)

static void _lump(QSP_ARG_DECL  Trial_Class *dst_tcp,Trial_Class *src_tcp)
{
	Data_Tbl *dtp_to;
	Data_Tbl *dtp_fr;
	int j;

	dtp_to=CLASS_DATA_TBL(dst_tcp);
	dtp_fr=CLASS_DATA_TBL(src_tcp);

	if( DTBL_SIZE(dtp_to) != DTBL_SIZE(dtp_fr) ){
		sprintf(ERROR_STRING,
			"lump:  data table sizes (%d,%d) do not match!?",
			DTBL_SIZE(dtp_to),DTBL_SIZE(dtp_fr));
		warn(ERROR_STRING);
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


COMMAND_FUNC( do_lump )
{
	Trial_Class *dst_tcp, *src_tcp;

	dst_tcp=index_class( QSP_ARG   (int) HOW_MANY("destination class") );
	src_tcp=index_class( QSP_ARG   (int) HOW_MANY("source class") );

	if( dst_tcp == NO_CLASS || src_tcp==NO_CLASS )
		return;

	lump(dst_tcp,src_tcp);
}

