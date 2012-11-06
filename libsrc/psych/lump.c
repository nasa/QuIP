#include "quip_config.h"

char VersionId_psych_lump[] = QUIP_VERSION_STRING;

#include "stc.h"

static void _lump(Trial_Class *,Trial_Class *);

COMMAND_FUNC( lump )
{
	Trial_Class *dst_clp, *src_clp;

	dst_clp=index_class( QSP_ARG   (int) HOW_MANY("destination class") );
	src_clp=index_class( QSP_ARG   (int) HOW_MANY("source class") );

	if( dst_clp == NO_CLASS || src_clp==NO_CLASS )
		return;

	_lump(dst_clp,src_clp);
}

static void _lump(Trial_Class *dst_clp,Trial_Class *src_clp)
{
	Data_Tbl *dpto;
	Data_Tbl *dpfr;
	int j;

	dpto=dst_clp->cl_dtp;
	dpfr=src_clp->cl_dtp;
	for(j=0;j<_nvals;j++){
		if( dpfr->d_data[j].ntotal > 0 ||
		dpto->d_data[j].ntotal > 0 ){
			dpto->d_data[j].ntotal+=dpfr->d_data[j].ntotal;
			dpto->d_data[j].ncorr+=dpfr->d_data[j].ncorr;
		}
	}
}

