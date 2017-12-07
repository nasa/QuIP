#include "quip_config.h"
#include "quip_prot.h"

#include <stdlib.h>		/* qsort() */

#include "polh_dev.h"

/* polhemus commands */

Ph_Cmd polh_cmds[N_PH_CMD_CODES] = {

/*                                                                              nargs		*/

{ "alignment",		"A", PH_ALIGNMENT, 	PH_SG,	PH_NEED_STAT, 9, ALIGN_REC,	},
{ "reset_alignment",	"R", PH_RESET_ALIGNMENT,PH_SET,	PH_NEED_STAT, 0, 0		},
{ "boresight",		"B", PH_BORESIGHT,	PH_SET,	PH_NEED_STAT, 0, 0		},
{ "ref_boresight",	"G", PH_REF_BORESIGHT,	PH_SG,	PH_NEED_STAT, 3, BORE_REC	},
{ "reset_boresight",	"b", PH_RESET_BORESIGHT,PH_SET,	PH_NEED_STAT, 0, 0		},
{ "xmtr_mount_angles",	"r", PH_XMTR_ANGLES,	PH_SG,	PH_NEED_XMTR, 3, XMTR_REC	},
{ "rcvr_angles",	"s", PH_RECV_ANGLES,	PH_SG,	PH_NEED_RECV, 3, RECV_REC	},
{ "att_filter",		"v", PH_ATT_FILTER,	PH_SG,	PH_NEED_NONE, 4, ATT_REC	},
{ "posn_filter",	"x", PH_POS_FILTER,	PH_SG,	PH_NEED_NONE, 4, POS_FTR_REC	},
#ifdef FOOBAR
{ "set_sync_internal",	"y0",PH_INTERNAL_SYNC,	PH_SET,	PH_NEED_NONE, 0, 0		},
{ "set_sync_external",	"y1",PH_EXTERNAL_SYNC,	PH_SET,	PH_NEED_NONE, 0, 0		},
{ "set_sync_software",	"y2",PH_SOFTWARE_SYNC,	PH_SET,	PH_NEED_NONE, 0, 0		},
{ "get_sync_mode",	"y", PH_SYNC_MODE,	PH_GET,	PH_NEED_NONE, 1, SYNC_REC	},
#endif
{ "sync_mode",		"y", PH_SYNC_MODE,	PH_SG,	PH_NEED_NONE, 1, SYNC_REC	},
{ "reinit_system",	"W", PH_REINIT_SYS,	PH_SET,	PH_NEED_NONE, 0, 0		},
{ "ang_envelope",	"Q", PH_ANGULAR_ENV,	PH_SG,	PH_NEED_STAT, 6, ANG_REC	},
{ "pos_envelope",	"V", PH_POSITIONAL_ENV, PH_SG,	PH_NEED_STAT, 6, POS_ENV_REC	},
{ "hemisphere",		"H", PH_HEMISPHERE,	PH_SG,	PH_NEED_STAT, 3, HEMI_REC	},
{ "start_continuous",	"C", PH_CONTINUOUS,	PH_SET,	PH_NEED_NONE, 0, 0		},
{ "stop_continuous",	"c", PH_NONCONTINUOUS,	PH_SET,	PH_NEED_NONE, 0, 0		},
{ "single_reading",	"P", PH_SINGLE_RECORD,	PH_SET,	PH_NEED_NONE, 0, 0		},
{ "units_inches",	"U", PH_INCHES_FMT,	PH_SET,	PH_NEED_NONE, 0, 0		},
{ "units_centimeters",	"u", PH_CM_FMT,		PH_SET,	PH_NEED_NONE, 0, 0		},
{ "active_station",	"l", PH_STATION,	PH_SG,	PH_NEED_STAT, 2, STAT_REC	},
{ "system_status",	"S", PH_STATUS,		PH_GET,	PH_NEED_NONE, 0, STATUS_REC	}

};


static int phc_cmp(const void *p1,const void *p2)
{
	const Ph_Cmd *pcp1,*pcp2;

	pcp1 = p1;
	pcp2 = p2;

	if( pcp1->pc_code > pcp2->pc_code ) return(1);
	else if( pcp1->pc_code < pcp2->pc_code ) return(-1);
	else return(0);
}

void _sort_table(SINGLE_QSP_ARG_DECL)
{
	int i;

	qsort(&polh_cmds[0],N_PH_CMD_CODES,sizeof(polh_cmds[0]),phc_cmp);

	/* now verify */

	for(i=0;i<N_PH_CMD_CODES;i++){
		if( polh_cmds[i].pc_code != i ){
			sprintf(ERROR_STRING,"Polhemus command table entry %d has code %d!?",
					i,polh_cmds[i].pc_code);
			error1(ERROR_STRING);
		}
	}
}


