#include "quip_config.h"

char VersionId_knox_knox_tbl[] = QUIP_VERSION_STRING;

#include "knox.h"

Knox_Cmd knox_tbl[N_KNOX_CMDS] = {
{ KNOX_RECALL_CROSSPOINT,	"R",	"recall crosspoint"		},
{ KNOX_STORE_CROSSPOINT,	"S",	"store crosspoint"		},
{ KNOX_SET_TIMER,		"T",	"set timer"			},
{ KNOX_STOP_TIMER,		"N",	"stop timer"			},
{ KNOX_LAMP_TEST,		"T",	"lamp test"			},
{ KNOX_MAP_REPORT,		"D",	"map report"			},
{ KNOX_CONDENSE_REPORT,		"D0",	"condensed report"		},
{ KNOX_TAKE_COMMAND,		"EE",	"execute command set"		},
{ KNOX_SET_BOTH,		"B",	"set audio/video source"	},
{ KNOX_SET_VIDEO,		"V",	"set video source"		},
{ KNOX_SET_AUDIO,		"A",	"set audio source"		},
{ KNOX_SET_DIFF,		"B",	"set audio/video sources"	},
{ KNOX_SALVO_BOTH,		"X",	"set audio/video source"	},
{ KNOX_SALVO_VIDEO,		"Y",	"set video source"		},
{ KNOX_SALVO_AUDIO,		"Z",	"set audio source"		},
{ KNOX_CONF_BOTH,		"J",	"set audio/video source"	},
{ KNOX_CONF_VIDEO,		"K",	"set video source"		},
{ KNOX_CONF_AUDIO,		"L",	"set audio source"		},
{ KNOX_CMD_BOTH,		"E",	"set audio/video source"	},
{ KNOX_CMD_VIDEO,		"F",	"set video source"		},
{ KNOX_CMD_AUDIO,		"G",	"set audio source"		}
};

