#include "quip_config.h"

#include "quip_prot.h"
#include "pic.h"

PIC_Cmd pic_tbl[N_PIC_CMDS] = {
{ PIC_RD_PGM,	"R",	"read program memory"			},
{ PIC_WR_PGM,	"W",	"read program memory"			},
{ PIC_RD_DATA,	"r",	"read data memory"			},
{ PIC_WR_DATA,	"w",	"read data memory"			},
{ PIC_RPT_VER,	"v",	"report firmware version"		},
{ PIC_SET_LED,	"l",	"set LED state"				},
{ PIC_GOTO,	"g",	"begin execution at new location"	},
{ PIC_ECHO,	"e",	"enable/disable command echo"		},
};

