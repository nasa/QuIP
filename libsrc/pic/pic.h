
#include "typedefs.h"		/* u_long */
#include "query.h"

typedef struct pic_state {
	int	ps_flags;
	u_long	ps_led[2];	/* what the led registers are holding */
} PIC_State;

typedef enum {
	PIC_RD_PGM,
	PIC_WR_PGM,
	PIC_RD_DATA,
	PIC_WR_DATA,
	PIC_RPT_VER,
	PIC_SET_LED,
	PIC_GOTO,
	PIC_ECHO,
	N_PIC_CMDS
} PIC_Cmd_Code;

typedef struct pic_cmd {
	PIC_Cmd_Code	pc_code;
	char *		pc_str;
	char *		pc_desc;
} PIC_Cmd;

extern PIC_Cmd pic_tbl[N_PIC_CMDS];

/* pic.c */

/* picmenu moved to submenus.h */

extern u_long pic_debug;

