

typedef struct knox_state {
	int	ks_video[8];	/* which input is connected to each of 8 outputs */
	int	ks_audio[8];	/* which input is connected to each of 8 outputs */
} Knox_State;

typedef enum {
	KNOX_RECALL_CROSSPOINT,
	KNOX_STORE_CROSSPOINT,
	KNOX_SET_TIMER,
	KNOX_STOP_TIMER,
	KNOX_LAMP_TEST,
	KNOX_MAP_REPORT,
	KNOX_CONDENSE_REPORT,
	KNOX_TAKE_COMMAND,
	KNOX_SET_BOTH,
	KNOX_SET_VIDEO,
	KNOX_SET_AUDIO,
	KNOX_SET_DIFF,
	KNOX_SALVO_BOTH,
	KNOX_SALVO_VIDEO,
	KNOX_SALVO_AUDIO,
	KNOX_CONF_BOTH,
	KNOX_CONF_VIDEO,
	KNOX_CONF_AUDIO,
	KNOX_CMD_BOTH,
	KNOX_CMD_VIDEO,
	KNOX_CMD_AUDIO,
	N_KNOX_CMDS
} Knox_Cmd_Code;

typedef struct knox_cmd {
	Knox_Cmd_Code	kc_code;
	const char *	kc_str;
	const char *	kc_desc;
} Knox_Cmd;

extern Knox_Cmd knox_tbl[N_KNOX_CMDS];

