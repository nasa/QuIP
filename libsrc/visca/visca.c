
#include "quip_config.h"

#ifdef VISCA_THREADS
#include <pthread.h>
#endif /* VISCA_THREADS */

#ifdef HAVE_STDLIB_H
#include <stdlib.h>     /* malloc() */
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>     /* usleep() */
#endif

#include "quip_prot.h"
#include "item_type.h"
#include "query_bits.h"	// LLEN - get rid of this! BUG
#include "fileck.h"	// path_exists()
#include "quip_menu.h"
#include "serial.h"
#include "ttyctl.h"
#include "visca.h"
#include "cam_params.h"	/* has the defines for the parameters */

#define INQ_RESULT_NAME	"inquiry_result"
static Visca_Cam *the_vcam_p=NULL;
#define vparam_p	the_vcam_p->vcam_param_p

#define hex_digit_to_ascii(dst,val)	_hex_digit_to_ascii(QSP_ARG  dst, val)
#define set_pan_speed(pkt,spd)	_set_pan_speed(QSP_ARG  pkt,spd)
#define set_tilt_speed(pkt,spd)	_set_tilt_speed(QSP_ARG  pkt,spd)

#define NO_VISCA_MSG(p,v)									\
												\
	sprintf(ERROR_STRING,"Sorry, no VISCA support in this build, can't set %s to %s.",p,v);	\
	WARN(ERROR_STRING);

#ifdef FOOBAR
#ifdef VISCA_THREADS
static void *camera_request_server(void *);
static void queue_visca_command( Visca_Queued_Cmd *vqcp );
static void queue_visca_inquiry( Visca_Inq_Def *vidp );
static void init_server_thread(Visca_Cam *vcam_p);
static int async_reqs=0;
#endif /* VISCA_THREADS */
#endif // FOOBAR


Visca_Params evi30_params = { 

		MIN_ZOOM_SPEED_EVI30,
		MAX_ZOOM_SPEED_EVI30,

		MIN_ZOOM_OPT_POS_EVI30,
		MAX_ZOOM_OPT_POS_EVI30,

		MIN_FOCUS_POSN_EVI30,
		MAX_FOCUS_POSN_EVI30,

		MIN_SHUTR_EVI30,
		MAX_SHUTR_EVI30,

		MIN_IRIS_POSN_EVI30,
		MAX_IRIS_POSN_EVI30,
		
		MIN_GAIN_POSN_EVI30,
		MAX_GAIN_POSN_EVI30,
		
		MIN_MEM_EVI30,
		MAX_MEM_EVI30,
	
		MIN_PAN_SPEED_EVI30,
		MAX_PAN_SPEED_EVI30,

		MIN_TILT_SPEED_EVI30,
		MAX_TILT_SPEED_EVI30,

		MIN_PAN_POSN_EVI30,
		MAX_PAN_POSN_EVI30,

		MIN_TILT_POSN_EVI30,
		MAX_TILT_POSN_EVI30,
	
		MIN_PAN_LMT_EVI30,
		MAX_PAN_LMT_EVI30,

		MIN_TILT_LMT_EVI30,
		MAX_TILT_LMT_EVI30,

		0, 0,	// this camera can't be flipped
};


Visca_Params evi100_params = {

		MIN_ZOOM_SPEED_EVI100,
		MAX_ZOOM_SPEED_EVI100,

		MIN_ZOOM_OPT_POS_EVI100,
		MAX_ZOOM_OPT_POS_EVI100,

		MIN_FOCUS_POSN_EVI100,
		MAX_FOCUS_POSN_EVI100,

		MIN_SHUTR_EVI100,
		MAX_SHUTR_EVI100,
		
		MIN_IRIS_POSN_EVI100,
		MAX_IRIS_POSN_EVI100,
	
		MIN_GAIN_POSN_EVI100,
		MAX_GAIN_POSN_EVI100,
	
		MIN_MEM_EVI100,
		MAX_MEM_EVI100,
	
		MIN_PAN_SPEED_EVI100,
		MAX_PAN_SPEED_EVI100,

		MIN_TILT_SPEED_EVI100,
		MAX_TILT_SPEED_EVI100,

		MIN_PAN_POSN_EVI100,
		MAX_PAN_POSN_EVI100,

		MIN_TILT_POSN_EVI100,
		MAX_TILT_POSN_EVI100,
	
		MIN_PAN_LMT_EVI100,
		MAX_PAN_LMT_EVI100,

		MIN_TILT_LMT_EVI100,
		MAX_TILT_LMT_EVI100,

		0, 0,	// this camera can't be flipped
};

Visca_Params evi70_params = { 

		MIN_ZOOM_SPEED_EVI70,
		MAX_ZOOM_SPEED_EVI70,

		MIN_ZOOM_OPT_POS_EVI70,
		MAX_ZOOM_OPT_POS_EVI70,

		MIN_FOCUS_POSN_EVI70,
		MAX_FOCUS_POSN_EVI70,

		MIN_SHUTR_EVI70,
		MAX_SHUTR_EVI70,
		
		MIN_IRIS_POSN_EVI70,
		MAX_IRIS_POSN_EVI70,
	
		MIN_GAIN_POSN_EVI70,
		MAX_GAIN_POSN_EVI70,
	
		MIN_MEM_EVI70,
		MAX_MEM_EVI70,
	
		MIN_PAN_SPEED_EVI70,
		MAX_PAN_SPEED_EVI70,

		MIN_TILT_SPEED_EVI70,
		MAX_TILT_SPEED_EVI70,

		MIN_PAN_POSN_EVI70,
		MAX_PAN_POSN_EVI70,

		MIN_TILT_POSN_EVI70,
		MAX_TILT_POSN_EVI70,
	
		MIN_PAN_LMT_EVI70,
		MAX_PAN_LMT_EVI70,

		MIN_TILT_LMT_EVI70,
		MAX_TILT_LMT_EVI70,
		
		// flipped limits
		-(MAX_TILT_POSN_EVI70),
		-(MIN_TILT_POSN_EVI70),
	
};

/* evi-d70 commands seem to include the evi-d100 commands but have a few extras:
 */

static Visca_Cmd_Def vcd_evi70_tbl[]={
{ "night_pwr",		"direct",	"8101044100000000ff",	PWR_ARG	},
{ "zoom",		"combine",	"8101043600ff",		NO_ARGS },
{ "zoom",		"separate",	"8101043601ff",		NO_ARGS },
{ "zoom",		"dig_zm_stop",	"8101040600ff",		NO_ARGS },
{ "zoom",		"dig_in_rate",	"8101040620ff",		DIG_ZOOM_SPEED },
{ "zoom",		"dig_out_rate",	"8101040630ff",		DIG_ZOOM_SPEED },
{ "zoom",		"x1_max",	"8101040610ff",		NO_ARGS },
{ "zoom_focus",		"direct",	"810104470000000000000000ff",	ZOOM_FOCUS_ARG },
{ "dzoom",		"on",		"8101040602ff",		NO_ARGS },
{ "dzoom",		"off",		"8101040603ff",		NO_ARGS },
{ "dzoom",		"combine",	"8101043600ff",		NO_ARGS },
{ "dzoom",		"separate",	"8101043601ff",		NO_ARGS },
{ "dzoom",		"stop",		"8101040600ff",		NO_ARGS },
{ "dzoom",		"tele",		"8101040620ff",		TELE_ARG },
{ "dzoom",		"wide",		"8101040630ff",		TELE_ARG },
{ "dzoom",		"x1Max",	"8101040610ff",		TELE_ARG },
{ "dzoom",		"direct",	"8101044600000000ff",	DZOOM_ARG },
{ "ir_mode",		"on",		"8101040102ff",		NO_ARGS },
{ "ir_mode",		"off",		"8101040103ff",		NO_ARGS },
{ "ir_mode",		"auto_on",	"8101045102ff",		NO_ARGS },
{ "ir_mode",		"auto_off",	"8101045103ff",		NO_ARGS },
};

#define N_EVI70_CMDS 	(sizeof(vcd_evi70_tbl)/sizeof(Visca_Cmd_Def))

static Visca_Cmd_Def vcd_evi100_tbl[]={
{ "auto_pwr",		"direct",	"8101044000000000ff",	PWR_ARG	},

{ "zoom",		"dig_direct",	"8101044600000000ff",	ZOOM_DIG_ARG },
{ "zoom",		"dig_zm_on",	"8101040602ff",		NO_ARGS },
{ "zoom",		"dig_zm_off",	"8101040603ff",		NO_ARGS },

{ "focus",		"far_rate",	"8101040820ff",		FOCUS_SPEED },
{ "focus",		"near_rate",	"8101040830ff",		FOCUS_SPEED },
{ "focus",		"one_psh_trig",	"8101041801ff",		NO_ARGS	},
{ "focus",		"infinity",	"8101041802ff",		NO_ARGS	},
{ "focus",		"af_sens_hgh",	"8101045802ff",		NO_ARGS	},
{ "focus",		"af_sens_low",	"8101045803ff",		NO_ARGS	},
{ "focus",		"near_limit",	"8101042800000000ff",	FOCUS_NEAR_LMT_ARG },
	
{ "white_bal",		"auto_trace_wb","8101043504ff",		NO_ARGS	},	
{ "white_bal",		"manual",	"8101043505ff",		NO_ARGS	},

{ "rgain",		"reset",	"8101040300ff",		NO_ARGS },
{ "rgain",		"up",		"8101040302ff",		NO_ARGS },
{ "rgain",		"down",		"8101040303ff",		NO_ARGS },
{ "rgain",		"direct",	"8101044300000000ff",	RGAIN_ARG },

{ "bgain",		"reset",	"8101040400ff",		NO_ARGS },
{ "bgain",		"up",		"8101040402ff",		NO_ARGS },
{ "bgain",		"down",		"8101040403ff",		NO_ARGS },
{ "bgain",		"direct",	"8101044400000000ff",	BGAIN_ARG },

{ "exposure",		"gain_priority", "810104390cff",	NO_ARGS	},
{ "exposure",		"shutter_auto", "810104391aff",		NO_ARGS	},
{ "exposure",		"iris_auto",	"810104392bff",		NO_ARGS	},
{ "exposure",		"gain_auto",	"810104393cff",		NO_ARGS	},

{ "slow_shutter",	"auto",		"8101045a02ff",		NO_ARGS	},
{ "slow_shutter",	"manual",	"8101045a03ff",		NO_ARGS	},

{ "bright",		"direct",	"8101044d00000000ff",	BRIGHT_ARG },

{ "exp_compensation",	"on",		"8101043e02ff",		NO_ARGS },
{ "exp_compensation",	"off",		"8101043e03ff",		NO_ARGS },
{ "exp_compensation",	"reset",	"8101040e00ff",		NO_ARGS },
{ "exp_compensation",	"up",		"8101040e02ff",		NO_ARGS },
{ "exp_compensation",	"down",		"8101040e03ff",		NO_ARGS },
{ "exp_compensation",	"direct",	"8101044e00000000ff",	EXP_COMP_ARG},


{ "aperture",		"reset",	"8101040200ff",		NO_ARGS },
{ "aperture",		"up",		"8101040202ff",		NO_ARGS },
{ "aperture",		"down",		"8101040203ff",		NO_ARGS },
/* manual says that the initial value of aperture gain is 5 !?*/
{ "aperture",		"direct",	"8101044200000005ff",	APERTURE_ARG },

{ "wide",		"off",		"8101046000ff",		NO_ARGS},
{ "wide",		"cinema",	"8101046001ff",		NO_ARGS},
{ "wide",		"full",		"8101046002ff",		NO_ARGS},

{ "lr_reverse",		"on",		"8101046102ff",		NO_ARGS},
{ "lr_reverse",		"off",		"8101046103ff",		NO_ARGS},

{ "freeze",		"on",		"8101046202ff",		NO_ARGS},
{ "freeze",		"off",		"8101046203ff",		NO_ARGS},

{ "pic_effect",		"off",		"8101046300ff",		NO_ARGS},
{ "pic_effect",		"pastel",	"8101046301ff",		NO_ARGS},
{ "pic_effect",		"negart",	"8101046302ff",		NO_ARGS},
{ "pic_effect",		"sepia",	"8101046303ff",		NO_ARGS},
{ "pic_effect",		"b&w",		"8101046304ff",		NO_ARGS},
{ "pic_effect",		"solarize",	"8101046305ff",		NO_ARGS},
{ "pic_effect",		"mosaic",	"8101046306ff",		NO_ARGS},
{ "pic_effect",		"slim",		"8101046307ff",		NO_ARGS},
{ "pic_effect",		"stretch",	"8101046308ff",		NO_ARGS},

{ "dig_effect",		"off",		"8101046400ff",		NO_ARGS},
{ "dig_effect",		"still",	"8101046401ff",		NO_ARGS},
{ "dig_effect",		"flash",	"8101046402ff",		NO_ARGS},
{ "dig_effect",		"lumi",		"8101046403ff",		NO_ARGS},
{ "dig_effect",		"trail",	"8101046404ff",		NO_ARGS},
{ "dig_effect",		"level",	"8101046400ff",		DIG_EFFECT_ARG },

};

#define N_EVI100_CMDS 	(sizeof(vcd_evi100_tbl)/sizeof(Visca_Cmd_Def))
	
static Visca_Cmd_Def vcd_common_tbl[]={
{ "addr_set",		"broadcast",	"883001ff",		NO_ARGS },

{ "clear",		"broadcast",	"88010001ff",		NO_ARGS },

{ "power",		"on",		"8101040002ff",		NO_ARGS },
{ "power",		"off",		"8101040003ff",		NO_ARGS },

{ "zoom",		"stop",		"8101040700ff",		NO_ARGS	},
{ "zoom",		"in",		"8101040702ff",		NO_ARGS	},
{ "zoom",		"out",		"8101040703ff",		NO_ARGS	},
{ "zoom",		"in_rate",	"8101040720ff",		ZOOM_SPEED },
{ "zoom",		"out_rate",	"8101040730ff",		ZOOM_SPEED },
{ "zoom",		"direct",	"8101044700000000ff",	ZOOM_OPT_ARG },

{ "focus",		"stop",		"8101040800ff",		NO_ARGS	}, 
{ "focus",		"far",		"8101040802ff",		NO_ARGS	},
{ "focus",		"near",		"8101040803ff",		NO_ARGS	},
{ "focus",		"direct",	"8101044800000000ff",	FOCUS_POS },
{ "focus",		"auto_focus",	"8101043802ff",		NO_ARGS	}, 
{ "focus",		"manual_focus",	"8101043803ff",		NO_ARGS	},

{ "white_bal",		"auto",		"8101043500ff",		NO_ARGS	},
{ "white_bal",		"indoor_mode",	"8101043501ff",		NO_ARGS	},
{ "white_bal",		"outdoor_mode",	"8101043502ff",		NO_ARGS	},
{ "white_bal",		"onepush_mode",	"8101043503ff",		NO_ARGS	},
{ "white_bal",		"onepush_trigr","8101041005ff",		NO_ARGS	},

{ "exposure",		"full_auto",	"8101043900ff",		NO_ARGS	},
{ "exposure",		"manual",	"8101043903ff",		NO_ARGS	},
{ "exposure",		"shutter_priority", "810104390aff",	NO_ARGS	},
{ "exposure",		"iris_priority", "810104390bff",	NO_ARGS	},
{ "exposure",		"bright_mode",	"810104390dff",		NO_ARGS	},

{ "shutter",		"reset",	"8101040a00ff",		NO_ARGS	},
{ "shutter",		"up",		"8101040a02ff",		NO_ARGS	},
{ "shutter",		"down",		"8101040a03ff",		NO_ARGS	}, 
{ "shutter",		"direct",	"8101044a00000000ff",	SHUTR_ARG },

{ "iris",		"reset",	"8101040b00ff",		NO_ARGS	},
{ "iris",		"up",		"8101040b02ff",		NO_ARGS	},
{ "iris",		"down",		"8101040b03ff",		NO_ARGS	}, 
{ "iris",		"direct",	"8101044b00000000ff",	IRIS_ARG },

{ "gain",		"reset",	"8101040c00ff",		NO_ARGS	},
{ "gain",		"up",		"8101040c02ff",		NO_ARGS	},
{ "gain",		"down",		"8101040c03ff",		NO_ARGS	},
{ "gain",		"direct",	"8101044c00000000ff",	GAIN_ARG },

{ "bright",		"reset",	"8101040d00ff",		NO_ARGS	},
{ "bright",		"up",		"8101040d02ff",		NO_ARGS	},
{ "bright",		"down",		"8101040d03ff",		NO_ARGS	},

{ "backlight",		"on",		"8101043302ff",		NO_ARGS	},
{ "backlight",		"off",		"8101043303ff",		NO_ARGS	},

{ "memory",		"reset",	"8101043f0000ff",	MEM_ARG	},
{ "memory",		"set",		"8101043f0100ff",	MEM_ARG	},
{ "memory",		"recall",	"8101043f0200ff",	MEM_ARG	},

{ "datascreen",		"on",		"8101060602ff",		NO_ARGS	},
{ "datascreen",		"off",		"8101060603ff",		NO_ARGS	},

{ "ir_receive",		"on",		"8101060802ff",		NO_ARGS	},
{ "ir_receive",		"off",		"8101060803ff",		NO_ARGS	},

{ "ir_rcvret",		"on",		"81017d01030000ff",	NO_ARGS	},
{ "ir_rcvret",		"off",		"81017d01130000ff",	NO_ARGS	}, 

{ "pantilt",		"up",		"8101060101010301ff",	TILT_SPEED },
{ "pantilt",		"down",		"8101060101010302ff",	TILT_SPEED },
{ "pantilt",		"left",		"8101060101010103ff",	PAN_SPEED },
{ "pantilt",		"right",	"8101060101010203ff",	PAN_SPEED },
{ "pantilt",		"upleft",	"8101060101010101ff",	PT_SPEED },
{ "pantilt",		"upright",	"8101060101010201ff",	PT_SPEED },
{ "pantilt",		"downleft",	"8101060101010102ff",	PT_SPEED },
{ "pantilt",		"downright",	"8101060101010202ff",	PT_SPEED },
{ "pantilt",		"stop",		"8101060101010303ff",	NO_ARGS	}, 
{ "pantilt",		"absolute_pos",	"8101060201010000000000000000ff",PT_POSN },
{ "pantilt",		"relative_pos",	"8101060301010000000000000000ff",PT_POSN },
{ "pantilt",		"home",		"81010604ff",		NO_ARGS	}, 
{ "pantilt",		"reset",	"81010605ff",		NO_ARGS	},

{ "limit",		"set",		"8101060700000000000000000000ff", PT_LMT_SET },
{ "limit",		"clear",	"810106070100070f0f0f070f0f0fff", PT_LMT_CLR }, 

};


#define N_COMMON_CMDS		(sizeof(vcd_common_tbl)/sizeof(Visca_Cmd_Def))


static Visca_Cmd_Def vcd_evi30_tbl[]={
{ "keylock",		"off",		"8101041700ff",		NO_ARGS	},
{ "keylock",		"on",		"8101041702ff",		NO_ARGS	},

{ "auto_track",		"enable",	"8101070100ff",		ATENB_ARG },
{ "auto_track",		"auto_exposure","8101070200ff",		ATAE_ARG },
{ "auto_track",		"auto_zoom",	"8101070310ff",		ATAZ_ARG },
{ "auto_track",		"offset",	"8101070500ff",		OFFSET_ARG },
{ "auto_track",		"chase",	"8101070700ff",		CHASE_ARG },
{ "auto_track",		"entry",	"8101071500ff",		ENTRY_ARG },
{ "auto_track",		"lostinfo",	"810106200720ff",	NO_ARGS	},

{ "motion_detect",	"enable",	"8101070800ff",		MDENB_ARG },
{ "motion_detect",	"frame_set",	"81010709ff",		NO_ARGS	},
{ "motion_detect",	"select_frame",	"8101070a10ff",		NO_ARGS	},
{ "motion_detect",	"y_level",	"8101070b0000ff",	MD_ARG	},
{ "motion_detect",	"hue_level",	"8101070c0000ff",	MD_ARG	},
{ "motion_detect",	"size",		"8101070d0000ff",	MD_ARG	},
{ "motion_detect",	"display_time",	"8101070f0000ff",	MD_ARG	},
{ "motion_detect",	"refresh_time",	"8101070b0000ff",	MD_ARG	},
{ "motion_detect",	"refresh_mode",	"8101071000ff",		REF_ARG	},
{ "motion_detect",	"lostinfo",	"810106200721ff",	NO_ARGS	},
{ "motion_detect",	"measure_mode1","8101072700ff",		MM_ARG	},
{ "motion_detect",	"measure_mode2","8101072800ff",		MM_ARG	}

};


#define N_EVI30_CMDS		(sizeof(vcd_evi30_tbl)/sizeof(Visca_Cmd_Def))


static Visca_Inq_Def vid_common_tbl[]={
{ "device_info",	INFO_INQ,	"81090002ff",	"00000000000000ff" },
{ "power",		POWER_INQ,	"81090400ff",	"02ff"		},
{ "zoom",		POSN_INQ,	"81090447ff",	"00000000ff"	},
{ "focus_mode",		FOCUS_MODE_INQ,	"81090438ff",	"00ff"		},
{ "focus_pos",		POSN_INQ,	"81090448ff",	"00000000ff"	},
{ "white_balance",	WBMODE_INQ,	"81090435ff",	"00ff"		},
{ "exposure_mode",	EXPMODE_INQ,	"81090439ff",	"00ff"		},
{ "shutter",		POSN_INQ,	"8109044aff",	"00000000ff"	},
{ "iris",		POSN_INQ,	"8109044Bff",	"00000000ff"	},
{ "gain",		POSN_INQ,	"8109044Cff",	"00000000ff"	},
{ "backlight",		BLMODE_INQ,	"81090433ff",	"02ff"		},
{ "memory",		MEMORY_INQ,	"8109043fff",	"00ff"		},
{ "datascreen",		DATASCRN_INQ,	"81090606ff",	"00ff"		},
{ "pantilt_mode",	PT_MODE_INQ,	"81090610ff",	"0000ff"	},
{ "pantilt_max_speed",	PT_MAX_SPEED_INQ, "81090611ff",	"0000ff"	},
{ "pantilt_posn",	PT_POSN_INQ,	"81090612ff",	"0000000000000000ff" },
{ "video",		VIDEO_INQ, 	"81090623ff",	"00ff"		},

/* this one is a strange one, it begins with 9007 instead of 9050 */

/* but this is ok since the part of response that we are checking 
 * is unique in that it begins with 7D
 */

{ "ir_recv_ret",	IR_RECV_INQ,	"",		"7d010000ff"	},

{ "invalid_common_inq",	NULL_INQ,	"",		""		}

};


#define N_COMMON_INQS		(sizeof(vid_common_tbl)/sizeof(Visca_Inq_Def))
 

static Visca_Inq_Def vid_evi100_tbl[]={
/* because all the responses begin with 9050, we omit that... */

{ "auto_pwr",		POSN_INQ,	"81090440ff",	"00000000ff"	},
{ "dig_zoom",		DIG_ZOOM_INQ,	"81090406ff",	"00ff"		},
{ "af_mode",		AFMODE_INQ,	"81090458ff",	"00ff"		},
{ "focus_limit",	POSN_INQ,	"81090428ff",	"00000000ff"	},
{ "rgain",		POSN_INQ,	"81090443ff",	"00000000ff"	},
{ "bgain",		POSN_INQ,	"81090444ff",	"00000000ff"	},
{ "slow_shutter",	SLOW_SHUTR_INQ,	"8109045aff",	"00ff"		},
{ "bright",		POSN_INQ,	"8109044Dff",	"00000000ff"	},
{ "exp_comp_mode",	EXPCMP_MOD_INQ,	"8109043eff",	"00000000ff"	},
{ "exp_comp_pos",	POSN_INQ,	"8109044eff",	"00ff"		},
{ "aperture",		POSN_INQ,	"81090442ff",	"00000000ff"	},
{ "wide_mode",		WID_MOD_INQ,	"81090460ff",	"00ff"		},
{ "lr_rev_mode",	LR_REV_INQ,	"81090461ff",	"00ff"		},
{ "freeze_mode",	FREEZE_MOD_INQ,	"81090462ff",	"00ff"		},
{ "pic_effect_mode",	PIC_EFFECT_MOD_INQ, "81090463ff", "00ff"	},
{ "dig_effect_mode",	DIG_EFFECT_MOD_INQ, "81090464ff", "00ff"	},
{ "dig_effect_level",	DIG_EFFECT_LVL_INQ, "81090465ff", "00ff"	},
//{ "device_info",	INFO_INQ,	"81090002ff",	"00000000000000ff" },
{ "invalid_evi100_inq",	NULL_INQ,	"",		""		}

};


#define N_EVI100_INQS		(sizeof(vid_evi100_tbl)/sizeof(Visca_Inq_Def))


static Visca_Inq_Def vid_evi30_tbl[]={
/* because all the responses begin with Z050, we omit that... */
/* Z = camera addr+8 */

{ "keylock",		LOCK_INQ,	"81090417ff",	"00ff"		},

/* an apparent error in the visca docs... */
/*
{ "id",			ID_INQ, 	"81090422ff",	"0000ff"	},
*/

{ "id",			ID_INQ, 	"81090422ff",	"00000000ff"	},
{ "atmd_mode",		ATMD_MODE_INQ,	"81090722ff",	"00ff"		},
{ "at_mode",		AT_MODE_INQ,	"81090723ff",	"0000ff"	},
{ "at_entry",		AT_ENTRY_INQ,	"81090715ff",	"00ff"		},
{ "md_mode",		MD_MODE_INQ,	"81090724ff",	"0000ff"	},
{ "at_obj_posn",	AT_POSN_INQ,	"81090720ff",	"000000ff"	},
{ "md_object_posn",	MD_POSN_INQ,	"81090721ff",	"000000ff"	},
{ "md_y_level",		MD_Y_INQ,	"8109070bff",	"0000ff"	},
{ "md_hue_level",	MD_HUE_INQ,	"8109070cff",	"0000ff"	},
{ "md_size",		MD_SIZE_INQ,	"8109070dff",	"0000ff"	},
{ "md_disp_time",	MD_DISP_TIME_INQ,"8109070fff",	"0000ff"	},
{ "md_refresh",		MD_REF_INQ,	"81090710ff",	"00ff"		},
{ "md_ref_time",	MD_REF_TIME_INQ,"81090711ff",	"0000ff"	},
{ "invalid_evi30_inq",	NULL_INQ,	"",		""		}

};

#define N_EVI30_INQS		(sizeof(vid_evi30_tbl)/sizeof(Visca_Inq_Def))


static Visca_Inq_Def vid_evi70_tbl[]={
{ "ir_mode",		IR_MODE_INQ,	"81090401ff",	"00ff"	},
{ "flip_mode",		FLIP_MODE_INQ,	"81090466ff",	"00ff"	},
{ "invalid_evi70_inq",	NULL_INQ,	"",		""		}
};

#define N_EVI70_INQS		(sizeof(vid_evi70_tbl)/sizeof(Visca_Inq_Def))

#ifdef HAVE_VISCA
static int n_vcams=0;
#endif // HAVE_VISCA

ITEM_INTERFACE_DECLARATIONS(Visca_Cam,vcam,0)
ITEM_INTERFACE_DECLARATIONS(Visca_Port,vport,0)

static const char *error_message(int code)
{
	switch(code){
		case 0x01: return("message length error");
		case 0x02: return("syntax error");
		case 0x03: return("cmd buffer full");
		case 0x04: return("cmd cancel");
		case 0x05: return("no sockets");
		case 0x41: return("not executable");
	}
	return("unrecognized error code");
}

#ifdef HAVE_VISCA

#define table_index_for_inq(tbl,code)	_table_index_for_inq(tbl,code)

static int _table_index_for_inq(QSP_ARG_DECL   Visca_Inq_Def *tbl, Inq_Type code )
{
	int i=0;

	while( tbl->vid_type != NULL_INQ ){
		if( tbl->vid_type == code ) return i;
		tbl++;
		i++;
	}
	sprintf(ERROR_STRING,"table_index_for_inq:  No visca inquiry definition found for code %d!?",code);
	WARN(ERROR_STRING);
	return -1;
}

/* Add a command definition to our database.
 * The main reasons that we bother with this is to get the automatic
 * name completion that comes with the item package...
 *
 */

#define init_visca_cmd(vcdp)	_init_visca_cmd(QSP_ARG  vcdp)

static void _init_visca_cmd( QSP_ARG_DECL  Visca_Cmd_Def *vcdp )
{
	Visca_Cmd_Set *vcsp;
	Visca_Command *vcmdp;

	vcsp = cmd_set_of(vcdp->vcd_set);
	if( vcsp == NULL ){
		vcsp = new_cmd_set(vcdp->vcd_set);
		assert( vcsp != NULL );
		vcsp->vcs_icp = create_visca_cmd_context(vcdp->vcd_set);
		if( vcsp->vcs_icp == NULL ){
			sprintf(ERROR_STRING,
				"Couldn't create item context %s",
				vcdp->vcd_set);
			error1(ERROR_STRING);
		}
	}

	push_visca_cmd_context(vcsp->vcs_icp);

	vcmdp = new_visca_cmd(vcdp->vcd_cmd);
	if( vcmdp == NULL ){
		sprintf(ERROR_STRING,"Couldn't create visca cmd %s",
			vcdp->vcd_cmd);
		error1(ERROR_STRING);
	}

	pop_visca_cmd_context(SINGLE_QSP_ARG);

	vcmdp->vcmd_vcdp = vcdp;

	// Make sure that MAX_PACKET_LEN is OK
	assert( strlen(vcdp->vcd_pkt) < MAX_PACKET_LEN );
} /* end init_visca_cmd() */

/* Call init_visca_inq with a pointer into the inquiry table.
 * An item is created that points to the table entry.
 */

#define init_visca_inq(vidp)	_init_visca_inq(QSP_ARG  vidp)

static void _init_visca_inq( QSP_ARG_DECL  Visca_Inq_Def *vidp )
{
	Visca_Inquiry *vip;

	vip = new_visca_inq(vidp->vid_inq);
	assert( vip != NULL );
	vip->vi_vidp = vidp;
	assert( strlen(vidp->vid_pkt) < MAX_PACKET_LEN );
}

/* Scan the table of command defns.
 */

#define load_visca_cmds()	_load_visca_cmds(SINGLE_QSP_ARG)

static void _load_visca_cmds(SINGLE_QSP_ARG_DECL)
{
	unsigned int i;
	static int cmds_inited=0;
	
	if( cmds_inited ) {
		WARN("load_visca_cmds:  unnecessary call");
		return;
	}

	for(i=0;i<N_EVI100_CMDS; i++)
		init_visca_cmd(&vcd_evi100_tbl[i]);

	for(i=0;i<N_COMMON_CMDS; i++)
		init_visca_cmd(&vcd_common_tbl[i]);

	for(i=0;i<N_EVI30_CMDS; i++)
		init_visca_cmd(&vcd_evi30_tbl[i]);

	for(i=0;i<N_EVI70_CMDS; i++)
		init_visca_cmd(&vcd_evi70_tbl[i]);

	
	for(i=0;i<N_EVI100_INQS; i++)
		init_visca_inq(&vid_evi100_tbl[i]);

	for(i=0;i<N_COMMON_INQS; i++)
		init_visca_inq(&vid_common_tbl[i]);

	for(i=0;i<N_EVI30_INQS; i++)
		init_visca_inq(&vid_evi30_tbl[i]);

	for(i=0;i<N_EVI70_INQS; i++)
		init_visca_inq(&vid_evi70_tbl[i]);

	cmds_inited=1;

}
#endif // HAVE_VISCA

/* The ACK messages should come back immediately after the transmission
 * of a command...  but is it sync'd to VBLANK???  Or is that just command execution?
 */

#define get_cmd_ack(vcam_p, vcdp)	_get_cmd_ack(QSP_ARG  vcam_p, vcdp)

static int _get_cmd_ack(QSP_ARG_DECL  Visca_Cam *vcam_p, Visca_Cmd_Def *vcdp)
{
	int n=3;
	int reply_addr;
	u_char ack_buf[LLEN];		/* probably could be MUCH shorter!? */

	
	/* wait until we have at least 3 chars before reading...
	 * BUG we should probably set an alarm here to wake us
	 * up in the case of something going really wrong.
	 */
	
	while( n_serial_chars(vcam_p->vcam_fd) < 3 )
		usleep(1000);

	n = recv_somex(vcam_p->vcam_fd,ack_buf,LLEN,n);

	/* Until we put the timeout in, we know we must have something if we are here... */
	assert( n!=0 );

	reply_addr = 0x80 + (vcam_p->vcam_index << 4);

	/* We expect the ack msg to be 0xZ0 0x40 0xff or 0xZ0 0x41 0xff (depending on socket number) 
	 * Z = device address + 8
	 */

	if( ack_buf[0] != reply_addr ){
		sprintf(ERROR_STRING,"Ack buffer camera address mismatch (expected 0x%x, received 0x%x)",
			reply_addr,ack_buf[0]);
		WARN(ERROR_STRING);
	}

	if(	ack_buf[0] == reply_addr 
	  &&	( ack_buf[1] == 0x40 || ack_buf[1] == 0x41 )
	  && 	ack_buf[2] == 0xff					){	/* all is normal */
		return(0);
	}

	if( (ack_buf[1] == 0x50 || ack_buf[1] == 0x51) && ack_buf[2] == 0xff ){
		//WARN("expected ACK response, but received completion response!?");
		/* This seems to happen reliably with certain commands,
		 * like home & manual_focus ...
		 */
		return(1);
	}

	if( ack_buf[0] == reply_addr && ( ack_buf[1]==0x60 || ack_buf[1]==0x61 ) ){	/* an error */
		const char *err_msg;

		err_msg = error_message(ack_buf[2]);
		sprintf(ERROR_STRING,"Command %s %s:  %s",
			vcdp->vcd_set,vcdp->vcd_cmd,err_msg);
		WARN(ERROR_STRING);

		/* eat up the last 0xff */

		while( n_serial_chars(vcam_p->vcam_fd) < 1 )
			usleep(1000);

		n = 1;
		n = recv_somex(vcam_p->vcam_fd,ack_buf,LLEN,n);

		if( ack_buf[0] != 0xff ){
			sprintf(ERROR_STRING,
				"Error string terminated with 0x%x, expected 0xff!?",
				ack_buf[0]);
			WARN(ERROR_STRING);
		}

		return(-1);
	}

	sprintf(ERROR_STRING,"get_cmd_ack:  Unexpected ACK msg (0x%.2x%.2x%.2x) for %s %s",
		ack_buf[0], ack_buf[1], ack_buf[2], vcdp->vcd_set, vcdp->vcd_cmd);
	advise(ERROR_STRING);

set_raw_len(ack_buf);
	dump_char_buf(ack_buf);
	return(-1);
} /* end get_cmd_ack() */

/* Unlike the ACK msgs, we don't know exaclty how long the cmds will
 * take to execute...
 */

#define get_cmd_completion(vcam_p,vcdp)	_get_cmd_completion(QSP_ARG  vcam_p, vcdp)

static void _get_cmd_completion( QSP_ARG_DECL  Visca_Cam *vcam_p, Visca_Cmd_Def *vcdp )
{
	int n=3;
	int reply_addr;
	u_char comp_buf[LLEN];		/* probably could be MUCH shorter!? */
	
	/* wait until we have at least 3 chars before reading...
	 * BUG we should probably set an alarm here to wake us
	 * up in the case of something going really wrong.
	 */

	while( n_serial_chars(vcam_p->vcam_fd) < 3 )
		usleep(1000);

	n = recv_somex(vcam_p->vcam_fd,comp_buf,LLEN,n);

	/* this check may not be cautious if we put in a timeout... */
	assert(n!=0);

	reply_addr = 0x80 + (vcam_p->vcam_index << 4);

	/* We expect the completion msg to be 0xZ0 0x50 0xff or 0xZ0 0x51 0xff (depending on sockeet number) 
	 * Z = device address + 8
	 */
	
	if( comp_buf[0] == reply_addr && ( comp_buf[1] == 0x50 || comp_buf[1] == 0x51 ) && comp_buf[2] == 0xff ) /* all is normal */
		return;

	sprintf(ERROR_STRING,"Expected 0x%.2x 0x50/0x51 , received 0x%.2x 0x%.2x!?",
			reply_addr, comp_buf[0], comp_buf[1]);
	WARN(ERROR_STRING);
	
	sprintf(ERROR_STRING,"Unexpected completion msg for %s %s",
		vcdp->vcd_set,vcdp->vcd_cmd);
	advise(ERROR_STRING);

set_raw_len(comp_buf);
	dump_char_buf(comp_buf);
} /* end get_cmd_completion */

#define compare_response(vidp,buf,proto,i,n)	_compare_response(QSP_ARG  vidp,buf,proto,i,n)

static int _compare_response(QSP_ARG_DECL  Visca_Inq_Def *vidp,
		unsigned char *buf,
		unsigned char * proto, int i,int n)
{
	int j;

	if( buf[i] != proto[i] ){
		sprintf(ERROR_STRING,
	"Inquiry %s, response (0x%.2x) differs from prototype (0x%.2x) at posn %d",
			vidp->vid_inq,buf[i],proto[i],i);
		WARN(ERROR_STRING);

		advise("\n\tproto\tresp\n");
		for(j=0;j<n;j++){
			sprintf(ERROR_STRING,"\t0x%.2x\t0x%.2x",
				proto[j],buf[j]);
			advise(ERROR_STRING);
		}
		return(-1);
	}
	return(0);
}

static long get_binary_number(u_char *buf,int n_bytes,int first_position)
{
	long n,i;

	n=0;
	for(i=0;i<n_bytes;i++){
		n <<= 8;
		n += buf[first_position+i];
	}
//sprintf(ERROR_STRING,"get_binary_number %d %d:  \"%s\" - returning 0x%lx",n_bytes,first_position,printable_string(&buf[first_position]),n);
//advise(ERROR_STRING);
	return(n);
}

/* a utility routine used in interpreting the response strings...
 * We have a problem:  some parameters (like pan value) are represented
 * as signed integers, represented by 4 hex digits...  We need to extend
 * the sign bit.
 */

static long get_number_reply(u_char *buf,int n_hex_digits,int first_position)
{
	long n,i;

	n=0;
	for(i=0;i<n_hex_digits;i++){
		n <<= 4;
		n += buf[first_position+i];
	}
	if( n_hex_digits == 4 && n&0x8000 )
		n |= 0xffff0000;

//sprintf(ERROR_STRING,"get_number_reply %d %d:  \"%s\" - returning 0x%lx",n_hex_digits,first_position,printable_string(buf),n);
//advise(ERROR_STRING);
	return(n);
}

#define process_inq_error(vcam_p, vidp)	_process_inq_error(QSP_ARG  vcam_p, vidp)

static void _process_inq_error(QSP_ARG_DECL  Visca_Cam *vcam_p, Visca_Inq_Def *vidp)
{
	int n;
	const char *err_msg;
	u_char err_buf[LLEN];		/* probably could be MUCH shorter!? */

	/* at this point, we've already read the first two chars
	 * of the error string...
	 */

	while( n_serial_chars(vcam_p->vcam_fd) < 2 )
		usleep(1000);
	n=2;
	n = recv_somex(vcam_p->vcam_fd,err_buf,LLEN,n);

	err_msg = error_message(err_buf[0]);
	sprintf(ERROR_STRING,"Inquiry %s:  %s",	vidp->vid_inq, err_msg);
	WARN(ERROR_STRING);
}

#define EXP_MODE_STRING		"exposure_mode"

#define INQ_RESULT( pname , rstring )	inq_result( QSP_ARG  pname , rstring )

static void inq_result(QSP_ARG_DECL  const char *parameter_name,
				const char *result_string)
{
	assign_var(INQ_RESULT_NAME, result_string);
	if( verbose ){
		if( parameter_name != NULL ){
			sprintf(msg_str,"%s is %s",parameter_name,result_string);
		} else {
			sprintf(msg_str,"%s",result_string);
		}

		prt_msg(msg_str);
	}
}

#define BAD_RESP( c )		sprintf(ERROR_STRING,			\
	"Unexpected response code 0x%.2x, inquiry %s",c, vidp->vid_inq);	\
				WARN(ERROR_STRING)

#define get_inq_reply(vcam_p,vidp)	_get_inq_reply(QSP_ARG  vcam_p,vidp)

static void _get_inq_reply(QSP_ARG_DECL  Visca_Cam *vcam_p, Visca_Inq_Def *vidp)
{
	u_char reply_buf[LLEN];		/* probably could be MUCH shorter!? */
	u_char proto[16];
	int i,n;
	int result;
	int mem_index;
	char result_str[80];	/* for passing answers back to scripts... */
	int reply_addr;

//advise("get_inq_reply BEGIN");
	/* If we get an error, we won't get the expected number of chars
	 * returned, so we read the first two to check for an error...
	 *
	 * The first two chars should be the address of the camera...
	 */

	while( n_serial_chars(vcam_p->vcam_fd) < 2 )
		usleep(1000);

	n=2;
	n = recv_somex(vcam_p->vcam_fd,reply_buf,LLEN,n);
	/* should we check that we got two??? */
	if( n != 2 ){
		sprintf(ERROR_STRING,"get_inq_reply:  expected 2 chars, got %d",n);
		WARN(ERROR_STRING);
	}
	
	reply_addr = 0x80 + (vcam_p->vcam_index << 4);

	if( reply_buf[0] == reply_addr && ( reply_buf[1]==0x60 || reply_buf[1]==0x61 ) ) {	/* an error */
		process_inq_error(vcam_p,vidp);
		return;
	
	}

//sprintf(ERROR_STRING,"reply_buf[0] = 0x%x",reply_buf[0]);
//advise(ERROR_STRING);
//sprintf(ERROR_STRING,"reply_buf[1] = 0x%x",reply_buf[1]);
//advise(ERROR_STRING);

	/* completion message is 0xZ0 0x50/51 0xFF  
	 * Z = device address + 8
	 */




	if( reply_buf[0] != reply_addr || ( reply_buf[1] != 0x50 && reply_buf[1] != 0x51 ) ) {
		sprintf(ERROR_STRING,
	"Inquiry %s:  expected 0x%.2x 0x50/51, received 0x%.2x 0x%.2x!?",
			vidp->vid_inq, reply_addr, reply_buf[0], reply_buf[1]);
		WARN(ERROR_STRING);
		if( reply_buf[1] != 0x41 && reply_buf[1] != 0x40 )
			return;
	}
		

	/* The replies can be of variable length...
	 * We look at the prototype reply string in the defn
	 * to determine how many chars to expect.
	 */

	n = strlen(vidp->vid_reply)/2;

	/* wait until we have the expected # of chars before reading...
	 * BUG we should probably set an alarm here to wake us
	 * up in the case of something going really wrong.
	 */


	while( (i=n_serial_chars(vcam_p->vcam_fd)) < n ){
		usleep(1000);
	}

	n = recv_somex(vcam_p->vcam_fd,reply_buf,LLEN,n);

//sprintf(ERROR_STRING,"Read %d chars",n);
//advise(ERROR_STRING);
//for(i=0;i<n;i++){
//sprintf(ERROR_STRING,"\treply_buf[%d] = %s (0x%x)",i,printable_version(reply_buf[i]),reply_buf[i]);
//advise(ERROR_STRING);
//}

	/* convert the prototype reply string to binary */

	for(i=0;i<n;i++)
		proto[i] = hex_byte((u_char *) &vidp->vid_reply[i*2]);

	/* compare the received string to the expected prototype */

	/* some positions must necessarily differ, because they
	 * contain the info!!
	 * Therefore, we only compare the first two, and last chars.
	 */

	if( compare_response(vidp,reply_buf,proto,n-1,n) < 0 ) return;

	/* Handle the reply based on the reply type */

	switch(vidp->vid_type){

#ifdef CAUTIOUS
		case NULL_INQ:
			sprintf(ERROR_STRING,"CAUTIOUS:  get_inq_reply:  should not see NULL_INQ!?");
			WARN(ERROR_STRING);
			break;
#endif // CAUTIOUS

		case POSN_INQ:
			/* The position is encoded in bytes 2-5 */
			result = get_number_reply(reply_buf,4,0);
			if( verbose ){
				sprintf(msg_str,"%s is %d (0x%x)",vidp->vid_inq,
					result,result);
				prt_msg(msg_str);
			}


			sprintf(result_str,"%d",result);
			assign_var(INQ_RESULT_NAME, result_str);

			break;

		case POWER_INQ:
			switch(reply_buf[0]){
				case 2: prt_msg("power is on"); break;
				case 3: prt_msg("power is off"); break;
				default:  BAD_RESP(reply_buf[0]); break;
			}
			break;

		case DIG_ZOOM_INQ:
			switch(reply_buf[0]){
				case 2: prt_msg("digital zoom is on"); break;
				case 3: prt_msg("digital zoom is off"); break;
				default:  BAD_RESP(reply_buf[0]); break;
			}
			break;

		case IR_MODE_INQ:
			switch(reply_buf[0]){
				case 2:
					//prt_msg("ir mode is on");
					assign_var(INQ_RESULT_NAME, "on");
					break;
				case 3:
					//prt_msg("ir mode is off");
					assign_var(INQ_RESULT_NAME, "off");
					break;
				default:  BAD_RESP(reply_buf[0]); break;
			}
			break;

		case FLIP_MODE_INQ:
			switch(reply_buf[0]){
				case 2:
					// prt_msg("flip mode is on");
					vcam_p->vcam_flipped = 1;
					assign_var(INQ_RESULT_NAME,"on");
					break;
				case 3:
					// prt_msg("flip mode is off");
					vcam_p->vcam_flipped = 0;
					assign_var(INQ_RESULT_NAME, "off");
					break;
				default:  BAD_RESP(reply_buf[0]); break;
			}
			break;

		case FOCUS_MODE_INQ:
			switch( reply_buf[0] ){
				case 2:
					//prt_msg("autofocus");
					assign_var(INQ_RESULT_NAME, "auto");
					break;
				case 3:
					//prt_msg("manual focus");
					assign_var(INQ_RESULT_NAME, "manual");
					break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

		case AFMODE_INQ:
			switch( reply_buf[0] ){
				case 2:  prt_msg("high autofocus sensitivity"); break;
				case 3:  prt_msg("low autofocus sensitivity"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

#define CAM_IS_70_OR_100(vcam_p)	(IS_EVI_D100(vcam_p) || IS_EVI_D70(vcam_p) )

		case WBMODE_INQ:
			if( CAM_IS_70_OR_100(vcam_p) ){
				switch(reply_buf[0]){
				case 4: prt_msg("ATW mode"); goto end_of_wbmode;
				case 5: prt_msg("manual white balance mode"); goto end_of_wbmode;
				}
			}

			switch(reply_buf[0]){
				case 0: prt_msg("auto white balance"); break;
				case 1: prt_msg("indoor mode"); break;
				case 2: prt_msg("outdoor mode"); break;
				case 3: prt_msg("onepush mode"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}

			end_of_wbmode:
			break;

		case EXPMODE_INQ:
			if( CAM_IS_70_OR_100(vcam_p) ) {
				switch(reply_buf[0]){
				case 0xc: INQ_RESULT(EXP_MODE_STRING,"gain_priority"); goto end_of_expmode;
				case 0x1a: INQ_RESULT(EXP_MODE_STRING,"shutter_auto"); goto end_of_expmode;
				case 0x1b: INQ_RESULT(EXP_MODE_STRING,"iris_auto"); goto end_of_expmode;
				case 0x1c: INQ_RESULT(EXP_MODE_STRING,"gain_auto"); goto end_of_expmode;
				}
			}

			switch(reply_buf[0]){
				case 0:  INQ_RESULT(EXP_MODE_STRING,"full_auto"); break;
				case 3:  INQ_RESULT(EXP_MODE_STRING,"manual"); break;
				case 0xa: INQ_RESULT(EXP_MODE_STRING,"shutter_priority"); break;
				case 0xb: INQ_RESULT(EXP_MODE_STRING,"iris_priority"); break;
				case 0xd: INQ_RESULT(EXP_MODE_STRING,"bright_mode"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}

			end_of_expmode:
			break;

		case SLOW_SHUTR_INQ:
			switch( reply_buf[0] ){
				case 2:  prt_msg("auto"); break;
				case 3:  prt_msg("manual"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

		case EXPCMP_MOD_INQ:
			switch( reply_buf[0] ){
				case 2:  prt_msg("exposure compensation mode on"); break;
				case 3:  prt_msg("exposure compensation mode off"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

		case BLMODE_INQ:
			switch(reply_buf[0]){
				case 2: prt_msg("backlight mode ON"); break;
				case 3: prt_msg("backlight mode OFF"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

		case WID_MOD_INQ:
			switch(reply_buf[0]){
				case 0: prt_msg("wide mode off"); break;
				case 1: prt_msg("cinema mode"); break;
				case 2: prt_msg("16:9 full mode"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

		case LR_REV_INQ:
			switch(reply_buf[0]){
				case 2: prt_msg("LR reverse mode ON"); break;
				case 3: prt_msg("LR reverse mode OFF"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

		case FREEZE_MOD_INQ:
			switch(reply_buf[0]){
				case 2: prt_msg("freeze mode ON"); break;
				case 3: prt_msg("freeze mode OFF"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

		case PIC_EFFECT_MOD_INQ:
			switch( reply_buf[0] ){
				case 0:  prt_msg("picture effect mode is off"); break;
				case 1:  prt_msg("picture effect mode is pastel"); break;
				case 2:  prt_msg("picture effect mode is neg art"); break;
				case 3:  prt_msg("picture effect mode is sepia"); break;
				case 4:  prt_msg("picture effect mode is b&w"); break;
				case 5:  prt_msg("picture effect mode is solarize"); break;
				case 6:  prt_msg("picture effect mode is mosaic"); break;
				case 7:  prt_msg("picture effect mode is slim"); break;
				case 8:  prt_msg("picture effect mode is stretch"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

		case DIG_EFFECT_MOD_INQ:
			switch(reply_buf[0]){
				case 0: prt_msg("digital effect mode is off"); break;
				case 1: prt_msg("digital effect mode is still"); break;
				case 2: prt_msg("digital effect mode is flash"); break;
				case 3: prt_msg("digital effect mode is lumi"); break;
				case 4: prt_msg("digital effect mode is trail"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

		case DIG_EFFECT_LVL_INQ:
			result = reply_buf[0];
			sprintf(msg_str,"%s is %d (0x%x)",vidp->vid_inq,result,result);
			prt_msg(msg_str);
			break;

		case MEMORY_INQ:
			mem_index = reply_buf[0];
			sprintf(msg_str,"Current memory index is %d", mem_index);
			prt_msg(msg_str);
			break;

		case DATASCRN_INQ:
			switch(reply_buf[0]){
				case 2:  prt_msg("datascreen is ON"); break;
				case 3:  prt_msg("datascreen is OFF"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

		case PT_MODE_INQ:
			sprintf(msg_str,"Pantilt mode status is 0x%.2x 0x%.2x",
				reply_buf[0],reply_buf[1]);
			prt_msg(msg_str);
			break;

		case PT_MAX_SPEED_INQ:
			sprintf(msg_str,"Pan max speed is %d (0x%x)\nTilt max speed is %d (0x%x)",
				reply_buf[0],reply_buf[0],reply_buf[1],reply_buf[1]);
			prt_msg(msg_str);
			break;

		case PT_POSN_INQ:
			result = get_number_reply(reply_buf,4,0);
			if( verbose ){
				sprintf(msg_str,"Pan position %d (0x%x) stored in $pan_posn", result,result);
				prt_msg(msg_str);
			}
			sprintf(msg_str,"%d",result);
			assign_var("pan_posn",msg_str);

			result = get_number_reply(reply_buf,4,4);
			if( verbose ){
				sprintf(msg_str,"Tilt position %d (0x%x) stored in $tilt_posn",result,result);
				prt_msg(msg_str);
			}
			sprintf(msg_str,"%d",result);
			assign_var("tilt_posn",msg_str);

			break;

		case VIDEO_INQ:
			switch(reply_buf[0]){
				case 0: prt_msg("NTSC"); break;
				case 1: prt_msg("PAL"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

		case INFO_INQ:
			vcam_p->vcam_vendor_id = get_binary_number(reply_buf,2,0);
			vcam_p->vcam_model_id = get_binary_number(reply_buf,2,2);
			vcam_p->vcam_rom_version = get_binary_number(reply_buf,2,4);
			vcam_p->vcam_max_socket = get_binary_number(reply_buf,1,6);

			if( verbose ){
				sprintf(msg_str,"Camera %d:",vcam_p->vcam_index);
				prt_msg(msg_str);

				sprintf(msg_str,"\tVendor ID:  0x%x", vcam_p->vcam_vendor_id);
				prt_msg(msg_str);

				sprintf(msg_str,"\tModel ID:   0x%x",vcam_p->vcam_model_id);
				prt_msg(msg_str);

				sprintf(msg_str,"\tROM Version:  0x%x",vcam_p->vcam_rom_version);
				prt_msg(msg_str);

				sprintf(msg_str,"\tSocket Number:  0x%x",vcam_p->vcam_max_socket);
				prt_msg(msg_str);
			}

			break;

		case IR_RECV_INQ:
			if( reply_buf[2]==4 ){
				switch(reply_buf[3]){
					case 0: prt_msg("power on/off"); break;
					case 7: prt_msg("zoom tele/wide"); break;
					case 0x38: prt_msg("AF on/off"); break;
					case 0x33: prt_msg("CAM backlight"); break;
					case 0x3f: prt_msg("CAM memory"); break;
					default: BAD_RESP(reply_buf[3]); break;
				}
			} else if( reply_buf[2] == 6 ){
				prt_msg("pan/tilt drive");
			} else {
				BAD_RESP(reply_buf[2]);
			}
			break;



/* The rest of the cases are for EVI-30 */


#ifdef FOOBAR
		case UNDEF_INQ:
			sprintf(ERROR_STRING,
				"Response format for inquiry %s is undefined!?",
				vidp->vid_inq);
			WARN(ERROR_STRING);
			break;
#endif // FOOBAR


		case LOCK_INQ:
			switch(reply_buf[0]){
				case 0: prt_msg("KeyLock is off"); break;
				case 2: prt_msg("KeyLock is on"); break;
				default:  BAD_RESP(reply_buf[0]); break;
			}
			break;

                case ID_INQ:
			result = get_number_reply(reply_buf,2,0);
			sprintf(msg_str,"Camera ID is %d",result);
			prt_msg(msg_str);
			break;


		case ATMD_MODE_INQ:
			switch(reply_buf[0]){
				case 0: prt_msg("normal mode (at/md off)");
					break;
				case 1: prt_msg("AT mode"); break;
				case 2: prt_msg("MD mode"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

		case AT_MODE_INQ:
			sprintf(msg_str,"AT status is 0x%.2x 0x%.2x",
				reply_buf[0],reply_buf[1]);
			prt_msg(msg_str);
			break;

		case AT_ENTRY_INQ:
			switch(reply_buf[0]){
				case 0: prt_msg("entry mode 0"); break;
				case 1: prt_msg("entry mode 1"); break;
				case 2: prt_msg("entry mode 2"); break;
				case 3: prt_msg("entry mode 3"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

		case MD_MODE_INQ:
			sprintf(msg_str,"MD status is 0x%.2x 0x%.2x",
				reply_buf[0],reply_buf[1]);
			prt_msg(msg_str);
			break;

		case AT_POSN_INQ:
			sprintf(msg_str,"AT posn codes:  0x%.2x 0x%.2x 0x%.2x",
				reply_buf[0],reply_buf[1],reply_buf[2]);
			prt_msg(msg_str);
			break;

		case MD_POSN_INQ:
			sprintf(msg_str,"MD posn codes:  X: 0x%.2x Y: 0x%.2x Z: 0x%x",
				reply_buf[0],reply_buf[1],reply_buf[2]);
			prt_msg(msg_str);
			break;
			
		case MD_Y_INQ:
			sprintf(msg_str,"MD Y Level: 0x%x", reply_buf[1]);
			prt_msg(msg_str);
			break;

		case MD_HUE_INQ:
			result = get_number_reply(reply_buf,1,3);
			sprintf(msg_str,"MD hue is %d",result);
			prt_msg(msg_str);
			break;

		case MD_SIZE_INQ:
			result = get_number_reply(reply_buf,1,3);
			sprintf(msg_str,"MD size is %d",result);
			prt_msg(msg_str);
			break;

		case MD_DISP_TIME_INQ:
			result = get_number_reply(reply_buf,1,3);
			sprintf(msg_str,"MD disp time is %d",result);
			prt_msg(msg_str);
			break;

		case MD_REF_INQ:
			switch(reply_buf[0]){
				case 0: prt_msg("refresh mode 1"); break;
				case 1: prt_msg("refresh mode 2"); break;
				case 2: prt_msg("refresh mode 3"); break;
				default: BAD_RESP(reply_buf[0]); break;
			}
			break;

		case MD_REF_TIME_INQ:
			result = get_number_reply(reply_buf,1,1);
			sprintf(msg_str,"MD ref time is %d",result);
			prt_msg(msg_str);
			break;


	}

} /* end get_inq_reply() */

static void _hex_digit_to_ascii(QSP_ARG_DECL  u_char *dest,int val)
{

	if( val>=0 && val<=9 ){
		*dest = '0' + val;
		return;
	}
	if( val>=0xa && val<=0xf ){
		*dest = 'a' + val - 0xa;
		return;
	}

	sprintf(ERROR_STRING,"bad hex digit value %d (0x%x)",val,val);

	WARN(ERROR_STRING);
}

static COMMAND_FUNC( get_pan_speed )
{
	char prompt[LLEN];
	unsigned int pan_speed;

	sprintf(prompt, "pan speed (0x%x - 0x%x)", vparam_p->pan_speed_min, vparam_p->pan_speed_max);

	pan_speed = HOW_MANY(prompt);

	if( pan_speed < vparam_p->pan_speed_min || pan_speed > vparam_p->pan_speed_max ){
		sprintf(ERROR_STRING,"Pan speed (0x%x) must be in the range 0x%x-0x%x",
			pan_speed,vparam_p->pan_speed_min, vparam_p->pan_speed_max);
		WARN(ERROR_STRING);
		advise("defaulting to 0x10");
		pan_speed = 0x10;
	}
	the_vcam_p->vcam_pan_speed = pan_speed;
}

static void _set_pan_speed(QSP_ARG_DECL  u_char *pkt,int speed)
{
	hex_digit_to_ascii(&pkt[9],speed&0x0f);
	hex_digit_to_ascii(&pkt[8],(speed&0xf0)>>4);
}

static COMMAND_FUNC( get_tilt_speed )
{
	char prompt[LLEN];
	unsigned int tilt_speed;

	sprintf(prompt, "tilt speed (0x%x - 0x%x)", vparam_p->tilt_speed_min, vparam_p->tilt_speed_max);
	tilt_speed = HOW_MANY(prompt);

	if( tilt_speed < vparam_p->tilt_speed_min || tilt_speed > vparam_p->tilt_speed_max ){
		sprintf(ERROR_STRING,"Tilt speed (0x%x) must be in the range 0x%x-0x%x",
			tilt_speed, vparam_p->tilt_speed_min, vparam_p->tilt_speed_max);
		WARN(ERROR_STRING);
		advise("defaulting to 0x10");
		tilt_speed = 0x10;
	}
	the_vcam_p->vcam_tilt_speed = tilt_speed;
}

static void _set_tilt_speed(QSP_ARG_DECL  u_char *pkt, int speed )
{
	/* the indices were wrong (was this right for the 30, or has it always been wrong?? */
	/*hex_digit_to_ascii(&pkt[9],speed&0x0f); */
	/*hex_digit_to_ascii(&pkt[8],(speed&0xf0)>>4); */
	hex_digit_to_ascii(&pkt[11],speed&0x0f);
	hex_digit_to_ascii(&pkt[10],(speed&0xf0)>>4);
}


#define GET_CAMERA_LOCK( vcam_p , code )						\
											\
	while( vcam_p->vcam_qlock != 0 )						\
		usleep(10);								\
	/* BUG?  could the other thread take the lock back here??? */			\
	vcam_p->vcam_qlock = code;

#define RLS_CAMERA_LOCK( vcam_p )							\
											\
	vcam_p->vcam_qlock = 0;

#ifdef VISCA_THREADS


/* Functions used to control the requests by a thread */

static void init_server_thread(Visca_Cam *vcam_p)
{
	pthread_attr_t attr1;

	vcam_p->vcam_qlock=0;
	pthread_attr_init(&attr1);
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);
	pthread_create(&vcam_p->vcam_ctl_thread,&attr1,camera_request_server,vcam_p);
}


/* This is the function executed by a thread to control the commands
 * sent to the camera.
 * If the asynchronous mode have been selected, we don't execute a
 * request when there are newer requests in the buffer of the same
 * type (command + set)
 */
static void *camera_request_server(void *arg)
{
	Visca_Cam *vcam_p;
	Node *np;
	int did_something;

	vcam_p = arg;
	np = NULL;
	while(1) {	
		did_something=0;
		/* Process inquiries first */
		if( vcam_p->vcam_vidp != NULL ){
			exec_visca_inquiry(THIS_QSP  vcam_p,vcam_p->vcam_vidp);
			did_something = 1;
			/* tell the client it can free the old request */
			vcam_p->vcam_old_vidp = vcam_p->vcam_vidp;
			vcam_p->vcam_vidp = NULL;
		}
		/* Lock the queue before we get the next command... */
		GET_CAMERA_LOCK(vcam_p,SERVER_LOCK)

		/* Now process a command if there is one... */ 
		np = QLIST_HEAD(vcam_p->vcam_cmd_lp);
		while( np != NULL ){
			Visca_Queued_Cmd *vqcp;

			vqcp = np->n_data;
			if( !vqcp->vqc_finished ){
				/* Now we're done with the queue */
				RLS_CAMERA_LOCK(vcam_p)
				exec_visca_command(vcam_p,vqcp->vqc_vcdp,pkt);
				vqcp->vqc_finished=1;
				np = NULL;
				did_something=1;
			}
			if( np != NULL )
				np = np->n_next;
		}
		if( vcam_p->vcam_qlock == SERVER_LOCK ){
			RLS_CAMERA_LOCK(vcam_p)
		}
		if( ! did_something )
			usleep(1000);		/* nothing to do? */
	}
	return(NULL);	/* not reached? */
}

#endif /* VISCA_THREADS */


/* we need to give the system time to send the code,
 * and send the ack back...
 * 9600 baud is about 1000 chars/sec, or 1 ms per char...
 * the number of chars is the number of hex digits/2,
 * plus 3 to return...  around 10 ms...
 * We wait our estimate, plus a little extra...
 *
 * This should work if the serial chars go out right away,
 * but in practice it doesn't!?  Sometimes even after the
 * sleep there are no readable chars!?
 *
 * The documentation says that the commands are sync'd to VBLANK!?
 * But we don't know exactly how - is the ACK delayed as well?
 *
 * It would be best to call select() instead of sleep()...
 *
 * But our original idea depended on baud rate...  if we are
 * only waiting for char transmission, we should use the baud
 * rate here when we calculate the needed delay...
 */

/* after adding the controlling thread, we needed to
 * increase this time... why ?
 */

#define exec_visca_command(vcam_p,vcdp,pkt)	_exec_visca_command(QSP_ARG  vcam_p,vcdp,pkt)

static void _exec_visca_command( QSP_ARG_DECL  Visca_Cam *vcam_p, Visca_Cmd_Def *vcdp, u_char *pkt )
{
	int r;

	send_hex(vcam_p->vcam_fd,pkt);
	
	/* BUG commands are sync'd to vertical refresh,
	 * should use select() instead of sleep()
	 */
	usleep( 2000*(strlen((char *)pkt)/2+5) );

	if( (r=get_cmd_ack(vcam_p,vcdp)) == 0 ){
//advise("cmd ack gotten, getting completion response...");
		get_cmd_completion(vcam_p,vcdp);
	} else if( r == 1 ){
		if( (r=get_cmd_ack(vcam_p,vcdp)) != 0 ){
			WARN("expected out-of-sequence ACK, but got something else!?");
		}
	} else {
		WARN("no command ack gotten!?");
	}

	/* Note that the echo of the command does not mean
	 * that the execution of the command has completed on the camera.
	 */
}

#define exec_visca_inquiry(vcam_p,vidp,pkt)	_exec_visca_inquiry(QSP_ARG  vcam_p,vidp,pkt)

static void _exec_visca_inquiry(QSP_ARG_DECL  Visca_Cam *vcam_p, Visca_Inq_Def *vidp, u_char *pkt)
{
	hex_digit_to_ascii(&(pkt[1]), vcam_p->vcam_index);
	send_hex(vcam_p->vcam_fd,(u_char *)pkt);
	get_inq_reply(vcam_p,vidp);
}

/* Make sure that this command is valid with the given camera.
 */

static int verify_cmd( Visca_Cam *vcam_p, Visca_Cmd_Def *vcdp )
{
	unsigned int i;

	for(i=0;i<N_COMMON_CMDS; i++) {
		if( strcmp(vcd_common_tbl[i].vcd_set,vcdp->vcd_set) == 0 ){
			if( strcmp(vcd_common_tbl[i].vcd_cmd,
					vcdp->vcd_cmd) == 0 ) 
				return 0;
		} 	
	
	}

	if( IS_EVI_D100(vcam_p) ) {
		for(i=0;i<N_EVI100_CMDS; i++) {
			if( strcmp(vcd_evi100_tbl[i].vcd_set,
					vcdp->vcd_set) == 0 ) 
				if( strcmp(vcd_evi100_tbl[i].vcd_cmd,
						vcdp->vcd_cmd) == 0 ) 
					return 0;
		}
	
	} else if( IS_EVI_D70(vcam_p) ) {
		/* use 100 cmds too! */
		for(i=0;i<N_EVI100_CMDS; i++) {
			if( strcmp(vcd_evi100_tbl[i].vcd_set,
					vcdp->vcd_set) == 0 ) 
				if( strcmp(vcd_evi100_tbl[i].vcd_cmd,
						vcdp->vcd_cmd) == 0 ) 
					return 0;
		}
		for(i=0;i<N_EVI70_CMDS; i++) {
			if( strcmp(vcd_evi70_tbl[i].vcd_set,
					vcdp->vcd_set) == 0 ) 
				if( strcmp(vcd_evi70_tbl[i].vcd_cmd,
						vcdp->vcd_cmd) == 0 ) 
					return 0;
		}
	} else {	

		for(i=0;i<N_EVI30_CMDS; i++) {
			if( strcmp(vcd_evi30_tbl[i].vcd_set,
					vcdp->vcd_set) == 0 ) 
				if( strcmp(vcd_evi30_tbl[i].vcd_cmd,
						vcdp->vcd_cmd) == 0 ) 
					return 0;
		}
	}

	return -1;
}

#define SET_ZOOM_ARG							\
									\
	sprintf(prompt, "zoom data, %d (wide) - 0x%x (tele)", 		\
	vparam_p->zoom_opt_pos_min, vparam_p->zoom_opt_pos_max );	\
									\
	zoom_data = HOW_MANY(prompt);					\
									\
	if( zoom_data < vparam_p->zoom_opt_pos_min || 			\
		zoom_data > vparam_p->zoom_opt_pos_max ){		\
									\
		sprintf(ERROR_STRING,					\
			"optical zoom data (0x%x) should be 0x%x-0x%x",	\
			zoom_data,vparam_p->zoom_opt_pos_min,		\
			vparam_p->zoom_opt_pos_max );			\
		WARN(ERROR_STRING);					\
		advise("defaulting to 0x200");				\
		zoom_data = 0x200;					\
	}								\
									\
	hex_digit_to_ascii(&(pkt[ 9]),(zoom_data>>12)&0xf);	\
	hex_digit_to_ascii(&(pkt[11]),(zoom_data>> 8)&0xf);	\
	hex_digit_to_ascii(&(pkt[13]),(zoom_data>> 4)&0xf);	\
	hex_digit_to_ascii(&(pkt[15]),(zoom_data>> 0)&0xf);

/* This is a misnomer, this code is only for evi70 combined zoom focus... */

#define DEFAULT_FOCUS_POSN	0x1000

#define SET_FOCUS_ARG							\
									\
	sprintf(prompt, "focus data, %d (near) - 0x%x (far)", 		\
		vparam_p->focus_pos_min, vparam_p->focus_pos_max );	\
									\
	focus_posn = HOW_MANY(prompt);					\
									\
	if( focus_posn < vparam_p->focus_pos_min || 			\
		focus_posn > vparam_p->focus_pos_max ){			\
									\
		sprintf(ERROR_STRING,					\
			"optical focus data should be 0x%x-0x%x", 	\
			vparam_p->focus_pos_min,			\
			vparam_p->focus_pos_max );			\
		WARN(ERROR_STRING);					\
		sprintf(ERROR_STRING,"defaulting to 0x%x",		\
						DEFAULT_FOCUS_POSN);	\
		advise(ERROR_STRING);					\
		focus_posn = DEFAULT_FOCUS_POSN;			\
	}								\
									\
	hex_digit_to_ascii(&(pkt[17]),(focus_posn>>12)&0xf);		\
	hex_digit_to_ascii(&(pkt[19]),(focus_posn>> 8)&0xf);		\
	hex_digit_to_ascii(&(pkt[21]),(focus_posn>> 4)&0xf);		\
	hex_digit_to_ascii(&(pkt[23]),(focus_posn>> 0)&0xf);

#define get_command_args(pkt,vcdp)	_get_command_args(QSP_ARG  pkt, vcdp)

static void _get_command_args(QSP_ARG_DECL  u_char *pkt, Visca_Cmd_Def *vcdp)
{
	char prompt[LLEN];
	int zoom_data;
	unsigned int focus_posn;
	int pan_posn, tilt_posn;
	int pt_lmt_choice ;
	const char *pt_lmt_set_strs[2] = {"downleft", "upright"};
	unsigned int gain;
	unsigned int speed;
	int n;

	switch(vcdp->vcd_argtype){
		
		case PWR_ARG:	/* evi100 */
			{
			int pwr_timer;
			
			sprintf(prompt,
	"auto power timeout (mins), %x (Timer off) - 0x%x (%d mins)", 
	MIN_TIMER_EVI100, MAX_TIMER_EVI100, MAX_TIMER_EVI100);
			pwr_timer = HOW_MANY(prompt);
			if( pwr_timer < MIN_TIMER_EVI100 ||
					pwr_timer > MAX_TIMER_EVI100 ){
				sprintf(ERROR_STRING,
			"power timer value should be 0x%x-0x%x",
			MIN_TIMER_EVI100, MAX_TIMER_EVI100);
				WARN(ERROR_STRING);
				advise("defaulting to 0 (Timer off)");
				pwr_timer = 0;
			}

			hex_digit_to_ascii(&(pkt[ 9]),(pwr_timer >>12)&0xf);
			hex_digit_to_ascii(&(pkt[11]),(pwr_timer >> 8)&0xf);
			hex_digit_to_ascii(&(pkt[13]),(pwr_timer >> 4)&0xf);
			hex_digit_to_ascii(&(pkt[15]),(pwr_timer >> 0)&0xf);
			}
			break;

		case ZOOM_SPEED:	/* evi100 & evi30 */

			sprintf(prompt,"focus speed (%d - %d)", vparam_p->zoom_speed_min, vparam_p->zoom_speed_max);
			speed = HOW_MANY(prompt);
			
			if( speed <  vparam_p->zoom_speed_min || speed > vparam_p->zoom_speed_max ){
				sprintf(ERROR_STRING, "zoom speed %d must be in the range %d - %d",speed,
					vparam_p->zoom_speed_min, vparam_p->zoom_speed_max);
				WARN(ERROR_STRING);
				advise("defaulting to 3");
				speed = 3;
			}

			pkt[9] = '0'+speed;
			break;
		

		case ZOOM_FOCUS_ARG:	/* evi70 */
			SET_ZOOM_ARG
			SET_FOCUS_ARG
			break;

		case ZOOM_OPT_ARG:	/* evi100 & evi30 */
			SET_ZOOM_ARG
			break;

		case TELE_ARG:
			sprintf(prompt,"wide/tele index 0 (low) - 7 (high)");
			zoom_data = HOW_MANY(prompt);
			if( zoom_data < 0 || zoom_data > 7 ){
				sprintf(ERROR_STRING, "wide/tele data should be in the range 0-7");
				WARN(ERROR_STRING);
				advise("defaulting to 0");
				zoom_data=0;
			}
			hex_digit_to_ascii(&(pkt[9]),zoom_data);
			break;

		case ZOOM_DIG_ARG:	/* evi100 */

			sprintf(prompt, "zoom data, 0x%x (X 1) - 0x%x (X 4)",MIN_ZOOM_DIG_EVI100, MAX_ZOOM_DIG_EVI100 );
			zoom_data = HOW_MANY(prompt);
			if( zoom_data < MIN_ZOOM_DIG_EVI100 || zoom_data > MAX_ZOOM_DIG_EVI100 ){
				sprintf(ERROR_STRING, "digital zoom data should be 0x%x-0x%x", MIN_ZOOM_DIG_EVI100, MAX_ZOOM_DIG_EVI100 );
				WARN(ERROR_STRING);
				advise("defaulting to 0x57ff");
				zoom_data = 0x57ff;
			}

			hex_digit_to_ascii(&(pkt[ 9]),(zoom_data>>12)&0xf);
			hex_digit_to_ascii(&(pkt[11]),(zoom_data>> 8)&0xf);
			hex_digit_to_ascii(&(pkt[13]),(zoom_data>> 4)&0xf);
			hex_digit_to_ascii(&(pkt[15]),(zoom_data>> 0)&0xf);
			break;

		case FOCUS_SPEED:	/* evi100 */

			sprintf(prompt,"focus speed (0x%x - 0x%x)", MIN_FOCUS_SPEED_EVI100, MAX_FOCUS_SPEED_EVI100);
			speed = HOW_MANY(prompt);
		
			if( speed < MIN_FOCUS_SPEED_EVI100 || speed > MAX_FOCUS_SPEED_EVI100 ){
				sprintf(ERROR_STRING, "Focus speed 0x%x must be in the range 0x%x - 0x%x",speed,
					MIN_FOCUS_SPEED_EVI100, MAX_FOCUS_SPEED_EVI100);
				WARN(ERROR_STRING);
				advise("defaulting to 3");
				speed = 3;
			}

			pkt[9] = '0'+speed;
			break;

		
		case FOCUS_POS:		/* evi100 & evi30 */

			sprintf(prompt,"focus setting (%x/far - %x/near)", vparam_p->focus_pos_min, vparam_p->focus_pos_max);
			focus_posn = HOW_MANY(prompt);

			if( focus_posn < vparam_p->focus_pos_min || focus_posn > vparam_p->focus_pos_max ){

				sprintf(ERROR_STRING, "Focus position (0x%x) must be in the range 0x%x - 0x%x",focus_posn,
					vparam_p->focus_pos_min,vparam_p->focus_pos_max);
				WARN(ERROR_STRING);
				advise("defaulting to 0x5000");
				focus_posn=0x5000;
			}
			hex_digit_to_ascii(&(pkt[15]),focus_posn & 0x000f);
			hex_digit_to_ascii(&(pkt[13]),(focus_posn & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[11]),(focus_posn & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[9]),(focus_posn & 0xf000)>>12);

			break;

		case FOCUS_NEAR_LMT_ARG:		/* evi100 */

			sprintf(prompt,"focus setting (%x/far - %x/near)", MIN_FOCUS_LMT_EVI100, MAX_FOCUS_LMT_EVI100);
			focus_posn = HOW_MANY(prompt);

			if( focus_posn < MIN_FOCUS_LMT_EVI100 || focus_posn > MAX_FOCUS_LMT_EVI100){

				sprintf(ERROR_STRING, "Focus position (0x%x) must be in the range 0x%x - 0x%x",focus_posn,
					MIN_FOCUS_LMT_EVI100, MAX_FOCUS_LMT_EVI100);
				WARN(ERROR_STRING);
				advise("defaulting to 0x5000");
				focus_posn=0x5000;
			}
			hex_digit_to_ascii(&(pkt[15]),focus_posn & 0x000f);
			hex_digit_to_ascii(&(pkt[13]),(focus_posn & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[11]),(focus_posn & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[9]),(focus_posn & 0xf000)>>12);

			break;
			
		case RGAIN_ARG:		/* evi100 */ 	
		
			sprintf(prompt,"R gain setting (%d - %d)", MIN_R_GAIN_EVI100, MAX_R_GAIN_EVI100);
			gain = HOW_MANY(prompt);
			
			if ( gain < MIN_R_GAIN_EVI100 || gain > MAX_R_GAIN_EVI100 ){

				sprintf(ERROR_STRING, "R gain (0x%x) must be in the range 0x%x - 0x%x",gain,
					MIN_R_GAIN_EVI100, MAX_R_GAIN_EVI100 );
				WARN(ERROR_STRING);
				advise("defaulting to 0x7f");
				gain=0x7f;
			}

			hex_digit_to_ascii(&(pkt[15]),gain & 0x000f);
			hex_digit_to_ascii(&(pkt[13]),(gain & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[11]),(gain & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[9]),(gain & 0xf000)>>12);
			
			break;
		
		case BGAIN_ARG: 	/* evi100 */

			sprintf(prompt,"B gain setting (0x%x - 0x%x)", MIN_B_GAIN_EVI100, MAX_B_GAIN_EVI100);
			gain = HOW_MANY(prompt);
	
			if ( gain < MIN_B_GAIN_EVI100 || gain > MAX_B_GAIN_EVI100 ){

				sprintf(ERROR_STRING, "B gain (0x%x) must be in the range 0x%x - 0x%x",gain,
					MIN_B_GAIN_EVI100, MAX_B_GAIN_EVI100 );
				WARN(ERROR_STRING);
				advise("defaulting to 0x7f");
				gain = 0x7f;
			}

			hex_digit_to_ascii(&(pkt[15]),gain & 0x000f);
			hex_digit_to_ascii(&(pkt[13]),(gain & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[11]),(gain & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[9]),(gain & 0xf000)>>12);
			
			break;

		case SHUTR_ARG:		/* evi100 & evi30 */
			{
			unsigned int shutter_speed;

			sprintf(prompt, "shutter speed %x(NTSC 1/4, PAL 1/3) - %x(1/10000)", 
				vparam_p->shutr_min, vparam_p->shutr_max);
			shutter_speed = HOW_MANY(prompt);

			if( shutter_speed < vparam_p->shutr_min || shutter_speed > vparam_p->shutr_max ){

				sprintf(ERROR_STRING, "Shutter speed (0x%x) must be in the range 0x%x - 0x%x",shutter_speed,
					vparam_p->shutr_min,vparam_p->shutr_max);
				WARN(ERROR_STRING);
				advise("defaulting to 0x0a");
				shutter_speed = 0x0a;
			}
			hex_digit_to_ascii(&(pkt[15]),shutter_speed & 0x000f);
			hex_digit_to_ascii(&(pkt[13]),(shutter_speed & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[11]),(shutter_speed & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[9]),(shutter_speed & 0xf000)>>12);
			}

			break;

		case IRIS_ARG: 		/* evi100 & evi30 */
			{
			unsigned int iris_posn;

			sprintf(prompt, "iris setting (%x/closed - %x/f1.8)",
				vparam_p->iris_pos_min, vparam_p->iris_pos_max);
			iris_posn = HOW_MANY(prompt);  
			
			if( iris_posn < vparam_p->iris_pos_min || iris_posn > vparam_p->iris_pos_max ){

				sprintf(ERROR_STRING,
					"Iris position (0x%x) must be in the range 0x%x - 0x%x",iris_posn,
						vparam_p->iris_pos_min, vparam_p->iris_pos_max);

				WARN(ERROR_STRING);
				advise("defaulting to 0x0008");
				iris_posn = 0x0008;
			}
			hex_digit_to_ascii(&(pkt[15]),iris_posn & 0x000f);
			hex_digit_to_ascii(&(pkt[13]),(iris_posn & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[11]),(iris_posn & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[9]),(iris_posn & 0xf000)>>12);
			}

			break;


		case GAIN_ARG: 		/* evi100 & evi30 */

			sprintf(prompt, "gain setting 0x%x - 0x%x",
				vparam_p->gain_pos_min, vparam_p->gain_pos_max);		
			gain = HOW_MANY(prompt);  
			
			if( gain < vparam_p->gain_pos_min || gain > vparam_p->gain_pos_max ){

				sprintf(ERROR_STRING, "Gain position (0x%x) must be in the range 0x%x - 0x%x",
					gain, vparam_p->gain_pos_min, vparam_p->gain_pos_max);
				
				WARN(ERROR_STRING);
				advise("defaulting to 0x0003");
				gain = 3;
			}
			hex_digit_to_ascii(&(pkt[15]),gain & 0x000f);
			hex_digit_to_ascii(&(pkt[13]),(gain & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[11]),(gain & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[9]),(gain & 0xf000)>>12);

			break;
		
		case BRIGHT_ARG: 		/* evi100 */
			{
			int bright_posn;

			sprintf(prompt, "bright_posn %x(close,0dB) - %x(F1.8,+18dB)", MIN_BRIGHT_EVI100, MAX_BRIGHT_EVI100);
 			bright_posn = HOW_MANY(prompt);  
			
			if( bright_posn < MIN_BRIGHT_EVI100 || bright_posn > MAX_BRIGHT_EVI100 ){

				sprintf(ERROR_STRING,
					"Bright position (0x%x) must be in the range 0x%x - 0x%x",bright_posn ,
					MIN_BRIGHT_EVI100,MAX_BRIGHT_EVI100);
				WARN(ERROR_STRING);
				advise("defaulting to 0x0010");
				bright_posn = 0x0010;
			}
			hex_digit_to_ascii(&(pkt[15]),bright_posn & 0x000f);
			hex_digit_to_ascii(&(pkt[13]),(bright_posn & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[11]),(bright_posn & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[9]),(bright_posn & 0xf000)>>12);
			}

			break;
			
		case EXP_COMP_ARG:		/* evi100 */
			{
			int exp_comp;

			sprintf(prompt, "exposure compensation setting %x(-10.5dB) - %x(+10.5dB)",MIN_EXP_COMP_EVI100, MAX_EXP_COMP_EVI100);
			exp_comp = HOW_MANY(prompt);		
			
			if( exp_comp < MIN_EXP_COMP_EVI100 || exp_comp > MAX_EXP_COMP_EVI100 ){

				sprintf(ERROR_STRING,
					"exposure compensation setting (0x%x) must be in the range 0x%x - 0x%x",exp_comp ,
					MIN_EXP_COMP_EVI100, MAX_EXP_COMP_EVI100);
				WARN(ERROR_STRING);
				advise("defaulting to 0x0007");
				exp_comp = 0x0007;
			}
			hex_digit_to_ascii(&(pkt[15]),exp_comp & 0x000f);
			hex_digit_to_ascii(&(pkt[13]),(exp_comp & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[11]),(exp_comp & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[9]),(exp_comp & 0xf000)>>12);
			}

			break;
			
		case APERTURE_ARG:		/* evi100 */

			sprintf(prompt, "gain setting (%x - %x)", MIN_APERTURE_GAIN_EVI100, MAX_APERTURE_GAIN_EVI100);
			gain = HOW_MANY(prompt);		
			
			if( gain < MIN_APERTURE_GAIN_EVI100 || gain > MAX_APERTURE_GAIN_EVI100){

				sprintf(ERROR_STRING,
					"Aperture position (0x%x) must be in the range 0x%x - 0x%x",gain,
					MIN_APERTURE_GAIN_EVI100, MAX_APERTURE_GAIN_EVI100);
				WARN(ERROR_STRING);
				advise("defaulting to 0x0005");
				gain = 5;
			}
			hex_digit_to_ascii(&(pkt[15]),gain & 0x000f);
			hex_digit_to_ascii(&(pkt[13]),(gain & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[11]),(gain & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[9]),(gain & 0xf000)>>12);

			break;
			
		case DIG_EFFECT_ARG:		/* evi100 */
			{
			int dig_choice;
			int dig_effect;
			const char *dig_strs[4] = {"still", "flash", "lumi", "trail"};

			dig_choice = WHICH_ONE("digital effect", 4, dig_strs); 
			dig_effect = 0x0;	// set default here to keep compiler quiet.
			
			if ( dig_choice == 0 || dig_choice == 2 ){	/* i.e. still or lumi */
				
				sprintf(prompt, "digital effect level for Still/Lumi (%x-%x)", 
						MIN_STILL_EFFECT_EVI100, MAX_STILL_EFFECT_EVI100);
				dig_effect = HOW_MANY(prompt);
				
				if ( dig_effect < MIN_STILL_EFFECT_EVI100 || dig_effect > MAX_STILL_EFFECT_EVI100 ) {
					sprintf(ERROR_STRING,
					"digital effect level (0x%x) must be in the range 0x%x - 0x%x",dig_effect,
					MIN_STILL_EFFECT_EVI100, MAX_STILL_EFFECT_EVI100 );
					WARN(ERROR_STRING);
					advise("defaulting to 0x0");			
					dig_effect = 0x0;
				}
			}
			
			if ( dig_choice == 1 || dig_choice == 3 ){	/* i.e. flash or trail */
				
				sprintf(prompt, "digital effect level for Flash/Trail (%x-%x)", 
						MIN_FLASH_EFFECT_EVI100, MAX_FLASH_EFFECT_EVI100);
				dig_effect = HOW_MANY(prompt);
				
				if ( dig_effect < MIN_FLASH_EFFECT_EVI100 || dig_effect > MAX_FLASH_EFFECT_EVI100 ) {
					sprintf(ERROR_STRING,
					"digital effect level (0x%x) must be in the range 0x%x - 0x%x",dig_effect,
					MIN_FLASH_EFFECT_EVI100, MAX_FLASH_EFFECT_EVI100 );
					WARN(ERROR_STRING);
					advise("defaulting to 0x0");				
					//dig_effect = 0x0;
				}
			}
			
			hex_digit_to_ascii(&(pkt[15]),dig_effect & 0x000f);
			hex_digit_to_ascii(&(pkt[13]),(dig_effect & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[11]),(dig_effect & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[9]),(dig_effect & 0xf000)>>12);
			}

			break;
			
		case MEM_ARG:		/* evi100 & evi30 */
			{
			unsigned int mem_index;

			sprintf(prompt, "memory index (0x%x-0x%x)", vparam_p->mem_min, vparam_p->mem_max);
			mem_index = HOW_MANY(prompt);
			if( mem_index < vparam_p->mem_min || mem_index > vparam_p->mem_max ){
				sprintf(ERROR_STRING,
			"Invalid memory index 0x%x - should be 0x%x-0x%x",mem_index, vparam_p->mem_min, vparam_p->mem_max);
				WARN(ERROR_STRING);
				advise("defaulting to 0");				
				mem_index=0;
			}
			hex_digit_to_ascii(&(pkt[11]),mem_index);
			}
			break;

		case PT_SPEED:		/* evi100 & evi30 */
			get_pan_speed(SINGLE_QSP_ARG);
			get_tilt_speed(SINGLE_QSP_ARG);
			set_pan_speed(pkt,the_vcam_p->vcam_pan_speed);
			set_tilt_speed(pkt,the_vcam_p->vcam_tilt_speed);
			break;

		case TILT_SPEED:		/* evi100 & evi30 */
			get_tilt_speed(SINGLE_QSP_ARG);
			set_tilt_speed(pkt,the_vcam_p->vcam_tilt_speed);
			break;

		case PAN_SPEED:		/* evi100 & evi30 */
			get_pan_speed(SINGLE_QSP_ARG);
			set_pan_speed(pkt,the_vcam_p->vcam_pan_speed);
			break;

		case PT_POSN:		/* evi100 & evi30 */

			get_pan_speed(SINGLE_QSP_ARG);
			get_tilt_speed(SINGLE_QSP_ARG);
			set_pan_speed(pkt,the_vcam_p->vcam_pan_speed);
			set_tilt_speed(pkt,the_vcam_p->vcam_tilt_speed);

			sprintf(prompt,"pan position (%d - %d)",
				vparam_p->pan_pos_min,vparam_p->pan_pos_max);
			pan_posn = HOW_MANY(prompt);
			if( pan_posn < vparam_p->pan_pos_min || pan_posn > vparam_p->pan_pos_max ){
				sprintf(ERROR_STRING,
		"Pan position (%d) must be in the range %d-%d",pan_posn,
					vparam_p->pan_pos_min,vparam_p->pan_pos_max);
				WARN(ERROR_STRING);
				advise("defaulting to 0");				
				pan_posn=0;
			}
			hex_digit_to_ascii(&(pkt[19]),pan_posn & 0x000f);
			hex_digit_to_ascii(&(pkt[17]),(pan_posn & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[15]),(pan_posn & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[13]),(pan_posn & 0xf000)>>12);

			sprintf(prompt,"tilt position (%d - %d)",
				vparam_p->tilt_pos_min,vparam_p->tilt_pos_max);
			tilt_posn = HOW_MANY(prompt);

			if( the_vcam_p->vcam_flipped ){
sprintf(ERROR_STRING,"Camera is flipped, limits are %d to %d",vparam_p->tilt_pos_min_flipped,vparam_p->tilt_pos_max_flipped);
advise(ERROR_STRING);
				if( tilt_posn < vparam_p->tilt_pos_min_flipped || tilt_posn > vparam_p->tilt_pos_max_flipped ){
					sprintf(ERROR_STRING,
		"Tilt position (%d) must be in the range %d - %d",tilt_posn,vparam_p->tilt_pos_min_flipped,vparam_p->tilt_pos_max_flipped);
					WARN(ERROR_STRING);
					advise("defaulting to 0");				
					tilt_posn=0;
				}
			} else {
				if( tilt_posn < vparam_p->tilt_pos_min || tilt_posn > vparam_p->tilt_pos_max ){
					sprintf(ERROR_STRING,
		"Tilt position (%d) must be in the range %d - %d",tilt_posn,vparam_p->tilt_pos_min,vparam_p->tilt_pos_max);
					WARN(ERROR_STRING);
					advise("defaulting to 0");				
					tilt_posn=0;
				}
			}
			hex_digit_to_ascii(&(pkt[27]),tilt_posn & 0x000f);
			hex_digit_to_ascii(&(pkt[25]),(tilt_posn & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[23]),(tilt_posn & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[21]),(tilt_posn & 0xf000)>>12);

			break;
		
		case PT_LMT_SET:	/* evi100 & evi30 */

			pt_lmt_choice = WHICH_ONE("pantilt downleft/upright", 2, pt_lmt_set_strs); 
			hex_digit_to_ascii(&(pkt[11]), (pt_lmt_choice & 0x0001));
		
			sprintf(prompt,"pan limit position (0x%x - 0x%x)",vparam_p->pan_lmt_min, vparam_p->pan_lmt_max);
			pan_posn = HOW_MANY(prompt);
			
			if( pan_posn < vparam_p->pan_lmt_min || pan_posn > vparam_p->pan_lmt_max ){
				sprintf(ERROR_STRING, "Pan limit position (0x%x) must be in the range 0x%x-0x%x",
					pan_posn, vparam_p->pan_lmt_min, vparam_p->pan_lmt_max);
				
				WARN(ERROR_STRING);
				advise("defaulting to 0");				
				pan_posn = 0;
			}

			hex_digit_to_ascii(&(pkt[19]),pan_posn & 0x000f);
			hex_digit_to_ascii(&(pkt[17]),(pan_posn & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[15]),(pan_posn & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[13]),(pan_posn & 0xf000)>>12);

			sprintf(prompt,"tilt limit position (0x%x - 0x%x)", vparam_p->tilt_lmt_min, vparam_p->tilt_lmt_max);
			tilt_posn = HOW_MANY(prompt);

			if( tilt_posn < vparam_p->tilt_lmt_min || tilt_posn > vparam_p->tilt_lmt_max ){
				sprintf(ERROR_STRING, "Tilt limit position (0x%x) must be in the range 0x%x-0x%x",
						tilt_posn, vparam_p->tilt_lmt_min, vparam_p->tilt_lmt_max);
				WARN(ERROR_STRING);
				advise("defaulting to 0");				
				tilt_posn=0;
			}
			
			hex_digit_to_ascii(&(pkt[27]),tilt_posn & 0x000f);
			hex_digit_to_ascii(&(pkt[25]),(tilt_posn & 0x00f0)>>4);
			hex_digit_to_ascii(&(pkt[23]),(tilt_posn & 0x0f00)>>8);
			hex_digit_to_ascii(&(pkt[21]),(tilt_posn & 0xf000)>>12);

		break;
		
		case PT_LMT_CLR:	/* evi100 & evi30 */
		
			pt_lmt_choice = WHICH_ONE("pantilt downleft/upright", 2, pt_lmt_set_strs); 
			hex_digit_to_ascii(&(pkt[11]), (pt_lmt_choice & 0x0001));
				
		break;
		

		case ATENB_ARG:		/* evi30 */
			
			if( ASKIF("enable auto track mode") )
				pkt[9] = '2';
			else
				pkt[9] = '3';
			break;

		case ATAE_ARG:		/* evi30 */
			
			if( ASKIF("auto-exposure for AT target") )
				pkt[9] = '2';
			else
				pkt[9] = '3';
			break;

		case ATAZ_ARG:		/* evi30 */
			
			if( ASKIF("auto-zoom for AT target") )
				pkt[9] = '2';
			else
				pkt[9] = '3';
			break;

			
		case OFFSET_ARG:	/* evi30 */
			
			if( ASKIF("shift sensing frame") )
				n=2;
			else	/* use pan-tilt drive */
				n=3;

			hex_digit_to_ascii(&(pkt[9]),n);
			break;

		case CHASE_ARG:		/* evi30 */
			
			n = HOW_MANY("chase mode (1,2,3)");
			if( n < 1 || n > 3 ){
				sprintf(ERROR_STRING,
			"Chase mode (%d) must be in the range 1-3",n);
				WARN(ERROR_STRING);
				n=1;
			}
			pkt[9] = '0'+n-1;
			break;
		

		case ENTRY_ARG:		/* evi30 */
			
			n = HOW_MANY("target study mode (1-4)");
			if( n < 1 || n > 4 ){
				sprintf(ERROR_STRING,
			"target study mode (%d) must be in the range 1-4",n);
				WARN(ERROR_STRING);
				n=1;
			}
			pkt[9] = '0'+n-1;
			break;
			

		case MDENB_ARG:		/* evi30 */
			
			if( ASKIF("enable motion detect mode") )
				pkt[9] = '2';
			else
				pkt[9] = '3';
			break;


		case MD_ARG:		/* evi30 */
			
			n = HOW_MANY("detecting condition (0-15)");
			if( n < 0 || n > 15 ){
				sprintf(ERROR_STRING, "Detecting level (%d) must be in the range 0-15",n);
				WARN(ERROR_STRING);
				
				n=7;
			}
			hex_digit_to_ascii(&(pkt[11]),n);
			break;

			
		case REF_ARG:		/* evi30 */
			
			n = HOW_MANY("refresh mode (1-3)");
			if( n < 1 || n > 3 ){
				sprintf(ERROR_STRING,
			"Refresh mode (%d) must be in the range 1-3",n);
				WARN(ERROR_STRING);
				n=1;
			}
			hex_digit_to_ascii(&(pkt[9]),n);
			break;

		case MM_ARG:		/* evi30 */
			
			if( ASKIF("enable measure mode") )
				n=2;
			else
				n=3;

			hex_digit_to_ascii(&(pkt[9]),n);
			break;

		 
		case NO_ARGS:
			/* nothing to do */
			break;

		case BACKLIGHT_ARG:
		case POWER_ARG:
		case ZOOM_DATA:
		case EXP_ARG:
			/* we got a compiler warning here after Saad added the evi100
			 * commands...
			 */
			WARN("Possibly improperly unhandled visca cmd arg case!?");
			break;

#ifdef CAUTIOUS
		default:
			WARN("CAUTIOUS:  unhandled visca cmd arg case");
			break;
#endif /* CAUTIOUS */
	}
} /* end get_command_args */

static COMMAND_FUNC( do_visca_cmd )
{
	Visca_Cmd_Def *vcdp;
	Visca_Cmd_Set *vcsp;
	Visca_Command *vcmd_p;
	u_char pkt[MAX_PACKET_LEN];

	vcsp = pick_cmd_set("command group");
	if( vcsp == NULL ){
		/* We eat a dummy word here to avoid a second error if there
		 * is a typo in the command group name.
		 */
		const char *s;
		s=NAMEOF("dummy word");
		// this message suppresses a compiler warning, var set but not used...
		if( verbose ){
			sprintf(ERROR_STRING,"Invalid command group, can't execute command '%s'",s);
			advise(ERROR_STRING);
		}
		return;
	}

	push_visca_cmd_context(vcsp->vcs_icp);

	vcmd_p = pick_visca_cmd("command");

	pop_visca_cmd_context();

	if( vcmd_p == NULL ) return;

//advise("do_visca_cmd");
	vcdp = vcmd_p->vcmd_vcdp;

	/* first make sure that we have a camera selected! */
	if( the_vcam_p == NULL ){
		WARN("No camera selected");
		advise("Please select a camera before issuing commands");
		return;
	}

	if( CAMERA_UNSPECIFIED(the_vcam_p) ){
		WARN("No camera type specified");
		advise("Please indicate camera type before issuing commands");
		return;
	}

	assert( the_vcam_p->vcam_param_p != NULL );
	
	if( verify_cmd(the_vcam_p,vcdp) < 0 ) {
		sprintf(ERROR_STRING, "%s %s not implemented by %s",
			vcdp->vcd_set, vcdp->vcd_cmd,
			the_vcam_p->vcam_name);
		WARN(ERROR_STRING);
		return;
	}	

	/* If this command has arguments, we need to get them into the
	 * packet code before we send it.
	 */
	
	strcpy((char *)pkt,(const char *)vcdp->vcd_pkt);
	get_command_args(pkt, vcdp);

	hex_digit_to_ascii(&(pkt[1]),the_vcam_p->vcam_index);
	
	/* To avoid waiting for the camera to execute a command,
	 * we're buffering the commands
	 * in asynchronous mode
	 */

#ifdef VISCA_THREADS
	if(async_reqs){
		Visca_Queued_Cmd *vqcp;
		Visca_Cmd_Def *q_vcdp;

		q_vcdp = getbuf(sizeof(Visca_Cmd_Def));
		*q_vcdp = *vcdp;
		q_vcdp->vcd_pkt = savestr(vcdp->vcd_pkt);

		vqcp = getbuf(sizeof(Visca_Queued_Cmd));
		vqcp->vqc_vcdp = q_vcdp;
		vqcp->vqc_finished = 0;

		queue_visca_command(vqcp);
	} else {
#endif /* VISCA_THREADS */
		exec_visca_command(the_vcam_p,vcdp,pkt);
#ifdef VISCA_THREADS
	}
#endif /* VISCA_THREADS */
} /* end do_visca_cmd() */

#ifdef VISCA_THREADS

static void cleanup_queue(Visca_Cam *vcam_p)
{
	Node *np;

	np = QLIST_HEAD(vcam_p->vcam_cmd_lp);
	while( np != NULL ){
		Visca_Queued_Cmd *vqcp;

		vqcp = np->n_data;
		if( vqcp->vqc_finished ){
			GET_CAMERA_LOCK(vcam_p,CLIENT_LOCK)
			np = remHead(vcam_p->vcam_cmd_lp);
			RLS_CAMERA_LOCK(vcam_p)

			givbuf(vqcp->vqc_vcdp->vcd_pkt);
			givbuf(vqcp->vqc_vcdp);
			givbuf(vqcp);
			rls_node(np);
			np = QLIST_HEAD(vcam_p->vcam_cmd_lp);
		} else {
			/* the head of the queue has not finished executing yet */
			return;
		}
	}
}

static void init_camera_queue(Visca_Cam *vcam_p)
{
	vcam_p->vcam_cmd_lp = new_list();
	init_server_thread(vcam_p);
}

static void queue_visca_command( Visca_Queued_Cmd *vqcp )
{
	Node *np;

	if( the_vcam_p->vcam_cmd_lp == NULL )
		init_camera_queue(the_vcam_p);

	/* Before we add this request, see if there are any commands at the head of the queue
	 * that can be removed.
	 */
	cleanup_queue(the_vcam_p);

	np = mk_node(vqcp);
	GET_CAMERA_LOCK(the_vcam_p,CLIENT_LOCK)
	addTail(the_vcam_p->vcam_cmd_lp,np);
	RLS_CAMERA_LOCK(the_vcam_p)
}

/* Because we want the answer to an inquiry RIGHT NOW there is no point
 * to queueing them in a list - but we have to let the server thread handle
 * them so they don't interfere with queued commands.
 */

static void queue_visca_inquiry( Visca_Inq_Def *vidp )
{
	Visca_Inq_Def *q_vidp;

	/* inquiries don't use the list, but we use the existence of the list
	 * as a flag to determine when to start the server thread.
	 */
	if( the_vcam_p->vcam_cmd_lp == NULL )
		init_camera_queue(the_vcam_p);

	q_vidp = getbuf(sizeof(Visca_Inq_Def));
	*q_vidp = *vidp;
	q_vidp->vid_pkt = savestr(vidp->vid_pkt);
	the_vcam_p->vcam_vidp = q_vidp;
	/* now wait... */
}

#endif /* VISCA_THREADS */

static int verify_inq(Visca_Inq_Def *vidp)
{
	unsigned int i;

	for(i=0;i<N_COMMON_INQS; i++) {
		if( strcmp(vid_common_tbl[i].vid_inq, vidp->vid_inq) == 0 ) {
				return 0;
		} 	
	}

	if( IS_EVI_D70(the_vcam_p) ){
		for(i=0;i<N_EVI70_INQS; i++) {
			if( strcmp(vid_evi70_tbl[i].vid_inq, vidp->vid_inq) == 0 ) 
				return 0;
		}
	}

	if( CAM_IS_70_OR_100(the_vcam_p) ) {
		for(i=0;i<N_EVI100_INQS; i++) {
			if( strcmp(vid_evi100_tbl[i].vid_inq, vidp->vid_inq) == 0 ) 
					return 0;
		}
	} else {	

		for(i=0;i<N_EVI30_INQS; i++) {
			if( strcmp(vid_evi30_tbl[i].vid_inq, vidp->vid_inq) == 0 ) 
					return 0;
		}
	}

	return -1;
} /* end verify_inq() */

static COMMAND_FUNC( do_visca_inq )
{
	Visca_Inq_Def *vidp;
	Visca_Inquiry *vip;
	u_char pkt[MAX_PACKET_LEN];
	
	vip = pick_visca_inq("inquiry");
	if( vip == NULL ){
		return;
	}
	
	vidp = vip->vi_vidp;
	
	if( the_vcam_p == NULL ) {
		assign_var(INQ_RESULT_NAME,"no_camera");
		return;
	}

	if( verify_inq(vidp) < 0 ) {
		sprintf(ERROR_STRING, "inquiry %s not implemented by %s", vidp->vid_inq,the_vcam_p->vcam_name);
		WARN(ERROR_STRING);
		return;
	}	

	/* We're moving this to exec_visca_inq... */
	//hex_digit_to_ascii(&(pkt[1]), the_vcam_p->vcam_index);
	strcpy((char *)pkt,vidp->vid_pkt);

#ifdef VISCA_THREADS
	/* BUG - if we are in asynchronous mode, we need
	 * to get the replies of outstanding commands first!
	 */
	if( async_reqs ){
		queue_visca_inquiry(vidp);
		/* We don't want to return until we have the answer!
		 * So we wait for it to finish...
		 *
		 * BUG?  processing the inquiry may call malloc (setting vars);
		 * Maybe should process the results from here instead...
		 */
		while( the_vcam_p->vcam_vidp != NULL ){
			usleep(500);
		}
		/* vcam_vidp moved to vcam_old_vidp */
		givbuf(the_vcam_p->vcam_old_vidp->vid_pkt);
		givbuf(the_vcam_p->vcam_old_vidp);
	} else {
		exec_visca_inquiry(the_vcam_p,vidp,pkt);
	}
#else /* ! VISCA_THREADS */
	exec_visca_inquiry(the_vcam_p,vidp,pkt);
#endif /* ! VISCA_THREADS */
} /* end do_visca_inq() */

static COMMAND_FUNC( do_set_async )
{
	int f;

	f = ASKIF("Set asynchronous mode ");
	// BUG?  generates compiler warning (set but not used) if no visca threads
#ifdef VISCA_THREADS
	if( f ){
		if( async_reqs ){
			WARN("Redundant request to set async VISCA mode");
		} else {
			async_reqs = 1;
		}
	} else {
		if( ! async_reqs ){
			WARN("Redundant request to clear async VISCA mode");
		} else {
			async_reqs = 0;	/* will cause thread to exit */
			/* BUG now we should wait for all threads to exit! */
		}
	}
#else	/* ! VISCA_THREADS */
	WARN("Sorry, not compiled for asynchronous execution within VISCA library");
	sprintf(ERROR_STRING,"Can't set async mode to %d",f);	// suppress compiler warning
	advise(ERROR_STRING);
#endif /* ! VISCA_THREADS */
}

static COMMAND_FUNC( select_cam )
{
	Visca_Cam *vcam_p;

	vcam_p=pick_vcam("");
	if( vcam_p == NULL ) return;

	if( vcam_p->vcam_param_p == NULL ){
		sprintf(ERROR_STRING,"Oops, the type of camera %s needs to be specified",
			vcam_p->vcam_name);
		WARN(ERROR_STRING);
		return;
	}

	the_vcam_p = vcam_p;
}

#ifdef HAVE_VISCA

/* add another camera to our databse.
 */

static void add_camera(QSP_ARG_DECL  Visca_Port *vport_p)
{
	Visca_Cam *vcam_p;
	char str[32];
	Node *np;
	u_char pkt[MAX_PACKET_LEN];
	int i;

	sprintf(str,"cam%d",++n_vcams);
	vcam_p = new_vcam(str);
	if( vcam_p == NULL ) return;

	np = mk_node(vcam_p);
	addTail(vport_p->vp_cam_lp,np);
	vcam_p->vcam_vport_p = vport_p;
	vcam_p->vcam_index = ++vport_p->vp_n_cams;	/* indices start at 1 - ? */
	vcam_p->vcam_type = N_CAM_TYPES;
	vcam_p->vcam_param_p = NULL;

#ifdef VISCA_THREADS
	vcam_p->vcam_cmd_lp = NULL;		/* this is the queue if in async mode... */
	vcam_p->vcam_vidp = NULL;
#endif /* VISCA_THREADS */

	/* Now ask the camera what is its type */
	/* INFO_INQ is the first entry... */
	i=table_index_for_inq(vid_common_tbl,INFO_INQ);
	assert( i >= 0 );

	strcpy((char *)pkt,(const char *)vid_common_tbl[i].vid_pkt);
	exec_visca_inquiry(vcam_p, &vid_common_tbl[i],pkt);

	/* Now set the camera type based on the info returned */
	switch( vcam_p->vcam_model_id ){
		case 0x402:
			vcam_p->vcam_type = EVI_D30;
			vcam_p->vcam_param_p = &evi30_params;
			break;
		case 0x40e:
			vcam_p->vcam_type = EVI_D70;
			vcam_p->vcam_param_p = &evi70_params;

			// Now query the camera to see if it is mounted upside-down (flipped)
			// We only do this if it is an evi-d70 - if the others support this,
			// they do not have a switch on the back panel.
			// FLIP_MODE_INQ is at index 1
			i=table_index_for_inq(vid_evi70_tbl,FLIP_MODE_INQ);
			assert( i >= 0 );

			strcpy((char *)pkt,(const char *)vid_evi70_tbl[i].vid_pkt);
			exec_visca_inquiry(vcam_p, &vid_evi70_tbl[i],pkt);
			if( vcam_p->vcam_flipped ){
				// Need to adjust limits for tilt!?
			}

			break;

		case 0x40d:
			vcam_p->vcam_type = EVI_D100;
			vcam_p->vcam_param_p = &evi100_params;
			break;
		default:
			sprintf(ERROR_STRING,"Unexpected camera model id:  0x%x",vcam_p->vcam_model_id);
			WARN(ERROR_STRING);
			break;
	}
}

/* was do_daisy_chain */

static void detect_daisy_chain(QSP_ARG_DECL  Visca_Port *vport_p)
{
/*#define ADDR_SET_BROADCAST 883001ff */

	int n=3;
	int n_cams = 0;
	u_char _buf[LLEN];		/* probably could be MUCH shorter!? */

	send_hex(vport_p->vp_fd, (u_char *)"883001ff");		/* address set command */
	
	/* wait until we have at least 3 chars before reading...
	 * BUG we should probably set an alarm here to wake us
	 * up in the case of something going really wrong.
	 */
	
	while( (n=n_serial_chars(vport_p->vp_fd)) < 3 ){
		/*usleep(1000); */
		/* sleep(1); */
		usleep(500000);
	}

	n = recv_somex(vport_p->vp_fd,_buf,LLEN,n);

	if( n==0 ){
		WARN("detect_daisy_chain:  no ack chars!?");
		return;
	}

	/* We expect the ack msg to be 0x88 0x30 0x0p 
	 * p = number of peripherals(cameras) + 1
	 */

	if( _buf[0] != 0x88 || _buf[1] != 0x30 || _buf[2] > MAX_NO_OF_PERIPHERALS ) { /* an error */
		const char *err_msg;

		err_msg = error_message(_buf[2]);
		sprintf(ERROR_STRING,"%s", err_msg);
		WARN(ERROR_STRING);
set_raw_len(_buf);
		dump_char_buf(_buf);
		return;
	}
	
	n_cams = _buf[2]-1;
	
	while(n_cams--)
		add_camera(vport_p);
	
	/* we initialize cam_type each time a new network is made */

	/* No completion response. */

} /* end detect_daisy_chain() */

#endif // HAVE_VISCA

#define vcam_info(vcam_p)	_vcam_info(QSP_ARG  vcam_p)

static void _vcam_info(QSP_ARG_DECL  Visca_Cam *vcam_p)
{
	const char *s;

	switch( vcam_p->vcam_type ) {
		case EVI_D100: s="EVI-D100"; break;
		case EVI_D70: s="EVI-D70"; break;
		case EVI_D30: s="EVI-D30"; break;
		default: s="(unspecified)"; break;
	}
	sprintf(msg_str, "\t%s\t%d\t%s:\t%s",
		vcam_p->vcam_vport_p->vp_name,
		vcam_p->vcam_index,vcam_p->vcam_name,s);
	prt_msg(msg_str);
}

#define vport_info(vport_p)	_vport_info(QSP_ARG  vport_p)

static void _vport_info(QSP_ARG_DECL  Visca_Port *vport_p)
{
	Node *np;

	assert( vport_p->vp_cam_lp != NULL );
	assert( eltcount(vport_p->vp_cam_lp) != 0 );

	np=QLIST_HEAD(vport_p->vp_cam_lp);
	while(np!=NULL){
		Visca_Cam *vcam_p;

		vcam_p = (Visca_Cam *)np->n_data;
		vcam_info(vcam_p);
		np=np->n_next;
	}
}

static COMMAND_FUNC( network_status )
{
	List *lp;
	Node *np;
	Visca_Port *vport_p;

	if( vport_itp == NULL ){
		WARN("network_status:  null vport_itp!?");
		return;
	}
	lp=item_list(vport_itp);
	if( lp == NULL || eltcount(lp)==0 ){
		WARN("No visca ports open");
		return;
	}
	np=QLIST_HEAD(lp);
	while(np!=NULL){
		vport_p = (Visca_Port *)np->n_data;
		vport_info(vport_p);
		np=np->n_next;
	}
}

static COMMAND_FUNC( do_vport_info )
{
	Visca_Port *vport_p;

	vport_p=pick_vport("");
	if( vport_p == NULL ) return;

	vport_info(vport_p);
}

#ifdef HAVE_VISCA

static Visca_Port *open_port(QSP_ARG_DECL  const char *name)
{
	Visca_Port *vport_p;
	int fd;

	fd = open_serial_device(name);
	if( fd < 0 ) return(NULL);

	/* Set the baud rate here in case somebody changed it by mistake */
	/* Perhaps should set all the flags too! */
	set_baud(fd,B9600);

	vport_p = new_vport(name);
	if( vport_p == NULL ){
		close(fd);			/* BUG? ignore return status? */
		return(vport_p);
	}

	vport_p->vp_fd = fd;
	vport_p->vp_cam_lp=new_list();
	vport_p->vp_n_cams=0;

	ttyraw(fd);

	/* here we should init the daisy chain */
	detect_daisy_chain(vport_p);

	return(vport_p);
} /* end open_port() */

#endif // HAVE_VISCA

static COMMAND_FUNC( do_vport_open )
{
#ifdef HAVE_VISCA
	Visca_Port *vport_p;
#endif // ! HAVE_VISCA
	const char *s;

	s=NAMEOF("Name of serial port device");
	// BUG generates set but not used compiler warning if no visca
#ifdef HAVE_VISCA
	vport_p = open_port(s);
#else // ! HAVE_VISCA
	NO_VISCA_MSG("serial port",s)
#endif // ! HAVE_VISCA
}

static COMMAND_FUNC(do_list_vports){list_vports(tell_msgfile());}

#define ADD_CMD(s,f,h)	ADD_COMMAND(visca_port_menu,s,f,h)

MENU_BEGIN(visca_port)
ADD_CMD( list,	do_list_vports,	list all open visca serial ports )
ADD_CMD( info,	do_vport_info,	print info about an open serial port )
ADD_CMD( open,	do_vport_open,	open visca serial port )
MENU_END( visca_port)

static COMMAND_FUNC( do_vport_menu )
{
	PUSH_MENU(visca_port);
}

static COMMAND_FUNC( do_vcam_info )
{
	Visca_Cam *vcam_p;

	vcam_p=pick_vcam("");
	if( vcam_p == NULL ) return;

	vcam_info(vcam_p);
}

static COMMAND_FUNC( do_get_cam_type )
{
	Visca_Cam *vcam_p;
	const char *s,*s2;

	vcam_p=pick_vcam("");
	s = NAMEOF("variable name");

	if( vcam_p == NULL ){
		assign_var(s,"no_camera");
		return;
	}

	switch( vcam_p->vcam_type ) {
		case EVI_D100: s2="EVI-D100"; break;
		case EVI_D70: s2="EVI-D70"; break;
		case EVI_D30: s2="EVI-D30"; break;
		default: s2="unidentified_camera"; break;
	}
	assign_var(s,s2);
}

static COMMAND_FUNC(do_list_vcams){list_vcams(tell_msgfile());}

static COMMAND_FUNC(do_get_n_cam)
{
	Visca_Port *vport_p;
	const char *s;
	int n,ntot=0;
	List *lp;
	Node *np;

	s=NAMEOF("variable name");

	if( vport_itp == NULL ){
		WARN("do_get_n_cam:  null vport_itp!?");
		return;
	}
	lp=item_list(vport_itp);
	if( lp == NULL || (n=eltcount(lp))==0 ){
		WARN("do_get_n_port:  No visca ports open");
		n=0;
	}

	np=QLIST_HEAD(lp);
	while(np!=NULL){
		vport_p = (Visca_Port *)np->n_data;
		assert( vport_p->vp_cam_lp != NULL );

		n = eltcount(vport_p->vp_cam_lp);
		assert( n != 0 );

		ntot += n;

		np=np->n_next;
	}

	sprintf(msg_str,"%d",ntot);
	assign_var(s,msg_str);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(visca_cam_menu,s,f,h)

MENU_BEGIN(visca_cam)
ADD_CMD( list,		do_list_vcams,		list all visca cameras )
ADD_CMD( info,		do_vcam_info,		print info about a camera )
ADD_CMD( get_n_cameras,	do_get_n_cam,		assign number of cameras to a script var )
ADD_CMD( get_cam_type,	do_get_cam_type,	assign number of cameras to a script var )
MENU_END(visca_cam)

static COMMAND_FUNC( do_vcam_menu )
{
	PUSH_MENU(visca_cam);
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(visca_menu,s,f,h)

MENU_BEGIN(visca)
// ADD_CMD( daisy_chain,	do_daisy_chain,	initialize camera addresses for daisy chain mode )
ADD_CMD( select,		select_cam,	select a camera )
ADD_CMD( network_status,	network_status,	show the types of cameras and their respective addresses )
ADD_CMD( command,		do_visca_cmd,	send a command to the camera )
ADD_CMD( inquire,		do_visca_inq,	send an inquiry to the camera )
ADD_CMD( set_asynchro,		do_set_async,	set the asynchronous mode to command the camera )
ADD_CMD( vports,		do_vport_menu,	visca port submenu )
ADD_CMD( cameras,		do_vcam_menu,	visca camera submenu )
MENU_END(visca)

#define DEFAULT_PORT_NAME "/dev/visca"

#ifdef HAVE_VISCA

static void default_camera(SINGLE_QSP_ARG_DECL)
{
	Visca_Port *vport_p;

	/* Don't try to open the default port if the directory entry doesn't exist... */

	/* BUG?  path_exists wants the pathname to refer to a directory or regular file
	 * - what will it do with a special file or symbolic link???
	 */
	if( ! path_exists(DEFAULT_PORT_NAME) ){
		/* say something here? */
		sprintf(ERROR_STRING,"Default visca port %s does not exist...",DEFAULT_PORT_NAME);
		advise(ERROR_STRING);
		return;
	}

	vport_p = open_port(DEFAULT_PORT_NAME);
	if( vport_p == NULL ){
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,"Unable to open default visca device %s",DEFAULT_PORT_NAME);
		return;
	}
	if( vport_p->vp_n_cams <= 0 ){
		sprintf(ERROR_STRING,"Default visca port %s is open, but no cameras detected",
			vport_p->vp_name);
		WARN(ERROR_STRING);
	} else {
		the_vcam_p = (Visca_Cam *)QLIST_HEAD(vport_p->vp_cam_lp)->n_data;
	}
}
#endif // HAVE_VISCA

COMMAND_FUNC( do_visca_menu )
{
#ifdef HAVE_VISCA
	if( the_vcam_p == NULL ){
		load_visca_cmds(SINGLE_QSP_ARG);
		default_camera(SINGLE_QSP_ARG);
	}
#else
	WARN("Program not compiled with VISCA support!?");
	advise("Will parse commands only.");
#endif // HAVE_VISCA

	PUSH_MENU(visca);
}

