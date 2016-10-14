

#ifndef _VISCA_H
#define _VISCA_H

#ifdef INC_VERSION
char VersionId_inc_visca[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#include "quip_prot.h"
#include "item_obj.h"

#define MAX_VISCA_REQS 10 /* size of the buffer of requests */

#define MAX_VISCA_REQS                  10 /* size of the buffer of requests */
#define MAX_NO_OF_PERIPHERALS           7
#define MIN_VISCA_ADDR                  1
#define MAX_VISCA_ADDR                  7

typedef enum {
	EVI_D30,			/* 0 */
	EVI_D100,			/* 1 */
	EVI_D70,			/* 2 */
	N_CAM_TYPES			/* must be last */
} Camera_Type;

/* The visca interfaces allows cameras to be daisy-chained; additionally,
 * we may find it convenient to use two or more serial ports, with one
 * or more cameras attached.  We have one Visca_Port structure per serial port.
 */

typedef struct visca_port {
	Item		vp_item;
#define vp_name	vp_item.item_name
	int		vp_fd;
	List *		vp_cam_lp;
	int		vp_n_cams;
} Visca_Port;

#define NO_VISCA_PORT	((Visca_Port *)NULL)

#define MAX_VISCA_PORTS 2

/* These are codes which represent the different types of arguments that
 * we might need to pass with a command.
 */

typedef enum {
	PWR_ARG,
	ZOOM_SPEED,
	DIG_ZOOM_SPEED,		/* 70 only? or 100 too? */
	ZOOM_OPT_ARG,
	TELE_ARG,
	DZOOM_ARG,
	ZOOM_DIG_ARG,
	ZOOM_FOCUS_ARG,
	FOCUS_SPEED,
	FOCUS_POS,
	FOCUS_NEAR_LMT_ARG,
	RGAIN_ARG,
	BGAIN_ARG,
	SHUTR_ARG,
	IRIS_ARG,
	GAIN_ARG,
	BRIGHT_ARG,
	EXP_COMP_ARG,
	BACKLIGHT_ARG,
	APERTURE_ARG,
	DIG_EFFECT_ARG,
	MEM_ARG,
	PT_SPEED,
	TILT_SPEED,
	PAN_SPEED,
	PT_POSN,
	PT_LMT_SET,
	PT_LMT_CLR,


/* the rest are only for EVI-30 */
	POWER_ARG,
	ZOOM_DATA,
	EXP_ARG,
	ATENB_ARG,
	ATAE_ARG,
	ATAZ_ARG,
	OFFSET_ARG,
	CHASE_ARG,
	ENTRY_ARG,
	MDENB_ARG,
	MD_ARG,
	REF_ARG,
	MM_ARG,
	NO_ARGS

} Command_Args;

/* This is the structure we use to build the tables of commands */

typedef struct visca_cmd_def {
	const char *	vcd_set;	/* command group */
	const char *	vcd_cmd;	/* particular command name */
	const char *	vcd_pkt;	/* the packet string... */
	Command_Args	vcd_argtype;	/* code indicating the type and layout of args */
} Visca_Cmd_Def;

#define NO_CMD_DEF	((Visca_Cmd_Def *)NULL)

#define MAX_PACKET_LEN	32	/* BUG - where do we check this? */

typedef struct visca_queued_cmd {
	Visca_Cmd_Def *	vqc_vcdp;
	int		vqc_finished;
} Visca_Queued_Cmd;

/* These are codes which tell us what type of data return to expect...
 * Inquiries which have the same data return format will use the same code.
 */

typedef enum {
	NULL_INQ,	// we use 0 to terminate the table
	POSN_INQ,
	POWER_INQ,
	DIG_ZOOM_INQ,
	FOCUS_MODE_INQ,
	IR_MODE_INQ,
	FLIP_MODE_INQ,
	AFMODE_INQ,
	WBMODE_INQ,
	EXPMODE_INQ,
	SLOW_SHUTR_INQ,
	EXPCMP_MOD_INQ,
	BLMODE_INQ,
	WID_MOD_INQ,
	LR_REV_INQ,
	FREEZE_MOD_INQ,
	PIC_EFFECT_MOD_INQ,
	DIG_EFFECT_MOD_INQ,
	DIG_EFFECT_LVL_INQ,
	MEMORY_INQ,
	DATASCRN_INQ,
	PT_MODE_INQ,
	PT_MAX_SPEED_INQ,
	PT_POSN_INQ,
	VIDEO_INQ,
	INFO_INQ,
	IR_RECV_INQ,

/* the rest are only for EVI-30 */

	LOCK_INQ,
	ID_INQ,
	ATMD_MODE_INQ,
	AT_MODE_INQ,
	AT_ENTRY_INQ,
	MD_MODE_INQ,
	AT_POSN_INQ,
	MD_POSN_INQ,
	MD_Y_INQ,
	MD_HUE_INQ,
	MD_SIZE_INQ,
	MD_DISP_TIME_INQ,
	MD_REF_INQ,
	MD_REF_TIME_INQ,
	//UNDEF_INQ			/* We should get rid of this one */
} Inq_Type;

/* This structure is used to build the tables of inquiry commands */

typedef struct visca_inq_def {
	const char *	vid_inq;
	Inq_Type	vid_type;
	const char *	vid_pkt;
	const char *	vid_reply;
} Visca_Inq_Def;

#define NO_INQ_DEF	((Visca_Inq_Def *)NULL)

/* This structure is used to make the commands into Items - the name is just the command name,
 * but because some of the names are recycled between different command groups, we have
 * to create separate name contexts for each command group.  It is not totally clear that
 * we really need this extra struture...
 */

typedef struct visca_cmd {
	const char *	vcmd_name;
	Visca_Cmd_Def *	vcmd_vcdp;
} Visca_Command;

#define NO_VISCA_CMD	((Visca_Command *)NULL)

typedef struct visca_inquiry {
	const char *	vi_name;
	Visca_Inq_Def *	vi_vidp;
} Visca_Inquiry;

#define NO_VISCA_INQ	((Visca_Inquiry *) NULL)


typedef struct visca_cmd_set {
	const char *	vcs_name;
	Item_Context *	vcs_icp;	/* context for the commands */
} Visca_Cmd_Set;

#define NO_CMD_SET	((Visca_Cmd_Set *)NULL)

ITEM_INTERFACE_PROTOTYPES(Visca_Command,visca_cmd)
ITEM_INTERFACE_PROTOTYPES(Visca_Cmd_Set,cmd_set)
ITEM_INTERFACE_PROTOTYPES(Visca_Inquiry,visca_inq)
#define PICK_VISCA_CMD(p)	pick_visca_cmd(QSP_ARG  p)
#define PICK_CMD_SET(p)		pick_cmd_set(QSP_ARG  p)
#define PICK_VISCA_INQ(p)	pick_visca_inq(QSP_ARG  p)

extern Item_Context *create_visca_cmd_context(QSP_ARG_DECL  const char *name);
extern void push_visca_cmd_context(QSP_ARG_DECL  Item_Context *);
#define PUSH_VISCA_CMD_CONTEXT(icp)	push_visca_cmd_context(QSP_ARG  icp)

extern Item_Context *pop_visca_cmd_context(SINGLE_QSP_ARG_DECL);
#define POP_VISCA_CMD_CONTEXT		pop_visca_cmd_context(SINGLE_QSP_ARG)

typedef struct visca_params {
	uint32_t	zoom_speed_min;
	uint32_t	zoom_speed_max;

	uint32_t	zoom_opt_pos_min;
	uint32_t	zoom_opt_pos_max;

	uint32_t	focus_pos_min;
	uint32_t	focus_pos_max;

	uint32_t	shutr_min;
	uint32_t	shutr_max;

	uint32_t	iris_pos_min;
	uint32_t	iris_pos_max;

	uint32_t	gain_pos_min;
	uint32_t	gain_pos_max;

	uint32_t	mem_min;
	uint32_t	mem_max;

	uint32_t	pan_speed_min;
	uint32_t	pan_speed_max;

	uint32_t	tilt_speed_min;
	uint32_t	tilt_speed_max;

	int32_t		pan_pos_min;
	int32_t		pan_pos_max;

	int32_t		tilt_pos_min;
	int32_t		tilt_pos_max;

	int32_t		pan_lmt_min;
	int32_t		pan_lmt_max;

	int32_t		tilt_lmt_min;
	int32_t		tilt_lmt_max;

	int32_t		tilt_pos_min_flipped;
	int32_t		tilt_pos_max_flipped;

} Visca_Params;

#define NO_VISCA_PARAMS	((Visca_Params *)NULL)


typedef struct visca_cam {
	Item		vcam_item;
#define vcam_name	vcam_item.item_name
	Visca_Port *	vcam_vport_p;
	Camera_Type	vcam_type;
	/* these 4 are returned by INFO_INQ */
	int		vcam_vendor_id;
	int		vcam_model_id;
	int		vcam_rom_version;
	int		vcam_max_socket;

	Visca_Params *	vcam_param_p;
	int		vcam_index;
	int		vcam_pan_speed;
	int		vcam_tilt_speed;
#ifdef HAVE_PTHREADS
	List *		vcam_cmd_lp;
	pthread_t	vcam_ctl_thread;
	Visca_Inq_Def *	vcam_vidp;
	Visca_Inq_Def *	vcam_old_vidp;
	int		vcam_qlock;
#define SERVER_LOCK	1
#define CLIENT_LOCK	2

#endif /* HAVE_PTHREADS */

	int		vcam_flipped;	// evi-d70 only

} Visca_Cam;

#define NO_CAMERA	((Visca_Cam *)NULL)

#define IS_EVI_D30(vcam_p)	(vcam_p->vcam_type == EVI_D30)
#define IS_EVI_D100(vcam_p)	(vcam_p->vcam_type == EVI_D100)
#define IS_EVI_D70(vcam_p)	(vcam_p->vcam_type == EVI_D70)
#define CAMERA_UNSPECIFIED(vcam_p)	(vcam_p->vcam_type == N_CAM_TYPES)

#define vcam_fd		vcam_vport_p->vp_fd


/* some prototypes */
ITEM_INTERFACE_PROTOTYPES(Visca_Cam,vcam)
ITEM_INTERFACE_PROTOTYPES(Visca_Port,vport)
#define PICK_VCAM(p)	pick_vcam(QSP_ARG  p)
#define PICK_VPORT(p)	pick_vport(QSP_ARG  p)


#endif /* undev _VISCA_H */
