#include "dataprot.h"
#include "ioctl_polhemus.h"

#ifdef INSIDE_TRACK
#define MAX_POLHEMUS_STATIONS	2
#else
#define MAX_POLHEMUS_STATIONS	4
#endif

typedef enum {
	PH_ERR_XMTR_ENV_CONN,
	PH_ERR_RCVR_CONN,
	PH_ERR_MOV_RAPID,
	PH_ERR_OUT_BOUNDS,
	PH_ERR_V_BOUNDS,
	PH_ERR_Q_BOUNDS,
	PH_ERR_XMTR_MAT,
	PH_ERR_RCVR_MAT,
	PH_ERR_COMP,
	N_POLH_ERRS
} Ph_Err_Code;

typedef struct ph_err {
	char *		pe_str;
	Ph_Err_Code	pe_code;
	char *		pe_msg;
} Ph_Err;


typedef struct output_datum {
	char *			od_name;
	Polh_Output_Type	od_type;
	int			od_code;
	int			od_words;	/* number of words transmitted by polhemus */
	int			od_strings;	/* number of strings in formatted output */
} Output_Datum;

extern Output_Datum od_tbl[N_OUTPUT_TYPES];

typedef struct formatted_point {
	char			fp_date_str[64];	/* BUG do we know this is long enough? */
	uint32_t		fp_seconds;
	uint32_t		fp_usecs;
	short			fp_station;
	float			fp_x;
	float			fp_y;
	float			fp_z;
	float			fp_azim;
	float			fp_elev;
	float			fp_roll;
	float			fp_xdc_x;
	float			fp_xdc_y;
	float			fp_xdc_z;
	float			fp_ydc_x;
	float			fp_ydc_y;
	float			fp_ydc_z;
	float			fp_zdc_x;
	float			fp_zdc_y;
	float			fp_zdc_z;
	float			fp_q1;
	float			fp_q2;
	float			fp_q3;
	float			fp_q4;
} Fmt_Pt;


typedef struct station_data {
	int 			sd_flags;		/* active, etc */
	Polh_Record_Format	sd_multi_prf;
	Polh_Record_Format	sd_single_prf;
} Station_Data;

#define STATION_ACTIVE		1

#define STATION_IS_ACTIVE(index)	(station_info[index].sd_flags & STATION_ACTIVE)


#define DEFAULT_SYNC_HZ		60			/* default softwaresync hz */
#ifdef INSIDE_TRACK
#define POLH_DEV		"/dev/polhemus"		/* polhemus device file (read-write) */
#else
#define POLH_DEV		"/dev/polhemus"		/* symlink to tty? */
//#define POLH_DEV		"/dev/ttyS15"		/* polhemus device file (read-write) */
//#define POLH_DEV		"/dev/ttyUSB0"		/* polhemus device file (read-write) */
//#define POLH_DEV		"/dev/ttyS0"		/* testing on dirac w/ port used by knox switcher */
#endif

#define POLH_DATA_DEV		"/dev/polhemusdata"	/* polhemus data device file w/ timestamps (read-only) */
#define POLH_STATION		1			/* active station number */
#define POLH_RECV		1			/* polhemus receiver number */
#define POLH_XMTR_NUM		1			/* polhemus transmitter number (always 1) */

/* Polhemus manual pg. A5 - A6 */
#define IN_PER_LSB		((300 / 2.54) / 32767.0)/* inches conversion */
#define CM_PER_LSB		(300 / 32767.0)		/* centimeter conversion */
#define DEG_PER_LSB		(180 / 32767.0) 	/* degree conversion */

#define RAW_TO_CM(raw)		(float)( *((short*)raw) * CM_PER_LSB)	/* raw data conversion */	
#define RAW_TO_IN(raw)		(float)( *((short*)raw) * IN_PER_LSB)	/* raw data conversion */	
#define RAW_TO_DEG(raw)		(float)( *((short*)raw) * DEG_PER_LSB)	/* raw data to degrees */


/* polhemus output record sizes (Polhemus manual: pg. A63 - A82) */
/* These are probably only applicable for binary data transfer, or with inside track? */
#define ALIGN_REC	68 
#define BORE_REC	26
#define XMTR_REC	30
#define RECV_REC	30
#define ATT_REC		34
#define POS_FTR_REC	34
#define SYNC_REC	6
#define ANG_REC		60
#define POS_ENV_REC	54
#define HEMI_REC	26

#ifdef INSIDE_TRACK
#define STAT_REC	8
#else
#define STAT_REC	9
#endif

#define STATUS_REC	56

#define PH_INTERNAL_SYNC	0
#define PH_EXTERNAL_SYNC	1
#define PH_SOFTWARE_SYNC	2

/* polhemus command codes */
typedef enum ph_cmd_code {
	PH_ALIGNMENT,
	PH_RESET_ALIGNMENT,
	PH_BORESIGHT,
	PH_REF_BORESIGHT,
	PH_RESET_BORESIGHT,
	PH_XMTR_ANGLES,
	PH_RECV_ANGLES,
	PH_ATT_FILTER,
	PH_POS_FILTER,
	/*
	PH_INTERNAL_SYNC,
	PH_EXTERNAL_SYNC,
	PH_SOFTWARE_SYNC,
	*/
	PH_SYNC_MODE,
	PH_REINIT_SYS,
	PH_ANGULAR_ENV,
	PH_POSITIONAL_ENV,
	PH_HEMISPHERE,
	PH_CONTINUOUS,
	PH_SINGLE_RECORD,
	PH_NONCONTINUOUS,
	PH_INCHES_FMT,
	PH_CM_FMT,
	PH_STATION,
	PH_STATUS,
	N_PH_CMD_CODES
} Ph_Cmd_Code;

#define IS_FILTER_CMD(c)	( c == PH_ATT_FILTER || c == PH_POS_FILTER )
#define IS_SYNC_CMD(c)		( c == PH_INTERNAL_SYNC || \
				  c == PH_EXTERNAL_SYNC || \
				  c == PH_SOFTWARE_SYNC || \
				  c == PH_SYNC_MODE )

/* polhemus command */
typedef enum ph_trs_num {
	PH_NEED_XMTR,		/* need transmitter number */
	PH_NEED_RECV,		/* need receiver number */
	PH_NEED_STAT,		/* need station number */
	PH_NEED_NONE,		/* needs nothing */
	N_PH_TRS_NUM
} Ph_Trs_Num;

typedef int ph_type_t;

typedef struct ph_cmd {
	const char *	pc_name;	/* our string */
	const char *	pc_cmdstr;	/* command string */
	Ph_Cmd_Code	pc_code;	/* polhemus command number */
	ph_type_t	pc_flags;	/* command type flag */
	Ph_Trs_Num	pc_trs;		/* polhemus transmitter/station/receiver flag */
	int		pc_n_args;	/* number of max. optional arguments */
	int		pc_rec_size;	/* output record size */
} Ph_Cmd;

#define STATION_CHAR(station_number)	('1'+station_number)

/* polhemus command types */
#define PH_SET		1		/* polhemus commands that set system attributes */
#define PH_GET		2		/* polhemus commands that get system attributes */
//  We used to have a reset type, but do we really need it to be distinct from SET?
//#define PH_RESET	4		/* polhemus command that reset system attributes */
#define PH_SG		(PH_SET|PH_GET) /* polhemus commands that set/get system attributes */

#define CHECK_PH_CMD( cmd, cmd_type )	( polh_cmds[cmd].pc_flags & cmd_type )
#define CAN_SET( cmd )		CHECK_PH_CMD( cmd, PH_SET )	
#define CAN_GET( cmd )		CHECK_PH_CMD( cmd, PH_GET )	
//#define CAN_RESET( cmd )	CHECK_PH_CMD( cmd, PH_RESET )	

//#define PH_CMD_TYPE_STR( type )		(type==PH_SET?"set":(type==PH_GET?"get":(type==PH_RESET?"reset":(type==PH_SG?"set/get":"invalid command type???"))))
#define PH_CMD_TYPE_STR( type )		(type==PH_SET?"set":(type==PH_GET?"get":(type==PH_SG?"set/get":"invalid command type???")))

/* polhemus device */
typedef enum polh_dev {
	POLH_NORMAL_DEVICE,
	POLH_DATA_DEVICE
} Ph_Dev;

/* globals */

extern int curr_station;
extern Station_Data station_info[2];
extern short resp_buf[];
extern debug_flag_t debug_polhemus;
extern int polh_fd;
extern Ph_Cmd polh_cmds[];
extern int which_receiver;
extern int polh_units;
extern int polh_continuous;	/* flag */
extern int n_active_stations;
extern int n_response_chars;

/* prototypes */

/* polh_dev.c */

extern char *read_polh_line(void);
extern int polh_getc(void);
extern int polh_ungetc(int);
extern void flush_polh_buffer(void);
extern void fill_polh_buffer(void);
extern void flush_input_data(void);
extern void show_output_data_format(int);
extern int polhemus_output_data_format( Polh_Record_Format *prfp );
extern void get_active_stations(void);

extern void display_buffer(short *buf,int n);
extern int read_polh_dev(short* databuf, int n_want);
extern int read_polh_data(void* raw_pdp, int n_want);

extern int init_polh_dev(void);
extern int send_string(const char *);
extern void read_response(int display_flag);
extern void clear_polh_dev(void);
extern int reopen_polh_dev(void);
extern int polhemus_word_count(void);
extern int polhemus_byte_count(void);

extern int send_polh_cmd(Ph_Cmd_Code, const char*);
extern int get_polh_info(Ph_Cmd_Code, const char*);
extern int read_single_polh_dp(Data_Obj *);
extern int read_next_polh_dp(Data_Obj *);
extern int read_cont_polh_dp(Data_Obj *);
extern void read_polh_vector(Data_Obj*);
extern int good_polh_vector(Data_Obj*);
extern void format_polh_vector(Data_Obj *);
extern void convert_polh_vector(Data_Obj *,Data_Obj *);
extern void display_formatted_point(Fmt_Pt *, Polh_Record_Format * );
extern int set_polh_sync_mode(int mode_code);
extern void set_polh_sync_rate(long);
extern int set_continuous_output_file(const char*);
extern COMMAND_FUNC( start_continuous_output_file );
extern COMMAND_FUNC( do_start_continuous_mode );
extern COMMAND_FUNC( do_stop_continuous_mode );
extern void start_continuous_mode(void);
extern void stop_continuous_mode(void);
extern void stop_continuous_output_file(void);
extern void sort_table(void);
extern void activate_station(int station,int flag);
extern void set_polh_units(Ph_Cmd_Code);
extern void dump_polh_data(void);
extern void raw_dump_polh_data(void);

/* polh_func.c */
extern int set_polh_angles(QSP_ARG_DECL Ph_Cmd_Code);
#define SET_POLH_ANGLES(code)				set_polh_angles(QSP_ARG code)
extern int set_polh_filter(QSP_ARG_DECL Ph_Cmd_Code);
#define SET_POLH_FILTER(code)				set_polh_filter(QSP_ARG code)
extern int ask_env(QSP_ARG_DECL const char* name, const char* type, float* max, float* min, float sys_max, float sys_min);
#define ASK_ENV(name,type,max,min,sys_max,sys_min)	ask_env(QSP_ARG name,type,max,min,sys_max,sys_min)

/* polh_err.c */
extern int check_polh_data(char *, Polh_Record_Format *);
extern int check_polh_output(char *, int, Ph_Cmd_Code );

/* acquire.c */
extern COMMAND_FUNC( polhemus_wait );
extern COMMAND_FUNC( polhemus_halt );
extern void polh_read_async(int);
extern int format_data( Fmt_Pt *fpp, Data_Obj *dp, Polh_Record_Format *prfp );
extern void assign_polh_var_data(const char *varname, Data_Obj *dp, Polh_Output_Type type, int index);

#ifdef FOOBAR
/* polhemus command macros */
#define SET_POLH_SYS(code)		(send_polh_cmd(code,NULL))
#define SET_POLH_SYS_ARGS(code,args)	(send_polh_cmd(code,args))
#define RESET_POLH_SYS(code)		(send_polh_cmd(code,NULL))
#endif /* FOOBAR */

