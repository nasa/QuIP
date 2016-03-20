/* usb2000.h */

#include "data_obj.h"

#define USE_SERIAL_LINE			/* comment this out if we ever figure out usb protocol */

#define MIN_CALIB_INDEX			0
#define MAX_CALIB_INDEX			16
			
#define MIN_LS450_CALIB_INDEX		0
#define MAX_LS450_CALIB_INDEX		16

#define MAX_SIZEOF_CALIB_CONST		16	/* max length is 15 pg.13 */

#define _8_BIT_TIMER			1
#define _16_BIT_TIMER			0

#define _16_BIT_BAUD_TIMER		1
#define _8_BIT_BAUD_TIMER		0

#define STX				0x02
#define ETX				0x03
#define ACK				0x06
#define LF				0x0a
#define CR				0x0d
#define NAK				0x15
#define SPACE				0x20

#define LAMP_LOW			0
#define LAMP_HIGH			1

#define MAX_RECV_DATA_SIZE		5000
#define MAX_PKT_SIZE			100
#define MAX_N_OF_PIXELS			2048

#define TRUE				1
#define FALSE				0

#define ENABLE_STRG			1
#define DISABLE_STRG			0

#define ENABLE				1
#define DISABLE				0

// how many times we try to read... was 100
#define MAX_WAITS			10000

#define MIN_WAVELENGTH			347.532928
#define MAX_WAVELENGTH			1002.446289
#define BIN_WIDTH			3.1271311932	/* pixels per nm */
#define WAVELENGTH_RANGE		(MAX_WAVELENGTH-MIN_WAVELENGTH)

#define CH( x ) sprintf(error_string, "ch%d",x); advise(error_string);	/* used for debugging */

typedef enum {

	ADD_SCANS,
	PB_WIDTH,
	COMP,	
	INTG_TIME,
	LAMP,	
	BAUD,
	CLR_MEM,
	STORGE,
	PXL_MODE,
	SPEC_ACQ,
	TRIG_MODE,
	N_MEM_SCANS,
	RD_MEM_SCAN,
	ASCII,	
	BINARY,
	CHKSUM,
	VER,	
	CALIB_CONSTS,
	TIMER,	
	INQ,
	ACCESSORIES,	
	USB_ID,

	LS_450_ANALOG_OP,
	LS_450_LED_MODE,
	LS_450_TEMPERATURE,
	LS_450_CALIB_CONST,
	LS_450_INQ

} Cmd_Index;


typedef enum {

 NO_ARG,
 ONE_ARG,
 N_ARGS

} Arg; 


typedef struct usb2000_cmd_def {
	Cmd_Index	ucd_index;
	const char *	ucd_cmd;
	Arg		ucd_arg;
	
} USB2000_Cmd_Def;

typedef struct spectral_data {
	float		sd_spec_data[MAX_N_OF_PIXELS];
	float 		sd_wavlen[MAX_N_OF_PIXELS];

} Spectral_Data; 

typedef struct pxl_mode_info {
	short		pxl_case;
	short		n;
	short		x;
	short		y;
	short		pixels[10];

} Pxl_Mode_Info;


extern u_short chk_sum_mode;
extern u_short data_comp_mode;
extern u_long usb2000_debug;
extern u_short intg_clk_timer;
extern u_short data_strg_mode;
extern u_short ascii_mode;

/* prototypes */

/* usb2000.c */
extern void make_pkt( char *pkt, const char *cmd, u_int arg );
extern int get_tail(SINGLE_QSP_ARG_DECL);
extern void pxl_mode_n_args(int n, int *nbuf, char *n_args);
extern int xmit_pkt(USB2000_Cmd_Def *ucdp, int data_word,char *n_args);
extern void init_usb2000(SINGLE_QSP_ARG_DECL);
extern void clear_input_buf(SINGLE_QSP_ARG_DECL);

extern int add_scans(QSP_ARG_DECL  int data_word);
extern int set_pb_width(QSP_ARG_DECL  int data_word);
extern int data_comp(QSP_ARG_DECL  int data_word);
extern int set_integ_time(QSP_ARG_DECL  int data_word);
extern int set_lamp(QSP_ARG_DECL  int data_word);
extern int clr_spectra(QSP_ARG_DECL  int data_word);
extern int data_strg(QSP_ARG_DECL  int data_word);
extern int get_n_of_scans(SINGLE_QSP_ARG_DECL);
extern int get_scan(QSP_ARG_DECL  Spectral_Data *sdp);

extern int set_trig_mode(QSP_ARG_DECL  int data_word);
extern int set_checksum(QSP_ARG_DECL  int data_word);
extern int set_timer(QSP_ARG_DECL  int data_word);
extern int get_ver(QSP_ARG_DECL  char *ver);
extern int spec_acq(QSP_ARG_DECL  Spectral_Data *sdp);
extern int set_data_mode(QSP_ARG_DECL  Cmd_Index data_mode);
extern int set_pxl_mode(QSP_ARG_DECL  int pxl_mode, float x, float y, float n, float *wavelengths_p);
extern int set_calib_const(QSP_ARG_DECL  Cmd_Index cmd_index, int calib_index, const char *coeff_value);
extern int set_analog_op(QSP_ARG_DECL  int analog_op_val);
extern int set_led_mode(QSP_ARG_DECL  int led_mode);

extern int get_n_scans(SINGLE_QSP_ARG_DECL);
extern int get_pb_width(SINGLE_QSP_ARG_DECL);
extern int get_integ_time(SINGLE_QSP_ARG_DECL);
extern int get_lamp_status(SINGLE_QSP_ARG_DECL);
extern int get_baud_rate(SINGLE_QSP_ARG_DECL);
extern int get_trig_mode(SINGLE_QSP_ARG_DECL);
extern int get_timer_type(SINGLE_QSP_ARG_DECL);
extern int get_analog_op(SINGLE_QSP_ARG_DECL); 
extern int get_led_mode(SINGLE_QSP_ARG_DECL);
extern int get_temp(QSP_ARG_DECL  char *temp);
extern int do_calib_inq(QSP_ARG_DECL  Cmd_Index cmd_index, int calib_index, char *calib_const);


extern int do_cmd_1_arg(QSP_ARG_DECL  Cmd_Index cmd_index, int data_word);

/* tty_funcs.c */
extern int recv_a_byte(SINGLE_QSP_ARG_DECL);
extern int send_pkt(QSP_ARG_DECL  const char *);
extern void send_usb2000_packet(QSP_ARG_DECL  const char *,int);
extern int recv_a_value(SINGLE_QSP_ARG_DECL);
extern int set_baud_rate(QSP_ARG_DECL  int data_word);
extern int xmit_pxl_mode_pkt(QSP_ARG_DECL  const char *pkt, int pxl_mode, int n );
extern int open_usb2000(void);

/* usb_menu.c */
extern COMMAND_FUNC( do_usb2000_menu );
