
#ifdef __cplusplus
extern "C" {
#endif


#ifndef CONVERT_MENU_NAME

/* warrior menus */

#define MORPH_MENU_NAME		"Morph"
#define CONVERT_MENU_NAME	"Convert"
#define	UNCONV_MENU_NAME	"Unconvert"
#define UNARY_MENU_NAME		"Unary"
#define TRIG_MENU_NAME		"Trig"
#define LOG_MENU_NAME		"Logical"
#define VV_MENU_NAME		"VVector"
#define SV_MENU_NAME		"SVector"
#define CSV_MENU_NAME		"CSVector"
#define QSV_MENU_NAME		"QSVector"
#define MINMAX_MENU_NAME	"MinMax"
#define MISC_MENU_NAME		"Misc"
#define IMGSYN_MENU_NAME	"Sythesize"
#define FFT_MENU_NAME		"FFT"
#define COMP_MENU_NAME		"Compare"
#define LIN_MENU_NAME		"Linear"
#define EXPR_MENU_NAME		"Expressions"
#define MLAB_MENU_NAME		"Mlab"
#define CPX_MENU_NAME		"Complex"
#define WCTL_MENU_NAME		"WarCtl"
#define REQUANT_MENU_NAME	"Requantize"
#define OUTER_MENU_NAME		"Outer"
#define SAMPLE_MENU_NAME	"Sample"



/* data menus */

#define DATA_MENU_NAME		"Data"
#define ASCII_MENU_NAME		"Ascii"
#define MEMOP_MENU_NAME		"Operate"
#define AREA_MENU_NAME		"Areas"
#define CONTEXT_MENU_NAME	"Contexts"
#define FIO_MENU_NAME		"Hips"

#define DATATEST_MENU_NAME	"Datatest"

/* graphics */
#define LUT_MENU_NAME		"Colors"
#define SEQ_MENU_NAME		"Anim"

/* experiment lib */
#define STIM_MENU_NAME		"Stimuli"
#define ADD_CMD				"Add"
#define ANAL_MENU_NAME		"Analyse"
#define EXP_MENU_NAME		"Experiment"
#define MOD_CMD				"Modify"
#define XV_CMD				"Xvals"
#define DEL_CMD				"Delete"
#define TST_CMD				"Test"
#define RUN_CMD				"Run"
#define REP_STC				"Report_Staircases"
#define TRL_CMD				"Trials"
#define STC_CMD				"Staircases"
#define RPD_CMD				"Report_Data"
#define ANAL_CMD			"Analyse"
#define RDD_CMD				"Read Data"
#define REP_COND			"Report # of conditions"

/* jbm lib */
#define PITEM_MENU_NAME		"Change"
#define HELP_MENU_NAME		"Help"
/* BUG should integrate mac help & unix help ... */
#define UHELP_MENU_NAME		"help"
#define FILE_MENU_NAME		"File"
#define MACRO_MENU_NAME		"Macros"
#define ITEM_MENU_NAME		"Items"
#define VAR_MENU_NAME		"Variables"
#define CTRL_MENU_NAME		"Control"
#define MENU_MENU_NAME		"Menus"
#define PIPE_MENU_NAME		"Pipes"

#define ADJUST_MENU_NAME	"Adjust"
#define VERSION_MENU_NAME	"Versions"



/* lutmenu */
#define BITP_MENU_NAME		"Bitplanes"
#define CMAP_MENU_NAME		"Cmaps"
#define GAMMA_MENU_NAME		"Gamma"
#define LUTBUF_MENU_NAME	"Lutbufs"

/* xclib digital camera (epix) */
#define XCLIB_MENU_NAME		"XCLib"

/* cuda */
#define NPP_MENU_NAME		"Cuda_NPP"

/* some mac prototypes from init_menus.c */
/* BUG shouldn't these go elsewhere??? */
void mac_menu_init(void);
void var_menu_init(void);
void ctrl_menu_init(void);
void rd_cmdfile(void);
void file_menu_init(void);

#endif /* !CONVERT_MENU_NAME */

#ifdef __cplusplus
}
#endif

