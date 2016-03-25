/* evi-d70 parameters */

#define MIN_TIMER_EVI70			0
#define MAX_TIMER_EVI70			0xffff

#define MIN_ZOOM_SPEED_EVI70			0
#define MAX_ZOOM_SPEED_EVI70			7

#define MIN_ZOOM_OPT_POS_EVI70			0
#define MAX_ZOOM_OPT_POS_EVI70			0x3fff

#define MIN_ZOOM_DIG_EVI70			0x4000
#define MAX_ZOOM_DIG_EVI70			0x7000

#define MIN_FOCUS_SPEED_EVI70			0
#define MAX_FOCUS_SPEED_EVI70			7

#define MIN_FOCUS_POSN_EVI70			0x1000
#define MAX_FOCUS_POSN_EVI70			0x8400

#define MIN_FOCUS_LMT_EVI70			0x1000
#define MAX_FOCUS_LMT_EVI70			0x8400

#define MIN_R_GAIN_EVI70			0x00
#define MAX_R_GAIN_EVI70			0xff

#define MIN_B_GAIN_EVI70			0x00
#define MAX_B_GAIN_EVI70			0xff

#define MIN_SHUTR_EVI70			0x0000
#define MAX_SHUTR_EVI70			0x0013

#define MIN_IRIS_POSN_EVI70			0x0000
#define MAX_IRIS_POSN_EVI70			0x0011

#define MIN_GAIN_POSN_EVI70			0x0000
#define MAX_GAIN_POSN_EVI70			0x0007

#define MIN_BRIGHT_EVI70			0x0000
#define MAX_BRIGHT_EVI70			0x0017

#define MIN_EXP_COMP_EVI70			0x0000
#define MAX_EXP_COMP_EVI70			0x000e

#define MIN_APERTURE_GAIN_EVI70		0x0000
#define MAX_APERTURE_GAIN_EVI70		0x000f

/* Since the min and max levels for still and lumi effects are the same 
 * therefore there is no need to define separate macros for lumi.
 * Same is the case for flash and trail.
 */
 
#define MIN_STILL_EFFECT_EVI70			0x00
#define MAX_STILL_EFFECT_EVI70			0x20

#define MIN_FLASH_EFFECT_EVI70			0x00
#define MAX_FLASH_EFFECT_EVI70			0x18

#define MIN_MEM_EVI70				0
#define MAX_MEM_EVI70				5

#define MIN_PAN_SPEED_EVI70			0x0001
#define MAX_PAN_SPEED_EVI70			0x0018		/* manual says 18, hex or decimal??? */

#define MIN_TILT_SPEED_EVI70			0x0001
#define MAX_TILT_SPEED_EVI70			0x0017		/* manual says 17, hex or decimal?? */

#define MIN_PAN_POSN_EVI70			-0x08db
#define MAX_PAN_POSN_EVI70			0x08db

#define MIN_TILT_POSN_EVI70			-0x0190
#define MAX_TILT_POSN_EVI70			0x04b0

#define MIN_PAN_LMT_EVI70			-0x08db
#define MAX_PAN_LMT_EVI70			0x08db

#define MIN_TILT_LMT_EVI70			-0x0190
#define MAX_TILT_LMT_EVI70			0x04b0

/* macros for evi100 */

#define MIN_TIMER_EVI100			0
#define MAX_TIMER_EVI100			0xffff

#define MIN_ZOOM_SPEED_EVI100			0
#define MAX_ZOOM_SPEED_EVI100			7

#define MIN_ZOOM_OPT_POS_EVI100			0
#define MAX_ZOOM_OPT_POS_EVI100			0x3fff

#define MIN_ZOOM_DIG_EVI100			0x4000
#define MAX_ZOOM_DIG_EVI100			0x7000

#define MIN_FOCUS_SPEED_EVI100			0
#define MAX_FOCUS_SPEED_EVI100			7

#define MIN_FOCUS_POSN_EVI100			0x1000
#define MAX_FOCUS_POSN_EVI100			0x8400

#define MIN_FOCUS_LMT_EVI100			0x1000
#define MAX_FOCUS_LMT_EVI100			0x8400

#define MIN_R_GAIN_EVI100			0x00
#define MAX_R_GAIN_EVI100			0xff

#define MIN_B_GAIN_EVI100			0x00
#define MAX_B_GAIN_EVI100			0xff

#define MIN_SHUTR_EVI100			0x0000
#define MAX_SHUTR_EVI100			0x0013

#define MIN_IRIS_POSN_EVI100			0x0000
#define MAX_IRIS_POSN_EVI100			0x0011

#define MIN_GAIN_POSN_EVI100			0x0000
#define MAX_GAIN_POSN_EVI100			0x0007

#define MIN_BRIGHT_EVI100			0x0000
#define MAX_BRIGHT_EVI100			0x0017

#define MIN_EXP_COMP_EVI100			0x0000
#define MAX_EXP_COMP_EVI100			0x000e

#define MIN_APERTURE_GAIN_EVI100		0x0000
#define MAX_APERTURE_GAIN_EVI100		0x000f

/* Since the min and max levels for still and lumi effects are the same 
 * therefore there is no need to define separate macros for lumi.
 * Same is the case for flash and trail.
 */
 
#define MIN_STILL_EFFECT_EVI100			0x00
#define MAX_STILL_EFFECT_EVI100			0x20

#define MIN_FLASH_EFFECT_EVI100			0x00
#define MAX_FLASH_EFFECT_EVI100			0x18

#define MIN_MEM_EVI100				0
#define MAX_MEM_EVI100				5

#define MIN_PAN_SPEED_EVI100			0x0001
#define MAX_PAN_SPEED_EVI100			0x0018

#define MIN_TILT_SPEED_EVI100			0x0001
#define MAX_TILT_SPEED_EVI100			0x0014

#define MIN_PAN_POSN_EVI100			-0x05a0
#define MAX_PAN_POSN_EVI100			0x05a0

#define MIN_TILT_POSN_EVI100			-0x0168
#define MAX_TILT_POSN_EVI100			0x0168

#define MIN_PAN_LMT_EVI100			-0x05a0
#define MAX_PAN_LMT_EVI100			0x05a0

#define MIN_TILT_LMT_EVI100			-0x0168
#define MAX_TILT_LMT_EVI100			0x0168


/* macros for evi30 */

#define MIN_ZOOM_SPEED_EVI30			2
#define MAX_ZOOM_SPEED_EVI30			7

#define MIN_ZOOM_OPT_POS_EVI30			0
#define MAX_ZOOM_OPT_POS_EVI30			0x03ff

#define MIN_FOCUS_POSN_EVI30			0x1000
#define MAX_FOCUS_POSN_EVI30			0x9fff

#define MIN_SHUTR_EVI30				0
#define MAX_SHUTR_EVI30				0x001b

#define MIN_IRIS_POSN_EVI30			0x0000
#define MAX_IRIS_POSN_EVI30			0x0011

#define MIN_GAIN_POSN_EVI30			0x0001
#define MAX_GAIN_POSN_EVI30			0x0007

#define MIN_MEM_EVI30				0
#define MAX_MEM_EVI30				0x0005

#define MIN_PAN_SPEED_EVI30			0
#define MAX_PAN_SPEED_EVI30			0x0018

#define MIN_TILT_SPEED_EVI30			0
#define MAX_TILT_SPEED_EVI30			0x0014

#define MIN_PAN_POSN_EVI30			-0x0370
#define MAX_PAN_POSN_EVI30			0x0370

#define MIN_TILT_POSN_EVI30			-0x012c
#define MAX_TILT_POSN_EVI30			0x012c

#define MIN_PAN_LMT_EVI30			-0x0370
#define MAX_PAN_LMT_EVI30			0x0370

#define MIN_TILT_LMT_EVI30			-0x012c
#define MAX_TILT_LMT_EVI30			0x012c



